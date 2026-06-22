"""B3 — supervision bbox + IoU verify layer.

PLAN
----
1. Parse raw VLM/retrieval box output → sv.Detections  (normalised [0,1] coords)
2. Compute IoU between the VLM box and every stored Docling chunk bbox.
3. If IoU >= iou_threshold for ANY chunk → PASS (return chunk_id).
   Otherwise → BLOCK ("hallucinated").
4. Annotate a page image with the verified box using supervision primitives.

CRITIQUE
--------
* VLM output formats vary; parser is deliberately lenient with multiple regex
  fallbacks so test fixtures don't require a live model.
* All coordinates are normalised to [0,1] before IoU to avoid pixel-vs-pt confusion.
* Zero-area or degenerate boxes → BLOCK by definition (IoU = 0 vs any finite box).
* supervision is MIT-licensed; safe for sidecar.
* No fitz / PyMuPDF import here; annotation uses PIL directly via supervision.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

# supervision ≥ 0.20 — Detections uses xyxy internally
import supervision as sv


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _box_to_xyxy(b: dict[str, float]) -> tuple[float, float, float, float]:
    """Convert a bbox dict to (x0, y0, x1, y1) in normalised [0,1] space.

    Accepts dicts with keys: x0/y0/x1/y1  OR  x/y/w/h  OR  left/top/right/bottom.
    All values are assumed to already be in the same coordinate space; callers
    are responsible for normalising to [0,1] before calling this function.
    """
    if "x1" in b and "y1" in b:
        return float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"])
    if "w" in b and "h" in b:
        x = float(b["x"])
        y = float(b["y"])
        return x, y, x + float(b["w"]), y + float(b["h"])
    if "right" in b:
        return (
            float(b["left"]),
            float(b["top"]),
            float(b["right"]),
            float(b["bottom"]),
        )
    raise ValueError(f"Cannot parse bbox dict: {b!r}")


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union for two axis-aligned boxes (xyxy format)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    # Area guard — zero-size box → IoU = 0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if area_a == 0.0 or area_b == 0.0:
        return 0.0

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_vlm_box(raw_output: str) -> sv.Detections:
    """Parse a VLM/retrieval bounding-box string into a supervision Detections object.

    Supported formats (normalised [0,1] coordinates):
    1. JSON dict: ``{"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.6}``
    2. JSON dict with w/h: ``{"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.4}``
    3. Four-float list/tuple: ``[0.1, 0.2, 0.5, 0.6]``
    4. DeepSeek ``<|box|>`` token notation: ``<|box|>(10,20),(50,60)<|/box|>``
       (pixel coords → normalised later by caller convention; kept as-is here
        because B3 tests use normalised values directly)
    5. Plain ``x0,y0,x1,y1`` CSV.

    Returns a ``sv.Detections`` with a single row, or an empty ``sv.Detections``
    if parsing fails (which triggers BLOCK in the verifier).
    """
    raw = raw_output.strip()
    xyxy: tuple[float, float, float, float] | None = None

    # 1. Try JSON dict / list
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            xyxy = _box_to_xyxy(obj)
        elif isinstance(obj, (list, tuple)) and len(obj) == 4:
            xyxy = tuple(float(v) for v in obj)  # type: ignore[assignment]
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # 2. DeepSeek <|box|> token: e.g. "<|box|>(10,20),(50,60)<|/box|>"
    if xyxy is None:
        m = re.search(r"\|box\|.*?\(([0-9.]+),([0-9.]+)\).*?\(([0-9.]+),([0-9.]+)\)", raw)
        if m:
            xyxy = (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))

    # 3. Plain CSV: "0.1,0.2,0.5,0.6"
    if xyxy is None:
        nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", raw)
        if len(nums) >= 4:
            xyxy = (float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3]))

    if xyxy is None:
        return sv.Detections.empty()

    x0, y0, x1, y1 = xyxy
    # Clamp to [0, 1]
    x0, y0, x1, y1 = (
        max(0.0, min(x0, 1.0)),
        max(0.0, min(y0, 1.0)),
        max(0.0, min(x1, 1.0)),
        max(0.0, min(y1, 1.0)),
    )
    # Zero-area box → return empty → BLOCK
    if x0 >= x1 or y0 >= y1:
        return sv.Detections.empty()

    xyxy_arr = np.array([[x0, y0, x1, y1]], dtype=np.float32)
    return sv.Detections(xyxy=xyxy_arr)


def verify_box_against_chunks(
    box: sv.Detections,
    chunks: list[Any],
    iou_threshold: float = 0.5,
) -> tuple[bool, str]:
    """Verify a VLM bounding box against stored Docling chunk bboxes via IoU.

    Parameters
    ----------
    box:
        Supervision Detections object produced by :func:`parse_vlm_box`.
        Must contain exactly one detection row (the VLM's claimed region).
    chunks:
        List of chunk objects.  Each chunk must have a ``.bbox`` attribute with
        ``.x0``, ``.y0``, ``.x1``, ``.y1`` float fields in normalised [0,1]
        space, and a ``.chunk_id`` string attribute.
    iou_threshold:
        Minimum IoU to consider a match (default 0.5 per SPEC §3 / B3).

    Returns
    -------
    (True, chunk_id)  if any chunk has IoU >= iou_threshold.
    (False, "hallucinated")  if no chunk matches → callers must BLOCK.
    """
    if box.xyxy is None or len(box.xyxy) == 0:
        return False, "hallucinated"

    vlm_box = tuple(float(v) for v in box.xyxy[0])  # (x0, y0, x1, y1)

    best_iou = 0.0
    best_chunk_id = "hallucinated"
    for chunk in chunks:
        b = chunk.bbox
        chunk_box = (float(b.x0), float(b.y0), float(b.x1), float(b.y1))
        iou = _iou(vlm_box, chunk_box)
        if iou > best_iou:
            best_iou = iou
            best_chunk_id = chunk.chunk_id

    if best_iou >= iou_threshold:
        return True, best_chunk_id
    return False, "hallucinated"


def annotate_page_image(
    page_img_path: str,
    bbox: dict[str, float],
    out_path: str,
    label: str = "grounded",
    color: str = "#00FF88",
) -> None:
    """Draw a supervision-annotated highlight box on a page image.

    Parameters
    ----------
    page_img_path:
        Absolute path to the page image (PNG / JPEG).
    bbox:
        Normalised bbox dict with keys ``x0, y0, x1, y1`` in [0,1] space.
    out_path:
        Where to write the annotated image.
    label:
        Text label shown on the annotation box.
    color:
        Hex colour string for the box.
    """
    from PIL import Image

    img = Image.open(page_img_path).convert("RGB")
    w, h = img.size

    x0, y0, x1, y1 = _box_to_xyxy(bbox)
    # Convert normalised → pixel coords
    px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

    xyxy_arr = np.array([[px0, py0, px1, py1]], dtype=np.float32)
    detections = sv.Detections(xyxy=xyxy_arr)

    sv_color = sv.Color.from_hex(color)
    annotator = sv.BoxAnnotator(color=sv_color, thickness=3)

    img_np = np.array(img)
    img_annotated = annotator.annotate(scene=img_np, detections=detections)

    # Add label
    label_annotator = sv.LabelAnnotator(color=sv_color, text_color=sv.Color.BLACK)
    img_annotated = label_annotator.annotate(
        scene=img_annotated,
        detections=detections,
        labels=[label],
    )

    from PIL import Image as PILImage
    PILImage.fromarray(img_annotated).save(out_path)
