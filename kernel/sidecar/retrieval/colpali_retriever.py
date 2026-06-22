"""B2 — ColQwen2 / ColPali late-interaction visual retrieval.

PLAN
----
1. ColPaliRetriever wraps the colpali_engine library (ColQwen2 model family).
2. On init, attempt to import colpali_engine; if absent, set _available=False.
3. index_page(page_img_bytes, page_meta): encode a page image into a multi-vector
   embedding (ColPali "late interaction" style) and cache it locally.
4. retrieve(query_text, top_k): encode the query, compute MaxSim scores against
   all indexed pages, return top_k results sorted by descending score.
5. Fallback: when colpali_engine is not installed, fall back to the existing
   hash embedding using cosine dot-product on the hash vector of the page metadata
   text. Results are still normalised and carry the same dict schema so callers
   need no branching logic.
6. VisualPatchRetriever extends the architecture to patch-grid retrieval: each
   indexed page is sliced into nrows×ncols patches, each patch is indexed
   independently, and the top-matching patch bbox is returned for B3 IoU
   verification. Controlled by a per-document ``enabled`` flag so text-native
   docs skip visual retrieval entirely.

CRITIQUE
--------
* ColPali/ColQwen2 requires a GPU and large model weights (~7 GB); the library is
  NOT required for tests that run in CI.  The fallback path is always exercised
  in offline test environments.
* The MaxSim score in the fallback is a dot product of 32-d hash vectors — not
  semantically meaningful, but deterministic and structurally correct.
* Page images are accepted as raw bytes to decouple this module from any specific
  file-system layout used by the sidecar.
* No GPU memory management is performed here; callers are responsible for model
  lifecycle (load once, reuse).
* VisualPatchRetriever has a PIL-based path (preferred) and a synthetic-patch
  fallback (for minimal-dep environments).
"""
from __future__ import annotations

import hashlib
import io
import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Internal: deterministic hash embedding (fallback when colpali absent)
# ---------------------------------------------------------------------------

def _hash_embed_text(text: str, dim: int = 32) -> list[float]:
    """SHA-256 based deterministic pseudo-embedding.

    SHA-256 produces a 32-byte digest.  dim is capped at 32; passing a larger
    value will silently return 32 elements.
    """
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    vec = [b / 255.0 for b in digest[:min(dim, len(digest))]]
    return vec


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-8
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-8
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# ColPaliRetriever — page-level visual retrieval
# ---------------------------------------------------------------------------

class ColPaliRetriever:
    """Late-interaction visual retrieval using ColQwen2 / ColPali.

    When colpali_engine is available, uses the real multi-vector MaxSim scoring.
    Otherwise falls back to deterministic hash embeddings for offline use.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the ColPali family.
        Default: ``"vidore/colqwen2-v1.0"`` (ColQwen2).
    device:
        Torch device string (``"cuda"``, ``"cpu"``, ``"mps"``).
        Ignored in fallback mode.
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._available = False
        self._model: Any = None
        self._processor: Any = None

        # Indexed pages: list of dicts with keys:
        #   page_index (int), chunk_id (str), embedding (list[float] or tensor)
        self._index: list[dict] = []

        try:
            from colpali_engine.models import ColQwen2, ColQwen2Processor  # type: ignore[import]
            import torch  # type: ignore[import]

            self._model = ColQwen2.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            ).to(device).eval()
            self._processor = ColQwen2Processor.from_pretrained(model_name)
            self._available = True
        except ImportError:
            # colpali_engine or torch not installed — use fallback
            pass
        except Exception:
            # Model load failure (no weights, CUDA OOM, etc.) — use fallback
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_page(self, page_img_bytes: bytes, page_meta: dict) -> None:
        """Encode and index a single page image.

        Parameters
        ----------
        page_img_bytes:
            Raw image bytes (PNG / JPEG).  Decoded by PIL internally.
        page_meta:
            Dict with at least ``page_index`` (int) and ``chunk_id`` (str).
            Additional keys (e.g. ``doc_id``) are stored verbatim.
        """
        page_index = int(page_meta.get("page_index", len(self._index)))
        chunk_id = str(page_meta.get("chunk_id", f"page-{page_index}"))

        if self._available and self._model is not None:
            embedding = self._encode_image_colpali(page_img_bytes)
        else:
            # Fallback: hash the page metadata text + raw bytes fingerprint
            fingerprint = hashlib.sha256(page_img_bytes[:256]).hexdigest()
            meta_text = f"page={page_index} chunk={chunk_id} fp={fingerprint}"
            embedding = _hash_embed_text(meta_text)

        self._index.append(
            {
                "page_index": page_index,
                "chunk_id": chunk_id,
                "meta": page_meta,
                "embedding": embedding,
            }
        )

    def retrieve(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Return top_k pages ranked by relevance to query_text.

        Returns
        -------
        List of dicts, each with keys:
            ``page_index`` (int), ``score`` (float in [0, 1]), ``chunk_id`` (str).
        Sorted by descending score.  Empty list if nothing is indexed.
        """
        if not self._index:
            return []

        if self._available and self._model is not None:
            query_emb = self._encode_query_colpali(query_text)
            scores = [
                self._maxsim(query_emb, entry["embedding"])
                for entry in self._index
            ]
        else:
            query_emb = _hash_embed_text(query_text)
            scores = [
                _cosine_sim(query_emb, entry["embedding"])
                for entry in self._index
            ]

        # Normalise scores to [0, 1] across the result set
        min_s = min(scores) if scores else 0.0
        max_s = max(scores) if scores else 1.0
        span = max_s - min_s if max_s != min_s else 1.0

        ranked = sorted(
            zip(scores, self._index),
            key=lambda t: t[0],
            reverse=True,
        )

        results = []
        for raw_score, entry in ranked[:top_k]:
            normalised = (raw_score - min_s) / span
            results.append(
                {
                    "page_index": entry["page_index"],
                    "score": float(round(normalised, 6)),
                    "chunk_id": entry["chunk_id"],
                }
            )
        return results

    # ------------------------------------------------------------------
    # ColPali-specific helpers (only called when _available is True)
    # ------------------------------------------------------------------

    def _encode_image_colpali(self, img_bytes: bytes) -> Any:
        """Encode a page image into ColPali multi-vector embeddings."""
        from PIL import Image  # type: ignore[import]
        import torch  # type: ignore[import]

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        batch = self._processor.process_images([img]).to(self._device)
        with torch.no_grad():
            emb = self._model(**batch)
        return emb[0]  # (seq_len, dim) tensor

    def _encode_query_colpali(self, query_text: str) -> Any:
        """Encode a text query into ColPali multi-vector embeddings."""
        import torch  # type: ignore[import]

        batch = self._processor.process_queries([query_text]).to(self._device)
        with torch.no_grad():
            emb = self._model(**batch)
        return emb[0]  # (seq_len, dim) tensor

    @staticmethod
    def _maxsim(query_emb: Any, page_emb: Any) -> float:
        """ColPali MaxSim score: sum of per-query-token max cosine over page tokens."""
        import torch  # type: ignore[import]

        # query_emb: (Nq, dim), page_emb: (Np, dim)
        scores = torch.einsum("id,jd->ij", query_emb, page_emb)  # (Nq, Np)
        return float(scores.max(dim=1).values.sum().item())

    # ------------------------------------------------------------------
    # Introspection helpers (useful in tests)
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True when colpali_engine is installed and the model loaded."""
        return self._available

    @property
    def index_size(self) -> int:
        """Number of pages currently in the index."""
        return len(self._index)


# ---------------------------------------------------------------------------
# VisualPatch — data container for a single patch
# ---------------------------------------------------------------------------

class VisualPatch:
    """A spatial patch of a page image with its normalised bounding box.

    Attributes
    ----------
    patch_idx : int
        Linear index of this patch in the grid (row-major).
    row : int
        Row of this patch in the grid (0-indexed).
    col : int
        Column of this patch in the grid (0-indexed).
    page_index : int
        Page number this patch belongs to.
    bbox : dict[str, float]
        Normalised [0, 1] bounding box: {x0, y0, x1, y1}.
    img_bytes : bytes
        Raw PNG bytes of the cropped patch region.
    """

    def __init__(
        self,
        patch_idx: int,
        row: int,
        col: int,
        page_index: int,
        bbox: dict[str, float],
        img_bytes: bytes,
    ) -> None:
        self.patch_idx = patch_idx
        self.row = row
        self.col = col
        self.page_index = page_index
        self.bbox = bbox
        self.img_bytes = img_bytes


# ---------------------------------------------------------------------------
# VisualPatchRetriever — patch-grid visual retrieval feeding B3
# ---------------------------------------------------------------------------

class VisualPatchRetriever:
    """Patch-grid visual retrieval with ColQwen2/ColPali late interaction.

    Divides each indexed page into an ``nrows × ncols`` patch grid. Encodes
    each patch independently with ColPali (or hash fallback), then ranks patches
    by MaxSim score against a query. The top-matching patch's bounding box is
    returned as a candidate region for B3 IoU verification.

    Per-document visual retrieval is controlled by the ``enabled`` flag. When
    ``enabled=False`` (text-native documents), all retrieval calls return an
    empty list immediately, preserving throughput for native-parse fast paths.

    Parameters
    ----------
    nrows, ncols : int
        Patch grid dimensions.  Default 4×4 (16 patches per page).
    colpali_model_name : str
        ColPali HuggingFace model identifier.  Ignored in fallback mode.
    device : str
        Torch device for ColPali.  Ignored in fallback mode.
    enabled : bool
        Per-document visual retrieval flag.  Set ``False`` for text-native docs.
    """

    def __init__(
        self,
        nrows: int = 4,
        ncols: int = 4,
        colpali_model_name: str = "vidore/colqwen2-v1.0",
        device: str = "cpu",
        enabled: bool = True,
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.enabled = enabled

        # Delegate embedding/scoring to the existing ColPaliRetriever
        self._retriever = ColPaliRetriever(
            model_name=colpali_model_name,
            device=device,
        )

        # Map from retriever chunk_id → VisualPatch for bbox lookup
        self._patch_map: dict[str, VisualPatch] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_page_patches(
        self,
        page_img_bytes: bytes,
        page_index: int,
        doc_id: str = "",
    ) -> list[VisualPatch]:
        """Slice a page image into a patch grid and index each patch.

        Parameters
        ----------
        page_img_bytes:
            Raw PNG/JPEG bytes of the full page render.
        page_index:
            0-based page number within the document.
        doc_id:
            Document identifier, used to form stable chunk IDs.

        Returns
        -------
        List of :class:`VisualPatch` objects for the indexed patches.
        Empty list if ``enabled=False``.
        """
        if not self.enabled:
            return []

        patches = self._slice_into_patches(page_img_bytes, page_index, doc_id)
        for patch in patches:
            chunk_id = f"{doc_id}:p{page_index}:r{patch.row}c{patch.col}"
            meta = {
                "page_index": page_index,
                "chunk_id": chunk_id,
                "patch_idx": patch.patch_idx,
                "doc_id": doc_id,
            }
            self._retriever.index_page(patch.img_bytes, meta)
            self._patch_map[chunk_id] = patch

        return patches

    def retrieve_patch(
        self,
        query_text: str,
        page_index: int | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Return the top-scoring patch region(s) for the given query.

        Parameters
        ----------
        query_text:
            Natural-language query (e.g. "Q4 revenue cell").
        page_index:
            If given, only patches from this page are considered.
            ``None`` searches across all indexed pages.
        top_k:
            Maximum number of patch results to return.

        Returns
        -------
        List of dicts with keys:
            ``chunk_id`` (str), ``score`` (float), ``bbox`` (dict x0/y0/x1/y1),
            ``page_index`` (int), ``patch_idx`` (int).
        Sorted by descending score.  Empty list if ``enabled=False`` or nothing indexed.
        """
        if not self.enabled or self._retriever.index_size == 0:
            return []

        # Retrieve all patches, then filter by page
        all_results = self._retriever.retrieve(
            query_text, top_k=self._retriever.index_size
        )

        results = []
        for entry in all_results:
            cid = entry["chunk_id"]
            patch = self._patch_map.get(cid)
            if patch is None:
                continue
            if page_index is not None and patch.page_index != page_index:
                continue
            results.append(
                {
                    "chunk_id": cid,
                    "score": entry["score"],
                    "bbox": patch.bbox,
                    "page_index": patch.page_index,
                    "patch_idx": patch.patch_idx,
                }
            )

        return results[:top_k]

    def top_patch_bbox(
        self,
        query_text: str,
        page_index: int | None = None,
    ) -> dict[str, float] | None:
        """Return the bounding box of the single best-scoring patch.

        Returns ``None`` when retrieval is disabled or index is empty.
        The bbox is in normalised [0, 1] coordinates: {x0, y0, x1, y1}.
        Intended to be passed directly to B3's ``parse_vlm_box`` /
        ``verify_box_against_chunks``.
        """
        hits = self.retrieve_patch(query_text, page_index=page_index, top_k=1)
        if not hits:
            return None
        return hits[0]["bbox"]

    @property
    def is_colpali_available(self) -> bool:
        """True when colpali_engine is loaded (not fallback mode)."""
        return self._retriever.is_available

    @property
    def total_patches_indexed(self) -> int:
        """Total number of patches indexed across all pages."""
        return self._retriever.index_size

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _slice_into_patches(
        self,
        page_img_bytes: bytes,
        page_index: int,
        doc_id: str,
    ) -> list[VisualPatch]:
        """Divide page image into nrows×ncols patches.

        Uses PIL when available; falls back to a synthetic patch generator
        when PIL is not installed (for minimal-dep environments). Each patch
        is returned as a fresh PNG-like blob.
        """
        try:
            return self._slice_with_pil(page_img_bytes, page_index, doc_id)
        except Exception:
            return self._slice_synthetic(page_img_bytes, page_index, doc_id)

    def _slice_with_pil(
        self,
        page_img_bytes: bytes,
        page_index: int,
        doc_id: str,
    ) -> list[VisualPatch]:
        """Patch extraction using Pillow (preferred path)."""
        from PIL import Image  # type: ignore[import]

        img = Image.open(io.BytesIO(page_img_bytes)).convert("RGB")
        w, h = img.size

        patches: list[VisualPatch] = []
        for row in range(self.nrows):
            for col in range(self.ncols):
                x0_px = int(col * w / self.ncols)
                y0_px = int(row * h / self.nrows)
                x1_px = int((col + 1) * w / self.ncols)
                y1_px = int((row + 1) * h / self.nrows)

                crop = img.crop((x0_px, y0_px, x1_px, y1_px))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                patch_bytes = buf.getvalue()

                bbox: dict[str, float] = {
                    "x0": x0_px / w,
                    "y0": y0_px / h,
                    "x1": x1_px / w,
                    "y1": y1_px / h,
                }
                patch_idx = row * self.ncols + col
                patches.append(
                    VisualPatch(
                        patch_idx=patch_idx,
                        row=row,
                        col=col,
                        page_index=page_index,
                        bbox=bbox,
                        img_bytes=patch_bytes,
                    )
                )
        return patches

    def _slice_synthetic(
        self,
        page_img_bytes: bytes,
        page_index: int,
        doc_id: str,
    ) -> list[VisualPatch]:
        """Synthetic patch fallback: generate virtual patches using hash seeds.

        When PIL is not available, we cannot crop the image. Instead we create
        virtual patches by combining the page bytes with a position seed. The
        patch image bytes are a minimal valid PNG derived from the position seed.
        This preserves the structural API contract (bbox, chunk_id, etc.) while
        working in minimal CI environments.
        """
        import struct
        import zlib

        def _seed_png(seed_int: int, fingerprint: bytes) -> bytes:
            """Minimal 1×1 PNG with colour derived from seed."""
            r = (seed_int * 73) & 0xFF
            g = (seed_int * 137) & 0xFF
            b = (seed_int * 211) & 0xFF
            header = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
            ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            raw_row = bytes([0, r, g, b])
            compressed = zlib.compress(raw_row)
            idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
            idat = (
                struct.pack(">I", len(compressed))
                + b"IDAT"
                + compressed
                + struct.pack(">I", idat_crc)
            )
            iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            # Mix in a per-page fingerprint so patches are distinct across pages
            return header + ihdr + idat + iend + fingerprint[:8]

        page_fingerprint = page_img_bytes[:16]
        patches: list[VisualPatch] = []
        for row in range(self.nrows):
            for col in range(self.ncols):
                bbox: dict[str, float] = {
                    "x0": col / self.ncols,
                    "y0": row / self.nrows,
                    "x1": (col + 1) / self.ncols,
                    "y1": (row + 1) / self.nrows,
                }
                patch_idx = row * self.ncols + col
                patch_bytes = _seed_png(patch_idx, page_fingerprint)
                patches.append(
                    VisualPatch(
                        patch_idx=patch_idx,
                        row=row,
                        col=col,
                        page_index=page_index,
                        bbox=bbox,
                        img_bytes=patch_bytes,
                    )
                )
        return patches


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_visual_retriever(
    *,
    enabled: bool = True,
    nrows: int = 4,
    ncols: int = 4,
    device: str = "cpu",
) -> VisualPatchRetriever:
    """Create a :class:`VisualPatchRetriever` with sensible defaults.

    Parameters
    ----------
    enabled:
        Set ``False`` for text-native documents to skip visual retrieval entirely.
    nrows, ncols:
        Patch grid dimensions.
    device:
        PyTorch device for ColPali.  Ignored in fallback mode.
    """
    return VisualPatchRetriever(
        nrows=nrows,
        ncols=ncols,
        enabled=enabled,
        device=device,
    )
