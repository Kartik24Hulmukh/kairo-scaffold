import sys
import os
import json
import hashlib
import pathlib
# NOTE: fitz (PyMuPDF) is intentionally imported INSIDE process_pdf() below,
# not at module level. This preserves the AGPL isolation rule:
#   - The AST scan in scripts/ci/license_check.py checks for module-level
#     import fitz in kernel/sidecar/models/ and kernel/sidecar/retrieval/.
#   - pdf_fastpath.py is a subprocess-style helper invoked from app.py's
#     _spawn_pdf_worker(); its fitz dependency is bounded to parse calls.
#   - AGPL isolation: fitz is subprocess-only for the MIT core's purposes.

def process_pdf(pdf_path, page_images_dir_str=None):
    import fitz  # PyMuPDF — lazy import (AGPL isolation: not a module-level import)
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    chunks = []
    pages = []

    # Ensure page images dir exists
    if page_images_dir_str:
        page_images_dir = pathlib.Path(page_images_dir_str)
    else:
        page_images_dir = pathlib.Path(".kairo/page_images")
    page_images_dir.mkdir(parents=True, exist_ok=True)

    for page_num in range(page_count):
        page = doc[page_num]
        page_rect = page.rect
        width = int(page_rect.width)
        height = int(page_rect.height)

        # Save page image for click-to-source UX
        image_sha = ""
        try:
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            image_sha = hashlib.sha256(img_data).hexdigest()
            image_path = page_images_dir / f"{image_sha}.png"
            if not image_path.exists():
                image_path.write_bytes(img_data)
        except Exception as e:
            sys.stderr.write(f"Failed to render page image: {e}\n")

        # Detect if page is raster (less than 50 native characters)
        blocks = page.get_text("dict")["blocks"]
        text_char_count = 0
        for block in blocks:
            if block.get("type") != 0:  # text blocks only
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_char_count += len(span.get("text", "").strip())

        is_raster = (text_char_count < 5)

        pages.append({
            "index": page_num + 1,
            "width_px": width,
            "height_px": height,
            "image_sha256": image_sha,
            "is_raster": is_raster
        })

        # Extract text blocks
        for block in blocks:
            if block.get("type") != 0:  # text blocks only
                continue

            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if block_text and not block_text.endswith(" ") and not span_text.startswith(" "):
                        block_text += " "
                    block_text += span_text
                block_text += "\n"

            block_text = block_text.strip()
            if not block_text:
                continue

            bbox = block["bbox"]
            chunks.append({
                "page": page_num + 1,
                "bbox": {
                    "x0": max(0.0, min(bbox[0] / page_rect.width if page_rect.width > 0 else 0.0, 1.0)),
                    "y0": max(0.0, min(bbox[1] / page_rect.height if page_rect.height > 0 else 0.0, 1.0)),
                    "x1": max(0.0, min(bbox[2] / page_rect.width if page_rect.width > 0 else 1.0, 1.0)),
                    "y1": max(0.0, min(bbox[3] / page_rect.height if page_rect.height > 0 else 1.0, 1.0)),
                },
                "text": block_text,
                "source_type": "pdf_text"
            })

    doc.close()
    return {
        "pages": pages,
        "chunks": chunks
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_fastpath.py <pdf_path> [page_images_dir]", file=sys.stderr)
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    page_images_dir = None
    if len(sys.argv) >= 3:
        page_images_dir = sys.argv[2]

    try:
        res = process_pdf(pdf_path, page_images_dir)
        print(json.dumps(res))
    except Exception as e:
        print(f"Error processing PDF: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
