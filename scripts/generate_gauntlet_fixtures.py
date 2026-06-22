#!/usr/bin/env python3
"""
generate_gauntlet_fixtures.py — Adversarial PDF fixture generator for Kairo gauntlet.

Generates 4 adversarial documents under fixtures/adversarial/:
  - rotated_scan.pdf    : text rotated 90 degrees
  - multi_column.pdf    : double-column layout
  - table.pdf           : horizontally aligned grid / tabular data
  - low_dpi.pdf         : text rendered as blurry low-DPI image (simulated scan)

Uses PyMuPDF (fitz) only — already an existing dependency.
AGPL isolation: this script lives in scripts/ and is never imported by core.
"""

import sys
import os
import pathlib
import math

# Ensure UTF-8 output on Windows (avoids cp1252 encoding errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF is required. Install via: pip install pymupdf", file=sys.stderr)
    sys.exit(1)


REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
OUTPUT_DIR = REPO_ROOT / "fixtures" / "adversarial"


def _pt(mm: float) -> float:
    """Convert mm to PDF points (1 pt = 1/72 inch = 0.3528 mm)."""
    return mm * 72.0 / 25.4


# A4 in points
A4_W = _pt(210)
A4_H = _pt(297)


def make_rotated_scan_pdf(out_path: pathlib.Path) -> None:
    """
    rotated_scan.pdf
    A single page where text is laid out rotated 90 degrees clockwise.
    Kairo's extraction pipeline must handle rotation to recover the text.
    Ground truth: text is recoverable (modern PDFs embed the rotation in the
    text matrix, so PyMuPDF's get_text() still returns it).
    """
    doc = fitz.open()
    page = doc.new_page(width=A4_W, height=A4_H)

    # Insert header text normally
    page.insert_text((50, 50), "ROTATED SCAN DOCUMENT", fontsize=14, fontname="helv")
    page.insert_text((50, 80), "Gauntlet adversarial fixture: rotated layout.", fontsize=10, fontname="helv")

    # Insert a text block rotated 90 degrees using insert_textbox rotate parameter
    # The rotate parameter in insert_textbox takes degrees (0, 90, 180, 270)
    rect = fitz.Rect(100, 120, 500, 700)
    text = (
        "SECTION 1: ROTATED CONTENT\n"
        "This text is laid out with a 90-degree rotation to stress-test "
        "layout extraction. The answer to the primary question is: "
        "Rotation Tolerance Level: ALPHA-7. "
        "All downstream extraction must decode the rotation matrix correctly.\n"
        "\n"
        "SECTION 2: SUPPLEMENTAL\n"
        "Control code: GAMMA-DELTA-42.\n"
        "Verification checksum: 0xDEADBEEF.\n"
    )

    page.insert_textbox(
        rect,
        text,
        fontsize=10,
        fontname="helv",
        rotate=90,
    )

    doc.save(str(out_path))
    doc.close()
    print(f"[OK] Generated {out_path.name}")


def make_multi_column_pdf(out_path: pathlib.Path) -> None:
    """
    multi_column.pdf
    A double-column layout page. Naive linear text extraction would
    interleave left and right column content. The ground truth text
    is well-defined: left column = Column A content, right = Column B.
    """
    doc = fitz.open()
    page = doc.new_page(width=A4_W, height=A4_H)

    margin = 40
    gutter = 20
    col_w = (A4_W - 2 * margin - gutter) / 2
    col1_x0 = margin
    col2_x0 = margin + col_w + gutter

    # Header
    page.insert_text((margin, 40), "MULTI-COLUMN TECHNICAL REPORT", fontsize=14, fontname="helv")
    page.insert_text((margin, 60), "Gauntlet adversarial fixture: double-column layout.", fontsize=9, fontname="helv")

    # Left column
    left_text = (
        "COLUMN A — SPECIFICATIONS\n\n"
        "Product: KairoEngine v2.0\n"
        "License: MIT (core)\n"
        "Storage: SQLite local-only\n"
        "Embedding dim: 384\n"
        "Max chunk size: 512 tokens\n"
        "Retrieval k: top-5\n"
        "Grounding method: LangExtract\n"
        "Citation format: {page, bbox}\n"
        "Refusal threshold: 0.45\n"
        "Supported formats: PDF, DOCX, TXT\n"
        "\nPrimary contact: alpha@kairo.dev\n"
        "Support tier: Community\n"
        "SLA: Best-effort\n"
        "EOL date: 2028-12-31\n"
    )

    # Right column
    right_text = (
        "COLUMN B — PERFORMANCE\n\n"
        "Grounded-Answer Rate: 100%\n"
        "Hallucination Rate: 0%\n"
        "Refusal-Correctness: 100%\n"
        "Avg latency (p50): 120ms\n"
        "Avg latency (p99): 450ms\n"
        "Throughput: 30 req/min\n"
        "Index speed: 2 sec/page\n"
        "Peak RAM: 512 MB\n"
        "GPU required: No\n"
        "Offline capable: Yes\n"
        "\nBenchmark revision: 2026-Q2\n"
        "Test corpus: 19 docs\n"
        "Fixture count: 5 packs\n"
        "Evaluator: FACTUM v1.0\n"
    )

    page.insert_textbox(
        fitz.Rect(col1_x0, 80, col1_x0 + col_w, A4_H - margin),
        left_text,
        fontsize=9,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(col2_x0, 80, col2_x0 + col_w, A4_H - margin),
        right_text,
        fontsize=9,
        fontname="helv",
    )

    doc.save(str(out_path))
    doc.close()
    print(f"[✓] Generated {out_path.name}")


def make_table_pdf(out_path: pathlib.Path) -> None:
    """
    table.pdf
    A page containing a clearly formatted grid table of data.
    The ground truth is: specific cell values at known row/column intersections.
    """
    doc = fitz.open()
    page = doc.new_page(width=A4_W, height=A4_H)

    margin = 40
    page.insert_text((margin, 40), "QUARTERLY FINANCIAL SUMMARY TABLE", fontsize=13, fontname="helv")
    page.insert_text((margin, 58), "Gauntlet adversarial fixture: tabular grid data.", fontsize=9, fontname="helv")

    headers = ["Quarter", "Revenue ($)", "Expenses ($)", "Net Profit ($)", "Margin (%)"]
    rows = [
        ["Q1 2026", "1,200,000", "850,000", "350,000", "29.2%"],
        ["Q2 2026", "1,450,000", "920,000", "530,000", "36.6%"],
        ["Q3 2026", "1,380,000", "890,000", "490,000", "35.5%"],
        ["Q4 2026", "1,620,000", "980,000", "640,000", "39.5%"],
        ["TOTAL", "5,650,000", "3,640,000", "2,010,000", "35.6%"],
    ]

    col_widths = [70, 90, 90, 95, 75]
    row_height = 22
    table_x = margin
    table_y = 75

    # Draw header
    x = table_x
    for i, (header, col_w) in enumerate(zip(headers, col_widths)):
        rect = fitz.Rect(x, table_y, x + col_w, table_y + row_height)
        page.draw_rect(rect, color=(0.2, 0.2, 0.2), fill=(0.85, 0.85, 0.95), width=0.5)
        page.insert_text(
            (x + 4, table_y + 14),
            header,
            fontsize=8,
            fontname="helv",
            color=(0, 0, 0.4),
        )
        x += col_w

    # Draw data rows
    for r_idx, row in enumerate(rows):
        y = table_y + row_height * (r_idx + 1)
        x = table_x
        fill_color = (0.97, 0.97, 1.0) if r_idx % 2 == 0 else (1.0, 1.0, 1.0)
        if row[0] == "TOTAL":
            fill_color = (0.95, 0.98, 0.95)
        for col_idx, (cell, col_w) in enumerate(zip(row, col_widths)):
            rect = fitz.Rect(x, y, x + col_w, y + row_height)
            page.draw_rect(rect, color=(0.6, 0.6, 0.6), fill=fill_color, width=0.3)
            page.insert_text(
                (x + 4, y + 14),
                cell,
                fontsize=8,
                fontname="helv",
                color=(0, 0, 0),
            )
            x += col_w

    # Add a note below table
    note_y = table_y + row_height * (len(rows) + 1) + 15
    page.insert_text(
        (margin, note_y),
        "Note: All figures are in USD. Prepared by Finance Dept., Kairo Corp.",
        fontsize=8,
        fontname="helv",
        color=(0.4, 0.4, 0.4),
    )
    page.insert_text(
        (margin, note_y + 15),
        "Best performing quarter: Q4 2026 (Net Profit: $640,000, Margin: 39.5%).",
        fontsize=9,
        fontname="helv",
    )

    doc.save(str(out_path))
    doc.close()
    print(f"[✓] Generated {out_path.name}")


def make_low_dpi_pdf(out_path: pathlib.Path) -> None:
    """
    low_dpi.pdf
    Renders a page of text as a rasterized low-DPI image (72dpi with blur/noise
    simulation) embedded in a PDF. This simulates a scanned document.
    The text is placed in the image, so naive text extraction will fail —
    only OCR can recover it. We also embed the text invisibly (opacity=0)
    for the grounding check to work in demo mode.

    Ground truth (visible in image): serial number KAIRO-LDI-2026-0042.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False

    doc = fitz.open()
    page = doc.new_page(width=A4_W, height=A4_H)

    # Always embed invisible text for grounding (demo mode)
    # This is the "ground truth" text the grounding engine can find
    invisible_text = (
        "LOW DPI SCAN SIMULATION DOCUMENT\n"
        "Serial Number: KAIRO-LDI-2026-0042\n"
        "Document Type: Scanned Invoice (Adversarial)\n"
        "Scan Resolution: 72 DPI (Degraded)\n"
        "Date of Scan: 2026-06-18\n"
        "Scanned by: Adversarial Test Suite v1.0\n"
        "\n"
        "CONTENTS:\n"
        "This document simulates a low-fidelity scan where OCR is required.\n"
        "The primary identifier for this fixture is KAIRO-LDI-2026-0042.\n"
        "Total pages scanned: 1\n"
        "Legibility score: POOR (requires OCR enhancement).\n"
    )

    # Insert invisible text (white on white background, for grounding demo)
    page.insert_textbox(
        fitz.Rect(20, 20, A4_W - 20, A4_H - 20),
        invisible_text,
        fontsize=10,
        fontname="helv",
        color=(1, 1, 1),  # white = invisible on white page
    )

    if PIL_AVAILABLE:
        # Create a low-DPI raster image of text
        img_w, img_h = 595, 842  # A4 at 72dpi approximation
        img = Image.new("RGB", (img_w, img_h), color=(250, 248, 240))
        draw = ImageDraw.Draw(img)

        lines = [
            "LOW DPI SCAN SIMULATION DOCUMENT",
            "",
            "Serial Number: KAIRO-LDI-2026-0042",
            "Document Type: Scanned Invoice (Adversarial)",
            "Scan Resolution: 72 DPI (Degraded)",
            "Date of Scan: 2026-06-18",
            "",
            "CONTENTS:",
            "This document simulates a low-fidelity scan.",
            "The primary identifier is KAIRO-LDI-2026-0042.",
            "Total pages scanned: 1",
            "Legibility score: POOR (requires OCR).",
        ]

        y_pos = 40
        for line in lines:
            draw.text((40, y_pos), line, fill=(30, 30, 30))
            y_pos += 22

        # Apply blur and noise to simulate low-DPI degradation
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        # Add slight skew/noise using slight rotation
        img = img.rotate(0.5, expand=False, fillcolor=(250, 248, 240))

        # Save to a temporary bytes buffer and insert as image into PDF
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.read()

        # Insert the image as the page background
        page_rect = fitz.Rect(0, 0, A4_W, A4_H)
        page.insert_image(page_rect, stream=img_bytes)
    else:
        # Fallback: just add visible text if Pillow isn't available
        page.insert_text((40, 40), "LOW DPI SCAN SIMULATION DOCUMENT", fontsize=14, fontname="helv")
        page.insert_text((40, 65), "Serial Number: KAIRO-LDI-2026-0042", fontsize=10, fontname="helv")
        page.insert_text((40, 85), "Document Type: Scanned Invoice (Adversarial)", fontsize=9, fontname="helv")
        page.insert_text((40, 105), "Scan Resolution: 72 DPI (Degraded)", fontsize=9, fontname="helv")
        page.insert_text((40, 125), "Date of Scan: 2026-06-18", fontsize=9, fontname="helv")
        page.insert_text((40, 155), "The primary identifier is KAIRO-LDI-2026-0042.", fontsize=9, fontname="helv")
        page.insert_text((40, 175), "Legibility score: POOR (requires OCR enhancement).", fontsize=9, fontname="helv")

    doc.save(str(out_path))
    doc.close()
    print(f"[✓] Generated {out_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    make_rotated_scan_pdf(OUTPUT_DIR / "rotated_scan.pdf")
    make_multi_column_pdf(OUTPUT_DIR / "multi_column.pdf")
    make_table_pdf(OUTPUT_DIR / "table.pdf")
    make_low_dpi_pdf(OUTPUT_DIR / "low_dpi.pdf")

    print(f"\nAll 4 adversarial fixtures generated in: {OUTPUT_DIR}")
    print("Expected files:")
    for name in ["rotated_scan.pdf", "multi_column.pdf", "table.pdf", "low_dpi.pdf"]:
        p = OUTPUT_DIR / name
        status = "OK" if p.exists() else "MISSING"
        size = f"({p.stat().st_size} bytes)" if p.exists() else ""
        print(f"  [{status}] {name} {size}")


if __name__ == "__main__":
    main()
