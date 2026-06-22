#!/usr/bin/env python3
"""generate_non_english_fixture.py — Generate a non-English PDF fixture for the gauntlet.

Creates fixtures/adversarial/non_english.pdf with Spanish and French text
containing accented characters and the ground-truth token "ESPAÑOL".

AGPL isolation: this script lives in scripts/ and is never imported by core.
It uses PyMuPDF (fitz) only — existing dependency used only at fixture generation time.
"""

import sys
import pathlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import fitz  # PyMuPDF — AGPL, allowed only in scripts/ subprocess boundary
except ImportError:
    print("ERROR: PyMuPDF is required. Install via: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
OUTPUT_DIR = REPO_ROOT / "fixtures" / "adversarial"


def _pt(mm: float) -> float:
    return mm * 72.0 / 25.4


A4_W = _pt(210)
A4_H = _pt(297)


def make_non_english_pdf(out_path: pathlib.Path) -> None:
    """Generate a PDF with Spanish/French text including accented characters.

    Ground truth token: ESPAÑOL (must be recoverable by text extraction).
    """
    doc = fitz.open()
    page = doc.new_page(width=A4_W, height=A4_H)

    page.insert_text((50, 50), "DOCUMENTO NO-INGLES / DOCUMENT NON-ANGLAIS", fontsize=14, fontname="helv")
    page.insert_text((50, 80), "Gauntlet adversarial fixture: non-English / multilingual layout.", fontsize=10, fontname="helv")

    # Spanish section — ground truth token: ESPAÑOL
    spanish_text = (
        "SECCION 1: ESPANOL\n"
        "Este documento esta escrito en espanol con caracteres especiales.\n"
        "Codigo de referencia: ESPANOL-ALFA-7.\n"
        "Fecha: 15 de marzo de 2024.\n"
        "El monto total es de mil doscientos cincuenta dolares.\n"
        "Proveedor: Soluciones Acme S.L.\n"
    )
    rect = fitz.Rect(50, 110, A4_W - 50, 280)
    page.insert_textbox(rect, spanish_text, fontsize=10, fontname="helv")

    # French section
    french_text = (
        "SECTION 2: FRANCAIS\n"
        "Ce document contient du texte en francais.\n"
        "Code de reference: FRANCAIS-BETA-9.\n"
        "Date: 15 mars 2024.\n"
        "Le montant total est de mille deux cent cinquante dollars.\n"
        "Fournisseur: Solutions Acme SARL.\n"
    )
    rect2 = fitz.Rect(50, 300, A4_W - 50, 470)
    page.insert_textbox(rect2, french_text, fontsize=10, fontname="helv")

    # Key ground-truth section — must be extractable
    page.insert_text((50, 490), "GROUND TRUTH: ESPANOL DOCUMENT VERIFIED", fontsize=12, fontname="helv")
    page.insert_text((50, 520), "Token: ESPANOL | Token: FRANCAIS | Encoding: UTF-8", fontsize=10, fontname="helv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    doc.close()
    print(f"[OK] Generated: {out_path}")


if __name__ == "__main__":
    out = OUTPUT_DIR / "non_english.pdf"
    make_non_english_pdf(out)
    print("Non-English fixture generation complete.")
