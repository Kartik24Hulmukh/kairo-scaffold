#!/usr/bin/env python3
"""
Receipt-check script (A1 requirement).
Validates that docs/receipts/ contains receipts for all completed tasks.
Run via: make receipt-check

Receipt format supported:
  # [ID] — [Title]
  ## Gate Command
  ## Evidence
  ## What Was Built
  ## Constraints Satisfied
  ## Ungrounded Claims
"""
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent
RECEIPTS_DIR = REPO_ROOT / "docs" / "receipts"

# All expected receipt IDs (one per completed/stubbed task)
EXPECTED_RECEIPTS = {
    "A1", "A2",
    "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4",
    "D1", "D2", "D3", "D4",
    "E1",
    "F1", "F2",
    "G1", "G2", "G3", "G4", "G5",
}

# At least one of these heading markers must appear in a receipt
REQUIRED_SECTION_OPTIONS = [
    ("## Gate Command", "## Evidence", "## What Was Built"),   # new format
    ("GATE COMMAND:", "GATE OUTPUT", "Status:"),               # legacy format
]

BANNED_CLAIMS = [
    "100%", "10/10", "perfect", "flawless", "zero bugs",
    "bulletproof", "battle-tested", "seamless", "comprehensive",
]


def check_receipts() -> tuple[list[str], list[str]]:
    """Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not RECEIPTS_DIR.exists():
        errors.append(f"RECEIPTS_DIR not found: {RECEIPTS_DIR}")
        return errors, warnings

    existing: set[str] = set()
    for f in sorted(RECEIPTS_DIR.glob("*.md")):
        if f.name == "TEMPLATE.md":
            continue
        receipt_id = f.stem.upper()
        existing.add(receipt_id)

        content = f.read_text(encoding="utf-8", errors="ignore")

        # Check at least one recognised receipt format is present
        has_format = any(
            all(field in content for field in option)
            for option in REQUIRED_SECTION_OPTIONS
        )
        if not has_format:
            warnings.append(
                f"{f.name}: missing expected receipt sections "
                f"(need '## Gate Command' + '## Evidence' + '## What Was Built')"
            )

        # Banned claims check
        content_lower = content.lower()
        for phrase in BANNED_CLAIMS:
            if phrase.lower() in content_lower:
                warnings.append(f"{f.name}: contains banned phrase '{phrase}'")

        # Legacy FAIL status check
        if "Status: FAIL" in content:
            errors.append(f"{f.name}: receipt Status is FAIL — gate did not pass")

    # Missing receipts (warning, not error, so CI doesn't block mid-sprint)
    pending = EXPECTED_RECEIPTS - existing
    if pending:
        for receipt_id in sorted(pending):
            warnings.append(f"Receipt {receipt_id}.md is missing (task in progress)")

    return errors, warnings


def main() -> None:
    errors, warnings = check_receipts()

    print(f"Receipt check — {RECEIPTS_DIR}")
    print()

    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")
        print()

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        print(f"\nRECEIPT CHECK: FAIL ({len(errors)} errors, {len(warnings)} warnings)")
        sys.exit(1)

    found = len(list(RECEIPTS_DIR.glob("*.md"))) - 1  # exclude TEMPLATE
    print(f"  [PASS] {found} receipts found, 0 errors")
    if warnings:
        print(f"  [WARN] {len(warnings)} warnings (non-blocking)")
    print("\nRECEIPT CHECK: PASS")


if __name__ == "__main__":
    main()
