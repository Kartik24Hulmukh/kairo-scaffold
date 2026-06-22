"""
kernel_purity_guard.py — CI guard: Rust core must NOT import fitz/PyMuPDF.

SPEC §1 isolation: All AGPL/BSL/source-available code is confined to the Python
sidecar. The Rust core must remain clean of any dependency on PyMuPDF, fitz,
or any other AGPL/BSL library.

Usage:
    python scripts/ci/kernel_purity_guard.py
    
Exit code:
    0 = clean (no violations)
    1 = violations found
"""
import pathlib
import sys
import re

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()

# Libraries that MUST NOT appear in Rust core (only permitted in Python sidecar)
FORBIDDEN_IN_RUST = [
    "fitz",         # PyMuPDF – AGPL-3.0
    "pymupdf",      # PyMuPDF – AGPL-3.0
    "MuPDF",        # underlying MuPDF – AGPL-3.0
    "docling",      # Docling – BSL-1.0 or AGPL
]

# Directories that form the Rust core (these must be pure)
RUST_CORE_DIRS = [
    REPO_ROOT / "kernel" / "core",   # Rust src/
    REPO_ROOT / "wasm-search-core",  # WASM Rust src/
]

# Directories that are allowed to use these libraries (Python sidecar)
ALLOWED_DIRS_STR = ["kernel/sidecar", "kernel/tests", "packs", "scripts", "bench"]


def check_rust_purity() -> list[str]:
    """Check Rust source files for forbidden AGPL/BSL library imports."""
    violations = []
    
    for rust_dir in RUST_CORE_DIRS:
        if not rust_dir.exists():
            continue
        for f in rust_dir.rglob("*.rs"):
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                for forbidden in FORBIDDEN_IN_RUST:
                    # Match use/extern/mod declarations
                    pattern = rf'(?i)(?:use|extern|mod|import)\s+{re.escape(forbidden)}'
                    if re.search(pattern, content) or forbidden.lower() in content.lower():
                        rel_path = f.relative_to(REPO_ROOT)
                        violations.append(
                            f"VIOLATION: '{forbidden}' found in Rust core file: {rel_path}"
                        )
            except Exception:
                pass
    
    return violations


def check_cargo_toml_purity() -> list[str]:
    """Check Cargo.toml files in core dirs for forbidden deps."""
    violations = []
    
    for rust_dir in RUST_CORE_DIRS:
        if not rust_dir.exists():
            continue
        for cargo in rust_dir.rglob("Cargo.toml"):
            try:
                content = cargo.read_text(encoding="utf-8", errors="ignore")
                for forbidden in FORBIDDEN_IN_RUST:
                    if forbidden.lower() in content.lower():
                        rel_path = cargo.relative_to(REPO_ROOT)
                        violations.append(
                            f"VIOLATION: '{forbidden}' in Cargo.toml: {rel_path}"
                        )
            except Exception:
                pass
    
    return violations


def main():
    print("Kairo Kernel Purity Guard — checking Rust core is AGPL/BSL-free...")
    
    all_violations = []
    all_violations.extend(check_rust_purity())
    all_violations.extend(check_cargo_toml_purity())
    
    if all_violations:
        print("\n[FAIL] Purity violations found:")
        for v in all_violations:
            print(f"  {v}")
        print(
            "\nAGPL/BSL libraries must only be used in kernel/sidecar/, never in Rust core.\n"
        )
        sys.exit(1)
    else:
        print("[PASS] Rust core imports zero AGPL/BSL symbols (fitz/PyMuPDF/docling).")
        sys.exit(0)


if __name__ == "__main__":
    main()
