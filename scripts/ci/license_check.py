"""CI License Compliance Check — SPEC §5 gate.

Enforces that no AGPL/BSL/GPL packages are directly imported in the MIT-core.

PyMuPDF (fitz) boundary rule:
  - PyMuPDF (AGPL 3.0) IS allowed in requirements.txt because it is used ONLY
    in scripts/ (generate_gauntlet_fixtures.py, generate_non_english_fixture.py)
    via subprocess isolation. It is NEVER imported in kernel/sidecar/models/ or
    kernel/sidecar/retrieval/.
  - This check enforces the boundary with an AST scan.

BANNED: Any package in requirements.txt with an AGPL/BSL/GPL comment.
BANNED: Direct `import fitz` in kernel/sidecar/models/ or kernel/sidecar/retrieval/.
ALLOWED: `import fitz` in scripts/ (subprocess boundary).
"""

import ast
import os
import pathlib
import re
import sys

BANNED_LICENSE_KEYWORDS = [
    r"\bagpl\b",
    r"\bgpl\b",
    r"\bbsl\b",
    r"\bcopyleft\b",
    r"license-agpl",
    r"license-gpl",
    r"license-bsl"
]

# Core directories where fitz (PyMuPDF AGPL) must NEVER be imported at module level
FITZ_BANNED_DIRS = [
    "kernel/sidecar/models",
    "kernel/sidecar/retrieval",
]


def check_requirements(filepath):
    """Scan requirements.txt for illegal dependencies.

    Note: PyMuPDF (fitz) is allowed in requirements.txt (scripts/ subprocess use).
    This check only fails on explicit AGPL/BSL license keywords in comments.
    """
    if not os.path.exists(filepath):
        return True

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line_clean = line.strip().lower()
        if not line_clean or line_clean.startswith('#'):
            continue
        # Skip pymupdf — it is allowed as a scripts/ subprocess dependency
        if "pymupdf" in line_clean or "fitz" in line_clean:
            continue
        # Look for comments indicating license or package names containing banned keywords
        for pattern in BANNED_LICENSE_KEYWORDS:
            if re.search(pattern, line_clean):
                print(f"[FAIL] Banned license pattern detected in requirements: {line_clean}")
                return False
    return True


def check_cargo(filepath):
    """Scan Cargo.toml for illegal dependencies."""
    if not os.path.exists(filepath):
        return True

    with open(filepath, 'r') as f:
        content = f.read().lower()

    # Search dependencies block
    for pattern in BANNED_LICENSE_KEYWORDS:
        if re.search(pattern, content):
            print(f"[FAIL] Banned license pattern detected in Cargo.toml")
            return False
    return True


def check_fitz_isolation(repo_root: str = ".") -> bool:
    """A4: AST scan — fitz must not be imported in kernel core (only scripts/).

    PyMuPDF is AGPL 3.0. It may only be used in scripts/ via subprocess.
    This function scans FITZ_BANNED_DIRS for any direct import fitz statements.
    """
    repo_path = pathlib.Path(repo_root)
    violations = []
    skip_dirs = {".venv", "__pycache__", "node_modules", ".git"}

    for banned_dir in FITZ_BANNED_DIRS:
        search_dir = repo_path / banned_dir
        if not search_dir.exists():
            continue
        for py_file in search_dir.rglob("*.py"):
            if any(skip in py_file.parts for skip in skip_dirs):
                continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "fitz" or alias.name.startswith("fitz."):
                            violations.append(
                                f"{py_file.relative_to(repo_path)}:{node.lineno}: "
                                f"import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (node.module == "fitz" or node.module.startswith("fitz.")):
                        names = [a.name for a in node.names]
                        violations.append(
                            f"{py_file.relative_to(repo_path)}:{node.lineno}: "
                            f"from {node.module} import {', '.join(names)}"
                        )

    if violations:
        for v in violations:
            print(f"[FAIL] AGPL boundary violation (fitz in core): {v}")
        return False

    print("[PASS] fitz isolation check: PyMuPDF not imported in kernel core (scripts/ only).")
    return True


def main():
    print("Running CI License Compliance Check...")
    ok = True
    ok = ok and check_requirements("kernel/sidecar/requirements.txt")
    ok = ok and check_cargo("kernel/core/Cargo.toml")
    ok = ok and check_fitz_isolation(".")

    if not ok:
        print("CI License Check: FAILED")
        sys.exit(1)

    print("CI License Check: PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()
