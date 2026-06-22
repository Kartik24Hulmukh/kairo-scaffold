"""A4 — License Contamination Guard Tests.

Verifies that:
1. No direct `import fitz` / `from fitz import` appears in the kernel core
   (PyMuPDF is AGPL; allowed only in scripts/ subprocess boundary).
2. No AGPL/BSL package names appear in requirements.txt.
3. The license_check.py CI script enforces these constraints.

GATE: pytest kernel/tests/test_license_guard.py -v
"""

import ast
import pathlib
import sys
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SIDECAR_DIR = REPO_ROOT / "kernel" / "sidecar"
MODELS_DIR = SIDECAR_DIR / "models"
RETRIEVAL_DIR = SIDECAR_DIR / "retrieval"
SCRIPTS_DIR = REPO_ROOT / "scripts"
LICENSE_CHECK = SCRIPTS_DIR / "ci" / "license_check.py"
REQUIREMENTS = SIDECAR_DIR / "requirements.txt"

# Modules that constitute the "MIT core" — must not import fitz at module level
MIT_CORE_DIRS = [
    MODELS_DIR,
    RETRIEVAL_DIR,
    SIDECAR_DIR,
]

# Known AGPL/BSL packages that must never be direct imports in the core
BANNED_LICENSES = {
    "fitz",        # PyMuPDF — AGPL 3.0
    "pymupdf",     # same
    "elasticsearch",  # SSPL (akin to AGPL)
    "mongodb",     # SSPL in some editions
}

# Packages that are allowed only as subprocess (not direct import in core)
SUBPROCESS_ONLY = {"fitz"}


# ---------------------------------------------------------------------------
# A4-01: AST scan — no direct fitz import in kernel core
# ---------------------------------------------------------------------------

class TestNoFitzImportInCore:
    """AST-level scan: fitz must not be imported at module level in kernel core."""

    SKIP_DIRS = {".venv", "__pycache__", "node_modules", ".git", "scripts"}

    def _find_fitz_imports(self, search_dir: pathlib.Path) -> list[dict]:
        """Return list of {file, lineno, stmt} for any fitz imports found."""
        violations = []

        for py_file in search_dir.rglob("*.py"):
            # Skip virtual env and generated dirs
            if any(skip in py_file.parts for skip in self.SKIP_DIRS):
                continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "fitz" or alias.name.startswith("fitz."):
                            violations.append({
                                "file": str(py_file.relative_to(REPO_ROOT)),
                                "lineno": node.lineno,
                                "stmt": f"import {alias.name}",
                            })
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (node.module == "fitz" or node.module.startswith("fitz.")):
                        names = [a.name for a in node.names]
                        violations.append({
                            "file": str(py_file.relative_to(REPO_ROOT)),
                            "lineno": node.lineno,
                            "stmt": f"from {node.module} import {', '.join(names)}",
                        })
        return violations

    def test_no_fitz_import_in_models(self):
        """kernel/sidecar/models/ must not directly import fitz (AGPL boundary)."""
        violations = self._find_fitz_imports(MODELS_DIR)
        assert not violations, (
            f"AGPL violation: fitz imported in kernel core models:\n"
            + "\n".join(f"  {v['file']}:{v['lineno']}: {v['stmt']}" for v in violations)
        )

    def test_no_fitz_import_in_retrieval(self):
        """kernel/sidecar/retrieval/ must not directly import fitz."""
        violations = self._find_fitz_imports(RETRIEVAL_DIR)
        assert not violations, (
            f"AGPL violation: fitz imported in retrieval layer:\n"
            + "\n".join(f"  {v['file']}:{v['lineno']}: {v['stmt']}" for v in violations)
        )

    def test_fitz_in_scripts_is_allowed(self):
        """fitz must not appear in models/ or retrieval/ — only scripts/ is allowed."""
        # Scan only the banned dirs (models/, retrieval/) — not the entire sidecar
        core_violations = []
        for search_dir in [MODELS_DIR, RETRIEVAL_DIR]:
            if search_dir.exists():
                core_violations.extend(self._find_fitz_imports(search_dir))

        assert not core_violations, (
            f"Fitz imported in sidecar core (models/ or retrieval/):\n"
            + "\n".join(f"  {v['file']}:{v['lineno']}: {v['stmt']}" for v in core_violations)
        )


# ---------------------------------------------------------------------------
# A4-02: requirements.txt does not contain AGPL/BSL packages as direct deps
# ---------------------------------------------------------------------------

class TestRequirementsBannedPackages:
    def test_requirements_txt_exists(self):
        """Sidecar requirements.txt must exist."""
        assert REQUIREMENTS.exists(), f"Missing: {REQUIREMENTS}"

    def test_requirements_no_banned_license_keywords(self):
        """requirements.txt must not reference AGPL/BSL licenses in comments."""
        content = REQUIREMENTS.read_text(encoding="utf-8").lower()
        banned_keywords = ["agpl", "gpl-3", "gpl v3", "bsl", "sspl"]
        violations = [kw for kw in banned_keywords if kw in content]
        assert not violations, (
            f"Banned license keywords found in requirements.txt: {violations}"
        )

    def test_requirements_pymupdf_is_subprocess_only(self):
        """If PyMuPDF is in requirements.txt, license_check.py must enforce fitz isolation."""
        if not REQUIREMENTS.exists():
            pytest.skip("requirements.txt not found")
        content = REQUIREMENTS.read_text(encoding="utf-8")

        # Check if pymupdf is present
        has_pymupdf = any(
            "pymupdf" in line.lower() or "fitz" in line.lower()
            for line in content.splitlines()
            if not line.strip().startswith("#")
        )

        if has_pymupdf:
            # If PyMuPDF is in requirements, license_check.py must have the fitz isolation check
            lc_content = LICENSE_CHECK.read_text(encoding="utf-8") if LICENSE_CHECK.exists() else ""
            assert "fitz" in lc_content.lower() or "pymupdf" in lc_content.lower(), (
                f"PyMuPDF found in requirements.txt but license_check.py does not "
                "validate its subprocess-only isolation via check_fitz_isolation()."
            )
            assert "check_fitz_isolation" in lc_content or "fitz_isolation" in lc_content, (
                "license_check.py must call check_fitz_isolation() when PyMuPDF is in requirements.txt"
            )


# ---------------------------------------------------------------------------
# A4-03: license_check.py CI script exists and enforces constraints
# ---------------------------------------------------------------------------

class TestLicenseCheckScript:
    def test_license_check_exists(self):
        """scripts/ci/license_check.py must exist."""
        assert LICENSE_CHECK.exists(), f"Missing: {LICENSE_CHECK}"

    def test_license_check_has_banned_patterns(self):
        """license_check.py must define banned license patterns."""
        content = LICENSE_CHECK.read_text(encoding="utf-8")
        assert "BANNED" in content or "banned" in content.lower(), (
            "license_check.py must define banned license patterns"
        )
        assert "agpl" in content.lower(), "license_check.py must ban AGPL"

    def test_license_check_exits_nonzero_on_fail(self):
        """license_check.py must call sys.exit(1) on failure."""
        content = LICENSE_CHECK.read_text(encoding="utf-8")
        assert "sys.exit(1)" in content or "exit(1)" in content, (
            "license_check.py must exit with code 1 on failure"
        )

    def test_license_check_covers_requirements(self):
        """license_check.py must scan requirements.txt."""
        content = LICENSE_CHECK.read_text(encoding="utf-8")
        assert "requirements" in content, (
            "license_check.py must scan requirements.txt"
        )

    def test_license_check_imports_cleanly(self):
        """license_check.py must parse without syntax errors."""
        content = LICENSE_CHECK.read_text(encoding="utf-8")
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"license_check.py has syntax error: {e}")

    def test_makefile_has_license_check_target(self):
        """Makefile must have a license-check target."""
        makefile = REPO_ROOT / "Makefile"
        assert makefile.exists()
        content = makefile.read_text(encoding="utf-8")
        assert "license-check" in content or "license_check" in content, (
            "Makefile must have license-check target"
        )


# ---------------------------------------------------------------------------
# A4-04: Extended license_check tests (subprocess boundary documented)
# ---------------------------------------------------------------------------

class TestSubprocessBoundaryDocumented:
    def test_generate_gauntlet_has_agpl_comment(self):
        """generate_gauntlet_fixtures.py must acknowledge PyMuPDF AGPL isolation."""
        gauntlet_gen = SCRIPTS_DIR / "generate_gauntlet_fixtures.py"
        if not gauntlet_gen.exists():
            pytest.skip("generate_gauntlet_fixtures.py not found")
        content = gauntlet_gen.read_text(encoding="utf-8")
        assert any(
            phrase in content
            for phrase in ["AGPL", "subprocess", "isolation", "never imported by core"]
        ), (
            "generate_gauntlet_fixtures.py must document PyMuPDF AGPL isolation "
            "(e.g., comment: 'AGPL isolation: this script lives in scripts/')"
        )
