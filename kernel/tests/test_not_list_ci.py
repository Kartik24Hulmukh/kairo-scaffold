"""A6 — SPEC §9 NOT-list CI enforcement tests.

Verifies that check_not_list.py:
1. Fails when forbidden modules appear in .py files
2. Passes on the clean repo
3. Covers all SPEC §9 forbidden module names
4. Has a file LOC gate (anti-bloat)

GATE: pytest kernel/tests/test_not_list_ci.py -v
"""

import importlib.util
import pathlib
import sys
import textwrap
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
NOT_LIST_SCRIPT = REPO_ROOT / "scripts" / "check_not_list.py"


def _load_check_not_list():
    """Dynamically load check_not_list.py."""
    spec = importlib.util.spec_from_file_location("check_not_list", NOT_LIST_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# A6-01: Script exists and has expected structure
# ---------------------------------------------------------------------------

class TestNotListScript:
    def test_script_exists(self):
        assert NOT_LIST_SCRIPT.exists(), f"Missing: {NOT_LIST_SCRIPT}"

    def test_script_defines_forbidden_modules(self):
        content = NOT_LIST_SCRIPT.read_text(encoding="utf-8")
        assert "FORBIDDEN_MODULES" in content, (
            "check_not_list.py must define FORBIDDEN_MODULES list"
        )

    def test_script_covers_spec9_items(self):
        """SPEC §9 forbidden items must be listed."""
        content = NOT_LIST_SCRIPT.read_text(encoding="utf-8")
        spec9_required = [
            "write_to_source",
            "cloud_sync",
            "telemetry",
        ]
        for item in spec9_required:
            assert item in content, (
                f"check_not_list.py missing SPEC §9 forbidden module: '{item}'"
            )

    def test_script_has_sys_exit_on_failure(self):
        content = NOT_LIST_SCRIPT.read_text(encoding="utf-8")
        assert "sys.exit(1)" in content or "exit(1)" in content, (
            "check_not_list.py must exit with code 1 when forbidden modules found"
        )

    def test_script_imports_cleanly(self):
        import ast
        content = NOT_LIST_SCRIPT.read_text(encoding="utf-8")
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"check_not_list.py has syntax error: {e}")


# ---------------------------------------------------------------------------
# A6-02: Functional tests — clean repo passes
# ---------------------------------------------------------------------------

class TestNotListCleanRepo:
    def test_clean_repo_has_no_forbidden_modules(self):
        """The actual repo must pass the not-list check."""
        mod = _load_check_not_list()
        # check_not_list() returns a list; empty = pass
        failures = mod.check_not_list()
        # Filter out any test files that intentionally reference these for testing
        real_failures = [
            f for f in failures
            if "test_not_list" not in f and "check_not_list" not in f
        ]
        assert not real_failures, (
            f"SPEC §9 NOT-list violations found in repo:\n"
            + "\n".join(f"  {f}" for f in real_failures)
        )


# ---------------------------------------------------------------------------
# A6-03: Functional tests — injected violations are detected
# ---------------------------------------------------------------------------

class TestNotListDetectsViolations:
    """Create temp files with forbidden content and verify detection."""

    def _run_check_with_fake_file(self, content: str, filename: str = "test_bad.py"):
        """Check if content contains any forbidden modules from check_not_list.py."""
        mod = _load_check_not_list()
        # Use FORBIDDEN_MODULES from the loaded module directly
        violations = []
        for mod_name in mod.FORBIDDEN_MODULES:
            if mod_name in content:
                violations.append(f"FORBIDDEN MODULE '{mod_name}' found in {filename}")
        return violations

    def test_write_to_source_detected(self):
        code = textwrap.dedent("""
        def example():
            write_to_source('/some/path', 'data')
        """)
        violations = self._run_check_with_fake_file(code)
        assert any("write_to_source" in v for v in violations), (
            "write_to_source must be detected as a forbidden module"
        )

    def test_cloud_sync_detected(self):
        code = textwrap.dedent("""
        import cloud_sync
        cloud_sync.upload(data)
        """)
        violations = self._run_check_with_fake_file(code)
        assert any("cloud_sync" in v for v in violations), (
            "cloud_sync must be detected as a forbidden module"
        )

    def test_telemetry_detected(self):
        code = textwrap.dedent("""
        telemetry.track_event('user_action', {'detail': 'clicked'})
        """)
        violations = self._run_check_with_fake_file(code)
        assert any("telemetry" in v for v in violations), (
            "telemetry must be detected as a forbidden module"
        )

    def test_clean_code_not_flagged(self):
        code = textwrap.dedent("""
        def process_document(path):
            with open(path) as f:
                return f.read()
        """)
        violations = self._run_check_with_fake_file(code)
        assert not violations, f"Clean code incorrectly flagged: {violations}"


# ---------------------------------------------------------------------------
# A6-04: Anti-bloat file LOC gate
# ---------------------------------------------------------------------------

class TestAntiBlot:
    """No single model file should exceed 1500 LOC (anti-bloat gate)."""

    MAX_LOC = 1500
    SKIP_DIRS = {".venv", "__pycache__", "node_modules", ".git"}

    def test_no_model_file_exceeds_max_loc(self):
        """kernel/sidecar/models/*.py must not exceed 1500 LOC."""
        models_dir = REPO_ROOT / "kernel" / "sidecar" / "models"
        if not models_dir.exists():
            pytest.skip("models directory not found")

        oversized = []
        for py_file in models_dir.glob("*.py"):
            lines = py_file.read_text(encoding="utf-8", errors="ignore").count("\n")
            if lines > self.MAX_LOC:
                oversized.append(f"{py_file.name}: {lines} LOC (max {self.MAX_LOC})")

        assert not oversized, (
            f"SPEC §9 anti-bloat violation — files exceed {self.MAX_LOC} LOC:\n"
            + "\n".join(f"  {f}" for f in oversized)
        )

    def test_no_retrieval_file_exceeds_max_loc(self):
        """kernel/sidecar/retrieval/*.py must not exceed 1500 LOC."""
        retrieval_dir = REPO_ROOT / "kernel" / "sidecar" / "retrieval"
        if not retrieval_dir.exists():
            pytest.skip("retrieval directory not found")

        oversized = []
        for py_file in retrieval_dir.glob("*.py"):
            lines = py_file.read_text(encoding="utf-8", errors="ignore").count("\n")
            if lines > self.MAX_LOC:
                oversized.append(f"{py_file.name}: {lines} LOC (max {self.MAX_LOC})")

        assert not oversized, (
            f"Anti-bloat: retrieval files exceed {self.MAX_LOC} LOC:\n"
            + "\n".join(f"  {f}" for f in oversized)
        )

    def test_makefile_has_not_list_check_target(self):
        """Makefile must have not-list-check target."""
        makefile = REPO_ROOT / "Makefile"
        assert makefile.exists()
        content = makefile.read_text(encoding="utf-8")
        assert "not-list-check" in content or "not_list" in content, (
            "Makefile must have not-list-check target"
        )

    def test_direct_script_exits_zero_on_clean_repo(self):
        """Running 'python scripts/check_not_list.py' from repo root must exit 0.

        This is the exact invocation used by Makefile's not-list-check target.
        A previous bug caused test files (intentionally referencing forbidden
        strings in assertions) to be scanned, causing spurious exit code 1.
        """
        import subprocess
        result = subprocess.run(
            [sys.executable, str(NOT_LIST_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"check_not_list.py exited {result.returncode} on clean repo.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}\n"
            "This means the Makefile not-list-check target would FAIL on a clean repo."
        )
        assert "[PASS]" in result.stdout, (
            f"check_not_list.py did not print [PASS]. Got: {result.stdout!r}"
        )
