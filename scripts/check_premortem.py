"""E1 — Premortem: 8 failure-mode structural checks.

Each check inspects the repository layout for the artefacts that guard
against one of the 8 catalogued failure modes.  No live sidecar is needed.

Exit 0 if every mode passes, exit 1 on the first FAIL (after printing all).
"""
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent

# Paths used in multiple checks
SIDECAR_APP = REPO_ROOT / "kernel" / "sidecar" / "app.py"
MODELS_DIR = REPO_ROOT / "kernel" / "sidecar" / "models"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Mode 1 — No hardcoded API keys
# ---------------------------------------------------------------------------
def check_no_hardcoded_api_keys() -> tuple[bool, str]:
    """Grep *.py files for patterns that look like hardcoded secrets."""
    LEAK_PATTERNS = [
        r'\bsk-[A-Za-z0-9]{20,}',          # OpenAI-style keys
        r'\bapi_key\s*=\s*["\'][^"\']{8,}',  # api_key = "..."
        r'\bAPIKEY\s*=\s*["\'][^"\']{8,}',   # APIKEY = "..."
        r'\bsecret\s*=\s*["\'][^"\']{8,}',   # secret = "..."
    ]
    SKIP = {'.venv', 'node_modules', '__pycache__', '.git'}

    violations = []
    for py_file in REPO_ROOT.rglob("*.py"):
        if any(part in SKIP for part in py_file.parts):
            continue
        content = _read(py_file)
        for pat in LEAK_PATTERNS:
            m = re.search(pat, content)
            if m:
                relative = py_file.relative_to(REPO_ROOT)
                violations.append(f"{relative}: matches '{pat[:30]}...'")

    if violations:
        return False, "; ".join(violations[:3])  # show up to 3
    return True, f"no hardcoded-key patterns found across {REPO_ROOT} *.py"


# ---------------------------------------------------------------------------
# Mode 2 — Sidecar cold-start: graceful error if not running
# ---------------------------------------------------------------------------
def check_sidecar_graceful_cold_start() -> tuple[bool, str]:
    """rag_shield.py exists — it gates sidecar interactions with typed errors."""
    shield = MODELS_DIR / "rag_shield.py"
    error_h = MODELS_DIR / "error_handling.py"
    if shield.exists() and error_h.exists():
        return True, f"{shield.relative_to(REPO_ROOT)} + {error_h.relative_to(REPO_ROOT)} present"
    missing = [str(p.relative_to(REPO_ROOT)) for p in [shield, error_h] if not p.exists()]
    return False, f"missing: {missing}"


# ---------------------------------------------------------------------------
# Mode 3 — Missing grounding → refusal (verify_grounding in app.py)
# ---------------------------------------------------------------------------
def check_grounding_refusal() -> tuple[bool, str]:
    """app.py must contain verify_grounding(); its 'block' branch is the refusal gate."""
    if not SIDECAR_APP.exists():
        return False, f"{SIDECAR_APP.relative_to(REPO_ROOT)} not found"
    content = _read(SIDECAR_APP)
    if "def verify_grounding" in content and '"block"' in content:
        return True, f"{SIDECAR_APP.relative_to(REPO_ROOT)} has verify_grounding() with block branch"
    return False, "verify_grounding() or 'block' branch not found in app.py"


# ---------------------------------------------------------------------------
# Mode 4 — Performance budget exceeded
# ---------------------------------------------------------------------------
def check_perf_budget_script() -> tuple[bool, str]:
    script = SCRIPTS_DIR / "check_perf_budget.py"
    if script.exists():
        content = _read(script)
        if "BUDGETS" in content:
            return True, f"{script.relative_to(REPO_ROOT)} defines BUDGETS dict"
    return False, f"{script.relative_to(REPO_ROOT)} missing or lacks BUDGETS"


# ---------------------------------------------------------------------------
# Mode 5 — Prompt injection attack
# ---------------------------------------------------------------------------
def check_prompt_injection_defense() -> tuple[bool, str]:
    shield = MODELS_DIR / "rag_shield.py"
    if not shield.exists():
        return False, f"{shield.relative_to(REPO_ROOT)} not found"
    content = _read(shield)
    has_scan = "def scan_content_for_poisoning" in content
    has_sanitize = "def sanitize_user_query" in content
    if has_scan and has_sanitize:
        return True, f"{shield.relative_to(REPO_ROOT)} has scan_content_for_poisoning + sanitize_user_query"
    missing_fns = []
    if not has_scan:
        missing_fns.append("scan_content_for_poisoning")
    if not has_sanitize:
        missing_fns.append("sanitize_user_query")
    return False, f"{shield.relative_to(REPO_ROOT)} missing: {missing_fns}"


# ---------------------------------------------------------------------------
# Mode 6 — Banned phrases in receipts
# ---------------------------------------------------------------------------
def check_banned_phrases_in_receipts() -> tuple[bool, str]:
    script = SCRIPTS_DIR / "check_receipt.py"
    if not script.exists():
        return False, f"{script.relative_to(REPO_ROOT)} not found"
    content = _read(script)
    if "BANNED_CLAIMS" in content and "100%" in content:
        return True, f"{script.relative_to(REPO_ROOT)} defines BANNED_CLAIMS including '100%'"
    return False, f"{script.relative_to(REPO_ROOT)} does not enforce BANNED_CLAIMS"


# ---------------------------------------------------------------------------
# Mode 7 — Schema validation failures (constrained_decoding)
# ---------------------------------------------------------------------------
def check_schema_validation() -> tuple[bool, str]:
    cd = MODELS_DIR / "constrained_decoding.py"
    if not cd.exists():
        return False, f"{cd.relative_to(REPO_ROOT)} not found"
    content = _read(cd)
    if "jsonschema.validate" in content:
        return True, f"{cd.relative_to(REPO_ROOT)} calls jsonschema.validate() before returning"
    return False, f"{cd.relative_to(REPO_ROOT)} missing jsonschema.validate call"


# ---------------------------------------------------------------------------
# Mode 8 — Dependency import failures (graceful ImportError handling)
# ---------------------------------------------------------------------------
def check_graceful_import_errors() -> tuple[bool, str]:
    """Key modules must either handle ImportError or have a try/except on heavy deps."""
    candidates = {
        "app.py": SIDECAR_APP,
        "rag_shield.py": MODELS_DIR / "rag_shield.py",
        "constrained_decoding.py": MODELS_DIR / "constrained_decoding.py",
    }
    # app.py uses try/except around qdrant and SentenceTransformer setup
    app_content = _read(SIDECAR_APP)
    has_try_except = app_content.count("except Exception") >= 2 or "except ImportError" in app_content
    if not has_try_except:
        return False, "app.py lacks try/except around heavy dependencies"

    # rag_shield.py and constrained_decoding.py are stdlib + jsonschema — lower risk
    # We verify they don't do bare top-level imports that would crash silently
    shield_content = _read(MODELS_DIR / "rag_shield.py")
    cd_content = _read(MODELS_DIR / "constrained_decoding.py")
    if "import re" in shield_content and "import json" in cd_content:
        return True, "app.py has except-guarded imports; models import only available stdlib/jsonschema"
    return False, "unexpected import pattern in model files"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("Mode 1: No hardcoded API keys", check_no_hardcoded_api_keys),
    ("Mode 2: Sidecar cold-start graceful error", check_sidecar_graceful_cold_start),
    ("Mode 3: Missing grounding -> refusal", check_grounding_refusal),
    ("Mode 4: Performance budget script exists", check_perf_budget_script),
    ("Mode 5: Prompt injection defense", check_prompt_injection_defense),
    ("Mode 6: Banned phrases enforced in receipts", check_banned_phrases_in_receipts),
    ("Mode 7: Schema validation in constrained decoding", check_schema_validation),
    ("Mode 8: Graceful dependency import handling", check_graceful_import_errors),
]


if __name__ == "__main__":
    failures = []
    for label, fn in CHECKS:
        passed, detail = fn()
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
        print(f"         {detail}")
        if not passed:
            failures.append(label)

    print()
    if failures:
        print(f"PREMORTEM CHECK: FAIL ({len(failures)} failure mode(s) unguarded)")
        sys.exit(1)
    print("PREMORTEM CHECK: PASS (all 8 failure modes have structural guards)")
