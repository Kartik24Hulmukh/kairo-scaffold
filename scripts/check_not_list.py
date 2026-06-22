"""CI check: SPEC §9 v1 NOT list enforcement.
Fails if out-of-scope modules appear in the codebase.
"""
import pathlib
import sys
import re

FORBIDDEN_MODULES = [
    # v1 NOT list per SPEC §9
    'write_to_source',     # v1 never writes to source apps
    'cloud_sync',          # no document content off-device
    'telemetry',           # no telemetry
    'kube', 'kubernetes',  # not cloud/K8s
]

FORBIDDEN_PHRASES_IN_UI = [
    '1000x',  # never said out loud
]


def check_not_list():
    root = pathlib.Path(".")
    failures = []

    # Check for forbidden modules
    for mod in FORBIDDEN_MODULES:
        for f in root.rglob("*.py"):
            str_f = str(f)
            # Exclude: venv, node_modules, this script, and test files
            # (test files intentionally reference forbidden names to test detection)
            if '.venv' in str_f or 'node_modules' in str_f:
                continue
            if 'check_not_list.py' in str_f or 'test_not_list' in str_f:
                continue
            try:
                content = f.read_text(encoding='utf-8', errors='ignore')
                if mod in content:
                    failures.append(f"FORBIDDEN MODULE '{mod}' found in {f}")
            except Exception:
                pass


    # Check for 1000x in UI/README
    for fname in ['README.md', 'overlay/index.html', 'web-demo/index.html']:
        p = root / fname
        if p.exists():
            content = p.read_text(encoding='utf-8', errors='ignore')
            for phrase in FORBIDDEN_PHRASES_IN_UI:
                if phrase in content:
                    failures.append(f"FORBIDDEN PHRASE '{phrase}' in {fname}")

    return failures


if __name__ == "__main__":
    failures = check_not_list()
    if failures:
        for f in failures:
            print(f"  [FAIL] {f}")
        sys.exit(1)
    print("  [PASS] v1 NOT-list check: no forbidden modules or phrases found")
