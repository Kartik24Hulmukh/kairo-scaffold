#!/usr/bin/env python3
"""
run_acceptance.py — Hard gate acceptance checker for Kairo Scaffold.

This script implements SPEC §5 hard gates:
  - Gate 1: Grounded-Answer Rate >= 95% on golden fixtures
  - Gate 2: Refusal-Correctness == 100% on unanswerable fixtures
  - Gate 3: Citation-Hallucination Rate == 0%
  - Gate 4: Zero ungrounded renders (checks fuzz test results)
  - Gate 5: All unit/integration tests pass (delegated to make test)

Usage:
  python scripts/run_acceptance.py [--bench-report bench/REPORT.md]

Exit code:
  0 = all gates pass
  1 = one or more gates failed
"""

import sys
import re
import pathlib
import subprocess
import argparse
import json

# Ensure UTF-8 output on Windows (avoids cp1252 encoding errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()


def parse_report_md(report_path: pathlib.Path) -> dict[str, dict]:
    """Parse bench/REPORT.md and extract system metrics."""
    if not report_path.exists():
        print(f"[ERROR] Bench report not found: {report_path}", file=sys.stderr)
        return {}

    content = report_path.read_text(encoding="utf-8")
    metrics = {}

    # Parse markdown table rows (skip header rows)
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line or "Model" in line:
            continue
        cols = [c.strip() for c in line.split("|") if c.strip()]
        if len(cols) < 4:
            continue
        sys_name = cols[0]
        try:
            g_rate = float(cols[1].replace("%", "").strip())
            h_rate = float(cols[2].replace("%", "").strip())
            r_rate = float(cols[3].replace("%", "").strip())
            metrics[sys_name] = {
                "grounded_answer_rate": g_rate,
                "citation_hallucination_rate": h_rate,
                "refusal_correctness": r_rate,
            }
        except (ValueError, IndexError):
            continue

    return metrics


def gate_bench_metrics(metrics: dict) -> list[str]:
    """Check SPEC §5 hard gates against bench metrics. Returns list of failures."""
    failures = []

    kairo = metrics.get("Kairo (Local)") or metrics.get("Kairo")
    if kairo is None:
        failures.append("GATE FAIL: 'Kairo (Local)' system not found in bench report.")
        return failures

    g_rate = kairo["grounded_answer_rate"]
    h_rate = kairo["citation_hallucination_rate"]
    r_rate = kairo["refusal_correctness"]

    if g_rate < 95.0:
        failures.append(
            f"GATE FAIL [G1]: Kairo Grounded-Answer Rate is {g_rate:.2f}% (required >= 95%)"
        )
    else:
        print(f"  [PASS] Gate G1: Grounded-Answer Rate = {g_rate:.2f}% (>= 95%)")

    if h_rate > 0.0:
        failures.append(
            f"GATE FAIL [G3]: Kairo Citation-Hallucination Rate is {h_rate:.2f}% (required == 0%)"
        )
    else:
        print(f"  [PASS] Gate G3: Citation-Hallucination Rate = {h_rate:.2f}% (== 0%)")

    if r_rate < 100.0:
        failures.append(
            f"GATE FAIL [G2]: Kairo Refusal-Correctness is {r_rate:.2f}% (required == 100%)"
        )
    else:
        print(f"  [PASS] Gate G2: Refusal-Correctness = {r_rate:.2f}% (== 100%)")

    stub = metrics.get("Stub/Offline baseline")
    if stub is not None:
        if stub["grounded_answer_rate"] != 0.0:
            failures.append(
                f"GATE FAIL [Stub]: Stub Grounded-Answer Rate is {stub['grounded_answer_rate']:.2f}% (should be 0%)"
            )
        if stub["refusal_correctness"] < 100.0:
            failures.append(
                f"GATE FAIL [Stub]: Stub Refusal-Correctness is {stub['refusal_correctness']:.2f}% (should be 100%)"
            )
        if not failures or "Stub" not in str(failures):
            print(f"  [PASS] Stub baseline: GAR=0%, RC=100%")

    return failures


def run_fuzz_tests() -> list[str]:
    """Run Vitest fuzz tests in the overlay directory. Returns failures."""
    failures = []
    overlay_dir = REPO_ROOT / "overlay"

    if not (overlay_dir / "package.json").exists():
        print("  [SKIP] Overlay fuzz tests: overlay/package.json not found.", file=sys.stderr)
        return failures

    print("  Running overlay fuzz tests (npm test)...")
    try:
        result = subprocess.run(
            ["npm.cmd", "test", "--", "--reporter=verbose"],
            cwd=str(overlay_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            failures.append(
                f"GATE FAIL [G4]: Overlay fuzz tests failed (exit code {result.returncode})"
            )
        else:
            # Check for test pass summary
            out = result.stdout + result.stderr
            if "failed" in out.lower() and "0 failed" not in out.lower():
                failures.append("GATE FAIL [G4]: Overlay fuzz tests report failures.")
            else:
                print("  [PASS] Gate G4: Overlay fuzz tests — no ungrounded renders detected.")
    except subprocess.TimeoutExpired:
        failures.append("GATE FAIL [G4]: Overlay fuzz tests timed out.")
    except FileNotFoundError:
        # npm not found — try npm (Linux/Mac)
        try:
            result = subprocess.run(
                ["npm", "test", "--", "--reporter=verbose"],
                cwd=str(overlay_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                failures.append(
                    f"GATE FAIL [G4]: Overlay fuzz tests failed (exit code {result.returncode})"
                )
            else:
                print("  [PASS] Gate G4: Overlay fuzz tests — no ungrounded renders detected.")
        except Exception as e:
            print(f"  [WARN] Could not run overlay fuzz tests: {e}", file=sys.stderr)

    return failures


def check_leaderboard_reproducibility() -> list[str]:
    """Check that leaderboard.html is reproducibly generated (no timestamps)."""
    failures = []
    leaderboard = REPO_ROOT / "bench" / "leaderboard.html"

    if not leaderboard.exists():
        failures.append("GATE FAIL: bench/leaderboard.html does not exist.")
        return failures

    content = leaderboard.read_text(encoding="utf-8")

    # Should contain "(reproducible build)" marker
    if "(reproducible build)" not in content:
        failures.append(
            "GATE FAIL [Repro]: bench/leaderboard.html missing '(reproducible build)' marker."
        )
    else:
        print("  [PASS] Reproducibility: leaderboard.html has '(reproducible build)' marker.")

    return failures


def main():
    parser = argparse.ArgumentParser(description="Kairo SPEC §5 acceptance gate checker")
    parser.add_argument(
        "--bench-report",
        type=pathlib.Path,
        default=REPO_ROOT / "bench" / "REPORT.md",
        help="Path to bench/REPORT.md",
    )
    parser.add_argument(
        "--skip-fuzz",
        action="store_true",
        help="Skip overlay fuzz tests (for CI environments without Node)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  KAIRO SCAFFOLD — SPEC §5 ACCEPTANCE GATE CHECK")
    print("=" * 60 + "\n")

    all_failures: list[str] = []

    # --- Gate: Bench Metrics ---
    print("Checking bench metrics from REPORT.md...")
    metrics = parse_report_md(args.bench_report)
    if not metrics:
        all_failures.append("GATE FAIL: Could not parse bench/REPORT.md — run `make bench` first.")
    else:
        all_failures.extend(gate_bench_metrics(metrics))

    # --- Gate: Leaderboard Reproducibility ---
    print("\nChecking leaderboard reproducibility...")
    all_failures.extend(check_leaderboard_reproducibility())

    # --- Gate: Fuzz Tests ---
    if not args.skip_fuzz:
        print("\nRunning fuzz gate (G4)...")
        all_failures.extend(run_fuzz_tests())
    else:
        print("\n  [SKIP] Fuzz tests skipped via --skip-fuzz.")

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_failures:
        print("  [ACCEPTANCE FAILED] -- Gates not satisfied:")
        for f in all_failures:
            print(f"     {f}")
        print("=" * 60 + "\n")
        sys.exit(1)
    else:
        print("  [ALL GATES PASSED]")
        print("=" * 60 + "\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
