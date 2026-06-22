"""Regression checker for Kairo grounding benchmark.

Compares the last two entries in bench/history.jsonl.
Fails (exits non-zero) if faithfulness regresses by more than 5 percentage points.
"""

import json
import pathlib
import sys

def main() -> None:
    history_file = pathlib.Path(__file__).parent / "history.jsonl"
    if not history_file.exists():
        print("No history file found. Regression check: PASS")
        sys.exit(0)

    lines = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception as e:
                    print(f"[WARN] Failed to parse history line: {e}")

    if len(lines) < 2:
        print(f"Only {len(lines)} entry/entries in history. Regression check: PASS")
        sys.exit(0)

    # Compare last two entries
    prev_entry = lines[-2]
    curr_entry = lines[-1]

    # In older schema, the key was 'grounded_answer_rate'.
    # In new schema, we have 'faithfulness' and 'grounded_answer_rate' (alias).
    # We will read 'faithfulness' first, falling back to 'grounded_answer_rate'.
    prev_report = prev_entry.get("report", {})
    curr_report = curr_entry.get("report", {})

    prev_faithfulness = prev_report.get("faithfulness", prev_report.get("grounded_answer_rate", 0.0))
    curr_faithfulness = curr_report.get("faithfulness", curr_report.get("grounded_answer_rate", 0.0))

    prev_sha = prev_report.get("git_sha", "unknown")
    curr_sha = curr_report.get("git_sha", "unknown")

    diff = prev_faithfulness - curr_faithfulness
    print(f"Regression Check:")
    print(f"  Previous commit ({prev_sha}) Faithfulness : {prev_faithfulness:.2%}")
    print(f"  Current commit ({curr_sha}) Faithfulness  : {curr_faithfulness:.2%}")
    print(f"  Difference                              : {diff:+.2%}")

    if diff > 0.05:
        print(f"GATE FAIL: Faithfulness regressed by {diff:.2%} (> 5pp)", file=sys.stderr)
        sys.exit(1)

    print("GATE PASS: No regression detected (<= 5pp drop)")
    sys.exit(0)

if __name__ == "__main__":
    main()
