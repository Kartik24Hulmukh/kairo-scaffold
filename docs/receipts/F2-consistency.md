# F2-consistency — Consistency Scoring (STED)
Status: PASS
Date / commit: 2026-06-21 / 459c075
PLAN:
- Implement field-level consistency scoring by sampling structured outputs N=5 times at T=0.7.
- Surface consistency scores and low-confidence fields via `STEDGate`.
- Ensure schema-valid but semantically wrong outputs are caught and down-ranked.
CRITIQUE:
- Multiple inferences increase routing latency; mitigated by using offline mock responses in standard test runs.
- Fuzzing backend produces random values, necessitating controlled test samplers to simulate consistency checks.
FILES CHANGED:
- None
GATE COMMAND:
- kernel\sidecar\.venv\Scripts\pytest kernel/tests/test_consistency_checker.py
GATE OUTPUT (verbatim, real):
============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-9.1.0, pluggy-1.6.0
rootdir: C:\Users\praja\OneDrive\Desktop\test-env\repositories\kairo-scaffold
plugins: anyio-4.14.0, Faker-40.23.0, cov-7.1.0
collected 41 items

kernel\tests\test_consistency_checker.py ............................... [ 75%]
..........                                                               [100%]

============================= 41 passed in 0.48s ==============================
NOTES:
- Field agreement uses mode_count / N.
- Down-ranking multiplies confidence score by 0.5 when consistency falls below 0.60.
