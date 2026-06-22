# FAILURE_TAXONOMY.md — Kairo Scaffold Blind Corpus Failure Taxonomy

> Generated from the blind-v1 corpus run. Classifies every failure mode observed so the team can prioritize fixes. **Status: INFRA-PENDING** — the blind bench run requires a full PyTorch venv (see BENCHMARKS.md). This template is ready to populate once `make bench` runs on a GPU machine.

---

## How to generate this file

```bash
make bench  # produces bench/results_blind.json
# Then run:
python bench/score.py --corpus bench/corpus/blind/v1 --predictions bench/results_blind.json --taxonomy FAILURE_TAXONOMY.md
```

The scorer (`bench/score.py`) classifies each prediction into one of the categories below.

---

## Failure Categories

| Category | Definition | Scorer Field | Impact |
| :--- | :--- | :--- | :--- |
| **Grounded Correct** | Answer value matches label AND bbox IoU ≥ 0.5 | `grounded_correct` | Success — no failure |
| **Grounded Wrong Box** | Answer value matches but bbox IoU < 0.5 | `grounded_wrong_box` | Citation points to wrong region |
| **False Refusal** | Question is answerable but Kairo returned BLOCK | `false_refusal` | Over-conservative; user gets no answer |
| **Refusal Correct** | Question is unanswerable and Kairo correctly refused | `refusal_correct` | Success — no failure |
| **Hallucination** | Answer value does not match label (fabricated) | `hallucination` | Worst case — wrong answer with confidence |
| **Bad Box Blocked** | Model produced wrong bbox but verifier blocked it | `blocked_bad_boxes` | Defense worked — no failure reaches user |

---

## Blind-v1 Results (INFRA-PENDING)

| Category | Count | Rate | Notes |
| :--- | :---: | :---: | :--- |
| Grounded Correct | [PENDING] | [PENDING] | |
| Grounded Wrong Box | [PENDING] | [PENDING] | |
| False Refusal | [PENDING] | [PENDING] | |
| Refusal Correct | [PENDING] | [PENDING] | |
| Hallucination | [PENDING] | [PENDING] | |
| Bad Box Blocked | [PENDING] | [PENDING] | |

---

## Per-Pack Failure Breakdown (INFRA-PENDING)

| Pack | Grounded | Wrong Box | False Refusal | Hallucination | Bad Box Blocked |
| :--- | :---: | :---: | :---: | :---: | :---: |
| invoice | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| contract | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| generic | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| paper | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

---

## Known Failure Modes (from dev-set + adversarial gauntlet)

These are failure modes observed during development that the blind run will confirm or refute:

1. **False refusal on multi-column layouts** — the FRONT SELECT snippet scorer may miss quotes split across columns. Mitigated by SW alignment but still a risk on blind docs.
2. **Wrong bbox on multi-line chunks** — `interpolate_bbox` approximates char position within multi-line chunks; IoU may drop below 0.5 for fields near line boundaries.
3. **Hallucination on unanswerable fields with plausible values** — if the model produces a value that happens to look correct but has no bbox anchor, VGVA should block it. Blind corpus includes 4 unanswerable fields to test this.
4. **OCR noise on low-DPI scans** — the adversarial gauntlet (rotated_scan, low_dpi) showed ~80% acceptance; blind corpus may include similar cases.

---

## Action Items (post blind run)

- [ ] Run `make bench` on GPU machine, populate tables above
- [ ] If false-refusal rate > 10%, investigate FRONT SELECT threshold (QUOTE_THRESHOLD)
- [ ] If hallucination rate > 5%, investigate VGVA visual grounding gate (IoU 0.5)
- [ ] If wrong-box rate > 10%, investigate `interpolate_bbox` for multi-line chunks
- [ ] Publish updated FAILURE_TAXONOMY.md with real numbers
