# BENCHMARKS.md — Kairo Scaffold Grounding Benchmark

> **Anti-bluff:** all numbers below are measured live by `make bench` on a frozen, content-addressed corpus. The headline is the **blind** number — measured on a corpus never seen during development or threshold tuning. The dev-set number (100% on ~19 golden fixtures) is quarantined to `legacy/` and is NOT the headline.

**Corpus:** blind-v1 (11 docs, 77 answerable + 4 unanswerable fields = 81 total)
**Packs:** invoice (4 docs), contract (3), generic (2), paper (2)
**Source:** [Kairo Phantom v2.2](https://github.com/KairoPhantom/Kairo-Phantom) — shared oracle, verified by `sha256sum -c CHECKSUMS.sha256`
**Scorer:** `bench/score.py` (shared verbatim with phantom)

---

## Headline — Blind Corpus (v1)

> **Why we report blind, not tuned.** The blind corpus was never used to tune thresholds (θ/φ/ψ), prompts, packs, or code. A skeptic can clone this repo, run `make bench`, and get the same number. The dev-set 100% is the overfit figure — it lives in `legacy/` for regression tracking only.

| Metric | Blind (v1) | Dev-set (golden, quarantined) |
| :--- | :---: | :---: |
| **Grounded-Answer Rate** | [INFRA-PENDING — run `make bench` on GPU machine] | 100.00% (overfit, quarantined) |
| **False-Refusal Rate** | [INFRA-PENDING] | 0.00% (overfit) |
| **Refusal-Correctness (Unanswerable)** | [INFRA-PENDING] | 100.00% (overfit) |
| **Hallucinated-Bbox Blocked Rate** | [INFRA-PENDING] | 100.00% (overfit) |

> **INFRA-PENDING:** The blind benchmark requires a full PyTorch venv (sentence-transformers, docling, torch+CUDA) for model inference. This sandbox cannot install the ~2 GB torch wheels (tmpfs disk limit). Run on a machine with ≥4 GB free disk + GPU (or CPU fallback):
> ```bash
> make build   # installs requirements.lock into kernel/sidecar/.venv
> make bench   # runs bench/run_bench.py → prints dev + blind columns
> ```
> Then paste the numbers into this table. The scorer (`bench/score.py`) and corpus (`bench/corpus/blind/v1/`) are ready — only the inference runtime is missing.

---

## Per-Pack Breakdown (Blind v1)

| Pack | Docs | Answerable | Unanswerable | Grounded Rate | False-Refusal | Refusal-Correct |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| invoice | 4 | 34 | 2 | [PENDING] | [PENDING] | [PENDING] |
| contract | 3 | 19 | 2 | [PENDING] | [PENDING] | [PENDING] |
| generic | 2 | 8 | 0 | [PENDING] | [PENDING] | [PENDING] |
| paper | 2 | 16 | 0 | [PENDING] | [PENDING] | [PENDING] |
| **Total** | **11** | **77** | **4** | **[PENDING]** | **[PENDING]** | **[PENDING]** |

---

## Competitor Comparison

Competitor rows are **cached** (captured during initial development on the dev-set, not re-run live on blind). Re-running live with dates is tracked as a follow-up.

| System / Model | Grounded Rate | False-Refusal | Refusal-Correct | Source |
| :--- | :---: | :---: | :---: | :--- |
| **Kairo (Local, blind)** | **[PENDING]** | **[PENDING]** | **[PENDING]** | `make bench` (blind-v1) |
| Kairo (Local, dev-set) | 100.00% | 0.00% | 100.00% | cached (overfit, quarantined) |
| GPT-4o-mini (BYO-key) | 84.62% | — | 75.00% | cached (dev-set, initial dev) |
| Claude Haiku (BYO-key) | 80.77% | — | 66.67% | cached (dev-set, initial dev) |
| Gemini Flash (BYO-key) | 76.92% | — | 58.33% | cached (dev-set, initial dev) |
| Stub/Offline baseline | 0.00% | 100.00% | 100.00% | `make bench` |

---

## Reproducibility

```bash
# 1. Verify the corpus is byte-identical to phantom's
cd bench/corpus/blind/v1 && sha256sum -c CHECKSUMS.sha256

# 2. Run the benchmark (deterministic + dated)
make bench
# Output: "Kairo Scaffold benchmark — date: YYYY-MM-DD HH:MM TZ | commit: <sha>"
#         followed by dev + blind columns

# 3. Run twice on a clean checkout — numbers must match
```

The bench output is stamped with date + commit (`bench/run_bench.py::_bench_stamp()`). Two clean-machine runs must produce identical numbers.

---

## Model Cards & Hardware Requirements

| Model | Role | VRAM (GPU) | CPU Fallback | Notes |
| :--- | :--- | :---: | :---: | :--- |
| sentence-transformers (all-MiniLM-L6-v2) | Semantic retrieval (SEMANTIC stage, φ0.86) | ~90 MB | Yes (slow) | Used in vector store for chunk retrieval |
| ColPali (visual retrieval) | Visual patch retrieval (VISUAL IoU ψ0.5) | ~2 GB | No (GPU only) | Optional; enabled per-document for scanned/visual docs |
| Docling | PDF/DOCX parsing + OCR | ~500 MB | Yes (slow) | Stateless sidecar ingestion |
| Local LLM (Ollama/llama.cpp) | Tier-1/Tier-2 inference (localhost:4000) | varies | Yes | BYO model; offline stub available for testing |
| OpenAI GPT-4o-mini | Tier-2 inference (BYO-key) | n/a (cloud) | n/a | Optional; requires OPENAI_API_KEY |
| Anthropic Claude Haiku | Tier-2 inference (BYO-key) | n/a (cloud) | n/a | Optional; requires ANTHROPIC_API_KEY |
| Google Gemini Flash | Tier-2 inference (BYO-key) | n/a (cloud) | n/a | Optional; requires GOOGLE_API_KEY |

**Offline mode:** `KAIRO_OFFLINE=1` or no gateway → deterministic stub responses. Zero network traffic. Used for `make bench` baseline and CI.

**Hardware for blind bench:**
- **Minimum (CPU):** 8 GB RAM, 4 GB free disk (torch CPU wheels + sentence-transformers). ColPali disabled. ~10× slower.
- **Recommended (GPU):** 8 GB VRAM, 4 GB free disk. ColPali enabled for visual retrieval stage. Full cascade.
- **This sandbox:** cannot install torch wheels (tmpfs 2 GB limit). Marked INFRA-PENDING.
