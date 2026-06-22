# Kairo Phantom -- Show HN

**Kairo Phantom** is an open-source document Q&A sidecar that refuses to answer when it cannot cite a specific passage.

## The problem

Language models hallucinate citations. Ask a model about a PDF and it will often produce confident-sounding answers that cite passages that do not exist. For document workflows -- legal review, research, support -- a wrong answer with a plausible-looking citation is worse than no answer at all.

## What Kairo does differently

Kairo indexes a document into text chunks with bounding-box coordinates, then for each query it retrieves candidate chunks and asks the model to answer *only from those chunks*. If the retrieved evidence does not support an answer, the sidecar returns `"blocked"` rather than fabricating one. The UI overlay displays the answer alongside the highlighted page region it came from.

## Try it

`cargo run --bin kairo -- run fixtures/golden/placeholder.txt --pack generic`

This starts the grounding pipeline on the bundled placeholder fixture. The sidecar exposes a REST API at http://127.0.0.1:7438.

## Benchmark

Results are tracked in a public leaderboard at bench/leaderboard.html, updated by running `make bench`. The current committed result shows Kairo at 0% Citation-Hallucination Rate on the golden fixture set. Optional baselines (GPT-4o-mini, Claude Haiku, Gemini Flash) can be added by exporting the relevant API key.

Metrics:
- **GAR** -- Grounded-Answer Rate: fraction of answerable questions that received a grounded answer
- **CHR** -- Citation-Hallucination Rate: fraction of non-refused answers with a fabricated citation
- **RC** -- Refusal-Correctness: fraction of unanswerable questions correctly refused

## Contribute

The easiest way to contribute is a new grounding fixture: a plain-text or PDF document plus a questions.json entry with at least one answerable and one unanswerable question. See bench/questions.json for the schema.

Pull requests for additional document pack types (packs/) and OCR backends (kernel/sidecar/ingest/) are also welcome.

---
*License: MIT. No telemetry.*
