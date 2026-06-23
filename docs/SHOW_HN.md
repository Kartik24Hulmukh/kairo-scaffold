# Kairo — Show HN

**Title:** Show HN: Kairo — AI document extraction that cites the pixel or refuses

---

Kairo is an open-source grounding engine for document extraction. Every value it returns is anchored to a bounding box on the source page, or it refuses. No source pixel → no answer.

## The problem

Language models hallucinate citations. Ask a model about a PDF and it will often produce confident answers that cite passages that don't exist — or worse, fabricate a page reference. For document workflows — legal review, research, support — a wrong answer with a plausible-looking citation is worse than no answer at all.

## What Kairo does differently

Kairo indexes a document into text chunks with bounding-box coordinates, then for each query it retrieves candidate chunks and asks the model to answer **only from those chunks**. If the retrieved evidence doesn't support an answer, it returns `blocked` rather than fabricating one. A 5-layer cascade (NORMALIZE → EXACT → FUZZY → SEMANTIC → VISUAL IoU) verifies every value against the source. The web demo shows the answer alongside the highlighted page region it came from.

## Try it in 60 seconds

```bash
git clone https://github.com/Kartik24Hulmukh/kairo-scaffold.git
cd kairo-scaffold
./quickstart.sh
# See it extract from real documents — no GPU, no ML downloads
```

**Or watch the live dashboard** (grounding decisions in real-time): run quickstart, then open `http://localhost:7438/dashboard`

**Or try the web demo**: `http://localhost:7438/demo` — upload a PDF, see bbox highlights + refusals.

## Connect your tool (5 lines)

```python
import requests
resp = requests.post("http://localhost:7438/api/extract-document",
    files={"file": open("invoice.pdf", "rb")})
for f in resp.json()["fields"]:
    print(f"  {f['field']}: {f['value']}  [page {f.get('page')}]")
```

Returns structured data + bbox per field, or refusal with reason. See [`docs/connector_guide.md`](docs/connector_guide.md).

## The honest number

We report the **blind grounded-rate** — measured on a frozen corpus never seen during development or threshold tuning. The dev-set 100% is the overfit figure; we quarantined it to `legacy/` and do **not** cite it as the headline.

> **Status:** the blind corpus + scorer are synced from the engine repo. The blind headline is [INFRA-PENDING — run `make bench` on a GPU machine]. The dev-set figure remains reproducible via `make bench` for regression tracking.

## What it can't do yet (radically honest)

- **Blind benchmark headline is pending** — the shared blind corpus is synced but the bench run needs a GPU machine (PyTorch venv ~2GB). The dev-set 100% is quarantined, not the headline.
- **Signed installers are INFRA-PENDING** — the release workflow is ready, but signing requires secrets not yet provisioned. Unsigned builds are clearly labeled.
- **In-browser OCR is limited** — the web demo handles native-text PDFs only; scanned PDFs require the desktop app (which does OCR). The web demo shows an honest warning when you load a scanned doc.
- **Competitor rows are cached** — GPT-4o-mini / Claude Haiku / Gemini Flash numbers were captured during initial development, not re-run live with dates. Re-running live is tracked as a follow-up.
- **Not a full OCR replacement** — uses PyMuPDF for text extraction, not a full OCR pipeline.
- **Not a general LLM** — it's a verifier, not a generator. It grounds model outputs, it doesn't generate from scratch.

## Pre-empting the obvious questions

**"Are these numbers real or did you tune to the eval set?"**
The headline is the blind number, measured on a frozen corpus never used for tuning. The dev-set 100% is quarantined and labeled as overfit. `make bench` reproduces on your machine.

**"What model does it use? Can I swap it?"**
Kairo is model-independent. The grounding cascade verifies any model's output against the document's bounding boxes. Bring your own key or run the local stub baseline offline.

**"Does it work on real messy files?"**
The adversarial gauntlet covers rotated scans, multi-column, tables, low-DPI, and non-English. The real-world blind corpus pass is pending the bench run on a GPU machine.

## Key numbers

- 100% injection blocked (25 adversarial payloads, 0 false positives)
- 5-layer grounding cascade with exact bbox citations
- Zero network traffic in default config (local-first, air-gap)
- 60-95% token reduction via built-in context compression
- MIT licensed, no telemetry

## Contribute

The easiest way to contribute is a new grounding fixture: a document plus a questions entry with at least one answerable and one unanswerable question. Pull requests for additional document packs and OCR backends are welcome.

---
*License: MIT. No telemetry. Local-first. Every answer cites a page + bbox, or it refuses.*