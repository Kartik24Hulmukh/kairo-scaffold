# Kairo Scaffold — Show HN Draft

**Kairo Scaffold** is an MIT, local-first document Q&A + field-extraction tool that **refuses to answer when it cannot cite a specific passage with an exact page + bounding box.** Same verified grounding core as Kairo Phantom v2.2, slimmed for a 5-minute clone-and-verify.

---

## The pitch (one paragraph)

Language models hallucinate citations. Ask a model about a PDF and it will often produce confident answers that cite passages that don't exist — or worse, fabricate a page reference. For legal review, research, and support, a wrong answer with a plausible citation is worse than no answer. Kairo indexes a document into text chunks with bounding-box coordinates, retrieves candidates per query, and asks the model to answer **only from those chunks**. If the evidence doesn't support an answer, it returns `blocked` rather than fabricating one. The desktop overlay shows the answer alongside the highlighted page region it came from.

## Try it in 5 minutes

```bash
git clone https://github.com/Kartik24Hulmukh/kairo-scaffold.git
cd kairo-scaffold
make build        # sets up the Python sidecar venv
make bench        # reproduce the benchmark on your machine
```

Then run a grounded extraction:
```bash
cargo run --bin kairo -- run fixtures/golden/sample_invoice_01.txt --pack invoice
```

The sidecar exposes a REST API at http://127.0.0.1:7438. Everything runs locally — no cloud, no telemetry.

## The honest number (why we report blind, not tuned)

We report the **blind grounded-rate** — measured on a frozen corpus never seen during development or threshold tuning. A tuned dev-set number (100% on ~19 golden fixtures) is the overfit figure; we quarantined it to `legacy/` and do **not** cite it as the headline. The blind number is the one you can re-run and trust.

> **Status:** the blind corpus + scorer are built in Kairo Phantom v2.2 (source of truth) and pulled verbatim into scaffold. Until the shared corpus is copied in and `sha256sum -c CHECKSUMS.sha256` passes, the blind headline is PENDING. The dev-set figure remains reproducible via `make bench` for regression tracking. See `CORE_SYNC.md` for the one-way sync flow.

## Reproduce it yourself

`make bench` is deterministic and stamped with date + commit. Run it twice on a clean checkout — you should get the same numbers. The benchmark harness lives in `bench/`; fixtures are version-controlled in `fixtures/golden/`.

## What it can't do yet (radically honest)

- **Blind benchmark headline is pending** — the shared blind corpus is not yet copied in from phantom. The dev-set 100% is quarantined, not the headline.
- **Signed installers are INFRA-PENDING** — the Tauri release workflow is ready, but signing requires secrets (`TAURI_SIGNING_PRIVATE_KEY`, Apple certificate) not yet provisioned. Unsigned builds are clearly labeled.
- **In-browser OCR is limited** — the web demo handles native-text PDFs only; scanned PDFs require the desktop app (which does OCR). The web demo shows an honest warning when you load a scanned doc.
- **Competitor rows are cached** — GPT-4o-mini / Claude Haiku / Gemini Flash numbers were captured during initial development, not re-run live with dates. Re-running live is tracked as a follow-up.
- **Real-world corpus pass is pending** — the blind grounded-rate on messy real-world documents + a failure taxonomy will be published once the shared corpus is in.

## Pre-empting the obvious questions

**"Are these numbers real or did you tune to the eval set?"**
The headline is the blind number, measured on a frozen corpus never used for tuning. The dev-set 100% is quarantined and labeled as overfit. `make bench` reproduces on your machine.

**"What model does it use? Can I swap it?"**
Kairo is model-independent. The grounding cascade (NORMALIZE → EXACT → FUZZY → SEMANTIC → VISUAL IoU → VGVA → BLOCK) verifies any model's output against the document's bounding boxes. Bring your own key (OpenAI / Anthropic / Google) or run the local stub baseline offline.

**"Does it work on real messy files, not just golden fixtures?"**
The real-world corpus pass (scans, rotation, multi-column, handwriting, multi-currency) is pending the shared blind corpus. The adversarial gauntlet (`fixtures/adversarial/`) already covers rotated scans, multi-column, tables, low-DPI, and non-English.

## Contribute

The easiest way to contribute is a new grounding fixture: a plain-text or PDF document plus a `bench/questions.json` entry with at least one answerable and one unanswerable question. Pull requests for additional document packs (`packs/`) and OCR backends (`kernel/sidecar/ingest/`) are welcome.

---
*License: MIT. No telemetry. Local-first. The grounding core is synced one-way from [Kairo Phantom v2.2](https://github.com/Kartik24Hulmukh) — see `CORE_SYNC.md`.*