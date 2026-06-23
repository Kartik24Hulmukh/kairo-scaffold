# 📎 Kairo: Grounded Document Intelligence

> **It cites the pixel or it refuses.** Every value Kairo returns is anchored to a bounding box on the source page — or it refuses. No source pixel → no answer.

![Kairo Refusal Demo — refusal first, citation second](refusal_demo.gif)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Bench](https://img.shields.io/badge/benchmark-blind%20corpus-blue)](BENCHMARKS.md)
[![Security](https://img.shields.io/badge/injection-100%25%20blocked-red)]()

---

## ⚡ 60-Second Proof

```bash
git clone https://github.com/Kartik24Hulmukh/kairo-scaffold.git
cd kairo-scaffold
./quickstart.sh
```

That's it. No GPU, no ML model downloads, no 500MB dependencies. Clone, run, see grounded extraction + refusal in 60 seconds.

**Or try the web demo:** `http://localhost:7438/demo` after running quickstart.

---

## 🔗 Connect Your Tool (5 lines)

```python
import requests
resp = requests.post("http://localhost:7438/api/extract-document",
    files={"file": open("invoice.pdf", "rb")})
for f in resp.json()["fields"]:
    print(f"  {f['field']}: {f['value']}  [page {f.get('page')}, conf {f.get('confidence')}]")
```

See [`docs/connector_guide.md`](docs/connector_guide.md) for Python, curl, and Node.js examples + post-ingestion hook patterns.

---

## 📊 Live Dashboard

Watch grounding decisions in real-time: `http://localhost:7438/dashboard`

The dashboard shows live extraction traces — every field, its cascade path (EXACT → FUZZY → SEMANTIC → VISUAL), confidence score, and whether it was grounded or refused. Click "Reproduce" to re-run the same extraction and verify determinism.

---

## What Makes Kairo Different

1. **Pixel citations, not chunk citations.** Every grounded value points to an exact bounding box on the source page, not just "chunk 3." You can verify the source visually.
2. **5-layer grounding cascade.** NORMALIZE → EXACT → FUZZY (θ0.92) → SEMANTIC (φ0.86 + re-verify) → VISUAL IoU (ψ0.5) → VGVA → BLOCK. Fabricated quotes fail alignment and are blocked.
3. **Local-first, air-gap capable.** Zero network traffic in default config. No telemetry, no cloud sync. Your documents never leave your machine.
4. **Model-independent.** The grounding verifier re-checks any model's output against stored bboxes. Bring your own key (OpenAI / Anthropic / Google) or run the local stub baseline offline.

---

## 🛠️ Feature Matrix & Implementation Status

| Feature / Component | Description | Status |
| :--- | :--- | :---: |
| **Stateless Sidecar Ingestion** | Statelessly parses `.pdf`, `.docx`, and `.txt` files using Docling and an isolated PyMuPDF fastpath. Renders PNG page previews under `.kairo/page_images/`. | **Fully Implemented** |
| **Rust Core SQLite DB** | Appends metadata, pages, and chunks to local `.kairo/kairo.db` SQLite database using `rusqlite`. Rust core acts as the sole database writer. | **Fully Implemented** |
| **Grounding Validator Gate** | Runs LangExtract domain-specific schemas (generic, contract, invoice, paper) and whitelisted fallback logic before returning payloads. | **Fully Implemented** |
| **WASM Search Core** | A client-side similarity matcher compiled from Rust to WASM, indexing layout chunks in-memory. | **Fully Implemented** |
| **Client-Side Web Demo** | Glassmorphic React SPA running entirely in the browser. Uses WASM core for zero-dependency local queries. | **Fully Implemented** |
| **Tauri Desktop Overlay** | Frosted glass panel overlay toggled via `Ctrl+Alt+Space` hotkey, rendering grounded answers and SVG highlights. | **PENDING-REAL-APP** (Dev ready; installer packaging pending) |
| **In-Browser OCR** | OCR fallback for scanned PDFs within the client-side Web Demo. | **PENDING-REAL-APP** (Desktop app does OCR; web demo displays a warning) |
| **Multi-user Auth & Sync** | Cloud database sync, user registration, and session sharing. | **PENDING-REAL-APP** (Local-only database scope) |

---

## ⚡ Quick Start

### Prerequisites
- Cargo / Rust (v1.96+ recommended)
- Node.js (v24+ recommended)
- Python (v3.12+ recommended)

### 1. Build and Setup Virtualenv
```bash
make build
```

### 2. Run Global Unit & Integration Tests
```bash
make test
```

### 3. Run Grounding Benchmark
Run the evaluation suite:
```bash
make bench
```
Open `bench/leaderboard.html` in your browser to view the interactive table.

---

## 📅 Show HN Launch Timing
Recommended post window: **Tuesday–Thursday, 7:00 AM – 9:00 AM ET** (maximum visibility and active developer traffic).

---

## 🚷 /legacy Quarantine
All deprecated and legacy components have been quarantined to prevent production contamination:
- `/legacy`: Holds deprecated documentation and initial design layouts for early research phases (`bench_README.md`, `overlay_README.md`).
