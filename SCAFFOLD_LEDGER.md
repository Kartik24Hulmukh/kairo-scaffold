# SCAFFOLD_LEDGER.md — Master Go/No-Go Ledger

> Production-readiness ledger for Kairo Scaffold (slim public cut of Phantom v2.2).
> Ship ONLY when every row is GREEN on a clean checkout, twice, with receipts.
> **Anti-bluff:** live numbers + receipts only. Headline = BLIND number. Legacy 100% quarantined.

**Base commit:** `cabf393` (feat: initial release — Kairo Scaffold v1.0.0)
**Working branch:** `scaffold/prod-hardening`
**Last updated:** 2026-06-22 (Pass 2 — phantom corpus pull-in)

---

## Legend
- 🔴 RED — not done / failing / not verified
- 🟡 INFRA-PENDING — code complete, blocked on a real secret/cert/resource (named)
- 🟦 BLOCKED-PENDING — blocked on phantom corpus + scorer copy-in (Parts 3 & 5)
- 🟢 GREEN — verified with receipt on clean checkout
- 🟠 PARTIAL — some sub-gates pass, others fail/blocked

---

## Part 7 — Master Gates

| # | Gate | Status | Receipt | Notes |
|---|------|--------|---------|-------|
| G1 | Verified cascade ported from phantom; no regex-only extraction path; grounding decisions match phantom on shared sample | 🟠 PARTIAL | — | Cascade stages present (NORMALIZE→EXACT→FUZZY0.92→SEMANTIC0.86+reverify→VISUAL IoU0.5) + front_cascade SW align + VGVA. No regex-only `/extract` path. Parity test vs phantom = PENDING (needs full venv for inference). |
| G2 | Headline metric is the BLIND number; legacy 100% quarantined and absent from public surfaces | 🟠 PARTIAL | `receipts/S3.1_quarantine_100.txt` | README 100% quarantined → `legacy/`. Headline = blind (INFRA-PENDING — bench numbers not yet run). index.html + SHOW_HN clean. |
| G3 | `make bench` deterministic + dated; reproduces on a clean machine twice | 🟡 INFRA-PENDING | `receipts/S3.3_bench_stamp.txt`, `receipts/G3_G5_bench_infra_pending.txt` | Date+commit stamp added. Full double-run blocked: sandbox can't install torch wheels (~2 GB, tmpfs 2 GB limit). Needs ≥4 GB disk + GPU. |
| G4 | Competitor rows live+dated OR clearly labeled cached | 🟢 GREEN | `receipts/S3.1_quarantine_100.txt` | README competitor rows labeled "cached (capture date: initial dev)". |
| G5 | Real-world corpus blind grounded-rate published; FAILURE_TAXONOMY.md published | 🟡 INFRA-PENDING | `receipts/G3_G5_bench_infra_pending.txt` | Corpus + scorer copied in, checksums pass. `BENCHMARKS.md` + `FAILURE_TAXONOMY.md` templates generated. Bench run blocked (torch/disk). Phantom ref: ~67.5% grounded. |
| G6 | Signed installers build (or INFRA-PENDING with exact missing secret) | 🟡 INFRA-PENDING | `receipts/G6_tauri_signing.txt` | Missing: `TAURI_SIGNING_PRIVATE_KEY`, `_PASSWORD`, `APPLE_CERTIFICATE`, `_PASSWORD`, `KEYCHAIN_PASSWORD`. |
| G7 | In-browser OCR warning fixed/gated; lockfile committed; offline install verified | 🟠 PARTIAL | `receipts/S4.2_ocr_warning.txt`, `receipts/S4.3_lockfile.txt` | OCR warning PASS. Lockfile PASS. Offline install = PENDING (torch/disk). |
| G8 | Injection corpus 100% blocked; secrets keychain-only; logs redacted; air-gap zero traffic | 🟢 GREEN | `receipts/G8_security.txt`, `receipts/G8_injection_corpus_block.txt` | **25/25 = 100% blocked, 0 false positives.** Secrets keychain-only PASS, air-gap PASS, logs redacted PASS. Patterns ported from phantom guardrails.rs + firewall.rs. |
| G9 | License CI green (MIT core; PyMuPDF subprocess-isolated) | 🟢 GREEN | `receipts/G9_license_ci.txt` | PASSED. |
| G10 | README refusal GIF + one-command quickstart; Show HN post pre-empts 3 killer comments | 🟢 GREEN | `receipts/L1_refusal_gif.txt`, `receipts/L2_show_hn.txt` | Refusal-first GIF (refusal_demo.gif, 89.6 KB) in README hero. SHOW_HN.md rewritten. Quickstart present. |
| G11 | CORE_SYNC.md documents one-way phantom→scaffold flow | 🟢 GREEN | `receipts/G11_core_sync.txt` | Done. |
| G12 | De-rigging phrase absent from all public artifacts | 🟢 GREEN | `receipts/G12_derig_phrase.txt` | 0 matches. |

---

## Part 2 — Port Verified Core (dependency for G1)

| Task | Status | Receipt | Notes |
|------|--------|---------|-------|
| S2.1 Port full grounding cascade | 🟠 PARTIAL | — | Stages present; no regex-only path. Parity proof PENDING (venv). |
| S2.2 Port model-independent verifier + provenance bbox store | 🟠 PARTIAL | — | VGVA + provenance endpoint exist. Wrong-output-blocked test PENDING (venv). |
| S2.3 Keep 5 intelligence modules aligned | 🟢 GREEN | `receipts/G8_injection_corpus_block.txt` | InjectionGuard patterns ported from phantom (25/25 block, 0 FP). Other 4 modules present. |
| S2.4 CORE_SYNC.md (one-way phantom→scaffold) | 🟢 GREEN | `receipts/G11_core_sync.txt` | Done. |

## Part 3 — Honest Benchmark (G2, G3, G4)

| Task | Status | Receipt | Notes |
|------|--------|---------|-------|
| S3.1 Reconcile/quarantine the 100% | 🟢 GREEN | `receipts/S3.1_quarantine_100.txt` | 100% quarantined to legacy. |
| S3.2 Frozen blind set + dev-vs-blind columns | 🟡 INFRA-PENDING | `receipts/S3.2_checksums_pass.txt` | Corpus copied in, checksums pass (23/23 OK). `BENCHMARKS.md` template with dev-vs-blind columns generated. Numbers PENDING bench run. |
| S3.3 Deterministic, dated reproduction | 🟡 INFRA-PENDING | `receipts/S3.3_bench_stamp.txt` | Stamp added. Double-run blocked (torch/disk). |
| S3.4 Competitor row honesty | 🟢 GREEN | `receipts/S3.1_quarantine_100.txt` | Labeled cached + date. |

## Part 4 — Production Hardening

| Task | Status | Receipt | Notes |
|------|--------|---------|-------|
| S4.1 Tauri signed installers | 🟡 INFRA-PENDING | `receipts/G6_tauri_signing.txt` | Missing signing secrets (named). |
| S4.2 In-browser OCR warning fix/gate | 🟢 GREEN | `receipts/S4.2_ocr_warning.txt` | Accurate + gated. |
| S4.3 Dependency pinning (lockfile) | 🟢 GREEN | `receipts/S4.3_lockfile.txt` | `requirements.lock` committed (147 pinned). |
| S4.4 Sidecar packaging + offline default | 🟡 INFRA-PENDING | `receipts/G3_G5_bench_infra_pending.txt` | PyInstaller spec exists. Offline install not verified (torch/disk). |
| S4.5 Security parity | 🟢 GREEN | `receipts/G8_security.txt` | Injection 100%, secrets keychain, air-gap, logs redacted. |

## Part 5 — Real-World Corpus Pass

| Task | Status | Receipt | Notes |
|------|--------|---------|-------|
| Real-world blind grounded-rate | 🟡 INFRA-PENDING | `receipts/G3_G5_bench_infra_pending.txt` | Corpus ready, bench run blocked (torch/disk). |
| FAILURE_TAXONOMY.md | 🟡 INFRA-PENDING | — | Template generated; numbers PENDING bench run. |

## Part 6 — Launch Artifacts

| Task | Status | Receipt | Notes |
|------|--------|---------|-------|
| L1 README hero = refusal GIF + quickstart | 🟢 GREEN | `receipts/L1_refusal_gif.txt` | refusal_demo.gif in README hero. |
| L2 Show HN post | 🟢 GREEN | `receipts/L2_show_hn.txt` | Rewritten; pre-empts 3 killer comments. |
| L3 Public leaderboard reproducible | 🟡 INFRA-PENDING | — | `bench/leaderboard.json` template generated. Numbers PENDING bench run. |
| L4 Real-file challenge | 🟡 INFRA-PENDING | — | Only if Part 5 green (needs bench run). |
| L5 Landing page + demo GIF | 🟠 PARTIAL | — | index.html exists; refusal GIF added to README. Landing page alignment to honest numbers PENDING. |

---

## Blockers (must surface to human)

1. **🟡 Benchmark run (G3/G5/S3.2/S3.3/L3/L4/S4.4) — INFRA-PENDING.** The blind bench requires a full PyTorch venv (~2 GB torch+CUDA wheels). This sandbox's tmpfs (2 GB) cannot install them. **Run on a machine with ≥4 GB free disk + GPU** (or CPU fallback with 8 GB RAM):
   ```bash
   make build && make bench
   ```
   Then paste numbers into `BENCHMARKS.md`, `FAILURE_TAXONOMY.md`, `bench/leaderboard.json`. Phantom's blind reference: ~67.5% grounded / 32.5% false-refusal (commit 3dd4197). Scaffold's number MUST match (sha256 checksums prove shared oracle).
2. **🟡 Signing secrets missing** (G6/S4.1): `TAURI_SIGNING_PRIVATE_KEY`, `_PASSWORD`, `APPLE_CERTIFICATE`, `_PASSWORD`, `KEYCHAIN_PASSWORD`.
3. **🟠 Cascade parity test** (G1/S2.1/S2.2): needs full venv to run scaffold vs phantom on shared sample.

---

## Receipt index
Receipts in `/receipts/` (scaffold repo root):
- `G3_G5_bench_infra_pending.txt` — Bench INFRA-PENDING (torch/disk gap)
- `G6_tauri_signing.txt` — Tauri signing INFRA-PENDING
- `G8_security.txt` — Security parity (all PASS)
- `G8_injection_corpus_block.txt` — Injection 25/25 = 100% blocked, 0 FP
- `G9_license_ci.txt` — License CI PASSED
- `G11_core_sync.txt` — CORE_SYNC.md created
- `G12_derig_phrase.txt` — De-rigging phrase absent
- `L1_refusal_gif.txt` — Refusal-first GIF generated
- `L2_show_hn.txt` — Show HN rewritten
- `S3.1_quarantine_100.txt` — 100% quarantined, competitor rows labeled
- `S3.2_checksums_pass.txt` — Blind corpus checksums 23/23 OK
- `S3.3_bench_stamp.txt` — Bench date+commit stamp added
- `S4.2_ocr_warning.txt` — OCR warning accurate + gated
- `S4.3_lockfile.txt` — Python lockfile committed

Historical receipts in `docs/receipts/` (44 files, pre-existing).