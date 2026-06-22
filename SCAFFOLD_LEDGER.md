# SCAFFOLD_LEDGER.md — Master Go/No-Go Ledger

> Production-readiness ledger for Kairo Scaffold (slim public cut of Phantom v2.2).
> Ship ONLY when every row is GREEN on a clean checkout, twice, with receipts.
> **Anti-bluff:** live numbers + receipts only. Headline = BLIND number. Legacy 100% quarantined.

**Base commit:** `cabf393` (feat: initial release — Kairo Scaffold v1.0.0)
**Working branch:** `scaffold/prod-hardening`
**Last updated:** 2026-06-22

---

## Legend
- 🔴 RED — not done / failing / not verified
- 🟡 INFRA-PENDING — code complete, blocked on a real secret/cert (named)
- 🟦 BLOCKED-PENDING — blocked on phantom corpus + scorer copy-in (Parts 3 & 5)
- 🟢 GREEN — verified with receipt on clean checkout
- 🟠 PARTIAL — some sub-gates pass, others fail/blocked

---

## Part 7 — Master Gates

| # | Gate | Status | Receipt | Notes |
|---|------|--------|---------|-------|
| G1 | Verified cascade ported from phantom; no regex-only extraction path; grounding decisions match phantom on shared sample | 🟠 PARTIAL | — | Cascade stages present (NORMALIZE→EXACT→FUZZY0.92→SEMANTIC0.86+reverify→VISUAL IoU0.5) + front_cascade SW align + VGVA. No regex-only `/extract` path found. Parity test vs phantom on shared sample = PENDING (needs phantom sample + corpus). |
| G2 | Headline metric is the BLIND number; legacy 100% quarantined and absent from public surfaces | 🟠 PARTIAL | `receipts/S3.1_quarantine_100.txt` | README 100% headline quarantined → `legacy/`. Headline now = blind (PENDING corpus). index.html + SHOW_HN clean. Goes GREEN when blind number filled. |
| G3 | `make bench` deterministic + dated; reproduces on a clean machine twice | 🟠 PARTIAL | `receipts/S3.3_bench_stamp.txt` | Date+commit stamp added to `run_bench.py`. Full double-run on clean machine PENDING (sandbox can't install torch/CUDA for full venv). |
| G4 | Competitor rows live+dated OR clearly labeled cached | 🟢 GREEN | `receipts/S3.1_quarantine_100.txt` | README competitor rows relabeled "cached (capture date: initial dev)". No undated/unlabeled claims. |
| G5 | Real-world corpus blind grounded-rate published; FAILURE_TAXONOMY.md published | 🟦 BLOCKED-PENDING | — | Blocked on phantom corpus + scorer copy-in (Part 5). `sha256sum -c CHECKSUMS.sha256` must pass first. |
| G6 | Signed installers build (or INFRA-PENDING with exact missing secret) | 🟡 INFRA-PENDING | `receipts/G6_tauri_signing.txt` | `release.yml` ready; conditional signing. Missing: `TAURI_SIGNING_PRIVATE_KEY`, `_PASSWORD`, `APPLE_CERTIFICATE`, `_PASSWORD`, `KEYCHAIN_PASSWORD`. Unsigned-labeled path confirmed. |
| G7 | In-browser OCR warning fixed/gated; lockfile committed; offline install verified | 🟠 PARTIAL | `receipts/S4.2_ocr_warning.txt`, `receipts/S4.3_lockfile.txt` | OCR warning = accurate + gated (PASS). Python lockfile `requirements.lock` committed (PASS). Offline install verified = PENDING (sandbox torch/CUDA limit). |
| G8 | Injection corpus 100% blocked; secrets keychain-only; logs redacted; air-gap zero traffic | 🟠 PARTIAL | `receipts/G8_security.txt`, `receipts/G8_injection_corpus_block.txt` | Secrets keychain-only PASS, air-gap PASS, logs redacted PASS. Injection corpus = **90% (FAIL, needs 100%)** — 2 pattern gaps; fix upstream in phantom, re-pull. |
| G9 | License CI green (MIT core; PyMuPDF subprocess-isolated) | 🟢 GREEN | `receipts/G9_license_ci.txt` | `scripts/ci/license_check.py` PASSED. PyMuPDF not imported in kernel core (scripts-only, subprocess-isolated). |
| G10 | README refusal GIF + one-command quickstart; Show HN post pre-empts 3 killer comments | 🟠 PARTIAL | `receipts/L2_show_hn.txt` | SHOW_HN.md rewritten (pre-empts 3 killer comments + "what it can't do yet"). Quickstart present. Refusal GIF (refusal-first) = PENDING (demo.gif is not refusal-first). |
| G11 | CORE_SYNC.md documents one-way phantom→scaffold flow | 🟢 GREEN | `receipts/G11_core_sync.txt` | `CORE_SYNC.md` created. Documents one-way phantom→scaffold flow, sync log, blockers. |
| G12 | De-rigging phrase absent from all public artifacts | 🟢 GREEN | `receipts/G12_derig_phrase.txt` | grep across all public surfaces = 0 matches. AUDIT.md P8 row renamed. |

---

## Part 2 — Port Verified Core (dependency for G1)

| Task | Status | Receipt | Notes |
|------|--------|---------|-------|
| S2.1 Port full grounding cascade | 🟠 PARTIAL | — | Stages present in code; no regex-only path. Parity proof vs phantom PENDING. |
| S2.2 Port model-independent verifier + provenance bbox store | 🟠 PARTIAL | — | VGVA + provenance endpoint exist. Wrong-output-blocked test PENDING (needs full venv). |
| S2.3 Keep 5 intelligence modules aligned | 🟠 PARTIAL | — | All 5 present (VGVA, RAGShield, InjectionGuard, TieredRouter, ConstrainedDecoding). InjectionGuard has pattern gap (90%) — fix upstream. |
| S2.4 CORE_SYNC.md (one-way phantom→scaffold) | 🟢 GREEN | `receipts/G11_core_sync.txt` | Done. |

## Part 3 — Honest Benchmark (G2, G3, G4)

| Task | Status | Receipt | Notes |
|------|--------|---------|-------|
| S3.1 Reconcile/quarantine the 100% | 🟢 GREEN | `receipts/S3.1_quarantine_100.txt` | 100% quarantined to legacy; headline = blind. |
| S3.2 Frozen blind set + dev-vs-blind columns | 🟦 BLOCKED-PENDING | — | Needs phantom corpus. |
| S3.3 Deterministic, dated reproduction | 🟠 PARTIAL | `receipts/S3.3_bench_stamp.txt` | Stamp added; double-run PENDING. |
| S3.4 Competitor row honesty | 🟢 GREEN | `receipts/S3.1_quarantine_100.txt` | Labeled cached + date. |

## Part 4 — Production Hardening

| Task | Status | Receipt | Notes |
|------|--------|---------|-------|
| S4.1 Tauri signed installers | 🟡 INFRA-PENDING | `receipts/G6_tauri_signing.txt` | Missing signing secrets (named). |
| S4.2 In-browser OCR warning fix/gate | 🟢 GREEN | `receipts/S4.2_ocr_warning.txt` | Accurate + gated. |
| S4.3 Dependency pinning (lockfile) | 🟢 GREEN | `receipts/S4.3_lockfile.txt` | `requirements.lock` committed (147 pinned). |
| S4.4 Sidecar packaging + offline default | 🔴 RED | — | PyInstaller spec exists; offline install not verified (sandbox limit). |
| S4.5 Security parity | 🟠 PARTIAL | `receipts/G8_security.txt` | Secrets/air-gap/logs PASS; injection 90% FAIL. |

## Part 5 — Real-World Corpus Pass — 🟦 BLOCKED-PENDING

| Task | Status | Receipt | Notes |
|------|--------|---------|-------|
| Real-world blind grounded-rate | 🟦 BLOCKED-PENDING | — | Needs phantom corpus. |
| FAILURE_TAXONOMY.md | 🟦 BLOCKED-PENDING | — | Needs corpus run. |

## Part 6 — Launch Artifacts

| Task | Status | Receipt | Notes |
|------|--------|---------|-------|
| L1 README hero = refusal GIF + quickstart | 🟠 PARTIAL | — | Quickstart present; refusal-first GIF PENDING. |
| L2 Show HN post | 🟢 GREEN | `receipts/L2_show_hn.txt` | Rewritten; pre-empts 3 killer comments. |
| L3 Public leaderboard reproducible | 🟦 BLOCKED-PENDING | — | Needs blind corpus. |
| L4 Real-file challenge | 🟦 BLOCKED-PENDING | — | Only if Part 5 green. |
| L5 Landing page + demo GIF | 🔴 RED | — | index.html exists; align to honest numbers. |

---

## Blockers (must surface to human)

1. **🟦 Phantom corpus + scorer not yet copied in.** Parts 3 & 5 (any published grounded %) are BLOCKED-PENDING until `sha256sum -c CHECKSUMS.sha256` passes against phantom's checksum file. This blocks G2 (blind headline), G3 (full bench), G5 (real-world pass), L3/L4. **Single biggest blocker.**
2. **🟠 InjectionGuard pattern gap (90% block, needs 100%).** Two pattern gaps in `kernel/sidecar/models/injection_guard.py`: (a) output-manipulation pattern requires trailing quote char, (b) "disregard" alternation missing "prior". Per divergence guard, fix UPSTREAM in phantom, then re-pull. Blocks G8 full green.
3. **🟡 Signing secrets missing** (G6/S4.1): `TAURI_SIGNING_PRIVATE_KEY`, `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`, `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`, `KEYCHAIN_PASSWORD`. Code/config ready; INFRA-PENDING.
4. **🟠 Sandbox environment limit:** full `make acceptance` / `make bench` can't run in this sandbox (torch/CUDA wheels exceed tmpfs). Lightweight tests (license, injection patterns, secrets, near-miss) pass. Full venv verification needs a machine with adequate disk/tmp.
5. **🔴 Refusal-first GIF** (G10/L1): `demo.gif` is not refusal-first. Need a 30s GIF showing refusal first, citation second.

---

## Receipt index
Receipts in `/receipts/` (scaffold repo root):
- `G6_tauri_signing.txt` — Tauri signing INFRA-PENDING
- `G8_security.txt` — Security parity (secrets/air-gap/logs PASS, injection 90%)
- `G8_injection_corpus_block.txt` — Injection block rate 9/10 = 90% (FAIL)
- `G9_license_ci.txt` — License CI PASSED
- `G11_core_sync.txt` — CORE_SYNC.md created
- `G12_derig_phrase.txt` — De-rigging phrase absent
- `S3.1_quarantine_100.txt` — 100% quarantined, competitor rows labeled
- `S3.3_bench_stamp.txt` — Bench date+commit stamp added
- `S4.2_ocr_warning.txt` — OCR warning accurate + gated
- `S4.3_lockfile.txt` — Python lockfile committed
- `L2_show_hn.txt` — Show HN rewritten

Historical receipts in `docs/receipts/` (44 files, pre-existing).