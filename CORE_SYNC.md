# CORE_SYNC.md — Phantom → Scaffold Core Sync

> Documents the ONE-WAY flow of the verified grounding core from **Kairo Phantom v2.2** (upstream, source of truth) to **Kairo Scaffold** (downstream, public slim cut).

## 1. Direction (non-negotiable)

```
phantom v2.2  ──── pull / subtree / release artifact ────►  scaffold
   (upstream)                                              (downstream)
```

- **Phantom is upstream.** Scaffold NEVER pushes core logic back.
- **Scaffold pulls.** Scaffold consumes the verified core; it does NOT re-derive, reimplement, or independently tune the grounding cascade, verifier, scorer, or benchmark corpus.
- **If a core fix is needed, fix it UPSTREAM in phantom, then re-pull.** Editing core logic in scaffold creates divergence — a launch-credibility bug.

## 2. What flows from phantom → scaffold

| Artifact | Location in scaffold | Source in phantom | Sync method |
|----------|---------------------|-------------------|-------------|
| Grounding cascade (NORMALIZE→EXACT→FUZZY θ0.92→SEMANTIC φ0.86+re-verify→VISUAL IoU ψ0.5→VGVA→BLOCK) | `kernel/sidecar/cascade/front_select.py`, `kernel/sidecar/app.py` (extract/ask endpoints) | phantom cascade module | subtree pull / release artifact |
| Model-independent verifier + provenance bbox store | `kernel/sidecar/models/vgva.py`, `kernel/sidecar/ingest/bbox_verify.py` | phantom verifier | subtree pull |
| 5 intelligence modules (VGVA, RAGShield, InjectionGuard, TieredRouter, ConstrainedDecoding) | `kernel/sidecar/models/{vgva,rag_shield,injection_guard,tier_client,constrained_decoding}.py` | phantom modules | subtree pull |
| Blind benchmark corpus + labels | `fixtures/blind/` (to be copied in) | phantom corpus | content-addressed copy + `sha256sum -c CHECKSUMS.sha256` |
| Scorer (`score.py`) | `bench/score.py` (to be copied in) | phantom scorer | verbatim copy |

## 3. What does NOT flow (scaffold-only)

- Tauri desktop overlay (`overlay/`) — scaffold's slim packaging surface
- Web demo (`web-demo/`) — scaffold's zero-install demo
- CLI (`cli/`) — scaffold's command-line interface
- Launch artifacts (README, SHOW_HN.md, landing page) — scaffold's public face
- These are scaffold's responsibility and may diverge from phantom's UI.

## 4. Verification of sync (the proof of no divergence)

After every pull:
1. `sha256sum -c CHECKSUMS.sha256` MUST pass in scaffold against phantom's checksum file (corpus + scorer).
2. Run the parity test: scaffold and phantom produce identical grounding decisions on a shared sample.
3. The blind grounded-rate published by scaffold MUST equal phantom's blind number (proven by matching sha256).

If any of these fail, the pull is incomplete or corrupted — do not ship.

## 5. How a core fix propagates

```
1. Bug found in scaffold's grounding behavior
2. Reproduce in phantom (upstream)
3. Fix in phantom; bump phantom version
4. Re-pull affected modules into scaffold
5. Re-run sha256sum -c + parity test + make acceptance
6. Update this file's sync log below
```

## 6. Sync log

| Date | Phantom version | Artifact pulled | sha256 verified | Parity test | Notes |
|------|----------------|-----------------|-----------------|-------------|-------|
| 2026-06-22 | v2.2 (initial) | cascade + 5 modules (present at clone) | PENDING (corpus not yet copied) | PENDING | Initial scaffold cut from phantom v2.2. Corpus + scorer copy-in pending. |

## 7. Current blockers

- **Corpus + scorer not yet copied in from phantom.** `CHECKSUMS.sha256` missing. Parts 3 & 5 (any published grounded %) remain BLOCKED-PENDING until `sha256sum -c` passes.
- **InjectionGuard pattern gap** (90% block, needs 100%): fix upstream in phantom, then re-pull `kernel/sidecar/models/injection_guard.py`.