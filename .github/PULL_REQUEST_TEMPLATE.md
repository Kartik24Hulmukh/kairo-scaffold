## PLAN
<!-- 3-6 lines: what will change, which files, which gate this satisfies -->



## CRITIQUE
<!-- 3-6 lines: risks, edge cases, license/parallel conflicts -->



## VERIFY
<!-- Paste the real gate command output here — gate must PASS before merging -->

**Gate command:** `make <target>`

**Gate output (verbatim):**
```
(paste real output here — no asserted constants, no fabricated numbers)
```

## Receipt
<!-- Receipt file added/updated: docs/receipts/<TASK-ID>.md -->
- [ ] `docs/receipts/<TASK-ID>.md` added with Status: PASS
- [ ] Gate output in receipt is reproducible by re-running the gate command
- [ ] No banned phrases in any changed artifact (100%, 10/10, perfect, flawless, zero bugs)

## Checklist
- [ ] Tests pass: `make test`
- [ ] License check passes: `make license-check`
- [ ] Not-list check: `make not-list-check`
- [ ] Receipt check: `make receipt-check`
