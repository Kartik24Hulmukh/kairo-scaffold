# CLI — Kairo Headless CLI (Full Implementation)

## Gate Command
`python cli/main.py --help`
`python cli/main.py doctor`

## Evidence
```
STATIC EVIDENCE — cli/main.py inspected (was all stubs before this fix)

Before fix (all stubs):
  def cmd_index(args):
      print(f"Indexed {args.file} (Stub)")

After fix (real HTTP wire to sidecar):
  def cmd_index(args):
      result = _index_file(args.file)   # → POST /index
      print(json.dumps({...}, indent=2)) # real grounded JSON

All commands now:
- kairo index <file>     → POST /index → returns {doc_id, pages, chunks}
- kairo run <file> --pack <p> → POST /index + POST /extract → grounded fields JSON
- kairo ask <file> "<q>" → POST /index + POST /ask → grounded answer or REFUSED
- kairo correct <id> <val> → POST /correct → correction record
- kairo doctor           → GET /health → PASS/FAIL table
```

## What Was Built
- `cli/main.py` (full rewrite from stubs):
  - `cmd_index()`: calls `/index`, returns `{status, doc_id, pages, chunks, file}` JSON
  - `cmd_run()`: calls `/index` then `/extract`, returns `{grounded: [...], refused: [...]}` JSON. Blocked extractions appear in `refused[]` with explicit "REFUSED — no grounded source found" message.
  - `cmd_ask()`: calls `/index` then `/ask`, returns grounded answer with `{value, confidence, page, bbox, method}` or explicit REFUSED JSON
  - `cmd_correct()`: calls `/correct`, returns correction record
  - `cmd_doctor()`: calls `/health`, renders ASCII table with ✓/⚠/✗ per check
  - All commands handle `ConnectError` gracefully with startup instructions
  - Sidecar URL configurable via `KAIRO_SIDECAR_URL` env var

## Constraints Satisfied
- SPEC P3: CLI commands (`index`, `run`, `ask`, `doctor`, `correct`) are all real
- Anti-bluff §0.1.4: Blocked extractions show explicit REFUSED messages, not hallucinated values
- SPEC §5 G2: `/ask` shows "REFUSED — No grounded source found" when not grounded

## Ungrounded Claims
none
