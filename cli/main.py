"""
Kairo CLI — grounded document intelligence.
Commands: index, run, ask, doctor, correct
All commands output grounded JSON with {value, confidence, page, bbox, method}.
"""
import argparse
import json
import sys
import os

SIDECAR_URL = os.environ.get("KAIRO_SIDECAR_URL", "http://127.0.0.1:7438")


def _get_http():
    """Get an httpx client, or raise a clear error if not installed."""
    try:
        import httpx
        return httpx
    except ImportError:
        print("ERROR: httpx not installed. Run: pip install httpx", file=sys.stderr)
        sys.exit(1)


def _index_file(filepath: str) -> dict:
    """POST /index and return the response dict."""
    httpx = _get_http()
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        print(f"ERROR: File not found: {abs_path}", file=sys.stderr)
        sys.exit(1)
    try:
        resp = httpx.post(f"{SIDECAR_URL}/index", json={"path": abs_path}, timeout=60.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        print(
            f"ERROR: Cannot connect to Kairo sidecar at {SIDECAR_URL}.\n"
            "Start it with: python -m uvicorn kernel.sidecar.app:app --port 7438 --host 127.0.0.1",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_index(args):
    result = _index_file(args.file)
    doc_id = result.get("doc_id", "unknown")
    pages = result.get("pages", 0)
    chunks = result.get("chunks", 0)
    print(json.dumps({
        "status": "indexed",
        "doc_id": doc_id,
        "pages": pages,
        "chunks": chunks,
        "file": args.file,
    }, indent=2))


def cmd_run(args):
    httpx = _get_http()
    # Step 1: index the document
    index_result = _index_file(args.file)
    doc_id = index_result.get("doc_id")

    # Step 2: extract fields using the pack
    try:
        resp = httpx.post(
            f"{SIDECAR_URL}/extract",
            json={"doc_id": doc_id, "pack": args.pack},
            timeout=60.0
        )
        resp.raise_for_status()
        extractions = resp.json()
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to Kairo sidecar at {SIDECAR_URL}.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR extracting: {e}", file=sys.stderr)
        sys.exit(1)

    grounded = []
    refused = []

    for ext in extractions:
        if ext.get("status") == "blocked" or ext.get("method") == "block":
            refused.append({
                "field": ext.get("field"),
                "status": "REFUSED — no grounded source found",
            })
        else:
            anchors = ext.get("anchors", [])
            page = anchors[0]["page"] if anchors else None
            bbox = anchors[0]["bbox"] if anchors else None
            grounded.append({
                "field": ext.get("field"),
                "value": ext.get("value"),
                "confidence": ext.get("confidence"),
                "page": page,
                "bbox": bbox,
                "method": ext.get("method"),
            })

    output = {
        "doc_id": doc_id,
        "pack": args.pack,
        "file": args.file,
        "grounded": grounded,
        "refused": refused,
    }

    if refused and not grounded:
        print("REFUSAL: No grounded extractions found.", file=sys.stderr)
    print(json.dumps(output, indent=2))


def cmd_ask(args):
    httpx = _get_http()
    # Step 1: index the document
    index_result = _index_file(args.file)
    doc_id = index_result.get("doc_id")

    # Step 2: ask the question
    try:
        resp = httpx.post(
            f"{SIDECAR_URL}/ask",
            json={"doc_id": doc_id, "query": args.query},
            timeout=60.0
        )
        resp.raise_for_status()
        answer = resp.json()
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to Kairo sidecar at {SIDECAR_URL}.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR asking: {e}", file=sys.stderr)
        sys.exit(1)

    grounded = answer.get("grounded", False)
    citations = answer.get("citations", [])
    text = answer.get("text", "")

    if not grounded or text in ("blocked", "", None):
        output = {
            "query": args.query,
            "status": "REFUSED — No grounded source found. Kairo declines to answer.",
            "grounded": False,
        }
    else:
        page = citations[0]["page"] if citations else None
        bbox = citations[0]["bbox"] if citations else None
        output = {
            "query": args.query,
            "value": text,
            "confidence": 1.0 if grounded else 0.0,
            "page": page,
            "bbox": bbox,
            "method": "grounded",
            "grounded": True,
        }

    print(json.dumps(output, indent=2))


def cmd_correct(args):
    httpx = _get_http()
    try:
        resp = httpx.post(
            f"{SIDECAR_URL}/correct",
            json={"extraction_id": args.extraction_id, "new_value": args.new_value},
            timeout=30.0
        )
        resp.raise_for_status()
        correction = resp.json()
        print(json.dumps(correction, indent=2, default=str))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_doctor(args):
    """Health check: verify sidecar reachable, DB writable, stores available."""
    httpx = _get_http()

    checks = {}

    # 1. Sidecar reachable
    try:
        resp = httpx.get(f"{SIDECAR_URL}/health", timeout=5.0)
        if resp.status_code == 200:
            health = resp.json()
            checks["sidecar_reachable"] = ("PASS", f"sidecar responded at {SIDECAR_URL}")
            checks["db_writable"] = (
                "PASS" if health.get("db_writable") else "FAIL",
                f"db at {health.get('db_path', 'unknown')}"
            )
            checks["qdrant_available"] = (
                "PASS" if health.get("qdrant_available") else "WARN",
                "qdrant_available=True" if health.get("qdrant_available") else "qdrant not available (fallback to SQLite)"
            )
            checks["embedding_model"] = ("PASS", f"model={health.get('embedding_model', 'unknown')}")
        else:
            checks["sidecar_reachable"] = ("FAIL", f"Unexpected HTTP {resp.status_code}")
    except Exception:
        checks["sidecar_reachable"] = ("FAIL", f"Cannot connect to {SIDECAR_URL} — is the sidecar running?")
        checks["db_writable"] = ("UNKNOWN", "cannot check, sidecar unreachable")
        checks["qdrant_available"] = ("UNKNOWN", "cannot check, sidecar unreachable")
        checks["embedding_model"] = ("UNKNOWN", "cannot check, sidecar unreachable")

    # Print table
    print("\nKairo Doctor — System Health Check")
    print("=" * 50)
    all_pass = True
    for check, (status, detail) in checks.items():
        icon = "[+]" if status == "PASS" else ("[!]" if status == "WARN" else ("[-]" if status == "UNKNOWN" else "[x]"))
        print(f"  {icon} {check:30s} {status:8s}  {detail}")
        if status == "FAIL":
            all_pass = False

    print("=" * 50)
    if all_pass:
        print("All checks passed.\n")
        sys.exit(0)
    else:
        print("Some checks FAILED. See above for details.\n")
        sys.exit(1)

def cmd_keys(args):
    # Ensure kernel is in path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from kernel.sidecar.models.secrets import get_api_key, set_api_key, clear_api_key, redact_key

    if args.keys_command == "set":
        set_api_key(args.provider, args.key)
        print(json.dumps({
            "status": "success",
            "message": f"API key for {args.provider} set successfully in OS keychain.",
            "provider": args.provider,
            "key": redact_key(args.key),
        }, indent=2))
    elif args.keys_command == "clear":
        clear_api_key(args.provider)
        print(json.dumps({
            "status": "success",
            "message": f"API key for {args.provider} cleared from OS keychain.",
            "provider": args.provider,
        }, indent=2))
    elif args.keys_command == "list":
        providers = ["openai", "anthropic", "google"]
        status = {}
        for p in providers:
            key = get_api_key(p)
            status[p] = {
                "configured": key is not None,
                "key": redact_key(key) if key else "None"
            }
        print(json.dumps(status, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Kairo — verifiable local document intelligence (refuse-or-cite)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index
    parser_index = subparsers.add_parser("index", help="Index a document into Kairo's provenance store")
    parser_index.add_argument("file", help="Path to file (.txt, .pdf, .docx)")

    # run
    parser_run = subparsers.add_parser("run", help="Extract grounded fields using a domain pack")
    parser_run.add_argument("file", help="Path to file")
    parser_run.add_argument("--pack", required=True,
                            choices=["generic", "invoice", "paper", "contract"],
                            help="Pack name")

    # ask
    parser_ask = subparsers.add_parser("ask", help="Ask a grounded question about a document")
    parser_ask.add_argument("file", help="Path to file")
    parser_ask.add_argument("query", help="Question to ask (quoted)")

    # correct
    parser_correct = subparsers.add_parser("correct", help="Record a human correction to an extraction")
    parser_correct.add_argument("extraction_id", help="ID of extraction to correct")
    parser_correct.add_argument("new_value", help="The corrected value")

    # doctor
    subparsers.add_parser("doctor", help="Check system health (sidecar, DB, stores)")

    # keys
    parser_keys = subparsers.add_parser("keys", help="Manage API keys in the OS keychain")
    keys_subparsers = parser_keys.add_subparsers(dest="keys_command", required=True)

    parser_keys_set = keys_subparsers.add_parser("set", help="Set an API key")
    parser_keys_set.add_argument("provider", choices=["openai", "anthropic", "google"], help="Provider name")
    parser_keys_set.add_argument("key", help="API key value")

    parser_keys_clear = keys_subparsers.add_parser("clear", help="Clear an API key")
    parser_keys_clear.add_argument("provider", choices=["openai", "anthropic", "google"], help="Provider name")

    keys_subparsers.add_parser("list", help="List configured API keys (redacted)")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "correct":
        cmd_correct(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "keys":
        cmd_keys(args)


if __name__ == "__main__":
    main()
