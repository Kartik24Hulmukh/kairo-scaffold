import pathlib
import sys
sys.stdout.reconfigure(encoding='utf-8')


repo_path = pathlib.Path(r"C:\Users\praja\OneDrive\Desktop\test-env\repositories\kairo-scaffold")
files_to_check = [
    # Group A
    "scripts/check_receipt.py",
    # Group B
    "kernel/sidecar/ingest/bbox_verify.py",
    "kernel/sidecar/ingest/quote_align.py",
    "kernel/sidecar/ingest/ocr_backends.py",
    "kernel/sidecar/models/constrained_decoding.py",
    # Group C/D
    "bench/eval_harness.py",
    "kernel/sidecar/retrieval/vector_store.py",
    "kernel/sidecar/models/embeddings.py",
    "kernel/sidecar/models/tier_router.py",
    "kernel/sidecar/models/consistency_gate.py",
    # Group E/F/G
    "scripts/check_perf_budget.py",
    "scripts/check_not_list.py",
    "kernel/sidecar/models/rag_shield.py",
]

for f in files_to_check:
    fp = repo_path / f
    if fp.exists():
        print(f"[FOUND] {f} | Size: {fp.stat().st_size} bytes")
        # Print first few lines
        with open(fp, "r", encoding="utf-8") as file_obj:
            lines = file_obj.readlines()
            head = "".join(lines[:10])
            print("  --- HEAD ---")
            print(head)
            print("  ------------")
    else:
        print(f"[MISSING] {f}")
