import os
import subprocess
import re
import json

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    golden_dir = os.path.join(root_dir, "fixtures", "golden")
    output_json = os.path.join(golden_dir, "ground_truth.json")

    print(f"Root dir: {root_dir}")
    print(f"Golden dir: {golden_dir}")

    # Identify files
    files = []
    for f in os.listdir(golden_dir):
        if f in ("placeholder.txt", "ground_truth.json"):
            continue
        if f.endswith(".pdf") or f.endswith(".docx") or f.endswith(".txt"):
            files.append(f)

    # Sort files to have a stable order in JSON
    files.sort()
    print(f"Found {len(files)} files to index: {files}")

    fixtures = []

    # Regex to capture ID, Pages, and Chunks
    # Successfully indexed. ID: <doc_id>, Pages: <page_count>, Chunks: <chunk_count>
    # Note: CLI output may contain cargo output on stderr, and CLI output on stdout
    pattern = re.compile(r"Successfully indexed\.\s+ID:\s+([a-zA-Z0-9_]+),\s+Pages:\s+(\d+),\s+Chunks:\s+(\d+)", re.IGNORECASE)

    for file_name in files:
        relative_path = f"fixtures/golden/{file_name}"
        cmd = ["cargo", "run", "--bin", "kairo", "--offline", "--", "index", relative_path]
        print(f"Running command: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        
        stdout = res.stdout
        stderr = res.stderr
        print(f"STDOUT:\n{stdout}")
        if res.returncode != 0:
            print(f"STDERR:\n{stderr}")
            print(f"Command failed with code {res.returncode}")
            continue

        match = pattern.search(stdout)
        if match:
            doc_id = match.group(1)
            pages = int(match.group(2))
            chunks = int(match.group(3))
            print(f"File: {file_name} -> doc_id={doc_id}, pages={pages}, chunks={chunks}")
            fixtures.append({
                "file": file_name,
                "expected_pages": pages,
                "expected_chunks": chunks
            })
        else:
            print(f"Could not parse output for {file_name}. Full stdout:\n{stdout}")

    result = {"fixtures": fixtures}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Successfully generated {output_json} with {len(fixtures)} entries.")

if __name__ == "__main__":
    main()
