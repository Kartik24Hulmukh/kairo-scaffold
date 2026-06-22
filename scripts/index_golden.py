import os
import re
import json
import subprocess

def main():
    fixtures_dir = r"fixtures/golden"
    ground_truth_path = os.path.join(fixtures_dir, "ground_truth.json")
    
    # Identify files
    files = [
        f for f in os.listdir(fixtures_dir)
        if f.endswith((".pdf", ".docx", ".txt")) and f != "placeholder.txt"
    ]
    # Sort files to have stable ordering
    files.sort()
    
    print(f"Found {len(files)} files to index.")
    
    fixtures_results = []
    
    # Regex to match the output:
    # "Successfully indexed. ID: <doc_id>, Pages: <page_count>, Chunks: <chunk_count>"
    pattern = re.compile(
        r"Successfully indexed\.\s+ID:\s+(?P<doc_id>[^,]+),\s+Pages:\s+(?P<pages>\d+),\s+Chunks:\s+(?P<chunks>\d+)"
    )
    
    for i, file_name in enumerate(files, 1):
        file_path = f"fixtures/golden/{file_name}"
        print(f"[{i}/{len(files)}] Indexing {file_path}...")
        
        # Run CLI command
        # Note: We run cargo run --bin kairo -- index fixtures/golden/<file_name>
        cmd = ["cargo", "run", "--bin", "kairo", "--", "index", file_path]
        
        try:
            # We capture stdout and stderr, merge them just in case
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            output = result.stdout
            print(f"CLI Output for {file_name}:")
            print(output)
            
            match = pattern.search(output)
            if not match:
                # Let's also check stderr or alternative outputs
                print(f"Stderr: {result.stderr}")
                raise ValueError(f"Could not parse indexing output for {file_name}. Output was:\n{output}")
                
            doc_id = match.group("doc_id")
            pages = int(match.group("pages"))
            chunks = int(match.group("chunks"))
            
            print(f"Parsed -> Pages: {pages}, Chunks: {chunks}")
            
            fixtures_results.append({
                "file": file_name,
                "expected_pages": pages,
                "expected_chunks": chunks
            })
            
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise e
            
    # Write to ground_truth.json
    data = {
        "fixtures": fixtures_results
    }
    
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print(f"Successfully wrote ground truth mapping to {ground_truth_path}")

if __name__ == "__main__":
    main()
