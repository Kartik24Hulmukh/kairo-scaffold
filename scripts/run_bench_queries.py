import json
import httpx

with open("bench/questions.json", "r") as f:
    fixtures = json.load(f)

for f in fixtures:
    filename = f["filename"]
    # get doc_id by indexing first
    filepath = f"C:\\Users\\praja\\OneDrive\\Desktop\\test-env\\repositories\\kairo-scaffold\\fixtures\\golden\\{filename}"
    if "adversarial" in filename:
        filepath = f"C:\\Users\\praja\\OneDrive\\Desktop\\test-env\\repositories\\kairo-scaffold\\fixtures\\{filename}"
        
    try:
        r = httpx.post("http://127.0.0.1:7438/index", json={"path": filepath})
        doc_id = r.json()["doc_id"]
    except Exception as e:
        print(f"Index fail for {filename}: {e}")
        continue
        
    for q in f["questions"]:
        if not q["answerable"]:
            res = httpx.post("http://127.0.0.1:7438/ask", json={"doc_id": doc_id, "query": q["query"]})
            data = res.json()
            print(f"Doc: {filename} | Query: {q['query']} | Text: {data['text']} | Grounded: {data['grounded']}")
