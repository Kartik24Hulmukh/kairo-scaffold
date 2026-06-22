import httpx

filename = "adversarial/low_dpi.pdf"
filepath = f"C:\\Users\\praja\\OneDrive\\Desktop\\test-env\\repositories\\kairo-scaffold\\fixtures\\{filename}"

# Index first
r = httpx.post("http://127.0.0.1:7438/index", json={"path": filepath})
doc_id = r.json()["doc_id"]

query = "What is the weight of the scanned document in grams?"
res = httpx.post("http://127.0.0.1:7438/ask", json={"doc_id": doc_id, "query": query})
print(res.text)
