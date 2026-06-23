# Kairo Connector Guide — 5-Minute Integration

> Connect your document manager to Kairo's grounding engine in 5 lines of code.
> Every extracted value comes with a page number + bounding box, or a refusal with a reason.

## Quick Start

### Python (5 lines)

```python
import requests

# Upload a document and get grounded extractions
resp = requests.post(
    "http://localhost:7438/api/extract-document",
    files={"file": open("invoice.pdf", "rb")}
)
data = resp.json()

for field, info in data["fields"]:
    if info["status"] == "grounded":
        print(f"  ✓ {field}: {info['value']}  [page {info['page']}, conf {info['confidence']}]")
    else:
        print(f"  ✗ {field}: REFUSED — {info.get('reason', 'no source')}")
```

### curl (1 line)

```bash
curl -F 'file=@invoice.pdf' http://localhost:7438/api/extract-document
```

### Node.js

```javascript
const formData = new FormData();
formData.append('file', fileBuffer, 'invoice.pdf');
const resp = await fetch('http://localhost:7438/api/extract-document', {
    method: 'POST',
    body: formData
});
const data = await resp.json();
console.log(data.fields);
```

## Response Format

```json
{
  "doc_id": "a1b2c3d4",
  "doc_type": "invoice",
  "fields": [
    {
      "field": "vendor_name",
      "value": "ACME Corp",
      "status": "grounded",
      "method": "EXACT",
      "confidence": 1.0,
      "page": 1,
      "bbox": [0.05, 0.02, 0.40, 0.04]
    },
    {
      "field": "tax_id",
      "value": null,
      "status": "blocked",
      "reason": "no grounded source found in document"
    }
  ]
}
```

**Key fields:**
- `status`: `"grounded"` (value found + cited) or `"blocked"` (refused — no source)
- `bbox`: `[x, y, width, height]` as fractions of page dimensions (0.0–1.0)
- `method`: which cascade stage matched (`EXACT`, `FUZZY`, `SEMANTIC`, `VISUAL`)
- `confidence`: 0.0–1.0 match confidence

## Ask a Question

```python
resp = requests.post(
    "http://localhost:7438/api/ask-document",
    json={"doc_id": "a1b2c3d4", "question": "What is the total amount?"}
)
answer = resp.json()
# {"status": "grounded", "answer": "$1,250.00", "anchors": [{"page": 1, "bbox": [...]}]}
```

If the question can't be answered from the document:
```json
{"status": "blocked", "reason": "no grounded source found for this question"}
```

## Integration Patterns

### Post-Ingestion Hook

Add Kairo as a post-ingestion step in your document pipeline:

```python
def after_document_upload(file_path):
    # 1. Your existing storage/indexing
    doc_record = save_to_your_db(file_path)

    # 2. Ground with Kairo
    kairo_resp = requests.post(
        "http://localhost:7438/api/extract-document",
        files={"file": open(file_path, "rb")}
    )

    # 3. Store grounded extractions with citations
    for field in kairo_resp.json()["fields"]:
        if field["status"] == "grounded":
            doc_record.add_extraction(
                field=field["field"],
                value=field["value"],
                page=field["page"],
                bbox=field["bbox"],
                confidence=field["confidence"]
            )

    return doc_record
```

### Batch Processing

```python
import concurrent.futures

def process_doc(filepath):
    resp = requests.post(
        "http://localhost:7438/api/extract-document",
        files={"file": open(filepath, "rb")}
    )
    return resp.json()

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(process_doc, filepaths))
```

## Auto Type Detection

Kairo automatically detects document type (invoice, contract, paper, generic) from content.
No need to specify a pack — the classifier uses keyword heuristics:

```python
# The classifier runs automatically on upload
# Or use it standalone:
from kairo.core.classifier import classify_document

doc_type = classify_document(text="INVOICE #1234 Total: $500")
# Returns: "invoice"
```

## Source Links

Every grounded extraction can generate a source link pointing to the exact page + bbox:

```python
from kairo.core.classifier import build_source_link

link = build_source_link(doc_id="a1b2c3d4", page=1, bbox=[0.05, 0.02, 0.40, 0.04])
# Returns a URL or reference string to the exact source location
```

## Error Handling

```python
resp = requests.post("http://localhost:7438/api/extract-document", files=...)
if resp.status_code == 200:
    data = resp.json()
    grounded = [f for f in data["fields"] if f["status"] == "grounded"]
    refused = [f for f in data["fields"] if f["status"] == "blocked"]
elif resp.status_code == 422:
    print("Unsupported file type")
elif resp.status_code == 500:
    print("Extraction failed — check sidecar logs")
```

## Running the Sidecar

```bash
# Quick start
./quickstart.sh

# Manual
python -m uvicorn kernel.sidecar.app:app --host 0.0.0.0 --port 7438

# Offline mode (no network, deterministic stub)
KAIRO_OFFLINE=1 python -m uvicorn kernel.sidecar.app:app --port 7438
```

## Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/extract-document` | POST | Upload doc → grounded extractions with bbox |
| `/api/ask-document` | POST | Ask question → answer with citation or refusal |
| `/demo` | GET | Interactive web demo (upload, view, ask) |
| `/dashboard` | GET | Live grounding trace dashboard |
| `/health` | GET | Sidecar health check |
| `/index` | POST | Index a document (internal) |
| `/extract` | POST | Extract fields by doc_id + pack (internal) |
| `/ask` | POST | Ask by doc_id + question (internal) |

---

*Every value Kairo returns is anchored to a bounding box on the source page, or it refuses. No source pixel → no answer.*