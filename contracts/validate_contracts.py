import os
import json
import sys

try:
    import jsonschema
except ImportError:
    print("jsonschema library is missing. Please install it to validate contracts.")
    sys.exit(1)

SCHEMAS_DIR = os.path.join(os.path.dirname(__file__), "schemas")

def load_schema(name):
    path = os.path.join(SCHEMAS_DIR, f"{name.lower()}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_validation():
    # 1. IndexRequest
    index_request_schema = load_schema("indexrequest")
    sample_index_request = {
        "doc_id": "doc_123",
        "source_path": "/path/to/doc.pdf",
        "pages": [
            {"width_px": 800.0, "height_px": 1000.0}
        ],
        "chunks": [
            {
                "id": "chunk_abc",
                "text": "This is a sample document chunk text.",
                "bbox": {"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.6},
                "page_index": 0
            }
        ]
    }
    jsonschema.validate(instance=sample_index_request, schema=index_request_schema)
    print("IndexRequest schema validation: PASS")

    # 2. Chunk
    chunk_schema = load_schema("chunk")
    sample_chunk = {
        "id": "chunk_abc",
        "doc_id": "doc_123",
        "page_index": 0,
        "bbox": {"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.6},
        "text": "This is a sample document chunk text.",
        "chunk_order": 1
    }
    jsonschema.validate(instance=sample_chunk, schema=chunk_schema)
    print("Chunk schema validation: PASS")

    # 3. Extraction
    extraction_schema = load_schema("extraction")
    sample_extraction = {
        "id": "ext_001",
        "doc_id": "doc_123",
        "field": "vendor",
        "value": "TechCorp Ltd",
        "confidence": 0.95,
        "status": "grounded",
        "method": "exact",
        "anchors": [
            {
                "chunk_id": "chunk_abc",
                "page": 0,
                "bbox": {"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.6},
                "char_start": 0,
                "char_end": 12
            }
        ]
    }
    jsonschema.validate(instance=sample_extraction, schema=extraction_schema)
    print("Extraction schema validation: PASS")

    # 4. Answer
    answer_schema = load_schema("answer")
    sample_answer = {
        "id": "ans_999",
        "query": "Who is the vendor?",
        "text": "TechCorp Ltd",
        "grounded": True,
        "citations": [
            {
                "chunk_id": "chunk_abc",
                "page": 0,
                "bbox": {"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.6},
                "char_start": 0,
                "char_end": 12
            }
        ]
    }
    jsonschema.validate(instance=sample_answer, schema=answer_schema)
    print("Answer schema validation: PASS")

if __name__ == "__main__":
    try:
        run_validation()
        print("All contract schema validations passed successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"Contract validation failure: {e}", file=sys.stderr)
        sys.exit(1)
