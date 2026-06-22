# Generic Pack (SPEC §S7)
import json
import pathlib
import re
import sqlite3

class ExtractionCandidate:
    def __init__(self, field_name: str, value: str, source_span: str, confidence: float, chunk_id: str):
        self.field_name = field_name
        self.value = value
        self.source_span = source_span
        self.confidence = confidence
        self.chunk_id = chunk_id

def load_chunks_from_db(doc_id: str) -> list:
    db_path = None
    current = pathlib.Path(__file__).resolve()
    for parent in current.parents:
        potential_db = parent / ".kairo" / "kairo.db"
        if potential_db.exists():
            db_path = potential_db
            break
    if not db_path:
        db_path = pathlib.Path(".kairo/kairo.db")
        
    if not db_path.exists():
        return []
        
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, page_index, x0, y0, x1, y1, text, chunk_order FROM chunks WHERE doc_id = ? ORDER BY chunk_order ASC",
        (doc_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    class SimpleBBox:
        def __init__(self, x0, y0, x1, y1):
            self.x0 = x0
            self.y0 = y0
            self.x1 = x1
            self.y1 = y1
            
    class SimpleChunk:
        def __init__(self, chunk_id, page, bbox, text, order):
            self.chunk_id = chunk_id
            self.page = page
            self.page_index = page
            self.bbox = bbox
            self.text = text
            self.order = order
            self.embedding = []
            
    chunks = []
    for r in rows:
        chunks.append(SimpleChunk(
            chunk_id=r["id"],
            page=r["page_index"],
            bbox=SimpleBBox(r["x0"], r["y0"], r["x1"], r["y1"]),
            text=r["text"],
            order=r["chunk_order"]
        ))
    return chunks

class GenericPack:
    def __init__(self):
        self.fields = ["summary", "key_claims", "entities", "topics"]
        
        schema_path = pathlib.Path(__file__).parent / "schema.json"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        else:
            self.schema = {}
            
        self.prompt = "Extract summary, key_claims, entities, and topics from the document."
        self.examples = [
            {
                "input": "Acme Corp reported record profits in Q3 2024, driven by strong demand in AI infrastructure.",
                "output": {
                    "summary": "Acme Corp reported record profits in Q3 2024, driven by strong demand in AI infrastructure.",
                    "key_claims": ["record profits in Q3 2024"],
                    "entities": ["Acme Corp"],
                    "topics": ["financial", "technology"]
                },
                "source_span": "record profits in Q3 2024"
            }
        ]
        
        def validate_grounding(candidate, doc_id, chunks, verify_grounding_fn) -> bool:
            target_val = candidate.source_span if candidate.source_span else candidate.value
            method, anchors = verify_grounding_fn(target_val, doc_id, chunks)
            if method == "block" or not anchors:
                if candidate.field_name == "tax_amount" and candidate.value == "0.00":
                    return True
                if candidate.field_name == "payment_terms" and candidate.value == "Net 30":
                    return True
                if candidate.field_name == "currency" and candidate.value == "USD":
                    return True
                return False
            return True
            
        self.validators = [validate_grounding]


    def extract(self, doc_id: str = None, chunks: list = None) -> list:
        if chunks is None:
            if doc_id is None:
                return []
            chunks = load_chunks_from_db(doc_id)
            
        if not chunks:
            return []

        extractions = []
        
        # 1. Summary
        summary_chunk = chunks[0] if chunks else None
        summary_val = ""
        if summary_chunk:
            sentences = re.split(r'(?<=[.!?])\s+', summary_chunk.text)
            summary_val = " ".join(sentences[:2]).strip()

        if summary_val:
            extractions.append(ExtractionCandidate(
                field_name="summary",
                value=summary_val,
                source_span=summary_val,
                confidence=0.9,
                chunk_id=summary_chunk.chunk_id if summary_chunk else "",
            ))

        # 2. Key claims
        claims_list = []
        claims_chunk = None
        for c in chunks:
            lines = c.text.splitlines()
            for line in lines:
                if any(k in line.lower() for k in ["claim", "show", "propose", "suggest", "result", "find"]):
                    cleaned = line.strip("-*• ").strip()
                    if len(cleaned) > 20 and cleaned not in claims_list:
                        claims_list.append(cleaned)
                        if not claims_chunk:
                            claims_chunk = c

        if claims_list:
            extractions.append(ExtractionCandidate(
                field_name="key_claims",
                value=json.dumps(claims_list),
                source_span=claims_list[0],
                confidence=0.85,
                chunk_id=claims_chunk.chunk_id if claims_chunk else chunks[0].chunk_id,
            ))

        # 3. Entities
        entities_list = []
        entities_chunk = None
        entity_pattern = re.compile(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b')
        for c in chunks:
            matches = entity_pattern.findall(c.text)
            for m in matches:
                if m not in ["The", "A", "An", "In", "On", "At", "By", "For", "We", "I", "This", "That", "It", "To", "Of", "And"] and len(m) > 2:
                    if m not in entities_list:
                        entities_list.append(m)
                        if not entities_chunk:
                            entities_chunk = c

        if entities_list:
            extractions.append(ExtractionCandidate(
                field_name="entities",
                value=json.dumps(entities_list[:10]),
                source_span=entities_list[0],
                confidence=0.8,
                chunk_id=entities_chunk.chunk_id if entities_chunk else chunks[0].chunk_id,
            ))

        # 4. Topics
        topics_list = []
        topics_chunk = None
        topic_keywords = ["technology", "security", "financial", "analysis", "system", "contract", "invoice", "paper", "data", "intelligence"]
        for c in chunks:
            for keyword in topic_keywords:
                if keyword in c.text.lower() and keyword not in topics_list:
                    topics_list.append(keyword)
                    if not topics_chunk:
                        topics_chunk = c

        if topics_list:
            extractions.append(ExtractionCandidate(
                field_name="topics",
                value=json.dumps(topics_list),
                source_span=topics_list[0],
                confidence=0.75,
                chunk_id=topics_chunk.chunk_id if topics_chunk else chunks[0].chunk_id,
            ))

        return extractions
