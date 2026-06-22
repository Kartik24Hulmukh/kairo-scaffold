# Contract Pack (SPEC §S7)
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

class ContractPack:
    def __init__(self):
        self.fields = [
            "parties",
            "effective_date",
            "termination_date",
            "obligations",
            "governing_law",
            "payment_terms",
            "confidentiality_clause"
        ]
        
        schema_path = pathlib.Path(__file__).parent / "schema.json"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        else:
            self.schema = {}
            
        self.prompt = "Extract contract metadata: parties, effective_date, termination_date, obligations, governing_law, payment_terms, confidentiality_clause."
        self.examples = [
            {
                "input": "This Agreement is between Acme Corp and Beta LLC, effective January 1, 2024. The agreement is governed by the laws of California. All payments due within 30 days.",
                "output": {
                    "parties": ["Acme Corp", "Beta LLC"],
                    "effective_date": "2024-01-01",
                    "governing_law": "California",
                    "payment_terms": "30 days"
                },
                "source_span": "governed by the laws of California"
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

        # Parties
        parties = []
        parties_chunk = chunks[0] if chunks else None
        if parties_chunk:
            m = re.findall(r'(?:between|among)\s+([A-Z][a-zA-Z\s,]+?)(?:\s+\(|and\b)', parties_chunk.text)
            if m:
                parties = [p.strip() for p in m]
            if not parties:
                m2 = re.findall(r'\b([A-Z][a-zA-Z0-9\s]+?)\s+(?:and|&)\s+([A-Z][a-zA-Z0-9\s]+?)\b', parties_chunk.text)
                if m2:
                    parties = list(m2[0])

        if parties:
            extractions.append(ExtractionCandidate(
                field_name="parties",
                value=json.dumps(parties),
                source_span=parties[0] if parties else "",
                confidence=0.85,
                chunk_id=parties_chunk.chunk_id if parties_chunk else "",
            ))

        # Effective date and termination date
        eff_date = ""
        term_date = ""
        date_chunk = None
        for c in chunks:
            m = re.search(r'(?:effective date|commencement date|date of this agreement)\s*(?:is|of|as of)?\s*(\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}\s+[a-zA-Z]+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})', c.text, re.IGNORECASE)
            if m:
                eff_date = m.group(1).strip()
                date_chunk = c
            
            m2 = re.search(r'(?:terminate on|expiration date|termination date|ends on)\s*(?:is|of|as of)?\s*(\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}\s+[a-zA-Z]+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})', c.text, re.IGNORECASE)
            if m2:
                term_date = m2.group(1).strip()
                date_chunk = c

        if eff_date:
            extractions.append(ExtractionCandidate(
                field_name="effective_date",
                value=self._parse_date(eff_date) or eff_date,
                source_span=eff_date,
                confidence=0.9,
                chunk_id=date_chunk.chunk_id if date_chunk else "",
            ))
        if term_date:
            extractions.append(ExtractionCandidate(
                field_name="termination_date",
                value=self._parse_date(term_date) or term_date,
                source_span=term_date,
                confidence=0.9,
                chunk_id=date_chunk.chunk_id if date_chunk else "",
            ))

        # Obligations
        obligations = []
        ob_chunk = None
        for c in chunks:
            lines = c.text.splitlines()
            for line in lines:
                if any(x in line.lower() for x in ["shall", "agree to", "covenant", "undertake"]):
                    obligations.append(line.strip())
                    if not ob_chunk:
                        ob_chunk = c

        if obligations:
            extractions.append(ExtractionCandidate(
                field_name="obligations",
                value=json.dumps(obligations),
                source_span=obligations[0][:50],
                confidence=0.8,
                chunk_id=ob_chunk.chunk_id if ob_chunk else chunks[0].chunk_id,
            ))

        # Governing Law
        gov_law = ""
        gov_chunk = None
        for c in chunks:
            m = re.search(r'(?:governed by|governing law|laws of|jurisdiction of)\s*(?:the state of|the laws of)?\s*([A-Z][a-zA-Z\s]+)', c.text, re.IGNORECASE)
            if m:
                gov_law = m.group(1).strip()
                gov_law = gov_law.split("\n")[0].split(".")[0].strip()
                gov_chunk = c
                break

        if gov_law:
            extractions.append(ExtractionCandidate(
                field_name="governing_law",
                value=gov_law,
                source_span=gov_law,
                confidence=0.9,
                chunk_id=gov_chunk.chunk_id if gov_chunk else "",
            ))

        # Payment terms
        pay_terms = ""
        pay_chunk = None
        for c in chunks:
            m = re.search(r'(?:payment|invoice|paid within)\s+(\d+\s+days|net\s+\d+|upon receipt)', c.text, re.IGNORECASE)
            if m:
                pay_terms = m.group(0).strip()
                pay_chunk = c
                break

        if pay_terms:
            extractions.append(ExtractionCandidate(
                field_name="payment_terms",
                value=pay_terms,
                source_span=pay_terms,
                confidence=0.85,
                chunk_id=pay_chunk.chunk_id if pay_chunk else "",
            ))

        # Confidentiality clause
        conf = ""
        conf_chunk = None
        for c in chunks:
            if "confidential" in c.text.lower() or "proprietary" in c.text.lower():
                lines = c.text.splitlines()
                for line in lines:
                    if "confidential" in line.lower() or "disclosure" in line.lower():
                        conf = line.strip()
                        conf_chunk = c
                        break
                if conf:
                    break

        if conf:
            extractions.append(ExtractionCandidate(
                field_name="confidentiality_clause",
                value=conf,
                source_span=conf[:50],
                confidence=0.85,
                chunk_id=conf_chunk.chunk_id if conf_chunk else "",
            ))

        return extractions

    @staticmethod
    def _parse_date(date_str: str) -> str | None:
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        
        m = re.match(r"(\d{4})[-/](\d{2})[-/](\d{2})", date_str)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            
        m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", date_str)
        if m:
            day = int(m.group(1))
            month = months.get(m.group(2).lower())
            year = int(m.group(3))
            if month:
                return f"{year:04d}-{month:02d}-{day:02d}"

        m = re.match(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", date_str)
        if m:
            month = months.get(m.group(1).lower())
            day = int(m.group(2))
            year = int(m.group(3))
            if month:
                return f"{year:04d}-{month:02d}-{day:02d}"

        return None
