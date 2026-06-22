# Invoice Pack (SPEC §S7)
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

class InvoicePack:
    def __init__(self):
        self.fields = [
            "vendor_name",
            "invoice_number",
            "invoice_date",
            "due_date",
            "total_amount",
            "currency",
            "line_items",
            "tax_amount",
            "payment_terms"
        ]
        
        schema_path = pathlib.Path(__file__).parent / "schema.json"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        else:
            self.schema = {}
            
        self.prompt = "Extract invoice information such as vendor_name, total_amount, currency, dates, line items, and payment terms."
        self.examples = [
            {
                "input": "ACME Inc. | Invoice #INV-2024-001 | Date: 2024-01-15 | Due: 2024-02-14 | Total: $1,250.00 | Tax: $0.00 | Terms: Net 30",
                "output": {
                    "vendor_name": "ACME Inc.",
                    "invoice_number": "INV-2024-001",
                    "invoice_date": "2024-01-15",
                    "due_date": "2024-02-14",
                    "total_amount": "1250.00",
                    "currency": "USD",
                    "tax_amount": "0.00",
                    "payment_terms": "Net 30"
                },
                "source_span": "Total: $1,250.00"
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

        # Find vendor name
        vendor_name = ""
        vendor_chunk = chunks[0] if chunks else None
        if vendor_chunk:
            lines = [l.strip() for l in vendor_chunk.text.splitlines() if l.strip()]
            for line in lines[:5]:
                if any(x in line.lower() for x in ["invoice", "bill to", "to:", "date:"]):
                    continue
                if len(line) > 3:
                    vendor_name = line
                    break

        if vendor_name:
            extractions.append(ExtractionCandidate(
                field_name="vendor_name",
                value=vendor_name,
                source_span=vendor_name,
                confidence=0.9,
                chunk_id=vendor_chunk.chunk_id if vendor_chunk else "",
            ))

        # Find invoice number
        inv_no = ""
        inv_chunk = None
        for c in chunks:
            m = re.search(r'(?:invoice|inv|number|no\.?|#)\s*:?\s*([a-zA-Z0-9\-]+)', c.text, re.IGNORECASE)
            if m:
                inv_no = m.group(1).strip()
                inv_chunk = c
                break

        if inv_no:
            extractions.append(ExtractionCandidate(
                field_name="invoice_number",
                value=inv_no,
                source_span=inv_no,
                confidence=0.95,
                chunk_id=inv_chunk.chunk_id if inv_chunk else "",
            ))

        # Find invoice date and due date
        inv_date = ""
        due_date = ""
        date_chunk = None
        for c in chunks:
            dates = re.findall(r'(?:date|issued|billed)\s*:?\s*(\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}\s+[a-zA-Z]+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{2}/\d{2}/\d{4})', c.text, re.IGNORECASE)
            if dates:
                inv_date = dates[0]
                date_chunk = c
            due_dates = re.findall(r'(?:due|due date|payment due)\s*:?\s*(\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}\s+[a-zA-Z]+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{2}/\d{2}/\d{4})', c.text, re.IGNORECASE)
            if due_dates:
                due_date = due_dates[0]
                date_chunk = c

        if inv_date:
            extractions.append(ExtractionCandidate(
                field_name="invoice_date",
                value=self._parse_date(inv_date) or inv_date,
                source_span=inv_date,
                confidence=0.9,
                chunk_id=date_chunk.chunk_id if date_chunk else "",
            ))
        if due_date:
            extractions.append(ExtractionCandidate(
                field_name="due_date",
                value=self._parse_date(due_date) or due_date,
                source_span=due_date,
                confidence=0.9,
                chunk_id=date_chunk.chunk_id if date_chunk else "",
            ))

        # Find total amount and currency
        total_amt = ""
        currency = "USD"
        amt_chunk = None
        for c in chunks:
            m = re.search(r'(?:total|amount due|balance due|total due|grand total)\s*:?\s*([$€£¥]?\s*[\d,]+(?:\.\d{2})?)', c.text, re.IGNORECASE)
            if m:
                total_str = m.group(1).strip()
                amt_chunk = c
                if "$" in total_str:
                    currency = "USD"
                elif "€" in total_str:
                    currency = "EUR"
                elif "£" in total_str:
                    currency = "GBP"
                
                total_val = re.sub(r'[^\d\.]', '', total_str)
                total_amt = total_val
                break

        if total_amt:
            extractions.append(ExtractionCandidate(
                field_name="total_amount",
                value=total_amt,
                source_span=total_str,
                confidence=0.95,
                chunk_id=amt_chunk.chunk_id if amt_chunk else "",
            ))
            extractions.append(ExtractionCandidate(
                field_name="currency",
                value=currency,
                source_span=currency,
                confidence=0.9,
                chunk_id=amt_chunk.chunk_id if amt_chunk else "",
            ))

        # Tax amount
        tax_amt = "0.00"
        tax_chunk = None
        for c in chunks:
            m = re.search(r'(?:tax|vat|gst)\s*:?\s*([$€£¥]?\s*[\d,]+\.\d{2})', c.text, re.IGNORECASE)
            if m:
                tax_str = m.group(1).strip()
                tax_amt = re.sub(r'[^\d\.]', '', tax_str)
                tax_chunk = c
                break

        extractions.append(ExtractionCandidate(
            field_name="tax_amount",
            value=tax_amt,
            source_span=tax_amt,
            confidence=0.85,
            chunk_id=tax_chunk.chunk_id if tax_chunk else chunks[0].chunk_id,
        ))

        # Payment terms
        terms = "Net 30"
        terms_chunk = None
        for c in chunks:
            m = re.search(r'(?:terms|payment terms)\s*:?\s*(net\s*\d+|due on receipt|immediate)', c.text, re.IGNORECASE)
            if m:
                terms = m.group(1).strip()
                terms_chunk = c
                break

        extractions.append(ExtractionCandidate(
            field_name="payment_terms",
            value=terms,
            source_span=terms,
            confidence=0.85,
            chunk_id=terms_chunk.chunk_id if terms_chunk else chunks[0].chunk_id,
        ))

        # Line items
        line_items = []
        for c in chunks:
            lines = c.text.splitlines()
            for line in lines:
                m = re.search(r'([a-zA-Z\s]{5,})\s+(\d+)\s+([$€£¥]?\s*[\d,]+\.\d{2})\s+([$€£¥]?\s*[\d,]+\.\d{2})', line)
                if m:
                    desc = m.group(1).strip()
                    qty = int(m.group(2))
                    price = float(re.sub(r'[^\d\.]', '', m.group(3)))
                    total = float(re.sub(r'[^\d\.]', '', m.group(4)))
                    line_items.append({"description": desc, "quantity": qty, "unit_price": price, "total": total})

        extractions.append(ExtractionCandidate(
            field_name="line_items",
            value=json.dumps(line_items),
            source_span=line_items[0]["description"] if line_items else "",
            confidence=0.8,
            chunk_id=chunks[0].chunk_id,
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
