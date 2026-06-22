import sys
import os
import argparse
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple

# Initialize FastAPI app
app = FastAPI(title="Kairo Mock Sidecar")

class IndexRequest(BaseModel):
    path: str

class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

class ParserPage(BaseModel):
    index: int
    width_px: int
    height_px: int
    image_sha256: str

class ParserChunk(BaseModel):
    page: int
    bbox: BBox
    text: str
    source_type: str

class ParseResponse(BaseModel):
    doc_id: str
    pages: List[ParserPage]
    chunks: List[ParserChunk]

class IndexResponsePage(BaseModel):
    index: int
    width_px: int
    height_px: int
    image_sha256: str

class IndexResponseChunk(BaseModel):
    text: str
    page_index: int
    bbox: BBox
    order: int

class IndexResponse(BaseModel):
    doc_id: str
    pages: int
    chunks: int
    pages_list: List[IndexResponsePage]
    chunks_list: List[IndexResponseChunk]

class Anchor(BaseModel):
    chunk_id: str
    char_span: Tuple[int, int]
    page: int
    bbox: BBox

class Extraction(BaseModel):
    id: str
    doc_id: str
    field: str
    value: str
    confidence: float
    status: str
    anchors: List[Anchor]
    method: str

class ExtractRequest(BaseModel):
    doc_id: str
    pack: str

class AskRequest(BaseModel):
    doc_id: str
    query: str

class Answer(BaseModel):
    id: str
    query: str
    text: str
    citations: List[Anchor]
    grounded: bool

class ProvenanceResponse(BaseModel):
    page: int
    bbox: BBox
    char_span: Tuple[int, int]
    image_ref: str

class CorrectRequest(BaseModel):
    extraction_id: str
    new_value: str

class Correction(BaseModel):
    extraction_id: str
    old_value: str
    new_value: str
    by: str
    at: datetime

@app.post("/parse", response_model=ParseResponse)
def parse_doc(req: IndexRequest):
    seed = int(os.environ.get("KAIRO_MOCK_SEED", "42"))
    doc_id = f"mock_{uuid.uuid5(uuid.NAMESPACE_DNS, req.path).hex[:12]}"
    return ParseResponse(
        doc_id=doc_id,
        pages=[
            ParserPage(index=1, width_px=800, height_px=1000, image_sha256=f"img_sha_{seed}_1"),
            ParserPage(index=2, width_px=800, height_px=1000, image_sha256=f"img_sha_{seed}_2")
        ],
        chunks=[
            ParserChunk(page=1, bbox=BBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5), text=f"Deterministic Chunk 1 with seed {seed}", source_type="text"),
            ParserChunk(page=1, bbox=BBox(x0=0.5, y0=0.0, x1=1.0, y1=0.5), text=f"Deterministic Chunk 2 with seed {seed}", source_type="text"),
            ParserChunk(page=2, bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0), text=f"Deterministic Chunk 3 with seed {seed}", source_type="text")
        ]
    )

@app.post("/index", response_model=IndexResponse)
def index_doc(req: IndexRequest):
    seed = int(os.environ.get("KAIRO_MOCK_SEED", "42"))
    doc_id = f"mock_{uuid.uuid5(uuid.NAMESPACE_DNS, req.path).hex[:12]}"
    return IndexResponse(
        doc_id=doc_id,
        pages=2,
        chunks=3,
        pages_list=[
            IndexResponsePage(index=1, width_px=800, height_px=1000, image_sha256=f"img_sha_{seed}_1"),
            IndexResponsePage(index=2, width_px=800, height_px=1000, image_sha256=f"img_sha_{seed}_2")
        ],
        chunks_list=[
            IndexResponseChunk(text=f"Deterministic Chunk 1 with seed {seed}", page_index=1, bbox=BBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5), order=0),
            IndexResponseChunk(text=f"Deterministic Chunk 2 with seed {seed}", page_index=1, bbox=BBox(x0=0.5, y0=0.0, x1=1.0, y1=0.5), order=1),
            IndexResponseChunk(text=f"Deterministic Chunk 3 with seed {seed}", page_index=2, bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0), order=2)
        ]
    )

@app.post("/extract", response_model=List[Extraction])
def extract_fields(req: ExtractRequest):
    seed = int(os.environ.get("KAIRO_MOCK_SEED", "42"))
    if req.pack.lower() == "invoice":
        return [
            Extraction(
                id="ext_inv_vendor",
                doc_id=req.doc_id,
                field="vendor",
                value=f"Mock Vendor {seed}",
                confidence=0.95,
                status="suggested",
                anchors=[
                    Anchor(
                        chunk_id=f"{req.doc_id}_p1_c0",
                        char_span=(0, 11),
                        page=1,
                        bbox=BBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
                    )
                ],
                method="exact"
            ),
            Extraction(
                id="ext_inv_total",
                doc_id=req.doc_id,
                field="total_amount",
                value="100.00",
                confidence=0.99,
                status="suggested",
                anchors=[
                    Anchor(
                        chunk_id=f"{req.doc_id}_p1_c1",
                        char_span=(0, 6),
                        page=1,
                        bbox=BBox(x0=0.5, y0=0.0, x1=1.0, y1=0.5)
                    )
                ],
                method="exact"
            )
        ]
    else:
        return [
            Extraction(
                id="ext_gen_field",
                doc_id=req.doc_id,
                field="generic_field",
                value=f"Mock Value {seed}",
                confidence=0.90,
                status="suggested",
                anchors=[
                    Anchor(
                        chunk_id=f"{req.doc_id}_p2_c2",
                        char_span=(0, 10),
                        page=2,
                        bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
                    )
                ],
                method="exact"
            )
        ]

@app.post("/ask", response_model=Answer)
def ask_question(req: AskRequest):
    import hashlib
    seed = int(os.environ.get("KAIRO_MOCK_SEED", "42"))
    ans_hash = hashlib.sha256(f"{seed}_{req.query}".encode()).hexdigest()
    ans_id = f"ans_{ans_hash[:12]}"
    if "unanswerable" in req.query.lower() or "refuse" in req.query.lower():
        return Answer(
            id=ans_id,
            query=req.query,
            text="blocked",
            citations=[],
            grounded=False
        )
    
    return Answer(
        id=ans_id,
        query=req.query,
        text=f"Mock answer text for query '{req.query}' under seed {seed}",
        citations=[
            Anchor(
                chunk_id=f"{req.doc_id}_p1_c0",
                char_span=(0, 10),
                page=1,
                bbox=BBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
            )
        ],
        grounded=True
    )

@app.get("/provenance/{extraction_id}", response_model=ProvenanceResponse)
def get_provenance(extraction_id: str):
    seed = int(os.environ.get("KAIRO_MOCK_SEED", "42"))
    return ProvenanceResponse(
        page=1,
        bbox=BBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5),
        char_span=(0, 10),
        image_ref=f"img_sha_{seed}_1"
    )

@app.post("/correct", response_model=Correction)
def correct_field(req: CorrectRequest):
    return Correction(
        extraction_id=req.extraction_id,
        old_value="old_mock_value",
        new_value=req.new_value,
        by="user",
        at=datetime.now(timezone.utc)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7439)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seed", type=int, default=42)
    args, unknown = parser.parse_known_args()
    os.environ["KAIRO_MOCK_SEED"] = str(args.seed)
    
    import uvicorn
    uvicorn.run("tests.mock_models.mock_sidecar:app", host=args.host, port=args.port, reload=False)
