from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
from datetime import datetime, timezone

app = FastAPI(title="Kairo Sidecar Stub")

class IndexRequest(BaseModel):
    path: str

class IndexResponse(BaseModel):
    doc_id: str
    pages: int
    chunks: int

class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

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

@app.post("/index", response_model=IndexResponse)
def index_doc(req: IndexRequest):
    return IndexResponse(doc_id="doc_123", pages=1, chunks=5)

@app.post("/extract", response_model=List[Extraction])
def extract_fields(req: ExtractRequest):
    return []

@app.post("/ask", response_model=Answer)
def ask_question(req: AskRequest):
    return Answer(id="ans_123", query=req.query, text="Stub answer", citations=[], grounded=True)

@app.get("/provenance/{extraction_id}", response_model=ProvenanceResponse)
def get_provenance(extraction_id: str):
    return ProvenanceResponse(
        page=1,
        bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        char_span=(0, 10),
        image_ref="img_123"
    )

@app.post("/correct", response_model=Correction)
def correct_field(req: CorrectRequest):
    return Correction(
        extraction_id=req.extraction_id,
        old_value="old",
        new_value=req.new_value,
        by="user",
        at=datetime.now(timezone.utc)
    )
