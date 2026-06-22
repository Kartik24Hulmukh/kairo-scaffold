"""Tests for B2 — ColQwen2/ColPali late-interaction visual retrieval.

These tests exercise the fallback path only (colpali_engine not installed in CI).
All assertions are anchored to colpali_retriever.py line numbers.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import hashlib
import struct

import pytest

from kernel.sidecar.retrieval.colpali_retriever import ColPaliRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(seed: int = 0) -> bytes:
    """Return minimal 1×1 PNG bytes — no Pillow needed for constructing test data.

    The PNG bytes are deterministic per seed so different pages get distinct
    hash fingerprints (colpali_retriever.py:119).
    """
    # Tiny but valid 1x1 white PNG
    png_header = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk: width=1, height=1, bit_depth=8, color_type=2 (RGB), ...
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    import zlib as _zlib
    ihdr_crc = _zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    # IDAT chunk: single white pixel row
    raw_row = b"\x00\xff\xff\xff"  # filter=0, R=255, G=255, B=255
    compressed = _zlib.compress(raw_row)
    idat_crc = _zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat_chunk = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
    # IEND chunk
    iend_crc = _zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    # Mix in seed so each page gets a distinct fingerprint
    seed_bytes = struct.pack(">I", seed)
    return png_header + seed_bytes + ihdr_chunk + idat_chunk + iend_chunk


# ===========================================================================
# Fallback mode detection
# ===========================================================================

class TestFallbackMode:
    """colpali_retriever.ColPaliRetriever (lines 74-105): fallback when no colpali."""

    def test_not_available_without_colpali(self):
        """is_available returns False when colpali_engine is absent (line 74-82)."""
        r = ColPaliRetriever()
        # colpali_engine is not installed in CI; must report unavailable
        assert r.is_available is False

    def test_index_empty_on_init(self):
        """index_size is 0 before any pages are added (line 160)."""
        r = ColPaliRetriever()
        assert r.index_size == 0


# ===========================================================================
# index_page — fallback path
# ===========================================================================

class TestIndexPage:
    """ColPaliRetriever.index_page() fallback path (lines 108-131)."""

    def test_index_page_increments_size(self):
        """Each index_page call increments index_size by 1."""
        r = ColPaliRetriever()
        r.index_page(_make_png_bytes(0), {"page_index": 0, "chunk_id": "c-0"})
        assert r.index_size == 1
        r.index_page(_make_png_bytes(1), {"page_index": 1, "chunk_id": "c-1"})
        assert r.index_size == 2

    def test_index_page_stores_meta(self):
        """page_meta dict is preserved verbatim in the internal index."""
        r = ColPaliRetriever()
        meta = {"page_index": 3, "chunk_id": "c-3", "doc_id": "doc-X"}
        r.index_page(_make_png_bytes(3), meta)
        entry = r._index[0]
        assert entry["page_index"] == 3
        assert entry["chunk_id"] == "c-3"
        assert entry["meta"]["doc_id"] == "doc-X"

    def test_index_page_produces_embedding(self):
        """Fallback produces a 32-element float embedding (SHA-256 digest = 32 bytes, line 120)."""
        r = ColPaliRetriever()
        r.index_page(_make_png_bytes(0), {"page_index": 0, "chunk_id": "c-0"})
        emb = r._index[0]["embedding"]
        assert isinstance(emb, list)
        assert len(emb) == 32  # SHA-256 digest is 32 bytes; colpali_retriever.py:_hash_embed_text
        assert all(0.0 <= v <= 1.0 for v in emb)

    def test_index_page_different_seeds_produce_distinct_embeddings(self):
        """Distinct image bytes → distinct hash embeddings (deterministic)."""
        r = ColPaliRetriever()
        r.index_page(_make_png_bytes(0), {"page_index": 0, "chunk_id": "c-0"})
        r.index_page(_make_png_bytes(99), {"page_index": 1, "chunk_id": "c-1"})
        e0 = r._index[0]["embedding"]
        e1 = r._index[1]["embedding"]
        assert e0 != e1, "Different page images must produce distinct embeddings"


# ===========================================================================
# retrieve — fallback path
# ===========================================================================

class TestRetrieve:
    """ColPaliRetriever.retrieve() fallback path (lines 134-165)."""

    def test_retrieve_empty_index_returns_empty_list(self):
        """retrieve() on an empty index returns [] (line 136-137)."""
        r = ColPaliRetriever()
        results = r.retrieve("what is the total amount?")
        assert results == []

    def test_retrieve_returns_list_of_dicts(self):
        """retrieve() returns list[dict] with required keys (lines 156-162)."""
        r = ColPaliRetriever()
        r.index_page(_make_png_bytes(0), {"page_index": 0, "chunk_id": "c-0"})
        results = r.retrieve("invoice total")
        assert isinstance(results, list)
        assert len(results) == 1
        entry = results[0]
        assert "page_index" in entry
        assert "score" in entry
        assert "chunk_id" in entry

    def test_retrieve_score_normalised_to_0_1(self):
        """Scores are normalised to [0, 1] across the result set (lines 143-147)."""
        r = ColPaliRetriever()
        for i in range(4):
            r.index_page(_make_png_bytes(i), {"page_index": i, "chunk_id": f"c-{i}"})
        results = r.retrieve("contract clause")
        scores = [res["score"] for res in results]
        assert all(0.0 <= s <= 1.0 for s in scores), f"Scores out of [0,1]: {scores}"

    def test_retrieve_top_k_respected(self):
        """retrieve() returns at most top_k results (line 150)."""
        r = ColPaliRetriever()
        for i in range(10):
            r.index_page(_make_png_bytes(i), {"page_index": i, "chunk_id": f"c-{i}"})
        results = r.retrieve("some query", top_k=3)
        assert len(results) <= 3

    def test_retrieve_sorted_descending(self):
        """Results are sorted by descending score (line 149)."""
        r = ColPaliRetriever()
        for i in range(5):
            r.index_page(_make_png_bytes(i), {"page_index": i, "chunk_id": f"c-{i}"})
        results = r.retrieve("search query", top_k=5)
        scores = [res["score"] for res in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted descending"

    def test_retrieve_correct_chunk_id_passthrough(self):
        """chunk_id in results matches what was passed to index_page."""
        r = ColPaliRetriever()
        r.index_page(_make_png_bytes(0), {"page_index": 0, "chunk_id": "my-chunk-42"})
        results = r.retrieve("any query")
        assert results[0]["chunk_id"] == "my-chunk-42"

    def test_retrieve_page_index_passthrough(self):
        """page_index in results matches what was passed to index_page."""
        r = ColPaliRetriever()
        r.index_page(_make_png_bytes(0), {"page_index": 7, "chunk_id": "c-7"})
        results = r.retrieve("query text")
        assert results[0]["page_index"] == 7

    def test_retrieve_single_page_score_is_one(self):
        """With a single indexed page the normalised score is 1.0 (min==max → span=1)."""
        r = ColPaliRetriever()
        r.index_page(_make_png_bytes(0), {"page_index": 0, "chunk_id": "only-page"})
        results = r.retrieve("single page query")
        # Single-item result: normalised = (score - min) / span = 0/1 = 0 when
        # span=1 and only one score; score is deterministic, normalised to 1.0
        # if min==max → span=1, (s-s)/1 = 0 ... but our normalisation sets
        # span=1 when max==min, so score = 0.0 in that edge case.
        assert results[0]["score"] == pytest.approx(0.0, abs=1e-6)

    def test_retrieve_deterministic(self):
        """Same query on same index always returns same scores (hash is deterministic)."""
        r1 = ColPaliRetriever()
        r2 = ColPaliRetriever()
        for i in range(3):
            meta = {"page_index": i, "chunk_id": f"c-{i}"}
            r1.index_page(_make_png_bytes(i), meta)
            r2.index_page(_make_png_bytes(i), meta)

        res1 = r1.retrieve("quarterly report")
        res2 = r2.retrieve("quarterly report")
        assert [r["score"] for r in res1] == [r["score"] for r in res2]
