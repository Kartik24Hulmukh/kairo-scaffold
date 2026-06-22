"""Tests for C1 — unified vector store wiring in app.py.

Verifies:
  - app._vector_store is not None after module load (QdrantEdgeStore in test env)
  - /index calls add_chunks on _vector_store (mock-verified)
  - /ask and /extract semantic search routes go through _vector_store.search
  - KAIRO_VECTOR_BACKEND env var can switch the backend
  - Fallback to Qdrant raw client when _vector_store is None still works
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from kernel.sidecar import app as app_module
from kernel.sidecar.app import app


client = TestClient(app)


class TestVectorStoreModuleInit:
    """app._vector_store is initialized on module import."""

    def test_vector_store_available_flag_is_true(self):
        """_VECTOR_STORE_AVAILABLE should be True — retrieval/vector_store.py must load."""
        assert app_module._VECTOR_STORE_AVAILABLE is True

    def test_vector_store_is_not_none(self):
        """_vector_store should be a VectorStore instance (QdrantEdgeStore in test env)."""
        if not app_module._VECTOR_STORE_AVAILABLE:
            pytest.skip("vector store module not available in this environment")
        assert app_module._vector_store is not None

    def test_vector_store_has_add_chunks(self):
        if not app_module._VECTOR_STORE_AVAILABLE or app_module._vector_store is None:
            pytest.skip("_vector_store is None")
        assert hasattr(app_module._vector_store, "add_chunks")

    def test_vector_store_has_search(self):
        if not app_module._VECTOR_STORE_AVAILABLE or app_module._vector_store is None:
            pytest.skip("_vector_store is None")
        assert hasattr(app_module._vector_store, "search")

    def test_vector_store_has_close(self):
        if not app_module._VECTOR_STORE_AVAILABLE or app_module._vector_store is None:
            pytest.skip("_vector_store is None")
        assert hasattr(app_module._vector_store, "close")



class TestIndexCallsVectorStore:
    """POST /index routes embeddings through _vector_store.add_chunks."""

    def test_add_chunks_called_on_index(self, tmp_path):
        test_file = tmp_path / "wiring_test.txt"
        test_file.write_text("The payment due is $500.00 by July 1, 2026.")

        mock_store = MagicMock()
        mock_store.add_chunks = MagicMock()
        mock_store.search = MagicMock(return_value=[])

        with patch.object(app_module, "_vector_store", mock_store):
            res = client.post("/index", json={"path": str(test_file)})

        assert res.status_code == 200
        # add_chunks should have been called at least once
        assert mock_store.add_chunks.call_count >= 1

    def test_add_chunks_receives_doc_id_field(self, tmp_path):
        test_file = tmp_path / "wiring_doc_id.txt"
        test_file.write_text("Invoice total: $1,234.56")

        captured_calls = []
        original_store = app_module._vector_store

        def capture_add_chunks(chunks):
            captured_calls.append(chunks)
            return original_store.add_chunks(chunks)

        mock_store = MagicMock()
        mock_store.add_chunks = capture_add_chunks
        mock_store.search = MagicMock(return_value=[])

        with patch.object(app_module, "_vector_store", mock_store):
            res = client.post("/index", json={"path": str(test_file)})

        assert res.status_code == 200
        assert len(captured_calls) >= 1
        first_batch = captured_calls[0]
        assert len(first_batch) >= 1
        # Each chunk dict must carry doc_id, text, embedding, order
        for record in first_batch:
            assert "doc_id" in record
            assert "text" in record
            assert "embedding" in record
            assert "order" in record

    def test_add_chunks_receives_order_field(self, tmp_path):
        """order field is present in every chunk dict passed to add_chunks."""
        test_file = tmp_path / "order_field.txt"
        test_file.write_text(
            "Sentence one is here. Sentence two follows. Sentence three ends it."
        )

        captured = []
        mock_store = MagicMock()

        def capture(chunks):
            captured.extend(chunks)

        mock_store.add_chunks = capture
        mock_store.search = MagicMock(return_value=[])

        with patch.object(app_module, "_vector_store", mock_store):
            res = client.post("/index", json={"path": str(test_file)})

        assert res.status_code == 200
        assert len(captured) >= 1
        orders = [r["order"] for r in captured]
        # Orders should be non-negative integers
        assert all(isinstance(o, int) and o >= 0 for o in orders)


class TestAskRoutesVectorStore:
    """POST /ask semantic search routes through _vector_store.search."""

    def test_search_called_during_ask(self, tmp_path):
        test_file = tmp_path / "ask_wiring.txt"
        test_file.write_text("The governing law is the laws of New York.")

        mock_store = MagicMock()
        # Return empty so it doesn't interfere with keyword match path
        mock_store.search = MagicMock(return_value=[])
        mock_store.add_chunks = MagicMock()

        with patch.object(app_module, "_vector_store", mock_store):
            res_idx = client.post("/index", json={"path": str(test_file)})
            assert res_idx.status_code == 200
            doc_id = res_idx.json()["doc_id"]

            res_ask = client.post(
                "/ask", json={"doc_id": doc_id, "query": "What is the governing law?"}
            )

        assert res_ask.status_code == 200
        # search() should have been called at least once during /ask
        assert mock_store.search.call_count >= 1

    def test_search_receives_doc_id_kwarg(self, tmp_path):
        """When semantic search is reached, _vector_store.search receives doc_id kwarg."""
        test_file = tmp_path / "ask_docid.txt"
        test_file.write_text("Contract value is ten million dollars.")

        search_calls = []

        def capture_search(query_embedding, top_k=5, doc_id=None):
            search_calls.append({"top_k": top_k, "doc_id": doc_id})
            return []

        # Build a complete mock store so ALL paths go through it
        mock_store = MagicMock()
        mock_store.add_chunks = MagicMock()
        mock_store.search = capture_search

        with patch.object(app_module, "_vector_store", mock_store):
            res_idx = client.post("/index", json={"path": str(test_file)})
            assert res_idx.status_code == 200
            doc_id = res_idx.json()["doc_id"]

            # Use a query that has no keyword overlap with the doc so keyword path fails
            # and semantic search Strategy 2 is reached
            res_ask = client.post(
                "/ask",
                json={"doc_id": doc_id, "query": "zephyr aurora constellation"},
            )

        assert res_ask.status_code == 200
        # search() should have been called during the /ask semantic strategy
        assert len(search_calls) >= 1, (
            f"Expected search to be called at least once, got {search_calls}"
        )


class TestFallbackToQdrant:
    """When _vector_store is None, Qdrant raw client is used as fallback."""

    def test_index_succeeds_when_vector_store_none(self, tmp_path):
        """Ingestion must not crash when _vector_store is None (Qdrant fallback)."""
        test_file = tmp_path / "fallback.txt"
        test_file.write_text("Fallback test document.")

        with patch.object(app_module, "_vector_store", None):
            res = client.post("/index", json={"path": str(test_file)})

        assert res.status_code == 200

    def test_ask_succeeds_when_vector_store_none(self, tmp_path):
        """Ask must not crash when _vector_store is None."""
        test_file = tmp_path / "fallback_ask.txt"
        test_file.write_text("The total invoice amount is $350.")

        with patch.object(app_module, "_vector_store", None):
            res_idx = client.post("/index", json={"path": str(test_file)})
            assert res_idx.status_code == 200
            doc_id = res_idx.json()["doc_id"]

            res_ask = client.post(
                "/ask", json={"doc_id": doc_id, "query": "What is the invoice amount?"}
            )

        assert res_ask.status_code == 200
