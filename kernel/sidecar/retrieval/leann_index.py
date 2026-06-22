"""D1 — LEANN graph-ANNS index for sub-linear retrieval.

LEANN (Learned Edge-Aware Nearest-Neighbor) is a graph-based approximate
nearest-neighbour index that prunes the search graph at query time via a
small learned router, achieving sub-linear recall at lower latency than
flat HNSW at large corpus sizes.

This module is a **v2.2 forward stub**.  The interface is defined and the
graceful fallback is wired; the full LEANN integration is deferred to the
D-series milestone.

GATE: pytest kernel/tests/test_d_series.py::test_leann_index_interface -v

Usage once kairo[leann] is installed::

    from kernel.sidecar.retrieval.leann_index import LeannGraphIndex
    idx = LeannGraphIndex()
    idx.index(chunks)
    results = idx.search(query_emb, top_k=5)
"""

from __future__ import annotations


class LeannGraphIndex:
    """Graph-ANNS index backed by LEANN for sub-linear approximate search.

    The index builds a proximity graph over chunk embeddings offline and
    navigates it at query time using a learned edge-pruning router, trading
    a small recall budget for significant latency reduction versus exhaustive
    HNSW at scale.

    This is a **v2.2 forward stub** — interface defined, full implementation
    deferred to D-series milestone.
    """

    def index(self, chunks: list[dict]) -> None:
        """Build the LEANN proximity graph from a list of chunk dicts.

        Each chunk dict must contain at least:
            - ``id``        (str)  : unique chunk identifier
            - ``embedding`` (list[float]) : dense vector
            - ``text``      (str)  : raw text payload

        Args:
            chunks: List of chunk dictionaries with embedding vectors.

        Raises:
            NotImplementedError: Always, until kairo[leann] is installed.
        """
        try:
            import leann  # noqa: F401 — optional dependency
        except ImportError:
            raise NotImplementedError(
                "LEANN not available: install kairo[leann]"
            )
        # Full graph-build implementation deferred to D-series milestone.
        raise NotImplementedError(
            "LEANN not available: install kairo[leann]"
        )  # pragma: no cover

    def search(self, query_emb: list[float], top_k: int = 5) -> list[dict]:
        """Search the LEANN graph for the top_k nearest chunks.

        Args:
            query_emb: Dense query vector; must match the dimensionality used
                       during :meth:`index`.
            top_k:     Number of nearest neighbours to return.

        Returns:
            List of chunk dicts (subset of what was indexed), ordered by
            ascending approximate distance.

        Raises:
            NotImplementedError: Always, until kairo[leann] is installed.
        """
        try:
            import leann  # noqa: F401 — optional dependency
        except ImportError:
            raise NotImplementedError(
                "LEANN not available: install kairo[leann]"
            )
        # Full search implementation deferred to D-series milestone.
        raise NotImplementedError(
            "LEANN not available: install kairo[leann]"
        )  # pragma: no cover
