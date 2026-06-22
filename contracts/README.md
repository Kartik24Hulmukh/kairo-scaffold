# Frozen Contracts Package

This directory contains the frozen JSON schema definitions for the 9 typed contracts (interfaces) shared between the Rust core and the Python sidecar. This package prevents silent API drift between the Rust and Python implementations.

## The 9 Typed Contracts

1. **IndexRequest**: Request body format for document indexing.
2. **Chunk**: Base text unit with page coordinates and order.
3. **Extraction**: Structured domain-specific field extraction results.
4. **Answer**: Free-form response containing citations and validation status.
5. **Citation**: Exact coordinate-anchored reference mapping an answer to source text.
6. **Correction**: User override schema for caching local corrections.
7. **ProvRequest**: Request schema for fetching provenance details.
8. **ProvResponse**: Coordinate, page, and image references for a citation.
9. **HealthResponse**: Diagnostics and capabilities reporting schema.

These interfaces are enforced at the boundary using schema validation to maintain high trust and prevent regression.
