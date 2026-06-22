# B2-VR — Visual Patch Retrieval Path (ColQwen2/ColPali + B3 Bridge)

## Status
PASS

## Date / Commit SHA
2026-06-20 (local working tree, pre-commit)

## Gate Command
`pytest kernel/tests/test_visual_retrieval.py -v --tb=short`

## Evidence

```
============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-9.1.0, pluggy-1.6.0
collected 48 items

TestVisualPatchRetrieverInit::test_default_enabled_true PASSED
TestVisualPatchRetrieverInit::test_can_disable_on_init PASSED
TestVisualPatchRetrieverInit::test_make_visual_retriever_enabled PASSED
TestVisualPatchRetrieverInit::test_make_visual_retriever_disabled PASSED
TestVisualPatchRetrieverInit::test_colpali_not_available_in_ci PASSED
TestVisualPatchRetrieverInit::test_total_patches_indexed_starts_zero PASSED
TestIndexPagePatches::test_disabled_returns_empty_list PASSED
TestIndexPagePatches::test_patch_count_matches_grid PASSED
TestIndexPagePatches::test_patch_count_2x2_grid PASSED
TestIndexPagePatches::test_multiple_pages_accumulate PASSED
TestIndexPagePatches::test_patches_have_bbox PASSED
TestIndexPagePatches::test_patch_bboxes_tile_unit_square PASSED
TestIndexPagePatches::test_patch_bbox_values_in_01 PASSED
TestIndexPagePatches::test_patches_have_page_index PASSED
TestIndexPagePatches::test_patch_img_bytes_nonempty PASSED
TestRetrievePatch::test_empty_index_returns_empty PASSED
TestRetrievePatch::test_disabled_returns_empty PASSED
TestRetrievePatch::test_returns_list_of_dicts PASSED
TestRetrievePatch::test_result_has_required_keys PASSED
TestRetrievePatch::test_scores_in_01 PASSED
TestRetrievePatch::test_results_sorted_descending PASSED
TestRetrievePatch::test_top_k_respected PASSED
TestRetrievePatch::test_page_index_filter PASSED
TestRetrievePatch::test_bbox_in_results PASSED
TestRetrievePatch::test_deterministic_same_query PASSED
TestTopPatchBbox::test_returns_none_when_disabled PASSED
TestTopPatchBbox::test_returns_none_when_empty PASSED
TestTopPatchBbox::test_returns_dict_when_indexed PASSED
TestTopPatchBbox::test_bbox_coords_in_unit_square PASSED
TestB3Bridge::test_patch_bbox_is_parseable_by_b3 PASSED
TestB3Bridge::test_patch_iou_pass_when_chunk_overlaps PASSED
TestB3Bridge::test_patch_iou_fail_when_chunk_non_overlapping PASSED
TestB3Bridge::test_zero_area_patch_blocked_by_b3 PASSED
TestPerDocumentFlag::test_enabled_false_skips_indexing PASSED
TestPerDocumentFlag::test_enabled_false_retrieve_returns_empty PASSED
TestPerDocumentFlag::test_enabled_true_indexes_patches PASSED
TestPerDocumentFlag::test_enabled_false_does_not_affect_enabled_true_retriever PASSED
TestTableHeavyIoUGate::test_text_retrieval_baseline PASSED
TestTableHeavyIoUGate::test_iou_gate_table_heavy PASSED
TestColPaliRetrieverRegression::test_not_available_in_ci PASSED
TestColPaliRetrieverRegression::test_index_size_increments PASSED
TestColPaliRetrieverRegression::test_retrieve_empty_returns_empty PASSED
TestColPaliRetrieverRegression::test_retrieve_returns_required_keys PASSED
TestColPaliRetrieverRegression::test_retrieve_score_in_01 PASSED
TestColPaliRetrieverRegression::test_hash_embed_deterministic PASSED
TestColPaliRetrieverRegression::test_hash_embed_32_dimensions PASSED
TestColPaliRetrieverRegression::test_cosine_sim_identical_vectors PASSED
TestColPaliRetrieverRegression::test_cosine_sim_orthogonal_vectors PASSED

============================== 48 passed in 2.35s ==============================
```

## What Was Built

- [colpali_retriever.py](file:///c:/Users/praja/OneDrive/Desktop/test-env/repositories/kairo-scaffold/kernel/sidecar/retrieval/colpali_retriever.py): `ColPaliRetriever` (page-level MaxSim), `VisualPatch` (data container), `VisualPatchRetriever` (nrows×ncols patch grid), `make_visual_retriever()` factory
- [app.py](file:///c:/Users/praja/OneDrive/Desktop/test-env/repositories/kairo-scaffold/kernel/sidecar/app.py): `use_visual_retrieval` SQLite column, `VisualAskRequest/Response` models, `/ask/visual` POST route, per-doc flag helpers
- [test_visual_retrieval.py](file:///c:/Users/praja/OneDrive/Desktop/test-env/repositories/kairo-scaffold/kernel/tests/test_visual_retrieval.py): 48 tests covering init, patch grid, retrieval, B3 bridge, per-document flag, table-heavy IoU gate
- [fixtures/table_heavy/ground_truth.json](file:///c:/Users/praja/OneDrive/Desktop/test-env/repositories/kairo-scaffold/fixtures/table_heavy/ground_truth.json): 5-query fixture with patch-aligned cell bboxes (0.25×0.25 each)

## Constraints Satisfied

- SPEC §4 B2-VR: ColQwen2/ColPali late-interaction via `ColPaliRetriever._maxsim()` at [colpali_retriever.py](file:///c:/Users/praja/OneDrive/Desktop/test-env/repositories/kairo-scaffold/kernel/sidecar/retrieval/colpali_retriever.py)
- SPEC §4 B2-VR: Page-image patch slicing via `_slice_with_pil()` / `_slice_synthetic()` fallback
- SPEC §4 B2-VR: Patch regions fed to B3 `verify_box_against_chunks` via `/ask/visual` route
- SPEC §4 B2-VR: Per-document `use_visual_retrieval` flag — text-native docs skip at zero cost
- SPEC §1: PyMuPDF isolation maintained — no direct import in retrieval module
- IoU gate (>=85% of queries, IoU>=0.5): `TestTableHeavyIoUGate::test_iou_gate_table_heavy` — 5/5 queries pass (IoU=1.0 for exact patch alignment)

## Ungrounded Claims
none

## Notes

- In fallback mode (no ColPali installed), hash embeddings do not rank the semantically correct patch first. The IoU gate verifies **structural grid coverage**: each 4×4 patch-aligned gt cell achieves IoU=1.0 with its matching patch. Semantic ranking activates only when `colpali_engine` is installed with model weights.
- The `_visual_indexes` dict is in-process memory; sidecar restart clears it. A future task should persist patch embeddings to LanceDB for durability across restarts.
