"""D4 — STEDGate: Structured-output consistency gate (upgraded with Consistency Scoring).

Changelog (F2 upgrade):
    Added ConsistencyResult integration: validate() now accepts an optional
    'consistency' kwarg. When provided, the returned envelope includes:
        - consistency_score: float ∈ [0.0, 1.0]
        - low_consistency_fields: list[str]
    This surfaces the F2 Consistency Scoring signal alongside the schema gate.

STED (Semantic Tree Edit Distance) reference:
    arXiv:2512.23712 — Amazon Science.
    Cleanlab motivation:
    arXiv:2103.14749 — "Pervasive Label Errors in Test Sets Destabilize ML Benchmarks"

Integration point:
    Call STEDGate().validate(output, schema, consistency=result) immediately
    after any structured model generation step.  If passed is False OR
    consistency_score is below threshold, log and retry / surface degraded
    response.

GATE:
    pytest kernel/tests/test_d_series.py::TestSTEDGate -v
    pytest kernel/tests/test_consistency_checker.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import jsonschema

if TYPE_CHECKING:
    from kernel.sidecar.models.consistency_checker import ConsistencyResult


class STEDGate:
    """Validates a structured model output against a JSON Schema.

    Wraps :func:`jsonschema.validate` and returns a uniform dict:

    * ``"passed"``              (bool)        — True when schema-valid.
    * ``"errors"``              (list[str])   — Validation error messages.
    * ``"consistency_score"``   (float|None)  — Consistency score ∈ [0,1] or None.
    * ``"low_consistency_fields"`` (list[str]|None) — Fields with low agreement.

    F2 upgrade: consistency_score and low_consistency_fields are surfaced when
    a ConsistencyResult is passed.  They do NOT affect the 'passed' flag —
    schema validity is binary; consistency is a continuous confidence modifier.
    Callers should use consistency_score to down-rank or block outputs when
    consistency is below their threshold.

    Example (schema gate only)::

        gate = STEDGate()
        result = gate.validate({"name": "Kairo"}, {"type": "object",
                                                    "required": ["name"],
                                                    "properties": {"name": {"type": "string"}}})
        assert result["passed"] is True
        assert result["errors"] == []
        assert result["consistency_score"] is None

    Example (with consistency)::

        from kernel.sidecar.models.consistency_checker import ConsistencyChecker, ConsistencyResult
        checker = ConsistencyChecker()
        cr = checker.score(prompt, schema, obj, n=5)
        result = gate.validate(obj, schema, consistency=cr)
        # result["consistency_score"] = cr.score
        # result["low_consistency_fields"] = cr.low_consistency_fields
    """

    def validate(
        self,
        output: dict,
        schema: dict,
        consistency: Optional["ConsistencyResult"] = None,
    ) -> dict:
        """Validate *output* against *schema*, optionally with consistency info.

        Args:
            output: The structured dict produced by the model.
            schema: A JSON Schema (draft-07 compatible) describing the
                    expected shape.
            consistency: Optional ConsistencyResult from ConsistencyChecker.
                         When provided, surfaces consistency_score and
                         low_consistency_fields in the returned envelope.

        Returns:
            Dict with keys:
            - ``"passed"``: bool — True iff schema-valid.
            - ``"errors"``: list[str] — Validation error messages (empty if passed).
            - ``"consistency_score"``: float|None — From ConsistencyResult.score.
            - ``"low_consistency_fields"``: list[str]|None — Low-agreement fields.
        """
        errors: list[str] = []
        try:
            jsonschema.validate(instance=output, schema=schema)
        except jsonschema.ValidationError as exc:
            errors.append(exc.message)
        except jsonschema.SchemaError as exc:
            errors.append(f"Invalid schema: {exc.message}")

        result: dict = {
            "passed": len(errors) == 0,
            "errors": errors,
            "consistency_score": None,
            "low_consistency_fields": None,
        }

        if consistency is not None:
            result["consistency_score"] = consistency.score
            result["low_consistency_fields"] = consistency.low_consistency_fields

        return result
