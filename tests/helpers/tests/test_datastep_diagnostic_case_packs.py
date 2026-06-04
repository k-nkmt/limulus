from __future__ import annotations

from pathlib import Path

import pytest

from limulus.models import ExecuteRequest
from limulus.runtime import DataStepExecutor
from tests.helpers.datastep_diagnostic_case_loader import (
    DataStepDiagnosticCase,
    DiagnosticExpectation,
    load_diagnostic_case_packs,
)


CASE_ROOT = Path(__file__).resolve().parents[2] / "cases" / "datastep_diagnostics"


def _find_matching_diagnostic(case_id: str, expected: DiagnosticExpectation, diagnostics):
    for diagnostic in diagnostics:
        if diagnostic.code == expected.code and diagnostic.stage == expected.stage:
            if expected.location is not None and diagnostic.location != expected.location:
                continue
            return diagnostic
    raise AssertionError(
        f"case={case_id}: expected diagnostic code={expected.code}, stage={expected.stage} not found"
    )


def _assert_expected_diagnostic(case: DataStepDiagnosticCase, expected: DiagnosticExpectation, diagnostics) -> None:
    actual = _find_matching_diagnostic(case.case_id, expected, diagnostics)
    assert actual.severity == expected.severity
    if expected.location is not None:
        assert actual.location == expected.location

    for fragment in expected.message_fragments:
        assert fragment in actual.message, f"case={case.case_id}: missing fragment '{fragment}'"

    if expected.span is not None:
        assert actual.span is not None
        assert actual.span.line == expected.span.get("line")
        assert actual.span.column == expected.span.get("column")

    if expected.labels:
        assert actual.labels

    if expected.source_text is not None:
        assert actual.source_text == expected.source_text


@pytest.mark.parametrize("case", load_diagnostic_case_packs(CASE_ROOT), ids=lambda c: c.case_id)
def test_diagnostic_case_packs_match_expected_diagnostics(case: DataStepDiagnosticCase) -> None:
    backends = ["python", "rust"] if case.backend == "auto" else [case.backend]

    for backend in backends:
        executor = DataStepExecutor(runtime_backend=backend, parser_backend="python")
        request_kwargs = {
            "dsl_text": case.program_text,
            "inputs": case.input_dataset_refs(),
        }
        if case.output_targets is not None:
            request_kwargs["output_targets"] = case.output_targets
        response = executor.execute(
            ExecuteRequest(**request_kwargs)
        )
        assert response.has_errors is True, f"case={case.case_id}, backend={backend}: expected diagnostics"
        assert response.diagnostics, f"case={case.case_id}, backend={backend}: diagnostics should not be empty"

        for expected in case.expected_diagnostics:
            _assert_expected_diagnostic(case, expected, response.diagnostics)
