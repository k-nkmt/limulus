from __future__ import annotations

import limulus
import pytest

from limulus import Session
from tests.helpers.public_workflow_cases import (
    MODULE_WORKFLOW_CASES,
    SESSION_DATASET_ACCESS_CASES,
    SESSION_OUTPUT_ROUNDTRIP_CASES,
)


@pytest.mark.parametrize("case", MODULE_WORKFLOW_CASES, ids=lambda case: case.case_id)
def test_module_public_workflows(case) -> None:
    result = getattr(limulus, case.api_name)(case.dsl, **case.inputs)

    assert result.success is True
    assert result.datasets["out"].to_pylist() == case.expected_output


@pytest.mark.parametrize("case", SESSION_DATASET_ACCESS_CASES, ids=lambda case: case.case_id)
def test_session_dataset_access_workflows(case) -> None:
    session = Session()
    session.load(case.input_name, case.input_payload)

    result = session.submit(case.dsl)

    assert result.success is True
    assert "out" in session.datasets
    assert session.work is session.datasets
    assert session["out"].to_pylist() == case.expected_output


@pytest.mark.parametrize("case", SESSION_OUTPUT_ROUNDTRIP_CASES, ids=lambda case: case.case_id)
def test_session_output_roundtrip_workflows(case) -> None:
    session = Session()
    session.load(case.input_name, case.input_payload)

    session.run(case.dsl)

    out_arrow = session.to_arrow("out")
    out_pandas = session.to_pandas("out")

    assert out_arrow.to_pylist() == case.expected_rows
    assert out_pandas.to_dict(orient="records") == case.expected_rows