from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend_matrix import build_session_payload


@dataclass(frozen=True)
class ModuleWorkflowCase:
    case_id: str
    api_name: str
    dsl: str
    inputs: dict[str, Any]
    expected_output: list[dict[str, Any]]


@dataclass(frozen=True)
class SessionDatasetAccessCase:
    case_id: str
    dsl: str
    input_name: str
    input_payload: Any
    expected_output: list[dict[str, Any]]


@dataclass(frozen=True)
class SessionOutputRoundtripCase:
    case_id: str
    dsl: str
    input_name: str
    input_payload: Any
    expected_rows: list[dict[str, Any]]


MODULE_WORKFLOW_CASES = (
    ModuleWorkflowCase(
        case_id="module-submit-basic",
        api_name="submit",
        dsl=(
            "data out; "
            "set inp; "
            "if x >= 0 then output out; "
            "run;"
        ),
        inputs={"inp": [{"x": 1}, {"x": -1}]},
        expected_output=[{"x": 1}],
    ),
    ModuleWorkflowCase(
        case_id="module-run-alias",
        api_name="run",
        dsl=(
            "data out; "
            "set inp; "
            "output out; "
            "run;"
        ),
        inputs={"inp": [{"x": 3}]},
        expected_output=[{"x": 3}],
    ),
)


SESSION_DATASET_ACCESS_CASES = (
    SessionDatasetAccessCase(
        case_id="session-load-submit-dataset-access",
        dsl=(
            "data out; "
            "set inp; "
            "if amount >= 0 then output out; "
            "run;"
        ),
        input_name="inp",
        input_payload=build_session_payload(
            "arrow_table",
            {"id": [1, 2], "amount": [10, -5]},
        ),
        expected_output=[{"id": 1, "amount": 10}],
    ),
)


SESSION_OUTPUT_ROUNDTRIP_CASES = (
    SessionOutputRoundtripCase(
        case_id="session-arrow-pandas-roundtrip",
        dsl=(
            "data out; "
            "set inp; "
            "output out; "
            "run;"
        ),
        input_name="inp",
        input_payload=[{"id": 1, "amount": 3}],
        expected_rows=[{"id": 1, "amount": 3}],
    ),
)


__all__ = [
    "MODULE_WORKFLOW_CASES",
    "ModuleWorkflowCase",
    "SESSION_DATASET_ACCESS_CASES",
    "SESSION_DATASET_ACCESS_CASES",
    "SessionDatasetAccessCase",
    "SESSION_OUTPUT_ROUNDTRIP_CASES",
    "SessionOutputRoundtripCase",
]