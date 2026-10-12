from __future__ import annotations

import inspect
from unittest.mock import patch

import pyarrow as pa

import limulus
from limulus.execution import DataStepExecutor
from limulus.models import ExecuteRequest, ExecuteResponse


def _stub_execute(observations: list[ExecuteRequest]):
    def execute(request: ExecuteRequest) -> ExecuteResponse:
        observations.append(request)
        return ExecuteResponse()

    return execute


def test_top_level_submit_captures_user_caller_and_explicit_functions_once() -> None:
    def caller_local(value: object) -> object:
        return value

    def explicit_target(value: object) -> object:
        return value

    observations: list[ExecuteRequest] = []
    with patch.object(DataStepExecutor, "execute", side_effect=_stub_execute(observations)):
        result = limulus.submit(
            "data out; set inp; output out; run;",
            functions={"explicit_target": explicit_target},
            inp=pa.table({"id": [1]}),
        )

    assert result.success is True
    assert len(observations) == 1
    context = observations[0].host_function_context
    assert context.explicit["explicit_target"] is explicit_target
    assert context.caller_locals["caller_local"] is caller_local


def test_top_level_run_does_not_reenter_submit_and_captures_direct_caller() -> None:
    def run_local(value: object) -> object:
        return value

    observations: list[ExecuteRequest] = []
    with patch.object(limulus, "submit") as submit_mock, patch.object(
        DataStepExecutor,
        "execute",
        side_effect=_stub_execute(observations),
    ):
        result = limulus.run(
            "data out; set inp; output out; run;",
            inp=pa.table({"id": [1]}),
        )

    assert result.success is True
    assert submit_mock.call_count == 0
    assert len(observations) == 1
    assert observations[0].host_function_context.caller_locals["run_local"] is run_local


def test_top_level_invalid_functions_fails_before_executor() -> None:
    with patch.object(DataStepExecutor, "execute") as execute_mock:
        result = limulus.submit(
            "data out; set inp; output out; run;",
            functions={"not_callable": 1},
            inp=pa.table({"id": [1]}),
        )

    assert result.success is False
    assert result.log[0].code == "RUNTIME_APPLY_FUNCTIONS_INVALID"
    assert execute_mock.call_count == 0


def test_top_level_functions_is_reserved_and_dataset_migration_uses_session_load() -> None:
    signature = inspect.signature(limulus.submit)
    assert signature.parameters["functions"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["datasets"].kind is inspect.Parameter.VAR_KEYWORD

    session = limulus.Session()
    session.load("functions", pa.table({"id": [1]}))

    assert "functions" in session.datasets


def test_top_level_requests_are_isolated_between_calls() -> None:
    def first_local(value: object) -> object:
        return value

    def second_local(value: object) -> object:
        return value

    observations: list[ExecuteRequest] = []

    def first_call() -> None:
        # Keep the callable in this direct caller frame.
        _ = first_local
        limulus.submit("data out; set inp; output out; run;", inp=pa.table({"id": [1]}))

    def second_call() -> None:
        _ = second_local
        limulus.submit("data out; set inp; output out; run;", inp=pa.table({"id": [1]}))

    with patch.object(DataStepExecutor, "execute", side_effect=_stub_execute(observations)):
        first_call()
        second_call()

    assert len(observations) == 2
    first_context = observations[0].host_function_context
    second_context = observations[1].host_function_context
    assert first_context.caller_locals["first_local"] is first_local
    assert "second_local" not in first_context.caller_locals
    assert second_context.caller_locals["second_local"] is second_local
    assert "first_local" not in second_context.caller_locals
