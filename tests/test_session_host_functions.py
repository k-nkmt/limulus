from __future__ import annotations

from unittest.mock import patch

import pyarrow as pa

from limulus import Session
from limulus.models import ExecuteRequest, ExecuteResponse


def _stub_execute(observations: list[ExecuteRequest]):
    def execute(request: ExecuteRequest) -> ExecuteResponse:
        observations.append(request)
        return ExecuteResponse()

    return execute


def test_session_submit_captures_direct_caller_and_explicit_functions_once() -> None:
    def caller_local(value: object) -> object:
        return value

    def explicit_target(value: object) -> object:
        return value

    session = Session()
    observations: list[ExecuteRequest] = []
    with patch.object(session._executor, "execute", side_effect=_stub_execute(observations)):
        result = session.submit(
            "data out; set inp; output out; run;",
            functions={"explicit_target": explicit_target},
        )

    assert result.success is True
    assert len(observations) == 1
    context = observations[0].host_function_context
    assert context.explicit["explicit_target"] is explicit_target
    assert context.caller_locals["caller_local"] is caller_local


def test_session_run_uses_common_entry_without_recapturing_through_submit() -> None:
    def run_local(value: object) -> object:
        return value

    session = Session()
    observations: list[ExecuteRequest] = []
    with patch.object(session, "submit") as submit_mock, patch.object(
        session._executor,
        "execute",
        side_effect=_stub_execute(observations),
    ):
        result = session.run("data out; set inp; output out; run;")

    assert result.success is True
    assert submit_mock.call_count == 0
    assert len(observations) == 1
    assert observations[0].host_function_context.caller_locals["run_local"] is run_local


def test_session_wrapper_local_and_explicit_function_snapshots_are_request_scoped() -> None:
    def explicit_target(value: object) -> object:
        return value

    session = Session()
    observations: list[ExecuteRequest] = []

    def wrapper() -> None:
        def wrapper_local(value: object) -> object:
            return value

        session.submit(
            "data out; set inp; output out; run;",
            functions={"explicit_target": explicit_target},
        )

    with patch.object(session._executor, "execute", side_effect=_stub_execute(observations)):
        wrapper()

    assert len(observations) == 1
    context = observations[0].host_function_context
    assert context.explicit["explicit_target"] is explicit_target
    assert context.caller_locals["wrapper_local"](1) == 1


def test_session_submit_rejects_invalid_functions_before_executor_and_restores_backend() -> None:
    session = Session(backend="auto")
    session.load("existing", pa.table({"id": [1]}))
    before_names = tuple(session.datasets)
    before_backend = session._executor._runtime_backend_preference

    with patch.object(session._executor, "execute") as execute_mock:
        result = session.submit(
            "data out; set existing; output out; run;",
            backend="rust",
            functions={"not_callable": 1},
        )

    assert result.success is False
    assert result.log[0].code == "RUNTIME_APPLY_FUNCTIONS_INVALID"
    assert execute_mock.call_count == 0
    assert session._executor._runtime_backend_preference == before_backend
    assert tuple(session.datasets) == before_names


def test_session_consecutive_requests_do_not_share_caller_snapshots() -> None:
    session = Session()
    observations: list[ExecuteRequest] = []

    def submit_with_first_local() -> None:
        def first_local(value: object) -> object:
            return value

        session.submit("data out; set inp; output out; run;")

    def submit_with_second_local() -> None:
        def second_local(value: object) -> object:
            return value

        session.submit("data out; set inp; output out; run;")

    # Keep each callable in its respective direct caller frame.
    with patch.object(session._executor, "execute", side_effect=_stub_execute(observations)):
        submit_with_first_local()
        submit_with_second_local()

    assert len(observations) == 2
    first_context = observations[0].host_function_context
    second_context = observations[1].host_function_context
    assert first_context.caller_locals["first_local"](1) == 1
    assert "second_local" not in first_context.caller_locals
    assert second_context.caller_locals["second_local"](1) == 1
    assert "first_local" not in second_context.caller_locals
