"""Public apply resolution regressions using self-contained Arrow inputs
and the real Rust runtime."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from limulus.backend_integration.contracts import RuntimeExecutionResult
from limulus.execution import DataStepExecutor
from limulus.models import DataSetRef, ExecuteRequest, ExecuteResponse, HostFunctionContext


# Permanent baseline fixture data.  Arrow tables are immutable, so
# sharing one table across requests cannot leak state between them.
_APPLY_BASELINE_TABLE = pa.table(
    {
        "id": [1, 2],
        "note": ["", "xy"],
        "amount": [4, 9],
    }
)


def _baseline_global_increment(value: object) -> object:
    """A representative module-global callable (resolved without stack search)."""

    return value + 3  # type: ignore[operator]


def _input_ref() -> DataSetRef:
    return DataSetRef(
        kind="arrow_table",
        location="dataset://apply_baseline_in",
        payload=_APPLY_BASELINE_TABLE,
    )


def _execute_with_native_observation(
    dsl_text: str,
    *,
    host_function_context: HostFunctionContext | None = None,
) -> tuple[ExecuteResponse, RuntimeExecutionResult]:
    """Observe the runtime result once while keeping the real native call."""

    executor = DataStepExecutor(runtime_backend="rust")
    original_execute = executor._rust_backend.execute_typed
    observations: list[RuntimeExecutionResult] = []

    def observe(context: Any) -> RuntimeExecutionResult:
        result = original_execute(context)
        observations.append(result)
        return result

    # A thin observation wrapper that still performs the real
    # adapter/native call instead of mocking the result.  The original
    # bound method is restored once the request completes.
    executor._rust_backend.execute_typed = observe  # type: ignore[method-assign]
    try:
        response = executor.execute(
            ExecuteRequest(
                dsl_text=dsl_text,
                inputs={"in": _input_ref()},
                output_targets=("out",),
                host_function_context=host_function_context or HostFunctionContext(),
            )
        )
    finally:
        executor._rust_backend.execute_typed = original_execute  # type: ignore[method-assign]

    assert len(observations) == 1, "baseline request must cross the Rust adapter exactly once"
    return response, observations[0]


def _assert_runtime_boundary(
    observed: RuntimeExecutionResult,
    *,
    case_id: str,
    native_invoked: bool,
) -> None:
    """Verify the execution-engine / native-invocation boundary contract."""

    assert observed.actual_engine == "rust", f"case={case_id}: actual engine changed"
    assert observed.native_invoked is native_invoked, (
        f"case={case_id}: expected native_invoked={native_invoked!r}, "
        f"got {observed.native_invoked!r}"
    )


def _assert_rows(
    response: ExecuteResponse,
    expected: list[dict[str, object]],
    *,
    case_id: str,
) -> None:
    """Backend-independent assertion on the public Arrow result."""

    assert response.has_errors is False, f"case={case_id}: {response.diagnostics!r}"
    output = response.outputs.get("out")
    assert output is not None, f"case={case_id}: output 'out' is missing"
    assert isinstance(output.payload, pa.Table), f"case={case_id}: output is not Arrow"
    assert output.payload.to_pylist() == expected, (
        f"case={case_id}: expected rows {expected!r}, got {output.payload.to_pylist()!r}"
    )


def _assert_unresolved_diagnostic(
    response: ExecuteResponse,
    *,
    target: str,
    case_id: str,
) -> None:
    """Verify the public diagnostic contract for an unresolved target."""

    assert response.has_errors is True, f"case={case_id}: unresolved target unexpectedly succeeded"
    assert response.diagnostics, f"case={case_id}: unresolved target has no diagnostic"
    assert target in response.diagnostics[0].message, (
        f"case={case_id}: target {target!r} is absent from diagnostic "
        f"{response.diagnostics[0].message!r}"
    )


def test_apply_baseline_builtin_len_and_qualified_math_sqrt() -> None:
    response, observed = _execute_with_native_observation(
        "data out; set in; "
        "note_length = apply('len', note); "
        "root = apply('math.sqrt', amount); "
        "output out; keep id note_length root; run;"
    )

    _assert_rows(
        response,
        [
            {"id": 1, "note_length": 0.0, "root": 2.0},
            {"id": 2, "note_length": 2.0, "root": 3.0},
        ],
        case_id="apply-baseline:builtin-len-qualified-math-sqrt",
    )
    _assert_runtime_boundary(
        observed,
        case_id="apply-baseline:builtin-len-qualified-math-sqrt",
        native_invoked=True,
    )


def test_apply_baseline_direct_local_callable() -> None:
    def baseline_local_increment(value: object) -> object:
        return value + 4  # type: ignore[operator]

    # Pins the host-context contract: the callable resolves through the
    # request-scoped caller-locals snapshot with no stack search.
    response, observed = _execute_with_native_observation(
        "data out; set in; shifted = apply('baseline_local_increment', amount); "
        "output out; keep id shifted; run;",
        host_function_context=HostFunctionContext(
            caller_locals={"baseline_local_increment": baseline_local_increment}
        ),
    )

    _assert_rows(
        response,
        [{"id": 1, "shifted": 8.0}, {"id": 2, "shifted": 13.0}],
        case_id="apply-baseline:local-callable",
    )
    _assert_runtime_boundary(
        observed,
        case_id="apply-baseline:local-callable",
        native_invoked=True,
    )


def test_apply_baseline_direct_global_callable() -> None:
    response, observed = _execute_with_native_observation(
        "data out; set in; shifted = apply('_baseline_global_increment', amount); "
        "output out; keep id shifted; run;",
        host_function_context=HostFunctionContext(
            caller_globals={"_baseline_global_increment": _baseline_global_increment}
        ),
    )

    _assert_rows(
        response,
        [{"id": 1, "shifted": 7.0}, {"id": 2, "shifted": 12.0}],
        case_id="apply-baseline:global-callable",
    )
    _assert_runtime_boundary(
        observed,
        case_id="apply-baseline:global-callable",
        native_invoked=True,
    )


def test_apply_baseline_unresolved_target_diagnostic() -> None:
    target = "apply_baseline_missing_target"
    response, observed = _execute_with_native_observation(
        f"data out; set in; shifted = apply('{target}', amount); "
        "output out; keep id shifted; run;"
    )

    _assert_unresolved_diagnostic(
        response,
        target=target,
        case_id="apply-baseline:unresolved-target",
    )
    # A resolution failure returns before payload creation and never
    # starts the native callback.
    _assert_runtime_boundary(
        observed,
        case_id="apply-baseline:unresolved-target",
        native_invoked=False,
    )
