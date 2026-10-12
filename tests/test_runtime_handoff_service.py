"""Focused tests for the strategy decision owner preceding typed handoff."""

from __future__ import annotations

import inspect

from limulus.execution.contracts import StrategyDecisionRequest
from limulus.execution.runtime_handoff import ExecutionRuntimeHandoffService


def test_strategy_decision_is_normalized_without_constructing_a_backend() -> None:
    service = ExecutionRuntimeHandoffService()

    result = service.decide_strategy(
        StrategyDecisionRequest(runtime_backend_preference="rust")
    )

    assert result.diagnostics == ()
    assert result.strategy_decision is not None
    assert result.strategy_decision.strategy == "row"
    assert result.strategy_decision.engine == "rust"
    assert result.strategy_decision.reason_codes == ("EXPLICIT_RUST_ROW",)
    assert "runtime_backend" not in inspect.signature(
        ExecutionRuntimeHandoffService
    ).parameters


def test_invalid_strategy_request_returns_its_validation_diagnostic() -> None:
    result = ExecutionRuntimeHandoffService().decide_strategy(
        StrategyDecisionRequest(runtime_backend_preference="columnar")
    )

    assert result.strategy_decision is None
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "REQ_INVALID_EXECUTION_MODE"
    assert result.diagnostics[0].stage == "validate"
