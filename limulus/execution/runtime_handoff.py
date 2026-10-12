"""Runtime handoff contracts and request-scoped strategy ownership."""

from __future__ import annotations

from .contracts import (
    StrategyDecisionRequest,
    StrategyDecisionResult,
)
from .strategy import StrategyRequestError, decide_execution_strategy


class ExecutionRuntimeHandoffService:
    """Own the execution strategy decision before typed native handoff."""

    def decide_strategy(
        self, request: StrategyDecisionRequest
    ) -> StrategyDecisionResult:
        if not isinstance(request, StrategyDecisionRequest):
            raise TypeError("request must be StrategyDecisionRequest")

        try:
            decision = decide_execution_strategy(request.runtime_backend_preference)
        except StrategyRequestError as error:
            return StrategyDecisionResult(
                strategy_decision=None,
                diagnostics=(error.diagnostic,),
            )

        return StrategyDecisionResult(
            strategy_decision=decision,
            diagnostics=(),
        )

__all__ = [
    "ExecutionRuntimeHandoffService",
    "StrategyDecisionRequest",
    "StrategyDecisionResult",
]
