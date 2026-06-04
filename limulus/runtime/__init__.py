from __future__ import annotations

"""Runtime semantics package entrypoint.

This package owns row-oriented execution state, condition evaluation, and
scalar expression evaluation for Python runtime execution.
"""

from .conditions import ConditionEvaluator as ConditionEvaluator
from .expressions import ExpressionEvaluator as ExpressionEvaluator
from .arrow_cursor import (
    ArrowRowCursorProtocol as ArrowRowCursorProtocol,
    ArrowTableRowCursor as ArrowTableRowCursor,
    create_arrow_row_cursor as create_arrow_row_cursor,
)
from .row_runtime import (
    FunctionArgumentError as FunctionArgumentError,
    FunctionRegistryService as FunctionRegistryService,
    LoopControlService as LoopControlService,
    RowRuntimeService as RowRuntimeService,
    ProgramExecutionService as ProgramExecutionService,
    RetainArrayRuntimeService as RetainArrayRuntimeService,
    RowView as RowView,
    RuntimeContext as RuntimeContext,
    RuntimeExecutionError as RuntimeExecutionError,
)


def __getattr__(name: str):
    if name in {"DataStepExecutor", "FormatSupportResult", "RuntimeRequirements"}:
        from ..execution import DataStepExecutor, FormatSupportResult, RuntimeRequirements

        return {
            "DataStepExecutor": DataStepExecutor,
            "FormatSupportResult": FormatSupportResult,
            "RuntimeRequirements": RuntimeRequirements,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConditionEvaluator",
    "ArrowRowCursorProtocol",
    "ArrowTableRowCursor",
    "create_arrow_row_cursor",
    "DataStepExecutor",
    "ExpressionEvaluator",
    "FormatSupportResult",
    "FunctionArgumentError",
    "FunctionRegistryService",
    "LoopControlService",
    "RowRuntimeService",
    "ProgramExecutionService",
    "RetainArrayRuntimeService",
    "RowView",
    "RuntimeContext",
    "RuntimeExecutionError",
    "RuntimeRequirements",
]