"""Expression-slot traversal for typed statements."""

from __future__ import annotations

from dataclasses import dataclass

from ..syntax.models import (
    AssignStatementSyntax,
    ArrayElementTargetSyntax,
    BlockSyntax,
    DeferredStatementError,
    ElseStatementSyntax,
    ExpressionSyntax,
    IfStatementSyntax,
    StatementSyntax,
    SumStatementSyntax,
    WhereStatementSyntax,
)


@dataclass(frozen=True)
class StatementExpressionSlot:
    """One classifiable expression owned by a typed statement.

    Whole deferred expressions have no structure to classify.
    """

    role: str
    expression: ExpressionSyntax


def statement_expression_slots(
    statement: StatementSyntax,
) -> tuple[StatementExpressionSlot, ...]:
    """Return the expression slots one typed statement owns, in source order.

    Structured IF/ELSE inline actions contribute their own slots through the
    same rule; a deferred inline action contributes nothing.
    """

    payload = statement.payload
    if isinstance(payload, DeferredStatementError):
        # Structural fault judged at initial parse; nothing to classify.
        return ()
    if isinstance(payload, AssignStatementSyntax):
        return _assign_slots(payload)
    if isinstance(payload, SumStatementSyntax):
        return (StatementExpressionSlot("sum_increment", payload.increment),)
    if isinstance(payload, IfStatementSyntax):
        slots = [StatementExpressionSlot("if_condition", payload.condition)]
        slots.extend(_structured_action_slots(payload.then_action))
        return tuple(slots)
    if isinstance(payload, ElseStatementSyntax):
        return _structured_action_slots(payload.action)
    if isinstance(payload, WhereStatementSyntax):
        return (StatementExpressionSlot("where_condition", payload.condition),)
    from ..syntax.models import DoStatementSyntax

    if isinstance(payload, DoStatementSyntax):
        if payload.start is None or payload.end is None:
            # Bare DO starts a plain block; its body statements are flat
            # siblings and contribute their own slots.
            return ()
        return (
            StatementExpressionSlot("do_start", payload.start),
            StatementExpressionSlot("do_end", payload.end),
        )
    return ()


def _assign_slots(
    payload: AssignStatementSyntax,
) -> tuple[StatementExpressionSlot, ...]:
    """ASSIGN slots in source order: array target index first, then the value.

    A plain name target is a defined assignment column, not a classifiable
    expression; only the ``array[i]`` element target has a typed index
    expression to expose.
    """

    slots: list[StatementExpressionSlot] = []
    if isinstance(payload.target, ArrayElementTargetSyntax):
        slots.append(StatementExpressionSlot("assign_target_index", payload.target.index))
    slots.append(StatementExpressionSlot("assign_value", payload.value))
    return tuple(slots)


def _structured_action_slots(
    action: StatementSyntax | DeferredStatementError | None,
) -> tuple[StatementExpressionSlot, ...]:
    """Slots of one structured IF/ELSE inline action.

    A ``DeferredStatementError`` action stays exactly the saved descriptor:
    it is never classified as an executable statement and its reachability is
    never judged here.  A structured action keeps the same slot contract as
    the equivalent flat statement.
    """

    if not isinstance(action, StatementSyntax) or isinstance(
        action.payload, DeferredStatementError
    ):
        return ()
    return statement_expression_slots(action)


def block_expression_slots(
    block: BlockSyntax,
) -> tuple[StatementExpressionSlot, ...]:
    """Aggregate the slots of one typed block in statement order."""

    slots: list[StatementExpressionSlot] = []
    for statement in block.statements:
        slots.extend(statement_expression_slots(statement))
    return tuple(slots)


__all__ = [
    "StatementExpressionSlot",
    "block_expression_slots",
    "statement_expression_slots",
]
