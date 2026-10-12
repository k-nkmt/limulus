"""Assemble typed helper and row-loop execution plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..naming import _column_key
from ..syntax.models import BlockSyntax, SourceStatementSyntax
from .contracts import (
    HelperColumnPlan,
    TypedPlanBuildRequest,
    TypedPlanBuildResult,
)
from .rewrites import (
    RewritePlanner,
    TypedHelperPlan,
)
from .strategy import ExecutionStrategyDecision


_PLANNER_TOGGLE_READY_KINDS = frozenset({"SUM", "DO", "ARRAY", "ASSIGN", "RETAIN", "STOP"})




class ExecutionPlanService:
    """Own typed helper discovery and final row-loop plan assembly."""

    def __init__(self, planner: RewritePlanner | None = None) -> None:
        self._planner = planner or RewritePlanner()

    def plan_typed_helpers(self, block: BlockSyntax) -> TypedHelperPlan:
        """Plan helper ownership directly from the parsed block structure."""

        return self._planner.build_typed_plan(block)

    def build_typed(self, request: TypedPlanBuildRequest) -> TypedPlanBuildResult:
        """Assemble a final row-loop plan from already-settled typed facts."""

        if not isinstance(request, TypedPlanBuildRequest):
            raise TypeError("request must be a TypedPlanBuildRequest")
        if request.output_target_plan.diagnostics:
            return TypedPlanBuildResult(
                diagnostics=request.output_target_plan.diagnostics
            )

        helper = request.helper_plan
        source_statement = next(
            (
                statement
                for statement in request.block.statements
                if statement.kind in {"SET", "MERGE"}
                and isinstance(statement.payload, SourceStatementSyntax)
            ),
            None,
        )
        source_aliases: tuple[str, ...] = ()
        if source_statement is not None and isinstance(
            source_statement.payload, SourceStatementSyntax
        ):
            source_aliases = tuple(
                reference.name for reference in source_statement.payload.refs
            )
        helper_columns = HelperColumnPlan(
            arrow_columns=tuple(helper.helper_source_columns),
            fallback_columns=(),
            source_aliases=source_aliases,
            requires_row_prepare=False,
            fallback_reason=None,
        )
        rewrite_plan = {
            "merge_mode": helper.merge_mode,
            "lag_lead_mode": helper.lag_lead_mode,
            "where_mode": helper.where_mode,
            "helper_source_columns": tuple(helper.helper_source_columns),
            "helper_runtime_columns": tuple(helper.helper_runtime_columns),
        }
        row_loop_guard = self._build_row_loop_guard(
            helper_column_plan=helper_columns,
        )
        source_policy = self._build_policy(
            stage="source_shaping",
            decision="apply",
            reason_code=None,
            details=None,
        )
        execution_readiness = self._build_execution_readiness(
            source_policy=source_policy,
            row_loop_guard=row_loop_guard,
        )
        row_loop_plan = self._build_row_loop_plan(
            helper_column_plan=helper_columns,
            output_handoff_plan=request.output_handoff_plan,
            row_loop_guard=row_loop_guard,
            slot_registry=request.slot_registry,
            execution_readiness=execution_readiness,
            rewrite_plan=rewrite_plan,
            strategy_decision=request.strategy_decision,
        )
        return TypedPlanBuildResult(
            execution_plan={
                "stage_order": [
                    "source_shaping",
                    "helper_column_plan",
                    "rewrite_plan",
                    "row_loop_guard",
                    "row_execution",
                    "output_handoff_plan",
                ],
                "helper_column_plan": {
                    "arrow_columns": tuple(helper_columns.arrow_columns),
                    "fallback_columns": (),
                    "source_aliases": source_aliases,
                    "requires_row_prepare": False,
                    "fallback_reason": None,
                },
                "rewrite_plan": rewrite_plan,
                "row_loop_guard": row_loop_guard,
                "row_loop_plan": row_loop_plan,
                "preparation_kind": request.prepared.preparation_kind,
                "output_handoff_plan": request.output_handoff_plan,
                "slot_registry": request.slot_registry,
                "execution_readiness": execution_readiness,
            }
        )

    @staticmethod
    def _build_policy(
        *,
        stage: str,
        decision: str,
        reason_code: str | None,
        details: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "decision": decision,
            "reason_code": reason_code,
            "details": details,
        }

    @staticmethod
    def _build_row_loop_guard(
        *,
        helper_column_plan: HelperColumnPlan,
    ) -> dict[str, Any]:
        reason_codes: list[str] = []
        if helper_column_plan.requires_row_prepare:
            reason_codes.append("ROW_LOOP_UNSUPPORTED_RUNTIME_HELPER_COLUMNS")
        return {
            "decision": "apply" if not reason_codes else "error",
            "reason_codes": tuple(dict.fromkeys(reason_codes)),
        }

    @staticmethod
    def _build_execution_readiness(
        *,
        source_policy: Mapping[str, Any],
        row_loop_guard: Mapping[str, Any],
    ) -> dict[str, Any]:
        decisions = [source_policy]
        decisions.append(
            {
                "stage": "row_loop_guard",
                "decision": row_loop_guard.get("decision", "apply"),
                "reason_code": (row_loop_guard.get("reason_codes") or (None,))[0],
                "details": (
                    None
                    if row_loop_guard.get("decision") == "apply"
                    else {"reasons": list(row_loop_guard.get("reason_codes", ()))}
                ),
            }
        )
        blocking = [
            policy for policy in decisions if policy.get("decision") in {"error", "fallback"}
        ]
        if blocking:
            return {
                "decision": "blocked",
                "policies": decisions,
                "blocking_policies": blocking,
                "blocking_reason_codes": [
                    policy.get("reason_code")
                    for policy in blocking
                    if policy.get("reason_code")
                ],
            }
        return {
            "decision": "ready",
            "policies": decisions,
            "blocking_policies": [],
            "blocking_reason_codes": [],
        }

    @staticmethod
    def _merge_row_loop_slot_order(*slot_groups: Sequence[str]) -> tuple[str, ...]:
        ordered: list[str] = []
        seen_exact: set[str] = set()
        seen_keys: set[str] = set()
        for group_index, slot_group in enumerate(slot_groups):
            for name in slot_group:
                if name in seen_exact:
                    continue
                if group_index > 0 and _column_key(name) in seen_keys:
                    continue
                seen_exact.add(name)
                seen_keys.add(_column_key(name))
                ordered.append(name)
        return tuple(ordered)

    @classmethod
    def _build_row_loop_plan(
        cls,
        *,
        helper_column_plan: HelperColumnPlan,
        output_handoff_plan: Any | None,
        row_loop_guard: Mapping[str, Any],
        slot_registry: Mapping[str, Mapping[str, int]],
        execution_readiness: Mapping[str, Any],
        rewrite_plan: Mapping[str, Any],
        strategy_decision: ExecutionStrategyDecision,
    ) -> dict[str, Any]:
        source_slot_order = cls._merge_row_loop_slot_order(
            tuple(slot_registry.get("source_slots", {}).keys()),
            helper_column_plan.arrow_columns,
            helper_column_plan.fallback_columns,
        )
        mutable_slot_order = tuple(slot_registry.get("mutable_slots", {}).keys())
        automatic_slot_order = tuple(slot_registry.get("automatic_slots", {}).keys())
        blocking_reason_codes = tuple(execution_readiness.get("blocking_reason_codes", ()) or ())
        guard_reason_codes = tuple(row_loop_guard.get("reason_codes", ()) or ())
        unsupported_reason = next(
            (
                reason
                for reason in (
                    *guard_reason_codes,
                    *blocking_reason_codes,
                    helper_column_plan.fallback_reason,
                    getattr(output_handoff_plan, "fallback_reason", None),
                )
                if reason
            ),
            None,
        )
        return {
            "mode": "arrow_row_loop",
            "requested_mode": strategy_decision.requested_mode,
            "strategy": strategy_decision.strategy,
            "engine": strategy_decision.engine,
            "reason_codes": strategy_decision.reason_codes,
            "where_mode": str(rewrite_plan.get("where_mode", "none")),
            "rewrite_mode": (
                "planner_owned_active"
                if rewrite_plan.get("merge_mode") != "none"
                or rewrite_plan.get("lag_lead_mode") != "none"
                else "planner_owned_pending"
            ),
            "cursor_kind": "arrow_row_cursor",
            "source_slot_order": source_slot_order,
            "mutable_slot_order": mutable_slot_order,
            "automatic_slot_order": automatic_slot_order,
            "builder_mode": "targeted_output_handoff",
            "output_mode": "targeted_output_handoff",
            "materialization_policy": "arrow_cursor",
            "unsupported_reason": unsupported_reason,
        }


__all__ = ["ExecutionPlanService"]
