from __future__ import annotations

import ast
from pathlib import Path

import pyarrow as pa

from limulus.execution.coordinator import DataStepExecutor
from limulus.execution.output_handoff import OutputHandoffPlan
from limulus.execution.planning import ExecutionPlanService
from limulus.execution.strategy import decide_execution_strategy
from limulus.execution.typed_output_planning import TypedOutputTargetPlan
from limulus.execution.contracts import TypedPlanBuildRequest
from limulus.execution.rewrites import RewritePlanner
from limulus.models import DataSetRef, PreparedExecutionInputs
from limulus.syntax import parse_program
from limulus.syntax.models import DatasetOptionSyntax


def _block(source: str):
    parsed = parse_program(source)
    assert parsed.program is not None
    assert not parsed.has_errors
    assert len(parsed.program.blocks) == 1
    return parsed.program.blocks[0]


def test_typed_plan_service_builds_from_settled_typed_facts() -> None:
    block = _block("data out; set source; value = lag(id); run;")
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"id": [1, 2]}),
    )
    service = ExecutionPlanService()
    helper_plan = service.plan_typed_helpers(block)
    target_plan = TypedOutputTargetPlan(
        targets=("out",),
        default_target="out",
        suppress_implicit_output=False,
        effective_output_options={"out": DatasetOptionSyntax()},
    )
    output_handoff = OutputHandoffPlan(
        targets=("out",),
        projected_columns_by_target={"out": ("id", "value")},
        rename_map_by_target={"out": {}},
        type_expectations_by_target={"out": {"id": "int64"}},
        fallback_reason=None,
    )

    result = service.build_typed(
        TypedPlanBuildRequest(
            block=block,
            prepared=PreparedExecutionInputs(inputs={"source": source}),
            helper_plan=helper_plan,
            output_target_plan=target_plan,
            output_handoff_plan=output_handoff,
            slot_registry={
                "source_slots": {"id": 0},
                "mutable_slots": {"value": 0},
                "automatic_slots": {},
            },
            strategy_decision=decide_execution_strategy("rust"),
        )
    )

    assert result.diagnostics == ()
    assert result.execution_plan is not None
    assert result.execution_plan["output_handoff_plan"] == output_handoff
    assert result.execution_plan["row_loop_plan"]["source_slot_order"] == (
        "id",
        helper_plan.lag_lead_specs[0].helper_column,
    )
    assert result.execution_plan["row_loop_plan"]["mutable_slot_order"] == ("value",)
    assert result.execution_plan["rewrite_plan"]["lag_lead_mode"] == (
        "planner_precomputed_columns"
    )


def test_legacy_text_planning_entrypoints_and_requests_are_removed() -> None:
    from limulus.execution import contracts

    assert not hasattr(ExecutionPlanService, "plan_rewrites")
    assert not hasattr(ExecutionPlanService, "build")
    assert not hasattr(RewritePlanner, "build_plan")
    assert not hasattr(RewritePlanner, "collect_lag_lead_specs")
    assert not hasattr(contracts, "RewritePlanningRequest")
    assert not hasattr(contracts, "PlanBuildRequest")
    assert not hasattr(contracts, "PlanBuildResult")


def test_rewrite_planner_has_one_typed_production_constructor() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    constructor_owners = []
    for source_path in sorted((repository_root / "limulus").rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "RewritePlanner":
                constructor_owners.append(source_path.relative_to(repository_root).as_posix())

    assert constructor_owners == ["limulus/execution/planning.py"]
