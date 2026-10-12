from __future__ import annotations

from unittest.mock import patch

from limulus.execution.planning import ExecutionPlanService
from limulus.expression_semantics import ExpressionSemantics
from limulus.syntax import parse_program
from limulus.syntax.models import (
    BinarySyntax,
    CallSyntax,
    ConcatSyntax,
    LiteralSyntax,
    NameSyntax,
)


def _block(source: str):
    parsed = parse_program(source)
    assert parsed.program is not None
    assert parsed.has_errors is False
    assert len(parsed.program.blocks) == 1
    return parsed.program.blocks[0]


def test_typed_helper_plan_reads_source_ownership_and_calls_from_syntax() -> None:
    block = _block(
        'data out; set source(in=flag) end=last indsname=source_name; by group; '
        'value = "prefix" || cats(name, .) || lag(amount, 2, .); '
        'where lag(amount, 1, "1" + "2") > 0; run;'
    )

    plan = ExecutionPlanService().plan_typed_helpers(block)

    assert plan.merge_mode == "none"
    assert plan.lag_lead_mode == "planner_precomputed_columns"
    assert plan.where_mode == "shared_pre_row_filter"
    assert plan.helper_source_columns[:5] == (
        "flag",
        "source_name",
        "last",
        "first.group",
        "last.group",
    )
    assert len(plan.lag_lead_specs) == 2

    source_owned, runtime_owned = plan.lag_lead_specs
    assert source_owned.function_name == "lag"
    assert source_owned.source_name == "amount"
    assert source_owned.offset == 2
    assert source_owned.default_expression is not None
    assert isinstance(source_owned.default_expression, LiteralSyntax)
    assert source_owned.default_expression.kind == "missing"
    assert source_owned.is_source_owned is True
    assert source_owned.helper_column in plan.helper_source_columns

    assert runtime_owned.function_name == "lag"
    assert runtime_owned.offset == 1
    assert isinstance(runtime_owned.default_expression, BinarySyntax)
    assert runtime_owned.is_source_owned is False
    assert runtime_owned.helper_column in plan.helper_runtime_columns
    assert isinstance(plan.pre_row_filter_expression, BinarySyntax)

    # The source expression remains intact for a later typed consumer: the
    # call, columns, and missing literal inside ConcatSyntax are all retained.
    assignment = plan.pre_evaluation_targets[0]
    assert assignment.role == "assign_value"
    assert isinstance(assignment.expression, ConcatSyntax)
    assert isinstance(assignment.expression.parts[1], CallSyntax)
    assert isinstance(assignment.expression.parts[1].arguments[0], NameSyntax)
    assert isinstance(assignment.expression.parts[1].arguments[1], LiteralSyntax)
    assert assignment.expression.parts[1].arguments[1].kind == "missing"


def test_typed_helper_plan_keeps_if_do_and_text_plus_defaults_at_runtime() -> None:
    block = _block(
        'data out; set source; if flag then result = lead(amount, 1, "a" || "b"); '
        'do i = lag(start_at) to lag(end_at); result = result + "1"; end; run;'
    )

    plan = ExecutionPlanService().plan_typed_helpers(block)

    assert [spec.function_name for spec in plan.lag_lead_specs] == [
        "lead",
        "lag",
        "lag",
    ]
    assert [spec.is_source_owned for spec in plan.lag_lead_specs] == [
        False,
        True,
        True,
    ]
    assert plan.lag_lead_specs[0].helper_column in plan.helper_runtime_columns
    assert all(
        spec.helper_column in plan.helper_source_columns
        for spec in plan.lag_lead_specs[1:]
    )
    assert [target.role for target in plan.pre_evaluation_targets] == [
        "if_condition",
        "assign_value",
        "do_start",
        "do_end",
        "assign_value",
    ]
    assert isinstance(plan.lag_lead_specs[0].default_expression, ConcatSyntax)
    assert isinstance(plan.lag_lead_specs[0].default_expression.parts[0], LiteralSyntax)
    assert plan.lag_lead_specs[0].default_expression.parts[0].kind == "string"


def test_typed_helper_plan_does_not_parse_or_warm_expression_text() -> None:
    block = _block(
        'data out; set source; value = lag(amount, 1, "1" + "2"); '
        'where value > 0; run;'
    )
    service = ExecutionPlanService()

    with (
        patch.object(
            ExpressionSemantics,
            "prepare_expression",
            side_effect=AssertionError("typed planning reparsed expression text"),
        ),
        patch.object(
            ExpressionSemantics,
            "parse_expression",
            side_effect=AssertionError("typed planning parsed expression text"),
        ),
        patch.object(
            ExpressionSemantics,
            "compile_expression",
            side_effect=AssertionError("typed planning warmed a text cache"),
        ),
    ):
        plan = service.plan_typed_helpers(block)

    assert plan.lag_lead_specs[0].is_source_owned is False
    assert isinstance(plan.lag_lead_specs[0].default_expression, BinarySyntax)


def test_typed_helper_names_follow_typed_candidate_occurrence_order() -> None:
    block = _block(
        "data out; set source; value = lag(amount, 0) + lag(other_amount); run;"
    )

    typed_plan = ExecutionPlanService().plan_typed_helpers(block)

    assert len(typed_plan.lag_lead_specs) == 1
    assert typed_plan.lag_lead_specs[0].helper_column == (
        "__rewrite_lag_other_amount_1_2__"
    )


def test_typed_helper_ownership_keeps_shift_default_and_offset_rules() -> None:
    statement_text = (
        'a = shift(name, -2); b = shift("name", -2); '
        'c = lag(name, -1); d = lead(name, 0); '
        'e = lag(name, 1, "fallback"); f = lag(name, 1, name + 1);'
    )
    block = _block(f"data out; set source; {statement_text} run;")

    typed_specs = ExecutionPlanService().plan_typed_helpers(block).lag_lead_specs

    assert [spec.function_name for spec in typed_specs] == [
        "shift",
        "shift",
        "lag",
        "lag",
    ]
    assert [spec.offset for spec in typed_specs] == [-2, -2, 1, 1]
    assert [spec.is_source_owned for spec in typed_specs] == [
        False,
        True,
        True,
        False,
    ]
    assert [spec.helper_column for spec in typed_specs] == [
        "__rewrite_shift_name_neg2_1__",
        "__rewrite_shift_name_neg2_2__",
        "__rewrite_lag_name_1_5__",
        "__rewrite_lag_name_1_6__",
    ]
