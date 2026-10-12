"""Typed output-target and shared option planning."""

from __future__ import annotations

from dataclasses import replace

from limulus.execution.contracts import TypedOutputPlanRequest
from limulus.execution.output_handoff import ExecutionOutputHandoffService
from limulus.syntax import parse_program
from limulus.syntax.models import (
    BlockSyntax,
    DatasetOptionSyntax,
    OutputStatementSyntax,
    WhereComparison,
)


def _request(
    dsl: str,
    *,
    explicit_output_targets: tuple[str, ...] = (),
) -> TypedOutputPlanRequest:
    parsed = parse_program(dsl)
    assert parsed.program is not None
    assert not parsed.has_errors, parsed.diagnostics
    return TypedOutputPlanRequest(
        block=parsed.program.blocks[0],
        source=parsed.program.source,
        explicit_output_targets=explicit_output_targets,
    )


def _plan(
    dsl: str,
    *,
    explicit_output_targets: tuple[str, ...] = (),
):
    return ExecutionOutputHandoffService().plan_typed(
        _request(dsl, explicit_output_targets=explicit_output_targets)
    )


def test_typed_plan_prefers_explicit_targets_and_preserves_existing_duplicates() -> None:
    plan = _plan(
        "data declared; set source; output named; run;",
        explicit_output_targets=("WORK.explicit", "explicit"),
    )

    assert plan.targets == ("explicit", "explicit")
    assert plan.default_target == "named"
    assert set(plan.effective_output_options) == {"EXPLICIT"}


def test_typed_plan_registers_data_and_top_level_but_not_inline_targets() -> None:
    plan = _plan(
        "data left; set source; if 0 then output inline; output right; run;"
    )

    assert plan.targets == ("left", "right")
    assert plan.default_target == "right"
    assert tuple(plan.effective_output_options) == ("LEFT", "RIGHT")


def test_typed_plan_uses_real_output_nodes_for_suppression_and_default() -> None:
    with_output = _plan("data out; set source; if 0 then output out; run;")
    without_output = _plan("data out; set source; note=' output'; run;")

    assert with_output.suppress_implicit_output is True
    assert without_output.suppress_implicit_output is False
    assert with_output.default_target == "out"
    assert without_output.default_target == "out"


def test_typed_plan_shares_first_nonempty_options_across_data_and_unreachable_inline() -> None:
    plan = _plan(
        'data out(keep=ID rename=(old=new) label="kept"); '
        "set source; if 0 then output WORK.out; output out; run;"
    )

    assert plan.diagnostics == ()
    assert plan.targets == ("out",)
    assert plan.effective_output_options["OUT"].keep_vars == ("ID",)
    assert plan.effective_output_options["OUT"].rename_map == (("old", "new"),)
    assert plan.effective_output_options["OUT"].label == "kept"


def test_typed_plan_ignores_empty_options_and_compares_column_keys_case_insensitively() -> None:
    plan = _plan(
        "data out(keep=ID rename=(old=new)); set source; "
        "output work.out; output OUT(keep=id rename=(OLD=NEW)); run;"
    )

    assert plan.diagnostics == ()
    assert plan.effective_output_options["OUT"].keep_vars == ("ID",)


def test_typed_plan_reports_conflict_at_later_reference() -> None:
    request = _request(
        "data out(keep=id); set source; output out(keep=amount); run;"
    )

    plan = ExecutionOutputHandoffService().plan_typed(request)

    diagnostic = plan.diagnostics[0]
    output_payload = request.block.statements[2].payload
    assert isinstance(output_payload, OutputStatementSyntax)
    later_ref = output_payload.targets[0]
    assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
    assert diagnostic.severity == "error"
    assert diagnostic.stage == "validate"
    assert diagnostic.span is not None
    assert diagnostic.span.start == later_ref.span.start
    assert "out" in diagnostic.message.lower()


def test_typed_plan_counts_unreachable_inline_options_in_source_order() -> None:
    plan = _plan(
        "data out; set source; if 0 then output out(keep=id); "
        "output out(keep=amount); run;"
    )

    assert [item.code for item in plan.diagnostics] == [
        "RUNTIME_DATASET_OPTION_INVALID"
    ]


def test_typed_plan_compares_where_structure_and_operand_order_without_equivalence() -> None:
    plan = _plan(
        "data out(where=(id > 0 and amount > 0)); set source; "
        "output out(where=(amount > 0 and id > 0)); run;"
    )

    assert [item.code for item in plan.diagnostics] == [
        "RUNTIME_DATASET_OPTION_INVALID"
    ]


def test_typed_plan_shares_equal_where_policy_with_normalized_columns_and_label() -> None:
    plan = _plan(
        'data out(where=(id=1) label="kept"); set source; '
        'output out(where=(ID=1) label="kept"); run;'
    )

    assert plan.diagnostics == ()
    assert plan.effective_output_options["OUT"].label == "kept"


def test_typed_plan_rejects_input_only_options_for_registered_output_targets() -> None:
    plan = _plan("data out; set source; output out(in=flag); run;")

    assert [item.code for item in plan.diagnostics] == [
        "RUNTIME_DATASET_OPTION_INVALID"
    ]
    assert plan.diagnostics[0].stage == "validate"


def test_typed_plan_does_not_register_unknown_inline_output() -> None:
    plan = _plan(
        "data out; set source; if 0 then output unknown(keep=id); run;"
    )

    assert plan.targets == ("out",)
    assert set(plan.effective_output_options) == {"OUT"}
    assert plan.diagnostics == ()


def test_typed_plan_rejects_data_duplicates_before_explicit_target_dedup() -> None:
    plan = _plan(
        "data out OUT; set source; run;",
        explicit_output_targets=("target",),
    )

    assert [item.code for item in plan.diagnostics] == [
        "VALIDATE_DUPLICATE_DATA_TARGET"
    ]
    assert plan.diagnostics[0].severity == "error"
    assert plan.diagnostics[0].stage == "validate"


def test_typed_plan_recomputes_from_a_transformed_block_without_text_reparse() -> None:
    original = _request("data left; set source; output right; run;")
    service = ExecutionOutputHandoffService()
    original_plan = service.plan_typed(original)
    transformed_statements = tuple(
        statement for statement in original.block.statements if statement.kind != "OUTPUT"
    )
    transformed_block = BlockSyntax(
        span=original.block.span,
        statements=transformed_statements,
    )
    transformed = TypedOutputPlanRequest(
        block=transformed_block,
        source=original.source,
    )

    transformed_plan = service.plan_typed(transformed)

    assert original_plan.targets == ("left", "right")
    assert original_plan.suppress_implicit_output is True
    assert transformed_plan.targets == ("left",)
    assert transformed_plan.default_target == "left"
    assert transformed_plan.suppress_implicit_output is False


def test_typed_plan_preserves_literal_scalar_types_when_comparing_where_options() -> None:
    request = _request("data out(where=(id=1)); set source; output out; run;")
    output_statement = request.block.statements[2]
    assert isinstance(output_statement.payload, OutputStatementSyntax)
    output_reference = output_statement.payload.targets[0]
    replacement_reference = replace(
        output_reference,
        options=DatasetOptionSyntax(
            where_expr=WhereComparison(column="ID", operator="=", value=True)
        ),
    )
    replacement_statement = replace(
        output_statement,
        payload=OutputStatementSyntax(targets=(replacement_reference,)),
    )
    transformed_block = replace(
        request.block,
        statements=(
            *request.block.statements[:2],
            replacement_statement,
            *request.block.statements[3:],
        ),
    )

    plan = ExecutionOutputHandoffService().plan_typed(
        replace(request, block=transformed_block)
    )

    assert [item.code for item in plan.diagnostics] == [
        "RUNTIME_DATASET_OPTION_INVALID"
    ]
