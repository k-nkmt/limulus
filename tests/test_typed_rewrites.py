from __future__ import annotations

import warnings
from dataclasses import replace
from unittest.mock import patch

import pytest

from limulus.execution.output_handoff import ExecutionOutputHandoffService
from limulus.execution.planning import ExecutionPlanService
from limulus.syntax import parse_program
from limulus.syntax.models import (
    ApplyReferenceSyntax,
    ArrayElementTargetSyntax,
    AssignStatementSyntax,
    BinarySyntax,
    CallSyntax,
    ConcatSyntax,
    DeferredStatementError,
    DatasetOptionSyntax,
    DatasetReferenceSyntax,
    DoStatementSyntax,
    ElseStatementSyntax,
    IfStatementSyntax,
    LiteralSyntax,
    NameSyntax,
    OutputStatementSyntax,
    StatementSyntax,
)
from limulus.execution.syntax_rewrites import rewrite_typed_block


def _parsed(source_text: str):
    parsed = parse_program(source_text)
    assert parsed.program is not None
    assert parsed.has_errors is False
    assert len(parsed.program.blocks) == 1
    return parsed.program.source, parsed.program.blocks[0]


def _rewrite(block, source):
    plan = ExecutionPlanService().plan_typed_helpers(block)
    return rewrite_typed_block(
        block,
        plan,
        source=source,
        output_handoff_service=ExecutionOutputHandoffService(),
    )


def test_rewrite_replaces_source_owned_calls_inside_concat_only() -> None:
    source, block = _parsed(
        'data out; set src; '
        'value = "prefix" || lag(amount) || lag(amount, 1, "1" + "2"); '
        'output out; run;'
    )
    result = _rewrite(block, source)

    assignment = next(
        statement
        for statement in result.block.statements
        if statement.kind == "ASSIGN"
    )
    expression = assignment.payload.value
    assert isinstance(expression, ConcatSyntax)
    assert isinstance(expression.parts[1], NameSyntax)
    source_owned = next(
        spec for spec in ExecutionPlanService().plan_typed_helpers(block).lag_lead_specs
        if spec.is_source_owned
    )
    assert expression.parts[1].name == source_owned.helper_column
    assert expression.parts[1].span == source_owned.call.span
    assert expression.parts[1].origin == (source_owned.call.origin or source_owned.call.span)
    assert isinstance(expression.parts[2], CallSyntax)
    assert expression.parts[2].name == "lag"


def test_rewrite_matches_original_call_identity_not_just_its_span() -> None:
    source, block = _parsed("data out; set src; value = lag(amount); run;")
    helper_plan = ExecutionPlanService().plan_typed_helpers(block)
    original = helper_plan.lag_lead_specs[0].call
    generated = CallSyntax(
        name=original.name,
        arguments=original.arguments,
        span=original.span,
        origin=original.origin,
    )
    assignment = next(statement for statement in block.statements if statement.kind == "ASSIGN")
    changed_assignment = replace(
        assignment,
        payload=replace(assignment.payload, value=generated),
    )
    changed_block = replace(
        block,
        statements=tuple(
            changed_assignment if statement is assignment else statement
            for statement in block.statements
        ),
    )

    result = rewrite_typed_block(
        changed_block,
        helper_plan,
        source=source,
        output_handoff_service=ExecutionOutputHandoffService(),
    )

    rewritten_assignment = next(
        statement for statement in result.block.statements if statement.kind == "ASSIGN"
    )
    assert isinstance(rewritten_assignment.payload, AssignStatementSyntax)
    assert rewritten_assignment.payload.value is generated


def test_rewrite_does_not_parse_source_or_rescan_apply_text() -> None:
    source, block = _parsed(
        'data out; set src; value="prefix" || lag(amount); output out; run;'
    )

    with (
        patch(
            "limulus.syntax.parse_program",
            side_effect=AssertionError("rewrite reparsed the source"),
        ),
        patch(
            "limulus.syntax.expressions.collect_apply_references",
            side_effect=AssertionError("rewrite rescanned source for apply calls"),
        ),
        patch("ast.parse", side_effect=AssertionError("rewrite reparsed source")),
        patch(
            "ast.literal_eval",
            side_effect=AssertionError("rewrite reparsed a literal token"),
        ),
    ):
        result = _rewrite(block, source)

    assignment = next(
        statement for statement in result.block.statements if statement.kind == "ASSIGN"
    )
    assert isinstance(assignment.payload.value, ConcatSyntax)
    assert isinstance(assignment.payload.value.parts[1], NameSyntax)


def test_rewrite_clears_display_metadata_for_changed_if_do_and_statement_nodes() -> None:
    source, block = _parsed(
        "data out; set src; value=lag(amount); "
        "if lag(flag) = 1 then value=lead(amount); else value=lag(other); "
        "do i=lag(start_at) to lag(end_at); items[lag(index)]=lag(value); end; "
        "run;"
    )
    result = _rewrite(block, source)

    if_statement = next(statement for statement in result.block.statements if statement.kind == "IF")
    if_payload = if_statement.payload
    assert isinstance(if_payload, IfStatementSyntax)
    assert isinstance(if_payload.condition, BinarySyntax)
    assert isinstance(if_payload.condition.left, NameSyntax)
    assert if_payload.condition_display_edits == ()
    assert if_payload.condition_display_span is None
    assert if_statement.legacy_display_span is None
    assert isinstance(if_payload.then_action, StatementSyntax)
    assert isinstance(if_payload.then_action.payload.value, NameSyntax)

    else_statement = next(statement for statement in result.block.statements if statement.kind == "ELSE")
    assert isinstance(else_statement.payload, ElseStatementSyntax)
    assert isinstance(else_statement.payload.action, StatementSyntax)
    assert isinstance(else_statement.payload.action.payload.value, NameSyntax)

    do_statement = next(statement for statement in result.block.statements if statement.kind == "DO")
    do_payload = do_statement.payload
    assert isinstance(do_payload, DoStatementSyntax)
    assert isinstance(do_payload.start, NameSyntax)
    assert isinstance(do_payload.end, NameSyntax)
    assert do_payload.boundary_display_spans is None
    assert do_statement.legacy_display_span is None

    array_assignment = next(
        statement
        for statement in result.block.statements
        if statement.kind == "ASSIGN"
        and isinstance(statement.payload.target, ArrayElementTargetSyntax)
    )
    assert isinstance(array_assignment.payload.target.index, NameSyntax)
    assert isinstance(array_assignment.payload.value, NameSyntax)
    assert array_assignment.legacy_display_span is None


def test_rewrite_preserves_deferred_nodes_and_apply_inventory() -> None:
    source, block = _parsed(
        'data out; set src; value=lag(amount); '
        'other=foo % apply("deferred", amount); run;'
    )
    deferred_assignment = next(
        statement
        for statement in block.statements
        if statement.kind == "ASSIGN" and statement.payload.target.name == "other"
    )
    deferred_expression = deferred_assignment.payload.value
    result = _rewrite(block, source)

    rewritten_deferred = next(
        statement
        for statement in result.block.statements
        if statement.kind == "ASSIGN" and statement.payload.target.name == "other"
    ).payload.value
    assert rewritten_deferred is deferred_expression
    assert result.apply_references == tuple(
        reference
        for statement in result.block.statements
        for reference in statement.apply_references
    )
    assert [reference.target for reference in result.apply_references] == ["deferred"]


def test_rewrite_preserves_inner_deferred_statement_action_and_its_inventory() -> None:
    source, block = _parsed(
        'data out; set src; value=lag(amount); '
        'if flag then mystery apply("inline", amount); run;'
    )
    if_statement = next(statement for statement in block.statements if statement.kind == "IF")
    deferred_action = if_statement.payload.then_action
    assert isinstance(deferred_action, DeferredStatementError)
    assert [reference.target for reference in if_statement.apply_references] == ["inline"]

    result = _rewrite(block, source)

    rewritten_if = next(statement for statement in result.block.statements if statement.kind == "IF")
    assert rewritten_if.payload.then_action is deferred_action
    assert rewritten_if.apply_references == if_statement.apply_references
    assert [reference.target for reference in result.apply_references] == ["inline"]


def test_rewrite_removes_apply_inventory_for_deleted_expression_and_inline_action() -> None:
    source, block = _parsed(
        'data out; set src; value=lag(amount); '
        'other=apply("removed-expression", amount); '
        'if flag then mystery apply("removed-action", amount); run;'
    )
    assignment = next(
        statement
        for statement in block.statements
        if statement.kind == "ASSIGN" and statement.payload.target.name == "other"
    )
    if_statement = next(statement for statement in block.statements if statement.kind == "IF")
    changed_assignment = replace(
        assignment,
        payload=replace(
            assignment.payload,
            value=NameSyntax(name="replacement", span=assignment.payload.value.span),
        ),
    )
    changed_if = replace(
        if_statement,
        payload=replace(if_statement.payload, then_action=None),
    )
    changed_block = replace(
        block,
        statements=tuple(
            changed_assignment
            if statement is assignment
            else changed_if
            if statement is if_statement
            else statement
            for statement in block.statements
        ),
    )

    result = _rewrite(changed_block, source)

    assert [reference.target for reference in result.apply_references] == []
    rewritten_assignment = next(
        statement
        for statement in result.block.statements
        if statement.kind == "ASSIGN" and statement.payload.target.name == "other"
    )
    assert rewritten_assignment.apply_references == ()
    rewritten_if = next(statement for statement in result.block.statements if statement.kind == "IF")
    assert rewritten_if.apply_references == ()


def test_rewrite_adds_explicit_inventory_for_generated_apply_call() -> None:
    source, block = _parsed("data out; set src; value=lag(amount); run;")
    helper_plan = ExecutionPlanService().plan_typed_helpers(block)
    assignment = next(statement for statement in block.statements if statement.kind == "ASSIGN")
    old_call = helper_plan.lag_lead_specs[0].call
    generated_call = CallSyntax(
        name="apply",
        arguments=(
            LiteralSyntax(kind="string", lexeme='"generated"', span=old_call.span),
            NameSyntax(name="amount", span=old_call.span),
        ),
        span=old_call.span,
        origin=old_call.span,
    )
    generated_reference = ApplyReferenceSyntax(
        target="generated",
        call_span=generated_call.span,
        owner_span=assignment.span,
    )
    changed_assignment = replace(
        assignment,
        payload=replace(assignment.payload, value=generated_call),
    )
    changed_block = replace(
        block,
        statements=tuple(
            changed_assignment if statement is assignment else statement
            for statement in block.statements
        ),
    )

    result = rewrite_typed_block(
        changed_block,
        helper_plan,
        source=source,
        output_handoff_service=ExecutionOutputHandoffService(),
        generated_apply_references=(generated_reference,),
    )

    assert result.apply_references == (generated_reference,)
    rewritten_assignment = next(
        statement for statement in result.block.statements if statement.kind == "ASSIGN"
    )
    assert rewritten_assignment.apply_references == (generated_reference,)
    assert isinstance(rewritten_assignment.payload, AssignStatementSyntax)
    assert rewritten_assignment.payload.value is generated_call


def test_rewrite_retains_parser_apply_references_by_target_literal_span() -> None:
    source, block = _parsed(
        'data out; set src; value=apply("old",amount); '
        'if apply("condition",flag) then '
        'value=cats(apply("action",amount),apply("nested",amount)); '
        'carry=lag(amount); run;'
    )
    expected = tuple(
        reference
        for statement in block.statements
        for reference in statement.apply_references
    )
    assert [reference.target for reference in expected] == [
        "old",
        "condition",
        "action",
        "nested",
    ]

    result = _rewrite(block, source)

    assert result.apply_references == expected


def test_rewrite_requires_matching_metadata_when_apply_target_changes_at_same_span() -> None:
    source, block = _parsed('data out; set src; value=apply("old",amount); run;')
    helper_plan = ExecutionPlanService().plan_typed_helpers(block)
    assignment = next(statement for statement in block.statements if statement.kind == "ASSIGN")
    original_call = assignment.payload.value
    assert isinstance(original_call, CallSyntax)
    original_reference = assignment.apply_references[0]
    changed_literal = replace(original_call.arguments[0], lexeme='"new"')
    generated_call = replace(
        original_call,
        arguments=(changed_literal, *original_call.arguments[1:]),
    )
    changed_assignment = replace(
        assignment,
        payload=replace(assignment.payload, value=generated_call),
    )
    changed_block = replace(
        block,
        statements=tuple(
            changed_assignment if statement is assignment else statement
            for statement in block.statements
        ),
    )
    output_service = ExecutionOutputHandoffService()

    with pytest.raises(ValueError, match="explicit apply metadata"):
        rewrite_typed_block(
            changed_block,
            helper_plan,
            source=source,
            output_handoff_service=output_service,
        )

    generated_reference = ApplyReferenceSyntax(
        target="new",
        call_span=original_reference.call_span,
        owner_span=assignment.span,
    )
    mismatched_reference = replace(generated_reference, target="stale")
    with pytest.raises(ValueError, match="target must match"):
        rewrite_typed_block(
            changed_block,
            helper_plan,
            source=source,
            output_handoff_service=output_service,
            generated_apply_references=(mismatched_reference,),
        )

    result = rewrite_typed_block(
        changed_block,
        helper_plan,
        source=source,
        output_handoff_service=output_service,
        generated_apply_references=(generated_reference,),
    )

    assert result.apply_references == (generated_reference,)


def test_rewrite_requires_metadata_for_a_new_quoted_apply_call() -> None:
    source, block = _parsed("data out; set src; value=lag(amount); run;")
    helper_plan = ExecutionPlanService().plan_typed_helpers(block)
    assignment = next(statement for statement in block.statements if statement.kind == "ASSIGN")
    original_call = assignment.payload.value
    assert isinstance(original_call, CallSyntax)
    generated_call = CallSyntax(
        name="apply",
        arguments=(
            LiteralSyntax(kind="string", lexeme='"new"', span=original_call.span),
            NameSyntax(name="amount", span=original_call.span),
        ),
        span=original_call.span,
        origin=original_call.span,
    )
    generated_assignment = replace(
        assignment,
        payload=replace(assignment.payload, value=generated_call),
    )
    generated_block = replace(
        block,
        statements=tuple(
            generated_assignment if statement is assignment else statement
            for statement in block.statements
        ),
    )

    with pytest.raises(ValueError, match="explicit apply metadata"):
        rewrite_typed_block(
            generated_block,
            helper_plan,
            source=source,
            output_handoff_service=ExecutionOutputHandoffService(),
        )


def test_rewrite_preserves_unicode_and_escaped_quote_apply_targets() -> None:
    source, block = _parsed(
        r'data out; set src; value=apply("café",amount); '
        r'other=apply("say \"hi\"",amount); run;'
    )

    result = _rewrite(block, source)

    assert [reference.target for reference in result.apply_references] == [
        "café",
        'say "hi"',
    ]


@pytest.mark.parametrize(
    ("literal_lexeme", "expected_target"),
    [
        (r'"\X41"', r"\X41"),
        (r'"\x41"', "A"),
        (r'"\N{LATIN CAPITAL LETTER A}"', "A"),
        (r'"\uD800"', chr(0xD800)),
    ],
)
def test_rewrite_matches_scanner_string_literal_target_semantics(
    literal_lexeme: str, expected_target: str
) -> None:
    source, block = _parsed(
        f"data out; set src; value=apply({literal_lexeme},amount); run;"
    )
    assignment = next(statement for statement in block.statements if statement.kind == "ASSIGN")
    assert [reference.target for reference in assignment.apply_references] == [
        expected_target
    ]

    result = _rewrite(block, source)

    assert [reference.target for reference in result.apply_references] == [
        expected_target
    ]


def test_unknown_apply_target_escape_does_not_emit_python_warning() -> None:
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", DeprecationWarning)
        warnings.simplefilter("always", SyntaxWarning)
        _, block = _parsed(
            r'data out; set src; value=apply("\X41",amount); run;'
        )

    assignment = next(
        statement for statement in block.statements if statement.kind == "ASSIGN"
    )
    assert [reference.target for reference in assignment.apply_references] == [r"\X41"]
    assert not any(
        "invalid escape sequence" in str(warning.message) for warning in seen
    )


@pytest.mark.parametrize(
    ("original_lexeme", "changed_lexeme", "new_target"),
    [
        (r'"\X41"', r'"\X42"', r"\X42"),
        (r'"\x41"', r'"\x42"', "B"),
        (r'"café"', r'"cafè"', "cafè"),
        (
            r'"\N{LATIN CAPITAL LETTER A}"',
            r'"\N{LATIN CAPITAL LETTER B}"',
            "B",
        ),
        (r'"\uD800"', r'"\uD801"', chr(0xD801)),
    ],
)
def test_rewrite_uses_scanner_target_semantics_for_generated_apply_metadata(
    original_lexeme: str, changed_lexeme: str, new_target: str
) -> None:
    source, block = _parsed(
        f"data out; set src; value=apply({original_lexeme},amount); run;"
    )
    helper_plan = ExecutionPlanService().plan_typed_helpers(block)
    assignment = next(statement for statement in block.statements if statement.kind == "ASSIGN")
    original_call = assignment.payload.value
    assert isinstance(original_call, CallSyntax)
    original_reference = assignment.apply_references[0]
    changed_literal = replace(original_call.arguments[0], lexeme=changed_lexeme)
    generated_call = replace(
        original_call,
        arguments=(changed_literal, *original_call.arguments[1:]),
    )
    changed_assignment = replace(
        assignment,
        payload=replace(assignment.payload, value=generated_call),
    )
    changed_block = replace(
        block,
        statements=tuple(
            changed_assignment if statement is assignment else statement
            for statement in block.statements
        ),
    )

    with pytest.raises(ValueError, match="explicit apply metadata"):
        rewrite_typed_block(
            changed_block,
            helper_plan,
            source=source,
            output_handoff_service=ExecutionOutputHandoffService(),
        )

    generated_reference = ApplyReferenceSyntax(
        target=new_target,
        call_span=original_reference.call_span,
        owner_span=assignment.span,
    )
    result = rewrite_typed_block(
        changed_block,
        helper_plan,
        source=source,
        output_handoff_service=ExecutionOutputHandoffService(),
        generated_apply_references=(generated_reference,),
    )

    assert result.apply_references == (generated_reference,)


@pytest.mark.parametrize(
    "literal_lexeme",
    [
        r'"\N{NOT A REAL UNICODE NAME}"',
        r'"\N{INCOMPLETE"',
    ],
)
def test_rewrite_does_not_require_inventory_for_unscannable_named_escape(
    literal_lexeme: str,
) -> None:
    source, block = _parsed(
        f"data out; set src; value=apply({literal_lexeme},amount); run;"
    )
    assignment = next(statement for statement in block.statements if statement.kind == "ASSIGN")
    assert assignment.apply_references == ()

    result = _rewrite(block, source)

    assert result.apply_references == ()


def test_rewrite_retains_scanner_adjacent_literal_target_for_doubled_quotes() -> None:
    source, block = _parsed(
        r"data out; set src; value=apply('a''b',amount); run;"
    )
    expected = tuple(
        reference
        for statement in block.statements
        for reference in statement.apply_references
    )
    assert [reference.target for reference in expected] == ["ab"]

    result = _rewrite(block, source)

    assert result.apply_references == expected


def test_rewrite_replans_generated_deleted_outputs_and_generated_options() -> None:
    source, original_block = _parsed(
        "data out; set src; value=lag(amount); output out; run;"
    )
    helper_plan = ExecutionPlanService().plan_typed_helpers(original_block)
    output_statement = next(
        statement for statement in original_block.statements if statement.kind == "OUTPUT"
    )
    original_reference = output_statement.payload.targets[0]
    generated_reference = DatasetReferenceSyntax(
        name="extra",
        options=DatasetOptionSyntax(keep_vars=("amount",)),
        span=original_reference.span,
    )
    generated_statement = replace(
        output_statement,
        payload=OutputStatementSyntax(targets=(generated_reference,)),
    )
    generated_block = replace(
        original_block,
        statements=tuple(
            generated_statement if statement is output_statement else statement
            for statement in original_block.statements
        ),
    )
    generated_result = rewrite_typed_block(
        generated_block,
        helper_plan,
        source=source,
        output_handoff_service=ExecutionOutputHandoffService(),
    )

    assert generated_result.output_plan.targets == ("out", "extra")
    assert generated_result.output_plan.default_target == "extra"
    assert generated_result.output_plan.suppress_implicit_output is True
    assert generated_result.output_plan.effective_output_options["EXTRA"].keep_vars == (
        "amount",
    )
    assert generated_result.block.statements[
        original_block.statements.index(output_statement)
    ].payload.targets[0] is generated_reference

    deleted_block = replace(
        original_block,
        statements=tuple(
            statement
            for statement in original_block.statements
            if statement is not output_statement
        ),
    )
    deleted_result = rewrite_typed_block(
        deleted_block,
        helper_plan,
        source=source,
        output_handoff_service=ExecutionOutputHandoffService(),
    )

    assert deleted_result.output_plan.targets == ("out",)
    assert deleted_result.output_plan.suppress_implicit_output is False
