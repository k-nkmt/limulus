"""Cross-language fixtures for the versioned typed native wire boundary."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from limulus.syntax.models import (
    ApplyReferenceSyntax,
    ArrayElementTargetSyntax,
    ArrayStatementSyntax,
    AssignStatementSyntax,
    BinarySyntax,
    BlockSyntax,
    CallSyntax,
    ConcatSyntax,
    DataStatementSyntax,
    DatasetOptionSyntax,
    DatasetReferenceSyntax,
    DeferredExpressionError,
    DeferredStatementError,
    DeleteStatementSyntax,
    DoStatementSyntax,
    ElseStatementSyntax,
    EndStatementSyntax,
    IfStatementSyntax,
    LabelAttributeSyntax,
    LabelStatementSyntax,
    LiteralSyntax,
    NameListStatementSyntax,
    NameSyntax,
    NameTargetSyntax,
    OutputStatementSyntax,
    RenameStatementSyntax,
    RunStatementSyntax,
    SetStatementOptionSyntax,
    SkippedStatementSyntax,
    SourceDocument,
    SourceStatementSyntax,
    StatementKind,
    StatementPayload,
    StatementSyntax,
    StopStatementSyntax,
    SubscriptSyntax,
    SumStatementSyntax,
    UnarySyntax,
    WhereComparison,
    WhereLogicalExpression,
    WhereNotExpression,
    WhereNullPredicate,
    WhereStatementSyntax,
)
from limulus.syntax.native_payload import (
    NATIVE_PAYLOAD_VERSION,
    NativePayloadValidationError,
    serialize_block,
    validate_native_program,
)


FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "native"
    / "limulus_native"
    / "tests"
    / "fixtures"
    / "typed_payload_v1.json"
)


def _typed_wire_fixture_program() -> dict[str, object]:
    source = SourceDocument("payload", source_id="typed-payload-fixture")
    span = source.span(0, len(source.text))
    origin = SourceDocument("original", source_id="typed-origin").span(0, 8)

    def name(value: str = "value") -> NameSyntax:
        return NameSyntax(name=value, span=span)

    numeric = LiteralSyntax(
        kind="numeric",
        lexeme="9223372036854775808",
        span=span,
        origin=origin,
    )
    string = LiteralSyntax(kind="string", lexeme="'東京 🦑'", span=span)
    missing = LiteralSyntax(kind="missing", lexeme=".", span=span)
    format_token = LiteralSyntax(kind="format", lexeme="z8.2", span=span)
    value_name = name()
    unary = UnarySyntax(operator="-", operand=numeric, span=span)
    binary = BinarySyntax(
        operator="**",
        left=value_name,
        right=BinarySyntax(operator="**", left=numeric, right=missing, span=span),
        span=span,
    )
    call = CallSyntax(
        name="apply.custom",
        arguments=(string, value_name, format_token),
        span=span,
    )
    subscript = SubscriptSyntax(base=name("items"), index=numeric, span=span)
    concat = ConcatSyntax(parts=(string, value_name, missing), span=span)

    apply_reference = ApplyReferenceSyntax(
        target="lookup",
        call_span=span,
        owner_span=span,
    )
    deferred_expression = DeferredExpressionError(
        error_kind="unsupported_expression",
        native_cause="unsupported expression syntax",
        span=span,
        where_syntax_summary="saved where summary",
        display_text="display-only expression",
        apply_references=(apply_reference,),
    )
    deferred_statement = DeferredStatementError(
        error_kind="unsupported_inline_action",
        native_cause="unsupported inline action: opaque action",
        span=span,
        where_syntax_summary="saved action summary",
        display_text="opaque action",
    )

    restrictive_where = WhereLogicalExpression(
        operator="and",
        operands=(
            WhereComparison(column="id", operator=">=", value=1),
            WhereNotExpression(
                operand=WhereNullPredicate(column="deleted", negated=False)
            ),
        ),
    )
    source_options = DatasetOptionSyntax(
        in_var="seen",
        keep_vars=("id",),
        drop_vars=("raw",),
        where_expr=restrictive_where,
        rename_map=(("id", "key"),),
        firstobs=2,
        obs=5,
        label="Python-owned metadata",
    )
    output_options = DatasetOptionSyntax(
        keep_vars=("id",),
        drop_vars=("raw",),
        where_expr=restrictive_where,
        rename_map=(("id", "key"),),
        firstobs=2,
        obs=5,
        label="Python-owned output metadata",
    )
    input_reference = DatasetReferenceSyntax(
        name="input",
        options=source_options,
        span=span,
    )
    output_reference = DatasetReferenceSyntax(
        name="out",
        options=DatasetOptionSyntax(keep_vars=("id",), label="output label"),
        span=span,
    )
    assignment_target = ArrayElementTargetSyntax(
        name="items",
        delimiter="[",
        index=subscript,
        span=span,
    )
    name_target = NameTargetSyntax(name="total", span=span)

    def statement(
        kind: StatementKind,
        payload: StatementPayload,
        *,
        references: tuple[ApplyReferenceSyntax, ...] = (),
        with_origin: bool = False,
    ) -> StatementSyntax:
        return StatementSyntax(
            kind=kind,
            payload=payload,
            span=span,
            apply_references=references,
            origin=origin if with_origin else None,
            legacy_display_span=span,
        )

    inline_assignment = statement(
        "ASSIGN",
        AssignStatementSyntax(target=name_target, value=call),
    )
    statements = (
        statement("DATA", DataStatementSyntax(outputs=(output_reference,))),
        statement(
            "SET",
            SourceStatementSyntax(
                refs=(input_reference,),
                options=SetStatementOptionSyntax(indsname_var="source", end_var="last"),
            ),
        ),
        statement(
            "MERGE",
            SourceStatementSyntax(refs=(input_reference,)),
        ),
        statement(
            "WHERE",
            WhereStatementSyntax(condition=deferred_expression),
            references=(apply_reference,),
        ),
        statement(
            "ASSIGN",
            AssignStatementSyntax(target=assignment_target, value=concat),
            references=(apply_reference,),
            with_origin=True,
        ),
        statement("ASSIGN", deferred_statement),
        statement("SUM", SumStatementSyntax(target=name_target, increment=unary)),
        statement("KEEP", NameListStatementSyntax(names=(name("id"),))),
        statement("DROP", NameListStatementSyntax(names=(name("raw"),))),
        statement("BY", NameListStatementSyntax(names=(name("group"),))),
        statement("RETAIN", NameListStatementSyntax(names=(name("total"),))),
        statement(
            "LABEL",
            LabelStatementSyntax(
                labels=(LabelAttributeSyntax(name="id", label="Python label", span=span),)
            ),
        ),
        statement("RENAME", RenameStatementSyntax(pairs=(("id", "key"),))),
        statement("OUTPUT", OutputStatementSyntax(targets=(output_reference,))),
        statement(
            "IF",
            IfStatementSyntax(condition=call, then_action=inline_assignment),
        ),
        statement(
            "IF",
            IfStatementSyntax(condition=value_name, then_action=deferred_statement),
            references=(apply_reference,),
        ),
        statement("IF", IfStatementSyntax(condition=binary, then_action=None)),
        statement("ELSE", ElseStatementSyntax(action=inline_assignment)),
        statement("ELSE", ElseStatementSyntax(action=deferred_statement)),
        statement("ELSE", ElseStatementSyntax(action=None)),
        statement(
            "DO",
            DoStatementSyntax(
                loop_var="index",
                start=binary,
                end=numeric,
                boundary_display_spans=(span, span),
            ),
        ),
        statement("DO", DoStatementSyntax()),
        statement(
            "ARRAY",
            ArrayStatementSyntax(
                array_name="items",
                variables=("first", "second"),
                span=span,
                declared_size=2,
                wildcard_size=False,
                character_array=True,
            ),
        ),
        statement("END", EndStatementSyntax()),
        statement("DELETE", DeleteStatementSyntax()),
        statement("STOP", StopStatementSyntax()),
        statement("RUN", RunStatementSyntax()),
        statement("SKIPPED", SkippedStatementSyntax()),
    )
    block = BlockSyntax(span=span, statements=statements)
    return serialize_block(
        block,
        source=source,
        output_targets=("WORK.OUT", "work.out", "work.empty"),
        default_target="explicit-unregistered-default",
        suppress_implicit_output=True,
        effective_output_options={
            "work.out": output_options,
            "WORK.EMPTY": DatasetOptionSyntax(),
        },
    )


def _load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _mutate_program(program: object, case: dict[str, object]) -> dict[str, object]:
    mutated = deepcopy(program)
    assert isinstance(mutated, dict)
    mutation = case["mutation"]
    assert isinstance(mutation, dict)
    path = mutation["path"]
    assert isinstance(path, list) and path
    parent: Any = mutated
    for part in path[:-1]:
        parent = parent[part]
    key = path[-1]
    operation = mutation["operation"]
    if operation == "remove":
        del parent[key]
    elif operation == "replace":
        parent[key] = mutation["value"]
    elif operation == "insert":
        parent[key] = mutation["value"]
    else:
        raise AssertionError(f"unknown fixture mutation {operation!r}")
    return mutated


def test_python_serializer_matches_the_shared_v1_fixture() -> None:
    fixture = _load_fixture()
    valid_program = fixture["valid_program"]

    assert NATIVE_PAYLOAD_VERSION == 1
    assert valid_program == _typed_wire_fixture_program()
    validate_native_program(valid_program)
    json.dumps(valid_program, ensure_ascii=False, allow_nan=False)


def test_required_nullable_wire_fields_accept_explicit_null() -> None:
    """Nullable program fields (e.g. ``default_target``) stay valid when
    they are set to an explicit JSON ``null`` — the validator returns
    instead of raising.
    """

    program = _typed_wire_fixture_program()
    program["default_target"] = None

    validate_native_program(program)


@pytest.mark.parametrize(
    ("case_name", "reason", "path"),
    [
        ("unsupported_version", "version", "version"),
        ("missing_default_target", "field", "program.default_target"),
        ("missing_output_option_field", "field", "effective_output_options['WORK.EMPTY'].obs"),
        ("unknown_statement_tag", "tag", "statements[0].payload.tag"),
        ("wrong_boolean_field_type", "field", "suppress_implicit_output"),
        ("extra_expression_field", "field", "statements[4].payload.value.unexpected_display"),
        ("missing_deferred_action_cause", "field", "statements[15].payload.then_action.native_cause"),
        ("missing_deferred_action_span", "field", "statements[15].payload.then_action.span"),
        ("missing_output_reference_options", "field", "statements[13].payload.targets[0].options"),
        ("output_option_map_mismatch", "field", "effective_output_options"),
        ("duplicate_normalized_option_map_key", "field", "effective_output_options['work.empty']"),
        ("null_span_source_id", "field", "statements[0].span.source_id"),
    ],
)
def test_shared_fixture_rejects_invalid_wire_cases(
    case_name: str,
    reason: str,
    path: str,
) -> None:
    fixture = _load_fixture()
    valid_program = fixture["valid_program"]
    cases = fixture["invalid_cases"]
    assert isinstance(cases, list)
    case = next(item for item in cases if item["name"] == case_name)
    assert case["reason"] == reason
    assert case["path"] == path

    invalid_program = _mutate_program(valid_program, case)
    with pytest.raises(NativePayloadValidationError) as error:
        validate_native_program(invalid_program)

    assert error.value.reason == reason
    assert error.value.path == path


def test_valid_fixture_keeps_deferred_action_explicit_and_label_python_owned() -> None:
    fixture = _load_fixture()
    program = fixture["valid_program"]
    assert isinstance(program, dict)
    statements = program["statements"]
    assert isinstance(statements, list)

    deferred_if = statements[15]
    assert isinstance(deferred_if, dict)
    deferred_payload = deferred_if["payload"]
    assert isinstance(deferred_payload, dict)
    action = deferred_payload["then_action"]
    assert isinstance(action, dict)
    assert action["tag"] == "deferred"
    assert "native_cause" in action and "span" in action
    assert "kind" not in action

    label_statement = statements[11]
    assert isinstance(label_statement, dict)
    label_payload = label_statement["payload"]
    assert label_payload == {"tag": "label"}

    source_payload = statements[1]["payload"]
    assert isinstance(source_payload, dict)
    input_options = source_payload["refs"][0]["options"]
    assert input_options["in_var"] == "seen"
    output_options = program["effective_output_options"]["work.out"]
    assert output_options["in_var"] is None


def _empty_options() -> dict[str, object]:
    return {
        "in_var": None,
        "keep_vars": [],
        "drop_vars": [],
        "where_expr": None,
        "rename_map": {},
        "firstobs": None,
        "obs": None,
    }
