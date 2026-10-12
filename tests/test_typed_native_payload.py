"""Contract tests for typed expression serialization at the native boundary."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import replace

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
    DeferredStatementError,
    DeleteStatementSyntax,
    DeferredExpressionError,
    DoStatementSyntax,
    ElseStatementSyntax,
    EndStatementSyntax,
    IfStatementSyntax,
    LabelAttributeSyntax,
    LabelStatementSyntax,
    LiteralKind,
    LiteralSyntax,
    NameListStatementSyntax,
    NameTargetSyntax,
    NameSyntax,
    OutputStatementSyntax,
    ParseDiagnosticDisplay,
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
import limulus.syntax.native_payload as native_payload
from limulus.syntax.native_payload import (
    NATIVE_PAYLOAD_VERSION,
    serialize_expression,
    serialize_span,
)


def _document(text: str = "a ** 2 + 'a ** 2'") -> SourceDocument:
    return SourceDocument(text, source_id="payload-test")


def test_native_payload_version_is_for_the_program_envelope() -> None:
    assert NATIVE_PAYLOAD_VERSION == 1

    expression = serialize_expression(
        LiteralSyntax(kind="numeric", lexeme="9223372036854775808", span=_document().span(0, 1))
    )
    assert "version" not in expression


def test_assign_statement_serializes_typed_target_value_inventory_and_display() -> None:
    document = _document("x = 9223372036854775808;")
    target = NameTargetSyntax(name="x", span=document.span(0, 1))
    value = LiteralSyntax(
        kind="numeric",
        lexeme="9223372036854775808",
        span=document.span(4, 23),
    )
    reference = ApplyReferenceSyntax(
        target="lookup",
        call_span=document.span(4, 23),
        owner_span=document.span(0, len(document.text)),
    )
    later_reference = ApplyReferenceSyntax(
        target="alternate",
        call_span=document.span(4, 23),
        owner_span=document.span(0, len(document.text)),
    )
    origin = document.span(0, 1)
    statement = StatementSyntax(
        kind="ASSIGN",
        payload=AssignStatementSyntax(target=target, value=value),
        span=document.span(0, len(document.text)),
        apply_references=(reference, later_reference),
        origin=origin,
        legacy_display_span=origin,
    )

    payload = native_payload.serialize_statement(statement, document)

    assert payload == {
        "kind": "ASSIGN",
        "payload": {
            "tag": "assign",
            "target": {
                "tag": "name",
                "name": "x",
                "span": serialize_span(target.span),
            },
            "value": serialize_expression(value),
        },
        "display_text": document.text,
        "span": serialize_span(statement.span),
        "origin": serialize_span(origin),
        "apply_references": [
            {
                "target": "lookup",
                "call_span": serialize_span(reference.call_span),
                "owner_span": serialize_span(reference.owner_span),
            },
            {
                "target": "alternate",
                "call_span": serialize_span(later_reference.call_span),
                "owner_span": serialize_span(later_reference.owner_span),
            },
        ],
    }


def test_block_serializes_explicit_output_plan_and_structured_output_options() -> None:
    document = _document("output out;")
    options = DatasetOptionSyntax(
        in_var=None,
        keep_vars=("id",),
        drop_vars=(),
        where_expr=WhereComparison(column="id", operator="=", value=1),
        rename_map=(("id", "key"),),
        firstobs=2,
        obs=5,
        label="Python metadata",
    )
    reference = DatasetReferenceSyntax(
        name="out",
        options=options,
        span=document.span(7, 10),
    )
    statement = StatementSyntax(
        kind="OUTPUT",
        payload=OutputStatementSyntax(targets=(reference,)),
        span=document.span(0, len(document.text)),
    )
    block = BlockSyntax(
        span=document.span(0, len(document.text)),
        statements=(statement,),
    )
    policies = {
        "work.out": options,
        "work.empty": DatasetOptionSyntax(),
    }

    payload = native_payload.serialize_block(
        block,
        source=document,
        output_targets=("work.out", "work.empty"),
        default_target="work.out",
        suppress_implicit_output=True,
        effective_output_options=policies,
    )

    assert payload == {
        "version": NATIVE_PAYLOAD_VERSION,
        "output_targets": ["work.out", "work.empty"],
        "default_target": "work.out",
        "suppress_implicit_output": True,
        "effective_output_options": {
            "work.out": {
                "in_var": None,
                "keep_vars": ["id"],
                "drop_vars": [],
                "where_expr": {
                    "kind": "comparison",
                    "column": "id",
                    "operator": "=",
                    "value": 1,
                },
                "rename_map": {"id": "key"},
                "firstobs": 2,
                "obs": 5,
            },
            "work.empty": {
                "in_var": None,
                "keep_vars": [],
                "drop_vars": [],
                "where_expr": None,
                "rename_map": {},
                "firstobs": None,
                "obs": None,
            },
        },
        "statements": [
            {
                "kind": "OUTPUT",
                "payload": {
                    "tag": "output",
                    "targets": [
                        {
                            "name": "out",
                            "options": {
                                "in_var": None,
                                "keep_vars": ["id"],
                                "drop_vars": [],
                                "where_expr": {
                                    "kind": "comparison",
                                    "column": "id",
                                    "operator": "=",
                                    "value": 1,
                                },
                                "rename_map": {"id": "key"},
                                "firstobs": 2,
                                "obs": 5,
                            },
                            "span": serialize_span(reference.span),
                        }
                    ],
                },
                "display_text": "output out;",
                "span": serialize_span(statement.span),
                "origin": None,
                "apply_references": [],
            }
        ],
    }

    output_options_payload = payload["effective_output_options"]
    assert isinstance(output_options_payload, dict)
    work_output_options = output_options_payload["work.out"]
    assert isinstance(work_output_options, dict)
    assert "label" not in work_output_options
    assert options.label == "Python metadata"


def test_statement_serializer_covers_each_typed_payload_kind() -> None:
    source = _document("x")
    span = source.span(0, 1)
    name = NameSyntax(name="x", span=span)
    literal = LiteralSyntax(kind="numeric", lexeme="1", span=span)
    target = NameTargetSyntax(name="x", span=span)
    reference = DatasetReferenceSyntax(
        name="input",
        options=DatasetOptionSyntax(in_var="seen"),
        span=span,
    )
    label_attribute = LabelAttributeSyntax(name="x", label="Python label", span=span)
    deferred = DeferredStatementError(
        error_kind="unsupported_syntax",
        native_cause="unsupported statement syntax",
        span=span,
        where_syntax_summary="saved where summary",
        display_text="saved display text",
        diagnostic_display=ParseDiagnosticDisplay("display projection only"),
    )
    cases: list[tuple[StatementKind, StatementPayload, str]] = [
        ("DATA", DataStatementSyntax(outputs=(reference,)), "data"),
        ("SET", SourceStatementSyntax(refs=(reference,)), "source"),
        ("MERGE", SourceStatementSyntax(refs=(reference,)), "source"),
        ("WHERE", WhereStatementSyntax(condition=name), "where"),
        (
            "ASSIGN",
            AssignStatementSyntax(target=target, value=literal),
            "assign",
        ),
        ("ASSIGN", deferred, "deferred"),
        ("SUM", SumStatementSyntax(target=target, increment=literal), "sum"),
        ("KEEP", NameListStatementSyntax(names=(name,)), "name_list"),
        ("DROP", NameListStatementSyntax(names=(name,)), "name_list"),
        ("BY", NameListStatementSyntax(names=(name,)), "name_list"),
        ("RETAIN", NameListStatementSyntax(names=(name,)), "name_list"),
        (
            "LABEL",
            LabelStatementSyntax(labels=(label_attribute,)),
            "label",
        ),
        ("RENAME", RenameStatementSyntax(pairs=(("old", "new"),)), "rename"),
        ("OUTPUT", OutputStatementSyntax(targets=(reference,)), "output"),
        ("IF", IfStatementSyntax(condition=name), "if"),
        ("ELSE IF", IfStatementSyntax(condition=name), "if"),
        ("ELSE", ElseStatementSyntax(), "else"),
        (
            "DO",
            DoStatementSyntax(
                loop_var="i",
                start=literal,
                end=literal,
                boundary_display_spans=(span, span),
            ),
            "do",
        ),
        ("DO", DoStatementSyntax(), "do"),
        ("END", EndStatementSyntax(), "end"),
        (
            "ARRAY",
            ArrayStatementSyntax(
                array_name="items",
                variables=("x", "y"),
                span=span,
                declared_size=2,
                wildcard_size=False,
                character_array=True,
            ),
            "array",
        ),
        ("DELETE", DeleteStatementSyntax(), "delete"),
        ("STOP", StopStatementSyntax(), "stop"),
        ("RUN", RunStatementSyntax(), "run"),
        ("SKIPPED", SkippedStatementSyntax(), "skipped"),
    ]

    for kind, typed_payload, expected_tag in cases:
        statement = StatementSyntax(
            kind=kind,
            payload=typed_payload,
            span=span,
            legacy_display_span=span,
        )
        serialized = native_payload.serialize_statement(statement, source)
        payload = serialized["payload"]

        assert serialized["kind"] == kind
        assert isinstance(payload, dict)
        assert payload["tag"] == expected_tag
        assert set(serialized) == {
            "kind",
            "payload",
            "display_text",
            "span",
            "origin",
            "apply_references",
        }
        assert serialized["display_text"] == "x"
        assert "legacy_display_span" not in serialized
        if isinstance(typed_payload, DeferredStatementError):
            assert payload["native_cause"] == "unsupported statement syntax"
            assert payload["where_syntax_summary"] == "saved where summary"
            assert "diagnostic_display" not in payload
        if isinstance(typed_payload, DoStatementSyntax):
            assert "boundary_display_spans" not in payload

    label_payload = native_payload.serialize_statement(
        StatementSyntax(
            kind="LABEL",
            payload=LabelStatementSyntax(labels=(label_attribute,)),
            span=span,
        ),
        source,
    )["payload"]
    assert isinstance(label_payload, dict)
    assert label_payload == {"tag": "label"}
    assert label_attribute.label == "Python label"


def test_dataset_options_keep_input_in_and_restrictive_where_dialect() -> None:
    source = _document("set input;")
    span = source.span(0, len(source.text))
    restrictive_where = WhereLogicalExpression(
        operator="and",
        operands=(
            WhereComparison(column="id", operator=">=", value=10),
            WhereNullPredicate(column="name", negated=True),
            WhereNotExpression(
                operand=WhereNullPredicate(column="deleted", negated=False)
            ),
        ),
    )
    options = DatasetOptionSyntax(
        in_var="seen",
        keep_vars=("id", "name"),
        drop_vars=("raw",),
        where_expr=restrictive_where,
        rename_map=(("id", "key"),),
        firstobs=2,
        obs=8,
        label="Python-owned label",
    )
    reference = DatasetReferenceSyntax(name="input", options=options, span=span)
    statement = StatementSyntax(
        kind="SET",
        payload=SourceStatementSyntax(
            refs=(reference,),
            options=SetStatementOptionSyntax(indsname_var="source", end_var="last"),
        ),
        span=span,
    )

    payload = native_payload.serialize_statement(statement, source)["payload"]

    assert payload == {
        "tag": "source",
        "refs": [
            {
                "name": "input",
                "options": {
                    "in_var": "seen",
                    "keep_vars": ["id", "name"],
                    "drop_vars": ["raw"],
                    "where_expr": {
                        "kind": "logical",
                        "operator": "and",
                        "operands": [
                            {
                                "kind": "comparison",
                                "column": "id",
                                "operator": ">=",
                                "value": 10,
                            },
                            {"kind": "null", "column": "name", "negated": True},
                            {
                                "kind": "not",
                                "operand": {
                                    "kind": "null",
                                    "column": "deleted",
                                    "negated": False,
                                },
                            },
                        ],
                    },
                    "rename_map": {"id": "key"},
                    "firstobs": 2,
                    "obs": 8,
                },
                "span": serialize_span(reference.span),
            }
        ],
        "options": {"indsname_var": "source", "end_var": "last"},
    }
    assert options.label == "Python-owned label"

    output_reference = DatasetReferenceSyntax(
        name="out",
        options=DatasetOptionSyntax(keep_vars=("id",), label="Python label"),
        span=span,
    )
    data_payload = native_payload.serialize_statement(
        StatementSyntax(
            kind="DATA",
            payload=DataStatementSyntax(outputs=(output_reference,)),
            span=span,
        ),
        source,
    )["payload"]
    assert data_payload == {
        "tag": "data",
        "outputs": [
            {
                "name": "out",
                "options": {
                    "in_var": None,
                    "keep_vars": ["id"],
                    "drop_vars": [],
                    "where_expr": None,
                    "rename_map": {},
                    "firstobs": None,
                    "obs": None,
                },
                "span": serialize_span(output_reference.span),
            }
        ],
    }
    assert output_reference.options.label == "Python label"


def test_array_assignment_target_and_if_else_action_union_are_structural() -> None:
    source = _document("arr[i] = 'x';")
    statement_span = source.span(0, len(source.text))
    index = NameSyntax(name="i", span=source.span(4, 5))
    target = ArrayElementTargetSyntax(
        name="arr",
        delimiter="[",
        index=index,
        span=source.span(0, 6),
    )
    value = LiteralSyntax(kind="string", lexeme="'x'", span=source.span(9, 12))
    assignment = StatementSyntax(
        kind="ASSIGN",
        payload=AssignStatementSyntax(target=target, value=value),
        span=statement_span,
    )
    assignment_payload = native_payload.serialize_statement(assignment, source)[
        "payload"
    ]
    assert isinstance(assignment_payload, dict)
    assert assignment_payload["target"] == {
        "tag": "array_element",
        "name": "arr",
        "delimiter": "[",
        "index": serialize_expression(index),
        "span": serialize_span(target.span),
    }

    condition = BinarySyntax(
        operator="==",
        left=NameSyntax(name="x", span=statement_span),
        right=LiteralSyntax(kind="numeric", lexeme="1", span=statement_span),
        span=statement_span,
    )
    deferred_action = DeferredStatementError(
        error_kind="unsupported_inline_action",
        native_cause="unsupported inline action: opaque action",
        where_syntax_summary="saved inline where summary",
        span=source.span(9, 12),
        display_text="opaque action",
        diagnostic_display=ParseDiagnosticDisplay("display-only action"),
    )
    actions: list[tuple[StatementSyntax | DeferredStatementError | None, str | None]] = [
        (assignment, "statement"),
        (deferred_action, "deferred"),
        (None, None),
    ]

    for action, expected_tag in actions:
        if_payload = native_payload.serialize_statement(
            StatementSyntax(
                kind="IF",
                payload=IfStatementSyntax(
                    condition=condition,
                    then_action=action,
                    is_subset=True,
                    is_then_do=False,
                    condition_display_edits=((source.span(4, 5), "not"),),
                    condition_display_span=statement_span,
                ),
                span=statement_span,
                legacy_display_span=statement_span,
            ),
            source,
        )
        serialized_if = if_payload["payload"]
        assert isinstance(serialized_if, dict)
        assert serialized_if["tag"] == "if"
        assert serialized_if["condition"] == serialize_expression(condition)
        assert serialized_if["is_subset"] is True
        assert serialized_if["is_then_do"] is False
        assert "condition_display_edits" not in serialized_if
        assert "condition_display_span" not in serialized_if
        assert "legacy_display_span" not in if_payload
        serialized_action = serialized_if["then_action"]
        if expected_tag is None:
            assert serialized_action is None
        else:
            assert isinstance(serialized_action, dict)
            assert serialized_action["tag"] == expected_tag
            if expected_tag == "statement":
                nested = serialized_action["statement"]
                assert isinstance(nested, dict)
                assert nested["kind"] == "ASSIGN"
            else:
                assert serialized_action["native_cause"] == (
                    "unsupported inline action: opaque action"
                )
                assert serialized_action["where_syntax_summary"] == (
                    "saved inline where summary"
                )
                assert serialized_action["span"] == serialize_span(
                    deferred_action.span
                )
                assert "diagnostic_display" not in serialized_action
                assert "kind" not in serialized_action

        else_payload = native_payload.serialize_statement(
            StatementSyntax(
                kind="ELSE",
                payload=ElseStatementSyntax(action=action),
                span=statement_span,
            ),
            source,
        )["payload"]
        assert isinstance(else_payload, dict)
        assert else_payload["tag"] == "else"
        assert else_payload["action"] == serialized_action


def test_block_display_changes_do_not_change_structure_inventory_or_routing() -> None:
    source = _document("output out;")
    changed_display = SourceDocument("output bad;", source_id=source.source_id)
    span = source.span(0, len(source.text))
    reference = DatasetReferenceSyntax(
        name="out",
        options=DatasetOptionSyntax(keep_vars=("id",)),
        span=source.span(7, 10),
    )
    inventory = ApplyReferenceSyntax(
        target="lookup",
        call_span=source.span(0, 6),
        owner_span=span,
    )
    statement = StatementSyntax(
        kind="OUTPUT",
        payload=OutputStatementSyntax(targets=(reference,)),
        span=span,
        apply_references=(inventory,),
    )
    block = BlockSyntax(span=span, statements=(statement,))
    plan = {
        "output_targets": ("work.out",),
        "default_target": "work.out",
        "suppress_implicit_output": True,
        "effective_output_options": {
            "work.out": DatasetOptionSyntax(keep_vars=("id",))
        },
    }

    before = native_payload.serialize_block(block, source=source, **plan)
    after = native_payload.serialize_block(block, source=changed_display, **plan)
    before_statement = before["statements"]
    after_statement = after["statements"]
    assert isinstance(before_statement, list)
    assert isinstance(after_statement, list)
    serialized_before = before_statement[0]
    serialized_after = after_statement[0]
    assert isinstance(serialized_before, dict)
    assert isinstance(serialized_after, dict)
    assert serialized_before["display_text"] == "output out;"
    assert serialized_after["display_text"] == "output bad;"
    assert {
        key: value
        for key, value in serialized_before.items()
        if key != "display_text"
    } == {
        key: value
        for key, value in serialized_after.items()
        if key != "display_text"
    }
    assert before["output_targets"] == ["work.out"]
    assert after["output_targets"] == ["work.out"]
    assert before["default_target"] == after["default_target"] == "work.out"
    assert before["effective_output_options"] == after["effective_output_options"]


def test_block_preserves_explicit_target_duplicates_keys_and_default() -> None:
    source = _document("")
    block = BlockSyntax(span=source.span(0, 0))

    payload = native_payload.serialize_block(
        block,
        source=source,
        output_targets=("WORK.OUT", "WORK.OUT"),
        default_target="not.in.outputs",
        suppress_implicit_output=False,
        effective_output_options={"Mixed.Key": DatasetOptionSyntax()},
    )

    assert payload["output_targets"] == ["WORK.OUT", "WORK.OUT"]
    assert payload["default_target"] == "not.in.outputs"
    assert payload["suppress_implicit_output"] is False
    effective_options = payload["effective_output_options"]
    assert isinstance(effective_options, dict)
    assert list(effective_options) == ["Mixed.Key"]


@pytest.mark.parametrize(
    ("kind", "lexeme"),
    [
        ("missing", "."),
        ("string", "'a ** 2'"),
        ("string", "'東京 🦑'"),
        ("numeric", "9223372036854775808"),
        ("format", "z8.2"),
    ],
)
def test_literal_payload_preserves_kind_raw_lexeme_and_source_span(
    kind: LiteralKind,
    lexeme: str,
) -> None:
    document = _document(lexeme)
    literal = LiteralSyntax(
        kind=kind,
        lexeme=lexeme,
        span=document.span(0, len(document.text)),
    )

    payload = serialize_expression(literal)

    assert payload == {
        "tag": "literal",
        "kind": kind,
        "lexeme": lexeme,
        "span": serialize_span(literal.span),
        "origin": None,
    }
    assert payload["lexeme"] == document.text


def test_span_payload_preserves_source_identity_and_coordinates() -> None:
    document = SourceDocument("é\nx", source_id="unicode-source")

    assert serialize_span(document.span(2, 3)) == {
        "start": 2,
        "end": 3,
        "line": 2,
        "column": 1,
        "end_line": 2,
        "end_column": 2,
        "source_id": "unicode-source",
    }


def test_name_payload_preserves_name_and_optional_origin() -> None:
    document = _document("mapped")
    origin_document = SourceDocument("prefix\nmapped", source_id="origin-source")
    origin = origin_document.span(7, 13)
    name = NameSyntax(
        name="mapped",
        span=document.span(0, 6),
        origin=origin,
    )

    assert serialize_expression(name) == {
        "tag": "name",
        "name": "mapped",
        "span": serialize_span(name.span),
        "origin": {
            "start": 7,
            "end": 13,
            "line": 2,
            "column": 1,
            "end_line": 2,
            "end_column": 7,
            "source_id": "origin-source",
        },
    }


def test_unary_payload_keeps_operator_and_operand_structure() -> None:
    document = _document("-a")
    operand = NameSyntax(name="a", span=document.span(1, 2))
    expression = UnarySyntax(operator="-", operand=operand, span=document.span(0, 2))

    assert serialize_expression(expression) == {
        "tag": "unary",
        "operator": "-",
        "operand": serialize_expression(operand),
        "span": serialize_span(expression.span),
        "origin": None,
    }


def test_binary_power_payload_preserves_tree_and_does_not_rewrite_quoted_text() -> None:
    document = _document("a ** 2 ** 3 + 'a ** 2'")
    a = NameSyntax(name="a", span=document.span(0, 1))
    two = LiteralSyntax(kind="numeric", lexeme="2", span=document.span(5, 6))
    three = LiteralSyntax(kind="numeric", lexeme="3", span=document.span(10, 11))
    right_associative = BinarySyntax(
        operator="**",
        left=a,
        right=BinarySyntax(
            operator="**",
            left=two,
            right=three,
            span=document.span(5, 11),
        ),
        span=document.span(0, 11),
    )
    quoted = LiteralSyntax(
        kind="string",
        lexeme="'a ** 2'",
        span=document.span(14, 22),
    )
    expression = BinarySyntax(
        operator="+",
        left=right_associative,
        right=quoted,
        span=document.span(0, 22),
    )

    payload = serialize_expression(expression)

    assert payload["operator"] == "+"
    left = payload["left"]
    assert isinstance(left, dict)
    assert left["tag"] == "binary"
    assert left["operator"] == "**"
    right = left["right"]
    assert isinstance(right, dict)
    assert right["operator"] == "**"
    quoted_payload = payload["right"]
    assert isinstance(quoted_payload, dict)
    assert quoted_payload["lexeme"] == "'a ** 2'"


def test_string_plus_remains_a_binary_operator_without_type_inference() -> None:
    document = _document("'1' + '2'")
    left = LiteralSyntax(kind="string", lexeme="'1'", span=document.span(0, 3))
    right = LiteralSyntax(kind="string", lexeme="'2'", span=document.span(6, 9))
    expression = BinarySyntax(
        operator="+",
        left=left,
        right=right,
        span=document.span(0, 9),
    )

    payload = serialize_expression(expression)
    serialized_left = payload["left"]
    serialized_right = payload["right"]

    assert payload["tag"] == "binary"
    assert payload["operator"] == "+"
    assert isinstance(serialized_left, dict)
    assert serialized_left["tag"] == "literal"
    assert serialized_left["kind"] == "string"
    assert isinstance(serialized_right, dict)
    assert serialized_right["tag"] == "literal"
    assert serialized_right["kind"] == "string"


def test_call_payload_keeps_name_and_argument_order() -> None:
    document = _document("cat(a, 'x')")
    first = NameSyntax(name="a", span=document.span(4, 5))
    second = LiteralSyntax(kind="string", lexeme="'x'", span=document.span(7, 10))
    expression = CallSyntax(
        name="cat",
        arguments=(first, second),
        span=document.span(0, 11),
    )

    payload = serialize_expression(expression)

    assert payload == {
        "tag": "call",
        "name": "cat",
        "arguments": [serialize_expression(first), serialize_expression(second)],
        "span": serialize_span(expression.span),
        "origin": None,
    }


def test_subscript_payload_keeps_base_and_index_as_expressions() -> None:
    document = _document("values[i]")
    base = NameSyntax(name="values", span=document.span(0, 6))
    index = NameSyntax(name="i", span=document.span(7, 8))
    expression = SubscriptSyntax(
        base=base,
        index=index,
        span=document.span(0, 9),
    )

    assert serialize_expression(expression) == {
        "tag": "subscript",
        "base": serialize_expression(base),
        "index": serialize_expression(index),
        "span": serialize_span(expression.span),
        "origin": None,
    }


def test_concat_payload_keeps_ordered_parts_without_reading_display_text() -> None:
    document = _document("a || '||' || b")
    first = NameSyntax(name="a", span=document.span(0, 1))
    quoted_pipe = LiteralSyntax(kind="string", lexeme="'||'", span=document.span(5, 9))
    last = NameSyntax(name="b", span=document.span(13, 14))
    expression = ConcatSyntax(
        parts=(first, quoted_pipe, last),
        span=document.span(0, 14),
    )

    payload = serialize_expression(expression)

    assert payload == {
        "tag": "concat",
        "parts": [
            serialize_expression(first),
            serialize_expression(quoted_pipe),
            serialize_expression(last),
        ],
        "span": serialize_span(expression.span),
        "origin": None,
    }


def test_deferred_payload_keeps_whole_descriptor_and_separate_inventory() -> None:
    document = _document("unsupported apply('missing', a) apply('later', b)")
    reference = ApplyReferenceSyntax(
        target="missing",
        call_span=document.span(12, 31),
        owner_span=document.span(0, len(document.text)),
    )
    later_reference = ApplyReferenceSyntax(
        target="later",
        call_span=document.span(32, 49),
        owner_span=document.span(0, len(document.text)),
    )
    expression = DeferredExpressionError(
        error_kind="unsupported_syntax",
        native_cause="unsupported native expression syntax",
        where_syntax_summary="invalid syntax (<unknown>, line 1)",
        span=document.span(0, len(document.text)),
        display_text="display-only source",
        apply_references=(reference, later_reference),
    )

    payload = serialize_expression(expression)

    assert payload == {
        "tag": "deferred",
        "error_kind": "unsupported_syntax",
        "native_cause": "unsupported native expression syntax",
        "where_syntax_summary": "invalid syntax (<unknown>, line 1)",
        "span": serialize_span(expression.span),
        "origin": None,
        "display_text": "display-only source",
        "apply_references": [
            {
                "target": "missing",
                "call_span": serialize_span(reference.call_span),
                "owner_span": serialize_span(reference.owner_span),
            },
            {
                "target": "later",
                "call_span": serialize_span(later_reference.call_span),
                "owner_span": serialize_span(later_reference.owner_span),
            },
        ],
    }
    assert not {"left", "right", "operand", "arguments", "parts"} & payload.keys()

    edited_display = serialize_expression(
        replace(expression, display_text="changed display only")
    )
    assert edited_display["display_text"] == "changed display only"
    assert {key: value for key, value in edited_display.items() if key != "display_text"} == {
        key: value for key, value in payload.items() if key != "display_text"
    }


def test_native_payload_import_and_serialization_need_no_parser_or_runtime() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        class DenyParserOrRuntime:
            def find_spec(self, fullname, path=None, target=None):
                denied = (
                    "limulus.syntax.program",
                    "limulus.parser",
                    "limulus.execution",
                    "pyarrow",
                    "polars",
                    "limulus_native",
                    "limulus.session",
                )
                if any(fullname == root or fullname.startswith(root + ".") for root in denied):
                    raise ImportError(f"denied import: {fullname}")
                return None

        sys.meta_path.insert(0, DenyParserOrRuntime())

        from limulus.syntax import legacy
        from limulus.syntax.models import (
            BlockSyntax,
            DatasetOptionSyntax,
            DatasetReferenceSyntax,
            LiteralSyntax,
            OutputStatementSyntax,
            SourceDocument,
            StatementSyntax,
        )
        from limulus.syntax.native_payload import (
            serialize_block,
            serialize_expression,
        )

        def fail_if_legacy_serializer_is_called(*args, **kwargs):
            raise AssertionError("native serializer called a legacy DTO serializer")

        legacy.ast_to_dict = fail_if_legacy_serializer_is_called
        legacy.statement_to_dict = fail_if_legacy_serializer_is_called

        source = SourceDocument("9223372036854775808")
        literal = LiteralSyntax(
            kind="numeric",
            lexeme=source.text,
            span=source.span(0, len(source.text)),
        )
        payload = serialize_expression(literal)
        assert payload["lexeme"] == "9223372036854775808"

        display_source = SourceDocument("output bad;")
        reference = DatasetReferenceSyntax(
            name="out",
            options=DatasetOptionSyntax(),
            span=display_source.span(7, 10),
        )
        statement = StatementSyntax(
            kind="OUTPUT",
            payload=OutputStatementSyntax(targets=(reference,)),
            span=display_source.span(0, len(display_source.text)),
        )
        block = serialize_block(
            BlockSyntax(
                span=display_source.span(0, len(display_source.text)),
                statements=(statement,),
            ),
            source=display_source,
            output_targets=("out",),
            default_target="out",
            suppress_implicit_output=True,
            effective_output_options={"out": DatasetOptionSyntax()},
        )
        assert block["statements"][0]["display_text"] == "output bad;"
        assert block["statements"][0]["payload"]["targets"][0]["name"] == "out"
        json.dumps(block)
        assert "limulus.syntax.program" not in sys.modules
        assert "limulus.parser" not in sys.modules
        assert "limulus.execution" not in sys.modules
        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"isolated payload import failed:\n{result.stderr}"
