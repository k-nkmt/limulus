"""Focused tests for the pure SyntaxModel contract.

These tests cover only the stdlib-backed pure types: source document,
codepoint spans, ordered program/block/skip-region types, the dedicated
dataset option type, and the restrictive option WHERE union moved into the
pure syntax layer.  No parser frontend, Arrow evaluation, or runtime
behavior is exercised here.
"""

from __future__ import annotations

import dataclasses
import types

import pytest

import limulus.dataset_option_expressions as legacy_where
import limulus.syntax.models as pure_models
from limulus.dataset_option_expressions import where_expression_to_dict
from limulus.models import Diagnostic
from limulus.syntax import (
    ApplyReferenceSyntax,
    ArrayElementTargetSyntax,
    ArrayStatementSyntax,
    AssignStatementSyntax,
    AssignmentTargetSyntax,
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
    ExpressionSyntax,
    IfStatementSyntax,
    LabelAttributeSyntax,
    LabelStatementSyntax,
    LiteralSyntax,
    NameListStatementSyntax,
    NameSyntax,
    NameTargetSyntax,
    OutputStatementSyntax,
    ParseDiagnosticDisplay,
    ProgramParseResult,
    ProgramProjectionFacts,
    ProgramSyntax,
    RenameStatementSyntax,
    RunStatementSyntax,
    SetStatementOptionSyntax,
    SkipRegionSyntax,
    SkippedStatementSyntax,
    SourceDocument,
    SourceSpan,
    SourceStatementSyntax,
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


# --- SourceDocument -------------------------------------------------------


def test_source_document_is_immutable_with_default_source_id() -> None:
    document = SourceDocument("data out; run;")
    assert document.text == "data out; run;"
    assert document.source_id == "<dsl>"

    named = SourceDocument("data out; run;", source_id="hook")
    assert named.source_id == "hook"

    with pytest.raises(dataclasses.FrozenInstanceError):
        document.text = "other"  # type: ignore[misc]


def test_source_document_rejects_non_string_fields() -> None:
    with pytest.raises(TypeError):
        SourceDocument(b"data out; run;")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SourceDocument("data out; run;", source_id=7)  # type: ignore[arg-type]


# --- source span derivation (codepoint half-open, 1-based line/column) ----

# Layout (codepoint offsets):
#   line 1: "a; b"   offsets 0..3, newline at 4
#   line 2: "あc d;"  offsets 5..9, newline at 10
#   line 3: "run;"   offsets 11..14
_MULTILINE_UNICODE_TEXT = "a; b\nあc d;\nrun;"


def test_source_document_span_derives_line_column_from_codepoints() -> None:
    document = SourceDocument(_MULTILINE_UNICODE_TEXT, source_id="doc")

    span = document.span(5, 7)

    assert span.start == 5
    assert span.end == 7
    assert span.line == 2
    assert span.column == 1
    assert span.end_line == 2
    assert span.end_column == 3
    assert span.source_id == "doc"
    # Codepoint indexing, not byte indexing: the covered slice starts with
    # the multi-byte ``あ`` and is exactly two codepoints wide.
    assert _MULTILINE_UNICODE_TEXT[span.start : span.end] == "あc"


def test_source_document_span_after_newline_starts_new_line() -> None:
    document = SourceDocument(_MULTILINE_UNICODE_TEXT, source_id="doc")

    newline_span = document.span(4, 4)
    assert newline_span.line == 1
    assert newline_span.column == 5

    first_of_line = document.span(5, 5)
    assert first_of_line.line == 2
    assert first_of_line.column == 1

    end_of_text = document.span(len(_MULTILINE_UNICODE_TEXT), len(_MULTILINE_UNICODE_TEXT))
    assert end_of_text.line == 3
    assert end_of_text.column == 5


def test_source_document_span_rejects_out_of_bounds_and_inverted_offsets() -> None:
    document = SourceDocument(_MULTILINE_UNICODE_TEXT, source_id="doc")

    with pytest.raises(ValueError):
        document.span(-1, 0)
    with pytest.raises(ValueError):
        document.span(0, len(_MULTILINE_UNICODE_TEXT) + 1)
    with pytest.raises(ValueError):
        document.span(3, 2)


def test_source_span_construction_validates_positions() -> None:
    valid = SourceSpan(start=0, end=4, line=1, column=1, source_id="doc")
    assert valid.end_line is None
    assert valid.end_column is None

    with pytest.raises(ValueError):
        SourceSpan(start=2, end=1, line=1, column=1)
    with pytest.raises(ValueError):
        SourceSpan(start=-1, end=1, line=1, column=1)
    with pytest.raises(ValueError):
        SourceSpan(start=0, end=1, line=0, column=1)
    with pytest.raises(ValueError):
        SourceSpan(start=0, end=1, line=1, column=0)
    with pytest.raises(ValueError):
        SourceSpan(start=0, end=1, line=1, column=1, end_line=0)
    with pytest.raises(ValueError):
        SourceSpan(start=0, end=1, line=1, column=1, end_column=0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        valid.start = 3  # type: ignore[misc]


# --- ordered program / block / skip region -------------------------------


def _block_span(start: int, end: int) -> SourceSpan:
    return SourceSpan(start=start, end=end, line=1, column=start + 1, source_id="doc")


def test_block_syntax_holds_flat_ordered_statements_with_default_empty() -> None:
    block = BlockSyntax(span=_block_span(0, 9))
    assert block.statements == ()

    populated = BlockSyntax(statements=[], span=_block_span(0, 9))
    assert populated.statements == ()

    with pytest.raises(dataclasses.FrozenInstanceError):
        populated.span = _block_span(9, 10)  # type: ignore[misc]


def test_block_syntax_requires_a_concrete_span_and_rejects_none() -> None:
    # The block span is a required pure SourceSpan value.
    with pytest.raises(TypeError):
        BlockSyntax()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        BlockSyntax(span=None)  # type: ignore[arg-type]

    # Zero-width spans are valid (empty block regions stay representable).
    zero_width = BlockSyntax(span=_block_span(5, 5))
    assert zero_width.span.start == zero_width.span.end == 5
    assert zero_width.statements == ()


def test_program_syntax_keeps_ordered_blocks_and_skip_regions() -> None:
    document = SourceDocument(_MULTILINE_UNICODE_TEXT, source_id="doc")
    first = BlockSyntax(span=document.span(0, 5))
    second = BlockSyntax(span=document.span(11, 15))
    skip_macro = SkipRegionSyntax(kind="macro", span=document.span(5, 11))
    skip_proc = SkipRegionSyntax(kind="proc", span=document.span(15, 15))

    program = ProgramSyntax(
        source=document,
        span=document.span(0, 15),
        blocks=(second, first),
        skip_regions=(skip_macro, skip_proc),
    )

    # Ordered exactly as given: the program preserves source order.
    assert program.blocks == (second, first)
    assert program.skip_regions == (skip_macro, skip_proc)

    with pytest.raises(dataclasses.FrozenInstanceError):
        program.blocks = ()  # type: ignore[misc]


def test_program_syntax_owns_the_full_immutable_source_document() -> None:
    # The program itself owns the request-scoped SourceDocument so
    # outside consumers can retrieve full-source spans and slices.
    document = SourceDocument(_MULTILINE_UNICODE_TEXT, source_id="doc")

    program = ProgramSyntax(source=document, span=document.span(0, len(document.text)))

    assert program.source is document
    assert program.source.text == _MULTILINE_UNICODE_TEXT
    assert program.source.source_id == "doc"

    with pytest.raises(TypeError):
        ProgramSyntax(span=document.span(0, 1))  # type: ignore[call-arg]


def test_program_syntax_span_must_match_source_identity_and_bounds() -> None:
    document = SourceDocument(_MULTILINE_UNICODE_TEXT, source_id="doc")

    # A program span pointing at another document is rejected.
    with pytest.raises(ValueError):
        ProgramSyntax(
            source=document,
            span=SourceSpan(start=0, end=1, line=1, column=1, source_id="other"),
        )

    # Spans beyond the owned document's codepoints are rejected.
    with pytest.raises(ValueError):
        ProgramSyntax(
            source=document,
            span=SourceSpan(
                start=0,
                end=len(document.text) + 1,
                line=1,
                column=1,
                source_id="doc",
            ),
        )
    with pytest.raises(ValueError):
        ProgramSyntax(
            source=document,
            span=SourceSpan(start=-1, end=1, line=1, column=1, source_id="doc"),
        )

    # Full-text and zero-width spans over the owned document stay valid.
    assert ProgramSyntax(source=document, span=document.span(0, len(document.text))).span.end == len(
        document.text
    )
    assert ProgramSyntax(source=document, span=document.span(0, 0)).span.start == 0


def test_skip_region_kind_is_restricted_to_standard_regions() -> None:
    assert SkipRegionSyntax(kind="macro", span=_block_span(0, 1)).kind == "macro"
    assert SkipRegionSyntax(kind="proc", span=_block_span(0, 1)).kind == "proc"

    with pytest.raises(ValueError):
        SkipRegionSyntax(kind="data", span=_block_span(0, 1))


def test_syntax_models_expose_no_framework_dependencies() -> None:
    banned_roots = {"pyarrow", "lark", "polars", "pandas"}
    for name, value in vars(pure_models).items():
        if isinstance(value, types.ModuleType):
            assert value.__name__.split(".")[0] not in banned_roots


# --- dedicated dataset option syntax -------------------------------------


def test_dataset_option_syntax_defaults_are_empty() -> None:
    options = DatasetOptionSyntax()
    assert options.in_var is None
    assert options.keep_vars == ()
    assert options.drop_vars == ()
    assert options.where_expr is None
    assert options.rename_map == ()
    assert options.firstobs is None
    assert options.obs is None
    assert options.label is None


def test_dataset_option_syntax_keeps_option_values_in_order() -> None:
    where = WhereLogicalExpression(
        "and",
        (WhereComparison("amount", ">=", 20), WhereNotExpression(WhereNullPredicate("tmp", False))),
    )
    options = DatasetOptionSyntax(
        in_var="ind",
        keep_vars=["id", "amount"],
        drop_vars=["tmp"],
        rename_map=[("amount", "score"), ("id", "key")],
        where_expr=where,
        firstobs=2,
        obs=1,
        label="out label",
    )

    assert options.keep_vars == ("id", "amount")
    assert options.drop_vars == ("tmp",)
    assert options.rename_map == (("amount", "score"), ("id", "key"))
    assert options.where_expr == where
    assert options.firstobs == 2
    assert options.obs == 1
    assert options.label == "out label"

    with pytest.raises(dataclasses.FrozenInstanceError):
        options.keep_vars = ()  # type: ignore[misc]


def test_dataset_option_syntax_where_uses_restrictive_union() -> None:
    options = DatasetOptionSyntax(where_expr=WhereNullPredicate("missing", negated=True))

    assert isinstance(options.where_expr, WhereNullPredicate)
    with pytest.raises(TypeError):
        DatasetOptionSyntax(where_expr="amount > 10")  # type: ignore[arg-type]


# --- restrictive WHERE union moved into the pure layer --------------------


def test_where_union_types_are_the_same_objects_as_legacy_reexports() -> None:
    shared_names = (
        "WhereComparison",
        "WhereComparisonOperator",
        "WhereExpression",
        "WhereLogicalExpression",
        "WhereLogicalOperator",
        "WhereNotExpression",
        "WhereNullPredicate",
        "WhereScalar",
    )
    for name in shared_names:
        assert getattr(legacy_where, name) is getattr(pure_models, name)


def test_moved_where_types_keep_constructor_and_serialization_identity() -> None:
    expression = WhereLogicalExpression(
        "or",
        (
            WhereComparison("amount", ">=", 20),
            WhereNullPredicate("missing", negated=True),
        ),
    )

    assert expression == legacy_where.WhereLogicalExpression(
        "or",
        (
            legacy_where.WhereComparison("amount", ">=", 20),
            legacy_where.WhereNullPredicate("missing", negated=True),
        ),
    )
    assert where_expression_to_dict(expression) == {
        "kind": "logical",
        "operator": "or",
        "operands": [
            {"kind": "comparison", "column": "amount", "operator": ">=", "value": 20},
            {"kind": "null", "column": "missing", "negated": True},
        ],
    }


# --- typed statement payloads, targets, expressions, deferred ---------------


def _doc(text: str = "data out; set inp(keep=a); b = a ** 2; if a > 1 then delete; run;") -> SourceDocument:
    return SourceDocument(text, source_id="doc")


def test_literal_preserves_kind_and_lexeme_without_decoding() -> None:
    document = _doc()
    missing = LiteralSyntax(kind="missing", lexeme=".", span=document.span(0, 1))
    quoted = LiteralSyntax(kind="string", lexeme="'a ** 2'", span=document.span(2, 10))
    numeric = LiteralSyntax(kind="numeric", lexeme="2", span=document.span(12, 13))
    format_token = LiteralSyntax(kind="format", lexeme="z8.2", span=document.span(14, 18))

    # The lexeme is the untouched source text of the literal.
    assert quoted.lexeme == "'a ** 2'"
    assert numeric.lexeme == "2"
    assert missing.lexeme == "."
    assert format_token.lexeme == "z8.2"

    with pytest.raises(ValueError):
        LiteralSyntax(kind="regex", lexeme="x", span=document.span(0, 1))
    with pytest.raises(ValueError):
        LiteralSyntax(kind="numeric", lexeme="", span=document.span(0, 1))
    with pytest.raises(TypeError):
        LiteralSyntax(kind="numeric", lexeme=2, span=document.span(0, 1))  # type: ignore[arg-type]


def test_name_syntax_is_a_typed_named_occurrence_with_span() -> None:
    node = NameSyntax(name="amount", span=_doc().span(0, 6))
    assert node.name == "amount"
    assert node.origin is None
    with pytest.raises(ValueError):
        NameSyntax(name="", span=_doc().span(0, 1))
    with pytest.raises(TypeError):
        NameSyntax(name="amount", span=None)  # type: ignore[arg-type]


def test_expression_union_covers_every_variant_with_required_spans() -> None:
    document = _doc()
    name = NameSyntax(name="a", span=document.span(0, 1))
    literal = LiteralSyntax(kind="numeric", lexeme="2", span=document.span(3, 4))
    binary = BinarySyntax(
        operator="**",
        left=name,
        right=literal,
        span=document.span(0, 4),
    )
    unary = UnarySyntax(operator="-", operand=name, span=document.span(0, 2))
    call = CallSyntax(
        name="array",
        arguments=(NameSyntax(name="i", span=document.span(6, 7)),),
        span=document.span(0, 8),
    )
    subscript = SubscriptSyntax(
        base=name,
        index=literal,
        span=document.span(0, 4),
    )
    concat = ConcatSyntax(
        parts=(LiteralSyntax(kind="string", lexeme="'x'", span=document.span(0, 3)), name),
        span=document.span(0, 5),
    )

    assert isinstance(literal, ExpressionSyntax)
    assert isinstance(name, ExpressionSyntax)
    assert isinstance(binary, ExpressionSyntax)
    assert isinstance(unary, ExpressionSyntax)
    assert isinstance(call, ExpressionSyntax)
    assert isinstance(subscript, ExpressionSyntax)
    assert isinstance(concat, ExpressionSyntax)

    # Every node carries its span; synthetic nodes keep an origin span.
    assert binary.origin is None
    synthetic = BinarySyntax(
        operator="+",
        left=name,
        right=literal,
        span=document.span(0, 4),
        origin=document.span(0, 4),
    )
    assert synthetic.origin == document.span(0, 4)

    with pytest.raises(TypeError):
        BinarySyntax(operator="**", left=name, right=literal, span=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BinarySyntax(operator="<>", left=name, right=literal, span=document.span(0, 4))
    with pytest.raises(ValueError):
        ConcatSyntax(parts=(name,), span=document.span(0, 1))


def test_deferred_expression_error_is_the_whole_failed_expression() -> None:
    document = _doc()
    reference = ApplyReferenceSyntax(
        target="lookup",
        call_span=document.span(4, 20),
        owner_span=document.span(0, 24),
    )
    deferred = DeferredExpressionError(
        error_kind="unsupported_syntax",
        native_cause="unsupported WHERE= arithmetic syntax",
        span=document.span(0, 24),
        display_text="amount + 1 > 10",
        where_syntax_summary="invalid syntax (<unknown>, line 1)",
        apply_references=(reference,),
    )

    assert isinstance(deferred, ExpressionSyntax)
    # Whole-expression failure: no operand/left/right/arguments payload field
    # exists, so no part of the failed expression can be partially evaluated.
    field_names = {field.name for field in dataclasses.fields(deferred)}
    assert not field_names & {"operand", "left", "right", "arguments", "parts"}
    assert deferred.error_kind == "unsupported_syntax"
    assert deferred.native_cause == "unsupported WHERE= arithmetic syntax"
    assert deferred.where_syntax_summary == "invalid syntax (<unknown>, line 1)"
    assert deferred.display_text == "amount + 1 > 10"
    assert deferred.apply_references == (reference,)

    with pytest.raises(dataclasses.FrozenInstanceError):
        deferred.error_kind = "other"  # type: ignore[misc]


def test_apply_reference_syntax_is_immutable_initial_metadata() -> None:
    document = _doc()
    reference = ApplyReferenceSyntax(
        target="unregistered",
        call_span=document.span(4, 20),
        owner_span=document.span(0, 24),
    )
    assert reference.target == "unregistered"
    assert reference.call_span == document.span(4, 20)
    assert reference.owner_span == document.span(0, 24)

    with pytest.raises(dataclasses.FrozenInstanceError):
        reference.target = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        ApplyReferenceSyntax(
            target="lookup",
            call_span=None,  # type: ignore[arg-type]
            owner_span=document.span(0, 24),
        )


def test_apply_reference_syntax_retains_decoded_empty_target() -> None:
    # Decoded empty targets are preserved in the inventory and the
    # resolver rejects them at the resolve stage. The pure inventory keeps the
    # reference untouched; it never resolves target validity.
    document = _doc()
    reference = ApplyReferenceSyntax(
        target="",
        call_span=document.span(4, 20),
        owner_span=document.span(0, 24),
    )
    assert reference.target == ""
    assert reference.call_span == document.span(4, 20)
    assert reference.owner_span == document.span(0, 24)

    # The empty target stays in the owning statement's inventory in order.
    statement = _statement(
        "ASSIGN",
        AssignStatementSyntax(
            target=NameTargetSyntax(name="b", span=document.span(0, 1)),
            value=LiteralSyntax(kind="numeric", lexeme="2", span=document.span(22, 23)),
        ),
        document,
        0,
        24,
        apply_references=(reference,),
    )
    assert statement.apply_references == (reference,)


def test_binary_operator_set_matches_native_pratt_vocabulary() -> None:
    # The binary vocabulary is the exact native Pratt set
    # (native/limulus_native/src/expressions/mod.rs 643-657).  "%" is not a
    # supported operator; "mod" is.
    document = _doc()
    name = NameSyntax(name="a", span=document.span(0, 1))
    literal = LiteralSyntax(kind="numeric", lexeme="2", span=document.span(5, 6))
    for operator in ("or", "and", "==", "!=", ">", ">=", "<", "<=", "+", "-", "*", "/", "**", "mod"):
        node = BinarySyntax(operator=operator, left=name, right=literal, span=document.span(0, 6))
        assert node.operator == operator

    with pytest.raises(ValueError):
        BinarySyntax(operator="%", left=name, right=literal, span=document.span(0, 6))


def test_binary_equality_node_keeps_raw_source_span() -> None:
    # The "==" operator is the normalized native Eq vocabulary; the source
    # "=" alias is normalized by the later frontend and the node keeps the
    # raw source occurrence span.  The model defines no normalization.
    document = _doc("if a = 1 then delete; run;")
    name = NameSyntax(name="a", span=document.span(3, 4))
    literal = LiteralSyntax(kind="numeric", lexeme="1", span=document.span(7, 8))
    node = BinarySyntax(operator="==", left=name, right=literal, span=document.span(5, 6))

    assert node.operator == "=="
    assert document.text[node.span.start : node.span.end] == "="
    assert node.origin is None


def test_assignment_target_variants_keep_delimiter_and_index() -> None:
    document = _doc()
    name_target = NameTargetSyntax(name="b", span=document.span(0, 1))
    index = NameSyntax(name="i", span=document.span(4, 5))
    array_target = ArrayElementTargetSyntax(
        name="arr",
        delimiter="(",
        index=index,
        span=document.span(0, 6),
    )

    assert isinstance(name_target, AssignmentTargetSyntax)
    assert isinstance(array_target, AssignmentTargetSyntax)
    assert array_target.delimiter == "("
    assert array_target.index == index

    with pytest.raises(ValueError):
        ArrayElementTargetSyntax(
            name="arr",
            delimiter="<",
            index=index,
            span=document.span(0, 6),
        )
    with pytest.raises(TypeError):
        ArrayElementTargetSyntax(
            name="arr",
            delimiter="(",
            index="i",  # type: ignore[arg-type]
            span=document.span(0, 6),
        )


def _statement(
    kind: str,
    payload: object,
    document: SourceDocument,
    start: int,
    end: int,
    apply_references: tuple[ApplyReferenceSyntax, ...] = (),
    legacy_display_span: SourceSpan | None = None,
) -> StatementSyntax:
    return StatementSyntax(
        kind=kind,
        payload=payload,
        span=document.span(start, end),
        apply_references=apply_references,
        legacy_display_span=legacy_display_span,
    )


def test_statement_syntax_validates_kind_against_payload_class() -> None:
    document = _doc()
    assign = _statement(
        "ASSIGN",
        AssignStatementSyntax(
            target=NameTargetSyntax(name="b", span=document.span(0, 1)),
            value=LiteralSyntax(kind="numeric", lexeme="2", span=document.span(4, 5)),
        ),
        document,
        0,
        5,
    )
    assert assign.kind == "ASSIGN"
    assert assign.origin is None

    # Kind and payload must agree.
    with pytest.raises(ValueError):
        _statement("KEEP", assign.payload, document, 0, 5)
    with pytest.raises(ValueError):
        _statement("THROW", assign.payload, document, 0, 5)

    # Apply references keep original source order and accept only typed items.
    reference = ApplyReferenceSyntax(
        target="lookup",
        call_span=document.span(4, 20),
        owner_span=document.span(0, 24),
    )
    statement = _statement(
        "ASSIGN",
        assign.payload,
        document,
        0,
        24,
        apply_references=(reference,),
    )
    assert statement.apply_references == (reference,)
    with pytest.raises(TypeError):
        _statement(
            "ASSIGN",
            assign.payload,
            document,
            0,
            24,
            apply_references=("lookup",),  # type: ignore[arg-type]
        )


def test_statement_payloads_represent_every_supported_kind_without_raw_parse() -> None:
    document = _doc()
    name = NameSyntax(name="x", span=document.span(0, 1))
    numeric = LiteralSyntax(kind="numeric", lexeme="1", span=document.span(2, 3))
    where_option = DatasetOptionSyntax(where_expr=WhereComparison("amount", ">=", 20))
    reference = DatasetReferenceSyntax(name="inp", options=where_option, span=document.span(4, 20))
    set_options = SetStatementOptionSyntax(indsname_var="dsname", end_var="eof")
    payload_by_kind: dict[str, object] = {
        "DATA": DataStatementSyntax(outputs=(reference,)),
        "SET": SourceStatementSyntax(refs=(reference,), options=set_options),
        "MERGE": SourceStatementSyntax(refs=(reference,)),
        "WHERE": WhereStatementSyntax(condition=BinarySyntax(
            operator="==",
            left=name,
            right=numeric,
            span=document.span(0, 3),
        )),
        "ASSIGN": AssignStatementSyntax(
            target=NameTargetSyntax(name="b", span=document.span(0, 1)),
            value=numeric,
        ),
        "SUM": SumStatementSyntax(
            target=NameTargetSyntax(name="b", span=document.span(0, 1)),
            increment=numeric,
        ),
        "KEEP": NameListStatementSyntax(names=(name,)),
        "DROP": NameListStatementSyntax(names=(name,)),
        "BY": NameListStatementSyntax(names=(name,)),
        "RETAIN": NameListStatementSyntax(names=(name,)),
        "LABEL": LabelStatementSyntax(labels=(
            LabelAttributeSyntax(name="x", label="the x", span=document.span(0, 12)),
        )),
        "RENAME": RenameStatementSyntax(pairs=(("x", "y"),)),
        "OUTPUT": OutputStatementSyntax(targets=(reference,)),
        "IF": IfStatementSyntax(
            condition=BinarySyntax(
                operator=">",
                left=name,
                right=numeric,
                span=document.span(0, 3),
            ),
            then_action=None,
            is_subset=True,
        ),
        "ELSE IF": IfStatementSyntax(
            condition=BinarySyntax(
                operator=">",
                left=name,
                right=numeric,
                span=document.span(0, 3),
            ),
            is_subset=True,
        ),
        "ELSE": ElseStatementSyntax(),
        "DO": DoStatementSyntax(
            loop_var="i",
            start=numeric,
            end=numeric,
        ),
        "ARRAY": ArrayStatementSyntax(
            array_name="arr",
            variables=("x", "y"),
            span=document.span(0, 12),
        ),
        "DELETE": DeleteStatementSyntax(),
        "STOP": StopStatementSyntax(),
        "END": EndStatementSyntax(),
        "RUN": RunStatementSyntax(),
        "SKIPPED": SkippedStatementSyntax(),
    }

    statements = [
        _statement(kind, payload, document, 0, 5)
        for kind, payload in payload_by_kind.items()
    ]
    # Flat order is preserved exactly as given ("ELSE IF" stays distinct).
    assert tuple(statement.kind for statement in statements) == tuple(payload_by_kind)
    # "ELSE IF" is not collapsed into "ELSE".
    assert "ELSE IF" in payload_by_kind
    assert "ELSE" in payload_by_kind


def test_if_do_array_payloads_keep_inline_action_and_flags() -> None:
    document = _doc()
    condition = BinarySyntax(
        operator=">",
        left=NameSyntax(name="a", span=document.span(3, 4)),
        right=LiteralSyntax(kind="numeric", lexeme="1", span=document.span(8, 9)),
        span=document.span(3, 9),
    )
    inline = StatementSyntax(
        kind="ASSIGN",
        payload=AssignStatementSyntax(
            target=NameTargetSyntax(name="b", span=document.span(14, 15)),
            value=LiteralSyntax(kind="numeric", lexeme="2", span=document.span(18, 19)),
        ),
        span=document.span(14, 19),
    )
    if_statement = _statement(
        "IF",
        IfStatementSyntax(condition=condition, then_action=inline, is_then_do=False),
        document,
        0,
        19,
    )
    payload = if_statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert payload.then_action == inline
    assert payload.is_then_do is False
    assert payload.is_subset is False

    # A bare DO keeps no loop bounds; a loop DO keeps all three parts.
    bare = _statement("DO", DoStatementSyntax(), document, 0, 4)
    assert isinstance(bare.payload, DoStatementSyntax)
    assert bare.payload.loop_var is None
    with pytest.raises(ValueError):
        DoStatementSyntax(loop_var="i")
    with pytest.raises(ValueError):
        DoStatementSyntax(loop_var="i", start=LiteralSyntax(
            kind="numeric", lexeme="1", span=document.span(0, 1)
        ))

    # DeferredStatementError is a valid payload for every supported kind and
    # keeps only the saved fault description, never a compile object.
    deferred_payload = DeferredStatementError(
        error_kind="invalid_do_spec",
        native_cause="incomplete DO= boundary expression",
        span=document.span(0, 9),
        display_text="do i=1 to ;",
    )
    for kind in ("DO", "ARRAY", "IF", "DATA", "SKIPPED"):
        statement = _statement(kind, deferred_payload, document, 0, 9)
        assert statement.payload is deferred_payload
        assert isinstance(statement.payload, DeferredStatementError)


def test_if_else_action_slots_accept_descriptor_union_only() -> None:
    document = _doc()
    condition = NameSyntax(name="flag", span=document.span(3, 7))
    action_deferred = DeferredStatementError(
        error_kind="parse_fault",
        native_cause="unsupported inline action: foo",
        span=document.span(13, 16),
        display_text="foo",
    )

    # The descriptor itself is a first-class action value (direct union).
    descriptor_if = IfStatementSyntax(condition=condition, then_action=action_deferred)
    assert descriptor_if.then_action is action_deferred
    assert isinstance(descriptor_if.then_action, DeferredStatementError)

    wrapper = StatementSyntax(
        kind="ASSIGN",
        payload=AssignStatementSyntax(
            target=NameTargetSyntax(name="b", span=document.span(8, 9)),
            value=LiteralSyntax(kind="numeric", lexeme="1", span=document.span(12, 13)),
        ),
        span=document.span(8, 13),
    )
    typed = IfStatementSyntax(condition=condition, then_action=wrapper)
    assert typed.then_action is wrapper
    bare = IfStatementSyntax(condition=condition)
    assert bare.then_action is None

    # ELSE mirrors the same three-way slot.
    else_with_descriptor = ElseStatementSyntax(action=action_deferred)
    assert else_with_descriptor.action is action_deferred
    else_with_wrapper = ElseStatementSyntax(action=wrapper)
    assert else_with_wrapper.action is wrapper
    assert ElseStatementSyntax().action is None

    # Strings / dicts / arbitrary objects are rejected; no filler kinds.
    for bad in ("foo", {"kind": "ASSIGN"}, 3, object()):
        with pytest.raises(TypeError):
            IfStatementSyntax(condition=condition, then_action=bad)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            ElseStatementSyntax(action=bad)  # type: ignore[arg-type]

    # The action slots are frozen like the rest of the syntax model.
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor_if.then_action = None  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        else_with_wrapper.action = None  # type: ignore[misc]


def test_dataset_reference_syntax_carries_typed_options_and_span() -> None:
    document = _doc()
    options = DatasetOptionSyntax(keep_vars=("id",), firstobs=2)
    reference = DatasetReferenceSyntax(name="inp", options=options, span=document.span(4, 20))
    assert reference.name == "inp"
    assert reference.options is options

    with pytest.raises(TypeError):
        DatasetReferenceSyntax(name="inp", options=options)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DatasetReferenceSyntax(  # type: ignore[arg-type]
            name="inp",
            options={"keep_vars": ("id",)},
            span=document.span(4, 20),
        )


def test_program_parse_result_is_pure_and_reports_has_errors() -> None:
    document = _doc()
    program = ProgramSyntax(source=document, span=document.span(0, len(document.text)))

    success = ProgramParseResult(program=program)
    assert success.program is program
    assert success.diagnostics == ()
    assert success.has_errors is False

    diagnostic = Diagnostic(
        code="PARSE_STATEMENT_INVALID",
        severity="error",
        message="invalid IF condition",
    )
    failure = ProgramParseResult(
        diagnostics=(diagnostic,),
    )
    assert failure.program is None
    assert failure.has_errors is True

    with pytest.raises(dataclasses.FrozenInstanceError):
        failure.program = program  # type: ignore[misc]


# --- IF condition display edits (shared projection info) ---------------------


def _edit_condition(document: SourceDocument) -> ExpressionSyntax:
    return BinarySyntax(
        operator="==",
        left=NameSyntax(name="x", span=document.span(0, 1)),
        right=LiteralSyntax(kind="numeric", lexeme="1", span=document.span(4, 5)),
        span=document.span(2, 3),
    )


def test_condition_display_edits_default_to_an_empty_tuple() -> None:
    document = _doc()

    payload = IfStatementSyntax(condition=_edit_condition(document))
    assert payload.condition_display_edits == ()

    subset = IfStatementSyntax(
        condition=_edit_condition(document),
        is_subset=True,
    )
    assert subset.condition_display_edits == ()


def test_condition_display_edits_stay_pure_immutable_pairs() -> None:
    document = _doc()
    edits = [
        (document.span(2, 3), "=="),
        (document.span(8, 10), "and"),
    ]

    payload = IfStatementSyntax(
        condition=_edit_condition(document),
        condition_display_edits=edits,  # type: ignore[arg-type]
    )
    # A list input is coerced into the pure tuple form; element pairs stay
    # plain (SourceSpan, str) values.
    assert isinstance(payload.condition_display_edits, tuple)
    assert payload.condition_display_edits == (
        (document.span(2, 3), "=="),
        (document.span(8, 10), "and"),
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        payload.condition_display_edits = ()  # type: ignore[misc]


def test_condition_display_edits_reject_non_pair_elements() -> None:
    document = _doc()
    bad_shapes = (
        "not a pair",
        (document.span(0, 1),),
        (document.span(0, 1), "==", "extra"),
        ("span", "=="),
        (document.span(0, 1), 3),
        (None, "=="),
    )
    for bad in bad_shapes:
        with pytest.raises((TypeError, ValueError)):
            IfStatementSyntax(
                condition=_edit_condition(document),
                condition_display_edits=(bad,),  # type: ignore[arg-type]
            )


def test_condition_display_edits_allow_only_the_known_replacements() -> None:
    document = _doc()
    for replacement in ("==", "!=", "and", "or", "not"):
        payload = IfStatementSyntax(
            condition=_edit_condition(document),
            condition_display_edits=((document.span(2, 3), replacement),),
        )
        assert payload.condition_display_edits[0][1] == replacement

    for replacement in ("=", "MOD", "mod", "<=", "===", "x", ""):
        with pytest.raises(ValueError):
            IfStatementSyntax(
                condition=_edit_condition(document),
                condition_display_edits=((document.span(2, 3), replacement),),
            )


def test_condition_display_edits_require_ordered_nonoverlapping_same_source_spans() -> None:
    document = _doc()

    # Out of order.
    with pytest.raises(ValueError):
        IfStatementSyntax(
            condition=_edit_condition(document),
            condition_display_edits=(
                (document.span(8, 10), "and"),
                (document.span(2, 3), "=="),
            ),
        )

    # Overlapping spans stay invalid; touching half-open spans are valid.
    with pytest.raises(ValueError):
        IfStatementSyntax(
            condition=_edit_condition(document),
            condition_display_edits=(
                (document.span(2, 4), "=="),
                (document.span(3, 6), "and"),
            ),
        )

    touching = IfStatementSyntax(
        condition=_edit_condition(document),
        condition_display_edits=(
            (document.span(2, 4), "=="),
            (document.span(4, 6), "and"),
        ),
    )
    assert touching.condition_display_edits == (
        (document.span(2, 4), "=="),
        (document.span(4, 6), "and"),
    )

    # Mixed source ids.
    other = SourceDocument("other", source_id="other")
    with pytest.raises(ValueError):
        IfStatementSyntax(
            condition=_edit_condition(document),
            condition_display_edits=(
                (document.span(2, 3), "=="),
                (other.span(0, 1), "and"),
            ),
        )

    # The ordered, same-source form is accepted and kept as-is.
    ordered = IfStatementSyntax(
        condition=_edit_condition(document),
        condition_display_edits=(
            (document.span(2, 3), "=="),
            (document.span(8, 10), "and"),
        ),
    )
    assert ordered.condition_display_edits == (
        (document.span(2, 3), "=="),
        (document.span(8, 10), "and"),
    )


# --- initial display ownership fields ----------------------------------------


def test_if_condition_display_span_accepts_source_span_or_none() -> None:
    document = _doc()

    payload = IfStatementSyntax(
        condition=_edit_condition(document),
        condition_display_span=document.span(0, 5),
    )
    assert payload.condition_display_span == document.span(0, 5)

    with pytest.raises(TypeError):
        IfStatementSyntax(
            condition=_edit_condition(document),
            condition_display_span=(0, 5),  # type: ignore[arg-type]
        )


def test_do_boundary_display_spans_stay_a_pipe_of_two_source_spans() -> None:
    document = _doc()

    payload = DoStatementSyntax(
        loop_var="i",
        start=LiteralSyntax(kind="numeric", lexeme="1", span=document.span(0, 1)),
        end=LiteralSyntax(kind="numeric", lexeme="5", span=document.span(2, 3)),
        boundary_display_spans=(document.span(0, 1), document.span(2, 3)),
    )
    assert payload.boundary_display_spans == (document.span(0, 1), document.span(2, 3))

    # A bare DO never carries boundary display spans.
    assert DoStatementSyntax().boundary_display_spans is None

    for bad in (
        (document.span(0, 1),),
        (document.span(0, 1), document.span(2, 3), document.span(4, 5)),
        document.span(0, 1),
        (document.span(0, 1), "5"),
    ):
        with pytest.raises(TypeError):
            DoStatementSyntax(
                loop_var="i",
                start=LiteralSyntax(kind="numeric", lexeme="1", span=document.span(0, 1)),
                end=LiteralSyntax(kind="numeric", lexeme="5", span=document.span(2, 3)),
                boundary_display_spans=bad,  # type: ignore[arg-type]
            )


def test_statement_legacy_display_span_keeps_the_initial_segment_shape() -> None:
    document = _doc()

    assert _statement("DO", DoStatementSyntax(), document, 0, 4).legacy_display_span is None

    carried = _statement(
        "DO",
        DoStatementSyntax(),
        document,
        0,
        4,
        legacy_display_span=document.span(0, 6),
    )
    assert carried.legacy_display_span == document.span(0, 6)

    with pytest.raises(TypeError):
        _statement("DO", DoStatementSyntax(), document, 0, 4, legacy_display_span=(0, 6))


def test_deferred_statement_error_accepts_display_or_none() -> None:
    document = _doc()

    deferred = DeferredStatementError(
        error_kind="parse_fault",
        native_cause="dataset option WHERE= is invalid: unsupported WHERE= arithmetic syntax",
        span=document.span(0, 5),
        diagnostic_display=ParseDiagnosticDisplay(source_text="a+b"),
    )
    assert deferred.diagnostic_display == ParseDiagnosticDisplay(source_text="a+b")

    with pytest.raises(TypeError):
        DeferredStatementError(
            error_kind="parse_fault",
            native_cause="cause",
            span=document.span(0, 5),
            diagnostic_display="a+b",  # type: ignore[arg-type]
        )


def test_parse_diagnostic_display_is_frozen_non_structure_text() -> None:
    display = ParseDiagnosticDisplay(source_text="a+b")

    with pytest.raises(dataclasses.FrozenInstanceError):
        display.source_text = "x"  # type: ignore[misc]

    with pytest.raises(TypeError):
        ParseDiagnosticDisplay(source_text=3)  # type: ignore[arg-type]


def _projection_facts_document() -> SourceDocument:
    return _doc()


def test_program_projection_facts_validate_elements_order_and_document() -> None:
    document = _projection_facts_document()

    facts = ProgramProjectionFacts(
        source=document,
        block_display_spans=((document.span(0, 5), True),),
    )
    assert facts.block_display_spans == ((document.span(0, 5), True),)

    # A non-pair sequence element stays invalid.
    with pytest.raises(TypeError):
        ProgramProjectionFacts(
            source=document,
            block_display_spans=([document.span(0, 5), True],),  # type: ignore[list-item]
        )

    # Out of order / overlapping blocks.
    with pytest.raises(ValueError):
        ProgramProjectionFacts(
            source=document,
            block_display_spans=(
                (document.span(10, 15), True),
                (document.span(0, 5), False),
            ),
        )
    with pytest.raises(ValueError):
        ProgramProjectionFacts(
            source=document,
            block_display_spans=(
                (document.span(0, 6), True),
                (document.span(5, 9), False),
            ),
        )

    # Outside the document.
    with pytest.raises(ValueError):
        ProgramProjectionFacts(
            source=document,
            block_display_spans=((document.span(0, len(document.text) + 1), True),),
        )

    # Mixed-source blocks and skip regions.
    other = SourceDocument("other", source_id="other")
    with pytest.raises(ValueError):
        ProgramProjectionFacts(
            source=document,
            block_display_spans=((other.span(0, 1), True),),
        )
    with pytest.raises(ValueError):
        ProgramProjectionFacts(
            source=document,
            skip_regions=(SkipRegionSyntax(kind="macro", span=other.span(0, 1)),),
        )

    # Bad element shapes and bad diagnostic displays.
    with pytest.raises(TypeError):
        ProgramProjectionFacts(
            source=document,
            skip_regions=("macro",),  # type: ignore[list-item]
        )
    with pytest.raises(TypeError):
        ProgramProjectionFacts(source=document, diagnostic_displays=("a+b",))  # type: ignore[list-item]

    with pytest.raises(TypeError):
        ProgramProjectionFacts(source="other")  # type: ignore[arg-type]


def test_program_parse_result_accepts_frozen_projection() -> None:
    document = _projection_facts_document()

    facts = ProgramProjectionFacts(source=document)
    result = ProgramParseResult(projection=facts)
    assert result.projection is facts
    assert result.projection.block_display_spans == ()
    assert result.projection.diagnostic_displays == ()
    assert result.projection.skip_regions == ()

    with pytest.raises(TypeError):
        ProgramParseResult(projection="facts")  # type: ignore[arg-type]
