"""General expression corpus covering frontend structure and real Rust
execution through Session.submit."""

from __future__ import annotations

import dataclasses
import re

import pyarrow as pa
import pytest

from limulus import Session
from limulus.dataset_option_expressions import parse_dataset_option_where
from limulus.syntax import (
    ApplyReferenceSyntax,
    BinarySyntax,
    CallSyntax,
    ConcatSyntax,
    DeferredExpressionError,
    LiteralSyntax,
    NameSyntax,
    SourceDocument,
    SourceSpan,
    SubscriptSyntax,
    UnarySyntax,
)
from limulus.syntax.expressions import (
    ExpressionParseError,
    GeneralExpressionResult,
    parse_expression,
    parse_general_expression,
)
from limulus.syntax.models import _EXPRESSION_MEMBERS


def _parsed(text: str, source_id: str = "doc"):
    document = SourceDocument(text, source_id=source_id)
    return document, parse_general_expression(document, document.span(0, len(text)))


def _strict(text: str, source_id: str = "doc") -> object:
    document = SourceDocument(text, source_id=source_id)
    return parse_expression(document, document.span(0, len(document.text)))


def _input_table() -> pa.Table:
    return pa.table({"a": [1.0], "b": [2.0]})


def _native_assign_cause(expression: str) -> str:
    """Native reality: the raw fault cause of one failing ASSIGN expression.

    The public rust submit reports the native parse cause verbatim as
    ``... assignment expression failed: '<expr>': <cause>``; no second
    evaluator is involved.
    """

    session = Session(backend="rust")
    session.load("in", _input_table())
    result = session.submit(f"data out; set in; y = {expression}; run;")
    assert result.success is False
    message = result.log[0].message
    match = re.search(r"assignment expression failed: '(.*)': (.+)$", message, re.DOTALL)
    assert match is not None, message
    return match.group(2)


def _native_where_failure(expression: str) -> tuple[str, str]:
    """Native reality: the statement-WHERE diagnostic for one failed slice."""

    session = Session(backend="rust")
    session.load("in", _input_table())
    result = session.submit(f"data out; set in; where {expression}; run;")
    assert result.success is False
    entry = result.log[0]
    return entry.code, entry.message


def _public_row(program: str) -> dict[str, object]:
    session = Session(backend="rust")
    session.load("in", _input_table())
    result = session.submit("data out; set in; " + program + " run;")
    assert result.success is True
    return result.datasets["out"].to_pylist()[0]


# --- native fault cause reality vs deferred descriptors -----------------------
#
# Every row was captured from the public rust submit path (assignments fail
# inside native with the verbatim cause; statement WHERE reports either the
# compile summary for the native expression-parse-error cause classes or the
# raw cause otherwise).  The deferred descriptor must carry the same cause,
# and the WHERE summary must equal the summary embedded in the public
# message, without compiling anything here.

_NATIVE_FAULT_CAUSES = (
    ("a||", "unexpected token: |"),
    ("a | b", "unexpected token: |"),
    ("[1,2]", "expression expected"),
    ("x[1,2]", "expected ']'"),
    ("1.2.3", "invalid number literal: 1.2.3"),
    ("18446744073709551616", "invalid integer literal: 18446744073709551616"),
    ("a b", "unexpected token at end of expression"),
    ("'abc", "unterminated string literal"),
    ("a + %", "unexpected token: %"),
    ("a\x1c> a", "unexpected token: \x1c"),
    ("a\x1d + 1", "unexpected token: \x1d"),
)


@pytest.mark.parametrize(("expression", "native_cause"), _NATIVE_FAULT_CAUSES)
def test_deferred_causes_match_native_reality(expression: str, native_cause: str) -> None:
    # Native side (public rust submit, live truth).
    assert _native_assign_cause(expression) == native_cause
    # Pure frontend side (deferred descriptor, no native call).
    _, result = _parsed(expression)
    deferred = result.expression
    assert isinstance(deferred, DeferredExpressionError), expression
    assert deferred.native_cause == native_cause, expression


_WHERE_SUMMARY_CASES = (
    ("a||", "RUNTIME_OPERATOR_NOT_SUPPORTED", "unexpected token: |", "invalid syntax (<limulus-condition>, line 1)"),
    ("a b", "RUNTIME_OPERATOR_NOT_SUPPORTED", "unexpected token at end of expression", "invalid syntax (<limulus-condition>, line 1)"),
    ("'abc", "RUNTIME_OPERATOR_NOT_SUPPORTED", "unterminated string literal", "unterminated string literal (detected at line 1) (<limulus-condition>, line 1)"),
)


@pytest.mark.parametrize(("expression", "code", "descriptor_cause", "summary"), _WHERE_SUMMARY_CASES)
def test_where_summary_matches_public_wrapper(
    expression: str, code: str, descriptor_cause: str, summary: str
) -> None:
    native_code, message = _native_where_failure(expression)
    assert native_code == code, message
    assert f": {summary}" in message, message

    document = SourceDocument(expression, source_id="doc")
    result = parse_general_expression(
        document, document.span(0, len(expression)), where_summary=True
    )
    deferred = result.expression
    assert isinstance(deferred, DeferredExpressionError)
    assert deferred.native_cause == descriptor_cause, expression
    assert deferred.where_syntax_summary == summary, expression


def test_where_non_compile_cause_keeps_the_raw_native_cause() -> None:
    # Not an expression-parse-error cause class: the owner reports the raw
    # native cause without compiling, so the descriptor keeps no summary.
    native_code, message = _native_where_failure("x[1,2]")
    assert native_code == "RUNTIME_EXPRESSION_EVALUATION_ERROR", message
    assert "expected ']'" in message

    _, result = _parsed("x[1,2]")
    deferred = result.expression
    assert isinstance(deferred, DeferredExpressionError)
    assert deferred.native_cause == "expected ']'"
    assert deferred.where_syntax_summary == ""


# --- public structural-expression compatibility ------------------------------


def test_typed_public_4_6_usual_assign_power() -> None:
    # Power contract: right-associative, parentheses win,
    # quoted strings are never rewritten.
    document = SourceDocument("a ** b ** c", source_id="doc")
    node = parse_expression(document, document.span(0, len(document.text)))
    assert isinstance(node, BinarySyntax)
    assert node.operator == "**"
    assert isinstance(node.right, BinarySyntax)
    assert node.right.operator == "**"

    grouped = SourceDocument("(a ** b) ** c", source_id="doc")
    grouped_node = parse_expression(grouped, grouped.span(0, len(grouped.text)))
    assert isinstance(grouped_node, BinarySyntax)
    assert isinstance(grouped_node.left, BinarySyntax)

    quoted = SourceDocument("'a ** 2'", source_id="doc")
    quoted_node = parse_expression(quoted, quoted.span(0, len(quoted.text)))
    assert isinstance(quoted_node, LiteralSyntax)
    assert quoted_node.lexeme == "'a ** 2'"

    # The public coordinator now uses the same structural expression as the
    # typed native evaluator: power is right-associative and quoted contents
    # remain string data.
    assert _public_row("a = 2; y = a ** 2 ** 3;")["y"] == 256.0
    assert _public_row("a = 2; y = (a ** 2) ** 3;")["y"] == 64.0
    assert _public_row("quoted = 'a ** 2';")["quoted"] == "a ** 2"


def test_power_values_match_across_typed_statement_owners() -> None:
    # Conditions, statement WHERE, and DO boundaries retain the same
    # right-associative power behavior as ordinary assignment.
    assert _public_row("a = 2; if a = 2 then y = a ** 2 ** 3;")["y"] == 256.0
    where_session = Session(backend="rust")
    where_session.load("in", pa.table({"a": [2.0], "b": [2.0]}))
    where_result = where_session.submit(
        "data out; set in; where a ** 2 ** 3 = 256; y = a; run;"
    )
    assert where_result.success is True, where_result.format_log()
    assert where_result.datasets["out"].to_pylist() == [
        {"a": 2.0, "b": 2.0, "y": 2.0}
    ]
    assert _public_row("a = 2; do i = 1 to 2 ** 2 ** 3; y = i; end;")["y"] == 256

    # Dataset-option WHERE remains its deliberately restrictive predicate
    # language, rather than becoming a general-expression owner.
    option_where = parse_dataset_option_where("a ** 2 > 0")
    assert option_where.expression is None
    assert option_where.diagnostics
    assert _native_assign_cause("'not numeric' * 2") == (
        "numeric value required, got 'not numeric'"
    )


def test_public_text_concat_is_accepted() -> None:
    # Parser contract: raw "||" is structural ConcatSyntax with
    # two or more non-empty parts; quoted contents are never rewritten.
    _, result = _parsed("a || b")
    assert isinstance(result.expression, ConcatSyntax)
    assert len(result.expression.parts) == 2

    _, result = _parsed("'a ** 2' || x")
    assert isinstance(result.expression, ConcatSyntax)
    assert result.expression.parts[0].lexeme == "'a ** 2'"

    assert _public_row("a = 'a'; b = 'B'; value = a || b;")["value"] == "aB"
    assert _public_row("a = 'a'; value = a || .;")["value"] == "a"


def test_expected_difference_7_text_plus_no_parser_type_inference() -> None:
    # Parser contract: "+" stays a plain binary node; the parser never
    # inspects operand types and never rewrites the call.
    _, result = _parsed("'a' + 'b'")
    assert isinstance(result.expression, BinarySyntax)
    assert result.expression.operator == "+"

    _, result = _parsed("1 + 'a'")
    assert isinstance(result.expression, BinarySyntax)
    assert result.expression.operator == "+"
    assert not isinstance(result.expression, CallSyntax)

    # Concatenation is restored for the shared Add evaluator when
    # both operands are text, while every mixed/non-text pair keeps arithmetic
    # coercion and NULL behavior. Existing string helpers keep their own rules.
    row = _public_row(
        "text = 'a' + 'B'; numeric_text = '1' + '2'; "
        "spaced = '' + '  雪  '; mixed = '1' + 2; null_text = 'a' + .; "
        "joined = cat('a', ' b'); stripped = cats(' a ', ' b '); "
        "right_trimmed = catt('a ', ' b '); delimited = catx('-', ' a ', ' b ');"
    )
    assert row["text"] == "aB"
    assert row["numeric_text"] == "12"
    assert row["spaced"] == "  雪  "
    assert row["mixed"] == 3.0
    assert row["null_text"] is None
    assert row["joined"] == "a b"
    assert row["stripped"] == "ab"
    assert row["right_trimmed"] == "a b"
    assert row["delimited"] == "a-b"
    assert _native_assign_cause("'a' + 2") == "numeric value required, got 'a'"


def test_typed_lowering_preserves_reserved_boolean_and_missing_names() -> None:
    # Parser contract: the reserved identifiers true/false/none/null
    # stay typed names; the native Bool/Null classification is applied when
    # the payload is lowered.  This corpus row prevents the native
    # classification from being lost while the parser keeps lexeme purity.
    for source in ("true", "false", "none", "null", "TRUE", "None"):
        document = SourceDocument(source, source_id="doc")
        node = parse_expression(document, document.span(0, len(source)))
        assert isinstance(node, NameSyntax), source
        assert node.name == source, source

    # The typed lowerer preserves the old native case-insensitive folding
    # while leaving ordinary names in the runtime binding path.
    row = _public_row(
        "t = true; f = false; n = none; m = null; upper_t = TRUE; upper_n = None;"
    )
    assert (row["t"], row["f"], row["n"], row["m"], row["upper_t"], row["upper_n"]) == (
        True,
        False,
        None,
        None,
        True,
        None,
    )


# --- native whitespace parity: U+001C..U+001F -----------------------------------
#
# Python ``str.isspace()`` and ``re \s`` treat the file/group/record/unit
# separators as whitespace; the native tokenizer follows the Unicode
# White_Space property (Rust ``char::is_whitespace``) and rejects them as
# unknown characters (live evidence: ``a\x1c + 1`` fails natively with
# "unexpected token: \x1c").  The frontend must refuse them in both entries
# while real whitespace (including U+00A0) stays ignorable on both sides.


@pytest.mark.parametrize("separator", ["\x1c", "\x1d", "\x1e", "\x1f"])
def test_control_only_slices_are_native_unknown_characters(separator: str) -> None:
    # Native tokenizer truth, taken directly from the tokenizer source
    # (native/limulus_native/src/expressions/mod.rs tokenize_expression):
    # the scan skips only ``ch.is_whitespace()`` and rejects every other
    # character with ``Err("unexpected token: {ch}")``; Rust's
    # ``char::is_whitespace`` follows the Unicode White_Space property,
    # which excludes U+001C..U+001F.  So a control-only slice is an unknown
    # character, never blank input ("expression expected").  Python
    # ``str.strip()`` disagrees and must not decide the blank check.
    for source in (separator, f" {separator}", f"{separator} ", f"  {separator}  "):
        document = SourceDocument(source, source_id="doc")
        with pytest.raises(ExpressionParseError) as error:
            parse_expression(document, document.span(0, len(source)))
        assert error.value.native_cause == f"unexpected token: {separator}", repr(source)

        _, result = _parsed(source)
        deferred = result.expression
        assert isinstance(deferred, DeferredExpressionError)
        assert deferred.native_cause == f"unexpected token: {separator}", repr(source)


def test_native_whitespace_only_slices_stay_expression_expected() -> None:
    # Real native whitespace (space, tab, U+00A0) alone is blank input on
    # both sides: the strict entry keeps "expression expected".
    for source in ("", "   ", " \xa0\t "):
        document = SourceDocument(source, source_id="doc")
        with pytest.raises(ExpressionParseError) as error:
            parse_expression(document, document.span(0, len(source)))
        assert error.value.native_cause == "expression expected", repr(source)


@pytest.mark.parametrize("separator", ["\x1c", "\x1d", "\x1e", "\x1f"])
def test_control_separators_are_refused_like_the_native_tokenizer(separator: str) -> None:
    document = SourceDocument(f"a{separator}> a", source_id="doc")
    with pytest.raises(ExpressionParseError) as error:
        parse_expression(document, document.span(0, len(document.text)))
    assert error.value.native_cause == f"unexpected token: {separator}"

    _, result = _parsed(f"a{separator}> a")
    deferred = result.expression
    assert isinstance(deferred, DeferredExpressionError)
    assert deferred.native_cause == f"unexpected token: {separator}"


@pytest.mark.parametrize("separator", ["\x1c", "\x1d", "\x1e", "\x1f"])
def test_control_separators_never_extend_identifiers(separator: str) -> None:
    _, result = _parsed(f"a b{separator}")
    deferred = result.expression
    assert isinstance(deferred, DeferredExpressionError)
    assert deferred.native_cause == f"unexpected token: {separator}"


def test_real_unicode_whitespace_stays_ignorable_like_native() -> None:
    document = SourceDocument("a\xa0+ b", source_id="doc")
    node = parse_expression(document, document.span(0, len(document.text)))
    assert isinstance(node, BinarySyntax)
    assert node.operator == "+"

    # Live native evidence: U+00A0 is skipped by the native tokenizer too.
    assert _public_row("y = a\xa0+ 1;")["y"] == 2.0


# --- new-acceptance structure corpus (compact rows) -----------------------------
#
# These rows freeze the established structural acceptance as one fixed
# corpus; they deliberately restate the contract without duplicating the
# deeper per-variant tests elsewhere.

_STRUCTURE_CORPUS = (
    # concat precedence: loosest, left-associative; parentheses group
    ("a and b || c + d", ConcatSyntax),
    ("a ** b || c", ConcatSyntax),
    ("(a||b)||c", ConcatSyntax),
    ("a||(b||c)", ConcatSyntax),
    # call commas never become concat operands
    ("cats(a||b,c)", CallSyntax),
    ("f(a || b || c, d)", CallSyntax),
    # quoted operators/format boundaries
    ("'a || b' || x", ConcatSyntax),
    ("put(x, best12.)", CallSyntax),
    ("put(x, 'z8.2')", CallSyntax),
    # "+" keeps binary structure with no type inference
    ("'1' + '2'", BinarySyntax),
)


@pytest.mark.parametrize(("source", "root_kind"), _STRUCTURE_CORPUS)
def test_structure_corpus(source: str, root_kind) -> None:
    _, result = _parsed(source)
    assert not isinstance(result.expression, DeferredExpressionError), source
    assert isinstance(result.expression, root_kind), source


@pytest.mark.parametrize("source", ("x[i||j]", "y{i||j}"))
def test_structure_corpus_subscript_concat(source: str) -> None:
    _, result = _parsed(source)
    assert isinstance(result.expression, SubscriptSyntax), source
    assert isinstance(result.expression.index, ConcatSyntax), source


def test_structure_corpus_call_argument_shapes() -> None:
    _, result = _parsed("cats(a||b,c)")
    call = result.expression
    assert isinstance(call, CallSyntax)
    assert len(call.arguments) == 2
    assert isinstance(call.arguments[0], ConcatSyntax)
    assert isinstance(call.arguments[1], NameSyntax)

    _, result = _parsed("put(x, best12.)")
    call = result.expression
    assert isinstance(call, CallSyntax)
    assert call.arguments[1].kind == "format"  # type: ignore[union-attr]
    assert call.arguments[1].lexeme == "best12."  # type: ignore[union-attr]

    _, result = _parsed("put(x, 'z8.2')")
    call = result.expression
    assert call.arguments[1].kind == "string"  # type: ignore[union-attr]
    assert call.arguments[1].lexeme == "'z8.2'"  # type: ignore[union-attr]


_LITERAL_LEXEME_CORPUS = (
    (".", LiteralSyntax, "missing"),
    (".5", LiteralSyntax, "numeric"),
    ("5.", LiteralSyntax, "numeric"),
    ("9223372036854775807", LiteralSyntax, "numeric"),
    ("18446744073709551615", LiteralSyntax, "numeric"),
    ("'a ** 2'", LiteralSyntax, "string"),
    ("'a+b'", LiteralSyntax, "string"),
    ("best12.", NameSyntax, None),
    ("-9223372036854775808", UnarySyntax, None),
)


@pytest.mark.parametrize(("lexeme", "node_kind", "literal_kind"), _LITERAL_LEXEME_CORPUS)
def test_literal_lexeme_corpus(lexeme: str, node_kind, literal_kind) -> None:
    document = SourceDocument(lexeme, source_id="doc")
    node = parse_expression(document, document.span(0, len(lexeme)))
    assert isinstance(node, node_kind), lexeme
    if literal_kind is not None:
        assert node.kind == literal_kind  # type: ignore[union-attr]
        assert node.lexeme == lexeme  # type: ignore[union-attr]
    if isinstance(node, UnarySyntax):
        # The parser transports lexemes; the minimum-i64 special case is a
        # lowering decision, never folded here.
        assert node.operator == "-"
        assert node.operand.lexeme == "9223372036854775808"  # type: ignore[union-attr]


def test_lexeme_fault_corpus_stays_whole_expression_deferred() -> None:
    for source, cause in (
        ("18446744073709551616", "invalid integer literal: 18446744073709551616"),
        ("1.2.3", "invalid number literal: 1.2.3"),
    ):
        _, result = _parsed(source)
        deferred = result.expression
        assert isinstance(deferred, DeferredExpressionError), source
        assert deferred.error_kind == "unsupported_literal", source
        assert deferred.native_cause == cause, source


# --- display-only raw text: typed trees carry the semantics ---------------------


def test_semantics_reconstruct_from_typed_fields_without_the_source() -> None:
    document = SourceDocument("1 + 'a'", source_id="doc")
    node = parse_expression(document, document.span(0, len(document.text)))
    expected = BinarySyntax(
        operator="+",
        left=LiteralSyntax(kind="numeric", lexeme="1", span=document.span(0, 1)),
        right=LiteralSyntax(kind="string", lexeme="'a'", span=document.span(4, 7)),
        span=document.span(2, 3),
    )
    assert node == expected


def test_tree_equality_is_independent_of_the_document_instance() -> None:
    first = SourceDocument("a + b", source_id="doc")
    second = SourceDocument("a + b", source_id="doc")
    assert parse_expression(first, first.span(0, 5)) == parse_expression(
        second, second.span(0, 5)
    )


def test_results_leak_no_framework_tree_or_compile_object() -> None:
    document = SourceDocument("apply('t', a) + a||", source_id="doc")
    result = parse_general_expression(
        document, document.span(0, len(document.text)), where_summary=True
    )

    def _walk(value: object) -> None:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            assert type(value) in (
                *_EXPRESSION_MEMBERS,
                ApplyReferenceSyntax,
                SourceSpan,
                GeneralExpressionResult,
            ), type(value)
            for field in dataclasses.fields(value):
                _walk(getattr(value, field.name))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)
            return
        assert isinstance(value, (str, int, bool, type(None))), value

    _walk(result)
    assert isinstance(result.expression, DeferredExpressionError)
    assert result.expression.error_kind in ("parse_fault", "unsupported_literal")


def test_descriptor_display_text_is_the_raw_slice_and_nothing_else() -> None:
    document = SourceDocument("prefix b = 1.2.3;", source_id="doc")
    expression_span = document.span(11, 16)
    result = parse_general_expression(document, expression_span, where_summary=False)
    deferred = result.expression
    assert isinstance(deferred, DeferredExpressionError)
    assert deferred.display_text == "1.2.3"
    assert deferred.span == expression_span
    assert deferred.native_cause == "invalid number literal: 1.2.3"
