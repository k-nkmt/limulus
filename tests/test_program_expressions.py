"""Focused structural tests for the initial general expression parser."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from limulus.syntax import (
    BinarySyntax,
    CallSyntax,
    ConcatSyntax,
    ExpressionSyntax,
    LiteralSyntax,
    NameSyntax,
    SourceDocument,
    SourceSpan,
    SubscriptSyntax,
    UnarySyntax,
)
from limulus.syntax.expressions import ExpressionParseError, parse_expression

REPO_ROOT = Path(__file__).resolve().parents[1]


def _expr(text: str, source_id: str = "doc") -> tuple[SourceDocument, ExpressionSyntax]:
    document = SourceDocument(text, source_id=source_id)
    return document, parse_expression(document, document.span(0, len(text)))


def _slice(document: SourceDocument, span) -> str:
    return document.text[span.start : span.end]


# --- scalar literals -------------------------------------------------------


def test_missing_dot_is_a_missing_literal() -> None:
    document, node = _expr(".")
    assert isinstance(node, LiteralSyntax)
    assert node.kind == "missing"
    assert node.lexeme == "."
    assert _slice(document, node.span) == "."


def test_decimal_float_lexemes_are_preserved_exactly() -> None:
    document, node = _expr(".5")
    assert isinstance(node, LiteralSyntax)
    assert node.kind == "numeric"
    assert node.lexeme == ".5"

    document, node = _expr("1.5")
    assert node.kind == "numeric"
    assert node.lexeme == "1.5"

    # ``5.`` keeps its trailing dot as a float lexeme (never rewritten).
    document, node = _expr("5.")
    assert node.kind == "numeric"
    assert node.lexeme == "5."


def test_quoted_strings_keep_quotes_and_inner_operators() -> None:
    document, node = _expr("'a ** 2'")
    assert isinstance(node, LiteralSyntax)
    assert node.kind == "string"
    assert node.lexeme == "'a ** 2'"

    document, node = _expr('"x+y"')
    assert node.kind == "string"
    assert node.lexeme == '"x+y"'


def test_quoted_dot_is_a_string_distinct_from_missing() -> None:
    document, node = _expr("'.'")
    assert isinstance(node, LiteralSyntax)
    assert node.kind == "string"
    assert node.lexeme == "'.'"

    document, node = _expr(".")
    assert node.kind == "missing"
    assert node.lexeme == "."


def test_dot_run_before_closer_is_a_string_literal_like_native() -> None:
    # The native tokenizer classifies a digit run ending in '.' directly
    # before ',' or ')' as text ("8." in put(..., 8.)), never as a number.
    document, node = _expr("(5.)")
    assert isinstance(node, LiteralSyntax)
    assert node.kind == "string"
    assert node.lexeme == "5."

    # Away from a closer the same lexeme stays numeric.
    document, node = _expr("5.")
    assert isinstance(node, LiteralSyntax)
    assert node.kind == "numeric"
    assert node.lexeme == "5."


def test_names_keep_raw_case_and_dots() -> None:
    document, node = _expr("amount")
    assert isinstance(node, NameSyntax)
    assert node.name == "amount"
    assert _slice(document, node.span) == "amount"

    document, node = _expr("work.a")
    assert isinstance(node, NameSyntax)
    assert node.name == "work.a"

    document, node = _expr("Amount")
    assert node.name == "Amount"


def test_integer_lexemes_within_i64_and_u64_boundaries() -> None:
    for lexeme in ("0", "5", "9223372036854775807", "9223372036854775808", "18446744073709551615"):
        document, node = _expr(lexeme)
        assert isinstance(node, LiteralSyntax)
        assert node.kind == "numeric"
        assert node.lexeme == lexeme
        assert _slice(document, node.span) == lexeme


# --- unary operators --------------------------------------------------------


def test_unary_neg_and_not_are_operator_occurrence_nodes() -> None:
    document, node = _expr("-a")
    assert isinstance(node, UnarySyntax)
    assert node.operator == "-"
    assert isinstance(node.operand, NameSyntax)
    assert node.operand.name == "a"
    assert _slice(document, node.span) == "-"

    document, node = _expr("not a")
    assert isinstance(node, UnarySyntax)
    assert node.operator == "not"
    assert _slice(document, node.span) == "not"

    document, node = _expr("NOT a")
    assert node.operator == "not"


def test_unary_neg_does_not_fold_into_a_literal() -> None:
    # The parser transports lexemes; the i64 -> u64 order and the minimum
    # i64 unary special case are reproduced later by the native lowering.
    document, node = _expr("-5")
    assert isinstance(node, UnarySyntax)
    assert node.operator == "-"
    operand = node.operand
    assert isinstance(operand, LiteralSyntax)
    assert operand.kind == "numeric"
    assert operand.lexeme == "5"


def test_unary_binds_looser_than_power_on_its_right() -> None:
    # Native binding powers: unary operand parses at 11, power is (11, 11),
    # so ``-a ** b`` is -(a ** b) and ``a ** -b`` is a ** (-b).
    document, node = _expr("-a ** b")
    assert isinstance(node, UnarySyntax)
    assert node.operator == "-"
    inner = node.operand
    assert isinstance(inner, BinarySyntax)
    assert inner.operator == "**"
    assert isinstance(inner.left, NameSyntax)
    assert isinstance(inner.right, NameSyntax)
    assert _slice(document, node.span) == "-"

    document, node = _expr("a ** -b")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "**"
    assert isinstance(node.left, NameSyntax)
    assert isinstance(node.right, UnarySyntax)
    assert node.right.operator == "-"


def test_double_prefix_unary_nesting() -> None:
    document, node = _expr("--a")
    assert isinstance(node, UnarySyntax)
    assert node.operator == "-"
    assert isinstance(node.operand, UnarySyntax)
    assert node.operand.operator == "-"
    assert isinstance(node.operand.operand, NameSyntax)

    document, node = _expr("not not a")
    assert isinstance(node, UnarySyntax)
    assert node.operator == "not"
    assert isinstance(node.operand, UnarySyntax)
    assert node.operand.operator == "not"


def test_power_chain_with_prefix_unary_inside() -> None:
    document, node = _expr("2 ** -2 ** 2")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "**"
    right = node.right
    assert isinstance(right, UnarySyntax)
    assert right.operator == "-"
    inner = right.operand
    assert isinstance(inner, BinarySyntax)
    assert inner.operator == "**"


def test_no_unary_plus_like_native() -> None:
    # The native primary grammar only accepts ``-`` and ``not`` prefixes.
    with pytest.raises(ExpressionParseError) as error:
        _expr("+5")
    assert error.value.native_cause == "expression expected"


# --- binary precedence ladder ----------------------------------------------


def test_or_is_looser_than_and() -> None:
    document, node = _expr("a or b and c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "or"
    assert isinstance(node.left, NameSyntax)
    right = node.right
    assert isinstance(right, BinarySyntax)
    assert right.operator == "and"


def test_and_binds_tighter_than_comparison() -> None:
    document, node = _expr("a and b == c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "and"
    right = node.right
    assert isinstance(right, BinarySyntax)
    assert right.operator == "=="


def test_comparison_binds_tighter_than_additive() -> None:
    document, node = _expr("a == b + c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "=="
    right = node.right
    assert isinstance(right, BinarySyntax)
    assert right.operator == "+"


def test_additive_binds_tighter_than_multiplicative() -> None:
    document, node = _expr("a + b * c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "+"
    right = node.right
    assert isinstance(right, BinarySyntax)
    assert right.operator == "*"


def test_comparison_chain_is_left_associative() -> None:
    document, node = _expr("a < b < c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "<"
    left = node.left
    assert isinstance(left, BinarySyntax)
    assert left.operator == "<"
    assert isinstance(left.left, NameSyntax)
    assert isinstance(left.right, NameSyntax)


def test_comparisons_share_one_level() -> None:
    document, node = _expr("a == b < c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "<"
    left = node.left
    assert isinstance(left, BinarySyntax)
    assert left.operator == "=="


def test_multiplicative_chain_is_left_associative() -> None:
    document, node = _expr("a * b mod c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "mod"
    left = node.left
    assert isinstance(left, BinarySyntax)
    assert left.operator == "*"

    document, node = _expr("a mod b mod c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "mod"
    assert isinstance(node.left, BinarySyntax)
    assert node.left.operator == "mod"


def test_additive_chain_is_left_associative() -> None:
    document, node = _expr("a - b - c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "-"
    left = node.left
    assert isinstance(left, BinarySyntax)
    assert left.operator == "-"


# --- operator aliases and normalization ------------------------------------


def test_equality_alias_normalizes_to_eq_keeping_raw_span() -> None:
    document, node = _expr("a = b")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "=="
    assert _slice(document, node.span) == "="
    assert isinstance(node.left, NameSyntax)
    assert isinstance(node.right, NameSyntax)
    assert _slice(document, node.left.span) == "a"
    assert _slice(document, node.right.span) == "b"

    document, node = _expr("a == b")
    assert node.operator == "=="
    assert _slice(document, node.span) == "=="


def test_not_equal_aliases_normalize_to_ne() -> None:
    for source, raw in (("a != b", "!="), ("a ^= b", "^="), ("a ~= b", "~=")):
        document, node = _expr(source)
        assert isinstance(node, BinarySyntax)
        assert node.operator == "!="
        assert _slice(document, node.span) == raw


def test_keyword_operators_match_any_case_and_normalize() -> None:
    document, node = _expr("a AND b OR c MOD d")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "or"
    left = node.left
    assert isinstance(left, BinarySyntax)
    assert left.operator == "and"
    right = node.right
    assert isinstance(right, BinarySyntax)
    assert right.operator == "mod"


# --- power associativity and explicit parentheses ----------------------------


def test_power_is_right_associative() -> None:
    document, node = _expr("a ** b ** c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "**"
    assert isinstance(node.left, NameSyntax)
    right = node.right
    assert isinstance(right, BinarySyntax)
    assert right.operator == "**"
    assert isinstance(right.left, NameSyntax)
    assert isinstance(right.right, NameSyntax)


def test_explicit_parentheses_override_right_associativity() -> None:
    document, node = _expr("(a ** b) ** c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "**"
    left = node.left
    assert isinstance(left, BinarySyntax)
    assert left.operator == "**"
    assert isinstance(left.left, NameSyntax)
    assert isinstance(left.right, NameSyntax)
    assert isinstance(node.right, NameSyntax)


def test_parentheses_group_across_all_levels() -> None:
    document, node = _expr("(a + b) * c")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "*"
    left = node.left
    assert isinstance(left, BinarySyntax)
    assert left.operator == "+"
    assert isinstance(left.left, NameSyntax)
    assert isinstance(left.right, NameSyntax)
    assert _slice(document, left.span) == "+"

    document, node = _expr("2 ** (1 + 2)")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "**"
    right = node.right
    assert isinstance(right, BinarySyntax)
    assert right.operator == "+"


def test_parentheses_are_transparent_around_scalars_and_unary() -> None:
    document, node = _expr("((a))")
    assert isinstance(node, NameSyntax)
    assert node.name == "a"

    document, node = _expr("-(a + b)")
    assert isinstance(node, UnarySyntax)
    assert node.operator == "-"
    inner = node.operand
    assert isinstance(inner, BinarySyntax)
    assert inner.operator == "+"

    document, node = _expr("(.)")
    assert isinstance(node, LiteralSyntax)
    assert node.kind == "missing"


# --- source-anchored spans ---------------------------------------------------


def test_spans_are_document_anchored_codepoint_positions() -> None:
    document, node = _expr("x = 1.5")
    assert isinstance(node, BinarySyntax)
    assert node.span.start == 2
    assert node.span.end == 3
    assert node.span.line == 1
    assert node.span.column == 3
    literal = node.right
    assert isinstance(literal, LiteralSyntax)
    assert literal.span.start == 4
    assert literal.span.end == 7
    assert literal.span.column == 5
    assert document.text[literal.span.start : literal.span.end] == "1.5"


def test_subspan_entry_keeps_document_anchored_positions() -> None:
    document = SourceDocument("x = 1.5; y = .5", source_id="doc")
    node = parse_expression(document, document.span(4, 7))
    assert isinstance(node, LiteralSyntax)
    assert node.lexeme == "1.5"
    assert document.text[node.span.start : node.span.end] == "1.5"
    assert node.span.line == 1
    assert node.span.column == 5
    assert node.span.source_id == "doc"


def test_multiline_slices_anchor_line_and_column() -> None:
    document = SourceDocument("x = 1.5\ny = .5", source_id="doc")
    node = parse_expression(document, document.span(12, 14))
    assert isinstance(node, LiteralSyntax)
    assert node.lexeme == ".5"
    assert node.span.line == 2
    assert node.span.column == 5


def test_multiline_expression_inside_one_slice() -> None:
    document, node = _expr("a **\n  b")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "**"
    assert document.text[node.span.start : node.span.end] == "**"
    assert node.right.span.line == 2


# --- internal parse failures --------------------------------------------------


def test_integer_beyond_u64_reports_the_native_cause() -> None:
    with pytest.raises(ExpressionParseError) as error:
        _expr("18446744073709551616")
    assert error.value.native_cause == "invalid integer literal: 18446744073709551616"


def test_number_with_two_dots_reports_the_native_cause() -> None:
    with pytest.raises(ExpressionParseError) as error:
        _expr("1.2.3")
    assert error.value.native_cause == "invalid number literal: 1.2.3"


def test_unterminated_string_reports_the_native_cause() -> None:
    with pytest.raises(ExpressionParseError) as error:
        _expr("'abc")
    assert error.value.native_cause == "unterminated string literal"


def test_unsupported_character_reports_the_native_cause() -> None:
    with pytest.raises(ExpressionParseError) as error:
        _expr("a + %")
    assert error.value.native_cause == "unexpected token: %"


def test_missing_operand_reports_expression_expected() -> None:
    for source in ("", "   ", "a +", "a *", "a and"):
        with pytest.raises(ExpressionParseError) as error:
            _expr(source)
        assert error.value.native_cause == "expression expected", source


def test_trailing_token_reports_end_of_expression() -> None:
    for source in ("a b", "a 1", "a not b"):
        with pytest.raises(ExpressionParseError) as error:
            _expr(source)
        assert error.value.native_cause == "unexpected token at end of expression", source


def test_unclosed_paren_reports_expected_rparen() -> None:
    for source in ("(a", "(a b)", "(a,"):
        with pytest.raises(ExpressionParseError) as error:
            _expr(source)
        assert error.value.native_cause == "expected ')'", source


def test_parse_error_carries_whole_expression_span_and_display_text() -> None:
    document = SourceDocument("b = 1.2.3;", source_id="doc")
    with pytest.raises(ExpressionParseError) as error:
        parse_expression(document, document.span(4, 9))
    assert document.text[error.value.span.start : error.value.span.end] == "1.2.3"
    assert error.value.display_text == "1.2.3"
    assert error.value.span.source_id == "doc"


# --- entry contract -----------------------------------------------------------


def test_entry_requires_a_span_from_the_same_source_document() -> None:
    document = SourceDocument("a + b", source_id="doc")
    other = SourceDocument("a + b", source_id="other")

    with pytest.raises(TypeError):
        parse_expression("a + b", document.span(0, 5))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        parse_expression(document, "a + b")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        parse_expression(document, other.span(0, 5))
    with pytest.raises(ValueError):
        parse_expression(document, SourceSpan(start=0, end=6, line=1, column=1, source_id="doc"))


def test_parsing_is_deterministic_and_pure() -> None:
    document = SourceDocument("a ** b ** c + .5", source_id="doc")
    span = document.span(0, len(document.text))
    first = parse_expression(document, span)
    second = parse_expression(document, span)
    assert first == second
    assert document.text == "a ** b ** c + .5"


def test_result_tree_contains_only_typed_syntax_nodes() -> None:
    document, node = _expr("(a + .5) ** -b or not c")
    stack = [node]
    allowed = (LiteralSyntax, NameSyntax, UnarySyntax, BinarySyntax)
    while stack:
        current = stack.pop()
        assert isinstance(current, allowed)
        if isinstance(current, UnarySyntax):
            stack.append(current.operand)
        elif isinstance(current, BinarySyntax):
            stack.extend((current.left, current.right))


def test_import_and_parse_stay_free_of_data_libraries() -> None:
    script = textwrap.dedent(
        """
        import sys

        import limulus.syntax.expressions as expressions

        from limulus.syntax import SourceDocument

        document = SourceDocument("(a ** b) ** c", source_id="doc")
        node = expressions.parse_expression(document, document.span(0, len(document.text)))
        assert node is not None

        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules, denied
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"isolated expression parse failed:\n{result.stdout}\n{result.stderr}"


# --- calls, subscripts, format tokens, concat ---------------------------------


def test_zero_argument_call_keeps_empty_arguments() -> None:
    document, node = _expr("f()")
    assert isinstance(node, CallSyntax)
    assert node.name == "f"
    assert node.arguments == ()
    assert _slice(document, node.span) == "f()"


def test_call_arguments_keep_order_and_nested_calls() -> None:
    document, node = _expr("f(g(1), h(2, 3))")
    assert isinstance(node, CallSyntax)
    assert node.name == "f"
    first, second = node.arguments
    assert isinstance(first, CallSyntax)
    assert first.name == "g"
    assert len(first.arguments) == 1
    assert isinstance(second, CallSyntax)
    assert second.name == "h"
    assert len(second.arguments) == 2
    assert _slice(document, node.span) == "f(g(1), h(2, 3))"


def test_call_arguments_are_concat_level_expressions() -> None:
    # Native call arguments parse at min_bp 0, so "||" inside call brackets
    # is a concat argument, matching the existing normalization.
    document, node = _expr("f(a || b)")
    assert isinstance(node, CallSyntax)
    assert len(node.arguments) == 1
    concat = node.arguments[0]
    assert isinstance(concat, ConcatSyntax)
    assert len(concat.parts) == 2


def test_array_element_call_stays_a_call_without_binding() -> None:
    # Native parses NAME "(" as FunctionCall for every identifier; array
    # resolution stays with the runtime owner, never the parser.
    document, node = _expr("arr(i)")
    assert isinstance(node, CallSyntax)
    assert node.name == "arr"
    assert len(node.arguments) == 1
    assert isinstance(node.arguments[0], NameSyntax)
    assert node.arguments[0].name == "i"

    document, node = _expr("ARR(1)")
    assert node.name == "ARR"


def test_call_with_expression_arguments_and_nested_operators() -> None:
    document, node = _expr("put(x, a + b, c ** 2)")
    assert isinstance(node, CallSyntax)
    assert node.name == "put"
    assert len(node.arguments) == 3
    middle = node.arguments[1]
    assert isinstance(middle, BinarySyntax)
    assert middle.operator == "+"
    last = node.arguments[2]
    assert isinstance(last, BinarySyntax)
    assert last.operator == "**"


def test_subscript_bracket_and_brace_forms_are_name_placement() -> None:
    # Native ArrayRef supports only NAME "[" expr "]" and NAME "{" expr "}"
    # placement; the model records no delimiter.
    document, node = _expr("arr[i]")
    assert isinstance(node, SubscriptSyntax)
    assert isinstance(node.base, NameSyntax)
    assert node.base.name == "arr"
    assert isinstance(node.index, NameSyntax)
    assert node.index.name == "i"
    assert _slice(document, node.span) == "arr[i]"

    document, node = _expr("arr{1}")
    assert isinstance(node, SubscriptSyntax)
    assert node.base.name == "arr"
    assert isinstance(node.index, LiteralSyntax)
    assert node.index.lexeme == "1"
    assert _slice(document, node.span) == "arr{1}"


def test_subscript_index_accepts_nested_calls() -> None:
    document, node = _expr("arr[f(1)]")
    assert isinstance(node, SubscriptSyntax)
    index = node.index
    assert isinstance(index, CallSyntax)
    assert index.name == "f"


# --- concat operand and comma boundaries --------------------------------------


def test_cats_concat_call_keeps_two_arguments() -> None:
    # cats(a||b,c) has exactly two arguments: the comma is an argument
    # delimiter, never absorbed into the concat operand list, and no empty
    # string operand is supplied anywhere.
    document, node = _expr("cats(a||b,c)")
    assert isinstance(node, CallSyntax)
    assert node.name == "cats"
    assert len(node.arguments) == 2
    first = node.arguments[0]
    assert isinstance(first, ConcatSyntax)
    assert len(first.parts) == 2
    assert isinstance(first.parts[0], NameSyntax)
    assert isinstance(first.parts[1], NameSyntax)
    second = node.arguments[1]
    assert isinstance(second, NameSyntax)
    assert second.name == "c"
    assert _slice(document, node.span) == "cats(a||b,c)"


def test_concat_call_arguments_stay_split_after_chains() -> None:
    # f(a || b || c, d): the flattened chain is one argument and "d" is the
    # second argument, never a concat part.
    document, node = _expr("f(a || b || c, d)")
    assert isinstance(node, CallSyntax)
    assert len(node.arguments) == 2
    concat = node.arguments[0]
    assert isinstance(concat, ConcatSyntax)
    assert len(concat.parts) == 3
    tail = node.arguments[1]
    assert isinstance(tail, NameSyntax)
    assert tail.name == "d"


def test_concat_argument_keeps_quoted_operands_untouched() -> None:
    # cats("a" || "b", "c") keeps two arguments and the quoted lexemes.
    document, node = _expr('cats("a" || "b", "c")')
    assert isinstance(node, CallSyntax)
    assert len(node.arguments) == 2
    first = node.arguments[0]
    assert isinstance(first, ConcatSyntax)
    assert len(first.parts) == 2
    assert isinstance(first.parts[0], LiteralSyntax)
    assert first.parts[0].lexeme == '"a"'
    assert isinstance(first.parts[1], LiteralSyntax)
    assert first.parts[1].lexeme == '"b"'


def test_put_call_arguments_keep_concat_argument_boundary() -> None:
    document, node = _expr("put(x,a||b)")
    assert isinstance(node, CallSyntax)
    assert node.name == "put"
    assert len(node.arguments) == 2
    concat = node.arguments[1]
    assert isinstance(concat, ConcatSyntax)
    assert len(concat.parts) == 2


def test_format_literal_stays_outside_the_concat_operand_list() -> None:
    # put(x,8.2.,a||b): three arguments; only the second top-level argument
    # is format-classified and the third argument is a plain concat.
    document, node = _expr("put(x,8.2.,a||b)")
    assert isinstance(node, CallSyntax)
    assert len(node.arguments) == 3
    second = node.arguments[1]
    assert isinstance(second, LiteralSyntax)
    assert second.kind == "format"
    assert second.lexeme == "8.2."
    third = node.arguments[2]
    assert isinstance(third, ConcatSyntax)
    assert len(third.parts) == 2


def test_valid_concat_admits_paren_child() -> None:
    # a||(b||c): explicit parentheses keep the nested concat structure as
    # the second part; the inner chain is not flattened into the outer one.
    document, node = _expr("a||(b||c)")
    assert isinstance(node, ConcatSyntax)
    assert len(node.parts) == 2
    inner = node.parts[1]
    assert isinstance(inner, ConcatSyntax)
    assert len(inner.parts) == 2


def test_valid_concat_admits_subscript_children() -> None:
    document, node = _expr("a||x[i||j]")
    assert isinstance(node, ConcatSyntax)
    assert len(node.parts) == 2
    inner = node.parts[1]
    assert isinstance(inner, SubscriptSyntax)
    index = inner.index
    assert isinstance(index, ConcatSyntax)
    assert len(index.parts) == 2

    document, node = _expr("a||y{i||j}")
    assert isinstance(node, ConcatSyntax)
    inner = node.parts[1]
    assert isinstance(inner, SubscriptSyntax)
    assert isinstance(inner.index, ConcatSyntax)


def test_concat_is_structured_inside_parens_and_subscripts() -> None:
    # Full general expressions accept "||" inside parentheses, subscript
    # brackets, and unary/binary operands; no general-expression position
    # keeps a raw "||" refusal any more.
    document, node = _expr("(b||c)")
    assert isinstance(node, ConcatSyntax)
    assert len(node.parts) == 2

    document, node = _expr("x[i||j]")
    assert isinstance(node, SubscriptSyntax)
    assert isinstance(node.index, ConcatSyntax)

    document, node = _expr("y{i||j}")
    assert isinstance(node, SubscriptSyntax)
    assert isinstance(node.index, ConcatSyntax)

    document, node = _expr("a + (b||c)")
    assert isinstance(node, BinarySyntax)
    assert node.operator == "+"
    right = node.right
    assert isinstance(right, ConcatSyntax)
    assert len(right.parts) == 2

    document, node = _expr("-(a||b)")
    assert isinstance(node, UnarySyntax)
    assert isinstance(node.operand, ConcatSyntax)

    document, node = _expr("a||(x+(y||z))")
    assert isinstance(node, ConcatSyntax)
    second = node.parts[1]
    assert isinstance(second, BinarySyntax)
    assert second.operator == "+"
    assert isinstance(second.right, ConcatSyntax)


def test_concat_operand_run_over_two_delimiters_is_refused() -> None:
    # No empty-operand supplementation: "||", "a||", "||b", and "a||||b"
    # stay invalid with the missing-operand cause.  Their whole-expression
    # Whole-expression Deferred conversion belongs to the deferred
    # frontend entry.
    for source in ("||", "a||", "|| b", "a || || b", "a||||b", "f(a||)", "x[i||]"):
        with pytest.raises(ExpressionParseError) as error:
            _expr(source)
        assert error.value.native_cause == "expression expected", source


def test_call_trailing_comma_drops_the_empty_tail() -> None:
    # The trailing-comma call shape stays a one-argument call (the empty
    # tail is a call-level allowance, not a concat operand).
    document, node = _expr("f(a,)")
    assert isinstance(node, CallSyntax)
    assert len(node.arguments) == 1
    assert isinstance(node.arguments[0], NameSyntax)
    assert node.arguments[0].name == "a"


def test_subscript_requires_native_name_placement() -> None:
    # "(a)[1]" is not native ArrayRef: the bracket is a top-level leftover.
    with pytest.raises(ExpressionParseError) as error:
        _expr("(a)[1]")
    assert error.value.native_cause == "unexpected token at end of expression"


def test_subscript_unclosed_reports_native_causes() -> None:
    with pytest.raises(ExpressionParseError) as error:
        _expr("arr[1")
    assert error.value.native_cause == "expected ']'"

    with pytest.raises(ExpressionParseError) as error:
        _expr("arr{1")
    assert error.value.native_cause == "expected '}'"

    with pytest.raises(ExpressionParseError) as error:
        _expr("arr[1 2]")
    assert error.value.native_cause == "expected ']'"

    with pytest.raises(ExpressionParseError) as error:
        _expr("arr[")
    assert error.value.native_cause == "expression expected"


def test_call_argument_faults_report_native_causes() -> None:
    with pytest.raises(ExpressionParseError) as error:
        _expr("f(1")
    assert error.value.native_cause == "expected ')'"

    with pytest.raises(ExpressionParseError) as error:
        _expr("f(1,")
    assert error.value.native_cause == "expression expected"

    with pytest.raises(ExpressionParseError) as error:
        _expr("f(a,,b)")
    assert error.value.native_cause == "expression expected"

    with pytest.raises(ExpressionParseError) as error:
        _expr("f(,a)")
    assert error.value.native_cause == "expression expected"

    with pytest.raises(ExpressionParseError) as error:
        _expr("f(a b)")
    assert error.value.native_cause == "expected ')'"


def test_call_prefix_postfix_chaining_is_leftover_like_native() -> None:
    with pytest.raises(ExpressionParseError) as error:
        _expr("f(a)(b)")
    assert error.value.native_cause == "unexpected token at end of expression"

    with pytest.raises(ExpressionParseError) as error:
        _expr("arr[1][2]")
    assert error.value.native_cause == "unexpected token at end of expression"


# --- format tokens -----------------------------------------------------------


def test_put_input_second_bare_argument_becomes_format_literal() -> None:
    for token in ("best12.", "$char.", "numeric5.", "z8.2.", "comma8.", "8.", "1.2."):
        document, node = _expr(f"put(x, {token})")
        assert isinstance(node, CallSyntax)
        assert node.name == "put"
        assert len(node.arguments) == 2
        second = node.arguments[1]
        assert isinstance(second, LiteralSyntax)
        assert second.kind == "format"
        assert second.lexeme == token
        assert _slice(document, second.span) == token

        document, node = _expr(f"INPUT(x, {token})")
        second = node.arguments[1]  # type: ignore[union-attr]
        assert second.kind == "format"


def test_format_classification_is_limited_to_put_input_second_argument() -> None:
    document, node = _expr("f(x, best12.)")
    assert isinstance(node, CallSyntax)
    second = node.arguments[1]
    assert isinstance(second, NameSyntax)
    assert second.name == "best12."

    document, node = _expr("put(best12.)")
    assert isinstance(node, CallSyntax)
    assert len(node.arguments) == 1
    assert isinstance(node.arguments[0], NameSyntax)

    document, node = _expr("put(x, best12., y)")
    third = node.arguments[2]
    assert isinstance(third, NameSyntax)
    assert third.name == "y"


def test_format_token_rejects_quoted_and_non_bare_lexemes() -> None:
    document, node = _expr("put(x, 'z8.2')")
    second = node.arguments[1]
    assert isinstance(second, LiteralSyntax)
    assert second.kind == "string"
    assert second.lexeme == "'z8.2'"

    document, node = _expr("put(x, 8.2)")
    second = node.arguments[1]
    assert second.kind == "numeric"
    assert second.lexeme == "8.2"

    document, node = _expr("put(x, work.a)")
    second = node.arguments[1]
    assert isinstance(second, NameSyntax)
    assert second.name == "work.a"


def test_digit_trailing_dot_ambiguity_outside_put_input() -> None:
    # "8." directly before ')' is a native text token (the format shape),
    # while the same lexeme away from a closer stays numeric.
    document, node = _expr("f(8.)")
    assert isinstance(node, CallSyntax)
    argument = node.arguments[0]
    assert isinstance(argument, LiteralSyntax)
    assert argument.kind == "string"
    assert argument.lexeme == "8."

    document, node = _expr("f(8.2)")
    argument = node.arguments[0]
    assert isinstance(argument, LiteralSyntax)
    assert argument.kind == "numeric"
    assert argument.lexeme == "8.2"


# --- concat ------------------------------------------------------------------


def test_concat_is_a_structural_two_part_node() -> None:
    document, node = _expr("a || b")
    assert isinstance(node, ConcatSyntax)
    assert len(node.parts) == 2
    assert isinstance(node.parts[0], NameSyntax)
    assert isinstance(node.parts[1], NameSyntax)
    assert _slice(document, node.span) == "a || b"


def test_concat_chain_flattens_and_is_left_associative() -> None:
    document, node = _expr("a || b || c")
    assert isinstance(node, ConcatSyntax)
    assert len(node.parts) == 3
    assert all(isinstance(part, NameSyntax) for part in node.parts)


def test_concat_binds_looser_than_every_operator() -> None:
    document, node = _expr("a and b || c + d")
    assert isinstance(node, ConcatSyntax)
    left, right = node.parts
    assert isinstance(left, BinarySyntax)
    assert left.operator == "and"
    assert isinstance(right, BinarySyntax)
    assert right.operator == "+"

    document, node = _expr("a ** b || c")
    assert isinstance(node, ConcatSyntax)
    left, right = node.parts
    assert isinstance(left, BinarySyntax)
    assert left.operator == "**"


def test_concat_inside_call_arguments() -> None:
    document, node = _expr("f(a, b || c)")
    assert isinstance(node, CallSyntax)
    second = node.arguments[1]
    assert isinstance(second, ConcatSyntax)
    assert len(second.parts) == 2

    document, node = _expr("f(a || b)")
    assert isinstance(node, CallSyntax)
    concat = node.arguments[0]
    assert isinstance(concat, ConcatSyntax)
    assert len(concat.parts) == 2


def test_quoted_concat_and_operators_are_untouched() -> None:
    document, node = _expr("'a || b'")
    assert isinstance(node, LiteralSyntax)
    assert node.kind == "string"
    assert node.lexeme == "'a || b'"

    document, node = _expr("'a ** 2' || x")
    assert isinstance(node, ConcatSyntax)
    first = node.parts[0]
    assert isinstance(first, LiteralSyntax)
    assert first.kind == "string"
    assert first.lexeme == "'a ** 2'"


def test_concat_missing_operands_are_refused_at_top_level() -> None:
    # No empty-string operand supplementation: "a||", "|| a", and
    # "a || || b" stay invalid instead of gaining a synthetic '' part.
    for source in ("a||", "|| a", "a || || b"):
        with pytest.raises(ExpressionParseError) as error:
            _expr(source)
        assert error.value.native_cause == "expression expected", source


def test_single_pipe_reports_native_cause() -> None:
    with pytest.raises(ExpressionParseError) as error:
        _expr("a | b")
    assert error.value.native_cause == "unexpected token: |"


def test_concat_span_and_part_spans_are_document_anchored() -> None:
    document = SourceDocument("x = 'a || b' || y", source_id="doc")
    node = parse_expression(document, document.span(4, len(document.text)))
    assert isinstance(node, ConcatSyntax)
    first, second = node.parts
    assert document.text[first.span.start : first.span.end] == "'a || b'"
    assert document.text[second.span.start : second.span.end] == "y"
    assert document.text[node.span.start : node.span.end] == "'a || b' || y"
