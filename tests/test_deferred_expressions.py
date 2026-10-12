"""Focused tests for the deferred expression frontend entry."""

from __future__ import annotations

from pathlib import Path

import pytest

from limulus.syntax import (
    ApplyReferenceSyntax,
    BinarySyntax,
    CallSyntax,
    ConcatSyntax,
    DeferredExpressionError,
    SourceDocument,
    SourceSpan,
    SubscriptSyntax,
)
from limulus.syntax.expressions import (
    ExpressionParseError,
    collect_apply_references,
    parse_expression,
    parse_general_expression,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parsed(text: str, source_id: str = "doc"):
    document = SourceDocument(text, source_id=source_id)
    return document, parse_general_expression(document, document.span(0, len(text)))


def _slice(document: SourceDocument, span) -> str:
    return document.text[span.start : span.end]


# --- valid expressions pass through with an empty inventory ------------------


def test_valid_expression_is_returned_undelayed_with_inventory() -> None:
    document, result = _parsed("a + b")
    assert isinstance(result.expression, BinarySyntax)
    assert result.expression.operator == "+"
    assert result.apply_references == ()

    document, result = _parsed("apply('len', a) + b")
    assert not isinstance(result.expression, DeferredExpressionError)
    assert [reference.target for reference in result.apply_references] == ["len"]


def test_valid_concat_is_never_deferred() -> None:
    for source in ("a||b", "a||(b||c)"):
        _, result = _parsed(source)
        assert not isinstance(result.expression, DeferredExpressionError), source
        assert isinstance(result.expression, ConcatSyntax), source

    _, result = _parsed("x[i||j]")
    assert not isinstance(result.expression, DeferredExpressionError)
    assert isinstance(result.expression, SubscriptSyntax)
    assert isinstance(result.expression.index, ConcatSyntax)

    _, result = _parsed("cats(a||b,c)")
    assert not isinstance(result.expression, DeferredExpressionError)
    assert isinstance(result.expression, CallSyntax)
    assert len(result.expression.arguments) == 2
    assert isinstance(result.expression.arguments[0], ConcatSyntax)

    # The native trailing-comma call shape is not a concat.
    _, result = _parsed("f(a,)")
    assert not isinstance(result.expression, DeferredExpressionError)
    assert result.expression.name == "f"


def test_entry_shares_the_source_and_span_contract() -> None:
    document = SourceDocument("a + b", source_id="doc")
    other = SourceDocument("a + b", source_id="other")

    with pytest.raises(TypeError):
        parse_general_expression("a + b", document.span(0, 5))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        parse_general_expression(document, "a + b")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        parse_general_expression(document, other.span(0, 5))
    with pytest.raises(ValueError):
        parse_general_expression(
            document, SourceSpan(start=0, end=6, line=1, column=1, source_id="doc")
        )


# --- whole-expression deferred descriptors ------------------------------------


def test_parse_faults_become_whole_deferred_descriptors() -> None:
    expected_causes = {
        "a +": "expression expected",
        "": "expression expected",
        "   ": "expression expected",
        "a b": "unexpected token at end of expression",
        "(a": "expected ')'",
        "f(a b)": "expected ')'",
        "a + %": "unexpected token: %",
    }
    for source, cause in expected_causes.items():
        document, result = _parsed(source)
        deferred = result.expression
        assert isinstance(deferred, DeferredExpressionError), source
        assert deferred.native_cause == cause, source
        assert deferred.error_kind == "parse_fault", source
        assert deferred.span == document.span(0, len(source)), source
        assert deferred.display_text == source, source
        assert deferred.where_syntax_summary == "", source
        assert deferred.apply_references == (), source


def test_malformed_raw_concat_becomes_deferred_with_native_pipe_cause() -> None:
    # The native tokenizer has no "|" token: malformed raw "||" fails at
    # tokenize time with "unexpected token: |", which rectifies the strict
    # entry's provisional "expression expected" in the deferred descriptor.
    for source in ("||", "a||", "|| b", "a || || b", "a||||b", "f(a||)", "x[i||]"):
        document, result = _parsed(source)
        deferred = result.expression
        assert isinstance(deferred, DeferredExpressionError), source
        assert deferred.native_cause == "unexpected token: |", source
        assert deferred.error_kind == "parse_fault", source
        assert _slice(document, deferred.span) == source, source


def test_single_pipe_becomes_deferred_with_native_pipe_cause() -> None:
    _, result = _parsed("a | b")
    deferred = result.expression
    assert isinstance(deferred, DeferredExpressionError)
    assert deferred.native_cause == "unexpected token: |"


def test_strict_entry_keeps_its_provisional_causes() -> None:
    # The strict ``parse_expression`` protocol is deliberately unchanged;
    # only the deferred frontend entry rectifies malformed "||" to the
    # native cause.
    document = SourceDocument("a||", source_id="doc")
    with pytest.raises(ExpressionParseError) as error:
        parse_expression(document, document.span(0, 3))
    assert error.value.native_cause == "expression expected"


def test_invalid_numeric_literals_are_unsupported_literal_descriptors() -> None:
    expected = {
        "18446744073709551616": "invalid integer literal: 18446744073709551616",
        "1.2.3": "invalid number literal: 1.2.3",
        "a + 18446744073709551616": "invalid integer literal: 18446744073709551616",
        "x[1.2.3]": "invalid number literal: 1.2.3",
    }
    for source, cause in expected.items():
        _, result = _parsed(source)
        deferred = result.expression
        assert isinstance(deferred, DeferredExpressionError), source
        assert deferred.native_cause == cause, source
        assert deferred.error_kind == "unsupported_literal", source


def test_list_and_struct_literals_stay_refused_in_descriptors() -> None:
    # Native primary parsing rejects standalone "[" / "{" with the same
    # "expression expected" cause; they never become executable structure.
    for source in ("[1, 2]", "a + [1, 2]", "{}"):
        _, result = _parsed(source)
        deferred = result.expression
        assert isinstance(deferred, DeferredExpressionError), source
        assert deferred.native_cause == "expression expected", source

    # A comma inside a subscript bracket is a delimiter fault instead.
    _, result = _parsed("x[1, 2]")
    deferred = result.expression
    assert isinstance(deferred, DeferredExpressionError)
    assert deferred.native_cause == "expected ']'"


def test_lexical_faults_win_in_native_tokenizer_order() -> None:
    # The native tokenizer runs before parsing, so the first offending
    # lexeme in source order decides the cause.
    expected = {
        "a||1.2.3": "unexpected token: |",
        "1.2.3 || b": "invalid number literal: 1.2.3",
        "1.2.3 +": "invalid number literal: 1.2.3",
        "f('abc": "unterminated string literal",
        "a b %": "unexpected token: %",
    }
    for source, cause in expected.items():
        _, result = _parsed(source)
        deferred = result.expression
        assert isinstance(deferred, DeferredExpressionError), source
        assert deferred.native_cause == cause, source


# --- statement-WHERE compile summary (limited diagnostic compat) ---------------


def _compile_summary(text: str) -> str:
    """The owner truth: one compile of the raw slice, message string only."""

    try:
        compile(text, "<limulus-condition>", "eval")
    except Exception as error:  # noqa: BLE001 - mirrors the native owner
        message = str(error)
        prefix = "SyntaxError: "
        return message[len(prefix) :] if message.startswith(prefix) else message
    return ""


def test_where_summary_is_stored_for_relevant_causes_only() -> None:
    # The statement-WHERE owner compiles the raw failed expression only for
    # the native expression-parse-error cause classes; the compile object is
    # never stored or executed.
    cases = {
        "a||": "unexpected token: |",
        "a b": "unexpected token at end of expression",
        "": "expression expected",
        "'abc": "unterminated string literal",
    }
    for source, cause in cases.items():
        document = SourceDocument(source, source_id="doc")
        result = parse_general_expression(
            document, document.span(0, len(source)), where_summary=True
        )
        deferred = result.expression
        assert isinstance(deferred, DeferredExpressionError), source
        assert deferred.native_cause == cause, source
        assert deferred.where_syntax_summary == _compile_summary(source), source
        assert deferred.where_syntax_summary != "", source


def test_where_summary_is_empty_for_non_compile_causes_and_successful_compile() -> None:
    # "invalid number/integer literal" is not an expression-parse-error
    # cause class (the owner reports it without compiling), and "a | b"
    # compiles cleanly in Python so the owner keeps the native cause.
    for source in ("a + 1.2.3", "18446744073709551616", "a | b"):
        document = SourceDocument(source, source_id="doc")
        result = parse_general_expression(
            document, document.span(0, len(source)), where_summary=True
        )
        deferred = result.expression
        assert isinstance(deferred, DeferredExpressionError), source
        assert deferred.where_syntax_summary == "", source


def test_where_summary_is_absent_by_default() -> None:
    _, result = _parsed("a||")
    deferred = result.expression
    assert isinstance(deferred, DeferredExpressionError)
    assert deferred.where_syntax_summary == ""


# --- initial apply-reference inventory -----------------------------------------


def test_inventory_corpus_preserves_decoded_targets_order_and_duplicates() -> None:
    # The frontend owns this inventory directly; explicit expected values keep
    # the behavior durable without retaining the removed bridge scanner as an
    # oracle. The resolver owns dedupe later.
    corpus = (
        ("apply('len', a)", ["len"]),
        ('apply("len", a)', ["len"]),
        ("x = apply('len', a) + apply('math.sqrt', b)", ["len", "math.sqrt"]),
        ("apply('a', 1) + apply('a', 2)", ["a", "a"]),
        ("APPLY('LEN', a)", ["LEN"]),
        ("apply ('len', a)", ["len"]),
        ("myapply('a', 1)", []),
        ("applyx('a', 1)", []),
        ('s = \'apply("q", 1)\'', []),
        ("/* apply('c', 1) */ x", []),
        ("apply(", []),
        ("apply(1)", []),
        ("apply(x)", []),
        ("apply('unterminated", []),
        ("apply('', 1)", [""]),
        ("note = apply('it\\'s', a) + b", ["it's"]),
        ("apply('it''s', a)", ["its"]),
    )
    for text, expected_targets in corpus:
        document = SourceDocument(text, source_id="doc")
        references = collect_apply_references(document, document.span(0, len(text)))
        targets = [reference.target for reference in references]
        assert targets == expected_targets, text


def test_inventory_is_collected_even_when_the_expression_fails() -> None:
    document, result = _parsed("apply(\"t\", a) + a||")
    deferred = result.expression
    assert isinstance(deferred, DeferredExpressionError)
    assert deferred.native_cause == "unexpected token: |"
    assert [reference.target for reference in deferred.apply_references] == ["t"]
    assert result.apply_references == deferred.apply_references

    document, result = _parsed("apply('q') + 1.2.3")
    deferred = result.expression
    assert isinstance(deferred, DeferredExpressionError)
    assert deferred.native_cause == "invalid number literal: 1.2.3"
    assert [reference.target for reference in deferred.apply_references] == ["q"]


def test_inventory_keeps_duplicates_in_source_order() -> None:
    _, result = _parsed("apply('a', 1) + apply('a', 2)")
    assert len(result.apply_references) == 2
    assert all(isinstance(reference, ApplyReferenceSyntax) for reference in result.apply_references)
    assert [reference.target for reference in result.apply_references] == ["a", "a"]


def test_apply_reference_spans_are_document_anchored() -> None:
    document = SourceDocument("y = apply('len', a) + b", source_id="doc")
    references = collect_apply_references(document, document.span(0, len(document.text)))
    assert len(references) == 1
    reference = references[0]
    assert reference.target == "len"
    # The call span covers the scanned "apply(..." occurrence through the
    # end of the decoded quoted target (the scanner never requires or
    # records the closing bracket).
    assert _slice(document, reference.call_span) == "apply('len'"
    assert reference.owner_span == document.span(0, len(document.text))
    assert reference.call_span.source_id == "doc"


def test_inventory_excludes_quoted_and_comment_occurrences() -> None:
    _, result = _parsed("s = 'apply(\"q\", 1)'")
    assert result.apply_references == ()

    # "/*" is not expression grammar: the slice fails as a parse fault, and
    # the lexical inventory still excludes the comment region.
    _, result = _parsed("/* apply('c', 1) */ x")
    assert isinstance(result.expression, DeferredExpressionError)
    assert result.expression.native_cause == "expression expected"
    assert result.apply_references == ()


def test_unicode_slices_keep_codepoint_anchoring() -> None:
    document = SourceDocument("x = 'ключ' || apply('цел', y)", source_id="doc")
    result = parse_general_expression(document, document.span(4, len(document.text)))
    assert not isinstance(result.expression, DeferredExpressionError)
    assert isinstance(result.expression, ConcatSyntax)
    assert [reference.target for reference in result.apply_references] == ["цел"]
    reference = result.apply_references[0]
    assert _slice(document, reference.call_span).startswith("apply(")
    assert reference.call_span.source_id == "doc"
    assert document.text[reference.call_span.start : reference.call_span.end] == "apply('цел'"


def test_pure_frontend_imports_no_bridge_execution_or_data_libraries() -> None:
    import subprocess
    import sys
    import textwrap as textwrap_module

    script = textwrap_module.dedent(
        """
        import sys

        import limulus.syntax.expressions as expressions

        from limulus.syntax import SourceDocument

        document = SourceDocument("apply('t', a) + a||", source_id="doc")
        result = expressions.parse_general_expression(
            document, document.span(0, len(document.text)), where_summary=True
        )
        assert isinstance(result.expression, expressions.DeferredExpressionError)
        assert result.expression.native_cause == "unexpected token: |"

        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session",
                       "limulus.backend_integration", "limulus.execution"):
            assert denied not in sys.modules, denied
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"isolated deferred parse failed:\n{proc.stdout}\n{proc.stderr}"
