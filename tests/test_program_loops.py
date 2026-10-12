"""Focused tests for the DO boundary frontend entry."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from limulus.syntax.models import (
    ApplyReferenceSyntax,
    BinarySyntax,
    CallSyntax,
    ConcatSyntax,
    DeferredExpressionError,
    DeferredStatementError,
    DoStatementSyntax,
    LiteralSyntax,
    NameSyntax,
    SourceDocument,
    StatementSyntax,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_DENY_HOOK = textwrap.dedent(
    """
    import sys

    _DENIED = ("pyarrow", "polars", "limulus_native", "limulus.session")


    class _DeniedDependencyGate:
        def find_spec(self, fullname, path=None, target=None):
            for root in _DENIED:
                if fullname == root or fullname.startswith(root + "."):
                    raise ImportError(
                        f"module {fullname!r} is denied by the loops frontend isolation hook"
                    )
            return None


    sys.meta_path.insert(0, _DeniedDependencyGate())
    """
)


def _run_isolated(snippet: str) -> None:
    script = _DENY_HOOK + "\n" + textwrap.dedent(snippet)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"isolated import failed:\n{result.stdout}\n{result.stderr}"


def _program(text: str, document_id: str = "loops"):
    from limulus.syntax.program import LarkProgramFrontend

    frontend = LarkProgramFrontend()
    document = SourceDocument(text, document_id)
    regions = frontend.extract_program_regions(document)
    assert regions is not None, f"region identification failed for {text!r}"
    return document, frontend, regions


def _parse_do_document(text: str, needle: str, document_id: str = "loops") -> tuple[str, StatementSyntax]:
    document, frontend, regions = _program(text, document_id)
    matches = [
        region
        for region in regions.statement_regions
        if needle.lower() in document.text[region.span.start : region.span.end].lower()
    ]
    assert len(matches) == 1, f"needle {needle!r} matched {len(matches)} regions in {text!r}"
    return document.text, frontend.parse_do_statement(document, matches[0])


# --- open-quote DO family ---------------------------------------------------------


def test_open_quote_do_boundary_keeps_lazy_fault_family() -> None:
    from limulus.syntax.models import DeferredStatementError
    from limulus.syntax.program import parse_program

    raw = "data out; do i = 1 to 'abc; end; run;"
    result = parse_program(raw, source_id="<loops>")
    assert result.program is not None
    assert result.diagnostics == ()
    statements = result.program.blocks[0].statements
    assert [statement.kind for statement in statements] == ["DATA", "DO", "END", "RUN"]
    do_statement = statements[1]
    payload = do_statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert payload.native_cause == "unsupported DO statement: do i = 1 to 'abc"


# --- typed boundaries ------------------------------------------------------------


def test_basic_do_boundaries_are_typed_with_exact_spans() -> None:
    text, statement = _parse_do_document("data out; do i = 1 to 3; x = i; end; run;", "do i")
    assert statement.kind == "DO"
    payload = statement.payload
    assert isinstance(payload, DoStatementSyntax)
    assert payload.loop_var == "i"
    start = payload.start
    assert isinstance(start, LiteralSyntax)
    assert start.lexeme == "1"
    assert text[start.span.start : start.span.end] == "1"
    end = payload.end
    assert isinstance(end, LiteralSyntax)
    assert end.lexeme == "3"
    assert text[end.span.start : end.span.end] == "3"
    # The parser adds no END matching, numeric coercion, or loop execution.
    assert statement.span.source_id == "loops"


def test_uppercase_and_dotted_loop_variables_keep_raw_text() -> None:
    text, statement = _parse_do_document("data out; DO I=1 TO N; end; run;", "DO I")
    payload = statement.payload
    assert isinstance(payload, DoStatementSyntax)
    assert payload.loop_var == "I"
    end = payload.end
    assert isinstance(end, NameSyntax)
    assert end.name == "N"
    assert text[end.span.start : end.span.end] == "N"

    _, statement = _parse_do_document("data out; do x.y=1 to 2; end; run;", "x.y")
    payload = statement.payload
    assert isinstance(payload, DoStatementSyntax)
    assert payload.loop_var == "x.y"


def test_bracket_power_concat_boundaries_are_structured() -> None:
    text, statement = _parse_do_document("data out; do i=(1+2)**2 to 3; end; run;", "do i")
    start_payload = statement.payload
    assert isinstance(start_payload, DoStatementSyntax)
    start = start_payload.start
    # Power binds over the addition inside the explicit group; the root
    # node keeps its operator occurrence span (model convention, like the
    # IF conditions), and the boundary slice is exactly typed there.
    assert isinstance(start, BinarySyntax)
    assert start.operator == "**"
    assert isinstance(start.left, BinarySyntax)
    assert start.left.operator == "+"
    assert text[start.left.span.start : start.left.span.end] == "+"

    text, statement = _parse_do_document(
        'data out; do i = 1 || "a" to 3; end; run;', "do i"
    )
    start_payload = statement.payload
    assert isinstance(start_payload, DoStatementSyntax)
    start = start_payload.start
    # Literal || is typable structurally — not a general expression failure.
    assert isinstance(start, ConcatSyntax)
    assert text[start.span.start : start.span.end] == '1 || "a"'
    end = start_payload.end
    assert isinstance(end, LiteralSyntax)
    assert end.lexeme == "3"


def test_call_boundaries_are_structured_without_host_exec() -> None:
    text, statement = _parse_do_document(
        'data out; if c then do; end; do i = apply("f", x) to 3; end; run;', "apply"
    )
    assert isinstance(statement.payload, DoStatementSyntax)
    start = statement.payload.start
    # The host call is pure reference structure (CallSyntax); the parser
    # still resolves nothing and evaluates nothing.
    assert isinstance(start, CallSyntax)
    assert start.name == "apply"
    end = statement.payload.end
    assert isinstance(end, LiteralSyntax)
    assert end.lexeme == "3"
    # Whole-statement inventory is the one original-order lexical scan.
    assert [r.target for r in statement.apply_references] == ["f"]
    for reference in statement.apply_references:
        assert isinstance(reference, ApplyReferenceSyntax)
        assert reference.owner_span == statement.span


def test_by_and_until_tails_stay_invalid_general_deferred() -> None:
    text, statement = _parse_do_document("data out; do i=1 to 3 by 2; end; run;", "by 2")
    payload = statement.payload
    assert isinstance(payload, DoStatementSyntax)
    assert payload.start is not None and payload.end is not None
    end = payload.end
    assert isinstance(end, DeferredExpressionError)
    # The tail still travels with the DO slice for the later reach decision.
    assert end.span.source_id == statement.span.source_id
    assert text[end.span.start : end.span.end] == "3 by 2"

    text, statement = _parse_do_document("data out; do i=1 to n by k*2; end; run;", "by k")
    payload = statement.payload
    assert isinstance(payload, DoStatementSyntax)
    end = payload.end
    assert isinstance(end, DeferredExpressionError)
    assert text[end.span.start : end.span.end] == "n by k*2"

    text, statement = _parse_do_document("data out; do i=1 by 2 to 3; end; run;", "by 2")
    payload = statement.payload
    assert isinstance(payload, DoStatementSyntax)
    start = payload.start
    assert isinstance(start, DeferredExpressionError)
    assert text[start.span.start : start.span.end] == "1 by 2"
    end = payload.end
    assert isinstance(end, LiteralSyntax)
    assert end.lexeme == "3"


# --- unsupported shapes: lazily saved, raw display, whole span -------------------


def test_bare_do_statement_is_lazily_unsupported() -> None:
    text, statement = _parse_do_document("data out; do; x=1; end; run;", "do")
    assert statement.kind == "DO"
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert payload.error_kind == "parse_fault"
    assert "unsupported DO statement: do" in payload.native_cause
    assert payload.display_text == "do"
    # The fault is lazily saved on the DO statement; IF/ELSE THEN DO bare
    # flags are unaffected and no new bare-DO loop form exists.
    assert payload.span == statement.span
    assert text[statement.span.start : statement.span.end] == "do"


def test_do_while_and_glued_equals_are_lazily_unsupported() -> None:
    text, statement = _parse_do_document("data out; do while(a); x=1; end; run;", "while")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert "unsupported DO statement: do while(a)" in payload.native_cause
    assert payload.display_text == "do while(a)"
    assert payload.span == statement.span

    text, statement = _parse_do_document("data out; do = 1; run;", "do =")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert "unsupported DO statement: do = 1" in payload.native_cause

    text, statement = _parse_do_document("data out; do:x=1; run;", "do:x")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert "unsupported DO statement: do:x=1" in payload.native_cause

    text, statement = _parse_do_document("data out; do i=1 to; x=1; end; run;", "i=1 to")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert "unsupported DO statement: do i=1 to" in payload.native_cause


def test_unicode_loop_variable_stays_lazily_unsupported() -> None:
    text, statement = _parse_do_document("data out; do 対象=1 to 3; end; run;", "対象")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert "unsupported DO statement" in payload.native_cause
    assert payload.display_text == "do 対象=1 to 3"
    # Raw Unicode offsets stay valid on the fault span.
    assert text[payload.span.start : payload.span.end].startswith("do 対象")


def test_tab_source_splits_like_the_space_spelling() -> None:
    text = "data out; do\ti=1\tto 3; end; run;"
    _, statement = _parse_do_document(text, "do")
    payload = statement.payload
    assert isinstance(payload, DoStatementSyntax)
    assert payload.loop_var == "i"
    assert payload.start is not None and payload.end is not None
    # Spans keep addressing the raw text with tabs.
    assert text[statement.span.start : statement.span.end] == "do\ti=1\tto 3"
    start = payload.start
    assert isinstance(start, LiteralSyntax)
    assert text[start.span.start : start.span.end] == "1"
    end = payload.end
    assert isinstance(end, LiteralSyntax)
    assert text[end.span.start : end.span.end] == "3"


def test_quoted_end_boundary_is_a_typed_literal() -> None:
    text, statement = _parse_do_document("data out; do i=1 to '3'; end; run;", "'3'")
    payload = statement.payload
    assert isinstance(payload, DoStatementSyntax)
    end = payload.end
    assert isinstance(end, LiteralSyntax)
    assert end.lexeme == "'3'"
    assert text[end.span.start : end.span.end] == "'3'"


# --- restricted entry, regions, inventory, and isolation ---------------------------


def test_word_glued_dox_is_not_this_entry() -> None:
    # The old facade classified "dox=1 to 2" as an assignment; the DO entry
    # does not claim word-glued prefixes.
    document, frontend, regions = _program("data out; dox=1 to 2; run;")
    with pytest.raises(ValueError, match="do"):
        frontend.parse_do_statement(document, regions.statement_regions[1])


def test_apply_inventory_survives_invalid_spec() -> None:
    text, statement = _parse_do_document(
        'data out; apply("pre"); do i=1 to apply("h", x) by 2; end; run;', "h"
    )
    payload = statement.payload
    # The invalid spec (BY tail) still keeps the whole-slice inventory once.
    assert isinstance(payload, DoStatementSyntax)
    assert isinstance(payload.end, DeferredExpressionError)
    assert [r.target for r in statement.apply_references] == ["h"]


def test_region_order_and_multiple_blocks_stay_flat() -> None:
    document, frontend, regions = _program("data a; do i=1 to 2; end; run; data b; do j=3 to 4; end; run;")
    spans = [r.span.start for r in regions.statement_regions]
    assert spans == sorted(spans)
    first = frontend.parse_do_statement(document, regions.statement_regions[1])
    second = frontend.parse_do_statement(document, regions.statement_regions[5])
    assert isinstance(first.payload, DoStatementSyntax)
    assert isinstance(second.payload, DoStatementSyntax)
    assert first.payload.loop_var == "i" and second.payload.loop_var == "j"
    # No END matching: every END region remains its own flat statement.
    assert document.text[regions.statement_regions[2].span.start : regions.statement_regions[2].span.end] == "end"
    assert document.text[regions.statement_regions[6].span.start : regions.statement_regions[6].span.end] == "end"


def test_non_do_region_is_rejected_by_the_limited_entry() -> None:
    document, frontend, regions = _program("data out; set inp; run;")
    with pytest.raises(ValueError, match="do"):
        frontend.parse_do_statement(document, regions.statement_regions[1])


def test_loops_entry_imports_without_data_libraries() -> None:
    _run_isolated(
        """
        from limulus.syntax.models import (
            DoStatementSyntax,
            SourceDocument,
        )
        from limulus.syntax.program import LarkProgramFrontend

        frontend = LarkProgramFrontend()
        document = SourceDocument("data out; do i=1 to 3; end; run;", "isolated")
        regions = frontend.extract_program_regions(document)
        assert regions is not None
        statement = frontend.parse_do_statement(document, regions.statement_regions[1])
        payload = statement.payload
        assert isinstance(payload, DoStatementSyntax)
        assert payload.loop_var == "i"
        """
    )


# --- old public path parity (original contracts only) ------------------------------


def test_legacy_public_parse_of_do_statements_is_untouched() -> None:
    """The legacy facade keeps the original public DO contract."""

    from limulus.parser import LarkParserFrontend

    corpus = (
        "data out; do i = 1 to 3; x = i; end; run;",
        "data out; do i=1 to 3 by 2; end; run;",
        "data out; DO I=1 TO N; end; run;",
        "data out; do x.y=1 to 2; end; run;",
        "data out; do i=1 to 3; end; run;",
    )
    expected = (
        ("i", "1", "3"),
        ("i", "1", "3 by 2"),
        ("I", "1", "N"),
        ("x.y", "1", "2"),
        ("i", "1", "3"),
    )
    frontend = LarkParserFrontend()
    for text, spec in zip(corpus, expected):
        result = frontend.parse(text)
        assert result.has_errors is False, text
        do = next(s for s in result.ast.statements if s.kind == "DO")
        assert do.do_spec is not None
        assert (do.do_spec.loop_var, do.do_spec.start_expr, do.do_spec.end_expr) == spec, text
