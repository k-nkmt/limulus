"""Focused tests for the ARRAY declaration frontend entry."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from limulus.syntax.models import (
    ArrayStatementSyntax,
    AssignStatementSyntax,
    DeferredStatementError,
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
                        f"module {fullname!r} is denied by the arrays frontend isolation hook"
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


def _program(text: str, document_id: str = "arrays"):
    from limulus.syntax.program import LarkProgramFrontend

    frontend = LarkProgramFrontend()
    document = SourceDocument(text, document_id)
    regions = frontend.extract_program_regions(document)
    assert regions is not None, f"region identification failed for {text!r}"
    return document, frontend, regions


def _parse_array_document(text: str, needle: str, document_id: str = "arrays") -> tuple[str, StatementSyntax]:
    document, frontend, regions = _program(text, document_id)
    matches = [
        region
        for region in regions.statement_regions
        if needle.lower() in document.text[region.span.start : region.span.end].lower()
    ]
    assert len(matches) == 1, f"needle {needle!r} matched {len(matches)} regions in {text!r}"
    return document.text, frontend.parse_array_statement(document, matches[0])


# --- open-quote ARRAY family ------------------------------------------------------


def test_open_quote_array_declaration_keeps_lazy_fault_family() -> None:
    from limulus.syntax.models import DeferredStatementError
    from limulus.syntax.program import parse_program

    raw = "data out; array xs[1] 'abc; run;"
    result = parse_program(raw, source_id="<arrays>")
    assert result.program is not None
    assert result.diagnostics == ()
    array_statement = result.program.blocks[0].statements[1]
    assert array_statement.kind == "ARRAY"
    payload = array_statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert payload.native_cause == "invalid ARRAY declaration: array xs[1] 'abc"


# --- accepted legacy corpus -------------------------------------------------------


def test_array_without_size_keeps_typed_variables() -> None:
    text, statement = _parse_array_document("data out; array y r; end; run;", "array y")
    assert statement.kind == "ARRAY"
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.array_name == "y"
    assert payload.variables == ("r",)
    assert payload.declared_size is None
    assert payload.wildcard_size is False
    assert payload.character_array is False
    # The model span is the whole statement span in raw document positions.
    assert text[payload.span.start : payload.span.end] == "array y r"
    assert statement.span == payload.span


def test_dollar_character_marker_and_wildcard_and_number_brackets() -> None:
    text, statement = _parse_array_document("data out; array c$ x y; end; run;", "array c$")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.array_name == "c"
    assert payload.variables == ("x", "y")
    assert payload.character_array is True

    text, statement = _parse_array_document("data out; array x[*] a b; end; run;", "[*]")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.wildcard_size is True
    assert payload.declared_size is None

    text, statement = _parse_array_document("data out; array x[4] a b c d; end; run;", "[4]")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.declared_size == 4
    assert payload.variables == ("a", "b", "c", "d")

    text, statement = _parse_array_document("data out; array x[ 3 ] a b c; end; run;", "[ 3 ]")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.declared_size == 3


def test_case_dotted_names_and_raw_int_size_preserved() -> None:
    text, statement = _parse_array_document("data out; array XY[3] A B C; end; run;", "XY")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.array_name == "XY"
    assert payload.variables == ("A", "B", "C")

    text, statement = _parse_array_document("data out; array p.q[3] a b c; end; run;", "p.q")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.array_name == "p.q"

    text, statement = _parse_array_document("data out; array x[99999999999999999999999] a; end; run;", "99999")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.declared_size == 99999999999999999999999


def test_glued_array_keyword_naming_matches_the_legacy_lex() -> None:
    text, statement = _parse_array_document("data out; arrayy r; run;", "arrayy")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.array_name == "y"
    assert payload.variables == ("r",)

    text, statement = _parse_array_document("data out; arrays5 x; run;", "arrays5")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.array_name == "s5"


# --- legacy-invalid shapes stay whole-statement faults (lazy) ----------------------


def test_invalid_array_shapes_keep_whole_statement_fault() -> None:
    corpus = {
        "data out; array x(3) a b c; end; run;": "array x(3) a b c",
        "data out; array 3 a; run;": "array 3 a",
        "data out; array x[3] 'a' b; end; run;": "array x[3] 'a' b",
        "data out; array x[3] a-b; end; run;": "array x[3] a-b",
        "data out; array x[3]; end; run;": "array x[3]",
        "data out; array; end; run;": "array",
    }
    for text, display in corpus.items():
        _, statement = _parse_array_document(text, "array")
        payload = statement.payload
        assert isinstance(payload, DeferredStatementError), text
        assert payload.error_kind == "parse_fault", text
        assert "invalid ARRAY declaration" in payload.native_cause, text
        # The cause keeps the raw display slice and the whole statement span.
        assert payload.display_text == display, text
        assert payload.span == statement.span, text
        assert text[payload.span.start : payload.span.end] == display, text


def test_unicode_array_name_stays_lazily_invalid_with_raw_offsets() -> None:
    text, statement = _parse_array_document("data out; array 配列[3] a; end; run;", "配列")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert "invalid ARRAY declaration" in payload.native_cause
    assert payload.display_text == "array 配列[3] a"
    assert text[payload.span.start : payload.span.end] == "array 配列[3] a"


def test_tab_positions_stay_raw_and_structured() -> None:
    text = "data out;\tarray x[3]\ta b c; end; run;"
    document, frontend, regions = _program(text, "arrays_tab")
    region = next(
        r
        for r in regions.statement_regions
        if document.text[r.span.start :].lstrip().lower().startswith("array")
    )
    statement = frontend.parse_array_statement(document, region)
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert text[statement.span.start : statement.span.end] == "array x[3]\ta b c"
    assert payload.variables == ("a", "b", "c")
    assert payload.declared_size == 3


# --- parser keeps no size validation and no runtime binding -----------------------


def test_mismatched_size_and_repeated_variables_parse_without_enforcement() -> None:
    text, statement = _parse_array_document("data out; array x[3] a b; end; run;", "array x[3]")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.declared_size == 3
    assert payload.variables == ("a", "b")
    # No declared-size vs len(variables) enforcement and no repeat rejection.
    text, statement = _parse_array_document("data out; array x[3] a b a; end; run;", "a b a")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    assert payload.variables == ("a", "b", "a")


def test_array_expression_and_assignment_touch_points_stay_other_units() -> None:
    text = "data out; array x[3] a b c; x[1] = v; z(2) = w; run;"
    document, frontend, regions = _program(text, "arrays_touch")
    array_statement = frontend.parse_array_statement(document, regions.statement_regions[1])
    assert isinstance(array_statement.payload, ArrayStatementSyntax)
    # The array-element expression/assignment target stays at the shared
    # general-expression structures — bound only by the existing runtime owner.
    assignment = frontend.parse_assignment_statement(document, regions.statement_regions[2])
    assert isinstance(assignment.payload, AssignStatementSyntax)
    assert assignment.kind == "ASSIGN"
    assert not hasattr(array_statement.payload, "binding")
    assert frontend.parse_array_statement(document, regions.statement_regions[1]).payload.variables == (
        "a", "b", "c"
    )


# --- inventory, guarded inputs, and isolation --------------------------------------


def test_apply_inventory_is_whole_statement_once() -> None:
    text, statement = _parse_array_document("data out; array x[3] a b c; end; run;", "array x[3]")
    payload = statement.payload
    assert isinstance(payload, ArrayStatementSyntax)
    # A clean declaration keeps the (empty) whole-slice lexical inventory.
    assert statement.apply_references == ()


def test_invalid_declaration_keeps_whole_statement_apply_inventory() -> None:
    text, statement = _parse_array_document(
        'data out; array x apply("h", c); end; run;', "apply"
    )
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert [r.target for r in statement.apply_references] == ["h"]


def test_span_guard_rejects_foreign_document_spans() -> None:
    from limulus.syntax.models import SourceSpan
    from limulus.syntax.program import LarkProgramFrontend, ProgramStatementRegion

    frontend = LarkProgramFrontend()
    document = SourceDocument("data out; array z; run;", "arrays_guard")
    mismatched = SourceDocument("data out; array z; run;", "other-id")
    region = ProgramStatementRegion(kind="OTHER", span=mismatched.span(10, 12))
    with pytest.raises(ValueError, match="source_id"):
        frontend.parse_array_statement(document, region)


# --- isolation + old facade parity --------------------------------------------------


def test_non_array_region_is_rejected_by_the_limited_entry():
    document, frontend, regions = _program("data out; set inp; run;")
    with pytest.raises(ValueError, match="array"):
        frontend.parse_array_statement(document, regions.statement_regions[1])


def test_arrays_entry_imports_without_data_libraries() -> None:
    _run_isolated(
        """
        from limulus.syntax.models import (
            ArrayStatementSyntax,
            SourceDocument,
        )
        from limulus.syntax.program import LarkProgramFrontend

        frontend = LarkProgramFrontend()
        document = SourceDocument("data out; array x[3] a b c; end; run;", "isolated")
        regions = frontend.extract_program_regions(document)
        assert regions is not None
        statement = frontend.parse_array_statement(document, regions.statement_regions[1])
        payload = statement.payload
        assert isinstance(payload, ArrayStatementSyntax)
        assert payload.declared_size == 3
        """
    )


def test_legacy_public_parse_of_array_statements_is_untouched() -> None:
    """The legacy facade keeps the original public ARRAY contract."""

    from limulus.parser import LarkParserFrontend

    corpus = (
        "data out; array x[*] a b; end; run;",
        "data out; array c$ x y; end; run;",
        "data out; array x[4] a b c d; end; run;",
        "data out; array x(3) a b c; end; run;",
    )
    frontend = LarkParserFrontend()
    for text in corpus:
        result = frontend.parse(text)
        old = next(s for s in result.ast.statements if s.kind == "ARRAY")
        if "(3)" in text:
            # The legacy facade never accepted paren sizes (spec None).
            assert old.array_spec is None, text
        else:
            assert old.array_spec is not None, text
