"""Focused tests for the legacy projection of the unified parser.

Classification faults retain the first outer diagnostic; per-statement
faults retain all collected diagnostics."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from limulus.block_splitter import DataStepBlockSplitter
from limulus.models import DiagnosticSpan
from limulus.parser import LarkParserFrontend, ParserService
from limulus.syntax import parse_program
from limulus.syntax.legacy import (
    ArrayStatementSpec,
    DataStepAst,
    DatasetReferenceOptionSpec,
    DoStatementSpec,
    IfStatementSpec,
    ParseResult,
    ParsedStatement,
    SetStatementOptionSpec,
    ast_to_dict,
    statement_to_dict,
)
from limulus.syntax.models import (
    DeferredExpressionError,
    DeferredStatementError,
    IfStatementSyntax,
    ParseDiagnosticDisplay,
    SourceDocument,
    SourceSpan,
)
from limulus.syntax.program import LarkProgramFrontend, ParserInitializationError

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
                        f"module {fullname!r} is denied by the compat projection test hook"
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
    assert result.returncode == 0, f"isolated run failed:\n{result.stdout}\n{result.stderr}"


def _kinds(result: ParseResult) -> list[str]:
    return [statement.kind for statement in result.ast.statements]


def _if_specs(result: ParseResult) -> list[IfStatementSpec]:
    return [statement.if_spec for statement in result.ast.statements if statement.if_spec]


def _do_specs(result: ParseResult) -> list[DoStatementSpec]:
    return [statement.do_spec for statement in result.ast.statements if statement.do_spec]


# --- facade identity, lazy initialization, and skip unification -------------------


def test_facade_alias_and_lazy_constructor() -> None:
    assert ParserService is LarkParserFrontend
    service = ParserService()
    # The lazy facade builds no grammar at construction time; the pure
    # program frontend owns every later parse.
    assert getattr(service, "_lark_parser", None) is None
    assert service._program_frontend is not None


def test_facade_parse_projects_dto_shapes() -> None:
    result = ParserService().parse("data out; set in; run;")

    assert result.has_errors is False
    assert _kinds(result) == ["DATA", "SET", "RUN"]
    data_statement, set_statement, run_statement = result.ast.statements
    assert data_statement.output_refs[0].name == "out"
    assert set_statement.dataset_refs[0].name == "in"
    assert isinstance(run_statement.span, DiagnosticSpan)
    assert (run_statement.span.start, run_statement.span.end) == (18, 21)


def test_facade_skip_unification_excludes_identified_regions() -> None:
    service = ParserService()

    # Skip only: the old direct-parse refusals (%put / PROC) and the %let
    # ASSIGN classification become an empty AST with no diagnostics.
    put_only = service.parse("%put skip;")
    assert _kinds(put_only) == []
    assert put_only.diagnostics == ()

    let_only = service.parse("%let x = 1;")
    assert _kinds(let_only) == []
    assert let_only.diagnostics == ()

    proc_only = service.parse("proc sort data=a; by id; run;")
    assert _kinds(proc_only) == []
    assert proc_only.diagnostics == ()

    macro_block = service.parse("%macro m(); data hidden; run; %mend m;")
    assert _kinds(macro_block) == []
    assert macro_block.diagnostics == ()

    # Mixed skip + DATA: only the DATA block statements project.
    mixed = service.parse("%let x=1; data out; set in; run;")
    assert _kinds(mixed) == ["DATA", "SET", "RUN"]

    data_before_let = service.parse("data out; set in; %let x=1; run;")
    assert _kinds(data_before_let) == ["DATA", "SET", "RUN"]

    trailing_proc = service.parse("data out; set in; run; proc print; run; data out2; run;")
    assert _kinds(trailing_proc) == ["DATA", "SET", "RUN", "DATA", "RUN"]


def test_facade_unknown_macro_and_ineligible_inputs_keep_legacy_truth() -> None:
    service = ParserService()

    # An unknown macro-like statement was never an identified skip: the
    # legacy ASSIGN classification truth survives the projection.
    unknown = service.parse("%foo a=1;")
    assert _kinds(unknown) == ["ASSIGN"]
    assert unknown.ast.statements[0].text == "%foo a=1"

    # Open-quote fragment requests establish no skip mask: the standard
    # %let keeps its raw legacy ASSIGN visibility and the unclosed
    # assignment stays a lazily failing ASSIGN (no parse stop).
    fragment = service.parse("%let q = 1; data out; x = 'abc;")
    assert _kinds(fragment) == ["ASSIGN", "DATA", "ASSIGN"]
    assert fragment.diagnostics == ()

    # A whole-parse stop (bare top-level DO) still stops with the legacy
    # diagnostic family.
    stopped = service.parse("data out; do; x=1; end; run;")
    assert stopped.has_errors is True
    diagnostic = stopped.diagnostics[0]
    assert diagnostic.code == "PARSE_UNSUPPORTED_STATEMENT"
    assert diagnostic.location == "statement:2"
    assert diagnostic.message == "Unsupported statement syntax near line 1, column 13."
    assert diagnostic.span is not None
    assert (diagnostic.span.start, diagnostic.span.end) == (12, 13)
    assert diagnostic.source_text == "data out; do; x=1; end; run;"


def test_facade_incomplete_where_precheck_keeps_legacy_family() -> None:
    result = ParserService().parse("data out(where=(x > ); set in; run;")

    assert result.has_errors is True
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_DATASET_WHERE_UNSUPPORTED"
    assert diagnostic.stage == "parse"
    assert diagnostic.message == (
        "dataset option WHERE= is invalid: incomplete WHERE= parenthesized expression"
    )
    assert diagnostic.location == "statement:1"


# --- IF display projection (stored condition_display_edits) -----------------------


def test_facade_if_condition_roundtrip_preserves_raw_layout() -> None:
    service = ParserService()

    spec = _if_specs(service.parse("data out; if (id=1) then foo; run;"))[0]
    assert spec.condition == "(id==1)"
    assert spec.then_action == "foo"

    spec = _if_specs(service.parse("data out; if [x=1] then foo; run;"))[0]
    assert spec.condition == "[x==1]"
    assert spec.then_action == "foo"

    spec = _if_specs(service.parse("data out; if x=1 and then foo; run;"))[0]
    assert spec.condition == "x==1 and"
    assert spec.then_action == "foo"

    spec = _if_specs(service.parse("data out; if x=1and then foo; run;"))[0]
    assert spec.condition == "x==1and"


def test_facade_if_quotes_parentheses_and_mod_stay_raw() -> None:
    service = ParserService()

    spec = _if_specs(service.parse("data out; if x = 'don''t' then foo; run;"))[0]
    assert spec.condition == "x == 'don''t'"

    spec = _if_specs(
        service.parse("data out; if x = 1 and (y = 2 or z = 3) then foo; run;")
    )[0]
    assert spec.condition == "x == 1 and (y == 2 or z == 3)"

    spec = _if_specs(
        service.parse("data out; if x mod 2 = 0 then foo; run;")
    )[0]
    # MOD is a typed general-expression operator, not a display edit: the
    # keyword keeps its raw spelling while the symbol equality normalizes.
    assert spec.condition == "x mod 2 == 0"

    spec = _if_specs(service.parse("data out; if x=1 THEN  THEN foo; run;"))[0]
    assert spec.condition == "x==1"
    assert spec.then_action == "THEN foo"


def test_facade_if_word_operators_keep_the_raw_display() -> None:
    """Only symbol equality aliases and logic words become display edits;
    COMPARE_WORD tokens are not edit anchors so their raw spelling slices
    through unchanged."""

    service = ParserService()

    spec = _if_specs(service.parse("data out; if x eq 1 then foo; run;"))[0]
    assert spec.condition == "x eq 1"

    spec = _if_specs(service.parse("data out; if x gt 1 then foo; run;"))[0]
    assert spec.condition == "x gt 1"

    # The broad symbol cases stay normalized (the kept compat promise).
    spec = _if_specs(service.parse("data out; if x ^= 3 then foo; run;"))[0]
    assert spec.condition == "x != 3"
    spec = _if_specs(service.parse("data out; if x ~= 3 then foo; run;"))[0]
    assert spec.condition == "x != 3"


def test_facade_if_deferred_condition_and_action_cause_remain() -> None:
    service = ParserService()

    result = service.parse("data out; if [x=1] then foo; run;")
    statement = result.ast.statements[1]
    assert statement.kind == "IF"
    spec = statement.if_spec
    assert spec is not None
    assert spec.condition == "[x==1]"
    assert spec.then_action == "foo"

    # The unsupported inline action keeps the raw descriptor cause family.
    unsupported = service.parse("data out; if q then put x; run;")
    typed = parse_program("data out; if q then put x; run;").program.blocks[0].statements[1]
    payload = typed.payload
    assert isinstance(payload, IfStatementSyntax)
    action = payload.then_action
    assert isinstance(action, DeferredStatementError)
    assert action.native_cause == "unsupported inline action: put x"
    unsupported_spec = unsupported.ast.statements[1].if_spec
    assert unsupported_spec is not None
    assert unsupported_spec.then_action == "put x"

    # A split-failure IF keeps the whole-statement fault with no if_spec.
    faulted = service.parse("data out; if x = 1 then; run;")
    assert _kinds(faulted) == ["DATA", "IF", "RUN"]
    assert _if_specs(faulted) == []


def test_facade_if_else_chain_projects_flat_legacy_kinds() -> None:
    result = ParserService().parse(
        "data out; set in; if a then x=1; else if b then y=2; else c=3; run;"
    )

    assert _kinds(result) == ["DATA", "SET", "IF", "ELSE IF", "ELSE", "RUN"]
    assert _if_specs(result) == [
        IfStatementSpec(condition="a", then_action="x=1"),
        IfStatementSpec(condition="b", then_action="y=2"),
    ]


def test_facade_if_then_do_keeps_flat_order_and_flag() -> None:
    result = ParserService().parse("data out; if c then do; x=1; end; run;")

    assert _kinds(result) == ["DATA", "IF", "ASSIGN", "END", "RUN"]
    spec = _if_specs(result)[0]
    assert spec.condition == "c"
    assert spec.then_action == "do"
    assert spec.is_then_do is True


# --- DO / ARRAY / dataset option projection ---------------------------------------


def test_facade_do_spec_recovers_full_raw_boundaries() -> None:
    service = ParserService()

    spec = _do_specs(service.parse("data out; do i = -1 to 5 by 2; x = i; end; run;"))[0]
    assert spec.loop_var == "i"
    assert spec.start_expr == "-1"
    assert spec.end_expr == "5 by 2"

    spec = _do_specs(service.parse("data out; do i = 1+2 to 10; end; run;"))[0]
    assert spec.start_expr == "1+2"
    assert spec.end_expr == "10"

    spec = _do_specs(service.parse("data out; do i = (1+2) to f(a,1); end; run;"))[0]
    assert spec.start_expr == "(1+2)"
    assert spec.end_expr == "f(a,1)"

    spec = _do_specs(
        service.parse("data out; do i = 1 to 10 until(x=3); end; run;")
    )[0]
    assert spec.start_expr == "1"
    assert spec.end_expr == "10 until(x=3)"

    spec = _do_specs(
        service.parse("data out; do i = 'a to b' to 'c'; end; run;")
    )[0]
    assert spec.start_expr == "'a to b'"
    assert spec.end_expr == "'c'"

    spec = _do_specs(service.parse("data out; do i = total to 10; end; run;"))[0]
    assert spec.start_expr == "total"
    assert spec.end_expr == "10"

    spec = _do_specs(service.parse("data out; do i = 1to5; end; run;"))[0]
    assert spec.start_expr == "1"
    assert spec.end_expr == "5"

    # A shape the legacy facade produced no spec for keeps do_spec None.
    result = service.parse("data out; do i = 1 to 5 by 2 to 9; end; run;")
    assert _do_specs(result) == []


def test_facade_do_bare_payload_keeps_no_spec() -> None:
    service = ParserService()
    typed = parse_program("data out; do; x=1; end; run;")
    assert typed.program is None  # bare top-level DO stopped the whole parse

    result = service.parse("data out; set in; if c then do; y=1; end; run;")
    assert _kinds(result) == ["DATA", "SET", "IF", "ASSIGN", "END", "RUN"]
    assert _do_specs(result) == []


def test_facade_array_and_dataset_option_projection() -> None:
    service = ParserService()

    result = service.parse("data out; array a[3] x1 x2 x3; run;")
    statements = result.ast.statements
    assert isinstance(statements[1].array_spec, ArrayStatementSpec)
    assert statements[1].array_spec == ArrayStatementSpec(
        array_name="a",
        variables=("x1", "x2", "x3"),
        declared_size=3,
        wildcard_size=False,
        character_array=False,
    )

    # Paren sizes were never accepted by the legacy facade: no spec.
    result = service.parse("data out; array a(3) x; run;")
    assert result.ast.statements[1].array_spec is None


def test_facade_data_dotted_names_and_nested_where_options() -> None:
    service = ParserService()

    result = service.parse(
        "data work.out(keep=id rename=(id=key) where=(id > 1));"
        " set work.Input(firstobs=2 obs=5 in=f); run;"
    )
    data_statement, set_statement = result.ast.statements[:2]
    assert data_statement.text == "data work.out(keep=id rename=(id=key) where=(id > 1))"
    assert data_statement.dataset_refs[0].name == "work.out"
    assert data_statement.output_refs == data_statement.dataset_refs
    options = data_statement.dataset_refs[0].options
    assert isinstance(options, DatasetReferenceOptionSpec)
    assert options.keep_vars == ("id",)
    assert options.rename_map == {"id": "key"}
    assert options.where_expr is not None

    assert set_statement.dataset_refs[0].name == "work.Input"
    set_options = set_statement.dataset_refs[0].options
    assert set_options.firstobs == 2
    assert set_options.obs == 5
    assert set_options.in_var == "f"

    # SET statement options project into the legacy DTO (INDSNAME on a
    # MERGE stays the legacy scope refusal).
    merged = service.parse("data out; set a(in=g) b indsname=src end=last; by id; run;")
    set_statement = merged.ast.statements[1]
    assert [ref.name for ref in set_statement.dataset_refs] == ["a", "b"]
    assert set_statement.statement_options == SetStatementOptionSpec(
        indsname_var="src", end_var="last"
    )
    refused = service.parse("data out; merge a indsname=src; by id; run;")
    assert refused.has_errors is True
    assert refused.diagnostics[0].code == "PARSE_SET_OPTION_SCOPE_ERROR"
    assert refused.diagnostics[0].message == (
        "INDSNAME= is only allowed as a SET statement option."
    )


# --- serializer goldens and raw text/positions ------------------------------------


def test_serializer_golden_keeps_legacy_shape() -> None:
    result = ParserService().parse("data out; x = 1; if x = 1 then y = 2; run;")

    serialized = ast_to_dict(result.ast)
    assert [item["kind"] for item in serialized["statements"]] == [
        "DATA", "ASSIGN", "IF", "RUN",
    ]
    if_statement = next(item for item in serialized["statements"] if item["kind"] == "IF")
    assert if_statement["text"] == "if x = 1 then y = 2"
    assert if_statement["span"]["source_id"] == "<dsl>"
    assert if_statement["if_spec"] == {
        "condition": "x == 1",
        "then_action": "y = 2",
        "is_subset": False,
        "is_then_do": False,
    }
    assert if_statement["dataset_refs"] == []
    assert if_statement["output_refs"] == []

    # OUTPUT keeps the old observational shape (no output_refs).
    output_result = ParserService().parse("data out; set in; output out; run;")
    output_statement = next(
        item for item in ast_to_dict(output_result.ast)["statements"] if item["kind"] == "OUTPUT"
    )
    assert output_statement["text"] == "output out"
    assert output_statement["output_refs"] == []


def test_projection_uses_raw_text_and_positions_with_unicode_and_tabs() -> None:
    service = ParserService()

    unicode_result = service.parse("data out; set in; value = '日本語メモ'; run;")
    assert unicode_result.has_errors is False
    value_statement = unicode_result.ast.statements[2]
    assert value_statement.text == "value = '日本語メモ'"

    tab_result = service.parse("data\tout;\tset in;\trun;")
    assert tab_result.has_errors is False
    assert _kinds(tab_result) == ["DATA", "SET", "RUN"]
    data_statement = tab_result.ast.statements[0]
    # The legacy text spans keep the raw tab layout (display contract).
    assert data_statement.text == "data\tout"
    assert data_statement.span is not None
    assert (data_statement.span.start, data_statement.span.end) == (0, 8)
    assert data_statement.span.line == 1


def test_projection_label_rename_maps_keep_legacy_dict_truth() -> None:
    result = ParserService().parse(
        "data out; label x = 'the x' y = 'the y' x = 'last wins';"
        " rename a = b a = c; run;"
    )
    label_statement, rename_statement = result.ast.statements[1:3]
    assert label_statement.label_map == {"x": "last wins", "y": "the y"}
    assert rename_statement.rename_map == {"a": "c"}


# --- stop diagnostics, multi-block projection, and block displays -----------------


def test_facade_multi_block_flattens_and_stops_with_legacy_diagnostics() -> None:
    service = ParserService()

    result = service.parse("data a; x=1; run; data b; y=2; run;")
    assert _kinds(result) == ["DATA", "ASSIGN", "RUN", "DATA", "ASSIGN", "RUN"]

    stopped = service.parse("data a; ???; run; data b; ???; run;")
    assert stopped.has_errors is True
    # The legacy view keeps the old single-stop shape: the old facade's
    # outer statement pass halted at the FIRST failing statement, so one
    # near-line/column diagnostic is projected even when the typed result
    # collected both block-level failures.
    assert len(stopped.diagnostics) == 1
    first = stopped.diagnostics[0]
    assert first.location == "statement:2"
    assert first.message == "Unsupported statement syntax near line 1, column 9."
    assert first.span is not None
    assert (first.span.start, first.span.end) == (8, 11)


def test_block_splitter_projects_typed_block_displays() -> None:
    splitter = DataStepBlockSplitter()

    assert splitter.split("data out; set in; run;") == ("data out; set in; run;",)
    assert splitter.split("data out; set in;") == ("data out; set in;",)
    assert splitter.split("data a; x=1; run; data b; y=2; run;") == (
        "data a; x=1; run;",
        "data b; y=2; run;",
    )
    # Missing-RUN blocks keep the tail comments up to the next DATA start.
    assert splitter.split("data out; set in; /* tail */ data b; run;") == (
        "data out; set in; /* tail */",
        "data b; run;",
    )
    assert splitter.split("data out; set in; * tail; data b; run;") == (
        "data out; set in; * tail;",
        "data b; run;",
    )
    assert splitter.split("data out; set in;\n/* multi\nline */\ndata b; run;") == (
        "data out; set in;\n/* multi\nline */",
        "data b; run;",
    )
    # Trailing comments after the last statement of the text stay outside.
    assert splitter.split("data out; set in; /* tail */") == ("data out; set in;",)

    # Skip regions are excluded from the block projection (unified skip).
    assert splitter.split("%let x=1; data out; set in; run;") == (
        "data out; set in; run;",
    )


def test_block_splitter_keeps_block_boundaries_on_stop_and_refusals() -> None:
    splitter = DataStepBlockSplitter()

    # Classification stops ("???") keep the region-level block boundaries
    # without any fake parsed program or second typed parse.
    assert splitter.split("data bad;\n  ???;\nrun;\ndata good;\n  set inp;\nrun;\n") == (
        "data bad;\n  ???;\nrun;",
        "data good;\n  set inp;\nrun;",
    )
    assert splitter.split("data a; ???; run; data b; ???; run;") == (
        "data a; ???; run;",
        "data b; ???; run;",
    )

    # Region-refusing inputs keep the old raw pseudo-block fallback.
    assert splitter.split("* only comment;") == ("* only comment;",)
    assert splitter.split("/* block only */") == ("/* block only */",)
    # Skip-only inputs are no blocks under the unified skip contract (the
    # legacy pipeline masked them before splitting, so the executed truth
    # was also no blocks).
    assert splitter.split("%put x;") == ()
    assert splitter.split("%let x = 1;") == ()


# --- compat pure helpers (advanced direct projection surface) ----------------------


def test_compat_pure_helpers_project_statement_block_program() -> None:
    from limulus.syntax.compat import (
        project_block,
        project_block_displays,
        project_program,
        project_result,
        project_statement,
    )

    program_text = "data out(keep=id); set in; if x = 1 then y = 2; do i = 1 to 2; z = i; end; run;"
    typed = parse_program(program_text)
    assert typed.program is not None
    source = typed.program.source

    # Statement-level: the raw slice is the legacy segment text.
    if_statement = typed.program.blocks[0].statements[2]
    projected_statement = project_statement(if_statement, source)
    assert isinstance(projected_statement, ParsedStatement)
    assert projected_statement.text == "if x = 1 then y = 2"
    assert projected_statement.if_spec is not None
    assert projected_statement.if_spec.condition == "x == 1"
    assert projected_statement.span is not None
    assert projected_statement.span.source_id == "<dsl>"

    # Block-level: flat statement order inside one block.
    projected_block = project_block(typed.program.blocks[0], source)
    assert [s.kind for s in projected_block] == [
        "DATA", "SET", "IF", "DO", "ASSIGN", "END", "RUN",
    ]

    # Program-level flattening and the result-level projection agree.
    ast = project_program(typed.program)
    assert ast.statements == tuple(projected_block)
    result_projection = project_result(typed)
    assert result_projection.ast == ast
    assert result_projection.has_errors is False

    # Skip regions never reach any projection level.
    mixed = parse_program("%let x=1; data out; set in; run;")
    assert mixed.program is not None
    assert [s.kind for s in project_program(mixed.program).statements] == [
        "DATA", "SET", "RUN",
    ]
    assert project_block_displays(mixed.program) == ("data out; set in; run;",)


# --- mandatory initialization failure and no-re-parse guards -----------------------


def test_facade_lazily_reports_mandatory_initialization_failure(monkeypatch) -> None:
    def broken_frontend(*args: object, **kwargs: object) -> None:
        raise ParserInitializationError("broken grammar")

    service = ParserService()
    monkeypatch.setattr(
        service._program_frontend, "parse_program", broken_frontend, raising=False
    )

    result = service.parse("data out; set in; run;")
    assert result.has_errors is True
    assert result.ast.statements == ()
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_BACKEND_CAPABILITY_MISSING"
    assert diagnostic.severity == "error"
    assert diagnostic.location == ""
    assert diagnostic.message == "lark parser is not available; parsing cannot proceed."


def test_projection_never_reparses_after_the_result_exists(monkeypatch) -> None:
    from lark import Lark
    from lark.lexer import Lexer

    program_text = "data out; if x = 1 then y = 2; do i = 1 to 2; z = i; end; run;"
    typed = parse_program(program_text)

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("the projection must never parse again")

    monkeypatch.setattr(Lark, "parse", explode)
    monkeypatch.setattr(Lexer, "lex", explode, raising=False)

    projected = __import__("limulus.syntax.compat", fromlist=["project_result"]).project_result(typed)

    assert _kinds(projected) == ["DATA", "IF", "DO", "ASSIGN", "END", "RUN"]
    assert _if_specs(projected)[0].condition == "x == 1"
    assert _do_specs(projected)[0].start_expr == "1"

    # The block display projection is equally parser-free.
    displays = __import__(
        "limulus.syntax.compat", fromlist=["project_block_displays"]
    ).project_block_displays(typed.program)
    assert displays == ("data out; if x = 1 then y = 2; do i = 1 to 2; z = i; end; run;",)


# --- cold imports with data libraries / native denied ------------------------------


def test_compat_module_imports_without_data_libraries_or_native() -> None:
    _run_isolated(
        """
        import sys

        from limulus.syntax.compat import (
            block_display_slices,
            project_block,
            project_block_displays,
            project_program,
            project_result,
            project_statement,
        )
        from limulus.syntax import parse_program

        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules, denied

        result = parse_program("data out; if x = 1 then y = 2; run;")
        projected = project_result(result)
        assert [s.kind for s in projected.ast.statements] == ["DATA", "IF", "RUN"]
        assert projected.ast.statements[1].if_spec.condition == "x == 1"
        assert project_block_displays(result.program) == (
            "data out; if x = 1 then y = 2; run;",
        )
        """
    )


def test_facade_real_parse_runs_without_data_libraries_or_native() -> None:
    _run_isolated(
        """
        import sys

        import limulus.parser as parser_module

        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules, denied

        service = parser_module.ParserService()
        result = service.parse(
            "data work.out(keep=id); set work.in(where=(x == 1));"
            " if x = 1 then y = 2; do i = 1 to 2; z = i; end; run;"
        )
        assert not result.has_errors
        assert [s.kind for s in result.ast.statements] == [
            "DATA", "SET", "IF", "DO", "ASSIGN", "END", "RUN",
        ]
        assert result.ast.statements[2].if_spec.condition == "x == 1"
        assert result.ast.statements[3].do_spec.start_expr == "1"
        assert "pyarrow" not in sys.modules
        assert "limulus_native" not in sys.modules
        """
    )


# --- legacy display boundary behaviors ---------------------------------------------


def test_facade_if_glued_condition_tokens_recover_stored_initial_span() -> None:
    """The condition display slices the stored initial condition span, so
    glued keyword spellings inside tokens never truncate the display
    (old oracle ``tthen==1``)."""

    service = ParserService()

    spec = _if_specs(service.parse("data out; if x.then=1 then foo; run;"))[0]
    assert spec.condition == "x.then==1"
    assert spec.then_action == "foo"

    spec = _if_specs(service.parse("data out; if tthen=1 then foo; run;"))[0]
    assert spec.condition == "tthen==1"

    spec = _if_specs(service.parse("data out; if total = 1 then foo; run;"))[0]
    assert spec.condition == "total == 1"


def test_facade_do_glued_boundary_tokens_recover_stored_initial_spans() -> None:
    """DO boundary displays slice the stored initial expression spans, so
    glued ``to`` spellings never split the START/END display
    (old oracle ``1+x.to`` / ``3``)."""

    service = ParserService()

    spec = _do_specs(service.parse("data out; do i=1+x.to to 3; end; run;"))[0]
    assert spec.loop_var == "i"
    assert spec.start_expr == "1+x.to"
    assert spec.end_expr == "3"

    spec = _do_specs(service.parse("data out; do i = -1 to 5 by 2; end; run;"))[0]
    assert spec.start_expr == "-1"
    assert spec.end_expr == "5 by 2"


def test_facade_where_fault_display_is_the_failing_option_owner_value() -> None:
    """The initial WHERE option owner unwraps the failing option's raw
    value into ``diagnostic_display``; the legacy diagnostic keeps it
    while the typed diagnostic source_text stays the full raw text
    (old oracle ``a+b``)."""

    service = ParserService()
    legacy = service.parse("data out; set i(where=(a=1)) j(where=(a+b)); run;")
    assert [d.code for d in legacy.diagnostics] == ["PARSE_DATASET_WHERE_UNSUPPORTED"]
    assert legacy.diagnostics[0].source_text == "a+b"

    text = "data out; set i(where=(a=1)) j(where=(a+b)); run;"
    typed = parse_program(text)
    assert typed.program is None
    assert len(typed.diagnostics) == 1
    assert typed.diagnostics[0].source_text == text
    assert typed.projection is not None
    assert len(typed.projection.diagnostic_displays) == len(typed.diagnostics)
    assert typed.projection.diagnostic_displays[0] == ParseDiagnosticDisplay(
        source_text="a+b"
    )

    # A failing FIRST option displays its own value with its raw interior
    # spacing; the later valid option is never re-scanned.
    typed_first = parse_program("data out; set i(where=(a + 1)) j(where=(a=1)); run;")
    assert typed_first.projection is not None
    assert len(typed_first.projection.diagnostic_displays) == len(typed_first.diagnostics)
    assert typed_first.projection.diagnostic_displays[0] == ParseDiagnosticDisplay(
        source_text="a + 1"
    )
    projected_first = ParserService().parse(
        "data out; set i(where=(a + 1)) j(where=(a=1)); run;"
    )
    assert projected_first.diagnostics[0].source_text == "a + 1"

    # The display value is sliced from the RAW document: tabs and Unicode
    # inside the WHERE value keep their raw codepoints (the normalized
    # analysis buffer is never the display source).
    tabs = service.parse("data out; set a(where=(a\t+b)); run;")
    assert tabs.diagnostics[0].source_text == "a\t+b"
    unicode_where = parse_program("data out; set a(where=(日本+1)); run;")
    assert unicode_where.projection is not None
    assert unicode_where.projection.diagnostic_displays[0] == ParseDiagnosticDisplay(
        source_text="日本+1"
    )


def test_facade_per_statement_fault_family_keeps_every_collected_diagnostic() -> None:
    """The old per-statement fault families (dataset option, LABEL, RENAME)
    never halted the old parse; the legacy view keeps every diagnostic in
    statement order, each carrying its own diagnostic display."""

    service = ParserService()

    result = service.parse(
        "data a(where=(a+b)); set in; run; data b(where=(a+b)); set in; run;"
    )
    assert result.has_errors is True
    assert len(result.diagnostics) == 2
    assert [d.location for d in result.diagnostics] == ["statement:1", "statement:4"]
    assert all(d.source_text == "a+b" for d in result.diagnostics)

    mixed = service.parse("data out; set a(where=(a+b)); label x = ; run;")
    assert [(d.code, d.location) for d in mixed.diagnostics] == [
        ("PARSE_DATASET_WHERE_UNSUPPORTED", "statement:2"),
        ("PARSE_UNSUPPORTED_STATEMENT", "statement:3"),
    ]
    assert mixed.diagnostics[0].source_text == "a+b"
    assert (
        mixed.diagnostics[1].source_text
        == "data out; set a(where=(a+b)); label x = ; run;"
    )


def test_facade_multi_block_classification_stop_returns_first_legacy_diagnostic() -> None:
    """The old outer statement pass halted at the first failing
    statement, so the classification-stop family projects one diagnostic
    even when the typed result collected both block failures."""

    service = ParserService()

    stopped = service.parse("data a; ???; run; data b; ???; run;")
    assert stopped.has_errors is True
    assert len(stopped.diagnostics) == 1
    diagnostic = stopped.diagnostics[0]
    assert diagnostic.code == "PARSE_UNSUPPORTED_STATEMENT"
    assert diagnostic.location == "statement:2"
    assert diagnostic.message == "Unsupported statement syntax near line 1, column 9."
    assert diagnostic.span is not None
    assert (diagnostic.span.start, diagnostic.span.end) == (8, 11)

    typed = parse_program("data a; ???; run; data b; ???; run;")
    assert typed.program is None
    assert len(typed.diagnostics) == 2
    assert typed.projection is not None
    assert typed.projection.diagnostic_displays == (None, None)


def test_facade_empty_and_blank_input_keep_legacy_rejection() -> None:
    """Empty/blank input keeps the old ``PARSE_UNSUPPORTED_STATEMENT``
    near-line/column rejection while ``parse_program`` itself stays an
    empty normal success."""

    service = ParserService()

    for text in ("", "   "):
        result = service.parse(text)
        assert result.has_errors is True
        assert result.ast.statements == ()
        diagnostic = result.diagnostics[0]
        assert diagnostic.code == "PARSE_UNSUPPORTED_STATEMENT"
        assert diagnostic.severity == "error"
        assert diagnostic.stage == ""
        assert diagnostic.location == "statement:1"
        assert diagnostic.message == "Unsupported statement syntax near line 1, column 1."
        assert diagnostic.span is not None
        assert (diagnostic.span.start, diagnostic.span.end, diagnostic.span.line, diagnostic.span.column) == (
            0,
            1,
            1,
            1,
        )
        assert diagnostic.span.end_column == 2
        assert diagnostic.source_text == text
        assert [(label.message, label.kind) for label in diagnostic.labels] == [
            ("syntax error", "primary")
        ]

    typed = parse_program("")
    assert typed.program is not None
    assert typed.program.blocks == ()
    assert typed.diagnostics == ()
    assert typed.projection is not None
    assert typed.projection.block_display_spans == ()


def test_skip_only_success_keeps_the_empty_ast_without_diagnostics() -> None:
    """Skip-only success stays the empty AST with no diagnostics
    (the skip-only AST is distinct from the empty/blank
    rejection)."""

    service = ParserService()

    result = service.parse("%put skip;")
    assert result.has_errors is False
    assert result.diagnostics == ()
    assert result.ast.statements == ()


def test_facade_statement_legacy_spans_carry_segment_leading_whitespace() -> None:
    """The legacy view spans slice the initial classification segment
    including its leading whitespace (old oracle
    ASSIGN 9..17) while the typed statement span keeps its region."""

    service = ParserService()

    result = service.parse("data out;   x = 1;  run;")
    data_statement, assign_statement, run_statement = result.ast.statements
    assert data_statement.span is not None
    assert assign_statement.span is not None
    assert run_statement.span is not None
    assert (data_statement.span.start, data_statement.span.end) == (0, 8)
    assert (assign_statement.span.start, assign_statement.span.end) == (9, 17)
    assert assign_statement.span.column == 10
    assert (run_statement.span.start, run_statement.span.end) == (20, 23)
    assert assign_statement.text == "x = 1"

    typed = parse_program("data out;   x = 1;  run;")
    typed_assign = typed.program.blocks[0].statements[1]
    assert typed_assign.span.start == 12
    assert typed_assign.legacy_display_span is not None
    assert (
        typed_assign.legacy_display_span.start,
        typed_assign.legacy_display_span.end,
    ) == (9, 17)


def test_facade_if_adjacent_not_display_edits_shift_without_valueerror() -> None:
    """Half-open adjacency between the ``=`` edit and a glued ``NOT`` edit
    is valid: ``NOT``/``NOT(y)`` display shifts must not leak a
    ``ValueError`` (old oracle ``x==not y`` /
    ``x==not(y)``)."""

    service = ParserService()

    spec = _if_specs(service.parse("data out; if x=NOT y then foo; run;"))[0]
    assert spec.condition == "x==not y"

    spec = _if_specs(service.parse("data out; if x=NOT(y) then foo; run;"))[0]
    assert spec.condition == "x==not(y)"


def test_program_parse_result_projection_facts_cover_success_stop_refusal() -> None:
    """``ProgramProjectionFacts`` travel on every pure ``parse_program``
    return: settled typed blocks on success, the initial region blocks on
    a stop, and the empty tuple when the region pass refused."""

    success = parse_program("data out; if x = 1 then y = 2; do i = 1 to 2; end; run;")
    facts = success.projection
    assert facts is not None
    assert facts.source is success.program.source
    assert [(span.start, span.end, terminated) for span, terminated in facts.block_display_spans] == [
        (0, 55, True)
    ]
    assert facts.skip_regions == ()
    assert facts.diagnostic_displays == ()

    stop = parse_program("data a; ???; run; data b; ???; run;")
    stop_facts = stop.projection
    assert stop_facts is not None
    assert stop_facts.source.text == "data a; ???; run; data b; ???; run;"
    assert [
        (span.start, span.end, terminated) for span, terminated in stop_facts.block_display_spans
    ] == [(0, 17, True), (18, 35, True)]
    assert stop_facts.diagnostic_displays == (None, None)

    refusal = parse_program("* only comment;")
    refusal_facts = refusal.projection
    assert refusal_facts is not None
    assert refusal_facts.block_display_spans == ()
    assert refusal_facts.diagnostic_displays == (None,)


def test_block_splitter_stop_keeps_the_initial_region_pass_single_call(monkeypatch) -> None:
    """A stopping split never re-runs the region pass: the projection
    facts of the SAME initial call carry the stop display blocks."""

    splitter = DataStepBlockSplitter()
    frontend = splitter._program_frontend
    assert frontend is not None
    original = frontend.extract_program_regions
    calls = {"count": 0}

    def counting(source):
        calls["count"] += 1
        return original(source)

    monkeypatch.setattr(frontend, "extract_program_regions", counting)

    blocks = splitter.split("data bad;\n  ???;\nrun;\ndata good;\n  set inp;\nrun;\n")
    assert calls["count"] == 1
    assert blocks == (
        "data bad;\n  ???;\nrun;",
        "data good;\n  set inp;\nrun;",
    )


def test_projection_stopping_result_never_reparses_after_the_result_exists(monkeypatch) -> None:
    """The legacy stopping view projects from the stored facts: parser and
    lexer calls after the initial result must throw (code-audit guard)."""

    from lark import Lark
    from lark.lexer import Lexer

    typed = parse_program("data out; set i(where=(a=1)) j(where=(a+b)); run;")
    assert typed.program is None

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("the projection must never parse again")

    monkeypatch.setattr(Lark, "parse", explode)
    monkeypatch.setattr(Lexer, "lex", explode, raising=False)

    projected = __import__(
        "limulus.syntax.compat", fromlist=["project_result"]
    ).project_result(typed)

    assert [d.code for d in projected.diagnostics] == ["PARSE_DATASET_WHERE_UNSUPPORTED"]
    assert projected.diagnostics[0].source_text == "a+b"
