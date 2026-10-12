"""Focused tests for the public pure ``parse_program`` API."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

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
                        f"module {fullname!r} is denied by the program parser isolation hook"
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


def _parse(text: str, source_id: str = "<parse>"):
    from limulus.syntax.program import parse_program

    return parse_program(text, source_id=source_id)


# --- the pure entry points -----------------------------------------------------


def test_module_function_and_frontend_share_the_owner() -> None:
    from limulus.syntax import ParserInitializationError, LarkProgramFrontend, parse_program
    from limulus.syntax.program import parse_program as module_owner

    assert parse_program is module_owner
    result = parse_program("data out; x=1; run;")
    assert result.program is not None
    assert result.has_errors is False

    frontend = LarkProgramFrontend()
    assert isinstance(frontend.parse_program("data out; run;"), type(result))


def test_lazy_syntax_exports_resolve_without_heavy_imports() -> None:
    _run_isolated(
        """
        import sys

        import limulus.syntax as syntax

        # Lazy: importing the syntax surface never loads Lark (the
        # grammar owner lives behind the module ``__getattr__``).
        assert "lark" not in sys.modules, "limulus.syntax must stay Lark-free"

        assert syntax.parse_program is not None
        assert syntax.LarkProgramFrontend is not None
        assert syntax.ParserInitializationError is not None

        result = syntax.parse_program("data out; x=1; run;", source_id="cold")
        assert result.program is not None
        assert len(result.program.blocks) == 1

        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules, denied
        """
    )


# --- structure, order, and full source spans -----------------------------------


def test_single_block_program_keeps_typed_statements_and_full_source_spans() -> None:
    from limulus.syntax.models import (
        AssignStatementSyntax,
        DataStatementSyntax,
        RunStatementSyntax,
    )

    text = "data out; x=1; run;"
    result = _parse(text)
    program = result.program
    assert program is not None
    assert program.source.text == text
    assert program.source.source_id == "<parse>"
    assert (program.span.start, program.span.end) == (0, len(text))
    assert len(program.blocks) == 1
    block = program.blocks[0]
    assert (block.span.start, block.span.end) == (0, 19)
    kinds = [statement.kind for statement in block.statements]
    assert kinds == ["DATA", "ASSIGN", "RUN"]
    payloads = [statement.payload for statement in block.statements]
    assert isinstance(payloads[0], DataStatementSyntax)
    assert isinstance(payloads[1], AssignStatementSyntax)
    assert isinstance(payloads[2], RunStatementSyntax)
    for statement in block.statements:
        # Spans slice the RAW document exactly.
        raw = text[statement.span.start : statement.span.end]
        assert raw in {"data out", "x=1", "run"}
        assert statement.span.source_id == "<parse>"


def test_multi_block_and_skip_regions_keep_ordered_structure() -> None:
    text = "%let x=1;\n\nproc print; run;\n\ndata a; y=2; run;\n\ndata b; z=3; run;\n"
    result = _parse(text)
    program = result.program
    assert program is not None
    assert [region.kind for region in program.skip_regions] == ["macro", "proc"]
    assert [len(block.statements) for block in program.blocks] == [3, 3]
    kinds = [statement.kind for block in program.blocks for statement in block.statements]
    assert kinds == ["DATA", "ASSIGN", "RUN", "DATA", "ASSIGN", "RUN"]
    first = program.blocks[0].statements[0]
    assert text[first.span.start : first.span.end] == "data a"


# --- statement kinds through the existing entries -------------------------------


def test_statement_classification_covers_the_bounded_entry_variants() -> None:
    text = (
        "data out; length a $4.; attrib b; format c; informat d; call foo;\n"
        "array arr[2] p q; label q='8'; rename p=r;\n"
        "keep a p q; drop b; retain t 0; by ascending g;\n"
        "sum_x + 1; s_total + z; output; delete; stop; end; run;\n"
    )
    result = _parse(text)
    program = result.program
    assert program is not None
    kinds = [statement.kind for block in program.blocks for statement in block.statements]
    assert kinds == [
        "DATA",
        "SKIPPED",
        "SKIPPED",
        "SKIPPED",
        "SKIPPED",
        "SKIPPED",
        "ARRAY",
        "LABEL",
        "RENAME",
        "KEEP",
        "DROP",
        "RETAIN",
        "BY",
        "SUM",
        "SUM",
        "OUTPUT",
        "DELETE",
        "STOP",
        "END",
        "RUN",
    ]


def test_where_statement_keeps_the_general_expression_owner() -> None:
    from limulus.syntax.models import BinarySyntax, WhereStatementSyntax

    text = "data out; set i; where a > 1; run;"
    result = _parse(text)
    program = result.program
    assert program is not None
    where = program.blocks[0].statements[2]
    assert where.kind == "WHERE"
    assert isinstance(where.payload, WhereStatementSyntax)
    condition = where.payload.condition
    assert isinstance(condition, BinarySyntax)
    assert condition.operator == ">"


def test_where_statement_deferred_condition_is_retained_not_stopping() -> None:
    from limulus.syntax.models import DeferredExpressionError, WhereStatementSyntax

    text = "data out; set i; where a > ; run;"
    result = _parse(text)
    program = result.program
    assert program is not None
    where = program.blocks[0].statements[2]
    assert where.kind == "WHERE"
    assert isinstance(where.payload, WhereStatementSyntax)
    condition = where.payload.condition
    assert isinstance(condition, DeferredExpressionError)
    assert condition.where_syntax_summary != ""


# --- star arithmetic and dotted names ---------------------------------------------


def test_parse_program_accepts_star_arithmetic_and_dotted_names() -> None:
    from limulus.syntax.models import DataStatementSyntax

    powered = _parse("data out; set in; x = 2 ** 3; run;")
    assert powered.program is not None
    assert powered.diagnostics == (), powered.diagnostics
    powered_statements = [
        statement for block in powered.program.blocks for statement in block.statements
    ]
    assert [statement.kind for statement in powered_statements] == [
        "DATA", "SET", "ASSIGN", "RUN"
    ]
    assert isinstance(powered_statements[0].payload, DataStatementSyntax)

    dotted = _parse("data work.out; set work.in; run;")
    assert dotted.program is not None
    assert dotted.diagnostics == (), dotted.diagnostics
    dotted_statements = [
        statement for block in dotted.program.blocks for statement in block.statements
    ]
    assert dotted_statements[0].kind == "DATA"
    assert dotted_statements[1].kind == "SET"


def test_parse_program_keeps_case_insensitive_references_program() -> None:
    text = (
        REPO_ROOT
        / "tests"
        / "cases"
        / "parser"
        / "core"
        / "programs"
        / "case_insensitive_keywords.txt"
    ).read_text(encoding="utf-8")
    result = _parse(text, source_id="<dsl>")
    assert result.program is not None
    assert result.diagnostics == (), result.diagnostics
    kinds = [
        statement.kind
        for block in result.program.blocks
        for statement in block.statements
    ]
    assert kinds == ["DATA", "SET", "WHERE", "OUTPUT", "RUN"]


# --- open-quote statements ---------------------------------------------------------


def test_open_quote_statements_keep_the_old_lazy_kind_families() -> None:
    expected = (
        ("x = 'abc;", "ASSIGN", "unterminated string literal"),
        ("x + 'abc;", "SUM", "unterminated string literal"),
        ("where id = 'abc;", "WHERE", "unterminated string literal"),
        ("if 0 then x = 'abc;", "IF", "invalid IF statement: if 0 then x = 'abc"),
        ("do i = 1 to 'abc;", "DO", "unsupported DO statement: do i = 1 to 'abc"),
        ("array xs[1] 'abc;", "ARRAY", "invalid ARRAY declaration: array xs[1] 'abc"),
    )
    for raw, kind, fragment in expected:
        result = _parse(raw)
        assert result.program is not None, (raw, [(d.code, d.message) for d in result.diagnostics])
        statements = [
            statement for block in result.program.blocks for statement in block.statements
        ]
        assert [statement.kind for statement in statements] == [kind], (raw, statements)
        text = result.program.source.text
        assert text[statements[0].span.start : statements[0].span.end] == raw[:-1], raw
        if kind == "ASSIGN":
            from limulus.syntax.models import AssignStatementSyntax

            assert isinstance(statements[0].payload, AssignStatementSyntax)
            assert statements[0].payload.value.error_kind == "parse_fault"
        elif kind in {"IF", "DO", "ARRAY"}:
            from limulus.syntax.models import DeferredStatementError

            assert isinstance(statements[0].payload, DeferredStatementError)
            assert fragment in statements[0].payload.native_cause


def test_open_quote_assignment_inside_block_keeps_one_block() -> None:
    raw = "data a; set in; x = 'abc; run;"
    result = _parse(raw)
    assert result.program is not None
    assert result.diagnostics == ()
    assert len(result.program.blocks) == 1
    assert [statement.kind for statement in result.program.blocks[0].statements] == [
        "DATA", "SET", "ASSIGN", "RUN"
    ]
    assignment = result.program.blocks[0].statements[2]
    # Old facade truth: the classification rule's leading whitespace stays
    # inside the span (the ``[^;]*`` terminal swallows the preceding space).
    assert result.program.source.text[assignment.span.start : assignment.span.end] == " x = 'abc"


def test_open_quote_splits_at_statement_semicolons_like_old_classification() -> None:
    for raw, kinds in (
        ("x = 'bad; y = 'bad;", ["ASSIGN", "ASSIGN"]),
        ("x = 'abc; y = 1;", ["ASSIGN", "ASSIGN"]),
        ("x = 'abc;\ndata a; run;", ["ASSIGN", "DATA", "RUN"]),
        ("x = 'it's;", ["ASSIGN"]),
        ("x = 'it''s; y = 1;", ["ASSIGN", "ASSIGN"]),
    ):
        result = _parse(raw)
        assert result.program is not None, (raw, [(d.code, d.message) for d in result.diagnostics])
        kinds_got = [
            statement.kind
            for block in result.program.blocks
            for statement in block.statements
        ]
        assert kinds_got == kinds, (raw, kinds_got, result.diagnostics)


def test_classification_failure_before_open_quote_still_stops() -> None:
    result = _parse("???; x = 'abc;")
    assert result.program is None
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_UNSUPPORTED_STATEMENT"
    assert diagnostic.message == "Unsupported statement syntax near line 1, column 1."
    assert diagnostic.location == "statement:1"


def test_open_quote_keeps_macro_and_proc_raw_visibility() -> None:
    from limulus.syntax.models import AssignStatementSyntax, DataStatementSyntax

    stopping = _parse("x = 'abc; %put skip; run;")
    assert stopping.program is None
    diagnostic = stopping.diagnostics[0]
    assert diagnostic.code == "PARSE_UNSUPPORTED_STATEMENT"
    assert diagnostic.location == "statement:2"

    accepted = _parse("x = 'abc; %let q = 1; data a; run;")
    assert accepted.program is not None, accepted.diagnostics
    statements = [
        statement for block in accepted.program.blocks for statement in block.statements
    ]
    assert [statement.kind for statement in statements] == [
        "ASSIGN", "ASSIGN", "DATA", "RUN"
    ]
    assert isinstance(statements[1].payload, AssignStatementSyntax)
    assert accepted.program.source.text[statements[1].span.start : statements[1].span.end] == (
        " %let q = 1"
    )
    assert accepted.program.skip_regions == ()
    assert isinstance(statements[2].payload, DataStatementSyntax)


def test_open_quote_text_alone_is_kept_not_stopped() -> None:
    from limulus.syntax.program import parse_program

    result = parse_program("x = 'abc;")
    assert result.program is not None
    assert result.diagnostics == ()


def test_closed_quoted_semicolon_inside_statement_keeps_old_parse_stop() -> None:
    result = _parse("x = \"abc;\";")
    assert result.program is None
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_UNSUPPORTED_STATEMENT"


def test_open_quote_apply_inventory_and_next_statement() -> None:
    raw = "x = 'abc; y = apply('ghost', a);"
    result = _parse(raw)
    assert result.program is not None
    statements = [
        statement for block in result.program.blocks for statement in block.statements
    ]
    assert [statement.kind for statement in statements] == ["ASSIGN", "ASSIGN"]
    # The parse fault lives inside the malformed literal; its span holds no
    # apply inventory, while the following statement's call stays collected
    # by the same scanner (old zero-row pre-resolve truth, no evaluation).
    assert statements[0].apply_references == ()
    assert [ref.target for ref in statements[1].apply_references] == ["ghost"]
    assert statements[1].apply_references[0].owner_span == statements[1].span


# --- lazily faulted statements stay deferred (no promotion) ---------------------


def test_lazy_statement_faults_are_retained_and_not_promoted() -> None:
    from limulus.syntax.models import (
        AssignStatementSyntax,
        DeferredExpressionError,
        DeferredStatementError,
    )

    result = _parse("data out; x=; run;")
    program = result.program
    assert program is not None
    assign = program.blocks[0].statements[1]
    assert assign.kind == "ASSIGN"
    assert isinstance(assign.payload, DeferredStatementError)
    assert assign.payload.native_cause == "invalid assignment statement: x="
    assert result.has_errors is False

    result = _parse("data out; y = unknown_fn(] ; run;")
    program = result.program
    assert program is not None
    assign = program.blocks[0].statements[1]
    assert isinstance(assign.payload, AssignStatementSyntax)
    assert isinstance(assign.payload.value, DeferredExpressionError)


def test_unsupported_do_shape_stays_a_lazy_statement_fault() -> None:
    from limulus.syntax.models import DeferredStatementError

    result = _parse("data out; do = 1; run;")
    program = result.program
    assert program is not None
    do_statement = program.blocks[0].statements[1]
    assert do_statement.kind == "DO"
    assert isinstance(do_statement.payload, DeferredStatementError)
    assert do_statement.payload.native_cause.startswith("unsupported DO statement: ")


def test_if_action_fault_stays_inside_the_program() -> None:
    result = _parse("data out; if 1 then foo; run;")
    program = result.program
    assert program is not None
    if_statement = program.blocks[0].statements[1]
    assert if_statement.kind == "IF"
    assert result.diagnostics == ()


# --- old parse-stage stopping faults (DATA/SET/MERGE/LABEL/RENAME) --------------


def test_incomplete_dataset_where_stops_with_the_where_family_diagnostic() -> None:
    result = _parse("data out; set i(where=(a=); x=1; run;")
    assert result.program is None
    assert result.has_errors is True
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_DATASET_WHERE_UNSUPPORTED"
    assert diagnostic.severity == "error"
    assert diagnostic.stage == "parse"
    assert (
        diagnostic.message
        == "dataset option WHERE= is invalid: incomplete WHERE= parenthesized expression"
    )
    assert diagnostic.location == "statement:2"
    assert diagnostic.source_text == text_of(result)


def text_of(result) -> str:
    return result.diagnostics[0].source_text


def test_dataset_option_fault_promotes_to_the_stop_diagnostic() -> None:
    text = "data out(size=big); x=1; run;"
    result = _parse(text)
    assert result.program is None
    assert result.has_errors is True
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_UNSUPPORTED_STATEMENT"
    assert diagnostic.severity == "error"
    assert diagnostic.message == "Unsupported dataset reference option: size=big"
    assert diagnostic.location == "statement:1"
    assert diagnostic.span is not None
    assert diagnostic.span.start == 0
    assert diagnostic.span.end == 18
    assert diagnostic.source_text == text
    assert diagnostic.labels


def test_label_fault_promotes_to_the_stop_diagnostic() -> None:
    text = "data out; label 'idx%'=1; run;"
    result = _parse(text)
    assert result.program is None
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_UNSUPPORTED_STATEMENT"
    assert diagnostic.message == "Unsupported LABEL statement syntax: label 'idx%'=1"
    assert diagnostic.location == "statement:2"
    assert diagnostic.span is not None
    assert diagnostic.span.start == 10
    assert diagnostic.span.end == 24
    assert diagnostic.source_text == text


def test_rename_fault_promotes_to_the_stop_diagnostic() -> None:
    text = "data out; rename x _a; run;"
    result = _parse(text)
    assert result.program is None
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_RENAME_STATEMENT_INVALID"
    assert diagnostic.message == "Invalid rename mapping: x _a"
    assert diagnostic.location == "statement:2"


# --- whole-text grammar failure gates -------------------------------------------


def test_last_block_parse_error_gates_the_whole_program() -> None:
    text = "data a; x=1; run;\ndata;\n"
    result = _parse(text)
    assert result.program is None
    assert result.has_errors is True
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_UNSUPPORTED_STATEMENT"
    assert diagnostic.message == "Unsupported statement syntax near line 2, column 5."
    assert diagnostic.location == "statement:4"
    assert diagnostic.span is not None
    assert diagnostic.span.line == 2
    assert diagnostic.span.column == 5
    assert diagnostic.source_text == text


def test_comment_only_text_stops_with_the_legacy_near_diagnostic() -> None:
    result = _parse("/* just a comment */")
    assert result.program is None
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_UNSUPPORTED_STATEMENT"
    assert diagnostic.message == "Unsupported statement syntax near line 1, column 1."


def test_empty_and_skip_only_programs_return_a_normal_empty_program() -> None:
    empty = _parse("")
    assert empty.program is not None
    assert empty.program.blocks == ()
    assert empty.diagnostics == ()

    whitespace = _parse("   \n\t\n")
    assert whitespace.program is not None
    assert whitespace.program.blocks == ()
    assert whitespace.diagnostics == ()

    skip_only = _parse("%put X;")
    assert skip_only.program is not None
    assert skip_only.program.blocks == ()
    assert len(skip_only.program.skip_regions) == 1
    assert skip_only.diagnostics == ()


# --- normalization, source identity, and apply inventory ------------------------


def test_tab_normalization_keeps_raw_spans_and_display() -> None:
    text = "data out; x\t= 1; run;"
    result = _parse(text, source_id="<tabs>")
    program = result.program
    assert program is not None
    assign = program.blocks[0].statements[1]
    assert assign.payload.target.name == "x"
    # Statement content is sliced RAW: the tab is untouched inside the span.
    assert "\t" in text[assign.span.start : assign.span.end]
    assert program.source.text == text


def test_custom_source_id_and_transformed_source_contract() -> None:
    result = _parse("data out; x=1; run;", source_id="hooks+1")
    program = result.program
    assert program is not None
    assert program.source.source_id == "hooks+1"
    assert program.span.source_id == "hooks+1"
    for block in program.blocks:
        for statement in block.statements:
            assert statement.span.source_id == "hooks+1"


def test_apply_inventory_is_present_on_the_typed_statements() -> None:
    text = "data out; set i; if x then y = [apply('fn', x)]; run;"
    result = _parse(text)
    program = result.program
    assert program is not None
    block = program.blocks[0]
    assert [statement.kind for statement in block.statements] == [
        "DATA",
        "SET",
        "IF",
        "RUN",
    ]
    for statement in block.statements:
        assert isinstance(statement.apply_references, tuple)
    assert block.statements[0].apply_references == ()
    if_statement = block.statements[2]
    assert if_statement.apply_references
    assert if_statement.apply_references[0].target == "fn"


# --- one upfront region parse and buffer lifetime -------------------------------


def test_parse_program_runs_exactly_one_upfront_region_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from limulus.syntax.models import SourceDocument
    from limulus.syntax.program import LarkProgramFrontend

    frontend = LarkProgramFrontend()
    original = frontend.extract_program_regions
    calls: list[SourceDocument] = []

    def counting(source: SourceDocument):
        calls.append(source)
        return original(source)

    monkeypatch.setattr(frontend, "extract_program_regions", counting)
    result = frontend.parse_program("data out; x=1; run;")
    assert result.program is not None
    assert len(calls) == 1


def test_repeated_calls_rebind_the_request_analysis_buffer() -> None:
    from limulus.syntax.program import LarkProgramFrontend

    frontend = LarkProgramFrontend()
    first = frontend.parse_program("data out; a\t= 1; run;")
    second = frontend.parse_program("data second; total + b; stop; end; run;")
    assert first.program is not None and second.program is not None
    assert [statement.kind for statement in first.program.blocks[0].statements] == [
        "DATA",
        "ASSIGN",
        "RUN",
    ]
    assert [statement.kind for statement in second.program.blocks[0].statements] == [
        "DATA",
        "SUM",
        "STOP",
        "END",
        "RUN",
    ]
    third = frontend.parse_program("data out; rename p=r; keep p; run;")
    assert third.program is not None
    kinds = [statement.kind for statement in third.program.blocks[0].statements]
    assert kinds == ["DATA", "RENAME", "KEEP", "RUN"]


# --- mandatory grammar initialization failures are never hidden -----------------


def test_region_grammar_initialization_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import limulus.syntax.program as program_module

    def boom() -> None:
        raise program_module.ParserInitializationError("program grammar boom")

    monkeypatch.setattr(program_module, "_build_program_parser", boom)
    with pytest.raises(program_module.ParserInitializationError):
        _parse("data out; x=1; run;")


def test_classification_grammar_initialization_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import limulus.syntax.program as program_module

    def boom() -> None:
        raise program_module.ParserInitializationError("classification grammar boom")

    monkeypatch.setattr(program_module, "_build_classification_parser", boom)
    with pytest.raises(program_module.ParserInitializationError):
        _parse("data out; x=1; run;")


# --- legacy classification parity matrix -----------------------------------------
# Old-public truth probes (Session.parse / hook+splitter+parse pipeline) for
# each matrix case are pinned below: root cause = the tree kind/token truth
# of the legacy classification grammar must reach the whole-text typing even
# when the bounded 6.x entries' own caller-misuse prefix gates stay
# unchanged, and the skip mask of the ONE upfront region parse must be
# shared with the incomplete-WHERE precheck and the per-block
# classification.  No new backend / success fallback / alias support.


@pytest.mark.parametrize(
    ("code", "expected_message"),
    [
        ("data out; set i(where=(a+b)); x=1; run;", "dataset option WHERE= is invalid: unsupported WHERE= arithmetic syntax"),
        ("data out; set i(where=(a>b)); x=1; run;", "dataset option WHERE= is invalid: unsupported WHERE= syntax (UnexpectedToken)"),
        ("data out; set i(where=(x.a=1)); x=1; run;", "dataset option WHERE= is invalid: unsupported WHERE= attribute syntax"),
    ],
)
def test_dataset_where_fault_family_prefix_mapping(code, expected_message):
    result = _parse(code)
    assert result.program is None
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_DATASET_WHERE_UNSUPPORTED"
    assert diagnostic.severity == "error"
    assert diagnostic.stage == "parse"
    assert diagnostic.message == expected_message
    assert diagnostic.location in {"statement:1", "statement:2"}
    assert diagnostic.span is not None
    assert diagnostic.span.source_id == "<parse>"


@pytest.mark.parametrize(
    ("code", "kind", "expected_ref_names"),
    [
        ("datafoo out", "DATA", ["foo", "out"]),
        ("dataset out", "DATA", ["set", "out"]),
        ("setfoo", "SET", ["foo"]),
        ("mergefoo", "MERGE", ["foo"]),
    ],
)
def test_classification_glued_dataset_variants_keep_old_refs(code, kind, expected_ref_names):
    result = _parse(code)
    program = result.program
    assert program is not None
    block = program.blocks[0]
    assert len(block.statements) == 1
    statement = block.statements[0]
    assert statement.kind == kind
    payload = statement.payload
    refs = payload.outputs if kind == "DATA" else payload.refs
    assert [ref.name for ref in refs] == expected_ref_names


@pytest.mark.parametrize(
    ("code", "expected_atom"),
    [
        ("data out; set ip; outputfoo; run;", "outputfoo"),
        ("data out; set ip; outputfoo bar; run;", "outputfoo bar"),
    ],
)
def test_classification_glued_output_keeps_the_whole_text_target_atom(code, expected_atom):
    result = _parse(code)
    program = result.program
    assert program is not None
    statements = program.blocks[0].statements
    assert [statement.kind for statement in statements] == ["DATA", "SET", "OUTPUT", "RUN"]
    output = statements[2]
    targets = output.payload.targets
    assert len(targets) == 1
    assert targets[0].name == expected_atom


def test_classification_glued_else_keeps_the_inner_deferred_action() -> None:
    result = _parse("data out; set ip; elsefoo; run;")
    program = result.program
    assert program is not None
    else_statement = program.blocks[0].statements[2]
    assert else_statement.kind == "ELSE"
    action = else_statement.payload.action
    from limulus.syntax.models import DeferredStatementError

    assert isinstance(action, DeferredStatementError)
    assert action.native_cause == "unsupported inline action: elsefoo"
    assert program is not None and result.diagnostics == ()
    raw = "data out; set ip; elsefoo; run;"
    assert raw[action.span.start : action.span.end] == "elsefoo"


def test_classification_glued_else_if_keeps_the_outer_deferred_family() -> None:
    from limulus.syntax.models import DeferredStatementError

    result = _parse("data out; set ip; elseif 1 then output; run;")
    program = result.program
    assert program is not None
    statement = program.blocks[0].statements[2]
    assert statement.kind == "ELSE IF"
    assert isinstance(statement.payload, DeferredStatementError)
    assert statement.payload.native_cause == "invalid IF statement: elseif 1 then output"
    assert result.diagnostics == ()


def test_classification_glued_do_keeps_the_do_fault_family() -> None:
    from limulus.syntax.models import DeferredStatementError

    result = _parse("data out; set ip; doover; run;")
    program = result.program
    assert program is not None
    statement = program.blocks[0].statements[2]
    assert statement.kind == "DO"
    assert isinstance(statement.payload, DeferredStatementError)
    assert statement.payload.native_cause == "unsupported DO statement: doover"


def test_classification_glued_by_keeps_body_names() -> None:
    result = _parse("data out; set ip; byfoo; run;")
    program = result.program
    assert program is not None
    statement = program.blocks[0].statements[2]
    assert statement.kind == "BY"
    assert [entry.name for entry in statement.payload.names] == ["byfoo"]


@pytest.mark.parametrize(
    ("code", "expected_kinds"),
    [
        ("data out; %let a=1; set i; x=1; run;", ["DATA", "SET", "ASSIGN", "RUN"]),
        ("data out; %put n; set i; run;", ["DATA", "SET", "RUN"]),
        ("data out; %* c; set i; run;", ["DATA", "SET", "RUN"]),
        ("data out; proc print; run; set i; x=1; run;", ["DATA", "SET", "ASSIGN", "RUN"]),
        ("%macro m; data fake(where=(a=1); %mend m; data out;set i;run;", ["DATA", "SET", "RUN"]),
    ],
)
def test_skip_masked_classification_matches_the_old_pipeline(code, expected_kinds):
    result = _parse(code)
    program = result.program
    assert program is not None, [d.message for d in result.diagnostics]
    kinds = [statement.kind for block in program.blocks for statement in block.statements]
    assert kinds == expected_kinds


def test_glued_data_frontier_splits_the_block_like_the_old_splitter() -> None:
    text = "data a;set i;x=1;datafoo b;set i;x=2;run;"
    result = _parse(text)
    program = result.program
    assert program is not None
    assert len(program.blocks) == 2
    first, second = program.blocks
    assert [s.kind for s in first.statements] == ["DATA", "SET", "ASSIGN"]
    assert (second.span.start, second.span.end) == (17, 41)
    assert text[second.span.start : second.span.end] == "datafoo b;set i;x=2;run;"
    assert [s.kind for s in second.statements] == ["DATA", "SET", "ASSIGN", "RUN"]
    assert text[first.span.start : first.span.end] == "data a;set i;x=1;"


def test_region_failure_keeps_incomplete_where_precheck_priority() -> None:
    code = "set i(where=(a=); z='p;\n"
    result = _parse(code)
    assert result.program is None
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_DATASET_WHERE_UNSUPPORTED"
    assert diagnostic.stage == "parse"
    assert (
        diagnostic.message
        == "dataset option WHERE= is invalid: incomplete WHERE= parenthesized expression"
    )
    assert diagnostic.location == "statement:1"
