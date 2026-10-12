"""Focused tests for the dataset statement frontend entry."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from limulus.syntax.models import (
    DataStatementSyntax,
    DatasetOptionSyntax,
    DatasetReferenceSyntax,
    DeferredStatementError,
    OutputStatementSyntax,
    SetStatementOptionSyntax,
    SourceDocument,
    SourceSpan,
    SourceStatementSyntax,
    StatementSyntax,
    WhereComparison,
    WhereLogicalExpression,
    WhereNullPredicate,
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
                        f"module {fullname!r} is denied by the dataset frontend isolation hook"
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


def _statement_frontend(text: str):
    from limulus.syntax.program import LarkProgramFrontend

    frontend = LarkProgramFrontend()
    document = SourceDocument(text, "datasets")
    regions = frontend.extract_program_regions(document)
    assert regions is not None
    return document, frontend, regions


def _parse_first_statement(text: str, keyword: str, *, statement_index: int = 0):
    document, frontend, regions = _statement_frontend(text)
    matching = [
        region
        for region in regions.statement_regions
        if text[region.span.start : region.span.start + len(keyword)].lower() == keyword
    ]
    assert matching, f"no {keyword!r} statement region found in {text!r}"
    return frontend.parse_dataset_statement(document, matching[statement_index])


# --- typed payloads and spans ----------------------------------------------------


def test_data_statement_outputs_keep_order_names_and_spans() -> None:
    text = "data out1(keep=a) out2; run;"
    statement = _parse_first_statement(text, "data")
    payload = statement.payload
    assert isinstance(payload, DataStatementSyntax)
    assert [ref.name for ref in payload.outputs] == ["out1", "out2"]
    first = payload.outputs[0]
    assert text[first.span.start : first.span.end] == "out1(keep=a)"
    assert payload.outputs[1].options.keep_vars == ()


def test_bare_data_statement_refuses_like_the_legacy_grammar() -> None:
    statement = _parse_first_statement("data;", "data")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert "Unsupported statement syntax" in payload.native_cause
    assert statement.kind == "DATA"


def test_bare_output_statement_is_empty_typed_payload() -> None:
    statement = _parse_first_statement("output;", "output")
    payload = statement.payload
    assert isinstance(payload, OutputStatementSyntax)
    assert payload.targets == ()


def test_set_statement_contains_refs_and_statement_options() -> None:
    text = "set indsname=src end=flag a b(keep=x);"
    statement = _parse_first_statement(text, "set")
    payload = statement.payload
    assert isinstance(payload, SourceStatementSyntax)
    assert [ref.name for ref in payload.refs] == ["a", "b"]
    assert text[payload.refs[1].span.start : payload.refs[1].span.end] == "b(keep=x)"
    assert payload.refs[1].options.keep_vars == ("x",)
    assert isinstance(payload.options, SetStatementOptionSyntax)
    assert payload.options.indsname_var == "src"
    assert payload.options.end_var == "flag"


def test_merge_statement_same_ref_and_statement_option_grammar() -> None:
    text = "merge a(drop=x keep=y) b;"
    statement = _parse_first_statement(text, "merge")
    payload = statement.payload
    assert isinstance(payload, SourceStatementSyntax)
    assert [ref.name for ref in payload.refs] == ["a", "b"]
    assert text[payload.refs[0].span.start : payload.refs[0].span.end] == "a(drop=x keep=y)"
    assert payload.refs[0].options.keep_vars == ("y",)
    assert payload.refs[0].options.drop_vars == ("x",)


def test_output_statement_targets_are_structured() -> None:
    text = "output first(in=x) second;"
    statement = _parse_first_statement(text, "output")
    payload = statement.payload
    assert isinstance(payload, OutputStatementSyntax)
    assert [ref.name for ref in payload.targets] == ["first", "second"]
    assert text[payload.targets[0].span.start : payload.targets[0].span.end] == "first(in=x)"


def _parse_named_output(text: str = "output first(in=x) second;"):
    return _parse_first_statement(text, "output")


# --- option boundary corpus (existing truth) -------------------------------------


def test_option_corpus_nested_and_quoted_parentheses_preserved() -> None:
    text = (
        "data out; set inp(keep=id amount drop=tmp "
        'where=((amount >= 20 and amount < 40) or name = "A) (B") '
        "rename=(amount=score) firstobs=2 obs=1);"
    )
    statement = _parse_first_statement(text, "set")
    payload = statement.payload
    assert isinstance(payload, SourceStatementSyntax)
    ref = payload.refs[0]
    options = ref.options
    assert isinstance(options, DatasetOptionSyntax)
    assert options.keep_vars == ("id", "amount")
    assert options.drop_vars == ("tmp",)
    assert options.where_expr == WhereLogicalExpression(
        "or",
        (
            WhereLogicalExpression(
                "and",
                (
                    WhereComparison("amount", ">=", 20),
                    WhereComparison("amount", "<", 40),
                ),
            ),
            WhereComparison("name", "=", "A) (B"),
        ),
    )
    assert options.rename_map == (("amount", "score"),)
    assert options.firstobs == 2
    assert options.obs == 1
    assert text[ref.span.start : ref.span.end] == (
        'inp(keep=id amount drop=tmp where=((amount >= 20 and amount < 40) '
        'or name = "A) (B") rename=(amount=score) firstobs=2 obs=1)'
    )


def test_option_corpus_quoted_semicolon_and_adjacent_options() -> None:
    statement = _parse_first_statement(
        "data out; set inp(where=(name = 'A;B') rename=(name=label) obs=2);",
        "set",
    )
    payload = statement.payload
    ref = payload.refs[0]
    assert ref.options.where_expr == WhereComparison("name", "=", "A;B")
    assert ref.options.rename_map == (("name", "label"),)
    assert ref.options.obs == 2


def test_option_corpus_null_predicate_and_rename_order() -> None:
    statement = _parse_first_statement(
        "merge inp(where=(amount is null) rename=(b=a2 a=b1));",
        "merge",
    )
    payload = statement.payload
    options = payload.refs[0].options
    assert options.where_expr == WhereNullPredicate("amount", False)
    assert options.rename_map == (("b", "a2"), ("a", "b1"))


def test_option_corpus_doubled_quotes_stay_inside_values() -> None:
    statement = _parse_first_statement(
        "set inp(where=(name = 'it''s; fine') keep=x5)",
        "set",
    )
    payload = statement.payload
    options = payload.refs[0].options
    assert options.where_expr == WhereComparison("name", "=", "it's; fine")
    assert options.keep_vars == ("x5",)


def test_label_option_value_is_unquoted_string() -> None:
    statement = _parse_first_statement(
        "set inp(label='monthly data');",
        "set",
    )
    payload = statement.payload
    assert payload.refs[0].options.label == "monthly data"


def test_in_option_value_is_kept_verbatim_with_case() -> None:
    statement = _parse_first_statement("merge a b(in=FLAG);", "merge")
    payload = statement.payload
    assert payload.refs[1].options.in_var == "FLAG"


# --- refusal behavior (existing semantics, typed plumbing) -----------------------


def _assert_deferred(payload, cause_fragment: str, kind: str) -> None:
    assert isinstance(payload, DeferredStatementError)
    assert cause_fragment in payload.native_cause
    assert isinstance(payload.span, SourceSpan)


def test_unsupported_dataset_option_refuses_with_cause() -> None:
    statement = _parse_first_statement("data out; set inp(bad=1);", "set")
    _assert_deferred(statement.payload, "Unsupported dataset reference option: bad=1", "SET")


def test_option_scope_refusal_for_indsname_inside_reference() -> None:
    statement = _parse_first_statement("set inp(indsname=s);", "set")
    _assert_deferred(
        statement.payload, "INDSNAME= is only allowed as a SET statement option.", "SET"
    )
    assert statement.kind == "SET"


def test_statement_level_indsname_refused_on_merge() -> None:
    statement = _parse_first_statement("merge m indsname=s;", "merge")
    _assert_deferred(
        statement.payload, "INDSNAME= is only allowed as a SET statement option.", "MERGE"
    )


def test_statement_level_in_requires_reference_position() -> None:
    statement = _parse_first_statement("set a in=flag;", "set")
    _assert_deferred(
        statement.payload,
        "IN= is only allowed as a dataset reference option",
        "SET",
    )


def test_where_refuses_functions_arithmetic_and_subscripts() -> None:
    for where_value, fragment in (
        ("sum(x) > 1", "function or incomplete parenthesized"),
        ("x + 1 > 2", "arithmetic"),
        ("x[1] > 1", "subscript"),
    ):
        statement = _parse_first_statement(f"set inp(where=({where_value}));", "set")
        _assert_deferred(
            statement.payload,
            "dataset option WHERE= is invalid",
            "SET",
        )
        assert fragment in statement.payload.native_cause


def test_where_refuses_arbitrary_python_without_evaluation() -> None:
    statement = _parse_first_statement(
        "set inp(where=(__import__('os').listdir()));",
        "set",
    )
    _assert_deferred(
        statement.payload,
        "dataset option WHERE= is invalid",
        "SET",
    )


def test_incomplete_where_option_refuses_with_missing_parenthesis_cause() -> None:
    statement = _parse_first_statement(
        "set inp(where=(name = 'A;B') keep=x;",
        "set",
    )
    _assert_deferred(
        statement.payload,
        "incomplete WHERE= parenthesized expression",
        "SET",
    )


def test_empty_option_group_refuses_like_statement_syntax() -> None:
    statement = _parse_first_statement("set inp();", "set")
    _assert_deferred(statement.payload, "Unsupported statement syntax", "SET")


# --- legacy parity battery (unchanged old parser as the corpus oracle) ---------


@pytest.mark.parametrize(
    "statement_text",
    [
        'set inp(keep=id amount drop=tmp where=((amount >= 20 and amount < 40) '
        'or name = "A) (B") rename=(amount=score) firstobs=2 obs=1);',
        "set inp(where=(name = 'A;B') rename=(name=label) obs=2);",
        "set inp(where=missing is null);",
        "merge a(in=flag) b(keep=x where=(x < 3));",
        "set indsname=src end=flag a b(keep=x);",
        "merge m1(in=a) m2(in=b) end=done;",
        "data out(keep=a) out2(obs=3);",
        # Existing corpus shapes (tests/cases/parser/core/programs).
        "set in_a(in=in_left) in_b(in=in_right) indsname=src end=last;",
        "merge left(keep=id amount drop=tmp where=(amount > 0) rename=(amount=amt)) right(in=in_right keep=id) end=eof;",
        "set inp end = eof;",
        "set inp end= eof;",
        "set inp(in = flag2);",
        "set inp(in= flag3);",
        "set inp(in =flag4);",
        "set in(keep=id amount drop=tmp where=(amount > 0) rename=(amount=amt) firstobs=2 obs=3);",
        "data;",
        'set inp(label="monthly data");',
    ],
)
def test_typed_option_payloads_match_legacy_parser_options(statement_text: str) -> None:
    from limulus.parser import ParserService
    from limulus.syntax.legacy import DatasetReferenceOptionSpec

    legacy = ParserService().parse(statement_text)
    keyword = statement_text.split(None, 1)[0].lower()
    statement = _parse_first_statement(statement_text, keyword)
    typed = statement.payload
    if legacy.diagnostics:
        # The legacy corpus oracle refused the statement: the typed entry
        # must refuse it too (no surprising acceptance drift).
        assert isinstance(typed, DeferredStatementError)
        return

    legacy_statement = legacy.ast.statements[0]
    if isinstance(typed, SourceStatementSyntax):
        assert legacy_statement.kind in {"SET", "MERGE"}
        assert [ref.name for ref in typed.refs] == [
            ref.name for ref in legacy_statement.dataset_refs
        ]
        for typed_ref, legacy_ref in zip(typed.refs, legacy_statement.dataset_refs):
            _assert_options_equal(typed_ref.options, legacy_ref.options)
        assert typed.options.indsname_var == legacy_statement.statement_options.indsname_var
        assert typed.options.end_var == legacy_statement.statement_options.end_var
        return
    assert isinstance(typed, DataStatementSyntax)
    assert legacy_statement.kind == "DATA"
    assert [ref.name for ref in typed.outputs] == [
        ref.name for ref in legacy_statement.output_refs
    ]
    for typed_ref, legacy_ref in zip(typed.outputs, legacy_statement.output_refs):
        _assert_options_equal(typed_ref.options, legacy_ref.options)


def _assert_options_equal(typed_options, legacy_options) -> None:
    from limulus.syntax.legacy import DatasetReferenceOptionSpec

    assert isinstance(legacy_options, DatasetReferenceOptionSpec)
    assert typed_options.in_var == legacy_options.in_var
    assert typed_options.keep_vars == legacy_options.keep_vars
    assert typed_options.drop_vars == legacy_options.drop_vars
    assert typed_options.where_expr == legacy_options.where_expr
    assert typed_options.rename_map == tuple(legacy_options.rename_map.items())
    assert typed_options.firstobs == legacy_options.firstobs
    assert typed_options.obs == legacy_options.obs
    assert typed_options.label == legacy_options.label


# --- dotted dataset names ---------------------------------------------------------


@pytest.mark.parametrize(
    "statement_text",
    [
        "set work.inp(keep=x);",
        "data work.out(in=flag);",
        "merge work.left(in=g) work.right;",
        "data Work.Out(keep=a obs=2) work.T2;",
        "set Work.Input;",
        "set income.table end=last_row;",
        "merge work.a(drop=x keep=y where=(amount > 0)) work.b;",
        "set work.inp(label='monthly data') end=fin;",
    ],
)
def test_dotted_typed_option_payloads_match_legacy_parser_options(
    statement_text: str,
) -> None:
    from limulus.parser import ParserService

    legacy = ParserService().parse(statement_text)
    keyword = statement_text.split(None, 1)[0].lower()
    statement = _parse_first_statement(statement_text, keyword)
    typed = statement.payload
    assert not isinstance(typed, DeferredStatementError), statement_text
    if legacy.diagnostics:
        # The legacy oracle refusal truth stays in force for dotted refs too.
        assert isinstance(typed, DeferredStatementError)
        return
    legacy_statement = legacy.ast.statements[0]
    if isinstance(typed, SourceStatementSyntax):
        assert legacy_statement.kind in {"SET", "MERGE"}
        assert [ref.name for ref in typed.refs] == [
            ref.name for ref in legacy_statement.dataset_refs
        ]
        for typed_ref, legacy_ref in zip(typed.refs, legacy_statement.dataset_refs):
            _assert_options_equal(typed_ref.options, legacy_ref.options)
        assert typed.options.indsname_var == legacy_statement.statement_options.indsname_var
        assert typed.options.end_var == legacy_statement.statement_options.end_var
        return
    assert isinstance(typed, DataStatementSyntax)
    assert legacy_statement.kind == "DATA"
    assert [ref.name for ref in typed.outputs] == [
        ref.name for ref in legacy_statement.output_refs
    ]
    for typed_ref, legacy_ref in zip(typed.outputs, legacy_statement.output_refs):
        _assert_options_equal(typed_ref.options, legacy_ref.options)


def test_dotted_names_keep_case_and_raw_reference_spans() -> None:
    text = "data Work.Out(keep=Id) T2;"
    statement = _parse_first_statement(text, "data")
    payload = statement.payload
    assert isinstance(payload, DataStatementSyntax)
    first = payload.outputs[0]
    assert first.name == "Work.Out"
    assert text[first.span.start : first.span.end] == "Work.Out(keep=Id)"
    assert first.options.keep_vars == ("Id",)


def test_dotted_output_statement_returns_structured_targets() -> None:
    text = "output work.T(keep=v) s2(in=flag);"
    statement = _parse_first_statement(text, "output")
    payload = statement.payload
    assert isinstance(payload, OutputStatementSyntax)
    assert not isinstance(payload, DeferredStatementError)
    assert [ref.name for ref in payload.targets] == ["work.T", "s2"]
    assert text[payload.targets[0].span.start : payload.targets[0].span.end] == (
        "work.T(keep=v)"
    )
    assert payload.targets[1].options.in_var == "flag"


def test_whole_text_parse_keeps_dotted_names_without_stopping() -> None:
    from limulus.syntax.program import parse_program

    for text in (
        "data work.out; set work.in; run;",
        "data out; set in.a (keep=x) b; run;",
        "data out; set in; if a then output work.b(keep=v); run;",
        "data out; set work.in; output work.out2; run;",
    ):
        result = parse_program(text)
        assert result.program is not None, (text, [(d.code, d.message) for d in result.diagnostics])
        assert result.diagnostics == (), (text, result.diagnostics)


def test_inline_output_dotted_target_is_structured_not_deferred() -> None:
    from limulus.syntax.models import IfStatementSyntax, OutputStatementSyntax
    from limulus.syntax.program import parse_program

    result = parse_program("data out; set in; if a then output work.b(keep=v); run;")
    assert result.program is not None
    statements = result.program.blocks[0].statements
    if_statement = statements[2]
    payload = if_statement.payload
    assert isinstance(payload, IfStatementSyntax)
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    action_payload = action.payload
    assert isinstance(action_payload, OutputStatementSyntax)
    assert [ref.name for ref in action_payload.targets] == ["work.b"]
    assert action_payload.targets[0].options.keep_vars == ("v",)


@pytest.mark.parametrize(
    "statement_text",
    [
        "set work.a(end=f);",
        "set work.x(ends=y);",
        "merge work.m indsname=s;",
        "set work.a in=flag;",
    ],
)
def test_dotted_ref_option_scope_refusals_keep_legacy_cause_family(
    statement_text: str,
) -> None:
    statement = _parse_first_statement(
        statement_text, statement_text.split()[0].lower()
    )
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert isinstance(statement.span, SourceSpan)
    assert statement.kind in {"SET", "MERGE"}


# --- group-external quotes are rejected (legacy rejection truth) ------------------


@pytest.mark.parametrize(
    ("statement_text", "cause"),
    [
        ("set a(keep=x) 'stray' b;", "Unsupported statement syntax near line 1, column 15."),
        ("set 'work.a'n;", "Unsupported statement syntax near line 1, column 5."),
        ("data 'out x'n;", "Unsupported statement syntax near line 1, column 6."),
        ('data a(keep=x) "d" b;', "Unsupported statement syntax near line 1, column 16."),
        ("set a 'stray' (keep=x) b;", "Unsupported statement syntax near line 1, column 7."),
    ],
)
def test_group_external_quoted_tokens_are_rejected_with_legacy_cause(
    statement_text: str, cause: str
) -> None:
    from limulus.syntax.legacy import DatasetReferenceOptionSpec  # noqa: F401

    statement = _parse_first_statement(statement_text, statement_text.split(None, 1)[0].lower())
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert payload.native_cause == cause
    assert payload.display_text == statement_text.rstrip(";").strip()
    assert statement.kind in {"DATA", "SET", "MERGE", "OUTPUT"}
    assert isinstance(statement.span, SourceSpan)


def test_valid_quoted_options_inside_groups_stay_accepted() -> None:
    for statement_text in (
        "set inp(label='monthly data') x2;",
        "set inp(where=(name = 'A;B') rename=(name=label) obs=2);",
        'set inp(where=(name = "A) (B") keep=x);',
    ):
        statement = _parse_first_statement(statement_text, "set")
        assert not isinstance(statement.payload, DeferredStatementError), statement_text


# --- entry boundary and region integration ------------------------------------------


def test_non_dataset_statement_is_rejected_without_filler() -> None:
    document, frontend, regions = _statement_frontend("x = 1;\ndata a; run;\n")
    with pytest.raises(ValueError):
        frontend.parse_dataset_statement(document, regions.statement_regions[0])


def test_entry_integration_uses_existing_regions_without_full_reparse() -> None:
    text = "data out; run;\nset a(keep=x) b;\n"
    document, frontend, regions = _statement_frontend(text)
    for region in regions.statement_regions:
        if region.kind == "RUN":
            continue
        statement = frontend.parse_dataset_statement(document, region)
        assert isinstance(statement, StatementSyntax)
        assert statement.kind in {"DATA", "SET", "MERGE", "OUTPUT"}
        assert statement.span == region.span


def test_entry_accepts_plain_span() -> None:
    document, frontend, regions = _statement_frontend("merge a b;")
    span = regions.statement_regions[0].span
    statement = frontend.parse_dataset_statement(document, span)
    assert isinstance(statement.payload, SourceStatementSyntax)
    assert statement.span == span


def test_statement_span_must_belong_to_document() -> None:
    document, frontend, regions = _statement_frontend("set a;")
    foreign = SourceSpan(start=0, end=6, line=1, column=1, source_id="other")
    with pytest.raises(ValueError):
        frontend.parse_dataset_statement(document, foreign)


# --- purity -----------------------------------------------------------------------


def test_dataset_frontend_imports_without_data_environment() -> None:
    _run_isolated(
        """
        import sys

        from limulus.syntax.program import LarkProgramFrontend
        from limulus.syntax.models import SourceDocument

        frontend = LarkProgramFrontend()
        document = SourceDocument("data out; set inp(keep=a where=(x is not null));")
        regions = frontend.extract_program_regions(document)
        assert regions is not None
        statement = frontend.parse_dataset_statement(document, regions.statement_regions[1])
        payload = statement.payload
        assert payload.refs[0].options.keep_vars == ("a",)
        assert payload.refs[0].options.where_expr is not None
        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules, denied
        """
    )
