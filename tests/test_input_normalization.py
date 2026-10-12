"""Focused tests for the initial input normalization."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from limulus.syntax.models import (
    AssignStatementSyntax,
    DataStatementSyntax,
    NameListStatementSyntax,
    NameTargetSyntax,
    OutputStatementSyntax,
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
                        f"module {fullname!r} is denied by the input-normalization"
                        " frontend isolation hook"
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


def _region_program(text: str):
    """Extract the typed program regions of one program text."""

    from limulus.syntax.program import LarkProgramFrontend

    frontend = LarkProgramFrontend()
    document = SourceDocument(text, "norm")
    regions = frontend.extract_program_regions(document)
    assert regions is not None
    return document, frontend, regions


def _parse_first_statement(text: str, keyword: str):
    document, frontend, regions = _region_program(text)
    matching = [
        region
        for region in regions.statement_regions
        if text[region.span.start : region.span.start + len(keyword)].lower() == keyword
    ]
    assert matching, f"no {keyword!r} statement region found in {text!r}"
    return document, frontend.parse_dataset_statement(document, matching[0])


def _region_summary(regions) -> dict[str, tuple] | None:
    """Structure snapshot without per-region object identity."""

    return {
        "statements": tuple(
            (statement.kind, statement.span.start, statement.span.end)
            for statement in regions.statement_regions
        ),
        "blocks": tuple(
            (block.span.start, block.span.end, block.terminated_by_run)
            for block in regions.blocks
        ),
    }


# --- program regions: tab and space spellings converge ---------------------------


def test_tab_program_regions_match_the_space_program() -> None:
    tabs = "data\tout;\n\tset\tinp;\n\toutput\tout;\nrun;"
    spaces = "data out;\n set inp;\n output out;\nrun;"
    document, _, tab_regions = _region_program(tabs)
    _, _, space_regions = _region_program(spaces)
    summary = _region_summary(tab_regions)
    assert summary is not None and summary == _region_summary(space_regions)

    # the source document keeps the raw tab characters and the regions
    # reference the original immutable document (identity, not a copy)
    assert "\t" in document.text
    assert tab_regions.source is document


def test_tab_statement_regions_slice_the_raw_source_unchanged() -> None:
    tabs = "data\tout;\n\toutput\tout;\nrun;"
    document, _, regions = _region_program(tabs)
    first = regions.statement_regions[0]
    assert document.text[first.span.start : first.span.end] == "data\tout"
    assert document.text[first.span.start : first.span.end] != "data out"
    assert first.span.line == 1
    assert first.span.end > first.span.start


# --- OUTPUT / dataset targets: tab converges with space -------------------------


def test_tab_output_targets_match_space_targets_with_spans() -> None:
    def _target_rows(text: str):
        document, statement = _parse_first_statement(text, "output")
        payload = statement.payload
        assert isinstance(payload, OutputStatementSyntax)
        return document, [
            (ref.name, ref.options, ref.span.start, ref.span.end)
            for ref in payload.targets
        ]

    document, tab_rows = _target_rows("output\tout;")
    _, space_rows = _target_rows("output out;")
    assert tab_rows == space_rows
    assert tab_rows[0][0] == "out"
    assert tab_rows[0][1].keep_vars == ()
    # the underlying document is still the raw (tab-carrying) request source
    assert "\t" in document.text


def test_tab_multiple_and_bare_output_targets_are_structured() -> None:
    document, frontend, regions = _region_program(
        "data\ta\tb;\n\toutput\ta\tb;\n\toutput;\nrun;"
    )
    parsed = [
        frontend.parse_dataset_statement(document, region)
        for region in regions.statement_regions
        if document.text[region.span.start :].lower().startswith(("data", "output"))
    ]
    assert [statement.kind for statement in parsed] == ["DATA", "OUTPUT", "OUTPUT"]

    data_payload = parsed[0].payload
    assert isinstance(data_payload, DataStatementSyntax)
    assert [ref.name for ref in data_payload.outputs] == ["a", "b"]

    named = parsed[1].payload
    assert isinstance(named, OutputStatementSyntax)
    assert [ref.name for ref in named.targets] == ["a", "b"]
    assert [ref.options.keep_vars for ref in named.targets] == [(), ()]

    bare = parsed[2].payload
    assert isinstance(bare, OutputStatementSyntax)
    assert bare.targets == ()


def _dataset_or_output_refs(text: str, keyword: str, kind: type):
    document, statement = _parse_first_statement(text, keyword)
    payload = statement.payload
    assert isinstance(payload, kind)
    refs = payload.outputs if isinstance(payload, DataStatementSyntax) else payload.targets  # type: ignore[union-attr]
    return document, refs


def test_tab_output_direct_options_share_the_data_option_structure() -> None:
    _, data_refs = _dataset_or_output_refs(
        "data out(keep=id rename=(id=key));", "data", DataStatementSyntax
    )
    _, output_refs = _dataset_or_output_refs(
        "output\tout(keep=id rename=(id=key));", "output", OutputStatementSyntax
    )
    assert [ref.name for ref in output_refs] == ["out"]
    assert [ref.name for ref in data_refs] == ["out"]
    assert output_refs[0].options.keep_vars == data_refs[0].options.keep_vars == ("id",)
    assert output_refs[0].options.rename_map == data_refs[0].options.rename_map == (
        ("id", "key"),
    )


# --- quoted literals, comments, and skip regions keep their tabs -----------------


def test_tab_inside_quoted_option_values_is_preserved() -> None:
    _, data_refs = _dataset_or_output_refs(
        "data out(label='a\tb');", "data", DataStatementSyntax
    )
    assert data_refs[0].name == "out"
    assert data_refs[0].options.label == "a\tb"


def test_tab_inside_where_quoted_value_is_preserved() -> None:
    _, data_refs = _dataset_or_output_refs(
        "data out(where=(x='a\tb'));", "data", DataStatementSyntax
    )
    assert data_refs[0].options.where_expr is not None


def test_comment_tabs_and_skip_region_raw_spans_stay_intact() -> None:
    # Comment content keeps its tab characters, and the skip-region span
    # keeps slicing the RAW document.  Macro/PROC content itself is not
    # exempt: its tabs outside quotes/comments normalize in the analysis
    # buffer like any other statement material, while the region span
    # (re-derived from the same offsets) and the raw slice stay unchanged
    # for exact display fidelity.
    text = "data out;\n\t* comment\tkeep;\nrun;\nproc\tprint;\nrun;"
    document, _, regions = _region_program(text)
    assert {region.kind for region in regions.skip_regions} == {"proc"}
    skip_span = regions.skip_regions[0].span
    assert document.text[skip_span.start : skip_span.end] == "proc\tprint;\nrun;"


# --- assignment targets and name lists converge ---------------------------------


def test_assignment_tab_target_becomes_one_structured_name() -> None:
    from limulus.syntax.program import LarkProgramFrontend

    text = "x\ty=1;"
    document = SourceDocument(text, "norm")
    frontend = LarkProgramFrontend()
    statement = frontend.parse_assignment_statement(document, document.span(0, len(text)))
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax)
    target = payload.target
    assert isinstance(target, NameTargetSyntax)
    # the structured target reads the normalized analysis text, not the raw
    # opaque parser string: the tab separator became one space
    assert target.name == "x y"
    assert target.span.start == 0
    assert target.span.end == 3
    # ...while the raw slice keeps the tab in the unmodified document
    assert document.text[target.span.start : target.span.end] == "x\ty"


def test_name_list_tab_separator_converges_with_single_space() -> None:
    from limulus.syntax.program import LarkProgramFrontend, name_list_consumer_views

    document = SourceDocument("keep\ta", "norm")
    frontend = LarkProgramFrontend()
    statement = frontend.parse_name_list_statement(document, document.span(0, 6))
    assert statement.kind == "KEEP"
    payload = statement.payload
    assert isinstance(payload, NameListStatementSyntax)
    assert tuple(name.name for name in payload.names) == ("a",)

    # the tab drift between the two legacy views disappears: the statement
    # parses like the space spelling now
    views = name_list_consumer_views(statement)
    assert views.native_names == ("a",)
    assert views.planning_names == ("a",)


def test_glued_keyword_and_newline_shapes_keep_the_6_4_views() -> None:
    from limulus.syntax.program import LarkProgramFrontend, name_list_consumer_views

    document = SourceDocument("keepx y", "norm")
    frontend = LarkProgramFrontend()
    statement = frontend.parse_name_list_statement(document, document.span(0, 7))
    views = name_list_consumer_views(statement)
    assert views.native_names == ("keepx", "y")
    assert views.planning_names == ("x", "y")

    document = SourceDocument("keep\n  a b", "norm")
    statement = frontend.parse_name_list_statement(document, document.span(0, 10))
    views = name_list_consumer_views(statement)
    payload = statement.payload
    assert isinstance(payload, NameListStatementSyntax)
    assert tuple(name.name for name in payload.names) == ("keep", "a", "b")
    assert views.planning_names == ("a", "b")


# --- shared analysis helper contract ---------------------------------------------

_LIT = "a\tb"
_DOUBLE = '"a\t""b\tc"'
_BACKSLASH = "'a\\\tb'"
_LINE = "* line\tcomment; after\ttab"
_BLOCK = "a/* block\tcomment */b\tc"
_CONTROL = "a\x1cb%"
_MACRO = "%let x=1;\tdata\tout;"

# Canonical region-lex matrix:
# a comment opener is lexical only at a region TOKEN START.  The greedy
# OTHER_TEXT chunks (``[;'"\s]`` terminators) swallow ``*``/``/*`` mid-chunk,
# there is no character-anywhere and no just-prev-char heuristic, and the
# macro / PROC region tokens create no comment boundaries and no tab
# exemption (their non-quoted tabs normalize like any other statement
# material).
_CANONICAL_LEX_MATRIX = (
    # a star INSIDE a greedy chunk is chunk material every time
    ("x=2*\t3;", "x=2* 3;"),
    ("x=2**\t3;", "x=2** 3;"),
    ("x*\ty=2", "x* y=2"),
    ("x/* block\tkeep */y\tz=2", "x/* block keep */y z=2"),
    ("a/* block\tcomment */b\tc", "a/* block comment */b c"),
    # a star at a token start still opens the real LINE_COMMENT; tabs inside
    # a comment stay comment content (the old datastep_split.lark truth)
    ("x=2 *\t3;", "x=2 * 3;"),
    ("* comment\tkeep; x\ty=2", "* comment\tkeep; x y=2"),
    ("/* block\tkeep */ x\ty=2", "/* block\tkeep */ x y=2"),
    # a closed quote resets the token boundary, so the star opens a comment
    ("x='a'* comment\tkeep;", "x='a'* comment keep;"),
    # Unterminated ordinary run: content stays raw until the statement ``;``
    # (quoted-content rule), then following statements normalize again.
    ("x = 'abc;\n\tset\ty;", "x = 'abc;\n set y;"),
    ("x = 'abc out\tz2;\tset y;", "x = 'abc out\tz2; set y;"),
    # doubled quotes keep the quoted tab and resume outside-token state
    ("x='a''\tb';\ty=2", "x='a''\tb'; y=2"),
    # macro / PROC region tokens: quote-aware, no tab exemption
    ("proc sql; select a\tb; run; data\tout;", "proc sql; select a b; run; data out;"),
    ("proc print;\n\trun;", "proc print;\n run;"),
    ("%let x=1;\tdata\tout;", "%let x=1; data out;"),
    ("%* macro\tcomment; after\there", "%* macro comment; after here"),
)


def test_initial_lex_state_follows_region_token_boundaries() -> None:
    from limulus.syntax.normalization import initial_analysis_text

    for source, expected in _CANONICAL_LEX_MATRIX:
        analysis = initial_analysis_text(source)
        assert len(analysis) == len(source), source
        assert analysis == expected, source
        # re-normalizing the already-normalized text is stable
        assert initial_analysis_text(analysis) == expected, source


_CANONICAL_LEX_TYPED_PAIRS = (
    # typed-convergence members of the same matrix: tab and space spellings
    # produce the equal assignment-target structure everywhere the region
    # lex allows normalization
    ("x*\ty=2", "x* y=2"),
    ("x=2*\t3;", "x=2* 3;"),
)


def test_assignment_targets_converge_under_region_lex_matrix() -> None:
    from limulus.syntax.models import AssignStatementSyntax, NameTargetSyntax
    from limulus.syntax.program import LarkProgramFrontend

    for raw, spaced in _CANONICAL_LEX_TYPED_PAIRS:
        document = SourceDocument(raw, "norm")
        frontend = LarkProgramFrontend()
        statement = frontend.parse_assignment_statement(
            document, document.span(0, len(raw))
        )
        payload = statement.payload
        assert isinstance(payload, AssignStatementSyntax), raw
        assert isinstance(payload.target, NameTargetSyntax), raw
        # the tab spelling and the space spelling converge on one target
        spaced_document = SourceDocument(spaced, "norm")
        spaced_statement = frontend.parse_assignment_statement(
            spaced_document, spaced_document.span(0, len(spaced))
        )
        spaced_payload = spaced_statement.payload
        assert isinstance(spaced_payload, AssignStatementSyntax), spaced
        assert payload.target.name == spaced_payload.target.name
        assert "\t" not in payload.target.name


def test_assignment_target_around_token_start_comment_keeps_comment_body() -> None:
    from limulus.syntax.models import AssignStatementSyntax, NameTargetSyntax
    from limulus.syntax.program import LarkProgramFrontend
    from limulus.syntax.normalization import initial_analysis_text

    # acceptance row: ``* comment\tkeep;`` is a real LINE_COMMENT at token
    # start, so its body tab stays comment CONTENT, while the next target
    # (``x\ty``) reads the normalized analysis text (``x y``)
    raw = "* comment\tkeep; x\ty=2"
    document = SourceDocument(raw, "norm")
    frontend = LarkProgramFrontend()
    statement = frontend.parse_assignment_statement(
        document, document.span(0, len(raw))
    )
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax), raw
    assert isinstance(payload.target, NameTargetSyntax), raw
    assert "\t" in initial_analysis_text(raw)
    assert initial_analysis_text(raw).endswith(" x y=2")
    assert payload.target.name == "* comment\tkeep; x y"


def test_analysis_length_and_literal_rules() -> None:
    from limulus.syntax.normalization import initial_analysis_text

    for sample in (
        _LIT,
        _DOUBLE,
        _BACKSLASH,
        _LINE,
        _BLOCK,
        _CONTROL,
        _MACRO,
        "/* unterminated\tcomment",
    ):
        analysis = initial_analysis_text(sample)
        assert len(analysis) == len(sample), sample

    assert initial_analysis_text("data\tout;") == "data out;"
    assert initial_analysis_text("no separator change") == "no separator change"
    assert initial_analysis_text(_LIT) == "a b"
    assert initial_analysis_text(_DOUBLE) == _DOUBLE
    assert initial_analysis_text(_BACKSLASH) == _BACKSLASH
    assert initial_analysis_text(_LINE) == "* line\tcomment; after tab"
    # the block opener is swallowed by the greedy OTHER_TEXT chunk "a/*" so
    # no comment ever opens here; tabs outside quotes normalize
    assert initial_analysis_text(_BLOCK) == "a/* block comment */b c"
    assert initial_analysis_text(_CONTROL) == _CONTROL
    assert initial_analysis_text(_MACRO) == "%let x=1; data out;"
    # the buffer is stable after re-normalizing already-normalized text
    assert initial_analysis_text(initial_analysis_text("data\tout;")) == "data out;"


# --- bounded, request-local analysis buffer ----------------------------------------


class _Marker(str):
    """Toy marker: a str subclass so the request text can be weakref-ed."""


def test_analysis_buffer_keeps_no_inter_request_history() -> None:
    import gc
    import weakref

    from limulus.syntax.models import SourceDocument
    from limulus.syntax.program import LarkProgramFrontend

    frontend = LarkProgramFrontend()
    marker = _Marker(" ".join(("request", "buffer", "attempt")))
    reference = weakref.ref(marker)
    frontend._analysis_text(SourceDocument(marker, "norm"))
    del frontend, marker
    gc.collect()
    # no global (inter-request) history may survive the frontend: the buffer
    # is bounded to the most recent document identity pair only
    assert reference() is None


def test_analysis_buffer_replaces_the_latest_document_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref

    from limulus.syntax import program
    from limulus.syntax.models import SourceDocument

    calls: list[str] = []
    real = program.initial_analysis_text

    def counting(text: str) -> str:
        calls.append(text)
        return real(text)

    monkeypatch.setattr(
        program, "initial_analysis_text", counting
    )
    frontend = program.LarkProgramFrontend()
    document = SourceDocument(
        "data\tout;\n\tset\tinp;\n\toutput\tout;\nrun;", "norm"
    )
    regions = frontend.extract_program_regions(document)
    assert regions is not None
    for region in list(regions.statement_regions)[:3]:
        frontend.parse_dataset_statement(document, region)
    # one document, many statement nodes: exactly one normalization
    assert len(calls) == 1

    # a NEW document with EQUAL text and EQUAL source_id is still a new
    # request pair: identity replaces the buffer (rescan), source_id alone
    # is never the key
    newer = SourceDocument(document.text, "norm")
    newer_regions = frontend.extract_program_regions(newer)
    assert newer_regions is not None
    assert len(calls) == 2
    assert calls[1] == document.text
    assert frontend._analysis_document is newer

    # only the latest pair is retained: the replaced document is collectable
    replaced = weakref.ref(document)
    del document, regions
    gc.collect()
    assert replaced() is None

    # truth identity is per frontend instance: a fresh frontend rescans
    fresh = program.LarkProgramFrontend()
    fresh.extract_program_regions(newer)
    assert len(calls) == 3

    # an empty request also replaces the buffered pair like any other
    # document (the buffer method has no blank shortcut; source_id alone
    # is never the key)
    empty = SourceDocument("", "norm")
    assert fresh._analysis_text(empty) == ""
    assert fresh._analysis_document is empty


def test_frontend_imports_without_data_environment() -> None:    _run_isolated(
        """
        from limulus.syntax.normalization import initial_analysis_text
        from limulus.syntax.program import LarkProgramFrontend

        assert initial_analysis_text("data\\tout;") == "data out;"

        frontend = LarkProgramFrontend()
        output = frontend.extract_program_regions(__import__(
            "limulus.syntax.models", fromlist=["SourceDocument"]
        ).SourceDocument("data\\tout;\\nrun;", "iso"))
        assert output is not None
        assert [region.kind for region in output.statement_regions] == ["DATA", "RUN"]
        """
    )
