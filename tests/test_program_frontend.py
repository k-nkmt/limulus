"""Focused tests for the whole-text program region frontend."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from limulus.syntax.models import SourceDocument

REPO_ROOT = Path(__file__).resolve().parents[1]

# Offset-preserving blanks.
_DENY_HOOK = textwrap.dedent(
    """
    import sys

    _DENIED = ("pyarrow", "polars", "limulus_native", "limulus.session")


    class _DeniedDependencyGate:
        def find_spec(self, fullname, path=None, target=None):
            for root in _DENIED:
                if fullname == root or fullname.startswith(root + "."):
                    raise ImportError(
                        f"module {fullname!r} is denied by the program frontend isolation hook"
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


def _hook_mask(text: str, ranges: list[tuple[int, int]]) -> str:
    masked = list(text)
    for start, end in ranges:
        for index in range(start, end):
            if masked[index] not in {"\n", "\r"}:
                masked[index] = " "
    return "".join(masked)


def _frontend():
    from limulus.syntax.program import LarkProgramFrontend

    return LarkProgramFrontend()


def _frontend_regions(text: str):
    frontend = _frontend()
    document = SourceDocument(text, "frontend")
    return document, frontend.extract_program_regions(document)


def _splitter_blocks(masked_text: str) -> tuple[str, ...] | None:
    from limulus.block_splitter import DataStepBlockSplitter

    return DataStepBlockSplitter().split(masked_text)


def _assert_block_texts_match_splitter_projection(text: str) -> None:
    """Frontend block spans equal the splitter projection of the masked text.

    Skip regions are masked before splitting blocks, so interior skip
    characters are compared as blanks inside the same span: masking the
    frontend skip spans must transform exactly the splitter's block texts,
    byte for byte, and mask nothing outside blocks.
    """

    _, regions = _frontend_regions(text)
    assert regions is not None
    ranges = [(r.span.start, r.span.end) for r in regions.skip_regions]
    masked = _hook_mask(text, ranges)
    legacy = _splitter_blocks(masked)
    frontend = tuple(
        masked[block.span.start : block.span.end].strip() for block in regions.blocks
    )
    assert frontend == legacy


# --- empty and skip-only programs ------------------------------------------------


def test_empty_and_whitespace_sources_yield_zero_regions() -> None:
    for text in ("", " \n\t ", "\n\n"):
        _, regions = _frontend_regions(text)
        assert regions is not None
        assert regions.blocks == ()
        assert regions.skip_regions == ()
        assert regions.statement_regions == ()
        assert regions.span.start == 0
        assert regions.span.end == len(text)
        assert regions.span.source_id == "frontend"


def test_skip_only_program_yields_zero_blocks() -> None:
    text = "%let x = 1;\n%macro noop();\ndata hidden;\nset inp;\nrun;\n%mend noop;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert regions.blocks == ()
    assert regions.statement_regions == ()


def test_skip_only_proc_program_yields_zero_blocks() -> None:
    text = "proc sort data=inp;\nby id;\nrun;\nproc print; quit;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert regions.blocks == ()
    assert all(
        region.kind == "proc" and text[region.span.start : region.span.end].startswith("proc")
        for region in regions.skip_regions
    )


# --- ordered blocks and skip spans ----------------------------------------------


def test_blocks_and_skips_are_ordered_in_source_order() -> None:
    text = (
        "proc sort data=inp; run;\n"
        "data a; set in; run;\n"
        "%let x = 1;\n"
        "data b; x = 1; run;\n"
        "%macro noop();\ndata hidden;\n%mend noop;\n"
        "data c; run;\n"
    )
    _, regions = _frontend_regions(text)
    assert regions is not None

    skip_starts = [region.span.start for region in regions.skip_regions]
    assert skip_starts == sorted(skip_starts)
    block_starts = [block.span.start for block in regions.blocks]
    assert block_starts == sorted(block_starts)

    assert [block.span.start for block in regions.blocks] == [
        text.index("data a;"),
        text.index("data b;"),
        text.index("data c;"),
    ]
    assert [region.kind for region in regions.skip_regions] == [
        "proc",
        "macro",
        "macro",
    ]

    _assert_block_texts_match_splitter_projection(text)


def test_skip_regions_are_not_blocks_or_statements() -> None:
    text = "%let x = 1;\ndata a; run;\n%put note;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None

    skip_spans = [(region.span.start, region.span.end) for region in regions.skip_regions]
    assert len(skip_spans) == 2
    for start, end in skip_spans:
        for block in regions.blocks:
            if block.span.start <= start and end <= block.span.end:
                pytest.fail(f"skip span {skip_spans} embedded inside block {block}")
        for statement in regions.statement_regions:
            assert (
                statement.span.end <= start or end <= statement.span.start
            ), "a skip region must not double as a statement region"


def test_macro_and_proc_run_vs_quit_skip_classification() -> None:
    text = (
        "%let total = 3;\n"
        "%* comment style macro line;\n"
        "proc sort data=inp; run;\n"
        "proc print data=inp; quit;\n"
        "%macro named(a);\n%mend named;\n"
    )
    _, regions = _frontend_regions(text)
    assert regions is not None
    kinds = [region.kind for region in regions.skip_regions]
    assert kinds == ["macro", "macro", "proc", "proc", "macro"]
    assert [text[r.span.start : r.span.end].strip().lower().splitlines()[0][:6] for r in regions.skip_regions] == [
        "%let t",
        "%* com",
        "proc s",
        "proc p",
        "%macro",
    ]


def test_bare_quit_after_run_stays_statement_not_skip() -> None:
    text = "data a; run; quit;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert [region.kind for region in regions.skip_regions] == []
    assert [block.terminated_by_run for block in regions.blocks] == [True, False]
    _assert_block_texts_match_splitter_projection(text)


# --- quoted delimiters, literals, comments --------------------------------------


def test_quoted_delimiters_keep_one_block_and_no_skips() -> None:
    text = (
        "data out;\n"
        "set inp;\n"
        'value = "proc sort data=inp; quit; %macro noop();";\n'
        "run;\n"
    )
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert len(regions.blocks) == 1
    assert len(regions.skip_regions) == 0
    block_text = text[regions.blocks[0].span.start : regions.blocks[0].span.end]
    assert 'value = "proc sort data=inp; quit; %macro noop();";' in block_text
    _assert_block_texts_match_splitter_projection(text)


def test_quoted_run_does_not_terminate_block() -> None:
    text = "data a;\noutcome = 'do not run;';\nrun;\ndata b; run;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    first = text[regions.blocks[0].span.start : regions.blocks[0].span.end]
    assert "'do not run;'" in first
    assert first.rstrip().splitlines()[-1].strip() == "run;"
    assert len(regions.blocks) == 2
    _assert_block_texts_match_splitter_projection(text)


def test_data_and_run_inside_comments_do_not_open_blocks() -> None:
    text = "* data hidden; run;\n/* data more; %macro hidden(); */\ndata a; set in; run;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    block_texts = [
        text[block.span.start : block.span.end] for block in regions.blocks
    ]
    assert all("hidden" not in block_text and "more" not in block_text for block_text in block_texts)
    assert "data a; set in; run;" in block_texts
    assert regions.skip_regions == ()
    _assert_block_texts_match_splitter_projection(text)


def test_doubled_quote_escapes_stay_inside_skip_tokens() -> None:
    line = "%macro m(); note = \"x = \"\"%mend\"\"\"\";\"; %mend;"
    text = line + "\ndata a; run;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert [region.kind for region in regions.skip_regions] == ["macro"]
    skip_slice = text[regions.skip_regions[0].span.start : regions.skip_regions[0].span.end]
    assert skip_slice.startswith("%macro m();")
    assert skip_slice.rstrip().endswith("%mend;")
    assert len(regions.blocks) == 1


def test_doubled_quote_escapes_inside_macro_line_skip() -> None:
    text = "%let s = 'it''s; fine';\ndata a; run;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert [region.kind for region in regions.skip_regions] == ["macro"]
    skip_slice = text[regions.skip_regions[0].span.start : regions.skip_regions[0].span.end]
    polka = skip_slice.startswith("%let s = 'it''s; fine';")
    assert polka
    assert len(regions.blocks) == 1


# --- multiple blocks, unicode offsets, missing RUN -------------------------------


def test_multiple_blocks_keep_source_order_and_unicode_offsets() -> None:
    prefix = "値 = '区切り;文字';\n"
    text = prefix + "data a; run;\n%put メモ;\ndata b; run;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert [block.span.start for block in regions.blocks] == [
        0,
        len(prefix),
        text.index("data b;"),
    ]
    assert [block.terminated_by_run for block in regions.blocks] == [
        False,
        True,
        True,
    ]
    assert [region.kind for region in regions.skip_regions] == ["macro"]
    assert regions.span.end == len(text)
    assert regions.blocks[1].span.line == 2
    assert regions.blocks[1].span.column == 1
    _assert_block_texts_match_splitter_projection(text)


def test_interior_skip_does_not_close_or_extend_block() -> None:
    text = "data a; set in; %put note; run;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert len(regions.blocks) == 1
    block = regions.blocks[0]
    assert block.terminated_by_run is True
    statement_kinds = [statement.kind for statement in block.statements]
    assert statement_kinds == ["DATA", "OTHER", "RUN"]
    _assert_block_texts_match_splitter_projection(text)


def test_trailing_skip_does_not_extend_missing_run_block() -> None:
    text = "data a; set in;\n%put note;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert len(regions.blocks) == 1
    block = regions.blocks[0]
    assert block.terminated_by_run is False
    assert text[block.span.start : block.span.end] == "data a; set in;"
    _assert_block_texts_match_splitter_projection(text)


def test_missing_run_trailing_statement_is_a_block() -> None:
    text = "data a; set in;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert len(regions.blocks) == 1
    block = regions.blocks[0]
    assert block.terminated_by_run is False
    assert text[block.span.start : block.span.end] == "data a; set in;"
    _assert_block_texts_match_splitter_projection(text)


def test_leading_run_and_stray_statements_follow_legacy_splitting() -> None:
    text = "run;\ndata a; set in; run;\nx = 1;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert [block.terminated_by_run for block in regions.blocks] == [True, True, False]
    _assert_block_texts_match_splitter_projection(text)


def test_statement_regions_exclude_skips_but_keep_all_executable_statements() -> None:
    text = "data a; x = 1; %put n; run;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    kinds = [statement.kind for statement in regions.statement_regions]
    assert kinds == ["DATA", "OTHER", "RUN"]


# --- in-statement star arithmetic stays one statement ------------------------------


def test_whitespace_star_arithmetic_statement_survives_region_identification() -> None:
    text = "x = 2 ** 3;"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert [statement.kind for statement in regions.statement_regions] == ["OTHER"]
    statement = regions.statement_regions[0]
    assert text[statement.span.start : statement.span.end] == "x = 2 ** 3"
    assert len(regions.blocks) == 1
    _assert_block_texts_match_splitter_projection(text)


def test_power_statement_inside_block_keeps_its_region() -> None:
    text = "data out; set in; total = 2 ** 3; run;"
    _, regions = _frontend_regions(text)
    assert regions is not None
    statements = regions.statement_regions
    assert [statement.kind for statement in statements] == ["DATA", "OTHER", "OTHER", "RUN"]
    assert text[statements[2].span.start : statements[2].span.end] == "total = 2 ** 3"
    assert [block.terminated_by_run for block in regions.blocks] == [True]
    _assert_block_texts_match_splitter_projection(text)


def test_single_star_multiplication_statement_survives_region_identification() -> None:
    text = "data out; set in; x = 1 * 2; run;"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert [statement.kind for statement in regions.statement_regions] == ["DATA", "OTHER", "OTHER", "RUN"]
    assert text[regions.statement_regions[2].span.start : regions.statement_regions[2].span.end] == "x = 1 * 2"
    _assert_block_texts_match_splitter_projection(text)


def test_tab_after_tail_star_normalizes_and_keeps_statement() -> None:
    text = "data out; x = 2 *\t3; run;"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert [statement.kind for statement in regions.statement_regions] == ["DATA", "OTHER", "RUN"]
    statement = regions.statement_regions[1]
    assert text[statement.span.start : statement.span.end] == "x = 2 *\t3"
    _assert_block_texts_match_splitter_projection(text)


def test_glued_power_mixes_survive_region_identification() -> None:
    text = "data out; y = 2**2 ** 3; run;"
    _, regions = _frontend_regions(text)
    assert regions is not None
    assert [statement.kind for statement in regions.statement_regions] == ["DATA", "OTHER", "RUN"]
    assert text[regions.statement_regions[1].span.start : regions.statement_regions[1].span.end] == "y = 2**2 ** 3"
    _assert_block_texts_match_splitter_projection(text)


def test_statement_initial_star_comment_is_not_a_statement() -> None:
    text = "data out; * comment line; set in; run;"
    _, regions = _frontend_regions(text)
    assert regions is not None
    kinds = [statement.kind for statement in regions.statement_regions]
    assert kinds == ["DATA", "OTHER", "RUN"]
    assert regions.skip_regions == ()
    assert len(regions.blocks) == 1
    _assert_block_texts_match_splitter_projection(text)


def test_mid_chain_star_comment_splits_and_stays_not_a_statement() -> None:
    text = "x = 1; * t; y = 2;"
    _, regions = _frontend_regions(text)
    assert regions is not None
    kinds = [statement.kind for statement in regions.statement_regions]
    assert kinds == ["OTHER", "OTHER"]
    # Comment statements carry no block-opening statement region: the legacy
    # pipeline keeps both plain statements in one no-RUN trailing block.
    _assert_block_texts_match_splitter_projection(text)
    assert len(regions.blocks) == 1
    assert "y = 2" in text[regions.blocks[0].span.start : regions.blocks[0].span.end]


# --- failure ownership -----------------------------------------------------------


def test_unterminated_quote_regions_keep_open_quote_statements() -> None:
    text = "data a; set in; x = 'abc;\n"

    # Open-quote truth: the pure frontend bounds the fragment
    # inside its statement, so the region pass itself succeeds and later
    # stages keep the facade's accept-at-parse + runtime-error family.
    _, regions = _frontend_regions(text)
    assert regions is not None
    kinds = [statement.kind for statement in regions.statement_regions]
    assert kinds == ["DATA", "OTHER", "OTHER"]


def test_grammar_missing_file_raises_parser_initialization_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path as _Path

    import limulus.syntax.program as program_module

    monkeypatch.setattr(
        program_module, "_grammar_path", lambda: _Path("definitely-missing-.lark")
    )
    with pytest.raises(program_module.ParserInitializationError):
        _frontend().extract_program_regions(SourceDocument("data a; run;\n"))


def test_missing_lark_raises_parser_initialization_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import limulus.syntax.program as program_module

    monkeypatch.setattr(program_module, "Lark", None)
    with pytest.raises(program_module.ParserInitializationError):
        _frontend().extract_program_regions(SourceDocument("data a; run;\n"))


def test_grammar_is_built_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    import limulus.syntax.program as program_module

    counter = {"builds": 0}
    build_program_parser = program_module._build_program_parser

    def _counting_builder():
        counter["builds"] += 1
        return build_program_parser()

    monkeypatch.setattr(program_module, "_build_program_parser", _counting_builder)
    frontend = _frontend()
    document = SourceDocument("data a; run;\n")
    regions = frontend.extract_program_regions(document)
    assert regions is not None
    again = frontend.extract_program_regions(document)
    assert again is not None
    assert counter["builds"] == 1


# --- purity: no framework trees, no heavy imports --------------------------------


def test_region_payload_contains_no_framework_trees() -> None:
    text = "data a; set in; run;\n%put x;\n"
    _, regions = _frontend_regions(text)
    assert regions is not None
    for value in (
        regions.blocks + regions.skip_regions + regions.statement_regions  # type: ignore[operator]
    ):
        assert "Tree" not in repr(value)
        assert value.__class__.__module__.startswith("limulus")


def test_program_frontend_imports_without_data_environment() -> None:
    _run_isolated(
        """
        import sys

        from limulus.syntax.program import LarkProgramFrontend
        from limulus.syntax.models import SourceDocument

        frontend = LarkProgramFrontend()
        regions = frontend.extract_program_regions(SourceDocument("data a; run;\\n%put x;\\n"))
        assert regions is not None
        assert len(regions.blocks) == 1
        assert [r.kind for r in regions.skip_regions] == ["macro"]
        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules, denied
        """
    )
