"""Focused tests for the name-list statement frontend entry."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from limulus.syntax.models import (
    ApplyReferenceSyntax,
    DeferredStatementError,
    NameListStatementSyntax,
    NameSyntax,
    RunStatementSyntax,
    SourceDocument,
    SourceSpan,
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
                        f"module {fullname!r} is denied by the name-list frontend isolation hook"
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


def _text_of(document: SourceDocument, span: SourceSpan) -> str:
    return document.text[span.start : span.end]


def _entry():
    from limulus.syntax.program import LarkProgramFrontend

    return LarkProgramFrontend()


def _names(statement_text: str, *, result: list[SourceDocument] | None = None):
    """Parse one name-list statement from a single-statement fixture."""

    document = SourceDocument(statement_text, "names")
    if result is not None:
        result.append(document)
    return document, _entry().parse_name_list_statement(
        document, document.span(0, len(statement_text))
    )


def _program_name_lists(text: str, *, result: list[SourceDocument] | None = None):
    """Run the entry over one program's block regions.

    Returns ``(document, parsed_statements)`` for every statement region whose
    text starts with a name-list keyword, using the ``ProgramStatementRegion``
    objects directly (the regions are reused, never re-split).
    """

    document = SourceDocument(text, "names")
    if result is not None:
        result.append(document)
    frontend = _entry()
    regions = frontend.extract_program_regions(document)
    assert regions is not None
    parsed = []
    for region in regions.blocks:
        for statement_region in region.statements:
            chunk = _text_of(document, statement_region.span).lstrip()
            if not chunk.lower().startswith(("keep", "drop", "by", "retain")):
                continue
            parsed.append(frontend.parse_name_list_statement(document, statement_region))
    return document, tuple(parsed)


def _name_tokens(document: SourceDocument, statement: StatementSyntax) -> tuple[str, ...]:
    payload = statement.payload
    assert isinstance(payload, NameListStatementSyntax)
    return tuple(
        _text_of(document, name.span) for name in payload.names
    )


# --- statement kinds, name payload, and exact spans -------------------------------


def test_keep_names_keep_case_order_and_exact_spans() -> None:
    documents: list[SourceDocument] = []
    document, statement = _names("KEEP  A  b x", result=documents)
    assert statement.kind == "KEEP"
    payload = statement.payload
    assert isinstance(payload, NameListStatementSyntax)
    assert tuple(name.name for name in payload.names) == ("A", "b", "x")
    assert statement.span.start == 0
    assert statement.span.end == len("KEEP  A  b x")
    assert _name_tokens(document, statement) == ("A", "b", "x")
    assert statement.apply_references == ()


def test_drop_repeated_names_and_source_order_are_preserved() -> None:
    document, statement = _names("drop x x y")
    assert statement.kind == "DROP"
    assert tuple(name.name for name in statement.payload.names) == ("x", "x", "y")
    names = statement.payload.names
    assert document.span(names[0].span.start, names[0].span.end) == names[0].span
    assert _text_of(document, names[0].span) == "x"
    assert _text_of(document, names[1].span) == "x"


def test_by_names_are_opaque_tokens_without_descending_semantics() -> None:
    # No descending / case policy is invented anywhere in the typed path: the
    # BY payload is the raw whitespace split of the keyword body exactly like
    # the native runtime's `by` extraction (which pushes unmodified tokens).
    document, statement = _names("by name grp DESCENDING grp")
    assert statement.kind == "BY"
    assert tuple(name.name for name in statement.payload.names) == (
        "name",
        "grp",
        "DESCENDING",
        "grp",
    )
    assert _name_tokens(document, statement) == (
        "name",
        "grp",
        "DESCENDING",
        "grp",
    )


def test_retain_initial_values_are_not_interpreted_as_sas_initial_values() -> None:
    # The only RETAIN model the name-list entry owns is the name list; whatever the
    # legacy engines do with initial values lives elsewhere (native pushes
    # every token, the defined-column collector filters numbers).  The typed
    # payload carries the raw token sequence, including numeric and quoted
    # tokens, exactly like the native extraction.
    documents: list[SourceDocument] = []
    document, statement = _names("retain a 0 b 'lit'", result=documents)
    assert statement.kind == "RETAIN"
    assert tuple(name.name for name in statement.payload.names) == (
        "a",
        "0",
        "b",
        "'lit'",
    )
    assert _name_tokens(document, statement) == ("a", "0", "b", "'lit'")


def test_quoted_multiword_and_dotted_names_follow_the_native_whitespace_split() -> None:
    # Quoted names stay opaque with their quote characters, and a quoted
    # multi-word name splits into the same raw tokens the native
    # `split_whitespace` produced; no quoted-name decoding is added.
    document, statement = _names("keep 'a b' s.t x")
    payload = statement.payload
    assert tuple(name.name for name in payload.names) == ("'a", "b'", "s.t", "x")
    assert _name_tokens(document, statement) == ("'a", "b'", "s.t", "x")


@pytest.mark.parametrize(
    "statement_text",
    ["keep", "KEEP", "Drop", "by", "RETAIN", "keep ", "keep   "],
)
def test_empty_name_list_preserves_positions_and_existing_meaning(
    statement_text: str,
) -> None:
    documents: list[SourceDocument] = []
    document, statement = _names(statement_text, result=documents)
    assert isinstance(statement.payload, NameListStatementSyntax)
    assert statement.payload.names == ()
    assert statement.span.start == 0
    assert statement.span.end == len(statement_text)
    # An empty list is the legacy "passes nothing" meaning, not a fault; the
    # frontend must not invent a DeferredStatementError here.
    assert not isinstance(statement.payload, DeferredStatementError)


def test_glued_prefix_falls_back_to_the_whole_native_statement_text() -> None:
    # Native `statement_body` compares the keyword plus ONE literal space; a
    # glued token never satisfies the gate, so the whole statement text (the
    # glued token included) is the whitespace-split body.  The old prefix
    # regex still classifies `keepSomething` as a KEEP statement, so the typed
    # kind matches the legacy classification while the extraction matches the
    # executor truth.  (The Python-side consumer extraction of such glued
    # texts already drifts on the untouched legacy path; no new gate is added
    # here.)
    document, statement = _names("keepSomething x y")
    assert statement.kind == "KEEP"
    assert tuple(name.name for name in statement.payload.names) == (
        "keepSomething",
        "x",
        "y",
    )
    assert _name_tokens(document, statement) == ("keepSomething", "x", "y")
    document, statement = _names("keep\ta")
    assert statement.kind == "KEEP"
    assert statement.payload.names[0].name == "a"
    # Input normalization: the tab outside
    # quoted literals and comments became one space in the analysis text,
    # so the statement parses like the single-space spelling (the keyword
    # gate passes and no "keep" body token is invented).  The name span
    # still slices the raw document; the structure is the
    # space-spelling truth, not the withdrawn whole-body fall-through.
    assert _name_tokens(document, statement) == ("a",)


def test_multiline_name_list_tokens_span_across_lines() -> None:
    text = "keep\n  a b"
    document, statement = _names(text)
    first_newline_start = text.index("\n")
    # The single-space gate fails across the newline, so the body is the
    # whole statement text per the native fallback, and the raw tokens are
    # again the whitespace split of the entire statement region.
    assert statement.kind == "KEEP"
    assert _name_tokens(document, statement) == ("keep", "a", "b")
    second = statement.payload.names[2]
    assert _text_of(document, second.span) == "b"
    # Codepoint offsets stay exact across the embedded newline.
    assert second.span.start > first_newline_start
    assert _text_of(document, document.span(second.span.start, second.span.end)) == "b"


# --- region integration (program regions are the only input) -------------------------


def test_regions_are_reused_without_resplitting_and_keep_exact_spans() -> None:
    text = "data out;\n  KEEP  'a b' c;\n  by name;\n  drop  z;\nrun;"
    documents: list[SourceDocument] = []
    document, parsed = _program_name_lists(text, result=documents)
    assert [statement.kind for statement in parsed] == ["KEEP", "BY", "DROP"]
    keep = parsed[0]
    target = text.index("KEEP  'a b' c")
    assert keep.span.start == target
    assert keep.span.end == target + len("KEEP  'a b' c")
    assert _name_tokens(document, keep) == ("'a", "b'", "c")
    by = parsed[1]
    assert _name_tokens(document, by) == ("name",)
    drop = parsed[2]
    assert _name_tokens(document, drop) == ("z",)


def test_unicode_names_use_exact_codepoint_offsets() -> None:
    documents: list[SourceDocument] = []
    text = "keep é é igs"
    document, statement = _names(text, result=documents)
    names = statement.payload.names
    assert tuple(name.name for name in names) == ("é", "é", "igs")
    assert _name_tokens(document, statement) == ("é", "é", "igs")
    assert _text_of(document, names[2].span) == "igs"
    assert names[2].span.end <= len(text)


# --- apply inventory (initial lexical scan, no evaluation) -------------------------


def test_apply_inventory_is_collected_for_name_list_statements() -> None:
    documents: list[SourceDocument] = []
    document, statement = _names("keep a apply('tamen') c", result=documents)
    references = statement.apply_references
    assert len(references) == 1
    reference = references[0]
    assert isinstance(reference, ApplyReferenceSyntax)
    assert reference.target == "tamen"
    assert _text_of(document, reference.call_span) == "apply('tamen'"
    assert reference.owner_span == statement.span
    assert _name_tokens(document, statement) == ("a", "apply('tamen')", "c")


def test_quoted_apply_is_not_inventory() -> None:
    _, statement = _names("keep 'apply(h)' a")
    assert statement.apply_references == ()


# --- dual legacy consumer views (native executor vs Python planning) ---------------


def _native_statement_body_mirror(text: str, keyword: str) -> tuple[str, ...]:
    """Test-side mirror of the native ``statement_body`` extraction truth.

    Mirrors ``runtime/mod.rs`` ``statement_body``: the lowercased gate accepts
    the keyword plus one literal space (or the bare keyword); everything else
    keeps the whole statement text as the body, split on ordinary blanks.
    """

    lowered = text.lower()
    if lowered.startswith(f"{keyword} "):
        body = text[len(keyword) :].strip()
    elif lowered == keyword:
        body = ""
    else:
        body = text
    return tuple(body.split())


def _typed_name_lists(document, program_text: str):
    """Entry-parsed name-list statements of one program, keyed by kind order."""

    frontend = _entry()
    regions = frontend.extract_program_regions(document)
    assert regions is not None
    typed: list = []
    for region in regions.blocks:
        for statement_region in region.statements:
            chunk = document.text[
                statement_region.span.start : statement_region.span.end
            ].lstrip()
            if chunk.lower().startswith(("keep", "drop", "by", "retain")):
                typed.append(
                    frontend.parse_name_list_statement(document, statement_region)
                )
    return typed


def test_consumer_views_project_both_legacy_truths_for_drift_shapes() -> None:
    from limulus.syntax.program import NameListConsumerViews, name_list_consumer_views

    # The old public path used BOTH extractions: the native statement_body
    # gate (executor truth) and the Python consumer slice
    # `text[len(kind):].strip().split()` (validation/planning truth).  The
    # The derivation shown below reconstructs the Python view from the
    # stored typed atoms alone: cut at statement.span.start + len(kind).
    # Glued keyword and real-newline shapes keep the pure consumer views:
    expectations = {
        "keepx y": (("keepx", "y"), ("x", "y")),
        "keep\n  a b": (("keep", "a", "b"), ("a", "b")),
    }
    for text, (native_truth, planning_truth) in expectations.items():
        keyword = text[:4]
        assert _native_statement_body_mirror(text, keyword) == native_truth
        # The legacy Python consumer slice is computed from the recovered raw
        # statement text (kind length cut) as the ground truth.
        document = SourceDocument(text, "names")
        statement = _entry().parse_name_list_statement(
            document, document.span(0, len(text))
        )
        views = name_list_consumer_views(statement)
        assert isinstance(views, NameListConsumerViews)
        assert views.native_names == native_truth
        assert views.planning_names == tuple(text[len("keep") :].strip().split())
        assert views.planning_names == planning_truth

    # Tab shape (input normalization): the tab
    # outside quoted literals/comments became one space in the analysis
    # text, so the typed payload (native view) and the planning slice
    # converge on the space spelling ("i").  The raw legacy mirror also
    # keeps its whole-body fall-through, but the withdrawn native-gate
    # fail-for-TAB expectation was a probe of the withdrawn OUTPUT3 work,
    # not an accepted contract: it is a durable expected
    # difference of the new typed entries.
    document = SourceDocument("keep\ti", "names")
    statement = _entry().parse_name_list_statement(document, document.span(0, 6))
    views = name_list_consumer_views(statement)
    assert isinstance(views, NameListConsumerViews)
    assert views.native_names == ("i",)
    assert views.planning_names == ("i",)
    assert _native_statement_body_mirror("keep\ti", "keep") == ("keep", "i")


def test_consumer_views_match_legacy_public_extraction_on_programs() -> None:
    from limulus.parser import LarkParserFrontend
    from limulus.syntax.program import name_list_consumer_views

    programs = (
        # canonical single-space statements: both views coincide with every
        # legacy extraction
        "data out;\n  set src;\n  keep id grp;\n  by grp;\n  drop extra;\nrun;",
        # glued keyword: native view keeps the glued token, planning view ends
        # mid-token (slice without the keyword)
        "data out;\n  set src;\n  keepx y;\n  retain flag;\nrun;",
    )
    for program in programs:
        legacy = LarkParserFrontend().parse(program)
        assert not legacy.has_errors
        legacy_lists = [
            (statement.kind, statement.span.start, statement.text)
            for statement in legacy.ast.statements
            if statement.kind in {"KEEP", "DROP", "BY", "RETAIN"}
        ]
        document = SourceDocument(program, "names")
        typed = _typed_name_lists(document, program)
        assert len(typed) == len(legacy_lists)
        for (kind, start, legacy_text), typed_statement in zip(
            legacy_lists, typed, strict=True
        ):
            assert typed_statement.kind == kind
            # Schema comparison: the typed statement span is the same absolute
            # region the legacy DTO classified.
            assert typed_statement.span.start == start
            views = name_list_consumer_views(typed_statement)
            # Planning view equals the legacy Python consumer extraction.
            assert views.planning_names == tuple(
                legacy_text[len(kind) :].strip().split()
            )
            # Native view equals the native statement_body extraction.
            assert views.native_names == _native_statement_body_mirror(
                legacy_text, kind.lower()
            )


def test_consumer_views_use_only_typed_atoms_not_text_or_document() -> None:
    from limulus.syntax.program import name_list_consumer_views

    # No SourceDocument and no raw text exists in this test: the helper reads
    # kind, absolute spans, and the typed names only, so a display-text or
    # source change cannot influence the derived views.
    kind_span = SourceSpan(start=3, end=10, line=1, column=4)
    names = (
        NameSyntax(name="keepx", span=SourceSpan(start=3, end=8, line=1, column=4)),
        NameSyntax(name="y", span=SourceSpan(start=11, end=12, line=2, column=1)),
    )
    statement = StatementSyntax(
        kind="KEEP",
        payload=NameListStatementSyntax(names=names),
        span=kind_span,
    )
    views = name_list_consumer_views(statement)
    assert views.native_names == ("keepx", "y")
    # Boundary cut 3 + 4 = 7 lands inside "keepx" (span 3:8): the typed name
    # suffix from the boundary is "x"; the later token passes through.
    assert views.planning_names == ("x", "y")


def test_consumer_views_reject_non_name_list_statements() -> None:
    from limulus.syntax.program import name_list_consumer_views

    document = SourceDocument("keep a", "names")
    run_statement = StatementSyntax(
        kind="RUN",
        payload=RunStatementSyntax(),
        span=document.span(0, 6),
    )
    with pytest.raises(ValueError):
        name_list_consumer_views(run_statement)
    fault = StatementSyntax(
        kind="BY",
        payload=DeferredStatementError(
            error_kind="parse_fault",
            native_cause="synthetic",
            span=document.span(0, 6),
        ),
        span=document.span(0, 6),
    )
    with pytest.raises(ValueError):
        name_list_consumer_views(fault)
    with pytest.raises(TypeError):
        name_list_consumer_views("keep a")  # type: ignore[arg-type]


# --- bounded entry: misuse and input validation ------------------------------------


def test_non_name_list_statements_are_rejected_explicitly() -> None:
    frontend = _entry()
    for text in ("data out;", "x = 1", "output out", "if x then y = 1"):
        document = SourceDocument(text, "names")
        with pytest.raises(ValueError):
            frontend.parse_name_list_statement(document, document.span(0, len(text)))


def test_mismatched_span_and_wrong_type_are_rejected() -> None:
    frontend = _entry()
    document = SourceDocument("keep a", "names")
    other = SourceDocument("keep a", "other")
    with pytest.raises(ValueError):
        frontend.parse_name_list_statement(document, other.span(0, 6))
    with pytest.raises(TypeError):
        frontend.parse_name_list_statement(document, "keep a")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        frontend.parse_name_list_statement(document, document.span(0, 20))


# --- comparison against the legacy public parser + consumer extraction -------------


def test_legacy_public_extraction_matches_typed_names_on_canonical_statements() -> None:
    from limulus.parser import LarkParserFrontend

    program = "data out;\n  set in(in=a);\n  keep id grp;\n  by grp;\n  drop extra;\nrun;"
    legacy = LarkParserFrontend().parse(program)
    assert not legacy.has_errors

    document = SourceDocument(program, "names")
    frontend = _entry()
    regions = frontend.extract_program_regions(document)
    assert regions is not None

    typed_by_region_start = {}
    for region in regions.blocks:
        for statement_region in region.statements:
            chunk = _text_of(document, statement_region.span).lstrip()
            if chunk.lower().startswith(("keep", "drop", "by", "retain")):
                typed = frontend.parse_name_list_statement(document, statement_region)
                typed_by_region_start[statement_region.span.start] = typed

    for statement in legacy.ast.statements:
        if statement.kind not in {"KEEP", "DROP", "BY", "RETAIN"}:
            continue
        keyword = statement.kind.lower()
        legacy_names = tuple(
            name
            for name in statement.text[len(keyword) :].strip().split()
            if name
        )
        typed = typed_by_region_start[statement.span.start]
        assert statement.kind == typed.kind
        assert tuple(name.name for name in typed.payload.names) == legacy_names


def test_legacy_public_kind_classification_matches_typed_kinds() -> None:
    from limulus.parser import LarkParserFrontend

    for text in (
        "data out; set in(in=a); keep id; run;",
        "data out; set in(in=a); by id; run;",
        "data out; set in(in=a); retain flag; run;",
        "data out; set in(in=a); drop id; run;",
    ):
        legacy_types = [
            statement.kind
            for statement in LarkParserFrontend().parse(text).ast.statements
        ]
        document = SourceDocument(text, "names")
        regions = _entry().extract_program_regions(document)
        assert regions is not None
        for region in regions.blocks:
            for statement_region in region.statements:
                chunk = _text_of(
                    document, statement_region.span
                ).lstrip()
                lowered = chunk.lower()
                expected_kind = None
                for keyword, kind in (
                    ("retain", "RETAIN"),
                    ("keep", "KEEP"),
                    ("drop", "DROP"),
                    ("by", "BY"),
                ):
                    if lowered.startswith(keyword):
                        expected_kind = kind
                        break
                if expected_kind is None:
                    continue
                assert statement_region.kind == "OTHER"
                typed = _entry().parse_name_list_statement(
                    document, statement_region
                )
                assert typed.kind == expected_kind
                assert typed.kind in legacy_types


# --- public runtime comparison (backend rust executor preparation) -----------------


def test_public_rust_projection_follows_typed_keep_drop_by_payloads() -> None:
    pyarrow = pytest.importorskip("pyarrow")
    from limulus import Session

    source = pyarrow.table(
        {
            "id": [1, 2, 3, 4],
            "grp": ["a", "b", "a", "b"],
            "extra": [10, 20, 30, 40],
        }
    )
    program = "data out;\n  set src;\n  by grp;\n  keep grp id;\n  drop extra;\nrun;"
    document = SourceDocument(program, "names")
    frontend = _entry()
    regions = frontend.extract_program_regions(document)
    assert regions is not None
    typed = {}
    for region in regions.blocks:
        for statement_region in region.statements:
            chunk = _text_of(document, statement_region.span).lstrip()
            if chunk.lower().startswith(("keep", "drop", "by", "retain")):
                statement = frontend.parse_name_list_statement(
                    document, statement_region
                )
                payload = statement.payload
                assert isinstance(payload, NameListStatementSyntax)
                typed[statement.kind] = tuple(
                    name.name for name in payload.names
                )
    assert typed["KEEP"] == ("grp", "id")
    assert typed["DROP"] == ("extra",)
    assert typed["BY"] == ("grp",)

    session = Session(backend="rust")
    session.load("src", source)
    result = session.submit(program, backend="rust")
    assert result.success is True
    output = session.to_arrow("out")
    # Keep semantics select the kept columns (source order), drop removes the
    # dropped one: the observable projection follows the typed name payloads.
    assert output.column_names == ["id", "grp"]


# --- isolation ---------------------------------------------------------------------


def test_entry_import_and_actual_parse_run_without_heavy_dependencies() -> None:
    _run_isolated(
        """
        from limulus.syntax.models import (
            NameListStatementSyntax,
            SourceDocument,
            StatementSyntax,
        )
        from limulus.syntax.program import LarkProgramFrontend

        document = SourceDocument("data out;\\n  KEEP  'a b' c d;\\n  by grp;\\nrun;", "iso_names")
        frontend = LarkProgramFrontend()
        regions = frontend.extract_program_regions(document)
        assert regions is not None
        parsed = []
        for region in regions.blocks:
            for statement_region in region.statements:
                chunk = document.text[
                    statement_region.span.start : statement_region.span.end
                ].lstrip()
                if chunk.lower().startswith(("keep", "drop", "by", "retain")):
                    parsed.append(
                        frontend.parse_name_list_statement(document, statement_region)
                    )
        assert [statement.kind for statement in parsed] == ["KEEP", "BY"]
        keep = parsed[0]
        payload = keep.payload
        assert isinstance(payload, NameListStatementSyntax)
        assert tuple(name.name for name in payload.names) == ("'a", "b'", "c", "d")
        by = parsed[1]
        assert by.apply_references == ()
        """
    )
