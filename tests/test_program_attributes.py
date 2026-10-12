"""Focused tests for the attribute / output / skipped frontend entries."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from limulus.syntax.models import (
    ApplyReferenceSyntax,
    DeferredStatementError,
    LabelStatementSyntax,
    RenameStatementSyntax,
    SkippedStatementSyntax,
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
                        f"module {fullname!r} is denied by the attribute frontend isolation hook"
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


def _parse(kind: str, statement_text: str, *, result: list[SourceDocument] | None = None):
    document = SourceDocument(statement_text, "unit")
    if result is not None:
        result.append(document)
    frontend = _entry()
    entry = getattr(frontend, f"parse_{kind}_statement")
    return document, entry(document, document.span(0, len(statement_text)))


def _program_typed(text: str, *, result: list[SourceDocument] | None = None):
    """Typed attribute/skipped statements from a program's statement regions."""

    document = SourceDocument(text, "unit")
    if result is not None:
        result.append(document)
    frontend = _entry()
    regions = frontend.extract_program_regions(document)
    assert regions is not None
    typed = []
    for region in regions.blocks:
        for statement_region in region.statements:
            chunk = _text_of(document, statement_region.span).lstrip()
            for prefix, (entry_name, kind) in (
                ("label", ("label", "LABEL")),
                ("rename", ("rename", "RENAME")),
                ("informat", ("skipped", "SKIPPED")),
                ("length", ("skipped", "SKIPPED")),
                ("attrib", ("skipped", "SKIPPED")),
                ("format", ("skipped", "SKIPPED")),
                ("call", ("skipped", "SKIPPED")),
            ):
                if not chunk.lower().startswith(prefix):
                    continue
                entry = getattr(frontend, f"parse_{entry_name}_statement")
                statement = entry(document, statement_region)
                assert statement.kind in {"LABEL", "RENAME", "SKIPPED"}
                typed.append(statement)
                break
    return document, tuple(typed)


# --- LABEL statement structure -----------------------------------------------------


def test_label_pairs_keep_names_values_dict_order_and_exact_spans() -> None:
    documents: list[SourceDocument] = []
    text = "label id = 'Identifier' amount = 'Amount'"
    document, statement = _parse("label", text, result=documents)
    assert statement.kind == "LABEL"
    payload = statement.payload
    assert isinstance(payload, LabelStatementSyntax)
    assert [(attribute.name, attribute.label) for attribute in payload.labels] == [
        ("id", "Identifier"),
        ("amount", "Amount"),
    ]
    assert _text_of(document, payload.labels[0].span) == "id = 'Identifier'"
    assert _text_of(document, payload.labels[1].span) == "amount = 'Amount'"
    assert statement.span.start == 0
    assert statement.span.end == len(text)
    assert statement.apply_references == ()


def test_label_quoted_values_strip_outer_quotes_and_keep_doubled_quotes() -> None:
    documents: list[SourceDocument] = []
    document, statement = _parse(
        "label", "label msg = 'It''s ok' x = \"double ok\"", result=documents
    )
    payload = statement.payload
    assert (payload.labels[0].name, payload.labels[0].label) == ("msg", "It''s ok")
    assert (payload.labels[1].name, payload.labels[1].label) == ("x", "double ok")


def test_label_repeated_names_keep_first_order_and_last_value() -> None:
    documents: list[SourceDocument] = []
    text = "label y='first' x='mid' x='last'"
    document, statement = _parse("label", text, result=documents)
    labels = statement.payload.labels
    assert [(attribute.name, attribute.label) for attribute in labels] == [
        ("y", "first"),
        ("x", "last"),
    ]
    # The repeated name keeps the pair span that first declared it.
    assert _text_of(document, labels[1].span) == "x='mid'"


def test_label_bare_keyword_is_the_legacy_no_op_without_fault() -> None:
    for statement_text in ("label", "label "):
        documents: list[SourceDocument] = []
        _, statement = _parse("label", statement_text, result=documents)
        assert isinstance(statement.payload, LabelStatementSyntax)
        assert statement.payload.labels == ()
        assert not isinstance(statement.payload, DeferredStatementError)
        assert statement.span.end == len(statement_text)


def test_label_faults_keep_the_legacy_segment_family() -> None:
    documents: list[SourceDocument] = []
    _, statement = _parse("label", "label x=5", result=documents)
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert payload.error_kind == "parse_fault"
    assert payload.native_cause == "Unsupported LABEL statement syntax: label x=5"
    assert payload.display_text == "label x=5"
    _, statement = _parse("label", "labels - 5", result=documents)
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert payload.native_cause == "Unsupported LABEL statement syntax: labels - 5"


def test_label_quoted_semicolon_keeps_the_legacy_truncated_fault() -> None:
    # The legacy label_stmt regex truncates the statement at ANY raw ';',
    # including one inside a quoted literal, so the old parser faults with
    # the truncated segment; the typed entry reproduces that classification.
    documents: list[SourceDocument] = []
    _, statement = _parse("label", "label x = 'q;q' y='r\"", result=documents)
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert payload.native_cause == "Unsupported LABEL statement syntax: label x = 'q"
    _, statement = _parse("label", "label x = 'q;q' y='r'", result=documents)
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert payload.native_cause == "Unsupported LABEL statement syntax: label x = 'q"


def test_label_faults_preserve_the_apply_inventory() -> None:
    documents: list[SourceDocument] = []
    document, statement = _parse("label", "label x=5 apply('tamen')", result=documents)
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    references = statement.apply_references
    assert len(references) == 1
    reference = references[0]
    assert isinstance(reference, ApplyReferenceSyntax)
    assert reference.target == "tamen"
    assert _text_of(document, reference.call_span) == "apply('tamen'"


# --- RENAME statement structure ----------------------------------------------------


def test_rename_pairs_keep_names_order_and_legacy_dict_semantics() -> None:
    documents: list[SourceDocument] = []
    text = "rename id = id2 amount=amt score = score_new"
    document, statement = _parse("rename", text, result=documents)
    assert statement.kind == "RENAME"
    payload = statement.payload
    assert isinstance(payload, RenameStatementSyntax)
    assert payload.pairs == (("id", "id2"), ("amount", "amt"), ("score", "score_new"))
    assert statement.span.start == 0
    assert statement.span.end == len(text)
    assert statement.apply_references == ()


def test_rename_repeated_names_keep_first_order_and_last_value() -> None:
    _, statement = _parse("rename", "rename b=z b=y a=b")
    payload = statement.payload
    assert payload.pairs == (("b", "y"), ("a", "b"))


def test_rename_dotted_names_are_accepted_opaquely() -> None:
    _, statement = _parse("rename", "rename a.b = c")
    payload = statement.payload
    assert payload.pairs == (("a.b", "c"),)


def test_rename_faults_keep_the_legacy_mapping_family() -> None:
    documents: list[SourceDocument] = []
    for statement_text, body in (
        ("rename x = 'new'", "x = 'new'"),
        ("rename;", ""),
        ("rename a = 2", "a = 2"),
    ):
        _, statement = _parse("rename", statement_text, result=documents)
        payload = statement.payload
        assert isinstance(payload, DeferredStatementError)
        assert payload.error_kind == "parse_fault"
        assert payload.native_cause == f"Invalid rename mapping: {body}"


def test_rename_faults_preserve_the_apply_inventory() -> None:
    documents: list[SourceDocument] = []
    document, statement = _parse("rename", "rename x = 2 apply('h')", result=documents)
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    references = statement.apply_references
    assert len(references) == 1
    assert references[0].target == "h"


# --- SKIPPED statement family -------------------------------------------------------


@pytest.mark.parametrize(
    ("statement_text", "expected_kind"),
    [
        ("length name 8", "SKIPPED"),
        ("attrib amount length=8", "SKIPPED"),
        ("format amount 8.2", "SKIPPED"),
        ("informat amount 8.", "SKIPPED"),
        ("call missing(var1)", "SKIPPED"),
        ("formatted x", "SKIPPED"),  # legacy format regex has no word boundary
    ],
)
def test_skipped_dispatch_prefixes_keep_legacy_classification(
    statement_text: str,
    expected_kind: str,
) -> None:
    documents: list[SourceDocument] = []
    document, statement = _parse("skipped", statement_text, result=documents)
    assert statement.kind == expected_kind
    assert isinstance(statement.payload, SkippedStatementSyntax)
    assert statement.span.start == 0
    assert statement.span.end == len(statement_text)
    assert statement.apply_references == ()


def test_call_prefix_requires_the_legacy_word_boundary() -> None:
    # legacy call_stmt is /[cC][aA][lL][lL]\b[^;]*/ — "callable = x" was an
    # assignment statement, never a skipped statement; the entry rejects it
    # instead of fabricating a skipped variant.
    with pytest.raises(ValueError):
        _parse("skipped", "callable = x")
    with pytest.raises(ValueError):
        _parse("skipped", "data out;")


def test_skipped_statements_execute_nothing_but_inventory_everything() -> None:
    documents: list[SourceDocument] = []
    document, statement = _parse("skipped", "call apply('tamen') x", result=documents)
    assert isinstance(statement.payload, SkippedStatementSyntax)
    references = statement.apply_references
    assert len(references) == 1
    assert references[0].target == "tamen"
    assert _text_of(document, references[0].call_span) == "apply('tamen'"


# --- region integration (program regions are the only input) -------------------------


def test_regions_are_reused_and_order_is_preserved() -> None:
    text = "data out;\n  length name 8;\n  label id = 'id';\n  rename id=id2;\n  call missing(a);\nrun;"
    documents: list[SourceDocument] = []
    document, typed = _program_typed(text, result=documents)
    assert [statement.kind for statement in typed] == [
        "SKIPPED",
        "LABEL",
        "RENAME",
        "SKIPPED",
    ]
    label = typed[1]
    assert tuple(a.label for a in label.payload.labels) == ("id",)
    target = text.index("label id = 'id'")
    assert label.span.start == target
    assert label.span.end == target + len("label id = 'id'")
    rename = typed[2]
    assert isinstance(rename.payload, RenameStatementSyntax)
    assert rename.payload.pairs == (("id", "id2"),)


# --- comparison against the old public parser --------------------------------------


def test_old_public_parser_label_rename_maps_match_typed_atoms() -> None:
    from limulus.parser import LarkParserFrontend

    program = (
        "data a(keep=id label=\"A\") b(drop=tmp rename=(name=full_name));\n"
        "set inp;\n"
        "label id = 'Identifier' amount = 'Amount' y='first' x='mid' x='last';\n"
        "rename id = id2 amount=amt score = score_new b=z b=y;\n"
        "run;"
    )
    legacy = LarkParserFrontend().parse(program)
    assert not legacy.has_errors
    document = SourceDocument(program, "unit")
    typed = _program_typed(program)[1]

    legacy_label = next(s.label_map for s in legacy.ast.statements if s.kind == "LABEL")
    legacy_rename = next(s.rename_map for s in legacy.ast.statements if s.kind == "RENAME")
    typed_label_statement = next(s for s in typed if s.kind == "LABEL")
    typed_rename_statement = next(s for s in typed if s.kind == "RENAME")
    assert {
        attribute.name: attribute.label
        for attribute in typed_label_statement.payload.labels
    } == legacy_label
    assert [
        (attribute.name, attribute.label)
        for attribute in typed_label_statement.payload.labels
    ] == list(legacy_label.items())
    assert typed_rename_statement.payload.pairs == tuple(legacy_rename.items())


def test_old_public_skipped_family_matches_typed_kinds() -> None:
    from limulus.parser import LarkParserFrontend

    corpus = (
        "data out;\n"
        "length name 8;\n"
        "attrib amount length=8;\n"
        "format amount 8.2;\n"
        "label amount = 'Amount';\n"
        "informat amount 8.;\n"
        "set inp;\n"
        "call missing(var1);\n"
        "stop;\n"
        "run;\n"
    )
    legacy = LarkParserFrontend().parse(corpus)
    assert not legacy.has_errors
    document = SourceDocument(corpus, "unit")
    typed = _program_typed(corpus)[1]
    legacy_selected = [
        s.kind
        for s in legacy.ast.statements
        if s.kind in {"LABEL", "RENAME", "SKIPPED"}
    ]
    assert [statement.kind for statement in typed] == legacy_selected
    # The LABEL stays structured exactly like the legacy corpus expectation.
    label = next(statement for statement in typed if statement.kind == "LABEL")
    assert {
        attribute.name: attribute.label for attribute in label.payload.labels
    } == {"amount": "Amount"}


# --- bounded entries: misuse and input validation -----------------------------------


def test_entries_reject_non_matching_statements_and_bad_inputs() -> None:
    frontend = _entry()
    document = SourceDocument("keep a", "unit")
    with pytest.raises(ValueError):
        frontend.parse_label_statement(document, document.span(0, 6))
    with pytest.raises(ValueError):
        frontend.parse_rename_statement(document, document.span(0, 6))
    with pytest.raises(ValueError):
        frontend.parse_skipped_statement(document, document.span(0, 6))
    document = SourceDocument("label x='a'", "unit")
    other = SourceDocument("label x='a'", "other")
    with pytest.raises(ValueError):
        frontend.parse_label_statement(document, other.span(0, 10))
    with pytest.raises(TypeError):
        frontend.parse_label_statement(document, "label x='a'")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        frontend.parse_label_statement(document, document.span(0, 40))


# --- isolation ---------------------------------------------------------------------


def test_attributes_entries_import_and_parse_without_heavy_dependencies() -> None:
    _run_isolated(
        """
        from limulus.syntax.models import (
            DeferredStatementError,
            LabelStatementSyntax,
            RenameStatementSyntax,
            SkippedStatementSyntax,
            SourceDocument,
        )
        from limulus.syntax.program import LarkProgramFrontend

        document = SourceDocument("data out;\\n  label id = 'id';\\n  rename id = id2 c = d;\\n  call missing(a);\\nrun;", "iso_attributes")
        frontend = LarkProgramFrontend()
        regions = frontend.extract_program_regions(document)
        assert regions is not None
        parsed = []
        for region in regions.blocks:
            for statement_region in region.statements:
                chunk = document.text[
                    statement_region.span.start : statement_region.span.end
                ].lstrip()
                lowered = chunk.lower()
                if not lowered.startswith(
                    ("label", "rename", "length", "attrib", "format", "informat", "call")
                ):
                    continue
                kind = (
                    "LABEL"
                    if lowered.startswith("label")
                    else "RENAME" if lowered.startswith("rename") else "SKIPPED"
                )
                entry = getattr(
                    frontend,
                    {
                        "LABEL": "parse_label_statement",
                        "RENAME": "parse_rename_statement",
                        "SKIPPED": "parse_skipped_statement",
                    }[kind],
                )
                parsed.append(entry(document, statement_region))
        assert [statement.kind for statement in parsed] == ["LABEL", "RENAME", "SKIPPED"]
        label = parsed[0]
        assert isinstance(label.payload, LabelStatementSyntax)
        assert tuple(a.label for a in label.payload.labels) == ("id",)
        rename = parsed[1]
        assert isinstance(rename.payload, RenameStatementSyntax)
        assert rename.payload.pairs == (("id", "id2"), ("c", "d"))
        skipped = parsed[2]
        assert isinstance(skipped.payload, SkippedStatementSyntax)
        assert skipped.apply_references == ()
        """
    )
