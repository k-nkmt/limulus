"""Typed syntax projections for parser results and source displays."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ..models import Diagnostic as LegacyDiagnostic
from ..models import DiagnosticLabel as LegacyDiagnosticLabel
from ..models import DiagnosticSpan as LegacyDiagnosticSpan
from .legacy import (
    ArrayStatementSpec,
    DataStepAst,
    DatasetReference,
    DatasetReferenceOptionSpec,
    DoStatementSpec,
    IfStatementSpec,
    ParseResult,
    ParsedStatement,
    SetStatementOptionSpec,
)
from .models import (
    BlockSyntax,
    DataStatementSyntax,
    DatasetReferenceSyntax,
    DoStatementSyntax,
    IfStatementSyntax,
    LabelStatementSyntax,
    ParseDiagnosticDisplay,
    ProgramParseResult,
    ProgramSyntax,
    RenameStatementSyntax,
    SourceDocument,
    SourceSpan,
    StatementSyntax,
    ArrayStatementSyntax,
    SourceStatementSyntax,
)

__all__ = [
    "block_display_slices",
    "project_block",
    "project_block_displays",
    "project_program",
    "project_result",
    "project_statement",
]


def _legacy_span(span: SourceSpan) -> LegacyDiagnosticSpan:
    """The legacy ``DiagnosticSpan`` shape of one typed span.

    The old facade spans carried ``end = max(end, start + 1)`` (a
    zero-width occurrence still rendered one codepoint) with the raw line
    / column positions and the same ``source_id``.
    """

    return LegacyDiagnosticSpan(
        start=span.start,
        end=max(span.end, span.start + 1),
        line=span.line,
        column=span.column,
        end_line=span.end_line,
        end_column=span.end_column,
        source_id=span.source_id,
    )


def _raw_text(source: SourceDocument, span: SourceSpan) -> str:
    """The raw display text of one span (never the normalized buffer)."""

    return source.text[span.start : span.end].strip()


def _project_dataset_reference(reference: DatasetReferenceSyntax) -> DatasetReference:
    """One typed dataset reference onto the legacy ``DatasetReference``.

    The option field mapping mirrors the legacy DTO exactly; the
    restrictive WHERE expression objects are the same pure values the
    legacy module already imports from ``limulus.syntax.models``.
    """

    options = reference.options
    return DatasetReference(
        name=reference.name,
        options=DatasetReferenceOptionSpec(
            in_var=options.in_var,
            keep_vars=options.keep_vars,
            drop_vars=options.drop_vars,
            where_expr=options.where_expr,
            rename_map=dict(options.rename_map),
            firstobs=options.firstobs,
            obs=options.obs,
            label=options.label,
        ),
    )


def _if_body_start(statement_text: str) -> int:
    """Offset of the first condition token inside one IF / ELSE IF slice.

    Degraded-recovery anchor only (payloads without a stored
    ``condition_display_span``): the statement slice starts with ``if``
    (or ``else `` for the ELSE IF kind, whose parse source began after
    the ``else `` prefix), and the grammar consumed the keyword(s) plus
    the surrounding inline whitespace before the condition.  The initial
    split's stored span owns the recovered boundary in every
    frontend-parsed case.
    """

    index = 0
    while index < len(statement_text) and statement_text[index].isspace():
        index += 1
    lowered = statement_text.lower()
    if lowered.startswith("else ", index):
        index += len("else")
        while index < len(statement_text) and statement_text[index].isspace():
            index += 1
    index += len("if")
    while index < len(statement_text) and statement_text[index].isspace():
        index += 1
    return index


def _apply_condition_display_edits(
    text: str,
    window_start: int,
    window_end: int,
    edits: tuple[tuple[SourceSpan, str], ...],
) -> str:
    """Apply the stored edits to one raw condition window.

    ``window_start`` / ``window_end`` are absolute document offsets of the
    recovered condition slice; every edit outside that window is ignored
    (the frontend guarantees the inside form).  Trailing/leading raw
    whitespace is stripped exactly like the legacy normalization output.
    """

    parts: list[str] = []
    cursor = window_start
    for edit_span, replacement in edits:
        if edit_span.start < window_start or edit_span.end > window_end:
            continue
        if edit_span.start < cursor:
            continue
        parts.append(text[cursor:edit_span.start])
        parts.append(replacement)
        cursor = edit_span.end
    parts.append(text[cursor:window_end])
    return "".join(parts).strip()


def _project_if_spec(
    payload: IfStatementSyntax,
    statement: StatementSyntax,
    source: SourceDocument,
) -> IfStatementSpec | None:
    """The legacy ``IfStatementSpec`` of one structured IF payload.

    The condition is the initial split's stored ``condition_display_span``
    raw slice with ONLY the stored ``condition_display_edits`` applied —
    no THEN/quote/paren re-scan, no re-lex, no re-parse: glued keyword
    spellings inside tokens, e.g.
    ``x.then=1``, can never truncate the display.  A payload without a
    stored span degrades to the whole post-keyword body slice instead of
    an invented boundary.  The THEN action display is the action's own raw
    span slice (typed statement or deferred descriptor alike).
    """

    span = payload.condition_display_span
    if span is not None:
        window_start, window_end = span.start, span.end
    else:
        raw = source.text[statement.span.start : statement.span.end]
        window_start = statement.span.start + _if_body_start(raw)
        window_end = statement.span.end

    if payload.is_subset or payload.then_action is None:
        return IfStatementSpec(
            condition=_apply_condition_display_edits(
                source.text, window_start, window_end, payload.condition_display_edits
            ),
            then_action=None,
            is_subset=payload.is_subset,
            is_then_do=payload.is_then_do,
        )

    action_span = payload.then_action.span
    return IfStatementSpec(
        condition=_apply_condition_display_edits(
            source.text, window_start, window_end, payload.condition_display_edits
        ),
        then_action=source.text[action_span.start : action_span.end].strip(),
        is_subset=payload.is_subset,
        is_then_do=payload.is_then_do,
    )


def _project_do_spec(
    payload: DoStatementSyntax,
    statement: StatementSyntax,
    source: SourceDocument,
) -> DoStatementSpec | None:
    """The legacy ``DoStatementSpec`` of one structured DO payload.

    The raw loop variable and the FULL raw boundary displays (parenthesized
    groups included and absorbed ``BY``/``until``/``while`` tails kept
    inside the END display) come from the stored initial
    ``boundary_display_spans`` — the same spans the initial ``do_loop``
    split computed, so a glued
    ``to`` spelling inside a bound expression, e.g. ``1+x.to``, can never
    split the display).  A bare DO and any payload without stored spans
    keep the legacy no-spec shape instead of an invented boundary; no
    top-level ``to`` re-scan and no re-parse ever runs.
    """

    loop_var = payload.loop_var
    spans = payload.boundary_display_spans
    if loop_var is None or spans is None:
        return None
    start_span, end_span = spans
    return DoStatementSpec(
        loop_var=loop_var,
        start_expr=source.text[start_span.start : start_span.end].strip(),
        end_expr=source.text[end_span.start : end_span.end].strip(),
    )


def project_statement(
    statement: StatementSyntax, source: SourceDocument
) -> ParsedStatement:
    """Project one typed statement onto the legacy ``ParsedStatement``.

    ``text`` is the raw legacy DISPLAY slice stripped (the initial
    classification segment, including leading whitespace in its span
    like the old facade) and ``span`` the
    legacy ``DiagnosticSpan`` of that slice.  A deferred payload keeps
    the legacy no-spec shape (``if_spec`` / ``do_spec`` / ``array_spec``
    all ``None``); recognized kinds copy their typed payload fields;
    OUTPUT keeps the old observational shape (no ``output_refs`` — the
    legacy runtime read OUTPUT targets from the display text) unless the
    public output-contract exception actually needs it.
    """

    payload = statement.payload
    display_span = statement.legacy_display_span
    if display_span is None:
        display_span = statement.span
    projected = ParsedStatement(
        kind=statement.kind,
        text=_raw_text(source, display_span),
        span=_legacy_span(display_span),
    )
    if isinstance(payload, DataStatementSyntax):
        refs = tuple(_project_dataset_reference(ref) for ref in payload.outputs)
        return replace(projected, dataset_refs=refs, output_refs=refs)
    if isinstance(payload, SourceStatementSyntax):
        return replace(
            projected,
            dataset_refs=tuple(_project_dataset_reference(ref) for ref in payload.refs),
            statement_options=SetStatementOptionSpec(
                indsname_var=payload.options.indsname_var,
                end_var=payload.options.end_var,
            ),
        )
    if isinstance(payload, LabelStatementSyntax):
        return replace(
            projected, label_map={attribute.name: attribute.label for attribute in payload.labels}
        )
    if isinstance(payload, RenameStatementSyntax):
        return replace(projected, rename_map=dict(payload.pairs))
    if isinstance(payload, ArrayStatementSyntax):
        return replace(
            projected,
            array_spec=ArrayStatementSpec(
                array_name=payload.array_name,
                variables=payload.variables,
                declared_size=payload.declared_size,
                wildcard_size=payload.wildcard_size,
                character_array=payload.character_array,
            ),
        )
    if isinstance(payload, IfStatementSyntax):
        return replace(projected, if_spec=_project_if_spec(payload, statement, source))
    if isinstance(payload, DoStatementSyntax):
        return replace(projected, do_spec=_project_do_spec(payload, statement, source))
    return projected


def project_block(
    block: BlockSyntax, source: SourceDocument
) -> tuple[ParsedStatement, ...]:
    """Project one typed block's statements in their flat source order."""

    return tuple(project_statement(statement, source) for statement in block.statements)


def project_program(program: ProgramSyntax) -> DataStepAst:
    """Project the whole typed program onto the legacy ``DataStepAst``.

    The legacy parse surface had no block grouping, so every block's
    statements flatten in source order and the identified skip regions
    are excluded (skip unification).
    """

    statements: list[ParsedStatement] = []
    for block in program.blocks:
        statements.extend(project_block(block, program.source))
    return DataStepAst(statements=tuple(statements))


def project_result(result: ProgramParseResult) -> ParseResult:
    """Project the whole-text pure result onto the legacy ``ParseResult``.

    Projected behavior:

    - a structured program flattens into the legacy ``DataStepAst``; an
      EMPTY/blank input (a success with neither blocks nor skip regions)
      keeps the old ``PARSE_UNSUPPORTED_STATEMENT`` near-line/column
      rejection, while a skip-only success stays the empty AST with no
      diagnostics;
    - a stopping result keeps its legacy-shaped diagnostics with the
      initial owners' stored ``diagnostic_display`` value per diagnostic
      (e.g. the failing dataset-option WHERE value).  Where the old
      facade's outer statement pass HALTED — the ``syntax error`` /
      ``unsupported statement`` label families — only the first
      diagnostic is projected (the old read had parsed nothing else);
      the old per-statement fault families (``statement`` labels) keep
      every collected diagnostic in order.

    No display value is ever re-scanned, and the typed result is
    untouched.
    """

    if result.program is None:
        diagnostics = result.diagnostics
        displays = (
            result.projection.diagnostic_displays if result.projection is not None else ()
        )
        if any(_legacy_halt_family(diagnostic) for diagnostic in diagnostics):
            # Old whole-text halt: exactly the first stop diagnostic (with
            # its own display metadata, position-matched by the frontend).
            display = displays[0] if displays else None
            return ParseResult(
                diagnostics=(_legacy_stopped_diagnostic(diagnostics[0], display),)
            )
        return ParseResult(
            diagnostics=tuple(
                _legacy_stopped_diagnostic(
                    diagnostic, displays[index] if index < len(displays) else None
                )
                for index, diagnostic in enumerate(diagnostics)
            )
        )

    program = result.program
    if not program.blocks:
        if program.skip_regions:
            # Adopted skip unification: skip-only input is the empty AST.
            return ParseResult(ast=DataStepAst())
        # Empty/blank input keeps the old rejection.
        return ParseResult(
            ast=DataStepAst(),
            diagnostics=(_legacy_empty_program_diagnostic(program.source),),
        )
    return ParseResult(ast=project_program(program))


_LEGACY_HALT_LABELS: frozenset[str] = frozenset(
    {"syntax error", "unsupported statement"}
)


def _legacy_halt_family(diagnostic: LegacyDiagnostic) -> bool:
    """Whether one typed stop diagnostic belongs to the old outer halt.

    The old facade's statement-segment pass raised these two label
    families from a SINGLE outer grammar call that aborted the whole
    parse; the per-statement fault families carried the ``statement``
    label instead and never halted the read.
    """

    return any(label.message in _LEGACY_HALT_LABELS for label in diagnostic.labels)


def _legacy_stopped_diagnostic(
    diagnostic: LegacyDiagnostic,
    display: ParseDiagnosticDisplay | None,
) -> LegacyDiagnostic:
    """One legacy stop diagnostic with its initial owner's display value.

    The typed diagnostic's ``source_text`` stays the full raw text; the
    stored ``ParseDiagnosticDisplay`` (e.g. the failing dataset-option
    WHERE value) is the ONLY display override the legacy view projects.
    """

    if display is not None and display.source_text != diagnostic.source_text:
        return replace(diagnostic, source_text=display.source_text)
    return diagnostic


def _legacy_empty_program_diagnostic(source: SourceDocument) -> LegacyDiagnostic:
    """The old empty/blank input rejection.

    Exactly the old facade's outer ``UnexpectedInput`` stop shape at the
    first codepoint: ``PARSE_UNSUPPORTED_STATEMENT``, ``statement:1``,
    ``near line 1, column 1``, the one-codepoint span 0..1 with the
    ``syntax error`` primary label, and the full raw input as
    ``source_text``.  The legacy span is the old DTO's independent
    constructor shape (end = start + 1 for the zero-width occurrence) —
    it is not an empty ``SourceSpan``.
    """

    span = LegacyDiagnosticSpan(
        start=0,
        end=1,
        line=1,
        column=1,
        end_line=1,
        end_column=2,
        source_id=source.source_id,
    )
    return LegacyDiagnostic(
        code="PARSE_UNSUPPORTED_STATEMENT",
        severity="error",
        location="statement:1",
        stage="",
        message="Unsupported statement syntax near line 1, column 1.",
        span=span,
        labels=(LegacyDiagnosticLabel(span=span, message="syntax error", kind="primary"),),
        source_text=source.text,
    )


def block_display_slices(
    text: str,
    blocks: Sequence[tuple[int, int, bool]],
    skip_spans: Sequence[tuple[int, int]] = (),
) -> tuple[str, ...]:
    """The old block splitter's display slices over the raw source text.

    Each block is a ``(start, end, terminated_by_run)`` triple.  A
    RUN-terminated block displays its own span (the frontier derivation
    already covers the RUN semicolon).  A missing-RUN block extends its
    display to the next frontier — the next block's start, or the next
    identified skip region's start when one sits between — so tail
    comments and whitespace before the frontier stay visible like the old
    next-DATA slicing; a trailing missing-RUN block keeps its own span
    end.  Every display is the stripped raw slice and nothing is parsed.
    """

    skip_starts = sorted(start for start, _end in skip_spans)
    displays: list[str] = []
    for index, (start, end, terminated_by_run) in enumerate(blocks):
        display_end = end
        if not terminated_by_run and index + 1 < len(blocks):
            frontier = blocks[index + 1][0]
            for skip_start in skip_starts:
                if start < skip_start < frontier:
                    frontier = skip_start
                    break
            display_end = frontier
        display = text[start:display_end].strip()
        if display:
            displays.append(display)
    return tuple(displays)


def project_block_displays(program: ProgramSyntax) -> tuple[str, ...]:
    """Project every typed block's display slice in source order.

    The last statement kind decides the RUN termination exactly like the
    frontier derivation (a block is closed by its own RUN statement).
    """

    def terminated_by_run(block: BlockSyntax) -> bool:
        return bool(block.statements) and block.statements[-1].kind == "RUN"

    return block_display_slices(
        program.source.text,
        [
            (block.span.start, block.span.end, terminated_by_run(block))
            for block in program.blocks
        ],
        [(region.span.start, region.span.end) for region in program.skip_regions],
    )
