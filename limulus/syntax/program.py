"""Whole-source Data Step parsing and DATA/skip region detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias, cast

from lark import Lark
from lark.exceptions import UnexpectedInput
from lark.lexer import Token
from lark.tree import Tree

from .models import (
    ApplyReferenceSyntax,
    AssignStatementSyntax,
    ArrayElementTargetSyntax,
    ArrayStatementSyntax,
    DataStatementSyntax,
    DatasetOptionSyntax,
    DatasetReferenceSyntax,
    DeferredStatementError,
    DeleteStatementSyntax,
    DoStatementSyntax,
    ElseStatementSyntax,
    IfStatementSyntax,
    NameTargetSyntax,
    NameListStatementSyntax,
    NameSyntax,
    OutputStatementSyntax,
    LabelAttributeSyntax,
    LabelStatementSyntax,
    ParseDiagnosticDisplay,
    ProgramProjectionFacts,
    RenameStatementSyntax,
    SetStatementOptionSyntax,
    SourceDocument,
    SourceSpan,
    SkipRegionSyntax,
    SkippedStatementSyntax,
    SourceStatementSyntax,
    StatementSyntax,
    StopStatementSyntax,
    SumStatementSyntax,
    WhereComparison,
    WhereComparisonOperator,
    WhereExpression,
    WhereLogicalExpression,
    WhereLogicalOperator,
    WhereNotExpression,
    WhereNullPredicate,
    WhereScalar,
)
from .expressions import collect_apply_references, parse_general_expression
from .models import ExpressionSyntax  # noqa: E402  (type alias reuse)
from .normalization import initial_analysis_text

# The pure result surface plus the pure diagnostic DTO the
# whole-text owner fills when a stopping parse fault occurs.  The diagnostic
# DTO is the stdlib-only ``limulus.models`` diagnostic; no framework parse
# tree or token is stored, and no alternate diagnostic type is introduced.
from ..models import DiagnosticLabel as PureDiagnosticLabel
from ..models import DiagnosticSpan as PureDiagnosticSpan
from ..models import Diagnostic as PureDiagnostic
from .models import BlockSyntax
from .models import EndStatementSyntax
from .models import ProgramParseResult
from .models import ProgramSyntax
from .models import RunStatementSyntax
from .models import StatementKind
from .models import WhereStatementSyntax



ProgramStatementKind = Literal["DATA", "RUN", "OTHER"]

_STATEMENT_RULE_KINDS: dict[str, ProgramStatementKind] = {
    "data_stmt": "DATA",
    "run_stmt": "RUN",
    "other_stmt": "OTHER",
}

_SKIP_RULE_KINDS: dict[str, Literal["macro", "proc"]] = {
    "macro_line_stmt": "macro",
    "macro_block_stmt": "macro",
    "proc_block_stmt": "proc",
}


class ParserInitializationError(Exception):
    """Mandatory grammar initialization failure.

    The frontend never degrades a failed grammar initialization into a
    parse result; the pipeline and the legacy parser facade convert this
    exception into the existing ``PARSE_BACKEND_CAPABILITY_MISSING``
    diagnostic contract in their own owners.
    """


@dataclass(frozen=True)
class ProgramStatementRegion:
    """One executable statement region: classified kind plus span."""

    kind: ProgramStatementKind
    span: SourceSpan


@dataclass(frozen=True)
class ProgramBlockRegion:
    """One ordered DATA block region.

    ``span`` covers the block from the DATA keyword through the extended
    end of its last statement.  ``statements`` lists the block's
    executable statement regions in source order; standard skip regions
    are never part of a block.  ``terminated_by_run`` records whether the
    block was closed by its own RUN statement or by the text (missing
    RUN) so the later stop-classification owner can decide without
    re-classifying.
    """

    span: SourceSpan
    statements: tuple[ProgramStatementRegion, ...]
    terminated_by_run: bool


@dataclass(frozen=True)
class ProgramRegions:
    """Internal whole-text region payload of one upfront parse.

    Immutable and document-scoped: spans anchor to ``source`` only.  The
    typed statement payloads and the public ``parse_program`` result are
    layered on top of these regions.
    """

    source: SourceDocument
    span: SourceSpan
    blocks: tuple[ProgramBlockRegion, ...] = ()
    skip_regions: tuple[SkipRegionSyntax, ...] = ()
    statement_regions: tuple[ProgramStatementRegion, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceDocument):
            raise TypeError(
                f"program regions source must be a SourceDocument, got "
                f"{type(self.source).__name__}"
            )
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"program regions span must be a SourceSpan, got {type(self.span).__name__}"
            )
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "skip_regions", tuple(self.skip_regions))
        object.__setattr__(self, "statement_regions", tuple(self.statement_regions))


def _grammar_path() -> Path:
    # The pure frontend owns its corrected region grammar (power-tail
    # truth plus ordinary open-quote fragments).  There is ONE
    # mandatory region parser path and no second-engine fallback.
    return Path(__file__).resolve().parents[1] / "grammar" / "program_regions.lark"


def _if_structured_grammar_path() -> Path:
    return Path(__file__).resolve().parents[1] / "grammar" / "if_structured.lark"


def _do_loop_grammar_path() -> Path:
    return Path(__file__).resolve().parents[1] / "grammar" / "do_loop.lark"


def _array_declaration_grammar_path() -> Path:
    return Path(__file__).resolve().parents[1] / "grammar" / "array_declaration.lark"


def _build_array_declaration_parser() -> Lark:
    """Build (once) the legacy ``array_declaration`` grammar.

    The ARRAY entry reuses the exact legacy facade grammar
    size / variable split, so the accepted declaration corpus (bracket
    ``[*]``/``[N]`` sizes, ``$`` character marker, glued ``arrayy`` naming)
    is reproduced instead of being re-guessed.  A mandatory initialization
    failure is raised instead of being hidden.
    """

    if Lark is None:
        raise ParserInitializationError(
            "lark is not available; the ARRAY declaration grammar cannot initialize."
        )
    try:
        grammar = _array_declaration_grammar_path().read_text(encoding="utf-8")
    except Exception as error:
        raise ParserInitializationError(
            f"ARRAY grammar initialization failed: {error}"
        ) from error
    try:
        return Lark(
            grammar,
            start="start",
            parser="lalr",
            propagate_positions=True,
        )
    except Exception as error:
        raise ParserInitializationError(
            f"ARRAY grammar initialization failed: {error}"
        ) from error


def _build_do_loop_parser() -> Lark:
    """Build (once) the legacy ``do_loop`` boundary splitter.

    The DO entry reuses the exact legacy facade grammar
    (``limulus/grammar/do_loop.lark``) for the DO loop variable / boundary
    split, so the accepted ``do i=1 to <expr>`` corpus (including the
    absorbed ``BY``/``until`` tails) is reproduced instead of being
    re-guessed.  A mandatory initialization failure is raised instead of
    being hidden.
    """

    if Lark is None:
        raise ParserInitializationError(
            "lark is not available; the DO boundary grammar cannot initialize."
        )
    try:
        grammar = _do_loop_grammar_path().read_text(encoding="utf-8")
    except Exception as error:
        raise ParserInitializationError(
            f"DO grammar initialization failed: {error}"
        ) from error
    try:
        return Lark(
            grammar,
            start="start",
            parser="lalr",
            propagate_positions=True,
        )
    except Exception as error:
        raise ParserInitializationError(
            f"DO grammar initialization failed: {error}"
        ) from error


def _build_if_structured_parser() -> Lark:
    """Build (once) the legacy ``if_structured`` condition/action splitter.

    The IF entry reuses the exact legacy facade grammar
    (``limulus/grammar/if_structured.lark``) for the IF / ELSE IF
    condition/action split, so the accepted splitting quirks of
    ``LarkParserFrontend._parse_if_statement_spec`` are reproduced instead
    of being re-guessed.  A mandatory initialization failure is raised
    instead of being hidden.
    """

    if Lark is None:
        raise ParserInitializationError(
            "lark is not available; the IF condition grammar cannot initialize."
        )
    try:
        grammar = _if_structured_grammar_path().read_text(encoding="utf-8")
    except Exception as error:
        raise ParserInitializationError(
            f"if grammar initialization failed: {error}"
        ) from error
    try:
        return Lark(
            grammar,
            start="start",
            parser="lalr",
            propagate_positions=True,
        )
    except Exception as error:
        raise ParserInitializationError(
            f"if grammar initialization failed: {error}"
        ) from error


def _build_program_parser() -> Lark:
    """Build (once) the LALR parser for the region identification grammar."""

    if Lark is None:
        raise ParserInitializationError(
            "lark is not available; the program grammar cannot initialize."
        )
    try:
        grammar = _grammar_path().read_text(encoding="utf-8")
    except Exception as error:
        raise ParserInitializationError(
            f"program grammar initialization failed: {error}"
        ) from error
    try:
        return Lark(
            grammar,
            start="start",
            parser="lalr",
            propagate_positions=True,
        )
    except Exception as error:
        raise ParserInitializationError(
            f"program grammar initialization failed: {error}"
        ) from error


def _extend_statement_end(text: str, end: int) -> int:
    """Extend a statement end through whitespace and a following ``;``.

    Same semantics as the legacy block splitter helper: the block span
    statement end covers the statement's closing semicolon but nothing
    beyond it.
    """

    cursor = end
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor < len(text) and text[cursor] == ";":
        return cursor + 1
    return end


@dataclass(frozen=True)
class _RawRegion:
    kind: ProgramStatementKind | Literal["SKIP", "MACRO", "PROC"] | None
    start: int
    end: int


def _region_nodes(tree: object) -> tuple[tuple[object, object, object], ...]:
    """Collect ``(rule_node, skip_rule_node_or_None, meta)`` per statement."""

    collected: list[tuple[object, object, object]] = []
    for statement_node in getattr(tree, "children", ()):
        children = getattr(statement_node, "children", ())
        if not children:
            continue
        rule_node = children[0]
        rule_name = str(getattr(rule_node, "data", ""))
        if rule_name == "skip_stmt":
            skip_children = getattr(rule_node, "children", ())
            skip_rule_node = skip_children[0] if skip_children else None
        else:
            skip_rule_node = None
        node_meta = getattr(rule_node, "meta", None)
        if node_meta is None:
            node_meta = getattr(statement_node, "meta", None)
        if node_meta is None:
            continue
        collected.append((rule_node, skip_rule_node, node_meta))
    return tuple(collected)


# --- initial request-local input normalization -------------
#
# Every initial syntax entry reads its STRUCTURE from one bounded,
# request-local analysis buffer (``LarkProgramFrontend._analysis_text`` ->
# ``limulus.syntax.normalization.initial_analysis_text``):
# the frontend keeps exactly ONE most-recent-document IDENTITY pair
# (document + analysis text); a new document — including an empty request
# and equal text/source_id — replaces the pair and is rescanned.  There is
# no inter-request global history, module cache, or retained wrapper.


class LarkProgramFrontend:
    """Pure frontend that owns the whole-text region identification parse."""

    def __init__(self) -> None:
        self._parser: Lark | None = None
        self._where_parser: Lark | None = None
        self._assignment_parser: Lark | None = None
        self._rename_pairs_parser: Lark | None = None
        self._label_pairs_parser: Lark | None = None
        self._if_structured_parser: Lark | None = None
        # Unclosed-quote fact (per-call local fact, no new raw model field): does the
        # ONE region parse contain an ordinary unclosed quote fragment token?
        self._region_open_quotes = False
        self._do_loop_parser: Lark | None = None
        self._array_declaration_parser: Lark | None = None
        self._classification_parser: Lark | None = None
        self._region_fail_position: int | None = None
        self._analysis_document: SourceDocument | None = None
        self._analysis_buffer: str = ""

    # --- request-local analysis text -----------------------

    def _analysis_text(self, source: SourceDocument) -> str:
        """The request-scoped normalized analysis text.

        Tabs outside quoted strings and comments became one space each and
        the length/newline layout is unchanged, so spans derived from it
        stay valid against the raw source.  The frontend holds exactly ONE
        most-recent-document pair (``SourceDocument`` identity + analysis
        text); the pair is replaced on every new request and ``source_id``
        or text equality alone is never the key.  The reported
        ``SourceDocument`` keeps its raw text.
        """

        if self._analysis_document is not source:
            self._analysis_document = source
            self._analysis_buffer = initial_analysis_text(source.text)
        return self._analysis_buffer

    def _get_parser(self) -> Lark:
        if self._parser is None:
            self._parser = _build_program_parser()
        return self._parser

    # --- dataset reference statement entry ----------------

    def _get_where_parser(self) -> Lark:
        if self._where_parser is None:
            self._where_parser = _build_dataset_where_parser()
        return self._where_parser

    def _get_assignment_parser(self) -> Lark:
        if self._assignment_parser is None:
            self._assignment_parser = _build_assignment_token_parser()
        return self._assignment_parser

    def _get_rename_pairs_parser(self) -> Lark:
        if self._rename_pairs_parser is None:
            self._rename_pairs_parser = _build_rename_pairs_parser()
        return self._rename_pairs_parser

    def parse_dataset_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one DATA/SET/MERGE/OUTPUT statement region into typed syntax.

        The entry is the bounded owner of the dataset reference
        statements: it takes the existing statement region (or an
        equivalent span into the same document) and returns a single
        ``StatementSyntax`` carrying the typed payload
        (``DataStatementSyntax`` / ``SourceStatementSyntax`` /
        ``OutputStatementSyntax``).  Option and syntax faults become a
        typed ``DeferredStatementError`` payload whose code, severity,
        and wrapper message stay with the existing statement owner; the
        statement keeps its keyword kind.  A non-dataset statement is a
        caller misuse and raises ``ValueError`` (there is no filler
        payload; further statement owners are completed by later review
        units).  Mandatory grammar initialization failures raise
        ``ParserInitializationError``.
        """

        if isinstance(statement, ProgramStatementRegion):
            span = statement.span
        elif isinstance(statement, SourceSpan):
            span = statement
        else:
            raise TypeError(
                "statement must be a ProgramStatementRegion or SourceSpan, "
                f"got {type(statement).__name__}"
            )
        if span.source_id != source.source_id:
            raise ValueError(
                f"statement span source_id {span.source_id!r} does not match "
                f"document source_id {source.source_id!r}"
            )
        if span.end > len(source.text) or span.start > span.end:
            raise ValueError(
                f"statement span {span.start}:{span.end} lies outside the document"
            )

        full_text = source.text[span.start : span.end]
        keyword_match = re.match(r"[A-Za-z_][\w]*", full_text.strip())
        if keyword_match is None:
            raise ValueError(
                "statement is not a dataset reference statement (expected DATA, "
                f"SET, MERGE, or OUTPUT): {full_text.strip()[:32]!r}"
            )
        keyword = keyword_match.group(0).lower()
        kind = _DATASET_KEYWORD_KINDS.get(keyword)
        if kind is None:
            raise ValueError(
                "statement is not a dataset reference statement (expected DATA, "
                f"SET, MERGE, or OUTPUT): {full_text.strip()[:32]!r}"
            )
        keyword_end = full_text.index(keyword_match.group(0)) + len(keyword)

        try:
            refs, statement_options = _scan_dataset_statement_body(
                self, source, span, keyword_end, kind
            )
            if kind in {"DATA", "SET", "MERGE"} and not refs:
                # The legacy grammar requires at least one dataset reference
                # component per DATA/SET/MERGE statement.
                raise _DatasetStatementFault("Unsupported statement syntax.")
        except _DatasetStatementFault as fault:
            return StatementSyntax(
                kind=kind,
                payload=DeferredStatementError(
                    error_kind="parse_fault",
                    native_cause=fault.cause,
                    span=span,
                    display_text=full_text.strip(),
                    diagnostic_display=fault.display,
                ),
                span=span,
            )

        if kind == "DATA":
            payload: DataStatementSyntax | SourceStatementSyntax | OutputStatementSyntax = (
                DataStatementSyntax(outputs=tuple(refs))
            )
        elif kind == "OUTPUT":
            payload = OutputStatementSyntax(targets=tuple(refs))
        else:
            options = _build_set_statement_options(statement_options)
            payload = SourceStatementSyntax(refs=tuple(refs), options=options)
        return StatementSyntax(kind=kind, payload=payload, span=span)

    # --- assignment / sum statement entries ---------------

    def parse_assignment_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one ordinary/inline assignment region into typed syntax.

        Bounded entry: the statement region (or an inline action span,
        reused by the IF / ELSE entries) is anchored to the same document and no
        full text is reparsed; target and RHS spans keep the exact original
        offsets.  The RHS reuses the shared deferred
        general-expression entry, so a failing RHS stays a whole-expression
        ``DeferredExpressionError`` without partial evaluation and no
        legacy normalizer/power regex is needed on the typed path; the
        power tree stays intact for the native evaluator.
        A failed target binding is a whole-statement
        ``DeferredStatementError`` keeping the legacy cause family
        (``invalid assignment statement: ...``).  Non-assignment input is
        rejected with ``ValueError`` (limited entry; the dispatcher owns
        statement classification).  Grammar initialization failures raise
        ``ParserInitializationError``.
        """

        span, statement_text = _require_execution_statement_span(source, statement)
        analysis_text = self._analysis_text(source)[span.start : span.end]
        apply_references = collect_apply_references(source, span)
        try:
            target, rhs_span = _assignment_target_and_rhs(
                source, span, statement_text, analysis_text
            )
        except _StatementEntryFault as fault:
            return _fault_statement(
                source, span, "ASSIGN", fault.cause, apply_references
            )
        result = parse_general_expression(source, rhs_span)
        return StatementSyntax(
            kind="ASSIGN",
            payload=AssignStatementSyntax(
                target=target,
                value=result.expression,
            ),
            span=span,
            apply_references=apply_references,
        )

    def parse_sum_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one SUM statement region into typed syntax.

        Single bare-name target plus the increment expression; the
        increment reuses the deferred general-expression entry.  A failed
        SUM shape (non-name target, blank increment) keeps the legacy
        ``Invalid SUM statement`` cause family as a whole-statement
        ``DeferredStatementError``; a failing increment expression stays
        deferred inside the payload.  Non-SUM input is rejected with
        ``ValueError``.
        """

        span, statement_text = _require_execution_statement_span(source, statement)
        analysis_text = self._analysis_text(source)[span.start : span.end]
        apply_references = collect_apply_references(source, span)
        try:
            target, increment_span = _sum_target_and_increment(
                source, span, statement_text, analysis_text
            )
        except _StatementEntryFault as fault:
            return _fault_statement(
                source, span, "SUM", fault.cause, apply_references
            )
        result = parse_general_expression(source, increment_span)
        return StatementSyntax(
            kind="SUM",
            payload=SumStatementSyntax(
                target=target,
                increment=result.expression,
            ),
            span=span,
            apply_references=apply_references,
        )

    # --- name-list statement entry -------------------------

    def parse_name_list_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one KEEP/DROP/BY/RETAIN statement region into typed syntax.

        Bounded name-list entry: the statement region (or an equivalent span
        into the same document) is the only input and its text must start
        with the keyword prefix; the payload is the ordered typed name list.

        The keyword gate mirrors the native ``statement_body`` helper over
        the normalized analysis text: the lowercased gate accepts the
        keyword plus one trailing space (or the bare keyword), and otherwise
        the whole statement text stays the body, whitespace-split with every
        token opaque.  This preserves the executor truth for glued prefixes
        (``keepSomething x``) and newline separators.

        Names accept quoted names (quotes kept), dotted names, repeated
        names, and RETAIN initial-value tokens as ordinary name entries; no
        descending/initial-value semantics is introduced here.  An empty
        list preserves the meaning and the full statement span.
        """

        span, statement_text = _require_execution_statement_span(source, statement)
        analysis_text = self._analysis_text(source)[span.start : span.end]
        apply_references = collect_apply_references(source, span)
        classification = _name_list_keyword(analysis_text)
        if classification is None:
            raise ValueError(
                "statement is not a name list statement (expected KEEP, DROP, "
                f"BY, or RETAIN): {statement_text.strip()[:32]!r}"
            )
        keyword, kind = classification
        names = _extract_name_list_names(source, span, analysis_text, keyword)
        return StatementSyntax(
            kind=kind,
            payload=NameListStatementSyntax(names=names),
            span=span,
            apply_references=apply_references,
        )

    # --- attribute output and skipped statement entries -----

    def _get_label_pairs_parser(self) -> Lark:
        if self._label_pairs_parser is None:
            self._label_pairs_parser = _build_label_pairs_parser()
        return self._label_pairs_parser

    def parse_label_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one LABEL statement region into typed syntax.

        Attribute pairs reuse the pinned ``label_pairs`` grammar: a quoted
        value keeps only its outer quotes removed, doubled quotes stay raw,
        order follows the legacy Python dict (first occurrence, last value
        wins, the stored span stays the pair that first declared the name),
        and the pair span covers ``name = value`` codepoints.  A failing
        body keeps the exact truncated legacy cause family; a bare ``label``
        keeps the legacy no-op payload.
        """
        span, statement_text = _require_execution_statement_span(source, statement)
        analysis_text = self._analysis_text(source)[span.start : span.end]
        apply_references = collect_apply_references(source, span)
        if not statement_text.lower().startswith("label"):
            raise ValueError(
                "statement is not a LABEL statement (expected the legacy label "
                f"keyword prefix): {statement_text.strip()[:32]!r}"
            )
        labels = _scan_label_pairs(self, source, span, analysis_text)
        if isinstance(labels, _StatementEntryFault):
            return _fault_statement(source, span, "LABEL", labels.cause, apply_references)
        return StatementSyntax(
            kind="LABEL",
            payload=LabelStatementSyntax(labels=labels),
            span=span,
            apply_references=apply_references,
        )

    def parse_rename_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one RENAME statement region into typed syntax.

        Bounded output entry over the same statement span contract as the
        LABEL entry, with the legacy ``rename`` keyword prefix.  The pairs
        reuse the pinned ``rename_pairs`` implementation shared with the
        dataset-reference option path, so valid pairs keep the legacy dict
        semantics (first-occurrence order, last value) and every invalid
        pair shape keeps the legacy ``Invalid rename mapping: {body}``
        cause family verbatim.  No schema, catalog, or runtime binding is
        applied, and a grammar backend failure raises
        ``ParserInitializationError``.
        """

        span, statement_text = _require_execution_statement_span(source, statement)
        analysis_text = self._analysis_text(source)[span.start : span.end]
        apply_references = collect_apply_references(source, span)
        if not statement_text.lower().startswith("rename"):
            raise ValueError(
                "statement is not a RENAME statement (expected the legacy "
                f"rename keyword prefix): {statement_text.strip()[:32]!r}"
            )
        try:
            pairs = _scan_rename_pairs(self, analysis_text)
        except _StatementEntryFault as fault:
            return _fault_statement(source, span, "RENAME", fault.cause, apply_references)
        return StatementSyntax(
            kind="RENAME",
            payload=RenameStatementSyntax(pairs=pairs),
            span=span,
            apply_references=apply_references,
        )

    def parse_skipped_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one legacy SKIPPED statement region into typed syntax.

        Bounded output entry over the same statement span contract as the
        LABEL / RENAME entries.  The keyword prefix family mirrors the
        legacy skipped regexes (``length``/``attrib``/``format``/
        ``informat`` without a word boundary, ``call`` with one), so glued
        spellings follow the legacy classification exactly.  The payload
        stays the position-only ``SkippedStatementSyntax``: the
        LENGTH/ATTRIB/FORMAT/INFORMAT/CALL spelling remains the display
        projection, skip-vs-reject classification is not
        duplicated here, and nothing is executed (no call evaluation, no
        host resolution).  Non-matching input is a caller-misuse
        ``ValueError``.
        """

        span, statement_text = _require_execution_statement_span(source, statement)
        apply_references = collect_apply_references(source, span)
        if _skipped_keyword_prefix(statement_text) is None:
            raise ValueError(
                "statement is not a legacy skipped statement (expected "
                "LENGTH, ATTRIB, FORMAT, INFORMAT, or CALL): "
                f"{statement_text.strip()[:32]!r}"
            )
        return StatementSyntax(
            kind="SKIPPED",
            payload=SkippedStatementSyntax(),
            span=span,
            apply_references=apply_references,
        )

    # --- IF / ELSE condition and inline action entries -------

    def _get_if_structured_parser(self) -> Lark:
        if self._if_structured_parser is None:
            self._if_structured_parser = _build_if_structured_parser()
        return self._if_structured_parser

    def parse_if_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one IF / ELSE IF statement region into typed syntax.

        The condition / THEN-action split reuses the legacy facade grammar
        (``if_structured.lark``; quote-aware THEN binding, subset shape when
        no top-level ``then`` exists) and the legacy ``THEN DO`` detection
        regexes, both read from the normalized analysis text, so every span
        stays anchored to the raw document.  No branch is ever executed and
        nothing beyond structure is computed here.

        - The condition reuses the shared deferred general-expression entry,
          so a failing condition stays a whole-expression
          ``DeferredExpressionError`` inside the payload and coexists with
          an action fault (the condition fault stays first).
        - The inline action is structured with the existing statement
          entries at exact original spans.  THEN DO is represented by
          ``is_then_do`` plus a bare no-loop-bound ``DO`` payload; the flat
          block statement order is untouched.
        - Per the deferred-action contract: an unsupported inline action is
          held as the ``DeferredStatementError`` descriptor INSIDE
          ``IfStatementSyntax.then_action`` — the action's own raw trimmed
          span, display slice, and the lazily failing cause family
          (``unsupported inline action: ...``); the condition always
          survives, and only a failed condition/action split keeps the
          outer whole-statement fault.  Neither shape becomes a SKIPPED
          statement or a program-stopping syntax error.
        """
        span, statement_text = _require_execution_statement_span(source, statement)
        analysis_text = self._analysis_text(source)[span.start : span.end]
        apply_references = collect_apply_references(source, span)
        split = _split_if_statement(self, source, span, analysis_text)
        if split is None:
            raise ValueError(
                "statement is not an IF / ELSE IF statement (expected the legacy "
                f"if or else if keyword prefix): {statement_text.strip()[:32]!r}"
            )
        return _build_if_statement(self, source, span, split, apply_references)

    def parse_else_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one ELSE statement region into typed syntax.

        The legacy ELSE statement has no condition: the inline body follows
        the ``else`` keyword with the exact legacy span and uses the
        deferred-action contract — a structured body keeps its typed
        statement, an unstructurable body keeps the descriptor inside the
        action slot, and a bare ``else`` keeps ``None``.
        """
        span, statement_text = _require_execution_statement_span(source, statement)
        analysis_text = self._analysis_text(source)[span.start : span.end]
        apply_references = collect_apply_references(source, span)
        body = _else_body_span(source, span, analysis_text)
        if body is None:
            raise ValueError(
                "statement is not an ELSE statement (expected the legacy else "
                f"keyword prefix): {statement_text.strip()[:32]!r}"
            )
        if (
            _ELSE_IF_KEYWORD_PATTERN.match(analysis_text) is not None
            and _THEN_KEYWORD_PATTERN.search(analysis_text) is not None
        ):
            # ``else if <condition> then ...`` belonged to the ELSE IF kind
            # under the legacy statement-kind classification.
            raise ValueError(
                "statement is not an ELSE statement (expected the legacy else "
                f"keyword prefix): {statement_text.strip()[:32]!r}"
            )
        if body.span is None:
            return StatementSyntax(
                kind="ELSE",
                payload=ElseStatementSyntax(),
                span=span,
                apply_references=apply_references,
            )
        action: StatementSyntax | DeferredStatementError = _structure_else_body(
            self, source, body.span, body.text
        )
        return StatementSyntax(
            kind="ELSE",
            payload=ElseStatementSyntax(action=action),
            span=span,
            apply_references=apply_references,
        )
    # --- initial classification and statement WHERE ----------

    def _get_classification_parser(self) -> Lark:
        if self._classification_parser is None:
            self._classification_parser = _build_classification_parser()
        return self._classification_parser

    def parse_where_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one statement WHERE region into typed syntax.

        Bounded statement-WHERE entry over the same statement span contract as the
        other limited entries.  The raw keyword body follows the exact
        native ``statement_body`` semantics the existing runtime owner
        keeps (``where `` keyword-plus-space consumed and trimmed, bare
        ``where`` yields an empty body, otherwise the whole statement
        slice stays the body), so no alias and no general-expression
        expansion is introduced.  The body reuses the shared
        deferred general-expression entry with ``where_summary=True``: a
        failing condition stays a whole-expression
        ``DeferredExpressionError`` inside the payload (the limited
        statement-WHERE compile summary rides on the descriptor; the
        compile object itself is never stored or executed), and the
        structured condition is never promoted to a program stop by
        this entry.  Non-WHERE input is a caller-misuse ``ValueError``
        (the classification dispatcher owns statement kinds), and
        grammar initialization failures raise
        ``ParserInitializationError``.
        """

        span, statement_text = _require_execution_statement_span(source, statement)
        analysis_text = self._analysis_text(source)[span.start : span.end]
        apply_references = collect_apply_references(source, span)
        lowered = analysis_text.lower()
        if lowered.startswith("where "):
            # Native ``statement_body`` truth: ``where `` keyword-plus-space
            # (with single whitespace already part of the prefix; further
            # whitespace is trimmed) is consumed and the tail trimmed.
            body_start = len("where")
        elif lowered == "where":
            body_start = None
        elif lowered.startswith("where"):
            # Glued ``where`` spellings keep the native whole-statement
            # body (the raw keyword body behavior of the statement-body
            # helper) and stay lazily failing general expressions.
            body_start = 0
        else:
            raise ValueError(
                "statement is not a WHERE statement (expected the legacy "
                f"where keyword prefix): {statement_text.strip()[:32]!r}"
            )
        if body_start is None:
            condition_span = None
        else:
            start = body_start
            end = len(analysis_text)
            while start < end and analysis_text[start].isspace():
                start += 1
            while end > start and analysis_text[end - 1].isspace():
                end -= 1
            if start >= end:
                condition_span = None
            else:
                condition_span = source.span(span.start + start, span.start + end)
        if condition_span is None:
            # Bare ``where`` (or a keyword-only tail) keeps the native
            # truth: the empty condition is a lazily failing general
            # expression, never a structured wildcard or a program stop.
            condition_span = source.span(span.end, span.end)
        result = parse_general_expression(
            source, condition_span, where_summary=True
        )
        return StatementSyntax(
            kind="WHERE",
            payload=WhereStatementSyntax(condition=result.expression),
            span=span,
            apply_references=apply_references,
        )

    # --- the public whole-text entry ------------------------

    def parse_program(
        self,
        text: str,
        *,
        source_id: str = "<dsl>",
    ) -> ProgramParseResult:
        """The public pure whole-text entry, module-level :func:`parse_program`.

        Same contract as :func:`limulus.syntax.program.parse_program`
        (optionally reused on an existing frontend): the whole text is
        structured in one upfront region-identification parse, statements
        are classified per block with the legacy ``datastep.lark``
        grammar, and the bounded statement entries complete the typing.  See
        :func:`parse_program` for the full contract.
        """

        return _parse_program_text(self, text, source_id=source_id)

    # --- DO boundary entry -----------------------------------

    def _get_do_loop_parser(self) -> Lark:
        if self._do_loop_parser is None:
            self._do_loop_parser = _build_do_loop_parser()
        return self._do_loop_parser

    def parse_do_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one DO statement region into typed syntax.

        The loop variable / boundary split reuses the exact legacy
        ``do_loop.lark`` grammar over the normalized analysis text, so every
        span stays anchored to the raw document.  ``BY``/``until`` tails
        that the legacy grammar absorbs into the END expression stay an
        invalid general slice in that slot (whole
        ``DeferredExpressionError``); no new loop forms, BY support, or
        matching-END bookkeeping is added here.

        Unsupported shapes (bare ``do``, ``do while``, ``do = 1``,
        non-word glued ``do:``/``do|``, missing END expression, non-ASCII
        loop variables) keep the lazily failing
        ``unsupported DO statement: ...`` native cause family as a whole
        statement ``DeferredStatementError``, never a program-stopping
        error, so an unmatched/untaken DO still avoids evaluation.
        """
        span, statement_text = _require_execution_statement_span(source, statement)
        analysis_text = self._analysis_text(source)[span.start : span.end]
        apply_references = collect_apply_references(source, span)
        if _DO_KEYWORD_PATTERN.match(analysis_text) is None:
            raise ValueError(
                "statement is not a DO statement (expected the legacy "
                f"do keyword prefix): {statement_text.strip()[:32]!r}"
            )
        return _build_do_statement(self, source, span, analysis_text, apply_references)

    # --- ARRAY declaration entry ----------------------------

    def _get_array_declaration_parser(self) -> Lark:
        if self._array_declaration_parser is None:
            self._array_declaration_parser = _build_array_declaration_parser()
        return self._array_declaration_parser

    def parse_array_statement(
        self,
        source: SourceDocument,
        statement: ProgramStatementRegion | SourceSpan,
    ) -> StatementSyntax:
        """Parse one ARRAY statement region into typed syntax.

        The split reuses the exact legacy ``array_declaration.lark``
        grammar and the old facade child reading (bracket ``[*]``/``[N]``
        /bare sizes, ``$`` character marker, glued ``arrayy`` naming), so
        the accepted legacy corpus is reproduced, NOT extended: ``(3)``
        sizes, bare leading numbers, quoted or hyphenated variables, and
        missing names were never accepted before.

        Size/variable consistency, repeat handling, and array registration
        are NOT this parser's job: they stay with the later owners.  Invalid
        declarations keep the lazily failing ``invalid ARRAY declaration:
        ...`` cause family as a whole-statement ``DeferredStatementError``
        — never a program-stopping syntax error — and stopped/untaken/
        array-fault code avoids evaluation as before.
        """
        span, statement_text = _require_execution_statement_span(source, statement)
        analysis_text = self._analysis_text(source)[span.start : span.end]
        apply_references = collect_apply_references(source, span)
        if _ARRAY_KEYWORD_PATTERN.match(analysis_text) is None:
            raise ValueError(
                "statement is not an ARRAY statement (expected the legacy "
                f"array keyword prefix): {statement_text.strip()[:32]!r}"
            )
        return _build_array_statement(self, source, span, analysis_text, apply_references)

    @staticmethod
    def _region_has_open_quote_fragments(tree: object) -> bool:
        """Whether the ONE region tree contains an unclosed quote fragment."""

        stack = [tree]
        while stack:
            node = stack.pop()
            children = getattr(node, "children", None)
            if children is None:
                if str(getattr(node, "type", "")) in {"SQ_OPEN", "DQ_OPEN"}:
                    return True
                continue
            stack.extend(children)
        return False

    def extract_program_regions(
        self, source: SourceDocument
    ) -> ProgramRegions | None:
        """Identify ordered DATA blocks, skip regions, and statement regions.

        Returns ``None`` only for full-text sources whose region grammar
        parse fails (no text fallback, diagnostics owned by the later
        whole-program unit); whitespace-only sources return an empty
        region payload, not ``None``.
        """

        text = source.text
        span = source.span(0, len(text))
        self._region_open_quotes = False
        if not text.strip():
            return ProgramRegions(source=source, span=span)

        parser = self._get_parser()
        try:
            tree = parser.parse(self._analysis_text(source))
        except UnexpectedInput as error:
            # Keep only the plain failing-codepoint offset
            # (no framework object retention) so the whole-text entry can
            # rebuild the legacy near-line/column stop diagnostic for the
            # full source; the return contract of this method stays None.
            self._region_fail_position = max(
                int(getattr(error, "pos_in_stream", 0) or 0), 0
            )
            return None

        # Open-quote fragments: the pure region
        # grammar keeps an ordinary unclosed quote run bounded inside its
        # statement (fragment token).  Such a fragment can only exist where
        # the legacy split grammar refused the text, which is exactly the
        # case where the legacy hook returned raw text and established NO
        # skip mask, so later stages keep the raw visibility below without
        # re-running the legacy grammar or any other backend.
        self._region_open_quotes = self._region_has_open_quote_fragments(tree)
        skipts: list[SkipRegionSyntax] = []
        statements: list[ProgramStatementRegion] = []
        open_quote_skips: list[ProgramStatementRegion] = []
        for rule_node, skip_rule_node, node_meta in _region_nodes(tree):
            rule_name = str(getattr(rule_node, "data", ""))
            start = max(getattr(node_meta, "start_pos", 0), 0)
            end = max(getattr(node_meta, "end_pos", start), start)
            if skip_rule_node is not None:
                skip_rule_name = str(getattr(skip_rule_node, "data", ""))
                skip_kind = _SKIP_RULE_KINDS.get(skip_rule_name)
                if skip_kind is None:
                    # Unknown skip alternative: mirror the legacy kind map
                    # and ignore the region instead of inventing one.
                    continue
                skip_end = end
                if end < len(text) and text[end] == ";":
                    skip_end = end + 1
                if self._region_open_quotes:
                    # Unclosed quote: the legacy hook returned raw visibility,
                    # so its standard macro/PROC regions established NO skip
                    # mask.  They stay ordinary statement regions (raw
                    # classification truth, e.g. ``%let q = 1`` reaches the
                    # classification as an ASSIGN) and the result keeps no
                    # skip regions for this request.
                    open_quote_skips.append(
                        ProgramStatementRegion(
                            kind="OTHER", span=source.span(start, skip_end)
                        )
                    )
                    continue
                skipts.append(
                    SkipRegionSyntax(kind=skip_kind, span=source.span(start, skip_end))
                )
                continue
            statement_kind = _STATEMENT_RULE_KINDS.get(rule_name)
            if statement_kind is None:
                # Unknown statement alternative: the legacy kind map yields
                # no kind, so the region participates in no block.
                continue
            statements.append(
                ProgramStatementRegion(
                    kind=statement_kind, span=source.span(start, end)
                )
            )

        if open_quote_skips:
            merged = sorted(
                [*statements, *open_quote_skips],
                key=lambda region: region.span.start,
            )
            statements = merged
        blocks = _derive_block_regions(source, statements)
        return ProgramRegions(
            source=source,
            span=span,
            blocks=blocks,
            skip_regions=tuple(skipts),
            statement_regions=tuple(statements),
        )


def _derive_block_regions(
    source: SourceDocument, statements: list[ProgramStatementRegion]
) -> tuple[ProgramBlockRegion, ...]:
    """Derive ordered block regions from the statement regions.

    Mirrors the legacy block splitter state machine, with standard skip
    regions already excluded upstream:  the first statement of any kind
    opens a pending block, a new DATA statement closes the pending block
    (missing RUN), a RUN statement closes its block at its extended end,
    and a pending block at end of text is the trailing block.
    """

    text = source.text
    blocks: list[ProgramBlockRegion] = []
    current_start: int | None = None
    current_end: int | None = None
    current_statements: list[ProgramStatementRegion] = []

    def _close(terminated_by_run: bool) -> None:
        assert current_start is not None and current_end is not None
        blocks.append(
            ProgramBlockRegion(
                span=source.span(current_start, current_end),
                statements=tuple(current_statements),
                terminated_by_run=terminated_by_run,
            )
        )

    for statement in statements:
        extended_end = _extend_statement_end(text, statement.span.end)
        if current_start is None:
            current_start = statement.span.start
            current_end = extended_end
            current_statements = [statement]
        elif statement.kind == "DATA":
            _close(False)
            current_start = statement.span.start
            current_end = extended_end
            current_statements = [statement]
        else:
            current_statements.append(statement)
            current_end = extended_end
        if statement.kind == "RUN":
            _close(True)
            current_start = None
            current_end = None
            current_statements = []

    if current_start is not None and current_end is not None:
        _close(False)
    return tuple(blocks)




# --- dataset statement grammar and option builders --------

_DATASET_KEYWORD_KINDS: dict[str, Literal["DATA", "SET", "MERGE", "OUTPUT"]] = {
    "data": "DATA",
    "set": "SET",
    "merge": "MERGE",
    "output": "OUTPUT",
}

_KEYWORD_PATTERN = re.compile(r"[A-Za-z_][\w]*")
_DATASET_NAME_PATTERN = re.compile(r"[A-Za-z_][\w\.]*")
_WHERE_PATTERN = re.compile(r"\bwhere\s*=", re.IGNORECASE)
_STATEMENT_OPTION_KEYS = frozenset({"indsname", "end"})


class _DatasetStatementFault(Exception):
    """Internal option/reference fault descriptor for one dataset statement.

    The frontend never raises user-facing parse diagnostics from here: the
    cause string is stored verbatim on the typed ``DeferredStatementError``
    payload, and the existing statement owner completes code, wrapper
    message, and stage later (unchanged option-refusal truth preserved).

    ``display`` is the frozen display text the initial owner
    computed for the actual failing PART — for a failed dataset-option
    WHERE it is the unwrapped RAW value of the failing option.  It is
    never re-scanned from display text downstream and it is never mixed
    into the native cause or the WHERE syntax summary.  ``None`` keeps
    the full-text source display.
    """

    def __init__(self, cause: str, display: ParseDiagnosticDisplay | None = None) -> None:
        super().__init__(cause)
        self.cause = cause
        self.display = display


def _build_dataset_where_parser() -> Lark:
    """Build (once) the restrictive dataset-option WHERE grammar."""

    if Lark is None:
        raise ParserInitializationError(
            "lark is not available; the dataset option WHERE grammar cannot initialize."
        )
    grammar_path = Path(__file__).resolve().parents[1] / "grammar" / "dataset_option_where.lark"
    try:
        grammar = grammar_path.read_text(encoding="utf-8")
    except Exception as error:
        raise ParserInitializationError(
            f"dataset option WHERE grammar initialization failed: {error}"
        ) from error
    try:
        return Lark(grammar, start="start", parser="lalr", propagate_positions=False)
    except Exception as error:
        raise ParserInitializationError(
            f"dataset option WHERE grammar initialization failed: {error}"
        ) from error


def _build_assignment_token_parser() -> Lark:
    """Build (once) the option ``key=value`` token grammar."""

    if Lark is None:
        raise ParserInitializationError(
            "lark is not available; the dataset option token grammar cannot initialize."
        )
    grammar_path = Path(__file__).resolve().parents[1] / "grammar" / "assignment_token.lark"
    try:
        grammar = grammar_path.read_text(encoding="utf-8")
    except Exception as error:
        raise ParserInitializationError(
            f"dataset option token grammar initialization failed: {error}"
        ) from error
    try:
        return Lark(grammar, start="start", parser="lalr", propagate_positions=False)
    except Exception as error:
        raise ParserInitializationError(
            f"dataset option token grammar initialization failed: {error}"
        ) from error


def _build_rename_pairs_parser() -> Lark:
    """Build (once) the rename-pair grammar."""

    if Lark is None:
        raise ParserInitializationError(
            "lark is not available; the rename pairs grammar cannot initialize."
        )
    grammar_path = Path(__file__).resolve().parents[1] / "grammar" / "rename_pairs.lark"
    try:
        grammar = grammar_path.read_text(encoding="utf-8")
    except Exception as error:
        raise ParserInitializationError(
            f"rename pairs grammar initialization failed: {error}"
        ) from error
    try:
        return Lark(grammar, start="start", parser="lalr", propagate_positions=False)
    except Exception as error:
        raise ParserInitializationError(
            f"rename pairs grammar initialization failed: {error}"
        ) from error


def _root_tree(parsed: object, expected: str) -> object:
    if getattr(parsed, "data", None) == expected:
        return parsed
    children = getattr(parsed, "children", ())
    if children and getattr(children[0], "data", None) == expected:
        return children[0]
    return parsed


def _parse_assignment_token(
    frontend: LarkProgramFrontend, token: str
) -> tuple[str, str] | None:
    try:
        parsed = frontend._get_assignment_parser().parse(token)
    except UnexpectedInput:
        return None

    root = _root_tree(parsed, "assignment_token")
    children = list(getattr(root, "children", ()))
    if len(children) != 3:
        return None
    return str(children[0]), str(children[2]).strip()


def _document_line_column(text: str, offset: int) -> tuple[int, int]:
    """Compute the 1-based line/column of a codepoint offset in the text."""

    if offset < 0:
        raise ValueError(f"codepoint offset {offset} must be non-negative")
    if offset > len(text):
        raise ValueError(
            f"codepoint offset {offset} is outside the source (length {len(text)})"
        )
    prefix = text[:offset]
    line = prefix.count("\n") + 1
    last_newline = prefix.rfind("\n")
    column = offset - last_newline if last_newline >= 0 else offset + 1
    return line, column


def _find_matching_parenthesis(text: str, start: int) -> int | None:
    """Find the matching ``)`` for one ``(``, balancing quoted literals."""

    depth = 0
    quote: str | None = None
    index = start
    while index < len(text):
        character = text[index]
        if quote is not None:
            if character == "\\" and index + 1 < len(text):
                index += 2
                continue
            if character == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
        index += 1
    return None


def _split_dataset_option_tokens(
    option_group: str,
) -> tuple[tuple[str, int], ...]:
    """Split one dataset option group at top-level ``key=`` boundaries.

    Keep/drop values may contain bare space-separated names, while WHERE
    and RENAME values may contain arbitrary nested parentheses and quoted
    text.  A boundary is therefore a top-level whitespace followed by an
    assignment key; no regular expression is applied to the option value
    itself.  Each result is ``(token, token_start)`` where ``token_start``
    is the offset of the token's first codepoint inside ``option_group``,
    so the initial fault owner can anchor a display slice to the RAW
    document without re-scanning the display text.
    """

    stripped = option_group.strip()
    stripped_start = option_group.index(stripped) if stripped else 0
    content = stripped
    content_start = stripped_start
    if stripped.startswith("("):
        matching_end = _find_matching_parenthesis(stripped, 0)
        if matching_end == len(stripped) - 1:
            content = stripped[1:-1]
            content_start = stripped_start + 1

    boundaries: list[int] = []
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(content):
        character = content[index]
        if quote is not None:
            if character == "\\" and index + 1 < len(content):
                index += 2
                continue
            if character == quote:
                if index + 1 < len(content) and content[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            index += 1
            continue
        if character == "(":
            depth += 1
            index += 1
            continue
        if character == ")":
            depth = max(depth - 1, 0)
            index += 1
            continue
        if depth == 0 and character.isspace():
            lookahead = index
            while lookahead < len(content) and content[lookahead].isspace():
                lookahead += 1
            if re.match(r"[A-Za-z_][\w]*\s*=", content[lookahead:]) is not None:
                boundaries.append(lookahead)
            index = lookahead
            continue
        index += 1

    starts = [0, *boundaries]
    tokens: list[tuple[str, int]] = []
    for start_index, end_index in zip(starts, [*boundaries, len(content)], strict=True):
        slice_ = content[start_index:end_index]
        token = slice_.strip()
        if token:
            lead = len(slice_) - len(slice_.lstrip())
            tokens.append((token, content_start + start_index + lead))
    return tuple(tokens)


def _unwrap_where_option(source: str) -> str:
    text = source.strip()
    if text[:6].lower() == "where=":
        text = text[6:].strip()
    if text.startswith("(") and text.endswith(")") and _outer_parentheses_enclose(text):
        text = text[1:-1].strip()
    return text


def _outer_parentheses_enclose(text: str) -> bool:
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(text):
        character = text[index]
        if quote is not None:
            if character == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 2
                    continue
                if index > 0 and text[index - 1] == "\\":
                    index += 1
                    continue
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
            if depth < 0:
                return False
        index += 1
    return depth == 0 and quote is None


def _parse_where_option(frontend: LarkProgramFrontend, value: str) -> WhereExpression:
    """Parse one dataset option WHERE value with the restrictive dialect.

    The dedicated grammar has no productions for functions, arithmetic,
    subscripts, attributes, or assignments, so those shapes fail closed
    with the existing refusal causes; nothing is evaluated.
    """

    text = _unwrap_where_option(value)
    if not text:
        raise _DatasetStatementFault("dataset option WHERE= is invalid: incomplete WHERE= expression")
    try:
        tree = frontend._get_where_parser().parse(text)
        return _expression_from_tree(tree)
    except (UnexpectedInput, ValueError, TypeError) as error:
        raise _DatasetStatementFault(
            "dataset option WHERE= is invalid: " + _failure_reason(text, error)
        ) from error
    except Exception as error:  # pragma: no cover - defensive parser boundary
        raise _DatasetStatementFault(
            f"dataset option WHERE= is invalid: unsupported WHERE= syntax ({type(error).__name__})"
        ) from error


def _expression_from_tree(tree: object) -> WhereExpression:
    root = tree
    if isinstance(root, Tree) and root.data == "start":
        if len(root.children) != 1:
            raise ValueError("incomplete WHERE= expression")
        root = root.children[0]
    return _node_from_tree(root)


def _node_from_tree(node: object) -> WhereExpression:
    if not isinstance(node, Tree):
        raise ValueError("incomplete WHERE= expression")

    data = str(node.data)
    children = list(node.children)
    if data == "grouped_expression":
        if len(children) != 1:
            raise ValueError("incomplete WHERE= parenthesized expression")
        return _node_from_tree(children[0])
    if data == "comparison":
        if len(children) != 3:
            raise ValueError("incomplete WHERE= comparison")
        column = _token_text(children[0])
        operator = _comparison_operator(_token_text(children[1]))
        value = _scalar_from_tree(children[2])
        return WhereComparison(column=column, operator=operator, value=value)
    if data == "null_predicate":
        columns = [child for child in children if isinstance(child, Token) and str(child.type) == "COLUMN"]
        if len(columns) != 1:
            raise ValueError("incomplete WHERE= null predicate")
        return WhereNullPredicate(column=_token_text(columns[0]), negated=False)
    if data == "null_predicate_negated":
        columns = [child for child in children if isinstance(child, Token) and str(child.type) == "COLUMN"]
        if len(columns) != 1:
            raise ValueError("incomplete WHERE= null predicate")
        return WhereNullPredicate(column=_token_text(columns[0]), negated=True)
    if data == "not_expression":
        expressions = [child for child in children if isinstance(child, Tree)]
        if len(expressions) != 1:
            raise ValueError("incomplete WHERE= NOT expression")
        return WhereNotExpression(operand=_node_from_tree(expressions[0]))
    if data in {"and_expression", "or_expression"}:
        operands = tuple(_node_from_tree(child) for child in children if isinstance(child, Tree))
        if len(operands) < 2:
            if len(operands) == 1:
                return operands[0]
            raise ValueError("incomplete WHERE= logical expression")
        return WhereLogicalExpression(operator=_logical_operator(data), operands=operands)
    # Lark may inline a single expression rule depending on grammar version.
    if len(children) == 1 and isinstance(children[0], Tree):
        return _node_from_tree(children[0])
    raise ValueError("unsupported WHERE= syntax")


def _scalar_from_tree(node: object) -> WhereScalar:
    if not isinstance(node, Tree) or len(node.children) != 1:
        raise ValueError("incomplete WHERE= scalar literal")
    token = node.children[0]
    raw = _token_text(token)
    if node.data == "string_literal":
        return _unquote_string(raw)
    if node.data == "boolean_literal":
        return raw.lower() == "true"
    if node.data == "number_literal":
        if any(marker in raw.lower() for marker in (".", "e")):
            return float(raw)
        return int(raw)
    raise ValueError("unsupported WHERE= scalar literal")


def _token_text(token: object) -> str:
    if isinstance(token, Token):
        return str(token.value)
    if isinstance(token, str):
        return token
    raise ValueError("incomplete WHERE= token")


def _comparison_operator(value: str) -> WhereComparisonOperator:
    if value == "=":
        return "="
    if value == "==":
        return "=="
    if value == "!=":
        return "!="
    if value == ">":
        return ">"
    if value == ">=":
        return ">="
    if value == "<":
        return "<"
    if value == "<=":
        return "<="
    raise ValueError("unsupported WHERE= comparison operator")


def _logical_operator(value: str) -> WhereLogicalOperator:
    if value == "and_expression":
        return "and"
    if value == "or_expression":
        return "or"
    raise ValueError("unsupported WHERE= logical operator")


def _unquote_string(raw: str) -> str:
    if len(raw) < 2 or raw[0] not in {"'", '"'} or raw[-1] != raw[0]:
        raise ValueError("incomplete WHERE= string literal")
    quote = raw[0]
    value = raw[1:-1]
    value = value.replace(quote * 2, quote)
    value = value.replace("\\" + quote, quote)
    return value


def _where_option_fault_display(
    source: SourceDocument,
    group_span: SourceSpan | None,
    token: str,
    token_start: int,
) -> ParseDiagnosticDisplay | None:
    """Unwrap the failing WHERE option's RAW display value (initial owner).

    Given the option group's RAW-anchored span and the failing option
    token's offset inside it, the display value is the balanced ``where=``
    value with the enclosing outer parentheses removed — sliced from the
    raw document text so tabs inside the value stay raw (never taken from
    the tab-normalized analysis buffer).  Any shape the unwrap cannot
    anchor (no ``where=``, unbalanced value, empty value) keeps ``None``,
    meaning the typed diagnostic's full-text ``source_text`` stays the
    display (the old truth for those shapes).  This is display-only
    information: no cause or summary is derived from it and nothing is
    re-scanned downstream.
    """

    if group_span is None:
        return None
    token_absolute_start = group_span.start + token_start
    token_absolute_end = token_absolute_start + len(token)
    if token_absolute_start < 0 or token_absolute_end > len(source.text):
        return None
    raw_token = source.text[token_absolute_start:token_absolute_end]
    match = re.search(r"where\s*=", raw_token, re.IGNORECASE)
    if match is None:
        return None
    cursor = match.end()
    while cursor < len(raw_token) and raw_token[cursor].isspace():
        cursor += 1
    if cursor < len(raw_token) and raw_token[cursor] == "(":
        end = _find_matching_parenthesis(raw_token, cursor)
        if end is None:
            return None
        value = raw_token[cursor + 1 : end].strip()
    else:
        value = raw_token[cursor:].strip()
    if not value:
        return None
    return ParseDiagnosticDisplay(source_text=value)


def _failure_reason(source: str, error: Exception) -> str:
    lowered = source.lower()
    if ":=" in source or ("=" in source and any(operator in source for operator in ("+=", "-=", "*=", "/="))):
        return "unsupported WHERE= assignment syntax"
    if "[" in source or "]" in source:
        return "unsupported WHERE= subscript syntax"
    if source.count("(") > source.count(")") or source.count(")") > source.count("("):
        return "incomplete WHERE= parenthesized expression"
    if "+" in source or "-" in source[1:] or "*" in source or "/" in source or "%" in source:
        return "unsupported WHERE= arithmetic syntax"
    if "(" in source or ")" in source and not _outer_parentheses_enclose(source):
        return "unsupported WHERE= function or incomplete parenthesized syntax"
    if "." in source and any(character.isalpha() for character in source):
        return "unsupported WHERE= attribute syntax"
    if any(word in lowered.split() for word in ("and", "or", "not", "is")) and source.rstrip().lower().endswith(("and", "or", "not", "is")):
        return "incomplete WHERE= logical or null predicate"
    if source.rstrip().endswith(("=", "==", "!=", ">", ">=", "<", "<=")):
        return "incomplete WHERE= comparison operand"
    if not source.strip() or getattr(error, "pos_in_stream", 0) >= len(source.rstrip()):
        return "incomplete WHERE= expression"
    if any(token.isidentifier() for token in source.replace("=", " ").split() if token):
        return "unsupported WHERE= column-to-column comparison"
    return f"unsupported WHERE= syntax ({type(error).__name__})"


def _parse_rename_pairs(frontend: LarkProgramFrontend, value_body: str) -> tuple[tuple[str, str], ...]:
    """Parse rename option pairs, preserving first-occurrence order."""

    try:
        parsed = frontend._get_rename_pairs_parser().parse(value_body)
    except UnexpectedInput:
        raise _DatasetStatementFault(f"Invalid rename mapping: {value_body}") from None

    ordered: dict[str, str] = {}
    for pair in getattr(parsed, "children", ()):
        if getattr(pair, "data", None) != "rename_pair":
            continue
        if len(getattr(pair, "children", ())) != 3:
            raise _DatasetStatementFault(f"Invalid rename mapping: {value_body}")
        old_name, _, new_name = pair.children
        normalized_old = old_name.strip()
        normalized_new = new_name.strip()
        if not normalized_old or not normalized_new:
            raise _DatasetStatementFault(f"Invalid rename mapping: {value_body}")
        ordered[normalized_old] = normalized_new
    return tuple(ordered.items())


def _strip_quoted_value(value: str) -> str:
    normalized = value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
        return normalized[1:-1]
    return normalized


def _build_dataset_options(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    group_span: SourceSpan | None,
    group_token: str | None,
) -> DatasetOptionSyntax:
    """Build one reference's options from its option-group token.

    The dispatch mirrors the existing option semantics verbatim, including
    keep/drop value collection, the INTEGER coercion of firstobs/obs, the
    unquoted label value, and the out-of-scope refusals for statement-only
    keys.  ``where_expr`` reuses the dedicated restrictive WHERE grammar.

    ``group_span`` anchors the option group in the RAW document (the
    analysis slice maps 1:1 onto it), so a failing WHERE option's display
    value is unwrapped from the raw option-token slice by its initial
    owner and
    travels on the fault — never re-scanned from display text, and never
    taken from the tab-normalized analysis buffer.
    """

    if group_token is None:
        return DatasetOptionSyntax()
    tokens = _split_dataset_option_tokens(group_token)
    if not tokens:
        # An empty option group was never accepted by the outer grammar.
        raise _DatasetStatementFault("Unsupported statement syntax.")

    in_var: str | None = None
    keep_vars: list[str] = []
    drop_vars: list[str] = []
    where_expr: WhereExpression | None = None
    rename_pairs: tuple[tuple[str, str], ...] = ()
    firstobs: int | None = None
    obs: int | None = None
    label: str | None = None
    active_collect: str | None = None

    for token, token_start in tokens:
        token_for_assignment = token
        if token.startswith("(") and token[1:].lstrip().lower().startswith("where="):
            token_for_assignment = token[1:].lstrip()
        parsed_token = _parse_assignment_token(frontend, token_for_assignment)
        if parsed_token is not None:
            key, raw_value = parsed_token
            normalized_key = key.strip().lower()
            value = raw_value.strip()

            if normalized_key in {"indsname", "end"}:
                raise _DatasetStatementFault(
                    f"{normalized_key.upper()}= is only allowed as a SET statement option."
                )

            if normalized_key == "in":
                in_var = value
                active_collect = None
                continue

            if normalized_key == "keep":
                keep_vars = [part for part in value.split() if part]
                active_collect = "keep"
                continue

            if normalized_key == "drop":
                drop_vars = [part for part in value.split() if part]
                active_collect = "drop"
                continue

            if normalized_key == "where":
                display = _where_option_fault_display(
                    source, group_span, token, token_start
                )
                try:
                    where_expr = _parse_where_option(frontend, value)
                except _DatasetStatementFault as fault:
                    raise _DatasetStatementFault(fault.cause, display) from None
                active_collect = None
                continue

            if normalized_key == "rename":
                value_body = value
                if value_body.startswith("(") and value_body.endswith(")"):
                    value_body = value_body[1:-1]
                rename_pairs = _parse_rename_pairs(frontend, value_body)
                active_collect = None
                continue

            if normalized_key == "firstobs":
                try:
                    firstobs = int(value)
                except ValueError:
                    raise _DatasetStatementFault(
                        f"Unsupported dataset reference option: {token}"
                    ) from None
                active_collect = None
                continue

            if normalized_key == "obs":
                try:
                    obs = int(value)
                except ValueError:
                    raise _DatasetStatementFault(
                        f"Unsupported dataset reference option: {token}"
                    ) from None
                active_collect = None
                continue

            if normalized_key == "label":
                label = _strip_quoted_value(value)
                active_collect = None
                continue

            raise _DatasetStatementFault(
                f"Unsupported dataset reference option: {token}"
            )

        if active_collect == "keep":
            keep_vars.append(token)
            continue

        if active_collect == "drop":
            drop_vars.append(token)
            continue

        raise _DatasetStatementFault(
            f"Invalid dataset reference option token: {token}"
        )

    return DatasetOptionSyntax(
        in_var=in_var,
        keep_vars=tuple(keep_vars),
        drop_vars=tuple(drop_vars),
        where_expr=where_expr,
        rename_map=rename_pairs,
        firstobs=firstobs,
        obs=obs,
        label=label,
    )


def _build_set_statement_options(
    statement_options: list[tuple[str, str, str]],
) -> SetStatementOptionSyntax:
    """Apply ordered SET/MERGE statement-level options (last one wins)."""

    options = SetStatementOptionSyntax()
    for key, value, _token in statement_options:
        if key == "indsname":
            options = SetStatementOptionSyntax(indsname_var=value, end_var=options.end_var)
        else:
            options = SetStatementOptionSyntax(indsname_var=options.indsname_var, end_var=value)
    return options


def _scan_dataset_statement_body(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    statement_span: SourceSpan,
    body_start: int,
    kind: str,
) -> tuple[list[DatasetReferenceSyntax], list[tuple[str, str, str]]]:
    """Scan the statement body into refs, option groups, statement options.

    Mirrors the legacy outer boundary grammar (`dataset_ref: dataset_name
    dataset_options?` plus the `set_statement_option` tokens) with balanced
    parentheses and quoted/escaped literal handling.  The statement slice
    is read from the request-local normalized analysis text: quoted option values and comment content keep
    their tab characters and every codepoint offset maps 1:1 onto the raw
    statement region.  No full-DSL reparse and no text fallback is
    involved.
    """

    text = frontend._analysis_text(source)[
        statement_span.start : statement_span.end
    ]
    base = statement_span.start
    refs: list[DatasetReferenceSyntax] = []
    statement_options: list[tuple[str, str, str]] = []
    pending_name: str | None = None
    pending_start: int | None = None
    pending_end: int | None = None
    pending_group: tuple[int, str] | None = None

    def close_pending() -> None:
        nonlocal pending_name, pending_start, pending_end, pending_group
        if pending_name is None or pending_start is None or pending_end is None:
            return
        group_token = pending_group[1] if pending_group is not None else None
        group_span = (
            source.span(base + pending_group[0], base + pending_group[0] + len(pending_group[1]))
            if pending_group is not None
            else None
        )
        options = _build_dataset_options(frontend, source, group_span, group_token)
        if pending_group is not None:
            ref_end = pending_group[0] + len(pending_group[1])
        else:
            assert pending_start is not None and pending_end is not None
            ref_end = pending_end
        refs.append(
            DatasetReferenceSyntax(
                name=pending_name,
                options=options,
                span=source.span(base + pending_start, base + ref_end),
            )
        )
        pending_name = None
        pending_start = None
        pending_end = None
        pending_group = None

    index = body_start
    while index < len(text):
        character = text[index]
        if character in {"'", '"'}:
            # A quoted literal only exists inside an option group; the group
            # branch consumes groups as balanced whole tokens, so any quote
            # reached here is group-external data the legacy statement
            # grammar never accepts (group-external quoted names, stray
            # quoted tokens after a group).
            close_pending()
            position = statement_span.start + index
            line, column = _document_line_column(source.text, position)
            raise _DatasetStatementFault(
                f"Unsupported statement syntax near line {line}, column {column}."
            )
        if character == ";":
            break
        if character == "(":
            if pending_name is None or pending_group is not None:
                close_pending()
                raise _DatasetStatementFault("Unsupported statement syntax.")
            group_end = _find_matching_parenthesis(text, index)
            if group_end is None:
                if _WHERE_PATTERN.search(text) is not None:
                    raise _DatasetStatementFault(
                        "dataset option WHERE= is invalid: incomplete WHERE= parenthesized expression"
                    )
                raise _DatasetStatementFault("Unsupported statement syntax.")
            pending_group = (index, text[index : group_end + 1])
            close_pending()
            index = group_end + 1
            continue
        statement_option_key = _KEYWORD_PATTERN.match(text, index)
        if statement_option_key is not None:
            name_end = statement_option_key.end()
            lookahead = name_end
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] == "=":
                close_pending()
                value_start = lookahead + 1
                while value_start < len(text) and text[value_start].isspace():
                    value_start += 1
                value_match = _DATASET_NAME_PATTERN.match(text, value_start)
                if value_match is None:
                    raise _DatasetStatementFault("Unsupported statement syntax.")
                key = statement_option_key.group(0).lower()
                token_text = text[index : value_match.end()].strip()
                if kind not in {"SET", "MERGE"}:
                    raise _DatasetStatementFault("Unsupported statement syntax.")
                if key == "indsname" and kind != "SET":
                    raise _DatasetStatementFault(
                        "INDSNAME= is only allowed as a SET statement option."
                    )
                if key == "in":
                    raise _DatasetStatementFault(
                        "IN= is only allowed as a dataset reference option (e.g. set in(in=flag))."
                    )
                if key not in _STATEMENT_OPTION_KEYS:
                    raise _DatasetStatementFault(
                        f"Unsupported SET statement option: {token_text}"
                    )
                statement_options.append((key, value_match.group(0), token_text))
                index = value_match.end()
                continue
        # Dotted dataset NAME: the legacy ``NAME`` terminal and the
        # old facade dataset-name truth accept the same ``[\w.]*`` tail
        # (``work.in``/``Work.Input``), so the existing name pattern owns
        # plain and dotted reference names alike.  Option keys for the
        # SET end/indsname family keep the old keyword pattern above.
        # Quoted names, ranges, and native aliases gain no acceptance.
        dataset_name_match = _DATASET_NAME_PATTERN.match(text, index)
        if dataset_name_match is not None:
            name_end = dataset_name_match.end()
            lookahead = name_end
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] == "(":
                close_pending()
                pending_name = dataset_name_match.group(0)
                pending_start = index
                pending_end = name_end
                index = lookahead
                continue
            close_pending()
            pending_name = dataset_name_match.group(0)
            pending_start = index
            pending_end = name_end
            index = name_end
            continue
        if character.isspace():
            index += 1
            continue
        close_pending()
        raise _DatasetStatementFault(
            f"Unsupported statement syntax near position {index}: {character!r}"
        )

    close_pending()
    return refs, statement_options


# --- assignment / sum statement scanning ------------------

_NAME_TOKEN_PATTERN = re.compile(r"[A-Za-z_][\w\.]*")
TargetDelimiter: TypeAlias = Literal["(", "[", "{"]
_BRACKET_PAIRS: tuple[
    tuple[TargetDelimiter, Literal[")", "]", "}"]], ...
] = (("(", ")"), ("[", "]"), ("{", "}"))


class _StatementEntryFault(Exception):
    """Internal statement-shape fault (legacy cause family, typed plumbing)."""

    def __init__(self, cause: str) -> None:
        super().__init__(cause)
        self.cause = cause


_FaultStatementKind: TypeAlias = Literal[
    "ASSIGN", "SUM", "LABEL", "RENAME", "SKIPPED", "IF", "ELSE IF", "ELSE", "DO", "ARRAY"
]


def _require_execution_statement_span(
    source: SourceDocument, statement: ProgramStatementRegion | SourceSpan
) -> tuple[SourceSpan, str]:
    if isinstance(statement, ProgramStatementRegion):
        span = statement.span
    elif isinstance(statement, SourceSpan):
        span = statement
    else:
        raise TypeError(
            "statement must be a ProgramStatementRegion or SourceSpan, "
            f"got {type(statement).__name__}"
        )
    if span.source_id != source.source_id:
        raise ValueError(
            f"statement span source_id {span.source_id!r} does not match "
            f"document source_id {source.source_id!r}"
        )
    if span.start < 0 or span.end > len(source.text) or span.start > span.end:
        raise ValueError(
            f"statement span {span.start}:{span.end} lies outside the document"
        )
    return span, source.text[span.start : span.end]


@dataclass(frozen=True)
class _SeparatorScan:
    separator: int | None
    candidate_seen: bool


def _scan_top_level_separator(text: str, character: str) -> _SeparatorScan:
    """Find the first raw separator occurrence, native split_once parity.

    The legacy native binding (``parse_assignment``/``parse_sum_statement``,
    native/limulus_native/src/runtime/mod.rs L209/L356) splits at the raw
    first ``=``/``+`` ignoring quotes, brackets, and comparison operators;
    statement-shape gates belong to the legacy grammar classification.
    """

    found = text.find(character)
    return _SeparatorScan(
        separator=found if found >= 0 else None,
        candidate_seen=found >= 0,
    )


def _assignment_target_and_rhs(
    source: SourceDocument,
    span: SourceSpan,
    statement_text: str,
    analysis_text: str,
) -> tuple[NameTargetSyntax | ArrayElementTargetSyntax, SourceSpan]:
    """Split one assignment statement into a typed target and the RHS span.

    ``analysis_text`` is the request-local normalized statement slice
   : the separator scan, the target
    trimming, and the structured target name read from it, so a tab
    separator between target tokens becomes one space (``x\ty`` keeps the
    structured name ``x y``).  ``statement_text`` (the raw slice) only
    feeds the legacy cause strings; every codepoint offset is shared,
    so the returned spans stay valid against the raw document.
    """

    scan = _scan_top_level_separator(analysis_text, "=")
    if scan.separator is None:
        if scan.candidate_seen:
            # An ``=`` exists (comparison alias / compound operator / inside
            # unbalanced brackets): the legacy assign statement shape fails
            # closed without pretending the separator was found.
            raise _StatementEntryFault(
                f"invalid assignment statement: {statement_text.strip()}"
            )
        raise ValueError(
            "statement is not an assignment (no assignment separator): "
            f"{statement_text.strip()!r}"
        )
    separator = scan.separator

    left_start = 0
    left_end = separator
    while left_start < left_end and analysis_text[left_start].isspace():
        left_start += 1
    while left_end > left_start and analysis_text[left_end - 1].isspace():
        left_end -= 1
    if left_start >= left_end:
        raise _StatementEntryFault(
            f"invalid assignment statement: {statement_text.strip()}"
        )
    token = analysis_text[left_start:left_end]
    target = _build_assignment_target(source, span, token, left_start)
    return target, _expression_rhs_span(source, span, statement_text, separator)


def _build_assignment_target(
    source: SourceDocument, span: SourceSpan, token: str, token_offset: int
) -> NameTargetSyntax | ArrayElementTargetSyntax:
    base = span.start
    # Legacy array-reference binding parity (parse_array_reference_token,
    # mod.rs L311): the trimmed target token becomes an array element
    # reference when a candidate opener exists (candidate order (), [], {})
    # and the token ends with the matching closer with nonempty trimmed
    # name/index parts; there is no identifier gate on the name part.  Every
    # other shape, including unbalanced brackets and quoted residue, stays
    # an opaque scalar name target with the whole trimmed token.
    for open_char, close_char in _BRACKET_PAIRS:
        open_index = token.find(open_char)
        if open_index <= 0 or not token.endswith(close_char):
            continue
        name = token[:open_index].strip()
        if not name or not token[open_index + 1 : len(token) - 1].strip():
            continue
        index_start = token_offset + open_index + 1
        return ArrayElementTargetSyntax(
            name=name,
            delimiter=open_char,
            index=_deferred_index_expression(
                source, span, index_start, token_offset + len(token) - 1
            ),
            span=source.span(base + token_offset, base + token_offset + len(token)),
        )
    return NameTargetSyntax(
        name=token, span=source.span(base + token_offset, base + token_offset + len(token))
    )


def _deferred_index_expression(
    source: SourceDocument,
    span: SourceSpan,
    index_start: int,
    index_end: int,
) -> ExpressionSyntax:
    """Parse the RAW bounded index slice (open+1 .. terminal closer).

    The index span covers every source character between the first opener
    and the terminal closer before any trim, computed from the bounded
    statement token only: no balanced-close search and no scan beyond the
    caller's span.  The shared deferred general-expression entry turns a
    failing index (comma lexemes, unterminated quotes, ...) into a
    whole-expression descriptor inside the typed target.
    """

    index_span = source.span(span.start + index_start, span.start + index_end)
    result = parse_general_expression(source, index_span)
    return result.expression


def _expression_rhs_span(
    source: SourceDocument, span: SourceSpan, statement_text: str, separator: int
) -> SourceSpan:
    start = separator + 1
    end = len(statement_text)
    while start < end and statement_text[start].isspace():
        start += 1
    while end > start and statement_text[end - 1].isspace():
        end -= 1
    if start >= end:
        raise _StatementEntryFault(
            f"invalid assignment statement: {statement_text.strip()}"
        )
    return source.span(span.start + start, span.start + end)


def _sum_target_and_increment(
    source: SourceDocument,
    span: SourceSpan,
    statement_text: str,
    analysis_text: str,
) -> tuple[NameTargetSyntax, SourceSpan]:
    """Split one SUM statement into a bare-name target and the increment span.

    ``analysis_text`` is the request-local normalized statement slice
   : the separator scan, the structured
    target name, and the increment trimming read from it; the legacy cause
    strings keep the raw slice.  Every codepoint offset is shared.
    """

    scan = _scan_top_level_separator(analysis_text, "+")
    if scan.separator is None:
        if scan.candidate_seen:
            raise _StatementEntryFault(
                f"Invalid SUM statement: {statement_text.strip()}"
            )
        raise ValueError(
            f"statement is not a SUM statement (no + separator): {statement_text.strip()!r}"
        )
    separator = scan.separator
    start = 0
    end = separator
    while start < end and analysis_text[start].isspace():
        start += 1
    while end > start and analysis_text[end - 1].isspace():
        end -= 1
    if start >= end:
        raise _StatementEntryFault(f"Invalid SUM statement: {statement_text.strip()}")
    token = analysis_text[start:end]
    # Legacy SUM statement shape check: datastep.lark's sum_stmt alternative
    # only classifies a bare `[A-Za-z_][\w\.]*` target followed by `+`; texts
    # like `x[1] + 2` were never SUM statements, so the limited entry refuses
    # them instead of fabricating an accepted SUM payload (the structural
    # structural classification owner handles the unsupported statement).
    if _NAME_TOKEN_PATTERN.fullmatch(token) is None:
        raise ValueError(
            f"statement is not a SUM statement (target is not a bare name): "
            f"{statement_text.strip()!r}"
        )
    target = NameTargetSyntax(
        name=token, span=source.span(span.start + start, span.start + end)
    )
    increment_start = separator + 1
    increment_end = len(statement_text)
    while increment_start < increment_end and statement_text[increment_start].isspace():
        increment_start += 1
    while increment_end > increment_start and statement_text[increment_end - 1].isspace():
        increment_end -= 1
    if increment_start >= increment_end:
        raise _StatementEntryFault(f"Invalid SUM statement: {statement_text.strip()}")
    return target, source.span(span.start + increment_start, span.start + increment_end)


def _fault_statement(
    source: SourceDocument,
    span: SourceSpan,
    kind: _FaultStatementKind,
    cause: str,
    apply_references: tuple[ApplyReferenceSyntax, ...],
) -> StatementSyntax:
    return StatementSyntax(
        kind=kind,
        payload=DeferredStatementError(
            error_kind="parse_fault",
            native_cause=cause,
            span=span,
            display_text=source.text[span.start : span.end].strip(),
        ),
        span=span,
        apply_references=apply_references,
    )


# --- name-list keyword gate and token extraction -----------

_NameListStatementKind: TypeAlias = Literal["KEEP", "DROP", "BY", "RETAIN"]

# Legacy prefix classification family (datastep.lark keep/drop/by/retain
# alternatives): the earliest keyword prefix decides the statement kind.
# Longer keywords come first; ``by`` cannot prefix the others, so this stays
# unambiguous while mirroring the legacy permutation-free prefix match.
_NAME_LIST_KEYWORDS: tuple[tuple[str, _NameListStatementKind], ...] = (
    ("retain", "RETAIN"),
    ("keep", "KEEP"),
    ("drop", "DROP"),
    ("by", "BY"),
)


def _name_list_keyword(
    statement_text: str,
) -> tuple[str, _NameListStatementKind] | None:
    """Classify one statement text by its legacy keyword prefix.

    The comparison is case-insensitive over the statement prefix, matching
    the legacy ``[kK][eE][eE][pP][^;]*``-style classification family.  No
    word-boundary check is added: the keyword kind decision is provisional
    here and the final structural classification owner is the classification
    pass.
    """

    lowered = statement_text.lower()
    for keyword, kind in _NAME_LIST_KEYWORDS:
        if lowered.startswith(keyword):
            return keyword, kind
    return None


def _extract_name_list_names(
    source: SourceDocument,
    span: SourceSpan,
    statement_text: str,
    keyword: str,
) -> tuple[NameSyntax, ...]:
    """Whitespace tokenization over the normalized analysis statement text.

    ``statement_text`` is the request-local normalized statement slice
   : tabs outside quoted literals and
    comments already became one space, so the gate below separates shapes
    the way the space spelling does.  Gate (runtime ``mod.rs``
    ``statement_body``; on the RAW slice the same helper keeps the legacy
    whole-body fall-through that the name-list consumer views project): the
    lowercased gate
    accepts ``keyword`` plus one literal trailing space, or the bare
    keyword for an empty body; every other shape (glued keyword, newline
    separator) keeps the WHOLE statement text as the body.  The
    token scan then splits that body on Python ``str.isspace`` separators
    exactly like ``str.split``: tokens stay opaque (no identifier gate, no
    quote decoding, no case change), repeated names keep their positions,
    and a whitespace-only body yields no entries.  This matches the native
    executor for ordinary blanks; interior U+001C–U+001F separators would
    differ from Rust ``char::is_whitespace``, but such separators never
    reach this entry through the region pass (the region grammar fragments
    at them), and the full acceptance edge is the structural pass, not the
    name-list entry scope.  Each token keeps the exact codepoint span of its source slice
    inside the statement region.
    """

    lowered = statement_text.lower()
    if lowered == keyword:
        return ()
    if lowered.startswith(f"{keyword} "):
        body_start = len(keyword)
    else:
        body_start = 0
    body_end = len(statement_text)
    index = body_start
    names: list[NameSyntax] = []
    while index < body_end:
        if statement_text[index].isspace():
            index += 1
            continue
        token_start = index
        while index < body_end and not statement_text[index].isspace():
            index += 1
        names.append(
            NameSyntax(
                name=statement_text[token_start:index],
                span=source.span(span.start + token_start, span.start + index),
            )
        )
    return tuple(names)


# --- dual legacy consumer views over one typed statement ----

_NAME_LIST_KINDS: frozenset[str] = frozenset({"KEEP", "DROP", "BY", "RETAIN"})


@dataclass(frozen=True)
class NameListConsumerViews:
    """The exact legacy consumer name views of one typed name-list statement.

    ``native_names`` is the sequence the executing native statement adapter
    receives today (the ``statement_body`` gate with whitespace tokens;
    identical to the typed payload of the name-list entry).  ``planning_names``
    is the legacy Python consumer promise applied by validation / planning
    / hand-off owners on the old public path:
    ``statement_text[len(kind):].strip().split()``.  The old public path
    used both owners, so neither view replaces the other and no engine
    preference is stored anywhere: consumers project the view they own
    from the same typed statement.
    """

    native_names: tuple[str, ...]
    planning_names: tuple[str, ...]


def name_list_consumer_views(statement: StatementSyntax) -> NameListConsumerViews:
    """Project both legacy consumer views from one typed name-list statement.

    Pure structural atom projection: the input
    is only the typed statement — kind plus the
    ``NameListStatementSyntax`` names with their absolute spans.  No
    ``SourceDocument``, no raw statement text, and no engine preference is
    consulted, so display text and source identity cannot influence the
    result.  The views derive exactly from the stored atoms:

    - native view: the payload token sequence itself (the name-list entry already
      mirrors the native ``statement_body`` extraction);
    - planning view: the legacy Python consumer slice
      ``text[len(kind):].strip().split()`` reconstructed by cutting at
      ``statement.span.start + len(kind)`` — a name whose span ends at or
      before the boundary is discarded (it lies inside the keyword prefix,
      e.g. the ``keep`` token of ``keep\\ta``), a name straddling the
      boundary contributes its typed-name suffix from the boundary
      (``keepx`` keeps ``x``), and later names pass through unchanged.

    For every statement the name-list entry accepts, this reproduces both
    recorded truths without re-reading any source text.
    """

    if not isinstance(statement, StatementSyntax):
        raise TypeError(
            "statement must be a StatementSyntax, got "
            f"{type(statement).__name__}"
        )
    if statement.kind not in _NAME_LIST_KINDS:
        raise ValueError(
            "name-list consumer views require a KEEP/DROP/BY/RETAIN "
            f"statement, got kind {statement.kind!r}"
        )
    if not isinstance(statement.payload, NameListStatementSyntax):
        raise ValueError(
            "name-list consumer views require a NameListStatementSyntax "
            f"payload, got {type(statement.payload).__name__}"
        )
    boundary = statement.span.start + len(statement.kind)
    native = tuple(name.name for name in statement.payload.names)
    planning: list[str] = []
    for name in statement.payload.names:
        if name.span.end <= boundary:
            continue
        if name.span.start < boundary:
            planning.append(name.name[boundary - name.span.start :])
        else:
            planning.append(name.name)
    return NameListConsumerViews(native_names=native, planning_names=tuple(planning))


# --- attribute and skipped statement scanning ---------------
#
# OUTPUT statements stay the dataset-reference declaration owner
# (``parse_dataset_statement`` → ``OutputStatementSyntax`` with parsed
# refs/options).  The legacy opaque-token output contract (Python ``io``
# effective targets ``tokens[1]`` + ``work.`` normalize; native ALL
# ``statement_body`` tokens overriding the resolved targets — see
# ``limulus/io.py`` ``extract_output_targets`` and runtime ``mod.rs``
# L1055/L1114) is NOT projected from the typed atoms here: option-group
# refs and gate-fail keyword spellings carry no raw atom field in the
# the models, and the runtime target contract for option-bearing
# output statements is the old public observation (those shapes
# fail at the execute stage; wiring the declared names would silently
# change that behavior).  Nothing below re-parses OUTPUT statements.

def _legacy_statement_effective_text(statement_text: str) -> str:
    """The legacy per-rule segment: truncate at the first raw semicolon.

    The legacy statement-family regexes (``label_stmt``/``rename_stmt`` fo
    rmat ``/…[^;]*/``) end their match at the first raw ``;`` character,
    including one inside a quoted literal, and every derived diagnostic
    (message text, body, location) used that truncated segment.  Typed
    regions only end before an UNquoted ``;``, so this helper
    reproduces the legacy truncation exactly without any reparse.
    """

    cut = statement_text.find(";")
    return statement_text[:cut] if cut >= 0 else statement_text


def _build_label_pairs_parser() -> Lark:
    """Build (once) the label-pairs grammar with exact pair spans.

    Same grammar file the legacy ``label_stmt`` parser owns
    (``label_pairs.lark``); ``propagate_positions=True`` supplies the
    ``NAME = STRING`` pair spans for the typed attribute atoms.  A
    mandatory failure raises ``ParserInitializationError`` instead of
    hiding a backend problem.
    """

    if Lark is None:
        raise ParserInitializationError(
            "lark is not available; the label pairs grammar cannot initialize."
        )
    grammar_path = Path(__file__).resolve().parents[1] / "grammar" / "label_pairs.lark"
    try:
        grammar = grammar_path.read_text(encoding="utf-8")
    except Exception as error:
        raise ParserInitializationError(
            f"label pairs grammar initialization failed: {error}"
        ) from error
    try:
        return Lark(grammar, start="start", parser="lalr", propagate_positions=True)
    except Exception as error:
        raise ParserInitializationError(
            f"label pairs grammar initialization failed: {error}"
        ) from error


def _scan_label_pairs(
    frontend: "LarkProgramFrontend",
    source: SourceDocument,
    span: SourceSpan,
    statement_text: str,
) -> "tuple[LabelAttributeSyntax, ...] | _StatementEntryFault":
    """Scan one LABEL body into typed attribute pairs (legacy dict truth).

    ``statement_text`` is the request-local normalized analysis slice of
    the statement: separator tabs
    already became one space while quoted label values kept their tab
    characters and every codepoint offset maps 1:1 onto the raw region.
    Mirrors ``_parse_label_statement``: the effective segment truncates at
    the first raw ``;``; an empty body stays a no-op payload; the
    ``label_pairs`` grammar maps ``NAME = STRING`` values after
    ``_strip_quoted_value`` (outer quotes only, doubled quotes raw); the
    repeated-name order keeps the first-occurrence position and pair span
    with the LAST assigned value; grammar faults keep the exact legacy
    cause strings.
    """

    effective_text = _legacy_statement_effective_text(statement_text)
    body = effective_text[len("label") :].strip()
    if not body:
        return ()
    keyword_offset = span.start + len("label")
    try:
        parsed = frontend._get_label_pairs_parser().parse(effective_text[len("label") :])
    except UnexpectedInput:
        return _StatementEntryFault(
            f"Unsupported LABEL statement syntax: {effective_text.strip()}"
        )
    except Exception as error:  # pragma: no cover - defensive grammar boundary
        raise ParserInitializationError(
            f"label pairs parse failed unexpectedly: {error}"
        ) from error

    ordered: dict[str, list] = {}
    for pair in getattr(parsed, "children", ()):
        if getattr(pair, "data", None) != "label_pair":
            continue
        if len(getattr(pair, "children", ())) != 3:
            return _StatementEntryFault(
                f"Unsupported LABEL statement syntax: {effective_text.strip()}"
            )
        name_token, _, value_token = pair.children
        name = str(name_token)
        value = _strip_quoted_value(str(value_token))
        pair_meta = getattr(pair, "meta", None)
        if pair_meta is None:
            return _StatementEntryFault(
                f"Unsupported LABEL statement syntax: {effective_text.strip()}"
            )
        pair_span = source.span(
            keyword_offset + getattr(pair_meta, "start_pos", 0),
            keyword_offset + getattr(pair_meta, "end_pos", 0),
        )
        if name in ordered:
            ordered[name][0] = value
            continue
        ordered[name] = [value, pair_span]  # first-occurrence position and span
    if not ordered:
        return _StatementEntryFault(
            f"Unsupported LABEL statement syntax: {effective_text.strip()}"
        )
    return tuple(
        LabelAttributeSyntax(name=name, label=value, span=pair_span)
        for name, (value, pair_span) in ordered.items()
    )


def _scan_rename_pairs(
    frontend: "LarkProgramFrontend",
    statement_text: str,
) -> "tuple[tuple[str, str], ...]":
    """Scan one RENAME body through the shared rename-pairs parser.

    ``statement_text`` is the request-local normalized analysis slice of
    the statement: separator tabs
    already became one space.  Mirrors ``_parse_rename_statement``: the
    effective segment truncates at the first raw ``;``, the stripped body
    is parsed by the same pinned ``rename_pairs`` implementation the
    dataset-option path uses, and every invalid shape keeps the legacy
    ``Invalid rename mapping: {body}`` cause.  The pairs follow the legacy
    dict semantics (first-occurrence order, last value).
    """

    effective_text = _legacy_statement_effective_text(statement_text)
    body = effective_text[len("rename") :].strip()
    try:
        return _parse_rename_pairs(frontend, body)
    except _DatasetStatementFault as fault:
        raise _StatementEntryFault(fault.cause) from None


_SKIPPED_KEYWORD_PREFIXES: tuple[str, ...] = (
    "informat",
    "length",
    "attrib",
    "format",
    "call",
)


def _skipped_keyword_prefix(statement_text: str) -> str | None:
    """The legacy skipped-family keyword prefix of one statement text.

    Mirrors the legacy ``skipped_stmt`` alternatives: ``length``/
    ``attrib``/``format``/``informat`` regexes have no word boundary
    (``formatted x`` stays skipped) while ``call_stmt`` requires one
    (``callable = x`` was an assignment statement).  Returns ``None`` when
    no skipped prefix matches, leaving the classification owner to
    reject the statement; no variant fields are invented for the payload.
    """

    lowered = statement_text.lower()
    for keyword in _SKIPPED_KEYWORD_PREFIXES:
        if not lowered.startswith(keyword):
            continue
        if keyword == "call":
            next_index = len(keyword)
            if next_index < len(statement_text):
                next_character = statement_text[next_index]
                if next_character.isalnum() or next_character == "_":
                    return None
        return keyword
    return None

# --- IF / ELSE condition and inline action structuring -----
#
# Branch statements keep the flat statement order of their block: no new
# control-flow tree is built here, and no branch is ever executed.  The
# condition / inline-action split reuses the exact legacy facade grammar
# ``if_structured.lark`` (quote-aware THEN binding; a statement without a
# top-level ``then`` stays a subset IF) and the legacy variant regexes for
# ``THEN DO``, both read from the request-local normalized analysis text,
# so every span stays anchored to the raw document.
# Inline actions are structured with the existing statement entries
# (assignment / array target / SUM / OUTPUT references / STOP / DELETE) at
# exact original spans, THEN DO keeps a bare no-loop-bound ``DO`` payload
# (the DO bounds belong to the DO boundary entry), and unsupported inline actions
# keep the lazily failing legacy runtime cause family
# (``unsupported inline action: ...``) as a whole-statement
# ``DeferredStatementError`` instead of becoming valid SKIPPED statements
# or program-stopping syntax errors.

_IfStatementKind: TypeAlias = Literal["IF", "ELSE IF"]

# Legacy statement-kind gates (datastep.lark equivalence, quote-unaware):
# ``if`` needs a word boundary (glued ``ifx`` was an assignment),
# ``ELSE IF`` needs whitespace between ``else`` and ``if`` plus any ``then``
# (glued ``elseif`` variants were classified elsewhere), and ``ELSE`` needs
# the ``else`` prefix without a glued word character (``elsefoo`` was an
# assignment; ``else:`` / ``else\n...`` stayed ELSE).
_IF_KEYWORD_PATTERN = re.compile(r"\Aif\b", re.IGNORECASE)
_ELSE_IF_KEYWORD_PATTERN = re.compile(r"\Aelse\s+if\b", re.IGNORECASE)
_ELSE_KEYWORD_PATTERN = re.compile(r"\Aelse\b", re.IGNORECASE)
_THEN_KEYWORD_PATTERN = re.compile(r"\bthen\b", re.IGNORECASE)
_THEN_DO_PATTERN = re.compile(r"\bthen\s+do\b", re.IGNORECASE)


@dataclass(frozen=True)
class _IfStatementSplit:
    """The legacy-facade split of one IF / ELSE IF statement slice.

    ``condition_span`` is ``None`` exactly when the legacy facade produced
    no ``if_spec`` (grammar refusals, missing condition or action pieces);
    the statement then keeps the lazily failing ``invalid IF statement: ...``
    cause family.  ``action_span`` is ``None`` for a subset IF.
    ``condition_edits`` carries the display
    edits collected from the SAME initial ``if_structured`` condition
    tokens (symbol equality aliases and logic words only); they exist only
    for the legacy condition display projection.
    """

    kind: _IfStatementKind
    condition_span: SourceSpan | None
    action_span: SourceSpan | None
    is_subset: bool
    is_then_do: bool
    condition_edits: tuple[tuple[SourceSpan, str], ...] = ()


def _empty_if_split(kind: _IfStatementKind) -> _IfStatementSplit:
    """The legacy no-``if_spec`` outcome of one IF / ELSE IF split."""

    return _IfStatementSplit(
        kind=kind,
        condition_span=None,
        action_span=None,
        is_subset=True,
        is_then_do=False,
    )


def _split_if_statement(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    span: SourceSpan,
    analysis_text: str,
) -> _IfStatementSplit | None:
    """Split one IF / ELSE IF statement into legacy-facade condition/action.

    The legacy facade (``_parse_if_statement_spec``) parses the stripped
    statement slice (for ``else if``: the slice after a lowered ``else ``
    prefix; glued ``elseif`` slices never matched and stayed invalid) with
    the ``if_structured`` grammar.  The split is quote-aware because THEN
    is a keyword token there, and it accepts odd forms structurally (for
    example ``if then do;`` became a subset IF whose condition is the text
    ``then do``).  Those accepted behaviors are reproduced, not re-guessed.
    """

    kind: _IfStatementKind | None
    if (
        _ELSE_IF_KEYWORD_PATTERN.match(analysis_text) is not None
        and _THEN_KEYWORD_PATTERN.search(analysis_text) is not None
    ):
        kind = "ELSE IF"
    elif _IF_KEYWORD_PATTERN.match(analysis_text) is not None:
        kind = "IF"
    else:
        return None
    assert kind is not None
    empty = _empty_if_split(kind)

    lead = len(analysis_text) - len(analysis_text.lstrip())
    body = analysis_text[lead:].rstrip()
    base = span.start + lead
    parse_source = body
    if kind == "ELSE IF":
        if not body.lower().startswith("else "):
            # Legacy glued ``else*`` prefixes kept the whole slice unparsed
            # and the statement stayed lazily invalid.
            return empty
        rest = body[4:]
        inner_lead = len(rest) - len(rest.lstrip())
        parse_source = rest[inner_lead:]
        base = base + 4 + inner_lead

    parser = frontend._get_if_structured_parser()
    try:
        parsed = parser.parse(parse_source)
    except UnexpectedInput:
        return empty
    except Exception as error:  # pragma: no cover - defensive grammar boundary
        raise ParserInitializationError(
            f"if statement parse failed unexpectedly: {error}"
        ) from error

    root = _root_if_tree(parsed)
    children = list(getattr(root, "children", ()))
    if len(children) < 2:
        return empty

    condition_node = children[1]
    condition_meta = getattr(condition_node, "meta", None)
    condition_start = getattr(condition_meta, "start_pos", None)
    condition_end = getattr(condition_meta, "end_pos", None)
    if (
        not isinstance(condition_start, int)
        or not isinstance(condition_end, int)
        or condition_end < condition_start
    ):
        return empty
    condition_span = source.span(base + condition_start, base + condition_end)
    condition_edits = _condition_display_edits(
        source, base, condition_node, condition_start, condition_end
    )

    if len(children) == 2:
        # Subset IF: no top-level THEN token outside quotes.
        return _IfStatementSplit(
            kind=kind,
            condition_span=condition_span,
            action_span=None,
            is_subset=True,
            is_then_do=False,
            condition_edits=condition_edits,
        )

    action_node = children[3] if len(children) >= 4 else None
    action_meta = getattr(action_node, "meta", None) if action_node is not None else None
    action_start = getattr(action_meta, "start_pos", None)
    action_end = getattr(action_meta, "end_pos", None)
    if (
        not isinstance(action_start, int)
        or not isinstance(action_end, int)
        or action_end < action_start
    ):
        return empty
    action_text = parse_source[action_start:action_end].strip()
    if not action_text:
        # The legacy facade produced no spec for an empty THEN action.
        return empty
    action_span = _exact_trimmed_span(source, base, parse_source, action_start, action_end)
    is_then_do = (
        _THEN_DO_PATTERN.search(analysis_text) is not None or action_text.lower() == "do"
    )
    return _IfStatementSplit(
        kind=kind,
        condition_span=condition_span,
        action_span=action_span,
        is_subset=False,
        is_then_do=is_then_do,
        condition_edits=condition_edits,
    )


# --- initial IF condition display edits ---------------------


def _condition_display_edits(
    source: SourceDocument,
    base: int,
    condition_node: object,
    condition_start: int,
    condition_end: int,
) -> tuple[tuple[SourceSpan, str], ...]:
    """Collect the legacy condition display edits from the initial split.

    The SAME ``if_structured`` parse that
    splits the IF statement already holds the condition tokens, so this
    records only the display replacements the old facade performed on
    them — symbol equality aliases (``=`` -> ``==``, ``^=``/``~=``/``¬=``
    -> ``!=``) and logic words (AND/OR/NOT -> lower case).  Quoted
    literals, case, whitespace, transparent groups, and MOD stay raw; no
    COMPARE_WORD / other token family is normalized and no re-lex or
    re-parse happens.  Edits are ordered ascending, non-overlapping, and
    anchored to the raw document inside the condition bounds.
    """

    edits: list[tuple[SourceSpan, str]] = []
    for token in sorted(
        _collect_condition_tokens(condition_node),
        key=lambda item: getattr(item, "start_pos", 0),
    ):
        token_start = getattr(token, "start_pos", None)
        token_end = getattr(token, "end_pos", None)
        if not isinstance(token_start, int) or not isinstance(token_end, int):
            continue
        if token_start < condition_start or token_end > condition_end:
            continue
        replacement = _condition_token_replacement(token)
        if replacement is None or replacement == str(token):
            # Identity edits (already-lower logic words) change nothing and
            # are never stored.
            continue
        edits.append((source.span(base + token_start, base + token_end), replacement))
    return tuple(edits)


def _collect_condition_tokens(node: object) -> list[object]:
    """Legacy facade ``_collect_tokens`` parity over the condition subtree."""

    tokens: list[object] = []
    for child in getattr(node, "children", ()):
        if hasattr(child, "children"):
            tokens.extend(_collect_condition_tokens(child))
            continue
        tokens.append(child)
    return tokens


def _condition_token_replacement(token: object) -> str | None:
    """The display replacement of one condition token, or ``None`` when raw.

    The exact legacy symbol/logic vocabulary of the old
    ``_normalize_if_token_text`` owner minus every token family the design
    leaves raw (COMPARE_WORD words, MOD, and everything else).
    """

    token_type = str(getattr(token, "type", ""))
    token_text = str(token)
    if token_type == "SYMBOL_OP":
        if token_text == "=":
            return "=="
        if token_text in {"^=", "~=", "¬="}:
            return "!="
        return None
    if token_type == "BOOL_WORD":
        lowered = token_text.lower()
        return lowered if lowered in {"and", "or", "not"} else None
    return None


def _root_if_tree(parsed: object) -> object:
    """Legacy facade ``_root_tree`` parity for the ``if_stmt`` rule."""

    if getattr(parsed, "data", None) == "if_stmt":
        return parsed
    children = getattr(parsed, "children", ())
    if children and getattr(children[0], "data", None) == "if_stmt":
        return children[0]
    return parsed


def _faulted_if_statement(
    source: SourceDocument,
    span: SourceSpan,
    kind: _IfStatementKind,
    apply_references: tuple[ApplyReferenceSyntax, ...],
) -> StatementSyntax:
    """The lazily faulted whole statement for a legacy no-spec IF split."""

    return _fault_statement(
        source,
        span,
        kind,
        f"invalid IF statement: {source.text[span.start : span.end].strip()}",
        apply_references,
    )


def _build_if_statement(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    span: SourceSpan,
    split: _IfStatementSplit,
    apply_references: tuple[ApplyReferenceSyntax, ...],
) -> StatementSyntax:
    """Build the typed IF / ELSE IF statement from the legacy-facade split."""

    if split.condition_span is None:
        return _faulted_if_statement(source, span, split.kind, apply_references)
    condition = parse_general_expression(source, split.condition_span).expression
    if split.action_span is None:
        payload: IfStatementSyntax = IfStatementSyntax(
            condition=condition,
            is_subset=True,
            condition_display_edits=split.condition_edits,
            condition_display_span=split.condition_span,
        )
    else:
        if split.is_then_do:
            # THEN DO keeps the legacy flag plus a bare DO payload; no loop
            # boundary is parsed here and the block body
            # stays inside the flat statement list.
            action: StatementSyntax | DeferredStatementError = StatementSyntax(
                kind="DO", payload=DoStatementSyntax(), span=split.action_span
            )
        else:
            action = _structure_inline_action(frontend, source, split.action_span)
        payload = IfStatementSyntax(
            condition=condition,
            then_action=action,
            is_subset=False,
            is_then_do=split.is_then_do,
            condition_display_edits=split.condition_edits,
            condition_display_span=split.condition_span,
        )
    return StatementSyntax(
        kind=split.kind,
        payload=payload,
        span=span,
        apply_references=apply_references,
    )


@dataclass(frozen=True)
class _ElseBody:
    """The legacy ``statement_body(text, \"else\")`` contract of an ELSE slice."""

    span: SourceSpan | None
    text: str


_BareElseBody = _ElseBody(span=None, text="")


def _else_body_span(
    source: SourceDocument,
    span: SourceSpan,
    analysis_text: str,
) -> _ElseBody | None:
    """Resolve the inline ELSE body with legacy ``statement_body`` semantics.

    ``else`` alone is a bare ELSE (``span is None``); a lowered ``else ``
    prefix carries the trimmed remainder; any other glued ``else`` prefix
    without a word boundary (``else:new = 1``, ``else\\nx = 1``) stayed an
    ELSE statement whose body is the whole statement slice, while a glued
    word character (``elsefoo x = 1``) was classified outside the ELSE
    kinds, so this restricted entry rejects it (``None``).
    """

    lowered = analysis_text.lower()
    if lowered == "else":
        return _BareElseBody
    if lowered.startswith("else "):
        rest = analysis_text[4:]
        lead = len(rest) - len(rest.lstrip())
        trail = len(rest) - len(rest.rstrip())
        body_span: SourceSpan = source.span(
            span.start + 4 + lead, span.end - trail
        )
        return _ElseBody(span=body_span, text=rest.strip())
    if _ELSE_KEYWORD_PATTERN.match(analysis_text) is not None:
        return _ElseBody(span=span, text=analysis_text.strip())
    return None


def _structure_else_body(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    body_span: SourceSpan,
    body_text: str,
) -> StatementSyntax | DeferredStatementError:
    """Structure one ELSE body (legacy native inline truth).

    A bare ``do`` body keeps the legacy ELSE DO shape as a no-loop-bound
    ``DO`` payload; everything else follows the shared inline action rules,
    where unsupported bodies return the inner
    ``DeferredStatementError`` action descriptor.
    """

    if body_text.lower() == "do":
        analysis = frontend._analysis_text(source)[body_span.start : body_span.end]
        action_span = _trimmed_occurrence(source, body_span, analysis)
        return StatementSyntax(
            kind="DO", payload=DoStatementSyntax(), span=action_span
        )
    return _structure_inline_action(frontend, source, body_span)


def _trimmed_occurrence(
    source: SourceDocument, span: SourceSpan, analysis: str
) -> SourceSpan:
    """The exact raw-document span of ``analysis.strip()`` inside ``span``."""

    lead = len(analysis) - len(analysis.lstrip())
    trail = len(analysis) - len(analysis.rstrip())
    return source.span(span.start + lead, span.end - trail)


def _exact_trimmed_span(
    source: SourceDocument, base: int, text: str, start: int, end: int
) -> SourceSpan:
    """The exact raw-document span of a trimmed slice ``text[start:end]``."""

    return source.span(
        base + start + (len(text[start:end]) - len(text[start:end].lstrip())),
        base + end - (len(text[start:end]) - len(text[start:end].rstrip())),
    )


def _action_descriptor(
    source: SourceDocument,
    action_span: SourceSpan,
    cause: str,
    diagnostic_display: ParseDiagnosticDisplay | None = None,
) -> DeferredStatementError:
    """The inner deferred-action descriptor for an unstructured action.

    The descriptor is the existing ``DeferredStatementError`` stored on the
    owning ``then_action`` / ``action`` slot itself: its span and display text address the ACTION's own raw
    trimmed slice and the native cause keeps the lazily failing runtime
    family.  Nothing is executed here; the reach-time owner fires the saved
    cause only when the branch reaches this action.
    """

    analysis_slice = source.text[action_span.start : action_span.end]
    trimmed = analysis_slice.strip()
    lead = len(analysis_slice) - len(analysis_slice.lstrip())
    trail = len(analysis_slice) - len(analysis_slice.rstrip())
    descriptor_span = source.span(
        action_span.start + lead, action_span.end - trail
    )
    return DeferredStatementError(
        error_kind="parse_fault",
        native_cause=cause,
        span=descriptor_span,
        display_text=trimmed,
        diagnostic_display=diagnostic_display,
    )


def _structure_inline_action(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    action_span: SourceSpan,
) -> StatementSyntax | DeferredStatementError:
    """Structure one inline branch action (legacy native execute truth).

    The native inline executor (``execute_inline_action``) classifies the
    trimmed action text in this exact order: bare ``delete``, bare
    ``stop``, an ``output`` prefix (with legacy ``statement_body`` body
    semantics), a SUM shape (raw first ``+`` with NO bare-name gate — the
    native inline parse differs from the legacy statement-classification
    regex, ``x = a + b`` inside THEN was a SUM on the opaque name ``x = a``),
    and finally an assignment shape.  Any action that matches NONE of these
    shapes is returned as the inner ``DeferredStatementError`` action
    descriptor with the lazily failing ``unsupported inline action: ...``
    runtime cause family: the owning condition and branch structure
    survive, and ONLY a failed condition/action split becomes the outer
    whole-statement fault.  For a RECOGNIZED action kind (OUTPUT), its own
    references/scan failure keeps the typed ``OUTPUT`` statement and stores
    the parse-side cause as that statement's payload (the reviewed
    ``parse_dataset_statement`` keyword-kind preservation), so later
    output-structure owners still see real OUTPUT structure without
    rescanning the display text.
    """

    analysis = frontend._analysis_text(source)[action_span.start : action_span.end]
    trimmed = analysis.strip()
    unsupported_cause = f"unsupported inline action: {trimmed}"
    if not trimmed:
        return _action_descriptor(source, action_span, unsupported_cause)
    action_statement_span = _trimmed_occurrence(source, action_span, analysis)
    lowered = trimmed.lower()

    if lowered == "delete":
        return StatementSyntax(
            kind="DELETE",
            payload=DeleteStatementSyntax(),
            span=action_statement_span,
        )
    if lowered == "stop":
        return StatementSyntax(
            kind="STOP",
            payload=StopStatementSyntax(),
            span=action_statement_span,
        )

    if lowered.startswith("output"):
        # Inline OUTPUT keeps the same typed reference structure the
        # dataset-reference entry produces for a top-level OUTPUT statement.
        if lowered == "output":
            return StatementSyntax(
                kind="OUTPUT",
                payload=OutputStatementSyntax(),
                span=action_statement_span,
            )
        if lowered.startswith("output "):
            rest = trimmed[6:]
            skip = len(rest) - len(rest.lstrip())
            body_rel = 6 + skip
        else:
            # Glued inline ``outputfoo`` behaved as a named output target.
            body_rel = 0
        body_span = source.span(
            action_statement_span.start + body_rel, action_statement_span.end
        )
        try:
            refs, _statement_options = _scan_dataset_statement_body(
                frontend, source, body_span, 0, "OUTPUT"
            )
        except _DatasetStatementFault as fault:
            # The OUTPUT kind was already recognized; per the
            # deferred-action contract its OWN referencing/scan failure is a
            # statement fault INSIDE the typed OUTPUT statement (mirror of
            # the reviewed ``parse_dataset_statement`` keyword-kind
            # preservation).  No bare descriptor fall-through, no new
            # statement kind, and no fabricated/partial target adoption.
            return StatementSyntax(
                kind="OUTPUT",
                payload=_action_descriptor(
                    source, action_span, fault.cause, fault.display
                ),
                span=action_statement_span,
            )
        return StatementSyntax(
            kind="OUTPUT",
            payload=OutputStatementSyntax(targets=tuple(refs)),
            span=action_statement_span,
        )

    # Native inline SUM shape: raw first ``+`` (quotes ignored), any
    # non-empty left/right parts, no bare-name gate.
    scan = _scan_top_level_separator(analysis, "+")
    if scan.separator is not None:
        separator = scan.separator
        left_token = analysis[:separator].strip()
        right_token = analysis[separator + 1 :].strip()
        if left_token and right_token:
            target = NameTargetSyntax(
                name=left_token,
                span=_exact_trimmed_span(source, action_span.start, analysis, 0, separator),
            )
            increment = parse_general_expression(
                source,
                _exact_trimmed_span(
                    source, action_span.start, analysis, separator + 1, len(analysis)
                ),
            ).expression
            return StatementSyntax(
                kind="SUM",
                payload=SumStatementSyntax(target=target, increment=increment),
                span=action_statement_span,
            )

    # Assignment shape through the reviewed assignment entry (same native split
    # parity: raw first ``=``, no identifier gate on the target token).
    try:
        return frontend.parse_assignment_statement(source, action_statement_span)
    except _StatementEntryFault:
        return _action_descriptor(source, action_span, unsupported_cause)
    except ValueError:
        return _action_descriptor(source, action_span, unsupported_cause)


# --- DO boundary structuring --------------------------------
#
# The flat DO statement's loop variable and boundary expressions travel in
# the typed ``DoStatementSyntax`` payload.  The split reuses the exact
# legacy ``do_loop.lark`` grammar (the old facade ``children == 6`` truth)
# over the request-local normalized analysis text, so
# every span stays anchored to the raw document.  Nothing is executed
# here: no matching-END search, no numeric coercion, no iteration, no host
# callable, no schema inference — the classification entry and the
# runtime keep their reach/execution decisions.  ``BY``/
# ``until`` tails absorbed by the legacy grammar stay invalid general
# slices in the END slot (whole ``DeferredExpressionError``); unsupported
# shapes keep the lazily failing ``unsupported DO statement: ...`` native
# cause family as a whole-statement ``DeferredStatementError`` with the
# raw display slice.

_DO_KEYWORD_PATTERN = re.compile(r"\Ado(?:\Z|[^A-Za-z0-9_])", re.IGNORECASE)


def _root_do_tree(parsed: object) -> object:
    """Legacy facade ``_root_tree`` parity for the ``do_stmt`` rule."""

    if getattr(parsed, "data", None) == "do_stmt":
        return parsed
    children = getattr(parsed, "children", ())
    if children and getattr(children[0], "data", None) == "do_stmt":
        return children[0]
    return parsed


def _build_do_statement(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    span: SourceSpan,
    analysis_text: str,
    apply_references: tuple[ApplyReferenceSyntax, ...],
) -> StatementSyntax:
    """Split one DO slice into a typed ``DoStatementSyntax``` or the saved fault.

    Grammar refusals keep the lazily failing legacy runtime cause family
    (``unsupported DO statement: ...``) as a whole-statement
    ``DeferredStatementError`` with the raw display slice; absolutely no
    loop execution, END matching, or numeric coercion happens here.
    """

    lead = len(analysis_text) - len(analysis_text.lstrip())
    parse_source = analysis_text[lead:].rstrip()
    base = span.start + lead

    def unsupported() -> StatementSyntax:
        return _fault_statement(
            source,
            span,
            "DO",
            f"unsupported DO statement: {source.text[span.start : span.end].strip()}",
            apply_references,
        )

    parser = frontend._get_do_loop_parser()
    try:
        parsed = parser.parse(parse_source)
    except UnexpectedInput:
        return unsupported()
    except Exception as error:  # pragma: no cover - defensive grammar boundary
        raise ParserInitializationError(
            f"DO statement parse failed unexpectedly: {error}"
        ) from error

    root = _root_do_tree(parsed)
    children = list(getattr(root, "children", ()))
    if len(children) != 6:
        return unsupported()

    loop_var_node = children[1]
    # The legacy facade stored the LOOP_VAR token string itself (the token
    # has no node meta); str() reproduces the exact raw variable text.
    loop_var = str(loop_var_node)
    start_node = children[3]
    start_meta = getattr(start_node, "meta", None)
    start_start = getattr(start_meta, "start_pos", None)
    start_end = getattr(start_meta, "end_pos", None)
    end_node = children[5]
    end_meta = getattr(end_node, "meta", None)
    end_start = getattr(end_meta, "start_pos", None)
    end_end = getattr(end_meta, "end_pos", None)
    if (
        start_start is None
        or start_end is None
        or end_start is None
        or end_end is None
        or start_end < start_start
        or end_end < end_start
    ):
        return unsupported()
    start_span = _exact_trimmed_span(source, base, parse_source, start_start, start_end)
    end_span = _exact_trimmed_span(source, base, parse_source, end_start, end_end)
    return StatementSyntax(
        kind="DO",
        payload=DoStatementSyntax(
            loop_var=loop_var,
            start=parse_general_expression(source, start_span).expression,
            end=parse_general_expression(source, end_span).expression,
            # The display spans are the full raw boundary expressions the
            # SAME initial grammar split computed: the legacy view slices them instead of re-scanning
            # for the top-level ``to`` keyword.  A bare DO never sets them.
            boundary_display_spans=(start_span, end_span),
        ),
        span=span,
        apply_references=apply_references,
    )


# --- ARRAY declaration structuring --------------------------
#
# The flat ARRAY declaration travels in the existing typed
# ``ArrayStatementSyntax`` payload (statement span included).  The split
# reuses the exact legacy ``array_declaration.lark`` grammar and the old
# facade child reading over the request-local normalized analysis text.
# The parser performs NO array binding: no registry
# store, no row/runtime ownership, no host callable, no size-versus-count
# enforcement, and no repeat rejection — declaration structure only.
# Shapes the legacy facade never accepted (``(3)`` sizes, bare leading
# numbers, quoted/hyphenated variables, missing names, non-ASCII names)
# keep the lazily failing ``invalid ARRAY declaration: ...`` native cause
# family as a whole-statement ``DeferredStatementError`` with the raw
# display slice.

_ARRAY_KEYWORD_PATTERN = re.compile(r"\Aarray", re.IGNORECASE)


def _root_array_tree(parsed: object) -> object:
    """Legacy facade ``_root_tree`` parity for the ``array_stmt`` rule."""

    if getattr(parsed, "data", None) == "array_stmt":
        return parsed
    children = getattr(parsed, "children", ())
    if children and getattr(children[0], "data", None) == "array_stmt":
        return children[0]
    return parsed


def _build_array_statement(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    span: SourceSpan,
    analysis_text: str,
    apply_references: tuple[ApplyReferenceSyntax, ...],
) -> StatementSyntax:
    """Split one ARRAY slice into the typed declaration or the saved fault.

    Grammar refusals keep the lazily failing legacy runtime cause family
    (``invalid ARRAY declaration: ...``) as a whole-statement
    ``DeferredStatementError`` with the raw display slice; no runtime
    binding, size enforcement, or repeat rejection happens here.
    """

    lead = len(analysis_text) - len(analysis_text.lstrip())
    parse_source = analysis_text[lead:].rstrip()

    def invalid() -> StatementSyntax:
        return _fault_statement(
            source,
            span,
            "ARRAY",
            f"invalid ARRAY declaration: {source.text[span.start : span.end].strip()}",
            apply_references,
        )

    parser = frontend._get_array_declaration_parser()
    try:
        parsed = parser.parse(parse_source)
    except UnexpectedInput:
        return invalid()
    except Exception as error:  # pragma: no cover - defensive grammar boundary
        raise ParserInitializationError(
            f"ARRAY statement parse failed unexpectedly: {error}"
        ) from error

    root = _root_array_tree(parsed)
    children = list(getattr(root, "children", ()))
    if len(children) < 3:
        # The legacy facade returned no spec below the name + one more
        # child; the runtime later fails with the exact cause family.
        return invalid()

    array_name = str(children[1])
    declared_size: int | None = None
    wildcard_size = False
    character_array = False
    variables: tuple[str, ...] = ()
    for child in children[2:]:
        child_data = getattr(child, "data", None)
        if child_data == "wildcard_size":
            wildcard_size = True
            continue
        if child_data == "bracket_size":
            declared_size = int(str(child.children[0]).strip("[] "))
            continue
        if child_data == "numeric_size":
            declared_size = int(str(child.children[0]))
            continue
        if child_data == "character_array":
            character_array = True
            continue
        if child_data == "variable_list":
            variables = tuple(str(token) for token in child.children)

    if not variables:
        # The legacy facade required at least one parsed variable name.
        return invalid()

    return StatementSyntax(
        kind="ARRAY",
        payload=ArrayStatementSyntax(
            span=span,
            array_name=array_name,
            variables=variables,
            declared_size=declared_size,
            wildcard_size=wildcard_size,
            character_array=character_array,
        ),
        span=span,
        apply_references=apply_references,
    )


# --- whole-text parse_program service ----------------------
#
# The public pure entry points are :func:`parse_program` (module level) and
# :meth:`LarkProgramFrontend.parse_program`.  The whole text is consumed in
# ONE upfront region-identification parse (``program_regions.lark``) and every statement inside a block is selected by ONE per-block
# classification parse over the same bounded block slice with the existing
# legacy ``datastep.lark`` initial classification grammar, so the statement
# order and every bounded statement entry prefix gate mirror the legacy
# ``_extract_statement_segments`` truth.  Full-text
# span ownership never shifts between stages: every statement span comes
# from the region pass and slices the RAW ``SourceDocument``, while invalid lazy expressions, IF/DO/ARRAY
# shapes, and statement-WHERE conditions stay interior
# ``DeferredStatementError`` / ``DeferredExpressionError`` payloads of the
# result -- never promoted into a program stop.  The old parse-stage
# stopping families (DATA/SET/MERGE dataset option faults, LABEL, RENAME)
# keep the exact legacy diagnostic code, severity, message, stage,
# full-span, raw-display and ``statement:N`` location rule, and a whole-text grammar failure (region parse or a per-block
# classification parse) gates the whole program to ``program=None`` with
# the legacy ``PARSE_UNSUPPORTED_STATEMENT`` near-line/column diagnostic.  Mandatory grammar initialization remains an explicit
# :class:`ParserInitializationError` and is never
# converted into a diagnostic or an empty result.  No data library, row
# runtime, schema inference, or legacy parser service is imported or
# consulted; no fallback, second backend, or new engine is added.

_CLASSIFICATION_RULE_KINDS: dict[str, str] = {
    "data_stmt": "DATA",
    "set_stmt": "SET",
    "merge_stmt": "MERGE",
    "if_stmt": "IF",
    "if_then_stmt": "IF",
    "if_then_do_stmt": "IF",
    "subset_if_stmt": "IF",
    "else_if_stmt": "ELSE IF",
    "else_if_then_stmt": "ELSE IF",
    "else_if_then_do_stmt": "ELSE IF",
    "else_stmt": "ELSE",
    "where_stmt": "WHERE",
    "by_stmt": "BY",
    "output_stmt": "OUTPUT",
    "drop_stmt": "DROP",
    "keep_stmt": "KEEP",
    "delete_stmt": "DELETE",
    "rename_stmt": "RENAME",
    "retain_stmt": "RETAIN",
    "array_stmt": "ARRAY",
    "stop_stmt": "STOP",
    "do_stmt": "DO",
    "end_stmt": "END",
    "run_stmt": "RUN",
    "sum_stmt": "SUM",
    "assign_stmt": "ASSIGN",
    "skipped_stmt": "SKIPPED",
    "length_stmt": "SKIPPED",
    "attrib_stmt": "SKIPPED",
    "format_stmt": "SKIPPED",
    "label_stmt": "LABEL",
    "informat_stmt": "SKIPPED",
    "call_stmt": "SKIPPED",
}

_SIMPLE_PAYLOAD_SYNTAX: dict[str, type] = {
    "STOP": StopStatementSyntax,
    "DELETE": DeleteStatementSyntax,
    "END": EndStatementSyntax,
    "RUN": RunStatementSyntax,
}


def _build_classification_parser() -> Lark:
    """Build (once) the legacy initial classification grammar.

    Reuses the exact legacy facade grammar
    (``limulus/grammar/datastep.lark``, the ``LarkParserFrontend``
    statement-kind classification parse) for the per-block initial
    classification of the whole-text service, so statement kinds,
    variants, and accepted shapes mirror the legacy ``_RULE_KIND_MAP``
    truth instead of being re-guessed.  A mandatory initialization
    failure raises :class:`ParserInitializationError` (never hidden).
    """

    if Lark is None:
        raise ParserInitializationError(
            "lark is not available; the classification grammar cannot initialize."
        )
    classification_path = (
        Path(__file__).resolve().parents[1] / "grammar" / "datastep.lark"
    )
    try:
        grammar = classification_path.read_text(encoding="utf-8")
    except Exception as error:
        raise ParserInitializationError(
            f"classification grammar initialization failed: {error}"
        ) from error
    try:
        return Lark(
            grammar,
            start="start",
            parser="lalr",
            propagate_positions=True,
        )
    except Exception as error:
        raise ParserInitializationError(
            f"classification grammar initialization failed: {error}"
        ) from error


def _mask_option_inner_parentheses(source: str) -> str:
    """Legacy ``_mask_dataset_option_inner_parentheses`` truth (ported).

    ``datastep.lark`` intentionally remains a lightweight statement
    grammar, but its historical terminal cannot express recursive
    parentheses or escaped/quoted delimiters.  For DATA/SET/MERGE
    statement references this strategy preserves the first option-group
    pair and replaces only inner delimiters with spaces.  The replacement
    is length preserving; extraction and diagnostics keep using the
    original source text.  The structural pass ports this narrow legacy helper
    instead of importing the legacy parser module (which pulls the
    production dependency chain the initial frontend must avoid); block
    slices start at a statement keyword so the statement-boundary state
    machine reproduces the legacy whole-text behavior block by block.
    """

    if not source:
        return source

    masked = list(source)
    index = 0
    statement_keyword: str | None = None
    at_statement_start = True
    quote: str | None = None

    while index < len(source):
        character = source[index]

        if quote is not None:
            if character == "\\" and index + 1 < len(source):
                index += 2
                continue
            if character == quote:
                if index + 1 < len(source) and source[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue

        if character in {"'", '"'}:
            quote = character
            index += 1
            continue

        if character == ";":
            statement_keyword = None
            at_statement_start = True
            index += 1
            continue

        if at_statement_start:
            if character.isspace():
                index += 1
                continue
            keyword_match = re.match(r"[A-Za-z_][\w]*", source[index:])
            if keyword_match is not None:
                statement_keyword = keyword_match.group(0).lower()
                at_statement_start = False
                index += len(keyword_match.group(0))
                continue
            at_statement_start = False

        if (
            character == "("
            and statement_keyword in {"data", "set", "merge"}
            and _mask_identifier_before(source, index) is not None
        ):
            end = _mask_matching_parenthesis(source, index)
            if end is not None:
                _mask_inner_option_parentheses(masked, source, index, end)
                index = end + 1
                continue

        index += 1

    return "".join(masked)


def _mask_identifier_before(source: str, position: int) -> str | None:
    match = re.search(r"([A-Za-z_][\w\.]*)\s*$", source[:position])
    return match.group(1) if match is not None else None


def _mask_matching_parenthesis(source: str, start: int) -> int | None:
    depth = 0
    quote: str | None = None
    index = start
    while index < len(source):
        character = source[index]
        if quote is not None:
            if character == "\\" and index + 1 < len(source):
                index += 2
                continue
            if character == quote:
                if index + 1 < len(source) and source[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
        index += 1
    return None


def _mask_inner_option_parentheses(
    masked: list[str], source: str, start: int, end: int
) -> None:
    """Replace inner or quoted parentheses in one outer option group."""

    depth = 1
    quote: str | None = None
    index = start + 1
    while index < end:
        character = source[index]
        if quote is not None:
            if character == "\\" and index + 1 < end:
                if source[index + 1] in {"(", ")"}:
                    masked[index + 1] = " "
                index += 2
                continue
            if character == quote:
                if index + 1 < end and source[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            if character in {"(", ")"}:
                masked[index] = " "
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            masked[index] = " "
            depth += 1
        elif character == ")":
            if depth > 1:
                masked[index] = " "
            depth -= 1
        index += 1


def _legacy_split_statement_texts(source: str) -> tuple[str, ...]:
    """Legacy ``_split_statement_texts`` truth: semicolons outside quotes.

    Read-only statement split for the incomplete dataset WHERE precheck;
    mirrors the legacy quote rules (backslash escapes and doubled quote
    escapes) exactly.  No comment stripping happens here: the legacy
    precheck read the same raw statement slices.
    """

    statements: list[str] = []
    start = 0
    quote: str | None = None
    index = 0
    while index < len(source):
        character = source[index]
        if quote is not None:
            if character == "\\" and index + 1 < len(source):
                index += 2
                continue
            if character == quote:
                if index + 1 < len(source) and source[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            index += 1
            continue
        if character == ";":
            statements.append(source[start:index])
            start = index + 1
        index += 1
    if start < len(source):
        statements.append(source[start:])
    return tuple(statements)


_DATASET_WHERE_INCOMPLETE_CAUSE = (
    "dataset option WHERE= is invalid: incomplete WHERE= parenthesized expression"
)


def _program_incomplete_dataset_where_diagnostic(
    source: SourceDocument,
    text: str | None = None,
) -> PureDiagnostic | None:
    """Legacy ``_incomplete_dataset_where_diagnostic`` truth, source-aware.

    The statement grammar is intentionally permissive and can consume a
    WHERE value's closing parenthesis as the dataset-option closing
    token, so the legacy parse gated this case BEFORE the outer Lark
    call.  The whole-text owner keeps the same precheck (quote/delimiter
    counting is length-neutral, and the legacy entry semantics are
    preserved verbatim over the raw document slices) with the legacy
    diagnostic code, severity, location enumeration, message, and
    ``parse`` stage; the legacy rule attaches no span and no labels to
    this diagnostic family, and the whole program stops immediately
    without any block parse (exactly the legacy entry behavior).
    ``text`` lets the whole-text entry share the ONE upfront skip mask
    with this precheck: the old pipeline had
    already masked the standard macro/PROC skip regions before any
    later parse, so ``data fake ...`` inside an identified ``%macro``
    region never reached the legacy precheck, and the same masked
    analysis buffer (semicolon coverage included, newline/length
    preserved) is the shared initial input here.
    """

    for statement_index, statement in enumerate(
        _legacy_split_statement_texts(source.text if text is None else text),
        start=1,
    ):
        stripped = statement.strip()
        if not stripped or not re.match(r"(?i)^(?:data|set|merge)\b", stripped):
            continue
        if re.search(r"(?i)\bwhere\s*=", stripped) is None:
            continue
        depth = 0
        quote: str | None = None
        index = 0
        while index < len(stripped):
            character = stripped[index]
            if quote is not None:
                if character == "\\" and index + 1 < len(stripped):
                    index += 2
                    continue
                if character == quote:
                    if index + 1 < len(stripped) and stripped[index + 1] == quote:
                        index += 2
                        continue
                    quote = None
                index += 1
                continue
            if character in {"'", '"'}:
                quote = character
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            index += 1
        if depth > 0:
            return PureDiagnostic(
                code="PARSE_DATASET_WHERE_UNSUPPORTED",
                severity="error",
                location=f"statement:{statement_index}",
                stage="parse",
                message=_DATASET_WHERE_INCOMPLETE_CAUSE,
                source_text=source.text,
            )
    return None


def _skip_masked_text(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    skip_regions: tuple[SkipRegionSyntax, ...],
) -> str:
    """The ONE upfront skip mask over the normalized analysis text.

    The legacy standard hook applied its mask before every later parse:
    the identified standard macro/PROC skip regions were blanked
    (newlines preserved, all other codepoints, including their trailing
    semicolons which the region span already covers) while every other
    region kept its text verbatim.  The whole-text entry rebuilds that
    single mask once from the ONE upfront region-identification result
    and shares it with the incomplete-WHERE precheck and the per-block
    classification parses.  No second full-text region parse, and
    unknown macro-like forms (``%foo a=1``) are NOT masked because the
    region pass did not identify them as skips (old public classified
    those assignments).
    """

    analysis = list(frontend._analysis_text(source))
    for region in skip_regions:
        for index in range(region.span.start, min(region.span.end, len(analysis))):
            if analysis[index] not in {"\n", "\r"}:
                analysis[index] = " "
    return "".join(analysis)


def _legacy_unexpected_span_end(text: str, start: int) -> int:
    """Legacy ``_unexpected_span_end`` truth for the near-line span."""

    if start >= len(text):
        return start + 1
    match = re.match(r"[^\s;]+", text[start:])
    if match is None:
        return min(start + 1, len(text))
    return start + max(len(match.group(0)), 1)


def _parse_failure_diagnostic(
    source: SourceDocument, position: int
) -> PureDiagnostic:
    """The legacy full-text ``UnexpectedInput`` stop diagnostic.

    Same code, severity, message family (``Unsupported statement syntax
    near line {line}, column {column}.``), ``statement:{index}`` location
    rule (semicolon count up to the failing position plus one), span, and
    ``syntax error`` label as the legacy ``_extract_statement_segments``
    UnexpectedInput handler, rebuilt as source-aware full-text spans for
    the whole-text result.  The diagnostic carries the
    raw full text as ``source_text`` (raw display contract: tabs, quotes,
    and comments are never rewritten in the display form).
    """

    clamped = max(min(position, len(source.text)), 0)
    line, column = _document_line_column(source.text, clamped)
    end = _legacy_unexpected_span_end(source.text, clamped)
    if clamped >= len(source.text):
        end = clamped + 1
    statement_index = source.text[:clamped].count(";") + 1
    span = PureDiagnosticSpan(
        start=clamped,
        end=max(end, clamped + 1),
        line=line,
        column=column,
        end_line=line,
        end_column=column + max(end - clamped, 1),
        source_id=source.source_id,
    )
    return PureDiagnostic(
        code="PARSE_UNSUPPORTED_STATEMENT",
        severity="error",
        location=f"statement:{statement_index}",
        message=f"Unsupported statement syntax near line {line}, column {column}.",
        span=span,
        labels=(PureDiagnosticLabel(span=span, message="syntax error"),),
        source_text=source.text,
    )


def _unsupported_segment_diagnostic(
    source: SourceDocument,
    span: SourceSpan,
    rule_name: str,
    statement_index: int,
) -> PureDiagnostic:
    """The legacy mapped-but-unaligned statement diagnostic.

    When the classification segmentation cannot be reconciled with the
    region statement spans (or a rule variant carries no kind), the
    legacy truth is ``PARSE_UNSUPPORTED_STATEMENT`` with the rule name,
    an ``unsupported statement`` label on the statement span, and the
    raw full text -- the whole program stops (never an empty-success
    result for a valid-grammar shape).
    """

    diagnostic_span = PureDiagnosticSpan(
        start=span.start,
        end=max(span.end, span.start + 1),
        line=span.line,
        column=span.column,
        end_line=span.end_line,
        end_column=span.end_column,
        source_id=span.source_id,
    )
    return PureDiagnostic(
        code="PARSE_UNSUPPORTED_STATEMENT",
        severity="error",
        location=f"statement:{statement_index}",
        message=f"Unsupported statement syntax: {rule_name}",
        span=diagnostic_span,
        labels=(PureDiagnosticLabel(span=diagnostic_span, message="unsupported statement"),),
        source_text=source.text,
    )


_STOP_STOPPING_KINDS: frozenset[str] = frozenset(
    {"DATA", "SET", "MERGE", "LABEL", "RENAME"}
)


def _segment_led_statements(
    source: SourceDocument,
    segments: list[tuple[str, int, int, str, int | None]],
    base: int,
) -> list[_ClassifiedStatement]:
    """Legacy-facade-native frames: one classified statement per segment.

    The old ``ParserService.parse`` iterated its own classification
    segments; the region frames are the typed overlay.  When the two
    disagree for legacy-string reasons (open-quote fragments, unclosed
    runs the legacy region lexer closed across statements), the single
    classification parse's own segments become the frames, so the typed
    statements, spans, and stop locations stay the old public truth
    without a second region parse.
    """

    return [
        _ClassifiedStatement(
            kind=kind,
            region=ProgramStatementRegion(
                kind=cast("ProgramStatementKind", kind),
                span=source.span(base + start, base + end),
            ),
            keyword_end_rel=keyword_end_rel,
            rule_name=rule_name,
            start_rel=start,
            end_rel=end,
            # The segment IS the frame here, so the initial classification
            # display span and the region share the exact span.
            legacy_display_span=source.span(base + start, base + end),
        )
        for (kind, start, end, rule_name, keyword_end_rel) in segments
    ]


def _statement_stop_code(kind: str, cause: str) -> tuple[str, str]:
    """Legacy per-statement diagnostic code/stage families for the cause.

    The bounded dataset/attribute entries preserve only the legacy cause strings
    on the typed ``DeferredStatementError``; this mapping restores the
    legacy code/stage owner rule:

    - the incomplete dataset option WHERE cause keeps its dedicated
      ``PARSE_DATASET_WHERE_UNSUPPORTED`` code and ``parse`` stage;
    - the SET/MERGE/reference scope-refusal causes keep
      ``PARSE_SET_OPTION_SCOPE_ERROR``;
    - the remaining dataset and LABEL families keep
      ``PARSE_UNSUPPORTED_STATEMENT`` with the cause as the message;
    - RENAME keeps ``PARSE_RENAME_STATEMENT_INVALID``, and
    - every other mapped family keeps
      ``PARSE_UNSUPPORTED_STATEMENT``.
    """

    if kind in {"DATA", "SET", "MERGE"}:
        if cause.startswith("dataset option WHERE= is invalid:"):
            # Every restrictive dataset-option WHERE cause family is the
            # legacy dedicated stage/parse diagnostic (the incomplete parenthesized
            # family is one prefix member, not the whole family).
            return "PARSE_DATASET_WHERE_UNSUPPORTED", "parse"
        if cause.endswith("= is only allowed as a SET statement option.") or cause.startswith(
            "IN= is only allowed as a dataset reference option"
        ):
            return "PARSE_SET_OPTION_SCOPE_ERROR", ""
        return "PARSE_UNSUPPORTED_STATEMENT", ""
    if kind == "RENAME":
        return "PARSE_RENAME_STATEMENT_INVALID", ""
    return "PARSE_UNSUPPORTED_STATEMENT", ""


def _statement_stop_diagnostic(
    source: SourceDocument,
    kind: str,
    span: SourceSpan,
    cause: str,
    statement_index: int,
) -> PureDiagnostic:
    """One full-text stop diagnostic for an old parse-stage fault family."""

    code, stage = _statement_stop_code(kind, cause)
    diagnostic_span = PureDiagnosticSpan(
        start=span.start,
        end=max(span.end, span.start + 1),
        line=span.line,
        column=span.column,
        end_line=span.end_line,
        end_column=span.end_column,
        source_id=span.source_id,
    )
    return PureDiagnostic(
        code=code,
        severity="error",
        location=f"statement:{statement_index}",
        stage=stage,
        message=cause,
        span=diagnostic_span,
        labels=(PureDiagnosticLabel(span=diagnostic_span, message="statement"),),
        source_text=source.text,
    )




def _analysis_statement_slice(
    frontend: LarkProgramFrontend, source: SourceDocument, span: SourceSpan
) -> str:
    """The request-local normalized statement slice (spans stay raw-true)."""

    return frontend._analysis_text(source)[span.start : span.end]


def _glued_output_statement(
    source: SourceDocument,
    span: SourceSpan,
    analysis_text: str,
    apply_references: tuple[ApplyReferenceSyntax, ...],
) -> StatementSyntax:
    """Old glued OUTPUT truth: the whole original text stays one target atom.

    The legacy ``output_stmt`` regex consumed the whole glued statement and
    the runtime ``statement_body(text, "output")`` body was the whole
    original text (``outputfoo`` / ``outputfoo bar`` as one opaque routing
    target), so the typed atom keeps the whole raw slice instead of
    splitting realigned tokens.  No alias, no
    general-expression expansion.
    """

    lead = len(analysis_text) - len(analysis_text.lstrip())
    trail = len(analysis_text) - len(analysis_text.rstrip())
    atom_span = source.span(span.start + lead, span.end - trail)
    return StatementSyntax(
        kind="OUTPUT",
        payload=OutputStatementSyntax(
            targets=(
                DatasetReferenceSyntax(
                    name=source.text[atom_span.start : atom_span.end],
                    options=DatasetOptionSyntax(),
                    span=atom_span,
                ),
            )
        ),
        span=span,
        apply_references=apply_references,
    )


def _dataset_statement_from_classification(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    kind: str,
    span: SourceSpan,
    keyword_end_rel: int | None,
) -> StatementSyntax:
    """The old DATA/SET/MERGE truth with the classification keyword offset.

    The legacy classification accepted glued keyword spellings (``datafoo
    out`` refs ``foo``/``out`` via the ``data_kw`` token end, ``setfoo`` /
    ``mergefoo`` refs ``foo``); the shared scanner keeps the exact original
    option/ref discipline, so the internal builder passes the tree
    keyword token end through to the shared option scanner instead of
    re-running the caller-misuse prefix gate (whose contract stays
    unchanged for the bounded entry's own callers).  Missing refs keep the
    legacy ``Unsupported statement syntax.`` deferred, never a stop.
    """

    full_text = source.text[span.start : span.end]
    keyword_end = (
        keyword_end_rel if keyword_end_rel is not None else len(kind)
    )
    try:
        refs, statement_options = _scan_dataset_statement_body(
            frontend, source, span, keyword_end, kind
        )
        if kind in {"DATA", "SET", "MERGE"} and not refs:
            # The legacy grammar requires at least one dataset reference
            # component per DATA/SET/MERGE statement.
            raise _DatasetStatementFault("Unsupported statement syntax.")
    except _DatasetStatementFault as fault:
        return StatementSyntax(
            kind=cast("StatementKind", kind),
            payload=DeferredStatementError(
                error_kind="parse_fault",
                native_cause=fault.cause,
                span=span,
                display_text=full_text.strip(),
                diagnostic_display=fault.display,
            ),
            span=span,
        )
    if kind == "DATA":
        payload: DataStatementSyntax | SourceStatementSyntax = (
            DataStatementSyntax(outputs=tuple(refs))
        )
    else:
        payload = SourceStatementSyntax(
            refs=tuple(refs),
            options=_build_set_statement_options(statement_options),
        )
    return StatementSyntax(kind=cast("StatementKind", kind), payload=payload, span=span)


def _dispatch_classified_statement(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    kind: str,
    statement: _ClassifiedStatement,
) -> StatementSyntax:
    """Dispatch one classified statement to its bounded statement entry.

    The tree kind plus the keyword/token truth of the classification
    decide the bounded entry; glued legacy-accepted spellings keep the
    old classification semantics through the internal builders (dataset
    keyword end, OUTPUT whole-text atom, ELSE glue body, glued ELSE IF
    outer family, glued DO family) while every other shape and every
    direct entry caller keep the reviewed behavior.  STOP / DELETE / END /
    RUN keep the old position-only simple payloads; the statement-level
    apply inventory is the whole statement slice scan in source order.
    """

    span = statement.region.span
    if kind in {"DATA", "SET", "MERGE"}:
        return _dataset_statement_from_classification(
            frontend, source, kind, span, statement.keyword_end_rel
        )
    apply_references = collect_apply_references(source, span)
    analysis_text = _analysis_statement_slice(frontend, source, span)
    if kind == "OUTPUT":
        lowered = analysis_text.lower()
        if lowered.startswith("output ") or lowered == "output":
            return frontend.parse_dataset_statement(source, statement.region)
        return _glued_output_statement(source, span, analysis_text, apply_references)
    if kind == "WHERE":
        return frontend.parse_where_statement(source, statement.region)
    if kind in {"KEEP", "DROP", "BY", "RETAIN"}:
        return frontend.parse_name_list_statement(source, statement.region)
    if kind == "ASSIGN":
        return frontend.parse_assignment_statement(source, statement.region)
    if kind == "SUM":
        return frontend.parse_sum_statement(source, statement.region)
    if kind == "LABEL":
        return frontend.parse_label_statement(source, statement.region)
    if kind == "RENAME":
        return frontend.parse_rename_statement(source, statement.region)
    if kind == "IF":
        return frontend.parse_if_statement(source, statement.region)
    if kind == "ELSE":
        if _ELSE_KEYWORD_PATTERN.match(analysis_text) is not None:
            return frontend.parse_else_statement(source, statement.region)
        # Glued ``elsefoo`` stayed an ELSE statement whose whole slice was
        # the body under the legacy ``statement_body`` rule (the reached
        # branch produced ``unsupported inline action: <body>``); the
        # inner descriptor keeps the raw span and never promotes a stop
        # The bounded entry's own caller-misuse
        # gate is unchanged for its other callers.
        body_text = analysis_text.strip()
        lead = len(analysis_text) - len(analysis_text.lstrip())
        trail = len(analysis_text) - len(analysis_text.rstrip())
        body_span = source.span(span.start + lead, span.end - trail)
        action = _structure_else_body(frontend, source, body_span, body_text)
        return StatementSyntax(
            kind="ELSE",
            payload=ElseStatementSyntax(action=action),
            span=span,
            apply_references=apply_references,
        )
    if kind == "ELSE IF":
        if (
            _ELSE_IF_KEYWORD_PATTERN.match(analysis_text) is not None
            or _IF_KEYWORD_PATTERN.match(analysis_text) is not None
        ):
            return frontend.parse_if_statement(source, statement.region)
        # A glued ``elseif`` no-spec shape kept the outer whole-statement
        # fault family (old ``ELSE IF`` without if_spec; the native reach
        # skipped it, so never a program stop).
        return _faulted_if_statement(source, span, "ELSE IF", apply_references)
    if kind == "DO":
        if _DO_KEYWORD_PATTERN.match(analysis_text) is not None:
            return frontend.parse_do_statement(source, statement.region)
        # Glued ``doover`` classification kept the lazily failing DO
        # family through the shared DO builder instead of a caller
        # misuse refusal.
        return _build_do_statement(
            frontend, source, span, analysis_text, apply_references
        )
    if kind == "ARRAY":
        return frontend.parse_array_statement(source, statement.region)
    if kind == "SKIPPED":
        return frontend.parse_skipped_statement(source, statement.region)
    payload_type = _SIMPLE_PAYLOAD_SYNTAX.get(kind)
    if payload_type is None:
        raise TypeError(f"classified statement kind {kind!r} has no bounded entry")
    return StatementSyntax(
        kind=cast("StatementKind", kind),
        payload=payload_type(),
        span=span,
        apply_references=apply_references,
    )


def _build_statement_from_classification(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    kind: str,
    statement: _ClassifiedStatement,
) -> StatementSyntax:
    """Build one classified statement and freeze its initial display span.

    The dispatch is the bounded-entry owner above; the ONLY extra step is
    the frozen display bookkeeping: the initial
    classification segment's absolute span is
    frozen on the statement at initial parse — never recomputed from the
    final block base — while the semantic ``span``/``origin`` stay
    untouched.
    """

    built = _dispatch_classified_statement(frontend, source, kind, statement)
    legacy_display_span = statement.legacy_display_span
    if built.legacy_display_span is None and legacy_display_span is not None:
        return replace(built, legacy_display_span=legacy_display_span)
    return built


@dataclass(frozen=True)
class _ClassifiedStatement:
    """One classified statement: kind, region, tree keyword end, rule name.

    ``keyword_end_rel`` is the classification-tree keyword token end
    relative to the block slice (``data_kw`` / ``set_kw`` / ``merge_kw``
    child meta) or ``None`` when the legacy rule was a single keyword
    regex (``output`` / ``do`` / ``else`` families).  It feeds the internal
    builders so glued legacy-accepted spellings (``datafoo out`` /
    ``setfoo`` / ``mergefoo``) keep the old refs instead of a caller-misuse
    refusal, without touching the bounded statement entries' own guards.

    ``legacy_display_span`` is the classification segment's absolute RAW span — the old facade's
    statement span, which included the segment's leading whitespace —
    frozen here at classification time and never recomputed from the
    final block base.  The semantic ``span``/``origin`` are untouched.
    """

    kind: str
    region: ProgramStatementRegion
    keyword_end_rel: int | None
    rule_name: str
    start_rel: int
    end_rel: int
    legacy_display_span: SourceSpan | None = None


@dataclass(frozen=True)
class _ClassifiedBlock:
    """One aligned block classification result (kinds plus regions)."""

    statements: tuple[_ClassifiedStatement, ...] = ()
    diagnostic: PureDiagnostic | None = None


def _classify_block_statements(
    frontend: LarkProgramFrontend,
    source: SourceDocument,
    block: ProgramBlockRegion,
    masked_text: str,
    *,
    segment_frame: bool = False,
) -> _ClassifiedBlock:
    """Classify one block's statement regions with the legacy grammar.

    The block slice is read from the request-local skip-masked analysis
    text (ONE upfront region parse result, so the
    standard macro/PROC statements inside a block span are blanked
    exactly like the legacy hook before any later parse) and parsed with
    the legacy ``datastep.lark`` initial classification grammar.
    Segments are aligned one-to-one with the region statement spans by
    source order and span containment; any disagreement, any
    classification grammar failure, or any unmapped rule variant keeps
    the legacy whole-parse stop semantics with a full-text source-aware
    span.  No full-text second region parse happens here for the final
    DATA/RUN frontiers: those are derived from the classified statement
    sequence.

    Open-quote fragments: the old facade was
    segment-native, so a disagreement between that older (legacy-medium)
    segmentation and the region frames falls back to the classification
    segments themselves instead of inventing a stop.  ``segment_frame``
    (open ordinary-quote fragments: the legacy hook returned raw text with
    NO skip mask) starts from the segment frame directly, and the fallback
    covers the same legacy settlement when the region frames disagree
    after the ONE classification parse, without a second parse.
    """

    parser = frontend._get_classification_parser()
    base = block.span.start
    # Compose the two legacy masks exactly once per block slice: the ONE
    # upfront skip mask plus the legacy option-inner-parentheses
    # mask the old classification parse applied before its outer Lark call.
    block_text = _mask_option_inner_parentheses(
        masked_text[base : block.span.end]
    )
    try:
        tree = parser.parse(block_text)
    except UnexpectedInput as error:
        position = base + max(getattr(error, "pos_in_stream", 0) or 0, 0)
        return _ClassifiedBlock(diagnostic=_parse_failure_diagnostic(source, position))
    except Exception as error:  # pragma: no cover - defensive grammar boundary
        raise ParserInitializationError(
            f"classification parse failed unexpectedly: {error}"
        ) from error

    segments: list[tuple[str, int, int, str, int | None]] = []
    for statement_node in getattr(tree, "children", ()):
        children = getattr(statement_node, "children", ())
        if not children:
            continue
        rule_node = children[0]
        rule_name = str(getattr(rule_node, "data", ""))
        variant_name = rule_name
        nested_children = getattr(rule_node, "children", ())
        if len(nested_children) == 1 and hasattr(nested_children[0], "data"):
            nested_rule = str(getattr(nested_children[0], "data", ""))
            if nested_rule in _CLASSIFICATION_RULE_KINDS:
                variant_name = nested_rule
        kind = _CLASSIFICATION_RULE_KINDS.get(variant_name)
        if kind is None:
            kind = _CLASSIFICATION_RULE_KINDS.get(rule_name)
        node_meta = getattr(rule_node, "meta", None)
        if node_meta is None:
            node_meta = getattr(statement_node, "meta", None)
        if node_meta is None:
            continue
        start = max(getattr(node_meta, "start_pos", 0), 0)
        end = max(getattr(node_meta, "end_pos", start), start)
        segment_text = block_text[start : end].strip()
        if not segment_text:
            continue
        keyword_end_rel = None
        for child in nested_children:
            child_data = getattr(child, "data", None)
            if child_data in {"data_kw", "set_kw", "merge_kw"}:
                child_meta = getattr(child, "meta", None)
                if child_meta is not None:
                    # Keep the offset statement-relative so the internal
                    # dataset builder can feed the shared option scanner
                    # with the exact keyword end of this statement.
                    keyword_end_rel = max(
                        getattr(child_meta, "end_pos", 0), 0
                    ) - start
                break
        segments.append((kind or "", start, end, rule_name, keyword_end_rel))

    if segment_frame:
        for (kind, start, end, rule_name, _keyword_end_rel) in segments:
            if not kind:
                return _ClassifiedBlock(
                    diagnostic=_unsupported_segment_diagnostic(
                        source, source.span(base + start, base + end), rule_name, 1
                    )
                )
        return _ClassifiedBlock(
            statements=tuple(_segment_led_statements(source, segments, base))
        )

    regions = block.statements
    if len(segments) != len(regions):
        probe = min(len(segments), len(regions) - 1) if regions else 0
        missing_span = regions[probe].span if regions else block.span
        rule_name = segments[probe][3] if segments else "statement"
        segment_led = _segment_led_statements(source, segments, base)
        if segment_led:
            return _ClassifiedBlock(statements=tuple(segment_led))
        return _ClassifiedBlock(
            diagnostic=_unsupported_segment_diagnostic(
                source, missing_span, rule_name, len(segments) + 1
            )
        )
    aligned: list[_ClassifiedStatement] = []
    for (kind, start, end, rule_name, keyword_end_rel), region in zip(
        segments, regions, strict=True
    ):
        if not kind:
            return _ClassifiedBlock(
                diagnostic=_unsupported_segment_diagnostic(
                    source, region.span, rule_name, len(aligned) + 1
                )
            )
        if region.span.start < base + start or region.span.end > base + end:
            segment_led = _segment_led_statements(source, segments, base)
            if segment_led:
                return _ClassifiedBlock(statements=tuple(segment_led))
            return _ClassifiedBlock(
                diagnostic=_unsupported_segment_diagnostic(
                    source, region.span, rule_name, len(aligned) + 1
                )
            )
        aligned.append(
            _ClassifiedStatement(
                kind=kind,
                region=region,
                keyword_end_rel=keyword_end_rel,
                rule_name=rule_name,
                start_rel=start,
                end_rel=end,
                # Absolute classification-segment
                # span (includes leading whitespace), frozen now.
                legacy_display_span=source.span(base + start, base + end),
            )
        )
    return _ClassifiedBlock(statements=tuple(aligned))


def _derive_classified_block_regions(
    source: SourceDocument,
    statements: tuple[_ClassifiedStatement, ...],
) -> tuple[_ClassifiedBlockSpan, ...]:
    """Final block frontiers from the classified statement sequence.

    Mirrors the legacy block-splitter state machine over the classified
    ordered statement kinds: the first statement of any kind opens a
    pending block, a classified ``DATA`` statement (including glued
    ``datafoo`` spellings the legacy classification grammar accepted)
    closes the pending block, a classified ``RUN`` closes at its extended
    end, and a pending block at the end of the sequence is the trailing
    block.  Statement spans keep their exact source offsets; no second
    full-text region parse and no ``text`` fallback.
    """

    blocks: list[_ClassifiedBlockSpan] = []
    current_start: int | None = None
    current_end: int | None = None
    current_statements: list[_ClassifiedStatement] = []

    def _close(terminated_by_run: bool) -> None:
        assert current_start is not None and current_end is not None
        blocks.append(
            _ClassifiedBlockSpan(
                span=source.span(current_start, current_end),
                statements=tuple(current_statements),
                terminated_by_run=terminated_by_run,
            )
        )

    for statement in statements:
        extended_end = _extend_statement_end(source.text, statement.region.span.end)
        if current_start is None:
            current_start = statement.region.span.start
            current_end = extended_end
            current_statements = [statement]
        elif statement.kind == "DATA":
            _close(False)
            current_start = statement.region.span.start
            current_end = extended_end
            current_statements = [statement]
        else:
            current_statements.append(statement)
            current_end = extended_end
        if statement.kind == "RUN":
            _close(True)
            current_start = None
            current_end = None
            current_statements = []

    if current_start is not None and current_end is not None:
        _close(False)
    return tuple(blocks)


@dataclass(frozen=True)
class _ClassifiedBlockSpan:
    """One final classified block: span plus ordered classified statements."""

    span: SourceSpan
    statements: tuple[_ClassifiedStatement, ...]
    terminated_by_run: bool


def _projection_stop_facts(
    source: SourceDocument,
    blocks: tuple[ProgramBlockRegion, ...],
    skip_regions: tuple[SkipRegionSyntax, ...],
    diagnostic_displays: tuple[ParseDiagnosticDisplay | None, ...],
) -> ProgramProjectionFacts:
    """Build the stop/refusal projection facts."""

    return ProgramProjectionFacts(
        source=source,
        block_display_spans=tuple(
            (block.span, block.terminated_by_run) for block in blocks
        ),
        skip_regions=skip_regions,
        diagnostic_displays=diagnostic_displays,
    )


def _parse_program_text(
    frontend: LarkProgramFrontend,
    text: str,
    *,
    source_id: str = "<dsl>",
) -> ProgramParseResult:
    """Whole-text service body shared by both public entry points."""

    source = SourceDocument(text=text, source_id=source_id)

    frontend._region_fail_position = None
    regions = frontend.extract_program_regions(source)
    if regions is None:
        # The whole-text region identification grammar failed: the legacy
        # full-parse stop contract gates the whole program (never an
        # empty-success, never a text fallback).  The old pseudo-block
        # entry kept the incomplete-WHERE precheck priority over the
        # generic statement failure, so the raw-text legacy WHERE-family
        # diagnostic is preserved first; otherwise the diagnostic keeps
        # the legacy near-line/column rule anchored at the region parse's
        # failing codepoint offset (EOF included), source-aware against
        # the raw document; there is no success fallback.
        # No display region was established, so the projection facts carry
        # the empty block/skip tuples while the
        # diagnostic display metadata stays position-matched.
        incomplete_where = _program_incomplete_dataset_where_diagnostic(source)
        if incomplete_where is not None:
            return ProgramParseResult(
                program=None,
                diagnostics=(incomplete_where,),
                projection=_projection_stop_facts(source, (), (), (None,)),
            )
        fail_position = frontend._region_fail_position
        position = 0 if not isinstance(fail_position, int) else fail_position
        return ProgramParseResult(
            program=None,
            diagnostics=(_parse_failure_diagnostic(source, position),),
            projection=_projection_stop_facts(source, (), (), (None,)),
        )

    # Open-quote fragments: an ordinary
    # unclosed quote fragment token means the legacy hook's region pass
    # refused the text and returned RAW visibility, establishing NO skip
    # mask.  Skip regions are therefore discarded for the whole request
    # (the old macro/PROC tokens stay ordinary classification statements,
    # e.g. ``%let q = 1`` reaches the classification as an ASSIGN) and
    # every later stage classifies from raw slices; the ONE region parse
    # plus the ONE per-block classification parse their own frames.  No
    # second engine, no legacy grammar re-run, and no other hidden mask
    # fallback is introduced.
    open_quotes = frontend._region_open_quotes
    skip_regions = () if open_quotes else regions.skip_regions
    masked_text = _skip_masked_text(frontend, source, skip_regions)
    incomplete_where = _program_incomplete_dataset_where_diagnostic(
        source, masked_text
    )
    if incomplete_where is not None:
        # The legacy parse gated this family BEFORE the statement
        # classification; keep the same immediate full-program gate.
        return ProgramParseResult(
            program=None,
            diagnostics=(incomplete_where,),
            projection=_projection_stop_facts(
                source, regions.blocks, skip_regions, (None,)
            ),
        )

    diagnostics: list[PureDiagnostic] = []
    diagnostic_displays: list[ParseDiagnosticDisplay | None] = []
    classified_statements: list[_ClassifiedStatement] = []
    for block in regions.blocks:
        classified = _classify_block_statements(
            frontend, source, block, masked_text, segment_frame=open_quotes
        )
        if classified.diagnostic is not None:
            diagnostics.append(classified.diagnostic)
            diagnostic_displays.append(None)
            continue
        classified_statements.extend(classified.statements)
    if diagnostics:
        # Classification stops: the initial region analysis' block regions
        # are the established display boundaries; the diagnostic display
        # metadata stays position-matched (all ``None`` here).
        return ProgramParseResult(
            program=None,
            diagnostics=tuple(diagnostics),
            projection=_projection_stop_facts(
                source,
                regions.blocks,
                skip_regions,
                tuple(diagnostic_displays),
            ),
        )

    # Final DATA/RUN frontiers derive from the classified ordered
    # statements (glued ``datafoo`` spellings the legacy classification
    # accepted open a new block exactly like the old splitter); the ONE
    # upfront region result plus statement spans cover every offset and
    # the skip positions pass through the original document verbatim
    #
    final_blocks = _derive_classified_block_regions(
        source, tuple(classified_statements)
    )

    statement_index = 0
    typed_blocks: list[BlockSyntax] = []
    for block in final_blocks:
        typed: list[StatementSyntax] = []
        for classified_statement in block.statements:
            statement_index += 1
            statement = _build_statement_from_classification(
                frontend, source, classified_statement.kind, classified_statement
            )
            payload = statement.payload
            if (
                isinstance(payload, DeferredStatementError)
                and classified_statement.kind in _STOP_STOPPING_KINDS
            ):
                # The old per-statement parse faults of the DATA/SET/MERGE
                # dataset-option, LABEL, and RENAME families were whole-
                # parse stopping diagnostics (never lazy); the bounded
                # entries preserve only their causes, so this owner
                # restores the stop decision without losing the typed
                # cause strings.  The restrictive
                # dataset WHERE family keeps its dedicated code and
                # ``parse`` stage.  Invalid lazy
                # expressions, IF / DO / ARRAY shapes, and statement-WHERE
                # conditions stay inside the typed statements (never
                # promoted here).
                diagnostics.append(
                    _statement_stop_diagnostic(
                        source,
                        classified_statement.kind,
                        classified_statement.region.span,
                        payload.native_cause,
                        statement_index,
                    )
                )
                # The display metadata stays position-matched with the
                # diagnostics: the initial WHERE option owner's stored
                # display value travels beside the diagnostic for the
                # legacy view only.
                diagnostic_displays.append(payload.diagnostic_display)
            typed.append(statement)
        typed_blocks.append(BlockSyntax(span=block.span, statements=tuple(typed)))

    if diagnostics:
        # Per-statement stop families: the initial region analysis' block
        # regions are the established display boundaries.
        return ProgramParseResult(
            program=None,
            diagnostics=tuple(diagnostics),
            projection=_projection_stop_facts(
                source,
                regions.blocks,
                skip_regions,
                tuple(diagnostic_displays),
            ),
        )

    return ProgramParseResult(
        program=ProgramSyntax(
            source=source,
            span=source.span(0, len(text)),
            blocks=tuple(typed_blocks),
            skip_regions=tuple(skip_regions),
        ),
        diagnostics=(),
        # Success facts: the settled typed blocks with their RUN closure
        # flags, the identified skip regions, and (no diagnostics) an
        # empty display-metadata tuple.
        projection=ProgramProjectionFacts(
            source=source,
            block_display_spans=tuple(
                (block.span, block.terminated_by_run) for block in final_blocks
            ),
            skip_regions=tuple(skip_regions),
            diagnostic_displays=(),
        ),
    )


def parse_program(
    text: str,
    *,
    source_id: str = "<dsl>",
) -> ProgramParseResult:
    """The public pure whole-text Data Step parse entry.

    Contract: the raw program text plus an
    optional ``source_id`` produce an immutable
    :class:`ProgramParseResult`.  The initial frontend consumes ONE
    upstream region-identification parse (``program_regions.lark``) and
    per-block classification with the existing legacy ``datastep.lark``
    grammar; independent expression/option parses are bounded
    frontend-owned parses; the full region text is never re-parsed.  A
    program without an old parse-stage stopping fault returns structured
    ``ProgramSyntax`` (ordered blocks plus standard skip regions) even
    when single statements hold deferred faults; the legacy stop
    families return ``program=None`` with legacy-shaped diagnostics;
    mandatory grammar initialization failures raise
    :class:`ParserInitializationError` and are never hidden.  No data
    library, execution host, schema inference, or legacy parser service
    is consulted, and no alternate backend, general engine, or failure
    success fallback is introduced.
    """

    return _parse_program_text(LarkProgramFrontend(), text, source_id=source_id)
