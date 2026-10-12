"""Immutable source, span, and typed syntax models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

from ..models import Diagnostic


SourceId = str


def _codepoint_line_column(text: str, offset: int) -> tuple[int, int]:
    """Return the 1-based line and column for a codepoint offset.

    Python strings are indexed by codepoint, so Unicode text maps without
    surrogate-pair adjustments.  ``offset == len(text)`` is allowed and
    returns the position just past the final character; any other
    out-of-bounds offset raises ``ValueError``.  Lines are delimited by
    ``"\\n"`` and columns count codepoints from the line start.
    """

    if offset < 0 or offset > len(text):
        raise ValueError(
            f"codepoint offset {offset} is outside the source (length {len(text)})"
        )
    prefix = text[:offset]
    line = prefix.count("\n") + 1
    last_newline = prefix.rfind("\n")
    column = offset - last_newline if last_newline >= 0 else offset + 1
    return line, column


@dataclass(frozen=True)
class SourceSpan:
    """Half-open codepoint span with 1-based line/column positions.

    ``start`` is inclusive and ``end`` exclusive; both are codepoint offsets
    into the associated source text.  ``end_line``/``end_column`` locate the
    position just past the final covered codepoint and may be omitted when a
    consumer only needs the start position.
    """

    start: int
    end: int
    line: int
    column: int
    source_id: SourceId = "<dsl>"
    end_line: int | None = None
    end_column: int | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"span start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"span end {self.end} must not precede start {self.start}"
            )
        if self.line < 1 or self.column < 1:
            raise ValueError(
                f"span line/column must be 1-based, got line {self.line}, column {self.column}"
            )
        if self.end_line is not None and self.end_line < 1:
            raise ValueError(f"span end_line must be 1-based, got {self.end_line}")
        if self.end_column is not None and self.end_column < 1:
            raise ValueError(f"span end_column must be 1-based, got {self.end_column}")


@dataclass(frozen=True)
class SourceDocument:
    """The full, request-scoped immutable source text of one parse request."""

    text: str
    source_id: SourceId = "<dsl>"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("source text must be a string")
        if not isinstance(self.source_id, str):
            raise TypeError("source_id must be a string")

    def span(self, start: int, end: int) -> SourceSpan:
        """Build a half-open codepoint span with derived line/column values."""

        line, column = _codepoint_line_column(self.text, start)
        end_line, end_column = _codepoint_line_column(self.text, end)
        return SourceSpan(
            start=start,
            end=end,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
            source_id=self.source_id,
        )


@dataclass(frozen=True)
class SkipRegionSyntax:
    """A standard skipped region: a macro definition or PROC block.

    Skipped regions are identified by the program frontend and carry only
    their source span.  Their contents are never parsed into statements.
    """

    kind: Literal["macro", "proc"]
    span: SourceSpan

    def __post_init__(self) -> None:
        if self.kind not in ("macro", "proc"):
            raise ValueError(f"unsupported skip region kind: {self.kind!r}")


@dataclass(frozen=True)
class BlockSyntax:
    """One DATA block with its flat, source-ordered statements.

    ``span`` is a required pure ``SourceSpan`` (zero-width spans are valid);
    ``None`` is rejected at construction.  ``statements`` preserves the
    original flat statement order; the ordinal of a block is its position
    inside ``ProgramSyntax.blocks``.
    """

    span: SourceSpan
    statements: tuple["StatementSyntax", ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"block span must be a SourceSpan, got {type(self.span).__name__}"
            )
        object.__setattr__(self, "statements", tuple(self.statements))

@dataclass(frozen=True)
class ProgramSyntax:
    """Structured whole-text result: ordered blocks and standard skip regions.

    ``source`` is the required, request-scoped immutable ``SourceDocument``
    that owns the full parsed text; there is no alternate mutable source
    ownership.  ``span`` must reference that document: its ``source_id``
    must match and its codepoint bounds must lie inside the document
    (full-text and zero-width spans are valid).  Blocks keep their original
    order, including blocks separated by skip regions.
    """

    source: SourceDocument
    span: SourceSpan
    blocks: tuple[BlockSyntax, ...] = ()
    skip_regions: tuple[SkipRegionSyntax, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceDocument):
            raise TypeError(
                f"program source must be a SourceDocument, got {type(self.source).__name__}"
            )
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"program span must be a SourceSpan, got {type(self.span).__name__}"
            )
        if self.span.source_id != self.source.source_id:
            raise ValueError(
                f"program span source_id {self.span.source_id!r} does not match "
                f"source document {self.source.source_id!r}"
            )
        if self.span.start < 0 or self.span.end > len(self.source.text):
            raise ValueError(
                f"program span [{self.span.start}, {self.span.end}) is outside "
                f"the source document (length {len(self.source.text)})"
            )
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "skip_regions", tuple(self.skip_regions))


# --- dedicated dataset option syntax (restrictive WHERE dialect) ----------

WhereComparisonOperator: TypeAlias = Literal["=", "==", "!=", ">", ">=", "<", "<="]
WhereLogicalOperator: TypeAlias = Literal["and", "or"]
WhereScalar: TypeAlias = str | int | float | bool


@dataclass(frozen=True)
class WhereComparison:
    """A column-to-scalar comparison preserving the source operator."""

    column: str
    operator: WhereComparisonOperator
    value: WhereScalar


@dataclass(frozen=True)
class WhereNullPredicate:
    """An ``IS NULL`` or ``IS NOT NULL`` column predicate."""

    column: str
    negated: bool


@dataclass(frozen=True)
class WhereLogicalExpression:
    """A boolean expression joined by ``and`` or ``or``."""

    operator: WhereLogicalOperator
    operands: tuple["WhereExpression", ...]


@dataclass(frozen=True)
class WhereNotExpression:
    """A boolean negation expression."""

    operand: "WhereExpression"


WhereExpression: TypeAlias = (
    WhereComparison | WhereNullPredicate | WhereLogicalExpression | WhereNotExpression
)


# The exact legacy display vocabulary of the IF condition projection:
# symbol equality aliases and logic words
# only.  Nothing else is ever stored as a display replacement.
_CONDITION_DISPLAY_REPLACEMENTS: frozenset[str] = frozenset(
    {"==", "!=", "and", "or", "not"}
)


@dataclass(frozen=True)
class DatasetOptionSyntax:
    """Structured dataset reference options with the restrictive WHERE dialect.

    Field names mirror the legacy parser option DTO so consumers migrate
    without renaming.  ``where_expr`` only accepts the restrictive
    ``WhereExpression`` union; free-form expressions are never representable.
    """

    in_var: str | None = None
    keep_vars: tuple[str, ...] = ()
    drop_vars: tuple[str, ...] = ()
    where_expr: WhereExpression | None = None
    rename_map: tuple[tuple[str, str], ...] = ()
    firstobs: int | None = None
    obs: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "keep_vars", tuple(self.keep_vars))
        object.__setattr__(self, "drop_vars", tuple(self.drop_vars))
        object.__setattr__(self, "rename_map", tuple(tuple(pair) for pair in self.rename_map))
        if self.where_expr is not None and not isinstance(
            self.where_expr,
            (WhereComparison, WhereNullPredicate, WhereLogicalExpression, WhereNotExpression),
        ):
            raise TypeError(
                "where_expr must be a restrictive WhereExpression, got "
                f"{type(self.where_expr).__name__}"
            )


# --- typed expressions, targets, statements, deferred errors ---------------

# Normalized operator vocabulary: the exact native Pratt binary operator set
# (native/limulus_native/src/expressions/mod.rs 643-657).  The source ``=``
# equality alias is normalized to ``==`` by the later frontend; nodes keep
# the raw source occurrence span.  ``%`` is not a supported operator.
BinaryOperator: TypeAlias = Literal[
    "or", "and", "==", "!=", ">", ">=", "<", "<=", "+", "-", "*", "/", "**", "mod"
]

LiteralKind: TypeAlias = Literal["missing", "string", "numeric", "format"]


@dataclass(frozen=True)
class LiteralSyntax:
    """A literal occurrence preserving its raw source lexeme.

    ``lexeme`` keeps the untouched source text (decimal integer lexemes are
    never reinterpreted, quoted strings keep their quotes).  ``kind``
    distinguishes missing (``.``), quoted strings, numeric lexemes, and the
    put/input format tokens.
    """

    kind: LiteralKind
    lexeme: str
    span: SourceSpan
    origin: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lexeme, str):
            raise TypeError("literal lexeme must be a string")
        if not self.lexeme:
            raise ValueError("literal lexeme must not be empty")
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"literal span must be a SourceSpan, got {type(self.span).__name__}"
            )
        if self.kind not in ("missing", "string", "numeric", "format"):
            raise ValueError(f"unsupported literal kind: {self.kind!r}")
        if self.origin is not None and not isinstance(self.origin, SourceSpan):
            raise TypeError("literal origin must be a SourceSpan or None")


def _require_expression(value: object, field_name: str) -> None:
    """Reject non-typed payloads for expression slots (no ``Any`` escape)."""

    if not isinstance(value, _EXPRESSION_MEMBERS):
        raise TypeError(
            f"{field_name} must be an ExpressionSyntax, got {type(value).__name__}"
        )


@dataclass(frozen=True)
class NameSyntax:
    """A named occurrence (expression variable or name-list entry)."""

    name: str
    span: SourceSpan
    origin: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name:
            raise ValueError("name must not be empty")
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"name span must be a SourceSpan, got {type(self.span).__name__}"
            )
        if self.origin is not None and not isinstance(self.origin, SourceSpan):
            raise TypeError("name origin must be a SourceSpan or None")


@dataclass(frozen=True)
class UnarySyntax:
    """Unary ``not`` or ``-`` over one operand."""

    operator: Literal["not", "-"]
    operand: "ExpressionSyntax"
    span: SourceSpan
    origin: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.operator not in ("not", "-"):
            raise ValueError(f"unsupported unary operator: {self.operator!r}")
        _require_expression(self.operand, "unary operand")
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"unary span must be a SourceSpan, got {type(self.span).__name__}"
            )
        if self.origin is not None and not isinstance(self.origin, SourceSpan):
            raise TypeError("unary origin must be a SourceSpan or None")


@dataclass(frozen=True)
class BinarySyntax:
    """A binary operator application at the existing native precedence.

    ``operator`` uses the normalized native Pratt vocabulary (the source
    ``=`` equality alias becomes ``==``); the node span keeps the raw source
    occurrence position and the model defines no normalization function.
    """

    operator: BinaryOperator
    left: "ExpressionSyntax"
    right: "ExpressionSyntax"
    span: SourceSpan
    origin: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.operator not in (
            "or", "and", "==", "!=", ">", ">=", "<", "<=",
            "+", "-", "*", "/", "**", "mod",
        ):
            raise ValueError(f"unsupported binary operator: {self.operator!r}")
        _require_expression(self.left, "binary left operand")
        _require_expression(self.right, "binary right operand")
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"binary span must be a SourceSpan, got {type(self.span).__name__}"
            )
        if self.origin is not None and not isinstance(self.origin, SourceSpan):
            raise TypeError("binary origin must be a SourceSpan or None")


@dataclass(frozen=True)
class CallSyntax:
    """A named call, including ``array(i)`` element calls.

    Function and array existence is never bound here; the existing runtime
    owner keeps that decision.
    """

    name: str
    arguments: tuple["ExpressionSyntax", ...]
    span: SourceSpan
    origin: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("call name must be a non-empty string")
        object.__setattr__(self, "arguments", tuple(self.arguments))
        for argument in self.arguments:
            _require_expression(argument, "call argument")
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"call span must be a SourceSpan, got {type(self.span).__name__}"
            )
        if self.origin is not None and not isinstance(self.origin, SourceSpan):
            raise TypeError("call origin must be a SourceSpan or None")


@dataclass(frozen=True)
class SubscriptSyntax:
    """A bracketed subscript over a base value."""

    base: "ExpressionSyntax"
    index: "ExpressionSyntax"
    span: SourceSpan
    origin: SourceSpan | None = None

    def __post_init__(self) -> None:
        _require_expression(self.base, "subscript base")
        _require_expression(self.index, "subscript index")
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"subscript span must be a SourceSpan, got {type(self.span).__name__}"
            )
        if self.origin is not None and not isinstance(self.origin, SourceSpan):
            raise TypeError("subscript origin must be a SourceSpan or None")


@dataclass(frozen=True)
class ConcatSyntax:
    """A structural ``||`` concatenation with at least two operands.

    Concatenation stays a structure node; lowering to the existing
    ``__dsl_concat__`` call never rewrites quoted string contents.
    """

    parts: tuple["ExpressionSyntax", ...]
    span: SourceSpan
    origin: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"concat span must be a SourceSpan, got {type(self.span).__name__}"
            )
        if len(self.parts) < 2:
            raise ValueError("concat requires at least two operands")
        for part in self.parts:
            _require_expression(part, "concat operand")
        if self.origin is not None and not isinstance(self.origin, SourceSpan):
            raise TypeError("concat origin must be a SourceSpan or None")


@dataclass(frozen=True)
class DeferredExpressionError:
    """A whole failed expression saved at initial parse.

    The descriptor holds only the failure description and the apply
    references collected from the failed expression.  No compile object,
    partial evaluation result, or sub-expression payload is stored; the
    existing statement/WHERE owner completes code, message, and stage at
    reach time.
    """

    error_kind: str
    native_cause: str
    span: SourceSpan
    where_syntax_summary: str = ""
    display_text: str = ""
    apply_references: tuple[ApplyReferenceSyntax, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"deferred span must be a SourceSpan, got {type(self.span).__name__}"
            )
        object.__setattr__(self, "apply_references", tuple(self.apply_references))
        for reference in self.apply_references:
            if not isinstance(reference, ApplyReferenceSyntax):
                raise TypeError("apply references must be ApplyReferenceSyntax values")


ExpressionSyntax: TypeAlias = (
    LiteralSyntax
    | NameSyntax
    | UnarySyntax
    | BinarySyntax
    | CallSyntax
    | SubscriptSyntax
    | ConcatSyntax
    | DeferredExpressionError
)

_EXPRESSION_MEMBERS: tuple[type, ...] = (
    LiteralSyntax,
    NameSyntax,
    UnarySyntax,
    BinarySyntax,
    CallSyntax,
    SubscriptSyntax,
    ConcatSyntax,
    DeferredExpressionError,
)


@dataclass(frozen=True)
class NameTargetSyntax:
    """A plain assignment target name."""

    name: str
    span: SourceSpan

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("assignment target name must be a non-empty string")
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"target span must be a SourceSpan, got {type(self.span).__name__}"
            )


@dataclass(frozen=True)
class ArrayElementTargetSyntax:
    """An array element assignment target with its source delimiter."""

    name: str
    delimiter: Literal["(", "[", "{"]
    index: "ExpressionSyntax"
    span: SourceSpan

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("assignment target name must be a non-empty string")
        if self.delimiter not in ("(", "[", "{"):
            raise ValueError(f"unsupported target delimiter: {self.delimiter!r}")
        _require_expression(self.index, "array element target index")
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"target span must be a SourceSpan, got {type(self.span).__name__}"
            )


AssignmentTargetSyntax: TypeAlias = NameTargetSyntax | ArrayElementTargetSyntax


@dataclass(frozen=True)
class ApplyReferenceSyntax:
    """Initial apply reference metadata, independent of expression success.

    ``target`` is the decoded first quoted argument as recorded by the
    initial lexical scan; an empty decoded target is retained here and the
    resolver rejects it at the resolve stage.  ``call_span`` locates the
    ``apply(...)`` occurrence and ``owner_span`` the expression or statement
    that owns it.  No schema, callable, or registry is stored.
    """

    target: str
    call_span: SourceSpan
    owner_span: SourceSpan

    def __post_init__(self) -> None:
        if not isinstance(self.target, str):
            raise TypeError("apply target must be a string")
        if not isinstance(self.call_span, SourceSpan):
            raise TypeError(
                f"apply call span must be a SourceSpan, got {type(self.call_span).__name__}"
            )
        if not isinstance(self.owner_span, SourceSpan):
            raise TypeError(
                f"apply owner span must be a SourceSpan, got {type(self.owner_span).__name__}"
            )


@dataclass(frozen=True)
class DatasetReferenceSyntax:
    """One dataset reference with its typed option structure and span."""

    name: str
    options: DatasetOptionSyntax
    span: SourceSpan

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("dataset reference name must be a non-empty string")
        if not isinstance(self.options, DatasetOptionSyntax):
            raise TypeError(
                f"dataset options must be DatasetOptionSyntax, got "
                f"{type(self.options).__name__}"
            )
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"dataset reference span must be a SourceSpan, got "
                f"{type(self.span).__name__}"
            )


@dataclass(frozen=True)
class SetStatementOptionSyntax:
    """SET/MERGE statement-level options mirroring the legacy DTO."""

    indsname_var: str | None = None
    end_var: str | None = None


@dataclass(frozen=True)
class DataStatementSyntax:
    """DATA statement payload: the ordered output dataset references."""

    outputs: tuple[DatasetReferenceSyntax, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", tuple(self.outputs))


@dataclass(frozen=True)
class SourceStatementSyntax:
    """SET/MERGE payload: ordered dataset references and statement options."""

    refs: tuple[DatasetReferenceSyntax, ...] = ()
    options: SetStatementOptionSyntax = field(default_factory=SetStatementOptionSyntax)

    def __post_init__(self) -> None:
        object.__setattr__(self, "refs", tuple(self.refs))
        if not isinstance(self.options, SetStatementOptionSyntax):
            raise TypeError(
                f"source statement options must be SetStatementOptionSyntax, got "
                f"{type(self.options).__name__}"
            )


@dataclass(frozen=True)
class WhereStatementSyntax:
    """Statement WHERE payload: the general expression condition.

    Statement WHERE keeps the general expression dialect (including
    ``DeferredExpressionError``); the restrictive dataset option dialect is
    unchanged and stays on ``DatasetOptionSyntax``.
    """

    condition: "ExpressionSyntax"

    def __post_init__(self) -> None:
        _require_expression(self.condition, "where statement condition")


@dataclass(frozen=True)
class AssignStatementSyntax:
    """ASSIGN payload: structured target and right-hand side."""

    target: AssignmentTargetSyntax
    value: "ExpressionSyntax"

    def __post_init__(self) -> None:
        if not isinstance(
            self.target, (NameTargetSyntax, ArrayElementTargetSyntax)
        ):
            raise TypeError(
                f"assignment target must be an AssignmentTargetSyntax, got "
                f"{type(self.target).__name__}"
            )
        _require_expression(self.value, "assign value")


@dataclass(frozen=True)
class SumStatementSyntax:
    """SUM payload: target name and increment expression."""

    target: NameTargetSyntax
    increment: "ExpressionSyntax"

    def __post_init__(self) -> None:
        if not isinstance(self.target, NameTargetSyntax):
            raise TypeError(
                f"sum target must be a NameTargetSyntax, got "
                f"{type(self.target).__name__}"
            )
        _require_expression(self.increment, "sum increment")


@dataclass(frozen=True)
class NameListStatementSyntax:
    """KEEP/DROP/BY/RETAIN payload: the ordered typed name list."""

    names: tuple[NameSyntax, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(self.names))
        for name in self.names:
            if not isinstance(name, NameSyntax):
                raise TypeError("name list entries must be NameSyntax values")


@dataclass(frozen=True)
class LabelAttributeSyntax:
    """One LABEL attribute pair, mirroring the legacy name/label mapping."""

    name: str
    label: str
    span: SourceSpan

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("label name must be a non-empty string")
        if not isinstance(self.label, str):
            raise TypeError("label text must be a string")
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"label attribute span must be a SourceSpan, got "
                f"{type(self.span).__name__}"
            )


@dataclass(frozen=True)
class LabelStatementSyntax:
    """LABEL payload: ordered attribute pairs in source order."""

    labels: tuple[LabelAttributeSyntax, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        for attribute in self.labels:
            if not isinstance(attribute, LabelAttributeSyntax):
                raise TypeError("label entries must be LabelAttributeSyntax values")


@dataclass(frozen=True)
class RenameStatementSyntax:
    """RENAME payload: ordered (old, new) pairs in source order."""

    pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", tuple(tuple(pair) for pair in self.pairs))
        for pair in self.pairs:
            if len(pair) != 2 or not all(isinstance(name, str) and name for name in pair):
                raise ValueError("rename pairs must be non-empty (old, new) name pairs")


@dataclass(frozen=True)
class OutputStatementSyntax:
    """OUTPUT payload: ordered output destination references."""

    targets: tuple[DatasetReferenceSyntax, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))


@dataclass(frozen=True)
class IfStatementSyntax:
    """IF / ELSE IF payload: condition, action slot, and flags.

    The action slot is
    the direct union ``StatementSyntax | DeferredStatementError | None``:
    a structured inline action travels as its typed statement, an action
    whose statement kind cannot be determined is kept as the existing
    ``DeferredStatementError`` descriptor itself (no filler ASSIGN/SKIPPED
    kind and no new generic statement kind), and no action is ``None``
    (subset IF / bare ELSE IF is represented by the flags).  A failed
    condition/action split is NOT represented here; it stays the outer
    whole-statement fault.  The exception is stored without being executed
    at parse time — reach-time evaluation is the later execution owner's.

    ``condition_display_edits`` is the LAST
    field: the initial ``if_structured`` condition parse records only the
    display replacements it already performed for symbol equality aliases
    (``=`` -> ``==``, ``^=``/``~=``/``¬=`` -> ``!=``) and logic words
    (AND/OR/NOT -> lower case) as ordered, non-overlapping ``(SourceSpan,
    str)`` pairs anchored to the raw document.  It exists only for the
    legacy ``if_spec.condition`` display projection (``limulus.syntax.
    compat``): quoted literals, case, whitespace, transparent groups, and
    MOD stay raw, no re-lex/re-parse ever runs on it, and it is never read
    by planning, the typed wire, native evaluation, or condition-rewriting
    passes.

    ``condition_display_span`` is the exact condition span the SAME initial
    split already computed: the projection slices the raw document at this span and applies the
    stored edits — no THEN/quote/paren re-scan ever recovers the boundary.
    Adjacent half-open spans are valid edits (start == previous end).
    """

    condition: "ExpressionSyntax"
    then_action: "StatementSyntax | DeferredStatementError | None" = None
    is_subset: bool = False
    is_then_do: bool = False
    condition_display_edits: tuple[tuple["SourceSpan", str], ...] = ()
    condition_display_span: "SourceSpan | None" = None

    def __post_init__(self) -> None:
        _require_expression(self.condition, "if condition")
        if self.then_action is not None and not isinstance(
            self.then_action, (StatementSyntax, DeferredStatementError)
        ):
            raise TypeError(
                "inline action must be a StatementSyntax, the "
                "DeferredStatementError descriptor itself, or None, got "
                f"{type(self.then_action).__name__}"
            )
        if self.condition_display_span is not None and not isinstance(
            self.condition_display_span, SourceSpan
        ):
            raise TypeError(
                "condition display span must be a SourceSpan or None, got "
                f"{type(self.condition_display_span).__name__}"
            )
        object.__setattr__(
            self, "condition_display_edits", tuple(self.condition_display_edits)
        )
        previous_end: int | None = None
        previous_source_id: str | None = None
        for edit in self.condition_display_edits:
            if not isinstance(edit, tuple) or len(edit) != 2:
                raise TypeError(
                    "condition display edits must be (SourceSpan, str) "
                    f"pairs, got {edit!r}"
                )
            edit_span, replacement = edit
            if not isinstance(edit_span, SourceSpan):
                raise TypeError(
                    "condition display edit spans must be SourceSpan "
                    f"values, got {type(edit_span).__name__}"
                )
            if not isinstance(replacement, str):
                raise TypeError(
                    "condition display replacements must be strings, got "
                    f"{type(replacement).__name__}"
                )
            if replacement not in _CONDITION_DISPLAY_REPLACEMENTS:
                raise ValueError(
                    f"unsupported condition display replacement: {replacement!r}"
                )
            if previous_end is not None and edit_span.start < previous_end:
                raise ValueError(
                    "condition display edits must be ordered and "
                    f"non-overlapping, got {edit_span.start} after {previous_end}"
                )
            if previous_source_id is not None and edit_span.source_id != previous_source_id:
                raise ValueError(
                    "condition display edits must share one source, got "
                    f"{edit_span.source_id!r} after {previous_source_id!r}"
                )
            previous_end = edit_span.end
            previous_source_id = edit_span.source_id


@dataclass(frozen=True)
class ElseStatementSyntax:
    """ELSE payload: the direct action union ``StatementSyntax |
    DeferredStatementError | None`` following ``else``.

    Same ownership rules as ``IfStatementSyntax.then_action``: a structured inline body keeps its
    typed statement, an undeterminable body keeps the existing
    ``DeferredStatementError`` descriptor itself (no filler statement kind),
    and a bare ``else`` keeps ``None``.  Stored without execution; the
    reach-time evaluation belongs to the later execution owner.
    """

    action: "StatementSyntax | DeferredStatementError | None" = None

    def __post_init__(self) -> None:
        if self.action is not None and not isinstance(
            self.action, (StatementSyntax, DeferredStatementError)
        ):
            raise TypeError(
                "else action must be a StatementSyntax, the "
                "DeferredStatementError descriptor itself, or None, got "
                f"{type(self.action).__name__}"
            )


@dataclass(frozen=True)
class DoStatementSyntax:
    """DO payload: loop variable and start/end boundary expressions.

    All three parts are ``None`` for a bare ``DO;`` block start, or all
    present for ``do <var> = <start> to <end>``.  ``boundary_display_spans``
    keeps the SAME initial ``do_loop`` split's full start/end
    expression spans so the legacy ``do_spec`` display slices them instead
    of re-scanning for the top-level ``to`` keyword; a bare DO never sets
    it.  The display info is projection-only and is never read by
    planning, the typed wire, or native evaluation.
    """

    loop_var: str | None = None
    start: "ExpressionSyntax | None" = None
    end: "ExpressionSyntax | None" = None
    boundary_display_spans: "tuple[SourceSpan, SourceSpan] | None" = None

    def __post_init__(self) -> None:
        parts = (self.loop_var is not None, self.start is not None, self.end is not None)
        if any(parts) and not all(parts):
            raise ValueError("DO loop bounds must be all present or all absent")
        if self.boundary_display_spans is not None:
            spans = self.boundary_display_spans
            if (
                not isinstance(spans, tuple)
                or len(spans) != 2
                or not all(isinstance(item, SourceSpan) for item in spans)
            ):
                raise TypeError(
                    "boundary display spans must be a (SourceSpan, SourceSpan) "
                    f"tuple or None, got {spans!r}"
                )
            object.__setattr__(self, "boundary_display_spans", tuple(spans))


@dataclass(frozen=True)
class ArrayStatementSyntax:
    """ARRAY declaration payload mirroring the legacy spec fields."""

    array_name: str
    variables: tuple[str, ...]
    span: SourceSpan
    declared_size: int | None = None
    wildcard_size: bool = False
    character_array: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.array_name, str) or not self.array_name:
            raise ValueError("array name must be a non-empty string")
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"array span must be a SourceSpan, got {type(self.span).__name__}"
            )
        object.__setattr__(self, "variables", tuple(self.variables))


@dataclass(frozen=True)
class DeleteStatementSyntax:
    """DELETE payload: no additional structure in the current dialect."""


@dataclass(frozen=True)
class StopStatementSyntax:
    """STOP payload: no additional structure in the current dialect."""


@dataclass(frozen=True)
class EndStatementSyntax:
    """END payload: no additional structure in the current dialect."""


@dataclass(frozen=True)
class RunStatementSyntax:
    """RUN payload: no additional structure in the current dialect."""


@dataclass(frozen=True)
class SkippedStatementSyntax:
    """SKIPPED payload: position-only representation of skipped statements."""


@dataclass(frozen=True)
class DeferredStatementError:
    """A statement-level structural fault saved at initial parse.

    Same ownership as ``DeferredExpressionError``: the existing statement
    owner completes code, message, and stage later; no compile object or
    re-parsed source is stored.
    """

    error_kind: str
    native_cause: str
    span: SourceSpan
    where_syntax_summary: str = ""
    display_text: str = ""
    diagnostic_display: "ParseDiagnosticDisplay | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"deferred span must be a SourceSpan, got {type(self.span).__name__}"
            )
        if self.diagnostic_display is not None and not isinstance(
            self.diagnostic_display, ParseDiagnosticDisplay
        ):
            raise TypeError(
                "deferred diagnostic display must be a ParseDiagnosticDisplay "
                f"or None, got {type(self.diagnostic_display).__name__}"
            )


StatementKind: TypeAlias = Literal[
    "DATA", "SET", "MERGE", "WHERE", "ASSIGN", "SUM", "KEEP", "DROP", "BY",
    "RETAIN", "LABEL", "RENAME", "OUTPUT", "IF", "ELSE IF", "ELSE", "DO",
    "END", "ARRAY", "DELETE", "STOP", "RUN", "SKIPPED",
]

_STATEMENT_PAYLOAD_KINDS: dict[str, tuple[type, ...]] = {
    "DATA": (DataStatementSyntax, DeferredStatementError),
    "SET": (SourceStatementSyntax, DeferredStatementError),
    "MERGE": (SourceStatementSyntax, DeferredStatementError),
    "WHERE": (WhereStatementSyntax, DeferredStatementError),
    "ASSIGN": (AssignStatementSyntax, DeferredStatementError),
    "SUM": (SumStatementSyntax, DeferredStatementError),
    "KEEP": (NameListStatementSyntax, DeferredStatementError),
    "DROP": (NameListStatementSyntax, DeferredStatementError),
    "BY": (NameListStatementSyntax, DeferredStatementError),
    "RETAIN": (NameListStatementSyntax, DeferredStatementError),
    "LABEL": (LabelStatementSyntax, DeferredStatementError),
    "RENAME": (RenameStatementSyntax, DeferredStatementError),
    "OUTPUT": (OutputStatementSyntax, DeferredStatementError),
    "IF": (IfStatementSyntax, DeferredStatementError),
    "ELSE IF": (IfStatementSyntax, DeferredStatementError),
    "ELSE": (ElseStatementSyntax, DeferredStatementError),
    "DO": (DoStatementSyntax, DeferredStatementError),
    "END": (EndStatementSyntax, DeferredStatementError),
    "ARRAY": (ArrayStatementSyntax, DeferredStatementError),
    "DELETE": (DeleteStatementSyntax, DeferredStatementError),
    "STOP": (StopStatementSyntax, DeferredStatementError),
    "RUN": (RunStatementSyntax, DeferredStatementError),
    "SKIPPED": (SkippedStatementSyntax, DeferredStatementError),
}

StatementPayload: TypeAlias = (
    DataStatementSyntax
    | SourceStatementSyntax
    | WhereStatementSyntax
    | AssignStatementSyntax
    | SumStatementSyntax
    | NameListStatementSyntax
    | LabelStatementSyntax
    | RenameStatementSyntax
    | OutputStatementSyntax
    | IfStatementSyntax
    | ElseStatementSyntax
    | DoStatementSyntax
    | ArrayStatementSyntax
    | DeleteStatementSyntax
    | StopStatementSyntax
    | EndStatementSyntax
    | RunStatementSyntax
    | SkippedStatementSyntax
    | DeferredStatementError
)


@dataclass(frozen=True)
class StatementSyntax:
    """One flat statement: kind, typed payload, span, and apply references.

    Statements keep the original flat control order of their block.  A
    ``DeferredStatementError`` payload marks a statement whose structural
    fault was judged at initial parse; every other kind accepts exactly the
    payload class mapped for it.  ``apply_references`` keeps the original
    source order of the collected references.
    """

    kind: StatementKind
    payload: StatementPayload
    span: SourceSpan
    apply_references: tuple[ApplyReferenceSyntax, ...] = ()
    origin: SourceSpan | None = None
    legacy_display_span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.kind not in _STATEMENT_PAYLOAD_KINDS:
            raise ValueError(f"unsupported statement kind: {self.kind!r}")
        if not isinstance(self.payload, _STATEMENT_PAYLOAD_KINDS[self.kind]):
            raise ValueError(
                f"statement kind {self.kind!r} does not accept payload "
                f"{type(self.payload).__name__}"
            )
        if not isinstance(self.span, SourceSpan):
            raise TypeError(
                f"statement span must be a SourceSpan, got {type(self.span).__name__}"
            )
        object.__setattr__(self, "apply_references", tuple(self.apply_references))
        for reference in self.apply_references:
            if not isinstance(reference, ApplyReferenceSyntax):
                raise TypeError("apply references must be ApplyReferenceSyntax values")
        if self.origin is not None and not isinstance(self.origin, SourceSpan):
            raise TypeError("statement origin must be a SourceSpan or None")
        if self.legacy_display_span is not None and not isinstance(
            self.legacy_display_span, SourceSpan
        ):
            raise TypeError(
                "statement legacy display span must be a SourceSpan or None, "
                f"got {type(self.legacy_display_span).__name__}"
            )


@dataclass(frozen=True)
class ParseDiagnosticDisplay:
    """Display-only source text of one parse fault.

    The initial fault owner keeps the display text it computed from the
    known raw boundaries — e.g. the failing dataset-option ``WHERE``
    value.  The public typed diagnostic keeps its full-text
    ``source_text`` untouched; this frozen display value travels beside
    it in :class:`ProgramProjectionFacts` and only the legacy view
    projects it.  It is never read by planning, the typed wire, or
    native evaluation, and no span or structure is recovered from it.
    """

    source_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_text, str):
            raise TypeError(
                "diagnostic display source text must be a string, got "
                f"{type(self.source_text).__name__}"
            )


@dataclass(frozen=True)
class ProgramProjectionFacts:
    """Frozen display facts for one pure whole-text parse result.

    ``source`` is the request-scoped document the result was parsed
    from.  ``block_display_spans`` are ``(span, terminated_by_run)`` pairs
    in ascending, non-overlapping document order: the settled typed
    blocks on success, the initial analysis' established regions on a
    stop, and the empty tuple when the region pass refused.  ``skip_
    regions`` repeats the result's identified standard skip regions, and
    ``diagnostic_displays`` is exactly as long as the result's
    diagnostics, position-matched (``None`` where no display override
    exists).  Pure display data only: no planning, typed wire, native
    input, callback, mutable cache, or second backend ever consumes it,
    and no structure is recovered from the displays.
    """

    source: SourceDocument
    block_display_spans: tuple[tuple[SourceSpan, bool], ...] = ()
    skip_regions: tuple[SkipRegionSyntax, ...] = ()
    diagnostic_displays: tuple["ParseDiagnosticDisplay | None", ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceDocument):
            raise TypeError(
                f"projection facts source must be a SourceDocument, got "
                f"{type(self.source).__name__}"
            )
        object.__setattr__(
            self, "block_display_spans", tuple(self.block_display_spans)
        )
        previous_end: int | None = None
        previous_source_id: str | None = None
        for pair in self.block_display_spans:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    "block display facts must be (SourceSpan, bool) pairs, "
                    f"got {pair!r}"
                )
            span, terminated = pair
            if not isinstance(span, SourceSpan) or not isinstance(terminated, bool):
                raise TypeError(
                    "block display facts must be (SourceSpan, bool) pairs, "
                    f"got {pair!r}"
                )
            if span.source_id != self.source.source_id:
                raise ValueError(
                    f"block display span source_id {span.source_id!r} does not "
                    f"match source document {self.source.source_id!r}"
                )
            if span.start < 0 or span.end > len(self.source.text):
                raise ValueError(
                    f"block display span [{span.start}, {span.end}) is outside "
                    f"the source document (length {len(self.source.text)})"
                )
            if previous_end is not None and span.start < previous_end:
                raise ValueError(
                    "block display spans must be ordered and non-overlapping, "
                    f"got {span.start} after {previous_end}"
                )
            previous_end = span.end
            previous_source_id = span.source_id
            entry = pair
        object.__setattr__(self, "skip_regions", tuple(self.skip_regions))
        for region in self.skip_regions:
            if not isinstance(region, SkipRegionSyntax):
                raise TypeError(
                    "projection fact skip regions must be SkipRegionSyntax "
                    f"values, got {type(region).__name__}"
                )
            if region.span.source_id != self.source.source_id:
                raise ValueError(
                    f"skip region source_id {region.span.source_id!r} does not "
                    f"match source document {self.source.source_id!r}"
                )
        object.__setattr__(
            self, "diagnostic_displays", tuple(self.diagnostic_displays)
        )
        for display in self.diagnostic_displays:
            if display is not None and not isinstance(display, ParseDiagnosticDisplay):
                raise TypeError(
                    "diagnostic displays must be ParseDiagnosticDisplay or None, "
                    f"got {type(display).__name__}"
                )


@dataclass(frozen=True)
class ProgramParseResult:
    """Whole-text parse result: optional program and pure diagnostics.

    ``program`` is ``None`` for a result without a usable program, such as a
    user-stopping grammar error reported through ``diagnostics``.  A
    mandatory grammar initialization failure is raised by its later owner
    instead of being returned as ``program=None``.  The existing pure
    ``Diagnostic`` DTO is reused; no alternate diagnostic type is introduced.
    ``projection`` is the frozen display
    facts the frontend generates on every return; it is projection-only and
    never read by planning, runtime, the typed wire, or native input.
    """

    program: ProgramSyntax | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    projection: ProgramProjectionFacts | None = None

    @property
    def has_errors(self) -> bool:
        return any(diagnostic.severity == "error" for diagnostic in self.diagnostics)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if self.projection is not None and not isinstance(
            self.projection, ProgramProjectionFacts
        ):
            raise TypeError(
                "projection must be a ProgramProjectionFacts or None, got "
                f"{type(self.projection).__name__}"
            )


__all__ = [
    "ApplyReferenceSyntax",
    "ArrayElementTargetSyntax",
    "ArrayStatementSyntax",
    "AssignStatementSyntax",
    "AssignmentTargetSyntax",
    "BinaryOperator",
    "BinarySyntax",
    "BlockSyntax",
    "CallSyntax",
    "ConcatSyntax",
    "DataStatementSyntax",
    "DatasetOptionSyntax",
    "DatasetReferenceSyntax",
    "DeferredExpressionError",
    "DeferredStatementError",
    "DeleteStatementSyntax",
    "DoStatementSyntax",
    "ElseStatementSyntax",
    "EndStatementSyntax",
    "ExpressionSyntax",
    "IfStatementSyntax",
    "LabelAttributeSyntax",
    "LabelStatementSyntax",
    "LiteralKind",
    "LiteralSyntax",
    "NameListStatementSyntax",
    "NameSyntax",
    "NameTargetSyntax",
    "OutputStatementSyntax",
    "ParseDiagnosticDisplay",
    "ProgramParseResult",
    "ProgramProjectionFacts",
    "ProgramSyntax",
    "RenameStatementSyntax",
    "RunStatementSyntax",
    "SetStatementOptionSyntax",
    "SkipRegionSyntax",
    "SkippedStatementSyntax",
    "SourceDocument",
    "SourceId",
    "SourceSpan",
    "StatementKind",
    "StatementPayload",
    "StatementSyntax",
    "StopStatementSyntax",
    "SubscriptSyntax",
    "SumStatementSyntax",
    "UnarySyntax",
    "WhereComparison",
    "WhereComparisonOperator",
    "WhereExpression",
    "WhereLogicalExpression",
    "WhereLogicalOperator",
    "WhereNotExpression",
    "WhereNullPredicate",
    "WhereScalar",
    "WhereStatementSyntax",
]
