"""General-expression parsing into typed syntax nodes."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from lark import Lark, Tree
from lark.exceptions import UnexpectedCharacters, UnexpectedToken
from lark.lexer import Token

from .models import (
    ApplyReferenceSyntax,
    BinaryOperator,
    BinarySyntax,
    CallSyntax,
    ConcatSyntax,
    DeferredExpressionError,
    ExpressionSyntax,
    LiteralSyntax,
    NameSyntax,
    SourceDocument,
    SourceSpan,
    SubscriptSyntax,
    UnarySyntax,
)

_I64_MAX = 2**63 - 1
_U64_MAX = 2**64 - 1

# put/input format-token shape owned by the existing evaluation semantics
# (limulus/expression_semantics.py _BARE_FORMAT_TOKEN); the structural
# parser reuses the exact regex so bare second arguments of put/input calls
# become format literals with the untouched raw lexeme.
_BARE_FORMAT_TOKEN = re.compile(
    r"^\$?(?:[A-Za-z_]\w*|\d+(?:\.\d+)?|z\d+(?:\.\d+)?|comma\d+(?:\.\d+)?)\.$",
    re.IGNORECASE,
)
_FORMAT_FUNCTION_NAMES = frozenset({"PUT", "INPUT"})

# Normalized native Pratt operator vocabulary per source occurrence.
# Typed as the node operator union so the static checker validates every
# alias value against ``BinaryOperator``.
_COMPARISON_ALIASES: dict[str, BinaryOperator] = {
    "==": "==",
    "=": "==",
    "!=": "!=",
    "~=": "!=",
    "^=": "!=",
    "¬=": "!=",
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
}

_BINARY_OPERATORS: dict[str, dict[str, str]] = {
    "or_expr": {"KW_OR": "or"},
    "and_expr": {"KW_AND": "and"},
    "cmp_expr": {"CMP_OP": "="},  # resolved through the alias table below
    "add_expr": {"PLUS": "+", "MINUS": "-"},
    "mul_expr": {"STAR": "*", "SLASH": "/", "KW_MOD": "mod"},
}
# The exact normalized node operator vocabulary (models.BinaryOperator);
# every mapping/alias value resolves into this set before a node is built,
# so the static checker sees the result as the ``BinaryOperator`` literal
# union without an unconstrained string cast.
_BINARY_OPERATOR_VALUES: frozenset[str] = frozenset(
    {"or", "and", "==", "!=", ">", ">=", "<", "<=", "+", "-", "*", "/", "**", "mod"}
)
# Terminals that can start a native primary; a parser failure on one of
# them means the expression stopped in front of a token the native parser
# accepted (leftover at top level or an unclosed explicit parenthesis).
_ATOM_START_TERMINALS = frozenset(
    {
        "STRING",
        "MISSING",
        "NUMBER_DOT",
        "NUMDOT_TRAILING",
        "NUMDOT_TEXT",
        "INTEGER",
        "NAME",
        "LPAREN",
        "MINUS",
        "KW_NOT",
    }
)
# Same set without the infix/prefix minus, which also appears as an
# operator candidate in merged LALR lookahead sets.
_ATOM_START_TERMINALS_NO_MINUS = _ATOM_START_TERMINALS - {"MINUS"}

# Bracket/closer tokens the native tokenizer accepts anywhere.
_CLOSER_TERMINALS = frozenset({"RPAREN", "RBRACKET", "RBRACE", "COMMA"})

_OPERATOR_CHARACTERS = frozenset("+-*/<>=!~^¬")

_PARSER: Lark | None = None


def _grammar_path() -> Path:
    return Path(__file__).resolve().parents[1] / "grammar" / "general_expression.lark"


def _get_parser() -> Lark:
    """Build (once) the LALR parser for the general expression grammar."""

    global _PARSER
    if _PARSER is None:
        grammar = _grammar_path().read_text(encoding="utf-8")
        _PARSER = Lark(
            grammar,
            start="start",
            parser="lalr",
            propagate_positions=True,
            keep_all_tokens=True,
        )
    return _PARSER


class ExpressionParseError(Exception):
    """Internal initial-parse failure carrying the native-shaped cause.

    This is the internal parser-frontend entry contract only: the
    whole-expression span and display slice are stored so a later frontend
    owner can build the user-facing ``DeferredExpressionError`` without
    re-parsing the source.  It is never raised from future public
    production paths directly; the later owner converts it.
    """

    def __init__(self, native_cause: str, span: SourceSpan, display_text: str = "") -> None:
        super().__init__(native_cause)
        self.native_cause = native_cause
        self.span = span
        self.display_text = display_text


def _validated_slice(source: SourceDocument, span: SourceSpan) -> str:
    """Shared entry validation; returns the expression slice text."""

    if not isinstance(source, SourceDocument):
        raise TypeError(
            f"source must be a SourceDocument, got {type(source).__name__}"
        )
    if not isinstance(span, SourceSpan):
        raise TypeError(f"span must be a SourceSpan, got {type(span).__name__}")
    if span.source_id != source.source_id:
        raise ValueError(
            f"span source_id {span.source_id!r} does not match source document "
            f"{source.source_id!r}"
        )
    if span.start < 0 or span.end > len(source.text):
        raise ValueError(
            f"span [{span.start}, {span.end}) is outside the source document "
            f"(length {len(source.text)})"
        )
    return source.text[span.start : span.end]


def parse_expression(source: SourceDocument, span: SourceSpan) -> ExpressionSyntax:
    """Parse the general expression slice ``source.text[span.start:span.end]``.

    ``span`` must belong to ``source`` (same ``source_id``) and stay inside
    the document bounds; it may cover any sub-range.  Returned nodes carry
    spans anchored to the full document so callers can always retrieve a
    trustworthy source slice.  Parse failures raise
    :class:`ExpressionParseError` with the native-shaped cause.

    This strict entry keeps the strict ``parse_expression`` protocol unchanged except
    for one native-tokenizer parity repair pinned by the compatibility corpus
    (U+001C..U+001F are unknown characters, never whitespace; the raw
    ``|`` rectification stays exclusive to the deferred entry).  The
    deferred frontend entry is :func:`parse_general_expression`.
    """

    text = _validated_slice(source, span)
    return _parse_slice(text, span, source, rectify_pipe=False)


def _parse_slice(
    text: str,
    span: SourceSpan,
    source: SourceDocument,
    *,
    rectify_pipe: bool,
) -> ExpressionSyntax:
    # Native tokenizes the whole slice before parsing, so the first
    # offending lexeme decides the cause even in an otherwise empty slice:
    # control-only U+001C..U+001F input is an unknown character, never
    # blank input (Python str.strip() would disagree; the blank decision
    # uses the native whitespace set instead).
    fault = _native_lexical_fault(text, rectify_pipe=rectify_pipe)
    if fault is not None:
        raise ExpressionParseError(fault, span, text)
    if not text or all(_is_native_whitespace(char) for char in text):
        # Native requires an expression before the first token; an
        # empty/native-whitespace slice rejects with the native cause.
        raise ExpressionParseError("expression expected", span, text)
    parser = _get_parser()
    try:
        tree = parser.parse(text)
    except UnexpectedCharacters as error:
        cause = _native_unexpected_characters_cause(error, text)
        raise ExpressionParseError(cause, span, text) from error
    except UnexpectedToken as error:
        cause = _native_unexpected_token_cause(error, text)
        raise ExpressionParseError(cause, span, text) from error

    return _build_node(tree, text, span.start, source, span)


# --- deferred frontend entry --------------------------------------------------


@dataclass(frozen=True)
class GeneralExpressionResult:
    """One initial general-expression parse with its apply inventory.

    ``expression`` is a typed :class:`ExpressionSyntax` node for accepted
    slices and a whole-expression :class:`DeferredExpressionError`
    descriptor for failed slices (valid concat stays a structural
    :class:`ConcatSyntax`, never deferred).  ``apply_references`` is the
    initial lexical inventory in source order, collected independently of
    the expression's success, for the later statement reference owner.
    """

    expression: ExpressionSyntax
    apply_references: tuple[ApplyReferenceSyntax, ...]


def parse_general_expression(
    source: SourceDocument,
    span: SourceSpan,
    *,
    where_summary: bool = False,
) -> GeneralExpressionResult:
    """Parse one general expression, deferring every failure as a descriptor.

    Same slice contract as :func:`parse_expression`; this entry never
    raises for expression faults.  Accepted slices return the structural
    node; failed slices return a whole-expression
    :class:`DeferredExpressionError` carrying the native-shaped cause
    (rectified by the initial lexical pass to the native tokenizer's
    first-offender cause), the ``error_kind``, the whole ``span``, the raw
    ``display_text``, and the lexical apply inventory.  With
    ``where_summary=True`` the statement-WHERE compile summary is stored
    for the cause classes the existing owner compiles; the compile object
    itself is never stored or executed.
    """

    text = _validated_slice(source, span)
    apply_references = collect_apply_references(source, span)
    try:
        expression = parse_expression(source, span)
    except ExpressionParseError as error:
        # The native tokenizer runs before parsing, so the first offending
        # lexeme decides the cause; the structural mapping is the fallback
        # and the malformed raw "|" rectification applies here.
        cause = _native_lexical_fault(text, rectify_pipe=True) or error.native_cause
        summary = ""
        if where_summary and _where_summary_needed(cause):
            summary = _where_compile_summary(text)
        deferred = DeferredExpressionError(
            error_kind=_descriptor_error_kind(cause),
            native_cause=cause,
            span=span,
            where_syntax_summary=summary,
            display_text=text,
            apply_references=apply_references,
        )
        return GeneralExpressionResult(
            expression=deferred, apply_references=apply_references
        )
    return GeneralExpressionResult(expression=expression, apply_references=apply_references)


def _descriptor_error_kind(cause: str) -> str:
    """Coarse descriptor kind: invalid numeric literals vs structure faults."""

    if cause.startswith(("invalid number literal:", "invalid integer literal:")):
        return "unsupported_literal"
    return "parse_fault"


_WHERE_SUMMARY_CAUSES = frozenset(
    {
        "expression expected",
        "unexpected token at end of expression",
        "unterminated string literal",
    }
)


def _where_summary_needed(cause: str) -> bool:
    """Whether the existing WHERE owner compiles this failed expression.

    Mirrors the native ``is_expression_parse_error`` gate exactly.
    """

    return cause in _WHERE_SUMMARY_CAUSES or cause.startswith("unexpected token:")


def _where_compile_summary(text: str) -> str:
    """One limited compile of the failed raw slice for WHERE diagnostics.

    The stored value is only the ``SyntaxError`` message string with the
    native "SyntaxError: " prefix stripped; a slice that compiles cleanly
    keeps no summary and the owner falls back to the native cause.  The
    compile object is never stored or executed.
    """

    try:
        compile(text, "<limulus-condition>", "eval")
    except Exception as error:  # noqa: BLE001 - the owner stringifies any failure
        message = str(error)
        prefix = "SyntaxError: "
        return message[len(prefix) :] if message.startswith(prefix) else message
    return ""


# --- initial lexical apply inventory (bridge scanner parity) ------------------


def collect_apply_references(
    document: SourceDocument,
    owner_span: SourceSpan,
) -> tuple[ApplyReferenceSyntax, ...]:
    """Collect the initial ``apply(...)`` reference inventory of a slice.

    Collects references from the source slice once during frontend parsing:
    quoted regions
    and comments are excluded, ``apply`` is recognized case-insensitively
    outside identifier prefixes/suffixes, the first argument must be a
    quoted string decoded with ``ast.literal_eval``, and incomplete calls
    contribute nothing.  Duplicates stay in source order (dedupe belongs
    to the resolver owner).  Pure reference information only: no resolver,
    host callable, schema, or registry is consulted.
    """

    text = document.text[owner_span.start : owner_span.end]
    references: list[ApplyReferenceSyntax] = []
    index = 0
    text_length = len(text)
    while index < text_length:
        if text.startswith("/*", index):
            index = _scanner_skip_comment(text, index)
            continue

        if text[index] in ("'", '"'):
            index = _scanner_skip_quoted(text, index)
            continue

        if text[index : index + 5].lower() == "apply":
            target, next_index = _scanner_read_apply_target(text, index)
            if target is not None:
                references.append(
                    ApplyReferenceSyntax(
                        target=target,
                        call_span=document.span(
                            owner_span.start + index,
                            owner_span.start + next_index,
                        ),
                        owner_span=owner_span,
                    )
                )
            index = max(next_index, index + 1)
            continue

        index += 1

    return tuple(references)


def _is_identifier_character(value: str) -> bool:
    return value.isalnum() or value == "_"


def _escape_unknown_python_string_escapes(literal: str) -> str:
    """Keep unknown backslash escapes literal without asking Python to warn.

    ``ast.literal_eval`` provides the legacy inventory's decoding for known
    escapes.  Doubling only unknown escapes preserves their backslash in the
    decoded value while avoiding Python's invalid-escape warning.
    """

    parts: list[str] = []
    index = 0
    simple_escapes = "\\'\"abfnrtv"
    while index < len(literal):
        if literal[index] != "\\" or index + 1 >= len(literal):
            parts.append(literal[index])
            index += 1
            continue

        escaped = literal[index + 1]
        recognized = (
            escaped in simple_escapes
            or escaped in "01234567xuU\r\n"
            or (
                escaped == "N"
                and index + 2 < len(literal)
                and literal[index + 2] == "{"
            )
        )
        if recognized:
            parts.append(literal[index : index + 2])
        else:
            parts.append("\\\\")
            parts.append(escaped)
        index += 2

    return "".join(parts)


def _scanner_read_apply_target(
    text: str,
    start: int,
) -> tuple[str | None, int]:
    """Old-scanner target read: keyword boundary, call opening, quoted target."""

    end = start + 5
    if start > 0 and _is_identifier_character(text[start - 1]):
        return None, end
    if end < len(text) and _is_identifier_character(text[end]):
        return None, end

    cursor = end
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "(":
        return None, end

    cursor += 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] not in ("'", '"'):
        return None, end

    literal_start = cursor
    literal_end = _scanner_skip_quoted(text, cursor)
    if literal_end <= literal_start + 1 or text[literal_end - 1] != text[cursor]:
        return None, literal_end

    try:
        literal = _escape_unknown_python_string_escapes(
            text[literal_start:literal_end]
        )
        target = ast.literal_eval(literal)
    except Exception:
        return None, literal_end
    if not isinstance(target, str):
        return None, literal_end
    return target, literal_end


def _scanner_skip_quoted(text: str, start: int) -> int:
    """Old-scanner quoted-region skip (backslash escapes, doubled quotes)."""

    quote = text[start]
    cursor = start + 1
    while cursor < len(text):
        current = text[cursor]
        if current == "\\":
            cursor += 2
            continue
        if current == quote:
            # Support doubled quote escaping in SAS-style source text.
            if cursor + 1 < len(text) and text[cursor + 1] == quote:
                cursor += 2
                continue
            return cursor + 1
        cursor += 1
    return cursor


def _scanner_skip_comment(text: str, start: int) -> int:
    """Old-scanner comment skip: block comment, else ``;``-terminated line."""

    block_end = text.find("*/", start + 2)
    if block_end >= 0:
        return block_end + 2
    line_end = text.find(";", start + 2)
    return len(text) if line_end < 0 else line_end + 1


# --- native tokenizer-order cause rectification -------------------------------


def _native_lexical_fault(text: str, *, rectify_pipe: bool) -> str | None:
    """First offending lexeme in native tokenizer order, or ``None``.

    Mirrors ``native/limulus_native/src/expressions/mod.rs``
    ``tokenize_expression`` failure causes and ordering (native quotes have
    no escapes and no doubled-quote merging, unlike the scanner skips
    above).  ``rectify_pipe`` adds the malformed raw ``|`` cause used by
    the deferred entry (``parse_general_expression``); the strict entry
    keeps its provisional causes and passes ``False``.  Valid
    expressions (including accepted ``||`` chains) never reach this pass.
    """

    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if _is_native_whitespace(char):
            index += 1
            continue
        if char in "()[]{},":
            index += 1
            continue
        if char in ("'", '"'):
            end = index + 1
            while end < length and text[end] != char:
                end += 1
            if end >= length:
                return "unterminated string literal"
            index = end + 1
            continue
        if "0" <= char <= "9" or (char == "." and index + 1 < length and "0" <= text[index + 1] <= "9"):
            start = index
            index += 1
            while index < length and (
                "0" <= text[index] <= "9" or text[index] == "."
            ):
                index += 1
            fault = _native_number_fault(text, start, index)
            if fault is not None:
                return fault
            continue
        if char == ".":
            # Lone dot is the native Missing token.
            index += 1
            continue
        if ("a" <= char <= "z" or "A" <= char <= "Z" or char in "_$"):
            index += 1
            while index < length and (
                "0" <= text[index] <= "9"
                or "a" <= text[index] <= "z"
                or "A" <= text[index] <= "Z"
                or text[index] in "._"
            ):
                index += 1
            continue
        if index + 1 < length:
            pair = text[index : index + 2]
            if pair in ("**", ">=", "<=", "!=", "==", "^=", "~=", "¬="):
                index += 2
                continue
        if char in "+-*/><=":
            index += 1
            continue
        if char == "|" and not rectify_pipe:
            # Strict entry: the malformed raw "|" keeps the
            # provisional structural causes; only the deferred entry
            # rectifies it to the native tokenize-time cause below.
            index += 1
            continue
        return f"unexpected token: {char}"
    return None


# Python str.isspace() and re \s treat U+001C..U+001F (file/group/record/
# unit separators) as whitespace, but the native tokenizer follows the
# Unicode White_Space property (Rust char::is_whitespace), which excludes
# them; native rejects those control separators as unknown characters
# (live public-path evidence).
_NATIVE_NON_WHITESPACE = frozenset("\x1c\x1d\x1e\x1f")


def _is_native_whitespace(char: str) -> bool:
    return char.isspace() and char not in _NATIVE_NON_WHITESPACE


def _native_number_fault(text: str, start: int, end: int) -> str | None:
    """Native digit/dot-run classification failure, if any."""

    raw = text[start:end]
    next_non_space = next(
        (character for character in text[end:] if not character.isspace()), None
    )
    if raw.endswith("."):
        if next_non_space in (",", ")"):
            return None  # native text token before ',' or ')'
        if "." in raw[:-1]:
            return None  # native dotted identifier ("1.2.")
        try:
            float(raw)
        except ValueError:
            return f"invalid number literal: {raw}"
        return None
    if "." in raw:
        try:
            float(raw)
        except ValueError:
            return f"invalid number literal: {raw}"
        return None
    value = int(raw)
    if value <= _I64_MAX:
        return None
    if value <= _U64_MAX:
        return None
    return f"invalid integer literal: {raw}"


# --- native-shaped failure causes ------------------------------------------


def _native_unexpected_characters_cause(error: UnexpectedCharacters, text: str) -> str:
    """Map a lexer error to the closest native tokenizer cause."""

    char = error.char or ""
    if char in ("'", '"'):
        return "unterminated string literal"
    return f"unexpected token: {char}"


def _native_unexpected_token_cause(error: UnexpectedToken, text: str) -> str:
    """Map a Lark token error to the closest native parse-failure cause.

    Native messages (mod.rs parse_expression / parse_expression_cached):
    "expression expected" when a primary is required, "expected ')'" from
    ``expect_rparen`` behind a completed parenthesized group, "expected
    ']'" / "expected '}'" from the subscript branches, and "unexpected
    token at end of expression" for top-level leftovers.
    """

    token = error.token
    token_type = token.type if token is not None else ""
    expected = set(error.expected or ())
    offset = token.start_pos if token is not None and token.start_pos is not None else len(text)

    if token_type == "CONCAT":
        if expected & _ATOM_START_TERMINALS_NO_MINUS:
            # "||b", "a || || b": a "||" landed where a concat operand is
            # required (expression start or after another operator).
            return "expression expected"
        # Any other unexpected "||" falls through to the generic mapping.

    if token_type == "$END":
        if expected & _ATOM_START_TERMINALS_NO_MINUS:
            # A primary was required when the slice ended ("a +", "f(1,").
            return "expression expected"
        cause = _unclosed_closer_cause(text, len(text), expected)
        return cause if cause is not None else "expression expected"

    if token_type in _CLOSER_TERMINALS:
        if expected & _ATOM_START_TERMINALS_NO_MINUS:
            # A closer arrived while a primary was required: "()", "(a +)".
            return "expression expected"
        opener = _innermost_unclosed(text, offset)
        if opener is not None:
            # "(a, b)" / "a[1, 2]": the delimiter lands on the unclosed
            # group's expect-closer fault.
            return _CLOSER_MESSAGES[opener]
        if _last_character_before(text, offset) in _OPERATOR_CHARACTERS | {","}:
            # "a || )": the closer stays inside the empty concat part.
            return "expression expected"
        return "unexpected token at end of expression"

    opener = _innermost_unclosed(text, offset)
    if opener is not None:
        # "(a b)" / "a[1 2]" / "f(a b)": the token lands on the unclosed
        # group's expect-closer fault.
        return _CLOSER_MESSAGES[opener]
    preceding = _last_character_before(text, offset)
    if preceding == "" or preceding in _OPERATOR_CHARACTERS | {","}:
        # "a + * b" / "+5": the failing token follows an operator or sits
        # at the start, where the native primary parse rejects it.
        return "expression expected"
    return "unexpected token at end of expression"


_CLOSER_MESSAGES = {
    "(": "expected ')'",
    "[": "expected ']'",
    "{": "expected '}'",
}


def _unclosed_closer_cause(text: str, offset: int, expected: set[str]) -> str | None:
    """Expected-closer message for the innermost unclosed bracket, if any."""

    opener = _innermost_unclosed(text, offset)
    if opener == "(" and "RPAREN" in expected:
        return "expected ')'"
    if opener == "[" and "RBRACKET" in expected:
        return "expected ']'"
    if opener == "{" and "RBRACE" in expected:
        return "expected '}'"
    return None


def _innermost_unclosed(text: str, offset: int) -> str | None:
    """Innermost unclosed bracket character just before ``offset``.

    Quoted segments (native string semantics: scan to the next matching
    quote character) never contribute brackets.
    """

    stack: list[str] = []
    index = 0
    length = len(text)
    while index < offset and index < length:
        char = text[index]
        if char in ("'", '"'):
            index += 1
            while index < length and text[index] != char:
                index += 1
            index += 1
            continue
        if char in ("(", "[", "{"):
            stack.append(char)
        elif char in (")", "]", "}"):
            if stack:
                stack.pop()
        index += 1
    return stack[-1] if stack else None


def _last_character_before(text: str, offset: int) -> str:
    stripped = text[:offset].rstrip()
    return stripped[-1] if stripped else ""


# --- typed node construction -------------------------------------------------


def _op_span(token: Token, base: int, source: SourceDocument) -> SourceSpan:
    # Structural grammar invariants: every real token occurrence carries
    # exact source positions (propagate_positions).  A position-less token
    # is an invalid tree and is rejected, never coerced to a fallback.
    start = token.start_pos
    end = token.end_pos
    if start is None or end is None:
        raise AssertionError(f"token without source positions: {token!r}")
    return source.span(base + start, base + end)


def _node_span(node: Tree[Token], base: int, source: SourceDocument) -> SourceSpan:
    meta = node.meta
    return source.span(base + meta.start_pos, base + meta.end_pos)


def _build_node(
    node: Tree[Token] | Token,
    text: str,
    base: int,
    source: SourceDocument,
    expression_span: SourceSpan,
) -> ExpressionSyntax:
    if isinstance(node, Token):
        return _build_token(node, base, source)

    data = node.data
    children = node.children

    if data == "start":
        return _build_node(children[0], text, base, source, expression_span)

    if data == "concat_expr":
        return _build_concat_chain(node, text, base, source, expression_span)

    if data == "call":
        return _build_call(node, text, base, source, expression_span)

    # "call_args" never reaches _build_node: the grammar root is
    # start: concat_expr and call arguments are built exclusively through
    # _build_call -> _build_call_arguments, so no tuple-returning branch
    # (and no raw-parser fallback) exists here.

    if data == "arg":
        # Call arguments are full concat-level expressions; the existing
        # normalization covers "||" inside call brackets unconditionally.
        return _build_node(children[0], text, base, source, expression_span)

    if data == "subscript":
        return _build_subscript(node, text, base, source, expression_span)

    if data in _BINARY_OPERATORS:
        if len(children) == 3:
            left, op_token, right = children
            if not isinstance(op_token, Token):
                raise AssertionError(f"unexpected operator child: {op_token!r}")
            operator = _operator_for(data, op_token)
            return BinarySyntax(
                operator=operator,
                left=_build_node(left, text, base, source, expression_span),
                right=_build_node(right, text, base, source, expression_span),
                span=_op_span(op_token, base, source),
            )
        return _build_node(children[0], text, base, source, expression_span)

    if data == "pow_expr":
        if len(children) == 3:
            left, op_token, right = children
            if not isinstance(op_token, Token):
                raise AssertionError(f"unexpected operator child: {op_token!r}")
            return BinarySyntax(
                operator="**",
                left=_build_node(left, text, base, source, expression_span),
                right=_build_node(right, text, base, source, expression_span),
                span=_op_span(op_token, base, source),
            )
        return _build_node(children[0], text, base, source, expression_span)

    if data == "unary":
        if len(children) == 2:
            op_token, operand = children
            if not isinstance(op_token, Token):
                raise AssertionError(f"unexpected operator child: {op_token!r}")
            operator = "not" if op_token.type == "KW_NOT" else "-"
            return UnarySyntax(
                operator=operator,
                operand=_build_node(operand, text, base, source, expression_span),
                span=_op_span(op_token, base, source),
            )
        return _build_node(children[0], text, base, source, expression_span)

    if data == "atom":
        if len(children) == 3:
            # Parenthesized group: structurally transparent, like the
            # native LParen branch; the inner node keeps its own span.
            return _build_node(children[1], text, base, source, expression_span)
        return _build_node(children[0], text, base, source, expression_span)

    raise AssertionError(f"unexpected grammar node: {data}")


def _build_concat_chain(
    node: Tree[Token],
    text: str,
    base: int,
    source: SourceDocument,
    expression_span: SourceSpan,
) -> ExpressionSyntax:
    """Build a concat node from the left-associative ``||`` chain.

    Chain node shapes (``concat_expr``): ``[item]`` (a plain expression
    with no ``||``) and ``[left, "||", right]`` (a chain step).  Left
    chains flatten into ordered parts; a parenthesized right operand keeps
    its nested concat structure because parentheses are transparent.  The
    grammar only produces non-empty operands, so every accepted chain has
    at least two parts.
    """

    children = node.children
    if len(children) == 1:
        # No "||" in the slice: the root is a plain expression, not a concat.
        return _build_node(children[0], text, base, source, expression_span)
    left, _concat_token, right = children
    left_node = _build_node(left, text, base, source, expression_span)
    parts: list[ExpressionSyntax] = (
        list(left_node.parts) if isinstance(left_node, ConcatSyntax) else [left_node]
    )
    parts.append(_build_node(right, text, base, source, expression_span))
    return ConcatSyntax(parts=tuple(parts), span=_node_span(node, base, source))


def _build_call(
    node: Tree[Token],
    text: str,
    base: int,
    source: SourceDocument,
    expression_span: SourceSpan,
) -> ExpressionSyntax:
    children = node.children
    name_token = children[0]
    name = str(name_token)

    if len(children) == 3:
        # NAME "(" ")"
        arguments: tuple[ExpressionSyntax, ...] = ()
    else:
        # NAME LPAREN call_args RPAREN.  The call_args chain is a tree by
        # the grammar (call: NAME LPAREN call_args RPAREN); a token here is
        # an invalid tree and is rejected.
        arg_chain = children[2]
        if not isinstance(arg_chain, Tree):
            raise AssertionError(f"unexpected call argument node: {arg_chain!r}")
        arguments = _build_call_arguments(
            arg_chain, text, base, source, expression_span, outermost=True
        )

    call = CallSyntax(
        name=name,
        arguments=arguments,
        span=_node_span(node, base, source),
    )
    return _classify_format_arguments(call)


def _build_call_arguments(
    node: Tree[Token],
    text: str,
    base: int,
    source: SourceDocument,
    expression_span: SourceSpan,
    outermost: bool,
) -> tuple[ExpressionSyntax, ...]:
    """Build ordered call arguments from the left-recursive chain.

    Shapes (``call_args``): ``[arg]``, ``[left, ",", arg]``, and
    ``[left, ","]``.  Only the outermost chain may end with a bare comma
    (the native argument loop exits on ``)`` after a trailing comma); an
    inner bare comma is the native "expression expected" fault.
    """

    children = node.children

    if len(children) == 3:
        left, _comma_token, arg = children
        if not isinstance(left, Tree):
            raise AssertionError(f"unexpected argument chain child: {left!r}")
        arguments = list(
            _build_call_arguments(left, text, base, source, expression_span, outermost=False)
        )
        if _is_empty_arg(arg):
            if not outermost:
                # "f(a,,b)": an empty argument between commas is a native
                # "expression expected" fault, not a trailing comma.
                raise ExpressionParseError("expression expected", expression_span, text)
            # "f(a,)" / "f(a, b,)": the native argument loop exits on ")"
            # after a trailing comma; the empty tail is not an argument.
            return tuple(arguments)
        arguments.append(_build_node(arg, text, base, source, expression_span))
        return tuple(arguments)

    if len(children) == 2:
        left, _comma_token = children
        if not outermost:
            # "f(a,,b)": an empty argument between commas is a native
            # "expression expected" fault, not a trailing comma.
            raise ExpressionParseError(
                "expression expected",
                expression_span,
                text,
            )
        if not isinstance(left, Tree):
            raise AssertionError(f"unexpected argument chain child: {left!r}")
        return _build_call_arguments(left, text, base, source, expression_span, outermost=False)

    arg = children[0]
    if _is_empty_arg(arg):
        # "f(,a)": the native argument loop requires a primary first.
        raise ExpressionParseError("expression expected", expression_span, text)
    return (_build_node(arg, text, base, source, expression_span),)


def _is_empty_arg(arg: object) -> bool:
    """Whether an ``arg`` node carries no expression (epsilon production).

    The ``arg`` shape is always a ``Tree`` by the grammar (``arg:
    concat_expr |``); the epsilon production yields ``Tree("arg", [])``.
    A token here violates the grammar invariant and is explicitly rejected.
    """

    if not isinstance(arg, Tree):
        raise AssertionError(f"unexpected argument node: {arg!r}")
    return len(arg.children) == 0


def _classify_format_arguments(call: CallSyntax) -> CallSyntax:
    """Quote-classify the second argument of put/input calls.

    Mirrors the existing format normalization: only the second top-level
    argument of a case-insensitive ``put``/``input`` call, only when its
    bare lexeme matches the legacy token shape.  A bare lexeme is one the
    grammar lexed without quotes (a name, a numeric trailing-dot lexeme, or
    a digit-dot text lexeme); explicitly quoted strings never match.
    """

    if call.name.strip().upper() not in _FORMAT_FUNCTION_NAMES:
        return call
    if len(call.arguments) < 2:
        return call

    second = call.arguments[1]
    lexeme: str | None = None
    if isinstance(second, NameSyntax):
        lexeme = second.name
    elif isinstance(second, LiteralSyntax) and second.kind in ("numeric", "string"):
        lexeme = second.lexeme
    if lexeme is None or _BARE_FORMAT_TOKEN.fullmatch(lexeme) is None:
        return call

    arguments = list(call.arguments)
    arguments[1] = LiteralSyntax(kind="format", lexeme=lexeme, span=second.span)
    return CallSyntax(
        name=call.name,
        arguments=tuple(arguments),
        span=call.span,
        origin=call.origin,
    )


def _build_subscript(
    node: Tree[Token],
    text: str,
    base: int,
    source: SourceDocument,
    expression_span: SourceSpan,
) -> ExpressionSyntax:
    # NAME "[" expr "]" / NAME "{" expr "}": the native ArrayRef placement,
    # keeping the base as the name literal and no delimiter on the model.
    children = node.children
    name_token = children[0]
    if not isinstance(name_token, Token):
        raise AssertionError(f"unexpected subscript base: {name_token!r}")
    index_node = children[2]
    return SubscriptSyntax(
        base=NameSyntax(name=str(name_token), span=_op_span(name_token, base, source)),
        index=_build_node(index_node, text, base, source, expression_span),
        span=_node_span(node, base, source),
    )


def _operator_for(rule: str, op_token: Token) -> BinaryOperator:
    if rule == "cmp_expr":
        operator = _COMPARISON_ALIASES.get(str(op_token))
        if operator is None:
            raise AssertionError(f"unexpected comparison token: {str(op_token)!r}")
        return operator
    operator = _BINARY_OPERATORS[rule].get(op_token.type)
    if operator is None:
        raise AssertionError(f"unexpected operator token: {op_token!r}")
    # The mapping value is validated against the exact BinaryOperator
    # vocabulary before the (type-checker-only) cast; no unconstrained
    # string ever flows into a node.
    if operator not in _BINARY_OPERATOR_VALUES:
        raise AssertionError(f"unexpected normalized operator: {operator!r}")
    return cast(BinaryOperator, operator)


def _build_token(token: Token, base: int, source: SourceDocument) -> ExpressionSyntax:
    span = _op_span(token, base, source)
    kind = token.type
    lexeme = str(token)

    if kind == "STRING":
        return LiteralSyntax(kind="string", lexeme=lexeme, span=span)
    if kind == "MISSING":
        return LiteralSyntax(kind="missing", lexeme=lexeme, span=span)
    if kind == "NUMDOT_TEXT":
        # Native tokenizes "5." directly before ',' or ')' as text.
        return LiteralSyntax(kind="string", lexeme=lexeme, span=span)
    if kind == "NUMDOT_TRAILING":
        # A digit run ending in '.': a float when the embedded part has no
        # dot ("5."), a dotted identifier otherwise ("1.2.").
        if "." in lexeme[:-1]:
            return NameSyntax(name=lexeme, span=span)
        return LiteralSyntax(kind="numeric", lexeme=lexeme, span=span)
    if kind == "NUMBER_DOT":
        if lexeme.count(".") > 1:
            raise ExpressionParseError(
                f"invalid number literal: {lexeme}", span, lexeme
            )
        return LiteralSyntax(kind="numeric", lexeme=lexeme, span=span)
    if kind == "INTEGER":
        if int(lexeme) > _U64_MAX:
            raise ExpressionParseError(
                f"invalid integer literal: {lexeme}", span, lexeme
            )
        return LiteralSyntax(kind="numeric", lexeme=lexeme, span=span)
    if kind == "NAME":
        return NameSyntax(name=lexeme, span=span)

    raise AssertionError(f"unexpected literal token: {token!r}")


__all__ = [
    "ExpressionParseError",
    "GeneralExpressionResult",
    "collect_apply_references",
    "parse_expression",
    "parse_general_expression",
]
