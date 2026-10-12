"""Expression normalization and syntax classification."""

from __future__ import annotations

import ast
import re
from collections import deque
from collections.abc import Iterable, Iterator
from types import CodeType

from .naming import _column_key
from .syntax.models import (
    BinarySyntax,
    CallSyntax,
    ConcatSyntax,
    DeferredExpressionError,
    ExpressionSyntax,
    LiteralSyntax,
    NameSyntax,
    SubscriptSyntax,
    UnarySyntax,
)


class ExpressionSemantics:
    """Normalize DSL expressions and classify their AST function calls.

    The caches are intentionally implementation details.  Callers receive
    immutable strings, ASTs, tuples, or compiled code objects and never need
    to manage a mutable evaluation scope.
    """

    _ASSIGNMENT_OPERATOR = re.compile(r"(?<![<>=!])=(?!=)")
    _MISSING_LITERAL = re.compile(r"(?<![\w])\.(?![\w])")
    _BARE_FORMAT_TOKEN = re.compile(
        r"^\$?(?:[A-Za-z_]\w*|\d+(?:\.\d+)?|z\d+(?:\.\d+)?|comma\d+(?:\.\d+)?)\.$",
        re.IGNORECASE,
    )

    _RESERVED_FUNCTION_NAMES = frozenset(
        _column_key(keyword) for keyword in ("and", "or", "not", "in")
    )

    def __init__(self) -> None:
        self._normalized_expression_cache: dict[str, str] = {}
        self._compiled_expression_cache: dict[str, CodeType] = {}

    def prepare_expression(self, expression: str) -> str:
        """Return the cached normalized form of *expression*."""

        normalized = self._normalized_expression_cache.get(expression)
        if normalized is None:
            normalized = self.normalize_expression(expression)
            self._normalized_expression_cache[expression] = normalized
        return normalized

    def normalize_expression(self, expression: str) -> str:
        """Apply the existing assignment/operator normalization semantics."""

        operator_normalized = expression.replace("^=", "!=").replace("¬=", "!=").replace("~=", "!=")
        operator_normalized = operator_normalized.replace("><", " @__unsupported_operator__ ")
        format_normalized = self.normalize_format_function_arguments(operator_normalized)
        concat_normalized = self.normalize_concat_operator(format_normalized)
        replaced_equality = self._ASSIGNMENT_OPERATOR.sub("==", concat_normalized)
        return self._MISSING_LITERAL.sub("None", replaced_equality)

    def normalize_format_function_arguments(self, expression: str) -> str:
        """Quote bare PUT/INPUT format tokens while preserving nested calls."""

        result: list[str] = []
        cursor = 0
        length = len(expression)

        while cursor < length:
            ch = expression[cursor]

            if ch in {"'", '"'}:
                quoted, cursor = self._consume_quoted_segment(expression, cursor)
                result.append(quoted)
                continue

            if ch.isalpha() or ch == "_":
                start = cursor
                while cursor < length and (expression[cursor].isalnum() or expression[cursor] == "_"):
                    cursor += 1
                name = expression[start:cursor]
                whitespace_start = cursor
                while cursor < length and expression[cursor].isspace():
                    cursor += 1
                if cursor < length and expression[cursor] == "(":
                    bracketed, cursor = self._consume_bracket_segment(expression, cursor)
                    inner = bracketed[1:-1]
                    rewritten = self._normalize_function_call_arguments(name, inner)
                    result.append(name)
                    result.append(expression[whitespace_start : cursor - len(bracketed)])
                    result.append(f"({rewritten})")
                    continue

                result.append(expression[start:cursor])
                continue

            result.append(ch)
            cursor += 1

        return "".join(result)

    def _normalize_function_call_arguments(self, function_name: str, argument_text: str) -> str:
        normalized_arguments = self.normalize_format_function_arguments(argument_text)
        if _column_key(function_name) not in {_column_key("put"), _column_key("input")}:
            return normalized_arguments

        arguments = self._split_top_level_arguments(normalized_arguments)
        if len(arguments) < 2:
            return normalized_arguments

        arguments[1] = self._normalize_format_token(arguments[1])
        return ", ".join(arguments)

    def _split_top_level_arguments(self, text: str) -> list[str]:
        if not text.strip():
            return []

        arguments: list[str] = []
        current: list[str] = []
        cursor = 0
        length = len(text)

        while cursor < length:
            ch = text[cursor]

            if ch in {"'", '"'}:
                quoted, cursor = self._consume_quoted_segment(text, cursor)
                current.append(quoted)
                continue

            if ch in "([{":
                bracketed, cursor = self._consume_bracket_segment(text, cursor)
                current.append(bracketed)
                continue

            if ch == ",":
                arguments.append("".join(current).strip())
                current = []
                cursor += 1
                continue

            current.append(ch)
            cursor += 1

        arguments.append("".join(current).strip())
        return arguments

    def _normalize_format_token(self, token: str) -> str:
        stripped = token.strip()
        if self._BARE_FORMAT_TOKEN.fullmatch(stripped) is None:
            return stripped
        return f"'{stripped}'"

    def normalize_concat_operator(self, expression: str) -> str:
        """Rewrite top-level DSL ``||`` concatenation into its helper call."""

        parts: list[str] = []
        current: list[str] = []
        cursor = 0
        length = len(expression)

        while cursor < length:
            ch = expression[cursor]

            if ch in {"'", '"'}:
                quoted, next_cursor = self._consume_quoted_segment(expression, cursor)
                current.append(quoted)
                cursor = next_cursor
                continue

            if ch in "([{":
                bracketed, next_cursor = self._consume_bracket_segment(expression, cursor)
                current.append(bracketed)
                cursor = next_cursor
                continue

            if cursor + 1 < length and expression[cursor : cursor + 2] == "||":
                parts.append("".join(current).strip())
                current = []
                cursor += 2
                continue

            current.append(ch)
            cursor += 1

        parts.append("".join(current).strip())

        if len(parts) <= 1:
            return expression

        normalized_parts = [part if part else "''" for part in parts]
        result = normalized_parts[0]
        for part in normalized_parts[1:]:
            result = f"__dsl_concat__({result}, {part})"
        return result

    def parse_expression(self, expression: str) -> ast.Expression | None:
        """Parse a normalized expression, returning ``None`` on syntax error."""

        normalized = self.prepare_expression(expression)
        try:
            parsed = ast.parse(normalized, mode="eval")
        except SyntaxError:
            return None
        return parsed

    def compile_expression(self, expression: str) -> CodeType:
        """Compile a normalized expression, reusing the per-service cache."""

        normalized = self.prepare_expression(expression)
        compiled = self._compiled_expression_cache.get(normalized)
        if compiled is None:
            compiled = compile(normalized, "<limulus-scalar>", "eval")
            self._compiled_expression_cache[normalized] = compiled
        return compiled

    def function_names(self, expression: str) -> tuple[str, ...]:
        """Return direct-name call targets found by walking the normalized AST."""

        parsed = self.parse_expression(expression)
        if parsed is None:
            return ()
        return tuple(
            node.func.id
            for node in ast.walk(parsed)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        )

    # Descriptive aliases keep the boundary readable for planner consumers.
    iter_function_names = function_names
    classify_function_names = function_names

    def contains_function_call(self, expression: str) -> bool:
        return bool(self.function_names(expression))

    def find_unsupported_function(
        self,
        expression: str,
        supported_functions: Iterable[str],
    ) -> str | None:
        """Return the first AST function name absent from ``supported_functions``."""

        supported = {_column_key(name) for name in supported_functions}
        for function_name in self.function_names(expression):
            normalized = _column_key(function_name)
            if normalized in self._RESERVED_FUNCTION_NAMES:
                continue
            if normalized not in supported:
                return function_name
        return None

    # -- typed tree classification ------------------------------------------
    #
    # Classify typed nodes directly. Deferred expressions have no structure;
    # concat chains use virtual left-folded calls matching native lowering.

    _CONCAT_HELPER_FUNCTION_NAME = "__dsl_concat__"

    def _walk_typed_expression(
        self,
        expression: ExpressionSyntax,
    ) -> Iterator[ExpressionSyntax]:
        """Walk typed expressions outer-first, level by level.

        Concat nodes expand into the left-folded helper-call chain used by
        native lowering without modifying or reparsing their parts.
        """

        queue: deque[ExpressionSyntax] = deque([expression])
        while queue:
            node = queue.popleft()
            if isinstance(node, ConcatSyntax):
                node = _folded_concat_node(node)
            yield node
            queue.extend(_typed_expression_children(node))

    def function_names_typed(
        self,
        expression: ExpressionSyntax,
    ) -> tuple[str, ...]:
        """Direct-name call targets found by walking the typed expression.

        The concat chain contributes one ``__dsl_concat__`` per left fold,
        exactly as lowering builds it.  Call names keep the source case and
        dotted names are reported verbatim, as the typed structure holds them.
        """

        names: list[str] = []
        for node in self._walk_typed_expression(expression):
            function_name = _typed_function_name(node)
            if function_name is not None:
                names.append(function_name)
        return tuple(names)

    def column_names_typed(
        self,
        expression: ExpressionSyntax,
    ) -> tuple[str, ...]:
        """Column reference names found by walking the typed expression.

        Only ``NameSyntax`` occurrences count: the call name is never a
        column, and the put/input format literal stays a format token.
        """

        return tuple(
            node.name
            for node in self._walk_typed_expression(expression)
            if isinstance(node, NameSyntax)
        )

    def contains_function_call_typed(
        self,
        expression: ExpressionSyntax,
    ) -> bool:
        return bool(self.function_names_typed(expression))

    def find_unsupported_function_typed(
        self,
        expression: ExpressionSyntax,
        supported_functions: Iterable[str],
    ) -> str | None:
        """Typed counterpart of :meth:`find_unsupported_function`."""

        supported = {_column_key(name) for name in supported_functions}
        for function_name in self.function_names_typed(expression):
            normalized = _column_key(function_name)
            if normalized in self._RESERVED_FUNCTION_NAMES:
                continue
            if normalized not in supported:
                return function_name
        return None

    @staticmethod
    def _consume_quoted_segment(text: str, start: int) -> tuple[str, int]:
        quote_char = text[start]
        cursor = start + 1
        length = len(text)

        while cursor < length:
            if text[cursor] == quote_char:
                if cursor + 1 < length and text[cursor + 1] == quote_char:
                    cursor += 2
                    continue
                cursor += 1
                break
            cursor += 1

        return text[start:cursor], cursor

    def _consume_bracket_segment(self, text: str, start: int) -> tuple[str, int]:
        bracket_map = {"(": ")", "[": "]", "{": "}"}
        opening = text[start]
        closing = bracket_map[opening]
        depth = 1
        cursor = start + 1
        length = len(text)

        while cursor < length:
            ch = text[cursor]

            if ch in {"'", '"'}:
                _, cursor = self._consume_quoted_segment(text, cursor)
                continue

            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    inner = text[start + 1 : cursor]
                    normalized_inner = self.normalize_concat_operator(inner)
                    return f"{opening}{normalized_inner}{closing}", cursor + 1

            cursor += 1

        return text[start:], length


def _typed_expression_children(node: ExpressionSyntax) -> tuple[ExpressionSyntax, ...]:
    """Typed child expressions in breadth-first traversal order."""

    if isinstance(node, UnarySyntax):
        return (node.operand,)
    if isinstance(node, BinarySyntax):
        return (node.left, node.right)
    if isinstance(node, CallSyntax):
        return node.arguments
    if isinstance(node, SubscriptSyntax):
        return (node.base, node.index)
    if isinstance(node, ConcatSyntax):
        return node.parts
    # LiteralSyntax / NameSyntax / DeferredExpressionError hold no children.
    return ()


def _folded_concat_node(node: ConcatSyntax) -> ExpressionSyntax:
    """Fold one ``ConcatSyntax`` into its lowering-shaped helper-call chain.

    Left fold, exactly as lowering builds ``__dsl_concat__`` calls: parts
    ``[p1..pn]`` become ``dc(dc(p1, p2), …)``.  The virtual nodes are
    classification-only (built with the existing ``CallSyntax`` constructor
    over the SAME part nodes); the original tree, its parts, the native wire,
    and row execution are never touched, and no source text is re-parsed.
    """

    folded: ExpressionSyntax = node.parts[0]
    for part in node.parts[1:]:
        folded = CallSyntax(
            name=ExpressionSemantics._CONCAT_HELPER_FUNCTION_NAME,
            arguments=(folded, part),
            span=node.span,
        )
    return folded


def _typed_function_name(node: ExpressionSyntax) -> str | None:
    """Function realization of one typed node, or ``None`` for non-calls.

    ``CallSyntax`` keeps the source-case name (dotted names verbatim);
    concat nodes never reach this point because the walk unfolds them into
    the lowering-shaped helper chain first.  Deferred descriptors have no
    structure to classify.
    """

    if isinstance(node, CallSyntax):
        return node.name
    return None


# The design names this boundary a service; retain both common spellings for
# callers while keeping one implementation and one cache contract.
ExpressionSemanticsService = ExpressionSemantics
ExpressionSemanticService = ExpressionSemantics


_DEFAULT_SEMANTICS = ExpressionSemantics()


def normalize_expression(expression: str) -> str:
    return _DEFAULT_SEMANTICS.prepare_expression(expression)


def prepare_expression(expression: str) -> str:
    return _DEFAULT_SEMANTICS.prepare_expression(expression)


def compile_expression(expression: str) -> CodeType:
    return _DEFAULT_SEMANTICS.compile_expression(expression)


def function_names(expression: str) -> tuple[str, ...]:
    return _DEFAULT_SEMANTICS.function_names(expression)


def contains_function_call(expression: str) -> bool:
    return _DEFAULT_SEMANTICS.contains_function_call(expression)


def find_unsupported_function(expression: str, supported_functions: Iterable[str]) -> str | None:
    return _DEFAULT_SEMANTICS.find_unsupported_function(expression, supported_functions)


def function_names_typed(expression: ExpressionSyntax) -> tuple[str, ...]:
    return _DEFAULT_SEMANTICS.function_names_typed(expression)


def column_names_typed(expression: ExpressionSyntax) -> tuple[str, ...]:
    return _DEFAULT_SEMANTICS.column_names_typed(expression)


def contains_function_call_typed(expression: ExpressionSyntax) -> bool:
    return _DEFAULT_SEMANTICS.contains_function_call_typed(expression)


def find_unsupported_function_typed(
    expression: ExpressionSyntax, supported_functions: Iterable[str]
) -> str | None:
    return _DEFAULT_SEMANTICS.find_unsupported_function_typed(
        expression, supported_functions
    )


__all__ = [
    "ExpressionSemanticService",
    "ExpressionSemantics",
    "ExpressionSemanticsService",
    "column_names_typed",
    "compile_expression",
    "contains_function_call",
    "contains_function_call_typed",
    "find_unsupported_function",
    "find_unsupported_function_typed",
    "function_names",
    "function_names_typed",
    "normalize_expression",
    "prepare_expression",
]
