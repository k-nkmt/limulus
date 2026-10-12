"""Typed parser contract for dataset-reference ``WHERE=`` expressions.

The dataset option language is deliberately smaller than the Data Step
expression language.  It is an allow-list grammar for column-to-scalar
comparisons, null predicates, and boolean composition.  Parsed expressions
are immutable values and never contain executable Python syntax.

The restrictive ``WHERE`` expression types live in the pure syntax layer
(``limulus.syntax.models``) and are re-exported below as the same objects;
this module owns only the Arrow evaluation side of the contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeAlias, cast

from lark import Token, Tree
from lark.exceptions import UnexpectedInput

if TYPE_CHECKING:
    import pyarrow as pa
    import pyarrow.compute as pc

from .lark_support import build_lark_parser_from_file
from .models import Diagnostic
from .syntax.models import (
    WhereComparison,
    WhereComparisonOperator,
    WhereExpression,
    WhereLogicalExpression,
    WhereLogicalOperator,
    WhereNotExpression,
    WhereNullPredicate,
    WhereScalar,
)

# The restrictive dataset option WHERE AST is owned by the pure syntax layer
# (``limulus.syntax.models``) and re-exported here as the same types so the
# legacy evaluator keeps its import, constructor, and type identity.


class _DeferredModule:
    """Forward attribute access to a module imported on first use.

    Parsing (Lark only) must not require Arrow.  The Arrow evaluation side
    below is the sole consumer, so the heavy import happens the first time
    an evaluation actually touches Arrow instead of at module import.  The
    ``pa``/``pc`` names below are typed as real modules under
    ``TYPE_CHECKING`` and deferred proxies at runtime.
    """

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module: ModuleType | None = None

    def __getattr__(self, name: str):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return getattr(self._module, name)


if not TYPE_CHECKING:
    pa = cast(ModuleType, _DeferredModule("pyarrow"))
    pc = cast(ModuleType, _DeferredModule("pyarrow.compute"))

if TYPE_CHECKING:
    _ArrowVector: TypeAlias = pa.Array | pa.ChunkedArray
else:
    _ArrowVector = Any


class _ArrowComputeProtocol(Protocol):
    equal: Callable[[_ArrowVector, pa.Scalar], _ArrowVector]
    not_equal: Callable[[_ArrowVector, pa.Scalar], _ArrowVector]
    greater: Callable[[_ArrowVector, pa.Scalar], _ArrowVector]
    greater_equal: Callable[[_ArrowVector, pa.Scalar], _ArrowVector]
    less: Callable[[_ArrowVector, pa.Scalar], _ArrowVector]
    less_equal: Callable[[_ArrowVector, pa.Scalar], _ArrowVector]
    invert: Callable[[_ArrowVector], _ArrowVector]
    and_kleene: Callable[[_ArrowVector, _ArrowVector], _ArrowVector]
    or_kleene: Callable[[_ArrowVector, _ArrowVector], _ArrowVector]
    is_null: Callable[[_ArrowVector], _ArrowVector]
    fill_null: Callable[[_ArrowVector, bool], _ArrowVector]


class _TableColumnLookupProtocol(Protocol):
    def __getitem__(self, name: str) -> _ArrowVector:
        ...


_ARROW_COMPUTE = cast(_ArrowComputeProtocol, pc)

# The stable plan/golden WHERE serializer moved to the pure syntax layer so
# the legacy DTO serialization stays importable without Arrow.  Re-exported
# here as the same function object to keep every existing import path.
from .syntax.legacy import where_expression_to_dict  # noqa: E402


@dataclass(frozen=True)
class DatasetWhereParseResult:
    """Result of parsing one dataset option WHERE expression."""

    expression: WhereExpression | None = None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(diagnostic.severity == "error" for diagnostic in self.diagnostics)


@dataclass(frozen=True)
class DatasetWhereEvaluationResult:
    """Arrow boolean mask or a structured dataset-option diagnostic."""

    mask: object | None = None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(diagnostic.severity == "error" for diagnostic in self.diagnostics)


class DatasetWhereParser:
    """Parse the dedicated, non-executable dataset option WHERE grammar."""

    def __init__(self) -> None:
        self._parser = build_lark_parser_from_file(
            "dataset_option_where.lark",
            start="start",
            parser="lalr",
            propagate_positions=False,
        )

    def parse(self, source: str) -> DatasetWhereParseResult:
        """Return a typed expression or one structured unsupported diagnostic.

        ``source`` may be either the WHERE body or an option-style
        ``WHERE=(...)`` wrapper.  Wrapper removal is intentionally performed
        here so callers can use this service independently of the main parser.
        """

        if not isinstance(source, str):
            return DatasetWhereParseResult(
                diagnostics=(
                    _where_diagnostic(
                        source,
                        reason="unsupported WHERE= input must be a string",
                    ),
                )
            )

        text = _unwrap_where_option(source)
        if not text:
            return DatasetWhereParseResult(
                diagnostics=(_where_diagnostic(text, reason="incomplete WHERE= expression"),)
            )
        if self._parser is None:
            return DatasetWhereParseResult(
                diagnostics=(_where_diagnostic(text, reason="unsupported WHERE= parser is unavailable"),)
            )

        try:
            tree = self._parser.parse(text)
            expression = _expression_from_tree(tree)
        except (UnexpectedInput, ValueError, TypeError) as error:
            return DatasetWhereParseResult(
                diagnostics=(
                    _where_diagnostic(text, reason=_failure_reason(text, error)),
                )
            )
        except Exception as error:  # pragma: no cover - defensive parser boundary
            return DatasetWhereParseResult(
                diagnostics=(
                    _where_diagnostic(text, reason=f"unsupported WHERE= syntax ({type(error).__name__})"),
                )
            )

        return DatasetWhereParseResult(expression=expression)


def parse_dataset_option_where(source: str) -> DatasetWhereParseResult:
    """Convenience function for parsing one dataset option WHERE expression."""

    return DatasetWhereParser().parse(source)


_WHERE_COMPARISON_KERNELS: dict[str, Callable[[Any, Any], Any]] | None = None


def _comparison_kernels() -> dict[str, Callable[[Any, Any], Any]]:
    """Build the Arrow comparison kernel table on first evaluation use."""

    global _WHERE_COMPARISON_KERNELS
    if _WHERE_COMPARISON_KERNELS is None:
        _WHERE_COMPARISON_KERNELS = {
            "=": _ARROW_COMPUTE.equal,
            "==": _ARROW_COMPUTE.equal,
            "!=": _ARROW_COMPUTE.not_equal,
            ">": _ARROW_COMPUTE.greater,
            ">=": _ARROW_COMPUTE.greater_equal,
            "<": _ARROW_COMPUTE.less,
            "<=": _ARROW_COMPUTE.less_equal,
        }
    return _WHERE_COMPARISON_KERNELS


def evaluate_dataset_where(
    expression: WhereExpression,
    table: object,
    source_name: str,
) -> DatasetWhereEvaluationResult:
    """Evaluate a typed WHERE expression into a deterministic Arrow mask.

    Recursive evaluation keeps Arrow's nullable (Kleene) boolean values until
    the root call.  The public result then fills unresolved nulls with
    ``False`` so filtering never treats an indeterminate row as selected.
    No row materialization or Python expression evaluation occurs here.
    """

    result = _evaluate_dataset_where_mask(expression, table, source_name)
    if result.mask is None or result.diagnostics:
        return result
    try:
        mask = cast(_ArrowVector, result.mask)
        return DatasetWhereEvaluationResult(mask=_ARROW_COMPUTE.fill_null(mask, False))
    except Exception as error:
        return DatasetWhereEvaluationResult(
            diagnostics=(
                _evaluation_diagnostic(
                    source_name=source_name,
                    column="<mask>",
                    operator="",
                    reason=f"final boolean mask normalization failed: {type(error).__name__}: {error}",
                ),
            )
        )


def _evaluate_dataset_where_mask(
    expression: WhereExpression,
    table: object,
    source_name: str,
) -> DatasetWhereEvaluationResult:
    """Evaluate one expression while preserving nullable Arrow mask values."""

    if isinstance(expression, (WhereComparison, WhereNullPredicate)):
        return _evaluate_dataset_where_leaf(expression, table, source_name)

    if isinstance(expression, WhereNotExpression):
        operand = _evaluate_dataset_where_mask(expression.operand, table, source_name)
        if operand.diagnostics or operand.mask is None:
            return operand
        try:
            mask = cast(_ArrowVector, operand.mask)
            return DatasetWhereEvaluationResult(mask=_ARROW_COMPUTE.invert(mask))
        except Exception as error:
            return DatasetWhereEvaluationResult(
                diagnostics=(
                    _evaluation_diagnostic(
                        source_name=source_name,
                        column="<expression>",
                        operator="NOT",
                        reason=f"boolean kernel failed: {type(error).__name__}: {error}",
                    ),
                )
            )

    if isinstance(expression, WhereLogicalExpression):
        if len(expression.operands) < 2:
            return DatasetWhereEvaluationResult(
                diagnostics=(
                    _evaluation_diagnostic(
                        source_name=source_name,
                        column="<expression>",
                        operator=expression.operator.upper(),
                        reason="logical expression requires at least two operands",
                    ),
                )
            )
        if expression.operator not in {"and", "or"}:
            return DatasetWhereEvaluationResult(
                diagnostics=(
                    _evaluation_diagnostic(
                        source_name=source_name,
                        column="<expression>",
                        operator=str(expression.operator),
                        reason="unsupported logical operator",
                    ),
                )
            )
        first = _evaluate_dataset_where_mask(expression.operands[0], table, source_name)
        if first.diagnostics or first.mask is None:
            return first
        mask = cast(_ArrowVector, first.mask)
        kernel = _ARROW_COMPUTE.and_kleene if expression.operator == "and" else _ARROW_COMPUTE.or_kleene
        for operand_expression in expression.operands[1:]:
            operand = _evaluate_dataset_where_mask(operand_expression, table, source_name)
            if operand.diagnostics or operand.mask is None:
                return operand
            try:
                operand_mask = cast(_ArrowVector, operand.mask)
                mask = kernel(mask, operand_mask)
            except Exception as error:
                return DatasetWhereEvaluationResult(
                    diagnostics=(
                        _evaluation_diagnostic(
                            source_name=source_name,
                            column="<expression>",
                            operator=expression.operator.upper(),
                            reason=f"boolean kernel failed: {type(error).__name__}: {error}",
                        ),
                    )
                )
        return DatasetWhereEvaluationResult(mask=mask)

    return DatasetWhereEvaluationResult(
        diagnostics=(
            _evaluation_diagnostic(
                source_name=source_name,
                column="<expression>",
                operator="",
                reason="unsupported WHERE= expression kind",
            ),
        )
    )


def _evaluate_dataset_where_leaf(
    expression: WhereComparison | WhereNullPredicate,
    table: object,
    source_name: str,
) -> DatasetWhereEvaluationResult:
    """Evaluate one comparison/null leaf without normalizing nullable values."""

    if isinstance(expression, WhereComparison):
        column_name = expression.column
        operator = expression.operator
    else:
        column_name = expression.column
        operator = "IS NOT NULL" if expression.negated else "IS NULL"

    names, names_error = _column_names(table)
    if names_error is not None:
        return DatasetWhereEvaluationResult(
            diagnostics=(
                _evaluation_diagnostic(
                    source_name=source_name,
                    column=column_name,
                    operator=operator,
                    reason=names_error,
                ),
            )
        )

    matches = [name for name in names if name.casefold() == column_name.casefold()]
    if not matches:
        return DatasetWhereEvaluationResult(
            diagnostics=(
                _evaluation_diagnostic(
                    source_name=source_name,
                    column=column_name,
                    operator=operator,
                    reason="column was not found (case-insensitive resolution)",
                ),
            )
        )
    if len(matches) > 1:
        return DatasetWhereEvaluationResult(
            diagnostics=(
                _evaluation_diagnostic(
                    source_name=source_name,
                    column=column_name,
                    operator=operator,
                    reason=f"column resolution is ambiguous ({', '.join(matches)})",
                ),
            )
        )

    try:
        column = _table_column(table, matches[0])
        if isinstance(expression, WhereNullPredicate):
            if _is_nested_where_operand(column):
                return DatasetWhereEvaluationResult(
                    diagnostics=(
                        _evaluation_diagnostic(
                            source_name=source_name,
                            column=column_name,
                            operator=operator,
                            reason=(
                                "complex access is unsupported for nested Arrow "
                                f"type {column.type}"
                            ),
                        ),
                    )
                )
            mask = _ARROW_COMPUTE.is_null(column)
            if expression.negated:
                mask = _ARROW_COMPUTE.invert(mask)
            return DatasetWhereEvaluationResult(mask=mask)

        if not isinstance(expression.value, (str, int, float, bool)):
            raise TypeError(
                f"scalar value has unsupported type {type(expression.value).__name__}"
            )
        scalar = pa.scalar(expression.value)
        mask = _comparison_kernels()[expression.operator](column, scalar)
        return DatasetWhereEvaluationResult(mask=mask)
    except Exception as error:
        return DatasetWhereEvaluationResult(
            diagnostics=(
                _evaluation_diagnostic(
                    source_name=source_name,
                    column=column_name,
                    operator=operator,
                    reason=f"comparison kernel/type conversion failed: {type(error).__name__}: {error}",
                ),
            )
        )


def _is_nested_where_operand(column: _ArrowVector) -> bool:
    data_type = getattr(column, "type", None)
    return any(
        predicate(data_type)
        for predicate in (
            pa.types.is_list,
            pa.types.is_large_list,
            pa.types.is_fixed_size_list,
            pa.types.is_struct,
        )
    )


def _column_names(table: object) -> tuple[tuple[str, ...], str | None]:
    names = getattr(table, "column_names", None)
    if names is None:
        schema = getattr(table, "schema", None)
        names = getattr(schema, "names", None)
    if names is None:
        return (), "input table does not expose Arrow column names"
    try:
        normalized = tuple(str(name) for name in names)
    except Exception as error:
        return (), f"input table column names are unavailable: {type(error).__name__}"
    return normalized, None


def _table_column(table: object, name: str) -> _ArrowVector:
    column_method = getattr(table, "column", None)
    if callable(column_method):
        return cast(_ArrowVector, column_method(name))
    return cast(_TableColumnLookupProtocol, table)[name]


def _evaluation_diagnostic(
    *,
    source_name: str,
    column: str,
    operator: str,
    reason: str,
) -> Diagnostic:
    return Diagnostic(
        code="RUNTIME_DATASET_OPTION_INVALID",
        severity="error",
        location=f"dataset-option WHERE={source_name}",
        stage="input preparation",
        message=(
            f"dataset option WHERE= evaluation failed for source {source_name!r}, "
            f"column {column!r}, operator {operator!r}: {reason}"
        ),
    )


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


def _where_diagnostic(source: object, *, reason: str) -> Diagnostic:
    return Diagnostic(
        code="PARSE_DATASET_WHERE_UNSUPPORTED",
        severity="error",
        location="dataset-option WHERE=",
        stage="parse",
        message=f"dataset option WHERE= is invalid: {reason}",
        source_text=source if isinstance(source, str) else None,
    )


__all__ = [
    "DatasetWhereEvaluationResult",
    "DatasetWhereParseResult",
    "DatasetWhereParser",
    "WhereComparison",
    "WhereComparisonOperator",
    "WhereExpression",
    "WhereLogicalExpression",
    "WhereLogicalOperator",
    "WhereNotExpression",
    "WhereNullPredicate",
    "WhereScalar",
    "evaluate_dataset_where",
    "parse_dataset_option_where",
    "where_expression_to_dict",
]
