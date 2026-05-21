from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from lark import Tree
from lark.exceptions import UnexpectedInput
import pyarrow as pa
import polars as pl

from ._naming import _column_key
from .evaluator import ExpressionEvaluator
from .format_registry import FormatRegistry
from .lark_support import build_lark_parser_from_file
from .models import Diagnostic, DiagnosticLabel, DiagnosticSpan, RenderRequest
from .renderer import render_diagnostics
from .runtime import PDVRuntimeService


_ASSIGNMENT_NORMALIZER = ExpressionEvaluator(lambda: {})
_UNSUPPORTED = object()
_ASSIGN_FUNCTION_REGISTRY = {
    "upcase",
    "lowcase",
    "propcase",
    "cat",
    "cats",
    "catt",
    "catx",
    "index",
    "find",
    "tranwrd",
    "translate",
    "length",
    "lengthn",
    "strip",
    "reverse",
    "repeat",
    "countw",
    "round",
    "put",
    "input",
    "hour",
}

_DEFAULT_FORMAT_REGISTRY = FormatRegistry()


@dataclass(frozen=True)
class _CaseWhenClause:
    condition: str
    value_expression: str


@dataclass(frozen=True)
class _AssignmentSpec:
    target: str
    kind: str
    value: Any = None
    expression: str | None = None
    clauses: tuple[_CaseWhenClause, ...] = ()
    else_expression: str | None = None


class _CaseWhenParseError(ValueError):
    def __init__(self, diagnostic: Diagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


class _CaseWhenParser:
    def __init__(self) -> None:
        self._parser = build_lark_parser_from_file("case_when.lark")
        self._expression_evaluator = ExpressionEvaluator(lambda: {})

    def parse(self, expression: str) -> tuple[tuple[_CaseWhenClause, ...], str | None] | None:
        if not expression.strip().lower().startswith("case"):
            return None

        if self._parser is None:
            raise _CaseWhenParseError(
                self._whole_expression_diagnostic(
                    expression,
                    message="CASE WHEN parser is unavailable.",
                )
            )

        try:
            parsed = self._parser.parse(expression)
        except UnexpectedInput as error:
            raise _CaseWhenParseError(self._unexpected_input_diagnostic(expression, error)) from error
        except Exception as error:
            raise _CaseWhenParseError(
                self._whole_expression_diagnostic(
                    expression,
                    message="Invalid CASE WHEN expression.",
                    notes=(str(error),),
                )
            ) from error

        root = parsed if parsed.data == "case_expr" else parsed.children[0]
        clauses: list[_CaseWhenClause] = []
        else_expression: str | None = None

        for child in root.children:
            if not isinstance(child, Tree):
                continue

            if child.data == "when_clause":
                expression_nodes = [node for node in child.children if isinstance(node, Tree) and node.data == "expression"]
                if len(expression_nodes) != 2:
                    raise _CaseWhenParseError(
                        self._whole_expression_diagnostic(
                            expression,
                            message="CASE WHEN clause must contain both condition and value expressions.",
                        )
                    )
                self._validate_expression_node(
                    expression,
                    expression_nodes[0],
                    message="Invalid CASE WHEN condition expression.",
                    label_message="when condition",
                )
                self._validate_expression_node(
                    expression,
                    expression_nodes[1],
                    message="Invalid CASE WHEN value expression.",
                    label_message="then expression",
                )
                clauses.append(
                    _CaseWhenClause(
                        condition=self._slice_text(expression, expression_nodes[0]),
                        value_expression=self._slice_text(expression, expression_nodes[1]),
                    )
                )
                continue

            if child.data == "else_clause":
                expression_nodes = [node for node in child.children if isinstance(node, Tree) and node.data == "expression"]
                if len(expression_nodes) != 1:
                    raise _CaseWhenParseError(
                        self._whole_expression_diagnostic(
                            expression,
                            message="ELSE clause must contain exactly one expression.",
                        )
                    )
                self._validate_expression_node(
                    expression,
                    expression_nodes[0],
                    message="Invalid ELSE expression.",
                    label_message="else expression",
                )
                else_expression = self._slice_text(expression, expression_nodes[0])

        if not clauses:
            raise _CaseWhenParseError(
                self._whole_expression_diagnostic(
                    expression,
                    message="CASE WHEN expression requires at least one WHEN ... THEN clause.",
                )
            )
        return tuple(clauses), else_expression

    @staticmethod
    def _slice_text(source: str, node: Tree) -> str:
        return source[node.meta.start_pos:node.meta.end_pos].strip()

    def _validate_expression_node(
        self,
        source_text: str,
        node: Tree,
        *,
        message: str,
        label_message: str,
    ) -> None:
        expression_text = self._slice_text(source_text, node)
        if not expression_text:
            span = self._span_from_node(node)
            raise _CaseWhenParseError(
                Diagnostic(
                    code="COLUMN_API_CASE_WHEN_PARSE_ERROR",
                    severity="error",
                    message=message,
                    span=span,
                    labels=(DiagnosticLabel(span=span, message="syntax error"),),
                    notes=(f"Context: {label_message}.",),
                    source_text=source_text,
                )
            )

        normalized = self._expression_evaluator.prepare_expression(expression_text)
        try:
            compile(normalized, "<limulus-scalar>", "eval")
        except Exception as error:
            span = self._span_from_node(node)
            raise _CaseWhenParseError(
                Diagnostic(
                    code="COLUMN_API_CASE_WHEN_PARSE_ERROR",
                    severity="error",
                    message=message,
                    span=span,
                    labels=(DiagnosticLabel(span=span, message="syntax error"),),
                    notes=(f"Context: {label_message}.", str(error)),
                    source_text=source_text,
                )
            ) from error

    def _unexpected_input_diagnostic(self, source_text: str, error: UnexpectedInput) -> Diagnostic:
        span = self._span_from_unexpected_input(source_text, error)
        return Diagnostic(
            code="COLUMN_API_CASE_WHEN_PARSE_ERROR",
            severity="error",
            message="Invalid CASE WHEN expression.",
            span=span,
            labels=(DiagnosticLabel(span=span, message="syntax error"),),
            notes=(str(error),),
            source_text=source_text,
        )

    def _whole_expression_diagnostic(
        self,
        source_text: str,
        *,
        message: str,
        notes: tuple[str, ...] = (),
    ) -> Diagnostic:
        span = self._whole_expression_span(source_text)
        labels = (DiagnosticLabel(span=span, message="case when"),) if span is not None else ()
        return Diagnostic(
            code="COLUMN_API_CASE_WHEN_PARSE_ERROR",
            severity="error",
            message=message,
            span=span,
            labels=labels,
            notes=notes,
            source_text=source_text,
        )

    def _span_from_unexpected_input(self, source_text: str, error: UnexpectedInput) -> DiagnosticSpan:
        start = max(getattr(error, "pos_in_stream", 0), 0)
        line = max(getattr(error, "line", 1), 1)
        column = max(getattr(error, "column", 1), 1)
        end = self._unexpected_span_end(source_text, start)
        return DiagnosticSpan(
            start=start,
            end=end,
            line=line,
            column=column,
            end_line=line,
            end_column=column + max(end - start, 1),
            source_id="<case-when>",
        )

    def _whole_expression_span(self, source_text: str) -> DiagnosticSpan | None:
        stripped = source_text.strip()
        if not stripped:
            return None
        start = source_text.find(stripped)
        end = start + len(stripped)
        line = source_text[:start].count("\n") + 1
        last_newline = source_text.rfind("\n", 0, start)
        column = start + 1 if last_newline < 0 else start - last_newline
        return DiagnosticSpan(
            start=start,
            end=end,
            line=line,
            column=column,
            end_line=line + stripped.count("\n"),
            end_column=(column + len(stripped)) if "\n" not in stripped else None,
            source_id="<case-when>",
        )

    @staticmethod
    def _span_from_node(node: Tree) -> DiagnosticSpan:
        return DiagnosticSpan(
            start=node.meta.start_pos,
            end=node.meta.end_pos,
            line=node.meta.line,
            column=node.meta.column,
            end_line=node.meta.end_line,
            end_column=node.meta.end_column,
            source_id="<case-when>",
        )

    @staticmethod
    def _unexpected_span_end(source_text: str, start: int) -> int:
        if start >= len(source_text):
            return start + 1
        match = re.match(r"[^\s]+", source_text[start:])
        if match is None:
            return min(start + 1, len(source_text))
        return start + max(len(match.group(0)), 1)


_CASE_WHEN_PARSER = _CaseWhenParser()


def _with_source_id(diagnostic: Diagnostic, source_id: str) -> Diagnostic:
    span = None
    if diagnostic.span is not None:
        span = DiagnosticSpan(
            start=diagnostic.span.start,
            end=diagnostic.span.end,
            line=diagnostic.span.line,
            column=diagnostic.span.column,
            end_line=diagnostic.span.end_line,
            end_column=diagnostic.span.end_column,
            source_id=source_id,
        )

    labels = tuple(
        DiagnosticLabel(
            span=DiagnosticSpan(
                start=label.span.start,
                end=label.span.end,
                line=label.span.line,
                column=label.span.column,
                end_line=label.span.end_line,
                end_column=label.span.end_column,
                source_id=source_id,
            ),
            message=label.message,
            kind=label.kind,
        )
        for label in diagnostic.labels
    )

    return Diagnostic(
        code=diagnostic.code,
        severity=diagnostic.severity,
        message=diagnostic.message,
        location=diagnostic.location,
        stage=diagnostic.stage,
        span=span,
        labels=labels,
        notes=diagnostic.notes,
        source_text=diagnostic.source_text,
    )


def _render_column_api_diagnostic(diagnostic: Diagnostic, *, source_id: str) -> str:
    updated = _with_source_id(diagnostic, source_id)
    return render_diagnostics(
        RenderRequest(
            source_id=source_id,
            source_text=updated.source_text,
            diagnostics=(updated,),
        )
    )


def normalize_column_sequence(columns: str | Sequence[str] | None) -> tuple[str, ...]:
    if columns is None:
        return ()
    if isinstance(columns, str):
        return (columns,)
    return tuple(str(column) for column in columns)


def resolve_column_name(table: pa.Table, column: str, *, parameter_name: str) -> str:
    if column in table.column_names:
        return column

    column_key = _column_key(column)
    matches = [name for name in table.column_names if _column_key(name) == column_key]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Ambiguous column reference for Session.{parameter_name}: {column}")
    raise KeyError(f"Column not found for Session.{parameter_name}: {column}")


def resolve_existing_columns(
    table: pa.Table,
    columns: str | Sequence[str] | None,
    *,
    parameter_name: str,
) -> tuple[str, ...]:
    normalized_columns = normalize_column_sequence(columns)
    return tuple(resolve_column_name(table, column, parameter_name=parameter_name) for column in normalized_columns)


def resolve_transpose_var_columns(
    table: pa.Table,
    *,
    by_columns: Sequence[str],
    id_columns: Sequence[str],
    var: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if var is not None:
        resolved = resolve_existing_columns(table, var, parameter_name="var")
        if resolved:
            return resolved

    excluded = {_column_key(name) for name in (*by_columns, *id_columns)}
    resolved = tuple(name for name in table.column_names if _column_key(name) not in excluded)
    if not resolved:
        raise ValueError("Session.transpose requires at least one VAR column")
    return resolved


def transpose_table(
    table: pa.Table,
    *,
    by_columns: tuple[str, ...],
    id_columns: tuple[str, ...],
    var_columns: tuple[str, ...],
    materialize_table: Callable[..., pa.Table],
) -> pa.Table:
    use_id_layout = bool(id_columns)
    max_group_width = 0
    type_hints: dict[str, pa.DataType] = {
        column_name: table.schema.field(column_name).type for column_name in by_columns
    }
    if use_id_layout and len(id_columns) != 1:
        raise ValueError("Session.transpose currently supports exactly one ID column")
    if use_id_layout and len(var_columns) != 1:
        raise ValueError("Session.transpose with id currently supports exactly one VAR column")

    source_rows = table.to_pylist()
    ordered_output_columns: list[str] = []
    output_rows: list[dict[str, Any]] = []

    for group_key, group_rows in _group_rows(source_rows, by_columns=by_columns):
        max_group_width = max(max_group_width, len(group_rows))
        base_row = {column: group_key[index] for index, column in enumerate(by_columns)}

        if not use_id_layout:
            for variable_name in var_columns:
                transposed_row = dict(base_row)
                transposed_row["_NAME_"] = variable_name
                for row_index, row in enumerate(group_rows, start=1):
                    transposed_row[f"COL{row_index}"] = row.get(variable_name)
                output_rows.append(transposed_row)
            continue

        id_column = id_columns[0]
        value_column = var_columns[0]
        transposed_row = dict(base_row)
        seen_ids: set[str] = set()

        for row in group_rows:
            raw_id_value = row.get(id_column)
            if raw_id_value is None or raw_id_value == "":
                raise ValueError(f"Missing ID value for Session.transpose: {id_column}")

            output_column = str(raw_id_value)
            if output_column in seen_ids:
                raise ValueError(f"Duplicate ID value for Session.transpose: {output_column}")

            seen_ids.add(output_column)
            if output_column not in ordered_output_columns:
                ordered_output_columns.append(output_column)
            transposed_row[output_column] = row.get(value_column)

        for output_column in ordered_output_columns:
            transposed_row.setdefault(output_column, None)
        output_rows.append(transposed_row)

    if use_id_layout:
        for output_row in output_rows:
            for output_column in ordered_output_columns:
                output_row.setdefault(output_column, None)

    if use_id_layout:
        value_type = table.schema.field(var_columns[0]).type
        type_hints.update({column_name: value_type for column_name in ordered_output_columns})
    elif var_columns:
        common_type = table.schema.field(var_columns[0]).type
        if all(table.schema.field(column_name).type == common_type for column_name in var_columns[1:]):
            for row_index in range(1, max_group_width + 1):
                type_hints[f"COL{row_index}"] = common_type

    return materialize_table(output_rows, table, type_hints=type_hints)


def assign_columns(
    table: pa.Table,
    assignments: Mapping[str, Any],
    *,
    finalize_table: Callable[[pa.Table, pa.Table], pa.Table],
    format_registry: FormatRegistry | None = None,
) -> pa.Table:
    specs = _normalize_assignments(assignments)
    if not specs:
        return table

    frame = pl.from_arrow(table)
    current_columns = tuple(frame.columns)
    registry = format_registry or _DEFAULT_FORMAT_REGISTRY

    for spec in specs:
        target_name = _resolve_assignment_target_name(current_columns, spec.target)
        expression = _build_assignment_expression(spec, column_names=current_columns, format_registry=registry)
        frame = frame.with_columns(expression.alias(target_name))
        current_columns = tuple(frame.columns)

    return finalize_table(table, frame.to_arrow())


def _build_assignment_expression(
    spec: _AssignmentSpec,
    *,
    column_names: Sequence[str],
    format_registry: FormatRegistry,
) -> pl.Expr:
    if spec.kind == "literal":
        return pl.lit(spec.value)

    if spec.kind in {"expr", "func"}:
        return _translate_expression_to_polars(
            spec.expression or "",
            column_names=column_names,
            format_registry=format_registry,
        )


    if spec.kind == "case_when":
        when_expr = None
        for clause in spec.clauses:
            condition = _translate_expression_to_polars(
                clause.condition,
                column_names=column_names,
                format_registry=format_registry,
            )
            value_expression = _translate_expression_to_polars(
                clause.value_expression,
                column_names=column_names,
                format_registry=format_registry,
            )
            when_expr = (
                pl.when(condition).then(value_expression)
                if when_expr is None
                else when_expr.when(condition).then(value_expression)
            )

        if when_expr is None:
            raise ValueError("CASE WHEN expression requires at least one WHEN clause")

        else_expression = (
            pl.lit(None)
            if spec.else_expression is None
            else _translate_expression_to_polars(
                spec.else_expression,
                column_names=column_names,
                format_registry=format_registry,
            )
        )
        return when_expr.otherwise(else_expression)

    raise ValueError(f"Unsupported assignment kind: {spec.kind}")


def _translate_expression_to_polars(
    expression: str,
    *,
    column_names: Sequence[str],
    format_registry: FormatRegistry,
) -> pl.Expr:
    unsupported_function = _find_unsupported_function(
        expression,
        supported_functions=_supported_assign_function_names(),
    )
    if unsupported_function is not None:
        raise ValueError(f"Unsupported function: {unsupported_function}")

    normalized = _ASSIGNMENT_NORMALIZER.prepare_expression(expression)
    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Invalid assign expression: {expression}: {error}") from error
    return _translate_ast_node(parsed.body, column_names=column_names, format_registry=format_registry)


def _translate_ast_node(node: ast.AST, *, column_names: Sequence[str], format_registry: FormatRegistry) -> pl.Expr:
    if isinstance(node, ast.Constant):
        return pl.lit(node.value)

    if isinstance(node, ast.Name):
        return pl.col(_resolve_assignment_source_name(column_names, node.id))

    if isinstance(node, ast.BinOp):
        left = _translate_ast_node(node.left, column_names=column_names, format_registry=format_registry)
        right = _translate_ast_node(node.right, column_names=column_names, format_registry=format_registry)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left.pow(right)
        raise ValueError(f"Unsupported operator in assign expression: {ast.dump(node.op)}")

    if isinstance(node, ast.UnaryOp):
        operand = _translate_ast_node(node.operand, column_names=column_names, format_registry=format_registry)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Not):
            return ~operand
        raise ValueError(f"Unsupported unary operator in assign expression: {ast.dump(node.op)}")

    if isinstance(node, ast.BoolOp):
        values = [
            _translate_ast_node(value, column_names=column_names, format_registry=format_registry)
            for value in node.values
        ]
        if not values:
            raise ValueError("Boolean expression requires at least one operand")
        combined = values[0]
        for value in values[1:]:
            if isinstance(node.op, ast.And):
                combined = combined & value
            elif isinstance(node.op, ast.Or):
                combined = combined | value
            else:
                raise ValueError(f"Unsupported boolean operator in assign expression: {ast.dump(node.op)}")
        return combined

    if isinstance(node, ast.Compare):
        left = _translate_ast_node(node.left, column_names=column_names, format_registry=format_registry)
        comparisons: list[pl.Expr] = []
        current_left = left
        for operator, comparator in zip(node.ops, node.comparators):
            right = _translate_ast_node(comparator, column_names=column_names, format_registry=format_registry)
            if isinstance(operator, ast.Eq):
                comparisons.append(current_left == right)
            elif isinstance(operator, ast.NotEq):
                comparisons.append(current_left != right)
            elif isinstance(operator, ast.Gt):
                comparisons.append(current_left > right)
            elif isinstance(operator, ast.GtE):
                comparisons.append(current_left >= right)
            elif isinstance(operator, ast.Lt):
                comparisons.append(current_left < right)
            elif isinstance(operator, ast.LtE):
                comparisons.append(current_left <= right)
            else:
                raise ValueError(f"Unsupported comparison in assign expression: {ast.dump(operator)}")
            current_left = right

        combined = comparisons[0]
        for comparison in comparisons[1:]:
            combined = combined & comparison
        return combined

    if isinstance(node, ast.Call):
        return _translate_call_node(node, column_names=column_names, format_registry=format_registry)

    raise ValueError(f"Unsupported assign expression: {ast.dump(node)}")


def _translate_call_node(node: ast.Call, *, column_names: Sequence[str], format_registry: FormatRegistry) -> pl.Expr:
    if not isinstance(node.func, ast.Name):
        raise ValueError("Unsupported function reference in assign expression")

    function_name = _column_key(node.func.id)
    arg_exprs = [
        _translate_ast_node(argument, column_names=column_names, format_registry=format_registry)
        for argument in node.args
    ]
    arg_literals = [_extract_literal_value(argument) for argument in node.args]

    if function_name == _column_key("upcase"):
        _require_arity(node.func.id, arg_exprs, expected=1)
        return _expr_upcase(arg_exprs[0])
    if function_name == _column_key("lowcase"):
        _require_arity(node.func.id, arg_exprs, expected=1)
        return _expr_lowcase(arg_exprs[0])
    if function_name == _column_key("propcase"):
        _require_arity(node.func.id, arg_exprs, expected=1)
        return _expr_propcase(arg_exprs[0])
    if function_name == _column_key("cat"):
        return _expr_cat(*arg_exprs)
    if function_name == _column_key("cats"):
        return _expr_cats(*arg_exprs)
    if function_name == _column_key("catt"):
        return _expr_catt(*arg_exprs)
    if function_name == _column_key("catx"):
        if len(arg_exprs) < 1:
            raise ValueError("catx() requires at least one argument")
        delimiter = arg_literals[0]
        if delimiter is _UNSUPPORTED:
            raise ValueError("Unsupported function: catx")
        return _expr_catx(str(delimiter), *arg_exprs[1:])
    if function_name == _column_key("index"):
        _require_arity(node.func.id, arg_exprs, expected=2)
        excerpt = arg_literals[1]
        if excerpt is _UNSUPPORTED:
            raise ValueError("Unsupported function: index")
        return _expr_index(arg_exprs[0], str(excerpt))
    if function_name == _column_key("find"):
        if len(arg_exprs) < 2 or len(arg_exprs) > 4:
            raise ValueError("find() requires between 2 and 4 arguments")
        excerpt = arg_literals[1]
        start = 1 if len(arg_literals) < 3 else arg_literals[2]
        modifiers = "" if len(arg_literals) < 4 else arg_literals[3]
        if excerpt is _UNSUPPORTED or start is _UNSUPPORTED or modifiers is _UNSUPPORTED:
            raise ValueError("Unsupported function: find")
        return _expr_find(
            arg_exprs[0],
            str(excerpt),
            start=int(start),
            modifiers=str(modifiers),
        )
    if function_name == _column_key("tranwrd"):
        _require_arity(node.func.id, arg_exprs, expected=3)
        target = arg_literals[1]
        replacement = arg_literals[2]
        if target is _UNSUPPORTED or replacement is _UNSUPPORTED:
            raise ValueError("Unsupported function: tranwrd")
        return _expr_tranwrd(arg_exprs[0], str(target), str(replacement))
    if function_name == _column_key("translate"):
        _require_arity(node.func.id, arg_exprs, expected=3)
        to_chars = arg_literals[1]
        from_chars = arg_literals[2]
        if to_chars is _UNSUPPORTED or from_chars is _UNSUPPORTED:
            raise ValueError("Unsupported function: translate")
        return _expr_translate(arg_exprs[0], str(to_chars), str(from_chars))
    if function_name == _column_key("length"):
        _require_arity(node.func.id, arg_exprs, expected=1)
        return _expr_length(arg_exprs[0])
    if function_name == _column_key("lengthn"):
        _require_arity(node.func.id, arg_exprs, expected=1)
        return _expr_lengthn(arg_exprs[0])
    if function_name == _column_key("strip"):
        _require_arity(node.func.id, arg_exprs, expected=1)
        return _expr_strip(arg_exprs[0])
    if function_name == _column_key("reverse"):
        _require_arity(node.func.id, arg_exprs, expected=1)
        return _expr_reverse(arg_exprs[0])
    if function_name == _column_key("repeat"):
        _require_arity(node.func.id, arg_exprs, expected=2)
        repeat_count = arg_literals[1]
        if repeat_count is _UNSUPPORTED:
            raise ValueError("Unsupported function: repeat")
        return _expr_repeat(arg_exprs[0], int(repeat_count))
    if function_name == _column_key("countw"):
        if len(arg_exprs) < 1 or len(arg_exprs) > 2:
            raise ValueError("countw() requires one or two arguments")
        delimiters = " " if len(arg_literals) == 1 else arg_literals[1]
        if delimiters is _UNSUPPORTED:
            raise ValueError("Unsupported function: countw")
        return _expr_countw(arg_exprs[0], str(delimiters))
    if function_name == _column_key("round"):
        if len(arg_exprs) < 1 or len(arg_exprs) > 2:
            raise ValueError("round() requires one or two arguments")
        unit_expr = pl.lit(1.0) if len(arg_exprs) == 1 else arg_exprs[1]
        unit_literal = 1.0 if len(arg_literals) == 1 else arg_literals[1]
        return _expr_round(arg_exprs[0], unit_expr=unit_expr, unit_literal=unit_literal)
    if function_name == _column_key("put"):
        _require_arity(node.func.id, arg_exprs, expected=2)
        format_name = arg_literals[1]
        if format_name is _UNSUPPORTED:
            raise ValueError("Unsupported function: put")
        return _expr_put(arg_exprs[0], format_name, format_registry=format_registry)
    if function_name == _column_key("input"):
        _require_arity(node.func.id, arg_exprs, expected=2)
        informat_name = arg_literals[1]
        if informat_name is _UNSUPPORTED:
            raise ValueError("Unsupported function: input")
        return _expr_input(arg_exprs[0], informat_name, format_registry=format_registry)
    if function_name == _column_key("hour"):
        _require_arity(node.func.id, arg_exprs, expected=1)
        return _expr_hour(arg_exprs[0], format_registry=format_registry)

    raise ValueError(f"Unsupported function: {node.func.id}")


def _resolve_assignment_target_name(column_names: Sequence[str], target: str) -> str:
    try:
        return _resolve_assignment_source_name(column_names, target)
    except KeyError:
        return target


def _resolve_assignment_source_name(column_names: Sequence[str], name: str) -> str:
    if name in column_names:
        return name

    normalized = _column_key(name)
    matches = [candidate for candidate in column_names if _column_key(candidate) == normalized]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Ambiguous column reference for Session.assign: {name}")
    raise KeyError(f"Column not found for Session.assign: {name}")


def _supported_assign_function_names() -> tuple[str, ...]:
    return tuple(sorted(_ASSIGN_FUNCTION_REGISTRY))


def _require_arity(function_name: str, args: Sequence[Any], *, expected: int) -> None:
    if len(args) != expected:
        raise ValueError(f"{function_name}() requires {expected} argument(s)")


def _extract_literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        if isinstance(node.operand.value, (int, float)):
            return -node.operand.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd) and isinstance(node.operand, ast.Constant):
        if isinstance(node.operand.value, (int, float)):
            return +node.operand.value
    return _UNSUPPORTED


def _as_string_expr(expr: pl.Expr) -> pl.Expr:
    return pl.coalesce([expr.cast(pl.Utf8), pl.lit("")])


def _expr_upcase(expr: pl.Expr) -> pl.Expr:
    return _as_string_expr(expr).str.to_uppercase()


def _expr_lowcase(expr: pl.Expr) -> pl.Expr:
    return _as_string_expr(expr).str.to_lowercase()


def _expr_propcase(expr: pl.Expr) -> pl.Expr:
    return _as_string_expr(expr).str.to_titlecase()


def _expr_catx(delimiter: str, *args: pl.Expr) -> pl.Expr:
    cleaned = []
    for expr in args:
        stripped = _as_string_expr(expr).str.strip_chars()
        cleaned.append(pl.when(stripped == "").then(None).otherwise(stripped))
    return pl.concat_str(cleaned, separator=delimiter, ignore_nulls=True).fill_null("")


def _expr_cat(*args: pl.Expr) -> pl.Expr:
    return pl.concat_str([_as_string_expr(expr) for expr in args], separator="", ignore_nulls=False)


def _expr_cats(*args: pl.Expr) -> pl.Expr:
    return pl.concat_str([_as_string_expr(expr).str.strip_chars() for expr in args], separator="", ignore_nulls=False)


def _expr_catt(*args: pl.Expr) -> pl.Expr:
    return pl.concat_str([_as_string_expr(expr).str.strip_chars_end() for expr in args], separator="", ignore_nulls=False)


def _expr_index(source: pl.Expr, excerpt: str) -> pl.Expr:
    source_expr = _as_string_expr(source)
    if excerpt == "":
        return pl.lit(1)
    found = source_expr.str.find(excerpt)
    return pl.when(found.is_null()).then(0).otherwise(found + 1)


def _expr_find(source: pl.Expr, excerpt: str, *, start: int = 1, modifiers: str = "") -> pl.Expr:
    source_expr = _as_string_expr(source)
    excerpt_text = excerpt
    start_index = max(int(start) - 1, 0)
    modifier_text = modifiers.lower()
    if "i" in modifier_text:
        source_expr = source_expr.str.to_lowercase()
        excerpt_text = excerpt_text.lower()
    if start_index:
        source_expr = source_expr.str.slice(start_index)
    found = source_expr.str.find(excerpt_text)
    return pl.when(found.is_null()).then(0).otherwise(found + 1 + start_index)


def _expr_tranwrd(source: pl.Expr, target: str, replacement: str) -> pl.Expr:
    return _as_string_expr(source).str.replace_all(re.escape(target), replacement, literal=True)


def _expr_translate(source: pl.Expr, to_chars: str, from_chars: str) -> pl.Expr:
    expr = _as_string_expr(source)
    for index, src in enumerate(from_chars):
        replacement = to_chars[index] if index < len(to_chars) else ""
        expr = expr.str.replace_all(re.escape(src), replacement, literal=True)
    return expr


def _expr_length(expr: pl.Expr) -> pl.Expr:
    return _as_string_expr(expr).str.len_chars()


def _expr_lengthn(expr: pl.Expr) -> pl.Expr:
    text = _as_string_expr(expr)
    return pl.when(text == "").then(0).otherwise(text.str.len_chars())


def _expr_strip(expr: pl.Expr) -> pl.Expr:
    return _as_string_expr(expr).str.strip_chars()


def _expr_reverse(expr: pl.Expr) -> pl.Expr:
    return _as_string_expr(expr).str.reverse()


def _expr_repeat(expr: pl.Expr, count: int) -> pl.Expr:
    if count <= 0:
        return pl.lit("")
    repeated = _as_string_expr(expr)
    for _ in range(count - 1):
        repeated = repeated + _as_string_expr(expr)
    return repeated


def _expr_countw(expr: pl.Expr, delimiters: str = " ") -> pl.Expr:
    splitter_chars = delimiters or " "
    splitter_pattern = "[" + re.escape(splitter_chars) + "]+"
    stripped = _as_string_expr(expr).str.strip_chars()
    normalized = stripped.str.replace_all(splitter_pattern, " ")
    return pl.when(normalized == "").then(0).otherwise(normalized.str.split(" ").list.len())


def _expr_round(value: pl.Expr, *, unit_expr: pl.Expr, unit_literal: Any) -> pl.Expr:
    numeric_value = value.cast(pl.Float64)
    numeric_unit = unit_expr.cast(pl.Float64)
    quotient = numeric_value / numeric_unit
    rounded_multiple = pl.when(quotient >= 0).then((quotient.abs() + 0.5).floor()).otherwise(-((quotient.abs() + 0.5).floor()))
    rounded = rounded_multiple * numeric_unit
    if unit_literal is not _UNSUPPORTED and isinstance(unit_literal, (int, float)):
        decimal_places = _decimal_places_from_unit(float(unit_literal))
        if decimal_places is not None:
            rounded = rounded.round(decimal_places)
    return pl.when(numeric_unit == 0).then(numeric_value).otherwise(rounded)


def _expr_put(value: pl.Expr, format_name: Any, *, format_registry: FormatRegistry) -> pl.Expr:
    return value.map_elements(
        lambda item: format_registry.put(item, format_name),
        return_dtype=pl.Utf8,
    )


def _expr_input(value: pl.Expr, informat_name: Any, *, format_registry: FormatRegistry) -> pl.Expr:
    try:
        kind = format_registry.infer_input_kind(informat_name)
    except ValueError:
        return value.map_elements(lambda item: format_registry.input(item, informat_name))

    if kind == "date":
        return_dtype = pl.Date
    elif kind == "datetime":
        return_dtype = pl.Datetime
    elif kind in {"integer", "int", "int64"}:
        return_dtype = pl.Int64
    elif kind in {"float", "float64"}:
        return_dtype = pl.Float64
    elif kind in {"string", "str", "utf8"}:
        return_dtype = pl.Utf8
    else:
        return_dtype = pl.Time

    return value.map_elements(
        lambda item: format_registry.input(item, informat_name),
        return_dtype=return_dtype,
    )


def _expr_hour(value: pl.Expr, *, format_registry: FormatRegistry) -> pl.Expr:
    return value.map_elements(
        lambda item: format_registry.hour(item),
        return_dtype=pl.Float64,
    )


def _decimal_places_from_unit(unit: float) -> int | None:
    abs_unit = abs(unit)
    for places in range(13):
        scaled = abs_unit * (10**places)
        if math.isclose(scaled, round(scaled), rel_tol=0.0, abs_tol=1e-9):
            return places
    return None


def _evaluate_assignment_spec(
    spec: _AssignmentSpec,
    *,
    row: Mapping[str, Any],
    evaluator: ExpressionEvaluator,
    runtime: PDVRuntimeService,
    context: Any,
) -> Any:
    if spec.kind == "literal":
        return spec.value

    if spec.kind in {"expr", "func"}:
        return _evaluate_expression_value(
            spec.expression or "",
            row=row,
            evaluator=evaluator,
            runtime=runtime,
            context=context,
        )

    if spec.kind == "case_when":
        for clause in spec.clauses:
            matched = _evaluate_expression_value(
                clause.condition,
                row=row,
                evaluator=evaluator,
                runtime=runtime,
                context=context,
            )
            if bool(matched):
                return _evaluate_expression_value(
                    clause.value_expression,
                    row=row,
                    evaluator=evaluator,
                    runtime=runtime,
                    context=context,
                )

        if spec.else_expression is None:
            return None

        return _evaluate_expression_value(
            spec.else_expression,
            row=row,
            evaluator=evaluator,
            runtime=runtime,
            context=context,
        )

    raise ValueError(f"Unsupported assignment kind: {spec.kind}")


def _evaluate_expression_value(
    expression: str,
    *,
    row: Mapping[str, Any],
    evaluator: ExpressionEvaluator,
    runtime: PDVRuntimeService,
    context: Any,
) -> Any:
    unsupported_function = _find_unsupported_function(
        expression,
        supported_functions=runtime.get_eval_scope().keys(),
    )
    if unsupported_function is not None:
        raise ValueError(f"Unsupported function: {unsupported_function}")

    value, diagnostic = evaluator.evaluate_scalar(expression, row=row, context=context, array_defs={})
    if diagnostic is not None:
        raise ValueError(diagnostic.message)
    return value


def _normalize_assignments(assignments: Mapping[str, Any]) -> tuple[_AssignmentSpec, ...]:
    normalized: list[_AssignmentSpec] = []
    for target, value in assignments.items():
        if not target or not target.strip():
            raise ValueError("Assignment target must be a non-empty column name")

        if isinstance(value, str):
            try:
                case_when = _CASE_WHEN_PARSER.parse(value)
            except _CaseWhenParseError as error:
                raise ValueError(
                    _render_column_api_diagnostic(error.diagnostic, source_id=f"assign:{target}")
                ) from error
            if case_when is not None:
                normalized.append(
                    _AssignmentSpec(
                        target=target,
                        kind="case_when",
                        clauses=case_when[0],
                        else_expression=case_when[1],
                    )
                )
                continue

            kind = "func" if _contains_function_call(value) else "expr"
            normalized.append(_AssignmentSpec(target=target, kind=kind, expression=value))
            continue

        normalized.append(_AssignmentSpec(target=target, kind="literal", value=value))

    return tuple(normalized)


def _contains_function_call(expression: str) -> bool:
    parsed = _parse_assignment_expression_ast(expression)
    if parsed is None:
        return False
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) for node in ast.walk(parsed))


def _find_unsupported_function(
    expression: str,
    *,
    supported_functions: Iterable[str],
) -> str | None:
    supported = {_column_key(name) for name in supported_functions}
    for function_name in _iter_assignment_function_names(expression):
        normalized = _column_key(function_name)
        if normalized in {_column_key(keyword) for keyword in {"and", "or", "not", "in"}}:
            continue
        if normalized not in supported:
            return function_name
    return None


def _parse_assignment_expression_ast(expression: str) -> ast.AST | None:
    normalized = _ASSIGNMENT_NORMALIZER.prepare_expression(expression)
    try:
        return ast.parse(normalized, mode="eval")
    except SyntaxError:
        return None


def _iter_assignment_function_names(expression: str) -> tuple[str, ...]:
    parsed = _parse_assignment_expression_ast(expression)
    if parsed is None:
        return ()
    return tuple(
        node.func.id
        for node in ast.walk(parsed)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    by_columns: Sequence[str],
) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    grouped: list[tuple[tuple[Any, ...], list[dict[str, Any]]]] = []
    grouped_index: dict[tuple[Any, ...], int] = {}

    for row in rows:
        group_key = tuple(row.get(column) for column in by_columns)
        group_position = grouped_index.get(group_key)
        if group_position is None:
            grouped_index[group_key] = len(grouped)
            grouped.append((group_key, [dict(row)]))
            continue
        grouped[group_position][1].append(dict(row))

    if grouped:
        return grouped
    return [(tuple(), [])]