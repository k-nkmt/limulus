from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from ..naming import _column_key
from ..syntax.models import (
    BinarySyntax,
    BlockSyntax,
    CallSyntax,
    ConcatSyntax,
    DeferredExpressionError,
    ExpressionSyntax,
    LiteralSyntax,
    NameSyntax,
    SourceStatementSyntax,
    SubscriptSyntax,
    UnarySyntax,
    WhereStatementSyntax,
)
from .statement_expressions import StatementExpressionSlot, block_expression_slots


def _freeze_rewrite_value(value: Any) -> Any:
    """Freeze nested policy values without retaining mutable containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_rewrite_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_rewrite_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_rewrite_value(item) for item in value)
    return value


def _thaw_rewrite_value(value: Any) -> Any:
    """Return serializer-friendly values from an immutable rewrite decision."""

    if isinstance(value, Mapping):
        return {key: _thaw_rewrite_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_thaw_rewrite_value(item) for item in value)
    if isinstance(value, frozenset):
        return tuple(_thaw_rewrite_value(item) for item in value)
    return value


@dataclass(frozen=True)
class RewritePlan:
    merge_mode: str
    lag_lead_mode: str
    where_mode: str
    helper_source_columns: tuple[str, ...]
    helper_runtime_columns: tuple[str, ...]
    lag_lead_specs: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "helper_source_columns", tuple(self.helper_source_columns))
        object.__setattr__(self, "helper_runtime_columns", tuple(self.helper_runtime_columns))
        object.__setattr__(
            self,
            "lag_lead_specs",
            tuple(_freeze_rewrite_value(spec) for spec in self.lag_lead_specs),
        )


@dataclass(frozen=True)
class HelperOwnershipDecision:
    helper_source_columns: tuple[str, ...]
    helper_runtime_columns: tuple[str, ...]


@dataclass(frozen=True)
class TypedLagLeadSpec:
    """One helper candidate extracted directly from a typed call node.

    The default is retained as syntax so a later materialization owner can
    decide what it supports without reparsing a display string or inferring
    its type.  ``is_source_owned`` follows the existing helper ownership
    policy: only absent or literal defaults can be prepared from input rows.
    """

    call: CallSyntax
    function_name: str
    argument_is_string_literal: bool
    source_name: str
    offset: int
    default_expression: ExpressionSyntax | None
    helper_column: str
    is_source_owned: bool


@dataclass(frozen=True)
class TypedHelperPlan:
    """Internal helper decisions derived from one ``BlockSyntax``.

    This parallel typed entry is owned by ``RewritePlanner``.  It does not
    bind input schemas or host callables and does not produce rewritten
    display text.
    """

    merge_mode: str
    lag_lead_mode: str
    where_mode: str
    helper_source_columns: tuple[str, ...]
    helper_runtime_columns: tuple[str, ...]
    lag_lead_specs: tuple[TypedLagLeadSpec, ...]
    pre_evaluation_targets: tuple[StatementExpressionSlot, ...]
    pre_row_filter_expression: ExpressionSyntax | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "helper_source_columns", tuple(self.helper_source_columns))
        object.__setattr__(self, "helper_runtime_columns", tuple(self.helper_runtime_columns))
        object.__setattr__(self, "lag_lead_specs", tuple(self.lag_lead_specs))
        object.__setattr__(self, "pre_evaluation_targets", tuple(self.pre_evaluation_targets))
        if self.pre_row_filter_expression is not None and not isinstance(
            self.pre_row_filter_expression,
            (BinarySyntax, CallSyntax, ConcatSyntax, DeferredExpressionError,
             LiteralSyntax, NameSyntax, SubscriptSyntax, UnarySyntax),
        ):
            raise TypeError("pre_row_filter_expression must be an ExpressionSyntax")


class HelperOwnershipPolicy:
    def classify(
        self,
        *,
        source_helper_columns: Sequence[str],
        lag_lead_runtime_columns: Sequence[str],
    ) -> HelperOwnershipDecision:
        return HelperOwnershipDecision(
            helper_source_columns=tuple(source_helper_columns),
            helper_runtime_columns=tuple(lag_lead_runtime_columns),
        )


_LAG_LEAD_PATTERN = re.compile(r"\b(lag|lead|shift)\s*\(", re.IGNORECASE)
_LAG_LEAD_CALL_PATTERN = re.compile(
    r"\b(?P<fn>lag|lead|shift)\s*\(\s*"
    r"(?P<arg>'[^']+'|\"[^\"]+\"|[A-Za-z_][\w\.]*)"
    r"\s*(?:,\s*(?P<offset>-?\d+))?"
    r"\s*(?:,\s*(?P<default>[^()]+?))?\s*\)",
    re.IGNORECASE,
)


class RewritePlanner:
    def __init__(self) -> None:
        self._helper_ownership_policy = HelperOwnershipPolicy()

    def build_typed_plan(self, block: BlockSyntax) -> TypedHelperPlan:
        """Discover helper ownership from typed syntax without text parsing."""

        if not isinstance(block, BlockSyntax):
            raise TypeError("block must be BlockSyntax")

        source_statement = next(
            (
                statement
                for statement in block.statements
                if statement.kind in {"SET", "MERGE"}
                and isinstance(statement.payload, SourceStatementSyntax)
            ),
            None,
        )
        helper_source_columns: list[str] = []
        if source_statement is not None:
            source_payload = source_statement.payload
            assert isinstance(source_payload, SourceStatementSyntax)
            for reference in source_payload.refs:
                if reference.options.in_var:
                    helper_source_columns.append(reference.options.in_var)
            if source_payload.options.indsname_var:
                helper_source_columns.append(source_payload.options.indsname_var)
            if source_payload.options.end_var:
                helper_source_columns.append(source_payload.options.end_var)

        by_statement = next(
            (statement for statement in block.statements if statement.kind == "BY"),
            None,
        )
        if by_statement is not None:
            names = getattr(by_statement.payload, "names", ())
            for name in names:
                helper_source_columns.extend((f"first.{name.name}", f"last.{name.name}"))

        expression_targets = block_expression_slots(block)
        lag_lead_specs = self._collect_typed_lag_lead_specs(expression_targets)
        source_owned = tuple(spec for spec in lag_lead_specs if spec.is_source_owned)
        runtime_owned = tuple(spec for spec in lag_lead_specs if not spec.is_source_owned)
        helper_source_columns.extend(spec.helper_column for spec in source_owned)
        ownership = self._helper_ownership_policy.classify(
            source_helper_columns=helper_source_columns,
            lag_lead_runtime_columns=tuple(spec.helper_column for spec in runtime_owned),
        )

        where_statement = next(
            (
                statement
                for statement in block.statements
                if statement.kind == "WHERE"
                and isinstance(statement.payload, WhereStatementSyntax)
            ),
            None,
        )
        pre_row_filter_expression = (
            where_statement.payload.condition
            if where_statement is not None
            and isinstance(where_statement.payload, WhereStatementSyntax)
            else None
        )
        return TypedHelperPlan(
            merge_mode=(
                "prepared_merge_rows"
                if source_statement is not None and source_statement.kind == "MERGE"
                else "none"
            ),
            lag_lead_mode=(
                "planner_precomputed_columns"
                if source_owned
                else "runtime_owned_helpers" if runtime_owned else "none"
            ),
            where_mode=(
                "shared_pre_row_filter"
                if any(statement.kind == "WHERE" for statement in block.statements)
                else "none"
            ),
            helper_source_columns=ownership.helper_source_columns,
            helper_runtime_columns=ownership.helper_runtime_columns,
            lag_lead_specs=lag_lead_specs,
            pre_evaluation_targets=expression_targets,
            pre_row_filter_expression=pre_row_filter_expression,
        )

    def _collect_typed_lag_lead_specs(
        self,
        expression_targets: Sequence[StatementExpressionSlot],
    ) -> tuple[TypedLagLeadSpec, ...]:
        candidates: list[CallSyntax] = []
        for target in expression_targets:
            candidates.extend(_typed_calls(target.expression))
        candidates.sort(key=lambda call: (call.span.start, -call.span.end))

        specs: list[TypedLagLeadSpec] = []
        occurrence = 0
        for call in candidates:
            function_name = call.name.lower()
            if function_name not in {"lag", "lead", "shift"}:
                continue
            parsed = _typed_lag_lead_arguments(call)
            if parsed is None:
                continue
            occurrence += 1
            source_name, argument_is_string_literal, offset, default_expression = parsed
            if function_name in {"lag", "lead"} and offset < 1:
                continue
            helper_column = self._build_helper_column_name(
                function_name,
                source_name,
                offset,
                occurrence,
            )
            source_owned = _is_typed_source_owned_lag_lead(
                function_name=function_name,
                argument_is_string_literal=argument_is_string_literal,
                default_expression=default_expression,
            )
            specs.append(
                TypedLagLeadSpec(
                    call=call,
                    function_name=function_name,
                    argument_is_string_literal=argument_is_string_literal,
                    source_name=source_name,
                    offset=offset,
                    default_expression=default_expression,
                    helper_column=helper_column,
                    is_source_owned=source_owned,
                )
            )
        return tuple(specs)


    @staticmethod
    def _build_helper_column_name(function_name: str, source_name: str, offset: int, occurrence: int) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", source_name).strip("_") or "value"
        offset_token = RewritePlanner._offset_token(offset)
        return f"__rewrite_{function_name}_{sanitized.lower()}_{offset_token}_{occurrence}__"

    @staticmethod
    def _offset_token(offset: int) -> str:
        if offset < 0:
            return f"neg{abs(offset)}"
        return str(offset)

def _typed_calls(expression: ExpressionSyntax) -> tuple[CallSyntax, ...]:
    """Walk a typed expression in source-shaped preorder, without text scans."""

    calls: list[CallSyntax] = []
    pending = [expression]
    while pending:
        current = pending.pop()
        children: tuple[ExpressionSyntax, ...] = ()
        if isinstance(current, CallSyntax):
            calls.append(current)
            children = current.arguments
        elif isinstance(current, BinarySyntax):
            children = (current.left, current.right)
        elif isinstance(current, UnarySyntax):
            children = (current.operand,)
        elif isinstance(current, ConcatSyntax):
            children = current.parts
        elif isinstance(current, SubscriptSyntax):
            children = (current.base, current.index)
        # Whole-expression deferred descriptors deliberately expose no
        # partial children; the saved statement/apply owner retains them.
        pending.extend(reversed(children))
    return tuple(calls)


def _typed_lag_lead_arguments(
    call: CallSyntax,
) -> tuple[str, bool, int, ExpressionSyntax | None] | None:
    """Project the existing helper-call shapes from their typed arguments."""

    if not 1 <= len(call.arguments) <= 3:
        return None
    source_argument = call.arguments[0]
    if isinstance(source_argument, NameSyntax):
        source_name = source_argument.name
        argument_is_string_literal = False
    elif isinstance(source_argument, LiteralSyntax) and source_argument.kind == "string":
        stripped = source_argument.lexeme.strip()
        if (
            len(stripped) < 2
            or stripped[0] not in {"'", '"'}
            or stripped[-1] != stripped[0]
        ):
            return None
        # Match the legacy helper's string-name projection.  Escaped-name
        # decoding remains with the same runtime/input owner as before.
        source_name = stripped[1:-1]
        argument_is_string_literal = True
    else:
        return None

    offset = 1
    default_expression: ExpressionSyntax | None = None
    if len(call.arguments) >= 2:
        offset_value = _typed_integer_value(call.arguments[1])
        if offset_value is not None:
            offset = offset_value
            if len(call.arguments) == 3:
                default_expression = call.arguments[2]
        elif len(call.arguments) == 2:
            # The legacy helper accepts a second non-offset argument as a
            # default.  Nested calls were outside that regex contract and
            # remain ordinary row evaluation on the typed path as well.
            default_expression = call.arguments[1]
        else:
            return None

    if default_expression is not None and _typed_calls(default_expression):
        return None
    return source_name, argument_is_string_literal, offset, default_expression


def _typed_integer_value(expression: ExpressionSyntax) -> int | None:
    sign = 1
    value = expression
    if isinstance(value, UnarySyntax):
        if value.operator != "-":
            return None
        sign = -1
        value = value.operand
    if not isinstance(value, LiteralSyntax) or value.kind != "numeric":
        return None
    if re.fullmatch(r"\d+", value.lexeme.strip()) is None:
        return None
    return sign * int(value.lexeme.strip())


def _is_typed_source_owned_lag_lead(
    *,
    function_name: str,
    argument_is_string_literal: bool,
    default_expression: ExpressionSyntax | None,
) -> bool:
    if function_name == "shift" and not argument_is_string_literal:
        return False
    if default_expression is None:
        return True
    value = default_expression
    if isinstance(value, UnarySyntax):
        if value.operator != "-":
            return False
        value = value.operand
        return isinstance(value, LiteralSyntax) and value.kind == "numeric"
    return isinstance(value, LiteralSyntax) and value.kind in {
        "missing",
        "numeric",
        "string",
    }


def is_source_owned_lag_lead_spec(spec: Mapping[str, Any]) -> bool:
    if "is_source_owned" in spec:
        return bool(spec["is_source_owned"])
    if spec.get("function_name") == "shift" and not spec.get("argument_is_string_literal"):
        return False
    default_expr = spec.get("default_expr")
    if default_expr is None:
        return True
    return default_expr.strip() in {".", "None", "none", "null", "NULL"}


def target_index_for_spec(*, index: int, spec: Mapping[str, Any]) -> int:
    function_name = spec["function_name"]
    offset = int(spec["offset"])
    if function_name == "lag":
        return index - offset
    if function_name == "lead":
        return index + offset
    return index + offset


def resolve_row_value(row: Mapping[str, Any], source_name: str) -> Any:
    if source_name in row:
        return row[source_name]
    normalized = _column_key(source_name)
    for candidate, value in row.items():
        if _column_key(candidate) == normalized:
            return value
    return None


def sortable_value(value: Any) -> tuple[int, int, Any]:
    if value is None:
        return (1, 0, 0)
    if isinstance(value, bool):
        return (0, 0, int(value))
    if isinstance(value, (int, float)):
        return (0, 1, float(value))
    if isinstance(value, dt.datetime):
        return (0, 2, value.timestamp())
    if isinstance(value, dt.date):
        return (0, 3, value.toordinal())
    if isinstance(value, dt.time):
        return (0, 4, value.isoformat())
    return (0, 5, str(value))


def detect_internal_variable_collision(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_name: str,
    internal_variable_names: set[str],
    allow_internal_names: bool,
) -> Any | None:
    if allow_internal_names or not internal_variable_names:
        return None
    for row in rows:
        collided_names = sorted(internal_variable_names.intersection(row.keys()))
        if not collided_names:
            continue
        return build_internal_var_collision(collided_names[0], source_name)
    return None


def annotate_by_group_flags(
    *,
    rows: list[dict[str, Any]],
    by_keys: Sequence[str],
    resolve_row_key: Callable[[Mapping[str, Any], str], str | None],
    resolve_row_value: Callable[[Mapping[str, Any], str], Any],
) -> tuple[list[dict[str, Any]], Any | None]:
    if not rows or not by_keys:
        return rows, None

    for by_key in by_keys:
        if any(resolve_row_key(row, by_key) is None for row in rows):
            return [], build_by_rows_precondition_failed(by_key)

    for by_key in by_keys:
        for index, row in enumerate(rows):
            previous_value = resolve_row_value(rows[index - 1], by_key) if index > 0 else object()
            next_value = resolve_row_value(rows[index + 1], by_key) if index < len(rows) - 1 else object()
            current_value = resolve_row_value(row, by_key)
            row[f"FIRST.{by_key}"] = 1 if current_value != previous_value else 0
            row[f"LAST.{by_key}"] = 1 if current_value != next_value else 0
            row[f"first.{by_key}"] = row[f"FIRST.{by_key}"]
            row[f"last.{by_key}"] = row[f"LAST.{by_key}"]
    return rows, None


def build_runtime_set_not_found(input_name: str) -> Any:
    from ..models import Diagnostic

    return Diagnostic(
        code="RUNTIME_SET_DATASET_NOT_FOUND",
        severity="error",
        message=f"Input dataset is not provided: {input_name}",
    )


def build_by_precondition_failed(by_key: str, input_name: str) -> Any:
    from ..models import Diagnostic

    return Diagnostic(
        code="RUNTIME_BY_PRECONDITION_FAILED",
        severity="error",
        message=f"BY key '{by_key}' is missing in source '{input_name}'.",
    )


def build_by_rows_precondition_failed(by_key: str) -> Any:
    from ..models import Diagnostic

    return Diagnostic(
        code="RUNTIME_BY_PRECONDITION_FAILED",
        severity="error",
        message=f"BY key '{by_key}' is missing in source rows.",
    )


def build_merge_duplicate_column(duplicate_label: str) -> Any:
    from ..models import Diagnostic

    return Diagnostic(
        code="RUNTIME_MERGE_DUPLICATE_COLUMN",
        severity="error",
        message=f"MERGE inputs contain duplicate non-BY columns: {duplicate_label}",
    )


def build_internal_var_collision(variable_name: str, source_name: str) -> Any:
    from ..models import Diagnostic

    return Diagnostic(
        code="RUNTIME_INTERNAL_VAR_NAME_COLLISION",
        severity="error",
        message=(
            "Input dataset contains a reserved internal reference variable name: "
            f"{variable_name} (source={source_name})"
        ),
    )


def build_row_sort_key(
    row: Mapping[str, Any],
    *,
    by_keys: Sequence[str],
    resolve_row_value: Callable[[Mapping[str, Any], str], Any],
) -> tuple[Any, ...]:
    return tuple(sortable_value(resolve_row_value(row, key)) for key in by_keys)


def lag_lead_default_value(spec: Mapping[str, Any]) -> Any:
    return spec["typed_default_value"]


def materialize_lag_lead_columns(
    *,
    rows: Sequence[dict[str, Any]],
    specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    specs = tuple(spec for spec in specs if is_source_owned_lag_lead_spec(spec))
    if not specs:
        return [dict(row) for row in rows]

    materialized_rows = [dict(row) for row in rows]
    for index, row in enumerate(materialized_rows):
        for spec in specs:
            target_index = target_index_for_spec(index=index, spec=spec)
            if target_index < 0 or target_index >= len(materialized_rows):
                row[spec["helper_column"]] = lag_lead_default_value(spec)
                continue
            row[spec["helper_column"]] = resolve_row_value(
                materialized_rows[target_index], spec["source_name"]
            )
    return materialized_rows


def build_row_sort_key(
    row: Mapping[str, Any],
    *,
    by_keys: Sequence[str],
    resolve_row_value: Callable[[Mapping[str, Any], str], Any],
) -> tuple[Any, ...]:
    return tuple(sortable_value(resolve_row_value(row, key)) for key in by_keys)


def prepare_merge_rows(
    *,
    source_refs: Sequence[Any],
    by_keys: Sequence[str],
    resolved_inputs: Mapping[str, Any],
    internal_variable_names: set[str],
    load_input_rows: Callable[[Any], tuple[list[dict[str, Any]], Any | None]],
    resolve_row_key: Callable[[Mapping[str, Any], str], str | None],
    resolve_row_value: Callable[[Mapping[str, Any], str], Any],
) -> tuple[list[tuple[str, dict[str, Any], str | None]], Any | None]:
    loaded_sources: list[tuple[str, str | None, list[dict[str, Any]]]] = []
    non_key_columns_by_source: list[set[str]] = []
    normalized_by_keys = {_column_key(key) for key in by_keys}
    canonical_by_names: dict[str, str] = {}

    for source_ref in source_refs:
        input_name = source_ref.name
        input_ref = resolved_inputs.get(input_name)
        if input_ref is None:
            return [], build_runtime_set_not_found(input_name)

        loaded_rows, load_error = load_input_rows(input_ref)
        if load_error is not None:
            return [], load_error

        allow_internal_names = (
            getattr(getattr(input_ref, "preparation", None), "kind", "none")
            == "prepared_merge"
        )
        collision = detect_internal_variable_collision(
            rows=loaded_rows,
            source_name=input_name,
            internal_variable_names=internal_variable_names,
            allow_internal_names=allow_internal_names,
        )
        if collision is not None:
            return [], collision

        # Dataset-reference options are applied by the shared Arrow input
        # preparation stage before this row transformation.
        option_rows = loaded_rows

        for row in option_rows:
            for by_key in by_keys:
                resolved_by_key = resolve_row_key(row, by_key)
                if resolved_by_key is None:
                    return [], build_by_precondition_failed(by_key, input_name)
                canonical_by_names.setdefault(_column_key(by_key), resolved_by_key)

        non_key_columns = {
            _column_key(column_name)
            for row in option_rows
            for column_name in row.keys()
            if _column_key(column_name) not in normalized_by_keys
        }
        non_key_columns_by_source.append(non_key_columns)
        loaded_sources.append((input_name, source_ref.options.in_var, option_rows))

    duplicate_columns: set[str] = set()
    for left_index in range(len(non_key_columns_by_source)):
        for right_index in range(left_index + 1, len(non_key_columns_by_source)):
            duplicate_columns.update(
                non_key_columns_by_source[left_index].intersection(
                    non_key_columns_by_source[right_index]
                )
            )

    if duplicate_columns:
        duplicate_label = ", ".join(sorted(duplicate_columns))
        return [], build_merge_duplicate_column(duplicate_label)

    if not by_keys:
        all_source_rows = [rows for _, _, rows in loaded_sources]
        max_len = max((len(rows) for rows in all_source_rows), default=0)
        merged_rows_no_by: list[dict[str, Any]] = []
        merged_sources_no_by: list[str] = []
        merged_in_vars_no_by: list[str | None] = []
        for row_index in range(max_len):
            merged_row: dict[str, Any] = {}
            contributing: list[str] = []
            for source_name, _in_var, rows in loaded_sources:
                if row_index < len(rows):
                    merged_row.update(rows[row_index])
                    contributing.append(source_name)
            merged_rows_no_by.append(merged_row)
            merged_sources_no_by.append(",".join(contributing) if contributing else "")
            merged_in_vars_no_by.append(None)
            for source_name, in_var, _ in loaded_sources:
                if in_var:
                    merged_row[in_var] = 1 if source_name in contributing else 0
        return list(zip(merged_sources_no_by, merged_rows_no_by, merged_in_vars_no_by)), None

    grouped_sources: list[dict[tuple[Any, ...], list[dict[str, Any]]]] = []
    merge_key_order: list[tuple[Any, ...]] = []
    seen_merge_keys: set[tuple[Any, ...]] = set()

    for _, _, rows in loaded_sources:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = tuple(resolve_row_value(row, key_name) for key_name in by_keys)
            grouped.setdefault(key, []).append(row)
            if key not in seen_merge_keys:
                seen_merge_keys.add(key)
                merge_key_order.append(key)
        grouped_sources.append(grouped)

    merge_key_order.sort(key=lambda item: tuple(sortable_value(value) for value in item))
    output_by_keys = tuple(canonical_by_names.get(_column_key(key), key) for key in by_keys)

    merged_rows: list[dict[str, Any]] = []
    merged_sources: list[str] = []
    merged_in_vars: list[str | None] = []

    for key in merge_key_order:
        group_lengths = [len(grouped.get(key, [])) for grouped in grouped_sources]
        max_group_length = max(group_lengths) if group_lengths else 0

        for row_index in range(max_group_length):
            merged_row: dict[str, Any] = {}
            contributing_sources: list[str] = []

            for source_index, (source_name, _in_var, _) in enumerate(loaded_sources):
                grouped_rows = grouped_sources[source_index].get(key, [])
                if row_index < len(grouped_rows):
                    merged_row.update(grouped_rows[row_index])
                    contributing_sources.append(source_name)
                elif grouped_rows:
                    merged_row.update(grouped_rows[-1])
                    contributing_sources.append(source_name)

            for by_index, by_key in enumerate(output_by_keys):
                normalized_by_key = _column_key(by_keys[by_index])
                duplicate_by_names = [
                    name
                    for name in tuple(merged_row)
                    if _column_key(name) == normalized_by_key and name != by_key
                ]
                for duplicate_name in duplicate_by_names:
                    merged_row.pop(duplicate_name, None)
                merged_row[by_key] = key[by_index]

            merged_rows.append(merged_row)
            merged_sources.append(",".join(contributing_sources) if contributing_sources else "")
            merged_in_vars.append(None)

            for source_name, in_var, _ in loaded_sources:
                if in_var:
                    merged_row[in_var] = 1 if source_name in contributing_sources else 0

    merged_rows, by_error = annotate_by_group_flags(
        rows=merged_rows,
        by_keys=by_keys,
        resolve_row_key=resolve_row_key,
        resolve_row_value=resolve_row_value,
    )
    if by_error is not None:
        return [], by_error

    return list(zip(merged_sources, merged_rows, merged_in_vars)), None


def prepare_set_rows(
    *,
    source_refs: Sequence[Any],
    by_keys: Sequence[str],
    in_option_vars: Sequence[str],
    indsname_var: str | None,
    end_var: str | None,
    resolved_inputs: Mapping[str, Any],
    internal_variable_names: set[str],
    load_input_rows: Callable[[Any], tuple[list[dict[str, Any]], Any | None]],
    resolve_row_key: Callable[[Mapping[str, Any], str], str | None],
    resolve_row_value: Callable[[Mapping[str, Any], str], Any],
) -> tuple[list[dict[str, Any]], Any | None]:
    rows_with_source: list[tuple[str, dict[str, Any], str | None]] = []

    for source_ref in source_refs:
        input_name = source_ref.name
        input_ref = resolved_inputs.get(input_name)
        if input_ref is None:
            return [], build_runtime_set_not_found(input_name)

        loaded_rows, load_error = load_input_rows(input_ref)
        if load_error is not None:
            return [], load_error

        # Source options have already been materialized by shared Arrow
        # preparation; this row path only owns SET ordering and flags.
        option_rows = loaded_rows
        allow_internal_names = (
            getattr(getattr(input_ref, "preparation", None), "kind", "none")
            == "prepared_merge"
        )
        collision = detect_internal_variable_collision(
            rows=option_rows,
            source_name=input_name,
            internal_variable_names=internal_variable_names,
            allow_internal_names=allow_internal_names,
        )
        if collision is not None:
            return [], collision

        for row in option_rows:
            rows_with_source.append((input_name, row, source_ref.options.in_var))

    if by_keys and len(rows_with_source) > 1:
        rows_with_source.sort(
            key=lambda item: build_row_sort_key(
                item[1],
                by_keys=by_keys,
                resolve_row_value=resolve_row_value,
            )
        )

    prepared_rows: list[dict[str, Any]] = []
    for index, (source_name, row, row_in_var) in enumerate(rows_with_source):
        enriched = dict(row)

        for in_var in in_option_vars:
            if row_in_var is None and in_var in enriched:
                continue
            enriched[in_var] = 1 if row_in_var == in_var else 0

        if indsname_var:
            enriched[indsname_var] = source_name

        if end_var:
            enriched[end_var] = 1 if index == len(rows_with_source) - 1 else 0

        prepared_rows.append(enriched)

    prepared_rows, by_error = annotate_by_group_flags(
        rows=prepared_rows,
        by_keys=by_keys,
        resolve_row_key=resolve_row_key,
        resolve_row_value=resolve_row_value,
    )
    if by_error is not None:
        return [], by_error

    return prepared_rows, None


__all__ = [
    "HelperOwnershipDecision",
    "HelperOwnershipPolicy",
    "RewritePlan",
    "RewritePlanner",
    "TypedHelperPlan",
    "TypedLagLeadSpec",
    "annotate_by_group_flags",
    "build_by_precondition_failed",
    "build_by_rows_precondition_failed",
    "build_internal_var_collision",
    "build_merge_duplicate_column",
    "build_row_sort_key",
    "build_runtime_set_not_found",
    "detect_internal_variable_collision",
    "is_source_owned_lag_lead_spec",
    "lag_lead_default_value",
    "materialize_lag_lead_columns",
    "prepare_merge_rows",
    "prepare_set_rows",
    "resolve_row_value",
    "sortable_value",
    "target_index_for_spec",
]
