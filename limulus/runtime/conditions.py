from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Callable

from ..naming import _column_key
from ..models import Diagnostic


class _DeferredConditionScope(dict[str, Any]):
    def __init__(
        self,
        *,
        row: Mapping[str, Any],
        context: Any,
        include_row_aliases: bool,
        eval_scope_provider: Callable[[], dict[str, Any]],
        resolve_row_value: Callable[[Mapping[str, Any], str], Any],
    ) -> None:
        super().__init__()
        self._row = row
        self._context = context
        self._include_row_aliases = include_row_aliases
        self._resolve_row_value = resolve_row_value

        self["_N_"] = context.get_variable("_N_")
        self["_ERROR_"] = context.get_variable("_ERROR_")
        self["_n_"] = context.get_variable("_N_")
        self["_error_"] = context.get_variable("_ERROR_")
        self.update(eval_scope_provider())
        self["__var__"] = self._var

    def _var(self, key: str) -> Any:
        context_value = self._context.get_variable(key)
        if context_value is not None:
            return context_value
        source_value = self._context.source_slot_value(key)
        if source_value is not None:
            return source_value
        return self._resolve_row_value(self._row, key)

    def __missing__(self, key: str) -> Any:
        if key in self._row:
            value = self._row[key]
            self[key] = value
            return value

        if self._include_row_aliases:
            normalized = _column_key(key)
            for candidate, value in self._row.items():
                if _column_key(candidate) == normalized:
                    self[key] = value
                    return value

            slot_ref = self._context.slot_index(key, sections=("source_slots",))
            if slot_ref is not None:
                source_value = self._context.source_slot_value(key)
                self[key] = source_value
                return source_value

        raise KeyError(key)


class ConditionEvaluator:
    def __init__(
        self,
        *,
        eval_scope_provider: Callable[[], dict[str, Any]],
        unsupported_function_finder: Callable[[str], str | None],
    ) -> None:
        self._eval_scope_provider = eval_scope_provider
        self._unsupported_function_finder = unsupported_function_finder
        self._compiled_condition_cache: dict[str, Any] = {}
        self._transformed_condition_cache: dict[tuple[str, tuple[str, ...]], str] = {}

    def prepare_expression(self, expression: str) -> None:
        transformed_expression = expression
        cache_key = (expression, tuple())
        self._transformed_condition_cache.setdefault(cache_key, transformed_expression)
        if transformed_expression in self._compiled_condition_cache:
            return
        try:
            self._compiled_condition_cache[transformed_expression] = compile(
                transformed_expression,
                "<limulus-condition>",
                "eval",
            )
        except Exception:
            return

    def evaluate(
        self,
        *,
        row: Mapping[str, Any],
        context: Any,
        expression: str,
        set_error: Callable[[Any, str], None],
        row_location_provider: Callable[[Any], str],
    ) -> bool:
        from .row_runtime import FunctionArgumentError, RuntimeExecutionError

        dotted_names = set(key for key in row.keys() if "." in key)
        for key in context.visible_variable_names():
            if "." in key:
                dotted_names.add(key)
        dotted_keys = tuple(sorted(dotted_names, key=len, reverse=True))
        transform_cache_key = (expression, dotted_keys)
        transformed_expression = self._transformed_condition_cache.get(transform_cache_key)
        if transformed_expression is None:
            transformed_expression = expression
            for key in dotted_keys:
                transformed_expression = re.sub(
                    rf"(?<![\w\']){re.escape(key)}(?![\w\'])",
                    f"__var__('{key}')",
                    transformed_expression,
                )
            self._transformed_condition_cache[transform_cache_key] = transformed_expression

        unsupported_function = self._unsupported_function_finder(expression)
        if unsupported_function is not None:
            set_error(context, unsupported_function)
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_UNSUPPORTED_FUNCTION",
                    severity="error",
                    location=row_location_provider(context),
                    message=f"Unsupported function: {unsupported_function}",
                )
            )

        try:
            compiled = self._compiled_condition_cache.get(transformed_expression)
            if compiled is None:
                compiled = compile(transformed_expression, "<limulus-condition>", "eval")
                self._compiled_condition_cache[transformed_expression] = compiled

            scope = self._build_scope(row=row, context=context, include_row_aliases=False)
            try:
                return bool(eval(compiled, {"__builtins__": {}}, scope))
            except NameError:
                fallback_scope = self._build_scope(row=row, context=context, include_row_aliases=True)
                return bool(eval(compiled, {"__builtins__": {}}, fallback_scope))
        except SyntaxError as error:
            set_error(context, str(error))
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_OPERATOR_NOT_SUPPORTED",
                    severity="error",
                    location=row_location_provider(context),
                    message=f"Unsupported operator in expression '{expression}': {error}",
                )
            ) from error
        except FunctionArgumentError as error:
            set_error(context, str(error))
            diagnostic = error.diagnostic
            if diagnostic is None:
                diagnostic = Diagnostic(
                    code="RUNTIME_FUNCTION_ARGUMENT_INVALID",
                    severity="error",
                    location=row_location_provider(context),
                    message=f"Invalid function arguments in expression '{expression}': {error}",
                )
            elif not diagnostic.location:
                diagnostic = Diagnostic(
                    code=diagnostic.code,
                    severity=diagnostic.severity,
                    message=diagnostic.message,
                    location=row_location_provider(context),
                    stage=diagnostic.stage,
                    span=diagnostic.span,
                    labels=diagnostic.labels,
                    notes=diagnostic.notes,
                    source_text=diagnostic.source_text,
                )
            raise RuntimeExecutionError(diagnostic) from error
        except NameError as error:
            set_error(context, str(error))
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_INVALID_REFERENCE",
                    severity="error",
                    location=row_location_provider(context),
                    message=f"Invalid reference while evaluating expression '{expression}': {error}",
                )
            ) from error
        except Exception as error:
            set_error(context, str(error))
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_EXPRESSION_EVALUATION_ERROR",
                    severity="error",
                    location=row_location_provider(context),
                    message=f"Expression evaluation failed for '{expression}': {error}",
                )
            ) from error

    def _build_scope(
        self,
        *,
        row: Mapping[str, Any],
        context: Any,
        include_row_aliases: bool,
    ) -> dict[str, Any]:
        def resolve_visible_value(key: str) -> Any:
            context_value = context.get_variable(key)
            if context_value is not None:
                return context_value
            source_value = context.source_slot_value(key)
            if source_value is not None:
                return source_value
            return self._resolve_row_value(row, key)

        scope = dict(row)
        for source_name, source_value in context.source_slot_items():
            scope[source_name] = source_value
            scope.setdefault(_column_key(source_name), source_value)
            scope.setdefault(source_name.lower(), source_value)
        if include_row_aliases:
            for column_name, value in row.items():
                scope.setdefault(_column_key(column_name), value)
                scope.setdefault(column_name.lower(), value)

        scope["_N_"] = context.get_variable("_N_")
        scope["_ERROR_"] = context.get_variable("_ERROR_")
        scope["_n_"] = context.get_variable("_N_")
        scope["_error_"] = context.get_variable("_ERROR_")
        scope["__var__"] = resolve_visible_value
        scope.update(self._eval_scope_provider())
        return scope

    @staticmethod
    def _resolve_row_value(row: Mapping[str, Any], key: str) -> Any:
        if key in row:
            return row.get(key)

        normalized = _column_key(key)
        for candidate, value in row.items():
            if _column_key(candidate) == normalized:
                return value
        return None


__all__ = ["ConditionEvaluator"]
