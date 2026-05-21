from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Callable

from ._naming import _column_key
from .models import Diagnostic


class ExpressionEvaluator:
    _ASSIGNMENT_OPERATOR = re.compile(r"(?<![<>=!])=(?!=)")
    _MISSING_LITERAL = re.compile(r"(?<![\w])\.(?![\w])")
    _ARRAY_REF = re.compile(r"^\s*([A-Za-z_][\w\.]*)\s*[\(\[\{]\s*(.+?)\s*[\)\]\}]\s*$")
    _BARE_FORMAT_TOKEN = re.compile(
        r"^(?:[A-Za-z_]\w*|\d+(?:\.\d+)?|z\d+(?:\.\d+)?|comma\d+(?:\.\d+)?)\.$",
        re.IGNORECASE,
    )

    def __init__(self, eval_scope_provider: Callable[[], dict[str, Any]]) -> None:
        self._eval_scope_provider = eval_scope_provider
        self._compiled_expression_cache: dict[str, Any] = {}
        self._normalized_expression_cache: dict[str, str] = {}
        self._transformed_expression_cache: dict[tuple[str, tuple[str, ...]], str] = {}

    def is_missing_value(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        return False

    def parse_array_reference(self, token: str) -> tuple[str, str] | None:
        matched = self._ARRAY_REF.match(token)
        if matched is None:
            return None
        return matched.group(1), matched.group(2).strip()

    def resolve_array_variable(
        self,
        array_name: str,
        index_value: Any,
        array_defs: Mapping[str, tuple[str, ...]],
    ) -> tuple[str, Diagnostic | None]:
        variables = array_defs.get(array_name)
        if variables is None:
            return "", Diagnostic(
                code="RUNTIME_ARRAY_INVALID",
                severity="error",
                message=f"Array is not defined: {array_name}",
            )

        try:
            index = int(index_value)
        except Exception:
            return "", Diagnostic(
                code="RUNTIME_ARRAY_INVALID",
                severity="error",
                message=f"Array index must be numeric: {array_name}[{index_value}]",
            )

        if index < 1 or index > len(variables):
            return "", Diagnostic(
                code="RUNTIME_ARRAY_INVALID",
                severity="error",
                message=f"Array index out of bounds: {array_name}[{index}] valid range is 1..{len(variables)}",
            )

        return variables[index - 1], None

    def transform_array_expression(self, expression: str, array_defs: Mapping[str, tuple[str, ...]]) -> str:
        transformed = expression
        for array_name in array_defs.keys():
            transformed = re.sub(
                rf"\bdim\s*\(\s*{re.escape(array_name)}\s*\)",
                f"__dim__('{array_name}')",
                transformed,
            )
            transformed = re.sub(
                rf"\bvname\s*\(\s*{re.escape(array_name)}\s*[\(\[\{{]\s*([^\)\]\}}]+)\s*[\)\]\}}]\s*\)",
                rf"__vname__('{array_name}', \1)",
                transformed,
            )
            transformed = re.sub(
                rf"\b{re.escape(array_name)}\s*[\(\[\{{]\s*([^\)\]\}}]+)\s*[\)\]\}}]",
                rf"__arr_get__('{array_name}', \1)",
                transformed,
            )
        return transformed

    def transform_dotted_variable_expression(self, expression: str, row: Mapping[str, Any]) -> str:
        transformed = expression
        dotted_keys = sorted((key for key in row.keys() if "." in key), key=len, reverse=True)
        for key in dotted_keys:
            transformed = re.sub(
                rf"(?<![\w\']){re.escape(key)}(?![\w\'])",
                f"__var__('{key}')",
                transformed,
            )
        return transformed

    def prepare_expression(self, expression: str) -> str:
        normalized = self._normalized_expression_cache.get(expression)
        if normalized is None:
            normalized = self.normalize_expression(expression)
            self._normalized_expression_cache[expression] = normalized
        return normalized

    def evaluate_scalar(
        self,
        expression: str,
        row: Mapping[str, Any],
        context: Any,
        array_defs: Mapping[str, tuple[str, ...]],
    ) -> tuple[Any, Diagnostic | None]:
        normalized = self.prepare_expression(expression)

        array_transformed = self.transform_array_expression(normalized, array_defs=array_defs)
        dotted_keys = tuple(sorted((key for key in row.keys() if "." in key), key=len, reverse=True))
        transform_cache_key = (array_transformed, dotted_keys)
        transformed = self._transformed_expression_cache.get(transform_cache_key)
        if transformed is None:
            transformed = array_transformed
            for key in dotted_keys:
                transformed = re.sub(
                    rf"(?<![\w\']){re.escape(key)}(?![\w\'])",
                    f"__var__('{key}')",
                    transformed,
                )
            self._transformed_expression_cache[transform_cache_key] = transformed

        try:
            compiled = self._compiled_expression_cache.get(transformed)
            if compiled is None:
                compiled = compile(transformed, "<limulus-scalar>", "eval")
                self._compiled_expression_cache[transformed] = compiled

            scope = self._build_scope(
                row=row,
                context=context,
                array_defs=array_defs,
                include_row_aliases=False,
            )
            try:
                return eval(compiled, {"__builtins__": {}}, scope), None
            except NameError:
                fallback_scope = self._build_scope(
                    row=row,
                    context=context,
                    array_defs=array_defs,
                    include_row_aliases=True,
                )
                return eval(compiled, {"__builtins__": {}}, fallback_scope), None
        except Exception as error:
            return None, Diagnostic(
                code="RUNTIME_EXPRESSION_EVALUATION_ERROR",
                severity="error",
                message=f"Expression evaluation failed for '{expression}': {error}",
            )

    def _build_scope(
        self,
        *,
        row: Mapping[str, Any],
        context: Any,
        array_defs: Mapping[str, tuple[str, ...]],
        include_row_aliases: bool,
    ) -> dict[str, Any]:
        scope = dict(row)
        if include_row_aliases:
            for column_name, value in row.items():
                scope.setdefault(_column_key(column_name), value)
                scope.setdefault(column_name.lower(), value)

        scope["_N_"] = context.get_variable("_N_")
        scope["_ERROR_"] = context.get_variable("_ERROR_")
        scope["_n_"] = context.get_variable("_N_")
        scope["_error_"] = context.get_variable("_ERROR_")
        scope.update(self._eval_scope_provider())

        def __dim__(array_name: str) -> int:
            return len(array_defs.get(array_name, ()))

        def __vname__(array_name: str, index_value: Any) -> str:
            variable_name, error = self.resolve_array_variable(array_name, index_value, array_defs)
            if error is not None:
                raise ValueError(error.message)
            return variable_name

        def __arr_get__(array_name: str, index_value: Any) -> Any:
            variable_name, error = self.resolve_array_variable(array_name, index_value, array_defs)
            if error is not None:
                raise ValueError(error.message)
            return self._resolve_row_value(row, variable_name)

        scope["__dim__"] = __dim__
        scope["__vname__"] = __vname__
        scope["__arr_get__"] = __arr_get__
        scope["__var__"] = lambda key: self._resolve_row_value(row, key)
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

    def normalize_expression(self, expression: str) -> str:
        operator_normalized = expression.replace("^=", "!=").replace("¬=", "!=").replace("~=", "!=")
        operator_normalized = operator_normalized.replace("><", " @__unsupported_operator__ ")
        format_normalized = self.normalize_format_function_arguments(operator_normalized)
        concat_normalized = self.normalize_concat_operator(format_normalized)
        replaced_equality = self._ASSIGNMENT_OPERATOR.sub("==", concat_normalized)
        return self._MISSING_LITERAL.sub("None", replaced_equality)

    def normalize_format_function_arguments(self, expression: str) -> str:
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
                    result.append(expression[whitespace_start:cursor - len(bracketed)])
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

            if ch in "([{" :
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

            if ch in "([{" :
                bracketed, next_cursor = self._consume_bracket_segment(expression, cursor)
                current.append(bracketed)
                cursor = next_cursor
                continue

            if cursor + 1 < length and expression[cursor:cursor + 2] == "||":
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

    def _consume_quoted_segment(self, text: str, start: int) -> tuple[str, int]:
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
                    inner = text[start + 1:cursor]
                    normalized_inner = self.normalize_concat_operator(inner)
                    return f"{opening}{normalized_inner}{closing}", cursor + 1

            cursor += 1

        return text[start:], length
