"""Row runtime services for row-by-row execution context management.

This module implements the core execution model that preserves Data Step-like
automatic variables and row-by-row iteration semantics.
"""
from collections.abc import Mapping, Sequence
import datetime as dt
import math
import re
from typing import Any, Callable, Optional

from .arrow_cursor import ArrowRowCursorProtocol

from ..naming import _column_key
from ..format_registry import FormatRegistry
from ..models import Diagnostic, DiagnosticLabel, DiagnosticSpan
from .conditions import ConditionEvaluator



class RuntimeExecutionError(Exception):
    """Raised when runtime expression evaluation fails."""

    def __init__(self, diagnostic: Diagnostic, fatal: bool = True) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic
        self.fatal = fatal


class FunctionArgumentError(Exception):
    """Raised when a supported function receives invalid arguments."""

    def __init__(self, message: str, diagnostic: Diagnostic | None = None) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


class RowView(Mapping):
    """Lightweight read-only view over a source row mapping.

    Avoids an immediate dict copy at row-loop entry in the basic execution path.
    All methods that only read rows accept ``Mapping[str, Any]``, so a ``RowView``
    can be passed directly.  Dict materialization is deferred to mutation
    boundaries (``evaluate_if_chain``, ``apply_drop_keep``, ``route_output_record``).
    """

    __slots__ = ("_row",)

    def __init__(self, row: Mapping[str, Any]) -> None:
        self._row = row

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        return self._row[key]

    def __iter__(self):
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

    def __contains__(self, key: object) -> bool:
        return key in self._row


class RuntimeContext:
    """Represents row-runtime execution state.

    The RuntimeContext maintains state for a single execution of a data step,
    including automatic variables (_N_, _ERROR_) and user-defined variables.
    Variables persist across row iterations when the data step retains them.
    """

    def __init__(self) -> None:
        self._variables: dict[str, Any] = {}
        self._original_names: dict[str, str] = {}
        self._slot_name_hints: dict[str, str] = {}
        self._slot_indices: dict[str, dict[str, int]] = {}
        self._slot_values: dict[str, dict[int, Any]] = {}
        self._source_slot_names: dict[int, str] = {}

    def get_variable(self, name: str) -> Optional[Any]:
        key = _column_key(name)
        for section_name in ("automatic_slots", "mutable_slots"):
            section = self._slot_indices.get(section_name, {})
            slot_index = section.get(key)
            if slot_index is None:
                continue
            if slot_index in self._slot_values.get(section_name, {}):
                return self._slot_values[section_name][slot_index]
        return self._variables.get(key)

    def remember_variable_name(self, name: str) -> str:
        key = _column_key(name)
        self._original_names.setdefault(key, name)
        return self._original_names[key]

    def original_name(self, name: str) -> str | None:
        return self._original_names.get(_column_key(name))

    def set_slot_registry(self, slot_registry: Mapping[str, Mapping[str, int]] | None) -> None:
        self._slot_name_hints = {}
        self._slot_indices = {}
        self._slot_values = {}
        self._source_slot_names = {}
        if not isinstance(slot_registry, Mapping):
            return
        for section_name in ("source_slots", "mutable_slots", "automatic_slots"):
            section = slot_registry.get(section_name)
            if not isinstance(section, Mapping):
                continue
            normalized_section: dict[str, int] = {}
            for name in section.keys():
                slot_index = section[name]
                if isinstance(name, str) and isinstance(slot_index, int):
                    normalized_name = _column_key(name)
                    self._slot_name_hints[normalized_name] = name
                    normalized_section[normalized_name] = slot_index
                    if section_name == "source_slots":
                        self._source_slot_names[slot_index] = name
            if normalized_section:
                self._slot_indices[section_name] = normalized_section
                self._slot_values[section_name] = {}

    def slot_name_hint(self, name: str) -> str | None:
        return self._slot_name_hints.get(_column_key(name))

    def slot_index(self, name: str, *, sections: tuple[str, ...]) -> tuple[str, int] | None:
        normalized = _column_key(name)
        for section_name in sections:
            section = self._slot_indices.get(section_name, {})
            slot_index = section.get(normalized)
            if slot_index is not None:
                return section_name, slot_index
        return None

    def bind_source_row(self, row: Mapping[str, Any]) -> None:
        source_section = self._slot_indices.get("source_slots", {})
        if not source_section:
            return
        values: dict[int, Any] = {}
        normalized_row = {_column_key(str(name)): value for name, value in row.items()}
        for normalized_name, slot_index in source_section.items():
            if normalized_name in normalized_row:
                values[slot_index] = normalized_row[normalized_name]
        self._slot_values["source_slots"] = values

    def bind_source_cursor(self, cursor: ArrowRowCursorProtocol, row_index: int) -> None:
        source_section = self._slot_indices.get("source_slots", {})
        if not source_section:
            return

        cursor.bind(row_index)
        values: dict[int, Any] = {}
        for slot_index in source_section.values():
            values[slot_index] = cursor.value_at_slot(slot_index)
        self._slot_values["source_slots"] = values

    def source_slot_value(self, name: str) -> Any:
        slot_ref = self.slot_index(name, sections=("source_slots",))
        if slot_ref is None:
            return None
        section_name, slot_index = slot_ref
        return self._slot_values.get(section_name, {}).get(slot_index)

    def source_slot_items(self) -> list[tuple[str, Any]]:
        return [
            (self._source_slot_names[slot_index], value)
            for slot_index, value in self._slot_values.get("source_slots", {}).items()
            if slot_index in self._source_slot_names
        ]

    def visible_variable_names(self) -> tuple[str, ...]:
        names: set[str] = set()
        names.update(self._original_names.values())
        names.update(self._slot_name_hints.values())
        names.update(self._variables.keys())
        return tuple(sorted(names))

    def set_variable(self, name: str, value: Any) -> None:
        key = _column_key(name)
        self._variables[key] = value
        self.remember_variable_name(name)
        slot_ref = self.slot_index(name, sections=("automatic_slots", "mutable_slots"))
        if slot_ref is not None:
            section_name, slot_index = slot_ref
            self._slot_values.setdefault(section_name, {})[slot_index] = value

    def has_variable(self, name: str) -> bool:
        return _column_key(name) in self._variables


class ProgramExecutionService:
    """Runtime backend execution orchestrator with selection-first routing."""

    def __init__(self, backend_selector: Any) -> None:
        self._backend_selector = backend_selector
        self._last_backend = "python"

    @property
    def last_backend(self) -> str:
        return self._last_backend

    def execute(self, context: Any, preferred_backend: str) -> tuple[dict[str, Any], list[Diagnostic]]:
        backend = self._backend_selector(preferred_backend, context)
        self._last_backend = backend.name
        outputs, diagnostics = backend.execute(context)
        # Keep the public auto-mode contract: fall back only when a rust execution
        # attempt returns diagnostics at runtime.
        if diagnostics and backend.name != "python" and preferred_backend.strip().lower() == "auto":
            fallback_backend = self._backend_selector("python_fallback", context)
            self._last_backend = fallback_backend.name
            return fallback_backend.execute(context)
        return outputs, diagnostics


class RowRuntimeService:
    """Service for managing row-runtime execution."""

    # Automatic variable names per DATA step convention
    _AUTO_VAR_N = "_N_"
    _AUTO_VAR_ERROR = "_ERROR_"
    RuntimeExecutionError = RuntimeExecutionError

    def create_context(self, slot_registry: Mapping[str, Mapping[str, int]] | None = None) -> RuntimeContext:
        context = RuntimeContext()
        context.set_slot_registry(slot_registry)
        context.set_variable(self._AUTO_VAR_N, 0)
        context.set_variable(self._AUTO_VAR_ERROR, 0)
        self._function_registry.reset_state()
        return context

    def __init__(self, format_registry: FormatRegistry | None = None) -> None:
        self._function_registry = FunctionRegistryService(format_registry=format_registry)
        self._condition_evaluator = ConditionEvaluator(
            eval_scope_provider=self._function_registry.get_eval_scope,
            unsupported_function_finder=self._function_registry.find_unsupported_function,
        )

    def set_row_view(self, row_index: int, rows: Sequence[Mapping[str, Any]]) -> None:
        self._function_registry.set_row_view(row_index=row_index, rows=rows)

    def bind_source(
        self,
        *,
        context: RuntimeContext,
        row: Mapping[str, Any],
        row_index: int,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        cursor = getattr(rows, "cursor", None)
        if cursor is not None:
            try:
                context.bind_source_cursor(cursor, row_index)
                return
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                pass
        context.bind_source_row(row)


    def get_eval_scope(self) -> dict[str, Any]:
        return self._function_registry.get_eval_scope()

    def list_registered_functions(self) -> tuple[str, ...]:
        return self._function_registry.list_registered_function_names()

    def begin_row(self, context: RuntimeContext) -> None:
        n_value = context.get_variable(self._AUTO_VAR_N)
        if n_value is None:
            # Context not properly initialized, set to 0
            n_value = 0
        context.set_variable(self._AUTO_VAR_N, n_value + 1)
        context.set_variable(self._AUTO_VAR_ERROR, 0)

    def set_error(self, context: RuntimeContext, message: str) -> None:
        context.set_variable(self._AUTO_VAR_ERROR, 1)

    def evaluate_if_chain(
        self,
        row: Mapping[str, Any],
        context: RuntimeContext,
        branches: list[tuple[str, Mapping[str, Any]]],
        else_assignments: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Evaluate IF/ELSE IF/ELSE style branches for a single row.

        Returns a new row dictionary with branch assignments applied.
        """
        updated_row = dict(row)

        for condition, assignments in branches:
            if self._evaluate_condition(updated_row, context, condition):
                updated_row.update(assignments)
                return updated_row

        if else_assignments:
            updated_row.update(else_assignments)

        return updated_row

    def passes_where(
        self,
        row: Mapping[str, Any],
        context: RuntimeContext,
        where_expression: str,
    ) -> bool:
        return self._evaluate_condition(row, context, where_expression)

    def prepare_condition_expression(self, expression: str) -> None:
        self._condition_evaluator.prepare_expression(expression)

    def apply_drop_keep(
        self,
        row: Mapping[str, Any],
        drop_vars: tuple[str, ...],
        keep_vars: tuple[str, ...],
    ) -> dict[str, Any]:
        working = dict(row)

        if keep_vars:
            keep_set = set(keep_vars)
            working = {name: value for name, value in working.items() if name in keep_set}

        if drop_vars:
            drop_set = set(drop_vars)
            working = {name: value for name, value in working.items() if name not in drop_set}

        return working

    def create_output_buffers(
        self,
        output_targets: tuple[str, ...],
    ) -> dict[str, list[dict[str, Any]]]:
        return {target: [] for target in output_targets}

    def route_output_record(
        self,
        context: RuntimeContext,
        row: Mapping[str, Any],
        target: str,
        routed_outputs: dict[str, list[dict[str, Any]]],
        diagnostics: list[Diagnostic],
    ) -> None:
        if target not in routed_outputs:
            diagnostics.append(
                Diagnostic(
                    code="RUNTIME_OUTPUT_TARGET_NOT_FOUND",
                    severity="error",
                    location=self._row_location(context),
                    message=f"Output target is not declared: {target}",
                )
            )
            self.set_error(context, f"Output target is not declared: {target}")
            return

        routed_outputs[target].append(dict(row))

    def register_row_names(self, context: RuntimeContext, row: Mapping[str, Any]) -> None:
        for name in row.keys():
            context.remember_variable_name(name)

    def has_row_name(self, row: Mapping[str, Any], name: str) -> bool:
        if name in row:
            return True

        normalized = _column_key(name)
        return any(_column_key(candidate) == normalized for candidate in row.keys())

    def resolve_row_name(self, row: Mapping[str, Any], context: RuntimeContext, name: str) -> str:
        if name in row:
            context.remember_variable_name(name)
            return name

        original = context.original_name(name)
        if original is not None and original in row:
            return original

        normalized = _column_key(name)
        for candidate in row.keys():
            if _column_key(candidate) == normalized:
                context.remember_variable_name(candidate)
                return candidate

        slot_hint = context.slot_name_hint(name)
        if slot_hint is not None:
            context.remember_variable_name(slot_hint)
            return slot_hint

        return context.remember_variable_name(name)

    def get_row_value(self, row: Mapping[str, Any], context: RuntimeContext, name: str) -> Any:
        slot_ref = context.slot_index(name, sections=("automatic_slots", "mutable_slots"))
        if slot_ref is not None:
            section_name, slot_index = slot_ref
            if slot_index in context._slot_values.get(section_name, {}):
                return context._slot_values[section_name][slot_index]
        source_value = context.source_slot_value(name)
        if source_value is not None:
            return source_value
        resolved = self.resolve_row_name(row, context, name)
        return row.get(resolved)

    def set_row_value(self, row: dict[str, Any], context: RuntimeContext, name: str, value: Any) -> str:
        resolved = self.resolve_row_name(row, context, name)
        row[resolved] = value
        context.set_variable(resolved, value)
        return resolved

    def export_row(self, row: Mapping[str, Any], context: RuntimeContext) -> dict[str, Any]:
        exported: dict[str, Any] = {}
        for name, value in row.items():
            exported[context.original_name(name) or name] = value
        return exported

    def _evaluate_condition(
        self,
        row: Mapping[str, Any],
        context: RuntimeContext,
        expression: str,
    ) -> bool:
        return self._condition_evaluator.evaluate(
            row=row,
            context=context,
            expression=expression,
            set_error=self.set_error,
            row_location_provider=self._row_location,
        )

    def _row_location(self, context: RuntimeContext) -> str:
        row_number = context.get_variable(self._AUTO_VAR_N)
        if isinstance(row_number, int):
            return f"row:{row_number}"
        return "row:0"

class FunctionRegistryService:
    _FUNCTION_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

    def __init__(self, format_registry: FormatRegistry | None = None) -> None:
        # Registry no longer used for user functions; kept for internal hooks
        self._registered_functions: dict[str, Any] = {}
        self._lag_queues: dict[int, list[Any]] = {}
        self._row_index = 0
        self._rows: list[dict[str, Any]] = []
        self._format_registry = format_registry or FormatRegistry()
        self._functions: dict[str, Any] = {
            "__dsl_concat__": self._dsl_concat,
            "prxmatch": self._prxmatch,
            "prxchange": self._prxchange,
            "substr": self._substr,
            "scan": self._scan,
            "compress": self._compress,
            "trim": self._trim,
            "upcase": self._upcase,
            "lowcase": self._lowcase,
            "propcase": self._propcase,
            "catx": self._catx,
            "cats": self._cats,
            "cat": self._cat,
            "catt": self._catt,
            "index": self._index,
            "find": self._find,
            "tranwrd": self._tranwrd,
            "translate": self._translate,
            "length": self._length,
            "lengthn": self._lengthn,
            "strip": self._strip,
            "reverse": self._reverse,
            "repeat": self._repeat,
            "countw": self._countw,
            "round": self._round,
            "ceil": self._ceil,
            "floor": self._floor,
            "int": self._int,
            "abs": self._abs,
            "mod": self._mod,
            "max": self._max,
            "min": self._min,
            "sum": self._sum,
            "mean": self._mean,
            "sqrt": self._sqrt,
            "log": self._log,
            "exp": self._exp,
            "sign": self._sign,
            "missing": self._missing,
            "nmiss": self._nmiss,
            "cmiss": self._cmiss,
            "shift": self._shift,
            "lag": self._lag,
            "lead": self._lead,
            "apply": self._apply,
            "mdy": self._mdy,
            "year": self._year,
            "intck": self._intck,
            "put": self._put,
            "input": self._input,
            "hour": self._hour,
        }
        self._eval_scope = dict(self._functions)
        for name, function in self._functions.items():
            self._eval_scope.setdefault(_column_key(name), function)
        self._supported_function_names = {_column_key(name) for name in self._functions}
        self._reserved_function_names = {_column_key(keyword) for keyword in {"and", "or", "not", "in"}}

    def list_registered_function_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._functions.keys()))

    def _dsl_concat(self, left: Any, right: Any) -> str:
        def to_text(value: Any) -> str:
            if value is None:
                return ""
            return str(value)

        return to_text(left) + to_text(right)


    def reset_state(self) -> None:
        self._lag_queues = {}
        self._row_index = 0
        self._rows: Sequence[Mapping[str, Any]] = []

    def set_row_view(self, row_index: int, rows: Sequence[Mapping[str, Any]]) -> None:
        self._row_index = row_index
        self._rows = rows

    def get_eval_scope(self) -> dict[str, Any]:
        return self._eval_scope


    def find_unsupported_function(self, expression: str) -> str | None:
        for name in self._FUNCTION_PATTERN.findall(expression):
            normalized = _column_key(name)
            if normalized in self._reserved_function_names:
                continue
            if normalized not in self._supported_function_names:
                return name
        return None

    def _prxmatch(self, pattern: Any, source: Any) -> int:
        compiled = self._compile_prx_pattern(pattern)
        source_text = "" if source is None else str(source)
        matched = compiled.search(source_text)
        if matched is None:
            return 0
        return matched.start() + 1

    def _prxchange(self, pattern: Any, times: Any, source: Any) -> str:
        compiled_pattern, replacement = self._parse_prxchange_pattern(pattern)
        source_text = "" if source is None else str(source)
        try:
            times_value = int(times)
        except (TypeError, ValueError) as error:
            raise FunctionArgumentError(f"Invalid prxchange times value: {times}") from error
        replace_count = 0 if times_value <= 0 else times_value
        return compiled_pattern.sub(replacement, source_text, count=replace_count)

    def _compile_prx_pattern(self, pattern: Any) -> re.Pattern[str]:
        raw = str(pattern)
        if len(raw) >= 2 and raw.startswith("/") and raw.count("/") >= 2:
            last_delimiter = raw.rfind("/")
            pattern_body = raw[1:last_delimiter]
            flags_part = raw[last_delimiter + 1 :]
            try:
                return re.compile(
                    pattern_body,
                    self._parse_regex_flags(
                        flags_part,
                        source_text=raw,
                        flags_offset=last_delimiter + 1,
                        source_id="<prxmatch-pattern>",
                    ),
                )
            except re.error as error:
                raise FunctionArgumentError(
                    f"Invalid prxmatch pattern: {pattern}",
                    diagnostic=self._build_regex_parse_diagnostic(
                        message=f"Invalid prxmatch pattern: {pattern}",
                        source_text=raw,
                        source_id="<prxmatch-pattern>",
                        start=1 + self._regex_error_offset(error),
                        end=2 + self._regex_error_offset(error),
                        label_message="regex syntax",
                        notes=(str(error),),
                    ),
                ) from error
        try:
            return re.compile(raw)
        except re.error as error:
            raise FunctionArgumentError(
                f"Invalid prxmatch pattern: {pattern}",
                diagnostic=self._build_regex_parse_diagnostic(
                    message=f"Invalid prxmatch pattern: {pattern}",
                    source_text=raw,
                    source_id="<prxmatch-pattern>",
                    start=self._regex_error_offset(error),
                    end=self._regex_error_offset(error) + 1,
                    label_message="regex syntax",
                    notes=(str(error),),
                ),
            ) from error

    def _parse_prxchange_pattern(self, pattern: Any) -> tuple[re.Pattern[str], str]:
        raw = str(pattern)
        if not raw.startswith("s/"):
            raise FunctionArgumentError(
                "prxchange pattern must start with s/",
                diagnostic=self._build_regex_parse_diagnostic(
                    message="prxchange pattern must start with s/",
                    source_text=raw,
                    source_id="<prxchange-pattern>",
                    start=0,
                    end=min(2, max(len(raw), 1)),
                    label_message="expected s/",
                ),
            )

        first_separator = raw.find("/", 2)
        if first_separator < 0:
            raise FunctionArgumentError(
                "prxchange pattern is missing replacement separator",
                diagnostic=self._build_regex_parse_diagnostic(
                    message="prxchange pattern is missing replacement separator",
                    source_text=raw,
                    source_id="<prxchange-pattern>",
                    start=0,
                    end=max(len(raw), 1),
                    label_message="missing separator",
                ),
            )
        second_separator = raw.find("/", first_separator + 1)
        if second_separator < 0:
            raise FunctionArgumentError(
                "prxchange pattern is missing closing separator",
                diagnostic=self._build_regex_parse_diagnostic(
                    message="prxchange pattern is missing closing separator",
                    source_text=raw,
                    source_id="<prxchange-pattern>",
                    start=first_separator + 1,
                    end=max(len(raw), first_separator + 2),
                    label_message="missing closing separator",
                ),
            )

        regex_body = raw[2:first_separator]
        replacement = raw[first_separator + 1 : second_separator]
        flags_part = raw[second_separator + 1 :]
        try:
            compiled = re.compile(
                regex_body,
                self._parse_regex_flags(
                    flags_part,
                    source_text=raw,
                    flags_offset=second_separator + 1,
                    source_id="<prxchange-pattern>",
                ),
            )
        except re.error as error:
            raise FunctionArgumentError(
                f"Invalid prxchange pattern: {pattern}",
                diagnostic=self._build_regex_parse_diagnostic(
                    message=f"Invalid prxchange pattern: {pattern}",
                    source_text=raw,
                    source_id="<prxchange-pattern>",
                    start=2 + self._regex_error_offset(error),
                    end=3 + self._regex_error_offset(error),
                    label_message="regex syntax",
                    notes=(str(error),),
                ),
            ) from error
        return compiled, replacement

    def _parse_regex_flags(
        self,
        flags_part: str,
        *,
        source_text: str,
        flags_offset: int,
        source_id: str,
    ) -> int:
        flags = 0
        for index, flag_char in enumerate(flags_part):
            lowered = flag_char.lower()
            if lowered == "i":
                flags |= re.IGNORECASE
                continue
            if lowered == "m":
                flags |= re.MULTILINE
                continue
            if lowered == "s":
                flags |= re.DOTALL
                continue
            if lowered == "x":
                flags |= re.VERBOSE
                continue
            start = flags_offset + index
            raise FunctionArgumentError(
                f"Unsupported PRX flag: {flag_char}",
                diagnostic=self._build_regex_parse_diagnostic(
                    message=f"Unsupported PRX flag: {flag_char}",
                    source_text=source_text,
                    source_id=source_id,
                    start=start,
                    end=start + 1,
                    label_message="unsupported flag",
                    notes=("Supported flags: i, m, s, x.",),
                ),
            )
        return flags

    def _build_regex_parse_diagnostic(
        self,
        *,
        message: str,
        source_text: str,
        source_id: str,
        start: int,
        end: int,
        label_message: str,
        notes: tuple[str, ...] = (),
    ) -> Diagnostic:
        span = self._regex_span(source_text, start, end, source_id=source_id)
        return Diagnostic(
            code="RUNTIME_FUNCTION_ARGUMENT_INVALID",
            severity="error",
            message=message,
            span=span,
            labels=(DiagnosticLabel(span=span, message=label_message),),
            notes=notes,
            source_text=source_text,
        )

    @staticmethod
    def _regex_error_offset(error: re.error) -> int:
        position = getattr(error, "pos", 0)
        if not isinstance(position, int):
            return 0
        return max(position, 0)

    @staticmethod
    def _regex_span(source_text: str, start: int, end: int, *, source_id: str) -> DiagnosticSpan:
        safe_start = max(min(start, len(source_text)), 0)
        safe_end = max(min(end, len(source_text)), safe_start + 1)
        prefix = source_text[:safe_start]
        line = prefix.count("\n") + 1
        last_newline = prefix.rfind("\n")
        column = safe_start + 1 if last_newline < 0 else safe_start - last_newline
        segment = source_text[safe_start:safe_end]
        end_line = line + segment.count("\n")
        end_column = column + max(safe_end - safe_start, 1) if "\n" not in segment else None
        return DiagnosticSpan(
            start=safe_start,
            end=safe_end,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
            source_id=source_id,
        )

    def _substr(self, value: Any, start: Any, length: Any | None = None) -> str:
        source = "" if value is None else str(value)
        start_index = max(int(start) - 1, 0)
        if length is None:
            return source[start_index:]
        return source[start_index : start_index + max(int(length), 0)]

    def _scan(self, value: Any, index: Any, delimiters: Any = " ") -> str:
        source = "" if value is None else str(value)
        delimiter_chars = str(delimiters) if delimiters is not None else " "
        splitter_pattern = "[" + re.escape(delimiter_chars) + "]+"
        parts = [part for part in re.split(splitter_pattern, source) if part]
        target_index = int(index) - 1
        if target_index < 0 or target_index >= len(parts):
            return ""
        return parts[target_index]

    def _compress(self, value: Any, chars: Any | None = None) -> str:
        source = "" if value is None else str(value)
        if chars is None:
            return "".join(part for part in source if not part.isspace())
        remove_set = set(str(chars))
        return "".join(part for part in source if part not in remove_set)

    def _trim(self, value: Any) -> str:
        return ("" if value is None else str(value)).rstrip()

    def _upcase(self, value: Any) -> str:
        return ("" if value is None else str(value)).upper()

    def _lowcase(self, value: Any) -> str:
        return ("" if value is None else str(value)).lower()

    def _propcase(self, value: Any) -> str:
        return ("" if value is None else str(value)).title()

    def _catx(self, delimiter: Any, *args: Any) -> str:
        delim = "" if delimiter is None else str(delimiter)
        cleaned = [str(arg).strip() for arg in args if arg is not None and str(arg).strip()]
        return delim.join(cleaned)

    def _cats(self, *args: Any) -> str:
        return "".join("" if arg is None else str(arg).strip() for arg in args)

    def _cat(self, *args: Any) -> str:
        return "".join("" if arg is None else str(arg) for arg in args)

    def _catt(self, *args: Any) -> str:
        return "".join("" if arg is None else str(arg).rstrip() for arg in args)

    def _index(self, source: Any, excerpt: Any) -> int:
        source_text = "" if source is None else str(source)
        excerpt_text = "" if excerpt is None else str(excerpt)
        if excerpt_text == "":
            return 1
        found = source_text.find(excerpt_text)
        return 0 if found < 0 else found + 1

    def _find(self, source: Any, excerpt: Any, start: Any = 1, modifiers: Any = "") -> int:
        source_text = "" if source is None else str(source)
        excerpt_text = "" if excerpt is None else str(excerpt)
        start_index = max(int(start) - 1, 0)
        modifier_text = "" if modifiers is None else str(modifiers).lower()
        if "i" in modifier_text:
            source_text = source_text.lower()
            excerpt_text = excerpt_text.lower()
        found = source_text.find(excerpt_text, start_index)
        return 0 if found < 0 else found + 1

    def _tranwrd(self, source: Any, target: Any, replacement: Any) -> str:
        source_text = "" if source is None else str(source)
        target_text = "" if target is None else str(target)
        replacement_text = "" if replacement is None else str(replacement)
        return source_text.replace(target_text, replacement_text)

    def _translate(self, source: Any, to_chars: Any, from_chars: Any) -> str:
        source_text = "" if source is None else str(source)
        to_text = "" if to_chars is None else str(to_chars)
        from_text = "" if from_chars is None else str(from_chars)
        replacement_map = {
            src: (to_text[index] if index < len(to_text) else "")
            for index, src in enumerate(from_text)
        }
        return "".join(replacement_map.get(char, char) for char in source_text)

    def _length(self, value: Any) -> int:
        return len("" if value is None else str(value))

    def _lengthn(self, value: Any) -> int:
        source = "" if value is None else str(value)
        return len(source) if source else 0

    def _strip(self, value: Any) -> str:
        return ("" if value is None else str(value)).strip()

    def _reverse(self, value: Any) -> str:
        return ("" if value is None else str(value))[::-1]

    def _repeat(self, value: Any, count: Any) -> str:
        source = "" if value is None else str(value)
        return source * max(int(count), 0)

    def _countw(self, value: Any, delimiters: Any = " ") -> int:
        source = "" if value is None else str(value)
        delimiter_chars = str(delimiters) if delimiters is not None else " "
        splitter_pattern = "[" + re.escape(delimiter_chars) + "]+"
        parts = [part for part in re.split(splitter_pattern, source.strip()) if part]
        return len(parts)

    def _round(self, value: Any, unit: Any = 1) -> float:
        numeric_value = float(value)
        numeric_unit = float(unit)
        if numeric_unit == 0:
            return numeric_value

        quotient = numeric_value / numeric_unit
        rounded_multiple = math.copysign(math.floor(abs(quotient) + 0.5), quotient)
        rounded_value = rounded_multiple * numeric_unit
        decimal_places = self._decimal_places_from_unit(numeric_unit)
        if decimal_places is None:
            return rounded_value

        normalized = round(rounded_value, decimal_places)
        if normalized == 0.0:
            return 0.0
        return normalized

    def _decimal_places_from_unit(self, unit: float) -> int | None:
        abs_unit = abs(unit)
        for places in range(13):
            scaled = abs_unit * (10**places)
            if math.isclose(scaled, round(scaled), rel_tol=0.0, abs_tol=1e-9):
                return places
        return None

    def _ceil(self, value: Any) -> int:
        return math.ceil(float(value))

    def _floor(self, value: Any) -> int:
        return math.floor(float(value))

    def _int(self, value: Any) -> int:
        return int(float(value))

    def _abs(self, value: Any) -> float:
        return abs(float(value))

    def _mod(self, value: Any, divisor: Any) -> float:
        return float(value) % float(divisor)

    def _max(self, *args: Any) -> Any:
        values = [arg for arg in args if not self._is_missing(arg)]
        if not values:
            return None
        return max(values)

    def _min(self, *args: Any) -> Any:
        values = [arg for arg in args if not self._is_missing(arg)]
        if not values:
            return None
        return min(values)

    def _sum(self, *args: Any) -> float:
        total = 0.0
        for arg in args:
            if self._is_missing(arg):
                continue
            total += float(arg)
        return total

    def _mean(self, *args: Any) -> float:
        values = [float(arg) for arg in args if not self._is_missing(arg)]
        if not values:
            return float("nan")
        return sum(values) / len(values)

    def _sqrt(self, value: Any) -> float:
        return math.sqrt(float(value))

    def _log(self, value: Any, base: Any | None = None) -> float:
        numeric_value = float(value)
        if base is None:
            return math.log(numeric_value)
        return math.log(numeric_value, float(base))

    def _exp(self, value: Any) -> float:
        return math.exp(float(value))

    def _sign(self, value: Any) -> int:
        numeric_value = float(value)
        if numeric_value > 0:
            return 1
        if numeric_value < 0:
            return -1
        return 0

    def _missing(self, value: Any) -> bool:
        return self._is_missing(value)

    def _nmiss(self, *args: Any) -> int:
        return sum(1 for arg in args if self._is_missing(arg))

    def _cmiss(self, *args: Any) -> int:
        return sum(1 for arg in args if self._is_missing(arg))

    def _shift(self, value_or_variable: Any, offset: Any = 1, default: Any = None) -> Any:
        steps = int(offset)
        if steps == 0:
            if isinstance(value_or_variable, str) and self._rows:
                if 0 <= self._row_index < len(self._rows):
                    return self._rows[self._row_index].get(value_or_variable, default)
                return default
            return value_or_variable

        if isinstance(value_or_variable, str) and self._rows:
            target_index = self._row_index - steps
            if target_index < 0 or target_index >= len(self._rows):
                return default
            return self._rows[target_index].get(value_or_variable, default)

        if steps < 0:
            raise FunctionArgumentError("shift with negative offset requires a variable name.")

        queue = self._lag_queues.setdefault(steps, [])
        queue.append(value_or_variable)
        if len(queue) <= steps:
            return default
        return queue.pop(0)

    def _lag(self, value: Any, offset: Any = 1, default: Any = None) -> Any:
        steps = int(offset)
        if steps < 1:
            raise FunctionArgumentError("lag offset must be >= 1")
        return self._shift(value, steps, default)

    def _lead(self, variable_name: Any, offset: Any = 1, default: Any = None) -> Any:
        steps = int(offset)
        if steps < 1:
            raise FunctionArgumentError("lead offset must be >= 1")
        key = str(variable_name)
        target_index = self._row_index + steps
        if target_index < 0 or target_index >= len(self._rows):
            return default
        return self._shift(key, -steps, default)

    def _apply(self, function_ref: Any, *args: Any) -> Any:
        if callable(function_ref):
            return function_ref(*args)

        function_name = str(function_ref)
        # look up in builtin function table, case-insensitive
        function = self._functions.get(function_name.lower())

        if function is None:
            # try builtin functions
            import builtins, importlib, inspect  # local import to keep module deps small

            function = getattr(builtins, function_name, None)

            # support dotted module paths (``math.sqrt`` etc)
            if function is None and "." in function_name:
                module_path, attr = function_name.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                except Exception:
                    module = None
                if module is not None:
                    function = getattr(module, attr, None)

            # if still missing try to resolve from caller's globals/locals
            if function is None:
                for frame_info in inspect.stack()[1:]:
                    f = frame_info.frame
                    if function_name in f.f_locals:
                        function = f.f_locals[function_name]
                        break
                    if function_name in f.f_globals:
                        function = f.f_globals[function_name]
                        break
                # avoid reference cycles
                del frame_info

        if function is None:
            raise FunctionArgumentError(f"Unknown apply function: {function_name}")

        if not callable(function):
            raise FunctionArgumentError(f"Resolved apply target is not callable: {function_name}")

        return function(*args)

    def _mdy(self, month: Any, day: Any, year: Any) -> dt.date:
        return dt.date(int(year), int(month), int(day))

    def _put(self, value: Any, format_name: Any) -> Any:
        return self._format_registry.put(value, format_name)

    def _input(self, value: Any, informat_name: Any) -> Any:
        return self._format_registry.input(value, informat_name)

    def _hour(self, value: Any) -> Any:
        return self._format_registry.hour(value)

    def _year(self, value: Any) -> int:
        if isinstance(value, dt.datetime):
            return value.year
        if isinstance(value, dt.date):
            return value.year
        return dt.date.fromisoformat(str(value)).year

    def _intck(self, interval: Any, start: Any, end: Any) -> int:
        interval_name = str(interval).lower()
        start_date = self._as_date(start)
        end_date = self._as_date(end)

        if interval_name == "day":
            return (end_date - start_date).days
        if interval_name == "month":
            return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        if interval_name == "year":
            return end_date.year - start_date.year
        raise ValueError(f"Unsupported intck interval: {interval}")

    def _as_date(self, value: Any) -> dt.date:
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, dt.date):
            return value
        return dt.date.fromisoformat(str(value))

    def _is_missing(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        if isinstance(value, float):
            return math.isnan(value)
        return False


class RetainArrayRuntimeService:
    def create_retain_state(self, variable_names: tuple[str, ...]) -> dict[str, Any]:
        return {name: None for name in variable_names}

    def apply_retain_values(
        self,
        row: Mapping[str, Any],
        retain_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        merged = dict(row)
        for name, value in retain_state.items():
            if name not in merged:
                merged[name] = value
        return merged

    def update_retain_state(
        self,
        row: Mapping[str, Any],
        retain_state: dict[str, Any],
    ) -> None:
        for name in list(retain_state.keys()):
            if name in row:
                retain_state[name] = row[name]

    def define_array(
        self,
        name: str,
        variables: tuple[str, ...],
    ) -> dict[str, tuple[str, ...]]:
        return {name: variables}

    def get_array_value(
        self,
        arrays: Mapping[str, tuple[str, ...]],
        row: Mapping[str, Any],
        array_name: str,
        index: int,
    ) -> Any:
        variables = arrays.get(array_name)
        if variables is None:
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_ARRAY_INVALID",
                    severity="error",
                    message=f"Array is not defined: {array_name}",
                )
            )

        if index < 1 or index > len(variables):
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_ARRAY_INVALID",
                    severity="error",
                    message=(
                        f"Array index out of bounds: {array_name}[{index}] "
                        f"valid range is 1..{len(variables)}"
                    ),
                )
            )

        variable_name = variables[index - 1]
        return row.get(variable_name)


class LoopControlService:
    def run_do_to(
        self,
        start: int,
        end: int,
        body: "Callable[[int], None]",
        step: int = 1,
        max_iterations: int = 10000,
    ) -> None:
        if step == 0:
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_LOOP_EVALUATION_ERROR",
                    severity="error",
                    message="DO TO loop step must not be zero.",
                )
            )

        iteration_count = 0
        current = start
        comparator = (lambda value: value <= end) if step > 0 else (lambda value: value >= end)

        while comparator(current):
            iteration_count += 1
            if iteration_count > max_iterations:
                raise RuntimeExecutionError(
                    Diagnostic(
                        code="RUNTIME_LOOP_LIMIT_EXCEEDED",
                        severity="error",
                        message="DO TO loop exceeded safety iteration limit.",
                    )
                )
            body(current)
            current += step

    def run_do_while(
        self,
        condition: "Callable[[], bool]",
        body: "Callable[[], None]",
        max_iterations: int = 10000,
    ) -> None:
        iteration_count = 0

        while True:
            try:
                should_continue = bool(condition())
            except Exception as error:
                raise RuntimeExecutionError(
                    Diagnostic(
                        code="RUNTIME_LOOP_EVALUATION_ERROR",
                        severity="error",
                        message=f"DO WHILE condition evaluation failed: {error}",
                    )
                ) from error

            if not should_continue:
                return

            iteration_count += 1
            if iteration_count > max_iterations:
                raise RuntimeExecutionError(
                    Diagnostic(
                        code="RUNTIME_LOOP_LIMIT_EXCEEDED",
                        severity="error",
                        message="DO WHILE loop exceeded safety iteration limit.",
                    )
                )
            body()

    def run_do_until(
        self,
        condition: "Callable[[], bool]",
        body: "Callable[[], None]",
        max_iterations: int = 10000,
    ) -> None:
        iteration_count = 0

        while True:
            iteration_count += 1
            if iteration_count > max_iterations:
                raise RuntimeExecutionError(
                    Diagnostic(
                        code="RUNTIME_LOOP_LIMIT_EXCEEDED",
                        severity="error",
                        message="DO UNTIL loop exceeded safety iteration limit.",
                    )
                )

            body()

            try:
                should_stop = bool(condition())
            except Exception as error:
                raise RuntimeExecutionError(
                    Diagnostic(
                        code="RUNTIME_LOOP_EVALUATION_ERROR",
                        severity="error",
                        message=f"DO UNTIL condition evaluation failed: {error}",
                    )
                ) from error

            if should_stop:
                return

__all__ = [
    "FunctionArgumentError",
    "FunctionRegistryService",
    "LoopControlService",
    "RowRuntimeService",
    "ProgramExecutionService",
    "RetainArrayRuntimeService",
    "RowView",
    "RuntimeContext",
    "RuntimeExecutionError",
]
