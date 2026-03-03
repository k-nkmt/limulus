"""PDV like Runtime Service for managing row-by-row execution context

This module implements the core runtime execution model that mimics 
Data Step's Program Data Vector (PDV) behavior with automatic variables
and row-by-row iteration semantics.
"""
from collections.abc import Mapping
import datetime as dt
import math
import re
from typing import Any, Callable, Optional

from .models import Diagnostic



class RuntimeExecutionError(Exception):
    """Raised when runtime expression evaluation fails."""

    def __init__(self, diagnostic: Diagnostic, fatal: bool = True) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic
        self.fatal = fatal


class FunctionArgumentError(Exception):
    """Raised when a supported function receives invalid arguments."""


class RuntimeContext:
    """Represents the PDV (Program Data Vector) execution context
    
    The RuntimeContext maintains state for a single execution of a data step,
    including automatic variables (_N_, _ERROR_) and user-defined variables.
    Variables persist across row iterations (PDV retention behavior).
    """

    def __init__(self) -> None:
        """Initialize an empty runtime context"""
        self._variables: dict[str, Any] = {}

    def get_variable(self, name: str) -> Optional[Any]:
        """Get variable value from context
        
        Args:
            name: Variable name to retrieve
            
        Returns:
            Variable value, or None if variable does not exist
        """
        return self._variables.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set variable value in context
        
        Args:
            name: Variable name to set
            value: Value to assign to the variable
        """
        self._variables[name] = value

    def has_variable(self, name: str) -> bool:
        """Check if variable exists in context
        
        Args:
            name: Variable name to check
            
        Returns:
            True if variable exists, False otherwise
        """
        return name in self._variables


class ProgramExecutionService:
    """Runtime backend execution orchestrator with python fallback."""

    def __init__(self, backend_selector: Any) -> None:
        self._backend_selector = backend_selector
        self._last_backend = "python"

    @property
    def last_backend(self) -> str:
        return self._last_backend

    def execute(self, context: Any, preferred_backend: str) -> tuple[dict[str, Any], list[Diagnostic]]:
        backend = self._backend_selector(preferred_backend)
        self._last_backend = backend.name
        outputs, diagnostics = backend.execute(context)
        if diagnostics and backend.name != "python":
            fallback_backend = self._backend_selector("python")
            self._last_backend = fallback_backend.name
            return fallback_backend.execute(context)
        return outputs, diagnostics


class PDVRuntimeService:
    """Service for managing PDV runtime execution
    
    This service implements the PDV lifecycle management, including:
    - Context creation with automatic variables
    - Row iteration with _N_ incrementing
    - Error state management with _ERROR_ flag
    
    Follows DATA step semantics where:
    - _N_ starts at 0 and increments at the start of each row
    - _ERROR_ is reset to 0 at the start of each row
    - Variables persist across rows (PDV retention)
    """

    # Automatic variable names per DATA step convention
    _AUTO_VAR_N = "_N_"
    _AUTO_VAR_ERROR = "_ERROR_"
    RuntimeExecutionError = RuntimeExecutionError

    def create_context(self) -> RuntimeContext:
        """Create a new execution context with automatic variables initialized
        
        Initializes:
        - _N_ to 0 (row counter, increments before first row)
        - _ERROR_ to 0 (error flag, 0 = no error, 1 = error occurred)
        
        Returns:
            New RuntimeContext ready for execution
        """
        context = RuntimeContext()
        context.set_variable(self._AUTO_VAR_N, 0)
        context.set_variable(self._AUTO_VAR_ERROR, 0)
        self._function_registry.reset_state()
        return context

    def __init__(self) -> None:
        self._function_registry = FunctionRegistryService()
        self._compiled_condition_cache: dict[str, Any] = {}
        self._transformed_condition_cache: dict[tuple[str, tuple[str, ...]], str] = {}
    def set_row_view(self, row_index: int, rows: list[dict[str, Any]]) -> None:
        self._function_registry.set_row_view(row_index=row_index, rows=rows)


    def get_eval_scope(self) -> dict[str, Any]:
        return self._function_registry.get_eval_scope()

    def list_registered_functions(self) -> tuple[str, ...]:
        return self._function_registry.list_registered_function_names()

    def begin_row(self, context: RuntimeContext) -> None:
        """Begin processing a new row - increment _N_ and reset _ERROR_
        
        Called at the start of each DATA step iteration to:
        1. Increment _N_ by 1 (row number)
        2. Reset _ERROR_ to 0 (clear error flag from previous row)
        
        Args:
            context: The execution context to update
        """
        n_value = context.get_variable(self._AUTO_VAR_N)
        if n_value is None:
            # Context not properly initialized, set to 0
            n_value = 0
        context.set_variable(self._AUTO_VAR_N, n_value + 1)
        context.set_variable(self._AUTO_VAR_ERROR, 0)

    def set_error(self, context: RuntimeContext, message: str) -> None:
        """Set error state in context
        
        Marks the current row as having an error by setting _ERROR_ to 1.
        Error flag persists for the current row until begin_row() is called.
        
        Args:
            context: The execution context to update
            message: Error description (reserved for future diagnostic logging)
        """
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
        """Evaluate WHERE expression and return whether row is eligible."""
        return self._evaluate_condition(row, context, where_expression)

    def prepare_condition_expression(self, expression: str) -> None:
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

    def apply_drop_keep(
        self,
        row: Mapping[str, Any],
        drop_vars: tuple[str, ...],
        keep_vars: tuple[str, ...],
    ) -> dict[str, Any]:
        """Apply KEEP first and then DROP to determine output variables."""
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
        """Create empty output buffers for all declared output targets."""
        return {target: [] for target in output_targets}

    def route_output_record(
        self,
        context: RuntimeContext,
        row: Mapping[str, Any],
        target: str,
        routed_outputs: dict[str, list[dict[str, Any]]],
        diagnostics: list[Diagnostic],
    ) -> None:
        """Route one record to the target dataset, or report unresolved target."""
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

    def _evaluate_condition(
        self,
        row: Mapping[str, Any],
        context: RuntimeContext,
        expression: str,
    ) -> bool:
        dotted_keys = tuple(sorted((key for key in row.keys() if "." in key), key=len, reverse=True))
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

        scope = dict(row)
        scope[self._AUTO_VAR_N] = context.get_variable(self._AUTO_VAR_N)
        scope[self._AUTO_VAR_ERROR] = context.get_variable(self._AUTO_VAR_ERROR)
        scope["_n_"] = context.get_variable(self._AUTO_VAR_N)
        scope["_error_"] = context.get_variable(self._AUTO_VAR_ERROR)
        scope["__var__"] = lambda key: row.get(key)
        # include built‑in and local functions via registry scope
        scope.update(self._function_registry.get_eval_scope())

        unsupported_function = self._function_registry.find_unsupported_function(expression)
        if unsupported_function is not None:
            self.set_error(context, unsupported_function)
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_UNSUPPORTED_FUNCTION",
                    severity="error",
                    location=self._row_location(context),
                    message=f"Unsupported function: {unsupported_function}",
                )
            )

        try:
            compiled = self._compiled_condition_cache.get(transformed_expression)
            if compiled is None:
                compiled = compile(transformed_expression, "<limulus-condition>", "eval")
                self._compiled_condition_cache[transformed_expression] = compiled
            return bool(eval(compiled, {"__builtins__": {}}, scope))
        except SyntaxError as error:
            self.set_error(context, str(error))
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_OPERATOR_NOT_SUPPORTED",
                    severity="error",
                    location=self._row_location(context),
                    message=f"Unsupported operator in expression '{expression}': {error}",
                )
            ) from error
        except FunctionArgumentError as error:
            self.set_error(context, str(error))
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_FUNCTION_ARGUMENT_INVALID",
                    severity="error",
                    location=self._row_location(context),
                    message=f"Invalid function arguments in expression '{expression}': {error}",
                )
            ) from error
        except NameError as error:
            self.set_error(context, str(error))
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_INVALID_REFERENCE",
                    severity="error",
                    location=self._row_location(context),
                    message=f"Invalid reference while evaluating expression '{expression}': {error}",
                )
            ) from error
        except Exception as error:
            self.set_error(context, str(error))
            raise RuntimeExecutionError(
                Diagnostic(
                    code="RUNTIME_EXPRESSION_EVALUATION_ERROR",
                    severity="error",
                    location=self._row_location(context),
                    message=f"Expression evaluation failed for '{expression}': {error}",
                )
            ) from error

    def _row_location(self, context: RuntimeContext) -> str:
        row_number = context.get_variable(self._AUTO_VAR_N)
        if isinstance(row_number, int):
            return f"row:{row_number}"
        return "row:0"


class FunctionRegistryService:
    _FUNCTION_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

    def __init__(self) -> None:
        # Registry no longer used for user functions; kept for internal hooks
        self._registered_functions: dict[str, Any] = {}
        self._lag_queues: dict[int, list[Any]] = {}
        self._row_index = 0
        self._rows: list[dict[str, Any]] = []
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
        }

    def _dsl_concat(self, left: Any, right: Any) -> str:
        def to_text(value: Any) -> str:
            if value is None:
                return ""
            return str(value)

        return to_text(left) + to_text(right)


    def reset_state(self) -> None:
        self._lag_queues = {}
        self._row_index = 0
        self._rows = []

    def set_row_view(self, row_index: int, rows: list[dict[str, Any]]) -> None:
        self._row_index = row_index
        self._rows = rows

    def get_eval_scope(self) -> dict[str, Any]:
        # only built-in functions are provided; user-defined functions
        # should be resolved via Python scope lookup in _apply or through
        # the eval scope provider when expressions are evaluated.
        return dict(self._functions)


    def find_unsupported_function(self, expression: str) -> str | None:
        for name in self._FUNCTION_PATTERN.findall(expression):
            lowered = name.lower()
            if lowered in {"and", "or", "not", "in"}:
                continue
            if lowered not in self._functions:
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
                return re.compile(pattern_body, self._parse_regex_flags(flags_part))
            except re.error as error:
                raise FunctionArgumentError(f"Invalid prxmatch pattern: {pattern}") from error
        try:
            return re.compile(raw)
        except re.error as error:
            raise FunctionArgumentError(f"Invalid prxmatch pattern: {pattern}") from error

    def _parse_prxchange_pattern(self, pattern: Any) -> tuple[re.Pattern[str], str]:
        raw = str(pattern)
        if not raw.startswith("s/"):
            raise FunctionArgumentError("prxchange pattern must start with s/")

        first_separator = raw.find("/", 2)
        if first_separator < 0:
            raise FunctionArgumentError("prxchange pattern is missing replacement separator")
        second_separator = raw.find("/", first_separator + 1)
        if second_separator < 0:
            raise FunctionArgumentError("prxchange pattern is missing closing separator")

        regex_body = raw[2:first_separator]
        replacement = raw[first_separator + 1 : second_separator]
        flags_part = raw[second_separator + 1 :]
        try:
            compiled = re.compile(regex_body, self._parse_regex_flags(flags_part))
        except re.error as error:
            raise FunctionArgumentError(f"Invalid prxchange pattern: {pattern}") from error
        return compiled, replacement

    def _parse_regex_flags(self, flags_part: str) -> int:
        flags = 0
        for flag_char in flags_part:
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
            raise FunctionArgumentError(f"Unsupported PRX flag: {flag_char}")
        return flags

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

# Re-export executor classes so callers can use `from limulus.runtime import …`
from .executor import DataStepExecutor, FormatSupportResult, RuntimeRequirements  # noqa: E402
