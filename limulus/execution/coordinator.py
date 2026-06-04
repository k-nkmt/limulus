"""Executor orchestration and backend-neutral pipeline coordination."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..naming import _column_key
from ..backend_integration.backend_dispatch_policy import BackendDispatchPolicy
from ..backend_integration import (
    PythonRuntimeBackend,
    RustNativeBlockExecutor,
    RuntimeBackendSelector,
    RustArrowIOBridge,
    RustRuntimeBackend,
)
from ..backend_integration.contracts import RuntimeExecutionContext
from ..backend_integration.transport import (
    _PREPARED_MERGE_ROWS_MARKER as BACKEND_PREPARED_MERGE_ROWS_MARKER,
    _PREPARED_SET_ROWS_MARKER as BACKEND_PREPARED_SET_ROWS_MARKER,
    _STANDARD_TRANSPORT_KIND,
)
from ..block_splitter import DataStepBlockSplitter
from .pipeline import (
    ExecutionPipelineCoordinator,
    MacroHook,
    UnsupportedSyntaxMacroHook,
)
from .input_preparation import apply_dataset_reference_options_to_rows, try_apply_dataset_reference_options_to_arrow_table
from .rewrites import RewritePlanner
from ..format_registry import FormatRegistry
from ..io_adapters import (
    DataFrameAdapterPandas,
    DataInputAdapterArrow,
    DataOutputAdapterArrow,
)
from ..io import ExecutorIOService
from ..models import CompatibilityNotice, DataSetRef, Diagnostic, DiagnosticLabel, ExecuteRequest, ExecuteResponse, OutputConversionResult
from ..parser import (
    DatasetReference,
    ParserBackendSelector,
    ParsedStatement,
    ParserService,
    PythonParserBackend,
    RustNativeParserBackend,
    SetStatementOptionSpec,
)
from ..runtime.expressions import ExpressionEvaluator
from ..runtime.row_runtime import ProgramExecutionService, RowRuntimeService
from .python_backend import PythonBackendExecutionService


@dataclass(frozen=True)
class RuntimeRequirements:
    required_python: str
    supported_os: tuple[str, ...]


@dataclass(frozen=True)
class FormatSupportResult:
    supported: bool
    reason_code: str
    message: str


_SUPPORTED_FORMATS: tuple[str, ...] = (
    "polars",
    "pandas",
    "arrow_table",
)

_MISSING_VALUE_NOTICE = CompatibilityNotice(
    id="COMPAT_MISSING_VALUE_SEMANTICS",
    category="missing-values",
    summary="Missing Value Semantics Difference: missing values '.' may have different comparison and ordering behavior than Python/Arrow None/NaN.",
    link="compat://missing-value-semantics",
)

def _collect_compatibility_notices(dsl_text: str, inputs: Mapping[str, DataSetRef]) -> tuple[CompatibilityNotice, ...]:
    _missing_literal = re.compile(r"(?<![\w])\.(?![\w])")

    def _payload_contains_none(payload: object) -> bool:
        if payload is None:
            return False
        if isinstance(payload, Mapping):
            return any(value is None for value in payload.values())
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            for item in payload:
                if isinstance(item, Mapping) and any(value is None for value in item.values()):
                    return True
                if item is None:
                    return True
        return False

    has_missing = bool(_missing_literal.search(dsl_text))
    if not has_missing:
        has_missing = any(_payload_contains_none(dataset.payload) for dataset in inputs.values())

    return (_MISSING_VALUE_NOTICE,) if has_missing else ()


class _LazyArrowOutputs(Mapping[str, Any]):
    def __init__(
        self,
        outputs: Mapping[str, DataSetRef],
        converter: Any,
    ) -> None:
        self._outputs = outputs
        self._converter = converter
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        dataset_ref = self._outputs.get(key)
        if dataset_ref is None:
            raise KeyError(key)
        arrow_table, error = self._converter(key, dataset_ref)
        if error is not None or arrow_table is None:
            raise KeyError(key)
        self._cache[key] = arrow_table
        return arrow_table

    def __iter__(self):
        return iter(self._outputs.keys())

    def __len__(self) -> int:
        return len(self._outputs)


class DataStepExecutor:
    _PREPARED_SET_ROWS_MARKER = BACKEND_PREPARED_SET_ROWS_MARKER
    _PREPARED_MERGE_ROWS_MARKER = BACKEND_PREPARED_MERGE_ROWS_MARKER
    _PREPARED_INTERNAL_VARS_MARKER = "|internal="
    _REWRITE_FALLBACK_REASON_MARKER = "|rewrite_fallback="

    def __init__(
        self,
        runtime_backend: str = "python",
        parser_backend: str = "python",
        *,
        format_registry: FormatRegistry | None = None,
        macro_hook: MacroHook | None = None,
    ) -> None:
        self._parser = ParserService()
        self._block_splitter = DataStepBlockSplitter(parser_service=self._parser)
        self._runtime = RowRuntimeService(format_registry=format_registry)
        self._evaluator = ExpressionEvaluator(eval_scope_provider=lambda: self._runtime.get_eval_scope())
        self._arrow_input = DataInputAdapterArrow()
        self._arrow_output = DataOutputAdapterArrow()
        self._pandas_adapter = DataFrameAdapterPandas()
        self._io_service = ExecutorIOService(
            arrow_input=self._arrow_input,
            arrow_output=self._arrow_output,
            pandas_adapter=self._pandas_adapter,
            supported_formats=_SUPPORTED_FORMATS,
        )
        self._dictionary_input_cache: dict[tuple[str, tuple[tuple[str, int, int], ...]], DataSetRef] = {}
        self._registered_tables: dict[str, DataSetRef] = {}
        self._generated_tables: dict[str, DataSetRef] = {}
        self._runtime_backend_preference = runtime_backend
        self._python_backend_service = PythonBackendExecutionService(
            runtime=self._runtime,
            evaluator=self._evaluator,
            io_service=self._io_service,
        )
        self._backend_selector = RuntimeBackendSelector(
            python_backend=PythonRuntimeBackend(self._python_backend_service.execute),
            rust_backend=RustRuntimeBackend(
                bridge=RustArrowIOBridge(),
                executor=RustNativeBlockExecutor(),
            ),
        )
        self._program_execution = ProgramExecutionService(backend_selector=self._backend_selector.select)
        self._parser_backend_preference = parser_backend
        self._last_parser_backend = "python"
        self._rewrite_planner = RewritePlanner()
        self._parser_backend_selector = ParserBackendSelector(
            python_backend=PythonParserBackend(self._parser),
            rust_backend=RustNativeParserBackend(self._parser),
        )
        self._pipeline = ExecutionPipelineCoordinator(
            executor=self,
            macro_hook=macro_hook or UnsupportedSyntaxMacroHook(),
        )

    def register_table(self, name: str, dataset: DataSetRef) -> None:
        self._registered_tables[name] = dataset

    def unregister_table(self, name: str, *, missing_ok: bool = True) -> bool:
        target_key = self._dataset_name_key(name)
        matched_name = next(
            (candidate for candidate in self._registered_tables if self._dataset_name_key(candidate) == target_key),
            None,
        )
        if matched_name is None:
            if missing_ok:
                return False
            raise KeyError(name)
        del self._registered_tables[matched_name]
        return True

    def unregister_tables(self, names: Sequence[str], *, missing_ok: bool = True) -> dict[str, bool]:
        return {name: self.unregister_table(name, missing_ok=missing_ok) for name in names}

    def register_tables(self, datasets: Mapping[str, Any]) -> None:
        for name, dataset in datasets.items():
            self._registered_tables[name] = self._coerce_dataset_ref(name=name, dataset=dataset)


    def set_runtime_backend(self, runtime_backend: str) -> None:
        self._runtime_backend_preference = runtime_backend

    @property
    def last_runtime_backend(self) -> str:
        return self._program_execution.last_backend

    @property
    def last_phase_metrics(self) -> dict[str, float]:
        rust_backend = getattr(self._backend_selector, "_rust_backend", None)
        executor = getattr(rust_backend, "_executor", None)
        metrics = getattr(executor, "last_phase_metrics", None)
        if isinstance(metrics, Mapping):
            return dict(metrics)
        return {}

    def set_parser_backend(self, parser_backend: str) -> None:
        self._parser_backend_preference = parser_backend

    @property
    def last_parser_backend(self) -> str:
        return self._last_parser_backend

    def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        self._dictionary_input_cache.clear()
        return self._pipeline.execute(request)

    def convert_outputs(self, response: ExecuteResponse, target_format: str) -> OutputConversionResult:
        return self._io_service.convert_outputs(response, target_format)

    def _execute_program(self, context: RuntimeExecutionContext) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        outputs, diagnostics = self._program_execution.execute(context, self._runtime_backend_preference)

        skipped_count = sum(1 for statement in context.ast_statements if statement.kind == "SKIPPED")
        if skipped_count:
            diagnostics = [
                *diagnostics,
                *[
                    Diagnostic(
                        code="RUNTIME_UNSUPPORTED_STATEMENT_SKIPPED",
                        severity="info",
                        message="unsupported statement skipped",
                    )
                    for _ in range(skipped_count)
                ],
            ]

        return outputs, diagnostics


    def _resolve_inputs(
        self,
        ast_statements: Sequence[Any],
        explicit_inputs: Mapping[str, DataSetRef],
        available_inputs: Mapping[str, DataSetRef] | None = None,
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        synthetic_inputs = self._resolve_synthetic_inputs_for_names(
            requested_names=self._extract_source_input_names(ast_statements),
            explicit_inputs=explicit_inputs,
            available_inputs=available_inputs,
        )
        return self._io_service.resolve_inputs(
            ast_statements=ast_statements,
            explicit_inputs=explicit_inputs,
            available_inputs=available_inputs,
            registered_tables=self._registered_tables,
            synthetic_inputs=synthetic_inputs,
        )

    def _validate_ast_block(
        self,
        *,
        ast_statements: Sequence[Any],
        dsl_text: str,
        explicit_inputs: Mapping[str, DataSetRef],
        available_inputs: Mapping[str, DataSetRef],
        explicit_output_targets: Sequence[str],
    ) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        diagnostics.extend(
            self._validate_reserved_output_targets(
                ast_statements=ast_statements,
                dsl_text=dsl_text,
                explicit_output_targets=explicit_output_targets,
            )
        )
        diagnostics.extend(
            self._validate_input_dataset_refs(
                ast_statements=ast_statements,
                dsl_text=dsl_text,
                explicit_inputs=explicit_inputs,
                available_inputs=available_inputs,
            )
        )
        if diagnostics:
            return tuple(diagnostics)
        diagnostics.extend(
            self._validate_static_column_refs(
                ast_statements=ast_statements,
                dsl_text=dsl_text,
                explicit_inputs=explicit_inputs,
                available_inputs=available_inputs,
            )
        )
        return tuple(diagnostics)

    def _validate_reserved_output_targets(
        self,
        *,
        ast_statements: Sequence[Any],
        dsl_text: str,
        explicit_output_targets: Sequence[str],
    ) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        if explicit_output_targets:
            targets = tuple(self._normalize_dataset_name(target) for target in explicit_output_targets)
            statement = None
        else:
            targets = self._merge_unique_targets(
                self._extract_data_targets(ast_statements),
                self._io_service.extract_output_targets(ast_statements),
            )
            statement = next((item for item in ast_statements if item.kind in {"DATA", "OUTPUT"}), None)

        for target in targets:
            key = self._dataset_name_key(target)
            if key != "DICTIONARY" and not key.startswith("DICTIONARY."):
                continue
            diagnostics.append(
                self._build_validate_diagnostic(
                    code="VALIDATE_RESERVED_OUTPUT_TARGET",
                    message=f"Reserved output target is not allowed: {target}",
                    statement=statement,
                    dsl_text=dsl_text,
                )
            )
        return diagnostics

    def _validate_input_dataset_refs(
        self,
        *,
        ast_statements: Sequence[Any],
        dsl_text: str,
        explicit_inputs: Mapping[str, DataSetRef],
        available_inputs: Mapping[str, DataSetRef],
    ) -> list[Diagnostic]:
        source_statement = next(
            (statement for statement in ast_statements if statement.kind in {"SET", "MERGE"}),
            None,
        )
        if source_statement is None:
            return []

        source_refs = tuple(getattr(source_statement, "dataset_refs", ()) or ())
        input_names = [ref.name for ref in source_refs if getattr(ref, "name", "")]
        if not input_names:
            source_tokens = getattr(source_statement, "text", "").split()
            if len(source_tokens) >= 2:
                input_names = [source_tokens[1]]
        if not input_names:
            return [
                self._build_validate_diagnostic(
                    code="RUNTIME_SET_DATASET_NOT_FOUND",
                    message="SET statement requires an input dataset name.",
                    statement=source_statement,
                    dsl_text=dsl_text,
                )
            ]

        candidates: dict[str, DataSetRef] = {}
        candidates.update(self._registered_tables)
        candidates.update(available_inputs)
        candidates.update(explicit_inputs)
        candidates.update(
            self._resolve_synthetic_inputs_for_names(
                requested_names=input_names,
                explicit_inputs=explicit_inputs,
                available_inputs=available_inputs,
            )
        )

        diagnostics: list[Diagnostic] = []
        for input_name in input_names:
            if self._resolve_dataset_alias(candidates, input_name) is not None:
                continue
            diagnostics.append(
                self._build_validate_diagnostic(
                    code="RUNTIME_SET_DATASET_NOT_FOUND",
                    message=f"Input dataset is not provided: {input_name}",
                    statement=source_statement,
                    dsl_text=dsl_text,
                )
            )
        return diagnostics

    def _build_validate_diagnostic(
        self,
        *,
        code: str,
        message: str,
        statement: Any | None,
        dsl_text: str,
        label_message: str | None = None,
    ) -> Diagnostic:
        span = getattr(statement, "span", None)
        resolved_label = label_message or self._validate_label_message(code)
        return Diagnostic(
            code=code,
            severity="error",
            message=message,
            span=span,
            labels=((DiagnosticLabel(span=span, message=resolved_label),) if span is not None else ()),
            source_text=dsl_text if span is not None else None,
        )

    @staticmethod
    def _validate_label_message(code: str) -> str:
        categories = {
            "RUNTIME_SET_DATASET_NOT_FOUND": "missing dataset",
            "VALIDATE_RESERVED_OUTPUT_TARGET": "reserved name",
            "VALIDATE_COLUMN_NOT_FOUND": "unknown variable",
            "RUNTIME_BY_PRECONDITION_FAILED": "missing BY key",
            "RUNTIME_RENAME_STATEMENT_INVALID": "invalid rename",
            "RUNTIME_DATASET_OPTION_INVALID": "invalid dataset option",
        }
        return categories.get(code, "validation issue")

    def _validate_static_column_refs(
        self,
        *,
        ast_statements: Sequence[Any],
        dsl_text: str,
        explicit_inputs: Mapping[str, DataSetRef],
        available_inputs: Mapping[str, DataSetRef],
    ) -> list[Diagnostic]:
        source_statement = next(
            (statement for statement in ast_statements if statement.kind in {"SET", "MERGE"}),
            None,
        )
        if source_statement is None:
            return []

        source_refs = tuple(getattr(source_statement, "dataset_refs", ()) or ())
        candidates: dict[str, DataSetRef] = {}
        candidates.update(self._registered_tables)
        candidates.update(available_inputs)
        candidates.update(explicit_inputs)
        candidates.update(
            self._resolve_synthetic_inputs_for_names(
                requested_names=[ref.name for ref in source_refs if getattr(ref, "name", "")],
                explicit_inputs=explicit_inputs,
                available_inputs=available_inputs,
            )
        )

        source_columns: dict[str, set[str]] = {}
        available_columns: set[str] = set()
        diagnostics: list[Diagnostic] = []

        for source_ref in source_refs:
            input_ref = self._resolve_dataset_alias(candidates, source_ref.name)
            if input_ref is None:
                continue
            columns = self._infer_dataset_columns(input_ref)
            if columns is None:
                continue
            transformed_columns, option_diagnostics = self._apply_source_option_columns_for_validate(
                columns=columns,
                source_name=source_ref.name,
                option_spec=source_ref.options,
                statement=source_statement,
                dsl_text=dsl_text,
            )
            if option_diagnostics:
                diagnostics.extend(option_diagnostics)
                return diagnostics
            source_key = self._dataset_name_key(source_ref.name)
            source_columns[source_key] = transformed_columns
            available_columns.update(transformed_columns)

        by_keys = self._extract_variable_list(ast_statements, "BY")
        in_option_vars = [
            ref.options.in_var
            for ref in source_refs
            if getattr(getattr(ref, "options", None), "in_var", None) is not None
        ]
        statement_options = getattr(source_statement, "statement_options", None)
        available_columns.update(
            self._collect_internal_variable_names(
                in_option_vars=in_option_vars,
                indsname_var=getattr(statement_options, "indsname_var", None),
                end_var=getattr(statement_options, "end_var", None),
                by_keys=by_keys,
            )
        )
        available_columns.update(self._collect_step_defined_columns(ast_statements))

        for statement in ast_statements:
            if statement.kind == "BY":
                for by_key in self._extract_variable_list((statement,), "BY"):
                    normalized_by_key = _column_key(by_key)
                    missing_in_sources = [
                        source_name
                        for source_name, columns in source_columns.items()
                        if normalized_by_key not in {_column_key(column) for column in columns}
                    ]
                    if missing_in_sources:
                        diagnostics.append(
                            self._build_validate_diagnostic(
                                code="RUNTIME_BY_PRECONDITION_FAILED",
                                message=f"BY key '{by_key}' is missing in source rows.",
                                statement=statement,
                                dsl_text=dsl_text,
                                label_message="missing BY key",
                            )
                        )
                        return diagnostics
                continue

            column_diagnostic = self._validate_statement_column_refs(
                statement=statement,
                available_columns=available_columns,
                dsl_text=dsl_text,
            )
            if column_diagnostic is not None:
                diagnostics.append(column_diagnostic)
                return diagnostics

        return diagnostics

    def _validate_statement_column_refs(
        self,
        *,
        statement: Any,
        available_columns: set[str],
        dsl_text: str,
    ) -> Diagnostic | None:
        if statement.kind == "RENAME":
            rename_validation = self._validate_rename_statement_contract(statement.rename_map)
            if rename_validation is not None:
                return self._build_validate_diagnostic(
                    code=rename_validation.code,
                    message=rename_validation.message,
                    statement=statement,
                    dsl_text=dsl_text,
                    label_message="invalid rename",
                )
            referenced_columns = tuple(statement.rename_map)
            diagnostic_code = "RUNTIME_RENAME_STATEMENT_INVALID"
        else:
            referenced_columns = self._extract_statement_column_refs(statement)
            diagnostic_code = "VALIDATE_COLUMN_NOT_FOUND"

        if not referenced_columns:
            return None

        normalized_available = {_column_key(name) for name in available_columns}
        missing = [name for name in referenced_columns if _column_key(name) not in normalized_available]
        if not missing:
            return None

        return self._build_validate_diagnostic(
            code=diagnostic_code,
            message=f"{statement.kind} statement references unknown variable: {missing[0]}",
            statement=statement,
            dsl_text=dsl_text,
            label_message="unknown variable",
        )

    def _extract_statement_column_refs(self, statement: Any) -> tuple[str, ...]:
        if statement.kind in {"KEEP", "DROP"}:
            return self._extract_variable_list((statement,), statement.kind)

        if statement.kind == "LABEL":
            return tuple(statement.label_map)

        return ()

    def _collect_step_defined_columns(self, ast_statements: Sequence[Any]) -> set[str]:
        defined: set[str] = set()

        for statement in ast_statements:
            if statement.kind == "ASSIGN":
                target = self._extract_assignment_target(statement.text)
                if target is not None:
                    defined.add(target)
                continue

            if statement.kind == "SUM":
                target = self._extract_sum_target(statement.text)
                if target is not None:
                    defined.add(target)
                continue

            if statement.kind in {"IF", "ELSE IF"}:
                target = self._extract_if_then_target(statement)
                if target is not None:
                    defined.add(target)
                continue

            if statement.kind == "DO":
                loop_var = getattr(getattr(statement, "do_spec", None), "loop_var", None)
                if loop_var:
                    defined.add(loop_var)
                continue

            if statement.kind == "ARRAY":
                variables = getattr(getattr(statement, "array_spec", None), "variables", ())
                defined.update(name for name in variables if name)
                continue

            if statement.kind == "RETAIN":
                defined.update(self._extract_retain_targets(statement.text))

        return defined

    def _collect_step_defined_columns_ordered(self, ast_statements: Sequence[Any]) -> tuple[str, ...]:
        defined: list[str] = []
        seen: set[str] = set()

        def add(name: str | None) -> None:
            if not name:
                return
            key = _column_key(name)
            if key in seen:
                return
            seen.add(key)
            defined.append(name)

        for statement in ast_statements:
            if statement.kind == "ASSIGN":
                add(self._extract_assignment_target(statement.text))
                continue
            if statement.kind == "SUM":
                add(self._extract_sum_target(statement.text))
                continue
            if statement.kind in {"IF", "ELSE IF"}:
                add(self._extract_if_then_target(statement))
                continue
            if statement.kind == "DO":
                add(getattr(getattr(statement, "do_spec", None), "loop_var", None))
                continue
            if statement.kind == "ARRAY":
                for name in getattr(getattr(statement, "array_spec", None), "variables", ()):
                    add(name)
                continue
            if statement.kind == "RETAIN":
                for name in self._extract_retain_targets(statement.text):
                    add(name)

        return tuple(defined)

    def _extract_assignment_target(self, statement_text: str) -> str | None:
        matched = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:[\(\[\{].*?[\)\]\}])?\s*=", statement_text)
        if matched is None:
            return None
        return matched.group(1)

    def _extract_sum_target(self, statement_text: str) -> str | None:
        matched = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*\+\s*(.+?)\s*$", statement_text)
        if matched is None:
            return None
        return matched.group(1)

    def _extract_retain_targets(self, statement_text: str) -> tuple[str, ...]:
        body = statement_text.strip()
        if body.lower().startswith("retain"):
            body = body[6:].strip()
        if not body:
            return ()

        tokens = body.split()
        names: list[str] = []
        for token in tokens:
            if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", token):
                continue
            if token.startswith(("'", '"')):
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*", token):
                names.append(token)
        return tuple(names)

    def _extract_if_then_target(self, statement: Any) -> str | None:
        if_spec = getattr(statement, "if_spec", None)
        action = getattr(if_spec, "then_action", None)
        if not isinstance(action, str) or not action.strip():
            return None
        target = self._extract_assignment_target(action)
        if target is not None:
            return target
        return self._extract_sum_target(action)

    def _infer_dataset_columns(self, input_ref: DataSetRef) -> set[str] | None:
        normalized_kind = input_ref.kind.strip().lower()
        payload = input_ref.payload
        if normalized_kind == "arrow_table" and hasattr(payload, "schema"):
            schema = getattr(payload, "schema", None)
            names = getattr(schema, "names", None)
            if names is not None:
                return set(str(name) for name in names)

        rows, load_error = self._io_service.load_input_rows(input_ref)
        if load_error is not None:
            return None
        columns: set[str] = set()
        for row in rows:
            if isinstance(row, Mapping):
                columns.update(str(name) for name in row.keys())
        return columns

    def _infer_dataset_column_order(self, input_ref: DataSetRef) -> tuple[str, ...] | None:
        normalized_kind = input_ref.kind.strip().lower()
        payload = input_ref.payload
        if normalized_kind == "arrow_table" and hasattr(payload, "schema"):
            schema = getattr(payload, "schema", None)
            names = getattr(schema, "names", None)
            if names is not None:
                return tuple(str(name) for name in names)

        rows, load_error = self._io_service.load_input_rows(input_ref)
        if load_error is not None:
            return None
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for name in row.keys():
                key = _column_key(str(name))
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(str(name))
        return tuple(ordered)

    def _apply_source_option_column_order(
        self,
        columns: Sequence[str],
        source_spec: Mapping[str, Any],
    ) -> tuple[str, ...]:
        ordered = list(columns)
        if keep_vars := tuple(source_spec.get("keep", ()) or ()):
            keep_keys = {_column_key(name) for name in keep_vars}
            ordered = [name for name in ordered if _column_key(name) in keep_keys]
        if drop_vars := tuple(source_spec.get("drop", ()) or ()):
            drop_keys = {_column_key(name) for name in drop_vars}
            ordered = [name for name in ordered if _column_key(name) not in drop_keys]
        if rename_map := dict(source_spec.get("rename", {}) or {}):
            normalized_rename = {_column_key(source): str(target) for source, target in rename_map.items()}
            ordered = [normalized_rename.get(_column_key(name), name) for name in ordered]
        return tuple(ordered)

    def _build_slot_registry(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        source_specs: Sequence[Mapping[str, Any]],
        runtime_helper_columns: Sequence[str] = (),
    ) -> dict[str, dict[str, int]]:
        source_columns: list[str] = []
        seen_source: set[str] = set()
        for source_spec in source_specs:
            source_name = source_spec.get("source")
            if not isinstance(source_name, str):
                continue
            input_ref = resolved_inputs.get(source_name)
            if input_ref is None:
                continue
            ordered_columns = self._infer_dataset_column_order(input_ref)
            if ordered_columns is None:
                continue
            if self._uses_prepared_row_input(input_ref):
                planned_columns = ordered_columns
            else:
                planned_columns = self._apply_source_option_column_order(ordered_columns, source_spec)
            for column_name in planned_columns:
                key = _column_key(column_name)
                if key in seen_source:
                    continue
                seen_source.add(key)
                source_columns.append(column_name)

        mutable_columns = list(self._collect_step_defined_columns_ordered(ast_statements))
        seen_mutable = {_column_key(name) for name in mutable_columns}
        for helper_name in runtime_helper_columns:
            helper_key = _column_key(helper_name)
            if helper_key in seen_mutable:
                continue
            seen_mutable.add(helper_key)
            mutable_columns.append(helper_name)
        return {
            "source_slots": {name: index for index, name in enumerate(source_columns)},
            "mutable_slots": {name: index for index, name in enumerate(mutable_columns)},
            "automatic_slots": {"_N_": 0, "_ERROR_": 1},
        }

    def _uses_prepared_row_input(self, dataset_ref: DataSetRef) -> bool:
        location = dataset_ref.location or ""
        return self._PREPARED_SET_ROWS_MARKER in location or self._PREPARED_MERGE_ROWS_MARKER in location

    @staticmethod
    def _resolve_row_key(row: Mapping[str, Any], name: str) -> str | None:
        if name in row:
            return name

        normalized = _column_key(name)
        for candidate in row.keys():
            if _column_key(candidate) == normalized:
                return candidate
        return None

    def _resolve_row_value(self, row: Mapping[str, Any], name: str) -> Any:
        resolved = self._resolve_row_key(row, name)
        if resolved is None:
            return None
        return row.get(resolved)

    @staticmethod
    def _build_case_insensitive_scope(row: Mapping[str, Any]) -> dict[str, Any]:
        scope = dict(row)
        for column_name, value in row.items():
            scope.setdefault(_column_key(column_name), value)
            scope.setdefault(column_name.lower(), value)
        return scope

    def _apply_source_option_columns_for_validate(
        self,
        *,
        columns: set[str],
        source_name: str,
        option_spec: Any,
        statement: Any,
        dsl_text: str,
    ) -> tuple[set[str], list[Diagnostic]]:
        working = set(columns)
        keep_vars = tuple(getattr(option_spec, "keep_vars", ()))
        drop_vars = tuple(getattr(option_spec, "drop_vars", ()))
        rename_map = dict(getattr(option_spec, "rename_map", {}))

        def resolve_column(name: str) -> str | None:
            normalized_name = _column_key(name)
            for candidate in working:
                if _column_key(candidate) == normalized_name:
                    return candidate
            return None

        if keep_vars:
            missing = [name for name in keep_vars if resolve_column(name) is None]
            if missing:
                return set(), [
                    self._build_validate_diagnostic(
                        code="VALIDATE_COLUMN_NOT_FOUND",
                        message=f"Dataset option KEEP= references unknown variable '{missing[0]}' for source '{source_name}'.",
                        statement=statement,
                        dsl_text=dsl_text,
                        label_message="unknown variable",
                    )
                ]
            working = {resolved for name in keep_vars if (resolved := resolve_column(name)) is not None}

        if drop_vars:
            missing = [name for name in drop_vars if resolve_column(name) is None]
            if missing:
                return set(), [
                    self._build_validate_diagnostic(
                        code="VALIDATE_COLUMN_NOT_FOUND",
                        message=f"Dataset option DROP= references unknown variable '{missing[0]}' for source '{source_name}'.",
                        statement=statement,
                        dsl_text=dsl_text,
                        label_message="unknown variable",
                    )
                ]
            drop_columns = {resolved for name in drop_vars if (resolved := resolve_column(name)) is not None}
            working.difference_update(drop_columns)

        if rename_map:
            if len({_column_key(value) for value in rename_map.values()}) != len(rename_map):
                return set(), [
                    self._build_validate_diagnostic(
                        code="RUNTIME_DATASET_OPTION_INVALID",
                        message=f"Dataset option RENAME= has duplicate target names for source '{source_name}'.",
                        statement=statement,
                        dsl_text=dsl_text,
                        label_message="invalid rename",
                    )
                ]
            missing = [name for name in rename_map if resolve_column(name) is None]
            if missing:
                return set(), [
                    self._build_validate_diagnostic(
                        code="RUNTIME_DATASET_OPTION_INVALID",
                        message=f"Dataset option RENAME= references unknown variable '{missing[0]}' for source '{source_name}'.",
                        statement=statement,
                        dsl_text=dsl_text,
                        label_message="unknown variable",
                    )
                ]
            resolved_rename_map = {
                resolved: new_name
                for old_name, new_name in rename_map.items()
                if (resolved := resolve_column(old_name)) is not None
            }
            working = {resolved_rename_map.get(name, name) for name in working}

        return working, []

    def _validate_rename_statement_contract(self, rename_map: Mapping[str, str]) -> Diagnostic | None:
        if len(set(rename_map.values())) != len(rename_map):
            return Diagnostic(
                code="RUNTIME_RENAME_STATEMENT_INVALID",
                severity="error",
                message="RENAME statement has duplicate target variable names.",
            )

        for source, target in rename_map.items():
            if target in rename_map and rename_map.get(target) == source:
                return Diagnostic(
                    code="RUNTIME_RENAME_STATEMENT_INVALID",
                    severity="error",
                    message="RENAME statement contains circular reference.",
                )

        return None

    def _prepare_runtime_inputs(
        self,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        current_inputs = dict(resolved_inputs)
        diagnostics: list[Diagnostic] = []

        by_keys_for_sort = self._extract_variable_list(ast_statements, "BY")
        has_merge = any(stmt.kind == "MERGE" for stmt in ast_statements)
        has_lag_lead = self._uses_lag_lead_functions(ast_statements)
        set_statement = next((statement for statement in ast_statements if statement.kind == "SET"), None)
        set_source_refs = list(getattr(set_statement, "dataset_refs", ()) or ()) if set_statement is not None else []
        has_repeated_set_source_names = len({_column_key(ref.name) for ref in set_source_refs}) < len(set_source_refs)
        has_source_dataset_options = any(
            getattr(getattr(ref, "options", None), "keep_vars", ())
            or getattr(getattr(ref, "options", None), "drop_vars", ())
            or getattr(getattr(ref, "options", None), "where_expr", None) is not None
            or getattr(getattr(ref, "options", None), "rename_map", {})
            or getattr(getattr(ref, "options", None), "firstobs", None) is not None
            or getattr(getattr(ref, "options", None), "obs", None) is not None
            for ref in set_source_refs
        )
        set_input_collision = self._detect_set_input_internal_variable_collisions(
            set_statement=set_statement,
            source_refs=set_source_refs,
            resolved_inputs=current_inputs,
            by_keys=by_keys_for_sort,
        )
        if set_input_collision is not None:
            diagnostics.append(set_input_collision)
            return current_inputs, diagnostics

        if (
            not has_merge
            and not has_lag_lead
            and BackendDispatchPolicy.prefers_rust_first_execution(self._runtime_backend_preference)
        ):
            interleaved = self._try_interleave_set_inputs(
                ast_statements, current_inputs, by_keys_for_sort
            )
            if interleaved is not None:
                current_inputs = interleaved
        elif by_keys_for_sort and not has_merge:
            current_inputs = self._sort_inputs_by_by_keys(
                current_inputs, by_keys_for_sort
            )

        # Use prepared SET carriers only when source-local shaping or rewrite ownership requires it.
        should_prepare_set_rows = bool(
            set_source_refs
            and (
                has_repeated_set_source_names
                or has_source_dataset_options
                or has_lag_lead
            )
        )

        if should_prepare_set_rows:
            prepared_inputs, prepare_error = self._prepare_set_rows_with_temporary_variables(
                ast_statements=ast_statements,
                resolved_inputs=current_inputs,
                by_keys=by_keys_for_sort,
                has_lag_lead=has_lag_lead,
            )
            if prepare_error is not None:
                diagnostics.append(prepare_error)
                return current_inputs, diagnostics
            if prepared_inputs is not None:
                current_inputs = prepared_inputs

        if has_lag_lead and not has_merge:
            rewritten_inputs, rewrite_error = self._materialize_lag_lead_rewrite_inputs(
                ast_statements=ast_statements,
                resolved_inputs=current_inputs,
            )
            if rewrite_error is not None:
                diagnostics.append(rewrite_error)
                return current_inputs, diagnostics
            if rewritten_inputs is not None:
                current_inputs = rewritten_inputs

        return current_inputs, diagnostics

    def _detect_set_input_internal_variable_collisions(
        self,
        *,
        set_statement: Any | None,
        source_refs: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        by_keys: Sequence[str],
    ) -> Diagnostic | None:
        if set_statement is None or not source_refs:
            return None

        in_option_vars = [
            ref.options.in_var
            for ref in source_refs
            if getattr(ref.options, "in_var", None) is not None
        ]
        indsname_var = getattr(getattr(set_statement, "statement_options", None), "indsname_var", None)
        end_var = getattr(getattr(set_statement, "statement_options", None), "end_var", None)
        internal_variable_names = self._collect_internal_variable_names(
            in_option_vars=in_option_vars,
            indsname_var=indsname_var,
            end_var=end_var,
            by_keys=by_keys,
        )
        if not internal_variable_names:
            return None

        import pyarrow as pa

        for source_ref in source_refs:
            input_ref = resolved_inputs.get(source_ref.name)
            if input_ref is None:
                continue
            if input_ref.kind != "arrow_table" or not isinstance(input_ref.payload, pa.Table):
                continue

            table, option_error = try_apply_dataset_reference_options_to_arrow_table(
                input_ref.payload,
                source_name=source_ref.name,
                option_spec=source_ref.options,
            )
            if option_error is not None or table is None:
                continue

            allow_internal_names = (
                isinstance(getattr(input_ref, "location", None), str)
                and self._PREPARED_MERGE_ROWS_MARKER in str(input_ref.location)
            )
            collision = self._detect_internal_variable_collision_in_schema(
                column_names=tuple(getattr(getattr(table, "schema", None), "names", ()) or ()),
                source_name=source_ref.name,
                internal_variable_names=internal_variable_names,
                allow_internal_names=allow_internal_names,
            )
            if collision is not None:
                return collision
        return None

    def _apply_output_stage_options(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        outputs: Mapping[str, DataSetRef],
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        diagnostics: list[Diagnostic] = []
        source_statement = next(
            (statement for statement in ast_statements if statement.kind in {"SET", "MERGE"}),
            None,
        )
        excluded_names = self._collect_internal_output_variable_names(
            ast_statements=ast_statements,
            resolved_inputs=resolved_inputs,
            additional_internal_names=tuple(
                self._rewrite_planner.build_plan(
                    ast_statements=ast_statements,
                    source_statement=source_statement,
                ).helper_runtime_columns
            ),
        )
        if not excluded_names:
            return dict(outputs), diagnostics

        filtered: dict[str, DataSetRef] = {}
        for target, dataset_ref in outputs.items():
            normalized_kind = dataset_ref.kind.strip().lower()

            if normalized_kind == "memory":
                payload = dataset_ref.payload
                if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
                    filtered_rows: list[dict[str, Any]] = []
                    for row in payload:
                        if not isinstance(row, Mapping):
                            diagnostics.append(
                                Diagnostic(
                                    code="RUNTIME_OUTPUT_TARGET_NOT_FOUND",
                                    severity="error",
                                    location=f"dataset:{target}",
                                    message="Output row must be a mapping.",
                                )
                            )
                            return {}, diagnostics
                        filtered_rows.append({k: v for k, v in dict(row).items() if k not in excluded_names})
                    filtered[target] = DataSetRef(
                        kind=dataset_ref.kind,
                        location=dataset_ref.location,
                        payload=filtered_rows,
                        metadata=dataset_ref.metadata,
                    )
                    continue
                filtered[target] = dataset_ref
                continue

            if normalized_kind == "arrow_table" and hasattr(dataset_ref.payload, "to_pylist"):
                try:
                    import pyarrow as pa

                    rows = dataset_ref.payload.to_pylist()
                    filtered_rows = [
                        {k: v for k, v in dict(row).items() if k not in excluded_names}
                        for row in rows
                        if isinstance(row, Mapping)
                    ]
                    all_keys: list[str] = []
                    seen_keys: set[str] = set()
                    for row in filtered_rows:
                        for key in row.keys():
                            if key in seen_keys:
                                continue
                            seen_keys.add(key)
                            all_keys.append(key)
                    normalized_rows = [{key: row.get(key) for key in all_keys} for row in filtered_rows]
                    filtered_table = pa.Table.from_pylist(normalized_rows)
                    filtered[target] = DataSetRef(
                        kind="arrow_table",
                        location=dataset_ref.location,
                        payload=filtered_table,
                        metadata=dataset_ref.metadata,
                    )
                except Exception as error:
                    diagnostics.append(
                        Diagnostic(
                            code="CONVERT_OUTPUT_FAILED",
                            severity="error",
                            location=f"dataset:{target}",
                            message=f"Failed to apply output-stage internal variable filtering: {error}",
                        )
                    )
                    return {}, diagnostics
                continue

            filtered[target] = dataset_ref

        return filtered, diagnostics

    def _apply_output_metadata(
        self,
        *,
        ast_statements: Sequence[Any],
        outputs: Mapping[str, DataSetRef],
        execution_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, DataSetRef]:
        data_statement = next((statement for statement in ast_statements if statement.kind == "DATA"), None)
        label_statement = next((statement for statement in ast_statements if statement.kind == "LABEL"), None)
        if data_statement is None and label_statement is None:
            return dict(outputs)

        column_labels = dict(getattr(label_statement, "label_map", {}))
        projection_spec: Mapping[str, Any] | None = None
        if isinstance(execution_plan, Mapping):
            raw_projection = execution_plan.get("post_projection")
            if isinstance(raw_projection, Mapping):
                projection_spec = raw_projection
        if projection_spec is None:
            keep_statement = next((statement for statement in ast_statements if statement.kind == "KEEP"), None)
            drop_statement = next((statement for statement in ast_statements if statement.kind == "DROP"), None)
            rename_statement = next((statement for statement in ast_statements if statement.kind == "RENAME"), None)
            projection_spec = {
                "keep": tuple(keep_statement.variables) if keep_statement is not None else (),
                "drop": tuple(drop_statement.variables) if drop_statement is not None else (),
                "rename": dict(rename_statement.rename_map) if rename_statement is not None else {},
            }
        column_labels = self._project_column_labels(
            column_labels=column_labels,
            keep_vars=projection_spec.get("keep", ()) if isinstance(projection_spec, Mapping) else (),
            drop_vars=projection_spec.get("drop", ()) if isinstance(projection_spec, Mapping) else (),
            rename_map=projection_spec.get("rename", {}) if isinstance(projection_spec, Mapping) else {},
        )
        dataset_labels: dict[str, str] = {}
        if data_statement is not None:
            for output_ref in getattr(data_statement, "output_refs", ()):
                label = getattr(getattr(output_ref, "options", None), "label", None)
                if label:
                    dataset_labels[self._dataset_name_key(output_ref.name)] = label

        updated: dict[str, DataSetRef] = {}
        for target, dataset_ref in outputs.items():
            metadata = dict(dataset_ref.metadata)
            dataset_label = dataset_labels.get(self._dataset_name_key(target))
            if dataset_label:
                metadata["memlabel"] = dataset_label
            if column_labels:
                metadata["column_labels"] = dict(column_labels)
            updated[target] = DataSetRef(
                kind=dataset_ref.kind,
                location=dataset_ref.location,
                payload=dataset_ref.payload,
                metadata=metadata,
            )

        return updated

    def _project_column_labels(
        self,
        *,
        column_labels: Mapping[str, str],
        keep_vars: Sequence[str],
        drop_vars: Sequence[str],
        rename_map: Mapping[str, str],
    ) -> dict[str, str]:
        if not column_labels:
            return {}

        entries = [(str(name), str(label)) for name, label in column_labels.items()]
        if keep_vars:
            keep_keys = {_column_key(name) for name in keep_vars}
            entries = [(name, label) for name, label in entries if _column_key(name) in keep_keys]
        if drop_vars:
            drop_keys = {_column_key(name) for name in drop_vars}
            entries = [(name, label) for name, label in entries if _column_key(name) not in drop_keys]
        if rename_map:
            normalized_rename = {_column_key(source): str(target) for source, target in rename_map.items()}
            entries = [(normalized_rename.get(_column_key(name), name), label) for name, label in entries]
        return dict(entries)

    def _collect_internal_output_variable_names(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        additional_internal_names: Sequence[str] = (),
    ) -> set[str]:
        source_statement = next(
            (statement for statement in ast_statements if statement.kind in {"SET", "MERGE"}),
            None,
        )

        source_refs = tuple(getattr(source_statement, "dataset_refs", ()) or ()) if source_statement is not None else ()
        in_option_vars = [
            ref.options.in_var
            for ref in source_refs
            if getattr(getattr(ref, "options", None), "in_var", None) is not None
        ]
        by_keys = self._extract_variable_list(ast_statements, "BY")
        statement_options = getattr(source_statement, "statement_options", None)

        excluded = self._collect_internal_variable_names(
            in_option_vars=in_option_vars,
            indsname_var=getattr(statement_options, "indsname_var", None),
            end_var=getattr(statement_options, "end_var", None),
            by_keys=by_keys,
        )

        for input_ref in resolved_inputs.values():
            location = input_ref.location or ""
            if self._PREPARED_INTERNAL_VARS_MARKER not in location:
                continue
            raw = location.split(self._PREPARED_INTERNAL_VARS_MARKER, maxsplit=1)[1]
            raw = raw.split("|", maxsplit=1)[0]
            excluded.update(name for name in raw.split(",") if name)

        rename_statement = next((statement for statement in ast_statements if statement.kind == "RENAME"), None)
        if rename_statement is not None:
            for source, target in rename_statement.rename_map.items():
                if source in excluded:
                    excluded.add(target)

        data_statement = next((statement for statement in ast_statements if statement.kind == "DATA"), None)
        if data_statement is not None:
            for output_ref in getattr(data_statement, "output_refs", ()):
                option_spec = getattr(output_ref, "options", None)
                if option_spec is None:
                    continue
                rename_map = dict(getattr(option_spec, "rename_map", {}))
                for source, target in rename_map.items():
                    if source in excluded:
                        excluded.add(target)

        excluded.update(name for name in additional_internal_names if name)

        return excluded

    def _prepare_runtime_evaluations(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> tuple[tuple[Any, ...], dict[str, DataSetRef], list[Diagnostic]]:
        diagnostics: list[Diagnostic] = []
        statements = tuple(ast_statements)
        current_inputs = dict(resolved_inputs)

        prepared_statements, prepared_inputs, merge_prepare_error = self._prepare_merge_for_execution(
            ast_statements=statements,
            resolved_inputs=current_inputs,
        )
        if merge_prepare_error is not None:
            diagnostics.append(merge_prepare_error)
            return statements, current_inputs, diagnostics
        statements = prepared_statements
        current_inputs = prepared_inputs

        rewritten_inputs, rewrite_error = self._materialize_lag_lead_rewrite_inputs(
            ast_statements=statements,
            resolved_inputs=current_inputs,
        )
        if rewrite_error is not None:
            diagnostics.append(rewrite_error)
            return statements, current_inputs, diagnostics
        if rewritten_inputs is not None:
            current_inputs = rewritten_inputs
            statements = self._rewrite_planner.rewrite_lag_lead_statements(statements)
            statements = self._collapse_rewrite_prepared_set_statement(
                ast_statements=statements,
                resolved_inputs=current_inputs,
            )

        self._warm_expression_evaluation_caches(statements)
        return statements, current_inputs, diagnostics

    def _materialize_lag_lead_rewrite_inputs(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> tuple[dict[str, DataSetRef] | None, Diagnostic | None]:
        lag_lead_specs = self._rewrite_planner.collect_lag_lead_specs(ast_statements)
        if not lag_lead_specs:
            return None, None

        set_statement = next((statement for statement in ast_statements if statement.kind == "SET"), None)
        if set_statement is None:
            return None, None

        source_refs = list(getattr(set_statement, "dataset_refs", ()) or ())
        if not source_refs:
            return None, None
        by_keys = self._extract_variable_list(ast_statements, "BY")

        first_source_name = source_refs[0].name
        source_input = resolved_inputs.get(first_source_name)
        if source_input is None:
            return None, Diagnostic(
                code="RUNTIME_SET_DATASET_NOT_FOUND",
                severity="error",
                message=f"Input dataset is not provided: {first_source_name}",
            )

        source_names = {ref.name for ref in source_refs}
        columnar_prepared_input = self._materialize_lag_lead_arrow_input_if_supported(
            ast_statements=ast_statements,
            source_refs=source_refs,
            by_keys=by_keys,
            source_name=first_source_name,
            source_input=source_input,
            lag_lead_specs=lag_lead_specs,
        )
        if columnar_prepared_input is not None:
            updated_inputs = {
                name: dataset
                for name, dataset in resolved_inputs.items()
                if name not in source_names
            }
            updated_inputs[first_source_name] = columnar_prepared_input
            return updated_inputs, None

        existing_rows: list[dict[str, Any]] = []
        participating_inputs: list[DataSetRef] = []
        source_inputs_missing = [ref.name for ref in source_refs[1:] if ref.name not in resolved_inputs]

        if source_inputs_missing:
            loaded_rows, load_error = self._io_service.load_input_rows(source_input)
            if load_error is not None:
                return None, load_error
            existing_rows = loaded_rows
            participating_inputs = [source_input]
        else:
            for source_ref in source_refs:
                input_ref = resolved_inputs.get(source_ref.name)
                if input_ref is None:
                    return None, Diagnostic(
                        code="RUNTIME_SET_DATASET_NOT_FOUND",
                        severity="error",
                        message=f"Input dataset is not provided: {source_ref.name}",
                    )
                loaded_rows, load_error = self._io_service.load_input_rows(input_ref)
                if load_error is not None:
                    return None, load_error
                option_rows, option_error = self._apply_dataset_reference_options_for_prepare(
                    rows=loaded_rows,
                    source_name=source_ref.name,
                    option_spec=source_ref.options,
                )
                if option_error is not None:
                    return None, option_error
                existing_rows.extend(option_rows)
                participating_inputs.append(input_ref)

        if by_keys and len(existing_rows) > 1:
            existing_rows.sort(
                key=lambda row: self._rewrite_planner.build_row_sort_key(
                    row,
                    by_keys=by_keys,
                    resolve_row_value=self._resolve_row_value,
                )
            )

        if existing_rows and all(
            all(spec["helper_column"] in row for spec in lag_lead_specs)
            for row in existing_rows
        ):
            return dict(resolved_inputs), None

        rewritten_rows = self._rewrite_planner.materialize_lag_lead_columns(
            rows=existing_rows,
            ast_statements=ast_statements,
        )
        prepared_input = self._coerce_rewrite_input(
            source_name=first_source_name,
            source_input=source_input,
            rewritten_rows=rewritten_rows,
            internal_variable_names={spec["helper_column"] for spec in lag_lead_specs},
            fallback_reason="arrow_columnar_not_supported",
        )

        updated_inputs = {
            name: dataset
            for name, dataset in resolved_inputs.items()
            if name not in source_names
        }
        updated_inputs[first_source_name] = prepared_input
        return updated_inputs, None

    def _materialize_lag_lead_arrow_input_if_supported(
        self,
        *,
        ast_statements: Sequence[Any],
        source_refs: Sequence[Any],
        by_keys: Sequence[str],
        source_name: str,
        source_input: DataSetRef,
        lag_lead_specs: Sequence[Mapping[str, Any]],
    ) -> DataSetRef | None:
        if source_input.kind != "arrow_table":
            return None

        location = str(getattr(source_input, "location", "") or "")
        is_prepared_carrier = self._PREPARED_SET_ROWS_MARKER in location
        if not is_prepared_carrier:
            if len(source_refs) != 1:
                return None
            option_spec = getattr(source_refs[0], "options", None)
            if option_spec is not None and (
                getattr(option_spec, "keep_vars", ())
                or getattr(option_spec, "drop_vars", ())
                or getattr(option_spec, "rename_map", {})
                or getattr(option_spec, "where_expr", None)
                or getattr(option_spec, "firstobs", None) is not None
                or getattr(option_spec, "obs", None) is not None
            ):
                return None
        table = source_input.payload
        if not hasattr(table, "schema") or not hasattr(table, "column"):
            return None

        source_owned_specs = tuple(
            spec for spec in lag_lead_specs if self._rewrite_planner._is_source_owned_lag_lead_spec(spec)
        )
        if not source_owned_specs:
            return None

        schema_names = tuple(getattr(table.schema, "names", ()) or ())
        if schema_names and all(spec["helper_column"] in schema_names for spec in source_owned_specs):
            return source_input

        if by_keys:
            resolved_sort_keys: list[tuple[str, str]] = []
            for by_key in by_keys:
                resolved_name = self._resolve_arrow_table_column_name(table, by_key)
                if resolved_name is None:
                    return None
                resolved_sort_keys.append((resolved_name, "ascending"))
            try:
                table = table.sort_by(resolved_sort_keys)
            except Exception:
                return None

        import pyarrow as pa

        materialized_columns: list[tuple[str, Any]] = []
        for spec in source_owned_specs:
            resolved_source_name = self._resolve_arrow_table_column_name(table, spec["source_name"])
            if resolved_source_name is None:
                return None
            values = table.column(resolved_source_name).combine_chunks().to_pylist()
            helper_values: list[Any] = []
            default_value = self._rewrite_planner._parse_default_expression(spec.get("default_expr"))
            for index in range(len(values)):
                target_index = self._rewrite_planner._target_index_for_spec(index=index, spec=spec)
                if 0 <= target_index < len(values):
                    helper_values.append(values[target_index])
                else:
                    helper_values.append(default_value)
            materialized_columns.append((spec["helper_column"], pa.array(helper_values)))

        rewritten_table = table
        for helper_name, helper_array in materialized_columns:
            if helper_name in rewritten_table.schema.names:
                field_index = rewritten_table.schema.get_field_index(helper_name)
                rewritten_table = rewritten_table.set_column(field_index, helper_name, helper_array)
            else:
                rewritten_table = rewritten_table.append_column(helper_name, helper_array)

        return self._coerce_rewrite_arrow_input(
            source_name=source_name,
            source_input=source_input,
            rewritten_table=rewritten_table,
            internal_variable_names={spec["helper_column"] for spec in source_owned_specs},
        )

    @staticmethod
    def _resolve_arrow_table_column_name(table: Any, name: str) -> str | None:
        schema_names = tuple(getattr(getattr(table, "schema", None), "names", ()) or ())
        if name in schema_names:
            return name
        normalized_name = _column_key(name)
        for candidate in schema_names:
            if _column_key(candidate) == normalized_name:
                return candidate
        return None

    def _collapse_rewrite_prepared_set_statement(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> tuple[Any, ...]:
        set_statement = next((statement for statement in ast_statements if statement.kind == "SET"), None)
        if set_statement is None:
            return tuple(ast_statements)

        source_refs = list(getattr(set_statement, "dataset_refs", ()) or ())
        if len(source_refs) <= 1:
            return tuple(ast_statements)

        if all(source_ref.name in resolved_inputs for source_ref in source_refs[1:]):
            return tuple(ast_statements)

        first_source_name = source_refs[0].name
        replacement_statement = ParsedStatement(
            kind="SET",
            text=f"set {first_source_name}",
            dataset_refs=(DatasetReference(name=first_source_name),),
            statement_options=getattr(set_statement, "statement_options", SetStatementOptionSpec()),
        )
        return tuple(
            replacement_statement if statement is set_statement else statement
            for statement in ast_statements
        )

    def _format_internal_variable_suffix(self, internal_variable_names: set[str] | None) -> str:
        internal_names = sorted(name for name in (internal_variable_names or set()) if name)
        if not internal_names:
            return ""
        return f"{self._PREPARED_INTERNAL_VARS_MARKER}{','.join(internal_names)}"

    def _format_rewrite_fallback_suffix(self, fallback_reason: str | None) -> str:
        if not fallback_reason:
            return ""
        return f"{self._REWRITE_FALLBACK_REASON_MARKER}{fallback_reason}"

    def _coerce_rewrite_input(
        self,
        *,
        source_name: str,
        source_input: DataSetRef,
        rewritten_rows: Sequence[Mapping[str, Any]],
        internal_variable_names: set[str] | None = None,
        fallback_reason: str | None = None,
    ) -> DataSetRef:
        normalized_rows = [dict(row) for row in rewritten_rows]
        base_location = source_input.location or f"dataset://{source_name}"
        internal_suffix = self._format_internal_variable_suffix(internal_variable_names)
        fallback_suffix = self._format_rewrite_fallback_suffix(fallback_reason)
        location = f"{base_location}{self._PREPARED_SET_ROWS_MARKER}{internal_suffix}{fallback_suffix}"
        if source_input.kind == "arrow_table":
            return DataSetRef(
                kind="arrow_table",
                location=location,
                payload=self._build_arrow_table_from_rows(normalized_rows),
            )
        return DataSetRef(
            kind="memory",
            location=location,
            payload=normalized_rows,
        )

    def _coerce_rewrite_arrow_input(
        self,
        *,
        source_name: str,
        source_input: DataSetRef,
        rewritten_table: Any,
        internal_variable_names: set[str] | None = None,
    ) -> DataSetRef:
        base_location = source_input.location or f"dataset://{source_name}"
        internal_suffix = self._format_internal_variable_suffix(internal_variable_names)
        location = f"{base_location}{self._PREPARED_SET_ROWS_MARKER}{internal_suffix}"
        return DataSetRef(
            kind="arrow_table",
            location=location,
            payload=rewritten_table,
        )

    def _prepare_merge_for_execution(
        self,
        *,
        ast_statements: tuple[Any, ...],
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> tuple[tuple[Any, ...], dict[str, DataSetRef], Diagnostic | None]:
        merge_statement = next((statement for statement in ast_statements if statement.kind == "MERGE"), None)
        if merge_statement is None:
            return ast_statements, dict(resolved_inputs), None

        source_refs = list(getattr(merge_statement, "dataset_refs", ()) or ())
        if not source_refs:
            return ast_statements, dict(resolved_inputs), None

        by_keys = self._extract_variable_list(ast_statements, "BY")
        in_option_vars = [
            ref.options.in_var
            for ref in source_refs
            if getattr(ref.options, "in_var", None) is not None
        ]
        internal_variable_names = self._collect_internal_variable_names(
            in_option_vars=in_option_vars,
            indsname_var=getattr(getattr(merge_statement, "statement_options", None), "indsname_var", None),
            end_var=getattr(getattr(merge_statement, "statement_options", None), "end_var", None),
            by_keys=by_keys,
        )

        prepared_arrow_input, prepared_arrow_error = self._prepare_merge_arrow_input_if_supported(
            source_refs=source_refs,
            by_keys=by_keys,
            resolved_inputs=resolved_inputs,
            internal_variable_names=internal_variable_names,
        )
        if prepared_arrow_error is not None:
            return ast_statements, dict(resolved_inputs), prepared_arrow_error
        if prepared_arrow_input is not None:
            first_source_name = source_refs[0].name
            updated_inputs = {
                name: dataset
                for name, dataset in resolved_inputs.items()
                if name not in {source_ref.name for source_ref in source_refs}
            }
            updated_inputs[first_source_name] = prepared_arrow_input

            replacement_statement = ParsedStatement(
                kind="SET",
                text=f"set {first_source_name}",
                dataset_refs=(DatasetReference(name=first_source_name),),
                statement_options=getattr(merge_statement, "statement_options", SetStatementOptionSpec()),
            )
            updated_statements = tuple(
                replacement_statement if statement is merge_statement else statement
                for statement in ast_statements
            )
            return updated_statements, updated_inputs, None

        merged_rows_with_source, merge_error = self._rewrite_planner.prepare_merge_rows(
            source_refs=source_refs,
            by_keys=by_keys,
            resolved_inputs=resolved_inputs,
            internal_variable_names=internal_variable_names,
            load_input_rows=self._io_service.load_input_rows,
            apply_dataset_reference_options=self._apply_dataset_reference_options_for_prepare,
            resolve_row_key=self._resolve_row_key,
            resolve_row_value=self._resolve_row_value,
            prepared_merge_marker=self._PREPARED_MERGE_ROWS_MARKER,
        )
        if merge_error is not None:
            return ast_statements, dict(resolved_inputs), merge_error

        merged_rows = [
            {key: value for key, value in dict(row).items() if value is not None}
            for _, row, _ in merged_rows_with_source
        ]
        first_source_name = source_refs[0].name
        prepared_input = self._coerce_prepared_merge_input(
            source_name=first_source_name,
            merged_rows=merged_rows,
            source_inputs=resolved_inputs,
            internal_variable_names=internal_variable_names,
        )

        updated_inputs = {
            name: dataset
            for name, dataset in resolved_inputs.items()
            if name not in {source_ref.name for source_ref in source_refs}
        }
        updated_inputs[first_source_name] = prepared_input

        replacement_statement = ParsedStatement(
            kind="SET",
            text=f"set {first_source_name}",
            dataset_refs=(DatasetReference(name=first_source_name),),
            statement_options=getattr(merge_statement, "statement_options", SetStatementOptionSpec()),
        )
        updated_statements = tuple(
            replacement_statement if statement is merge_statement else statement
            for statement in ast_statements
        )
        return updated_statements, updated_inputs, None

    def _prepare_merge_arrow_input_if_supported(
        self,
        *,
        source_refs: Sequence[Any],
        by_keys: Sequence[str],
        resolved_inputs: Mapping[str, DataSetRef],
        internal_variable_names: set[str],
    ) -> tuple[DataSetRef | None, Diagnostic | None]:
        import pyarrow as pa

        if not by_keys:
            return None, None

        normalized_by_keys = {_column_key(key) for key in by_keys}
        source_infos: list[dict[str, Any]] = []
        canonical_by_names: dict[str, str] = {}
        non_key_columns_by_source: list[set[str]] = []

        for source_ref in source_refs:
            input_ref = resolved_inputs.get(source_ref.name)
            if input_ref is None:
                return None, None
            if input_ref.kind != "arrow_table" or not isinstance(input_ref.payload, pa.Table):
                return None, None

            table, option_error = try_apply_dataset_reference_options_to_arrow_table(
                input_ref.payload,
                source_name=source_ref.name,
                option_spec=source_ref.options,
            )
            if option_error is not None:
                return None, option_error
            if table is None:
                return None, None

            allow_internal_names = (
                isinstance(getattr(input_ref, "location", None), str)
                and self._PREPARED_MERGE_ROWS_MARKER in str(input_ref.location)
            )
            collision = self._detect_internal_variable_collision_in_schema(
                column_names=tuple(getattr(getattr(table, "schema", None), "names", ()) or ()),
                source_name=source_ref.name,
                internal_variable_names=internal_variable_names,
                allow_internal_names=allow_internal_names,
            )
            if collision is not None:
                return None, collision

            resolved_by_names: list[str] = []
            for by_key in by_keys:
                resolved_name = self._resolve_arrow_table_column_name(table, by_key)
                if resolved_name is None:
                    return None, self._rewrite_planner._build_by_precondition_failed(by_key, source_ref.name)
                canonical_by_names.setdefault(_column_key(by_key), resolved_name)
                resolved_by_names.append(resolved_name)

            column_values = table.to_pydict()
            key_to_row_index: dict[tuple[Any, ...], int] = {}
            for row_index in range(int(getattr(table, "num_rows", 0) or 0)):
                key = tuple(column_values[name][row_index] for name in resolved_by_names)
                if key in key_to_row_index:
                    return None, None
                key_to_row_index[key] = row_index

            non_key_columns = {
                _column_key(column_name)
                for column_name in getattr(table, "column_names", ())
                if _column_key(column_name) not in normalized_by_keys
            }
            non_key_columns_by_source.append(non_key_columns)
            source_infos.append(
                {
                    "source_name": source_ref.name,
                    "table": table,
                    "column_values": column_values,
                    "key_to_row_index": key_to_row_index,
                }
            )

        duplicate_columns: set[str] = set()
        for left_index in range(len(non_key_columns_by_source)):
            for right_index in range(left_index + 1, len(non_key_columns_by_source)):
                duplicate_columns.update(
                    non_key_columns_by_source[left_index].intersection(non_key_columns_by_source[right_index])
                )
        if duplicate_columns:
            duplicate_label = ", ".join(sorted(duplicate_columns))
            return None, self._rewrite_planner._build_merge_duplicate_column(duplicate_label)

        merge_key_order = sorted(
            {key for info in source_infos for key in info["key_to_row_index"].keys()},
            key=lambda item: tuple(self._rewrite_planner._sortable_value(value) for value in item),
        )
        output_by_keys = tuple(canonical_by_names.get(_column_key(key), key) for key in by_keys)

        output_columns: list[tuple[str, list[Any]]] = []
        for index, output_name in enumerate(output_by_keys):
            output_columns.append((output_name, [key[index] for key in merge_key_order]))

        for info in source_infos:
            for column_name in getattr(info["table"], "column_names", ()):
                if _column_key(column_name) in normalized_by_keys:
                    continue
                values: list[Any] = []
                column_data = info["column_values"][column_name]
                for key in merge_key_order:
                    row_index = info["key_to_row_index"].get(key)
                    values.append(None if row_index is None else column_data[row_index])
                output_columns.append((column_name, values))

        output_table = pa.table({name: values for name, values in output_columns})
        output_table, by_error = self._annotate_arrow_by_group_flags(output_table, by_keys=by_keys)
        if by_error is not None:
            return None, by_error

        first_source_name = source_refs[0].name
        first_source_input = resolved_inputs.get(first_source_name)
        if first_source_input is None:
            return None, None
        return DataSetRef(
            kind="arrow_table",
            location=f"{first_source_input.location}{self._PREPARED_MERGE_ROWS_MARKER}{self._format_internal_variable_suffix(internal_variable_names)}",
            payload=output_table,
        ), None

    def _coerce_prepared_merge_input(
        self,
        *,
        source_name: str,
        merged_rows: Sequence[Mapping[str, Any]],
        source_inputs: Mapping[str, DataSetRef],
        internal_variable_names: set[str] | None = None,
    ) -> DataSetRef:
        normalized_rows = [dict(row) for row in merged_rows]
        source_input = source_inputs.get(source_name)
        source_location = source_input.location if source_input is not None else f"dataset://{source_name}"
        internal_names = sorted(name for name in (internal_variable_names or set()) if name)
        internal_suffix = ""
        if internal_names:
            internal_suffix = f"{self._PREPARED_INTERNAL_VARS_MARKER}{','.join(internal_names)}"

        if source_input is not None and source_input.kind == "arrow_table":
            return DataSetRef(
                kind="arrow_table",
                location=f"{source_location}{self._PREPARED_MERGE_ROWS_MARKER}{internal_suffix}",
                payload=self._build_arrow_table_from_rows(normalized_rows),
            )

        return DataSetRef(
            kind="memory",
            location=f"{source_location}{self._PREPARED_MERGE_ROWS_MARKER}{internal_suffix}",
            payload=normalized_rows,
        )

    def _warm_expression_evaluation_caches(self, ast_statements: Sequence[Any]) -> None:
        for statement in ast_statements:
            statement_text = getattr(statement, "text", "")
            if not isinstance(statement_text, str):
                continue

            if statement.kind == "WHERE":
                expression = statement_text[len("where"):].strip() if statement_text.lower().startswith("where") else ""
                if expression:
                    self._runtime.prepare_condition_expression(expression)
                continue

            if statement.kind == "IF":
                lowered = statement_text.lower()
                then_index = lowered.find(" then ")
                if then_index > 0:
                    condition = statement_text[len("if "):then_index].strip()
                    if condition:
                        self._runtime.prepare_condition_expression(condition)
                continue

            if statement.kind == "ASSIGN":
                matched = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*(?:\s*[\(\[\{].*?[\)\]\}])?)\s*=\s*(.+?)\s*$", statement_text)
                if matched is not None:
                    self._evaluator.prepare_expression(matched.group(2).strip())
                continue

            if statement.kind == "SUM":
                matched = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*\+\s*(.+?)\s*$", statement_text)
                if matched is not None:
                    self._evaluator.prepare_expression(matched.group(2).strip())

    def _apply_dataset_reference_options_for_prepare(
        self,
        *,
        rows: list[dict[str, Any]],
        source_name: str,
        option_spec: Any,
    ) -> tuple[list[dict[str, Any]], Diagnostic | None]:
        return apply_dataset_reference_options_to_rows(
            rows,
            source_name=source_name,
            option_spec=option_spec,
        )

    @staticmethod
    def _build_arrow_table_from_rows(rows: Sequence[Mapping[str, Any]]):
        import pyarrow as pa

        normalized_rows = [dict(row) for row in rows]
        if not normalized_rows:
            return pa.Table.from_pylist([])

        all_keys: list[str] = []
        seen_keys: set[str] = set()
        for row in normalized_rows:
            for key in row.keys():
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_keys.append(key)

        stabilized_rows = [
            {key: row.get(key) for key in all_keys}
            for row in normalized_rows
        ]
        return pa.Table.from_pylist(stabilized_rows)

    def _try_interleave_set_inputs(
        self,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        by_keys: tuple[str, ...],
    ) -> dict[str, DataSetRef] | None:
        """Interleave multiple SET sources via Arrow concat + sort when possible.

        """
        import pyarrow as pa

        set_stmt = next((s for s in ast_statements if s.kind == "SET"), None)
        if set_stmt is None:
            return None
        source_refs = getattr(set_stmt, "dataset_refs", ())
        if len(source_refs) <= 1:
            return None
        # IN= requires per-row source identification; leave to Python backend
        if any(getattr(getattr(r, "options", None), "in_var", None) is not None for r in source_refs):
            return None
        # Per-source dataset options must be applied before any concatenation shortcut.
        if any(
            getattr(getattr(r, "options", None), "keep_vars", ())
            or getattr(getattr(r, "options", None), "drop_vars", ())
            or getattr(getattr(r, "options", None), "where_expr", None) is not None
            or getattr(getattr(r, "options", None), "rename_map", {})
            or getattr(getattr(r, "options", None), "firstobs", None) is not None
            or getattr(getattr(r, "options", None), "obs", None) is not None
            for r in source_refs
        ):
            return None

        tables: list[pa.Table] = []
        for ref in source_refs:
            dr = resolved_inputs.get(ref.name)
            if dr is None or dr.kind != "arrow_table" or not isinstance(dr.payload, pa.Table):
                return None  # non-Arrow inputs: delegate to Python backend row-level interleave
            tables.append(dr.payload)

        try:
            combined = pa.concat_tables(tables, promote_options="default")
            sort_keys = [(k, "ascending") for k in by_keys if k in combined.schema.names]
            if sort_keys:
                combined = combined.sort_by(sort_keys)
        except Exception:
            return None

        first_name = source_refs[0].name
        has_repeated_source_names = len({_column_key(ref.name) for ref in source_refs}) < len(source_refs)
        set_names = {r.name for r in source_refs}
        result: dict[str, DataSetRef] = {}
        for name, dr in resolved_inputs.items():
            if name in set_names:
                if name == first_name:
                    location = dr.location
                    if has_repeated_source_names and isinstance(location, str):
                        location = f"{location}{self._PREPARED_SET_ROWS_MARKER}"
                    result[name] = DataSetRef(
                        kind="arrow_table",
                        location=location,
                        payload=combined,
                    )
                # other SET sources omitted; Rust skips missing stream names
            else:
                result[name] = dr
        return result

    def _prepare_set_rows_with_temporary_variables(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        by_keys: tuple[str, ...],
        has_lag_lead: bool,
    ) -> tuple[dict[str, DataSetRef] | None, Diagnostic | None]:
        set_stmt = next((s for s in ast_statements if s.kind == "SET"), None)
        if set_stmt is None:
            return None, None
        if any(s.kind == "MERGE" for s in ast_statements):
            return None, None

        source_refs = list(getattr(set_stmt, "dataset_refs", ()) or ())
        if not source_refs:
            return None, None

        in_option_vars = [
            ref.options.in_var
            for ref in source_refs
            if getattr(ref.options, "in_var", None) is not None
        ]
        indsname_var = set_stmt.statement_options.indsname_var
        end_var = set_stmt.statement_options.end_var
        has_position_window_options = any(
            getattr(ref.options, "firstobs", None) is not None
            or getattr(ref.options, "obs", None) is not None
            for ref in source_refs
        )
        needs_prepare = bool(
            in_option_vars
            or indsname_var
            or end_var
            or by_keys
            or has_lag_lead
            or (
                BackendDispatchPolicy.prefers_rust_first_execution(self._runtime_backend_preference)
                and has_position_window_options
            )
        )
        if not needs_prepare:
            return None, None

        internal_names = self._collect_internal_variable_names(
            in_option_vars=in_option_vars,
            indsname_var=indsname_var,
            end_var=end_var,
            by_keys=by_keys,
        )
        prepared_arrow_inputs, prepared_arrow_error = self._prepare_set_arrow_inputs_if_supported(
            source_refs=source_refs,
            by_keys=by_keys,
            in_option_vars=in_option_vars,
            indsname_var=indsname_var,
            end_var=end_var,
            resolved_inputs=resolved_inputs,
            internal_variable_names=internal_names,
        )
        if prepared_arrow_error is not None:
            return None, prepared_arrow_error
        if prepared_arrow_inputs is not None:
            return prepared_arrow_inputs, None

        prepared_rows, prepare_error = self._rewrite_planner.prepare_set_rows(
            source_refs=source_refs,
            by_keys=by_keys,
            in_option_vars=in_option_vars,
            indsname_var=indsname_var,
            end_var=end_var,
            resolved_inputs=resolved_inputs,
            internal_variable_names=internal_names,
            load_input_rows=self._io_service.load_input_rows,
            apply_dataset_reference_options=self._apply_dataset_reference_options_for_prepare,
            resolve_row_key=self._resolve_row_key,
            resolve_row_value=self._resolve_row_value,
            prepared_merge_marker=self._PREPARED_MERGE_ROWS_MARKER,
        )
        if prepare_error is not None:
            return None, prepare_error

        first_source = source_refs[0].name
        first_input = resolved_inputs.get(first_source)
        if first_input is None:
            return None, None

        prepared_ref = self._coerce_rewrite_input(
            source_name=first_source,
            source_input=first_input,
            rewritten_rows=prepared_rows,
            internal_variable_names=internal_names,
        )

        source_names = {ref.name for ref in source_refs}
        updated_inputs: dict[str, DataSetRef] = {}
        for name, dataset in resolved_inputs.items():
            if name in source_names:
                if name == first_source:
                    updated_inputs[name] = prepared_ref
                continue
            updated_inputs[name] = dataset
        return updated_inputs, None

    def _prepare_set_arrow_inputs_if_supported(
        self,
        *,
        source_refs: Sequence[Any],
        by_keys: Sequence[str],
        in_option_vars: Sequence[str],
        indsname_var: str | None,
        end_var: str | None,
        resolved_inputs: Mapping[str, DataSetRef],
        internal_variable_names: set[str],
    ) -> tuple[dict[str, DataSetRef] | None, Diagnostic | None]:
        import pyarrow as pa

        prepared_tables: list[Any] = []
        source_names = {ref.name for ref in source_refs}
        first_source_name = source_refs[0].name
        first_source_input = resolved_inputs.get(first_source_name)
        if first_source_input is None:
            return None, None

        for source_order, source_ref in enumerate(source_refs):
            input_ref = resolved_inputs.get(source_ref.name)
            if input_ref is None:
                return None, None
            if input_ref.kind != "arrow_table" or not isinstance(input_ref.payload, pa.Table):
                return None, None

            table, option_error = try_apply_dataset_reference_options_to_arrow_table(
                input_ref.payload,
                source_name=source_ref.name,
                option_spec=source_ref.options,
            )
            if option_error is not None:
                return None, option_error
            if table is None:
                return None, None

            allow_internal_names = (
                isinstance(getattr(input_ref, "location", None), str)
                and self._PREPARED_MERGE_ROWS_MARKER in str(input_ref.location)
            )
            collision = self._detect_internal_variable_collision_in_schema(
                column_names=tuple(getattr(getattr(table, "schema", None), "names", ()) or ()),
                source_name=source_ref.name,
                internal_variable_names=internal_variable_names,
                allow_internal_names=allow_internal_names,
            )
            if collision is not None:
                return None, collision

            row_count = int(getattr(table, "num_rows", 0) or 0)
            table = table.append_column("__prepare_source_order__", pa.array([source_order] * row_count, type=pa.int64()))
            table = table.append_column("__prepare_row_order__", pa.array(list(range(row_count)), type=pa.int64()))

            for in_var in in_option_vars:
                values = [1 if getattr(source_ref.options, "in_var", None) == in_var else 0] * row_count
                table = self._set_arrow_column(table, in_var, pa.array(values, type=pa.int8()))
            if indsname_var:
                table = self._set_arrow_column(table, indsname_var, pa.array([source_ref.name] * row_count))

            prepared_tables.append(table)

        try:
            combined = pa.concat_tables(prepared_tables, promote_options="default")
            if by_keys:
                sort_keys: list[tuple[str, str]] = []
                for by_key in by_keys:
                    resolved_name = self._resolve_arrow_table_column_name(combined, by_key)
                    if resolved_name is None:
                        return None, self._rewrite_planner._build_by_rows_precondition_failed(by_key)
                    sort_keys.append((resolved_name, "ascending"))
                sort_keys.extend(
                    [
                        ("__prepare_source_order__", "ascending"),
                        ("__prepare_row_order__", "ascending"),
                    ]
                )
                combined = combined.sort_by(sort_keys)
        except Exception:
            return None, None

        if end_var:
            row_count = int(getattr(combined, "num_rows", 0) or 0)
            end_values = [0] * row_count
            if end_values:
                end_values[-1] = 1
            combined = self._set_arrow_column(combined, end_var, pa.array(end_values, type=pa.int8()))

        combined, by_error = self._annotate_arrow_by_group_flags(combined, by_keys=by_keys)
        if by_error is not None:
            return None, by_error

        visible_names = [
            name
            for name in getattr(combined, "column_names", ())
            if name not in {"__prepare_source_order__", "__prepare_row_order__"}
        ]
        combined = combined.select(visible_names)

        updated_inputs = {
            name: dataset
            for name, dataset in resolved_inputs.items()
            if name not in source_names
        }
        updated_inputs[first_source_name] = self._coerce_rewrite_arrow_input(
            source_name=first_source_name,
            source_input=first_source_input,
            rewritten_table=combined,
            internal_variable_names=internal_variable_names,
        )
        return updated_inputs, None

    @staticmethod
    def _set_arrow_column(table: Any, name: str, values: Any) -> Any:
        field_index = table.schema.get_field_index(name) if hasattr(table, "schema") else -1
        if field_index >= 0:
            return table.set_column(field_index, name, values)
        return table.append_column(name, values)

    def _annotate_arrow_by_group_flags(
        self,
        table: Any,
        *,
        by_keys: Sequence[str],
    ) -> tuple[Any, Diagnostic | None]:
        import pyarrow as pa

        if not by_keys or int(getattr(table, "num_rows", 0) or 0) == 0:
            return table, None

        value_columns: list[list[Any]] = []
        for by_key in by_keys:
            resolved_name = self._resolve_arrow_table_column_name(table, by_key)
            if resolved_name is None:
                return table, self._rewrite_planner._build_by_rows_precondition_failed(by_key)
            value_columns.append(table.column(resolved_name).combine_chunks().to_pylist())

        for by_key, values in zip(by_keys, value_columns):
            first_values: list[int] = []
            last_values: list[int] = []
            for index, current_value in enumerate(values):
                previous_value = values[index - 1] if index > 0 else object()
                next_value = values[index + 1] if index < len(values) - 1 else object()
                first_values.append(1 if current_value != previous_value else 0)
                last_values.append(1 if current_value != next_value else 0)
            table = self._set_arrow_column(table, f"FIRST.{by_key}", pa.array(first_values, type=pa.int8()))
            table = self._set_arrow_column(table, f"LAST.{by_key}", pa.array(last_values, type=pa.int8()))
            table = self._set_arrow_column(table, f"first.{by_key}", pa.array(first_values, type=pa.int8()))
            table = self._set_arrow_column(table, f"last.{by_key}", pa.array(last_values, type=pa.int8()))

        return table, None

    def _detect_internal_variable_collision_in_schema(
        self,
        *,
        column_names: Sequence[str],
        source_name: str,
        internal_variable_names: set[str],
        allow_internal_names: bool,
    ) -> Diagnostic | None:
        if allow_internal_names or not internal_variable_names:
            return None
        normalized_internal_names = {_column_key(name) for name in internal_variable_names}
        collided_names = sorted(
            (name for name in column_names if _column_key(name) in normalized_internal_names),
            key=_column_key,
        )
        if not collided_names:
            return None
        return self._rewrite_planner._build_internal_var_collision(collided_names[0], source_name)

    def _apply_dataset_reference_options_for_prepare(
        self,
        *,
        rows: list[dict[str, Any]],
        source_name: str,
        option_spec: Any,
    ) -> tuple[list[dict[str, Any]], Diagnostic | None]:
        if (
            not option_spec.keep_vars
            and not option_spec.drop_vars
            and not option_spec.rename_map
            and not option_spec.where_expr
            and getattr(option_spec, "firstobs", None) is None
            and getattr(option_spec, "obs", None) is None
        ):
            return list(rows), None

        processed: list[dict[str, Any]] = []

        if option_spec.rename_map and len({_column_key(value) for value in option_spec.rename_map.values()}) != len(option_spec.rename_map):
            return [], Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message=f"Dataset option RENAME= has duplicate target names for source '{source_name}'.",
            )

        for row in rows:
            working = dict(row)

            if option_spec.keep_vars:
                keep_set = {_column_key(name) for name in option_spec.keep_vars}
                working = {
                    name: value
                    for name, value in working.items()
                    if _column_key(name) in keep_set
                }

            if option_spec.drop_vars:
                drop_set = {_column_key(name) for name in option_spec.drop_vars}
                working = {
                    name: value
                    for name, value in working.items()
                    if _column_key(name) not in drop_set
                }

            if option_spec.where_expr:
                try:
                    passes = bool(
                        eval(
                            option_spec.where_expr,
                            {"__builtins__": {}},
                            self._build_case_insensitive_scope(working),
                        )
                    )
                except Exception as error:
                    return [], Diagnostic(
                        code="RUNTIME_DATASET_OPTION_INVALID",
                        severity="error",
                        message=(
                            f"Dataset option WHERE= evaluation failed for source '{source_name}': {error}"
                        ),
                    )
                if not passes:
                    continue

            if option_spec.rename_map:
                resolved_rename_map: dict[str, str] = {}
                for old_name, new_name in option_spec.rename_map.items():
                    resolved_old_name = self._resolve_row_key(working, old_name)
                    if resolved_old_name is None:
                        return [], Diagnostic(
                            code="RUNTIME_DATASET_OPTION_INVALID",
                            severity="error",
                            message=(
                                f"Dataset option RENAME= references unknown variable '{old_name}' "
                                f"for source '{source_name}'."
                            ),
                        )
                    resolved_rename_map[resolved_old_name] = new_name
                renamed_row: dict[str, Any] = {}
                for key, value in working.items():
                    renamed_row[resolved_rename_map.get(key, key)] = value
                working = renamed_row

            processed.append(working)

        firstobs = getattr(option_spec, "firstobs", None)
        obs = getattr(option_spec, "obs", None)
        if firstobs is not None and firstobs <= 0:
            return [], Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message=f"Dataset option FIRSTOBS= must be positive for source '{source_name}'.",
            )
        if obs is not None and obs < 0:
            return [], Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message=f"Dataset option OBS= must be non-negative for source '{source_name}'.",
            )
        start_index = max((firstobs or 1) - 1, 0)
        processed = processed[start_index:]
        if obs is not None:
            processed = processed[:obs]

        return processed, None

    def _collect_internal_variable_names(
        self,
        *,
        in_option_vars: Sequence[str],
        indsname_var: str | None,
        end_var: str | None,
        by_keys: Sequence[str],
    ) -> set[str]:
        names = {name for name in in_option_vars if name}
        if indsname_var:
            names.add(indsname_var)
        if end_var:
            names.add(end_var)
        for key in by_keys:
            names.add(f"FIRST.{key}")
            names.add(f"LAST.{key}")
            names.add(f"first.{key}")
            names.add(f"last.{key}")
        return names

    def _uses_lag_lead_functions(self, ast_statements: Sequence[Any]) -> bool:
        return self._rewrite_planner.uses_lag_lead_functions(ast_statements)

    def _sort_inputs_by_by_keys(
        self,
        resolved_inputs: Mapping[str, DataSetRef],
        by_keys: tuple[str, ...],
    ) -> dict[str, DataSetRef]:

        import pyarrow as pa  # type: ignore

        sorted_inputs: dict[str, DataSetRef] = {}
        for name, ref in resolved_inputs.items():
            if ref.kind == "arrow_table" and isinstance(ref.payload, pa.Table):
                # Sort in Arrow space – preserves kind so Rust backend is unaffected.
                table: pa.Table = ref.payload
                active_keys = []
                for by_key in by_keys:
                    resolved_key = next(
                        (column_name for column_name in table.column_names if _column_key(column_name) == _column_key(by_key)),
                        None,
                    )
                    if resolved_key is not None:
                        active_keys.append((resolved_key, "ascending"))
                if not active_keys:
                    sorted_inputs[name] = ref
                    continue
                sorted_table = table.sort_by(active_keys)
                sorted_inputs[name] = DataSetRef(
                    kind="arrow_table",
                    location=ref.location,
                    payload=sorted_table,
                )
            else:
                # For memory / other kinds, fall back to row-based sort.
                rows, error = self._io_service.load_input_rows(ref)
                if error is not None or not rows:
                    sorted_inputs[name] = ref
                    continue
                active_keys_str = [k for k in by_keys if self._resolve_row_key(rows[0], k) is not None]
                if not active_keys_str:
                    sorted_inputs[name] = ref
                    continue

                def _sort_key(row: dict[str, Any], keys: list[str] = active_keys_str) -> tuple:
                    return self._rewrite_planner.build_row_sort_key(
                        row,
                        by_keys=tuple(keys),
                        resolve_row_value=self._resolve_row_value,
                    )

                rows.sort(key=_sort_key)
                sorted_inputs[name] = DataSetRef(
                    kind="memory",
                    location=ref.location,
                    payload=rows,
                )
        return sorted_inputs


    def _extract_variable_list(self, ast_statements: Sequence[Any], kind: str) -> tuple[str, ...]:
        statement = next((item for item in ast_statements if item.kind == kind), None)
        if statement is None:
            return ()
        keyword = kind.lower()
        variables = statement.text[len(keyword):].strip().split()
        return tuple(name for name in variables if name)


    def _extract_data_targets(self, ast_statements: Sequence[Any]) -> tuple[str, ...]:
        return self._io_service.extract_data_targets(ast_statements)

    def _resolve_output_targets(
        self,
        ast_statements: Sequence[Any],
        explicit_output_targets: Sequence[str],
    ) -> tuple[tuple[str, ...], list[Diagnostic]]:
        return self._io_service.resolve_output_targets(
            ast_statements=ast_statements,
            explicit_output_targets=explicit_output_targets,
        )

    def _merge_unique_targets(self, *target_groups: Sequence[str]) -> tuple[str, ...]:
        return self._io_service.merge_unique_targets(*target_groups)

    def _normalize_dataset_name(self, name: str) -> str:
        return self._io_service.normalize_dataset_name(name)

    def _dataset_name_key(self, name: str) -> str:
        return self._io_service.dataset_name_key(name)

    def _resolve_dataset_alias(
        self,
        datasets: Mapping[str, DataSetRef],
        requested_name: str,
    ) -> DataSetRef | None:
        return self._io_service.resolve_dataset_alias(datasets, requested_name)

    def _extract_source_input_names(self, ast_statements: Sequence[Any]) -> list[str]:
        source_statement = next(
            (statement for statement in ast_statements if statement.kind in {"SET", "MERGE"}),
            None,
        )
        if source_statement is None:
            return []

        input_names = [ref.name for ref in getattr(source_statement, "dataset_refs", ()) if getattr(ref, "name", "")]
        if input_names:
            return input_names

        source_tokens = getattr(source_statement, "text", "").split()
        if len(source_tokens) >= 2:
            return [source_tokens[1]]
        return []

    def _resolve_synthetic_inputs_for_names(
        self,
        *,
        requested_names: Sequence[str],
        explicit_inputs: Mapping[str, DataSetRef],
        available_inputs: Mapping[str, DataSetRef] | None,
    ) -> dict[str, DataSetRef]:
        if not requested_names:
            return {}
        return self._io_service.resolve_synthetic_inputs_for_names(
            requested_names=requested_names,
            explicit_inputs=explicit_inputs,
            available_inputs=available_inputs,
            registered_tables=self._registered_tables,
            cache=self._dictionary_input_cache,
        )


    def _split_data_step_blocks(self, dsl_text: str) -> tuple[str, ...]:
        return self._block_splitter.split(dsl_text)

    def _with_block_location(
        self,
        diagnostics: tuple[Diagnostic, ...],
        block_index: int,
        include_block_location: bool,
    ) -> tuple[Diagnostic, ...]:
        if not include_block_location:
            return diagnostics

        updated: list[Diagnostic] = []
        for diagnostic in diagnostics:
            if diagnostic.location:
                location = f"block:{block_index},{diagnostic.location}"
            else:
                location = f"block:{block_index}"
            updated.append(
                Diagnostic(
                    code=diagnostic.code,
                    severity=diagnostic.severity,
                    message=diagnostic.message,
                    location=location,
                    stage=diagnostic.stage,
                    span=diagnostic.span,
                    labels=diagnostic.labels,
                    notes=diagnostic.notes,
                    source_text=diagnostic.source_text,
                )
            )
        return tuple(updated)

    def _coerce_dataset_ref(self, name: str, dataset: Any) -> DataSetRef:
        if isinstance(dataset, DataSetRef):
            return dataset

        if isinstance(dataset, Sequence) and not isinstance(dataset, (str, bytes, bytearray)):
            if all(isinstance(item, Mapping) for item in dataset):
                return DataSetRef(kind="memory", location=f"dataset://{name}", payload=list(dataset))

        if hasattr(dataset, "to_pylist"):
            return DataSetRef(kind=_STANDARD_TRANSPORT_KIND, location=f"dataset://{name}", payload=dataset)

        raise ValueError(f"Unsupported dataset type for register_tables: {name}")


    def list_supported_formats(self) -> tuple[str, ...]:
        return _SUPPORTED_FORMATS

    def get_runtime_requirements(self) -> RuntimeRequirements:
        return RuntimeRequirements(required_python=">=3.10", supported_os=("windows", "linux", "macos"))

    def check_format_support(self, format_name: str) -> FormatSupportResult:
        normalized = format_name.strip().lower()
        if normalized in _SUPPORTED_FORMATS:
            return FormatSupportResult(supported=True, reason_code="", message="")
        return FormatSupportResult(
            supported=False,
            reason_code="CAP_UNSUPPORTED_FORMAT",
            message=f"Unsupported format: {format_name}",
        )
