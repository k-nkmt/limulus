import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ._naming import _column_key
from .backends import (
    PythonRuntimeBackend,
    RustNativeBlockExecutor,
    RuntimeBackendSelector,
    RuntimeExecutionContext,
    RustArrowIOBridge,
    RustRuntimeBackend,
)
from .block_splitter import DataStepBlockSplitter
from .evaluator import ExpressionEvaluator
from .format_registry import FormatRegistry
from .io_adapters import (
    DataFrameAdapterPandas,
    DataInputAdapterArrow,
    DataOutputAdapterArrow,
)
from .io import ExecutorIOService
from .models import CompatibilityNotice, DataSetRef, Diagnostic, DiagnosticLabel, ExecuteRequest, ExecuteResponse, OutputConversionResult
from .parser import (
    DatasetReference,
    ParserBackendSelector,
    ParserExecutionContext,
    ParsedStatement,
    ParserService,
    PythonParserBackend,
    RustNativeParserBackend,
    SetStatementOptionSpec,
    SplitStageParserService,
)
from .runtime import PDVRuntimeService, ProgramExecutionService
from .executor_python import PythonBackendExecutionService


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


class MacroHook(Protocol):
    def evaluate(self, dsl_text: str) -> str:
        ...


class NoOpMacroHook:
    def evaluate(self, dsl_text: str) -> str:
        return dsl_text


class UnsupportedSyntaxMacroHook:
    def __init__(self, parser_service: SplitStageParserService | None = None) -> None:
        self._parser_service = parser_service or SplitStageParserService()

    def evaluate(self, dsl_text: str) -> str:
        regions = self._parser_service.extract_statement_regions(dsl_text)
        if regions is None:
            return dsl_text

        ranges = [self._expand_skip_region(dsl_text, region.start, region.end) for region in regions if region.kind == "SKIP"]
        if not ranges:
            return dsl_text

        masked = list(dsl_text)
        for start, end in ranges:
            for index in range(start, end):
                if masked[index] not in {"\n", "\r"}:
                    masked[index] = " "
        return "".join(masked)

    @staticmethod
    def _expand_skip_region(dsl_text: str, start: int, end: int) -> tuple[int, int]:
        if end < len(dsl_text) and dsl_text[end] == ";":
            return start, end + 1
        return start, end


class ExecutionPipelineCoordinator:
    _STAGE_SPLIT = "split blocks"
    _STAGE_MACRO = "macro hook"
    _STAGE_PARSE = "parse"
    _STAGE_VALIDATE = "validate"
    _STAGE_INPUT_OPTION = "resolve inputs"
    _STAGE_INPUT_PREPARE = "pre-processing inputs"
    _STAGE_EVAL_PREPARE = "pre-evaluations"
    _STAGE_PLAN = "plan generation"
    _STAGE_EXECUTE = "execute"
    _STAGE_OUTPUT_OPTION = "resolve outputs"

    def __init__(self, executor: "DataStepExecutor", macro_hook: MacroHook | None = None) -> None:
        self._executor = executor
        self._macro_hook: MacroHook = macro_hook or NoOpMacroHook()

    def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        request_diagnostics = self._validate_request(request)
        if request_diagnostics:
            return ExecuteResponse(diagnostics=self._tag_stage(request_diagnostics, self._STAGE_SPLIT))

        transformed_dsl, macro_diagnostics = self._apply_macro_hook(request.dsl_text)
        if macro_diagnostics:
            return ExecuteResponse(diagnostics=self._tag_stage(macro_diagnostics, self._STAGE_MACRO))

        split_blocks = self._executor._split_data_step_blocks(transformed_dsl)
        if not split_blocks:
            return ExecuteResponse(diagnostics=tuple())

        parser_backend = self._executor._parser_backend_selector.select(self._executor._parser_backend_preference)
        self._executor._last_parser_backend = parser_backend.name

        parsed_blocks: list[tuple[str, Any]] = []
        include_block_location = len(split_blocks) > 1
        for block_index, block_dsl in enumerate(split_blocks, start=1):
            parse_result = parser_backend.parse(ParserExecutionContext(dsl_text=block_dsl))
            if parse_result.has_errors:
                if parser_backend.name != "python":
                    fallback = self._executor._parser_backend_selector.select("python")
                    self._executor._last_parser_backend = fallback.name
                    parse_result = fallback.parse(ParserExecutionContext(dsl_text=block_dsl))
                if parse_result.has_errors:
                    return ExecuteResponse(
                        diagnostics=self._tag_stage(
                            self._executor._with_block_location(
                                diagnostics=parse_result.diagnostics,
                                block_index=block_index,
                                include_block_location=include_block_location,
                            ),
                            self._STAGE_PARSE,
                        )
                    )
            parsed_blocks.append((block_dsl, parse_result.ast.statements))

        required_input_names_by_block: list[set[str]] = []
        for _, ast_statements in parsed_blocks:
            source_statement = next(
                (statement for statement in ast_statements if statement.kind in {"SET", "MERGE"}),
                None,
            )
            names: set[str] = set()
            if source_statement is not None:
                names.update(self._executor._dataset_name_key(ref.name) for ref in source_statement.dataset_refs)
            required_input_names_by_block.append(names)

        future_required_after_block: list[set[str]] = [set() for _ in parsed_blocks]
        needed_future: set[str] = set()
        for index in range(len(parsed_blocks) - 1, -1, -1):
            future_required_after_block[index] = set(needed_future)
            needed_future.update(required_input_names_by_block[index])

        explicit_inputs = request.inputs or {}
        combined_outputs: dict[str, DataSetRef] = {}
        combined_notices: list[Any] = []
        available_inputs: dict[str, DataSetRef] = dict(self._executor._generated_tables)
        explicit_input_names = {self._executor._dataset_name_key(name) for name in explicit_inputs.keys()}

        for block_index, (block_dsl, ast_statements) in enumerate(parsed_blocks, start=1):
            validation_diagnostics = self._executor._validate_ast_block(
                ast_statements=ast_statements,
                dsl_text=block_dsl,
                explicit_inputs=explicit_inputs,
                available_inputs=available_inputs,
                explicit_output_targets=request.output_targets,
            )
            if validation_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=validation_diagnostics,
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_VALIDATE,
                    )
                )

            resolved_output_targets, output_target_diagnostics = self._executor._resolve_output_targets(
                ast_statements=ast_statements,
                explicit_output_targets=request.output_targets,
            )
            if output_target_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(output_target_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_OUTPUT_OPTION,
                    )
                )

            resolved_inputs, resolution_diagnostics = self._executor._resolve_inputs(
                ast_statements=ast_statements,
                explicit_inputs=explicit_inputs,
                available_inputs=available_inputs,
            )
            if resolution_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(resolution_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_INPUT_OPTION,
                    )
                )

            prepared_inputs, prepare_diagnostics = self._executor._prepare_runtime_inputs(
                ast_statements=ast_statements,
                resolved_inputs=resolved_inputs,
            )
            if prepare_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(prepare_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_INPUT_PREPARE,
                    )
                )
            resolved_inputs = prepared_inputs

            prepared_statements, prepared_eval_inputs, eval_prepare_diagnostics = self._executor._prepare_runtime_evaluations(
                ast_statements=ast_statements,
                resolved_inputs=resolved_inputs,
            )
            if eval_prepare_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(eval_prepare_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_EVAL_PREPARE,
                    )
                )
            ast_statements = prepared_statements
            resolved_inputs = prepared_eval_inputs
            execution_plan = self._generate_execution_plan(
                request=request,
                ast_statements=ast_statements,
                resolved_inputs=resolved_inputs,
                resolved_output_targets=resolved_output_targets,
            )

            execution_outputs, execution_diagnostics = self._executor._execute_program(
                RuntimeExecutionContext(
                    request=request,
                    ast_statements=ast_statements,
                    resolved_inputs=resolved_inputs,
                    resolved_output_targets=resolved_output_targets,
                    execution_plan=execution_plan,
                )
            )
            if execution_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(execution_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_EXECUTE,
                    )
                )

            output_stage_outputs, output_stage_diagnostics = self._executor._apply_output_stage_options(
                ast_statements=ast_statements,
                resolved_inputs=resolved_inputs,
                outputs=execution_outputs,
            )
            if output_stage_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(output_stage_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_OUTPUT_OPTION,
                    )
                )
            execution_outputs = output_stage_outputs
            execution_outputs = self._executor._apply_output_metadata(
                ast_statements=ast_statements,
                outputs=execution_outputs,
            )

            for name, dataset in execution_outputs.items():
                combined_outputs[name] = dataset
                available_inputs[name] = dataset
                self._executor._generated_tables[name] = dataset

            notices = _collect_compatibility_notices(dsl_text=block_dsl, inputs=resolved_inputs)
            combined_notices.extend(notices)

            future_required = future_required_after_block[block_index - 1]
            if future_required:
                protected_inputs = explicit_input_names.union(future_required)
                available_inputs = {
                    name: dataset
                    for name, dataset in available_inputs.items()
                    if self._executor._dataset_name_key(name) in protected_inputs
                }
            else:
                available_inputs = {
                    name: dataset
                    for name, dataset in available_inputs.items()
                    if self._executor._dataset_name_key(name) in explicit_input_names
                }

        outputs_arrow = _LazyArrowOutputs(combined_outputs, self._executor._io_service.dataset_ref_to_arrow_table)
        return ExecuteResponse(outputs=combined_outputs, outputs_arrow=outputs_arrow, notices=tuple(combined_notices))

    def _generate_execution_plan(
        self,
        *,
        request: ExecuteRequest,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        resolved_output_targets: tuple[str, ...],
    ) -> Any | None:
        del request, ast_statements, resolved_inputs, resolved_output_targets
        return None

    def _validate_request(self, request: ExecuteRequest) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []

        if not request.dsl_text or not request.dsl_text.strip():
            diagnostics.append(
                Diagnostic(
                    code="REQ_EMPTY_DSL",
                    severity="error",
                    message="dsl_text must not be empty.",
                )
            )

        explicit_inputs = request.inputs or {}

        if any(not name.strip() for name in explicit_inputs):
            diagnostics.append(
                Diagnostic(
                    code="REQ_INVALID_INPUTS",
                    severity="error",
                    message="inputs must contain non-empty dataset names.",
                )
            )

        if request.output_targets and any(not target.strip() for target in request.output_targets):
            diagnostics.append(
                Diagnostic(
                    code="REQ_INVALID_OUTPUT_TARGETS",
                    severity="error",
                    message="output_targets must contain non-empty target names.",
                )
            )

        return tuple(diagnostics)

    def _apply_macro_hook(self, dsl_text: str) -> tuple[str, tuple[Diagnostic, ...]]:
        try:
            return self._macro_hook.evaluate(dsl_text), ()
        except Exception as error:
            return "", (
                Diagnostic(
                    code="MACRO_EVALUATION_FAILED",
                    severity="error",
                    message=f"macro hook failed: {error}",
                ),
            )

    def _tag_stage(self, diagnostics: Sequence[Diagnostic], stage: str) -> tuple[Diagnostic, ...]:
        tagged: list[Diagnostic] = []
        for diagnostic in diagnostics:
            tagged.append(
                Diagnostic(
                    code=diagnostic.code,
                    severity=diagnostic.severity,
                    message=diagnostic.message,
                    location=diagnostic.location,
                    stage=diagnostic.stage or stage,
                    span=diagnostic.span,
                    labels=diagnostic.labels,
                    notes=diagnostic.notes,
                    source_text=diagnostic.source_text,
                )
            )
        return tuple(tagged)


class DataStepExecutor:
    _PREPARED_SET_ROWS_MARKER = "#prepared_set_rows"
    _PREPARED_MERGE_ROWS_MARKER = "#prepared_merge_rows"
    _PREPARED_INTERNAL_VARS_MARKER = "|internal="

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
        self._runtime = PDVRuntimeService(format_registry=format_registry)
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

    def set_parser_backend(self, parser_backend: str) -> None:
        self._parser_backend_preference = parser_backend

    @property
    def last_parser_backend(self) -> str:
        return self._last_parser_backend

    def execute(self, request: ExecuteRequest) -> ExecuteResponse:
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
        return self._io_service.resolve_inputs(
            ast_statements=ast_statements,
            explicit_inputs=explicit_inputs,
            available_inputs=available_inputs,
            registered_tables=self._registered_tables,
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

        candidates: dict[str, DataSetRef] = {}
        candidates.update(self._registered_tables)
        candidates.update(available_inputs)
        candidates.update(explicit_inputs)

        source_refs = tuple(getattr(source_statement, "dataset_refs", ()) or ())
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

        runtime_backend = self._runtime_backend_preference.strip().lower()
        if not has_merge and runtime_backend in {"rust", "auto"}:
            interleaved = self._try_interleave_set_inputs(
                ast_statements, current_inputs, by_keys_for_sort
            )
            if interleaved is not None:
                current_inputs = interleaved
        elif by_keys_for_sort and not has_merge:
            current_inputs = self._sort_inputs_by_by_keys(
                current_inputs, by_keys_for_sort
            )

        should_prepare_set_rows = self._runtime_backend_preference.strip().lower() == "python"
        if not should_prepare_set_rows:
            set_stmt = next((s for s in ast_statements if s.kind == "SET"), None)
            source_refs = list(getattr(set_stmt, "dataset_refs", ()) or ()) if set_stmt is not None else []
            should_prepare_set_rows = bool(
                source_refs
                and any(
                    ref.options.keep_vars
                    or ref.options.drop_vars
                    or ref.options.rename_map
                    or ref.options.where_expr
                    or getattr(ref.options, "firstobs", None) is not None
                    or getattr(ref.options, "obs", None) is not None
                    for ref in source_refs
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

        return current_inputs, diagnostics

    def _apply_output_stage_options(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        outputs: Mapping[str, DataSetRef],
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        diagnostics: list[Diagnostic] = []
        excluded_names = self._collect_internal_output_variable_names(
            ast_statements=ast_statements,
            resolved_inputs=resolved_inputs,
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
    ) -> dict[str, DataSetRef]:
        data_statement = next((statement for statement in ast_statements if statement.kind == "DATA"), None)
        label_statement = next((statement for statement in ast_statements if statement.kind == "LABEL"), None)
        if data_statement is None and label_statement is None:
            return dict(outputs)

        column_labels = dict(getattr(label_statement, "label_map", {}))
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

    def _collect_internal_output_variable_names(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
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
            if self._PREPARED_MERGE_ROWS_MARKER not in location:
                continue
            if self._PREPARED_INTERNAL_VARS_MARKER not in location:
                continue
            raw = location.split(self._PREPARED_INTERNAL_VARS_MARKER, maxsplit=1)[1]
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

        self._warm_expression_evaluation_caches(statements)
        return statements, current_inputs, diagnostics

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

        merged_rows_with_source, merge_error = self._python_backend_service._build_merge_rows(
            source_refs=source_refs,
            by_keys=by_keys,
            resolved_inputs=resolved_inputs,
            internal_variable_names=internal_variable_names,
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
        set_names = {r.name for r in source_refs}
        result: dict[str, DataSetRef] = {}
        for name, dr in resolved_inputs.items():
            if name in set_names:
                if name == first_name:
                    result[name] = DataSetRef(
                        kind="arrow_table",
                        location=dr.location,
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
        runtime_backend = self._runtime_backend_preference.strip().lower()
        has_source_dataset_options = any(
            ref.options.keep_vars
            or ref.options.drop_vars
            or ref.options.rename_map
            or ref.options.where_expr
            or getattr(ref.options, "firstobs", None) is not None
            or getattr(ref.options, "obs", None) is not None
            for ref in source_refs
        )
        needs_prepare = bool(
            in_option_vars
            or indsname_var
            or end_var
            or by_keys
            or has_lag_lead
            or (runtime_backend in {"rust", "auto"} and has_source_dataset_options)
        )
        if not needs_prepare:
            return None, None

        rows_with_source: list[tuple[str, dict[str, Any], str | None]] = []
        internal_names = self._collect_internal_variable_names(
            in_option_vars=in_option_vars,
            indsname_var=indsname_var,
            end_var=end_var,
            by_keys=by_keys,
        )
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

            is_prepared_merge_rows = (
                isinstance(input_ref.location, str)
                and self._PREPARED_MERGE_ROWS_MARKER in input_ref.location
            )

            for row in option_rows:
                if not is_prepared_merge_rows:
                    collided = sorted(internal_names.intersection(row.keys()))
                    if collided:
                        return None, Diagnostic(
                            code="RUNTIME_INTERNAL_VAR_NAME_COLLISION",
                            severity="error",
                            message=(
                                "Input dataset contains a reserved internal reference variable name: "
                                f"{collided[0]} (source={source_ref.name})"
                            ),
                        )
                rows_with_source.append((source_ref.name, row, source_ref.options.in_var))

        if by_keys and len(rows_with_source) > 1:
            def _sort_key(item: tuple[str, dict[str, Any], str | None], keys: tuple[str, ...] = by_keys) -> tuple:
                row = item[1]
                return tuple(
                    (row.get(k) is None, str(row.get(k)) if row.get(k) is not None else "")
                    for k in keys
                )
            rows_with_source.sort(key=_sort_key)

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

        if by_keys:
            if any(any(self._resolve_row_key(row, by_key) is None for by_key in by_keys) for row in prepared_rows):
                missing_key = next(
                    by_key
                    for by_key in by_keys
                    if any(self._resolve_row_key(row, by_key) is None for row in prepared_rows)
                )
                return None, Diagnostic(
                    code="RUNTIME_BY_PRECONDITION_FAILED",
                    severity="error",
                    message=f"BY key '{missing_key}' is missing in source rows.",
                )

            for by_key in by_keys:
                for index, row in enumerate(prepared_rows):
                    previous_value = self._resolve_row_value(prepared_rows[index - 1], by_key) if index > 0 else object()
                    next_value = self._resolve_row_value(prepared_rows[index + 1], by_key) if index < len(prepared_rows) - 1 else object()
                    current_value = self._resolve_row_value(row, by_key)
                    row[f"FIRST.{by_key}"] = 1 if current_value != previous_value else 0
                    row[f"LAST.{by_key}"] = 1 if current_value != next_value else 0
                    row[f"first.{by_key}"] = row[f"FIRST.{by_key}"]
                    row[f"last.{by_key}"] = row[f"LAST.{by_key}"]

        first_source = source_refs[0].name
        first_input = resolved_inputs.get(first_source)
        if first_input is None:
            return None, None

        marker_location = f"{first_input.location}{self._PREPARED_SET_ROWS_MARKER}"
        prepared_ref = DataSetRef(
            kind="memory",
            location=marker_location,
            payload=prepared_rows,
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
        lag_lead_pattern = re.compile(r"\b(lag|lead)\s*\(", re.IGNORECASE)
        for statement in ast_statements:
            statement_text = getattr(statement, "text", "")
            if isinstance(statement_text, str) and lag_lead_pattern.search(statement_text):
                return True
        return False

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
                    return tuple(
                        (
                            self._resolve_row_value(row, k) is None,
                            str(self._resolve_row_value(row, k)) if self._resolve_row_value(row, k) is not None else "",
                        )
                        for k in keys
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
            return DataSetRef(kind="arrow_table", location=f"dataset://{name}", payload=dataset)

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
