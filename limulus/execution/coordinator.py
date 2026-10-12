"""Public execution facade and dependency composition root."""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

from ..backend_integration import (
    RustNativeBlockExecutor,
    RustRuntimeBackend,
)
from ..backend_integration.contracts import TypedRustExecutionContext
from ..backend_integration.typed_bridge import TypedRustArrowIOBridge
from ..backend_integration.transport import _STANDARD_TRANSPORT_KIND
from ..format_registry import FormatRegistry
from ..io import ExecutorIOService
from ..io_adapters import (
    DataFrameAdapterPandas,
    DataInputAdapterArrow,
    DataOutputAdapterArrow,
)
from ..models import (
    DataSetRef,
    Diagnostic,
    ExecuteRequest,
    ExecuteResponse,
    NormalizedExecutionInputs,
    OutputConversionResult,
    PreparedExecutionInputs,
    SourceMetadataLineage,
)
from ..syntax.program import LarkProgramFrontend, ParserInitializationError
from ..syntax.compat import block_display_slices, project_block_displays
from .contracts import (
    ExecutionObservations,
    PipelineExecutionRequest,
    PipelineExecutionResult,
    TypedInputPreparationRequest,
    TypedInputResolutionRequest,
    TypedOutputFinalizeRequest,
    TypedOutputPlanRequest,
    TypedOutputShapingRequest,
    TypedPlanBuildRequest,
    TypedValidationRequest,
    StrategyDecisionRequest,
)
from .compatibility import _collect_compatibility_notices
from .input_preparation import (
    ExecutionInputPreparationService,
    normalize_execution_inputs,
)
from .output_handoff import ExecutionOutputHandoffService
from .hooks import MacroHook, NoOpMacroHook
from .planning import ExecutionPlanService
from .syntax_rewrites import rewrite_typed_block
from .runtime_handoff import ExecutionRuntimeHandoffService
from .strategy import ExecutionStrategyDecision
from .validation import ExecutionValidationService


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

class DataStepExecutor:
    """Stable public facade around the request-scoped execution pipeline."""

    _STAGE_SPLIT = "split blocks"
    _STAGE_MACRO = "macro hook"
    _STAGE_PARSE = "parse"
    _STAGE_VALIDATE = "validate"
    _STAGE_INPUT_OPTION = "resolve inputs"
    _STAGE_INPUT_PREPARE = "pre-processing inputs"
    _STAGE_EVAL_PREPARE = "pre-evaluations"
    _STAGE_EXECUTE = "execute"
    _STAGE_OUTPUT_OPTION = "resolve outputs"

    def __init__(
        self,
        runtime_backend: str = "auto",
        *,
        format_registry: FormatRegistry | None = None,
        macro_hook: MacroHook | None = None,
    ) -> None:
        # The compatibility parser remains a standalone public service;
        # execution consumes one whole-program typed parse.
        self._program_frontend = LarkProgramFrontend()
        self._macro_hook = macro_hook if macro_hook is not None else NoOpMacroHook()
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
        self._last_strategy_decision: ExecutionStrategyDecision | None = None
        self._last_runtime_backend: str | None = None
        self._last_phase_metrics: dict[str, float] = {}
        self._last_prepared_execution_inputs = PreparedExecutionInputs()
        self._last_output_source_metadata_lineage = SourceMetadataLineage()

        self._rust_backend = RustRuntimeBackend(
            executor=RustNativeBlockExecutor(),
            typed_bridge=TypedRustArrowIOBridge(),
        )
        self._plan_service = ExecutionPlanService()
        self._runtime_handoff_service = ExecutionRuntimeHandoffService()
        self._validation_service = ExecutionValidationService(
            io_service=self._io_service
        )
        self._input_preparation_service = ExecutionInputPreparationService(
            io_service=self._io_service,
        )
        self._output_handoff_service = ExecutionOutputHandoffService(
            io_service=self._io_service
        )

    @staticmethod
    def _tag_stage(
        diagnostics: Sequence[Diagnostic], stage: str
    ) -> tuple[Diagnostic, ...]:
        return tuple(
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
            for diagnostic in diagnostics
        )

    @staticmethod
    def _with_block_location(
        *,
        diagnostics: Sequence[Diagnostic],
        block_index: int,
        include_block_location: bool,
    ) -> tuple[Diagnostic, ...]:
        if not include_block_location:
            return tuple(diagnostics)
        return tuple(
            Diagnostic(
                code=diagnostic.code,
                severity=diagnostic.severity,
                message=diagnostic.message,
                location=(
                    f"block:{block_index},{diagnostic.location}"
                    if diagnostic.location
                    else f"block:{block_index}"
                ),
                stage=diagnostic.stage,
                span=diagnostic.span,
                labels=diagnostic.labels,
                notes=diagnostic.notes,
                source_text=diagnostic.source_text,
            )
            for diagnostic in diagnostics
        )

    @staticmethod
    def _validate_request(request: ExecuteRequest) -> tuple[Diagnostic, ...]:
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
        if request.output_targets is not None and any(
            not target.strip() for target in request.output_targets
        ):
            diagnostics.append(
                Diagnostic(
                    code="REQ_INVALID_OUTPUT_TARGETS",
                    severity="error",
                    message="output_targets must contain non-empty target names.",
                )
            )
        return tuple(diagnostics)

    @staticmethod
    def _project_full_document_diagnostics(
        diagnostics: Sequence[Diagnostic],
        *,
        full_source: str,
        block_display: str,
    ) -> tuple[Diagnostic, ...]:
        """Project only full-document diagnostic displays onto one block."""

        return tuple(
            replace(diagnostic, source_text=block_display)
            if diagnostic.source_text == full_source
            else diagnostic
            for diagnostic in diagnostics
        )

    @staticmethod
    def _project_parse_stop_diagnostics(parsed) -> tuple[Diagnostic, ...]:
        """Project parser diagnostics using the frontend's saved display facts."""

        projection = parsed.projection
        if projection is None:
            return parsed.diagnostics

        source = projection.source
        block_spans = projection.block_display_spans
        skip_spans = tuple(
            (region.span.start, region.span.end)
            for region in projection.skip_regions
        )
        block_displays = block_display_slices(
            source.text,
            tuple(
                (span.start, span.end, terminated_by_run)
                for span, terminated_by_run in block_spans
            ),
            skip_spans,
        )
        if len(block_displays) != len(block_spans):
            # Region spans normally have non-empty displays. If a frontend
            # refusal produced an empty region, retain the raw source rather
            # than associate a diagnostic with the wrong block.
            block_displays = ()

        projected: list[Diagnostic] = []
        for index, diagnostic in enumerate(parsed.diagnostics):
            if diagnostic.source_text != source.text:
                projected.append(diagnostic)
                continue

            display = (
                projection.diagnostic_displays[index]
                if index < len(projection.diagnostic_displays)
                else None
            )
            if display is not None:
                display_text = display.source_text
            elif block_displays:
                diagnostic_start = (
                    diagnostic.span.start if diagnostic.span is not None else 0
                )
                containing = [
                    block_index
                    for block_index, (span, _terminated_by_run) in enumerate(
                        block_spans
                    )
                    if span.start <= diagnostic_start < span.end
                ]
                if containing:
                    block_index = containing[0]
                else:
                    preceding = [
                        block_index
                        for block_index, (span, _terminated_by_run) in enumerate(
                            block_spans
                        )
                        if span.start <= diagnostic_start
                    ]
                    block_index = preceding[-1] if preceding else 0
                display_text = block_displays[block_index]
            else:
                display_text = source.text
            projected.append(replace(diagnostic, source_text=display_text))

        return tuple(projected)

    @staticmethod
    def _request_input_alias_conflict_diagnostics(
        *, explicit_inputs: Mapping[str, Any], dataset_name_key: Any
    ) -> tuple[Diagnostic, ...]:
        seen_raw_names: dict[str, str] = {}
        for raw_name in explicit_inputs:
            canonical_key = dataset_name_key(raw_name)
            first_raw_name = seen_raw_names.setdefault(canonical_key, raw_name)
            if first_raw_name != raw_name:
                return (
                    Diagnostic(
                        code="RUNTIME_REQUEST_INPUT_ALIAS_CONFLICT",
                        severity="error",
                        message=(
                            "Request inputs contain duplicate canonical dataset key "
                            f"'{canonical_key}': '{first_raw_name}', '{raw_name}'. "
                            "Ambiguous request inputs are rejected instead of "
                            "silently choosing one."
                        ),
                    ),
                )
        return ()

    def _apply_macro_hook(
        self, dsl_text: str
    ) -> tuple[str, tuple[Diagnostic, ...]]:
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

    def register_table(self, name: str, dataset: DataSetRef) -> None:
        canonical_key = self._io_service.dataset_name_key(name)
        self._registered_tables[canonical_key] = dataset

    def unregister_table(self, name: str, *, missing_ok: bool = True) -> bool:
        target_key = self._io_service.dataset_name_key(name)
        if target_key not in self._registered_tables:
            if missing_ok:
                return False
            raise KeyError(name)
        del self._registered_tables[target_key]
        return True

    def unregister_tables(
        self, names: Sequence[str], *, missing_ok: bool = True
    ) -> dict[str, bool]:
        return {
            name: self.unregister_table(name, missing_ok=missing_ok)
            for name in names
        }

    def register_tables(self, datasets: Mapping[str, Any]) -> None:
        for name, dataset in datasets.items():
            self._registered_tables[self._io_service.dataset_name_key(name)] = (
                self._coerce_dataset_ref(name=name, dataset=dataset)
            )

    def set_runtime_backend(self, runtime_backend: str) -> None:
        self._runtime_backend_preference = runtime_backend

    @property
    def last_runtime_backend(self) -> str | None:
        return self._last_runtime_backend

    @property
    def last_strategy_decision(self) -> ExecutionStrategyDecision | None:
        return self._last_strategy_decision

    @property
    def last_phase_metrics(self) -> dict[str, float]:
        return dict(self._last_phase_metrics)

    @property
    def last_prepared_execution_inputs(self) -> PreparedExecutionInputs:
        return self._last_prepared_execution_inputs

    @property
    def last_source_metadata_lineage(self) -> SourceMetadataLineage:
        return self._last_prepared_execution_inputs.source_metadata_lineage

    @property
    def last_output_source_metadata_lineage(self) -> SourceMetadataLineage:
        return self._last_output_source_metadata_lineage

    def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        self._last_strategy_decision = None
        self._last_runtime_backend = None
        self._last_phase_metrics = {}
        self._last_prepared_execution_inputs = PreparedExecutionInputs()
        self._last_output_source_metadata_lineage = SourceMetadataLineage()
        pipeline_request = PipelineExecutionRequest(
            request=request,
            registered_tables=self._registered_tables,
            generated_tables=self._generated_tables,
            runtime_backend_preference=(
                request.runtime_backend
                if request.runtime_backend is not None
                else self._runtime_backend_preference
            ),
        )
        result = self._execute_typed_program(pipeline_request)
        self._commit_pipeline_result(result)
        return result.response

    def _execute_typed_program(
        self, request: PipelineExecutionRequest
    ) -> PipelineExecutionResult:
        """Parse the transformed request once, then execute its typed blocks."""

        execute_request = request.request

        def result(
            response: ExecuteResponse,
            *,
            catalog_updates: Mapping[str, DataSetRef] | None = None,
        ) -> PipelineExecutionResult:
            return PipelineExecutionResult(
                response=response,
                catalog_updates=catalog_updates or {},
                observations=ExecutionObservations(
                    strategy_decision=self._last_strategy_decision,
                    runtime_backend=self._last_runtime_backend,
                    phase_metrics=self._last_phase_metrics,
                    prepared_inputs=self._last_prepared_execution_inputs,
                    output_source_metadata_lineage=(
                        self._last_output_source_metadata_lineage
                    ),
                ),
            )

        request_diagnostics = self._validate_request(execute_request)
        if request_diagnostics:
            return result(
                ExecuteResponse(
                    diagnostics=self._tag_stage(request_diagnostics, self._STAGE_SPLIT)
                )
            )

        transformed_source, macro_diagnostics = self._apply_macro_hook(execute_request.dsl_text)
        if macro_diagnostics:
            return result(
                ExecuteResponse(
                    diagnostics=self._tag_stage(macro_diagnostics, self._STAGE_MACRO)
                )
            )

        try:
            parsed = self._program_frontend.parse_program(transformed_source)
        except ParserInitializationError:
            return result(
                ExecuteResponse(
                    diagnostics=(
                        Diagnostic(
                            code="PARSE_BACKEND_CAPABILITY_MISSING",
                            severity="error",
                            message="lark parser is not available; parsing cannot proceed.",
                            location="",
                            stage=self._STAGE_PARSE,
                        ),
                    )
                )
            )
        if parsed.program is None or parsed.has_errors:
            return result(
                ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._project_parse_stop_diagnostics(parsed),
                        self._STAGE_PARSE,
                    )
                )
            )

        program = parsed.program

        # Syntax diagnostics have historically taken precedence over request
        # catalog alias conflicts. Check aliases only after the transformed
        # whole program has passed the parse stop gate.
        alias_conflicts = self._request_input_alias_conflict_diagnostics(
            explicit_inputs=dict(execute_request.inputs or {}),
            dataset_name_key=self._io_service.dataset_name_key,
        )
        if alias_conflicts:
            return result(
                ExecuteResponse(
                    diagnostics=self._tag_stage(alias_conflicts, self._STAGE_INPUT_OPTION)
                )
            )

        if not program.blocks:
            return result(ExecuteResponse())

        # Parse diagnostics for every block are gated above before this loop
        # can call native. Runtime/catalog validation remains block ordered.
        typed_request = replace(execute_request, dsl_text=transformed_source)
        available_inputs = dict(request.generated_tables)
        catalog_updates: dict[str, DataSetRef] = {}
        public_outputs: dict[str, DataSetRef] = {}
        arrow_output_maps: list[Mapping[str, Any]] = []
        notices = []
        warnings: list[Diagnostic] = []
        include_block_location = len(program.blocks) > 1
        block_displays = project_block_displays(program)

        for block_index, block in enumerate(program.blocks, start=1):
            block_response = self._execute_typed_block(
                source=program.source,
                block=block,
                request=typed_request,
                available_inputs=available_inputs,
            )
            skipped_count = sum(
                statement.kind == "SKIPPED" for statement in block.statements
            )
            skipped_diagnostics = tuple(
                Diagnostic(
                    code="RUNTIME_UNSUPPORTED_STATEMENT_SKIPPED",
                    severity="info",
                    message="unsupported statement skipped",
                    stage=self._STAGE_EXECUTE,
                )
                for _ in range(skipped_count)
            )
            block_display = (
                block_displays[block_index - 1]
                if block_index - 1 < len(block_displays)
                else program.source.text[block.span.start : block.span.end].strip()
            )
            block_diagnostics = self._with_block_location(
                diagnostics=self._project_full_document_diagnostics(
                    (*block_response.diagnostics, *skipped_diagnostics),
                    full_source=program.source.text,
                    block_display=block_display,
                ),
                block_index=block_index,
                include_block_location=include_block_location,
            )
            if block_response.has_errors:
                return result(
                    ExecuteResponse(
                        diagnostics=(*warnings, *block_diagnostics),
                        notices=tuple(notices),
                    ),
                    catalog_updates=catalog_updates,
                )

            warnings.extend(block_diagnostics)
            public_outputs.update(block_response.outputs)
            arrow_output_maps.append(block_response.outputs_arrow)
            available_inputs.update(block_response.outputs)
            catalog_updates.update(block_response.outputs)
            notices.extend(
                _collect_compatibility_notices(
                    block=block,
                    inputs=self._last_prepared_execution_inputs.inputs,
                )
            )

        return result(
            ExecuteResponse(
                outputs=public_outputs,
                # ChainMap stubs require mutable maps; this response exposes
                # the chain read-only and only performs lazy lookups.
                outputs_arrow=MappingProxyType(
                    ChainMap(
                        *(
                            cast(MutableMapping[str, Any], output_map)
                            for output_map in reversed(arrow_output_maps)
                        )
                    )
                ),
                diagnostics=tuple(warnings),
                notices=tuple(notices),
            ),
            catalog_updates=catalog_updates,
        )

    def _execute_typed_block(
        self,
        *,
        source: Any,
        block: Any,
        request: ExecuteRequest,
        available_inputs: Mapping[str, DataSetRef] | None = None,
    ) -> ExecuteResponse:
        """Run one internal BlockSyntax through the settled typed stage owners."""

        from ..syntax.models import BlockSyntax, SourceDocument

        if not isinstance(source, SourceDocument):
            raise TypeError("source must be a SourceDocument")
        if not isinstance(block, BlockSyntax):
            raise TypeError("block must be a BlockSyntax")
        if not isinstance(request, ExecuteRequest):
            raise TypeError("request must be an ExecuteRequest")

        warnings: list[Diagnostic] = []

        def stop_on_errors(diagnostics: Sequence[Diagnostic]) -> ExecuteResponse | None:
            if any(diagnostic.severity == "error" for diagnostic in diagnostics):
                return ExecuteResponse(diagnostics=(*warnings, *diagnostics))
            warnings.extend(diagnostics)
            return None

        explicit_inputs = dict(request.inputs or {})
        available_snapshot = dict(
            self._generated_tables if available_inputs is None else available_inputs
        )
        registered_snapshot = dict(self._registered_tables)

        validation = self._validation_service.validate_typed(
            TypedValidationRequest(
                block=block,
                source=source,
                explicit_inputs=explicit_inputs,
                available_inputs=available_snapshot,
                registered_tables=registered_snapshot,
                explicit_output_targets=request.output_targets,
            )
        )
        validation_diagnostics = self._tag_stage(
            validation.diagnostics,
            self._STAGE_VALIDATE,
        )
        stopped = stop_on_errors(validation_diagnostics)
        if stopped is not None:
            return stopped

        helper_plan = self._plan_service.plan_typed_helpers(block)
        rewrite_result = rewrite_typed_block(
            block,
            helper_plan,
            source=source,
            output_handoff_service=self._output_handoff_service,
            explicit_output_targets=request.output_targets,
        )
        stopped = stop_on_errors(rewrite_result.output_plan.diagnostics)
        if stopped is not None:
            return stopped
        if not rewrite_result.output_plan.targets:
            return ExecuteResponse(
                diagnostics=(
                    Diagnostic(
                        code="REQ_INVALID_OUTPUT_TARGETS",
                        severity="error",
                        message=(
                            "output_targets is omitted and no DATA/OUTPUT target "
                            "can be resolved."
                        ),
                        stage=self._STAGE_OUTPUT_OPTION,
                    ),
                )
            )

        resolution = self._input_preparation_service.resolve_typed(
            TypedInputResolutionRequest(
                block=rewrite_result.block,
                explicit_inputs=explicit_inputs,
                available_inputs=available_snapshot,
                registered_tables=registered_snapshot,
            )
        )
        resolution_stage = self._STAGE_INPUT_OPTION
        if any(
            diagnostic.code
            in {
                "RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW",
                "RUNTIME_DATASET_OPTION_INVALID",
            }
            for diagnostic in resolution.diagnostics
        ):
            # The selected runtime owner is recorded even when Arrow
            # normalization fails before native is invoked.
            self._last_runtime_backend = "rust"
            resolution_stage = self._STAGE_INPUT_PREPARE
        resolution_diagnostics = self._tag_stage(
            resolution.diagnostics,
            resolution_stage,
        )
        stopped = stop_on_errors(resolution_diagnostics)
        if stopped is not None:
            return stopped

        preparation = self._input_preparation_service.prepare_typed(
            TypedInputPreparationRequest(
                block=rewrite_result.block,
                helper_plan=helper_plan,
                input_resolution=resolution,
            )
        )
        preparation_diagnostic_stage = (
            self._STAGE_EVAL_PREPARE
            if any(statement.kind == "MERGE" for statement in rewrite_result.block.statements)
            else self._STAGE_INPUT_PREPARE
        )
        preparation_diagnostics = self._tag_stage(
            preparation.diagnostics,
            preparation_diagnostic_stage,
        )
        stopped = stop_on_errors(preparation_diagnostics)
        if stopped is not None:
            return stopped

        # Keep the complete prepared carrier as the stage observation.  The
        # native handoff may project case-insensitive internal aliases, but
        # that transport view must not erase prepared MERGE lineage columns
        # from the preparation result exposed by the executor.
        self._last_prepared_execution_inputs = preparation.prepared
        resolved_inputs = self._input_preparation_service.project_prepared_internal_aliases_for_handoff(
            preparation.prepared.inputs
        )
        prepared = PreparedExecutionInputs(
            inputs=resolved_inputs,
            preparation_kind=preparation.prepared.preparation_kind,
            source_metadata_lineage=preparation.prepared.source_metadata_lineage,
        )
        shaping = self._output_handoff_service.plan_typed_shaping(
            TypedOutputShapingRequest(
                block=preparation.block,
                source=source,
                output_target_plan=rewrite_result.output_plan,
                resolved_inputs=resolved_inputs,
                helper_plan=helper_plan,
            )
        )
        stopped = stop_on_errors(shaping.diagnostics)
        if stopped is not None:
            return stopped
        if shaping.output_plan is None:
            return ExecuteResponse(diagnostics=tuple(warnings))

        strategy_decision = self._runtime_handoff_service.decide_strategy(
            StrategyDecisionRequest(
                runtime_backend_preference=(
                    request.runtime_backend
                    if request.runtime_backend is not None
                    else self._runtime_backend_preference
                )
            )
        )
        stopped = stop_on_errors(strategy_decision.diagnostics)
        if stopped is not None:
            return stopped
        if strategy_decision.strategy_decision is None:
            return ExecuteResponse(diagnostics=tuple(warnings))

        self._last_strategy_decision = strategy_decision.strategy_decision
        slot_registry = self._output_handoff_service.build_typed_slot_registry(
            block=preparation.block,
            resolved_inputs=resolved_inputs,
            helper_plan=helper_plan,
        )
        plan_result = self._plan_service.build_typed(
            TypedPlanBuildRequest(
                block=preparation.block,
                prepared=prepared,
                helper_plan=helper_plan,
                output_target_plan=rewrite_result.output_plan,
                output_handoff_plan=shaping.output_plan,
                slot_registry=slot_registry,
                strategy_decision=strategy_decision.strategy_decision,
            )
        )
        stopped = stop_on_errors(plan_result.diagnostics)
        if stopped is not None:
            return stopped
        if plan_result.execution_plan is None:
            return ExecuteResponse(diagnostics=tuple(warnings))

        runtime_result = self._rust_backend.execute_typed(
            TypedRustExecutionContext(
                source=source,
                block=preparation.block,
                request=request,
                resolved_inputs=resolved_inputs,
                preparation_kind=prepared.preparation_kind,
                output_plan=rewrite_result.output_plan,
                execution_plan=plan_result.execution_plan,
            )
        )
        self._last_runtime_backend = runtime_result.actual_engine
        self._last_phase_metrics = dict(runtime_result.phase_metrics)
        if runtime_result.diagnostics:
            runtime_diagnostics = self._tag_stage(
                runtime_result.diagnostics,
                self._STAGE_EXECUTE,
            )
            return ExecuteResponse(
                diagnostics=(*warnings, *runtime_diagnostics)
            )

        finalized = self._output_handoff_service.finalize_typed(
            TypedOutputFinalizeRequest(
                block=preparation.block,
                output_target_plan=rewrite_result.output_plan,
                output_plan=shaping.output_plan,
                resolved_inputs=resolved_inputs,
                outputs=runtime_result.outputs,
                helper_plan=helper_plan,
                source_metadata_lineage=prepared.source_metadata_lineage,
                preparation_kind=prepared.preparation_kind,
                preserve_memory_presentation=(
                    resolution.resolved_inputs.preserve_memory_presentation
                ),
            )
        )
        self._last_output_source_metadata_lineage = (
            finalized.output_source_metadata_lineage
        )
        finalized_diagnostics = self._tag_stage(
            finalized.diagnostics,
            self._STAGE_OUTPUT_OPTION,
        )
        return ExecuteResponse(
            outputs=finalized.public_outputs,
            outputs_arrow=finalized.outputs_arrow,
            diagnostics=(*warnings, *finalized_diagnostics),
        )

    def _commit_pipeline_result(self, result: PipelineExecutionResult) -> None:
        """Commit one result exactly once, including an error response."""

        # ``catalog_updates`` is the successful prefix only. The pipeline
        # never places a failing block's output in this mapping.
        self._generated_tables.update(result.catalog_updates)
        observations = result.observations
        self._last_strategy_decision = observations.strategy_decision
        self._last_runtime_backend = observations.runtime_backend
        self._last_phase_metrics = dict(observations.phase_metrics)
        self._last_prepared_execution_inputs = observations.prepared_inputs
        self._last_output_source_metadata_lineage = (
            observations.output_source_metadata_lineage
        )

    def convert_outputs(
        self, response: ExecuteResponse, target_format: str
    ) -> OutputConversionResult:
        return self._io_service.convert_outputs(response, target_format)

    def normalize_execution_inputs(
        self,
        inputs: Mapping[str, DataSetRef],
    ) -> tuple[NormalizedExecutionInputs | None, Diagnostic | None]:
        """Public adapter for the input-stage normalization contract."""

        return normalize_execution_inputs(inputs)

    def _coerce_dataset_ref(self, name: str, dataset: Any) -> DataSetRef:
        if isinstance(dataset, DataSetRef):
            return dataset

        if isinstance(dataset, Sequence) and not isinstance(
            dataset, (str, bytes, bytearray)
        ):
            if all(isinstance(item, Mapping) for item in dataset):
                return DataSetRef(
                    kind="memory",
                    location=f"dataset://{name}",
                    payload=list(dataset),
                )

        if hasattr(dataset, "to_pylist"):
            return DataSetRef(
                kind=_STANDARD_TRANSPORT_KIND,
                location=f"dataset://{name}",
                payload=dataset,
            )

        try:
            import polars as pl
        except ImportError:
            pass
        else:
            if isinstance(dataset, pl.DataFrame):
                return DataSetRef(
                    kind="polars",
                    location=f"dataset://{name}",
                    payload=dataset,
                )

        try:
            import pandas as pd
        except ImportError:
            pass
        else:
            if isinstance(dataset, pd.DataFrame):
                return DataSetRef(
                    kind="pandas",
                    location=f"dataset://{name}",
                    payload=dataset,
                )

        raise ValueError(f"Unsupported dataset type for register_tables: {name}")

    def list_supported_formats(self) -> tuple[str, ...]:
        return _SUPPORTED_FORMATS

    def get_runtime_requirements(self) -> RuntimeRequirements:
        return RuntimeRequirements(
            required_python=">=3.10",
            supported_os=("windows", "linux", "macos"),
        )

    def check_format_support(self, format_name: str) -> FormatSupportResult:
        normalized = format_name.strip().lower()
        if normalized in _SUPPORTED_FORMATS:
            return FormatSupportResult(supported=True, reason_code="", message="")
        return FormatSupportResult(
            supported=False,
            reason_code="CAP_UNSUPPORTED_FORMAT",
            message=f"Unsupported format: {format_name}",
        )
