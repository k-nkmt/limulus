"""Durable checks for the strategy-neutral input/output boundaries."""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, fields as dataclass_fields, replace
from types import MappingProxyType, SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest

from limulus import arrow_bridge as arrow_bridge_module
from limulus.backend_integration import typed_bridge as typed_bridge_module
from limulus.dataset_option_expressions import WhereComparison
from limulus.execution import DataStepExecutor
from limulus.execution.contracts import (
    InputPreparationRequest,
    InputResolutionResult,
    TypedOutputFinalizeRequest,
    TypedOutputPlanRequest,
)
from limulus.execution import input_preparation as input_module
from limulus.arrow_bridge import (
    project_lineage_for_output,
    restore_arrow_schema_from_merge_lineage,
)
from limulus.execution.output_handoff import (
    ExecutionOutputHandoffService,
    OutputLineageProjectionPlan,
    OutputProjectionStage,
)
from limulus.execution.rewrites import RewritePlan
from limulus.models import (
    DataSetRef,
    ExecuteRequest,
    NormalizedExecutionInputs,
    PreparedExecutionInputs,
    SourceMetadataLineage,
    SourceMetadataSnapshot,
)
from limulus.parser import ParserService
from limulus.syntax import parse_program


def _rewrite_plan(executor: DataStepExecutor, statements, inputs):
    del executor, statements, inputs
    return RewritePlan(
        merge_mode="none",
        lag_lead_mode="none",
        where_mode="none",
        helper_source_columns=(),
        helper_runtime_columns=(),
    )


def _prepare_runtime_inputs_for_test(executor, ast_statements, resolved_inputs, **kwargs):
    kwargs.setdefault(
        "rewrite_plan",
        _rewrite_plan(executor, ast_statements, resolved_inputs),
    )
    result = executor._input_preparation_service.prepare(
        InputPreparationRequest(
            ast_statements=tuple(ast_statements),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(inputs=resolved_inputs),
            ),
            rewrite_plan=kwargs["rewrite_plan"],
        )
    )
    return dict(result.prepared.inputs), list(result.diagnostics)


def test_shared_input_module_has_no_python_row_owner() -> None:
    source = inspect.getsource(input_module)

    assert "_PythonDatasetExecutionMixin" not in source
    assert "_ArrowRowCarrier" not in source
    assert "_PythonInputPreparationService" not in source
    assert "limulus.runtime" not in source
    assert not hasattr(input_module, "_PythonDatasetExecutionMixin")
    assert not hasattr(input_module, "_ArrowRowCarrier")
    assert not hasattr(input_module, "_PythonInputPreparationService")


def test_shared_input_keeps_arrow_dataset_option_preprocessing() -> None:
    option_spec = SimpleNamespace(
        keep_vars=(),
        drop_vars=(),
        rename_map={"amount": "value"},
        where_expr=WhereComparison("id", ">=", 2),
        firstobs=None,
        obs=None,
    )

    table, diagnostic = input_module.try_apply_dataset_reference_options_to_arrow_table(
        pa.table({"id": [1, 2], "amount": [10, 20]}),
        source_name="source",
        option_spec=option_spec,
    )

    assert diagnostic is None
    assert table is not None
    assert table.to_pylist() == [{"id": 2, "value": 20}]


def test_typed_source_plan_and_arrow_handoff_preserve_schema_metadata() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    dsl = "data out; set source(where=(amount >= 20) rename=(amount=score)); run;"
    parsed = ParserService().parse(dsl)
    assert not parsed.has_errors
    source_schema = pa.schema(
        [
            pa.field("id", pa.int64(), metadata={b"label": b"Identifier"}),
            pa.field("amount", pa.int64(), metadata={b"label": b"Amount"}),
        ],
        metadata={b"memlabel": b"Source"},
    )
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.Table.from_arrays(
            [pa.array([1, 2]), pa.array([10, 20])],
            schema=source_schema,
        ),
    )

    prepared, diagnostics = _prepare_runtime_inputs_for_test(executor,
        parsed.ast.statements,
        {"source": source},
    )

    assert diagnostics == []
    prepared_table = prepared["source"].payload
    assert prepared_table.to_pylist() == [{"id": 2, "score": 20}]
    assert prepared_table.schema.metadata == {b"memlabel": b"Source"}
    assert prepared_table.schema.field("id").metadata == {b"label": b"Identifier"}
    assert prepared_table.schema.field("score").metadata == {b"label": b"Amount"}


def test_dataset_options_parse_and_prepare_combined_filter() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    dsl = (
        'data out; set source(keep=id amount status drop=temporary '
        'where=((amount >= 20 and status = "ready") or amount is null) '
        'rename=(amount=score) firstobs=1 obs=10); run;'
    )
    parsed = ParserService().parse(dsl)
    assert not parsed.has_errors
    prepared, diagnostics = _prepare_runtime_inputs_for_test(executor,
        parsed.ast.statements,
        {
            "source": DataSetRef(
                kind="arrow_table",
                location="dataset://source",
                payload=pa.table(
                    {
                        "id": [1, 2, 3],
                        "amount": [10, 20, None],
                        "status": ["ready", "ready", "pending"],
                        "temporary": ["a", "b", "c"],
                    }
                ),
            )
        },
    )

    assert diagnostics == []
    assert prepared["source"].payload.to_pylist() == [
        {"id": 2, "score": 20, "status": "ready"},
        {"id": 3, "score": None, "status": "pending"},
    ]


def test_schema_lineage_is_schema_only_and_stays_out_of_typed_bridge() -> None:
    snapshot_fields = {field.name for field in dataclass_fields(SourceMetadataSnapshot)}
    assert {"source_ordinal", "source_name", "schema", "dataset_ref_metadata"} <= snapshot_fields
    assert snapshot_fields.isdisjoint({"table", "rows", "payload", "data"})

    arrow_source = inspect.getsource(arrow_bridge_module)
    typed_bridge_source = inspect.getsource(typed_bridge_module)
    assert "SourceMetadataLineage" in arrow_source
    assert "OutputLineageProjectionPlan" not in typed_bridge_source
    assert "SourceMetadataLineage" not in typed_bridge_source


def test_metadata_restoration_preserves_target_types_and_arrow_buffers() -> None:
    source_schema = pa.schema(
        [
            pa.field("id", pa.int32(), metadata={b"label": b"source id"}),
            pa.field("score", pa.string(), metadata={b"label": b"source score"}),
        ],
        metadata={b"memlabel": b"source"},
    )
    lineage = SourceMetadataLineage(
        ordered_sources=(
            SourceMetadataSnapshot(
                source_ordinal=0,
                source_name="source",
                schema=source_schema,
            ),
        )
    )
    target_schema = pa.schema(
        [
            pa.field("id", pa.int64(), metadata={b"label": b"target id"}),
            pa.field("score", pa.int64(), metadata={b"label": b"target score"}),
        ],
        metadata={b"memlabel": b"target"},
    )
    target = pa.Table.from_arrays(
        [pa.array([1, 2], type=pa.int64()), pa.array([10, None], type=pa.int64())],
        schema=target_schema,
    )

    def buffer_addresses(table: pa.Table) -> tuple[tuple[int | None, ...], ...]:
        return tuple(
            tuple(
                None if buffer is None else buffer.address
                for chunk in table.column(index).chunks
                for buffer in chunk.buffers()
            )
            for index in range(table.num_columns)
        )

    target_types = target.schema.types
    target_buffers = buffer_addresses(target)
    restored = restore_arrow_schema_from_merge_lineage(
        target,
        lineage=lineage,
        by_keys=("id",),
    )

    assert restored.schema.types == target_types
    assert buffer_addresses(restored) == target_buffers
    assert restored.schema.metadata == {b"memlabel": b"source"}
    assert restored.schema.field("id").metadata == {b"label": b"source id"}
    assert restored.schema.field("score").metadata == {b"label": b"source score"}


@dataclass(frozen=True)
class _OwnerRow:
    symbol: str
    lifecycle: str
    owner_path: str
    allowed_refs: str
    export_contract: str


_OWNER_SCHEMA = tuple(
    _OwnerRow(*row)
    for row in (
        ("DataStepExecutor", "PUBLIC_RETAINED", "limulus/execution/coordinator.py", "limulus/session.py; limulus/__init__.py; limulus/execution/__init__.py; limulus/runtime/__init__.py; DataStepExecutor(...)", "limulus/__init__.py::__all__; limulus/execution/__init__.py::__all__; limulus/runtime/__init__.py::__all__"),
        ("ExecutionStrategyDecision", "ADVANCED_RETAINED", "limulus/execution/strategy.py", "limulus/execution/__init__.py; limulus/execution/contracts.py; limulus/execution/coordinator.py; limulus/execution/planning.py; limulus/execution/strategy.py; ExecutionStrategyDecision(...)", "limulus/execution/__init__.py::__all__; limulus/execution/strategy.py::__all__"),
        ("_ExecutionPipelineCoordinator", "REMOVED_INTERNAL", "-", "none; typed coordinator is the only execution path", "absent from module and package exports"),
        ("PipelineDependencies", "REMOVED_INTERNAL", "-", "none; typed coordinator owns stage order", "absent from module exports"),
        ("PipelineBlockSplitter", "REMOVED_INTERNAL", "-", "none; whole-program frontend owns block regions", "absent from module exports"),
        ("PipelineParserFrontend", "REMOVED_INTERNAL", "-", "none; whole-program frontend is called directly", "absent from module exports"),
        ("PipelineOutputTargetResolver", "REMOVED_INTERNAL", "-", "none; typed validation owns output targets", "absent from module exports"),
        ("PipelineSyntheticInputResolver", "REMOVED_INTERNAL", "-", "none; typed input owner resolves sources", "absent from module exports"),
        ("PipelineRuntimeEvaluationPreparer", "REMOVED_INTERNAL", "-", "none; typed input owner prepares inputs", "absent from module exports"),
        ("PipelinePreparedInputProjector", "REMOVED_INTERNAL", "-", "none; typed coordinator uses output handoff owner", "absent from module exports"),
        ("PipelineSlotRegistryBuilder", "REMOVED_INTERNAL", "-", "none; typed output owner builds the slot registry", "absent from module exports"),
        ("PipelineInternalOutputNameCollector", "REMOVED_INTERNAL", "-", "none; typed output owner owns internal names", "absent from module exports"),
        ("PipelineVariableListExtractor", "REMOVED_INTERNAL", "-", "none; typed syntax carries name lists", "absent from module exports"),
        ("PipelineRuntimeService", "REMOVED_INTERNAL", "-", "none; typed runtime handoff owner is used directly", "absent from module exports"),
        ("PipelineValidationService", "REMOVED_INTERNAL", "-", "none; typed validation owner is used directly", "absent from module exports"),
        ("PipelineInputPreparationService", "REMOVED_INTERNAL", "-", "none; typed input owner is used directly", "absent from module exports"),
        ("PipelinePlanService", "REMOVED_INTERNAL", "-", "none; typed plan owner is used directly", "absent from module exports"),
        ("PipelineOutputHandoffService", "REMOVED_INTERNAL", "-", "none; typed output handoff owner is used directly", "absent from module exports"),
        ("PipelineMacroHook", "REMOVED_INTERNAL", "-", "none; public hook protocol lives in hooks.py", "absent from module exports"),
        ("PipelineExecutionRequest", "INTERNAL_RETAINED", "limulus/execution/contracts.py", "limulus/execution/coordinator.py; immutable registered/generated DataSetRef snapshots", "limulus/execution/contracts.py::__all__"),
        ("ExecutionValidationService", "INTERNAL_RETAINED", "limulus/execution/validation.py", "limulus/execution/coordinator.py; limulus/execution/output_handoff.py; ExecutionValidationService(...)", "limulus/execution/validation.py::__all__"),
        ("ExecutionInputPreparationService", "INTERNAL_RETAINED", "limulus/execution/input_preparation.py", "limulus/execution/coordinator.py; ExecutionInputPreparationService(...)", "none; module-internal"),
        ("ExecutionPlanService", "INTERNAL_RETAINED", "limulus/execution/planning.py", "limulus/execution/coordinator.py; ExecutionPlanService(...); sole RewritePlanner() construction", "limulus/execution/planning.py::__all__"),
        ("RewritePlanner", "INTERNAL_RETAINED", "limulus/execution/rewrites.py", "limulus/execution/planning.py; limulus/execution/rewrites.py; ExecutionPlanService -> RewritePlanner()", "limulus/execution/rewrites.py::__all__"),
        ("ExecutionRuntimeHandoffService", "INTERNAL_RETAINED", "limulus/execution/runtime_handoff.py", "limulus/execution/coordinator.py; ExecutionRuntimeHandoffService(...)", "limulus/execution/runtime_handoff.py::__all__"),
        ("ExecutionOutputHandoffService", "INTERNAL_RETAINED", "limulus/execution/output_handoff.py", "limulus/execution/coordinator.py; limulus/execution/syntax_rewrites.py; ExecutionOutputHandoffService(...)", "limulus/execution/output_handoff.py::__all__"),
        ("OutputHandoffPlan", "INTERNAL_RETAINED", "limulus/execution/output_handoff.py", "limulus/execution/output_handoff.py; OutputHandoffPlan(...)", "limulus/execution/output_handoff.py::__all__"),
        ("OutputLineageProjectionPlan", "INTERNAL_RETAINED", "limulus/execution/output_handoff.py", "limulus/execution/output_handoff.py; OutputLineageProjectionPlan(...)", "limulus/execution/output_handoff.py::__all__"),
        ("ExecutorIOService", "INTERNAL_RETAINED", "limulus/io.py", "limulus/execution/coordinator.py; limulus/execution/input_preparation.py; limulus/execution/output_handoff.py; limulus/execution/planning.py; limulus/execution/validation.py; ExecutorIOService(...)", "none; module-internal"),
        ("_collect_compatibility_notices", "MOVED_INTERNAL", "limulus/execution/compatibility.py", "limulus/execution/coordinator.py; limulus/execution/compatibility.py", "none; not exported"),
        ("_LazyArrowOutputs", "MOVED_INTERNAL", "limulus/execution/output_handoff.py", "limulus/execution/output_handoff.py only", "none; not exported"),
        ("ExecutionPipelineCoordinator", "REMOVED_ADVANCED", "-", "none; no production reference or revival", "absent from package export"),
    )
)

_OWNER_TABLE_HEADER = (
    "| symbol | lifecycle | owner_path | allowed production refs/constructors "
    "| export contract |"
)
_PATH_RE = re.compile(r"limulus/[A-Za-z0-9_./-]+\.py")


def _normalize_owner_cell(value: str) -> str:
    return " ".join(value.strip().replace("`", "").split())


def _parse_owner_table(document: str) -> tuple[_OwnerRow, ...]:
    rows: list[_OwnerRow] = []
    in_table = False
    for line in document.splitlines():
        stripped = line.strip()
        if stripped == _OWNER_TABLE_HEADER:
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped.startswith("|"):
            break
        cells = [_normalize_owner_cell(cell) for cell in stripped.strip("|").split("|")]
        if cells == ["---"] * 5:
            continue
        if len(cells) != 5:
            raise AssertionError(f"invalid execution owner table row: {line}")
        rows.append(_OwnerRow(*cells))
    return tuple(rows)


def _owner_schema_mismatches(
    actual: tuple[_OwnerRow, ...], expected: tuple[_OwnerRow, ...]
) -> tuple[str, ...]:
    if actual == expected:
        return ()
    mismatches: list[str] = []
    for index, (actual_row, expected_row) in enumerate(
        zip(actual, expected, strict=False)
    ):
        if actual_row != expected_row:
            mismatches.append(
                f"row {index}: actual={actual_row!r}, expected={expected_row!r}"
            )
    if len(actual) != len(expected):
        mismatches.append(f"row count: actual={len(actual)}, expected={len(expected)}")
    return tuple(mismatches)


@dataclass(frozen=True)
class _OwnerEvidence:
    definitions: dict[str, tuple[str, ...]]
    references: dict[str, tuple[str, ...]]
    exports: dict[str, frozenset[str]]


def _collect_owner_evidence(repository_root: Path) -> _OwnerEvidence:
    source_trees: dict[str, ast.Module] = {}
    source_root = repository_root / "limulus"
    for source_path in sorted(source_root.rglob("*.py")):
        relative_path = source_path.relative_to(repository_root).as_posix()
        source_trees[relative_path] = ast.parse(
            source_path.read_text(encoding="utf-8"), filename=relative_path
        )

    symbols = {row.symbol for row in _OWNER_SCHEMA if "=" not in row.symbol}
    definitions: dict[str, list[str]] = {symbol: [] for symbol in symbols}
    references: dict[str, set[str]] = {symbol: set() for symbol in symbols}
    exports: dict[str, frozenset[str]] = {}

    for relative_path, tree in source_trees.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in definitions:
                    definitions[node.name].append(relative_path)
            if isinstance(node, ast.Name) and node.id in references:
                references[node.id].add(relative_path)
            if isinstance(node, ast.Attribute) and node.attr in references:
                references[node.attr].add(relative_path)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    imported_name = alias.asname or alias.name.rsplit(".", maxsplit=1)[-1]
                    if alias.name in references:
                        references[alias.name].add(relative_path)
                    if imported_name in references:
                        references[imported_name].add(relative_path)
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                continue
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                continue
            exports[relative_path] = frozenset(
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )

    return _OwnerEvidence(
        definitions={name: tuple(paths) for name, paths in definitions.items()},
        references={name: tuple(sorted(paths)) for name, paths in references.items()},
        exports=exports,
    )


def _audit_owner_rows(
    rows: tuple[_OwnerRow, ...], evidence: _OwnerEvidence
) -> tuple[str, ...]:
    errors: list[str] = []
    for row in rows:
        definitions = evidence.definitions.get(row.symbol, ())
        references = set(evidence.references.get(row.symbol, ()))
        if row.lifecycle.startswith("REMOVED"):
            if definitions or references:
                errors.append(f"removed symbol revived: {row.symbol}")
            if any(
                row.symbol in names for names in evidence.exports.values()
            ):
                errors.append(f"removed symbol exported: {row.symbol}")
            continue

        if definitions != (row.owner_path,):
            errors.append(
                f"{row.symbol} definitions {definitions!r} != {(row.owner_path,)!r}"
            )
        allowed_paths = set(_PATH_RE.findall(row.allowed_refs))
        allowed_paths.add(row.owner_path)
        unexpected_refs = references - allowed_paths
        if unexpected_refs:
            errors.append(f"{row.symbol} references outside owner map: {sorted(unexpected_refs)}")

        export_targets = _PATH_RE.findall(row.export_contract)
        if row.export_contract.startswith("absent") or row.export_contract.startswith("none"):
            if any(row.symbol in names for names in evidence.exports.values()):
                errors.append(f"{row.symbol} is unexpectedly exported")
        else:
            for export_path in export_targets:
                if row.symbol not in evidence.exports.get(export_path, frozenset()):
                    errors.append(f"{row.symbol} missing from {export_path} __all__")
    return tuple(errors)


def test_execution_owner_audit_uses_the_same_rows_for_source_and_export_contracts() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    evidence = _collect_owner_evidence(repository_root)
    assert _audit_owner_rows(_OWNER_SCHEMA, evidence) == ()

    from limulus.execution.contracts import PipelineExecutionRequest

    assert dataclass_fields(PipelineExecutionRequest)
    assert {field.name for field in dataclass_fields(PipelineExecutionRequest)} >= {
        "registered_tables",
        "generated_tables",
    }
    assert PipelineExecutionRequest.__dataclass_params__.frozen is True
    snapshot = PipelineExecutionRequest(
        request=ExecuteRequest(dsl_text=""),
        registered_tables={
            "source": DataSetRef(
                kind="arrow_table", location="dataset://source", payload=pa.table({})
            )
        },
        generated_tables={
            "prior": DataSetRef(
                kind="arrow_table", location="dataset://prior", payload=pa.table({})
            )
        },
    )
    assert isinstance(snapshot.registered_tables, MappingProxyType)
    assert isinstance(snapshot.generated_tables, MappingProxyType)
    with pytest.raises(TypeError):
        snapshot.registered_tables["new"] = snapshot.registered_tables["source"]


def test_owner_audit_rejects_wrong_path_lifecycle_and_removed_symbol_revival() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    evidence = _collect_owner_evidence(repository_root)
    executor_row = next(row for row in _OWNER_SCHEMA if row.symbol == "DataStepExecutor")
    removed_row = next(
        row for row in _OWNER_SCHEMA if row.symbol == "ExecutionPipelineCoordinator"
    )

    wrong_path = replace(executor_row, owner_path="limulus/execution/pipeline.py")
    assert _audit_owner_rows((wrong_path,), evidence)

    wrong_lifecycle = replace(executor_row, lifecycle="INTERNAL_RETAINED")
    assert _owner_schema_mismatches((wrong_lifecycle,), (executor_row,))

    revived = replace(
        removed_row,
        lifecycle="INTERNAL_RETAINED",
        owner_path="limulus/execution/pipeline.py",
        allowed_refs="limulus/execution/pipeline.py",
        export_contract="none; module-internal",
    )
    assert _audit_owner_rows((revived,), evidence)


@pytest.mark.parametrize(
    ("dsl", "expected_code"),
    (
        (
            "data out; set source; rename a=b; output out; run;",
            "RUNTIME_RENAME_STATEMENT_INVALID",
        ),
        (
            "data out(keep=a b rename=(a=b)); set source; output out; run;",
            "RUNTIME_DATASET_OPTION_INVALID",
        ),
    ),
    ids=("statement-rename", "data-target-rename"),
)
def test_output_rename_collision_is_rejected_before_native_invocation(
    dsl: str,
    expected_code: str,
) -> None:
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"a": [1], "b": [2]}),
    )
    executor = DataStepExecutor(runtime_backend="rust")

    with patch.object(executor._rust_backend, "execute_typed", wraps=executor._rust_backend.execute_typed) as native_execute:
        response = executor.execute(ExecuteRequest(dsl_text=dsl, inputs={"source": source}))

    assert response.has_errors is True
    assert response.outputs_arrow == {}
    diagnostic = next(diagnostic for diagnostic in response.diagnostics if diagnostic.severity == "error")
    assert diagnostic.code == expected_code
    assert diagnostic.stage == "validate"
    native_execute.assert_not_called()


def test_statement_keep_then_rename_accepts_destination_removed_by_keep() -> None:
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"amount": [10], "other": [99]}),
    )
    executor = DataStepExecutor(runtime_backend="rust")

    with patch.object(executor._rust_backend, "execute_typed", wraps=executor._rust_backend.execute_typed) as native_execute:
        response = executor.execute(
            ExecuteRequest(
                dsl_text="data out; set source; keep amount; rename amount=other; output out; run;",
                inputs={"source": source},
            )
        )

    assert response.has_errors is False
    output = response.outputs_arrow["out"]
    assert output.to_pylist() == [{"other": 10}]
    assert output.schema.names == ["other"]
    native_execute.assert_called_once()


def test_one_target_collision_rejects_all_output_targets_before_native_invocation() -> None:
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"a": [1], "b": [2]}),
    )
    executor = DataStepExecutor(runtime_backend="rust")

    with patch.object(executor._rust_backend, "execute_typed", wraps=executor._rust_backend.execute_typed) as native_execute:
        response = executor.execute(
            ExecuteRequest(
                dsl_text=(
                    "data good(keep=a) bad(keep=a b rename=(a=b)); "
                    "set source; if a = 1 then output good; else output bad; run;"
                ),
                inputs={"source": source},
            )
        )

    assert response.has_errors is True
    assert response.outputs_arrow == {}
    diagnostic = next(diagnostic for diagnostic in response.diagnostics if diagnostic.severity == "error")
    assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
    assert diagnostic.stage == "validate"
    native_execute.assert_not_called()


def test_output_lineage_plan_keeps_statement_then_target_stages_isolated() -> None:
    dsl = (
        "data left(keep=id score rename=(score=left_score)) "
        "right(keep=id score rename=(score=right_score)); "
        "set source; keep id amount discard; drop discard; rename amount=score; run;"
    )
    parsed = parse_program(dsl)
    assert parsed.program is not None
    assert not parsed.has_errors
    block = parsed.program.blocks[0]
    service = ExecutionOutputHandoffService()
    target_plan = service.plan_typed(
        TypedOutputPlanRequest(block=block, source=parsed.program.source)
    )

    plan = service._build_typed_output_lineage_projection_plan(
        block=block,
        output_target_plan=target_plan,
        targets=("left", "right"),
    )

    assert isinstance(plan, OutputLineageProjectionPlan)
    assert tuple(plan.stages_by_target) == ("left", "right")
    assert plan.stages_by_target["left"] == (
        OutputProjectionStage(
            keep=("id", "amount", "discard"),
            drop=("discard",),
            rename={"amount": "score"},
        ),
        OutputProjectionStage(
            keep=("id", "score"),
            drop=(),
            rename={"score": "left_score"},
        ),
    )
    assert plan.stages_by_target["right"] == (
        OutputProjectionStage(
            keep=("id", "amount", "discard"),
            drop=("discard",),
            rename={"amount": "score"},
        ),
        OutputProjectionStage(
            keep=("id", "score"),
            drop=(),
            rename={"score": "right_score"},
        ),
    )
    assert plan.stages_by_target["left"] is not plan.stages_by_target["right"]
    assert plan.stages_by_target["left"][0] is not plan.stages_by_target["right"][0]


def test_output_lineage_plan_copies_and_freezes_nested_mappings() -> None:
    rename_map = {"amount": "score"}
    stages_by_target = {
        "out": (
            OutputProjectionStage(keep=["amount"], drop=[], rename=rename_map),
        )
    }
    plan = OutputLineageProjectionPlan(stages_by_target=stages_by_target)

    rename_map["amount"] = "changed"
    stages_by_target["out"] = ()

    assert plan.stages_by_target["out"][0].rename == {"amount": "score"}
    with pytest.raises(TypeError):
        plan.stages_by_target["new"] = ()
    with pytest.raises(TypeError):
        plan.stages_by_target["out"][0].rename["amount"] = "changed"


def test_output_lineage_plan_stays_out_of_execution_plan_and_typed_payload() -> None:
    dsl = "data out(keep=id amount rename=(amount=score)); set source; run;"
    parsed = parse_program(dsl)
    assert parsed.program is not None
    assert not parsed.has_errors
    service = ExecutionOutputHandoffService()
    target_plan = service.plan_typed(
        TypedOutputPlanRequest(block=parsed.program.blocks[0], source=parsed.program.source)
    )
    output_lineage_plan = service._build_typed_output_lineage_projection_plan(
        block=parsed.program.blocks[0],
        output_target_plan=target_plan,
        targets=("out",),
    )
    execution_plan = {
        "post_projection": {"keep": (), "drop": (), "rename": {}}
    }

    assert output_lineage_plan.stages_by_target["out"]
    assert isinstance(execution_plan, dict)
    assert "output_lineage_projection_plan" not in execution_plan
    assert all(value is not output_lineage_plan for value in execution_plan.values())

    from limulus.syntax.models import DatasetOptionSyntax
    from limulus.syntax.native_payload import serialize_block

    typed_result = parse_program(dsl)
    assert typed_result.program is not None
    serialized = serialize_block(
        typed_result.program.blocks[0],
        source=typed_result.program.source,
        output_targets=("out",),
        default_target="out",
        suppress_implicit_output=False,
        effective_output_options={"out": DatasetOptionSyntax()},
    )
    assert "OutputLineageProjectionPlan" not in serialized
    assert "stages_by_target" not in serialized
    assert "source_metadata_lineage" not in serialized


def test_execute_delegates_output_finalization_to_handoff_service() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"id": [1], "amount": [10]}),
    )
    dsl = "data out(keep=id amount rename=(amount=score)); set source; output out; run;"

    with patch.object(
        executor._output_handoff_service,
        "finalize_typed",
        wraps=executor._output_handoff_service.finalize_typed,
    ) as finalization:
        response = executor.execute(ExecuteRequest(dsl_text=dsl, inputs={"source": source}))

    assert response.has_errors is False
    assert finalization.call_count == 1
    request = finalization.call_args.args[0]
    assert isinstance(request, TypedOutputFinalizeRequest)
    assert request.output_target_plan.targets == ("out",)


def test_output_lineage_projector_composes_stages_and_preserves_schema_contract() -> None:
    lineage = SourceMetadataLineage(
        ordered_sources=(
            SourceMetadataSnapshot(
                source_ordinal=3,
                source_name="source",
                schema=pa.schema(
                    [
                        pa.field("id", pa.int64(), nullable=False, metadata={b"label": b"Identifier"}),
                        pa.field("amount", pa.int64(), nullable=False, metadata={b"label": b"Amount"}),
                        pa.field("discard", pa.string(), metadata={b"label": b"Discard"}),
                    ],
                    metadata={b"memlabel": b"Source"},
                ),
                dataset_ref_metadata={"origin": "source", "ordinal": 3},
            ),
        )
    )
    stages = (
        OutputProjectionStage(
            keep=("id", "amount", "discard"),
            drop=("discard",),
            rename={"amount": "score"},
        ),
        OutputProjectionStage(
            keep=("id", "score"),
            drop=(),
            rename={"score": "renamed_score"},
        ),
    )

    result = project_lineage_for_output(
        lineage,
        stages=stages,
        final_columns=("id", "renamed_score"),
    )

    assert result.invariant_error is None
    assert result.lineage is not None
    snapshot = result.lineage.ordered_sources[0]
    assert snapshot.source_ordinal == 3
    assert snapshot.source_name == "source"
    assert snapshot.dataset_ref_metadata == {"origin": "source", "ordinal": 3}
    assert snapshot.schema.names == ["id", "renamed_score"]
    assert snapshot.schema.types == [pa.int64(), pa.int64()]
    assert snapshot.schema.field("id").nullable is False
    assert snapshot.schema.field("renamed_score").nullable is False
    assert snapshot.schema.field("id").metadata == {b"label": b"Identifier"}
    assert snapshot.schema.field("renamed_score").metadata == {b"label": b"Amount"}
    assert snapshot.schema.metadata == {b"memlabel": b"Source"}


def test_output_lineage_projector_reports_only_validated_canonical_duplicates() -> None:
    lineage = SourceMetadataLineage(
        ordered_sources=(
            SourceMetadataSnapshot(
                source_ordinal=0,
                source_name="source",
                schema=pa.schema(
                    [
                        pa.field("a", pa.int64(), metadata={b"label": b"A"}),
                        pa.field("b", pa.int64(), metadata={b"label": b"B"}),
                    ],
                    metadata={b"memlabel": b"Source"},
                ),
            ),
        )
    )

    result = project_lineage_for_output(
        lineage,
        stages=(OutputProjectionStage(keep=(), drop=(), rename={"a": "b"}),),
        final_columns=("b",),
    )

    assert result.lineage is None
    assert result.invariant_error is not None
    assert "duplicate" in result.invariant_error.lower()
