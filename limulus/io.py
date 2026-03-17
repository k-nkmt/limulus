from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .io_adapters import DataFrameAdapterPandas, DataInputAdapterArrow, DataOutputAdapterArrow, InputSpec, OutputSpec
from .models import DataSetRef, Diagnostic, ExecuteResponse, OutputConversionResult


def _apply_arrow_metadata(table: Any, metadata: Mapping[str, Any]) -> Any:
    if not metadata:
        return table

    dataset_label = metadata.get("memlabel") or metadata.get("dataset_label")
    column_labels = metadata.get("column_labels") or {}
    if not dataset_label and not column_labels:
        return table

    schema = table.schema
    schema_metadata = dict(schema.metadata or {})
    if dataset_label:
        schema_metadata[b"memlabel"] = str(dataset_label).encode("utf-8")

    fields = []
    for field in schema:
        field_metadata = dict(field.metadata or {})
        label = column_labels.get(field.name)
        if label:
            field_metadata[b"label"] = str(label).encode("utf-8")
        fields.append(field.with_metadata(field_metadata or None))

    import pyarrow as pa

    updated_schema = pa.schema(fields, metadata=schema_metadata or None)
    return pa.Table.from_arrays([table.column(index) for index in range(table.num_columns)], schema=updated_schema)


class ExecutorIOService:
    def __init__(
        self,
        *,
        arrow_input: DataInputAdapterArrow,
        arrow_output: DataOutputAdapterArrow,
        pandas_adapter: DataFrameAdapterPandas,
        supported_formats: Sequence[str],
    ) -> None:
        self._arrow_input = arrow_input
        self._arrow_output = arrow_output
        self._pandas_adapter = pandas_adapter
        self._supported_formats = tuple(supported_formats)

    def convert_outputs(self, response: ExecuteResponse, target_format: str) -> OutputConversionResult:
        normalized = target_format.strip().lower()
        if normalized not in {"arrow_table", "pylist", "pandas"}:
            return OutputConversionResult(
                diagnostics=(
                    Diagnostic(
                        code="CONVERT_OUTPUT_FORMAT_UNSUPPORTED",
                        severity="error",
                        message=f"Unsupported output conversion format: {target_format}",
                    ),
                )
            )

        converted: dict[str, Any] = {}
        diagnostics: list[Diagnostic] = []

        for dataset_name, dataset_ref in response.outputs.items():
            value, error = self.convert_single_output(
                dataset_name=dataset_name,
                dataset_ref=dataset_ref,
                outputs_arrow=response.outputs_arrow,
                target_format=normalized,
            )
            if error is not None:
                diagnostics.append(error)
                continue
            converted[dataset_name] = value

        return OutputConversionResult(outputs=converted, diagnostics=tuple(diagnostics))

    def dataset_ref_to_arrow_table(
        self,
        dataset_name: str,
        dataset_ref: DataSetRef,
    ) -> tuple[Any | None, Diagnostic | None]:
        try:
            if dataset_ref.kind == "arrow_table" and hasattr(dataset_ref.payload, "to_pylist"):
                return _apply_arrow_metadata(dataset_ref.payload, dataset_ref.metadata), None

            if isinstance(dataset_ref.payload, Sequence) and not isinstance(dataset_ref.payload, (str, bytes, bytearray)):
                stored = self._arrow_output.store(
                    dataset_ref.payload,
                    OutputSpec(format="arrow_table", location=f"memory://{dataset_name}"),
                )
                return _apply_arrow_metadata(stored.payload, dataset_ref.metadata), None

            return None, Diagnostic(
                code="CONVERT_OUTPUT_FAILED",
                severity="error",
                location=f"dataset:{dataset_name}",
                message=f"Unable to convert dataset '{dataset_name}' to arrow_table.",
            )
        except Exception as error:
            return None, Diagnostic(
                code="CONVERT_OUTPUT_FAILED",
                severity="error",
                location=f"dataset:{dataset_name}",
                message=f"Output conversion failed for '{dataset_name}': {error}",
            )

    def convert_single_output(
        self,
        *,
        dataset_name: str,
        dataset_ref: DataSetRef,
        outputs_arrow: Mapping[str, Any],
        target_format: str,
    ) -> tuple[Any, Diagnostic | None]:
        try:
            if target_format == "arrow_table":
                arrow_table = outputs_arrow.get(dataset_name)
                if arrow_table is not None:
                    return arrow_table, None

                return self.dataset_ref_to_arrow_table(dataset_name, dataset_ref)

            if target_format == "pylist":
                if isinstance(dataset_ref.payload, Sequence) and not isinstance(dataset_ref.payload, (str, bytes, bytearray)):
                    if all(isinstance(row, Mapping) for row in dataset_ref.payload):
                        return [dict(row) for row in dataset_ref.payload], None

                arrow_table = outputs_arrow.get(dataset_name)
                if arrow_table is not None and hasattr(arrow_table, "to_pylist"):
                    return [dict(row) for row in arrow_table.to_pylist()], None

                if dataset_ref.kind == "arrow_table" and hasattr(dataset_ref.payload, "to_pylist"):
                    return [dict(row) for row in dataset_ref.payload.to_pylist()], None

                if dataset_ref.kind == "pandas" and hasattr(dataset_ref.payload, "to_dict"):
                    return [dict(row) for row in dataset_ref.payload.to_dict(orient="records")], None

                return None, Diagnostic(
                    code="CONVERT_OUTPUT_FAILED",
                    severity="error",
                    location=f"dataset:{dataset_name}",
                    message=f"Unable to convert dataset '{dataset_name}' to pylist.",
                )

            pylist_value, pylist_error = self.convert_single_output(
                dataset_name=dataset_name,
                dataset_ref=dataset_ref,
                outputs_arrow=outputs_arrow,
                target_format="pylist",
            )
            if pylist_error is not None:
                return None, pylist_error
            return self._pandas_adapter.from_canonical(pylist_value), None
        except Exception as error:
            return None, Diagnostic(
                code="CONVERT_OUTPUT_FAILED",
                severity="error",
                location=f"dataset:{dataset_name}",
                message=f"Output conversion failed for '{dataset_name}': {error}",
            )

    def resolve_inputs(
        self,
        *,
        ast_statements: Sequence[Any],
        explicit_inputs: Mapping[str, DataSetRef],
        available_inputs: Mapping[str, DataSetRef] | None,
        registered_tables: Mapping[str, DataSetRef],
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        resolved_inputs = dict(explicit_inputs)
        diagnostics: list[Diagnostic] = []
        available = available_inputs or {}

        source_statement = next(
            (
                statement
                for statement in ast_statements
                if statement.kind in {"SET", "MERGE"}
            ),
            None,
        )
        if source_statement is None:
            return resolved_inputs, diagnostics

        source_input_names = [ref.name for ref in source_statement.dataset_refs]
        if not source_input_names:
            source_tokens = source_statement.text.split()
            if len(source_tokens) >= 2:
                source_input_names = [source_tokens[1]]

        for input_name in source_input_names:
            explicit_match = self.resolve_dataset_alias(resolved_inputs, input_name)
            if explicit_match is not None:
                resolved_inputs[input_name] = explicit_match
                continue

            registered = self.resolve_dataset_alias(registered_tables, input_name)
            if registered is not None:
                resolved_inputs[input_name] = registered
                continue

            generated = self.resolve_dataset_alias(available, input_name)
            if generated is not None:
                resolved_inputs[input_name] = generated
                continue

            diagnostics.append(
                Diagnostic(
                    code="RUNTIME_SET_DATASET_NOT_FOUND",
                    severity="error",
                    message=f"Input dataset is not provided: {input_name}",
                )
            )
            return {}, diagnostics

        return resolved_inputs, diagnostics

    def load_input_rows(self, input_ref: DataSetRef) -> tuple[list[dict[str, Any]], Diagnostic | None]:
        normalized_kind = input_ref.kind.strip().lower()

        if normalized_kind == "memory":
            payload = input_ref.payload
            if payload is None:
                return [], None
            if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
                if all(isinstance(item, Mapping) for item in payload):
                    return list(payload), None
                return [], Diagnostic(
                    code="IO_MEMORY_INPUT_ERROR",
                    severity="error",
                    message="memory input payload must be a sequence of mappings.",
                )
            return [], Diagnostic(
                code="IO_MEMORY_INPUT_ERROR",
                severity="error",
                message="memory input payload must be a sequence of mappings.",
            )

        if normalized_kind == "arrow_table":
            payload = input_ref.payload
            if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
                if all(isinstance(item, Mapping) for item in payload):
                    return list(payload), None
                return [], Diagnostic(
                    code="IO_INPUT_ERROR",
                    severity="error",
                    message="arrow_table sequence payload must be a sequence of mappings.",
                )

        if normalized_kind not in self._supported_formats:
            return [], Diagnostic(
                code="CAP_UNSUPPORTED_FORMAT",
                severity="error",
                message=f"Unsupported format: {input_ref.kind}",
            )

        try:
            if normalized_kind == "arrow_table":
                return self._arrow_input.load(
                    InputSpec(format="arrow_table", payload=input_ref.payload)
                ), None
            if normalized_kind == "polars":
                try:
                    import polars as pl
                except ImportError as error:
                    return [], Diagnostic(
                        code="IO_INPUT_ERROR",
                        severity="error",
                        message="polars is required for polars input format.",
                    )
                payload = input_ref.payload
                if not isinstance(payload, pl.DataFrame):
                    return [], Diagnostic(
                        code="IO_INPUT_ERROR",
                        severity="error",
                        message="polars input requires a polars.DataFrame payload.",
                    )
                return payload.to_dicts(), None
            if normalized_kind == "pandas":
                return self._pandas_adapter.load(InputSpec(format="pandas", payload=input_ref.payload)), None
        except Exception as error:
            return [], Diagnostic(
                code="IO_INPUT_ERROR",
                severity="error",
                message=str(error),
            )

        return [], Diagnostic(
            code="CAP_UNSUPPORTED_FORMAT",
            severity="error",
            message=f"Unsupported format: {input_ref.kind}",
        )

    def iterate_input_rows(self, input_ref: DataSetRef) -> tuple[Iterable[dict[str, Any]], Diagnostic | None]:
        normalized_kind = input_ref.kind.strip().lower()

        if normalized_kind == "memory":
            payload = input_ref.payload
            if payload is None:
                return (), None
            if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
                if all(isinstance(item, Mapping) for item in payload):
                    return (dict(item) if not isinstance(item, dict) else item for item in payload), None
                return (), Diagnostic(
                    code="IO_MEMORY_INPUT_ERROR",
                    severity="error",
                    message="memory input payload must be a sequence of mappings.",
                )
            return (), Diagnostic(
                code="IO_MEMORY_INPUT_ERROR",
                severity="error",
                message="memory input payload must be a sequence of mappings.",
            )

        if normalized_kind == "arrow_table":
            payload = input_ref.payload
            if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
                if all(isinstance(item, Mapping) for item in payload):
                    return (dict(item) if not isinstance(item, dict) else item for item in payload), None
                return (), Diagnostic(
                    code="IO_INPUT_ERROR",
                    severity="error",
                    message="arrow_table sequence payload must be a sequence of mappings.",
                )

            if hasattr(payload, "to_batches"):
                def _iter_arrow_batches() -> Iterable[dict[str, Any]]:
                    for batch in payload.to_batches(max_chunksize=4096):
                        for row in batch.to_pylist():
                            yield row

                return _iter_arrow_batches(), None

        rows, load_error = self.load_input_rows(input_ref)
        if load_error is not None:
            return (), load_error
        return rows, None

    def resolve_output_targets(
        self,
        *,
        ast_statements: Sequence[Any],
        explicit_output_targets: Sequence[str],
    ) -> tuple[tuple[str, ...], list[Diagnostic]]:
        diagnostics: list[Diagnostic] = []

        if explicit_output_targets:
            return tuple(self.normalize_dataset_name(target) for target in explicit_output_targets), diagnostics

        inferred = self.merge_unique_targets(
            self.extract_data_targets(ast_statements),
            self.extract_output_targets(ast_statements),
        )
        if inferred:
            return inferred, diagnostics

        diagnostics.append(
            Diagnostic(
                code="REQ_INVALID_OUTPUT_TARGETS",
                severity="error",
                message="output_targets is omitted and no DATA/OUTPUT target can be resolved.",
            )
        )
        return (), diagnostics

    def extract_output_targets(self, ast_statements: Sequence[Any]) -> tuple[str, ...]:
        targets: list[str] = []
        for statement in ast_statements:
            if statement.kind != "OUTPUT":
                continue
            tokens = statement.text.split()
            if len(tokens) > 1:
                targets.append(self.normalize_dataset_name(tokens[1]))
        return tuple(targets)

    def extract_data_targets(self, ast_statements: Sequence[Any]) -> tuple[str, ...]:
        statement = next((item for item in ast_statements if item.kind == "DATA"), None)
        if statement is None:
            return ()
        output_refs = getattr(statement, "output_refs", ())
        if output_refs:
            return tuple(self.normalize_dataset_name(ref.name) for ref in output_refs if getattr(ref, "name", ""))

        dataset_refs = getattr(statement, "dataset_refs", ())
        if dataset_refs:
            return tuple(self.normalize_dataset_name(ref.name) for ref in dataset_refs if getattr(ref, "name", ""))

        tokens = statement.text.split()
        if len(tokens) <= 1:
            return ()
        return tuple(self.normalize_dataset_name(token) for token in tokens[1:] if token)

    def merge_unique_targets(self, *target_groups: Sequence[str]) -> tuple[str, ...]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in target_groups:
            for target in group:
                normalized_target = self.normalize_dataset_name(target)
                target_key = self.dataset_name_key(normalized_target)
                if not normalized_target or target_key in seen:
                    continue
                merged.append(normalized_target)
                seen.add(target_key)
        return tuple(merged)

    def normalize_dataset_name(self, name: str) -> str:
        normalized = name.strip()
        if not normalized:
            return normalized
        lowered = normalized.lower()
        if lowered.startswith("work."):
            return normalized[5:]
        return normalized

    def dataset_name_key(self, name: str) -> str:
        return self.normalize_dataset_name(name).lower()

    def resolve_dataset_alias(
        self,
        datasets: Mapping[str, DataSetRef],
        requested_name: str,
    ) -> DataSetRef | None:
        normalized_requested = self.dataset_name_key(requested_name)
        for candidate_name, dataset in datasets.items():
            if self.dataset_name_key(candidate_name) == normalized_requested:
                return dataset
        return None

    def resolve_declared_output_target(
        self,
        requested_target: str,
        declared_targets: Sequence[str],
    ) -> str:
        requested_key = self.dataset_name_key(requested_target)
        for declared in declared_targets:
            if self.dataset_name_key(declared) == requested_key:
                return declared
        return self.normalize_dataset_name(requested_target)
