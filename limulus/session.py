"""Public session and dataset-view APIs for limulus workflows.

This module owns the user-facing session catalog, DATA step submission entry
points, and the dataset-scoped helper methods that are surfaced in the API docs.
"""

from __future__ import annotations

from pathlib import Path
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pyarrow as pa
import polars as pl

from .naming import _column_key, _dataset_key
from .arrow_bridge import (
    materialize_rebuilt_table,
    polars_result_to_arrow,
    preserve_arrow_metadata,
)
from .column_api import (
    assign_columns as apply_assignment_columns,
    resolve_column_name as resolve_column_api_name,
    resolve_existing_columns,
    resolve_transpose_var_columns,
    transpose_table as build_transposed_table,
)
from .execution import DataStepExecutor
from .format_registry import FormatRegistry
from .session_parsing import classify_sql, parse_simple_filter, raise_session_sql_execution_error
from .models import (
    DatasetCatalog,
    ExecuteRequest,
    LogEntry,
    SubmitResult,
)


_DICTIONARY_TABLES_SCHEMA = pa.schema(
    [
        pa.field("LIBNAME", pa.string()),
        pa.field("MEMNAME", pa.string()),
        pa.field("MEMTYPE", pa.string()),
        pa.field("MEMLABEL", pa.string()),
        pa.field("NOBS", pa.int64()),
        pa.field("NVAR", pa.int64()),
    ]
)
_DICTIONARY_COLUMNS_SCHEMA = pa.schema(
    [
        pa.field("LIBNAME", pa.string()),
        pa.field("MEMNAME", pa.string()),
        pa.field("MEMTYPE", pa.string()),
        pa.field("NAME", pa.string()),
        pa.field("TYPE", pa.string()),
        pa.field("VARNUM", pa.int64()),
        pa.field("LABEL", pa.string()),
        pa.field("FORMAT", pa.string()),
        pa.field("INFORMAT", pa.string()),
    ]
)


def _empty_table(schema: pa.Schema) -> pa.Table:
    arrays = [pa.array([], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def _decode_metadata(metadata: Mapping[bytes, bytes] | None, key: bytes) -> str:
    if metadata is None:
        return ""
    value = metadata.get(key, b"")
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


class DictionaryProvider:
    def __init__(self, catalog: DatasetCatalog) -> None:
        self._catalog = catalog

    @property
    def tables(self) -> pa.Table:
        rows: list[dict[str, Any]] = []
        for name in self._catalog:
            table = self._catalog[name]
            rows.append(
                {
                    "LIBNAME": "WORK",
                    "MEMNAME": _dataset_key(name),
                    "MEMTYPE": "DATA",
                    "MEMLABEL": _decode_metadata(table.schema.metadata, b"memlabel"),
                    "NOBS": table.num_rows,
                    "NVAR": table.num_columns,
                }
            )
        return pa.Table.from_pylist(rows, schema=_DICTIONARY_TABLES_SCHEMA) if rows else _empty_table(_DICTIONARY_TABLES_SCHEMA)

    @property
    def columns(self) -> pa.Table:
        rows: list[dict[str, Any]] = []
        for name in self._catalog:
            table = self._catalog[name]
            rows.extend(self._column_rows(name, table))
        return pa.Table.from_pylist(rows, schema=_DICTIONARY_COLUMNS_SCHEMA) if rows else _empty_table(_DICTIONARY_COLUMNS_SCHEMA)

    def __call__(self, name: str) -> pa.Table:
        table = self._catalog[name]
        rows = self._column_rows(name, table)
        return pa.Table.from_pylist(rows, schema=_DICTIONARY_COLUMNS_SCHEMA) if rows else _empty_table(_DICTIONARY_COLUMNS_SCHEMA)

    @staticmethod
    def _column_rows(name: str, table: pa.Table) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for varnum, field in enumerate(table.schema, start=1):
            rows.append(
                {
                    "LIBNAME": "WORK",
                    "MEMNAME": _dataset_key(name),
                    "MEMTYPE": "DATA",
                    "NAME": field.name,
                    "TYPE": str(field.type),
                    "VARNUM": varnum,
                    "LABEL": _decode_metadata(field.metadata, b"label"),
                    "FORMAT": "",
                    "INFORMAT": "",
                }
            )
        return rows


class DatasetView:
    """A view for chained operations on a dataset.

    Obtained via :meth:`Session.dataset`. You can chain :meth:`select` / :meth:`keep` / :meth:`drop` /
    :meth:`where` / :meth:`rename` / :meth:`sort` calls.

    Examples:
        >>> view = session.dataset("bmi")
        >>> df = view.where("bmi > 20").keep(["name", "bmi"]).to_pandas()
    """

    def __init__(self, session: "Session", name: str) -> None:
        self._session = session
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def dictionary(self) -> pa.Table:
        return self._session.dictionary(self._name)

    def to_arrow(self):
        """Returns this dataset as a ``pyarrow.Table``."""
        return self._session.to_arrow(self._name)

    def to_pandas(self):
        """Converts this dataset to a ``pandas.DataFrame`` and returns it."""
        return self._session.to_pandas(self._name)

    def to_polars(self):
        """Converts this dataset to a ``polars.DataFrame`` and returns it."""
        return self._session.to_polars(self._name)

    def select(self, columns: Sequence[str], out: str | None = None) -> "DatasetView":
        """Retains only the specified columns.

        Args:
            columns: List of column names to keep.
            out: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = self._session._resolve_output_name(self._name, out=out)
        self._session.select(self._name, columns, out=output_name)
        return DatasetView(self._session, output_name)

    def keep(self, columns: Sequence[str], out: str | None = None) -> "DatasetView":
        """Retains only the specified columns (equivalent to KEEP).

        Args:
            columns: List of column names to keep.
            out: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        return self.select(columns, out=out)

    def drop(self, columns: Sequence[str], out: str | None = None) -> "DatasetView":
        """Removes the specified columns (equivalent to DROP).

        Args:
            columns: List of column names to remove.
            out: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = self._session._resolve_output_name(self._name, out=out)
        self._session.drop(self._name, columns, out=output_name)
        return DatasetView(self._session, output_name)

    def where(self, expression: str, out: str | None = None) -> "DatasetView":
        """Filters rows using a simple comparison expression (equivalent to WHERE).

        Args:
            expression: Filter expression, e.g. ``"age > 13"``, ``"sex = 'M'"``.
            out: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = self._session._resolve_output_name(self._name, out=out)
        self._session.where(self._name, expression, out=output_name)
        return DatasetView(self._session, output_name)

    def rename(self, mapping: Mapping[str, str], out: str | None = None) -> "DatasetView":
        """Renames columns (equivalent to RENAME).

        Args:
            mapping: A ``{old_name: new_name}`` mapping dict.
            out: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = self._session._resolve_output_name(self._name, out=out)
        self._session.rename(self._name, mapping, out=output_name)
        return DatasetView(self._session, output_name)

    def astype(
        self,
        mapping: Mapping[str, str],
        out: str | None = None,
        *,
        alias: str | Mapping[str, str] | None = None,
    ) -> "DatasetView":
        """Casts columns using a ``{column: dtype}`` mapping.

        Args:
            mapping: Mapping from column name to dtype string.
            out: Output dataset name. If omitted, overwrites this view's dataset.
            alias: Optional new column name, or ``{source: alias}`` mapping, for
                writing cast results into new columns instead of overwriting the source.

        Returns:
            A :class:`DatasetView` for the resulting dataset.

        Note:
            Column names are resolved case-insensitively when the match is unique.
        """
        output_name = self._session._resolve_output_name(self._name, out=out)
        self._session._astype_dataset(self._name, mapping, out=output_name, alias=alias)
        return DatasetView(self._session, output_name)

    def cast(
        self,
        mapping: Mapping[str, str],
        out: str | None = None,
        *,
        alias: str | Mapping[str, str] | None = None,
    ) -> "DatasetView":
        """Alias for :meth:`astype`."""
        return self.astype(mapping, out=out, alias=alias)

    def apply_options(
        self,
        *,
        keep: Sequence[str] | None = None,
        drop: Sequence[str] | None = None,
        rename: Mapping[str, str] | None = None,
        where: str | None = None,
        out: str | None = None,
    ) -> "DatasetView":
        """Applies ``keep``/``drop``/``rename``/``where`` in a single call.

        Args:
            keep: List of column names to retain.
            drop: List of column names to remove.
            rename: A ``{old_name: new_name}`` mapping dict.
            where: Filter expression (e.g. ``"age > 13"``).
            out: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = self._session._resolve_output_name(self._name, out=out)
        self._session.apply_dataset_options(
            self._name,
            keep=keep,
            drop=drop,
            rename=rename,
            where=where,
            out=output_name,
        )
        return DatasetView(self._session, output_name)
    
    def sort(
        self,
        by: Sequence[str],
        out: str | None = None,
        *,
        nodupkey: bool = False,
    ) -> "DatasetView":
        """Sorts the dataset by columns.

        Args:
            by: List of column names to sort by. Can also be a list of
                ``(column_name, "ascending" | "descending")`` tuples to specify direction.
            out: Output dataset name. If omitted, overwrites this view's dataset.
            nodupkey: When ``True``, keeps only the first row for each unique key
                defined by ``by`` after sorting.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = self._session._resolve_output_name(self._name, out=out)
        self._session.sort(self._name, by, out=output_name, nodupkey=nodupkey)
        return DatasetView(self._session, output_name)

    def transpose(
        self,
        *,
        by: str | Sequence[str] | None = None,
        id: str | None = None,
        var: str | Sequence[str] | None = None,
        out: str | None = None,
    ) -> "DatasetView":
        """Transposes the dataset using PROC TRANSPOSE-like defaults.

        Args:
            by: Grouping columns preserved in the output.
            id: Single column name expanded into output column names.
            var: Value columns to transpose. When omitted, uses all non-``by``/``id`` columns.
            out: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.

        Note:
            This helper is intended as a convenience API with PROC TRANSPOSE-like defaults.
            Performance is not a primary goal. For performance-critical reshaping, prefer
            Arrow or Polars directly.
        """
        output_name = self._session._resolve_output_name(self._name, out=out)
        self._session.transpose(self._name, by=by, id=id, var=var, out=output_name)
        return DatasetView(self._session, output_name)

    def assign(self, out: str | None = None, **assignments: Any) -> "DatasetView":
        """Adds or replaces columns using ordered expressions or literals.

        Args:
            out: Output dataset name. If omitted, overwrites this view's dataset.
            **assignments: Column assignments evaluated from left to right. String values are
                interpreted as expressions. Non-string values are treated as literals.

        Returns:
            A :class:`DatasetView` for the resulting dataset.

        Note:
            Assignments are translated to ordered Polars expressions and evaluated from
            left to right. Expressions or functions that cannot be translated to the
            column-oriented assign pipeline raise ``ValueError`` instead of falling back
            to Python row materialization.
        """
        output_name = self._session._resolve_output_name(self._name, out=out)
        self._session.assign(self._name, out=output_name, **assignments)
        return DatasetView(self._session, output_name)

    def unload(self, *, missing_ok: bool = True) -> bool:
        """Removes this dataset from the session.

        Args:
            missing_ok: If ``False``, raises ``KeyError`` when the dataset does not exist.

        Returns:
            ``True`` if the dataset was successfully removed.
        """
        return self._session.unload(self._name, missing_ok=missing_ok)


class Session:
    def __init__(
        self,
        *,
        backend: str = "auto",
        runtime_backend: str | None = None,
        parser_backend: str = "python",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        """Creates a limulus execution session.

        A session holds a dataset catalog and execution engine, sharing datasets
        across multiple :meth:`submit` calls.

        Args:
            backend: Preferred backend. One of ``"auto"`` (recommended), ``"python"``, or ``"rust"``.
                With ``"auto"``, limulus prefers the Rust runtime when the current workload is supported and falls back to Python otherwise.
            runtime_backend: Explicit override for runtime selection. Takes precedence over ``backend``.
            parser_backend: Parser backend. Currently only ``"python"`` (lark) is stable.
            options: Session-level option dictionary. These attributes are currently reserved for
                future use, but are propagated with each submit request.

        Examples:
            >>> import limulus
            >>> session = limulus.Session()
            >>> session = limulus.Session(backend="python")
        """
        selected_runtime_backend = runtime_backend or backend
        self._format_registry = FormatRegistry()
        self._executor = DataStepExecutor(
            runtime_backend=selected_runtime_backend,
            parser_backend=parser_backend,
            format_registry=self._format_registry,
        )
        self._datasets = DatasetCatalog()
        self._dictionary_provider = DictionaryProvider(self._datasets)
        self._last_submit_result: SubmitResult | None = None
        self._options: dict[str, Any] = dict(options or {})

    def register_format(self, name: str, formatter: Any, *, namespace: str | None = None) -> None:
        self._format_registry.register_format(name, formatter, namespace=namespace)

    def register_informat(self, name: str, parser: Any, *, kind: str | None = None) -> None:
        self._format_registry.register_informat(name, parser, kind=kind)

    def load(self, name: str, data: Any) -> None:
        """Registers a single dataset in the session catalog.

        Args:
            name: Dataset name referenced from Data Step code (case-insensitive).
            data: Data to register. Supported types:

                - ``pyarrow.Table``
                - ``polars.DataFrame``
                - ``pandas.DataFrame``
                - ``list[dict]`` (list of dicts; normalized to Arrow-backed storage)

        Examples:
            >>> import limulus, pyarrow as pa
            >>> session = limulus.Session()
            >>> session.load("mydata", pa.table({"x": [1, 2, 3]}))
        """
        if self._is_dictionary_reserved_name(name):
            raise ValueError(f"Cannot overwrite dictionary table: {name}")
        dataset = self._to_registerable_dataset(data)
        self._executor.register_tables({name: dataset})
        self._datasets.set(name, self._to_arrow(dataset))

    def loads(self, datasets: Mapping[str, Any] | None = None, **named_datasets: Any) -> None:
        """Registers multiple datasets at once.

        Args:
            datasets: A mapping of dataset names to data.
            **named_datasets: Can also be passed as keyword arguments (compatible with ``datasets``).

        Examples:
            >>> session.loads({"a": df_a, "b": df_b})
            >>> session.loads(a=df_a, b=df_b)  # also accepted as keyword arguments
        """
        merged: dict[str, Any] = {}
        if datasets:
            merged.update(datasets)
        if named_datasets:
            merged.update(named_datasets)
        for name, data in merged.items():
            self.load(name, data)

    def unload(self, *names: str | Sequence[str], missing_ok: bool = True) -> bool:
        """Removes one or more datasets from the session.

        Args:
            *names: Dataset name(s) to remove. Accepts strings or sequences,
                e.g. ``"a", "b"`` or ``["a", "b"]``.
            missing_ok: If ``False``, raises ``KeyError`` for datasets that do not exist.
                Defaults to ``True`` (silently ignores missing datasets).

        Returns:
            ``True`` if at least one dataset was removed.

        Examples:
            >>> session.unload("tmp")
            >>> session.unload("a", "b")
            >>> session.unload(["a", "b"])
        """
        removed_any = False
        for name in self._iter_dataset_names(names):
            removed_catalog = self._datasets.delete(name, missing_ok=missing_ok)
            removed_registered = self._executor.unregister_table(name, missing_ok=missing_ok)
            removed_any = removed_any or removed_catalog or removed_registered
        return removed_any

    def delete(self, *names: str | Sequence[str], missing_ok: bool = True) -> bool:
        """Alias for :meth:`unload`."""
        return self.unload(*names, missing_ok=missing_ok)

    def dataset(self, name: str) -> DatasetView:
        """Returns an operation view for the specified dataset.

        Use this when you want to chain :meth:`keep` / :meth:`drop` / :meth:`where` /
        :meth:`rename` / :meth:`sort` calls.

        Args:
            name: Name of the dataset to create a view for.

        Returns:
            A :class:`DatasetView` instance.

        Examples:
            >>> df = session.dataset("bmi").where("bmi > 20").keep(["name", "bmi"]).to_pandas()
        """
        return DatasetView(self, name)

    def submit(
        self,
        code: str,
        *,
        backend: str | None = None,
        show_result: bool = False,
    ) -> SubmitResult:
        """Submits Data Step code to the session and executes it.

        Resulting datasets are added to or updated in the session catalog.
        Datasets created by a previous :meth:`submit` call can be referenced in subsequent calls.

        Args:
            code: Data Step DSL text to execute.
                Multiple ``DATA ... RUN;`` blocks may be included.
            backend: Optional runtime backend override for this call only.
                One of ``"rust"``, ``"python"``, or ``"auto"``.
                When given, temporarily overrides the session-level backend setting.
            show_result: Controls notebook/REPL representation of the returned
                :class:`~limulus.SubmitResult`. Defaults to ``False`` for
                ``Session.submit()`` to keep successful calls quiet.

        Returns:
            A :class:`~limulus.SubmitResult`.
            Check ``result.success`` to determine whether errors occurred.

        Examples:
            >>> rc = session.submit(\"\"\"
            ... data result;
            ...   set mydata;
            ...   bmi = round(weight / height**2, 0.1);
            ...   keep name bmi;
            ... run;
            ... \"\"\")
            >>> print(session["result"].to_pandas())
        """
        started = time.perf_counter()
        _original_backend: str | None = None
        if backend is not None:
            _original_backend = self._executor._runtime_backend_preference
            self._executor.set_runtime_backend(backend)
        try:
            request_options = dict(self._options)
            if self._format_registry.has_custom_registrations():
                dispatch_hints = dict(request_options.get("dispatch_hints", {}) or {})
                dispatch_hints.update(self._format_registry.dispatch_hints())
                request_options["dispatch_hints"] = dispatch_hints
                format_catalog_payload = self._format_registry.catalog_payload()
                if format_catalog_payload:
                    request_options["format_catalog_payload"] = format_catalog_payload
            response = self._executor.execute(ExecuteRequest(dsl_text=code, options=request_options))
        finally:
            if backend is not None:
                self._executor.set_runtime_backend(_original_backend)
        elapsed_seq = (time.perf_counter() - started) 

        converted = self._executor.convert_outputs(response, "arrow_table")
        self._datasets.update(converted.outputs)
        log_entries = tuple(
            LogEntry(
                code=diagnostic.code,
                severity=diagnostic.severity,
                message=diagnostic.message,
                location=diagnostic.location,
                stage=diagnostic.stage,
                span=diagnostic.span,
                labels=diagnostic.labels,
                notes=diagnostic.notes,
                source_text=diagnostic.source_text,
            )
            for diagnostic in response.diagnostics
        )

        success = not response.has_errors and not converted.has_errors
        if converted.has_errors:
            log_entries = (
                *log_entries,
                *tuple(
                    LogEntry(
                        code=diagnostic.code,
                        severity=diagnostic.severity,
                        message=diagnostic.message,
                        location=diagnostic.location,
                        stage=diagnostic.stage,
                        span=diagnostic.span,
                        labels=diagnostic.labels,
                        notes=diagnostic.notes,
                        source_text=diagnostic.source_text,
                    )
                    for diagnostic in converted.diagnostics
                ),
            )

        result = SubmitResult(
            success=success,
            datasets=dict(converted.outputs),
            log=log_entries,
            elapsed_seq=elapsed_seq,
            display_result=show_result,
        )
        self._last_submit_result = result
        if not result.success:
            result.print_log()
        return result

    def run(
        self,
        code: str,
        *,
        backend: str | None = None,
    ) -> SubmitResult:
        """Alias for :meth:`submit`."""
        return self.submit(code, backend=backend)

    @property
    def datasets(self) -> DatasetCatalog:
        """Returns the session catalog (:class:`~limulus.DatasetCatalog`).

        Supports dict-like access (``session.datasets["name"]``),
        ``in`` operator, and iteration.

        Note:
            ``session["name"]`` is syntactic sugar for ``session.datasets["name"]``.
        """
        return self._datasets

    @property
    def work(self) -> DatasetCatalog:
        """Alias for :attr:`datasets`."""
        return self._datasets

    @property
    def dictionary(self) -> DictionaryProvider:
        return self._dictionary_provider

    def __getitem__(self, name: str) -> Any:
        """Returns a dataset as a ``pyarrow.Table``.

        Args:
            name: Dataset name.

        Returns:
            ``pyarrow.Table``.

        Raises:
            KeyError: If no dataset with the given name exists.

        Examples:
            >>> table = session["result"]
            >>> df = session["result"].to_pandas()
        """
        return self._datasets[name]

    def to_pandas(self, name: str):
        """Converts a dataset to a ``pandas.DataFrame`` and returns it.

        Args:
            name: Dataset name.

        Returns:
            ``pandas.DataFrame``.
        """
        return self.to_arrow(name).to_pandas()

    def to_polars(self, name: str):
        """Converts a dataset to a ``polars.DataFrame`` and returns it.

        Args:
            name: Dataset name.

        Returns:
            ``polars.DataFrame``.
        """
        return pl.from_arrow(self.to_arrow(name))

    def to_arrow(self, name: str):
        """Returns a dataset as a ``pyarrow.Table``.

        Args:
            name: Dataset name.

        Returns:
            ``pyarrow.Table``.
        """
        return self._datasets[name]

    @property
    def log(self) -> SubmitResult | None:
        """Most recent :class:`~limulus.SubmitResult`.

        Returns ``None`` until the first :meth:`submit`/:meth:`run` call.
        """
        return self._last_submit_result

    def get_log(self) -> SubmitResult | None:
        """Most recent :class:`~limulus.SubmitResult`.

        This method is equivalent to :attr:`log` and is provided for cases where
        method-style access is preferred.

        Returns ``None`` until the first :meth:`submit`/:meth:`run` call.
        """
        return self.log

    def set_option(self, options: Mapping[str, Any]) -> Session:
        """Merges session options from a dictionary.

        These options are currently reserved for future use, but are forwarded
        to the executor on each :meth:`submit` call.
        """
        self._options.update(dict(options))
        return self

    def get_option(self, name: str | Sequence[str] | None = None, default: Any = None) -> Any:
        """Returns session options as a value or dictionary.

        When ``name`` is omitted, returns a copy of all options. When a list of
        keys is given, returns a dictionary for those keys. When a single key is
        given, returns its value or ``default``.
        """
        if name is None:
            return dict(self._options)
        if isinstance(name, str):
            return self._options.get(name, default)
        return {key: self._options.get(key, default) for key in name}

    def include(self, path: str) -> SubmitResult:
        """Reads a DSL file and executes it via :meth:`submit`.

        Args:
            path: Path to a UTF-8 encoded Data Step script.

        Returns:
            The resulting :class:`~limulus.SubmitResult`.
        """
        return self.submit(Path(path).read_text(encoding="utf-8"))


    def select(self, source: str, columns: Sequence[str], out: str | None = None, *, target: str | None = None) -> Session:
        """Column selection (alias for :meth:`keep`).

        """
        table = self.to_arrow(source)
        selected = table.select(list(resolve_existing_columns(table, columns, parameter_name="select")))
        output_name = self._resolve_output_name(source, out=out, target=target)
        self.load(output_name, selected)
        return self

    def keep(self, source: str, columns: Sequence[str], out: str | None = None, *, target: str | None = None) -> Session:
        """Retains only the specified columns from a dataset (equivalent to KEEP).

        Args:
            source: Source dataset name.
            columns: List of column names to keep.
            out: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).

        Examples:
            >>> session.keep("bmi", ["name", "sex", "bmi"])
            >>> session.keep("bmi", ["name", "bmi"], out="bmi_slim")
        """
        return self.select(source, columns, out=out, target=target)

    def rename(self, source: str, mapping: Mapping[str, str], out: str | None = None, *, target: str | None = None) -> Session:
        """Renames columns (equivalent to RENAME).

        Args:
            source: Source dataset name.
            mapping: A ``{old_name: new_name}`` mapping dict.
            out: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).

        Examples:
            >>> session.rename("bmi", {"height_m": "height_meter"})
        """
        table = self.to_arrow(source)
        resolved_mapping = {
            self._resolve_column_name(table, source_name, parameter_name="rename"): target_name
            for source_name, target_name in mapping.items()
        }
        renamed_columns = [resolved_mapping.get(name, name) for name in table.column_names]
        renamed = table.rename_columns(renamed_columns)
        output_name = self._resolve_output_name(source, out=out, target=target)
        self.load(output_name, renamed)
        return self

    def _astype_dataset(
        self,
        source: str,
        mapping: Mapping[str, str],
        out: str | None = None,
        *,
        alias: str | Mapping[str, str] | None = None,
        target: str | None = None,
    ) -> Session:
        table = self.to_arrow(source)
        frame = pl.from_arrow(table)
        resolved_mapping = {
            self._resolve_column_name(table, column_name, parameter_name="astype"): dtype
            for column_name, dtype in mapping.items()
        }
        resolved_alias = self._resolve_cast_aliases(table, resolved_mapping, alias)
        polars_dtype_mapping = {
            name: self._resolve_polars_dtype(dtype) for name, dtype in resolved_mapping.items()
        }
        if resolved_alias:
            casted_frame = frame.with_columns(
                [
                    pl.col(name).cast(polars_dtype_mapping[name]).alias(resolved_alias.get(name, name))
                    for name in resolved_mapping
                ]
            )
        else:
            casted_frame = frame.cast(polars_dtype_mapping)
        casted = polars_result_to_arrow(casted_frame, table)
        output_name = self._resolve_output_name(source, out=out, target=target)
        self.load(output_name, casted)
        return self

    def astype(
        self,
        source: str,
        mapping: Mapping[str, str],
        out: str | None = None,
        *,
        alias: str | Mapping[str, str] | None = None,
        target: str | None = None,
    ) -> Session:
        """Casts dataset columns using a ``{column: dtype}`` mapping.

        When ``alias=`` is provided, cast results are written into new columns
        instead of replacing the original source columns.
        """
        return self._astype_dataset(source, mapping, out=out, alias=alias, target=target)

    def cast(
        self,
        source: str,
        mapping: Mapping[str, str],
        out: str | None = None,
        *,
        alias: str | Mapping[str, str] | None = None,
        target: str | None = None,
    ) -> Session:
        """Alias for dataset-scoped column casting.

        This is a convenience alias for the DatasetView-style :meth:`DatasetView.astype`
        workflow and forwards to the same internal implementation.
        """
        return self._astype_dataset(source, mapping, out=out, alias=alias, target=target)

    def transpose(
        self,
        source: str,
        *,
        by: str | Sequence[str] | None = None,
        id: str | None = None,
        var: str | Sequence[str] | None = None,
        out: str | None = None,
        target: str | None = None,
    ) -> Session:
        """Transposes a dataset using a minimal PROC TRANSPOSE-like contract.

        Args:
            source: Source dataset name.
            by: Grouping columns preserved in the output.
            id: Single column name expanded into output column names.
            var: Value columns to transpose. When omitted, uses all non-``by``/``id`` columns.
            out: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).

        Note:
            ``var`` defaults to all non-``by``/``id`` columns. Without ``id``, output rows contain
            ``_NAME_`` and ``COL1..COLn``. With ``id``, the current implementation supports exactly
            one ``id`` column and one value column. Duplicate or missing ``id`` values within a
            group raise ``ValueError``. This helper is a convenience API rather than a
            performance-oriented transpose implementation.
        """
        table = self.to_arrow(source)
        by_columns = resolve_existing_columns(table, by, parameter_name="by")
        if id is None:
            id_columns: tuple[str, ...] = ()
        else:
            if not isinstance(id, str):
                raise TypeError("Session.transpose id must be a single column name")
            id_columns = (self._resolve_column_name(table, id, parameter_name="id"),)
        var_columns = resolve_transpose_var_columns(table, by_columns=by_columns, id_columns=id_columns, var=var)

        transposed = build_transposed_table(
            table,
            by_columns=by_columns,
            id_columns=id_columns,
            var_columns=var_columns,
            materialize_table=materialize_rebuilt_table,
        )
        output_name = self._resolve_output_name(source, out=out, target=target)
        self.load(output_name, transposed)
        return self

    def assign(self, source: str, out: str | None = None, *, target: str | None = None, **assignments: Any) -> Session:
        """Adds or replaces columns using ordered Data Step-style assignments.

        Args:
            source: Source dataset name.
            out: Output dataset name. If omitted, overwrites ``source``.
            **assignments: Column assignments evaluated from left to right. String values are
                interpreted as expressions. Non-string values are treated as literals.

        Returns:
            ``self`` (for method chaining).

        Note:
            Within one call, assignments are evaluated from left to right, and later expressions can
            reference columns created earlier in the same call. Supported assignment syntax is
            translated into ordered Polars expressions; unsupported constructs raise ``ValueError``
            instead of silently falling back to Python row materialization.
        """
        table = self.to_arrow(source)
        updated = apply_assignment_columns(
            table,
            assignments,
            finalize_table=preserve_arrow_metadata,
            format_registry=self._format_registry,
        )
        output_name = self._resolve_output_name(source, out=out, target=target)
        self.load(output_name, updated)
        return self

    def sql(self, query: str, out: str | None = None, *, target: str | None = None):
        """Executes SQL against session datasets.

        Args:
            query: SQL text executed against datasets currently loaded in the session.
            out: Optional explicit output dataset name.

        Returns:
            A ``pyarrow.Table`` containing the query result.

        SQL execution is backed by the Polars SQL engine. 
        https://docs.pola.rs/api/python/stable/reference/sql/index.html
        If the SQL starts with ``CREATE TABLE name AS ...``, the result is also 
        stored in the session catalog under ``name``. 
        """
        classification = classify_sql(query)
        if classification.kind == "drop_table":
            self.unload(classification.target, missing_ok=False)
            return None

        if classification.target is not None and self._is_dictionary_reserved_name(classification.target):
            raise ValueError(f"Cannot overwrite dictionary table: {classification.target}")

        source_tables = {name: self.to_arrow(name) for name in self._datasets}
        context = self._build_sql_context()
        output_name = self._resolve_output_name(classification.target, out=out, target=target)
        try:
            result = context.execute(classification.query)
            table = polars_result_to_arrow(result, *source_tables.values())
        except Exception as error:
            raise_session_sql_execution_error(classification.query, error)
        if output_name is not None:
            self.load(output_name, table)
        return table

    def filter(self, source: str, expression: str, out: str | None = None, *, target: str | None = None) -> Session:
        """Row filtering (alias for :meth:`where`).

        """
        filter_spec = parse_simple_filter(source, expression)

        import pyarrow as pa
        import pyarrow.compute as pc

        table = self.to_arrow(source)
        variable_name = self._resolve_column_name(table, filter_spec.variable_name, parameter_name="filter")

        column = table[variable_name]
        scalar = pa.scalar(filter_spec.scalar_value)
        if filter_spec.operator == ">":
            mask = pc.greater(column, scalar)
        elif filter_spec.operator == ">=":
            mask = pc.greater_equal(column, scalar)
        elif filter_spec.operator == "<":
            mask = pc.less(column, scalar)
        elif filter_spec.operator == "<=":
            mask = pc.less_equal(column, scalar)
        elif filter_spec.operator in {"=", "=="}:
            mask = pc.equal(column, scalar)
        else:
            mask = pc.not_equal(column, scalar)

        filtered = table.filter(mask)
        output_name = self._resolve_output_name(source, out=out, target=target)
        self.load(output_name, filtered)
        return self

    def where(self, source: str, expression: str, out: str | None = None, *, target: str | None = None) -> Session:
        """Filters rows using a simple comparison expression (equivalent to WHERE).

        Supported expression format: ``"column op value"`` (e.g. ``"age > 13"``, ``"sex = 'M'``").

        Args:
            source: Source dataset name.
            expression: Filter expression. 
            out: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).

        Raises:
            ValueError: If the expression format is invalid.

        Examples:
            >>> session.where("class", "age > 13")
            >>> session.where("class", "sex = 'M'", out="male")
        """
        return self.filter(source, expression, out=out, target=target)

    def drop(self, source: str, columns: Sequence[str], out: str | None = None, *, target: str | None = None) -> Session:
        """Removes the specified columns from a dataset (equivalent to DROP).

        Args:
            source: Source dataset name.
            columns: List of column names to remove.
            out: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).
        """
        table = self.to_arrow(source)
        drop_columns = set(resolve_existing_columns(table, columns, parameter_name="drop"))
        keep_columns = [name for name in table.column_names if name not in drop_columns]
        return self.select(source, keep_columns, out=out, target=target)
    
    def sort(
        self,
        source: str,
        by: Sequence[str],
        out: str | None = None,
        *,
        nodupkey: bool = False,
        target: str | None = None,
    ) -> Session:
        """Sorts a dataset by columns.

        Args:
            source: Source dataset name.
            by: List of column names to sort by. Can also be a list of
                ``(column_name, "ascending" | "descending")`` tuples to specify direction.
            out: Output dataset name. If omitted, overwrites ``source``.
            nodupkey: When ``True``, keeps only the first row for each unique key
                defined by ``by`` after sorting.

        Returns:
            ``self`` (for method chaining).

        Examples:
            >>> session.sort("class", "age")
            >>> session.sort("class", ["age", "name"])
            >>> session.sort("class", [("age", "descending")])
            >>> session.sort("class", ["age"], nodupkey=True)
        """
        table = self.to_arrow(source)
        key, key_names = self._normalize_sort_key(table, by)

        sorted_table = table.sort_by(key)
        if nodupkey:
            sorted_table = self._sort_unique_by_keys(sorted_table, key_names)
        output_name = self._resolve_output_name(source, out=out, target=target)
        self.load(output_name, sorted_table)
        return self

    def apply_dataset_options(
        self,
        source: str,
        *,
        keep: Sequence[str] | None = None,
        drop: Sequence[str] | None = None,
        rename: Mapping[str, str] | None = None,
        where: str | None = None,
        out: str | None = None,
        target: str | None = None,
    ) -> Session:
        """Applies ``keep``/``drop``/``rename``/``where`` in a single call.

        Multiple operations can be performed in one call.
        Application order: ``keep`` → ``drop`` → ``where`` → ``rename``.

        Args:
            source: Source dataset name.
            keep: List of column names to retain.
            drop: List of column names to remove.
            rename: A ``{old_name: new_name}`` mapping dict.
            where: Filter expression (same format as :meth:`where`).
            out: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).
        """
        output_name = self._resolve_output_name(source, out=out, target=target)
        source_name = source

        if keep:
            self.keep(source_name, list(keep), out=output_name)
            source_name = output_name

        if drop:
            self.drop(source_name, list(drop), out=output_name)
            source_name = output_name

        if where:
            self.where(source_name, where, out=output_name)
            source_name = output_name

        if rename:
            self.rename(source_name, rename, out=output_name)

        return self

    @staticmethod
    def _resolve_column_name(table: pa.Table, column: str, *, parameter_name: str) -> str:
        return resolve_column_api_name(table, column, parameter_name=parameter_name)

    @staticmethod
    def _to_registerable_dataset(data: Any) -> Any:
        if isinstance(data, pa.Table):
            return data

        if isinstance(data, pl.DataFrame):
            return data.to_arrow()

        try:
            import pandas as pd  # type: ignore

            if isinstance(data, pd.DataFrame):
                return pa.Table.from_pandas(data)
        except Exception:
            pass

        if isinstance(data, list):
            if all(isinstance(item, Mapping) for item in data):
                return pa.Table.from_pylist(data)
            return data
        return data

    @staticmethod
    def _to_arrow(data: Any):
        if isinstance(data, pa.Table):
            return data

        if isinstance(data, pl.DataFrame):
            return data.to_arrow()

        try:
            import pandas as pd  # type: ignore

            if isinstance(data, pd.DataFrame):
                return pa.Table.from_pandas(data)
        except Exception:
            pass

        if isinstance(data, list):
            return pa.Table.from_pylist(data)
        raise ValueError("Unsupported dataset type for Session.load")

    @staticmethod
    def _resolve_output_name(source: str | None, *, out: str | None = None, target: str | None = None) -> str | None:
        if out is not None and target is not None and out != target:
            raise ValueError("out and target must match when both are provided")
        if out is not None:
            return out
        if target is not None:
            return target
        return source

    def _resolve_cast_aliases(
        self,
        table: pa.Table,
        resolved_mapping: Mapping[str, Any],
        alias: str | Mapping[str, str] | None,
    ) -> dict[str, str]:
        if alias is None:
            return {}

        if isinstance(alias, str):
            if len(resolved_mapping) != 1:
                raise TypeError("alias must be a mapping when casting multiple columns")
            return {next(iter(resolved_mapping)): alias}

        resolved_alias = {
            self._resolve_column_name(table, source_name, parameter_name="alias"): target_name
            for source_name, target_name in alias.items()
        }
        unexpected = sorted(name for name in resolved_alias if name not in resolved_mapping)
        if unexpected:
            joined = ", ".join(unexpected)
            raise ValueError(f"alias entries must reference cast columns: {joined}")

        existing_names = set(table.column_names)
        for source_name, target_name in resolved_alias.items():
            if target_name in existing_names and target_name != source_name:
                raise ValueError(f"alias target already exists: {target_name}")
        if len(set(resolved_alias.values())) != len(resolved_alias):
            raise ValueError("alias targets must be unique")

        return resolved_alias

    def _build_sql_context(self) -> pl.SQLContext:
        context = pl.SQLContext()
        for name in self._datasets:
            context.register(name, pl.from_arrow(self.to_arrow(name)))

        dictionary_tables = self.dictionary.tables
        dictionary_columns = self.dictionary.columns
        context.register("dictionary.tables", pl.from_arrow(dictionary_tables))
        context.register("dictionary.columns", pl.from_arrow(dictionary_columns))
        context.register("dictionary_tables", pl.from_arrow(dictionary_tables))
        context.register("dictionary_columns", pl.from_arrow(dictionary_columns))
        return context

    @staticmethod
    def _is_dictionary_reserved_name(name: str) -> bool:
        return _dataset_key(name) in {"DICTIONARY", "DICTIONARY.TABLES", "DICTIONARY.COLUMNS"}

    def _normalize_sort_key(self, table: pa.Table, by: str | Sequence[str]) -> tuple[str | list[tuple[str, str]], tuple[str, ...]]:
        if isinstance(by, str):
            resolved_name = self._resolve_column_name(table, by, parameter_name="sort")
            return resolved_name, (resolved_name,)

        sort_items = list(by)
        if not sort_items:
            raise ValueError("Session.sort requires at least one sort key")
        if isinstance(sort_items[0], str):
            names = tuple(self._resolve_column_name(table, str(item), parameter_name="sort") for item in sort_items)
            return [(name, "ascending") for name in names], names

        normalized_items = [
            (self._resolve_column_name(table, str(item[0]), parameter_name="sort"), str(item[1]))
            for item in sort_items
        ]
        return normalized_items, tuple(name for name, _ in normalized_items)

    @classmethod
    def _sort_unique_by_keys(cls, table: pa.Table, key_names: Sequence[str]) -> pa.Table:
        if not key_names:
            return table

        rows = table.to_pylist()
        seen: set[tuple[Any, ...]] = set()
        unique_rows: list[dict[str, Any]] = []
        for row in rows:
            key = tuple(row.get(name) for name in key_names)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(dict(row))

        deduped = pa.Table.from_pylist(unique_rows)
        return preserve_arrow_metadata(table, deduped)

    @staticmethod
    def _iter_dataset_names(names: Sequence[str | Sequence[str]]) -> Iterable[str]:
        for name in names:
            if isinstance(name, str):
                yield name
                continue
            if isinstance(name, Sequence):
                for nested in name:
                    if not isinstance(nested, str):
                        raise TypeError("Dataset names must be strings.")
                    yield nested
                continue
            raise TypeError("Dataset names must be strings or sequences of strings.")

    @staticmethod
    def _resolve_polars_dtype(dtype: Any) -> Any:
        if not isinstance(dtype, str):
            return dtype

        normalized = dtype.strip().lower()
        mapping = {
            "int8": pl.Int8,
            "int16": pl.Int16,
            "int32": pl.Int32,
            "int64": pl.Int64,
            "uint8": pl.UInt8,
            "uint16": pl.UInt16,
            "uint32": pl.UInt32,
            "uint64": pl.UInt64,
            "float32": pl.Float32,
            "float64": pl.Float64,
            "bool": pl.Boolean,
            "boolean": pl.Boolean,
            "str": pl.String,
            "string": pl.String,
            "utf8": pl.String,
        }
        resolved = mapping.get(normalized)
        if resolved is None:
            raise ValueError(f"Unsupported dtype for DatasetView.astype: {dtype}")
        return resolved
