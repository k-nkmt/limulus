from __future__ import annotations

import re
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pyarrow as pa
import polars as pl

from .runtime import DataStepExecutor
from .models import DatasetCatalog, ExecuteRequest, LogEntry, SubmitResult


class DatasetView:
    """A view for chained operations on a dataset.

    Obtained via :meth:`Session.dataset`. You can chain :meth:`keep` / :meth:`drop` /
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

    def to_arrow(self):
        """Returns this dataset as a ``pyarrow.Table``."""
        return self._session.to_arrow(self._name)

    def to_pandas(self):
        """Converts this dataset to a ``pandas.DataFrame`` and returns it."""
        return self._session.to_pandas(self._name)

    def to_polars(self):
        """Converts this dataset to a ``polars.DataFrame`` and returns it."""
        return self._session.to_polars(self._name)

    def keep(self, columns: Sequence[str], target: str | None = None) -> "DatasetView":
        """Retains only the specified columns (equivalent to KEEP).

        Args:
            columns: List of column names to keep.
            target: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = target or self._name
        self._session.keep(self._name, columns, target=output_name)
        return DatasetView(self._session, output_name)

    def drop(self, columns: Sequence[str], target: str | None = None) -> "DatasetView":
        """Removes the specified columns (equivalent to DROP).

        Args:
            columns: List of column names to remove.
            target: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = target or self._name
        self._session.drop(self._name, columns, target=output_name)
        return DatasetView(self._session, output_name)

    def where(self, expression: str, target: str | None = None) -> "DatasetView":
        """Filters rows using a simple comparison expression (equivalent to WHERE).

        Args:
            expression: Filter expression, e.g. ``"age > 13"``, ``"sex = 'M'"``.
            target: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = target or self._name
        self._session.where(self._name, expression, target=output_name)
        return DatasetView(self._session, output_name)

    def rename(self, mapping: Mapping[str, str], target: str | None = None) -> "DatasetView":
        """Renames columns (equivalent to RENAME).

        Args:
            mapping: A ``{old_name: new_name}`` mapping dict.
            target: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = target or self._name
        self._session.rename(self._name, mapping, target=output_name)
        return DatasetView(self._session, output_name)

    def apply_options(
        self,
        *,
        keep: Sequence[str] | None = None,
        drop: Sequence[str] | None = None,
        rename: Mapping[str, str] | None = None,
        where: str | None = None,
        target: str | None = None,
    ) -> "DatasetView":
        """Applies ``keep``/``drop``/``rename``/``where`` in a single call.

        Args:
            keep: List of column names to retain.
            drop: List of column names to remove.
            rename: A ``{old_name: new_name}`` mapping dict.
            where: Filter expression (e.g. ``"age > 13"``).
            target: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = target or self._name
        self._session.apply_dataset_options(
            self._name,
            keep=keep,
            drop=drop,
            rename=rename,
            where=where,
            target=output_name,
        )
        return DatasetView(self._session, output_name)
    
    def sort(self, by: Sequence[str], target: str | None = None) -> "DatasetView":
        """Sorts the dataset by columns.

        Args:
            by: List of column names to sort by. Can also be a list of
                ``(column_name, "ascending" | "descending")`` tuples to specify direction.
            target: Output dataset name. If omitted, overwrites this view's dataset.

        Returns:
            A :class:`DatasetView` for the resulting dataset.
        """
        output_name = target or self._name
        self._session.sort(self._name, by, target=output_name)
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
    _SIMPLE_FILTER = re.compile(r"^\s*([A-Za-z_][\w\.]*)\s*(>=|<=|!=|=|>|<)\s*(.+?)\s*$")

    def __init__(
        self,
        *,
        backend: str = "auto",
        runtime_backend: str | None = None,
        parser_backend: str = "python",
    ) -> None:
        """Creates a limulus execution session.

        A session holds a dataset catalog and execution engine, sharing datasets
        across multiple :meth:`submit` calls.

        Args:
            backend: Preferred backend. One of ``"auto"`` (recommended), ``"python"``, or ``"rust"``.
                With ``"auto"``, the Rust backend is used when input is an Arrow table; otherwise Python.
            runtime_backend: Explicit override for the backend. Takes precedence over ``backend``.
            parser_backend: Parser backend. Currently only ``"python"`` (lark) is stable.

        Examples:
            >>> import limulus
            >>> session = limulus.Session()
            >>> session = limulus.Session(backend="python")  # Force Python backend
        """
        selected_runtime_backend = runtime_backend or backend
        self._executor = DataStepExecutor(
            runtime_backend=selected_runtime_backend,
            parser_backend=parser_backend,
        )
        self._datasets = DatasetCatalog()
        self._last_submit_result: SubmitResult | None = None

    def load(self, name: str, data: Any) -> None:
        """Registers a single dataset in the session catalog.

        Args:
            name: Dataset name referenced from Data Step code (case-insensitive).
            data: Data to register. Supported types:

                - ``pyarrow.Table``
                - ``polars.DataFrame``
                - ``pandas.DataFrame``
                - ``list[dict]`` (list of dicts)

        Examples:
            >>> import limulus, pyarrow as pa
            >>> session = limulus.Session()
            >>> session.load("mydata", pa.table({"x": [1, 2, 3]}))
        """
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
            response = self._executor.execute(ExecuteRequest(dsl_text=code))
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


    def select(self, source: str, columns: Sequence[str], target: str | None = None) -> Session:
        """Column selection (alias for :meth:`keep`).

        """
        table = self.to_arrow(source)
        selected = table.select(list(columns))
        output_name = target or source
        self.load(output_name, selected)
        return self

    def keep(self, source: str, columns: Sequence[str], target: str | None = None) -> Session:
        """Retains only the specified columns from a dataset (equivalent to KEEP).

        Args:
            source: Source dataset name.
            columns: List of column names to keep.
            target: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).

        Examples:
            >>> session.keep("bmi", ["name", "sex", "bmi"])
            >>> session.keep("bmi", ["name", "bmi"], target="bmi_slim")
        """
        return self.select(source, columns, target=target)

    def rename(self, source: str, mapping: Mapping[str, str], target: str | None = None) -> Session:
        """Renames columns (equivalent to RENAME).

        Args:
            source: Source dataset name.
            mapping: A ``{old_name: new_name}`` mapping dict.
            target: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).

        Examples:
            >>> session.rename("bmi", {"height_m": "height_meter"})
        """
        table = self.to_arrow(source)
        renamed_columns = [mapping.get(name, name) for name in table.column_names]
        renamed = table.rename_columns(renamed_columns)
        output_name = target or source
        self.load(output_name, renamed)
        return self

    def filter(self, source: str, expression: str, target: str | None = None) -> Session:
        """Row filtering (alias for :meth:`where`).

        """
        matched = self._SIMPLE_FILTER.match(expression.strip())
        if matched is None:
            raise ValueError(f"Unsupported filter expression for Session.filter: {expression}")

        variable_name = matched.group(1)
        operator = matched.group(2)
        raw_value = matched.group(3).strip()
        if not raw_value:
            raise ValueError(f"Unsupported filter expression for Session.filter: {expression}")

        scalar_value: Any
        if (raw_value.startswith('"') and raw_value.endswith('"')) or (
            raw_value.startswith("'") and raw_value.endswith("'")
        ):
            scalar_value = raw_value[1:-1]
        else:
            try:
                scalar_value = float(raw_value) if "." in raw_value else int(raw_value)
            except Exception as error:
                raise ValueError(f"Unsupported filter literal for Session.filter: {raw_value}") from error

        import pyarrow as pa
        import pyarrow.compute as pc

        table = self.to_arrow(source)
        if variable_name not in table.column_names:
            raise KeyError(f"Column not found for Session.filter: {variable_name}")

        column = table[variable_name]
        scalar = pa.scalar(scalar_value)
        if operator == ">":
            mask = pc.greater(column, scalar)
        elif operator == ">=":
            mask = pc.greater_equal(column, scalar)
        elif operator == "<":
            mask = pc.less(column, scalar)
        elif operator == "<=":
            mask = pc.less_equal(column, scalar)
        elif operator in {"=", "=="}:
            mask = pc.equal(column, scalar)
        else:
            mask = pc.not_equal(column, scalar)

        filtered = table.filter(mask)
        output_name = target or source
        self.load(output_name, filtered)
        return self

    def where(self, source: str, expression: str, target: str | None = None) -> Session:
        """Filters rows using a simple comparison expression (equivalent to WHERE).

        Supported expression format: ``"column op value"`` (e.g. ``"age > 13"``, ``"sex = 'M'``").

        Args:
            source: Source dataset name.
            expression: Filter expression. 
            target: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).

        Raises:
            ValueError: If the expression format is invalid.

        Examples:
            >>> session.where("class", "age > 13")
            >>> session.where("class", "sex = 'M'", target="male")
        """
        return self.filter(source, expression, target=target)

    def drop(self, source: str, columns: Sequence[str], target: str | None = None) -> Session:
        """Removes the specified columns from a dataset (equivalent to DROP).

        Args:
            source: Source dataset name.
            columns: List of column names to remove.
            target: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).
        """
        table = self.to_arrow(source)
        keep_columns = [name for name in table.column_names if name not in set(columns)]
        return self.select(source, keep_columns, target=target)
    
    def sort(self, source: str, by: Sequence[str], target: str | None = None) -> Session:
        """Sorts a dataset by columns.

        Args:
            source: Source dataset name.
            by: List of column names to sort by. Can also be a list of
                ``(column_name, "ascending" | "descending")`` tuples to specify direction.
            target: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).

        Examples:
            >>> session.sort("class", "age")
            >>> session.sort("class", ["age", "name"])
            >>> session.sort("class", [("age", "descending")])
        """
        table = self.to_arrow(source)
        if isinstance(by, str):
            key = by
        elif isinstance(by, list):
            if isinstance(by[0], str):
                key = [(col, "ascending") for col in by]
            else:
                key = by

        sorted_table = table.sort_by(key)
        output_name = target or source
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
        target: str | None = None,
    ) -> Session:
        """Applies ``keep``/``drop``/``rename``/``where`` in a single call.

        Multiple operations can be performed in one call.
        Application order: ``where`` → ``keep`` → ``drop`` → ``rename``.

        Args:
            source: Source dataset name.
            keep: List of column names to retain.
            drop: List of column names to remove.
            rename: A ``{old_name: new_name}`` mapping dict.
            where: Filter expression (same format as :meth:`where`).
            target: Output dataset name. If omitted, overwrites ``source``.

        Returns:
            ``self`` (for method chaining).
        """
        output_name = target or source
        if where:
            self.where(source, where, target=output_name)
            source_name = output_name
        else:
            source_name = source

        if keep:
            self.keep(source_name, list(keep), target=output_name)
            source_name = output_name

        if drop:
            self.drop(source_name, list(drop), target=output_name)
            source_name = output_name

        if rename:
            self.rename(source_name, rename, target=output_name)

        return self

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
