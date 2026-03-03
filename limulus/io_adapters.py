import os
from dataclasses import dataclass
from typing import Any, Iterable

from .models import DataSetRef


@dataclass(frozen=True)
class InputSpec:
    format: str
    location: str = ""
    payload: Any | None = None


@dataclass(frozen=True)
class OutputSpec:
    format: str
    location: str = ""


class DataAdapterError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DataInputAdapterArrow:
    def load(self, spec: InputSpec) -> list[dict[str, Any]]:
        normalized = spec.format.strip().lower()
        pyarrow, _, _ = _require_pyarrow()

        if normalized == "arrow_table":
            if not isinstance(spec.payload, pyarrow.Table):
                raise DataAdapterError(
                    "IO_ARROW_INPUT_ERROR",
                    "arrow_table input requires a pyarrow.Table payload.",
                )
            return self._to_canonical_rows(spec.payload)

        raise DataAdapterError("IO_ARROW_UNSUPPORTED_FORMAT", f"Unsupported Arrow input format: {spec.format}")

    def _to_canonical_rows(self, table: Any) -> list[dict[str, Any]]:
        strategy = os.getenv("LIMULUS_ARROW_ROW_READER", "polars_iter_rows").strip().lower()

        if strategy == "pyarrow_to_pylist":
            return table.to_pylist()

        if strategy == "polars_iter_rows":
            try:
                import polars as pl  # type: ignore

                frame = pl.from_arrow(table)
                buffer_size_raw = os.getenv("LIMULUS_POLARS_ITER_ROWS_BUFFER_SIZE", "4096").strip()
                iter_kwargs: dict[str, Any] = {"named": True}
                if buffer_size_raw:
                    iter_kwargs["buffer_size"] = max(1, int(buffer_size_raw))
                return list(frame.iter_rows(**iter_kwargs))
            except Exception:
                return table.to_pylist()

        return table.to_pylist()


class DataOutputAdapterArrow:
    def store(
        self,
        table: Iterable[dict[str, Any]],
        spec: OutputSpec,
    ) -> DataSetRef:
        normalized = spec.format.strip().lower()
        pyarrow, _, _ = _require_pyarrow()

        records = list(table)
        arrow_table = pyarrow.Table.from_pylist(records)

        if normalized == "arrow_table":
            location = spec.location or "memory://arrow_table"
            return DataSetRef(kind="arrow_table", location=location, payload=arrow_table)

        raise DataAdapterError("IO_ARROW_UNSUPPORTED_FORMAT", f"Unsupported Arrow output format: {spec.format}")


class DataFrameAdapterPandas:
    def to_canonical(self, dataframe: Any) -> list[dict[str, Any]]:
        pandas = _require_pandas()
        if not isinstance(dataframe, pandas.DataFrame):
            raise DataAdapterError(
                "IO_PANDAS_INPUT_ERROR",
                "pandas input requires a pandas.DataFrame payload.",
            )
        return [dict(row) for row in dataframe.to_dict(orient="records")]

    def from_canonical(self, rows: Iterable[dict[str, Any]]) -> Any:
        pandas = _require_pandas()
        return pandas.DataFrame(list(rows))

    def load(self, spec: InputSpec) -> list[dict[str, Any]]:
        normalized = spec.format.strip().lower()
        if normalized != "pandas":
            raise DataAdapterError("IO_PANDAS_UNSUPPORTED_FORMAT", f"Unsupported pandas input format: {spec.format}")
        return self.to_canonical(spec.payload)

    def store(self, rows: Iterable[dict[str, Any]], spec: OutputSpec) -> DataSetRef:
        normalized = spec.format.strip().lower()
        if normalized != "pandas":
            raise DataAdapterError(
                "IO_PANDAS_UNSUPPORTED_FORMAT",
                f"Unsupported pandas output format: {spec.format}",
            )

        frame = self.from_canonical(rows)
        location = spec.location or "memory://pandas"
        return DataSetRef(kind="pandas", location=location, payload=frame)


def _require_pyarrow() -> tuple[Any, Any, Any]:
    try:
        import pyarrow  # type: ignore
        import pyarrow.ipc as ipc  # type: ignore
        import pyarrow.parquet as parquet  # type: ignore

        return pyarrow, parquet, ipc
    except Exception as error:  # pragma: no cover - covered by importorskip tests
        raise DataAdapterError(
            "IO_ARROW_DEPENDENCY_MISSING",
            "pyarrow is required for Arrow/Parquet I/O.",
        ) from error


def _require_pandas() -> Any:
    try:
        import pandas  # type: ignore

        return pandas
    except Exception as error:  # pragma: no cover - covered by importorskip tests
        raise DataAdapterError(
            "IO_PANDAS_DEPENDENCY_MISSING",
            "pandas is required for DataFrame conversion.",
        ) from error


__all__ = [
    "DataAdapterError",
    "InputSpec",
    "OutputSpec",
    "DataInputAdapterArrow",
    "DataOutputAdapterArrow",
    "DataFrameAdapterPandas",
]