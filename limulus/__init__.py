from __future__ import annotations

from typing import Any

# Public API
from .session import Session
from .models import DatasetCatalog, LogEntry, SubmitResult

# Internal / legacy API (still importable for advanced use and internal benchmarks)
from .runtime import DataStepExecutor, FormatSupportResult, RuntimeRequirements
from .io_adapters import (
    DataAdapterError,
    DataFrameAdapterPandas,
    DataInputAdapterArrow,
    DataOutputAdapterArrow,
    InputSpec,
    OutputSpec,
)
from .models import (
    CompatibilityNotice,
    DataSetRef,
    Diagnostic,
    ExecuteRequest,
    ExecuteResponse,
    OutputConversionResult,
)
from .parser import DataStepAst, ParseResult, ParsedStatement, ParserService


def submit(code: str, *, backend: str = "auto", **datasets: Any) -> SubmitResult:
    """One-shot execution (when session management is not needed).

    Args:
        code: Data Step DSL text to execute.
        backend: Runtime backend to use, one of ``"rust"``, ``"python"``, or ``"auto"``.
            Defaults to ``"auto"``.

    Examples:
        >>> import limulus, pyarrow as pa
        >>> result = limulus.submit("data out; set inp; run;", inp=pa.table({"x": [1, 2]}))
        >>> result.success
        True
    """
    session = Session(backend=backend)
    for name, data in datasets.items():
        session.load(name, data)
    return session.submit(code, show_result=True)


def run(code: str, *, backend: str = "auto", **datasets: Any) -> SubmitResult:
    """Alias for :func:`submit`."""
    return submit(code, backend=backend, **datasets)


# Public surface
__all__ = [
    "Session",
    "submit",
    "run",
    "SubmitResult",
    "LogEntry",
    "DatasetCatalog",
    "DataStepExecutor",
    "FormatSupportResult",
    "RuntimeRequirements",
    "DataAdapterError",
    "DataFrameAdapterPandas",
    "DataInputAdapterArrow",
    "DataOutputAdapterArrow",
    "InputSpec",
    "OutputSpec",
    "CompatibilityNotice",
    "DataSetRef",
    "Diagnostic",
    "ExecuteRequest",
    "ExecuteResponse",
    "OutputConversionResult",
    "DataStepAst",
    "ParseResult",
    "ParsedStatement",
    "ParserService",
]
