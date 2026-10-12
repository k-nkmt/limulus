"""Top-level public imports for the limulus workflow surface.

Import from here when you want the stable user-facing entry points such as
``Session``, ``submit(...)``, and ``run(...)``.

Importing ``limulus`` itself is deliberately light: pure models stay eager,
while data-processing exports (Session, execution, adapters, parser
service) are resolved lazily through module ``__getattr__`` so pure syntax
imports never load Arrow, Polars, the native extension, or the Session
runtime.  ``submit``/``run`` remain module-level functions and still
capture the direct caller context inside their own bodies.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

# Public API: pure models are stdlib-only and safe to import eagerly.
from .host_functions import HostFunctionContextCapture, capture_host_function_context
from .models import (
    CompatibilityNotice,
    DatasetCatalog,
    DataSetRef,
    Diagnostic,
    ExecuteRequest,
    ExecuteResponse,
    LogEntry,
    OutputConversionResult,
    SubmitResult,
)

# Data-processing exports resolved on first attribute access (PackageExports).
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Session": ("limulus.session", "Session"),
    "DataStepExecutor": ("limulus.execution", "DataStepExecutor"),
    "FormatSupportResult": ("limulus.execution", "FormatSupportResult"),
    "RuntimeRequirements": ("limulus.execution", "RuntimeRequirements"),
    "DataAdapterError": ("limulus.io_adapters", "DataAdapterError"),
    "DataFrameAdapterPandas": ("limulus.io_adapters", "DataFrameAdapterPandas"),
    "DataInputAdapterArrow": ("limulus.io_adapters", "DataInputAdapterArrow"),
    "DataOutputAdapterArrow": ("limulus.io_adapters", "DataOutputAdapterArrow"),
    "InputSpec": ("limulus.io_adapters", "InputSpec"),
    "OutputSpec": ("limulus.io_adapters", "OutputSpec"),
    "DataStepAst": ("limulus.parser", "DataStepAst"),
    "ParseResult": ("limulus.parser", "ParseResult"),
    "ParsedStatement": ("limulus.parser", "ParsedStatement"),
    "ParserService": ("limulus.parser", "ParserService"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = spec
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def _submit_top_level_with_context(
    code: str,
    *,
    backend: str,
    datasets: dict[str, Any],
    host_capture: HostFunctionContextCapture,
) -> SubmitResult:
    """Create the one-shot Session and execute an already-captured request."""

    # Imported here so importing ``limulus`` stays free of Session and the
    # data-processing stack until a top-level call actually uses it.
    from .session import Session

    session = Session(backend=backend)
    for name, data in datasets.items():
        session.load(name, data)
    # Calling Session.submit here would capture this module frame rather than
    # the user's direct caller. The private common entry preserves the one
    # top-level snapshot and avoids a second capture through ``run``.
    return session._submit_with_host_context(
        code,
        backend=None,
        show_result=True,
        host_capture=host_capture,
    )


def submit(
    code: str,
    *,
    backend: str = "auto",
    functions: object | None = None,
    **datasets: Any,
) -> SubmitResult:
    """One-shot execution (when session management is not needed).

    Args:
        code: Data Step DSL text to execute.
        backend: Runtime backend to use, one of ``"rust"``, ``"python"``, or ``"auto"``.
            Defaults to ``"auto"``.
        functions: Optional request-scoped mapping of string names to callables
            for ``apply``. This keyword is reserved and is not a dataset name.
            To use a dataset named ``functions``, register it through
            ``Session.load("functions", table)`` and call ``Session.submit``.

    Examples:
        >>> import limulus, pyarrow as pa
        >>> result = limulus.submit("data out; set inp; run;", inp=pa.table({"x": [1, 2]}))
        >>> result.success
        True
    """
    started_frame = inspect.currentframe()
    caller_frame = None
    try:
        caller_frame = started_frame.f_back if started_frame is not None else None
        caller_locals = dict(caller_frame.f_locals) if caller_frame is not None else {}
        caller_globals = dict(caller_frame.f_globals) if caller_frame is not None else {}
    finally:
        del caller_frame
        del started_frame
    host_capture = capture_host_function_context(
        caller_locals,
        caller_globals,
        functions=functions,
    )
    del caller_locals
    del caller_globals
    return _submit_top_level_with_context(
        code,
        backend=backend,
        datasets=datasets,
        host_capture=host_capture,
    )


def run(
    code: str,
    *,
    backend: str = "auto",
    functions: object | None = None,
    **datasets: Any,
) -> SubmitResult:
    """Run one-shot code with the same direct-caller and functions contract."""

    started_frame = inspect.currentframe()
    caller_frame = None
    try:
        caller_frame = started_frame.f_back if started_frame is not None else None
        caller_locals = dict(caller_frame.f_locals) if caller_frame is not None else {}
        caller_globals = dict(caller_frame.f_globals) if caller_frame is not None else {}
    finally:
        del caller_frame
        del started_frame
    host_capture = capture_host_function_context(
        caller_locals,
        caller_globals,
        functions=functions,
    )
    del caller_locals
    del caller_globals
    return _submit_top_level_with_context(
        code,
        backend=backend,
        datasets=datasets,
        host_capture=host_capture,
    )


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
