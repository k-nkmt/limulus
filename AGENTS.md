# limulus Developer Guide

Technical information for developers and AI agents. For end-user information, see [README.md](README.md).

---

## Project Overview

**limulus** is a library that executes Data Step syntax in Python.  
It receives DSL text, parses and executes it, and returns results as Apache Arrow tables.

- Language: Python 3.10+ / Rust (PyO3)
- Package manager: **uv**
- Build system: **maturin** (Rust → Python bindings)
- Parser: **lark** (Python, EBNF grammar)
- License: PolyForm Noncommercial License 1.0.0

The software may be used for personal, educational, academic, and noncommercial research purposes.
Commercial use is not permitted under the current license terms.

This is an independent implementation, no SAS source code or proprietary materials have been used.
Compatibility with SAS software is not guaranteed and is not a project goal. Certain behaviors intentionally differ to provide modern semantics.

---

## Repository Structure

```
limulus/              Python package
  __init__.py         Public API (Session, submit, run, ...)
  session.py          Session class (user-facing interface layer)
  runtime.py          Execution services (PDVRuntime, ProgramExecution, etc.)
  executor.py         Backend selection & management
  executor_python.py  Python backend
  parser.py           Parser backend selection & AST models
  models.py           Shared data models (Diagnostic, SubmitResult, etc.)
  io.py               Arrow ↔ row-list conversion
  io_adapters.py      Input/output adapters
  backends.py         Backend capability checks
  grammar/
    datastep.lark     EBNF grammar definition (lark)

native/
  limulus_native/     Rust crate
    src/lib.rs        PDV loop engine / expression evaluation engine (core)
    Cargo.toml

tests/                pytest test suite
docs/                 Sphinx documentation (MyST / myst-nb)
```

---

## Current Architecture

```
user code
    │ session.submit(dsl_text)
    ▼
Session (session.py)
    │ ExecuteRequest
    ▼
DataStepExecutor (executor.py)
    │ ExecutionPipelineCoordinator
    │  1) split blocks
    │  2) macro hook(in future)
    │  3) parse (parser.py + grammar/datastep.lark)
    │  4) resolve inputs
    │  5) pre-processing inputs
    │  6) pre-evaluations
    │  7) execute runtime backend
    │      ├─ Python backend → PythonBackendExecutionService (executor_python.py)
    │      │                     └─ PDVRuntimeService (runtime.py)
    │      └─ Rust backend   → limulus_native.execute_datastep (lib.rs)
    │  8) resolve outputs (temporary/internal variable filtering)
    │  9) output conversion (arrow_table)
    ▼
ExecuteResponse → Session catalog (Arrow tables)
```

### Key Design Decisions

| Item | Detail |
|------|------|
| Parser | lark only (Python). Rust is not involved in parsing |
| Data representation | Storage/I/O uses Apache Arrow. During PDV loop, converted to row dicts |
| Rust scope | Native block execution via Arrow C stream bridge; orchestrated by Python pipeline |
| Backend selection | `auto` or `rust` preference may fall back to Python when Rust capability/inputs are unsupported (e.g. non-arrow inputs, `apply()`) |
| BY groups | executor sorts ascending by BY variables (Arrow-native) before execution |
| SUM statement behavior | Based on current row value; `sum_totals` acts as fallback when variable is absent from row |

### Pipeline Stage Diagnostics

- Each diagnostic now carries an independent `stage` field (instead of stage text being prefixed in `message`).
- `Session.submit()` maps this field into `LogEntry.stage`.
- `SubmitResult.format_log()` renders stage as a dedicated item: `Severity [stage: ...]: message`.

### Temporary/Internal Variables

- Internal helper variables from execution (`IN=`, `INDSNAME=`, `END=`, `FIRST./LAST.` and their renamed aliases)
    are filtered centrally in the pipeline output stage (`resolve outputs`).
- This filtering is backend-agnostic and is applied after runtime execution so behavior is consistent across
    Python and Rust backends.

---

## Development Commands

```bash
# Install dependencies (including dev)
uv sync --group dev

# Rebuild Rust (--release is required to reflect changes)
uv run maturin develop --release
```

> ⚠️ **`maturin develop` (without options) generates a debug build but does NOT update the `.so` in site-packages.**  
> After modifying Rust code, always run with `--release`.  


---

## Testing

```bash
uv run pytest               # all tests
uv run pytest -q            # concise output
uv run pytest -x            # stop on first failure
uv run pytest tests/test_session.py::TestSessionSubmit -v
```

Test file mapping:

| File | Target |
|----------|------|
| `test_session.py` | Session API · end-to-end |
| `test_submit.py` | submit/run convenience functions |
| `test_runtime.py` | PDV runtime service |
| `test_parser.py` | Parser |
| `test_io.py` / `test_io_adapters.py` | I/O conversion |
| `test_pdv_runtime.py` | PDV loop (Python backend) |
| `test_advanced_runtime_services.py` | Advanced features: RETAIN / ARRAY / BY, etc. |
| `test_functions.py` | Functions |


---

## Building Documentation

```bash
cd docs
uv run sphinx-build -b html . _build/html
```

---

## Notes When Modifying Key Files

### When modifying `limulus/executor.py` or `limulus/executor_python.py`
- `executor.py` contains `DataStepExecutor`: orchestration, backend selection, I/O resolution, multi-block coordination
- `executor_python.py` contains `PythonBackendExecutionService`: complete row-level PDV loop for the Python backend
- Prefer pipeline-level post-processing for cross-backend behavior (e.g. internal temporary variable filtering)
- When adding new statement types or row-processing logic, changes typically go in `executor_python.py`
- When modifying backend selection or I/O handling, changes go in `executor.py`
- Always verify with `uv run pytest` after changes to either file

### When modifying `native/limulus_native/src/lib.rs`
1. Recompile with `uv run maturin develop --release`
2. If using Jupyter, restart the kernel
3. Verify with tests: `uv run pytest`

### When modifying `limulus/grammar/datastep.lark`
- lark caches the parser, so verify grammar changes are reflected in existing tests

### When modifying the public API in `limulus/__init__.py`
- Use `__all__` to explicitly declare the public API
- If removing existing re-exports for compatibility, check all usages (tests/benchmarks/docs)

---

## Known Warnings (from Lark)

- When running `uv run pytest` or `uv run sphinx-build`, a `DeprecationWarning` (`Flags not at the start of the expression`) from Lark's internal implementation may appear.
- This is a known non-fatal warning and does not affect build/test results (pass/fail).
- Resolution will be considered when the upstream (Lark) update becomes necessary.

---

## Module Visibility

| Module | Visibility |
|-----------|---------|
| `Session`, `submit`, `run` | Public user API |
| `SubmitResult`, `LogEntry`, `DatasetCatalog` | Public (return types) |
| `DataStepExecutor`, `RuntimeRequirements` | Internal / advanced use |
| `io_adapters.*`, `models.*` (except `SubmitResult`, etc.) | Internal |
| `parser.*` (`DataStepAst`, etc.) | Internal / advanced use |
| `limulus_native.*` | Rust bindings, not for direct use |

---

## Key Rust Backend Structures (`lib.rs`)

| Function / Struct | Role |
|------------|------|
| `execute_datastep()` | PyO3 entry point |
| `execute_statement_block()` | Recursive execution of statement lists |
| `execute_inline_action()` | Single-line action execution for IF THEN |
| `evaluate_scalar_expression()` | Expression evaluation (Rust-native + Python fallback) |
| `EvalRuntimeState` | Execution state (RETAIN values, SUM accumulations, ARRAY definitions) |
| `AstStatement` | AST node passed from parser (via PyO3) |
