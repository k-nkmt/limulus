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
  session_parsing.py  Session.filter / Session.sql parsing and diagnostics
  column_api.py       assign / transpose internals and case-when parsing
  parser.py           Parser backend selection and AST models
  models.py           Shared data models (Diagnostic, SubmitResult, etc.)
  io.py               Arrow conversion helpers and synthetic dictionary inputs
  io_adapters.py      Input/output adapters
  arrow_bridge.py     Arrow metadata restoration and table stabilization
  native_bridge.py    Native extension loading helpers
  renderer.py         Diagnostic rendering helpers
  block_splitter.py   Multi-block splitting helpers
  format_registry.py  Custom format / informat registry
  naming.py           Shared dataset / column normalization helpers
  execution/
    coordinator.py        DataStepExecutor orchestration and input resolution
    pipeline.py           Shared pipeline dataclasses and block coordination
    input_preparation.py  Shared input preparation and dataset option handling
    output_handoff.py     Shared output filtering and Arrow handoff
    rewrites.py           Rewrite planning helpers
    python_backend.py     Residual Python backend entry point
    python_statement_blocks.py  Residual Python statement-block execution helpers
    python_statements.py        Residual Python backend dispatch helpers
  backend_integration/
    backend_dispatch_policy.py  Backend routing policy
    selection.py                Runtime selection outcomes
    contracts.py                Rust handoff contracts
    rust_bridge.py              Python -> Rust bridge helpers
    rust_executor.py            Native execution entry point
    transport.py                Arrow / row transport rules
  runtime/
    row_runtime.py      Row runtime state and apply helpers
    expressions.py      Expression evaluation helpers
    conditions.py       Conditional evaluation helpers
    arrow_cursor.py     Arrow row cursor helpers
  grammar/
    *.lark              EBNF grammar definitions

native/
  limulus_native/      Rust crate
    Cargo.toml
    src/lib.rs                  PyO3 entry point
    src/ast.rs                  AST adapters
    src/diagnostics.rs          Native diagnostics
    src/expressions.rs          Native expression evaluation
    src/expressions/functions.rs  Native function dispatch
    src/io.rs                   Arrow / Python I/O bridging
    src/output.rs               Output assembly
    src/output/accumulator.rs   Output accumulation helpers
    src/row_cursor.rs           Native row cursor
    src/runtime.rs              Native row-loop runtime
    src/runtime/state.rs        Native runtime state
    src/source_view.rs          Source span helpers

tests/                pytest release-facing public suite
tests_dev/            internal residual contract, boundary, routing, and benchmark suites
tests/cases/          external success/diagnostic case packs and inventory/report JSON
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
DataStepExecutor (execution/coordinator.py)
    │ ExecutionPipelineCoordinator
    │  1) macro hook / unsupported syntax skip
    │  2) split blocks
    │  3) parse (parser.py + grammar/*.lark)
    │  4) validate
    │  5) resolve inputs
    │  6) pre-processing inputs
    │  7) pre-evaluations
    │  8) plan generation
    │  9) execute runtime backend
    │      ├─ Rust backend (default)
    │      │    └─ backend_integration/rust_bridge.py
    │      │         -> rust_executor.py
    │      │         -> limulus_native.execute_datastep (src/lib.rs)
    │      └─ Residual Python backend
    │           └─ execution/python_backend.py
    │ 10) resolve outputs (temporary/internal variable filtering)
    │ 11) apply output metadata
    ▼
ExecuteResponse -> Session catalog (Arrow tables)
```

### Key Design Decisions

| Item | Detail |
|------|------|
| Parser | lark only (Python). Rust is not involved in parsing |
| Data representation | Storage/I/O uses Apache Arrow. Arrow is the canonical table representation |
| Rust scope | Standard row-loop execution is Rust-first; shared preprocessing and postprocessing stay in Python |
| Backend selection | `auto`, `rust`, and `python` act as routing preferences or seam controls; normal Arrow-backed execution is not expected to use Python row-loop by default |
| BY groups | Coordinator sorts ascending by BY variables before execution when required |
| SUM statement behavior | Based on current row value; `sum_totals` acts as fallback when variable is absent from row |

### Pipeline Stage Diagnostics

- Each diagnostic carries an independent `stage` field.
- `Session.submit()` maps this field into `LogEntry.stage`.
- `SubmitResult.format_log()` renders stage as a dedicated item: `Severity [stage: ...]: message`.

### Temporary/Internal Variables

- Internal helper variables from execution (`IN=`, `INDSNAME=`, `END=`, `FIRST./LAST.` and their renamed aliases)
  are filtered centrally in the shared output handoff stage.
- This filtering is backend-agnostic and is applied after runtime execution so behavior stays consistent across
  Rust and residual Python execution.

### Backend Boundary

- Files under `execution/python_*` are residual Python-backend-specific.
- Shared behavior that must also apply to Rust execution, such as backend-agnostic input normalization,
  source dataset option preprocessing, synthetic dictionary table synthesis, and shared output shaping, should live in backend-neutral modules.
- Rust handoff contracts, routing policy, and transport rules belong in `backend_integration/`.

---

## Development Commands

```bash
# Install dependencies (including dev)
uv sync --group dev

# Rebuild Rust (--release is required to reflect changes)
uv run --no-sync maturin develop --release
```

> ⚠️ **`maturin develop` (without options) generates a debug build but does NOT update the `.so` in site-packages.**  
> After modifying Rust code, always run with `--release`.

---

## Testing

```bash
uv run --no-sync pytest     # all tests
uv run --no-sync pytest -q  # concise output
uv run --no-sync pytest -x  # stop on first failure
```

Test file mapping:

| File | Target |
|----------|------|
| `tests/test_session.py` | Session integration / end-to-end scenarios and public API contracts |
| `tests/test_submit.py` | submit/run convenience functions and dictionary Data Step regressions |
| `tests/test_runtime.py` | Runtime routing and execution contracts |
| `tests/test_runtime_backend_contract.py` | Public backend preference smoke wrappers |
| `tests/test_session_backend_contract.py` | Public session/backend preference wrappers |
| `tests/test_session_renderer_adapter_contract.py` | Public renderer/adapter wrappers |
| `tests/test_submit_renderer_adapter_contract.py` | Public submit adapter/convert_outputs wrapper |
| `tests_dev/backend_selection/test_runtime_backend_contract.py` | Internal backend parity and dataset-option matrix |
| `tests_dev/backend_selection/test_session_backend_contract.py` | Internal session/backend residual wrapper |
| `tests_dev/boundary_contracts/test_session_renderer_internal_contracts.py` | Internal renderer availability, fallback, and lazy materialization contracts |
| `tests_dev/boundary_contracts/test_release_surface_inventory_v05o.py` | Inventory/report and release-surface split guards |

Public minimum wrappers stay under `tests/`. Internal residual wrappers stay under `tests_dev/`. When moving contract wrappers, keep the delegated contract targets disjoint and update the release-surface inventory/report at the same time.

---

## Building Documentation

```bash
uv run sphinx-build -b html docs docs/_build/html
```

---

## Notes When Modifying Key Files

### When modifying `limulus/execution/coordinator.py` or `limulus/execution/python_backend.py`
- `coordinator.py` contains `DataStepExecutor`: orchestration, backend selection, I/O resolution, and multi-block coordination.
- `python_backend.py` contains the residual Python backend entry point and high-level Python row-loop orchestration.
- `input_preparation.py` contains shared input preparation such as dataset options and source-row planning.
- `output_handoff.py` contains shared post-runtime filtering and Arrow output shaping.
- `python_statement_blocks.py` contains residual Python statement/block execution helpers (`IF`, `DO`, `ARRAY`, `SUM`, `ASSIGN`, etc.).
- `python_statements.py` contains residual Python backend dispatch helpers.
- Prefer shared pipeline post-processing for cross-backend behavior.
- When modifying backend selection or Rust routing behavior, changes typically go in `coordinator.py` or `backend_integration/*`.
- Always verify with `uv run --no-sync pytest -q` after changes to these files.

### When modifying `native/limulus_native/src`
1. Recompile with `uv run --no-sync maturin develop --release`
2. If native runtime behavior changed, inspect related files under `src/runtime.rs`, `src/expressions.rs`, `src/io.rs`, and `src/output.rs`
3. Verify with focused backend/runtime tests before broader validation

### When modifying `limulus/grammar/*.lark`
- lark caches the parser, so verify grammar changes are reflected in existing tests

### When modifying the public API in `limulus/__init__.py`
- Use `__all__` to explicitly declare the public API
- If removing existing re-exports for compatibility, check all usages (tests / docs / examples)

---

## Known Warnings (from Lark)

- When running `uv run --no-sync pytest` or `uv run sphinx-build`, a `DeprecationWarning`
  (`Flags not at the start of the expression`) from Lark's internal implementation may appear.
- This is a known non-fatal warning and does not affect build/test results (pass/fail).

---

## Module Visibility

| Module | Visibility |
|-----------|---------|
| `Session`, `submit`, `run` | Public user API |
| `SubmitResult`, `LogEntry`, `DatasetCatalog` | Public return types |
| `DataStepExecutor`, `RuntimeBackendSelector` | Internal / advanced use |
| `io_adapters.*`, `models.*` (except `SubmitResult`, etc.) | Internal |
| `parser.*` (`DataStepAst`, etc.) | Internal / advanced use |
| `limulus_native.*` | Rust bindings, not for direct use |