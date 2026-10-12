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
  expression_semantics.py  Planner-neutral normalization/classification/cache; typed classification `*_typed` entries are internal
  parser.py           Parser facade and flat result model re-exports
  syntax/
    __init__.py       Pure syntax API and lazy parse_program export
    models.py         Source/span/typed syntax and deferred-error models
    program.py        Whole-text Lark frontend, DATA boundaries, skip regions
    expressions.py    Structured general expressions and deferred errors
    normalization.py  Position-preserving analysis tab normalization
    legacy.py         Flat parser-result models and serializers
    compat.py         Typed syntax to parser-result/display projections
    native_payload.py Version 1 wire serializer and pure validator
  models.py           Shared data models (Diagnostic, SubmitResult, etc.)
  io.py               Arrow conversion helpers and synthetic dictionary inputs
  io_adapters.py      Input/output adapters
  arrow_bridge.py     Arrow metadata restoration and table stabilization
  native_bridge.py    Native extension loading helpers
  renderer.py         Diagnostic rendering helpers
  block_splitter.py   Multi-block splitting helpers
  format_registry.py  Dictionary format / informat catalogs
  host_functions.py   Named apply callable resolution
  naming.py           Shared dataset / column normalization helpers
  execution/
    coordinator.py        DataStepExecutor facade, composition root, catalog commit
    contracts.py          Immutable request/result and dependency contracts
    hooks.py              Public MacroHook protocol and NoOpMacroHook
    compatibility.py      Shared compatibility notices
    validation.py         Block validation and canonical diagnostic construction
    input_preparation.py  Input resolution, Arrow preparation, and pre-evaluations
    planning.py           Typed helper planning and execution-plan assembly
    runtime_handoff.py    Strategy decision and one native runtime handoff
    output_handoff.py     Target output planning and post-runtime finalization
    typed_output_planning.py Typed output-target and shared option planning
    strategy.py           Strategy normalization and execution-engine decision
    rewrites.py           Typed helper planning algorithms
    syntax_rewrites.py    Structural source-owned helper replacement
    statement_expressions.py Statement expression-slot traversal
  backend_integration/
    contracts.py                Rust handoff and typed result contracts
    rust_runtime.py             Sole row-runtime adapter
    typed_bridge.py             Build-only typed payload bridge
    bridge_support.py            Typed bridge input validation and Arrow export
    rust_executor.py            Native execution entry point
    transport.py                Arrow classifier and preparation provenance
  runtime/
    __init__.py         Minimal advanced-import compatibility exports
  grammar/
    *.lark              EBNF grammar definitions

native/
  limulus_native/      Rust crate
    Cargo.toml
    src/lib.rs                  PyO3 entry point
    src/ast.rs                  AST adapters
    src/typed_ast.rs            Versioned typed payload intake and expression lowering
    src/typed_lowering.rs       Pure typed statement lowering
    src/diagnostics.rs          Native diagnostics
    src/expressions/mod.rs      Native expression evaluation
    src/expressions/functions.rs  Native function dispatch
    src/output/mod.rs           Output assembly
    src/row_cursor.rs           Native row cursor
    src/row_view.rs             Logical row access
    src/runtime/mod.rs          Native row-loop runtime
    src/runtime/state.rs        Native runtime state
    src/source_registry.rs      Source stream registry
    src/source_view.rs          Source-backed value access
    src/value_ref.rs            Borrowed/owned value references

tests/                pytest release-facing public suite
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
    │  1) apply MacroHook
    │  2) parse the complete source, DATA blocks, and recognized skip regions
    │  3) reject program-level parse errors before executing any block
    │  for each DATA block:
    │  4) validate references and declarations
    │  5) plan source-owned helpers and output targets
    │  6) resolve and prepare Arrow inputs
    │  7) assemble execution and output-shaping plans
    │  8) execute the Rust row runtime
    │      └─ backend_integration/rust_runtime.py -> typed_bridge.py
    │           -> rust_executor.py -> limulus_native.execute_block (src/lib.rs)
    │  9) filter internal variables and restore output types/metadata
    │ 10) publish successful outputs to the executor's generated catalog
    ▼
ExecuteResponse -> Session catalog (on successful submission)
```

### Key Design Decisions

| Item | Detail |
|------|------|
| Parser | Lark in Python structures statements and expressions; Rust executes the lowered structure |
| Data representation | Storage/I/O uses Apache Arrow. Arrow is the canonical table representation |
| Rust scope | Rust is the only Data Step row engine; shared preprocessing and postprocessing stay in Python |
| Backend selection | `auto`, `rust`, and `python` select the Rust row strategy |
| Python callbacks | `apply("name", ...)` is the explicit host-extension boundary |
| BY groups | Input preparation sorts and aligns rows before execution when required |
| MERGE warnings | Repeated BY keys across sources produce a non-stopping warning without changing occurrence alignment |
| SUM statement behavior | An absent target uses the prior total or zero; an explicit NULL resets the base to zero |
| Preparation provenance | Typed carrier provenance is authoritative; public `DataSetRef.location` remains opaque |
| Output publication | Each DATA block publishes to the executor after success; Session.datasets updates only when the whole submission succeeds |

### Pipeline Stage Diagnostics

- Each diagnostic carries an independent `stage` field.
- `Session.submit()` maps this field into `LogEntry.stage`.
- `SubmitResult.format_log()` renders stage as a dedicated item: `Severity [stage: ...]: message`.

### Temporary/Internal Variables

- Internal helper variables from execution (`IN=`, `INDSNAME=`, `END=`, `FIRST./LAST.` and their renamed aliases)
  are filtered centrally in the shared output handoff stage.
- This filtering is applied after Rust runtime execution.

### Backend Boundary

- Public API, parsing, input preparation, Arrow contracts, metadata, and output shaping stay in Python.
- Rust owns row iteration, expression evaluation, statement execution, and data projection. Unsupported expressions return diagnostics without falling back to Python `eval`.
- Shared input normalization, source dataset options, synthetic dictionaries, and output shaping belong in the shared Python stage owners.
- Rust handoff contracts and transport rules belong in `backend_integration/`.
- Session-level catalogs, format registries, and observations are not thread-safe.

---

## Development Commands

```bash
# Install dependencies (including dev)
uv sync --locked --group dev --no-install-project

# Rebuild and install the optimized Rust extension
uv run --no-sync maturin develop --release
```

> After modifying Rust code, rebuild the extension before testing. Use `--release`
> for performance measurements; builds without this option use the debug profile.

---

## Testing

```bash
uv run --locked --no-sync pytest tests     # all release-facing tests
uv run --locked --no-sync pytest -q tests  # concise output
uv run --locked --no-sync pytest -x tests  # stop on first failure
```

Canonical release validation:

```bash
uv run --locked --no-sync ty check
uv run --no-sync maturin develop --release
cargo test --locked --manifest-path native/limulus_native/Cargo.toml --release
uv run --locked --no-sync python scripts/validate_wheel.py
uv run --locked --no-sync sphinx-build -W --keep-going -b html docs docs/_build/html
```

Test file mapping:

| File | Target |
|----------|------|
| `tests/test_session.py` | Session integration / end-to-end scenarios and public API contracts |
| `tests/test_submit.py` | submit/run convenience functions and dictionary Data Step regressions |
| `tests/test_runtime_semantics.py` | Missing, NULL, SUM, BY, and atomic-output behavior |
| `tests/test_native_scalar_fidelity.py` | Arrow scalar and nested-value preservation |
| `tests/test_input_preparation_service.py` | SET/MERGE input preparation contracts |
| `tests/test_apply_public_regression.py` | Public `apply()` boundary and failures |
| `tests/test_parser_isolation.py` | Parser independent import / lazy export / caller capture regression |
| `tests/helpers/tests/test_datastep_case_packs.py` | Release-facing Data Step case packs |

Public tests stay under `tests/`. Internal development, audit, and benchmark assets are local-only and are not included in the public repository.

---

## Building Documentation

```bash
uv run --locked --no-sync sphinx-build -W --keep-going -b html docs docs/_build/html
```

---

## Notes When Modifying Key Files

### When modifying `limulus/execution/coordinator.py` or strategy/runtime boundaries
- `coordinator.py` contains `DataStepExecutor`: service construction, stage order, whole-program gating, multi-block coordination, catalog commit, and public observations.
- Input precedence is explicit `ExecuteRequest.inputs`, generated outputs, registered inputs, then synthetic dictionaries. Successful prefix outputs survive a later block failure in the executor catalog; a failed response does not publish them to `Session.datasets`.
- `input_preparation.py` owns input resolution, dataset options, SET/MERGE preparation, and pre-evaluations. `prepare_typed` builds an `InputPreparationRequest` and delegates to the shared `prepare` core without reparsing source text.
- Preserve typed preparation provenance, lineage, and metadata. Prepared-input observations retain the complete carrier; case-insensitive helper-alias projection belongs only in the native handoff map.
- SET shared columns and MERGE BY keys promote mixed Arrow `string` / `large_string` to `large_string` during input preparation; preserve metadata ownership and keep other type conflicts strict.
- `planning.py`, `runtime_handoff.py`, and `output_handoff.py` own plan construction, native handoff, and output finalization.
- Strategy changes belong in `execution/strategy.py`; Rust invocation and payload changes belong in `backend_integration/*`.
- Always verify with `uv run --no-sync pytest -q` after changes to these files.

### When modifying `native/limulus_native/src`
1. Recompile with `uv run --no-sync maturin develop --release`
2. If native runtime behavior changed, inspect related files under `src/runtime/`, `src/expressions/`, `src/output/`, and the source/view modules
3. Verify with focused backend/runtime tests before broader validation

### When modifying `limulus/grammar/*.lark`
- Lark caches the parser, so verify grammar changes are reflected in existing tests

### When modifying the public API in `limulus/__init__.py`
- Use `__all__` to explicitly declare the public API
- If removing existing re-exports for compatibility, check all usages (tests / docs / examples)
- Keep pure models and host-function contracts immediately importable; resolve data-processing exports lazily through module `__getattr__`.
- Keep caller-frame capture inside the public `submit` / `run` functions. Verify lazy-export identity, star-import, and caller capture with `tests/test_parser_isolation.py`.

### When modifying `limulus/syntax` or `limulus/parser.py`
- `limulus.syntax.parse_program(text, *, source_id="<dsl>")` returns `ProgramParseResult`. `syntax/program.py` owns DATA boundaries and recognized macro/PROC skip regions. The coordinator calls it once after the macro hook and checks stopping diagnostics before any block executes.
- Stopping diagnostics yield `program=None`. Some malformed expressions and IF/DO/ARRAY forms remain whole deferred faults until reached. Dataset-option WHERE retains its restricted dialect.
- Syntax models, parser-result models, and display projections must not import Arrow, Polars, native bindings, Session, or execution services. Standalone parsing requires only Lark and the standard library; importing `limulus.syntax` does not load Lark. Grammar initialization is lazy.
- Pure parsing raises `ParserInitializationError` for mandatory grammar initialization failures. Execution and compatibility facades convert it to `PARSE_BACKEND_CAPABILITY_MISSING`, severity `error`, stage `parse`, without native execution or catalog updates.
- `ParserService` / `LarkParserFrontend` projects the same parse facts through `syntax/compat.py`. Preserve parser-result class identity, source positions, and display contracts. Never reparse display text or use it to reconstruct execution input.
- Keep `SourceDocument.text` raw. Analysis replaces tabs outside strings/comments with one space per tab, preserving codepoint offsets and newlines. Keep only the most recent document/analysis pair by identity; do not expand tabs or introduce global history.
- Binary/Unary spans identify operators, not whole expressions. Stored display edits and boundary spans belong only to compatibility presentation.
- Tests: `test_program_parser.py`, `test_parser_isolation.py`, `test_input_normalization.py`, and the statement-specific `test_program_*.py` modules.

### When modifying validation, expression planning, or rewrites
- `validation.py` consumes typed statements and shares schema binding and diagnostic construction. Validate static KEEP/DROP/BY/RENAME/LABEL references without evaluating row expressions. Duplicate DATA targets fail before native execution.
- `expression_semantics.py` keeps string entries/caches for Column API consumers; typed submit uses the `*_typed` entries. `statement_expressions.py` owns expression-slot traversal.
- `rewrites.py` plans helpers; `syntax_rewrites.py` replaces source-owned calls by original node identity. Preserve span/origin and ordered apply references, update stale display information, and leave whole deferred faults opaque. Runtime-owned helpers/default expressions stay in the row loop.
- Tests: `test_typed_validation.py`, `test_typed_expression_classification.py`, `test_typed_helper_planning.py`, `test_typed_rewrites.py`, and `test_typed_input_preparation.py`.

### When modifying output planning or finalization
- `typed_output_planning.py` owns registered targets, default output, implicit-output suppression, and per-target `effective_output_options`. Rewrites use the same owner to replan outputs.
- Duplicate DATA targets, conflicting nonempty options, and output `IN=` are validation errors. Inline OUTPUT options, including unreachable branches, declare policy for existing targets.
- `output_handoff.py` applies base KEEP/DROP/RENAME before target options, checks normalized rename collisions before deduplication, and restores types, labels, and lineage through existing owners. Metadata restoration must not change arrays, buffers, or computed result types.
- Native owns output WHERE/FIRSTOBS/OBS and per-target counters shared across statements and branches. Python finalization must not filter or count those rows again. A target's schema cannot change midway through a DATA step.
- Tests: `test_typed_output_planning.py`, `test_typed_output_handoff.py`, and `test_native_scalar_fidelity.py`.

### When modifying typed native payloads or lowering
- `syntax/native_payload.py` serializes typed syntax and settled output policy to version 1; structural validation checks required fields, tags, types, and version without replanning or interpreting source text.
- `typed_bridge.py` resolves saved apply references and validates the wire before Arrow export. Registering a callable must not execute it; LABEL values stay with Python metadata owners.
- Native `execute_block` accepts the versioned payload and lowers it once through `typed_ast.rs` / `typed_lowering.rs`. Preserve literal precision, argument order, spans, and deferred causes. Lowering must not evaluate expressions, allocate arrays, or bind functions.
- Typed evaluation uses lowered expressions without a production DSL parser/cache. String `+` concatenates only two text operands after existing NULL checks; `||` uses an ordered concat fold with NULL as empty text. These extensions do not apply to Column API or dataset-option WHERE.
- Shared Python/Rust fixtures live in `native/limulus_native/tests/fixtures/typed_payload_v1.json`. Check both acceptance and rejection cases when changing the wire.
- Tests: `test_typed_wire_contract.py`, `test_typed_native_payload.py`, `test_typed_native_execution.py`, and `test_expression_compatibility.py`, plus Rust typed lowering/evaluation tests.

### When modifying assign result assembly in `limulus/column_api.py`
- `assign_columns` / `_assemble_assign_result` own result assembly. Reuse unchanged columns' resolved Arrow arrays and fields, including nested metadata, nullability, buffers, and chunks.
- Replaced/new columns use computed values and types; do not restore stale source values/types.
- Detect row-count mismatches before `Session.load` and leave the catalog unchanged on failure. Preserve the type/buffer contracts of `finalize_table` and `restore_arrow_schema_from_sources`.
- Tests: `test_assign_schema_preservation.py` and the assign/metadata cases in `test_session_methods.py`.

---

## Module Visibility

| Module | Visibility |
|-----------|---------|
| `Session`, `submit`, `run` | Public user API |
| `SubmitResult`, `LogEntry`, `DatasetCatalog` | Public return types |
| `DataStepExecutor`, `ExecutionStrategyDecision` | Internal / advanced use |
| `io_adapters.*`, `models.*` (except `SubmitResult`, etc.) | Internal |
| `parser.*` (`DataStepAst`, etc.) | Internal / advanced use |
| `limulus.syntax.*` (`parse_program` / syntax models / parser-result models) | Pure syntax layer, independently usable without the execution environment |
| `limulus_native.*` | Rust bindings, not for direct use |
