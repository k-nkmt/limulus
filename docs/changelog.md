# Changelog

All notable changes to this project are documented on this page.

## [v0.5.0] - 2026-06-04

`v0.3.x` and `v0.4.x` are treated as internal development milestones, and this section summarizes the changes rolled up into `v0.5.0`.

### Added

- `Session.transpose(...)` and `DatasetView.transpose(...)` for minimal PROC TRANSPOSE-like reshaping with `by`, `id`, `var`, and `out`.
- `Session.assign(...)` and `DatasetView.assign(...)` for ordered Data Step-style column creation with literals, expressions, built-in functions, and `case when ... then ... else ... end`, executed left-to-right through Polars column expressions for supported assignment syntax.
- Built-in `put(...)` and `input(...)` support in helper expressions, backed by a shared session format / informat registry.
- `Session.dictionary.tables`, `Session.dictionary.columns`, `Session.dictionary(name)`, and `DatasetView.dictionary` for dynamic Arrow-metadata-backed dictionary views.
- Structured diagnostics now carry span/label/source metadata across parse and validation failures.

### Changed

- `SubmitResult.format_log()` now renders structured diagnostics through the diagnostic renderer, includes error counts, and shows source excerpts when available.
- Case-insensitive resolution is now applied more consistently across Session/DatasetView column helpers, runtime variable lookup, and `WORK.`-prefixed dataset references.
- The execution pipeline now reserves a `validate` stage ahead of input resolution, and rejects unresolved input datasets plus `DICTIONARY.*` output targets before execution.
- Unsupported `%let`, `%put`, `%*`, `%macro ... %mend`, and `proc ... run|quit;` blocks are now skipped through parser-based split-stage region detection so surrounding supported code can still execute without string-scanner false positives.
- Numeric `put(...)` formatting now treats `w` as total width, defaults `d` to `0`, supports `zw.d` zero-filled output, and keeps limulus's no-space-padding rule for short values. `astype(...)` / `cast(...)` also support `alias=` for writing typed results into new columns.
- `Session.sql(...)` now supports `DROP TABLE ...` and DICTIONARY table queries in addition to read queries and `CREATE TABLE ... AS ...`.
- Session dataset lookup now strips a `WORK.` prefix consistently and normalizes through a shared dataset-key helper.
- The Rust backend now consumes parser-provided structured AST payloads for `IF` / `DO` / `ARRAY` and dataset references instead of maintaining a separate subset parse path.
- Helper pipelines now preserve Arrow metadata and numeric column types more consistently across transpose and Arrow/Polars roundtrips.
- The Python backend now has a narrower role, and data handoff is now Arrow-based.

### Fixed

- Multi-block `Session.submit(...)` now prefers datasets created earlier in the same submit call when a later `SET` reuses the same dataset name, preventing stale rows from pre-existing session tables from being mixed into rewritten outputs.

## [v0.2.0] - 2026-03-17

### Added

- `firstobs=` and `obs=` source dataset options
- `Session.sql(query)`.
- `DatasetView.astype(mapping)` for dataset-scoped type conversion in chained workflows.
- Dataset labels in Arrow schema metadata under `memlabel`, and column labels in per-field Arrow metadata.
- `Session.include(path)` for loading and executing external Data Step scripts from the current session.
- `DatasetView.select(columns, out=...)` for explicit column selection.
- Dictionary-based `Session.set_option({...})` and `Session.get_option(...)` as reserved session options for future use.

### Changed

- Source-side dataset option processing order is now `keep/drop` → `where` → `rename` → `obs/firstobs`.
- `out=` is now the primary user-facing output parameter for Session/DatasetView column-oriented helpers; `DatasetView` no longer exposes `target=` and `Session` keeps it only as a compatibility alias.
- The Python backend implementation was split into `executor_python.py`, `executor_py_stage.py`, `executor_py_stmt.py`, and `executor_py_data.py`.

### Fixed
- Parsing now accepts `MERGE ... END=`.


## [v0.1.0] - 2026-03-03

Initial release.
