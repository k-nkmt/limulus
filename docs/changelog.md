# Changelog

All notable changes to this project are documented on this page.

## [Unreleased]

### Added

- `Session.transpose(...)` and `DatasetView.transpose(...)` for minimal PROC TRANSPOSE-like reshaping with `by`, `id`, `var`, and `out`.
- `Session.assign(...)` and `DatasetView.assign(...)` for ordered Data Step-style column creation with literals, expressions, built-in functions, and `case when ... then ... else ... end`.

### Changed

- Column-oriented reshape logic now uses shared column-resolution and table-rebuild helpers in `session.py` to keep future case-insensitive and metadata-stability work localized.

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