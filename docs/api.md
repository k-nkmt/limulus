# API Reference

Public API for using limulus from Python.

## Column-Oriented Helpers

`Session` and `DatasetView` expose reshape and derived-column helpers as part of the public API.

`assign(...)` expressions and DATA step expressions share the same limited format-aware helper surface:

- `put(value, format_name)`
- `input(value, informat_name)`
- `hour(value)`

Built-in format / informat families currently cover `w.`, `w.d`, `w.d.`, `zw.`, `zw.d`, `zw.d.`, `commaw.`, `commaw.d`, `commaw.d.`, plus named forms such as `e8601da.`, `e8601dt.`, `yymmdd6.`, `yymmdd8.`, `yymmdd10.`, and `time.`. For numeric formats, `w` is the total width, `d` is the number of fractional digits, and omitted `d` defaults to `0`. Bare `w` / `zw` / `commaw` without any period are rejected. limulus does not space-pad values when the rendered text is shorter than `w`; `zw.d` zero-fills instead.

`DatasetView.astype(...)` / `DatasetView.cast(...)` also support `out=` so type-converted results can be written into a new dataset instead of replacing the source. When the cast result should be written into a new column instead of replacing the source, use `alias=`.

For advanced cases, `Session.register_format(...)` and `Session.register_informat(..., kind=...)` extend the same registry used by helper expressions and the Python runtime.

They also expose dynamic dictionary metadata views:

- `session.dictionary.tables`: dataset-level metadata for the current session
- `session.dictionary.columns`: column-level metadata for all session datasets
- `session.dictionary("name")`: column metadata for one dataset
- `session.dataset("name").dictionary`: dataset-scoped dictionary view

Dataset names are resolved case-insensitively, and a `work.` prefix is stripped during lookup.

```{eval-rst}
.. automethod:: limulus.Session.transpose

.. automethod:: limulus.Session.assign

.. automethod:: limulus.session.DatasetView.transpose

.. automethod:: limulus.session.DatasetView.assign
```

## Top-Level Functions

Convenience functions that can be executed as one-shot operations without creating a session.

```{eval-rst}
.. autofunction:: limulus.submit

.. autofunction:: limulus.run
```

## Session

The main class responsible for dataset management and Data Step execution.

`Session.sql(...)` supports read queries, `CREATE TABLE ... AS ...`, and `DROP TABLE ...` against session datasets. DICTIONARY tables are available in SQL as `dictionary.tables` / `dictionary.columns`, and also through the aliases `dictionary_tables` / `dictionary_columns` for engines that prefer simple identifiers.

`Session.register_format(...)` / `Session.register_informat(..., kind=...)` are extension points for the shared format registry. They are intended for Python-side custom formatter / parser registration rather than for implementing SAS `FORMAT` / `INFORMAT` statements.

```{eval-rst}
.. autoclass:: limulus.Session
   :members:
   :undoc-members: False

   :exclude-members: select, filter, transpose, assign
```

## DatasetView

A view class returned by :meth:`Session.dataset` for chained operations.

```{eval-rst}
.. autoclass:: limulus.session.DatasetView
   :members:
   :undoc-members: False

   :exclude-members: transpose, assign
```

## SubmitResult

The return value of :meth:`Session.submit`.

```{eval-rst}
.. autoclass:: limulus.SubmitResult
   :members:
```

## LogEntry

A class representing a single entry in the execution log.

```{eval-rst}
.. autoclass:: limulus.LogEntry
   :members:
```

## Diagnostics

Execution diagnostics now carry structured source metadata in addition to message text.

- `DiagnosticSpan`: source offset and line/column range for a primary location
- `DiagnosticLabel`: labeled span metadata for renderer input
- `LogEntry`: stage, span, labels, notes, and optional source text preserved from execution diagnostics

`SubmitResult.format_log()` uses these fields to render excerpt-based plain-text diagnostics when source text is available, and falls back to the older message-only format otherwise.

The same renderer path is used across `parse`, `validate`, and `execute` stages. This also applies to non-DSL parse points such as `case when`, simple filter / SQL classification, and runtime regex parsing, so source-aware diagnostics share the same display contract regardless of where the error originated.
