# API Reference

Public API for using limulus from Python.

## Top-Level Functions

Convenience functions that can be executed as one-shot operations without creating a session.

```{eval-rst}
.. autofunction:: limulus.submit

.. autofunction:: limulus.run
```

## Session

The main class responsible for dataset management and Data Step execution.

```{eval-rst}
.. autoclass:: limulus.Session
    :members: dataset, datasets, work, submit, run, include, load, loads, delete, unload, sql, to_arrow, to_pandas, to_polars, dictionary, register_format, register_informat, get_log, get_option, log
    :undoc-members:
```

## DatasetView

A view class returned by :meth:`Session.dataset` for chained operations.

```{eval-rst}
.. autoclass:: limulus.session.DatasetView
    :members:
    :undoc-members:
```

## SubmitResult

The return value of :meth:`Session.submit`.

```{eval-rst}
.. autoclass:: limulus.SubmitResult
    :members:
    :undoc-members:
```

## LogEntry

A class representing a single entry in the execution log.

```{eval-rst}
.. autoclass:: limulus.LogEntry
    :members:
    :undoc-members:
```

## Column-Oriented Helpers

`Session` and `DatasetView` expose reshape and derived-column helpers as part of the public API.  
Dataset names are resolved case-insensitively, and a `work.` prefix is stripped during lookup.
These helpers include procedure-like operations such as `sort(...)` and `transpose(...)`, along with `assign(...)` for fast, simple column-oriented transformations.

`assign(...)` evaluates assignments from left to right and treats string values as expressions rather than string literals. It is intended for simple derived-column work such as literals, arithmetic and comparison expressions, `case when`, and the built-in helper functions. Unsupported constructs raise `ValueError`.


## Registries

Built-in format / informat families currently cover `w.`, `w.d`, `w.d.`, `zw.`, `zw.d`, `zw.d.`, `commaw.`, `commaw.d`, `commaw.d.`, plus named forms such as `e8601da.`, `e8601dt.`, `yymmdd6.`, `yymmdd8.`, `yymmdd10.`, and `time.`. For numeric formats, `w` is the total width, `d` is the number of fractional digits, and omitted `d` defaults to `0`. Bare `w` / `zw` / `commaw` without any period are rejected. limulus does not space-pad values when the rendered text is shorter than `w`; `zw.d` zero-fills instead.

The shared registry also supports exact-match dict catalogs:

- `Session.register_format(name, mapping)` for numeric `put(...)` lookups
- `Session.register_format(name, mapping, namespace="character")` for `$name.` character-format lookups
- `Session.register_informat(name, mapping)` for float64-only `input(...)` lookups

Dict `put(...)` catalogs return text and fall back to the original value on no-match. Dict `input(...)` catalogs return `null` on no-match. Callable formatter / parser registration remains supported for advanced Python-side hooks; callable custom `input(...)` may still need `kind=` so `assign(...)` can determine the output dtype.

`DatasetView.astype(...)` / `DatasetView.cast(...)` also support `out=` so type-converted results can be written into a new dataset instead of replacing the source. When the cast result should be written into a new column instead of replacing the source, use `alias=`.

They also expose dynamic dictionary metadata views:

- `session.dictionary.tables`: dataset-level metadata for the current session
- `session.dictionary.columns`: column-level metadata for all session datasets
- `session.dictionary("name")`: column metadata for one dataset
- `session.dataset("name").dictionary`: dataset-scoped dictionary view


## Diagnostics

Execution diagnostics now carry structured source metadata in addition to message text.

- `DiagnosticSpan`: source offset and line/column range for a primary location
- `DiagnosticLabel`: labeled span metadata for renderer input
- `LogEntry`: stage, span, labels, notes, and optional source text preserved from execution diagnostics

`SubmitResult.format_log()` uses these fields to render excerpt-based plain-text diagnostics when source text is available, and falls back to the older message-only format otherwise.

The same renderer path is used across `parse`, `validate`, and `execute` stages. This also applies to non-DSL parse points such as `case when`, simple filter / SQL classification, and runtime regex parsing, so source-aware diagnostics share the same display contract regardless of where the error originated.
