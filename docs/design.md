# Design


## Concept

Bringing Data Step Skills into Python Workflows

While the rapid evolution of Python's data processing ecosystem is a strength, keeping up with changes, managing dependencies, and shifting from row-oriented to column-oriented thinking can impose unexpected costs.  
Data Step, on the other hand, has been used for decades with virtually unchanged syntax, and is unlikely to change significantly going forward.  
The name *limulus* — the horseshoe crab, a living fossil — reflects this same enduring stability.  
Although it has some limitations, row-oriented and procedural notation can often express certain types of processing more naturally.

While generative AI has made Python coding easier, thorough review remains especially important for data processing logic.  
This library was created with the belief that carrying Data Step skills into Python can open new doors — whether for reviewing code, working on personal projects, or working independently outside a dedicated analytics environment.

For these reasons, this library is not a complete reimplementation or drop-in replacement for SAS language.  
Rather than rarely-used edge cases or SAS language-specific quirks, it prioritizes practical functionality, safe and predictable behavior, and draws on design ideas from other modern libraries.


## Architecture Overview

```text
Session.submit()
  -> DataStepExecutor.execute()
    -> ExecutionPipelineCoordinator
      1) macro hook / unsupported syntax skip
      2) split blocks
      3) parse
      4) validate
      5) resolve inputs
      6) pre-processing inputs
      7) pre-evaluations
      8) plan generation
      9) execute runtime backend
     10) resolve outputs (temporary/internal variable filtering)
     11) apply output metadata
  -> Session.datasets update
```


### 1. Unsupported PROC / Macro Skip
Before block splitting, limulus masks a narrow subset of unsupported legacy syntax so mixed source files can still reach the Data Step parser.

Current skip scope:

- `%let ...;`
- `%put ...;`
- `%* ...;` macro comments
- `%macro ... %mend;` and `%macro ... %mend name;`
- `proc ... run;` and `proc ... quit;`

These constructs are skipped rather than executed. Open-code macro expansion and procedure semantics are still outside the supported surface.

### 2. Block Splitting
Code passed to `submit()` is split into `DATA ... RUN;` units and executed sequentially.  
This means that when multiple blocks are passed at once, the output of an earlier block can be referenced as input by subsequent blocks.

### 3. Parsing
Data Step code is currently parsed using [lark](https://github.com/lark-parser/lark).
The parser produces a structured internal representation that can be inspected programmatically.  

For example, code like `data out; set iris(in=in1) ; where sepal_length > 5; if species ^= 'setosa'; keep species sepal_length sepal_width ;run;` is currently recognized as the following statement structure:

```text
start
  statement
    data_stmt
      data_kw	data
      dataset_ref
        dataset_name	out
  statement
    set_stmt
      set_kw	set
      set_component
        dataset_ref
          dataset_name	iris
          dataset_options
            dataset_option	in=in1
  statement
    where_stmt	where sepal_length > 5
  statement
    if_stmt
      subset_if_stmt	if species ^= 'setosa' 
  statement
    keep_stmt	keep species sepal_length sepal_width  
  statement
    run_stmt	run
```

The exact internal rule names may evolve over time, but the important point is that statements, dataset references, and options are separated before execution starts.

### 4. Input Resolution

References specified in `SET` / `MERGE` are resolved in the following order:

- Specified at `submit()` time
- Inputs registered via `Session.loads()`
- Outputs produced by earlier blocks within the same `submit()` call

Dataset name resolution is case-insensitive and handles the `work.` prefix transparently. Column-oriented helpers such as `select`, `where`, `transpose`, `assign`, and `cast` also resolve column names case-insensitively when the match is unique.

### 5. Backend Selection
Row-oriented processing tends to be less efficient than column-oriented processing.  
To improve execution speed, a Rust-based runtime module is provided.  
The Rust backend is used by default.  
Supported `apply()` calls do not force a whole-block Python row loop. When the function target can be resolved up front from a string name, the row loop stays on Rust and only the callable invocation crosses a narrow Python gateway. This keeps builtins, dotted module functions, and in-scope helper functions available without restoring broad Python runtime ownership.

Input-stage dataset options such as `keep=`, `drop=`, `where=`, `rename=`, `firstobs=`, and `obs=` are normalized in shared Python-side preprocessing before the runtime loop when needed. This keeps the row-loop semantics consistent across Python and Rust backends without duplicating the same preparation rules in multiple runtimes.

By contrast, `MERGE` preparation and `lag(...)` / `lead(...)` rewrite materialization remain part of `pre-evaluations`. Those steps depend on statement semantics and runtime rewrite planning, not only on source dataset option normalization.

### 6. Session Helpers and Metadata Views

In addition to `submit()`, `Session` and `DatasetView` provide user-facing helpers for common column-oriented tasks.

- `transpose(...)` offers a compact reshape helper for common wide/long conversions.
- `assign(...)` adds or replaces columns from expressions, literals, and `case when` logic while keeping left-to-right assignment order.
- `assign(...)` and the DSL both support limited built-in `put(...)`, `input(...)`, and `hour(...)` conversion helpers.
- `Session` maintains a shared format / informat registry. Built-in families cover `best.` / `w.` / `w.d` / `w.d.`, `zw.` / `zw.d` / `zw.d.`, `commaw.` / `commaw.d` / `commaw.d.`, and named forms such as `e8601da.`, `e8601dt.`, `yymmdd6.`, `yymmdd8.`, `yymmdd10.`, and `time.`. 
- `astype(...)` / `cast(...)` remain helper-oriented type conversion APIs, can materialize into a new dataset by using `out=`, and can write cast results into new columns by using `alias=`.
- `sql(...)` provides practical session-level querying, including read queries, `CREATE TABLE ... AS ...`, and `DROP TABLE ...`.
- `dictionary.tables`, `dictionary.columns`, and dataset-scoped dictionary views expose session metadata such as labels, row counts, and Arrow column types.

### 7. Row Loop Processing
Manages the basic per-row loop processing and its associated automatic variables.

- `_N_`: incremented on each row iteration
- `_ERROR_`: initialized to 0 at the start of each row; set to 1 on execution errors
- `BY` assigns `FIRST.<var>` / `LAST.<var>` automatic variables per row

- Output destination is controlled with `DATA out1 out2;` and `OUTPUT out1;`
- If no explicit `OUTPUT` is present, the default output destination (usually the first DATA target) is used

### 8. Output Conversion and Session Update
Processed results are converted to `arrow_table` and then reflected in the catalog.  
This allows `session["name"]` to be retrieved as an Arrow Table.

In the output stage, internal temporary variables are also removed in a backend-agnostic way.
This includes helper variables created by `IN=`, `INDSNAME=`, `END=`, and `FIRST./LAST.` (including renamed aliases).

Dataset labels are stored in Arrow schema metadata under `memlabel`, and column labels are stored in each Arrow field's custom metadata. This allows label information produced by `DATA ... (label="...")` and `LABEL` statements to survive round-trips through `Session`.

### 9. Stage-aware Diagnostics and Logs

Pipeline diagnostics keep stage information as a dedicated field (`stage`) rather than embedding it in diagnostic messages.
`Session.submit()` propagates this value into log entries so each log line can identify the pipeline stage independently.

When source text is available, log output also includes short source excerpts. This makes parse, validation, and execution errors easier to read without requiring users to inspect internal parser details.


## Input Data

Arrow, Polars, and Pandas are supported as input formats.  
For CSV, Parquet, sas7bdat, or other file formats, load them first with any library of your choice.  
Internally, limulus uses Arrow for data exchange, so Arrow or Polars inputs are recommended for best performance.


## Performance

Handling large datasets is a key motivation for using Python, so performance is an explicit design consideration.

Row-oriented processing is inherently slower than columnar processing, which operates on entire columns at once.  
For that reason, the main Data Step runtime focuses on predictable row-loop semantics, while user-facing helpers such as `assign()` can use a column-oriented expression path for supported syntax.

For wide or performance-sensitive pure column operations, `assign(...)`, `cast(...)` / `astype(...)`, and direct Arrow / Polars usage are often better fits than forcing row-oriented logic. limulus aims to keep those boundaries explicit rather than promising blanket parity with columnar engines.

For reference, limulus uses the following iris-like neutral scenario for benchmark comparisons:

Processing scenario:
```sas
data setosa_like others;
  set flowers;
  where sepal_length > 4.5;
  sepal_area = round(sepal_length * sepal_width, 0.01);
  petal_ratio = round(petal_length / petal_width, 0.01);
  if petal_length > 2.5 then do;
    segment = "others";
    output others;
  end ;
  else do;
    segment = "setosa_like";
    output setosa_like;
  end ;
run;
```

For row-oriented execution, limulus is still slower than `polars.iter_rows`, but it remains substantially faster than `pandas.iterrows` for the same workload.
limulus also carries a fixed overhead of roughly 50 ms for parsing, preprocessing, and related setup work regardless of input size. As the dataset grows and that fixed-cost share becomes smaller, limulus tends to run at about four times the runtime of Polars and about one third of the runtime of pandas for this row-oriented benchmark.
This comparison uses `pandas.iterrows()`, which is often chosen for its simple column-name-based access even though `pandas.itertuples()` is usually much faster and can be much closer to Polars row iteration performance for the same workload.

| rows | limulus_ms | pandas_iterrows_ms | polars_iterrows_ms |
|---:|---:|---:|---:|
| 10000 | 94.17 | 147.32 | 9.06 |
| 100000 | 455.67 | 1433.75 | 111.30 |
| 1000000 | 4220.09  | 14675.22 | 980.06 |

When the same transformation is expressed through `limulus.assign(...)`, performance improves substantially because the helper uses a Polars-based columnar execution path internally. In practice, this makes it much closer to Polars than to row-oriented execution.
This path also carries a fixed overhead of roughly 50 ms, so it remains somewhat slower than other columnar-style processing at small sizes. However, at around 100k rows it can already run faster than `polars.iter_rows` for the same transformation.

| rows | limulus_assign_ms | pandas_ms | polars_ms |
|---:|---:|---:|---:|
| 10000 | 54.11 | 1.93 | 0.81 |
| 100000 | 63.57 | 8.57 | 2.72 |
| 1000000 | 108.52 | 105.83 | 28.25 |


## Roadmap (Under Consideration)

### Short-term (v0.x)

- [ ] Improved stability (bug fixes, expanded parser coverage, etc.)  
  -> Parser/runtime coverage and validation have improved.
- [ ] Additional supported functions (string-related, `put`, etc.)  
  -> Support for `put(...)` and `input(...)` has been added.
- [x] Broader helper coverage and improved ergonomics around the existing column-oriented API
- [ ] Performance improvements in non-runtime processing areas  
  -> Some pipeline cleanup and helper improvements have landed. Data handoff costs were reduced and performance improved, but further code cleanup is still needed.
- [x] Support for label-based metadata settings

### Mid-term (beta release v0.x – v1.0)

- [ ] Improved reliability through expanded and organized test coverage
  -> Migrated tests to a case-based structure for easier integration with other systems.
- [ ] Enhanced logging and debugging capabilities  
  -> Stage-aware and source-excerpt-based diagnostics are available, but deeper debugging support is still planned.
- [x] Support for Dataset-JSON  
  -> [dsjframe](https://github.com/k-nkmt/dsjframe) has been released as a standalone library.

### Long-term Outlook

- [ ] Support for macro variables and open-code macros
- [x] Support for dictionary tables
- [ ] Further runtime performance improvements
