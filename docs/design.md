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
      1) split blocks
      2) macro hook
      3) parse
      4) resolve inputs
      5) pre-processing inputs
      6) pre-evaluations
      7) execute runtime backend
      8) resolve outputs (temporary/internal variable filtering)
    -> Output conversion (arrow_table)
  -> Session.datasets update
```


### 1. Block Splitting
Code passed to `submit()` is split into `DATA ... RUN;` units and executed sequentially.  
This means that when multiple blocks are passed at once, the output of an earlier block can be referenced as input by subsequent blocks.

### 2. Parsing
Data Step code is currently parsed using [lark](https://github.com/lark-parser/lark).

For example, code like `data out; set iris(in=in1) ; where sepal_length > 5; if species ^= 'setosa'; keep species sepal_length sepal_width ;run;` is currently parsed as follows:
```
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

### 3. Input Resolution

References specified in `SET` / `MERGE` are resolved in the following order:

- Specified at `submit()` time
- Inputs registered via `Session.loads()`
- Outputs produced by earlier blocks within the same `submit()` call

Name resolution is case-insensitive and handles the `work.` prefix transparently.

### 4. Backend Selection
Row-oriented processing tends to be less efficient than column-oriented processing.  
To improve execution speed, a Rust-based runtime module is provided.  
The Rust backend is used by default.  
However, if a diagnostic error occurs for cases that cannot be handled by the Rust backend (e.g., `apply()`), the Python backend is automatically used as a fallback.

### 5. Row Loop Processing
Manages the basic per-row loop processing and its associated automatic variables.

- `_N_`: incremented on each row iteration
- `_ERROR_`: initialized to 0 at the start of each row; set to 1 on execution errors
- `BY` assigns `FIRST.<var>` / `LAST.<var>` automatic variables per row

- Output destination is controlled with `DATA out1 out2;` and `OUTPUT out1;`
- If no explicit `OUTPUT` is present, the default output destination (usually the first DATA target) is used

### 6. Output Conversion and Session Update
Processed results are converted to `arrow_table` and then reflected in the catalog.  
This allows `session["name"]` to be retrieved as an Arrow Table.

In the output stage, internal temporary variables are also removed in a backend-agnostic way.
This includes helper variables created by `IN=`, `INDSNAME=`, `END=`, and `FIRST./LAST.` (including renamed aliases).

### 7. Stage-aware Diagnostics and Logs

Pipeline diagnostics keep stage information as a dedicated field (`stage`) rather than embedding it in diagnostic messages.
`Session.submit()` propagates this value into log entries so each log line can identify the pipeline stage independently.


## Input Data

Arrow, Polars, and Pandas are supported as input formats.  
For CSV, Parquet, sas7bdat, or other file formats, load them first with any library of your choice.  
Internally, limulus uses Arrow for data exchange, so Arrow or Polars inputs are recommended for best performance.


## Performance

Handling large datasets is a key motivation for using Python, so performance is an explicit design consideration.

Row-oriented processing is inherently slower than columnar processing, which operates on entire columns at once.  
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

With the Rust runtime, processing time is reduced to about half compared with the Python backend.  
Compared with column-oriented processing in pandas or polars, limulus is at a disadvantage because those libraries can process entire columns in bulk. However, when compared against row-wise patterns such as `iterrows`, performance drops sharply in pandas, and the Rust runtime in limulus still runs faster than pandas `iterrows`.


| rows | limulus rust(ms) | limulus python(ms) | pandas(ms) | polars(ms) | pandas iterrows(ms) | polars iterrows(ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 10000 | 87.70 | 201.63 | 2.38 | 1.62 | 141.35 | 8.78 |
| 100000 | 757.76 | 1952.34 | 8.86 | 3.05 | 1432.81 | 89.39 |
| 1000000 | 8640.51 | 20336.69 | 112.09 | 30.96 | 14621.46 | 884.21 |


---


## Roadmap (Under Consideration)

### Short-term (v0.x)

1. Improved stability (bug fixes, expanded parser coverage, etc.)
2. Additional supported functions (string-related, `put`, etc.)
3. Column-oriented API additions (basic data operations, SQL query execution)
4. Performance improvements in non-runtime processing areas
5. Support for label-based metadata settings

### Mid-term (beta release v0.x – v1.0)

1. Improved reliability through expanded and organized test coverage
2. Enhanced logging and debugging capabilities

### Long-term (TBD)

1. Support for macro variables and open-code macros
2. Support for dictionary tables
3. Further runtime performance improvements
