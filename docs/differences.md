# Differences from SAS language Data Step

This page summarizes the key differences between the SAS language DATA step and limulus.

## IO Support

- Standard support: `pyarrow.Table` / `polars.DataFrame`
- Optional support: `pandas.DataFrame` (`pip install limulus[pandas]`)
- `session.loads()` accepts all three formats listed above

Support for sas7bdat input/output is not planned, as the format specification is proprietary.  
If you need to work with sas7bdat files, use pandas or other libraries to load them first.

---

## Type Mapping

| Arrow Type | SAS language Equivalent | Python |
|-----------|----------|--------|
| `int64`, etc. | Numeric | `int` |
| `float64`, etc. | Numeric | `float` |
| `string` | Character | `str` |
| `bool` | Numeric (0/1) | `bool` |
| `date`, etc. | Numeric | `datetime` |

Arrow supports a wide variety of data types.  
Complex types such as structs that do not exist in SAS are not expected to be used in logic, but can be retained as columns without issue.

---

## Behavioral Differences

### Labels
Dataset labels and column labels are supported, but they are stored as Arrow metadata rather than as a separate display-layer construct. Dataset labels live in schema metadata under `memlabel`, and column labels live in per-field custom metadata.

### Dataset Option Type Preservation
Physical types are usually preserved through processing.  
If exact physical types matter, it is still recommended to perform explicit type conversion at the final output stage. In practice, prefer `DatasetView.cast(...)` or `DatasetView.astype(...)` as the last transformation before using the result.

### SQL API
`Session.sql()` is available for read-oriented queries, `CREATE TABLE ... AS ...` style result persistence, and `DROP TABLE ...` dataset removal. The feature is backed by the Polars SQL engine and is intended as a practical session-level query helper rather than a full PROC SQL reimplementation.

`dictionary.tables` and `dictionary.columns` are available in SQL. limulus also registers `dictionary_tables` and `dictionary_columns` aliases because some SQL engines treat dotted names as schema-qualified identifiers.

### Dictionary Tables
Dictionary metadata is generated dynamically from Arrow schema metadata rather than from a separate catalog store.

- `MEMLABEL` comes from table schema metadata key `memlabel`
- `LABEL` comes from per-field metadata key `label`
- `TYPE` reports the Arrow type string such as `int64` or `string`
- `LENGTH` is not exposed in `dictionary.columns`

This means the metadata contract is Arrow-centric even when the API shape follows DICTIONARY-style naming.

### Name Resolution
Dataset names are case-insensitive, and a `work.` prefix is stripped during lookup. Column-oriented Session / DatasetView helpers also resolve column names case-insensitively when the match is unique.

### Transpose / Assign API
`Session.transpose()` / `DatasetView.transpose()` and `Session.assign()` / `DatasetView.assign()` are Python-side column APIs rather than DATA step statements.

Current transpose scope is intentionally narrow:

- Without `id`, output follows a PROC TRANSPOSE-like shape with `_NAME_` and `COL1..COLn` columns.
- With `id`, the current release accepts exactly one `id` column name and one `var` column.
- Duplicate or missing `id` values within a group raise an error instead of applying name mangling rules.

`transpose()` is a convenience helper. Performance is not a primary design goal; for performance-critical reshaping, prefer Arrow or Polars directly.

For `assign()`, string values are interpreted as expressions. To assign a string literal, quote it inside the expression, for example `flag="'A'"`.

`assign()` is now executed through a column-oriented expression pipeline. The current release guarantees column-oriented execution for literals, arithmetic / comparison expressions, `case when`, and the built-in functions.
### PUT / INPUT Scope
limulus now provides a limited `put(...)` / `input(...)` helper surface in both helper expressions and the Python runtime.

Current built-in families are:

- numeric to text: `best.`, `w.d.`, `zw.d`, `zw.d.`, `commaw.d.`
- text to numeric: `best.`
- text to date / datetime / time: `e8601da.`, `e8601dt.`, `yymmdd6.`, `yymmdd8.`, `yymmdd10.`, `time.`
- date / datetime / time to text: `e8601da.`, `e8601dt.`, `yymmdd6.`, `yymmdd8.`, `yymmdd10.`, `time.`

`input(...)` returns Python / Arrow-compatible numeric, `date`, `datetime`, and `time` values rather than date/time numeric values. This is intentionally a practical subset, not a full format catalog.

`Session.register_format(...)` and `Session.register_informat(...)` now support two extension styles:

- exact-match dict catalogs, which are serializable and can stay on the Rust-first DATA step path when the rest of the workload is eligible
- callable hooks, which remain Python-side extension points

For dict catalogs, `put(...)` returns text and falls back to the original value on no-match, while `input(...)` returns `null` on no-match. Character custom formats use `$name.` in DATA step syntax but register as `name` plus `namespace="character"` in Python. Dict custom `input(...)` is intentionally float64-only in this phase; broader typed custom informat support is deferred.

### PROC / Macro Skip Scope
Some unsupported syntax is skipped before Data Step parsing so mixed legacy source can still be processed.

Skipped forms:

- `%let ...;`
- `%put ...;`
- `%* ...;`
- `%macro ... %mend;`
- `%macro ... %mend name;`
- `proc ... run;`
- `proc ... quit;`

Skipped syntax is ignored rather than executed. Macro variable expansion, open-code macros, and procedure semantics remain unsupported.

### length
Character length is variable by default, so no character truncation occurs.

### BY Groups
Pre-sorting is not required to use BY groups.

### Missing Values
SAS language numeric missing values use the special `.` symbol, but limulus treats them as `null` (Arrow).  
SAS language character missing values are `""`, but in limulus, following the Arrow specification, `""` and `null` are treated as distinct values.  
Use `missing()` to check for both `""` and `null` together.  
Assign `null` using `=.` for numeric types, or `=None` for both numeric and character types. (Support for `call missing` is also under consideration.)

### Data Joining
Merge operates like SQL; if there are duplicate column names, an error is raised.

### Array
Both numeric and character types can be included in the same array without issues.

```python
result = session.submit("""
data out; 
  set iris; 
  array var [*] species sepal_length ; 
  do i = 1 to dim(var); 
    vname = vname(var[i]); 
    output ;
  end ; 
  stop ;
  keep vname ;
run;
""")
```

### Additional Operators
| Operator | Added |
|---|---|
|`=` | `==`|
|`^=` |`!=`|
|`\|\|`| `+`| 

### Custom Functions

**`lead` function**  
Looks ahead to a future row's value, opposite to `lag`. Useful for cases where you need the next value, such as computing differences.

**`shift` function**  
A general-purpose shift function: negative values behave like `lag`, positive values like `lead`.

**`apply` function**  
Allows calling Python lambda functions or custom functions from within a Data Step.  
Useful for extending functionality not covered by limulus's built-in functions, such as complex string operations.



## Unsupported Features

| SAS language Feature | Notes |
|---------|------|
| Attrib | Full ATTRIB parity is not implemented yet; use dataset labels and LABEL statements for supported metadata cases |
| Format (`FORMAT`, `INFORMAT`) | Exact-match dict catalogs and Python callable registry hooks are available; SAS `FORMAT` / `INFORMAT` statements are not implemented |
| Macro variables (`%let`, `&var`) | `%let` / `%put` / `%macro ... %mend` are skipped, not executed; `&var` expansion is not implemented |
| PROC statements | `proc ... run;` / `proc ... quit;` blocks are skipped so surrounding DATA steps can still be parsed, but procedure semantics are not implemented |
| `INFILE` / `FILE` | Not planned; handle on the Python side |
| `INPUT` / `DATALINES` | Not planned; handle on the Python side |
| Colon-based options (`=:`, `aa:`) | Workarounds available; under consideration |
| Range notation (`a-z`, `a1-a3`) | Workarounds available; under consideration |
| Variable groups (`_numeric_`, `_character_`, `_all_`) | Workarounds available; under consideration |
| Character operators (e.g., `eq`) | May be implemented if there is demand |
| CALL subroutines | `symputx` and `missing` are planned |

In general, features that have no practical workaround and are commonly used will be prioritized.  
For features that are costly to implement and have good Python-side alternatives, those alternatives are recommended instead, freeing resources for areas like performance.  
For example, external file I/O (`INFILE` / `FILE`) is well served by pandas, polars, or jinja2.

---

## Sorting and Column Reordering

Operations typically performed by procedures such as sorting are executed through the column-oriented API.  
The current release exposes practical session-local sorting and column reordering helpers rather than a separate procedure layer.  
Column reordering can be done with `select` or `keep`.

```python
session.dataset("ds").sort("x")
session.dataset("ds").sort(["x", "y"])
session.dataset("ds").sort(["x"], nodupkey=True)
session.dataset("ds").sort([("x", "Ascending"),("y", "Descending")])

session.dataset("ds").select(["x","y"])
```

---

## Recommended Patterns

- [ ] `%LET` macro variables → Python variables + f-string to build DSL strings
- [ ] `INFILE` → pass Polars / Pandas to `session.loads()`
- [ ] Missing value checks → use `missing()` function or `cmiss()|nmiss()`
