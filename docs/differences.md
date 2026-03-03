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
| Label / Attrib | Planned for future implementation |
| Data transposition | Handle on the Python side; `retain` workaround available; `transpose` API planned separately |
| Numeric format (`PUT(x, 8.2)`) | Planned for future implementation; `apply` can be used as a workaround |
| Format (`FORMAT`, `INFORMAT`) | Handle with if statements and merge; to be revisited |
| Macro variables (`%let`, `&var`) | Planned; use Python f-string as an alternative |
| `INFILE` / `FILE` | Not planned; handle on the Python side |
| `INPUT` / `DATALINES` | Not planned; handle on the Python side |
| Colon-based options (`=:`, `aa:`) | Workarounds available; under consideration |
| Range notation (`a-z`, `a1-a3`) | Workarounds available; under consideration |
| Variable groups (`_numeric_`, `_character_`, `_all_`) | Workarounds available; under consideration |
| Character operators (e.g., `eq`) | May be implemented if there is demand |
| CALL subroutines | `symputx` and `missing` are planned |
| Dictionary tables | Planned for future implementation |

In general, features that have no practical workaround and are commonly used will be prioritized.  
For features that are costly to implement and have good Python-side alternatives, those alternatives are recommended instead, freeing resources for areas like performance.  
For example, external file I/O (`INFILE` / `FILE`) is well served by pandas, polars, or jinja2.

---

## Sorting and Column Reordering

Operations typically performed by procedures such as sorting are executed through the column-oriented API.  
A high-performance polars-based wrapper is planned, but currently only basic operations are available.  
Column reordering can be done with `select` or `keep`.

```python
session.dataset("ds").sort("x")
session.dataset("ds").sort(["x", "y"])
session.dataset("ds").sort([("x", "Ascending"),("y", "Descending")])

session.dataset("ds").select(["x","y"])
```

---

## Recommended Patterns

- [ ] `%LET` macro variables → Python variables + f-string to build DSL strings
- [ ] `INFILE` → pass Polars / Pandas to `session.loads()`
- [ ] `FORMAT` → handle with if statements and merge
- [ ] Missing value checks → use `missing()` function or `cmiss()|nmiss()`
