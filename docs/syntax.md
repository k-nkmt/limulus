# Syntax Reference

A reference for the Data Step statements supported by limulus.  
For differences from SAS language, see [Differences from SAS language](differences.md).
For column-oriented API and session-level helpers such as `sort`, `sql`, and `include`, see [API Reference](api.md) and [Changelog](changelog.md).

---

## DATA Statement

```sas
data <output1> [<output2> ...];
  ...
run;
```

Specifying multiple output destinations creates a multi-output DATA step.  
Using an explicit `output` statement allows row-level routing.

---

## SET Statement

```sas
set <dataset> [<dataset2> ...] [end=<var>] [indsname=<var>];
```

| Option | Description |
|-----------|------|
| `end=last` | Defines a variable that becomes `1` at the last row |
| `indsname=src` | Stores the current dataset name in a variable |

Listing multiple datasets performs a vertical concatenation.
Shared columns must have matching types; mixed Arrow `string` and `large_string`
columns are promoted to `large_string`.

### Interleaving with BY

```sas
set <dataset1> <dataset2> ...;
by <key>;
```

When `by` is specified together with multiple datasets, rows from all sources are
merge-sorted by the BY key rather than concatenated in source order.
This matches "interleaving" semantics.

```sas
data combined;
  set sales2023 sales2024;
  by date;
run;
```

---

## MERGE Statement

```sas
merge <dataset>(in=<var>) [<dataset2>(in=<var>)];
by <key>;
```

Duplicate non-`BY` column names raise an error instead of being overwritten.
Repeated BY values across multiple inputs produce a warning.
BY keys must have matching types; mixed Arrow `string` and `large_string` keys
are promoted to `large_string`.

---

## WHERE Statement

```sas
where <condition>;
```

Placed after SET/MERGE. Applies a filter at data read time.


## Subsetting IF Statement

```sas
if <condition>;
```

Applies a filter after data is read.

---

## IF / ELSE IF / ELSE Statement

```sas
if <condition> then <statement>;
else if <condition> then <statement>;
else <statement>;
```

Use `DO...END` to group multiple statements:

```sas
if x > 0 then do;
  y = x * 2;
  z = 1;
end;
```

---

## DO / END Statement

```sas
do <var> = <start> to <stop> [by <step>];
  ...
end;
```

Used for counter loops as well as conditional blocks (`if...then do;`).

---

## BY Statement

```sas
by <var> [<var2> ...];
```

Used in combination with SET/MERGE.  
Makes `FIRST.<var>` / `LAST.<var>` automatic variables available.

---

## OUTPUT Statement

```sas
output [<dataset>];
```

Without arguments, writes to the default destination: the first named `OUTPUT`
target, or the first `DATA` target if no named `OUTPUT` exists.
With names, writes to the specified destinations.

Output references accept dataset options such as `KEEP`, `DROP`, and `RENAME`.
References to the same destination share its options.

---

## STOP Statement

```sas
stop ;
```

Stops the Data Step processing.

---

## KEEP / DROP Statement

```sas
keep <var1> [<var2> ...];
drop <var1> [<var2> ...];
```

Specifies variables to include in / exclude from the output dataset.

---

## RENAME Statement

```sas
rename <oldname>=<newname> [<oldname>=<newname> ...];
```

---

## RETAIN Statement

```sas
retain <var> [<initial>] [<var2> [<initial2>] ...];
```

Retains the value of a variable across iterations. If the initial value is omitted, defaults to `null`.

---

## ARRAY Statement

```sas
array <name> <var1> [<var2> ...];
```

Assigns a name to a group of variables. Index access (`name[1]`) is supported.

---

## SUM Statement (Cumulative Addition)

```sas
<var> + <expression>;
```

Performs cumulative addition without `RETAIN`.

---

## ASSIGN Statement (Assignment)

```sas
<varname> = <expression>;
```

### String Concatenation in General Expressions

Two strings can be joined with `+` or `||`, preserving spaces:

```sas
text_value = "a" + "B";    /* "aB" */
digits = "1" + "2";        /* "12" */
joined = "a" || "B";       /* "aB" */
```

These forms apply to Data Step expressions, not the Column API or dataset-option
`WHERE=`.

---

## DELETE Statement

```sas
delete;
```

Does not output the current row (moves to the next iteration of the PDV loop).

---

## Built-in Functions

### String

String positions and lengths use Unicode characters rather than bytes.

| Function | Description | Example |
|------|------|--------|
| `substr(s, pos, len)` | Substring | `substr(name, 1, 3)` |
| `upcase(s)` | Convert to uppercase | `upcase(sex)` |
| `lowcase(s)` | Convert to lowercase | `lowcase(name)` |
| `propcase(s)` | Capitalize first letter | `propcase(city)` |
| `trim(s)` | Remove trailing spaces | `trim(raw_name)` |
| `strip(s)` | Remove leading and trailing spaces | `strip(raw_name)` |
| `length(s)` | String length | `length(name)` |
| `lengthn(s)` | NULL-safe string length | `lengthn(comment)` |
| `reverse(s)` | Reverse string | `reverse(code)` |
| `scan(s, n, dlm)` | n-th token | `scan(path, 2, '/')` |
| `compress(s, chars)` | Remove specified characters | `compress(phone, '-')` |
| `index(s, sub)` | Position of substring | `index(name, 'AL')` |
| `find(s, sub)` | Position of substring | `find(name, 'al', 1, 'i')` |
| `tranwrd(s, from, to)` | Word replacement | `tranwrd(note, 'old', 'new')` |
| `translate(s, to, from)` | Character translation | `translate(code, 'AB', '12')` |
| `countw(s, dlm)` | Word count | `countw(text, ' ')` |
| `cat(a, b, ...)` | Concatenate | `cat(first, last)` |
| `cats(a, b, ...)` | Concatenate with trim | `cats(first, last)` |
| `catt(a, b, ...)` | Concatenate with trailing trim | `catt(first, last)` |
| `catx(dlm, a, b, ...)` | Concatenate with delimiter | `catx('-', y, m, d)` |
| `repeat(s, n)` | Repeat n times | `repeat('*', 3)` |

### Numeric

| Function | Description | Example |
|------|------|--------|
| `abs(x)` | Absolute value | `abs(delta)` |
| `round(x, n)` | Round (0.5 rounds away from zero) | `round(bmi, 0.1)` |
| `ceil(x)` | Ceiling | `ceil(value)` |
| `floor(x)` | Floor | `floor(value)` |
| `int(x)` | Integer part | `int(score)` |
| `mod(x, y)` | Remainder | `mod(id, 2)` |
| `max(a, b, ...)` | Maximum | `max(v1, v2, v3)` |
| `min(a, b, ...)` | Minimum | `min(v1, v2, v3)` |
| `sum(a, b, ...)` | Sum | `sum(v1, v2, v3)` |
| `mean(a, b, ...)` | Mean | `mean(v1, v2, v3)` |
| `sqrt(x)` | Square root | `sqrt(var)` |
| `log(x)` | Natural logarithm | `log(value)` |
| `exp(x)` | Exponent | `exp(value)` |
| `sign(x)` | Sign | `sign(change)` |

### Missing Values / Dates

| Function | Description | Example |
|------|------|--------|
| `missing(x)` | Returns `1` if missing | `missing(result)` |
| `nmiss(a, b, ...)` | Count of missing values | `nmiss(v1, v2, v3)` |
| `cmiss(a, b, ...)` | Count of missing values (mixed types) | `cmiss(name, age, score)` |
| `lag(x, n)` | Previous row's value | `lag(amount)` |
| `mdy(m, d, y)` | Month/day/year to date | `mdy(1, 15, 2025)` |
| `year(d)` | Year from date | `year(mdy(1, 15, 2025))` |
| `intck(unit, from, to)` | Date difference | `intck('day', mdy(1,1,2025), mdy(1,10,2025))` |

### Format Helpers

| Function | Description | Example |
|------|------|--------|
| `put(value, format_name)` | Render a value as text with a built-in or registered format | `put(amount, 8.2.)` |
| `input(value, informat_name)` | Parse text into a numeric, date, datetime, or time value | `input(date_text, yymmdd10.)` |
| `hour(value)` | Convert a time-like value to decimal hours | `hour(clock_text)` |

Built-in `put(...)` formats currently support:

- numeric to text: `best.`, `w.`, `w.d`, `w.d.`, `zw.`, `zw.d`, `zw.d.`, `commaw.`, `commaw.d`, `commaw.d.`
- date / datetime / time to text: `e8601da.`, `e8601dt.`, `yymmdd6.`, `yymmdd8.`, `yymmdd10.`, `time.`

Built-in `input(...)` informats currently support:

- text to numeric: `best.`
- text to date / datetime / time: `e8601da.`, `e8601dt.`, `yymmdd6.`, `yymmdd8.`, `yymmdd10.`, `time.`

Session-level custom catalogs can also be registered for exact-match lookups:

- `Session.register_format(name, mapping)` registers a numeric `put(...)` catalog.
- `Session.register_format(name, mapping, namespace="character")` registers a character `put(...)` catalog referenced as `$name.`.
- `Session.register_informat(name, mapping)` registers a float64-only `input(...)` catalog.

Current dict-catalog semantics are intentionally narrow:

- `put(value, myfmt.)` returns text and falls back to the original value when no dict entry matches.
- `put(value, $myfmt.)` uses the same base name with the character-format namespace.
- `input(value, myinf.)` returns `null` when no dict entry matches.
- dict custom `input(...)` is float64-only in this phase; typed custom informats remain deferred.
- callable format/informat registrations are unsupported; custom catalogs must be serializable mappings.


### Regular Expressions

| Function | Description | Example |
|------|------|--------|
| `prxmatch(pattern, s)` | Regex match position | `prxmatch('/abc/i', text)` |
| `prxchange(pattern, times, s)` | Regex substitution | `prxchange('s/ +/ /', -1, text)` |

### ARRAY Helpers

| Function | Description | Example |
|------|------|--------|
| `dim(array)` | Number of elements in array | `dim(vars)` |
| `vname(array[i])` | Variable name of array element | `vname(vars[2])` |


### Custom Functions

| Function | Description | Example |
|------|------|--------|
| `lead(x, n)` | Next row's value | `lead(x)` |
| `shift(x, n)` | negative n behaves like `lag`, positive n like `lead` | `shift(x, -1)`<br>`shift(x, 1)` |
| `apply` | Apply a Python callable by name. | `apply('double',amount)`<br>`apply('len',note)`<br>`apply('math.sqrt',value)` |

> **Note:** `apply()` supports builtins, dotted functions such as `math.sqrt`, and
> your own functions registered with `functions={"name": callable}`.
