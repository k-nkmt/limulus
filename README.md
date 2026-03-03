# limulus — Data Step for Your Workspace

[日本語](docs/README_ja.md)

---

![limulus](docs/_static/limulus.png)

**limulus** is a Python library for data processing using Data Step syntax.  
Its goal is to bring the simplicity and long-term stability of Data Step into Python workflows.

This is currently an alpha release. Please note that breaking changes to the API and other interfaces may occur before the beta release.

---

## Installation

```bash
pip install limulus
```

---

## Usage

### 1. Prepare Your Data

Load a DataFrame (Arrow / Polars / Pandas) from a CSV or other source.

```python
import pandas as pd
import limulus

health_df = pd.DataFrame({
	"name": ["Alice", "Bob", "Charlie", "David"],
	"age": [25, 30, 35, 40],
	"height": [65, 70, 68, 72],  # inches
	"weight": [140, 180, 130, 200]  # pounds
})

# Load data into a Session
session = limulus.Session()
session.loads({"health": health_df})
```

### 2. Run a Data Step

```python
session.submit("""
data result;
  set health;
  where age > 25;
  height_m = height * 0.0254;
  weight_kg = weight * 0.454;
  bmi = round(weight_kg / (height_m**2), 0.1);
  keep name age bmi;
run;
""")
```

Multiple Data Steps can be submitted at once. Datasets created by earlier steps can be referenced by subsequent steps.

### 3. Retrieve Results

The simplest way is to retrieve the result from the session as an Arrow table using subscript notation.  
You can convert it to pandas using Arrow's methods.

```python
df_out = session["result"].to_pandas()
print(df_out)
```
---


## Documentation

https://k-nkmt.github.io/limulus/

## License

[PolyForm Noncommercial License 1.0.0](LICENSE)

This project is distributed under the PolyForm Noncommercial License.
Creative Commons licenses are not used because they are generally not recommended for software distribution.

The software may be used for personal, educational, academic, and noncommercial research purposes.
Commercial use is not permitted under the current license terms.

I may consider adopting a different licensing model in the future as the project evolves.

contact: info@knworx.com

---

## Notices

**Project Positioning**  
limulus is a modern data-step–inspired data transformation framework implemented independently in Python and Rust. It is not affiliated with or endorsed by SAS Institute Inc.

**Trademark Notice**  
SAS® is a registered trademark of SAS Institute Inc. All other trademarks are the property of their respective owners.

**Independence Statement**  
This is an independent implementation. No SAS source code or proprietary materials have been used.

**Compatibility Disclaimer**  
Compatibility with SAS software is not guaranteed and is not a project goal. Certain behaviors intentionally differ to provide modern semantics.