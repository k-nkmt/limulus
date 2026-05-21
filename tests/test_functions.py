import math
import unittest
import datetime as dt

import pyarrow as pa
import pytest

import limulus
from limulus import Session
from limulus.models import DataSetRef, ExecuteRequest
from limulus.runtime import DataStepExecutor

FUNCTION_SCENARIOS = {

    "where_string_functions": {
        "overview": "WHERE string functions and operators filter expected rows under python/rust runtimes",
        "upcase_inputs": {"name": ["Alice", "Bob", "Catherine"], "sex": ["F", "M", "F"]},
        "upcase_dsl": """
        data out;
        set inp;
        where upcase(sex) ^= "M";
        run;
        """,
        "upcase_expected": [{"name": "Alice", "sex": "F"}, {"name": "Catherine", "sex": "F"}],
        "lowcase_inputs": {"name": ["ALICE", "Bob"]},
        "lowcase_dsl": """
        data out;
        set inp;
        where lowcase(name) = "alice";
        run;
        """,
        "lowcase_expected": [{"name": "ALICE"}],
        "length_inputs": {"name": ["Al", "Robert", "Ann"]},
        "length_dsl": """
        data out;
        set inp;
        where length(name) > 5;
        run;
        """,
        "length_expected": [{"name": "Robert"}],
    },

    "where_builtin_funcs": {
        "overview": "Native runtime evaluates abs/mod/missing in WHERE expressions",
        "dsl": (
            "data out; set in; "
            "where abs(delta) >= 2 and mod(id, 2) = 0 and missing(note); "
            "output out; run;"
        ),
        "inputs": [
            {"id": 1, "delta": -3, "note": None},
            {"id": 2, "delta": -2, "note": None},
            {"id": 4, "delta": -1, "note": None},
        ],
        "expected_output": [{"id": 2, "delta": -2, "note": None}],
    },
    "where_string_native_funcs": {
        "overview": "Native runtime evaluates substr/trim/upcase/scan/compress in WHERE expressions",
        "dsl": (
            "data out; set in; "
            "where substr(trim(name), 1, 1) = 'a' and upcase(code) = 'X1' "
            "and scan(path, 2, '/') = 'usr' and compress('a b c') = 'abc'; "
            "output out; run;"
        ),
        "inputs": [
            {"id": 1, "name": " alice ", "code": "x1", "path": "root/usr/bin"},
            {"id": 2, "name": "bob", "code": "x2", "path": "root/opt/bin"},
        ],
        "expected_output": [{"id": 1, "name": " alice ", "code": "x1", "path": "root/usr/bin"}],
    },
    "where_date_funcs": {
        "overview": "Native runtime evaluates year/mdy/intck in WHERE expressions",
        "dsl": (
            "data out; set in; "
            "where year(mdy(month, day, year_num)) = 2024 "
            "and intck('day', mdy(1, 1, 2024), mdy(1, 3, 2024)) = 2; "
            "output out; run;"
        ),
        "inputs": [
            {"id": 1, "month": 2, "day": 10, "year_num": 2024},
            {"id": 2, "month": 2, "day": 10, "year_num": 2023},
        ],
        "expected_output": [{"id": 1, "month": 2, "day": 10, "year_num": 2024}],
    },
    "where_lag": {
        "overview": "Native runtime supports lag() in WHERE expressions",
        "dsl": "data out; set in; where lag(id) = 1; output out; run;",
        "inputs": [{"id": 1}, {"id": 2}, {"id": 3}],
        "expected_output": [{"id": 2}],
    },

    "round_tie_values": {
        "overview": "round() uses half-away-from-zero rounding for tie values on both backends",
        "dsl": (
            "data out; set in; "
            "pos = round(2.5); neg = round(-2.5); "
            "pos_unit = round(2.25, 0.1); neg_unit = round(-2.25, 0.1); "
            "output out; keep pos neg pos_unit neg_unit; run;"
        ),
        "inputs": [{"id": 1}],
        "expected_output": [{"pos": 3.0, "neg": -3.0, "pos_unit": 2.3, "neg_unit": -2.3}],
    },
    "round_bmi_decimal_stability": {
        "overview": "round(..., 0.1) keeps stable decimal output for BMI-like calculations on both backends",
        "dsl": (
            "data out; set in; "
            "height_m = height * 0.0254; weight_kg = weight * 0.454; "
            "bmi = round(weight_kg / (height_m**2), 0.1); "
            "keep name bmi; run;"
        ),
        "inputs": [{"name": "Bob", "height": 70.0, "weight": 180.0}],
        "expected_output": [{"name": "Bob", "bmi": 25.9}],
    },

    "string_functions": {
        "overview": "String functions produce correct scalar results for Rust and Python backends",
        "inputs": {"name": ["hello world", "foo bar"], "x": [10, 20]},
        "propcase": {
            "dsl": "data out; set inp; y = propcase(name); run;",
            "expected": ["Hello World", "Foo Bar"],
        },
        "cat": {
            "dsl": "data out; set inp; y = cat(name, ' test'); run;",
            "expected": ["hello world test", "foo bar test"],
        },
        "cats": {
            "dsl": "data out; set inp; y = cats(' a ', ' b '); run;",
            "expected": ["ab", "ab"],
        },
        "catt": {
            "dsl": "data out; set inp; y = catt('hello ', ' world'); run;",
            "expected": ["hello world", "hello world"],
        },
        "catx": {
            "dsl": "data out; set inp; y = catx('-', 'a', 'b', 'c'); run;",
            "expected": ["a-b-c", "a-b-c"],
        },
        "index": {
            "dsl": "data out; set inp; y = index(name, 'world'); run;",
            "expected": [7.0, 0.0],
        },
        "find": {
            "dsl": "data out; set inp; y = find(name, 'WORLD', 1, 'i'); run;",
            "expected": [7.0, 0.0],
        },
        "tranwrd": {
            "dsl": "data out; set inp; y = tranwrd(name, 'hello', 'hi'); run;",
            "expected": ["hi world", "foo bar"],
        },
        "translate": {
            "dsl": "data out; set inp; y = translate(name, 'HW', 'hw'); run;",
            "expected": ["Hello World", "foo bar"],
        },
        "length": {
            "dsl": "data out; set inp; y = length(name); run;",
            "expected": [11.0, 7.0],
        },
        "lengthn": {
            "dsl": "data out; set inp; y = lengthn(name); run;",
            "expected": [11.0, 7.0],
        },
        "strip": {
            "dsl": "data out; set inp; y = strip(name); run;",
            "expected": ["hello world", "foo bar"],
        },
        "reverse": {
            "dsl": "data out; set inp; y = reverse(name); run;",
            "expected": ["dlrow olleh", "rab oof"],
        },
        "repeat": {
            "dsl": "data out; set inp; y = repeat('ab', 3); run;",
            "expected": ["ababab", "ababab"],
        },
        "countw": {
            "dsl": "data out; set inp; y = countw(name); run;",
            "expected": [2.0, 2.0],
        },
    },
    "numeric_functions": {
        "overview": "Numeric functions produce correct scalar results for Rust and Python backends",
        "inputs": {"name": ["hello world", "foo bar"], "x": [10, 20]},
        "int": {
            "dsl": "data out; set inp; y = int(3.7); run;",
            "expected": [3.0, 3.0],
        },
        "sum": {
            "dsl": "data out; set inp; y = sum(1, 2, 3); run;",
            "expected": [6.0, 6.0],
        },
        "mean": {
            "dsl": "data out; set inp; y = mean(2, 4, 6); run;",
            "expected": [4.0, 4.0],
        },
        "sqrt": {
            "dsl": "data out; set inp; y = sqrt(16); run;",
            "expected": [4.0, 4.0],
        },
        "log": {
            "dsl": "data out; set inp; y = log(1); run;",
            "expected": [0.0, 0.0],
        },
        "exp": {
            "dsl": "data out; set inp; y = exp(0); run;",
            "expected": [1.0, 1.0],
        },
        "sign": {
            "dsl": "data out; set inp; y = sign(-5); run;",
            "expected": [-1.0, -1.0],
        },
    },
    "missing_functions": {
        "overview": "cmiss counts missing values correctly on both backends",
        "inputs": {"name": ["hello world", "foo bar"], "x": [10, 20]},
        "cmiss": {
            "dsl": "data out; set inp; y = cmiss(name, x); run;",
            "expected": [0.0, 0.0],
        },
    },
    "regex_functions": {
        "overview": "prxmatch and prxchange support Perl regular expressions on Rust backend",
        "inputs": {"name": ["hello world", "foo bar"], "x": [10, 20]},
        "prxmatch": {
            "dsl": "data out; set inp; y = prxmatch('/world/', name); run;",
            "expected": [7.0, 0.0],
        },
        "prxmatch_ci": {
            "dsl": "data out; set inp; y = prxmatch('/WORLD/i', name); run;",
            "expected": [7.0, 0.0],
        },
        "prxchange": {
            "dsl": "data out; set inp; y = prxchange('s/world/earth/', 1, name); run;",
            "expected": ["hello earth", "foo bar"],
        },
    },
    "function_parity": {
        "overview": "All newly added functions produce identical results on both Python and Rust backends",
        "inputs": {"name": ["hello world", "foo bar"], "x": [10, 20]},
        "expressions": [
            "propcase(name)",
            "cat(name, ' test')",
            "cats(' a ', ' b ')",
            "catt('hello ', ' world')",
            "catx('-', 'a', 'b', 'c')",
            "index(name, 'world')",
            "find(name, 'o', 1)",
            "tranwrd(name, 'foo', 'baz')",
            "length(name)",
            "lengthn(name)",
            "strip(name)",
            "reverse(name)",
            "repeat('x', 3)",
            "countw(name)",
            "int(3.9)",
            "sum(1, 2, 3)",
            "mean(10, 20)",
            "sqrt(25)",
            "log(1)",
            "exp(0)",
            "sign(-3)",
            "cmiss(name, x)",
            "prxmatch('/foo/', name)",
            "prxchange('s/bar/baz/', 1, name)",
        ],
    },
    "submit_backend_override": {
        "overview": (
            "session.submit(backend=...) and limulus.submit(..., backend=...) "
            "temporarily override the runtime backend for a single call"
        ),
        "inputs": {"name": ["hello world", "foo bar"], "x": [10, 20]},
        "dsl": "data out; set inp; where x >= 15; output out; run;",
        "expected_output": [{"name": "foo bar", "x": 20}],
    },
}


def _arrow_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist([dict(row) for row in rows])


def _output_rows(response, target: str) -> list[dict]:
    dataset = response.outputs[target]
    if dataset.kind == "arrow_table" and hasattr(dataset.payload, "to_pylist"):
        return dataset.payload.to_pylist()
    return dataset.payload


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_where_upcase_not_equals_filters_expected_rows(backend: str) -> None:
    scenario = FUNCTION_SCENARIOS["where_string_functions"]
    session = Session(runtime_backend=backend, parser_backend="python")
    session.load("inp", pa.table(scenario["upcase_inputs"]))

    result = session.submit(scenario["upcase_dsl"])

    assert result.success is True
    assert session["out"].to_pylist() == scenario["upcase_expected"]


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_where_lowcase_equals_matches_single_row(backend: str) -> None:
    scenario = FUNCTION_SCENARIOS["where_string_functions"]
    session = Session(runtime_backend=backend, parser_backend="python")
    session.load("inp", pa.table(scenario["lowcase_inputs"]))

    result = session.submit(scenario["lowcase_dsl"])

    assert result.success is True
    assert session["out"].to_pylist() == scenario["lowcase_expected"]


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_where_length_greater_than_filters_long_names(backend: str) -> None:
    scenario = FUNCTION_SCENARIOS["where_string_functions"]
    session = Session(runtime_backend=backend, parser_backend="python")
    session.load("inp", pa.table(scenario["length_inputs"]))

    result = session.submit(scenario["length_dsl"])

    assert result.success is True
    assert session["out"].to_pylist() == scenario["length_expected"]


class TestBuiltinFunctionsInWhere(unittest.TestCase):
    def test_abs_mod_missing_in_where_expression(self) -> None:
        scenario = FUNCTION_SCENARIOS["where_builtin_funcs"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, "rust")

    def test_substr_trim_upcase_scan_compress_in_where_expression(self) -> None:
        scenario = FUNCTION_SCENARIOS["where_string_native_funcs"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, "rust")

    def test_year_mdy_intck_in_where_expression(self) -> None:
        scenario = FUNCTION_SCENARIOS["where_date_funcs"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, "rust")

    def test_lag_in_where_expression(self) -> None:
        scenario = FUNCTION_SCENARIOS["where_lag"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, "rust")


class TestRoundTieValues(unittest.TestCase):
    def _run(self, backend: str) -> dict:
        scenario = FUNCTION_SCENARIOS["round_tie_values"]
        executor = DataStepExecutor(runtime_backend=backend)
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )
        response = executor.execute(request)
        self.assertFalse(response.has_errors)
        rows = _output_rows(response, "out")
        self.assertEqual(len(rows), 1)
        return rows[0]

    def _assert_row(self, row: dict) -> None:
        scenario = FUNCTION_SCENARIOS["round_tie_values"]
        expected = scenario["expected_output"][0]
        self.assertAlmostEqual(row["pos"], expected["pos"], places=9)
        self.assertAlmostEqual(row["neg"], expected["neg"], places=9)
        self.assertAlmostEqual(row["pos_unit"], expected["pos_unit"], places=9)
        self.assertAlmostEqual(row["neg_unit"], expected["neg_unit"], places=9)

    def test_python_runtime_round_uses_rounding_for_tie_values(self) -> None:
        row = self._run("python")
        self._assert_row(row)

    def test_native_runtime_round_uses_rounding_for_tie_values(self) -> None:
        row = self._run("rust")
        self._assert_row(row)

    def test_round_bmi_decimal_stability_python_runtime(self) -> None:
        scenario = FUNCTION_SCENARIOS["round_bmi_decimal_stability"]
        executor = DataStepExecutor(runtime_backend="python")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])

    def test_round_bmi_decimal_stability_native_runtime(self) -> None:
        scenario = FUNCTION_SCENARIOS["round_bmi_decimal_stability"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])

class TestStringFunctions:
    """Verify newly added string functions produce correct values on Rust backend."""

    @pytest.mark.parametrize("func_key", [
        "propcase", "cat", "cats", "catt", "catx",
        "index", "find", "tranwrd", "translate",
        "length", "lengthn", "strip", "reverse", "repeat", "countw",
    ])
    def test_string_function_rust(self, func_key: str) -> None:
        scenario = FUNCTION_SCENARIOS["string_functions"]
        s = Session(runtime_backend="rust", parser_backend="python")
        s.load("inp", pa.table(scenario["inputs"]))
        case = scenario[func_key]

        result = s.submit(case["dsl"])

        assert result.success is True
        assert list(s.to_pandas("out")["y"]) == case["expected"]


class TestNumericFunctions:
    """Verify newly added numeric functions produce correct values on Rust backend."""

    @pytest.mark.parametrize("func_key", ["int", "sum", "mean", "sqrt", "log", "exp", "sign"])
    def test_numeric_function_rust(self, func_key: str) -> None:
        scenario = FUNCTION_SCENARIOS["numeric_functions"]
        s = Session(runtime_backend="rust", parser_backend="python")
        s.load("inp", pa.table(scenario["inputs"]))
        case = scenario[func_key]

        result = s.submit(case["dsl"])

        assert result.success is True
        assert list(s.to_pandas("out")["y"]) == case["expected"]


class TestMissingValueFunction:
    """Verify cmiss counts missing values on Rust backend."""

    def test_cmiss_no_missing_values(self) -> None:
        scenario = FUNCTION_SCENARIOS["missing_functions"]
        s = Session(runtime_backend="rust", parser_backend="python")
        s.load("inp", pa.table(scenario["inputs"]))
        case = scenario["cmiss"]

        result = s.submit(case["dsl"])

        assert result.success is True
        assert list(s.to_pandas("out")["y"]) == case["expected"]

class TestRegexFunctions:
    """Verify prxmatch / prxchange support Perl-compatible patterns on Rust backend."""

    @pytest.mark.parametrize("func_key", ["prxmatch", "prxmatch_ci", "prxchange"])
    def test_regex_function_rust(self, func_key: str) -> None:
        scenario = FUNCTION_SCENARIOS["regex_functions"]
        s = Session(runtime_backend="rust", parser_backend="python")
        s.load("inp", pa.table(scenario["inputs"]))
        case = scenario[func_key]

        result = s.submit(case["dsl"])

        assert result.success is True
        assert list(s.to_pandas("out")["y"]) == case["expected"]


class TestFormatFunctionsRust:
    def test_put_input_and_hour_are_supported_on_rust_backend(self) -> None:
        session = Session(runtime_backend="rust", parser_backend="python")
        session.load(
            "inp",
            pa.table(
                {
                    "id": [7],
                    "amount": [12345.6],
                    "best_text": ["12345.6"],
                    "date_text": ["2024-02-03"],
                    "timestamp_text": ["2024-02-03T16:24:43"],
                    "clock_text": ["11:30"],
                }
            ),
        )

        result = session.submit(
            """
            data out;
            set inp;
            code = put(id, z5.);
            fixed_text = put(amount, 8.1.);
            rounded_text = put(amount, 8.);
            comma_text = put(amount, comma8.1.);
            zero_scaled = put(amount, z8.1.);
            best_rendered = put(amount, best.);
            best_value = input(best_text, best.);
            visit_date = input(date_text, yymmdd10.);
            visit_iso = put(visit_date, e8601da.);
            timestamp_value = input(timestamp_text, e8601dt.);
            timestamp_iso = put(timestamp_value, e8601dt.);
            clock_value = input(clock_text, time.);
            clock_iso = put(clock_value, time.);
            clock_hour = hour(clock_text);
            run;
            """
        )

        assert result.success is True
        assert session._executor.last_runtime_backend == "rust"
        assert session["out"].to_pylist() == [
            {
                "id": 7,
                "amount": 12345.6,
                "best_text": "12345.6",
                "date_text": "2024-02-03",
                "timestamp_text": "2024-02-03T16:24:43",
                "clock_text": "11:30",
                "code": "00007",
                "fixed_text": "12345.6",
                "rounded_text": "12346",
                "comma_text": "12,345.6",
                "zero_scaled": "012345.6",
                "best_rendered": "12345.6",
                "best_value": 12345.6,
                "visit_date": dt.date(2024, 2, 3),
                "visit_iso": "2024-02-03",
                "timestamp_value": dt.datetime(2024, 2, 3, 16, 24, 43),
                "timestamp_iso": "2024-02-03T16:24:43",
                "clock_value": dt.time(11, 30),
                "clock_iso": "11:30:00",
                "clock_hour": 11.5,
            }
        ]


class TestFunctionParity:
    """All newly added functions produce identical results on Python and Rust backends."""

    @pytest.fixture
    def rust_session(self) -> Session:
        s = Session(backend="rust")
        s.load("inp", pa.table(FUNCTION_SCENARIOS["function_parity"]["inputs"]))
        return s

    @pytest.fixture
    def python_session(self) -> Session:
        s = Session(backend="python")
        s.load("inp", pa.table(FUNCTION_SCENARIOS["function_parity"]["inputs"]))
        return s

    @pytest.mark.parametrize("expr", FUNCTION_SCENARIOS["function_parity"]["expressions"])
    def test_parity(self, rust_session: Session, python_session: Session, expr: str) -> None:
        code = f"data out; set inp; y = {expr}; run;"
        python_session.submit(code)
        py_values = list(python_session.to_pandas("out")["y"])
        rust_session.submit(code)
        rust_values = list(rust_session.to_pandas("out")["y"])

        for py_val, rust_val in zip(py_values, rust_values):
            if isinstance(py_val, float) and isinstance(rust_val, float):
                if math.isnan(py_val) and math.isnan(rust_val):
                    continue
                assert abs(py_val - rust_val) < 1e-10, (
                    f"Mismatch for {expr}: py={py_val}, rust={rust_val}"
                )
            else:
                assert py_val == rust_val, f"Mismatch for {expr}: py={py_val}, rust={rust_val}"


class TestSubmitBackendOverride:
    """session.submit(backend=...) and limulus.submit(..., backend=...) override the runtime."""

    def test_session_submit_backend_kwarg_uses_specified_backend(self) -> None:
        scenario = FUNCTION_SCENARIOS["submit_backend_override"]
        session = Session(backend="python")
        session.load("inp", pa.table(scenario["inputs"]))

        result = session.submit(scenario["dsl"], backend="rust")

        assert result.success is True
        assert session["out"].to_pylist() == scenario["expected_output"]
        assert session._executor.last_runtime_backend == "rust"

    def test_session_submit_backend_kwarg_restores_session_backend_after_call(self) -> None:
        scenario = FUNCTION_SCENARIOS["submit_backend_override"]
        session = Session(backend="python")
        session.load("inp", pa.table(scenario["inputs"]))

        session.submit(scenario["dsl"], backend="rust")

        assert session._executor._runtime_backend_preference == "python"

    def test_limulus_submit_backend_kwarg_uses_specified_backend(self) -> None:
        scenario = FUNCTION_SCENARIOS["submit_backend_override"]

        result = limulus.submit(
            scenario["dsl"],
            backend="rust",
            inp=pa.table(scenario["inputs"]),
        )

        assert result.success is True
        assert result.datasets["out"].to_pylist() == scenario["expected_output"]
