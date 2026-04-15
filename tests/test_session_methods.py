import pyarrow as pa
import pytest

from limulus import Session
from limulus.models import ExecuteResponse


SESSION_METHOD_SCENARIOS = {
    "column_api_select_rename_filter": {
        "overview": "filter/select/rename chain keeps positive rows and renames amount to amt",
        "inputs": {
            "inp": {
                "id": [1, 2, 3],
                "amount": [10, -1, 30],
                "name": ["Alice", "Bob", "Catherine"],
            }
        },
        "expected_output": [{"id": 1, "amt": 10}, {"id": 3, "amt": 30}],
    },
    "sort_nodupkey": {
        "overview": "sort with nodupkey preserves the first row in sorted order for duplicate keys",
        "inputs": {
            "src": {
                "group_id": [2, 1, 1, 2],
                "amount": [40, 20, 20, 30],
                "label": ["d", "b", "b", "c"],
            }
        },
        "expected_output": [
            {"group_id": 1, "amount": 20, "label": "b"},
            {"group_id": 2, "amount": 30, "label": "c"},
            {"group_id": 2, "amount": 40, "label": "d"},
        ],
    },
    "apply_dataset_options": {
        "overview": "apply_dataset_options enforces where/keep/drop/rename in one call",
        "inputs": {
            "src": {
                "id": [1, 2, 3],
                "amount": [10, -1, 30],
                "tmp": ["a", "b", "c"],
            }
        },
        "expected_output": [{"id": 1, "amt": 10}, {"id": 3, "amt": 30}],
    },
    "loads_unload_delete_alias": {
        "overview": "loads registers multiple datasets and unload/delete remove them from catalog",
        "inputs": {"a": {"id": [1]}, "b": {"id": [2]}, "c": {"id": [3]}},
    },
    "unload_sequence_names": {
        "overview": "unload accepts multiple dataset names and removes only the requested entries",
        "inputs": {"a": {"id": [1]}, "b": {"id": [2]}, "c": {"id": [3]}},
        "expected_remaining": ["c"],
    },
    "step_style_methods": {
        "overview": "where/keep/drop/rename step-style APIs return the same filtered rows as the column API",
        "inputs": {
            "inp": {
                "id": [1, 2, 3],
                "amount": [10, -1, 30],
                "name": ["Alice", "Bob", "Catherine"],
            }
        },
        "expected_output": [{"id": 1, "amt": 10}, {"id": 3, "amt": 30}],
    },
    "submit_option_propagation": {
        "overview": "session options are stored locally and forwarded on submit requests",
        "options": {"execution.trace": True, "execution.mode": "debug"},
        "expected_missing_default": "fallback",
    },
    "include_submits_file_contents": {
        "overview": "include reads a DSL file and materializes the expected filtered output",
        "inputs": {"inp": {"id": [1, 2], "amount": [5, 20]}},
        "dsl": "data out; set inp; where amount > 10; output out; run;",
        "expected_output": [{"id": 2, "amount": 20}],
    },
    "cast_alias": {
        "overview": "cast keeps the Session alias and writes the converted Arrow type via out=",
        "inputs": {"src": {"id": [1, 2], "amount": [1.25, 2.5]}},
        "expected_type": pa.float32(),
    },
    "sql_query_and_store": {
        "overview": "sql returns Arrow rows for a query and stores CREATE TABLE results in the session catalog",
        "inputs": {"src": {"id": [1, 2, 3], "amount": [5, 20, 30]}},
        "expected_query": [{"id": 2, "amount": 20}, {"id": 3, "amount": 30}],
        "expected_stored": [{"id": 2}, {"id": 3}],
    },
    "sql_metadata_roundtrip": {
        "overview": "sql restores table and field metadata after the internal Polars roundtrip",
        "source_rows": {"id": [1, 2], "amount": [10, 20]},
        "source_schema": pa.schema(
            [
                pa.field("id", pa.int64(), metadata={b"label": b"Identifier"}),
                pa.field("amount", pa.int64(), metadata={b"label": b"Amount"}),
            ],
            metadata={b"memlabel": b"Source Label"},
        ),
        "expected_memlabel": b"Source Label",
        "expected_id_label": b"Identifier",
        "expected_amount_label": b"Amount",
    },
    "transpose_default": {
        "overview": "transpose without id keeps PROC TRANSPOSE-like _NAME_/COLn output",
        "inputs": {"src": {"id": [1, 2], "x": [10, 20], "y": [100, 200]}},
        "expected_output": [
            {"_NAME_": "x", "COL1": 10, "COL2": 20},
            {"_NAME_": "y", "COL1": 100, "COL2": 200},
        ],
    },
    "transpose_by": {
        "overview": "transpose with by groups observations into COLn rows per variable",
        "inputs": {
            "src": {
                "grp": ["a", "a", "b"],
                "x": [1, 2, 3],
                "y": [10, 20, 30],
            }
        },
        "expected_output": [
            {"grp": "a", "_NAME_": "x", "COL1": 1, "COL2": 2},
            {"grp": "a", "_NAME_": "y", "COL1": 10, "COL2": 20},
            {"grp": "b", "_NAME_": "x", "COL1": 3, "COL2": None},
            {"grp": "b", "_NAME_": "y", "COL1": 30, "COL2": None},
        ],
    },
    "transpose_id": {
        "overview": "transpose with id pivots one value column into wide output columns",
        "inputs": {
            "src": {
                "grp": ["a", "a", "b"],
                "visit": ["v1", "v2", "v1"],
                "score": [10, 20, 30],
            }
        },
        "expected_output": [
            {"grp": "a", "v1": 10, "v2": 20},
            {"grp": "b", "v1": 30, "v2": None},
        ],
    },
    "transpose_case_insensitive": {
        "overview": "transpose resolves by/id/var columns case-insensitively",
        "inputs": {
            "src": {
                "Grp": ["a", "a", "b"],
                "Visit": ["v1", "v2", "v1"],
                "Score": [10, 20, 30],
            }
        },
        "expected_output": [
            {"Grp": "a", "v1": 10, "v2": 20},
            {"Grp": "b", "v1": 30, "v2": None},
        ],
    },
    "transpose_duplicate_id": {
        "overview": "transpose raises when duplicate id values appear within a by group",
        "inputs": {
            "src": {
                "grp": ["a", "a"],
                "visit": ["v1", "v1"],
                "score": [10, 20],
            }
        },
        "error_fragment": "Duplicate ID value",
    },
    "assign_basic": {
        "overview": "assign supports literals, expressions, functions, case when, and left-to-right references",
        "inputs": {
            "src": {
                "name": ["Alice", "Bob", "Cara"],
                "weight": [50.0, 80.0, 45.0],
                "height_m": [1.60, 1.80, 1.50],
            }
        },
        "expected_output": [
            {
                "name": "Alice",
                "weight": 50.0,
                "height_m": 1.60,
                "cohort": "A",
                "name_up": "ALICE",
                "bmi": 19.53,
                "bmi_flag": "normal",
                "summary": "ALICE:A",
            },
            {
                "name": "Bob",
                "weight": 80.0,
                "height_m": 1.80,
                "cohort": "A",
                "name_up": "BOB",
                "bmi": 24.69,
                "bmi_flag": "normal",
                "summary": "BOB:A",
            },
            {
                "name": "Cara",
                "weight": 45.0,
                "height_m": 1.50,
                "cohort": "A",
                "name_up": "CARA",
                "bmi": 20.0,
                "bmi_flag": "normal",
                "summary": "CARA:A",
            },
        ],
    },
    "assign_unsupported_function": {
        "overview": "assign fails with a clear error for unsupported functions",
        "inputs": {"src": {"name": ["Alice"]}},
        "error_fragment": "Unsupported function",
    },
    "assign_case_insensitive": {
        "overview": "assign expressions can reference source columns without matching the exact original case",
        "inputs": {
            "src": {
                "Name": ["Alice", "Bob"],
                "Weight": [50.0, 80.0],
                "Height_M": [1.60, 1.80],
            }
        },
        "expected_output": [
            {
                "Name": "Alice",
                "Weight": 50.0,
                "Height_M": 1.60,
                "name_up": "ALICE",
                "bmi": 19.53,
            },
            {
                "Name": "Bob",
                "Weight": 80.0,
                "Height_M": 1.80,
                "name_up": "BOB",
                "bmi": 24.69,
            },
        ],
    },
    "assign_case_when_keywords_in_strings": {
        "overview": "assign case when parsing does not treat keyword text inside strings as clause boundaries",
        "inputs": {
            "src": {
                "note": ["contains then", "plain"],
            }
        },
        "expected_output": [
            {"note": "contains then", "flag": "contains then:end"},
            {"note": "plain", "flag": "else"},
        ],
    },
}


def test_session_column_api_select_rename_filter() -> None:
    scenario = SESSION_METHOD_SCENARIOS["column_api_select_rename_filter"]
    session = Session()
    session.load("inp", pa.table(scenario["inputs"]["inp"]))

    session.filter("inp", "amount > 0", out="flt")
    session.select("flt", ["id", "amount"], out="sel")
    session.rename("sel", {"amount": "amt"}, out="out")

    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_sort_nodupkey_keeps_first_row_after_sort() -> None:
    scenario = SESSION_METHOD_SCENARIOS["sort_nodupkey"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    session.sort(
        "src",
        [("group_id", "ascending"), ("amount", "ascending"), ("label", "ascending")],
        out="out",
        nodupkey=True,
    )

    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_column_api_apply_dataset_options() -> None:
    scenario = SESSION_METHOD_SCENARIOS["apply_dataset_options"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    session.apply_dataset_options(
        "src",
        where="amount > 0",
        keep=["id", "amount", "tmp"],
        drop=["tmp"],
        rename={"amount": "amt"},
        out="out",
    )

    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_loads_and_unload_delete_alias() -> None:
    scenario = SESSION_METHOD_SCENARIOS["loads_unload_delete_alias"]
    session = Session()
    session.loads({name: pa.table(data) for name, data in scenario["inputs"].items() if name != "c"})
    session.loads(c=pa.table(scenario["inputs"]["c"]))

    assert "a" in session.datasets
    assert "b" in session.datasets
    assert "c" in session.datasets

    assert session.unload("a") is True
    assert "a" not in session.datasets

    assert session.delete("b") is True
    assert "b" not in session.datasets


def test_session_unload_accepts_sequence_names() -> None:
    scenario = SESSION_METHOD_SCENARIOS["unload_sequence_names"]
    session = Session()
    session.loads({name: pa.table(data) for name, data in scenario["inputs"].items()})

    assert session.unload(["a", "b"]) is True
    assert "a" not in session.datasets
    assert "b" not in session.datasets
    assert sorted(session.datasets.keys()) == scenario["expected_remaining"]


def test_session_step_style_methods() -> None:
    scenario = SESSION_METHOD_SCENARIOS["step_style_methods"]
    session = Session()
    session.load("inp", pa.table(scenario["inputs"]["inp"]))

    session.where("inp", "amount > 0", out="flt")
    session.keep("flt", ["id", "amount", "name"], out="k")
    session.drop("k", ["name"], out="d")
    session.rename("d", {"amount": "amt"}, out="out")

    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_set_option_get_option_and_submit_propagation() -> None:
    scenario = SESSION_METHOD_SCENARIOS["submit_option_propagation"]
    session = Session()
    captured: dict[str, object] = {}

    def fake_execute(request):
        captured["options"] = request.options
        return ExecuteResponse(outputs={}, outputs_arrow={})

    session._executor.execute = fake_execute
    session.set_option(scenario["options"])

    result = session.submit("data out; run;")

    assert result.success is True
    assert session.get_option("execution.trace") is True
    assert session.get_option("missing", scenario["expected_missing_default"]) == scenario["expected_missing_default"]
    assert session.get_option(["execution.trace", "execution.mode", "missing"], default=scenario["expected_missing_default"]) == {
        "execution.trace": True,
        "execution.mode": "debug",
        "missing": scenario["expected_missing_default"],
    }
    assert session.get_option() == scenario["options"]
    assert captured["options"] == scenario["options"]


def test_session_include_submits_file_contents(tmp_path) -> None:
    scenario = SESSION_METHOD_SCENARIOS["include_submits_file_contents"]
    session = Session(runtime_backend="python", parser_backend="python")
    session.load("inp", pa.table(scenario["inputs"]["inp"]))
    include_file = tmp_path / "program.dsl"
    include_file.write_text(scenario["dsl"], encoding="utf-8")

    result = session.include(str(include_file))

    assert result.success is True
    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_include_raises_for_missing_file(tmp_path) -> None:
    session = Session()

    with pytest.raises(FileNotFoundError):
        session.include(str(tmp_path / "missing.dsl"))


def test_session_cast_alias_uses_out_parameter() -> None:
    scenario = SESSION_METHOD_SCENARIOS["cast_alias"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    session.cast("src", {"amount": "float32"}, out="typed")

    assert session.to_arrow("typed").column("amount").type == scenario["expected_type"]


def test_session_sql_returns_arrow_and_can_store_target_from_create_table() -> None:
    scenario = SESSION_METHOD_SCENARIOS["sql_query_and_store"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    query_result = session.sql("select id, amount from src where amount > 10 order by id")
    stored_result = session.sql("create table out as select id from src where amount > 10 order by id")

    assert query_result.to_pylist() == scenario["expected_query"]
    assert stored_result.to_pylist() == scenario["expected_stored"]
    assert session["out"].to_pylist() == scenario["expected_stored"]


def test_session_sql_restores_arrow_metadata_after_polars_roundtrip() -> None:
    scenario = SESSION_METHOD_SCENARIOS["sql_metadata_roundtrip"]
    session = Session()
    session.load("src", pa.Table.from_pydict(scenario["source_rows"], schema=scenario["source_schema"]))

    query_result = session.sql("select id, amount from src order by id")
    session.sql("create table out as select id from src order by id")

    assert query_result.schema.metadata[b"memlabel"] == scenario["expected_memlabel"]
    assert query_result.schema.field("id").metadata[b"label"] == scenario["expected_id_label"]
    assert query_result.schema.field("amount").metadata[b"label"] == scenario["expected_amount_label"]
    assert session.to_arrow("out").schema.metadata[b"memlabel"] == scenario["expected_memlabel"]
    assert session.to_arrow("out").schema.field("id").metadata[b"label"] == scenario["expected_id_label"]


def test_session_transpose_matches_proc_transpose_style_defaults() -> None:
    scenario = SESSION_METHOD_SCENARIOS["transpose_default"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    session.transpose("src", var=["x", "y"], out="out")

    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_transpose_groups_by_columns_before_emitting_coln_rows() -> None:
    scenario = SESSION_METHOD_SCENARIOS["transpose_by"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    session.transpose("src", by=["grp"], var=["x", "y"], out="out")

    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_transpose_with_id_pivots_one_value_column_to_wide_output() -> None:
    scenario = SESSION_METHOD_SCENARIOS["transpose_id"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    session.transpose("src", by=["grp"], id="visit", var=["score"], out="out")

    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_transpose_resolves_columns_case_insensitively() -> None:
    scenario = SESSION_METHOD_SCENARIOS["transpose_case_insensitive"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    session.transpose("src", by=["grp"], id="visit", var=["score"], out="out")

    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_transpose_raises_for_duplicate_id_values_within_group() -> None:
    scenario = SESSION_METHOD_SCENARIOS["transpose_duplicate_id"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    with pytest.raises(ValueError, match=scenario["error_fragment"]):
        session.transpose("src", by=["grp"], id="visit", var=["score"], out="out")


def test_session_transpose_rejects_sequence_id_arguments() -> None:
    session = Session()
    session.load("src", pa.table({"grp": ["a"], "visit": ["v1"], "score": [10]}))

    with pytest.raises(TypeError, match="single column name"):
        session.transpose("src", by=["grp"], id=["visit"], var=["score"], out="out")


def test_session_assign_supports_literals_expressions_functions_case_when_and_ordering() -> None:
    scenario = SESSION_METHOD_SCENARIOS["assign_basic"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    session.assign(
        "src",
        out="out",
        cohort="'A'",
        name_up="upcase(name)",
        bmi="round(weight / (height_m * height_m), 0.01)",
        bmi_flag="case when bmi >= 25 then 'high' when bmi >= 18.5 then 'normal' else 'low' end",
        summary="catx(':', name_up, cohort)",
    )

    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_assign_raises_for_unsupported_functions() -> None:
    scenario = SESSION_METHOD_SCENARIOS["assign_unsupported_function"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    with pytest.raises(ValueError, match=scenario["error_fragment"]):
        session.assign("src", out="out", invalid="unknown_func(name)")


def test_session_assign_resolves_expression_columns_case_insensitively() -> None:
    scenario = SESSION_METHOD_SCENARIOS["assign_case_insensitive"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    session.assign(
        "src",
        out="out",
        name_up="upcase(name)",
        bmi="round(weight / (height_m * height_m), 0.01)",
    )

    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_assign_case_when_parser_ignores_keywords_inside_string_literals() -> None:
    scenario = SESSION_METHOD_SCENARIOS["assign_case_when_keywords_in_strings"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    session.assign(
        "src",
        out="out",
        flag="case when find(note, 'then') > 0 then catx(':', note, 'end') else 'else' end",
    )

    assert session["out"].to_pylist() == scenario["expected_output"]