import datetime as dt

import pyarrow as pa
import pytest
import re
from unittest.mock import patch

from limulus.arrow_bridge import restore_arrow_schema_from_sources
from limulus import Session
from limulus.models import ExecuteResponse


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


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


def test_session_step_style_methods_resolve_columns_case_insensitively() -> None:
    session = Session()
    session.load(
        "src",
        pa.table(
            {
                "Id": [1, 2],
                "Amount": [10, -1],
                "Name": ["Alice", "Bob"],
            }
        ),
    )

    session.where("src", "amount > 0", out="flt")
    session.keep("flt", ["ID", "amount", "name"], out="kept")
    session.drop("kept", ["NAME"], out="trimmed")
    session.rename("trimmed", {"amount": "amt"}, out="renamed")
    session.cast("renamed", {"AMT": "float32"}, out="out")

    assert session["out"].to_pylist() == [{"Id": 1, "amt": 10.0}]
    assert session.to_arrow("out").column("amt").type == pa.float32()


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


def test_session_astype_alias_adds_new_typed_column() -> None:
    session = Session()
    session.load("src", pa.table({"id": [1, 2], "amount": [1.25, 2.5]}))

    session.astype("src", {"amount": "float32"}, alias="amount_f32", out="typed")

    typed = session.to_arrow("typed")
    assert typed.column("amount").type == pa.float64()
    assert typed.column("amount_f32").type == pa.float32()
    assert typed.to_pylist() == [
        {"id": 1, "amount": 1.25, "amount_f32": 1.25},
        {"id": 2, "amount": 2.5, "amount_f32": 2.5},
    ]


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


def test_restore_arrow_schema_from_sources_matches_columns_case_insensitively() -> None:
    source = pa.Table.from_pydict(
        {"Amount": pa.array([10, 20], type=pa.int32())},
        schema=pa.schema(
            [pa.field("Amount", pa.int32(), metadata={b"label": b"Amount"})],
            metadata={b"memlabel": b"Case Source"},
        ),
    )
    target = pa.Table.from_pydict({"amount": pa.array([10, 20], type=pa.int64())})

    restored = restore_arrow_schema_from_sources(target, source_tables=(source,))

    assert restored.schema.metadata[b"memlabel"] == b"Case Source"
    assert restored.schema.field("amount").metadata[b"label"] == b"Amount"


def test_session_cast_keeps_requested_arrow_type_after_polars_roundtrip() -> None:
    session = Session()
    session.load(
        "src",
        pa.table({"Amount": pa.array([1.25, 2.5], type=pa.float64())}),
    )

    session.cast("src", {"amount": "float32"}, out="typed")

    assert session.to_arrow("typed").column("Amount").type == pa.float32()


def test_session_sort_assign_and_transpose_preserve_arrow_metadata_after_helper_pipeline() -> None:
    schema = pa.schema(
        [
            pa.field("grp", pa.string(), metadata={b"label": b"Group"}),
            pa.field("visit", pa.string(), metadata={b"label": b"Visit"}),
            pa.field("score", pa.int64(), metadata={b"label": b"Score"}),
        ],
        metadata={b"memlabel": b"Visits"},
    )
    source = pa.Table.from_pydict(
        {
            "grp": ["a", "a", "b"],
            "visit": ["v2", "v1", "v1"],
            "score": [20, 10, 30],
        },
        schema=schema,
    )
    session = Session()
    session.load("src", source)

    session.sort("src", ["grp", "visit"], out="sorted", nodupkey=True)
    session.assign("sorted", out="assigned", score_up="round(score, 1)")
    session.transpose("assigned", by=["grp"], id="visit", var=["score"], out="wide")

    sorted_table = session.to_arrow("sorted")
    assigned_table = session.to_arrow("assigned")
    wide_table = session.to_arrow("wide")

    assert sorted_table.schema.metadata[b"memlabel"] == b"Visits"
    assert sorted_table.schema.field("grp").metadata[b"label"] == b"Group"
    assert sorted_table.schema.field("score").metadata[b"label"] == b"Score"
    assert assigned_table.schema.metadata[b"memlabel"] == b"Visits"
    assert assigned_table.schema.field("grp").metadata[b"label"] == b"Group"
    assert assigned_table.schema.field("score").metadata[b"label"] == b"Score"
    assert wide_table.schema.metadata[b"memlabel"] == b"Visits"
    assert wide_table.schema.field("grp").metadata[b"label"] == b"Group"


def test_session_load_normalizes_work_prefixed_dataset_names_case_insensitively() -> None:
    session = Session()
    session.load("WORK.MixedCase", pa.table({"id": [1, 2]}))

    assert "mixedcase" in session.datasets
    assert "WORK.MIXEDCASE" in session.datasets
    assert session.to_arrow("mixedcase").to_pylist() == [{"id": 1}, {"id": 2}]
    assert session.to_arrow("WORK.MIXEDCASE").to_pylist() == [{"id": 1}, {"id": 2}]


def test_session_dictionary_tables_and_columns_reflect_arrow_schema_metadata() -> None:
    schema = pa.schema(
        [
            pa.field("subject_id", pa.int64(), metadata={b"label": b"Subject ID"}),
            pa.field("visit", pa.string(), metadata={b"label": b"Visit"}),
        ],
        metadata={b"memlabel": b"Clinical Visits"},
    )
    session = Session()
    session.load(
        "work.visits",
        pa.Table.from_pydict(
            {"subject_id": [1001, 1002], "visit": ["BASELINE", "WEEK4"]},
            schema=schema,
        ),
    )

    tables = session.dictionary.tables
    columns = session.dictionary.columns

    assert tables.to_pylist() == [
        {
            "LIBNAME": "WORK",
            "MEMNAME": "VISITS",
            "MEMTYPE": "DATA",
            "MEMLABEL": "Clinical Visits",
            "NOBS": 2,
            "NVAR": 2,
        }
    ]
    assert columns.column_names == [
        "LIBNAME",
        "MEMNAME",
        "MEMTYPE",
        "NAME",
        "TYPE",
        "VARNUM",
        "LABEL",
        "FORMAT",
        "INFORMAT",
    ]
    assert columns.to_pylist() == [
        {
            "LIBNAME": "WORK",
            "MEMNAME": "VISITS",
            "MEMTYPE": "DATA",
            "NAME": "subject_id",
            "TYPE": "int64",
            "VARNUM": 1,
            "LABEL": "Subject ID",
            "FORMAT": "",
            "INFORMAT": "",
        },
        {
            "LIBNAME": "WORK",
            "MEMNAME": "VISITS",
            "MEMTYPE": "DATA",
            "NAME": "visit",
            "TYPE": "string",
            "VARNUM": 2,
            "LABEL": "Visit",
            "FORMAT": "",
            "INFORMAT": "",
        },
    ]


def test_session_dictionary_dataset_helpers_and_reserved_names() -> None:
    session = Session()
    session.load("src", pa.table({"id": [1], "name": ["Alice"]}))

    assert session.dictionary("SRC").to_pylist() == [
        {
            "LIBNAME": "WORK",
            "MEMNAME": "SRC",
            "MEMTYPE": "DATA",
            "NAME": "id",
            "TYPE": "int64",
            "VARNUM": 1,
            "LABEL": "",
            "FORMAT": "",
            "INFORMAT": "",
        },
        {
            "LIBNAME": "WORK",
            "MEMNAME": "SRC",
            "MEMTYPE": "DATA",
            "NAME": "name",
            "TYPE": "string",
            "VARNUM": 2,
            "LABEL": "",
            "FORMAT": "",
            "INFORMAT": "",
        },
    ]

    with pytest.raises(ValueError, match="dictionary table"):
        session.load("dictionary.tables", pa.table({"id": [1]}))


def test_session_dictionary_reflects_submit_outputs_and_unload_removals() -> None:
    session = Session(runtime_backend="python", parser_backend="python")
    session.load("inp", pa.table({"id": [1, 2], "amount": [10, -1]}))

    result = session.submit("data out; set inp; if amount > 0 then output out; run;")

    assert result.success is True
    assert session.dictionary.tables.to_pylist() == [
        {"LIBNAME": "WORK", "MEMNAME": "INP", "MEMTYPE": "DATA", "MEMLABEL": "", "NOBS": 2, "NVAR": 2},
        {"LIBNAME": "WORK", "MEMNAME": "OUT", "MEMTYPE": "DATA", "MEMLABEL": "", "NOBS": 1, "NVAR": 2},
    ]

    assert session.unload("out") is True
    assert session.dictionary.tables.to_pylist() == [
        {"LIBNAME": "WORK", "MEMNAME": "INP", "MEMTYPE": "DATA", "MEMLABEL": "", "NOBS": 2, "NVAR": 2},
    ]


def test_session_sql_can_query_dictionary_tables_and_drop_datasets() -> None:
    session = Session()
    session.load("src", pa.table({"id": [1, 2], "name": ["Alice", "Bob"]}))

    dotted = session.sql(
        "select MEMNAME, NAME, TYPE from dictionary.columns where MEMNAME = 'SRC' order by VARNUM"
    )
    aliased = session.sql(
        "select MEMNAME, NAME, TYPE from dictionary_columns where MEMNAME = 'SRC' order by VARNUM"
    )

    assert dotted.to_pylist() == [
        {"MEMNAME": "SRC", "NAME": "id", "TYPE": "int64"},
        {"MEMNAME": "SRC", "NAME": "name", "TYPE": "string"},
    ]
    assert aliased.to_pylist() == dotted.to_pylist()

    assert session.sql("drop table src") is None
    assert "src" not in session.datasets


def test_session_sql_rejects_reserved_dictionary_create_targets() -> None:
    session = Session()
    session.load("src", pa.table({"id": [1]}))

    with pytest.raises(ValueError, match="dictionary table"):
        session.sql("create table dictionary.columns as select id from src")


def test_session_sql_drop_missing_table_raises_key_error() -> None:
    session = Session()

    with pytest.raises(KeyError, match="missing"):
        session.sql("drop table missing")


def test_session_sql_wraps_polars_execution_errors_with_rendered_message() -> None:
    session = Session()
    session.load("src", pa.table({"id": [1]}))

    with pytest.raises(ValueError) as error_info:
        session.sql("select * from missing")

    rendered = _strip_ansi(str(error_info.value))
    assert "session_sql_execution_error" in rendered.lower()
    assert "select * from missing" in rendered.lower()
    assert "sql execution failed" in rendered.lower()


def test_session_filter_parse_error_uses_renderer_output() -> None:
    session = Session()
    session.load("src", pa.table({"amount": [10, 20]}))

    with pytest.raises(ValueError) as error_info:
        session.filter("src", "amount between 10 and 20", out="flt")

    rendered = _strip_ansi(str(error_info.value))
    assert "session_filter_parse_error" in rendered.lower()
    assert "filter:src" in rendered
    assert "amount between 10 and 20" in rendered
    assert "simple comparison expression" in rendered.lower()


def test_session_sql_classification_error_uses_renderer_output() -> None:
    session = Session()
    session.load("src", pa.table({"id": [1, 2]}))

    with pytest.raises(ValueError) as error_info:
        session.sql("create table out select id from src")

    rendered = _strip_ansi(str(error_info.value))
    assert "session_sql_classification_error" in rendered.lower()
    assert "sql:1:1" in rendered.lower()
    assert "create table out select id from src" in rendered.lower()
    assert "create table name as <query>" in rendered.lower()


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


def test_session_transpose_preserves_value_column_arrow_type_for_wide_output() -> None:
    session = Session()
    session.load(
        "src",
        pa.table(
            {
                "grp": pa.array(["a", "a", "b"]),
                "visit": pa.array(["v1", "v2", "v1"]),
                "score": pa.array([10, 20, 30], type=pa.int32()),
            }
        ),
    )

    session.transpose("src", by=["grp"], id="visit", var=["score"], out="out")

    output = session.to_arrow("out")
    assert output.column("v1").type == pa.int32()
    assert output.column("v2").type == pa.int32()


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


def test_session_assign_case_when_parse_error_uses_renderer_output() -> None:
    session = Session()
    session.load("src", pa.table({"name": ["Alice"], "bmi": [22.0]}))

    with pytest.raises(ValueError) as error_info:
        session.assign(
            "src",
            out="out",
            bmi_flag="case when bmi >= then 'high' else 'low' end",
        )

    rendered = _strip_ansi(str(error_info.value))
    assert "column_api_case_when_parse_error" in rendered.lower()
    assert "assign:bmi_flag" in rendered
    assert "case when bmi >= then 'high' else 'low' end" in rendered
    assert "syntax error" in rendered.lower()


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


def test_session_assign_supports_case_insensitive_function_names() -> None:
    session = Session()
    session.load(
        "src",
        pa.table(
            {
                "name": ["Alice"],
                "weight": [50.0],
                "height_m": [1.60],
            }
        ),
    )

    session.assign(
        "src",
        out="out",
        name_up="UPCASE(name)",
        bmi="ROUND(weight / (height_m * height_m), 0.01)",
    )

    assert session["out"].to_pylist() == [
        {
            "name": "Alice",
            "weight": 50.0,
            "height_m": 1.60,
            "name_up": "ALICE",
            "bmi": 19.53,
        }
    ]


def test_session_assign_keeps_existing_numeric_column_types() -> None:
    session = Session()
    session.load(
        "src",
        pa.table(
            {
                "weight": pa.array([50.0, 80.0], type=pa.float32()),
                "height_m": pa.array([1.60, 1.80], type=pa.float32()),
            }
        ),
    )

    session.assign(
        "src",
        out="out",
        bmi="round(weight / (height_m * height_m), 0.01)",
    )

    output = session.to_arrow("out")
    assert output.column("weight").type == pa.float32()
    assert output.column("height_m").type == pa.float32()
    assert output.column("bmi").type == pa.float64()


def test_session_assign_uses_columnar_execution_without_row_evaluator() -> None:
    session = Session()
    session.load(
        "src",
        pa.table(
            {
                "name": ["Alice", "Bob"],
                "weight": [50.0, 80.0],
                "height_m": [1.60, 1.80],
            }
        ),
    )

    with patch(
        "limulus.column_api.ExpressionEvaluator.evaluate_scalar",
        side_effect=AssertionError("assign should not use row-wise scalar evaluation"),
    ):
        session.assign(
            "src",
            out="out",
            cohort="'A'",
            name_up="upcase(name)",
            bmi="round(weight / (height_m * height_m), 0.01)",
            summary="catx(':', name_up, cohort)",
        )

    assert session["out"].to_pylist() == [
        {
            "name": "Alice",
            "weight": 50.0,
            "height_m": 1.60,
            "cohort": "A",
            "name_up": "ALICE",
            "bmi": 19.53,
            "summary": "ALICE:A",
        },
        {
            "name": "Bob",
            "weight": 80.0,
            "height_m": 1.80,
            "cohort": "A",
            "name_up": "BOB",
            "bmi": 24.69,
            "summary": "BOB:A",
        },
    ]


def test_session_assign_supports_documented_columnar_function_registry() -> None:
    session = Session()
    session.load(
        "src",
        pa.table(
            {
                "name": ["hello world", "foo bar"],
                "phrase": ["hello world", "foo bar"],
            }
        ),
    )

    session.assign(
        "src",
        out="out",
        proper="propcase(name)",
        joined="cat(name, ' test')",
        stripped="cats(' a ', ' b ')",
        right_trimmed="catt('hello ', ' world')",
        position="index(name, 'world')",
        replaced="tranwrd(phrase, 'hello', 'hi')",
        translated="translate(phrase, 'HW', 'hw')",
        len_all="length(name)",
        len_nonblank="lengthn(name)",
        trimmed="strip('  x  ')",
        reversed="reverse(name)",
        repeated="repeat('ab', 3)",
        words="countw(name)",
    )

    assert session["out"].to_pylist() == [
        {
            "name": "hello world",
            "phrase": "hello world",
            "proper": "Hello World",
            "joined": "hello world test",
            "stripped": "ab",
            "right_trimmed": "hello world",
            "position": 7,
            "replaced": "hi world",
            "translated": "Hello World",
            "len_all": 11,
            "len_nonblank": 11,
            "trimmed": "x",
            "reversed": "dlrow olleh",
            "repeated": "ababab",
            "words": 2,
        },
        {
            "name": "foo bar",
            "phrase": "foo bar",
            "proper": "Foo Bar",
            "joined": "foo bar test",
            "stripped": "ab",
            "right_trimmed": "hello world",
            "position": 0,
            "replaced": "foo bar",
            "translated": "foo bar",
            "len_all": 7,
            "len_nonblank": 7,
            "trimmed": "x",
            "reversed": "rab oof",
            "repeated": "ababab",
            "words": 2,
        },
    ]


def test_session_assign_supports_put_input_and_hour_functions() -> None:
    session = Session()
    session.load(
        "src",
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

    session.assign(
        "src",
        out="out",
        code="put(id, 'z5')",
        fixed_text="put(amount, '8.1.')",
        rounded_text="put(amount, '8.')",
        comma_text="put(amount, 'comma8.1.')",
        zero_scaled="put(amount, 'z8.1.')",
        best_rendered="put(amount, 'best.')",
        best_value="input(best_text, 'best.')",
        visit_date="input(date_text, 'yymmdd10.')",
        visit_iso="put(visit_date, 'e8601da.')",
        timestamp_value="input(timestamp_text, 'e8601dt.')",
        timestamp_iso="put(timestamp_value, 'e8601dt.')",
        clock_value="input(clock_text, 'time.')",
        clock_iso="put(clock_value, 'time.')",
        clock_hour="hour(clock_text)",
    )

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


def test_session_assign_ignores_function_like_text_inside_string_literals() -> None:
    session = Session()
    session.load("src", pa.table({"name": ["Alice"]}))

    session.assign(
        "src",
        out="out",
        marker="cat(name, ' unknown_func(')",
    )

    assert session["out"].to_pylist() == [{"name": "Alice", "marker": "Alice unknown_func("}]