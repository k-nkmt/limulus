import pyarrow as pa
import pytest

from limulus import Session


DATASET_VIEW_METHOD_SCENARIOS = {
    "sort_nodupkey": {
        "overview": "DatasetView.sort uses out= and keeps the first sorted row per duplicate key when nodupkey is enabled",
        "inputs": {
            "src": {
                "group_id": [1, 1, 2],
                "amount": [20, 20, 30],
            }
        },
        "expected_name": "deduped",
        "expected_output": [
            {"group_id": 1, "amount": 20},
            {"group_id": 2, "amount": 30},
        ],
    },
    "select": {
        "overview": "DatasetView.select projects the requested columns and materializes the result via out=",
        "inputs": {"src": {"id": [1, 2], "amount": [10, 20], "name": ["a", "b"]}},
        "expected_name": "selected",
        "expected_output": [
            {"name": "a", "amount": 10},
            {"name": "b", "amount": 20},
        ],
    },
    "rejects_target": {
        "overview": "DatasetView methods are out-only and reject the legacy target parameter",
        "inputs": {"src": {"id": [1]}},
    },
    "chained_methods": {
        "overview": "DatasetView chaining keeps the intermediate style readable while preserving expected rows",
        "inputs": {
            "inp": {
                "id": [1, 2, 3],
                "amount": [10, -1, 30],
                "name": ["Alice", "Bob", "Catherine"],
            }
        },
        "expected_name": "v2",
        "expected_output": [{"id": 1, "amount": 10}, {"id": 3, "amount": 30}],
    },
    "cast": {
        "overview": "DatasetView.cast converts a column and exposes the typed result through the returned view",
        "inputs": {"src": {"id": [1, 2], "amount": ["10", "20"]}},
        "expected_type": pa.int64(),
        "expected_output": [{"id": 1, "amount": 10}, {"id": 2, "amount": 20}],
    },
    "astype_materialized": {
        "overview": "DatasetView.astype materializes the requested Arrow type on the stored output table",
        "inputs": {"src": {"id": [1, 2], "amount": [1.25, 2.5]}},
        "expected_type": pa.float32(),
    },
    "cast_metadata_roundtrip": {
        "overview": "DatasetView.cast preserves table and field metadata after the internal Polars conversion",
        "source_rows": {"id": [1, 2], "amount": ["10", "20"]},
        "source_schema": pa.schema(
            [
                pa.field("id", pa.int64(), metadata={b"label": b"Identifier"}),
                pa.field("amount", pa.string(), metadata={b"label": b"Amount"}),
            ],
            metadata={b"memlabel": b"Source Label"},
        ),
        "expected_memlabel": b"Source Label",
        "expected_id_label": b"Identifier",
        "expected_amount_label": b"Amount",
        "expected_type": pa.int64(),
    },
    "transpose": {
        "overview": "DatasetView.transpose materializes the wide result through out=",
        "inputs": {
            "src": {
                "grp": ["a", "a", "b"],
                "visit": ["v1", "v2", "v1"],
                "score": [10, 20, 30],
            }
        },
        "expected_name": "wide_scores",
        "expected_output": [
            {"grp": "a", "v1": 10, "v2": 20},
            {"grp": "b", "v1": 30, "v2": None},
        ],
    },
    "assign": {
        "overview": "DatasetView.assign supports chained column creation using out=",
        "inputs": {
            "src": {
                "name": ["Alice", "Bob"],
                "weight": [50.0, 80.0],
                "height_m": [1.60, 1.80],
            }
        },
        "expected_name": "scored",
        "expected_output": [
            {
                "name": "Alice",
                "weight": 50.0,
                "height_m": 1.60,
                "name_up": "ALICE",
                "bmi": 19.53,
            },
            {
                "name": "Bob",
                "weight": 80.0,
                "height_m": 1.80,
                "name_up": "BOB",
                "bmi": 24.69,
            },
        ],
    },
}


def test_dataset_view_sort_nodupkey_uses_out_parameter() -> None:
    scenario = DATASET_VIEW_METHOD_SCENARIOS["sort_nodupkey"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    view = session.dataset("src").sort(
        [("group_id", "ascending"), ("amount", "ascending")],
        out="deduped",
        nodupkey=True,
    )

    assert view.name == scenario["expected_name"]
    assert view.to_arrow().to_pylist() == scenario["expected_output"]


def test_dataset_view_select_uses_out_parameter() -> None:
    scenario = DATASET_VIEW_METHOD_SCENARIOS["select"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    view = session.dataset("src").select(["name", "amount"], out="selected")

    assert view.name == scenario["expected_name"]
    assert view.to_arrow().to_pylist() == scenario["expected_output"]


def test_dataset_view_methods_do_not_accept_target_parameter() -> None:
    scenario = DATASET_VIEW_METHOD_SCENARIOS["rejects_target"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    with pytest.raises(TypeError):
        session.dataset("src").keep(["id"], target="out")


def test_dataset_view_chained_methods() -> None:
    scenario = DATASET_VIEW_METHOD_SCENARIOS["chained_methods"]
    session = Session()
    session.load("inp", pa.table(scenario["inputs"]["inp"]))

    view = session.dataset("inp").where("amount > 0", out="v1").keep(["id", "amount"], out="v2")

    assert view.name == scenario["expected_name"]
    assert view.to_arrow().to_pylist() == scenario["expected_output"]


def test_dataset_view_chained_methods_resolve_columns_case_insensitively() -> None:
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

    view = (
        session.dataset("src")
        .where("amount > 0", out="flt")
        .keep(["ID", "name"], out="out")
    )

    assert view.name == "out"
    assert view.to_arrow().to_pylist() == [{"Id": 1, "Name": "Alice"}]


def test_dataset_view_astype() -> None:
    scenario = DATASET_VIEW_METHOD_SCENARIOS["cast"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    view = session.dataset("src").cast({"amount": "int64"}, out="typed_view")

    assert view.to_arrow().column("amount").type == scenario["expected_type"]
    assert view.to_arrow().to_pylist() == scenario["expected_output"]


def test_dataset_view_astype_preserves_arrow_type_on_materialized_view() -> None:
    scenario = DATASET_VIEW_METHOD_SCENARIOS["astype_materialized"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    session.dataset("src").astype({"amount": "float32"}, out="typed_view")

    assert session.to_arrow("typed_view").column("amount").type == scenario["expected_type"]


def test_dataset_view_astype_alias_adds_new_typed_column() -> None:
    session = Session()
    session.load("src", pa.table({"id": [1, 2], "amount": [1.25, 2.5]}))

    view = session.dataset("src").astype({"amount": "float32"}, alias="amount_f32", out="typed_view")

    assert view.to_arrow().column("amount").type == pa.float64()
    assert view.to_arrow().column("amount_f32").type == pa.float32()
    assert view.to_arrow().to_pylist() == [
        {"id": 1, "amount": 1.25, "amount_f32": 1.25},
        {"id": 2, "amount": 2.5, "amount_f32": 2.5},
    ]


def test_dataset_view_cast_preserves_arrow_metadata_after_polars_roundtrip() -> None:
    scenario = DATASET_VIEW_METHOD_SCENARIOS["cast_metadata_roundtrip"]
    session = Session()
    session.load("src", pa.Table.from_pydict(scenario["source_rows"], schema=scenario["source_schema"]))

    session.dataset("src").cast({"amount": "int64"}, out="typed_view")

    table = session.to_arrow("typed_view")
    assert table.schema.metadata[b"memlabel"] == scenario["expected_memlabel"]
    assert table.schema.field("id").metadata[b"label"] == scenario["expected_id_label"]
    assert table.schema.field("amount").metadata[b"label"] == scenario["expected_amount_label"]
    assert table.column("amount").type == scenario["expected_type"]


def test_dataset_view_transpose_uses_out_parameter() -> None:
    scenario = DATASET_VIEW_METHOD_SCENARIOS["transpose"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    view = session.dataset("src").transpose(by=["grp"], id="visit", var=["score"], out="wide_scores")

    assert view.name == scenario["expected_name"]
    assert view.to_arrow().to_pylist() == scenario["expected_output"]


def test_dataset_view_transpose_rejects_sequence_id_arguments() -> None:
    session = Session()
    session.load("src", pa.table({"grp": ["a"], "visit": ["v1"], "score": [10]}))

    with pytest.raises(TypeError, match="single column name"):
        session.dataset("src").transpose(by=["grp"], id=["visit"], var=["score"], out="wide_scores")


def test_dataset_view_assign_uses_out_parameter_and_left_to_right_evaluation() -> None:
    scenario = DATASET_VIEW_METHOD_SCENARIOS["assign"]
    session = Session()
    session.load("src", pa.table(scenario["inputs"]["src"]))

    view = session.dataset("src").assign(
        out="scored",
        name_up="upcase(name)",
        bmi="round(weight / (height_m * height_m), 0.01)",
    )

    assert view.name == scenario["expected_name"]
    assert view.to_arrow().to_pylist() == scenario["expected_output"]


def test_dataset_view_dictionary_returns_dataset_specific_columns() -> None:
    session = Session()
    session.load("src", pa.table({"id": [1], "name": ["Alice"]}))

    assert session.dataset("src").dictionary.to_pylist() == [
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