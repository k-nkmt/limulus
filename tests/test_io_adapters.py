import pytest
import pyarrow as pa
import polars as pl

from limulus import Session
from limulus.io_adapters import (
    DataFrameAdapterPandas,
    DataInputAdapterArrow,
    DataOutputAdapterArrow,
    InputSpec,
    OutputSpec,
)

IO_ADAPTER_SCENARIOS = {
    "arrow_roundtrip": {
        "overview": "Arrow table canonicalization and in-memory roundtrip preserve row content",
        "arrow_rows": [{"id": 1, "amount": 10}, {"id": 2, "amount": 20}],
    },
    "dataframe_adapter_support": {
        "overview": "Pandas adapter converts canonical rows both directions and honors input/output specs",
        "rows": [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}],
        "spec_rows": [{"id": 10, "group": "A"}],
        "spec_location": "memory://out",
    },
    "session_dataframe_ingest": {
        "overview": "Session accepts polars/pandas dataframes as first-class inputs for DATA step execution",
        "dsl": "data out; set inp; where amount > 0; output out; run;",
        "polars_rows": {"id": [1, 2], "amount": [10, -1]},
        "polars_expected": [{"id": 1, "amount": 10}],
        "pandas_rows": [{"id": 1, "amount": 5}, {"id": 2, "amount": -1}],
        "pandas_expected": [{"id": 1, "amount": 5}],
    },
}

def test_arrow_table_input_adapter_converts_arrow_table_to_canonical() -> None:
    scenario = IO_ADAPTER_SCENARIOS["arrow_roundtrip"]
    pyarrow = pytest.importorskip("pyarrow")
    adapter = DataInputAdapterArrow()

    table = pyarrow.Table.from_pylist(scenario["arrow_rows"])
    rows = adapter.load(InputSpec(format="arrow_table", payload=table))

    assert rows == scenario["arrow_rows"]


def test_arrow_table_input_adapter_supports_pyarrow_to_pylist_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = IO_ADAPTER_SCENARIOS["arrow_roundtrip"]
    pyarrow = pytest.importorskip("pyarrow")
    adapter = DataInputAdapterArrow()
    monkeypatch.setenv("LIMULUS_ARROW_ROW_READER", "pyarrow_to_pylist")

    table = pyarrow.Table.from_pylist(scenario["arrow_rows"])
    rows = adapter.load(InputSpec(format="arrow_table", payload=table))

    assert rows == scenario["arrow_rows"]


def test_arrow_table_input_adapter_supports_polars_iter_rows_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = IO_ADAPTER_SCENARIOS["arrow_roundtrip"]
    pyarrow = pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    adapter = DataInputAdapterArrow()
    monkeypatch.setenv("LIMULUS_ARROW_ROW_READER", "polars_iter_rows")
    monkeypatch.setenv("LIMULUS_POLARS_ITER_ROWS_BUFFER_SIZE", "4096")

    table = pyarrow.Table.from_pylist(scenario["arrow_rows"])
    rows = adapter.load(InputSpec(format="arrow_table", payload=table))

    assert rows == scenario["arrow_rows"]


def test_arrow_output_adapter_stores_arrow_table() -> None:
    scenario = IO_ADAPTER_SCENARIOS["arrow_roundtrip"]
    pytest.importorskip("pyarrow")
    output_adapter = DataOutputAdapterArrow()

    output_ref = output_adapter.store(
        scenario["arrow_rows"],
        OutputSpec(format="arrow_table", location="memory://out"),
    )
    loaded = DataInputAdapterArrow().load(InputSpec(format="arrow_table", payload=output_ref.payload))
    assert loaded == scenario["arrow_rows"]


def test_pandas_adapter_converts_both_directions() -> None:
    scenario = IO_ADAPTER_SCENARIOS["dataframe_adapter_support"]
    pandas = pytest.importorskip("pandas")
    adapter = DataFrameAdapterPandas()

    frame = pandas.DataFrame(scenario["rows"])
    rows = adapter.to_canonical(frame)
    restored = adapter.from_canonical(rows)

    assert rows == scenario["rows"]
    assert restored.to_dict(orient="records") == rows


def test_pandas_adapter_load_and_store_with_specs() -> None:
    scenario = IO_ADAPTER_SCENARIOS["dataframe_adapter_support"]
    pandas = pytest.importorskip("pandas")
    adapter = DataFrameAdapterPandas()

    frame = pandas.DataFrame(scenario["spec_rows"])
    loaded = adapter.load(InputSpec(format="pandas", payload=frame))
    ref = adapter.store(loaded, OutputSpec(format="pandas", location=scenario["spec_location"]))

    assert ref.kind == "pandas"
    assert ref.location == scenario["spec_location"]
    assert isinstance(ref.payload, pandas.DataFrame)
    assert ref.payload.to_dict(orient="records") == scenario["spec_rows"]


def test_session_load_accepts_polars_dataframe_as_first_class_input() -> None:
    scenario = IO_ADAPTER_SCENARIOS["session_dataframe_ingest"]
    session = Session()
    frame = pl.DataFrame(scenario["polars_rows"])

    session.load("inp", frame)
    result = session.submit(scenario["dsl"])

    assert result.success is True
    assert session["out"].to_pylist() == scenario["polars_expected"]


def test_session_load_accepts_pandas_dataframe_when_available() -> None:
    scenario = IO_ADAPTER_SCENARIOS["session_dataframe_ingest"]
    pandas = pytest.importorskip("pandas")
    session = Session()
    frame = pandas.DataFrame(scenario["pandas_rows"])

    session.load("inp", frame)
    result = session.submit(scenario["dsl"])

    assert result.success is True
    assert session.to_arrow("out").to_pylist() == scenario["pandas_expected"]
