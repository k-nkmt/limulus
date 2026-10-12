from __future__ import annotations

import pyarrow as pa

from limulus import Session
from limulus.execution import DataStepExecutor
from limulus.models import DataSetRef, ExecuteRequest
from tests.helpers.assertion_helpers import (
    assert_rows_match,
    assert_table_metadata,
)


def _arrow_ref(name: str, table: pa.Table) -> DataSetRef:
    return DataSetRef(
        kind="arrow_table",
        location=f"dataset://{name}",
        payload=table,
    )


def test_rust_row_success_baseline_reports_actual_engine() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set inp; where amount >= 10; output out; run;",
            inputs={
                "inp": _arrow_ref(
                    "inp",
                    pa.table({"id": [1, 2], "amount": [5, 12]}),
                )
            },
        )
    )

    assert response.has_errors is False
    assert_rows_match(
        response.outputs["out"].payload.to_pylist(),
        [{"id": 2, "amount": 12}],
        case_id="rust-row-success:out",
        backend="rust",
    )


def test_rust_runtime_diagnostic_baseline_reports_actual_engine() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set inp; if id === 1 then output out; run;",
            inputs={"inp": _arrow_ref("inp", pa.table({"id": [1]}))},
        )
    )

    assert response.has_errors is True
    assert len(response.diagnostics) == 1
    diagnostic = response.diagnostics[0]
    assert diagnostic.code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert diagnostic.stage == "execute"
    assert diagnostic.severity == "error"
    assert "Rust statement execution failed" in diagnostic.message
    assert "expression expected" in diagnostic.message


def test_rust_arrow_io_and_metadata_baseline_is_preserved() -> None:
    session = Session(runtime_backend="rust")
    session.load("inp", pa.table({"id": [1], "amount": [10]}))

    result = session.submit(
        """
        data dm(label="DM");
        set inp;
        label id = "Identifier" amount = "Amount";
        output dm;
        run;
        """
    )

    assert result.success is True
    output = session.to_arrow("dm")
    assert isinstance(output, pa.Table)
    assert_rows_match(
        output.to_pylist(),
        [{"id": 1, "amount": 10}],
        case_id="rust-arrow-metadata:dm",
        backend="rust",
    )
    assert_table_metadata(
        output,
        memlabel="DM",
        column_labels={"id": "Identifier", "amount": "Amount"},
        case_id="rust-arrow-metadata:dm",
        backend="rust",
    )


def test_rust_dict_format_catalog_baseline_reports_actual_engine() -> None:
    session = Session(runtime_backend="rust")
    session.register_format(
        "tag",
        {"A": "ALPHA", "B": "BETA"},
        namespace="character",
    )
    session.register_informat("pipeint", {"A|12": 12.0, "B|03": 3.0})
    session.load("src", pa.table({"code": ["A", "Z"], "raw": ["A|12", "B|00"]}))

    result = session.submit(
        """
        data out;
        set src;
        text_out = put(code, $tag.);
        parsed = input(raw, pipeint.);
        output out;
        run;
        """
    )

    assert result.success is True
    assert_rows_match(
        session.to_arrow("out").to_pylist(),
        [
            {"code": "A", "raw": "A|12", "text_out": "ALPHA", "parsed": 12.0},
            {"code": "Z", "raw": "B|00", "text_out": "Z", "parsed": None},
        ],
        case_id="rust-dict-format-catalog:out",
        backend="rust",
    )
