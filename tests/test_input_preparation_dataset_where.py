from __future__ import annotations

from types import SimpleNamespace
import inspect

import pyarrow as pa

from limulus.dataset_option_expressions import WhereComparison
from limulus.execution import input_preparation


def _options(**overrides: object) -> SimpleNamespace:
    values = {
        "keep_vars": (),
        "drop_vars": (),
        "where_expr": None,
        "rename_map": {},
        "firstobs": None,
        "obs": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _table() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), metadata={b"label": b"Identifier"}),
            pa.field("Amount", pa.int64(), metadata={b"label": b"Amount"}),
            pa.field("temp", pa.string()),
        ],
        metadata={b"memlabel": b"Input"},
    )
    return pa.Table.from_arrays(
        [
            pa.array([1, 2, 3, 4], type=pa.int64()),
            pa.array([10, 20, 30, 40], type=pa.int64()),
            pa.array(["a", "b", "c", "d"]),
        ],
        schema=schema,
    )


def test_arrow_input_preparation_applies_typed_options_in_contract_order() -> None:
    result, diagnostic = input_preparation.try_apply_dataset_reference_options_to_arrow_table(
        _table(),
        source_name="input",
        option_spec=_options(
            keep_vars=("id", "Amount", "temp"),
            drop_vars=("temp",),
            where_expr=WhereComparison("amount", ">=", 20),
            rename_map={"Amount": "score"},
            firstobs=2,
            obs=1,
        ),
    )

    assert diagnostic is None
    assert result is not None
    assert result.to_pylist() == [{"id": 3, "score": 30}]
    assert result.schema.names == ["id", "score"]
    assert result.schema.metadata == {b"memlabel": b"Input"}
    assert result.schema.field("id").metadata == {b"label": b"Identifier"}
    assert result.schema.field("score").metadata == {b"label": b"Amount"}


def test_arrow_input_preparation_reports_unknown_where_column() -> None:
    result, diagnostic = input_preparation.try_apply_dataset_reference_options_to_arrow_table(
        _table(),
        source_name="input",
        option_spec=_options(where_expr=WhereComparison("missing", "=", 1)),
    )

    assert result is None
    assert diagnostic is not None
    assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
    assert "input" in diagnostic.message
    assert "missing" in diagnostic.message


def test_arrow_input_preparation_reports_where_type_mismatch() -> None:
    result, diagnostic = input_preparation.try_apply_dataset_reference_options_to_arrow_table(
        _table(),
        source_name="input",
        option_spec=_options(where_expr=WhereComparison("Amount", ">", "not numeric")),
    )

    assert result is None
    assert diagnostic is not None
    assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
    assert "input" in diagnostic.message
    assert "amount" in diagnostic.message.lower()
    assert ">" in diagnostic.message


def test_arrow_input_preparation_fails_closed_for_non_arrow_input() -> None:
    result, diagnostic = input_preparation.try_apply_dataset_reference_options_to_arrow_table(
        [{"id": 1}],
        source_name="input",
        option_spec=_options(),
    )

    assert result is None
    assert diagnostic is not None
    assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
    assert "Arrow" in diagnostic.message


def test_input_preparation_has_no_python_where_evaluation_or_regex_fallback() -> None:
    source = inspect.getsource(input_preparation)

    assert "eval(" not in source
    assert "_parse_simple_arrow_dataset_option_where" not in source
    assert "re.compile" not in source


def test_row_preparation_refuses_where_instead_of_evaluating_it() -> None:
    rows, diagnostic = input_preparation.apply_dataset_reference_options_to_rows(
        [{"id": 1}],
        source_name="input",
        option_spec=_options(where_expr=WhereComparison("id", "=", 1)),
    )

    assert rows == []
    assert diagnostic is not None
    assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
    assert "Arrow" in diagnostic.message
