from __future__ import annotations

from pathlib import Path

import pytest

from limulus import Session
from tests.helpers.assertion_helpers import assert_rows_match, assert_table_metadata
from tests.helpers.datastep_case_loader import load_case_packs


CASE_ROOT = Path(__file__).resolve().parents[2] / "cases" / "datastep"


def _run_case(case, *, runtime_backend: str) -> Session:
    session = Session(runtime_backend=runtime_backend, parser_backend="python")
    for input_name, table in case.arrow_inputs().items():
        session.load(input_name, table)

    result = session.submit(case.program_text)
    assert result.success is True, f"case={case.case_id}, backend={runtime_backend} failed"
    return session


@pytest.mark.parametrize(
    "case",
    load_case_packs(CASE_ROOT),
    ids=lambda c: c.case_id,
)
def test_external_case_packs_match_limulus_expected(case) -> None:
    backends = ["python", "rust"] if case.backend == "auto" else [case.backend]

    for backend in backends:
        session = _run_case(case, runtime_backend=backend)
        for dataset_name, expected_dataset in case.expected_datasets.items():
            assert_rows_match(
                session[dataset_name].to_pylist(),
                expected_dataset.rows,
                case_id=f"{case.case_id}:{dataset_name}",
                backend=backend,
            )
            assert_table_metadata(
                session.to_arrow(dataset_name),
                memlabel=expected_dataset.memlabel,
                column_labels=expected_dataset.column_labels,
                case_id=f"{case.case_id}:{dataset_name}",
                backend=backend,
            )
