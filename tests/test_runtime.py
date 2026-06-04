import unittest
from unittest.mock import patch

import pyarrow as pa

from limulus.execution import DataStepExecutor
from limulus.models import DataSetRef, ExecuteRequest


def _arrow_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist([dict(row) for row in rows])


def _arrow_ref(name: str, rows: list[dict]) -> DataSetRef:
    return DataSetRef(
        kind="arrow_table",
        location=f"dataset://{name}",
        payload=_arrow_table(rows),
    )


def _output_rows(response, target: str) -> list[dict]:
    dataset = response.outputs[target]
    if dataset.kind == "arrow_table" and hasattr(dataset.payload, "to_pylist"):
        return dataset.payload.to_pylist()
    return dataset.payload


class RuntimeBackendContractTests(unittest.TestCase):
    def test_explicit_rust_backend_executes_eligible_workload_natively(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")

        response = executor.execute(
            ExecuteRequest(
                dsl_text="data out; set in; where amount >= 10; output out; run;",
                inputs={
                    "in": _arrow_ref(
                        "in",
                        [
                            {"id": 1, "amount": 5},
                            {"id": 2, "amount": 12},
                        ],
                    )
                },
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(executor.last_runtime_backend, "rust")
        self.assertEqual(_output_rows(response, "out"), [{"id": 2, "amount": 12}])

    def test_python_preference_routes_eligible_workload_to_rust(self) -> None:
        executor = DataStepExecutor(runtime_backend="python")

        response = executor.execute(
            ExecuteRequest(
                dsl_text="data out; set in; where amount >= 10; output out; run;",
                inputs={
                    "in": _arrow_ref(
                        "in",
                        [
                            {"id": 1, "amount": 5},
                            {"id": 2, "amount": 12},
                        ],
                    )
                },
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(executor.last_runtime_backend, "rust")
        self.assertEqual(_output_rows(response, "out"), [{"id": 2, "amount": 12}])

    def test_python_preference_falls_back_when_native_module_is_unavailable(self) -> None:
        executor = DataStepExecutor(runtime_backend="python")

        with patch(
            "limulus.backend_integration.rust_executor.load_native_module",
            return_value=(None, RuntimeError("missing native module")),
        ):
            response = executor.execute(
                ExecuteRequest(
                    dsl_text="data out; set in; where amount >= 10; output out; run;",
                    inputs={
                        "in": _arrow_ref(
                            "in",
                            [
                                {"id": 1, "amount": 5},
                                {"id": 2, "amount": 12},
                            ],
                        )
                    },
                )
            )

        self.assertFalse(response.has_errors)
        self.assertEqual(executor.last_runtime_backend, "python")
        self.assertEqual(_output_rows(response, "out"), [{"id": 2, "amount": 12}])

if __name__ == "__main__":
    unittest.main()
