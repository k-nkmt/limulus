"""Minimal smoke test for the currently built limulus wheel.

This module is executed by the isolated wheel gate.  It deliberately imports
only the installed package and exercises the public top-level ``submit``
entrypoint with the Rust backend.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

import limulus
import limulus_native


def _assert_installed_modules() -> None:
    """Reject imports that resolve back to this source checkout.

    The wheel gate must prove that the packaged modules work in a clean
    environment; an editable install or an accidental source-tree import
    would make that check ineffective.
    """

    source_root = Path(__file__).resolve().parents[1]
    for module in (limulus, limulus_native):
        module_file = getattr(module, "__file__", None)
        assert module_file is not None, f"{module.__name__} has no module origin"
        resolved_file = Path(module_file).resolve()
        assert source_root not in resolved_file.parents, (
            f"{module.__name__} imported from source checkout: {resolved_file}"
        )


def _run_session_load_submit_smoke() -> None:
    """Exercise ``Session.load`` registration followed by ``Session.submit``.

    A small Arrow input is registered on the session catalog, a data step
    transforms it, and both the submit result and the retrieval-side dataset
    must report the same expected rows.
    """

    session = limulus.Session()
    session.load("smoke_input", pa.table({"id": [1, 2, 3]}))

    result = session.submit(
        """
        data smoke_output;
          set smoke_input;
          if id > 1;
        run;
        """,
    )

    expected_rows = [{"id": 2}, {"id": 3}]
    assert result.success is True
    assert result.log == ()
    assert set(result.datasets) == {"smoke_output"}
    assert result.datasets["smoke_output"].to_pylist() == expected_rows
    assert session.dataset("smoke_output").to_arrow().to_pylist() == expected_rows


def _run_explicit_apply_smoke() -> None:
    """Exercise explicit ``apply`` through ``Session.submit(functions=...)``.

    The Rust row loop evaluates each row's arguments and calls the explicitly
    registered Python callable with them, so a plain local function must be
    reachable by name and produce the expected per-row values.
    """

    def double(value: int) -> int:
        return value * 2

    session = limulus.Session()
    session.load("apply_input", pa.table({"id": [1, 2, 3]}))

    result = session.submit(
        """
        data apply_output;
          set apply_input;
          doubled = apply("double", id);
        run;
        """,
        functions={"double": double},
    )

    expected_rows = [
        {"id": 1, "doubled": 2.0},
        {"id": 2, "doubled": 4.0},
        {"id": 3, "doubled": 6.0},
    ]
    assert result.success is True
    assert result.log == ()
    assert set(result.datasets) == {"apply_output"}
    assert result.datasets["apply_output"].to_pylist() == expected_rows
    assert session.dataset("apply_output").to_arrow().to_pylist() == expected_rows


def main() -> None:
    _assert_installed_modules()
    assert callable(limulus_native.execute_block)

    result = limulus.submit(
        "data out; set inp; output out; run;",
        backend="rust",
        inp=pa.table({"id": [1]}),
    )

    assert result.success is True
    assert set(result.datasets) == {"out"}
    assert result.datasets["out"].to_pylist() == [{"id": 1}]

    _run_session_load_submit_smoke()
    _run_explicit_apply_smoke()


if __name__ == "__main__":
    main()
