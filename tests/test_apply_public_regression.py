"""Public apply/native regression guarantees for v06 host safety."""

from __future__ import annotations

from decimal import Decimal

import pyarrow as pa

import limulus
from limulus import Session
from limulus.backend_integration.contracts import RuntimeExecutionResult


def _input_table() -> pa.Table:
    return pa.table({"id": [1, 2], "note": ["", "xy"], "amount": [4, 9]})


def _rows(table: pa.Table) -> list[dict[str, object]]:
    return table.to_pylist()


def test_session_submit_executes_builtin_qualified_and_explicit_apply_natively() -> None:
    def explicit_target(value: object) -> object:
        return value + 5  # type: ignore[operator]

    session = Session(backend="rust")
    session.load("in", _input_table())
    observations: list[RuntimeExecutionResult] = []
    original_execute = session._executor._rust_backend.execute_typed

    def observe(context):
        result = original_execute(context)
        observations.append(result)
        return result

    session._executor._rust_backend.execute_typed = observe
    try:
        def wrapper_submit():
            return session.submit(
                "data out; set in; "
                "note_length = apply('len', note); "
                "root = apply('math.sqrt', amount); "
                "shifted = apply('explicit_target', amount); "
                "output out; keep id note_length root shifted; run;",
                functions={"explicit_target": explicit_target},
            )

        result = wrapper_submit()
    finally:
        session._executor._rust_backend.execute_typed = original_execute

    assert result.success is True
    assert _rows(session.to_arrow("out")) == [
        {"id": 1, "note_length": 0.0, "root": 2.0, "shifted": 9.0},
        {"id": 2, "note_length": 2.0, "root": 3.0, "shifted": 14.0},
    ]
    assert len(observations) == 1
    assert observations[0].actual_engine == "rust"
    assert observations[0].native_invoked is True


def test_session_run_resolves_direct_caller_local_natively() -> None:
    def local_increment(value: object) -> object:
        return value + 2  # type: ignore[operator]

    session = Session(backend="rust")
    session.load("in", _input_table())
    result = session.run(
        "data out; set in; shifted = apply('local_increment', amount); "
        "output out; keep id shifted; run;"
    )

    assert result.success is True
    assert _rows(session.to_arrow("out")) == [
        {"id": 1, "shifted": 6.0},
        {"id": 2, "shifted": 11.0},
    ]


def test_top_level_submit_explicit_mapping_cannot_shadow_reserved_builtin() -> None:
    def shadowed_len(value: object) -> object:
        return 99

    def explicit_target(value: object) -> object:
        return value + 1  # type: ignore[operator]

    result = limulus.submit(
        "data out; set in; "
        "note_length = apply('len', note); "
        "root = apply('math.sqrt', amount); "
        "shifted = apply('explicit_target', amount); "
        "output out; keep id note_length root shifted; run;",
        functions={"len": shadowed_len, "explicit_target": explicit_target},
        **{"in": _input_table()},
    )

    assert result.success is True
    assert _rows(result.datasets["out"]) == [
        {"id": 1, "note_length": 0.0, "root": 2.0, "shifted": 5.0},
        {"id": 2, "note_length": 2.0, "root": 3.0, "shifted": 10.0},
    ]


def test_top_level_run_resolves_direct_caller_local_and_isolates_requests() -> None:
    def first_call():
        def shared_target(value: object) -> object:
            return value + 1  # type: ignore[operator]

        return limulus.run(
            "data out; set in; value = apply('shared_target', amount); "
            "output out; keep value; run;",
            **{"in": pa.table({"amount": [4]})},
        )

    first = first_call()
    assert first.success is True
    assert _rows(first.datasets["out"]) == [{"value": 5.0}]

    second = limulus.run(
        "data out; set in; value = apply('shared_target', amount); "
        "output out; keep value; run;",
        **{"in": pa.table({"amount": [4]})},
    )
    assert second.success is False
    assert second.log[0].code == "RUNTIME_APPLY_TARGET_UNRESOLVED"
    assert "shared_target" in second.log[0].message


def test_session_unresolved_target_reports_failure_before_native_invocation() -> None:
    session = Session(backend="rust")
    session.load("in", _input_table())
    observations: list[RuntimeExecutionResult] = []
    original_execute = session._executor._rust_backend.execute_typed

    def observe(context):
        result = original_execute(context)
        observations.append(result)
        return result

    session._executor._rust_backend.execute_typed = observe
    try:
        result = session.submit(
            "data out; set in; value = apply('missing_public_target', amount); "
            "output out; keep value; run;"
        )
    finally:
        session._executor._rust_backend.execute_typed = original_execute

    assert result.success is False
    assert result.log[0].code == "RUNTIME_APPLY_TARGET_UNRESOLVED"
    assert "missing_public_target" in result.log[0].message
    assert len(observations) == 1
    assert observations[0].actual_engine == "rust"
    assert observations[0].native_invoked is False


class TestBlockAtomicPublishContract:
    """Publication is atomic per DATA block.

    - a mid-row host callback failure or expression failure publishes no
      target for the failed block, so no partial rows or partial targets are
      exposed;
    - a pre-existing same-name target is preserved unchanged when its
      re-publish fails;
    - the diagnostic keeps the failing expression and its stage so the cause
      stays identifiable. The invalid-Decimal diagnostic
      text itself is pinned in ``tests/test_runtime_semantics.py``;
      this section pins only publication absence and the execute stage.
    """

    @staticmethod
    def _multi_row_table() -> pa.Table:
        return pa.table({"id": [1, 2, 3], "amount": [1.0, 2.0, 3.0]})

    def test_apply_callback_failure_publishes_no_partial_target(self) -> None:
        def boom(value: object) -> object:
            if value == 2:
                raise ValueError("boom on 2")
            return value * 10  # type: ignore[operator]

        session = Session(backend="rust")
        session.load("c", pa.table({"id": [1, 2, 3]}))
        result = session.submit(
            "data out; set c; y = apply('boom', id); run;",
            functions={"boom": boom},
        )

        assert result.success is False
        assert "out" not in result.datasets
        assert any(
            entry.severity == "error"
            and entry.code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
            and entry.stage == "execute"
            and "apply('boom', id)" in entry.message
            and "ValueError: boom on 2" in entry.message
            for entry in result.log
        )

    def test_expression_failure_publishes_no_partial_target(self) -> None:
        def snan(value: object) -> object:
            return Decimal("sNaN")

        session = Session(backend="rust")
        session.load("inp", self._multi_row_table())
        result = session.submit(
            "data out; set inp; val = apply('snan', amount); y = val + 1; run;",
            functions={"snan": snan},
        )

        assert result.success is False
        assert "out" not in result.datasets
        assert any(
            entry.severity == "error"
            and entry.code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
            and entry.stage == "execute"
            for entry in result.log
        )

    def test_failed_block_preserves_preexisting_same_name_target(self) -> None:
        def snan(value: object) -> object:
            return Decimal("sNaN")

        session = Session(backend="rust")
        session.load("inp", self._multi_row_table())

        initial = session.submit("data out; set inp; y = id; run;")
        assert initial.success is True
        assert _rows(session.to_arrow("out")) == [
            {"id": 1, "amount": 1.0, "y": 1},
            {"id": 2, "amount": 2.0, "y": 2},
            {"id": 3, "amount": 3.0, "y": 3},
        ]

        failed = session.submit(
            "data out; set inp; val = apply('snan', amount); y = val + 1; run;",
            functions={"snan": snan},
        )
        assert failed.success is False
        assert "out" not in failed.datasets

        # The failed re-publish of `out` must not have replaced the original
        # target: a later read still observes the 3 pre-existing rows.
        reread = session.submit("data w; set out; x = id; run;")
        assert reread.success is True
        assert _rows(session.to_arrow("w")) == [
            {"id": 1, "amount": 1.0, "y": 1, "x": 1},
            {"id": 2, "amount": 2.0, "y": 2, "x": 2},
            {"id": 3, "amount": 3.0, "y": 3, "x": 3},
        ]
