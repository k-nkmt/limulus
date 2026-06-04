import datetime as dt

import pyarrow as pa
import polars as pl
import pytest

from limulus import Session
from limulus.backend_integration.backend_dispatch_policy import BackendDispatchPolicy
from limulus.backend_integration.contracts import RuntimeExecutionContext
from limulus.format_registry import FormatRegistry
from limulus.models import ExecuteRequest


def test_format_registry_builtin_families_cover_numeric_and_temporal_formats() -> None:
    registry = FormatRegistry()

    assert registry.put(7, "z5.") == "00007"
    assert registry.put(-7, "z5.") == "-0007"
    assert registry.put(123456, "z5.") == "123456"
    assert registry.put(12.3456, "8.") == "12"
    assert registry.put(12.3456, "8.2.") == "12.35"
    assert registry.put(12.3456, "z5.") == "00012"
    assert registry.put(12.3456, "z8.2.") == "00012.35"
    assert registry.put(12345.6, "comma8.") == "12,346"
    assert registry.put(12345.6, "comma8.1.") == "12,345.6"
    assert registry.put(12.0, "best.") == "12"
    assert registry.put(12.3456, "best.") == "12.3456"
    assert registry.put(dt.date(2024, 2, 3), "e8601da.") == "2024-02-03"
    assert registry.put(dt.date(2024, 2, 3), "yymmdd6.") == "240203"
    assert registry.put(dt.date(2024, 2, 3), "yymmdd8.") == "20240203"
    assert registry.put(dt.date(2024, 2, 3), "yymmdd10.") == "2024-02-03"
    assert registry.put(dt.datetime(2024, 2, 3, 16, 24, 43), "e8601dt.") == "2024-02-03T16:24:43"
    assert registry.put(dt.time(11, 30), "time.") == "11:30:00"

    assert registry.input("12", "best.") == 12.0
    assert registry.input("12.3456", "best.") == 12.3456
    assert registry.input("20240203", "yymmdd8.") == dt.date(2024, 2, 3)
    assert registry.input("240203", "yymmdd6.") == dt.date(2024, 2, 3)
    assert registry.input("2024-02-03", "yymmdd10.") == dt.date(2024, 2, 3)
    assert registry.input("2024-02-03", "e8601da.") == dt.date(2024, 2, 3)
    assert registry.input("2024-02-03T16:24:43", "e8601dt.") == dt.datetime(2024, 2, 3, 16, 24, 43)
    assert registry.input("11:30", "time.") == dt.time(11, 30)
    assert registry.hour("11:30") == 11.5


def test_format_registry_raises_for_invalid_numeric_and_temporal_inputs() -> None:
    registry = FormatRegistry()

    with pytest.raises(ValueError, match="Unsupported format"):
        registry.put(7, "unknown")

    with pytest.raises(ValueError, match="Unsupported format"):
        registry.put(7, "8")

    with pytest.raises(ValueError, match="Unsupported format"):
        registry.put(7, "comma8")

    with pytest.raises(ValueError):
        registry.input("2024/02/03", "yymmdd10")

    with pytest.raises(ValueError):
        registry.input("bad", "time")

    with pytest.raises(ValueError):
        registry.input("bad", "best.")

    assert registry.put(None, "z5.") is None
    assert registry.input("", "yymmdd10") is None


def test_session_registry_custom_format_and_informat_apply_to_assign_and_submit() -> None:
    session = Session(backend="python")
    session.register_format("tag", lambda value: f"TAG-{value}")
    session.register_informat("pipeint", lambda value: int(str(value).split("|")[-1]), kind="integer")
    session.load("src", pa.table({"id": [7], "raw": ["A|12"]}))

    session.assign(
        "src",
        out="helper",
        tag_text="put(id, 'tag')",
        parsed="input(raw, 'pipeint')",
    )
    session.submit(
        """
        data dsl;
        set src;
        tag_text = put(id, tag.);
        parsed = input(raw, pipeint.);
        output dsl;
        run;
        """
    )

    expected = [{"id": 7, "raw": "A|12", "tag_text": "TAG-7", "parsed": 12}]
    assert session["helper"].to_pylist() == expected
    assert session["dsl"].to_pylist() == expected


def test_session_auto_backend_routes_custom_format_and_informat_to_python_facade() -> None:
    session = Session(backend="auto")
    session.register_format("tag", lambda value: f"TAG-{value}")
    session.register_informat("pipeint", lambda value: int(str(value).split("|")[-1]), kind="integer")
    session.load("src", [{"id": 7, "raw": "A|12"}])

    result = session.submit(
        """
        data dsl;
        set src;
        tag_text = put(id, tag.);
        parsed = input(raw, pipeint.);
        output dsl;
        run;
        """
    )

    assert result.success is True
    assert session["dsl"].to_pylist() == [{"id": 7, "raw": "A|12", "tag_text": "TAG-7", "parsed": 12}]


def test_format_registry_supports_exact_match_dict_catalogs_and_character_namespace() -> None:
    registry = FormatRegistry()
    registry.register_format("tag", {7: "TAG-7"})
    registry.register_format("tag", {"A": "ALPHA"}, namespace="character")
    registry.register_informat("pipeint", {"A|12": 12.0})

    assert registry.put(7, "tag.") == "TAG-7"
    assert registry.put("A", "$tag.") == "ALPHA"
    assert registry.put("B", "$tag.") == "B"
    assert registry.input("A|12", "pipeint.") == 12.0
    assert registry.input("B|00", "pipeint.") is None
    assert registry.infer_input_kind("pipeint.") == "float64"


def test_dict_catalog_dispatch_hints_do_not_force_python_limited_format_routing() -> None:
    registry = FormatRegistry()
    registry.register_format("tag", {"A": "ALPHA"}, namespace="character")
    registry.register_informat("pipeint", {"A|12": 12.0})

    context = RuntimeExecutionContext(
        request=ExecuteRequest(
            dsl_text="data out; set src; text_out = put(code, $tag.); parsed = input(raw, pipeint.); run;",
            options={"dispatch_hints": registry.dispatch_hints()},
        ),
        ast_statements=(
            type(
                "Statement",
                (),
                {"text": "text_out = put(code, $tag.); parsed = input(raw, pipeint.);"},
            )(),
        ),
        resolved_inputs={},
        resolved_output_targets=("out",),
    )

    assert BackendDispatchPolicy.python_limited_reason_codes(context) == ()

    registry.register_format("callable_tag", lambda value: f"TAG-{value}")
    callable_context = RuntimeExecutionContext(
        request=ExecuteRequest(
            dsl_text="data out; set src; text_out = put(code, callable_tag.); run;",
            options={"dispatch_hints": registry.dispatch_hints()},
        ),
        ast_statements=(type("Statement", (), {"text": "text_out = put(code, callable_tag.);"})(),),
        resolved_inputs={},
        resolved_output_targets=("out",),
    )

    assert BackendDispatchPolicy.python_limited_reason_codes(callable_context) == (
        "PYTHON_LIMITED_FORMAT_REGISTRY",
    )


def test_session_rust_backend_supports_dict_custom_formats_and_informats() -> None:
    session = Session(backend="rust")
    session.register_format("tag", {"A": "ALPHA", "B": "BETA"}, namespace="character")
    session.register_informat("pipeint", {"A|12": 12.0, "B|03": 3.0})
    session.load("src", pa.table({"code": ["A", "Z"], "raw": ["A|12", "B|00"]}))

    result = session.submit(
        """
        data dsl;
        set src;
        text_out = put(code, $tag.);
        parsed = input(raw, pipeint.);
        output dsl;
        run;
        """,
        backend="rust",
    )

    assert result.success is True
    assert session["dsl"].to_pylist() == [
        {"code": "A", "raw": "A|12", "text_out": "ALPHA", "parsed": 12.0},
        {"code": "Z", "raw": "B|00", "text_out": "Z", "parsed": None},
    ]


def test_session_assign_uses_columnar_mapping_for_dict_catalog_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    session = Session(backend="python")
    session.register_format("tag", {"A": "ALPHA", "B": "BETA"}, namespace="character")
    session.register_informat("pipeint", {"A|12": 12.0, "B|03": 3.0})
    session.load("src", pa.table({"code": ["A", "Z"], "raw": ["A|12", "B|00"]}))

    original_map_elements = pl.Expr.map_elements

    def fail_map_elements(self: pl.Expr, *args: object, **kwargs: object) -> pl.Expr:
        raise AssertionError("dict catalog assign path should stay columnar")

    monkeypatch.setattr(pl.Expr, "map_elements", fail_map_elements)

    session.assign(
        "src",
        out="helper",
        text_out="put(code, '$tag.')",
        parsed="input(raw, 'pipeint.')",
    )

    monkeypatch.setattr(pl.Expr, "map_elements", original_map_elements)

    assert session["helper"].to_pylist() == [
        {"code": "A", "raw": "A|12", "text_out": "ALPHA", "parsed": 12.0},
        {"code": "Z", "raw": "B|00", "text_out": "Z", "parsed": None},
    ]
