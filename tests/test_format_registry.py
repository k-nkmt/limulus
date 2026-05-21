import datetime as dt

import pyarrow as pa
import pytest

from limulus import Session
from limulus.format_registry import FormatRegistry


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