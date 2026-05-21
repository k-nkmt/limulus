from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from typing import Any


_Z_FORMAT = re.compile(r"^z(\d+)(?:\.(\d+))?$", re.IGNORECASE)
_WD_FORMAT = re.compile(r"^(\d+)(?:\.(\d+))?$")
_COMMA_WD_FORMAT = re.compile(r"^comma(\d+)(?:\.(\d+))?$", re.IGNORECASE)
_YYMMDD_FORMAT = re.compile(r"^yymmdd(6|8|10)$", re.IGNORECASE)


class FormatRegistry:
    def __init__(self) -> None:
        self._formatters: dict[str, Callable[[Any], Any]] = {}
        self._informats: dict[str, Callable[[Any], Any]] = {}
        self._informat_kinds: dict[str, str] = {}

    def register_format(self, name: str, formatter: Callable[[Any], Any]) -> None:
        self._formatters[self.normalize_name(name)] = formatter

    def register_informat(self, name: str, parser: Callable[[Any], Any], *, kind: str | None = None) -> None:
        normalized = self.normalize_name(name)
        self._informats[normalized] = parser
        if kind is not None:
            self._informat_kinds[normalized] = kind.strip().lower()

    def put(self, value: Any, format_name: Any) -> Any:
        if value is None:
            return None

        raw_text = self._coerce_name_text(format_name)
        normalized = self.normalize_name(raw_text)
        formatter = self._formatters.get(normalized)
        if formatter is not None:
            return formatter(value)

        if self._requires_numeric_format_dot(raw_text, normalized):
            raise ValueError(f"Unsupported format: {format_name}")

        z_match = _Z_FORMAT.fullmatch(normalized)
        if z_match is not None:
            return self._format_z(
                value,
                width=int(z_match.group(1)),
                decimals=int(z_match.group(2) or "0"),
            )

        wd_match = _WD_FORMAT.fullmatch(normalized)
        if wd_match is not None:
            return self._format_wd(
                value,
                width=int(wd_match.group(1)),
                decimals=int(wd_match.group(2) or "0"),
            )

        comma_wd_match = _COMMA_WD_FORMAT.fullmatch(normalized)
        if comma_wd_match is not None:
            return self._format_comma_wd(
                value,
                width=int(comma_wd_match.group(1)),
                decimals=int(comma_wd_match.group(2) or "0"),
            )

        if normalized == "best":
            return self._format_best(value)

        if normalized == "e8601da":
            return self._coerce_date(value).isoformat()
        if normalized == "e8601dt":
            return self._coerce_datetime(value).isoformat(timespec="seconds")
        if normalized == "time":
            return self._coerce_time(value).isoformat(timespec="seconds")

        raise ValueError(f"Unsupported format: {format_name}")

    def input(self, value: Any, informat_name: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None

        normalized = self.normalize_name(informat_name)
        parser = self._informats.get(normalized)
        if parser is not None:
            return parser(value)

        if normalized == "e8601da":
            return self._coerce_date(value)
        if normalized == "e8601dt":
            return self._coerce_datetime(value)
        if normalized == "time":
            return self._coerce_time(value)

        if normalized == "best":
            return self._parse_best(value)

        yymmdd_match = _YYMMDD_FORMAT.fullmatch(normalized)
        if yymmdd_match is not None:
            return self._parse_yymmdd(value, width=int(yymmdd_match.group(1)))

        raise ValueError(f"Unsupported informat: {informat_name}")

    def hour(self, value: Any) -> Any:
        if value is None:
            return None
        time_value = self._coerce_time(value)
        return (
            time_value.hour
            + time_value.minute / 60.0
            + time_value.second / 3600.0
            + time_value.microsecond / 3_600_000_000.0
        )

    def infer_input_kind(self, informat_name: Any) -> str:
        normalized = self.normalize_name(informat_name)
        registered_kind = self._informat_kinds.get(normalized)
        if registered_kind is not None:
            return registered_kind
        if normalized in {"e8601da", "yymmdd6", "yymmdd8", "yymmdd10"}:
            return "date"
        if normalized == "e8601dt":
            return "datetime"
        if normalized == "time":
            return "time"
        if normalized == "best":
            return "float"
        raise ValueError(f"Unsupported informat: {informat_name}")

    @staticmethod
    def normalize_name(name: Any) -> str:
        normalized = FormatRegistry._coerce_name_text(name)

        if normalized.endswith("."):
            normalized = normalized[:-1]
        return normalized.lower()

    @staticmethod
    def _coerce_name_text(name: Any) -> str:
        if isinstance(name, str):
            normalized = name.strip()
            if len(normalized) >= 2 and normalized[0] in {"'", '"'} and normalized[-1] == normalized[0]:
                normalized = normalized[1:-1]
        else:
            normalized = str(name).strip()

        return normalized

    @staticmethod
    def _requires_numeric_format_dot(raw_text: str, normalized: str) -> bool:
        if "." in raw_text:
            return False
        return any(
            pattern.fullmatch(normalized) is not None
            for pattern in (_WD_FORMAT, _COMMA_WD_FORMAT)
        )

    def _format_z(self, value: Any, *, width: int, decimals: int) -> str:
        rendered = self._format_fixed(value, decimals=decimals)
        sign = "-" if rendered.startswith("-") else ""
        magnitude = rendered[1:] if sign else rendered
        return sign + magnitude.zfill(max(width - len(sign), 0))

    @staticmethod
    def _format_wd(value: Any, *, width: int, decimals: int) -> str:
        del width
        return FormatRegistry._format_fixed(value, decimals=decimals)

    @staticmethod
    def _format_comma_wd(value: Any, *, width: int, decimals: int) -> str:
        del width
        numeric_value = float(value)
        return f"{numeric_value:,.{decimals}f}"

    @staticmethod
    def _format_fixed(value: Any, *, decimals: int) -> str:
        numeric_value = float(value)
        return f"{numeric_value:.{decimals}f}"

    @staticmethod
    def _format_best(value: Any) -> str:
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return str(numeric_value)

    @staticmethod
    def _parse_best(value: Any) -> float:
        text = str(value).strip()
        return float(text)

    @staticmethod
    def _coerce_int(value: Any, *, family: str) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ValueError(f"{family} expects an integer-compatible value: {value}")
        return int(str(value).strip())

    @staticmethod
    def _coerce_date(value: Any) -> dt.date:
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, dt.date):
            return value
        return dt.date.fromisoformat(str(value).strip())

    @staticmethod
    def _coerce_datetime(value: Any) -> dt.datetime:
        if isinstance(value, dt.datetime):
            return value.replace(microsecond=0)
        if isinstance(value, dt.date):
            return dt.datetime.combine(value, dt.time())
        return dt.datetime.fromisoformat(str(value).strip()).replace(microsecond=0)

    @staticmethod
    def _coerce_time(value: Any) -> dt.time:
        if isinstance(value, dt.datetime):
            return value.time().replace(microsecond=0)
        if isinstance(value, dt.time):
            return value.replace(microsecond=0)
        return dt.time.fromisoformat(str(value).strip()).replace(microsecond=0)

    @staticmethod
    def _parse_yymmdd(value: Any, *, width: int) -> dt.date:
        text = str(value).strip()
        if width == 6:
            return dt.datetime.strptime(text, "%y%m%d").date()
        if width == 8:
            return dt.datetime.strptime(text, "%Y%m%d").date()
        if width == 10:
            return dt.datetime.strptime(text, "%Y-%m-%d").date()
        raise ValueError(f"Unsupported yymmdd width: {width}")