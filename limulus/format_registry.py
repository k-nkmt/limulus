from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Mapping
from typing import Any


_Z_FORMAT = re.compile(r"^z(\d+)(?:\.(\d+))?$", re.IGNORECASE)
_WD_FORMAT = re.compile(r"^(\d+)(?:\.(\d+))?$")
_COMMA_WD_FORMAT = re.compile(r"^comma(\d+)(?:\.(\d+))?$", re.IGNORECASE)
_YYMMDD_FORMAT = re.compile(r"^yymmdd(6|8|10)$", re.IGNORECASE)
_NUMERIC_NAMESPACE = "numeric"
_CHARACTER_NAMESPACE = "character"


class FormatRegistry:
    def __init__(self) -> None:
        self._formatters: dict[tuple[str, str], Callable[[Any], Any]] = {}
        self._format_catalogs: dict[tuple[str, str], dict[Any, str | None]] = {}
        self._informats: dict[str, Callable[[Any], Any]] = {}
        self._informat_catalogs: dict[str, dict[Any, float | None]] = {}
        self._informat_kinds: dict[str, str] = {}

    def register_format(
        self,
        name: str,
        formatter: Callable[[Any], Any] | Mapping[Any, Any],
        *,
        namespace: str | None = None,
    ) -> None:
        normalized_name, normalized_namespace = self._normalize_format_spec(name, namespace=namespace)
        catalog_key = (normalized_namespace, normalized_name)
        if callable(formatter):
            self._formatters[catalog_key] = formatter
            return
        if not isinstance(formatter, Mapping):
            raise TypeError("register_format() expects a callable formatter or mapping catalog")
        self._format_catalogs[catalog_key] = {
            key: self._coerce_put_output(value)
            for key, value in formatter.items()
        }

    def register_informat(self, name: str, parser: Callable[[Any], Any] | Mapping[Any, Any], *, kind: str | None = None) -> None:
        normalized = self.normalize_name(name)
        if callable(parser):
            self._informats[normalized] = parser
            if kind is not None:
                self._informat_kinds[normalized] = kind.strip().lower()
            return
        if not isinstance(parser, Mapping):
            raise TypeError("register_informat() expects a callable parser or mapping catalog")
        self._informat_catalogs[normalized] = {
            key: self._coerce_informat_output(value)
            for key, value in parser.items()
        }
        self._informat_kinds[normalized] = "float64"

    def has_custom_registrations(self) -> bool:
        return bool(self._formatters or self._format_catalogs or self._informats or self._informat_catalogs)

    def has_callable_registrations(self) -> bool:
        return bool(self._formatters or self._informats)

    def has_dict_catalogs(self) -> bool:
        return bool(self._format_catalogs or self._informat_catalogs)

    def dispatch_hints(self) -> dict[str, bool]:
        return {
            "has_custom_format_registry": self.has_custom_registrations(),
            "has_callable_format_registry": self.has_callable_registrations(),
            "has_dict_format_catalogs": self.has_dict_catalogs(),
        }

    def catalog_payload(self) -> dict[str, Any]:
        if not self.has_dict_catalogs():
            return {}
        numeric_formats: dict[str, dict[Any, str | None]] = {}
        character_formats: dict[str, dict[Any, str | None]] = {}
        for (namespace, name), catalog in self._format_catalogs.items():
            if namespace == _CHARACTER_NAMESPACE:
                character_formats[name] = dict(catalog)
            else:
                numeric_formats[name] = dict(catalog)
        return {
            "formats": {
                _NUMERIC_NAMESPACE: numeric_formats,
                _CHARACTER_NAMESPACE: character_formats,
            },
            "informats": {
                name: dict(catalog)
                for name, catalog in self._informat_catalogs.items()
            },
        }

    def get_format_catalog(self, format_name: Any) -> dict[Any, str | None] | None:
        normalized_name, normalized_namespace = self._normalize_format_spec(format_name)
        return self._format_catalogs.get((normalized_namespace, normalized_name))

    def get_informat_catalog(self, informat_name: Any) -> dict[Any, float | None] | None:
        return self._informat_catalogs.get(self.normalize_name(informat_name))

    def put(self, value: Any, format_name: Any) -> Any:
        if value is None:
            return None

        raw_text = self._coerce_name_text(format_name)
        normalized, namespace = self._normalize_format_spec(raw_text)
        formatter = self._formatters.get((namespace, normalized))
        if formatter is not None:
            return formatter(value)

        catalog = self._format_catalogs.get((namespace, normalized))
        if catalog is not None:
            matched = catalog.get(value)
            if matched is not None:
                return matched
            return self._coerce_put_output(value)

        if namespace != _NUMERIC_NAMESPACE:
            raise ValueError(f"Unsupported format: {format_name}")

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

        yymmdd_match = _YYMMDD_FORMAT.fullmatch(normalized)
        if yymmdd_match is not None:
            return self._format_yymmdd(self._coerce_date(value), width=int(yymmdd_match.group(1)))

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

        catalog = self._informat_catalogs.get(normalized)
        if catalog is not None:
            return catalog.get(value)

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
        if normalized in self._informat_catalogs:
            return "float64"
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
        if normalized.startswith("$"):
            normalized = normalized[1:]

        if normalized.endswith("."):
            normalized = normalized[:-1]
        return normalized.lower()

    @staticmethod
    def _normalize_format_spec(name: Any, *, namespace: str | None = None) -> tuple[str, str]:
        raw_text = FormatRegistry._coerce_name_text(name)
        inferred_namespace = _CHARACTER_NAMESPACE if raw_text.startswith("$") else _NUMERIC_NAMESPACE
        normalized_name = raw_text[1:] if raw_text.startswith("$") else raw_text
        if normalized_name.endswith("."):
            normalized_name = normalized_name[:-1]
        return normalized_name.lower(), FormatRegistry._normalize_namespace(namespace or inferred_namespace)

    @staticmethod
    def _normalize_namespace(namespace: Any) -> str:
        normalized = str(namespace).strip().lower()
        if normalized in {_NUMERIC_NAMESPACE, "num", "numeric_format"}:
            return _NUMERIC_NAMESPACE
        if normalized in {_CHARACTER_NAMESPACE, "char", "$", "character_format"}:
            return _CHARACTER_NAMESPACE
        raise ValueError(f"Unsupported format namespace: {namespace}")

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
    def _coerce_put_output(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, dt.datetime):
            return value.replace(microsecond=0).isoformat(timespec="seconds")
        if isinstance(value, dt.date):
            return value.isoformat()
        if isinstance(value, dt.time):
            return value.replace(microsecond=0).isoformat(timespec="seconds")
        return str(value)

    @staticmethod
    def _coerce_informat_output(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

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

    @staticmethod
    def _format_yymmdd(value: dt.date, *, width: int) -> str:
        if width == 6:
            return value.strftime("%y%m%d")
        if width == 8:
            return value.strftime("%Y%m%d")
        if width == 10:
            return value.strftime("%Y-%m-%d")
        raise ValueError(f"Unsupported yymmdd width: {width}")
