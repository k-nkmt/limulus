from __future__ import annotations

from importlib import import_module
from typing import Any


def load_native_module() -> tuple[Any | None, str | None]:
    try:
        module = import_module("limulus_native")
        return module, None
    except Exception as error:
        return None, str(error)
