from __future__ import annotations

from importlib import import_module
from typing import Any


def load_native_module() -> tuple[Any | None, str | None]:
    try:
        module = import_module("limulus_native")
        render_fn = getattr(module, "render_diagnostics_ariadne", None)
        execute_fn = getattr(module, "execute_block", None)
        parse_fn = getattr(module, "parse_subset", None)
        if callable(render_fn) or callable(execute_fn) or callable(parse_fn):
            return module, None

        native_submodule = import_module("limulus_native.limulus_native")
        return native_submodule, None
    except Exception as error:
        return None, str(error)
