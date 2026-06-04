from __future__ import annotations

from importlib import machinery, util
from importlib import import_module
from pathlib import Path
import sys
from typing import Any


def _validate_native_module(module: Any) -> str | None:
    execute_fn = getattr(module, "execute_block", None)
    render_fn = getattr(module, "render_diagnostics_ariadne", None)
    parse_fn = getattr(module, "parse_subset", None)
    if not (callable(execute_fn) or callable(render_fn) or callable(parse_fn)):
        return "limulus native module does not expose required callable entry points"
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _expected_native_runtime_sources(repo_root: Path | None = None) -> tuple[Path, ...]:
    resolved_repo_root = repo_root or _repo_root()
    native_src_root = resolved_repo_root / "native" / "limulus_native" / "src"
    split_modules = [
        "ast.rs",
        "diagnostics.rs",
        "expressions/mod.rs",
        "expressions/functions.rs",
        "runtime/mod.rs",
        "runtime/state.rs",
        "output/mod.rs",
        "output/accumulator.rs",
        "row_cursor.rs",
        "value_ref.rs",
    ]
    staged_modules = ("source_view.rs",)
    for module_name in staged_modules:
        if (native_src_root / module_name).exists():
            split_modules.append(module_name)
    return (native_src_root / "lib.rs", *(native_src_root / name for name in split_modules))


def _newest_native_source(repo_root: Path | None = None) -> Path | None:
    resolved_repo_root = repo_root or _repo_root()
    native_root = resolved_repo_root / "native" / "limulus_native"
    src_root = native_root / "src"
    cargo_toml = native_root / "Cargo.toml"
    if not src_root.exists() or not cargo_toml.exists():
        return None

    expected_sources = _expected_native_runtime_sources(resolved_repo_root)
    if any(not path.exists() for path in expected_sources):
        return None

    return max((cargo_toml, *expected_sources), key=lambda path: path.stat().st_mtime)


def _workspace_native_artifact_candidates(repo_root: Path | None = None) -> tuple[Path, ...]:
    resolved_repo_root = repo_root or _repo_root()
    target_root = resolved_repo_root / "native" / "limulus_native" / "target"
    return (
        target_root / "maturin" / "liblimulus_native.dylib",
        target_root / "release" / "liblimulus_native.dylib",
        target_root / "debug" / "liblimulus_native.dylib",
    )


def _load_extension_module_from_path(module_name: str, extension_path: Path) -> tuple[Any | None, str | None]:
    loader = machinery.ExtensionFileLoader(module_name, str(extension_path))
    spec = util.spec_from_file_location(module_name, extension_path, loader=loader)
    if spec is None or spec.loader is None:
        return None, f"could not create import spec for native extension at {extension_path}"

    previous_module = sys.modules.get(module_name)
    try:
        module = util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, None
    except Exception as error:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        return None, str(error)


def _load_workspace_native_module() -> tuple[Any | None, str | None]:
    repo_root = _repo_root()
    newest_source = _newest_native_source(repo_root)
    if newest_source is None:
        return None, None

    load_error: str | None = None
    required_mtime = newest_source.stat().st_mtime
    for candidate in _workspace_native_artifact_candidates(repo_root):
        if not candidate.exists():
            continue
        if candidate.stat().st_mtime < required_mtime:
            continue
        module, load_error = _load_extension_module_from_path("limulus_native.limulus_native", candidate)
        if module is None:
            continue
        validation_error = _validate_native_module(module)
        if validation_error is None:
            return module, None
        load_error = validation_error

    return None, load_error


def _detect_native_source_drift(module: Any) -> str | None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        return None

    extension_path = Path(module_file)
    if not extension_path.exists():
        return None

    repo_root = _repo_root()
    native_root = repo_root / "native" / "limulus_native"
    src_root = native_root / "src"
    cargo_toml = native_root / "Cargo.toml"
    if not src_root.exists() or not cargo_toml.exists():
        return None

    expected_sources = _expected_native_runtime_sources(repo_root)
    missing_sources = [path for path in expected_sources if not path.exists()]
    if missing_sources:
        missing_labels = ", ".join(path.name for path in missing_sources)
        return (
            "limulus native source tree is missing expected split runtime modules "
            f"({missing_labels}); rebuild or restore the native runtime module group before running rust/auto workloads"
        )

    newest_source = _newest_native_source(repo_root)
    if newest_source is None:
        return None
    extension_mtime = extension_path.stat().st_mtime
    if newest_source.stat().st_mtime <= extension_mtime:
        return None

    relative_source = newest_source.relative_to(repo_root)
    return (
        "limulus native extension is older than workspace Rust sources "
        f"({relative_source}); rebuild with "
        "`uv run --no-sync maturin develop --release` before running rust/auto workloads"
    )


def load_native_module() -> tuple[Any | None, str | None]:
    try:
        # Prefer compiled extension submodule explicitly to avoid stale wrapper behavior.
        native_submodule = import_module("limulus_native.limulus_native")
        validation_error = _validate_native_module(native_submodule)
        if validation_error is None:
            validation_error = _detect_native_source_drift(native_submodule)
        if validation_error is None:
            return native_submodule, None
        workspace_module, workspace_error = _load_workspace_native_module()
        if workspace_module is not None:
            return workspace_module, None

        module = import_module("limulus_native")
        validation_error = _validate_native_module(module)
        if validation_error is None:
            validation_error = _detect_native_source_drift(module)
        if validation_error is None:
            return module, None
        if workspace_error is not None:
            return None, workspace_error
        return None, validation_error
    except Exception as error:
        workspace_module, workspace_error = _load_workspace_native_module()
        if workspace_module is not None:
            return workspace_module, None
        return None, workspace_error or str(error)
