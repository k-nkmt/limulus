"""Build and smoke-test exactly one wheel from the current source tree.

The output directory is intentionally retained so a successful artifact can be
inspected after the command completes.  This script owns only the wheel build
and installed-wheel smoke boundaries; the other local validation commands stay
visible to the developer-facing gate.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SELECTION_FAILURE_STATUS = 2


def _run_boundary(label: str, command: Sequence[str]) -> int:
    """Run one child command after displaying its boundary and exact argv."""

    print(f"\n=== {label} ===", flush=True)
    print(shlex.join(command), flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        print(
            f"[validate_wheel] {label} failed with exit status {completed.returncode}",
            file=sys.stderr,
            flush=True,
        )
    return completed.returncode


def _create_fresh_output_directory() -> Path:
    """Create and retain a wheel output directory outside the repository."""

    output_directory = Path(tempfile.mkdtemp(prefix="limulus-wheel-")).resolve()
    if output_directory == REPOSITORY_ROOT or REPOSITORY_ROOT in output_directory.parents:
        raise RuntimeError(
            f"wheel output directory must be outside the repository: {output_directory}"
        )
    return output_directory


def _select_single_wheel(output_directory: Path) -> Path | None:
    """Return the only top-level regular wheel, or report a selection failure."""

    candidates = sorted(output_directory.glob("*.whl"))
    if len(candidates) != 1 or not candidates[0].is_file() or candidates[0].is_symlink():
        found = ", ".join(str(path) for path in candidates) or "none"
        print(
            "[validate_wheel] expected exactly one top-level regular wheel; "
            f"found {len(candidates)} ({found})",
            file=sys.stderr,
            flush=True,
        )
        return None
    return candidates[0].resolve()


def main() -> int:
    output_directory = _create_fresh_output_directory()
    print(
        f"[validate_wheel] retaining fresh wheel artifact directory: {output_directory}",
        flush=True,
    )

    build_command = [
        "uv",
        "run",
        "--no-sync",
        "maturin",
        "build",
        "--release",
        "--out",
        str(output_directory),
    ]
    build_status = _run_boundary("wheel build", build_command)
    if build_status != 0:
        return build_status

    wheel_path = _select_single_wheel(output_directory)
    if wheel_path is None:
        return SELECTION_FAILURE_STATUS
    print(f"[validate_wheel] selected wheel: {wheel_path}", flush=True)

    smoke_command = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        str(wheel_path),
        "python",
        "tests/wheel_smoke.py",
    ]
    smoke_status = _run_boundary("isolated wheel smoke", smoke_command)
    if smoke_status != 0:
        return smoke_status

    print(
        f"[validate_wheel] wheel smoke passed; artifact directory retained: {output_directory}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
