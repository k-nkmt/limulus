#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python="$repo_root/.venv/bin/python"

if [ ! -x "$python" ]; then
    echo "Cargo tests require the repository uv environment at .venv" >&2
    exit 1
fi

python_home=$("$python" -c 'import sys; print(sys.base_prefix)')
site_packages=$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')

export PYTHONHOME="$python_home"
export PYTHONPATH="$site_packages${PYTHONPATH:+:$PYTHONPATH}"

exec "$@"
