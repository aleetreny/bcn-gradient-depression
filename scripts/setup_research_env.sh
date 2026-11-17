#!/usr/bin/env bash
# Build the research environment (Python 3.11, uv) and verify it.
#
#   bash scripts/setup_research_env.sh
#
# macOS note: the LightGBM wheel links @rpath/libomp.dylib and only looks in Homebrew/MacPorts
# prefixes. Without Homebrew, we point it at the libomp.dylib that the scikit-learn wheel already
# ships, so a single OpenMP runtime is loaded per process (two copies can deadlock or crash).
# With Homebrew, `brew install libomp` works too and this step is skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv311"
PY="$VENV/bin/python"

[ -x "$PY" ] || uv venv --python 3.11 "$VENV"
uv pip install --python "$PY" -r "$ROOT/requirements-research-lock.txt"

if [ "$(uname)" = "Darwin" ] && [ ! -e /opt/homebrew/opt/libomp/lib/libomp.dylib ]; then
    SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    LIB="$SITE/lightgbm/lib/lib_lightgbm.dylib"
    RPATH="@loader_path/../../sklearn/.dylibs"
    [ -e "$SITE/sklearn/.dylibs/libomp.dylib" ] || { echo "sklearn libomp.dylib not found" >&2; exit 1; }
    if ! otool -l "$LIB" | grep -q "$RPATH"; then
        install_name_tool -add_rpath "$RPATH" "$LIB"
        codesign --force --sign - "$LIB"   # editing the binary invalidates its signature on arm64
        echo "Patched LightGBM to use scikit-learn's libomp.dylib"
    fi
fi

"$PY" "$ROOT/scripts/check_env.py"
