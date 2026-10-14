#!/usr/bin/env bash
# Isolated raw logs -> CSVs -> validation -> figures; each run has an immutable source snapshot.
# Usage: PY=/path/to/venv/bin/python bash release/run_all.sh full --output-dir /path/to/new-run
# Resume the same captured source/commands with: ... full --output-dir /path/to/run --resume
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
exec "$PY" "$ROOT/paper/scripts/reproduce.py" "$@"
