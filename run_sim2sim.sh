#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mjpython="${MJLAB_PYTHON:-$HOME/dev-envs/robotics-mjlab-py313/bin/python}"
lab="${ISAACLAB_DIR:-$HOME/nvidia/isaac-lab-3.0.0}"
prepare_only=false
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h ]]; then
    exec "$mjpython" "$root/scripts/sim2sim_isaacsim.py" --help
  fi
  if [[ "$arg" == --prepare-only ]]; then prepare_only=true; fi
done
[[ -x "$mjpython" ]] || { echo "Missing existing MJLab Python: $mjpython" >&2; exit 1; }
if ! "$prepare_only"; then
  [[ -x "$lab/.venv/bin/python" ]] || { echo "Missing existing Isaac Lab environment: $lab/.venv/bin/python" >&2; exit 1; }
  command -v uv >/dev/null || { echo 'uv is required; no dependencies will be installed' >&2; exit 1; }
fi
mkdir -p "$root/outputs/sim2sim_isaacsim"
bundle="$(mktemp -d "$root/outputs/sim2sim_isaacsim/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
cd "$root"
"$mjpython" "$root/scripts/prepare_sim2sim.py" --bundle "$bundle" "$@"
if "$prepare_only"; then exit 0; fi
echo "[Isaac runtime] $lab (existing environment, uv --no-sync --offline)"
exec env -u PYTHONPATH uv run --project "$lab" --no-sync --offline python \
  "$root/scripts/sim2sim_isaacsim.py" --bundle "$bundle" "$@"
