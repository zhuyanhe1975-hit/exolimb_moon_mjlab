#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lab="${ISAACLAB_DIR:-$HOME/nvidia/isaac-lab-3.0.0}"
mjpython="${MJLAB_PYTHON:-$HOME/dev-envs/robotics-mjlab-py313/bin/python}"
export ZBOT_MOON_DIR="${ZBOT_MOON_DIR:-$HOME/myWorks/zbot_exoarm_moon}"
export OMNILRS_DIR="${OMNILRS_DIR:-$HOME/Github/OmniLRS}"
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h ]]; then
    exec "$mjpython" "$root/scripts/preview_isaacsim.py" --help
  fi
done
[[ -x "$mjpython" ]] || { echo "Missing MJLab Python: $mjpython" >&2; exit 1; }
[[ -x "$lab/.venv/bin/python" ]] || { echo "Missing existing Isaac Lab environment: $lab/.venv/bin/python" >&2; exit 1; }
command -v uv >/dev/null || { echo 'uv is required (existing Isaac Lab environment; no installation attempted)' >&2; exit 1; }
for suffix in diff rough nor_gl ao; do
  file="$ZBOT_MOON_DIR/assets/lunar_sites/southpole_site20/LunarRegolith4k/LunarRegolith4k_$suffix.jpg"
  [[ -f "$file" ]] || { echo "Missing cached scene texture: $file" >&2; exit 1; }
done
echo "[Isaac runtime] $lab (uv --no-sync --offline)"
echo "[Scene assets] $ZBOT_MOON_DIR"
echo "[OmniLRS] $OMNILRS_DIR"
echo '[Mode] static scene preview; no policy, training or physics stepping'
if [[ "${1:-}" == --dry-run ]]; then exit 0; fi
data="$root/outputs/isaacsim_preview/terrain.npz"
"$mjpython" "$root/scripts/export_isaacsim_scene.py" --output "$data"
cd "$root"
exec env -u PYTHONPATH uv run --project "$lab" --no-sync --offline python \
  "$root/scripts/preview_isaacsim.py" --terrain "$data" "$@"
