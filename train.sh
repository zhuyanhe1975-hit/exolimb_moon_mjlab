#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$root"
task=Wangyu-Moon-Stand-Lunar-v15
case "${1:-}" in
  hop-moon|jump-moon) task=Wangyu-Moon-Hop-Lunar-v2; shift ;;
  stand-moon|stand-v15) task=Wangyu-Moon-Stand-Lunar-v15; shift ;;
  stand-contact|stand-v11) task=Wangyu-Moon-Stand-Support-Contact-v11; shift ;;
  stand-lqr|stand-lqr-flat|stand-v7) task=Wangyu-Moon-Stand-LQR-Flat-v7; shift ;;
  stand-v6) task=Wangyu-Moon-Stand-LQR-Flat-v6; shift ;;
  stand-lqr-terrain|stand-v5) task=Wangyu-Moon-Stand-LQR-v5; shift ;;
  stand|stand-terrain) shift ;;
  stand-plane|stand-v14) task=Wangyu-Moon-Stand-Flat-v14; shift ;;
  stand-v13) task=Wangyu-Moon-Stand-Flat-v13; shift ;;
  stand-v12) task=Wangyu-Moon-Stand-Support-Flat-v12; shift ;;
  stand-v10) task=Wangyu-Moon-Stand-Support-v10; shift ;;
  stand-v9) task=Wangyu-Moon-Stand-Push-v9; shift ;;
  stand-v8) task=Wangyu-Moon-Stand-Push-v8; shift ;;
  stand-v4) task=Wangyu-Moon-Stand-Balance-v4; shift ;;
  stand-v3) task=Wangyu-Moon-Stand-Balance-v3; shift ;;
  stand-terrain-v2) task=Wangyu-Moon-Stand-Terrain-v2; shift ;;
  stand-flat) task=Wangyu-Moon-Stand-v1; shift ;;
  --physics) task=Wangyu-Moon-Jump-Physics-v1; shift ;;
  --jump) task=Wangyu-Moon-Jump-v0; shift ;;
esac
if [[ "$task" == Wangyu-Moon-Stand* || "$task" == Wangyu-Moon-Hop* ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
fi
prepare_lqr=true
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h ]]; then prepare_lqr=false; fi
done
if [[ "$task" == Wangyu-Moon-Stand-LQR* && "$prepare_lqr" == true ]]; then
  export WANGYU_LQR_FILE="${WANGYU_LQR_FILE:-$root/outputs/lqr/stand_lqr_v1.npz}"
  if [[ ! -f "$WANGYU_LQR_FILE" ]]; then
    "${MJLAB_PYTHON:-$HOME/dev-envs/robotics-mjlab-py313/bin/python}" scripts/prepare_stand_lqr.py --output "$WANGYU_LQR_FILE"
  fi
fi
exec "${MJLAB_PYTHON:-$HOME/dev-envs/robotics-mjlab-py313/bin/python}" launch.py train "$task" "$@"
