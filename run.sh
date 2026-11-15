#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$root"
python_bin="${MJLAB_PYTHON:-$HOME/dev-envs/robotics-mjlab-py313/bin/python}"
case "${1:-}" in
  hop-moon|jump-moon)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_jump.py "$@"
    ;;
  stand-moon|stand-v15)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --lunar-v15 "$@"
    ;;
  stand-contact|stand-v11)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --contact-v11 "$@"
    ;;
  stand-lqr|stand-lqr-flat|stand-v7)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --lqr-flat "$@"
    ;;
  stand-v6)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --lqr-flat-v6 "$@"
    ;;
  stand-lqr-terrain|stand-v5)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --lqr "$@"
    ;;
  --show-model)
    shift
    export MUJOCO_GL=glfw
    exec "$python_bin" scripts/view_model.py "$@"
    ;;
  --physics)
    shift
    if [[ $# -eq 0 || "${1:-}" == "--simulate" || "${1:-}" == "--skin" || "${1:-}" == "--proxy" ]]; then
      export MUJOCO_GL=glfw
      exec "$python_bin" scripts/view_physics.py "$@"
    fi
    exec "$python_bin" launch.py play Wangyu-Moon-Jump-Physics-v1 "$@"
    ;;
  --jump)
    shift
    exec "$python_bin" launch.py play Wangyu-Moon-Jump-v0 "$@"
    ;;
  stand-flat)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --flat "$@"
    ;;
  stand-terrain-v2)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --terrain-v2 "$@"
    ;;
  stand-v3)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --balance-v3 "$@"
    ;;
  stand-v4)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --balance-v4 "$@"
    ;;
  stand-v8)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --push-v8 "$@"
    ;;
  stand-v9)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --push-v9 "$@"
    ;;
  stand-v10)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --support-v10 "$@"
    ;;
  stand-v12)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --flat-v12 "$@"
    ;;
  stand-v13)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --flat-v13 "$@"
    ;;
  stand-plane|stand-v14)
    shift
    export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
    exec "$python_bin" scripts/play_stand.py --flat-v14 "$@"
    ;;
  stand|stand-terrain)
    shift
    ;;
esac
export MUJOCO_GL="${MUJOCO_GL_STAND:-glfw}"
exec "$python_bin" scripts/play_stand.py "$@"
