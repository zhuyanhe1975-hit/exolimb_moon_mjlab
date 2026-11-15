# Source from interactive Bash; never starts the training process.
_wangyu_train_complete() {
  local cur prev command_path project_root
  cur=${COMP_WORDS[COMP_CWORD]}
  prev=${COMP_WORDS[COMP_CWORD-1]}
  project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  command_path="$(realpath -- "${COMP_WORDS[0]}" 2>/dev/null)" || return
  [[ "$command_path" == "$project_root/train.sh" ]] || return
  COMPREPLY=()
  case "$prev" in
    --resume-from) compopt -o filenames; mapfile -t COMPREPLY < <(compgen -f -- "$cur"); return ;;
    --push-force|--push-torque|--agent.max-iterations|--env.scene.num-envs|--agent.load-run|--agent.load-checkpoint) return ;;
    --agent.resume) mapfile -t COMPREPLY < <(compgen -W 'True False' -- "$cur"); return ;;
  esac
  local choices='--help --pushes --no-pushes --push-force --push-torque --resume-from --terrain-transfer --agent.max-iterations --env.scene.num-envs --agent.resume --agent.load-run --agent.load-checkpoint'
  if (( COMP_CWORD == 1 )); then
    choices="stand stand-moon stand-terrain stand-plane stand-contact stand-lqr stand-lqr-flat stand-lqr-terrain stand-flat stand-terrain-v2 stand-v3 stand-v4 stand-v5 stand-v6 stand-v7 stand-v8 stand-v9 stand-v10 stand-v11 stand-v12 stand-v13 stand-v14 stand-v15 --physics --jump $choices"
  fi
  mapfile -t COMPREPLY < <(compgen -W "$choices" -- "$cur")
}
complete -F _wangyu_train_complete ./train.sh train.sh /home/yhzhu/myWorks/exolimb_moon_mjlab/train.sh
