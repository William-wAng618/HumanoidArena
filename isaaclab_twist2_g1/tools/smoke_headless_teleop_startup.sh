#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/script/common/runtime_paths.sh"
cd "${SCRIPT_DIR}" || exit 1

export PROJECT_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/smoke_headless_teleop_startup/${RUN_ID}}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-240}"
SMOKE_STEPS="${SMOKE_STEPS:-1}"
RUN_DEVICE="${RUN_DEVICE:-cpu}"
BACKENDS="${BACKENDS:-twist2 sonic}"
TASKS="${TASKS:-boxing doubledesk football_single open_door pp_box sit_sofa vision_navi}"
PYTHON_BIN="${PYTHON_BIN:-${ISAACLAB_PYTHON}}"

mkdir -p "${LOG_ROOT}"

runner_py="${LOG_ROOT}/_smoke_env_startup_runner.py"
cat > "${runner_py}" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import traceback
import functools
from types import SimpleNamespace

project_root = os.environ.get("PROJECT_ROOT")
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

from isaaclab.app import AppLauncher
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser(description="HumanoidArena headless env startup smoke runner")
parser.add_argument("--task", required=True)
parser.add_argument("--env_config_yaml", required=True)
parser.add_argument("--backend", required=True, choices=["twist2", "sonic"])
parser.add_argument("--smoke_steps", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Keep this smoke focused on local path/assets/env creation. Runtime profiles are
# still applied because open-door and vision-navigation choose different assets
# for live inference vs replay/rerecord contracts.
profile_args = SimpleNamespace(
    task=args.task,
    task_runtime_profile="auto",
    input_source=f"pico_{args.backend}",
    gmt_backend=args.backend,
    replay_file="",
    replay_mode="",
    record_during_replay=False,
)

simulation_app = None
env = None
try:
    from task_runtime_profiles import apply_task_runtime_profile

    apply_task_runtime_profile(profile_args)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import gymnasium as gym
    # Import only the seven maintained task modules used by this smoke test.
    # This avoids unrelated legacy task packages from blocking deployment checks.
    import importlib
    for module_name in (
        "tasks.g1_tasks.move_boxing_bag_g1_29dof_dex3_wholebody",
        "tasks.g1_tasks.move_pickplace_doubledesk_g1_29dof_dex3_wholebody",
        "tasks.g1_tasks.move_football_single_g1_29dof_dex3_wholebody",
        "tasks.g1_tasks.move_open_door_g1_29dof_dex3_wholebody",
        "tasks.g1_tasks.move_pickplace_box_g1_29dof_dex3_wholebody",
        "tasks.g1_tasks.move_sit_sofa_g1_29dof_dex3_wholebody",
        "tasks.g1_tasks.move_small_warehouse_vision_navigation_g1_29dof_dex3_wholebody",
    ):
        importlib.import_module(module_name)
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tasks.common_env_config import apply_env_config_yaml

    print(f"[SMOKE] task={args.task}")
    print(f"[SMOKE] backend={args.backend}")
    print(f"[SMOKE] env_config_yaml={args.env_config_yaml}")
    print(f"[SMOKE] device={args.device}")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.env_name = args.task
    resolved_yaml = apply_env_config_yaml(
        env_cfg,
        args.env_config_yaml,
        task_name=args.task,
        route_name=args.backend,
    )
    env_cfg.seed = 0
    print(f"[SMOKE] resolved_yaml={resolved_yaml}")

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    print("[SMOKE] create_environment=ok")
    print(f"[SMOKE] robot_usd={env.cfg.scene.robot.spawn.usd_path}")

    env.sim.reset()
    env.reset()
    print("[SMOKE] reset=ok")

    for step_idx in range(max(0, int(args.smoke_steps))):
        shape = getattr(env.action_space, "shape", None)
        if not shape:
            print("[SMOKE] action_space has no shape; skip env.step")
            break
        if len(shape) == 1:
            action_shape = (getattr(env, "num_envs", 1), shape[0])
        else:
            action_shape = shape
        action = torch.zeros(action_shape, device=env.device, dtype=torch.float32)
        env.step(action)
        print(f"[SMOKE] step={step_idx + 1}=ok")

    print("[SMOKE] PASS")
except Exception as exc:  # noqa: BLE001 - print full failure for smoke logs
    print(f"[SMOKE] FAIL: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    raise
finally:
    if env is not None:
        try:
            env.close()
            print("[SMOKE] env_closed=ok")
        except Exception as close_exc:  # noqa: BLE001
            print(f"[SMOKE] env_close_failed={close_exc}")
    if simulation_app is not None:
        try:
            simulation_app.close()
            print("[SMOKE] simulation_app_closed=ok")
        except Exception as close_exc:  # noqa: BLE001
            print(f"[SMOKE] simulation_app_close_failed={close_exc}")
PY

get_env_config() {
  local task="$1"
  local backend="$2"
  case "${task}:${backend}" in
    boxing:twist2) echo "tasks/common_env_config/boxing_bag_twist2.yaml" ;;
    boxing:sonic) echo "tasks/common_env_config/boxing_bag_sonic.yaml" ;;
    doubledesk:twist2) echo "tasks/common_env_config/doubledesk_twist2.yaml" ;;
    doubledesk:sonic) echo "tasks/common_env_config/doubledesk_sonic.yaml" ;;
    football_single:twist2) echo "tasks/common_env_config/football_single_twist2.yaml" ;;
    football_single:sonic) echo "tasks/common_env_config/football_single_sonic.yaml" ;;
    open_door:twist2) echo "tasks/common_env_config/opendoor_twist2.yaml" ;;
    open_door:sonic) echo "tasks/common_env_config/opendoor_sonic.yaml" ;;
    pp_box:twist2) echo "tasks/common_env_config/pickplace_box_twist2.yaml" ;;
    pp_box:sonic) echo "tasks/common_env_config/pickplace_box_sonic.yaml" ;;
    sit_sofa:twist2) echo "tasks/common_env_config/livingroom_sitsofa_twist2.yaml" ;;
    sit_sofa:sonic) echo "tasks/common_env_config/livingroom_sitsofa_sonic.yaml" ;;
    vision_navi:twist2) echo "tasks/common_env_config/small_warehouse_vision_navigation_twist2.yaml" ;;
    vision_navi:sonic) echo "tasks/common_env_config/small_warehouse_vision_navigation_sonic.yaml" ;;
    *) return 1 ;;
  esac
}

get_task_name() {
  "${PYTHON_BIN}" - "${SCRIPT_DIR}/tasks/common_env_config/loader.py" "$1" <<'PY'
import importlib.util
import pathlib
import sys
loader_path = pathlib.Path(sys.argv[1])
config_path = sys.argv[2]
spec = importlib.util.spec_from_file_location("common_env_config_loader", loader_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(module.get_env_config_task_name(config_path))
PY
}

summary_tsv="${LOG_ROOT}/summary.tsv"
echo -e "task\tbackend\tstatus\texit_code\tlog" > "${summary_tsv}"

echo "[smoke] log_root=${LOG_ROOT}"
echo "[smoke] python=${PYTHON_BIN}"
echo "[smoke] tasks=${TASKS}"
echo "[smoke] backends=${BACKENDS}"
echo "[smoke] timeout=${CASE_TIMEOUT_SEC}s smoke_steps=${SMOKE_STEPS} device=${RUN_DEVICE}"

for task in ${TASKS}; do
  for backend in ${BACKENDS}; do
    env_config="$(get_env_config "${task}" "${backend}")"
    task_name="$(get_task_name "${env_config}")"
    log_file="${LOG_ROOT}/${task}_${backend}.log"
    echo "[smoke] START task=${task} backend=${backend} env_config=${env_config}"
    set +e
    timeout "${CASE_TIMEOUT_SEC}" "${PYTHON_BIN}" "${runner_py}" \
      --headless \
      --enable_cameras \
      --device "${RUN_DEVICE}" \
      --task "${task_name}" \
      --env_config_yaml "${env_config}" \
      --backend "${backend}" \
      --smoke_steps "${SMOKE_STEPS}" \
      >"${log_file}" 2>&1
    exit_code=$?
    set -e
    if [[ "${exit_code}" -eq 0 ]] && ! grep -q "Traceback\|\[SMOKE\] FAIL" "${log_file}"; then
      status="PASS"
    elif [[ "${exit_code}" -eq 124 ]]; then
      status="TIMEOUT"
    else
      status="FAIL"
    fi
    echo -e "${task}\t${backend}\t${status}\t${exit_code}\t${log_file}" >> "${summary_tsv}"
    echo "[smoke] ${status} task=${task} backend=${backend} exit=${exit_code} log=${log_file}"
  done
done

echo "[smoke] summary=${summary_tsv}"
column -t -s $'\t' "${summary_tsv}" || cat "${summary_tsv}"
