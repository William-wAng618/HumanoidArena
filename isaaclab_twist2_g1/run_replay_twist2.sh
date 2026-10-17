#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/script/common/runtime_paths.sh"
cd "${SCRIPT_DIR}" || exit 1

# ------------------------------------------------------------------
# User config: edit here
# ------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-${ISAACLAB_PYTHON}}"
REPLAY_FILE="${REPLAY_FILE:-}"
REPLAY_MODE="direct"   # inference | direct
REPLAY_LOOP=0             # 1 | 0
TASK_NAME=""              # 留空则从 replay 文件读取
# ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/opendoor_twist2.yaml}"
# ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/livingroom_sitsofa_twist2.yaml}"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-}"
# ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/boxing_bag_twist2.yaml}"
RUN_DEVICE="cpu"
ROBOT_TYPE="g129"
ROBOT_COLLIDER_MODE="box" # box | fourpoints
ENABLE_CAMERAS=1
ENABLE_WRIST_CAMERAS=0
ENABLE_DEX3_DDS=1
HEADLESS=0
MODEL_PATH="${MODEL_PATH:-${TWIST2_ROOT}/assets/ckpts/twist2_1017_20k.onnx}"
IMAGE_TRANSPORT="zmq"
IMAGE_ZMQ_PORT="5555"
LEFT_WRIST_CAMERA_PORT="5557"
RIGHT_WRIST_CAMERA_PORT="5558"
IMAGE_FPS="30"
STATS_INTERVAL="10.0"
STEP_HZ="50"
VIDEO_FPS="30"
SEED="42"

export PROJECT_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

if [ -z "${REPLAY_FILE}" ]; then
  echo "Usage: REPLAY_FILE=/path/to/episode.npz bash ${0}"
  echo "Error: REPLAY_FILE is empty"
  exit 1
fi
if [ ! -f "${REPLAY_FILE}" ]; then
  echo "Error: replay file not found: ${REPLAY_FILE}"
  exit 1
fi
if [ "${REPLAY_MODE}" != "inference" ] && [ "${REPLAY_MODE}" != "direct" ]; then
  echo "Error: replay mode must be inference or direct, got: ${REPLAY_MODE}"
  exit 1
fi

REPLAY_FILE="$(realpath "${REPLAY_FILE}")"
if [ -z "${TASK_NAME}" ]; then
  TASK_NAME="$("${PYTHON_BIN}" - "${REPLAY_FILE}" <<'PY'
import sys
import numpy as np

replay_file = sys.argv[1]
with np.load(replay_file, allow_pickle=True) as data:
    task = data.get("task")
    if task is None:
        raise SystemExit("Replay file missing 'task' metadata")
    if hasattr(task, "item"):
        task = task.item()
    print(task)
PY
)"
fi

resolve_twist2_env_config_yaml() {
  case "$1" in
    Isaac-Move-Open-Door-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/opendoor_twist2.yaml"
      ;;
    Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/pickplace_box_twist2.yaml"
      ;;
    Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/doubledesk_twist2.yaml"
      ;;
    Isaac-Move-Football-Single-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/football_single_twist2.yaml"
      ;;
    Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/livingroom_sitsofa_twist2.yaml"
      ;;
    Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/boxing_bag_twist2.yaml"
      ;;
    Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/small_warehouse_vision_navigation_twist2.yaml"
      ;;
    *)
      return 1
      ;;
  esac
}

if [ -z "${ENV_CONFIG_YAML}" ]; then
  if ! ENV_CONFIG_YAML="$(resolve_twist2_env_config_yaml "${TASK_NAME}")"; then
    echo "Error: cannot infer TWIST2 env config for task: ${TASK_NAME}"
    echo "Set ENV_CONFIG_YAML explicitly."
    exit 1
  fi
fi

if [ "${ROBOT_COLLIDER_MODE}" = "fourpoints" ]; then
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
else
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
fi

echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"
echo "[twist2 replay] file=${REPLAY_FILE}"
echo "[twist2 replay] mode=${REPLAY_MODE}"
echo "[twist2 replay] task=${TASK_NAME}"
echo "[twist2 replay] env_config=${ENV_CONFIG_YAML}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sim_main.py"
  --device "${RUN_DEVICE}"
  --task "${TASK_NAME}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --robot_type "${ROBOT_TYPE}"
  --input_source replay
  --gmt_backend twist2
  --replay_file "${REPLAY_FILE}"
  --replay_mode "${REPLAY_MODE}"
  --model_path "${MODEL_PATH}"
  --image_transport "${IMAGE_TRANSPORT}"
  --image_fps "${IMAGE_FPS}"
  --image_zmq_port "${IMAGE_ZMQ_PORT}"
  --stats_interval "${STATS_INTERVAL}"
  --step_hz "${STEP_HZ}"
  --video_fps "${VIDEO_FPS}"
  --seed "${SEED}"
)

if [ "${ENABLE_CAMERAS}" = "1" ]; then
  cmd+=(--enable_cameras)
fi
if [ "${ENABLE_WRIST_CAMERAS}" = "1" ]; then
  cmd+=(--enable_wrist_cameras)
  cmd+=(--left_wrist_camera_port "${LEFT_WRIST_CAMERA_PORT}")
  cmd+=(--right_wrist_camera_port "${RIGHT_WRIST_CAMERA_PORT}")
fi
if [ "${ENABLE_DEX3_DDS}" = "1" ]; then
  cmd+=(--enable_dex3_dds)
fi
if [ "${HEADLESS}" = "1" ]; then
  cmd+=(--headless)
fi
if [ "${REPLAY_LOOP}" = "1" ]; then
  cmd+=(--replay_loop)
fi

exec "${cmd[@]}"
