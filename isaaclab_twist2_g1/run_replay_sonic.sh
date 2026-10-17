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
REPLAY_MODE="${REPLAY_MODE:-direct_replay}"   # inference_replay | direct_replay
REPLAY_LOOP="${REPLAY_LOOP:-0}"               # 1 | 0
TASK_NAME="${TASK_NAME:-}"       # 留空则从 replay 文件读取
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-}"

RUN_DEVICE="${RUN_DEVICE:-cpu}"
ROBOT_TYPE="${ROBOT_TYPE:-g129}"
ROBOT_COLLIDER_MODE="${ROBOT_COLLIDER_MODE:-box}"        # box | fourpoints
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
ENABLE_WRIST_CAMERAS="${ENABLE_WRIST_CAMERAS:-0}"
ENABLE_DEX3_DDS="${ENABLE_DEX3_DDS:-1}"
HEADLESS="${HEADLESS:-0}"
EXIT_WHEN_REPLAY_COMPLETE="${EXIT_WHEN_REPLAY_COMPLETE:-0}"
SONIC_ENCODER_PATH="${SONIC_ENCODER_PATH:-${SONIC_POLICY_ROOT}/model_encoder.onnx}"
SONIC_DECODER_PATH="${SONIC_DECODER_PATH:-${SONIC_POLICY_ROOT}/model_decoder.onnx}"
IMAGE_TRANSPORT="${IMAGE_TRANSPORT:-zmq}"
IMAGE_ZMQ_PORT="${IMAGE_ZMQ_PORT:-5555}"
LEFT_WRIST_CAMERA_PORT="${LEFT_WRIST_CAMERA_PORT:-5557}"
RIGHT_WRIST_CAMERA_PORT="${RIGHT_WRIST_CAMERA_PORT:-5558}"
IMAGE_FPS="${IMAGE_FPS:-30}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/sonic_replay}"
SEED="${SEED:-42}"

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
if [ "${REPLAY_MODE}" != "direct_replay" ] && [ "${REPLAY_MODE}" != "inference_replay" ]; then
  echo "Error: replay mode must be direct_replay or inference_replay, got: ${REPLAY_MODE}"
  exit 1
fi
if [ ! -f "${SONIC_ENCODER_PATH}" ]; then
  echo "Error: SONIC encoder not found: ${SONIC_ENCODER_PATH}"
  exit 1
fi
if [ ! -f "${SONIC_DECODER_PATH}" ]; then
  echo "Error: SONIC decoder not found: ${SONIC_DECODER_PATH}"
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import onnxruntime" >/dev/null 2>&1; then
  echo "Error: usable python with onnxruntime not found"
  echo "Set PYTHON_BIN or ISAACLAB_PYTHON to an environment with onnxruntime."
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

resolve_sonic_env_config_yaml() {
  case "$1" in
    Isaac-Move-Open-Door-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/opendoor_sonic.yaml"
      ;;
    Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/pickplace_box_sonic.yaml"
      ;;
    Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/doubledesk_sonic.yaml"
      ;;
    Isaac-Move-Football-Single-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/football_single_sonic.yaml"
      ;;
    Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/livingroom_sitsofa_sonic.yaml"
      ;;
    Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/boxing_bag_sonic.yaml"
      ;;
    Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody)
      printf '%s\n' "tasks/common_env_config/small_warehouse_vision_navigation_sonic.yaml"
      ;;
    *)
      return 1
      ;;
  esac
}

if [ -z "${ENV_CONFIG_YAML}" ]; then
  if ! ENV_CONFIG_YAML="$(resolve_sonic_env_config_yaml "${TASK_NAME}")"; then
    echo "Error: cannot infer SONIC env config for task: ${TASK_NAME}"
    echo "Set ENV_CONFIG_YAML explicitly."
    exit 1
  fi
fi

export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"


redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  isaac_input_ready_sonic_unitree_g1_with_hands \
  controller_data \
  t_action >/dev/null 2>&1 || true

mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SIM_LOG="${LOG_DIR}/sim_main_${RUN_TS}.log"

echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"
echo "[sonic replay] file=${REPLAY_FILE}"
echo "[sonic replay] mode=${REPLAY_MODE}"
echo "[sonic replay] task=${TASK_NAME}"
echo "[sonic replay] seed=${SEED}"
echo "[sonic replay] encoder=${SONIC_ENCODER_PATH}"
echo "[sonic replay] decoder=${SONIC_DECODER_PATH}"
echo "[sonic replay] python=${PYTHON_BIN}"
echo "[sonic replay] log=${SIM_LOG}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sim_main.py"
  --device "${RUN_DEVICE}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --task "${TASK_NAME}"
  --robot_type "${ROBOT_TYPE}"
  --input_source replay
  --gmt_backend sonic
  --sonic_encoder_path "${SONIC_ENCODER_PATH}"
  --sonic_decoder_path "${SONIC_DECODER_PATH}"
  --replay_file "${REPLAY_FILE}"
  --replay_mode "${REPLAY_MODE}"
  --image_transport "${IMAGE_TRANSPORT}"
  --image_fps "${IMAGE_FPS}"
  --image_zmq_port "${IMAGE_ZMQ_PORT}"
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
if [ "${EXIT_WHEN_REPLAY_COMPLETE}" = "1" ]; then
  cmd+=(--exit_when_replay_complete)
fi

{
  echo "[$(date '+%F %T')] Starting SONIC replay"
  echo "[$(date '+%F %T')] file=${REPLAY_FILE}"
  echo "[$(date '+%F %T')] mode=${REPLAY_MODE}"
  echo "[$(date '+%F %T')] task=${TASK_NAME}"
  echo "[$(date '+%F %T')] python=${PYTHON_BIN}"
} | tee -a "${SIM_LOG}"

"${cmd[@]}" 2>&1 | tee -a "${SIM_LOG}"
