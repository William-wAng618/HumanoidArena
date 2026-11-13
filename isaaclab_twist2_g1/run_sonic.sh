#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISAACLAB_CONDA_ENV_NAME="${ISAACLAB_CONDA_ENV_NAME:-unitree_sim_env}"
source "${SCRIPT_DIR}/script/common/runtime_paths.sh"
cd "${SCRIPT_DIR}" || exit 1

# ------------------------------------------------------------------
# User config: edit here
# ------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-${ISAACLAB_PYTHON}}"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/boxing_bag_sonic.yaml}"
RUN_DEVICE="cpu"
ROBOT_TYPE="g129"
ROBOT_COLLIDER_MODE="box"   # box | DD
SEED="42"
ENABLE_CAMERAS=1
ENABLE_DEX3_DDS=1
HEADLESS="${HEADLESS:-0}"
SONIC_REDIS_HOST="localhost"
SONIC_REDIS_PORT="6379"
SONIC_ENCODER_PATH="${SONIC_ENCODER_PATH:-${SONIC_POLICY_ROOT}/model_encoder.onnx}"
SONIC_DECODER_PATH="${SONIC_DECODER_PATH:-${SONIC_POLICY_ROOT}/model_decoder.onnx}"
IMAGE_TRANSPORT="xrobot"
IMAGE_XROBOT_HOST="${IMAGE_XROBOT_HOST:-10.42.0.35}"
IMAGE_XROBOT_PORT="12345"
IMAGE_XROBOT_BITRATE="2097152"
IMAGE_FPS="30"
IMAGE_XROBOT_FFMPEG="/usr/bin/ffmpeg"
RECORDING_SAVE_DIR="${RECORDING_SAVE_DIR:-${SCRIPT_DIR}/recording_data/HSI_boxing/sonic/yb}"

export PROJECT_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

load_task_name_from_yaml() {
  "${PYTHON_BIN}" - "${SCRIPT_DIR}/tasks/common_env_config/loader.py" "${1}" <<'PY'
import importlib.util
import pathlib
import sys

loader_path = pathlib.Path(sys.argv[1])
config_path = sys.argv[2]
spec = importlib.util.spec_from_file_location("common_env_config_loader", loader_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

task_name = module.get_env_config_task_name(config_path)
if not task_name:
    raise SystemExit(
        f"Error: env config YAML must define a top-level 'task_name': {config_path}"
    )
print(task_name)
PY
}

TASK_NAME="${TASK_NAME:-$(load_task_name_from_yaml "${ENV_CONFIG_YAML}")}"

if [ ! -f "${SONIC_ENCODER_PATH}" ]; then
  echo "Error: SONIC encoder not found: ${SONIC_ENCODER_PATH}"
  exit 1
fi
if [ ! -f "${SONIC_DECODER_PATH}" ]; then
  echo "Error: SONIC decoder not found: ${SONIC_DECODER_PATH}"
  exit 1
fi

redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  human_smplx_data_unitree_g1_with_hands \
  recording_control_unitree_g1_with_hands \
  isaac_reset_trigger \
  isaac_reset_complete_unitree_g1_with_hands \
  isaac_input_ready_sonic_unitree_g1_with_hands \
  controller_data \
  t_action >/dev/null 2>&1 || true

if [ "${ROBOT_COLLIDER_MODE}" = "fourpoints" ]; then
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
else
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
fi

echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"
echo "[sonic] task=${TASK_NAME}"
echo "[sonic] env_config=${ENV_CONFIG_YAML}"
echo "[sonic] seed=${SEED}"
echo "[sonic] encoder=${SONIC_ENCODER_PATH}"
echo "[sonic] decoder=${SONIC_DECODER_PATH}"
echo "[recording] save_dir=${RECORDING_SAVE_DIR}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sim_main.py"
  --device "${RUN_DEVICE}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --task "${TASK_NAME}"
  --robot_type "${ROBOT_TYPE}"
  --input_source pico_sonic
  --gmt_backend sonic
  --sonic_pose_source redis
  --sonic_redis_host "${SONIC_REDIS_HOST}"
  --sonic_redis_port "${SONIC_REDIS_PORT}"
  --sonic_encoder_path "${SONIC_ENCODER_PATH}"
  --sonic_decoder_path "${SONIC_DECODER_PATH}"
  --image_transport "${IMAGE_TRANSPORT}"
  --image_xrobot_host "${IMAGE_XROBOT_HOST}"
  --image_xrobot_port "${IMAGE_XROBOT_PORT}"
  --image_xrobot_bitrate "${IMAGE_XROBOT_BITRATE}"
  --image_fps "${IMAGE_FPS}"
  --image_xrobot_ffmpeg "${IMAGE_XROBOT_FFMPEG}"
  --recording_save_dir "${RECORDING_SAVE_DIR}"
  --seed "${SEED}"
)

if [ "${ENABLE_CAMERAS}" = "1" ]; then
  cmd+=(--enable_cameras)
fi
if [ "${ENABLE_DEX3_DDS}" = "1" ]; then
  cmd+=(--enable_dex3_dds)
fi
if [ "${HEADLESS}" = "1" ]; then
  cmd+=(--headless)
fi

if [ "$#" -gt 0 ]; then
  cmd+=("$@")
fi

exec "${cmd[@]}"

# 可切换：
#一般沙袋: tasks/common_env_config/boxing_bag_sonic.yaml
#吊挂沙袋: tasks/common_env_config/boxing_bag_hanging_sonic.yaml
#足球: tasks/common_env_config/football_sonic.yaml
#单足球: tasks/common_env_config/football_single_sonic.yaml
#双桌面拾放: tasks/common_env_config/doubledesk_sonic.yaml
#Push-T: tasks/common_env_config/push_t_sonic.yaml
#三级台阶平台：tasks/common_env_config/three_step_platform_sonic.yaml
#开门：tasks/common_env_config/opendoor_sonic.yaml
#小推车：tasks/common_env_config/pickplace_small_trolley_sonic.yaml
