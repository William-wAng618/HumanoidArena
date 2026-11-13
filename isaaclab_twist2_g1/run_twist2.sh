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
# ENV_CONFIG_YAML="tasks/common_env_config/small_warehouse_vision_navigation_twist2.yaml"
# ENV_CONFIG_YAML="tasks/common_env_config/pickplace_box_twist2.yaml"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/opendoor_twist2.yaml}"
# ENV_CONFIG_YAML="tasks/common_env_config/livingroom_sitsofa_twist2.yaml"
# ENV_CONFIG_YAML="tasks/common_env_config/boxing_bag_twist2.yaml"
# ENV_CONFIG_YAML="tasks/common_env_config/football_single_twist2.yaml"
# ENV_CONFIG_YAML="tasks/common_env_config/doubledesk_twist2.yaml"
RUN_DEVICE="cpu"
ROBOT_TYPE="g129"
ROBOT_COLLIDER_MODE="box"   # box | fourpoints
SEED="42"
ENABLE_CAMERAS=1
ENABLE_DEX3_DDS=1
HEADLESS="${HEADLESS:-0}"
IMAGE_TRANSPORT="xrobot"
IMAGE_XROBOT_HOST="${IMAGE_XROBOT_HOST:-10.42.0.35}" # eth
# IMAGE_XROBOT_HOST="192.168.101.174" #｜ wifi
IMAGE_XROBOT_PORT="12345"
IMAGE_XROBOT_BITRATE="16777216"
IMAGE_FPS="30"
IMAGE_XROBOT_FFMPEG="/usr/bin/ffmpeg"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_vision_target/twist2/yb"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_pp_box/twist2/yb"
#RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_pp_box/twist2/zk"
RECORDING_SAVE_DIR="${RECORDING_SAVE_DIR:-${SCRIPT_DIR}/recording_data/HSI_open_door/twist2/zz}"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_sit_sofa/twist2/zz"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_sitingsofa/twist2/zk"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_boxing/twist2/zz"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_vision_target/twist2/yb"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_double_desk/twist2/tw_supply"
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

redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  controller_data \
  t_action \
  isaac_reset_trigger \
  isaac_input_ready_twist2_unitree_g1_with_hands >/dev/null 2>&1 || true

if [ "${ROBOT_COLLIDER_MODE}" = "fourpoints" ]; then
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
else
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
fi

echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"
echo "[twist2] task=${TASK_NAME}"
echo "[twist2] env_config=${ENV_CONFIG_YAML}"
echo "[twist2] recording_save_dir=${RECORDING_SAVE_DIR}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sim_main.py"
  --device "${RUN_DEVICE}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --input_source pico_twist2
  --gmt_backend twist2
  --task "${TASK_NAME}"
  --robot_type "${ROBOT_TYPE}"
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
#一般沙袋: tasks/common_env_config/boxing_bag_twist2.yaml
#吊挂沙袋: tasks/common_env_config/boxing_bag_hanging_twist2.yaml
#足球: tasks/common_env_config/football_twist2.yaml
#单足球: tasks/common_env_config/football_single_twist2.yaml
#双桌面拾放: tasks/common_env_config/doubledesk_twist2.yaml
#Push-T: tasks/common_env_config/push_t_twist2.yaml
#三级台阶平台：tasks/common_env_config/three_step_platform_twist2.yaml
#开门：tasks/common_env_config/opendoor_twist2.yaml
#小推车：tasks/common_env_config/pickplace_small_trolley_twist2.yaml
# 箱子：tasks/common_env_config/pickplace_box_twist2.yaml
