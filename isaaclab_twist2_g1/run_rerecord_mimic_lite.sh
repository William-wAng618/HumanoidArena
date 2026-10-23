#!/usr/bin/env bash
# Batch replay MimicLite recordings and re-record them as multicam episodes.
#
# Usage:
#   MIMIC_LITE_SOURCE_ROOT=/path/to/recording_data/HSI_boxing/mimic_lite/kk0908 \
#     bash isaaclab_twist2_g1/run_rerecord_mimic_lite.sh
#
#   # Dry run (print commands only)
#   DRY_RUN=1 MIMIC_LITE_SOURCE_ROOT=... bash isaaclab_twist2_g1/run_rerecord_mimic_lite.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Must be exported BEFORE runtime_paths.sh: it derives ISAACLAB_PYTHON from
# ISAACLAB_CONDA_ENV_NAME and defaults to `unitree_sim_env` (Isaac Sim 5.0).
# The recordings and the sonic/twist2 rerecord baseline all run under Isaac Sim 5.1
# (`unitree_sim_env_isaaclab5_0`); replaying on 5.0 diverges and the robot falls.
export ISAACLAB_CONDA_ENV_NAME="${ISAACLAB_CONDA_ENV_NAME:-unitree_sim_env_isaaclab5_0}"
source "${SCRIPT_DIR}/script/common/runtime_paths.sh"
cd "${SCRIPT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-${ISAACLAB_PYTHON}}"

MIMIC_LITE_SOURCE_ROOT="${MIMIC_LITE_SOURCE_ROOT:-}"
MIMIC_LITE_PARALLEL_JOBS="${MIMIC_LITE_PARALLEL_JOBS:-1}"
MIMIC_LITE_IMAGE_PORT_BASE="${MIMIC_LITE_IMAGE_PORT_BASE:-6200}"
MIMIC_LITE_IMAGE_XROBOT_PORT_BASE="${MIMIC_LITE_IMAGE_XROBOT_PORT_BASE:-12800}"
MIMIC_LITE_SHM_PREFIX="${MIMIC_LITE_SHM_PREFIX:-isaac_multi_image_shm_mimic_lite}"
MIMIC_LITE_OUTPUT_SUFFIX="${MIMIC_LITE_OUTPUT_SUFFIX:-_multicam_rerecord}"
MIMIC_LITE_SEED="${MIMIC_LITE_SEED:-42}"
MIMIC_LITE_LIMIT="${MIMIC_LITE_LIMIT:-0}"
MIMIC_LITE_TIMEOUT_SECONDS="${MIMIC_LITE_TIMEOUT_SECONDS:-0}"
MIMIC_LITE_DISABLE_WRIST_CAMERAS="${MIMIC_LITE_DISABLE_WRIST_CAMERAS:-0}"
MIMIC_LITE_ENABLE_PERSPECTIVE_CAMERA="${MIMIC_LITE_ENABLE_PERSPECTIVE_CAMERA:-0}"
# Task runtime profile forwarded to sim_main.py.
#   auto          -> task/context default (vision-navi rerecord => replay_compat room)
#   inference     -> validation room; use this for vision-navi data recorded by
#                    run_mimic_lite.sh, whose live_inference context also selects it.
# Boxing has no profile rules, so the value is ignored there.
MIMIC_LITE_TASK_RUNTIME_PROFILE="${MIMIC_LITE_TASK_RUNTIME_PROFILE:-auto}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
  "${PYTHON_BIN}" -u
  "${SCRIPT_DIR}/tools/data_tools/rerecord_mimic_lite_recordings_to_multicam.py"
)

if [ -n "${MIMIC_LITE_SOURCE_ROOT}" ]; then
  cmd+=("${MIMIC_LITE_SOURCE_ROOT}")
fi

cmd+=(
  --parallel-jobs "${MIMIC_LITE_PARALLEL_JOBS}"
  --image-port-base "${MIMIC_LITE_IMAGE_PORT_BASE}"
  --image-port-stride 10
  --image-xrobot-port-base "${MIMIC_LITE_IMAGE_XROBOT_PORT_BASE}"
  --image-xrobot-port-stride 10
  --shm-prefix "${MIMIC_LITE_SHM_PREFIX}"
  --output-suffix "${MIMIC_LITE_OUTPUT_SUFFIX}"
  --seed "${MIMIC_LITE_SEED}"
  --limit "${MIMIC_LITE_LIMIT}"
  --timeout-seconds "${MIMIC_LITE_TIMEOUT_SECONDS}"
  --task-runtime-profile "${MIMIC_LITE_TASK_RUNTIME_PROFILE}"
)

if [ "${MIMIC_LITE_DISABLE_WRIST_CAMERAS}" = "1" ]; then
  cmd+=(--disable-wrist-cameras)
fi
if [ "${MIMIC_LITE_ENABLE_PERSPECTIVE_CAMERA}" = "1" ]; then
  cmd+=(--enable-perspective-camera)
fi
if [ "${DRY_RUN}" = "1" ]; then
  cmd+=(--dry-run)
fi

echo "[rerecord_mimic_lite] python=${PYTHON_BIN}"
echo "[rerecord_mimic_lite] source_root=${MIMIC_LITE_SOURCE_ROOT:-<script default>}"
echo "[rerecord_mimic_lite] workers=${MIMIC_LITE_PARALLEL_JOBS} port_base=${MIMIC_LITE_IMAGE_PORT_BASE} shm_prefix=${MIMIC_LITE_SHM_PREFIX}"
echo "[rerecord_mimic_lite] seed=${MIMIC_LITE_SEED} limit=${MIMIC_LITE_LIMIT} suffix=${MIMIC_LITE_OUTPUT_SUFFIX}"
echo "[rerecord_mimic_lite] task_runtime_profile=${MIMIC_LITE_TASK_RUNTIME_PROFILE}"
if [ "${MIMIC_LITE_PARALLEL_JOBS}" != "1" ]; then
  echo "[rerecord_mimic_lite] WARNING: parallel-jobs>1 is unsafe until sim_main.py stops using" >&2
  echo "  'pgrep -f sim_main.py' in its exit cleanup: each worker SIGTERMs its siblings, which" >&2
  echo "  truncates their in-flight episode saves (observed 6/20 lost at --parallel-jobs 2)." >&2
fi

exec "${cmd[@]}"
