from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from action_provider.action_base import ActionProvider, ReplayComplete
from action_provider.mimic_lite_runtime import MIMIC_LITE_JOINT_ORDER, MimicLitePolicyRuntime
from action_provider.recording_common import AsyncEpisodeRecorder
from action_provider.recording_control import (
    get_recording_control_identity,
    should_accept_recording_control,
)
from action_provider.reset_control import (
    GMR_BODY_POS_KEY,
    GMR_BODY_QUAT_W_KEY,
    GMR_FULL_QPOS_KEY,
    GMR_JOINT_POS_KEY,
    GMR_JOINT_VEL_KEY,
    MIMIC_LITE_INPUT_READY_KEY,
    consume_reset_complete,
    get_input_ready_key,
    publish_reset_command,
)
from action_provider.vision_video import write_rgb_video_mp4
from common_env_objects import (
    add_env_object_frame_arrays,
    add_episode_init_env_object_fields,
    collect_recordable_env_object_states,
    get_current_episode_object_seed_info,
    resolve_env_object_scene_key,
)
from tools.get_reward import get_step_reward_value

try:
    import redis
except ImportError:
    redis = None

# VLA support (aligned with SonicActionProvider VLA pipeline)
from collections import deque
from action_provider.lerobot_vla_http_client import LeRobotVLAHttpClient
from action_provider.vla_robot_current_local_runtime_v3 import (
    UnifiedRobotCurrentLocalActionRuntimeV3,
    build_sonic_joint29_payload_v3,
    build_vla_rotlocal_v3_observation_state,
    VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM as SONIC_VLA_ACTION_DIM,
    VLA_ROBOT_CURRENT_LOCAL_V3_STATE_DIM as SONIC_VLA_STATE_DIM,
)
from action_provider.vla_smpl_runtime import (
    quat_from_roll_pitch_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_to_rot6d_wxyz,
    rot6d_to_quat_wxyz_with_layout,
    yaw_from_quat_wxyz,
)
from pico_server.data_utils.params import DEFAULT_HAND_POSE
from action_provider.action_provider_sonic import SONIC_HAND_POSE_ROBOT_NAME


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_MIMIC_LITE_ONNX = REPO_ROOT / "isaaclab_twist2_g1" / "assets" / "checkpoints" / "mimic_lite" / "policy-xua2csee-4000.onnx"
DEFAULT_MIMIC_LITE_YAML = REPO_ROOT / "isaaclab_twist2_g1" / "assets" / "checkpoints" / "mimic_lite" / "policy-xua2csee-4000.yaml"


def _decode_json_array(raw_value, dtype=np.float32):
    if raw_value is None:
        return None
    payload = raw_value.decode("utf-8") if isinstance(raw_value, bytes) else raw_value
    return np.asarray(json.loads(payload), dtype=dtype)


def _json_string(obj):
    return json.dumps(obj, ensure_ascii=False)


def _reward_scalar(env) -> float:
    try:
        reward = get_step_reward_value(env)
        if isinstance(reward, torch.Tensor):
            if reward.numel() == 0:
                return 0.0
            return float(reward.detach().reshape(-1)[0].item())
        return float(reward)
    except Exception:
        return 0.0


def _reward_success_flag(reward: float, tol: float = 1e-6) -> bool:
    try:
        return float(reward) > float(tol)
    except Exception:
        return False


def _store_camera_stream(
    organized: dict[str, Any],
    *,
    save_dir: str | None,
    task_name: str,
    timestamp_us: int,
    fps: float,
    frames: list[np.ndarray],
    depth_frames: list[np.ndarray],
    frame_indices: list[int],
    indices_key: str,
    video_path_key: str,
    video_fps_key: str,
    video_num_frames_key: str,
    depth_key: str,
    raw_rgb_key: str,
    video_suffix: str,
) -> bool:
    if not frames:
        return False
    organized[indices_key] = np.array(frame_indices, dtype=np.int32)
    organized[video_fps_key] = np.array(fps, dtype=np.float32)
    if save_dir:
        basename = f"{task_name}_{timestamp_us}_{video_suffix}.mp4"
        video_relpath = os.path.join("videos", basename)
        video_abspath = os.path.join(save_dir, video_relpath)
        written = write_rgb_video_mp4(frames, video_abspath, fps=fps)
        organized[video_path_key] = np.array(video_relpath)
        organized[video_num_frames_key] = np.array(written, dtype=np.int32)
    else:
        organized[raw_rgb_key] = np.array(frames, dtype=np.uint8)
    if depth_frames:
        organized[depth_key] = np.asarray(depth_frames, dtype=np.float16)
    return True


def _organize_mimic_lite_episode(
    data_buffer: list[dict[str, Any]],
    timestamp_us: int,
    save_dir: str | None = None,
    task_name: str | None = None,
) -> dict[str, Any]:
    if not data_buffer:
        raise ValueError("empty mimic_lite recording buffer")

    first = data_buffer[0]
    meta = first["meta"]
    num_frames = len(data_buffer)

    def _stack(path: tuple[str, ...], dtype=None):
        values = []
        for frame in data_buffer:
            current = frame
            for key in path:
                current = current[key]
            values.append(np.asarray(current))
        return np.stack(values).astype(dtype) if dtype is not None else np.stack(values)

    organized: dict[str, Any] = {
        "schema_version": np.array("mimic_lite_episode_v1"),
        "task": np.array(meta["task"]),
        "episode_id": np.array(meta["episode_id"], dtype=np.int64),
        "save_timestamp_us": np.array(timestamp_us, dtype=np.int64),
        "num_frames": np.array(num_frames, dtype=np.int32),
        "meta_control_dt": np.array(meta["control_dt"], dtype=np.float32),
        "meta_physics_dt": np.array(meta["physics_dt"], dtype=np.float32),
        "meta_decimation": np.array(meta["decimation"], dtype=np.int32),
        "meta_onnx_path": np.array(meta.get("onnx_path", "")),
        "meta_yaml_path": np.array(meta.get("yaml_path", "")),
        "episode_object_seed": np.array(meta.get("episode_object_seed") if meta.get("episode_object_seed") is not None else -1, dtype=np.int64),
        "episode_object_seed_source": np.array(meta.get("episode_object_seed_source") or ""),
        "frame_index": _stack(("markers", "frame_index"), np.int64),
        "episode_step": _stack(("markers", "episode_step"), np.int64),
        "timestamp_wall": _stack(("markers", "timestamp_wall"), np.float64),
        "recording_command": np.array([frame["markers"]["recording_command"] for frame in data_buffer]),
        "reset_requested": _stack(("markers", "reset_requested"), np.bool_),
        "reset_completed": _stack(("markers", "reset_completed"), np.bool_),
        "save_triggered": _stack(("markers", "save_triggered"), np.bool_),
        "human_recording_control_json": np.array(
            [_json_string(frame["human_raw"]["recording_control"]) for frame in data_buffer]
        ),
        "human_raw_body_quat_w": _stack(("human_raw", "body_quat_w"), np.float32),
        "human_raw_body_pos": _stack(("human_raw", "body_pos"), np.float32),
        "human_raw_joint_pos": _stack(("human_raw", "joint_pos"), np.float32),
        "human_raw_joint_vel": _stack(("human_raw", "joint_vel"), np.float32),
        "human_raw_full_qpos": _stack(("human_raw", "full_qpos"), np.float32),
        "human_raw_left_hand": _stack(("human_raw", "left_hand"), np.float32),
        "human_raw_right_hand": _stack(("human_raw", "right_hand"), np.float32),
        "human_raw_controller_data_json": np.array(
            [_json_string(frame["human_raw"]["controller_data"]) for frame in data_buffer]
        ),
        "mimic_command": _stack(("mimic_model_io", "command"), np.float32),
        "mimic_policy": _stack(("mimic_model_io", "policy"), np.float32),
        "mimic_policy_action": _stack(("mimic_model_io", "policy_action"), np.float32),
        "robot_qpos_before_decimation": _stack(("robot", "qpos_before_decimation"), np.float32),
        "robot_qvel_before_decimation": _stack(("robot", "qvel_before_decimation"), np.float32),
        "robot_root_position": _stack(("robot", "root_position"), np.float32),
        "robot_root_orientation": _stack(("robot", "root_orientation"), np.float32),
        "robot_root_lin_vel_local": _stack(("robot", "root_lin_vel_local"), np.float32),
        "robot_root_ang_vel_local": _stack(("robot", "root_ang_vel_local"), np.float32),
        "robot_root_lin_vel_world": _stack(("robot", "root_lin_vel_world"), np.float32),
        "robot_root_ang_vel_world": _stack(("robot", "root_ang_vel_world"), np.float32),
        "action_body_29dof": _stack(("action", "body_action_29dof"), np.float32),
        "action_full_action": _stack(("action", "full_action"), np.float32),
        "action_body_effort_target": _stack(("action", "body_effort_target"), np.float32),
        "action_hand_action_left": _stack(("action", "hand_action_left"), np.float32),
        "action_hand_action_right": _stack(("action", "hand_action_right"), np.float32),
        "env": {},
    }
    # Vision: extract per-frame camera data and encode RGB as MP4 video files.
    front_rgb_frames = []
    front_depth_frames = []
    front_vision_indices = []
    world_rgb_frames = []
    world_depth_frames = []
    world_vision_indices = []
    left_wrist_rgb_frames = []
    left_wrist_depth_frames = []
    left_wrist_indices = []
    right_wrist_rgb_frames = []
    right_wrist_depth_frames = []
    right_wrist_indices = []
    for idx, frame in enumerate(data_buffer):
        vision = frame["env"].get("vision") or {}
        rgb = vision.get("rgb")
        depth = vision.get("depth")
        if rgb is not None:
            front_rgb_frames.append(rgb)
            front_vision_indices.append(idx)
        if depth is not None:
            front_depth_frames.append(depth)
        world_rgb = vision.get("world_rgb")
        world_depth = vision.get("world_depth")
        if world_rgb is not None:
            world_rgb_frames.append(world_rgb)
            world_vision_indices.append(idx)
        if world_depth is not None:
            world_depth_frames.append(world_depth)
        left_rgb = vision.get("left_wrist_rgb")
        left_depth = vision.get("left_wrist_depth")
        if left_rgb is not None:
            left_wrist_rgb_frames.append(left_rgb)
            left_wrist_indices.append(idx)
        if left_depth is not None:
            left_wrist_depth_frames.append(left_depth)
        right_rgb = vision.get("right_wrist_rgb")
        right_depth = vision.get("right_wrist_depth")
        if right_rgb is not None:
            right_wrist_rgb_frames.append(right_rgb)
            right_wrist_indices.append(idx)
        if right_depth is not None:
            right_wrist_depth_frames.append(right_depth)
    if front_rgb_frames or world_rgb_frames or left_wrist_rgb_frames or right_wrist_rgb_frames:
        organized["vision_storage_format"] = np.array("video_v1")
        control_dt = float(meta.get("control_dt", 0.0) or 0.0)
        video_fps = float(1.0 / control_dt) if control_dt > 1e-6 else 30.0
        _store_camera_stream(
            organized,
            save_dir=save_dir,
            task_name=task_name or meta["task"],
            timestamp_us=timestamp_us,
            fps=video_fps,
            frames=front_rgb_frames,
            depth_frames=front_depth_frames,
            frame_indices=front_vision_indices,
            indices_key="vision_frame_indices",
            video_path_key="vision_rgb_video_path",
            video_fps_key="vision_rgb_video_fps",
            video_num_frames_key="vision_rgb_video_num_frames",
            depth_key="vision_depth",
            raw_rgb_key="vision_rgb",
            video_suffix="front_rgb",
        )
        if _store_camera_stream(
            organized,
            save_dir=save_dir,
            task_name=task_name or meta["task"],
            timestamp_us=timestamp_us,
            fps=video_fps,
            frames=world_rgb_frames,
            depth_frames=world_depth_frames,
            frame_indices=world_vision_indices,
            indices_key="vision_world_frame_indices",
            video_path_key="vision_world_rgb_video_path",
            video_fps_key="vision_world_rgb_video_fps",
            video_num_frames_key="vision_world_rgb_video_num_frames",
            depth_key="vision_world_depth",
            raw_rgb_key="vision_world_rgb",
            video_suffix="world_rgb",
        ):
            organized["schema_version"] = np.array("mimic_lite_episode_v2_multicam")
        if _store_camera_stream(
            organized,
            save_dir=save_dir,
            task_name=task_name or meta["task"],
            timestamp_us=timestamp_us,
            fps=video_fps,
            frames=left_wrist_rgb_frames,
            depth_frames=left_wrist_depth_frames,
            frame_indices=left_wrist_indices,
            indices_key="vision_left_wrist_frame_indices",
            video_path_key="vision_left_wrist_rgb_video_path",
            video_fps_key="vision_left_wrist_rgb_video_fps",
            video_num_frames_key="vision_left_wrist_rgb_video_num_frames",
            depth_key="vision_left_wrist_depth",
            raw_rgb_key="vision_left_wrist_rgb",
            video_suffix="left_wrist_rgb",
        ):
            organized["schema_version"] = np.array("mimic_lite_episode_v2_multicam")
        if _store_camera_stream(
            organized,
            save_dir=save_dir,
            task_name=task_name or meta["task"],
            timestamp_us=timestamp_us,
            fps=video_fps,
            frames=right_wrist_rgb_frames,
            depth_frames=right_wrist_depth_frames,
            frame_indices=right_wrist_indices,
            indices_key="vision_right_wrist_frame_indices",
            video_path_key="vision_right_wrist_rgb_video_path",
            video_fps_key="vision_right_wrist_rgb_video_fps",
            video_num_frames_key="vision_right_wrist_rgb_video_num_frames",
            depth_key="vision_right_wrist_depth",
            raw_rgb_key="vision_right_wrist_rgb",
            video_suffix="right_wrist_rgb",
        ):
            organized["schema_version"] = np.array("mimic_lite_episode_v2_multicam")
    add_env_object_frame_arrays(organized, data_buffer)
    add_episode_init_env_object_fields(organized, first.get("episode_init_env"))
    return organized


class MimicLiteActionProvider(ActionProvider):
    """MimicLite backend driven by existing GMR/SONIC joint29 Redis references."""

    def __init__(self, env, args_cli):
        super().__init__("MimicLiteActionProvider")
        self.env = env
        self.device = env.device
        self.redis_host = getattr(args_cli, "mimic_lite_redis_host", "localhost")
        self.redis_port = int(getattr(args_cli, "mimic_lite_redis_port", 6379))
        self.enable_dex3 = bool(getattr(args_cli, "enable_dex3_dds", False))
        self.enable_gripper = bool(getattr(args_cli, "enable_dex1_dds", False))
        self._debug = bool(int(os.environ.get("MIMIC_LITE_DEBUG", "0") or "0"))
        self._log_every = int(os.environ.get("MIMIC_LITE_LOG_EVERY", "100") or 100)
        self._startup_blend_sec = float(os.environ.get("MIMIC_LITE_STARTUP_BLEND_SEC", "0") or 0.0)
        self._frame_count = 0
        self.task_name = getattr(args_cli, "task", "mimic_lite")
        # --- VLA support (aligned with SonicActionProvider) ---
        self._input_source = getattr(args_cli, "input_source", "") or ""
        self._gmt_backend = getattr(args_cli, "gmt_backend", "") or ""
        self._use_lerobot_vla = self._input_source == "vla"
        self._enable_robot = str(getattr(args_cli, "robot_type", "") or "")
        self._lerobot_server_url = getattr(args_cli, "lerobot_server_url", "") or ""
        self._lerobot_server_timeout = float(getattr(args_cli, "lerobot_server_timeout", 5.0))
        self._lerobot_server_verify_ssl = bool(getattr(args_cli, "lerobot_server_verify_ssl", False))
        self._lerobot_policy = None
        self._lerobot_preprocessor = None
        self._lerobot_postprocessor = None
        self._lerobot_predict_action = None
        self._lerobot_device = None
        self._lerobot_http_client = None
        self._lerobot_vla_runtime = UnifiedRobotCurrentLocalActionRuntimeV3(
            root_rot6d_layout="auto",
            max_root_delta_deg=26.0,
        )
        self._vla_initial_robot_quat_wxyz = None
        # Yaw calibration for mimic_lite VLA data convention mismatch.
        # mimic_lite training data has state[0:6] and action[3:9]/[0:2] with a fixed
        # +yaw(robot_reset) offset (raw world frame) instead of heading-canonicalized (yaw=0).
        # This injects +calib to state and -calib to action to match the training distribution.
        # Default 0.0 = off (Convention A). Set to 90.0 for mimic_lite data trained VLA.
        self._vla_yaw_calib_deg = float(os.environ.get("VLA_MIMICLITE_YAW_CALIB_DEG", "0.0") or "0.0")
        self._vla_yaw_calib_rad = float(np.deg2rad(self._vla_yaw_calib_deg)) if abs(self._vla_yaw_calib_deg) > 1e-6 else 0.0
        self._vla_yaw_calib_quat_wxyz = (
            quat_from_roll_pitch_yaw_wxyz(0.0, 0.0, self._vla_yaw_calib_rad)
            if abs(self._vla_yaw_calib_deg) > 1e-6
            else None
        )
        self._vla_yaw_calib_quat_inv_wxyz = (
            quat_from_roll_pitch_yaw_wxyz(0.0, 0.0, -self._vla_yaw_calib_rad)
            if abs(self._vla_yaw_calib_deg) > 1e-6
            else None
        )
        self._lerobot_hand_binary_threshold = float(getattr(args_cli, "lerobot_gripper_threshold", 0.5))
        self._lerobot_action_chunk_queue = deque()
        self._latest_vla_action = None
        # Replay state early init (needed before VLA/replay mutual-exclusion check below).
        self._replay_file = getattr(args_cli, "replay_file", "") or ""
        self._replay_mode = self._normalize_replay_mode(getattr(args_cli, "replay_mode", "inference_replay"))
        self._replay_loop = bool(getattr(args_cli, "replay_loop", False))
        self._replay_enabled = bool(self._replay_file)
        self._record_during_replay = bool(getattr(args_cli, "record_during_replay", False))
        self._exit_when_replay_complete = bool(getattr(args_cli, "exit_when_replay_complete", False))
        if self._use_lerobot_vla and self._replay_enabled:
            raise ValueError("[MimicLiteActionProvider] input_source=vla and replay_file are mutually exclusive")

        onnx_path = getattr(args_cli, "mimic_lite_onnx_path", "") or str(DEFAULT_MIMIC_LITE_ONNX)
        yaml_path = getattr(args_cli, "mimic_lite_yaml_path", "") or str(DEFAULT_MIMIC_LITE_YAML)
        self._onnx_path = onnx_path
        self._yaml_path = yaml_path
        self.runtime = MimicLitePolicyRuntime(
            env=env,
            onnx_path=onnx_path,
            yaml_path=yaml_path,
            expected_joint_order=MIMIC_LITE_JOINT_ORDER,
        )
        self._setup_joint_mapping()
        if self._replay_enabled:
            self.dex3_dds = None
            self.gripper_dds = None
        else:
            self._setup_hand_interfaces()
        self._redis_client = None
        self._redis_control_client = None
        if not self._use_lerobot_vla and not self._replay_enabled:
            self._setup_redis()
        # Hand target buffers (aligned with SonicActionProvider, needed by VLA path).
        self._left_hand_target = np.zeros(7, dtype=np.float32)
        self._right_hand_target = np.zeros(7, dtype=np.float32)
        self._left_hand_binary_state = False
        self._right_hand_binary_state = False
        self._last_ref_joint_pos = self.runtime.default_joint_pos.copy()
        self._last_ref_joint_vel = np.zeros((29,), dtype=np.float32)
        robot = self.env.scene["robot"].data
        root_state = robot.root_state_w[0].detach().cpu().numpy().astype(np.float32)
        self._last_ref_body_pos = root_state[:3].copy()
        self._last_ref_body_quat = root_state[3:7].copy()
        self._startup_ref_joint_pos = self.runtime.default_joint_pos.copy()
        self._startup_ref_body_pos = self._last_ref_body_pos.copy()
        self._startup_ref_body_quat = self._last_ref_body_quat.copy()
        self._full_action_buf = torch.zeros(len(self.all_joint_names), dtype=torch.float32, device=self.device)
        self._use_self_torque = bool(int(os.environ.get("MIMIC_LITE_USE_SELF_TORQUE", "0") or "0"))
        self._torque_mode_configured = False
        self._mimic_kp_t = torch.tensor(self.runtime.joint_kp, dtype=torch.float32, device=self.device)
        self._mimic_kd_t = torch.tensor(self.runtime.joint_kd, dtype=torch.float32, device=self.device)
        self._mimic_effort_limit_t = self._resolve_effort_limits()
        if self._use_self_torque and not torch.all(torch.isfinite(self._mimic_effort_limit_t)):
            raise RuntimeError(
                "MIMIC_LITE_USE_SELF_TORQUE=1 requires finite Isaac joint effort limits; "
                "refusing to run with unlimited torque."
            )
        self._zero_body_vel_t = torch.zeros(len(MIMIC_LITE_JOINT_ORDER), dtype=torch.float32, device=self.device)
        self._last_body_effort = torch.zeros(len(MIMIC_LITE_JOINT_ORDER), dtype=torch.float32, device=self.device)
        self._decimation = int(getattr(getattr(env, "cfg", None), "decimation", 1) or 1)
        self._render_interval = max(1, int(os.environ.get("MIMIC_LITE_RENDER_INTERVAL", "1") or 1))
        self._prev_ref_debug_pos = None
        self._prev_ref_debug_joint = None
        self._last_tracking_log_frame = 0
        self._last_teleop_state = None

        # Recording / reset state machine (aligned with SONIC complete version).
        self._disable_eval_recording = False
        self._record_world_camera = bool(
            getattr(args_cli, "enable_world_camera", False)
            or getattr(args_cli, "enable_perspective_camera", False)
        )
        self._episode_init_env_state = self._collect_env_state()
        self.recording_manager = AsyncEpisodeRecorder(
            save_dir=getattr(args_cli, "recording_save_dir", "./recording_data"),
            task_name=f"{self.task_name}_mimic_lite",
            organize_fn=_organize_mimic_lite_episode,
            max_frames=10000,
            max_save_workers=int(getattr(args_cli, "recording_save_workers", 1)),
            max_queue_size=int(getattr(args_cli, "recording_save_queue_size", 10)),
        )
        self._should_start_recording_on_first_call = False  # set after replay state init
        self._recording_command = "none"
        self._recording_active = False
        self._recording_display_state = "idle"
        self._recording_display_counter = 0
        self._recording_display_duration = 10
        self._save_in_progress = False
        self._save_completion_state = None
        self._pending_save_jobs = 0
        self._waiting_for_reset_complete = False
        self._reset_complete_received = False
        self._input_ready_key = get_input_ready_key("mimic_lite")
        self._input_ready_epoch_id = -1
        self._input_ready_timestamp_realtime = 0.0
        self._input_ready_timestamp_monotonic = 0.0
        self._stale_input_drop_logged_epoch = -1
        self._episode_id = 0
        self._latest_recording_control = None
        self._latest_recording_control_sequence = -1
        self._latest_recording_control_identity = None
        self._raw_controller_data = None
        self._command_edge_this_frame = "none"
        self._default_full_pos_t = self.env.scene["robot"].data.default_joint_pos.clone().squeeze(0).to(self.device)

        # Replay state (aligned with SONIC/TWIST2 replay support).
        # NOTE: _replay_file/_replay_mode/_replay_loop/_replay_enabled/_record_during_replay/
        # _exit_when_replay_complete were already initialized early (before the VLA check above).
        self._replay_body_targets = None
        self._replay_commands = None
        self._replay_policies = None
        self._replay_policy_actions = None
        self._replay_recorded_joint_pos = None
        self._replay_hand_left = None
        self._replay_hand_right = None
        self._replay_object_states: dict[str, dict[str, Any]] = {}
        self._replay_num_frames = 0
        self._replay_cursor = 0
        self._replay_joint_mae_sum = 0.0
        self._replay_joint_mae_count = 0
        self._replay_joint_err_log_interval = 10
        self._replay_object_err_sums: dict[tuple[str, str], float] = {}
        self._replay_object_err_counts: dict[tuple[str, str], int] = {}
        self._replay_completion_requested = False
        self._replay_reward_max = None
        self._replay_any_success = False
        if self._replay_enabled:
            self._setup_local_replay()
        # Align with SONIC: only auto-start recording in live mode or when record_during_replay is set
        self._should_start_recording_on_first_call = (not self._replay_enabled) or self._record_during_replay

        if self._use_lerobot_vla:
            self._setup_lerobot_vla(args_cli)

        print(
            f"[MimicLiteActionProvider] ready redis={self.redis_host}:{self.redis_port} "
            f"onnx={onnx_path} yaml={yaml_path} "
            f"physics_dt={float(getattr(self.env, 'physics_dt', 0.0) or 0.0):.6f} "
            f"decimation={self._decimation} "
            f"control_dt={float(getattr(self.env, 'physics_dt', 0.0) or 0.0) * self._decimation:.6f} "
            f"prev_actions_newest_first={self.runtime.prev_actions_newest_first} "
            f"yaml_vs_robot_default_abs_max={float(np.max(np.abs(self.runtime.default_joint_pos - self.runtime.robot_default_joint_pos))):.3f} "
            f"motion_dt={self.runtime.motion_dt_s:.3f} motion_delay={self.runtime._delay_ns / 1e9:.3f} "
            f"self_torque={self._use_self_torque} "
            f"kp_range=({float(self._mimic_kp_t.min()):.3f},{float(self._mimic_kp_t.max()):.3f}) "
            f"kd_range=({float(self._mimic_kd_t.min()):.3f},{float(self._mimic_kd_t.max()):.3f}) "
            f"effort_limit_range=({float(self._mimic_effort_limit_t.min()):.3f},{float(self._mimic_effort_limit_t.max()):.3f}) "
            f"input_ready_key={self._input_ready_key} recording_task={self.task_name}_mimic_lite "
            f"replay_enabled={self._replay_enabled} replay_mode={self._replay_mode if self._replay_enabled else 'n/a'} "
            f"replay_frames={self._replay_num_frames} replay_file={self._replay_file or 'n/a'}"
        )


    @staticmethod
    def _quat_lerp_normalized_wxyz(q0, q1, alpha: float):
        q0 = np.asarray(q0, dtype=np.float32).reshape(4)
        q1 = np.asarray(q1, dtype=np.float32).reshape(4)
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        q = (1.0 - float(alpha)) * q0 + float(alpha) * q1
        norm = float(np.linalg.norm(q))
        if norm > 1e-6:
            return (q / norm).astype(np.float32)
        return q1.astype(np.float32, copy=True)

    def _startup_blended_reference(self):
        if self._startup_blend_sec <= 0.0:
            return self._last_ref_joint_pos, self._last_ref_body_pos, self._last_ref_body_quat, 1.0
        physics_dt = float(getattr(self.env, "physics_dt", 0.005) or 0.005)
        policy_dt = physics_dt * max(1, self._decimation)
        warmup_frames = max(1, int(round(self._startup_blend_sec / max(policy_dt, 1e-6))))
        alpha = min(1.0, float(self._frame_count) / float(warmup_frames))
        joint_pos = ((1.0 - alpha) * self._startup_ref_joint_pos + alpha * self._last_ref_joint_pos).astype(np.float32)
        body_pos = ((1.0 - alpha) * self._startup_ref_body_pos + alpha * self._last_ref_body_pos).astype(np.float32)
        body_quat = self._quat_lerp_normalized_wxyz(self._startup_ref_body_quat, self._last_ref_body_quat, alpha)
        return joint_pos, body_pos, body_quat, alpha

    def _setup_joint_mapping(self):
        self.all_joint_names = list(self.env.scene["robot"].data.joint_names)
        idx_map = {name: i for i, name in enumerate(self.all_joint_names)}
        missing = [name for name in MIMIC_LITE_JOINT_ORDER if name not in idx_map]
        if missing:
            raise ValueError(f"[MimicLiteActionProvider] joints missing: {missing}")
        self._body_idx = torch.tensor([idx_map[name] for name in MIMIC_LITE_JOINT_ORDER], dtype=torch.long, device=self.device)
        self._default_full_pos = self.env.scene["robot"].data.default_joint_pos.clone().squeeze(0).to(self.device)

        self.left_hand_joint_mapping = {
            "left_hand_thumb_0_joint": 0,
            "left_hand_thumb_1_joint": 1,
            "left_hand_thumb_2_joint": 2,
            "left_hand_middle_0_joint": 3,
            "left_hand_middle_1_joint": 4,
            "left_hand_index_0_joint": 5,
            "left_hand_index_1_joint": 6,
        }
        self.right_hand_joint_mapping = {
            "right_hand_thumb_0_joint": 0,
            "right_hand_thumb_1_joint": 1,
            "right_hand_thumb_2_joint": 2,
            "right_hand_middle_0_joint": 3,
            "right_hand_middle_1_joint": 4,
            "right_hand_index_0_joint": 5,
            "right_hand_index_1_joint": 6,
        }
        self._left_hand_target_idx = []
        self._left_hand_source_idx = []
        self._right_hand_target_idx = []
        self._right_hand_source_idx = []
        for name, source_idx in self.left_hand_joint_mapping.items():
            if name in idx_map:
                self._left_hand_target_idx.append(idx_map[name])
                self._left_hand_source_idx.append(source_idx)
        for name, source_idx in self.right_hand_joint_mapping.items():
            if name in idx_map:
                self._right_hand_target_idx.append(idx_map[name])
                self._right_hand_source_idx.append(source_idx)
        self._left_hand_target_idx_t = torch.tensor(self._left_hand_target_idx, dtype=torch.long, device=self.device)
        self._right_hand_target_idx_t = torch.tensor(self._right_hand_target_idx, dtype=torch.long, device=self.device)

        gripper_map = {
            "right_hand_Joint1_1": 0,
            "right_hand_Joint2_1": 0,
            "left_hand_Joint1_1": 1,
            "left_hand_Joint2_1": 1,
        }
        self._gripper_target_idx = []
        self._gripper_source_idx = []
        for name, source_idx in gripper_map.items():
            if name in idx_map:
                self._gripper_target_idx.append(idx_map[name])
                self._gripper_source_idx.append(source_idx)
        self._gripper_target_idx_t = torch.tensor(self._gripper_target_idx, dtype=torch.long, device=self.device)

    def _resolve_effort_limits(self):
        robot_data = self.env.scene["robot"].data
        for attr in ("joint_effort_limits", "soft_joint_effort_limits"):
            limits = getattr(robot_data, attr, None)
            if limits is None:
                continue
            try:
                out = limits[0, self._body_idx].detach().to(device=self.device, dtype=torch.float32)
                if torch.all(torch.isfinite(out)) and torch.any(out > 0.0):
                    return out
            except Exception:
                pass
        return torch.full((len(MIMIC_LITE_JOINT_ORDER),), float("inf"), dtype=torch.float32, device=self.device)

    def _ensure_self_torque_mode(self, env):
        if not self._use_self_torque or self._torque_mode_configured:
            return
        robot = env.scene["robot"]
        robot.write_joint_stiffness_to_sim(0.0, joint_ids=self._body_idx)
        robot.write_joint_damping_to_sim(0.0, joint_ids=self._body_idx)
        self._torque_mode_configured = True
        print(
            "[MimicLiteActionProvider] self torque mode configured: "
            "body joint stiffness/damping set to 0"
        )

    def _compute_body_effort(self, target: torch.Tensor) -> torch.Tensor:
        robot_data = self.env.scene["robot"].data
        current_pos = robot_data.joint_pos[0, self._body_idx]
        current_vel = robot_data.joint_vel[0, self._body_idx]
        effort = self._mimic_kp_t * (target - current_pos) + self._mimic_kd_t * (self._zero_body_vel_t - current_vel)
        limits = self._mimic_effort_limit_t
        effort = torch.clamp(effort, -limits, limits)
        self._last_body_effort = effort.detach().clone()
        return effort

    def _setup_hand_interfaces(self):
        self.dex3_dds = None
        self.gripper_dds = None
        try:
            from dds.dds_master import dds_manager

            if self.enable_dex3:
                self.dex3_dds = dds_manager.get_object("dex3")
            elif self.enable_gripper:
                self.gripper_dds = dds_manager.get_object("dex1")
        except Exception as exc:
            print(f"[MimicLiteActionProvider] hand DDS unavailable: {exc}")

    def _setup_redis(self):
        if redis is None:
            raise ImportError("redis package is required for MimicLite GMR input")
        self._redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=False)
        self._redis_control_client = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
        self._redis_client.ping()
        self._redis_control_client.ping()

    # ==================================================================
    # VLA support methods (aligned with SonicActionProvider)
    # ==================================================================

    def _setup_lerobot_vla(self, args_cli) -> None:
        if self._lerobot_server_url:
            self._lerobot_http_client = LeRobotVLAHttpClient(
                base_url=self._lerobot_server_url,
                timeout_s=self._lerobot_server_timeout,
                verify_ssl=self._lerobot_server_verify_ssl,
            )
            print(
                f"[MimicLiteActionProvider] LeRobot VLA remote client enabled: "
                f"url={self._lerobot_server_url} verify_ssl={self._lerobot_server_verify_ssl}"
            )
            return
        lerobot_policy_path = Path(getattr(args_cli, "lerobot_policy_path", "")).expanduser().resolve()
        if not lerobot_policy_path.is_dir():
            raise FileNotFoundError(
                f"[MimicLiteActionProvider] LeRobot policy directory not found: {lerobot_policy_path}"
            )
        project_root = os.environ.get("PROJECT_ROOT")
        isaaclab_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        lerobot_src = isaaclab_root.parent / "lerobot" / "src"
        if not lerobot_src.is_dir():
            raise FileNotFoundError(
                f"[MimicLiteActionProvider] LeRobot src directory not found: {lerobot_src}"
            )
        import sys as _sys
        if str(lerobot_src) not in _sys.path:
            _sys.path.insert(0, str(lerobot_src))
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import _reconnect_relative_absolute_steps, get_policy_class
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        from lerobot.utils.constants import (
            POLICY_POSTPROCESSOR_DEFAULT_NAME,
            POLICY_PREPROCESSOR_DEFAULT_NAME,
        )
        from lerobot.utils.control_utils import predict_action

        device_name = getattr(args_cli, "lerobot_policy_device", "") or getattr(args_cli, "device", "cuda:0")
        config = PreTrainedConfig.from_pretrained(lerobot_policy_path)
        config.device = device_name
        state_feature = (config.input_features or {}).get("observation.state")
        action_feature = (config.output_features or {}).get("action")
        state_shape = tuple(getattr(state_feature, "shape", ()) or ())
        action_shape = tuple(getattr(action_feature, "shape", ()) or ())
        if state_shape and state_shape != (SONIC_VLA_STATE_DIM,):
            raise ValueError(
                f"[MimicLiteActionProvider] VLA policy must use observation.state shape {(SONIC_VLA_STATE_DIM,)}, "
                f"got {state_shape}"
            )
        if action_shape and action_shape != (SONIC_VLA_ACTION_DIM,):
            raise ValueError(
                f"[MimicLiteActionProvider] VLA policy action shape mismatch: expected {(SONIC_VLA_ACTION_DIM,)}, "
                f"got {action_shape}"
            )
        policy_cls = get_policy_class(config.type)
        self._lerobot_policy = policy_cls.from_pretrained(lerobot_policy_path, config=config)
        self._lerobot_preprocessor = PolicyProcessorPipeline.from_pretrained(
            lerobot_policy_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        )
        self._lerobot_postprocessor = PolicyProcessorPipeline.from_pretrained(
            lerobot_policy_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        _reconnect_relative_absolute_steps(self._lerobot_preprocessor, self._lerobot_postprocessor)
        self._lerobot_predict_action = predict_action
        self._lerobot_device = torch.device(device_name)
        self._lerobot_policy.reset()
        self._lerobot_preprocessor.reset()
        self._lerobot_postprocessor.reset()
        print(
            f"[MimicLiteActionProvider] LeRobot VLA enabled: "
            f"path={lerobot_policy_path} type={config.type} device={self._lerobot_device}"
        )

    def _get_front_camera_rgb_for_vla(self) -> np.ndarray:
        if "front_camera" not in self.env.scene.keys():
            raise RuntimeError(
                "[MimicLiteActionProvider] front_camera not found in scene for LeRobot VLA inference"
            )
        camera = self.env.scene["front_camera"]
        if "rgb" not in camera.data.output:
            raise RuntimeError(
                "[MimicLiteActionProvider] front_camera has no rgb output for LeRobot VLA inference"
            )
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.ndim != 3:
            raise RuntimeError(f"[MimicLiteActionProvider] Unexpected front_camera rgb shape: {rgb.shape}")
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _get_current_robot_root_pose_for_vla(self) -> tuple[np.ndarray, np.ndarray]:
        robot = self.env.scene["robot"].data
        root_state = robot.root_state_w[0].detach().cpu().numpy().astype(np.float32)
        root_xy_world = root_state[0:2].astype(np.float32, copy=True)
        root_quat_wxyz = root_state[3:7].astype(np.float32, copy=True)
        return root_quat_wxyz, root_xy_world

    def _build_lerobot_vla_observation_state(self) -> np.ndarray:
        robot = self.env.scene["robot"].data
        joint_pos = robot.joint_pos[0, self._body_idx].cpu().numpy().astype(np.float32)
        joint_vel = robot.joint_vel[0, self._body_idx].cpu().numpy().astype(np.float32)
        base_quat_wxyz, _ = self._get_current_robot_root_pose_for_vla()
        if self._vla_initial_robot_quat_wxyz is None:
            self._vla_initial_robot_quat_wxyz = base_quat_wxyz.copy()
        state = build_vla_rotlocal_v3_observation_state(
            initial_robot_orientation_wxyz=self._vla_initial_robot_quat_wxyz,
            root_orientation_wxyz=base_quat_wxyz,
            joint_pos_canonical_29=joint_pos,
            joint_vel_canonical_29=joint_vel,
        )
        # Yaw calibration: inject +calib yaw to state[0:6] to match mimic_lite
        # training data convention (raw world-frame quat instead of heading-canonicalized).
        if self._vla_yaw_calib_quat_wxyz is not None:
            heading_canonical_quat = rot6d_to_quat_wxyz_with_layout(
                state[0:6], layout="row"
            ).reshape(4).astype(np.float32)
            calibrated_quat = quat_mul_wxyz(self._vla_yaw_calib_quat_wxyz, heading_canonical_quat)
            state[0:6] = quat_to_rot6d_wxyz(calibrated_quat).reshape(6).astype(np.float32)
        if state.shape != (SONIC_VLA_STATE_DIM,):
            raise RuntimeError(
                f"[MimicLiteActionProvider] expected VLA observation.state shape {(SONIC_VLA_STATE_DIM,)}, "
                f"got {state.shape}"
            )
        return state

    def _fetch_lerobot_action_chunk(self) -> np.ndarray:
        rgb = self._get_front_camera_rgb_for_vla()
        state = self._build_lerobot_vla_observation_state()
        if self._lerobot_http_client is not None:
            action_chunk = self._lerobot_http_client.infer_chunk(
                front_rgb=rgb,
                observation_state=state,
                robot_type=self._enable_robot,
                task=self.task_name,
            )
        else:
            if self._lerobot_policy is None or self._lerobot_predict_action is None:
                raise RuntimeError(
                    "[MimicLiteActionProvider] LeRobot VLA requested before initialization"
                )
            observation = {
                "observation.images.front": rgb,
                "observation.state": state,
            }
            action = self._lerobot_predict_action(
                observation=observation,
                policy=self._lerobot_policy,
                device=self._lerobot_device,
                preprocessor=self._lerobot_preprocessor,
                postprocessor=self._lerobot_postprocessor,
                use_amp=self._lerobot_device.type == "cuda",
                task=self.task_name,
                robot_type=self._enable_robot,
            )
            if isinstance(action, torch.Tensor):
                action_chunk = action.detach().cpu().numpy().astype(np.float32)
            else:
                action_chunk = np.asarray(action, dtype=np.float32)
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(1, -1)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != SONIC_VLA_ACTION_DIM:
            raise ValueError(
                f"[MimicLiteActionProvider] Expected VLA action chunk shape [N, {SONIC_VLA_ACTION_DIM}], "
                f"got {action_chunk.shape}"
            )
        return action_chunk

    def _pop_lerobot_action(self) -> np.ndarray:
        if not self._lerobot_action_chunk_queue:
            for action in self._fetch_lerobot_action_chunk():
                self._lerobot_action_chunk_queue.append(
                    np.asarray(action, dtype=np.float32).copy()
                )
        return np.asarray(
            self._lerobot_action_chunk_queue.popleft(), dtype=np.float32
        ).reshape(-1)

    def _infer_lerobot_semantic_action(self) -> np.ndarray:
        action = self._pop_lerobot_action()
        if action.shape != (SONIC_VLA_ACTION_DIM,):
            raise ValueError(
                f"[MimicLiteActionProvider] Expected VLA action dim {SONIC_VLA_ACTION_DIM}, got {action.shape}"
            )
        return action

    def _should_refresh_lerobot_visuals_next_step(self) -> bool:
        return (not self._use_lerobot_vla) or (len(self._lerobot_action_chunk_queue) == 0)

    def _get_hand_pose_from_binary(self, side: str, closed: bool) -> np.ndarray:
        pose_name = "close" if closed else "open"
        pose = DEFAULT_HAND_POSE[SONIC_HAND_POSE_ROBOT_NAME][side][pose_name]
        return np.asarray(pose, dtype=np.float32).reshape(-1)

    def _apply_hand_binary_targets(self, left_closed: bool, right_closed: bool) -> None:
        self._left_hand_binary_state = bool(left_closed)
        self._right_hand_binary_state = bool(right_closed)
        self._left_hand_target[:] = self._get_hand_pose_from_binary("left", left_closed)
        self._right_hand_target[:] = self._get_hand_pose_from_binary("right", right_closed)

    def _apply_lerobot_vla_action(self, action: np.ndarray) -> dict:
        # Decode 40-dim VLA action and step mimic_lite runtime. Returns result dict.
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._latest_vla_action = action.copy()
        current_robot_quat_wxyz, current_robot_xy_world = self._get_current_robot_root_pose_for_vla()
        # Yaw calibration: correct action[0:2] (xy_delta) and action[3:9] (rot6d) by -calib
        # to cancel the fixed +yaw offset in mimic_lite training data convention.
        if self._vla_yaw_calib_quat_inv_wxyz is not None:
            action = action.copy()
            # action[0:2]: 2D rotate by -calib_deg (cancel training data +calib rotation)
            c, s = float(np.cos(-self._vla_yaw_calib_rad)), float(np.sin(-self._vla_yaw_calib_rad))
            xy = action[0:2].astype(np.float32).copy()
            action[0] = c * xy[0] - s * xy[1]
            action[1] = s * xy[0] + c * xy[1]
            # action[3:9]: decode -> left-multiply inv-calib yaw -> re-encode
            decoded_quat = rot6d_to_quat_wxyz_with_layout(
                action[3:9], layout="row"
            ).reshape(4).astype(np.float32)
            corrected_quat = quat_mul_wxyz(self._vla_yaw_calib_quat_inv_wxyz, decoded_quat)
            action[3:9] = quat_to_rot6d_wxyz(corrected_quat).reshape(6).astype(np.float32)
        runtime_frame = self._lerobot_vla_runtime.step(
            action,
            current_robot_quat_wxyz=current_robot_quat_wxyz,
            current_robot_xy_world=current_robot_xy_world,
        )
        control_dt = float(self.env.physics_dt * self._decimation)
        payload = build_sonic_joint29_payload_v3(
            runtime_frame=runtime_frame,
            control_dt=control_dt,
        )
        # Feed decoded reference to mimic_lite runtime (interfaces are aligned)
        timestamp_ns = time.time_ns()
        result = self.runtime.step(
            ref_joint_pos=payload["joint_pos"],
            ref_body_pos_w=payload["body_pos"],
            ref_body_quat_wxyz=payload["body_quat_w"],
            timestamp_ns=timestamp_ns,
        )
        # Update hand targets
        left_closed = bool(runtime_frame.hand_binary[0] >= self._lerobot_hand_binary_threshold)
        right_closed = bool(runtime_frame.hand_binary[1] >= self._lerobot_hand_binary_threshold)
        self._apply_hand_binary_targets(left_closed=left_closed, right_closed=right_closed)
        return {"result": result, "left_closed": left_closed, "right_closed": right_closed}

    def _collect_env_state(self) -> dict[str, Any]:
        return collect_recordable_env_object_states(self.env, self.env.cfg)

    def _collect_vision_state(self) -> dict[str, Any]:
        vision: dict[str, Any] = {
            "rgb": None,
            "depth": None,
            "world_rgb": None,
            "world_depth": None,
            "left_wrist_rgb": None,
            "left_wrist_depth": None,
            "right_wrist_rgb": None,
            "right_wrist_depth": None,
        }
        try:
            if "front_camera" in self.env.scene.keys():
                camera = self.env.scene["front_camera"]
                if "rgb" in camera.data.output:
                    vision["rgb"] = camera.data.output["rgb"][0].cpu().numpy().copy()
                if "distance_to_image_plane" in camera.data.output:
                    vision["depth"] = camera.data.output["distance_to_image_plane"][0].cpu().numpy().copy()
            if self._record_world_camera and "world_camera" in self.env.scene.keys():
                camera = self.env.scene["world_camera"]
                if "rgb" in camera.data.output:
                    vision["world_rgb"] = camera.data.output["rgb"][0].cpu().numpy().copy()
                if "distance_to_image_plane" in camera.data.output:
                    vision["world_depth"] = camera.data.output["distance_to_image_plane"][0].cpu().numpy().copy()
            if "left_wrist_camera" in self.env.scene.keys():
                camera = self.env.scene["left_wrist_camera"]
                if "rgb" in camera.data.output:
                    vision["left_wrist_rgb"] = camera.data.output["rgb"][0].cpu().numpy().copy()
                if "distance_to_image_plane" in camera.data.output:
                    vision["left_wrist_depth"] = camera.data.output["distance_to_image_plane"][0].cpu().numpy().copy()
            if "right_wrist_camera" in self.env.scene.keys():
                camera = self.env.scene["right_wrist_camera"]
                if "rgb" in camera.data.output:
                    vision["right_wrist_rgb"] = camera.data.output["rgb"][0].cpu().numpy().copy()
                if "distance_to_image_plane" in camera.data.output:
                    vision["right_wrist_depth"] = camera.data.output["distance_to_image_plane"][0].cpu().numpy().copy()
        except Exception:
            pass
        return vision

    def on_env_reset(self):
        self.runtime.reset()
        self._frame_count = 0
        self._prev_ref_debug_pos = None
        self._prev_ref_debug_joint = None
        self._episode_init_env_state = self._collect_env_state()
        # VLA reset (aligned with SonicActionProvider)
        if self._use_lerobot_vla:
            self._lerobot_action_chunk_queue.clear()
            self._latest_vla_action = None
            try:
                root_quat_wxyz, root_xy_world = self._get_current_robot_root_pose_for_vla()
                self._vla_initial_robot_quat_wxyz = root_quat_wxyz.copy()
                self._lerobot_vla_runtime.reset(
                    body_xy_world=root_xy_world,
                    target_root_quat_wxyz=root_quat_wxyz,
                )
            except Exception:
                self._vla_initial_robot_quat_wxyz = None
                self._lerobot_vla_runtime.reset()
            self._apply_hand_binary_targets(left_closed=False, right_closed=False)
            if self._lerobot_http_client is not None:
                self._lerobot_http_client.reset()
            if self._lerobot_policy is not None:
                self._lerobot_policy.reset()
            if self._lerobot_preprocessor is not None:
                self._lerobot_preprocessor.reset()
            if self._lerobot_postprocessor is not None:
                self._lerobot_postprocessor.reset()
        if getattr(self, "_disable_eval_recording", False):
            try:
                self.recording_manager.cancel_recording()
            except Exception:
                pass
            self._recording_active = False
            self._recording_command = "none"
            self._recording_display_state = "idle"
            self._save_in_progress = False
            self._pending_save_jobs = 0
        self._torque_mode_configured = False
        self._latest_recording_control = None
        self._raw_controller_data = None
        self._command_edge_this_frame = "none"
        if self._replay_enabled:
            self._replay_cursor = 0
            self._replay_joint_mae_sum = 0.0
            self._replay_joint_mae_count = 0
            self._replay_object_err_sums = {}
            self._replay_object_err_counts = {}
            self._replay_completion_requested = False
            self._replay_reward_max = None
            self._replay_any_success = False
        print("[MimicLite] on_env_reset: frame_count and history reset")

    def on_env_objects_reset(self):
        self._episode_init_env_state = self._collect_env_state()

    def _recording_enabled_for_current_mode(self) -> bool:
        if getattr(self, "_disable_eval_recording", False):
            return False
        return True

    def _begin_episode_recording(self) -> None:
        if getattr(self, "_disable_eval_recording", False):
            return
        if not self.recording_manager.is_recording:
            self.recording_manager.start_recording()
        self._recording_active = True
        self._recording_display_state = "recording"
        self._recording_display_counter = 0
        self._save_in_progress = False
        self._pending_save_jobs = self.recording_manager.get_pending_save_count()

    def is_recording_active(self):
        return self._recording_active

    def get_recording_command(self):
        return self._recording_command

    def get_pending_save_jobs(self) -> int:
        return int(self._pending_save_jobs)

    def _update_input_ready_guard(self, raw_guard) -> None:
        guard = None
        if raw_guard is not None:
            try:
                if isinstance(raw_guard, (bytes, bytearray)):
                    raw_guard = raw_guard.decode("utf-8")
                parsed = json.loads(raw_guard)
                if isinstance(parsed, dict):
                    guard = parsed
            except Exception:
                guard = None
        new_epoch = int(guard.get("epoch_id", -1)) if guard is not None else -1
        new_realtime = float(guard.get("ready_timestamp_realtime", 0.0)) if guard is not None else 0.0
        new_monotonic = float(guard.get("ready_timestamp_monotonic", 0.0)) if guard is not None else 0.0
        if (
            new_epoch != self._input_ready_epoch_id
            or new_realtime != self._input_ready_timestamp_realtime
            or new_monotonic != self._input_ready_timestamp_monotonic
        ):
            self._stale_input_drop_logged_epoch = -1
        self._input_ready_epoch_id = new_epoch
        self._input_ready_timestamp_realtime = new_realtime
        self._input_ready_timestamp_monotonic = new_monotonic

    def _is_stale_controller_payload(self, controller_data: dict | None) -> bool:
        if self._input_ready_timestamp_monotonic <= 0.0 and self._input_ready_timestamp_realtime <= 0.0:
            return False
        if not isinstance(controller_data, dict):
            return True
        monotonic_ts = controller_data.get("timestamp_monotonic")
        realtime_ts = controller_data.get("timestamp_realtime")
        timestamp_ms = controller_data.get("timestamp")
        try:
            monotonic_ts = float(monotonic_ts) if monotonic_ts is not None else None
        except Exception:
            monotonic_ts = None
        try:
            realtime_ts = float(realtime_ts) if realtime_ts is not None else None
        except Exception:
            realtime_ts = None
        try:
            timestamp_ms = float(timestamp_ms) if timestamp_ms is not None else None
        except Exception:
            timestamp_ms = None
        if self._input_ready_timestamp_monotonic > 0.0 and monotonic_ts is not None:
            return monotonic_ts <= self._input_ready_timestamp_monotonic
        if self._input_ready_timestamp_realtime > 0.0 and realtime_ts is not None:
            return realtime_ts <= self._input_ready_timestamp_realtime
        if self._input_ready_timestamp_realtime > 0.0 and timestamp_ms is not None:
            return (timestamp_ms / 1000.0) <= self._input_ready_timestamp_realtime
        return False

    def _update_recording_display_state(self) -> None:
        self._pending_save_jobs = self.recording_manager.get_pending_save_count()
        self._save_in_progress = self._pending_save_jobs > 0
        if self._save_completion_state is not None:
            if self._save_completion_state == "success":
                self._recording_display_state = "saved"
            else:
                self._recording_display_state = "discard"
            self._recording_display_counter = 0
            self._save_completion_state = None
        elif self._save_in_progress:
            return
        elif self._recording_display_state in ["saved", "discard"]:
            self._recording_display_counter += 1
            if self._recording_display_counter >= self._recording_display_duration:
                self._recording_display_counter = 0
                if not self.recording_manager.is_recording and not self._waiting_for_reset_complete:
                    self._begin_episode_recording()
                else:
                    self._recording_display_state = "recording" if self.recording_manager.is_recording else "idle"
        elif self.recording_manager.is_recording:
            self._recording_display_state = "recording"
        elif self._recording_display_state == "recording":
            self._recording_display_state = "idle"

    def _trigger_complete_reset(self) -> None:
        try:
            publish_reset_command(
                reset_category="3",
                redis_client=self._redis_control_client,
                host=self.redis_host,
                port=self.redis_port,
            )
            print("[MimicLite] complete reset requested via Redis")
        except Exception as exc:
            print(f"[MimicLite] failed to send reset trigger: {exc}")

    def _check_reset_complete(self) -> bool:
        try:
            return consume_reset_complete(
                redis_client=self._redis_control_client,
                host=self.redis_host,
                port=self.redis_port,
            )
        except Exception as exc:
            print(f"[MimicLite] failed to check reset complete: {exc}")
            return False

    def _handle_recording_command(self) -> None:
        command = self._recording_command
        if command == "none":
            return
        print(f"[MimicLite] recording command received: {command}")

        def on_save_complete(success: bool) -> None:
            self._save_completion_state = "success" if success else "failure"

        if command == "start":
            self._begin_episode_recording()

        elif command == "save":
            if self.recording_manager.is_recording:
                print("[MimicLite] saving recording...")
                self._recording_display_state = "saving"
                self._recording_display_counter = 0
                self._save_in_progress = True
                self.recording_manager.save_recording(completion_callback=on_save_complete)
                self._pending_save_jobs = self.recording_manager.get_pending_save_count()
                print(f"[MimicLite] save queued (pending={self._pending_save_jobs})")
                self._recording_active = False
                self._episode_id += 1

        elif command == "cancel":
            self.recording_manager.cancel_recording()
            self._recording_active = False
            self._recording_display_state = "discard"
            self._recording_display_counter = 0
            self._pending_save_jobs = self.recording_manager.get_pending_save_count()
            self._save_in_progress = self._pending_save_jobs > 0

        elif command == "save_and_reset":
            if self.recording_manager.is_recording:
                print("[MimicLite] save_and_reset command received")
                self._recording_display_state = "saving"
                self._recording_display_counter = 0
                self._save_in_progress = True
                print("[MimicLite] saving recording...")
                self.recording_manager.save_recording(completion_callback=on_save_complete)
                self._pending_save_jobs = self.recording_manager.get_pending_save_count()
                print(f"[MimicLite] save queued (pending={self._pending_save_jobs})")
            self._recording_active = False
            print("[MimicLite] triggering complete reset...")
            self._trigger_complete_reset()
            self._waiting_for_reset_complete = True
            self._reset_complete_received = False

        elif command == "discard_and_reset":
            print("[MimicLite] discard_and_reset command received")
            self.recording_manager.cancel_recording()
            self._recording_active = False
            self._recording_display_state = "discard"
            self._recording_display_counter = 0
            self._pending_save_jobs = self.recording_manager.get_pending_save_count()
            self._save_in_progress = self._pending_save_jobs > 0
            print("[MimicLite] triggering complete reset...")
            self._trigger_complete_reset()
            self._waiting_for_reset_complete = True
            self._reset_complete_received = False

        self._recording_command = "none"

    @staticmethod
    def _normalize_replay_mode(replay_mode: str) -> str:
        if replay_mode in ("direct", "direct_replay"):
            return "direct_replay"
        if replay_mode in ("inference", "inference_replay"):
            return "inference_replay"
        raise ValueError(f"[MimicLite] Unsupported replay_mode: {replay_mode}")

    def _load_replay_array(self, replay_data, *keys, required=False):
        for key in keys:
            if key in replay_data:
                return np.asarray(replay_data[key]).copy()
        if required:
            raise KeyError(f"[MimicLite] replay npz missing keys: {keys}")
        return None

    def _setup_local_replay(self):
        replay_path = Path(self._replay_file).expanduser().resolve()
        if not replay_path.is_file():
            raise FileNotFoundError(f"[MimicLite] replay file not found: {replay_path}")

        with np.load(replay_path, allow_pickle=True) as replay_data:
            self._replay_body_targets = self._load_replay_array(
                replay_data, "action_body_29dof", required=self._replay_mode == "direct_replay",
            )
            self._replay_commands = self._load_replay_array(replay_data, "mimic_command")
            self._replay_policies = self._load_replay_array(replay_data, "mimic_policy")
            self._replay_policy_actions = self._load_replay_array(replay_data, "mimic_policy_action")
            if self._replay_mode == "inference_replay":
                if self._replay_commands is None or self._replay_policies is None:
                    raise KeyError(
                        "[MimicLite] inference_replay requires mimic_command and mimic_policy arrays"
                    )
            self._replay_recorded_joint_pos = self._load_replay_array(
                replay_data, "robot_qpos_before_decimation",
            )
            self._replay_hand_left = self._load_replay_array(
                replay_data, "action_hand_action_left", "human_raw_left_hand",
            )
            self._replay_hand_right = self._load_replay_array(
                replay_data, "action_hand_action_right", "human_raw_right_hand",
            )
            self._replay_object_states = {}
            replay_object_suffixes = {
                "_position": "position",
                "_orientation": "orientation",
                "_linear_velocity": "linear_velocity",
                "_angular_velocity": "angular_velocity",
            }
            for key in replay_data.files:
                if not key.startswith("env_obj_"):
                    continue
                for suffix, field_name in replay_object_suffixes.items():
                    if key.endswith(suffix):
                        object_name = key[len("env_obj_") : -len(suffix)]
                        self._replay_object_states.setdefault(object_name, {})[field_name] = np.asarray(
                            replay_data[key]
                        ).copy()
                        break

        candidate_arrays = [
            arr for arr in [
                self._replay_body_targets,
                self._replay_commands,
                self._replay_policies,
                self._replay_hand_left,
                self._replay_hand_right,
            ] if arr is not None
        ]
        if not candidate_arrays:
            raise ValueError("[MimicLite] replay npz contains no usable frame arrays")

        self._replay_num_frames = int(candidate_arrays[0].shape[0])
        self._replay_file = str(replay_path)
        self._replay_cursor = 0
        print(
            f"[MimicLite] replay loaded: {self._replay_num_frames} frames "
            f"mode={self._replay_mode} body_targets={'Y' if self._replay_body_targets is not None else 'N'} "
            f"commands={'Y' if self._replay_commands is not None else 'N'} "
            f"policies={'Y' if self._replay_policies is not None else 'N'} "
            f"objects={list(self._replay_object_states.keys())}"
        )

    def _next_replay_frame_idx(self) -> int | None:
        if not self._replay_enabled or self._replay_num_frames <= 0:
            return None
        if self._replay_cursor >= self._replay_num_frames:
            if not self._replay_loop:
                return None
            self._replay_cursor = 0
        frame_idx = int(self._replay_cursor)
        self._replay_cursor += 1
        return frame_idx

    def _set_replay_hand_targets(self, frame_idx: int) -> tuple[Any, Any]:
        left_hand = None
        right_hand = None
        if self._replay_hand_left is not None and frame_idx < len(self._replay_hand_left):
            left_hand = np.asarray(self._replay_hand_left[frame_idx], dtype=np.float32).reshape(-1)
        if self._replay_hand_right is not None and frame_idx < len(self._replay_hand_right):
            right_hand = np.asarray(self._replay_hand_right[frame_idx], dtype=np.float32).reshape(-1)
        return left_hand, right_hand

    def _run_replay_policy(self, frame_idx: int) -> dict[str, np.ndarray]:
        """Return a result dict matching MimicLitePolicyRuntime.step output."""
        if self._replay_mode == "inference_replay":
            command = np.asarray(self._replay_commands[frame_idx], dtype=np.float32).reshape(304)
            policy = np.asarray(self._replay_policies[frame_idx], dtype=np.float32).reshape(535)
            action = np.asarray(
                self.runtime.session.run(["action"], {"command": command, "policy": policy})[0],
                dtype=np.float32,
            ).reshape(29)
            target = self.runtime.default_joint_pos + action * self.runtime.action_scale
            self.runtime.last_action = action.copy()
            self.runtime.last_target = target.astype(np.float32, copy=True)
            self.runtime.prev_action_hist.append(action.copy())
            self.runtime.last_command = command.copy()
            self.runtime.last_policy = policy.copy()
            return {
                "target_joint_pos": target.astype(np.float32, copy=True),
                "policy_action": action.copy(),
                "command": command.copy(),
                "policy": policy.copy(),
            }
        # direct_replay
        target = np.asarray(self._replay_body_targets[frame_idx], dtype=np.float32).reshape(-1)
        if target.shape[0] != 29:
            raise ValueError(f"[MimicLite] direct_replay target must have 29 dims, got {target.shape}")
        if self._replay_policy_actions is not None and frame_idx < len(self._replay_policy_actions):
            action = np.asarray(self._replay_policy_actions[frame_idx], dtype=np.float32).reshape(-1)
            if action.shape[0] != 29:
                raise ValueError(f"[MimicLite] replay policy action must have 29 dims, got {action.shape}")
        else:
            action = np.zeros((29,), dtype=np.float32)
        self.runtime.last_action = action.copy()
        self.runtime.last_target = target.copy()
        self.runtime.prev_action_hist.append(action.copy())
        command = (
            np.asarray(self._replay_commands[frame_idx], dtype=np.float32).reshape(304)
            if self._replay_commands is not None and frame_idx < len(self._replay_commands)
            else np.zeros((304,), dtype=np.float32)
        )
        policy = (
            np.asarray(self._replay_policies[frame_idx], dtype=np.float32).reshape(535)
            if self._replay_policies is not None and frame_idx < len(self._replay_policies)
            else np.zeros((535,), dtype=np.float32)
        )
        self.runtime.last_command = command.copy()
        self.runtime.last_policy = policy.copy()
        return {
            "target_joint_pos": target.copy(),
            "policy_action": action.copy(),
            "command": command.copy(),
            "policy": policy.copy(),
        }

    def _finalize_replay_if_needed(self) -> None:
        if self._replay_completion_requested:
            return
        self._replay_completion_requested = True
        try:
            setattr(self.env, "_request_main_loop_exit", True)
            print("[MimicLite] Marked env for main-loop exit after replay completion")
        except Exception:
            pass
        if self._record_during_replay and self.recording_manager.is_recording:
            self._update_replay_reward_stats()
            final_reward = _reward_scalar(self.env)
            max_reward = final_reward if self._replay_reward_max is None else max(self._replay_reward_max, final_reward)
            any_success = bool(self._replay_any_success or _reward_success_flag(max_reward))
            if self.recording_manager.recording_buffer:
                last_frame = self.recording_manager.recording_buffer[-1]
                last_frame["rerecord_final_reward"] = final_reward
                last_frame["rerecord_max_reward"] = max_reward
                last_frame["rerecord_any_success"] = any_success
            print(f"[MimicLite] Final rerecord reward={final_reward:.4f}")
            print(f"[MimicLite] Max rerecord reward={max_reward:.4f} any_success={str(any_success).lower()}")
            print("[MimicLite] Finalizing replay rerecord before exit")
            self.recording_manager.save_recording()
    def should_exit_after_replay_complete(self) -> bool:
        if not self._exit_when_replay_complete:
            return False
        return bool(self._replay_completion_requested)

    def _update_replay_reward_stats(self) -> None:
        if not (self._replay_enabled and self._record_during_replay):
            return
        if self._replay_cursor <= 0:
            return
        reward = _reward_scalar(self.env)
        if self._replay_reward_max is None:
            self._replay_reward_max = reward
        else:
            self._replay_reward_max = max(self._replay_reward_max, reward)
        if _reward_success_flag(reward):
            self._replay_any_success = True

    def _log_replay_joint_position_error(self, frame_idx: int) -> None:
        if (
            not self._replay_enabled
            or self._replay_recorded_joint_pos is None
            or len(self._replay_recorded_joint_pos) == 0
        ):
            return
        compare_idx = min(frame_idx + 1, len(self._replay_recorded_joint_pos) - 1)
        recorded = np.asarray(self._replay_recorded_joint_pos[compare_idx], dtype=np.float32).reshape(-1)
        current = (
            self.env.scene["robot"].data.joint_pos[0, self._body_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(-1)
        )
        dims = min(current.shape[0], recorded.shape[0])
        if dims <= 0:
            return
        err = np.abs(current[:dims] - recorded[:dims])
        mae = float(err.mean())
        max_err = float(err.max())
        self._replay_joint_mae_sum += mae
        self._replay_joint_mae_count += 1
        running_mae = self._replay_joint_mae_sum / max(1, self._replay_joint_mae_count)
        if frame_idx < 3 or frame_idx % self._replay_joint_err_log_interval == 0:
            print(
                f"[MimicLite] Replay joint err: frame={frame_idx} "
                f"compare_to_recorded_frame={compare_idx} "
                f"mae={mae:.6f} rad max={max_err:.6f} rad "
                f"running_mae={running_mae:.6f} rad"
            )

    def _get_current_replay_object_state(self, object_name: str) -> dict | None:
        scene_key = resolve_env_object_scene_key(self.env, self.env.cfg, object_name)
        if scene_key is None:
            return None
        try:
            obj = self.env.scene[scene_key]
            root_state = obj.data.root_state_w
            return {
                "position": root_state[0, 0:3].detach().cpu().numpy().astype(np.float32),
                "linear_velocity": root_state[0, 7:10].detach().cpu().numpy().astype(np.float32),
                "angular_velocity": root_state[0, 10:13].detach().cpu().numpy().astype(np.float32),
            }
        except Exception:
            return None

    def _log_replay_object_state_error(self, frame_idx: int) -> None:
        if not self._replay_enabled or not self._replay_object_states:
            return
        if not (frame_idx < 3 or frame_idx % self._replay_joint_err_log_interval == 0):
            return

        field_units = {
            "position": "m",
            "linear_velocity": "m/s",
            "angular_velocity": "rad/s",
        }
        field_labels = {
            "position": "pos",
            "linear_velocity": "lin_vel",
            "angular_velocity": "ang_vel",
        }

        for object_name, recorded_fields in self._replay_object_states.items():
            current_state = self._get_current_replay_object_state(object_name)
            if current_state is None:
                continue

            parts = []
            object_compare_idx = None
            for field_name in ("position", "linear_velocity", "angular_velocity"):
                recorded_series = recorded_fields.get(field_name)
                if recorded_series is None or len(recorded_series) == 0:
                    continue
                compare_idx = min(frame_idx + 1, len(recorded_series) - 1)
                object_compare_idx = compare_idx
                recorded = np.asarray(recorded_series[compare_idx], dtype=np.float32).reshape(-1)
                current = np.asarray(current_state[field_name], dtype=np.float32).reshape(-1)
                dims = min(current.shape[0], recorded.shape[0])
                if dims <= 0:
                    continue
                err = np.abs(current[:dims] - recorded[:dims])
                mae = float(err.mean())
                max_err = float(err.max())
                stat_key = (object_name, field_name)
                self._replay_object_err_sums[stat_key] = self._replay_object_err_sums.get(stat_key, 0.0) + mae
                self._replay_object_err_counts[stat_key] = self._replay_object_err_counts.get(stat_key, 0) + 1
                running_mae = self._replay_object_err_sums[stat_key] / self._replay_object_err_counts[stat_key]
                unit = field_units[field_name]
                parts.append(
                    f"{field_labels[field_name]}_mae={mae:.6f} {unit} "
                    f"max={max_err:.6f} {unit} running={running_mae:.6f} {unit}"
                )

            if parts:
                print(
                    f"[MimicLite] Replay env err: frame={frame_idx} "
                    f"compare_to_recorded_frame={object_compare_idx if object_compare_idx is not None else frame_idx} "
                    f"object={object_name} "
                    + " ".join(parts)
                )

    def _read_reference(self):
        raw = self._redis_client.mget(
            [
                GMR_JOINT_POS_KEY,
                GMR_JOINT_VEL_KEY,
                GMR_BODY_POS_KEY,
                GMR_BODY_QUAT_W_KEY,
                GMR_FULL_QPOS_KEY,
                "action_hand_left_unitree_g1_with_hands",
                "action_hand_right_unitree_g1_with_hands",
                "teleop_state_unitree_g1_with_hands",
                "t_action",
                "recording_control_unitree_g1_with_hands",
                "controller_data",
                MIMIC_LITE_INPUT_READY_KEY,
            ]
        )
        joint_pos = _decode_json_array(raw[0], dtype=np.float32)
        joint_vel = _decode_json_array(raw[1], dtype=np.float32)
        body_pos = _decode_json_array(raw[2], dtype=np.float32)
        body_quat = _decode_json_array(raw[3], dtype=np.float32)
        full_qpos = _decode_json_array(raw[4], dtype=np.float32)
        left_hand = _decode_json_array(raw[5], dtype=np.float32)
        right_hand = _decode_json_array(raw[6], dtype=np.float32)
        teleop_state = raw[7].decode("utf-8") if raw[7] is not None else None
        timestamp_ns = None
        if raw[8] is not None:
            try:
                timestamp_ns = int(float(raw[8].decode("utf-8") if isinstance(raw[8], bytes) else raw[8]) * 1_000_000)
            except (TypeError, ValueError):
                timestamp_ns = None

        recording_control_raw = raw[9]
        controller_data_raw = raw[10]
        input_ready_raw = raw[11]

        self._update_input_ready_guard(input_ready_raw)

        self._latest_recording_control = None
        if recording_control_raw is not None:
            try:
                payload = (
                    recording_control_raw.decode("utf-8")
                    if isinstance(recording_control_raw, bytes)
                    else recording_control_raw
                )
                self._latest_recording_control = json.loads(payload)
                sequence = int(self._latest_recording_control.get("sequence", -1))
                command = str(self._latest_recording_control.get("command", "none"))
                recording_active = bool(self._latest_recording_control.get("active", False))
                identity = get_recording_control_identity(self._latest_recording_control)
                if should_accept_recording_control(
                    self._latest_recording_control,
                    last_identity=self._latest_recording_control_identity,
                    waiting_for_reset_complete=self._waiting_for_reset_complete,
                ):
                    self._latest_recording_control_sequence = sequence
                    self._latest_recording_control_identity = identity
                    self._command_edge_this_frame = command
                    self._recording_command = command
                    self._recording_active = recording_active
                elif command == "none":
                    self._recording_active = recording_active
            except Exception:
                self._latest_recording_control = None

        self._raw_controller_data = None
        if controller_data_raw is not None:
            try:
                controller_payload = (
                    controller_data_raw.decode("utf-8")
                    if isinstance(controller_data_raw, bytes)
                    else controller_data_raw
                )
                self._raw_controller_data = json.loads(controller_payload)
            except Exception:
                self._raw_controller_data = None

        if teleop_state == "teleop" and self._last_teleop_state != "teleop":
            self.runtime.reset()
            self._prev_ref_debug_pos = None
            self._prev_ref_debug_joint = None
            if self._debug:
                print("[MimicLite] teleop entered, reset reference history and heading alignment")
        self._last_teleop_state = teleop_state
        if (
            joint_pos is not None
            and joint_pos.shape == (29,)
            and np.all(np.isfinite(joint_pos))
            and not np.allclose(joint_pos, 0.0, atol=1e-6)
        ):
            self._last_ref_joint_pos = joint_pos.astype(np.float32, copy=True)
        if joint_vel is not None and joint_vel.shape == (29,) and np.all(np.isfinite(joint_vel)):
            self._last_ref_joint_vel = joint_vel.astype(np.float32, copy=True)
        if full_qpos is not None and full_qpos.shape[0] >= 7 and np.all(np.isfinite(full_qpos[:7])):
            self._last_ref_body_pos = full_qpos[:3].astype(np.float32, copy=True)
            qpos_quat = full_qpos[3:7].astype(np.float32, copy=True)
            if np.linalg.norm(qpos_quat) > 1e-6:
                self._last_ref_body_quat = qpos_quat
        else:
            if body_pos is not None and body_pos.shape == (3,) and np.all(np.isfinite(body_pos)):
                self._last_ref_body_pos = body_pos.astype(np.float32, copy=True)
            if body_quat is not None and body_quat.shape == (4,) and np.all(np.isfinite(body_quat)) and np.linalg.norm(body_quat) > 1e-6:
                self._last_ref_body_quat = body_quat.astype(np.float32, copy=True)
        return left_hand, right_hand, timestamp_ns

    def _apply_hands(self, full_action: torch.Tensor, left_hand, right_hand):
        if self.enable_dex3 and left_hand is not None and right_hand is not None:
            if len(self._left_hand_target_idx) and len(self._right_hand_target_idx):
                left = np.asarray(left_hand, dtype=np.float32).reshape(-1)
                right = np.asarray(right_hand, dtype=np.float32).reshape(-1)
                left_vals = torch.tensor(left[self._left_hand_source_idx], dtype=torch.float32, device=self.device)
                right_vals = torch.tensor(right[self._right_hand_source_idx], dtype=torch.float32, device=self.device)
                full_action.index_copy_(0, self._left_hand_target_idx_t, left_vals)
                full_action.index_copy_(0, self._right_hand_target_idx_t, right_vals)
                return
        if self.enable_dex3 and self.dex3_dds is not None and len(self._left_hand_target_idx):
            hand_cmds = self.dex3_dds.get_hand_commands()
            if hand_cmds:
                left_cmd = hand_cmds.get("left_hand_cmd", {}).get("positions", [])
                right_cmd = hand_cmds.get("right_hand_cmd", {}).get("positions", [])
                if len(left_cmd) >= 7 and len(right_cmd) >= 7:
                    full_action.index_copy_(0, self._left_hand_target_idx_t, torch.tensor(left_cmd[:7], dtype=torch.float32, device=self.device))
                    full_action.index_copy_(0, self._right_hand_target_idx_t, torch.tensor(right_cmd[:7], dtype=torch.float32, device=self.device))
        elif self.enable_gripper and self.gripper_dds is not None and len(self._gripper_target_idx):
            cmd = self.gripper_dds.get_gripper_command()
            if cmd:
                right = cmd.get("right_gripper_cmd", {}).get("positions", [])
                left = cmd.get("left_gripper_cmd", {}).get("positions", [])
                values = list(right[:1]) + list(left[:1])
                if len(values) == 2:
                    source_vals = torch.tensor(values, dtype=torch.float32, device=self.device)
                    mapped = source_vals[torch.tensor(self._gripper_source_idx, dtype=torch.long, device=self.device)]
                    full_action.index_copy_(0, self._gripper_target_idx_t, mapped)

    def _collect_recording_data(
        self,
        *,
        full_action: torch.Tensor,
        target_joint_pos: np.ndarray,
        body_effort_target: np.ndarray,
        left_hand,
        right_hand,
        command: np.ndarray,
        policy: np.ndarray,
        policy_action: np.ndarray,
    ) -> dict[str, Any]:
        robot = self.env.scene["robot"].data
        root_state = robot.root_state_w
        joint_pos = robot.joint_pos[0, self._body_idx].cpu().numpy().astype(np.float32)
        joint_vel = robot.joint_vel[0, self._body_idx].cpu().numpy().astype(np.float32)
        seed_info = get_current_episode_object_seed_info(self.env.cfg)
        left_hand_arr = np.asarray(left_hand, dtype=np.float32).reshape(-1) if left_hand is not None else np.zeros((7,), dtype=np.float32)
        right_hand_arr = np.asarray(right_hand, dtype=np.float32).reshape(-1) if right_hand is not None else np.zeros((7,), dtype=np.float32)
        return {
            "meta": {
                "task": self.task_name,
                "episode_id": self._episode_id,
                "control_dt": float(self.env.physics_dt * self._decimation),
                "physics_dt": float(self.env.physics_dt),
                "decimation": int(self._decimation),
                "onnx_path": self._onnx_path,
                "yaml_path": self._yaml_path,
                "episode_object_seed": seed_info.get("seed"),
                "episode_object_seed_source": seed_info.get("source"),
            },
            "episode_init_env": self._episode_init_env_state,
            "markers": {
                "frame_index": int(self._frame_count),
                "episode_step": int(self.recording_manager.frame_count),
                "timestamp_wall": float(time.time()),
                "recording_command": self._command_edge_this_frame,
                "reset_requested": bool(
                    self._waiting_for_reset_complete
                    or self._command_edge_this_frame in {"save_and_reset", "discard_and_reset"}
                ),
                "reset_completed": bool(self._reset_complete_received),
                "save_triggered": bool(self._command_edge_this_frame == "save"),
            },
            "human_raw": {
                "left_hand": left_hand_arr.copy(),
                "right_hand": right_hand_arr.copy(),
                "controller_data": self._raw_controller_data,
                "recording_control": self._latest_recording_control,
                "body_quat_w": self._last_ref_body_quat.copy(),
                "body_pos": self._last_ref_body_pos.copy(),
                "joint_pos": self._last_ref_joint_pos.copy(),
                "joint_vel": self._last_ref_joint_vel.copy(),
                "full_qpos": np.asarray(self._last_ref_body_pos.tolist() + self._last_ref_body_quat.tolist() + self._last_ref_joint_pos.tolist(), dtype=np.float32) if self._last_ref_joint_pos is not None else np.zeros((7,), dtype=np.float32),
            },
            "mimic_model_io": {
                "command": command.copy(),
                "policy": policy.copy(),
                "policy_action": policy_action.copy(),
            },
            "robot": {
                "qpos_before_decimation": joint_pos,
                "qvel_before_decimation": joint_vel,
                "root_position": root_state[0, 0:3].cpu().numpy(),
                "root_orientation": root_state[0, 3:7].cpu().numpy(),
                "root_lin_vel_local": robot.root_lin_vel_b[0].cpu().numpy(),
                "root_ang_vel_local": robot.root_ang_vel_b[0].cpu().numpy(),
                "root_lin_vel_world": root_state[0, 7:10].cpu().numpy(),
                "root_ang_vel_world": root_state[0, 10:13].cpu().numpy(),
            },
            "action": {
                "body_action_29dof": target_joint_pos.astype(np.float32).copy(),
                "full_action": full_action.detach().cpu().numpy().astype(np.float32),
                "body_effort_target": body_effort_target.astype(np.float32).copy(),
                "hand_action_left": left_hand_arr.copy(),
                "hand_action_right": right_hand_arr.copy(),
            },
            "env": {
                **self._collect_env_state(),
                "vision": self._collect_vision_state(),
            },
        }

    def get_action(self, env) -> Optional[torch.Tensor]:
        self._frame_count += 1
        self._command_edge_this_frame = "none"
        self._update_replay_reward_stats()

        replay_frame_idx = None
        if self._replay_enabled:
            replay_frame_idx = self._next_replay_frame_idx()
            if replay_frame_idx is None:
                print("[MimicLite] Replay finished")
                self._finalize_replay_if_needed()
                if self._exit_when_replay_complete:
                    raise ReplayComplete("MimicLiteActionProvider replay rerecord complete")
                return self._default_full_pos_t.clone()
            if replay_frame_idx % 100 == 0:
                print(
                    f"[MimicLite] Replay progress: {replay_frame_idx}/{self._replay_num_frames} "
                    f"mode={self._replay_mode}"
                )

        if self._waiting_for_reset_complete:
            if self._check_reset_complete():
                print("[MimicLite] reset complete received")
                self._waiting_for_reset_complete = False
                self._reset_complete_received = True
                self.on_env_reset()
                self._episode_id += 1
                if self._recording_enabled_for_current_mode():
                    self._begin_episode_recording()
                else:
                    self._recording_active = False
                    self._recording_display_state = "idle"
                    self._recording_display_counter = 0
            else:
                return self._default_full_pos_t.clone()

        if (
            self._should_start_recording_on_first_call
            and self._recording_enabled_for_current_mode()
            and not self.recording_manager.is_recording
            and not self._waiting_for_reset_complete
        ):
            self._should_start_recording_on_first_call = False
            self._begin_episode_recording()

        left_hand = right_hand = None
        timestamp_ns = None
        startup_alpha = 1.0
        ref_joint_pos = ref_body_pos = ref_body_quat = None
        if self._replay_enabled and replay_frame_idx is not None:
            left_hand, right_hand = self._set_replay_hand_targets(replay_frame_idx)
            result = self._run_replay_policy(replay_frame_idx)
        elif self._use_lerobot_vla:
            action = self._infer_lerobot_semantic_action()
            vla_out = self._apply_lerobot_vla_action(action)
            result = vla_out["result"]
            left_hand = self._left_hand_target.copy()
            right_hand = self._right_hand_target.copy()
        else:
            left_hand, right_hand, timestamp_ns = self._read_reference()
            ref_joint_pos, ref_body_pos, ref_body_quat, startup_alpha = self._startup_blended_reference()
            result = self.runtime.step(
                ref_joint_pos=ref_joint_pos,
                ref_body_pos_w=ref_body_pos,
                ref_body_quat_wxyz=ref_body_quat,
                timestamp_ns=timestamp_ns,
            )
        full_action = self._full_action_buf
        full_action.copy_(self._default_full_pos)
        target = torch.tensor(result["target_joint_pos"], dtype=torch.float32, device=self.device)
        full_action.index_copy_(0, self._body_idx, target)
        self._apply_hands(full_action, left_hand, right_hand)

        if self._debug and (self._frame_count <= 3 or self._frame_count % self._log_every == 0):
            command = result["command"]
            policy = result["policy"]
            policy_action = result["policy_action"]
            target_joint_pos = result["target_joint_pos"]
            command_pos = command[:24].reshape(8, 3)
            command_ori = command[24:72]
            command_joint = command[72:304].reshape(8, 29)
            policy_joint_rel = policy[42:245]
            policy_joint_vel = policy[245:448]
            policy_prev_actions = policy[448:535].reshape(3, 29)
            live_ref_pos_delta = (
                np.zeros(3, dtype=np.float32)
                if self._prev_ref_debug_pos is None
                else self._last_ref_body_pos - self._prev_ref_debug_pos
            )
            live_ref_joint_delta_abs_max = 0.0 if self._prev_ref_debug_joint is None else float(np.max(np.abs(self._last_ref_joint_pos - self._prev_ref_debug_joint)))
            self._prev_ref_debug_pos = self._last_ref_body_pos.copy()
            self._prev_ref_debug_joint = self._last_ref_joint_pos.copy()
            debug = self.runtime.last_debug
            ref_range_str = (
                f"({ref_joint_pos.min():.3f},{ref_joint_pos.max():.3f})"
                if ref_joint_pos is not None else "n/a"
            )
            ref_body_pos_str = (
                np.array2string(ref_body_pos, precision=3)
                if ref_body_pos is not None else "n/a"
            )
            ref_quat_norm_str = (
                f"{np.linalg.norm(ref_body_quat):.3f}"
                if ref_body_quat is not None else "n/a"
            )
            print(
                f"[MimicLite] frame={self._frame_count} "
                f"replay={self._replay_enabled and replay_frame_idx is not None} "
                f"startup_alpha={startup_alpha:.2f} "
                f"robot_yaw={debug.get('robot_yaw', 0.0):.3f} "
                f"robot_aligned_yaw={debug.get('robot_aligned_yaw', 0.0):.3f} "
                f"ref_yaw={debug.get('ref_current_yaw', 0.0):.3f} "
                f"rel_yaw={debug.get('ref_current_b_yaw', 0.0):.3f} "
                f"ref_range={ref_range_str} "
                f"live_ref_range=({self._last_ref_joint_pos.min():.3f},{self._last_ref_joint_pos.max():.3f}) "
                f"ref_body_pos={ref_body_pos_str} "
                f"live_ref_body_pos={np.array2string(self._last_ref_body_pos, precision=3)} "
                f"ref_quat_norm={ref_quat_norm_str} "
                f"cmd_pos_range=({command_pos.min():.3f},{command_pos.max():.3f}) "
                f"cmd_pos_t0={np.array2string(command_pos[3], precision=3)} "
                f"cmd_pos_future4={np.array2string(command_pos[-1], precision=3)} "
                f"cmd_xy_span=({command_pos[:,0].min():.3f},{command_pos[:,0].max():.3f};{command_pos[:,1].min():.3f},{command_pos[:,1].max():.3f}) "
                f"cmd_ori_range=({command_ori.min():.3f},{command_ori.max():.3f}) "
                f"cmd_joint_t0_range=({command_joint[3].min():.3f},{command_joint[3].max():.3f}) "
                f"live_ref_pos_delta={np.array2string(live_ref_pos_delta, precision=3)} "
                f"live_ref_joint_delta_abs_max={live_ref_joint_delta_abs_max:.3f} "
                f"policy_joint_rel_range=({policy_joint_rel.min():.3f},{policy_joint_rel.max():.3f}) "
                f"policy_joint_vel_range=({policy_joint_vel.min():.3f},{policy_joint_vel.max():.3f}) "
                f"prev_action0_range=({policy_prev_actions[0].min():.3f},{policy_prev_actions[0].max():.3f}) "
                f"action_range=({policy_action.min():.3f},{policy_action.max():.3f}) "
                f"target_range=({target_joint_pos.min():.3f},{target_joint_pos.max():.3f}) "
                f"effort_range=({self._last_body_effort.min().item():.3f},{self._last_body_effort.max().item():.3f})"
            )

        if self._recording_command != "none":
            self._handle_recording_command()
            if self._waiting_for_reset_complete:
                return self._default_full_pos_t.clone()

        if self._recording_enabled_for_current_mode() and self.recording_manager.is_recording:
            body_effort_preview = self._last_body_effort.detach().cpu().numpy().astype(np.float32)
            self.recording_manager.add_frame(
                self._collect_recording_data(
                    full_action=full_action,
                    target_joint_pos=result["target_joint_pos"],
                    body_effort_target=body_effort_preview,
                    left_hand=left_hand,
                    right_hand=right_hand,
                    command=result["command"],
                    policy=result["policy"],
                    policy_action=result["policy_action"],
                )
            )
            if self._reset_complete_received:
                self._reset_complete_received = False
        if self._recording_enabled_for_current_mode():
            self._update_recording_display_state()

        # Wholebody controller mode skips env.step(); advance physics here like SONIC/TWIST2.
        should_render = self._frame_count % self._render_interval == 0
        self._ensure_self_torque_mode(env)
        for substep in range(self._decimation):
            env.scene["robot"].set_joint_position_target(full_action)
            if self._use_self_torque:
                body_effort = self._compute_body_effort(target)
                env.scene["robot"].set_joint_effort_target(body_effort, joint_ids=self._body_idx)
            env.scene.write_data_to_sim()
            env.sim.step(render=should_render and substep == self._decimation - 1)
            env.scene.update(dt=env.physics_dt)

        if self._replay_enabled and replay_frame_idx is not None:
            self._log_replay_joint_position_error(replay_frame_idx)
            self._log_replay_object_state_error(replay_frame_idx)

        if self._debug and self._frame_count - self._last_tracking_log_frame >= self._log_every:
            self._last_tracking_log_frame = self._frame_count
            current = env.scene["robot"].data.joint_pos[0, self._body_idx].detach()
            err = (current - target).abs()
            print(
                f"[MimicLiteTracking] frame={self._frame_count} render={should_render} "
                f"current_range=({current.min().item():.3f},{current.max().item():.3f}) "
                f"target_range=({target.min().item():.3f},{target.max().item():.3f}) "
                f"mean_abs_err={err.mean().item():.3f} max_abs_err={err.max().item():.3f}"
            )

        return full_action.clone()

    def cleanup(self):
        try:
            self.recording_manager.shutdown()
        except Exception:
            pass
        if getattr(self, "_redis_client", None) is not None:
            try:
                self._redis_client.close()
            except Exception:
                pass
        if getattr(self, "_redis_control_client", None) is not None:
            try:
                self._redis_control_client.close()
            except Exception:
                pass
