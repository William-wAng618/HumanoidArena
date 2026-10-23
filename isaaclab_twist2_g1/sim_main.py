import sys
import coverage, coverage.types
print("PY:", sys.executable)
print("coverage:", coverage.__version__, coverage.__file__)
print("has Tracer:", hasattr(coverage.types, "Tracer"))
print("sys.path head:", sys.path[:8])
#exit()
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
#!/usr/bin/env python3
# main.py
import os

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

import argparse
import time
import sys
import signal
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from task_runtime_profiles import apply_task_runtime_profile

REPO_ROOT = Path(project_root).parent
DEFAULT_TWIST2_MODEL_PATH = os.environ.get(
    "TWIST2_MODEL_PATH",
    str(REPO_ROOT / "TWIST2" / "assets" / "ckpts" / "twist2_1017_20k.onnx"),
)
DEFAULT_SMPLX_MODEL_PATH = os.environ.get("SMPLX_MODEL_PATH", "")

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

from image_server.image_server import ImageServer
from dds.dds_create import create_dds_objects,create_dds_objects_replay
# add command line arguments
parser = argparse.ArgumentParser(description="Unitree Simulation")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G129-Head-Waist-Fix", help="task name")
parser.add_argument(
    "--env_config_yaml",
    type=str,
    default="",
    help="YAML file with env config overrides. Relative paths resolve under tasks/common_env_config.",
)
parser.add_argument("--action_source", type=str, default="dds",
                   choices=["dds", "file", "trajectory", "policy", "replay",
                            "twist2_wholebody", "sonic_wholebody", "mimic_lite_wholebody"],
                   help="Action source")
parser.add_argument(
    "--input_source",
    type=str,
    default="",
    choices=["", "pico_twist2", "pico_sonic", "pico_mimic_lite", "vla", "replay"],
    help="Optional high-level input source alias",
)
parser.add_argument(
    "--gmt_backend",
    type=str,
    default="",
    choices=["", "twist2", "sonic", "sonic_joint29", "sonic_low_latency", "mimic_lite"],
    help="Optional GMT backend alias",
)
parser.add_argument(
    "--replay_file",
    type=str,
    default="",
    help="Path to local replay npz. When set, the selected backend provider replays frames locally.",
)
parser.add_argument(
    "--replay_mode",
    type=str,
    default="inference_replay",
    choices=["direct", "inference", "direct_replay", "inference_replay"],
    help="Replay mode: direct_replay/direct uses recorded targets, inference_replay/inference reruns the backend model.",
)
parser.add_argument(
    "--replay_loop",
    action="store_true",
    default=False,
    help="Loop local replay frames when reaching the end of the npz.",
)
parser.add_argument(
    "--task_runtime_profile",
    "--task-runtime-profile",
    type=str,
    default="auto",
    choices=["auto", "inference", "replay_compat"],
    help="Task-specific runtime profile. 'auto' selects defaults from task + replay/rerecord context.",
)

# SONIC-specific arguments (used when action_source=sonic_wholebody)
parser.add_argument("--sonic_pose_source", type=str, default="redis",
                    choices=["redis", "zmq"],
                    help="Transport used by SonicActionProvider for Pico pose input")
parser.add_argument("--sonic_zmq_host", type=str, default="localhost",
                    help="ZMQ host for SONIC pose topic (pico_manager_thread_server)")
parser.add_argument("--sonic_zmq_port", type=int, default=5556,
                    help="ZMQ port for SONIC pose topic")
parser.add_argument("--sonic_redis_host", type=str, default="localhost",
                    help="Redis host for SONIC pose data")
parser.add_argument("--sonic_redis_port", type=int, default=6379,
                    help="Redis port for SONIC pose data")
parser.add_argument("--sonic_encoder_path", type=str, default="",
                    help="Path to GEAR-SONIC encoder ONNX model")
parser.add_argument("--sonic_decoder_path", type=str, default="",
                    help="Path to GEAR-SONIC decoder ONNX model")
parser.add_argument("--sonic_input_timeout_s", type=float, default=0.25,
                    help="Seconds without a fresh SONIC pose frame before holding the default standing pose; 0 disables the guard")

# VLA-specific arguments
parser.add_argument("--language_instruction", type=str, default="",
                    help="Language instruction for VLA")
parser.add_argument("--lerobot_policy_path", type=str, default="",
                    help="Path to a LeRobot pretrained_model directory")
parser.add_argument("--lerobot_policy_device", type=str, default="",
                    help="Inference device for LeRobot policy. Defaults to --device when empty.")
parser.add_argument("--lerobot_gripper_threshold", type=float, default=0.5,
                    help="Threshold for binarizing LeRobot grip outputs in vla mode")
parser.add_argument("--lerobot_server_url", type=str, default="",
                    help="HTTP(S) endpoint for remote LeRobot VLA inference")
parser.add_argument("--lerobot_server_timeout", type=float, default=5.0,
                    help="Timeout in seconds for LeRobot HTTP(S) inference requests")
parser.add_argument("--lerobot_server_verify_ssl", action="store_true", default=False,
                    help="Verify TLS certificates for LeRobot HTTPS connections")
parser.add_argument("--smplx_model_path", type=str,
                    default=DEFAULT_SMPLX_MODEL_PATH,
                    help="Path to SMPL-X model files")
parser.add_argument("--human_height", type=float, default=1.75,
                    help="Human height in meters for GMR scaling")
parser.add_argument("--twist2_model_path", type=str, default="",
                    help="TWIST2 policy used downstream of VLA. Defaults to --model_path when empty.")
parser.add_argument("--video_save_dir", type=str, default="./videos/vla",
                    help="Directory to save VLA videos")
parser.add_argument("--video_fps", type=int, default=30,
                    help="Video frame rate for VLA recording")
parser.add_argument("--enable_smpl_vis", action="store_true", default=True,
                    help="Enable SMPL visualization for VLA video recording")



# MimicLite-specific arguments
parser.add_argument("--mimic_lite_onnx_path", type=str, default="",
                    help="Path to exported MimicLite ONNX policy")
parser.add_argument("--mimic_lite_yaml_path", type=str, default="",
                    help="Path to exported MimicLite deploy YAML")
parser.add_argument("--mimic_lite_redis_host", type=str, default="localhost",
                    help="Redis host for MimicLite GMR reference input")
parser.add_argument("--mimic_lite_redis_port", type=int, default=6379,
                    help="Redis port for MimicLite GMR reference input")

parser.add_argument("--robot_type", type=str, default="unitree_g1_rotlocal_v3", help="robot type")
parser.add_argument("--enable_dex1_dds", action="store_true", help="enable gripper DDS")
parser.add_argument("--enable_dex3_dds", action="store_true", help="enable dexterous hand DDS")
parser.add_argument("--enable_inspire_dds", action="store_true", help="enable inspire hand DDS")
parser.add_argument("--stats_interval", type=float, default=10.0, help="statistics print interval (seconds)")

parser.add_argument("--file_path", type=str, default="", help="file path (when action_source=file)")
parser.add_argument("--generate_data_dir", type=str, default="./data", help="save data dir")
parser.add_argument("--generate_data", action="store_true", default=False, help="generate data")
parser.add_argument("--rerun_log", action="store_true", default=False, help="rerun log")
parser.add_argument("--replay_data",  action="store_true", default=False, help="replay data")

parser.add_argument("--modify_light",  action="store_true", default=False, help="modify light")
parser.add_argument("--modify_camera",  action="store_true", default=False,    help="modify camera")

# image streaming parameters
parser.add_argument(
    "--image_transport",
    type=str,
    default="zmq",
    choices=["zmq", "redis", "dds", "xrobot"],
    help="image transport for streaming (zmq/redis/dds/xrobot)",
)
parser.add_argument("--image_fps", type=int, default=30, help="image streaming fps cap")
parser.add_argument("--image_zmq_port", type=int, default=5555, help="ZMQ port for image streaming")
parser.add_argument("--image_redis_host", type=str, default="localhost", help="Redis host for image streaming")
parser.add_argument("--image_redis_port", type=int, default=6379, help="Redis port for image streaming")
parser.add_argument("--image_redis_db", type=int, default=0, help="Redis db for image streaming")
parser.add_argument(
    "--image_redis_key_prefix",
    type=str,
    default="isaac_image",
    help="Redis key prefix for image streaming",
)
parser.add_argument(
    "--image_redis_channel",
    type=str,
    default="",
    help="Redis pubsub channel (optional) for image streaming",
)
parser.add_argument("--image_dds_topic", type=str, default="rt/isaac_image", help="DDS topic for image streaming")
parser.add_argument("--image_xrobot_host", type=str, default="172.20.10.2", help="XRobot/Pico IP for image streaming")
parser.add_argument("--image_xrobot_port", type=int, default=12345, help="XRobot/Pico port for image streaming")
parser.add_argument("--image_xrobot_bitrate", type=int, default=4000000, help="XRobot/Pico H264 bitrate (bps)")
parser.add_argument("--image_xrobot_ffmpeg", type=str, default="", help="ffmpeg path for XRobot streaming")

# performance analysis parameters
parser.add_argument("--step_hz", type=int, default=500, help="control frequency")
parser.add_argument("--enable_profiling", action="store_true", default=True, help="enable performance analysis")
parser.add_argument("--profile_interval", type=int, default=500, help="performance analysis report interval (steps)")

parser.add_argument("--model_path", type=str, default=DEFAULT_TWIST2_MODEL_PATH, help="model path")
parser.add_argument("--enable_wholebody_dds", action="store_true", default=False, help="enable wh dds")
parser.add_argument("--setpgrp", action="store_true", default=False, help="detach to a new process group")

# recording parameters
parser.add_argument("--recording_save_dir", type=str, default="./recording_data", help="directory to save recording data")
parser.add_argument("--auto_start_recording", action="store_true", default=False, help="automatically start recording on startup (for testing from-reset reproducibility)")
parser.add_argument(
    "--record_during_replay",
    action="store_true",
    default=False,
    help="re-record a fresh episode while replaying an input npz; disabled by default to preserve legacy replay behavior",
)
parser.add_argument(
    "--exit_when_replay_complete",
    action="store_true",
    default=False,
    help="stop the main loop after replay reaches EOF (useful for batch replay/rerecord jobs)",
)
parser.add_argument(
    "--recording_save_workers",
    type=int,
    default=10,
    help="max concurrent background save workers (keep low to avoid CPU contention)",
)
parser.add_argument(
    "--recording_save_queue_size",
    type=int,
    default=10,
    help="max queued save jobs before producer blocks",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="stop after this many main-loop control steps; 0 keeps the legacy infinite loop",
)
parser.add_argument(
    "--eval_video_path",
    type=str,
    default="",
    help="optional mp4 path for a short eval recording from front_camera rgb frames",
)

# random seed for reproducibility
parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility (default: None)")

# world camera parameters
parser.add_argument("--enable_world_camera", action="store_true", default=False, help="enable world camera (third-person view)")
parser.add_argument(
    "--enable_perspective_camera",
    action="store_true",
    default=False,
    help="alias for --enable_world_camera; enables /World/PerspectiveCamera",
)
parser.add_argument("--world_camera_port", type=int, default=5556, help="ZMQ port for world camera streaming")
parser.add_argument(
    "--enable_wrist_cameras",
    action="store_true",
    default=False,
    help="enable left/right wrist camera sensors and streams",
)
parser.add_argument(
    "--disable_front_camera",
    "--disable-front-camera",
    action="store_true",
    default=False,
    help="disable the front camera sensor and stream even when camera rendering is enabled",
)
parser.add_argument(
    "--disable_wrist_cameras",
    "--disable-wrist-cameras",
    action="store_true",
    default=False,
    help="disable left/right wrist camera sensors and streams",
)
parser.add_argument("--left_wrist_camera_port", type=int, default=5557, help="ZMQ port for left wrist camera streaming")
parser.add_argument("--right_wrist_camera_port", type=int, default=5558, help="ZMQ port for right wrist camera streaming")
parser.add_argument(
    "--viewport_camera",
    type=str,
    default="world",
    choices=["world", "front", "auto"],
    help="GUI viewport camera preference: world third-person, front robot camera, or auto.",
)

# add AppLauncher parameters
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "enable_perspective_camera", False):
    args_cli.enable_world_camera = True
if getattr(args_cli, "disable_wrist_cameras", False) and getattr(args_cli, "enable_wrist_cameras", False):
    print("[sim_main] disable_wrist_cameras requested; overriding enable_wrist_cameras")
    args_cli.enable_wrist_cameras = False
if getattr(args_cli, "enable_world_camera", False) and not getattr(args_cli, "enable_cameras", False):
    print("[sim_main] enable_world_camera requested; forcing enable_cameras for sensor rendering")
    args_cli.enable_cameras = True
if getattr(args_cli, "enable_wrist_cameras", False) and not getattr(args_cli, "enable_cameras", False):
    print("[sim_main] enable_wrist_cameras requested; forcing enable_cameras for sensor rendering")
    args_cli.enable_cameras = True
if getattr(args_cli, "record_during_replay", False) and not (
    getattr(args_cli, "input_source", "") == "replay" or getattr(args_cli, "replay_file", "")
):
    raise ValueError("--record_during_replay requires replay input (--replay_file / input_source=replay)")

apply_task_runtime_profile(args_cli)


if args_cli.enable_dex3_dds and args_cli.enable_dex1_dds and args_cli.enable_inspire_dds:
    print("Error: enable_dex3_dds and enable_dex1_dds and enable_inspire_dds cannot be enabled at the same time")
    print("Please select one of the options")
    sys.exit(1)


# import pinocchio  # 注释掉：与 NumPy 2.x 不兼容，且当前未使用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from layeredcontrol.robot_control_system import (
    RobotController, 
    ControlConfig,
)

from dds.reset_pose_dds import *

# Bind MimicLite-aligned PD actuator config (G129_CFG_MIMIC_LITE) to gmt_backend=mimic_lite.
# This must happen BEFORE `import tasks` because task env_cfg class attributes
# (robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody()) are evaluated at
# module-import time, and get_base_config() reads MIMIC_LITE_ROBOT_CFG then.
# Training-side PD from mimic-lite/assets/g1.py:
#   stiffness = armature * (10*2*pi)^2, damping = 2*2*armature*(10*2*pi), friction=0.01
# Other backends (twist2/sonic/sonic_low_latency) keep G129_CFG_WITH_DEX3_WHOLEBODY.
if getattr(args_cli, "gmt_backend", "") == "mimic_lite" or getattr(args_cli, "action_source", "") == "mimic_lite_wholebody":
    os.environ["MIMIC_LITE_ROBOT_CFG"] = "1"
    print("[sim_main] gmt_backend=mimic_lite -> MIMIC_LITE_ROBOT_CFG=1 (BeyondMimic PD actuator config)")

import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from tools.data_json_load import sim_state_to_json
from dds.sim_state_dds import *
from action_provider.action_base import ReplayComplete
from action_provider.create_action_provider import create_action_provider
from action_provider.reset_control import (
    clear_reset_trigger,
    create_redis_client,
    publish_input_ready,
    publish_reset_complete,
    read_reset_trigger,
)
from common_env_objects import (
    apply_deterministic_object_resets_with_seed,
    apply_explicit_env_object_states as _apply_explicit_env_object_states_common,
    get_recordable_env_object_specs,
)
from tasks.common_env_config import apply_env_config_yaml
from tasks.common_runtime import (
    apply_optional_runtime_augments,
)
from tools.get_stiffness import get_robot_stiffness_from_env
from tools.get_reward import get_reward_debug_string
# Use text-based tracker instead of GUI visualizer to avoid matplotlib issues
from tools.joint_position_tracker import JointPositionTracker


def _capture_front_camera_rgb(env):
    try:
        if "front_camera" not in env.scene.keys():
            return None
        camera = env.scene["front_camera"]
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            return None
        frame = rgb[0].detach().cpu().numpy()
        if frame.ndim != 3:
            return None
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame
    except Exception:
        return None


class _EvalVideoWriter:
    def __init__(self, path: str, fps: int):
        self.path = Path(path).expanduser()
        self.fps = int(fps)
        self.frames = []

    def add(self, frame):
        if frame is not None:
            self.frames.append(frame.copy())

    def close(self):
        if not self.path or not self.frames:
            return ""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio
            imageio.mimsave(str(self.path), self.frames, fps=self.fps)
            return str(self.path)
        except Exception as exc:
            print(f"[eval_video] failed to save {self.path}: {exc}")
            return ""


def _normalize_control_routing(args_cli):
    """Normalize input_source/gmt_backend aliases back to the legacy action_source."""
    if args_cli.input_source and args_cli.gmt_backend:
        route_map = {
            ("pico_twist2", "twist2"): "twist2_wholebody",
            ("pico_sonic", "sonic"): "sonic_wholebody",
            ("pico_sonic", "sonic_low_latency"): "sonic_wholebody",
            ("pico_twist2", "sonic_joint29"): "sonic_wholebody",
            ("pico_mimic_lite", "mimic_lite"): "mimic_lite_wholebody",
            ("vla", "twist2"): "twist2_wholebody",
            ("vla", "sonic"): "sonic_wholebody",
            ("vla", "sonic_low_latency"): "sonic_wholebody",
            ("vla", "mimic_lite"): "mimic_lite_wholebody",
            ("replay", "twist2"): "twist2_wholebody",
            ("replay", "sonic"): "sonic_wholebody",
            ("replay", "sonic_low_latency"): "sonic_wholebody",
            ("replay", "sonic_joint29"): "sonic_wholebody",
            ("replay", "mimic_lite"): "mimic_lite_wholebody",
        }
        mapped = route_map.get((args_cli.input_source, args_cli.gmt_backend))
        if mapped is None:
            raise ValueError(
                f"Unsupported input/gmt route: input_source={args_cli.input_source}, "
                f"gmt_backend={args_cli.gmt_backend}"
            )
        args_cli.action_source = mapped
    elif args_cli.action_source == "sonic_wholebody":
        args_cli.input_source = args_cli.input_source or "pico_sonic"
        args_cli.gmt_backend = args_cli.gmt_backend or "sonic"
    elif args_cli.action_source == "twist2_wholebody":
        args_cli.input_source = args_cli.input_source or "pico_twist2"
        args_cli.gmt_backend = args_cli.gmt_backend or "twist2"
    elif args_cli.action_source == "mimic_lite_wholebody":
        args_cli.input_source = args_cli.input_source or "pico_mimic_lite"
        args_cli.gmt_backend = args_cli.gmt_backend or "mimic_lite"
    elif args_cli.action_source == "replay":
        args_cli.input_source = args_cli.input_source or "replay"
        if args_cli.gmt_backend == "twist2":
            args_cli.action_source = "twist2_wholebody"
        elif args_cli.gmt_backend in ("sonic", "sonic_low_latency"):
            args_cli.action_source = "sonic_wholebody"
        elif args_cli.gmt_backend == "sonic_joint29":
            args_cli.action_source = "sonic_wholebody"
        elif args_cli.gmt_backend == "mimic_lite":
            args_cli.action_source = "mimic_lite_wholebody"


def _initialize_task_scene(env, env_cfg, args_cli):
    try:
        initialize_task_scene = getattr(env_cfg, "initialize_task_scene", None)
        if callable(initialize_task_scene):
            initialize_task_scene(env, args_cli)
        else:
            apply_optional_runtime_augments(args_cli)
            legacy_runtime_setup = getattr(env_cfg, "apply_runtime_setup", None)
            if callable(legacy_runtime_setup):
                legacy_runtime_setup(env, args_cli)
    except Exception as exc:
        print(f"[env_runtime] init setup failed: {exc}")
    try:
        setup_vision_test_light_from_cfg(env_cfg)
    except Exception as exc:
        print(f"[vision_randomization] init setup failed: {exc}")



def _resolve_wrist_camera_pair(args_cli, env_cfg):
    from tasks.common_config import CameraPresets

    usd_path = ""
    try:
        usd_path = str(env_cfg.scene.robot.spawn.usd_path).lower()
    except Exception:
        usd_path = ""

    if getattr(args_cli, "enable_dex3_dds", False) or "dex3" in usd_path:
        return CameraPresets.left_dex3_wrist_camera(), CameraPresets.right_dex3_wrist_camera(), "dex3"
    if getattr(args_cli, "enable_inspire_dds", False) or "inspire" in usd_path:
        return CameraPresets.left_inspire_wrist_camera(), CameraPresets.right_inspire_wrist_camera(), "inspire"
    if getattr(args_cli, "enable_dex1_dds", False) or "dex1" in usd_path or "gripper" in usd_path:
        return CameraPresets.left_gripper_wrist_camera(), CameraPresets.right_gripper_wrist_camera(), "gripper"
    # Most replay/live G1 whole-body tasks in this repo use Dex3 hands.
    return CameraPresets.left_dex3_wrist_camera(), CameraPresets.right_dex3_wrist_camera(), "dex3(default)"


def _augment_env_cfg_with_wrist_cameras(env_cfg, args_cli):
    if not getattr(args_cli, "enable_wrist_cameras", False):
        return

    left_cfg, right_cfg, hand_variant = _resolve_wrist_camera_pair(args_cli, env_cfg)
    env_cfg.scene.left_wrist_camera = left_cfg
    env_cfg.scene.right_wrist_camera = right_cfg
    print(
        "[sim_main] Wrist cameras enabled in scene config "
        f"(variant={hand_variant}, left={left_cfg.prim_path}, right={right_cfg.prim_path})"
    )


def _disable_env_cfg_front_camera(env_cfg, args_cli):
    if not getattr(args_cli, "disable_front_camera", False):
        return

    scene_cfg = getattr(env_cfg, "scene", None)
    if scene_cfg is None:
        raise ValueError("disable_front_camera requested but env_cfg has no scene config")
    if getattr(scene_cfg, "front_camera", None) is None:
        print("[sim_main] Front camera disabled; scene config has no front_camera")
        return

    scene_cfg.front_camera = None
    print("[sim_main] Front camera disabled in scene config")


def _augment_env_cfg_with_perspective_camera(env_cfg, args_cli):
    if not getattr(args_cli, "enable_world_camera", False):
        return

    from tasks.common_config import CameraPresets

    scene_cfg = getattr(env_cfg, "scene", None)
    if scene_cfg is None:
        raise ValueError("PerspectiveCamera requested but env_cfg has no scene config")
    existing_camera = getattr(scene_cfg, "world_camera", None)
    if existing_camera is not None:
        print(
            "[sim_main] PerspectiveCamera enabled using existing scene config "
            f"(world_camera={existing_camera.prim_path})"
        )
        return

    scene_cfg.world_camera = CameraPresets.g1_world_camera()
    print(
        "[sim_main] PerspectiveCamera enabled using default scene config "
        f"(world_camera={scene_cfg.world_camera.prim_path})"
    )


def _create_image_server(args_cli, *, port: int, camera_name: str, redis_suffix: str, dds_suffix: str, xrobot_port_offset: int):
    redis_channel = args_cli.image_redis_channel + redis_suffix if args_cli.image_redis_channel else ""
    return ImageServer(
        fps=args_cli.image_fps,
        port=port,
        Unit_Test=False,
        transport=args_cli.image_transport,
        redis_host=args_cli.image_redis_host,
        redis_port=args_cli.image_redis_port,
        redis_db=args_cli.image_redis_db,
        redis_key_prefix=args_cli.image_redis_key_prefix + redis_suffix,
        redis_channel=redis_channel,
        dds_topic=args_cli.image_dds_topic + dds_suffix,
        xrobot_host=args_cli.image_xrobot_host,
        xrobot_port=args_cli.image_xrobot_port + xrobot_port_offset,
        xrobot_bitrate=args_cli.image_xrobot_bitrate,
        xrobot_width=None,
        xrobot_height=None,
        xrobot_ffmpeg=args_cli.image_xrobot_ffmpeg or None,
        camera_name=camera_name,
    )


def _trigger_task_reset_event(env_cfg, event_name, env):
    event_manager = getattr(env_cfg, "event_manager", None)
    triggered = False
    if event_manager is not None:
        event_manager.trigger(event_name, env)
        triggered = True
    _apply_vision_light_randomization_for_reset(env_cfg)
    return triggered


def _apply_vision_light_randomization_for_reset(env_cfg, *, episode_seed=None, seed_source=None):
    try:
        return apply_vision_light_randomization_from_cfg(
            env_cfg,
            episode_seed=episode_seed,
            seed_source=seed_source,
        )
    except Exception as exc:
        print(f"[vision_randomization] reset hook failed: {exc}")
        return False


def _resolve_input_guard_backend(args_cli):
    if getattr(args_cli, "input_source", "") == "replay" or getattr(args_cli, "replay_file", ""):
        return None
    if getattr(args_cli, "gmt_backend", "") == "twist2" or getattr(args_cli, "action_source", "") == "twist2_wholebody":
        return "twist2"
    if getattr(args_cli, "gmt_backend", "") == "sonic" or getattr(args_cli, "action_source", "") == "sonic_wholebody":
        if getattr(args_cli, "gmt_backend", "") == "sonic_low_latency":
            return "sonic_low_latency"
        return "sonic"
    if getattr(args_cli, "gmt_backend", "") == "sonic_joint29":
        return "sonic_joint29"
    if getattr(args_cli, "gmt_backend", "") == "mimic_lite":
        return "mimic_lite"
    return None


def _notify_action_provider_env_reset(action_provider):
    if action_provider is None:
        return
    for method_name in ("on_env_reset", "_reset_internal_buffers"):
        method = getattr(action_provider, method_name, None)
        if callable(method):
            method()
            return


def _notify_action_provider_env_objects_reset(action_provider):
    if action_provider is None:
        return
    method = getattr(action_provider, "on_env_objects_reset", None)
    if callable(method):
        method()


def _should_exit_after_replay_complete(action_provider, args_cli) -> bool:
    if action_provider is None or not getattr(args_cli, "exit_when_replay_complete", False):
        return False
    env = getattr(action_provider, "env", None)
    if env is not None and bool(getattr(env, "_request_main_loop_exit", False)):
        return True
    method = getattr(action_provider, "should_exit_after_replay_complete", None)
    if callable(method):
        try:
            return bool(method())
        except Exception as exc:
            print(f"[sim_main] replay completion check failed: {exc}")
            return False
    return False


def _publish_backend_input_ready(args_cli, *, redis_client=None, source="startup"):
    backend = _resolve_input_guard_backend(args_cli)
    if backend is None:
        return None
    payload = publish_input_ready(backend=backend, redis_client=redis_client, source=source)
    print(
        f"[input_guard] backend={backend} source={source} "
        f"ready_timestamp_ms={payload.get('ready_timestamp_ms')} epoch_id={payload.get('epoch_id')}"
    )
    return payload


def _load_replay_env_init_cache(args_cli):
    replay_file = getattr(args_cli, "replay_file", "")
    if not replay_file:
        return {"object_states": {}, "episode_object_seed": None, "episode_object_seed_source": ""}

    cached = getattr(args_cli, "_replay_env_init_cache", None)
    if cached is not None:
        return cached

    replay_path = Path(replay_file).expanduser().resolve()
    object_states = {}
    episode_object_seed = None
    episode_object_seed_source = ""
    suffixes = {
        "_position": "position",
        "_orientation": "orientation",
        "_linear_velocity": "linear_velocity",
        "_angular_velocity": "angular_velocity",
    }

    with np.load(replay_path, allow_pickle=True) as replay_data:
        for key in replay_data.files:
            if not key.startswith("episode_init_env_obj_"):
                continue
            for suffix, field_name in suffixes.items():
                if key.endswith(suffix):
                    object_name = key[len("episode_init_env_obj_") : -len(suffix)]
                    object_states.setdefault(object_name, {})[field_name] = np.asarray(replay_data[key]).copy()
                    break

        if not object_states:
            for key in replay_data.files:
                if not key.startswith("env_obj_"):
                    continue
                for suffix, field_name in suffixes.items():
                    if key.endswith(suffix):
                        object_name = key[len("env_obj_") : -len(suffix)]
                        value = np.asarray(replay_data[key]).copy()
                        if value.ndim >= 1:
                            value = value[0].copy()
                        object_states.setdefault(object_name, {})[field_name] = value
                        break
            if object_states:
                print(
                    "[replay_env_init] episode_init_env_obj_* missing, "
                    "falling back to frame-0 env_obj_* state from replay file"
                )

        if "episode_object_seed" in replay_data:
            episode_object_seed = int(np.asarray(replay_data["episode_object_seed"]).item())
        if "episode_object_seed_source" in replay_data:
            episode_object_seed_source = str(np.asarray(replay_data["episode_object_seed_source"]).item())

    cache = {
        "object_states": object_states,
        "episode_object_seed": episode_object_seed,
        "episode_object_seed_source": episode_object_seed_source,
    }
    setattr(args_cli, "_replay_env_init_cache", cache)
    return cache


def _load_replay_initial_env_object_states(args_cli):
    return _load_replay_env_init_cache(args_cli)["object_states"]


def _load_replay_episode_object_seed_info(args_cli):
    cache = _load_replay_env_init_cache(args_cli)
    return cache.get("episode_object_seed"), cache.get("episode_object_seed_source", "")

def _apply_explicit_env_object_states(env, env_cfg, object_states, *, log_prefix="replay_env_init"):
    return _apply_explicit_env_object_states_common(env, env_cfg, object_states, log_prefix=log_prefix)


def _restore_replay_initial_env_state_if_needed(env, args_cli):
    object_states = _load_replay_initial_env_object_states(args_cli)
    specs = get_recordable_env_object_specs(env.cfg)
    configured_names = {
        str(spec.get("record_name"))
        for spec in specs
        if spec.get("record_name")
    }
    spec_by_name = {
        str(spec.get("record_name")): spec
        for spec in specs
        if spec.get("record_name")
    }
    episode_seed, episode_seed_source = _load_replay_episode_object_seed_info(args_cli)
    seed_applied_names: set[str] = set()
    task_handled_names: set[str] = set()

    restore_replay_initial_env_state = getattr(env.cfg, "restore_replay_initial_env_state", None)
    if callable(restore_replay_initial_env_state):
        try:
            handled_names = restore_replay_initial_env_state(
                env,
                episode_seed=episode_seed,
                episode_seed_source=episode_seed_source or "recorded",
                object_states=object_states,
            )
            if handled_names:
                task_handled_names = {str(name) for name in handled_names}
                print(
                    "[replay_env_init] task-specific restore handled "
                    f"{sorted(task_handled_names)}"
                )
        except Exception as exc:
            print(f"[replay_env_init] task-specific restore failed: {exc}")

    # For prim-path objects such as the doubledesk basket, explicit world-pose snapshots can be
    # polluted by authored xform stacks on the referenced USD. When a recorded episode seed exists,
    # rebuild these objects through the same deterministic reset path used during recording.
    seed_preferred_names = {
        name
        for name, spec in spec_by_name.items()
        if spec.get("prim_paths") and not spec.get("scene_keys") and name not in task_handled_names
    }
    if episode_seed is not None and seed_preferred_names:
        applied = apply_deterministic_object_resets_with_seed(
            env.cfg,
            env,
            episode_seed=episode_seed,
            seed_source=episode_seed_source or "recorded",
            selected_record_names={str(name) for name in seed_preferred_names},
        )
        if applied:
            seed_applied_names = {
                entry.split("->", 1)[0]
                for entry in applied
                if "->" in entry
            }
            print(
                "[replay_env_init] rebuilt prim-path object init state from "
                f"episode_object_seed={episode_seed} source={episode_seed_source or 'recorded'} "
                f"for {sorted(seed_applied_names)}"
            )

    explicit_object_states = {
        name: state
        for name, state in object_states.items()
        if name not in seed_applied_names and name not in task_handled_names
    }
    missing_names = configured_names - seed_applied_names - task_handled_names - set(explicit_object_states)

    if missing_names and episode_seed is not None:
        applied = apply_deterministic_object_resets_with_seed(
            env.cfg,
            env,
            episode_seed=episode_seed,
            seed_source=episode_seed_source or "recorded",
            selected_record_names={str(name) for name in missing_names},
        )
        if applied:
            print(
                "[replay_env_init] replay file missing explicit init state for "
                f"{sorted(missing_names)}; rebuilt from recorded episode_object_seed={episode_seed} "
                f"source={episode_seed_source or 'recorded'}"
            )

    if episode_seed is not None:
        _apply_vision_light_randomization_for_reset(
            env.cfg,
            episode_seed=episode_seed,
            seed_source=episode_seed_source or "recorded",
        )

    if not explicit_object_states:
        return bool(seed_applied_names or task_handled_names)
    return _apply_explicit_env_object_states(env, env.cfg, explicit_object_states)


def _perform_complete_reset(env, env_cfg, args_cli, redis_client=None, action_provider=None):
    print("🔄 Complete reset (PhysX + all entities)...")
    env.sim.reset()
    env.reset()
    _trigger_task_reset_event(env_cfg, "reset_all_self", env)
    _restore_replay_initial_env_state_if_needed(env, args_cli)
    _notify_action_provider_env_reset(action_provider)
    _publish_backend_input_ready(args_cli, redis_client=redis_client, source="complete_reset")
    print("✅ Complete reset finished")
    try:
        publish_reset_complete(redis_client=redis_client)
        print("✅ Reset complete signal sent via Redis")
    except Exception as exc:
        print(f"❌ Failed to send reset complete signal: {exc}")


def setup_signal_handlers(controller, dds_manager=None, image_servers=None, simulation_app=None):
    """set signal handlers

    Args:
        controller: robot controller instance
        dds_manager: DDS manager instance
        image_servers: list of ImageServer instances or single ImageServer
        simulation_app: simulation app instance
    """
    _handling = {"in_progress": False}
    def signal_handler(signum, frame):
        print(f"\nreceived signal {signum}, stopping controller...")
        # Prevent running cleanup multiple times (Ctrl-C can be pressed repeatedly)
        if _handling["in_progress"]:
            print("signal handler already running; forcing exit")
            os._exit(0)
        _handling["in_progress"] = True
        try:
            controller.stop()
        except Exception as e:
            print(f"Failed to stop controller: {e}")

        # Close image servers (support single or multiple servers)
        try:
            if image_servers is not None:
                # Handle both single server and list of servers
                servers_list = image_servers if isinstance(image_servers, list) else [image_servers]
                for idx, server in enumerate(servers_list):
                    if server is not None:
                        try:
                            server._close()
                            print(f"[sim_main] Image server {idx} closed successfully")
                        except Exception as e:
                            print(f"[sim_main] Failed to close image server {idx}: {e}")
        except Exception as e:
            print(f"Failed to stop image servers: {e}")

        # Clean up global shared memory writer from camera_state.py
        try:
            from tasks.common_observations.camera_state import multi_image_writer
            print("[sim_main] Cleaning up global camera multi_image_writer...")
            multi_image_writer.cleanup()
        except Exception as e:
            print(f"[sim_main] Failed to cleanup camera shared memory: {e}")

        try:
            if dds_manager is not None:
                dds_manager.stop_all_communication()
        except Exception as e:
            print(f"Failed to stop DDS: {e}")
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception as e:
            print(f"Failed to close simulation app: {e}")
        # If any background threads (DDS / OmniKit) keep the process alive, force exit.
        try:
            import os as _os
            _os._exit(0)
        except Exception:
            raise SystemExit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def _close_image_servers(image_servers):
    try:
        if image_servers is None:
            return
        servers_list = image_servers if isinstance(image_servers, list) else [image_servers]
        for idx, server in enumerate(servers_list):
            if server is None:
                continue
            try:
                server._close()
                print(f"[sim_main] Image server {idx} closed successfully")
            except Exception as exc:
                print(f"[sim_main] Failed to close image server {idx}: {exc}")
    except Exception as exc:
        print(f"[sim_main] Failed to stop image servers: {exc}")

def main():
    """main function"""
    import os
    import atexit
    _normalize_control_routing(args_cli)
    local_gmt_replay = bool(args_cli.replay_file) and args_cli.input_source == "replay" and args_cli.gmt_backend in {
        "twist2",
        "sonic",
        "sonic_joint29",
        "sonic_low_latency",
        "mimic_lite",
    }
    image_servers = None
    dds_manager = None
    try:
        if args_cli.setpgrp:
            os.setpgrp()
            current_pgid = os.getpgrp()
            print(f"Setting process group: {current_pgid}")
        
            def cleanup_process_group():
                try:
                    print(f"Cleaning up process group: {current_pgid}")
                    import signal
                    os.killpg(current_pgid, signal.SIGTERM)
                except Exception as e:
                    print(f"Failed to clean up process group: {e}")
        
            atexit.register(cleanup_process_group)
        
    except Exception as e:
        print(f"Failed to set process group: {e}")
    print("=" * 60)
    print("robot control system started")
    print(f"Task: {args_cli.task}")
    print(f"Action source: {args_cli.action_source}")
    if args_cli.input_source or args_cli.gmt_backend:
        print(f"Input source: {args_cli.input_source or 'legacy'}")
        print(f"GMT backend: {args_cli.gmt_backend or 'legacy'}")
    if args_cli.input_source == "vla":
        if args_cli.gmt_backend and args_cli.gmt_backend not in {"twist2", "sonic", "sonic_joint29", "sonic_low_latency", "mimic_lite"}:
            raise ValueError("input_source=vla currently only supports --gmt_backend twist2, sonic, or sonic_joint29")
        if not args_cli.lerobot_server_url and not args_cli.lerobot_policy_path:
            raise ValueError("--lerobot_server_url or --lerobot_policy_path is required when using input_source=vla")
        if float(args_cli.human_height) <= 0.0:
            raise ValueError("--human_height must be positive when using input_source=vla")
        print("VLA runtime schema: unitree_g1_gmt_refpose_v3_1, observation.state=64D, action=40D ref-pose local output")
    print("=" * 60)

    # parse environment configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
        _augment_env_cfg_with_perspective_camera(env_cfg, args_cli)
        apply_env_config_yaml(
            env_cfg,
            args_cli.env_config_yaml,
            task_name=args_cli.task,
            route_name=args_cli.gmt_backend or args_cli.action_source,
        )
        _disable_env_cfg_front_camera(env_cfg, args_cli)
        _augment_env_cfg_with_wrist_cameras(env_cfg, args_cli)
        # Set seed: command line argument takes priority, otherwise use default 42
        seed_value = args_cli.seed if args_cli.seed is not None else 42
        env_cfg.seed = seed_value
        print(f"[CONFIG] Setting environment seed: {seed_value}")
    except Exception as e:
        print(f"Failed to parse environment configuration: {e}")
        return
    
    # create environment
    print("\ncreate environment...")
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env._enable_world_camera_stream = bool(getattr(args_cli, "enable_world_camera", False))
        env._request_main_loop_exit = False
        print(f"\ncreate environment success ...")
        print("robot cfg init pos:", env.cfg.scene.robot.init_state.pos)
        print("robot usd:", env.cfg.scene.robot.spawn.usd_path)

        # Set rendering mode via viewport API
        # try:
        #     from omni.kit.viewport.utility import get_active_viewport
        #     viewport = get_active_viewport()
        #     if viewport:
        #         # Switch to Real-Time mode (RaytracedLighting) - fastest RTX mode
        #         viewport.set_hd_engine("rtx", "RaytracedLighting")
        #         print("[RENDER] ✅ Switched to Real-Time (RaytracedLighting) mode via viewport API")
        #         # === Other render modes (commented out for comparison) ===
        #         # viewport.set_hd_engine("rtx", "PathTracing")        # Path Tracing (slower, higher quality)
        #         # viewport.set_hd_engine("iray", "iray")              # iray (not available in Isaac Sim)
        #
        #         # Verify viewport settings
        #         print(f"[RENDER] Viewport hydra_engine: {viewport.hydra_engine}")
        #         print(f"[RENDER] Viewport render_mode: {viewport.render_mode}")
        #
        #         # Print current render settings
        #         import carb
        #         settings = carb.settings.get_settings()
        #         print("\n" + "="*60)
        #         print(" REAL-TIME RENDER PARAMETERS")
        #         print("="*60)
        #         print(f"  /rtx/rendermode: {settings.get('/rtx/rendermode')}")
        #         print(f"  Antialiasing: DLAA (configured in env_cfg)")
        #         print("="*60 + "\n")
        #
        #         # === Path Tracing code (commented out for comparison) ===
        #         # viewport.set_hd_engine("rtx", "PathTracing")
        #         # print("[RENDER] ✅ Switched to Path Tracing mode via viewport API")
        #         # settings.set("/rtx/pathtracing/spp", 1)  # Samples per pixel per frame = 1
        #         # settings.set("/rtx/pathtracing/totalSpp", 1)  # Total samples per pixel
        #         # settings.set("/rtx/pathtracing/maxBounces", 4)  # Max light bounces
        #         # pt_settings = [
        #         #     "/rtx/rendermode",
        #         #     "/rtx/pathtracing/spp",
        #         #     "/rtx/pathtracing/totalSpp",
        #         #     "/rtx/pathtracing/maxBounces",
        #         #     "/rtx/pathtracing/maxSpecularAndTransmissionBounces",
        #         #     "/rtx/pathtracing/maxVolumeBounces",
        #         #     "/rtx/pathtracing/clampSpp",
        #         #     "/rtx/pathtracing/optixDenoiser/enabled",
        #         #     "/rtx/pathtracing/cached/enabled",
        #         #     "/rtx/pathtracing/aa/op",
        #         # ]
        #         # for setting in pt_settings:
        #         #     value = settings.get(setting)
        #         #     print(f"  {setting}: {value}")
        #         # === End Path Tracing code ===
        #     else:
        #         print("[RENDER] ⚠️ No active viewport found, cannot set render mode")
        # except Exception as e:
        #     print(f"[RENDER] ⚠️ Failed to set render mode: {e}")
    except Exception as e:
        print(f"\nFailed to create environment: {e}")
        return
    
    # get robot stiffness and damping parameters from runtime environment
    print("\n" + "="*60)
    print("🔍 Getting robot stiffness and damping parameters from runtime environment")
    print("="*60)
    
    try:
        stiffness_data = get_robot_stiffness_from_env(env)
        if stiffness_data:
            print("✅ Successfully got robot parameters!")
        else:
            print("⚠️ Failed to get robot parameters, will try again after environment reset")
    except Exception as e:
        print(f"⚠️ Error getting robot parameters: {e}")
    
    print("="*60)
    
    print("\n")
    print("***  Please left-click on the Sim window to activate rendering. ***")
    print("\n")
    _initialize_task_scene(env, env_cfg, args_cli)
    env.sim.reset()
    env.reset()
    if getattr(env_cfg, "startup_task_reset_enabled", True):
        try:
            _trigger_task_reset_event(env_cfg, "reset_all_self", env)
            _restore_replay_initial_env_state_if_needed(env, args_cli)
            debug_after_startup_reset = getattr(env_cfg, "debug_after_startup_reset", None)
            if callable(debug_after_startup_reset):
                debug_after_startup_reset(env, args_cli)
        except Exception as exc:
            print(f"[env_runtime] startup reset_all_self failed: {exc}")
    else:
        print("[env_runtime] startup reset_all_self skipped by task config")

    # --- set default viewport camera (GUI only) ---
    try:
        # Viewport camera switching only makes sense when rendering is enabled
        import omni.kit.viewport.utility as vp_utils
        from pxr import Usd, UsdGeom

        # Debug: List all cameras in the scene
        stage = env.sim.stage
        print("\n🔍 [DEBUG] Available cameras in USD stage:")
        camera_paths = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                cam_path_str = str(prim.GetPath())
                camera_paths.append(cam_path_str)
                print(f"  - {cam_path_str}")

        if not camera_paths:
            print("  ⚠️ No cameras found in stage!")

        # Select the GUI viewport camera. This does not change image streaming cameras.
        world_cam_candidates = [p for p in camera_paths if "PerspectiveCamera" in p or "world_camera" in p.lower()]
        front_cam_candidates = [p for p in camera_paths if "front_cam" in p.lower()]

        viewport_camera = getattr(args_cli, "viewport_camera", "world")
        if viewport_camera == "front" and front_cam_candidates:
            cam_path = front_cam_candidates[0]
            print(f"\n✅ Found front_cam at: {cam_path}")
        elif viewport_camera == "front":
            cam_path = "/World/envs/env_0/Robot/d435_link/front_cam"
            print(f"\n⚠️ No front_cam found, trying expected path: {cam_path}")
        elif world_cam_candidates:
            cam_path = world_cam_candidates[0]
            print(f"\n✅ Found world_camera at: {cam_path}")
        elif front_cam_candidates:
            cam_path = front_cam_candidates[0]
            print(f"\n✅ Found front_cam at: {cam_path}")
        else:
            # Fallback to the expected path
            cam_path = "/World/envs/env_0/Robot/d435_link/front_cam"
            print(f"\n⚠️ No camera found, trying expected path: {cam_path}")

        vp = vp_utils.get_active_viewport()
        if vp is not None:
            # Check current camera before switching
            current_cam = vp.get_active_camera()
            print(f"  Current viewport camera: {current_cam}")

            # Try to set the camera
            vp.set_active_camera(cam_path)

            # Verify the switch
            new_cam = vp.get_active_camera()
            if new_cam == cam_path:
                print(f"  ✅ Successfully switched viewport to: {cam_path}")
            else:
                print(f"  ❌ Failed to switch! Viewport still at: {new_cam}")
        else:
            print("⚠️ No active viewport found; skip setting default camera")
    except Exception as e:
        # Likely running headless / no viewport
        import traceback
        print(f"⚠️ Failed to set viewport default camera: {e}")
        print(traceback.format_exc())


    # create simplified control configuration
    try:
        wholebody_sources = {"twist2_wholebody", "sonic_wholebody", "mimic_lite_wholebody"}
        if args_cli.action_source == "sonic_wholebody":
            use_wholebody = True
            physics_dt = getattr(env, "physics_dt", None) or env_cfg.sim.dt
            policy_hz = int(round(1.0 / (physics_dt * env_cfg.decimation)))
            if args_cli.step_hz != policy_hz:
                print(f"⚠️  Overriding step_hz {args_cli.step_hz} -> {policy_hz} to match TWIST2 policy rate")
            step_hz = policy_hz
        elif args_cli.action_source in wholebody_sources:
            use_wholebody = True
            if args_cli.action_source == "twist2_wholebody":
                args_cli.enable_wholebody_dds = True
            physics_dt = getattr(env, "physics_dt", None) or env_cfg.sim.dt
            policy_hz = int(round(1.0 / (physics_dt * env_cfg.decimation)))
            step_hz = policy_hz
            print(
                f"✅ Wholebody runtime: source={args_cli.action_source}, step_hz={step_hz}Hz "
                f"(physics_dt={physics_dt}s, decimation={env_cfg.decimation})"
            )
        elif "Wholebody" in args_cli.task or args_cli.enable_wholebody_dds:
            use_wholebody = True
            args_cli.action_source = "twist2_wholebody"
            args_cli.enable_wholebody_dds = True
            # Match step_hz with physics frequency for TWIST2
            physics_dt = getattr(env, "physics_dt", None) or env_cfg.sim.dt
            policy_hz = int(round(1.0 / (physics_dt * env_cfg.decimation)))
            step_hz = policy_hz  # Use physics frequency instead of args_cli.step_hz
            print(f"✅ TWIST2 Wholebody: step_hz set to {step_hz}Hz (physics_dt={physics_dt}s, decimation={env_cfg.decimation})")
        else:
            use_wholebody = False
            step_hz = args_cli.step_hz

        control_config = ControlConfig(
            step_hz=step_hz,
            replay_mode=args_cli.replay_data,
            use_rl_action_mode=use_wholebody,
        )
    except Exception as e:
        print(f"Failed to create control configuration: {e}")
        return
    
    # create controller

    if local_gmt_replay:
        image_servers = []
        print("[sim_main] Local GMT replay: skipping live image servers and DDS")
    elif not args_cli.replay_data:
        print("========= create image server(s) =========")
        image_servers = []  # List to hold all image servers

        if args_cli.disable_front_camera:
            print("[sim_main] Front camera ImageServer disabled")
        else:
            try:
                front_server = _create_image_server(
                    args_cli,
                    port=args_cli.image_zmq_port,
                    camera_name="front_camera",
                    redis_suffix="",
                    dds_suffix="",
                    xrobot_port_offset=0,
                )
                image_servers.append(front_server)
                print(f"[sim_main] Front camera ImageServer started on port {args_cli.image_zmq_port}")
            except Exception as e:
                print(f"Failed to create front camera image server: {e}")
                return

        # Create world camera image server (optional, based on --enable_world_camera)
        if args_cli.enable_world_camera:
            try:
                world_server = _create_image_server(
                    args_cli,
                    port=args_cli.world_camera_port,
                    camera_name="world_camera",
                    redis_suffix="_world",
                    dds_suffix="_world",
                    xrobot_port_offset=1,
                )
                image_servers.append(world_server)
                print(f"[sim_main] World camera ImageServer started on port {args_cli.world_camera_port}")
            except Exception as e:
                print(f"Warning: Failed to create world camera image server: {e}")
                print("[sim_main] Continuing without world camera...")
        else:
            print("[sim_main] World camera disabled (use --enable_world_camera to enable)")

        if args_cli.enable_wrist_cameras:
            wrist_specs = [
                ("left_wrist_camera", args_cli.left_wrist_camera_port, "_left_wrist", 2, "Left"),
                ("right_wrist_camera", args_cli.right_wrist_camera_port, "_right_wrist", 3, "Right"),
            ]
            for camera_name, port, suffix, xrobot_offset, label in wrist_specs:
                try:
                    server = _create_image_server(
                        args_cli,
                        port=port,
                        camera_name=camera_name,
                        redis_suffix=suffix,
                        dds_suffix=suffix,
                        xrobot_port_offset=xrobot_offset,
                    )
                    image_servers.append(server)
                    print(f"[sim_main] {label} wrist camera ImageServer started on port {port}")
                except Exception as e:
                    print(f"Warning: Failed to create {camera_name} image server: {e}")
                    print(f"[sim_main] Continuing without {camera_name} stream...")

        print(f"========= created {len(image_servers)} image server(s) success =========")
        print("========= create dds =========")
        try:
            reset_pose_dds,sim_state_dds,dds_manager = create_dds_objects(args_cli,env)
        except Exception as e:
            print(f"Failed to create dds: {e}")
            return
        print("========= create dds success =========")
    else:
        print("========= create dds =========")
        try:
            create_dds_objects_replay(args_cli,env)
        except Exception as e:
            print(f"Failed to create dds: {e}")
            return
        print("========= create dds success =========")
        from tools.data_json_load import get_data_json_list
        print("========= get data json list =========")
        data_idx=0
        data_json_list = get_data_json_list(args_cli.file_path)
        if args_cli.action_source != "replay":
            args_cli.action_source = "replay"
        print("========= get data json list success =========")
    # create action provider

    print(f"\ncreate action provider: {args_cli.action_source}...")
    try:
        print(f"args_cli.task: {args_cli.task}")
        # import pdb; pdb.set_trace()
        action_provider = create_action_provider(env, args_cli)
        if action_provider is None:
            print("action provider creation failed, exiting")
            return
    except Exception as e:
        print(f"Failed to create action provider: {e}")
        return
    
    # set action provider
    print("========= create controller =========")
    controller = RobotController(env, control_config)
    controller.set_action_provider(action_provider)

    # Also set action_provider on env for camera_state.py to access
    env.action_provider = action_provider
    print(f"[sim_main] Set action_provider on env: {type(action_provider)}")

    # 立即启动录制（在第一次 get_action() 之前）
    # 这确保从 env.reset() 后就开始捕获所有状态，避免随机序列不同步
    if hasattr(action_provider, '_should_start_recording_on_first_call'):
        if action_provider._should_start_recording_on_first_call:
            print("\n" + "="*80)
            print("🔴 STARTING RECORDING IMMEDIATELY AFTER ENV.RESET()")
            print("="*80)
            if hasattr(action_provider, "_begin_episode_recording") and callable(action_provider._begin_episode_recording):
                action_provider._begin_episode_recording()
            else:
                action_provider.recording_manager.start_recording()
            action_provider._should_start_recording_on_first_call = False
            print("✅ Recording started to capture all random state from the beginning")
            print("="*80 + "\n")

    print("========= create controller success =========")
    
    # configure performance analysis
    if args_cli.enable_profiling:
        controller.set_profiling(True, args_cli.profile_interval)
        print(f"performance analysis enabled, report every {args_cli.profile_interval} steps")
    else:
        controller.set_profiling(False)
        print("performance analysis disabled")


    # set signal handlers
    if not args_cli.replay_data and not local_gmt_replay:
        setup_signal_handlers(controller, dds_manager, image_servers, simulation_app)
    else:
        setup_signal_handlers(controller, None, None, simulation_app)
    print("Note: The DDS in Sim transmits messages on channel 1. Please ensure that other DDS instances use the same channel for message exchange by setting: ChannelFactoryInitialize(1).")

    # Initialize joint position tracker (text-based, no GUI)
    # joint_tracker = None
    try:
        print("\n" + "="*80)
        print("Initializing Joint Position Tracker (Text-based)")
        print("="*80)
        robot = env.scene["robot"]
        # joint_tracker = JointPositionTracker(
        #     num_joints=robot.num_joints,
        #     window_size=200
        # )
        # Update joint names
        # if hasattr(robot.data, 'joint_names'):
        #     joint_tracker.joint_names = robot.data.joint_names
        print(f"Tracking {robot.num_joints} joints")
        print("Statistics will be printed every 10 seconds")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Warning: Failed to initialize joint tracker: {e}")
        print("Continuing without tracking...")

    try:
        # start controller - start asynchronous components
        print("========= start controller =========")
        controller.start()
        print("========= start controller success =========")
        redis_reset_client = None
        if not args_cli.replay_data and not local_gmt_replay:
            try:
                redis_reset_client = create_redis_client(
                    host=getattr(args_cli, "sonic_redis_host", "localhost"),
                    port=getattr(args_cli, "sonic_redis_port", 6379),
                    decode_responses=True,
                )
            except Exception as exc:
                print(f"[WARN] Failed to initialize reset Redis client: {exc}")
        try:
            _publish_backend_input_ready(args_cli, redis_client=redis_reset_client, source="startup")
        except Exception as exc:
            print(f"[WARN] Failed to publish startup input guard: {exc}")

        # main loop - execute in main thread to support rendering
        last_stats_time = time.time()
        loop_start_time = time.time()
        loop_count = 0
        last_loop_time = time.time()
        recent_loop_times = []  # for calculating moving average frequency
        camera_update_fn = None
        if args_cli.enable_cameras and not local_gmt_replay:
            try:
                from tasks.common_observations.camera_state import get_camera_image

                camera_update_fn = get_camera_image
                print("[sim_main] Camera shared-memory refresh enabled in main loop")
            except Exception as exc:
                print(f"[sim_main] Camera shared-memory refresh unavailable: {exc}")
        eval_video = _EvalVideoWriter(args_cli.eval_video_path, args_cli.video_fps) if args_cli.eval_video_path else None

        # use torch.inference_mode() and handle KeyboardInterrupt
        try:
            with torch.inference_mode():
                first_control_step_debug_pending = True
                while simulation_app.is_running() and controller.is_running:
                    current_time = time.time()
                    loop_count += 1
                    if not args_cli.replay_data and not local_gmt_replay:
                        # Only update state and reward every 10 loops to improve performance
                        if loop_count % 10 == 0:
                            try:
                                env_state = env.scene.get_state()
                                env_state_json = sim_state_to_json(env_state)
                                sim_state = {"init_state": env_state_json, "task_name": args_cli.task}
                            except Exception as e:
                                print(f"Failed to get env state: {e}")
                                raise e
                            try:
                                # sim_state = json.dumps(sim_state)
                                sim_state_dds.write_sim_state_data(sim_state)
                            except Exception as e:
                                print(f"Failed to write sim state: {e}")
                                raise e
                        # Check for reset command from Redis (from Pico controller)
                        try:
                            reset_cmd_redis = read_reset_trigger(redis_client=redis_reset_client)
                            if loop_count % 50 == 0:
                                print(f"[DEBUG] Checking Redis reset trigger... (value: {reset_cmd_redis})")
                            if reset_cmd_redis:
                                reset_category = str(reset_cmd_redis.get("reset_category", "2"))
                                print(f"[DEBUG] ✅ Received reset from Redis: category={reset_category}")
                                if reset_category == "1":
                                    print("🔄 Resetting object...")
                                    _trigger_task_reset_event(env_cfg, "reset_object_self", env)
                                    _restore_replay_initial_env_state_if_needed(env, args_cli)
                                    _notify_action_provider_env_objects_reset(action_provider)
                                elif reset_category == "2":
                                    print("🔄 Resetting all (robot + objects)...")
                                    _trigger_task_reset_event(env_cfg, "reset_all_self", env)
                                    _restore_replay_initial_env_state_if_needed(env, args_cli)
                                    _notify_action_provider_env_reset(action_provider)
                                    _publish_backend_input_ready(
                                        args_cli,
                                        redis_client=redis_reset_client,
                                        source="reset_all",
                                    )
                                elif reset_category == "3":
                                    _perform_complete_reset(
                                        env,
                                        env_cfg,
                                        args_cli,
                                        redis_client=redis_reset_client,
                                        action_provider=action_provider,
                                    )
                                clear_reset_trigger(redis_client=redis_reset_client)
                                print("[DEBUG] Reset trigger cleared from Redis")
                        except Exception as e:
                            if loop_count % 100 == 0:
                                print(f"[WARN] Failed to check Redis reset trigger: {e}")

                        # Check for reset command from DDS (original method)
                        # print(f"reset_pose_dds: {reset_pose_dds}")
                        try:
                            reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
                            # Debug: print when command is received
                            if reset_pose_cmd is not None:
                                print(f"[DEBUG] Received reset command: {reset_pose_cmd}")
                        except Exception as e:
                            print(f"Failed to get reset pose command: {e}")
                            raise e
                        if reset_pose_cmd is not None:
                            try:
                                reset_category = reset_pose_cmd.get("reset_category")
                                # print(f"reset_category: {reset_category}")

                                if reset_category == '1':
                                    # Reset object only
                                    print("🔄 Resetting object...")
                                    _trigger_task_reset_event(env_cfg, "reset_object_self", env)
                                    _restore_replay_initial_env_state_if_needed(env, args_cli)
                                    _notify_action_provider_env_objects_reset(action_provider)
                                    reset_pose_dds.write_reset_pose_command(-1)
                                elif reset_category == '2':
                                    # Reset all (robot + objects)
                                    print("🔄 Resetting all (robot + objects)...")
                                    _trigger_task_reset_event(env_cfg, "reset_all_self", env)
                                    _restore_replay_initial_env_state_if_needed(env, args_cli)
                                    _notify_action_provider_env_reset(action_provider)
                                    _publish_backend_input_ready(
                                        args_cli,
                                        redis_client=redis_reset_client,
                                        source="reset_all",
                                    )
                                    reset_pose_dds.write_reset_pose_command(-1)
                            except Exception as e:
                                print(f"Failed to write reset pose command: {e}")
                                raise e
                    elif args_cli.replay_data:
                        if action_provider.get_start_loop() and data_idx < len(data_json_list):
                            print(f"data_idx: {data_idx}")
                            try:
                                sim_state, task_name = action_provider.load_data(data_json_list[data_idx])
                                if task_name != args_cli.task:
                                    raise ValueError(
                                        f" The {task_name} in the dataset is different from the {args_cli.task} being executed ."
                                    )
                            except Exception as e:
                                print(f"Failed to load data: {e}")
                                raise e
                            try:
                                env.reset_to(sim_state, torch.tensor([0], device=env.device), is_relative=True)
                                env.sim.reset()
                                time.sleep(1)
                                action_provider.start_replay()
                                data_idx += 1
                            except Exception as e:
                                print(f"Failed to start replay: {e}")
                                raise e
                    # print(f"env_state: {env_state}")
                    # calculate instantaneous loop time
                    loop_dt = current_time - last_loop_time
                    last_loop_time = current_time
                    recent_loop_times.append(loop_dt)

                    # keep recent 100 loop times
                    if len(recent_loop_times) > 100:
                        recent_loop_times.pop(0)

                    # execute control step (in main thread, support rendering)
                    try:
                        controller.step()
                    except ReplayComplete as exc:
                        print(f"[sim_main] {exc}")
                        controller.stop()
                        break

                    if camera_update_fn is not None and loop_count % 3 == 0:
                        try:
                            camera_update_fn(env)
                        except Exception as exc:
                            if loop_count % 300 == 0:
                                print(f"[sim_main] Camera shared-memory refresh failed: {exc}")

                    if eval_video is not None:
                        eval_video.add(_capture_front_camera_rgb(env))

                    max_steps = int(getattr(args_cli, "max_steps", 0) or 0)
                    if max_steps > 0 and loop_count >= max_steps:
                        print(f"[sim_main] max_steps={max_steps} reached; stopping controller")
                        controller.stop()
                        break

                    if _should_exit_after_replay_complete(action_provider, args_cli):
                        print("[sim_main] Replay reached EOF and requested exit; stopping controller")
                        controller.stop()
                        break

                    if loop_count % 10 == 0:
                        try:
                            print(get_reward_debug_string(env))
                        except Exception as e:
                            print(f"奖励输出失败: {e}")

                    # Update joint tracker
                    # if joint_tracker and loop_count % 2 == 0:  # Update every 2 steps
                        # try:
                            # target_pos = env.scene["robot"].data.joint_pos_target[0].cpu().numpy()
                            # current_pos = env.scene["robot"].data.joint_pos[0].cpu().numpy()
                            # joint_tracker.update_data(target_pos, current_pos, timestamp=loop_count)

                            # Print compact summary every 100 steps
                            # if loop_count % 100 == 0:
                            #     joint_tracker.print_compact_summary()
                        # except Exception as e:
                        #     if loop_count % 500 == 0:  # Print error occasionally
                        #         print(f"Warning: Joint tracker update failed: {e}")

                    # print statistics and loop frequency periodically
                    if current_time - last_stats_time >= args_cli.stats_interval:
                        # calculate while loop execution frequency
                        elapsed_time = current_time - loop_start_time
                        loop_frequency = loop_count / elapsed_time if elapsed_time > 0 else 0

                        # calculate moving average frequency (based on recent loop times)
                        if recent_loop_times:
                            avg_loop_time = sum(recent_loop_times) / len(recent_loop_times)
                            moving_avg_frequency = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
                            min_loop_time = min(recent_loop_times)
                            max_loop_time = max(recent_loop_times)
                            max_freq = 1.0 / min_loop_time if min_loop_time > 0 else 0
                            min_freq = 1.0 / max_loop_time if max_loop_time > 0 else 0
                        else:
                            moving_avg_frequency = 0
                            min_freq = max_freq = 0

                        print(f"\n=== While loop execution frequency statistics ===")
                        print(f"loop execution count: {loop_count}")
                        print(f"running time: {elapsed_time:.2f} seconds")
                        print(f"overall average frequency: {loop_frequency:.2f} Hz")
                        print(
                            f"moving average frequency: {moving_avg_frequency:.2f} Hz (last {len(recent_loop_times)} times)"
                        )
                        print(f"frequency range: {min_freq:.2f} - {max_freq:.2f} Hz")
                        print(f"average loop time: {(elapsed_time/loop_count*1000):.2f} ms")
                        if recent_loop_times:
                            print(f"recent loop time: {(avg_loop_time*1000):.2f} ms")
                        print(f"=============================")

                        # Print joint tracking statistics
                        # if joint_tracker:
                        #     try:
                        #         joint_tracker.print_statistics(top_n=joint_tracker.num_joints)  # Print all joints
                        #     except Exception as e:
                        #         print(f"Warning: Failed to print joint statistics: {e}")

                        # print_stats(controller)
                        last_stats_time = current_time

                    # check environment state
                    if env.sim.is_stopped():
                        print("\nenvironment stopped")
                        break
                    # rate_limiter.sleep(env)
        except KeyboardInterrupt:
            print("\nuser interrupted program")
    
    except Exception as e:
        print(f"\nprogram exception: {e}")
    
    finally:
        # clean up resources
        print("\nclean up resources...")
        if "eval_video" in locals() and eval_video is not None:
            saved_video = eval_video.close()
            if saved_video:
                print(f"[eval_video] saved {saved_video} frames={len(eval_video.frames)}")
            else:
                print(f"[eval_video] no video saved frames={len(eval_video.frames)}")
        _close_image_servers(image_servers)
        controller.cleanup()
        env.close()
        print("cleanup completed")


if __name__ == "__main__":
    try:
        main()
    finally:
        print("Performing final cleanup...")
        
        # Get current process information
        import os
        import subprocess
        import signal
        import time
        
        current_pid = os.getpid()
        print(f"Current main process PID: {current_pid}")
        
        try:
            # Find all related Python processes
            result = subprocess.run(['pgrep', '-f', 'sim_main.py'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                print(f"Found related processes: {pids}")
                
                for pid in pids:
                    if pid and pid != str(current_pid):
                        try:
                            print(f"Terminating child process: {pid}")
                            os.kill(int(pid), signal.SIGTERM)
                        except ProcessLookupError:
                            print(f"Process {pid} does not exist")
                        except Exception as e:
                            print(f"Failed to terminate process {pid}: {e}")
                
                # Wait for processes to exit
                time.sleep(2)
                
                # Check if there are any remaining processes, force kill them
                result2 = subprocess.run(['pgrep', '-f', 'sim_main.py'],
                                       capture_output=True, text=True)
                if result2.returncode == 0:
                    remaining_pids = result2.stdout.strip().split('\n')
                    for pid in remaining_pids:
                        if pid and pid != str(current_pid):
                            try:
                                print(f"Force killing process: {pid}")
                                os.kill(int(pid), signal.SIGKILL)
                            except Exception as e:
                                print(f"Failed to force kill process {pid}: {e}")
                                
        except Exception as e:
            print(f"Error during process cleanup: {e}")
        
        try:
            simulation_app.close()
        except Exception as e:
            print(f"Failed to close simulation application: {e}")
            
        print("Program exit completed")
        
        # Force exit
        os._exit(0)

# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-Cylinder-G129-Dex1-Joint    --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129

# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-RedBlock-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129


# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex3-Joint     --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Inspire-Joint     --enable_inspire_dds --robot_type g129




# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex1-Wholebody  --robot_type g129 --enable_dex1_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex3-Wholebody  --robot_type g129 --enable_dex3_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Inspire-Wholebody  --robot_type g129 --enable_inspire_dds 
