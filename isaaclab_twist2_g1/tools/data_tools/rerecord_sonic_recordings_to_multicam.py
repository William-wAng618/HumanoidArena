#!/usr/bin/env python3
"""Batch replay SONIC recordings and re-record them as multicam episodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

try:
    from tools.data_tools.rerecord_parallel_utils import (
        DEFAULT_RERECORD_SUMMARY_FILENAME,
        DEFAULT_IMAGE_DDS_TOPIC,
        DEFAULT_IMAGE_REDIS_KEY_PREFIX,
        DEFAULT_IMAGE_XROBOT_PORT_BASE,
        DEFAULT_SHM_PREFIX,
        allocate_job_output_dir,
        append_text_log,
        append_image_runtime_args,
        build_worker_env,
        build_worker_runtime_config,
        chunk_round_robin,
        format_rerecord_summary_entry,
        move_tree_contents,
        remove_dir_if_empty,
        reset_text_log,
    )
except ModuleNotFoundError:
    from rerecord_parallel_utils import (
        DEFAULT_RERECORD_SUMMARY_FILENAME,
        DEFAULT_IMAGE_DDS_TOPIC,
        DEFAULT_IMAGE_REDIS_KEY_PREFIX,
        DEFAULT_IMAGE_XROBOT_PORT_BASE,
        DEFAULT_SHM_PREFIX,
        allocate_job_output_dir,
        append_text_log,
        append_image_runtime_args,
        build_worker_env,
        build_worker_runtime_config,
        chunk_round_robin,
        format_rerecord_summary_entry,
        move_tree_contents,
        remove_dir_if_empty,
        reset_text_log,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_ROOT = REPO_ROOT / "isaaclab_twist2_g1"
SIM_MAIN = ISAACLAB_ROOT / "sim_main.py"
DEFAULT_ISAACLAB_PY = Path(
    "/home/dreams/miniconda3/envs/unitree_sim_env_isaaclab5_0/bin/python"
)
DEFAULT_ENCODER_PATH = Path(
    "/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
)
DEFAULT_DECODER_PATH = Path(
    "/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
)
DEFAULT_BOX_ROBOT_USD = (
    ISAACLAB_ROOT
    / "assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
).resolve()
# DEFAULT_BOX_ROBOT_USD = (
#     ISAACLAB_ROOT
#     / "assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2_thumd.usd"
# ).resolve()
DEFAULT_FOURPOINTS_ROBOT_USD = (
    ISAACLAB_ROOT
    / "assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
).resolve()
DEFAULT_SOURCE_ROOTS = []
TASK_TO_ENV_CONFIG = {
      "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody": (
          ISAACLAB_ROOT / "tasks/common_env_config/small_warehouse_vision_navigation_sonic.yaml"
      ).resolve(),
      "Isaac-Move-Football-Single-G129-Dex3-Wholebody": (
          ISAACLAB_ROOT / "tasks/common_env_config/football_single_sonic.yaml"
      ).resolve(),
      "Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody": (
          ISAACLAB_ROOT / "tasks/common_env_config/pickplace_box_sonic.yaml"
      ).resolve(),
      "Isaac-Move-Open-Door-G129-Dex3-Wholebody": (
          ISAACLAB_ROOT / "tasks/common_env_config/opendoor_sonic.yaml"
      ).resolve(),
      "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody": (
          ISAACLAB_ROOT / "tasks/common_env_config/livingroom_sitsofa_sonic.yaml"
      ).resolve(),
      "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody": (
          ISAACLAB_ROOT / "tasks/common_env_config/boxing_bag_sonic.yaml"
      ).resolve(),
      "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody": (
          ISAACLAB_ROOT / "tasks/common_env_config/doubledesk_sonic.yaml"
      ).resolve(),
      

}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch replay SONIC recordings and re-record them as multicam episodes."
    )
    parser.add_argument(
        "source_roots",
        nargs="*",
        default=[str(path) for path in DEFAULT_SOURCE_ROOTS],
        help="Source directories that contain SONIC v2 .npz files.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="auto",
        help="Python interpreter used to launch sim_main.py. Defaults to an onnxruntime-capable interpreter.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device passed to sim_main.py.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="g129",
        help="Robot type passed to sim_main.py.",
    )
    parser.add_argument(
        "--robot-collider-mode",
        type=str,
        default="box",
        choices=["box", "fourpoints", "default"],
        help=(
            "Robot collider/USD preset used for replay rerecord jobs. "
            "'box' matches run_replay_sonic.sh and exports ROBOT_USD_OVERRIDE to the m2 USD."
        ),
    )
    parser.add_argument(
        "--robot-usd-override",
        type=str,
        default="",
        help="Explicit ROBOT_USD_OVERRIDE path. Takes precedence over --robot-collider-mode.",
    )
    parser.add_argument(
        "--replay-mode",
        type=str,
        default="direct_replay",
        choices=["direct", "inference", "direct_replay", "inference_replay"],
        help="Replay mode used during migration.",
    )
    parser.add_argument(
        "--task-runtime-profile",
        type=str,
        default="auto",
        choices=["auto", "inference", "replay_compat"],
        help="Task-specific runtime profile forwarded to sim_main.py.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_multicam_rerecord",
        help="Suffix appended beside each source root to store rerecorded outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only migrate the first N eligible files across all source roots. 0 means no limit.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Per-file timeout. 0 disables the timeout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed passed to sim_main.py.",
    )
    parser.add_argument(
        "--recording-save-workers",
        type=int,
        default=1,
        help="Background save workers for the rerecorded output.",
    )
    parser.add_argument(
        "--recording-save-queue-size",
        type=int,
        default=4,
        help="Background save queue size for the rerecorded output.",
    )
    parser.add_argument(
        "--sonic-encoder-path",
        type=str,
        default=str(DEFAULT_ENCODER_PATH),
        help="GEAR-SONIC encoder path.",
    )
    parser.add_argument(
        "--sonic-decoder-path",
        type=str,
        default=str(DEFAULT_DECODER_PATH),
        help="GEAR-SONIC decoder path.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run sim_main.py in headless mode.",
    )
    parser.add_argument(
        "--disable-cameras",
        action="store_true",
        default=False,
        help="Disable cameras during rerecord. By default cameras stay enabled so vision data is regenerated.",
    )
    parser.add_argument(
        "--enable-perspective-camera",
        action="store_true",
        default=False,
        help=(
            "Enable the third-person /World/PerspectiveCamera stream and save it "
            "as vision_world_* data beside the enabled front and wrist cameras."
        ),
    )
    parser.add_argument(
        "--disable-front-camera",
        action="store_true",
        default=False,
        help="Do not create or record the front camera during rerecord.",
    )
    parser.add_argument(
        "--disable-wrist-cameras",
        action="store_true",
        default=False,
        help="Do not create or record left/right wrist cameras during rerecord.",
    )
    parser.add_argument(
        "--disable-dex3",
        action="store_true",
        default=False,
        help="Do not pass --enable_dex3_dds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Ignore successful manifest entries and rerun every source file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the planned commands without launching Isaac Lab.",
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=5,
        help="Number of concurrent headless Isaac Lab rerecord jobs.",
    )
    parser.add_argument(
        "--image-port-base",
        type=int,
        default=5555,
        help="Base ZMQ image port for worker 0. Each worker gets its own port bundle.",
    )
    parser.add_argument(
        "--image-port-stride",
        type=int,
        default=10,
        help="Port stride reserved per worker for image/world/wrist streams.",
    )
    parser.add_argument(
        "--image-xrobot-port-base",
        type=int,
        default=DEFAULT_IMAGE_XROBOT_PORT_BASE,
        help="Base XRobot image port for worker 0.",
    )
    parser.add_argument(
        "--image-xrobot-port-stride",
        type=int,
        default=10,
        help="XRobot port stride reserved per worker.",
    )
    parser.add_argument(
        "--image-redis-key-prefix",
        type=str,
        default=DEFAULT_IMAGE_REDIS_KEY_PREFIX,
        help="Base Redis key prefix for image transport; worker runtime suffix is appended automatically.",
    )
    parser.add_argument(
        "--image-dds-topic",
        type=str,
        default=DEFAULT_IMAGE_DDS_TOPIC,
        help="Base DDS topic for image transport; worker runtime suffix is appended automatically.",
    )
    parser.add_argument(
        "--shm-prefix",
        type=str,
        default=DEFAULT_SHM_PREFIX,
        help="Shared-memory name prefix; worker runtime suffix is appended automatically.",
    )
    return parser.parse_args()


def resolve_python_bin(requested: str) -> str:
    if requested != "auto":
        return requested
    candidates = [Path(sys.executable)]
    if DEFAULT_ISAACLAB_PY not in candidates:
        candidates.append(DEFAULT_ISAACLAB_PY)
    for candidate in candidates:
        if not candidate.exists():
            continue
        result = subprocess.run(
            [str(candidate), "-c", "import onnxruntime"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return str(candidate)
    raise SystemExit("No usable python interpreter with onnxruntime was found.")


def read_npz_meta(path: Path) -> tuple[str, str]:
    with np.load(path, allow_pickle=True) as data:
        schema = str(np.asarray(data["schema_version"]).item()) if "schema_version" in data else ""
        task = str(np.asarray(data["task"]).item()) if "task" in data else ""
    return schema, task


def read_rerecord_metrics(path: Path) -> tuple[float | None, float | None, bool | None]:
    with np.load(path, allow_pickle=True) as data:
        def _read_float(key: str) -> float | None:
            if key not in data:
                return None
            value = np.asarray(data[key], dtype=np.float32).reshape(-1)
            if value.size == 0:
                return None
            return float(value[0])

        def _read_bool(key: str) -> bool | None:
            if key not in data:
                return None
            value = np.asarray(data[key]).reshape(-1)
            if value.size == 0:
                return None
            return bool(value[0])

        final_reward = _read_float("rerecord_final_reward")
        max_reward = _read_float("rerecord_max_reward")
        if max_reward is None:
            max_reward = final_reward
        any_success = _read_bool("rerecord_any_success")
        if any_success is None and max_reward is not None:
            any_success = bool(max_reward > 1e-6)
        return final_reward, max_reward, any_success


def resolve_robot_usd_override(args: argparse.Namespace) -> str:
    if args.robot_usd_override:
        robot_usd = Path(args.robot_usd_override).expanduser().resolve()
        if not robot_usd.is_file():
            raise SystemExit(f"ROBOT_USD_OVERRIDE not found: {robot_usd}")
        return str(robot_usd)
    if args.robot_collider_mode == "box":
        if not DEFAULT_BOX_ROBOT_USD.is_file():
            raise SystemExit(f"Default box robot USD not found: {DEFAULT_BOX_ROBOT_USD}")
        return str(DEFAULT_BOX_ROBOT_USD)
    if args.robot_collider_mode == "fourpoints":
        if not DEFAULT_FOURPOINTS_ROBOT_USD.is_file():
            raise SystemExit(f"Default fourpoints robot USD not found: {DEFAULT_FOURPOINTS_ROBOT_USD}")
        return str(DEFAULT_FOURPOINTS_ROBOT_USD)
    return ""


def build_subprocess_env(robot_usd_override: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PROJECT_ROOT", str(ISAACLAB_ROOT))
    pythonpath_parts = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    isaaclab_root_str = str(ISAACLAB_ROOT)
    if isaaclab_root_str not in pythonpath_parts:
        pythonpath_parts.insert(0, isaaclab_root_str)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if robot_usd_override:
        env["ROBOT_USD_OVERRIDE"] = robot_usd_override
    return env


def read_successful_sources(manifest_path: Path) -> set[str]:
    if not manifest_path.is_file():
        return set()
    sources: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "success" and payload.get("source"):
                sources.add(str(payload["source"]))
    return sources


def append_manifest(manifest_path: Path, payload: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def infer_failure_status_from_log(log_path: Path) -> str | None:
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if "Failed to parse environment configuration" in text:
        return "failed_to_parse_env_config"
    if "Failed to create environment:" in text:
        return "failed_to_create_environment"
    if "Failed to create action provider:" in text:
        return "failed_to_create_action_provider"
    if "Failed to create dds:" in text:
        return "failed_to_create_dds"
    if "program exception:" in text:
        return "program_exception"
    if "Simulation App Shutting Down" in text and "STARTING RECORDING IMMEDIATELY AFTER ENV.RESET()" not in text:
        return "app_shutdown_before_control_loop"
    return None


def build_jobs(args: argparse.Namespace) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for source_root_raw in args.source_roots:
        source_root = Path(source_root_raw).expanduser().resolve()
        if not source_root.is_dir():
            raise SystemExit(f"Source root not found: {source_root}")
        output_root = source_root.parent / f"{source_root.name}{args.output_suffix}"
        manifest_path = output_root / "rerecord_manifest.jsonl"
        completed_sources = set() if args.force else read_successful_sources(manifest_path)
        for source_file in sorted(source_root.rglob("*.npz")):
            source_file = source_file.resolve()
            if source_file.name.endswith("_temp.npz"):
                print(f"[skip] ignoring temporary source file: {source_file}")
                continue
            if not args.force and str(source_file) in completed_sources:
                continue
            try:
                schema_version, task_name = read_npz_meta(source_file)
            except (zipfile.BadZipFile, pickle.UnpicklingError, OSError, EOFError, ValueError) as exc:
                print(f"[skip] unreadable source npz: {source_file} ({exc})")
                continue
            if schema_version not in {"sonic_episode_v2", "sonic_episode_v3", "sonic_episode_v4_multicam"}:
                continue
            env_config = TASK_TO_ENV_CONFIG.get(task_name)
            if not env_config:
                raise SystemExit(
                    f"No env_config mapping configured for task '{task_name}' from {source_file}"
                )
            jobs.append(
                {
                    "source_root": source_root,
                    "source_file": source_file,
                    "output_root": output_root,
                    "manifest_path": manifest_path,
                    "task_name": task_name,
                    "env_config": env_config,
                }
            )
    return jobs


def find_new_npz_files(output_dir: Path, before_files: set[Path]) -> list[Path]:
    after_files = {path.resolve() for path in output_dir.glob("*.npz")}
    return sorted(after_files - before_files, key=lambda path: path.stat().st_mtime)


def build_command(
    args: argparse.Namespace,
    job: dict[str, object],
    output_dir: Path,
    python_bin: str,
    runtime_config,
) -> list[str]:
    source_file = Path(job["source_file"])
    command = [
        python_bin,
        "-u",
        str(SIM_MAIN),
        "--device",
        args.device,
        "--env_config_yaml",
        str(job["env_config"]),
        "--task",
        str(job["task_name"]),
        "--robot_type",
        args.robot_type,
        "--input_source",
        "replay",
        "--gmt_backend",
        "sonic",
        "--sonic_encoder_path",
        args.sonic_encoder_path,
        "--sonic_decoder_path",
        args.sonic_decoder_path,
        "--replay_file",
        str(source_file),
        "--replay_mode",
        args.replay_mode,
        "--task_runtime_profile",
        args.task_runtime_profile,
        "--record_during_replay",
        "--exit_when_replay_complete",
        "--recording_save_dir",
        str(output_dir),
        "--recording_save_workers",
        str(args.recording_save_workers),
        "--recording_save_queue_size",
        str(args.recording_save_queue_size),
        "--seed",
        str(args.seed),
    ]
    append_image_runtime_args(command, runtime_config)
    if not args.disable_cameras:
        command.append("--enable_cameras")
        if args.disable_front_camera:
            command.append("--disable_front_camera")
        if args.disable_wrist_cameras:
            command.append("--disable_wrist_cameras")
        else:
            command.append("--enable_wrist_cameras")
        if args.enable_perspective_camera:
            command.append("--enable_perspective_camera")
    if not args.disable_dex3:
        command.append("--enable_dex3_dds")
    if args.headless:
        command.append("--headless")
    return command


def wait_for_rerecord_completion(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_handle,
    output_dir: Path,
    timeout_seconds: float,
) -> tuple[int | None, bool, bool]:
    before_npz = {path.resolve() for path in output_dir.glob("*.npz")}
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    start_time = time.time()
    stable_count = 0
    detected_output = False
    timed_out = False
    last_output_file = ""
    last_size = -1

    while True:
        if timeout_seconds > 0 and time.time() - start_time > timeout_seconds:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            break

        new_npz_files = find_new_npz_files(output_dir, before_npz)
        if new_npz_files:
            latest_output = new_npz_files[-1]
            current_size = latest_output.stat().st_size
            if str(latest_output) == last_output_file and current_size == last_size:
                stable_count += 1
            else:
                stable_count = 0
                last_output_file = str(latest_output)
                last_size = current_size

            if stable_count >= 2:
                detected_output = True
                log_handle.write(f"[watchdog] detected stable rerecord output: {latest_output}\n")
                log_handle.flush()
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                break

        if process.poll() is not None:
            return process.returncode, bool(new_npz_files), timed_out

        time.sleep(1.0)

    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10.0)

    new_npz_files = find_new_npz_files(output_dir, before_npz)
    return process.returncode, bool(new_npz_files), timed_out


def log_runtime_details(log_handle, runtime_config) -> None:
    log_handle.write(f"RUNTIME_TAG: {runtime_config.runtime_tag}\n")
    log_handle.write(f"SHM_NAME: {runtime_config.shm_name}\n")
    log_handle.write(
        "IMAGE_PORTS: "
        f"front={runtime_config.image_zmq_port} "
        f"world={runtime_config.world_camera_port} "
        f"left={runtime_config.left_wrist_camera_port} "
        f"right={runtime_config.right_wrist_camera_port} "
        f"xrobot={runtime_config.image_xrobot_port}\n"
    )
    log_handle.write(f"IMAGE_REDIS_KEY_PREFIX: {runtime_config.image_redis_key_prefix}\n")
    log_handle.write(f"IMAGE_DDS_TOPIC: {runtime_config.image_dds_topic}\n")


def print_job_header(
    *,
    print_lock: threading.Lock,
    total_jobs: int,
    job: dict[str, object],
    log_path: Path,
    temp_output_dir: Path,
    runtime_config,
) -> None:
    with print_lock:
        print(f"[{job['index']}/{total_jobs}] rerecording {job['source_file']}")
        print(f"  output_dir={job['output_dir']}")
        print(f"  temp_output_dir={temp_output_dir}")
        print(f"  log={log_path}")
        print(
            "  runtime="
            f"{runtime_config.runtime_tag} "
            f"ports={runtime_config.image_zmq_port}/{runtime_config.world_camera_port}/"
            f"{runtime_config.left_wrist_camera_port}/{runtime_config.right_wrist_camera_port}"
        )
        print(f"  shm={runtime_config.shm_name}")


def execute_job(
    *,
    args: argparse.Namespace,
    total_jobs: int,
    job: dict[str, object],
    python_bin: str,
    base_env: dict[str, str],
    runtime_config,
    print_lock: threading.Lock,
    manifest_lock: threading.Lock,
    summary_lock: threading.Lock,
) -> dict[str, object]:
    source_file = Path(job["source_file"]).resolve()
    output_root = Path(job["output_root"]).resolve()
    output_dir = Path(job["output_dir"]).resolve()
    manifest_path = Path(job["manifest_path"]).resolve()
    summary_path = Path(job["summary_path"]).resolve()
    temp_output_dir = allocate_job_output_dir(output_root, source_file.stem, runtime_config.runtime_tag)
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, job, temp_output_dir, python_bin, runtime_config)
    log_dir = output_root / "rerecord_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{source_file.stem}.log"

    print_job_header(
        print_lock=print_lock,
        total_jobs=total_jobs,
        job=job,
        log_path=log_path,
        temp_output_dir=temp_output_dir,
        runtime_config=runtime_config,
    )
    if args.dry_run:
        with print_lock:
            print("  command=" + " ".join(command))
        return {"status": "dry_run", "source_file": source_file}

    before_npz = {path.resolve() for path in temp_output_dir.glob("*.npz")}
    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND: " + " ".join(command) + "\n")
        log_handle.write(f"ROBOT_USD_OVERRIDE: {base_env.get('ROBOT_USD_OVERRIDE', '<default>')}\n")
        log_runtime_details(log_handle, runtime_config)
        log_handle.flush()
        return_code, detected_output, timed_out = wait_for_rerecord_completion(
            command=command,
            cwd=REPO_ROOT,
            env=build_worker_env(base_env, runtime_config),
            log_handle=log_handle,
            output_dir=temp_output_dir,
            timeout_seconds=args.timeout_seconds,
        )

    rerecorded_tmp_npz = None
    new_npz_files = find_new_npz_files(temp_output_dir, before_npz)
    if new_npz_files:
        rerecorded_tmp_npz = new_npz_files[-1]

    rerecorded_schema = ""
    if rerecorded_tmp_npz is not None:
        rerecorded_schema, _ = read_npz_meta(rerecorded_tmp_npz)
    final_reward = None
    max_reward = None
    any_success = None
    if rerecorded_tmp_npz is not None:
        final_reward, max_reward, any_success = read_rerecord_metrics(rerecorded_tmp_npz)

    status = "success"
    if timed_out:
        status = "timeout"
    elif rerecorded_tmp_npz is None:
        status = "missing_output"
    elif rerecorded_schema not in {"sonic_episode_v3", "sonic_episode_v4_multicam"}:
        status = f"unexpected_schema:{rerecorded_schema or 'missing'}"
    elif detected_output and return_code in {130, -2, 143, -15}:
        status = "success"

    rerecorded_npz = None
    if status == "success" and rerecorded_tmp_npz is not None:
        move_tree_contents(temp_output_dir, output_dir)
        remove_dir_if_empty(temp_output_dir, output_root / ".tmp_rerecord")
        rerecorded_npz = (output_dir / rerecorded_tmp_npz.name).resolve()

    payload = {
        "duration_sec": round(time.time() - started_at, 3),
        "env_config": str(job["env_config"]),
        "log_path": str(log_path.resolve()),
        "rerecorded_npz": str(rerecorded_npz.resolve()) if rerecorded_npz is not None else "",
        "return_code": return_code,
        "rerecord_final_reward": final_reward,
        "rerecord_max_reward": max_reward,
        "rerecord_any_success": any_success,
        "source": str(source_file),
        "status": status,
        "task_name": job["task_name"],
        "timestamp": int(time.time()),
    }
    if status == "missing_output":
        inferred_status = infer_failure_status_from_log(log_path)
        if inferred_status:
            payload["status"] = inferred_status
            status = inferred_status

    with manifest_lock:
        append_manifest(manifest_path, payload)

    with summary_lock:
        append_text_log(
            summary_path,
            format_rerecord_summary_entry(
                index=int(job["index"]),
                total_jobs=total_jobs,
                source_file=source_file,
                output_dir=output_dir,
                log_path=log_path,
                status=status,
                rerecorded_npz=rerecorded_npz,
                final_reward=final_reward,
                max_reward=max_reward,
                any_success=any_success,
                return_code=return_code,
            ),
        )

    with print_lock:
        if status == "success":
            if final_reward is None and max_reward is None:
                print(f"  success -> {rerecorded_npz}")
            else:
                final_text = "<missing>" if final_reward is None else f"{final_reward:.4f}"
                max_text = "<missing>" if max_reward is None else f"{max_reward:.4f}"
                any_success_text = "<missing>" if any_success is None else str(bool(any_success)).lower()
                print(
                    f"  success -> {rerecorded_npz}"
                    f" final_reward={final_text}"
                    f" max_reward={max_text}"
                    f" any_success={any_success_text}"
                )
        else:
            print(f"  {status}")

    return payload


def worker_loop(
    *,
    args: argparse.Namespace,
    jobs: list[dict[str, object]],
    total_jobs: int,
    python_bin: str,
    base_env: dict[str, str],
    runtime_config,
    print_lock: threading.Lock,
    manifest_lock: threading.Lock,
    summary_lock: threading.Lock,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for job in jobs:
        results.append(
            execute_job(
                args=args,
                total_jobs=total_jobs,
                job=job,
                python_bin=python_bin,
                base_env=base_env,
                runtime_config=runtime_config,
                print_lock=print_lock,
                manifest_lock=manifest_lock,
                summary_lock=summary_lock,
            )
        )
    return results


def main() -> int:
    args = parse_args()
    if args.parallel_jobs <= 0:
        raise SystemExit("--parallel-jobs must be >= 1")
    if args.enable_perspective_camera and args.disable_cameras:
        raise SystemExit("--enable-perspective-camera requires camera rendering; remove --disable-cameras")
    python_bin = resolve_python_bin(args.python_bin)
    robot_usd_override = resolve_robot_usd_override(args)
    subprocess_env = build_subprocess_env(robot_usd_override)
    jobs = build_jobs(args)
    if args.limit > 0:
        jobs = jobs[: args.limit]
    if not jobs:
        print("No SONIC recordings matched the rerecord criteria.")
        return 0

    for job in jobs:
        source_root = Path(job["source_root"]).resolve()
        output_root = Path(job["output_root"]).resolve()
        try:
            output_root.relative_to(source_root)
        except ValueError:
            continue
        raise SystemExit(
            f"output_root must not be inside input_root:\n"
            f"  input_root={source_root}\n"
            f"  output_root={output_root}"
        )

    for index, job in enumerate(jobs, start=1):
        source_root = Path(job["source_root"])
        source_file = Path(job["source_file"])
        output_root = Path(job["output_root"])
        rel_parent = source_file.relative_to(source_root).parent
        output_dir = (output_root / rel_parent).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        job["index"] = index
        job["output_dir"] = output_dir
        job["summary_path"] = output_root / DEFAULT_RERECORD_SUMMARY_FILENAME

    worker_count = min(args.parallel_jobs, len(jobs))
    if not args.dry_run:
        for summary_path in sorted({Path(job["summary_path"]).resolve() for job in jobs}):
            reset_text_log(summary_path)
    worker_configs = [
        build_worker_runtime_config(
            worker_index=worker_index,
            image_port_base=args.image_port_base,
            image_port_stride=args.image_port_stride,
            image_xrobot_port_base=args.image_xrobot_port_base,
            image_xrobot_port_stride=args.image_xrobot_port_stride,
            shm_prefix=args.shm_prefix,
            image_redis_key_prefix=args.image_redis_key_prefix,
            image_dds_topic=args.image_dds_topic,
        )
        for worker_index in range(worker_count)
    ]
    job_chunks = chunk_round_robin(jobs, worker_count)

    print(f"python_bin={python_bin}")
    print(f"robot_usd_override={robot_usd_override or '<default>'}")
    print(f"jobs={len(jobs)}")
    print(f"parallel_jobs={worker_count}")
    if not args.dry_run:
        for summary_path in sorted({Path(job["summary_path"]).resolve() for job in jobs}):
            print(f"summary_log={summary_path}")

    print_lock = threading.Lock()
    manifest_lock = threading.Lock()
    summary_lock = threading.Lock()
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                worker_loop,
                args=args,
                jobs=chunk,
                total_jobs=len(jobs),
                python_bin=python_bin,
                base_env=subprocess_env,
                runtime_config=runtime_config,
                print_lock=print_lock,
                manifest_lock=manifest_lock,
                summary_lock=summary_lock,
            )
            for chunk, runtime_config in zip(job_chunks, worker_configs)
            if chunk
        ]
        for future in futures:
            results.extend(future.result())

    success_count = sum(1 for result in results if result["status"] == "success")
    failure_count = sum(1 for result in results if result["status"] not in {"success", "dry_run"})

    print(f"done: success={success_count} failure={failure_count}")
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
