#!/usr/bin/env python3
"""Batch replay TWIST2 recordings and re-record them as multicam episodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import signal
import subprocess
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
DEFAULT_INPUT_ROOT = ISAACLAB_ROOT / "recording_data/HSI_boxing/twist2"
DEFAULT_ENV_CONFIG_YAML = (
    ISAACLAB_ROOT / "tasks/common_env_config/boxing_bag_twist2.yaml"
).resolve()
PERSPECTIVE_RERECORD_SHM_SIZE_BYTES = 64 * 1024 * 1024
TASK_TO_ENV_CONFIG = {
    "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody": (
        ISAACLAB_ROOT / "tasks/common_env_config/small_warehouse_vision_navigation_twist2.yaml"
    ).resolve(),
    "Isaac-Move-Football-Single-G129-Dex3-Wholebody": (
        ISAACLAB_ROOT / "tasks/common_env_config/football_single_twist2.yaml"
    ).resolve(),
    "Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody": (
        ISAACLAB_ROOT / "tasks/common_env_config/pickplace_box_twist2.yaml"
    ).resolve(),
    "Isaac-Move-Open-Door-G129-Dex3-Wholebody": (
        ISAACLAB_ROOT / "tasks/common_env_config/opendoor_twist2.yaml"
    ).resolve(),
    "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody": (
        ISAACLAB_ROOT / "tasks/common_env_config/livingroom_sitsofa_twist2.yaml"
    ).resolve(),
    "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody": (
        ISAACLAB_ROOT / "tasks/common_env_config/boxing_bag_twist2.yaml"
    ).resolve(),
    "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody": (
        ISAACLAB_ROOT / "tasks/common_env_config/doubledesk_twist2.yaml"
    ).resolve(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch replay TWIST2 recordings and re-record them as multicam episodes."
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        default=str(DEFAULT_INPUT_ROOT),
        type=str,
        help="Root directory containing source TWIST2 .npz files.",
    )
    parser.add_argument(
        "--env-config-yaml",
        type=str,
        default="",
        help=(
            "Environment config YAML used for every replay job. "
            "Defaults to a per-task TWIST2 config inferred from each recording."
        ),
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="auto",
        help="Python interpreter used to launch sim_main.py.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--robot-type", type=str, default="g129")
    parser.add_argument(
        "--robot-collider-mode",
        type=str,
        default="box",
        choices=["box", "fourpoints", "default"],
    )
    parser.add_argument(
        "--replay-mode",
        type=str,
        default="direct",
        choices=["direct", "inference", "direct_replay", "inference_replay"],
    )
    parser.add_argument(
        "--task-runtime-profile",
        type=str,
        default="auto",
        choices=["auto", "inference", "replay_compat"],
        help="Task-specific runtime profile forwarded to sim_main.py.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Explicit output root. Defaults to <input_root>_multicam_rerecord.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recording-save-workers", type=int, default=1)
    parser.add_argument("--recording-save-queue-size", type=int, default=4)
    parser.add_argument("--headless", action="store_true", default=True)
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
        "--force",
        action="store_true",
        default=False,
        help="Compatibility flag matching SONIC rerecord; TWIST2 currently rerecords all matched files.",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print planned commands without launching Isaac Lab.",
    )
    return parser.parse_args()


def resolve_python_bin(requested: str) -> str:
    if requested != "auto":
        return requested
    if DEFAULT_ISAACLAB_PY.is_file():
        return str(DEFAULT_ISAACLAB_PY)
    return "python"


def resolve_robot_usd_override(args: argparse.Namespace) -> str | None:
    if args.robot_collider_mode == "default":
        return None
    if args.robot_collider_mode == "fourpoints":
        return str(DEFAULT_FOURPOINTS_ROBOT_USD)
    return str(DEFAULT_BOX_ROBOT_USD)


def build_env(robot_usd_override: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(ISAACLAB_ROOT)
    env["PYTHONPATH"] = str(ISAACLAB_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if robot_usd_override:
        env["ROBOT_USD_OVERRIDE"] = robot_usd_override
    else:
        env.pop("ROBOT_USD_OVERRIDE", None)
    return env


def find_npz_files(input_root: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in input_root.rglob("*.npz")
        if not path.name.endswith("_temp.npz")
    )


def read_task_name(npz_path: Path) -> str:
    with np.load(npz_path, allow_pickle=True) as data:
        task = data.get("task")
        if task is None:
            raise KeyError(f"{npz_path} missing task field")
        if hasattr(task, "item"):
            task = task.item()
        return str(task)


def resolve_env_config_yaml(args: argparse.Namespace, task_name: str) -> str:
    if args.env_config_yaml:
        return args.env_config_yaml
    env_config = TASK_TO_ENV_CONFIG.get(task_name)
    if env_config is None:
        raise SystemExit(f"No TWIST2 env_config mapping configured for task '{task_name}'")
    return str(env_config)


def read_rerecord_metrics(npz_path: Path) -> tuple[float | None, float | None, bool | None]:
    with np.load(npz_path, allow_pickle=True) as data:
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


def find_new_npz_files(output_dir: Path, before_files: set[Path]) -> list[Path]:
    after_files = {path.resolve() for path in output_dir.glob("*.npz")}
    return sorted(after_files - before_files, key=lambda path: path.stat().st_mtime)


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
    if "Failed to create any GPU devices" in text:
        return "failed_to_create_gpu_device"
    if "no CUDA-capable device is detected" in text:
        return "failed_to_create_cuda_context"
    if "Simulation App Shutting Down" in text and "STARTING RECORDING IMMEDIATELY AFTER ENV.RESET()" not in text:
        return "app_shutdown_before_control_loop"
    return None


def build_command(
    args: argparse.Namespace,
    *,
    python_bin: str,
    replay_file: Path,
    output_dir: Path,
    task_name: str,
    env_config_yaml: str,
    runtime_config,
) -> list[str]:
    command = [
        python_bin,
        "-u",
        str(SIM_MAIN),
        "--device",
        args.device,
        "--env_config_yaml",
        env_config_yaml,
        "--task",
        task_name,
        "--robot_type",
        args.robot_type,
        "--input_source",
        "replay",
        "--gmt_backend",
        "twist2",
        "--replay_file",
        str(replay_file),
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
        "--enable_cameras",
        "--enable_dex3_dds",
    ]
    append_image_runtime_args(command, runtime_config)
    if args.disable_front_camera:
        command.append("--disable_front_camera")
    if args.disable_wrist_cameras:
        command.append("--disable_wrist_cameras")
    else:
        command.append("--enable_wrist_cameras")
    if args.enable_perspective_camera:
        command.append("--enable_perspective_camera")
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


def build_jobs(
    args: argparse.Namespace,
    input_root: Path,
    output_root: Path,
    npz_paths: list[Path],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for index, npz_path in enumerate(npz_paths, start=1):
        try:
            task_name = read_task_name(npz_path)
        except (zipfile.BadZipFile, pickle.UnpicklingError, OSError, EOFError, ValueError, KeyError) as exc:
            print(f"[skip] unreadable source npz: {npz_path} ({exc})")
            continue
        env_config_yaml = resolve_env_config_yaml(args, task_name)
        rel_parent = npz_path.relative_to(input_root).parent
        output_dir = (output_root / rel_parent).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "index": index,
                "source_file": npz_path,
                "output_dir": output_dir,
                "task_name": task_name,
                "env_config_yaml": env_config_yaml,
            }
        )
    return jobs


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
    job: dict[str, Any],
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
    job: dict[str, Any],
    python_bin: str,
    base_env: dict[str, str],
    output_root: Path,
    runtime_config,
    print_lock: threading.Lock,
    summary_lock: threading.Lock,
    summary_path: Path,
) -> dict[str, Any]:
    source_file = Path(job["source_file"]).resolve()
    output_dir = Path(job["output_dir"]).resolve()
    temp_output_dir = allocate_job_output_dir(output_root, source_file.stem, runtime_config.runtime_tag)
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(
        args,
        python_bin=python_bin,
        replay_file=source_file,
        output_dir=temp_output_dir,
        task_name=str(job["task_name"]),
        env_config_yaml=str(job["env_config_yaml"]),
        runtime_config=runtime_config,
    )
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
    elapsed = time.time() - started_at
    new_npz_files = find_new_npz_files(temp_output_dir, before_npz)
    rerecorded_tmp_npz = new_npz_files[-1] if new_npz_files else None

    if timed_out:
        status = "timeout"
    elif return_code not in {0, 130, -2, 143, -15} and not detected_output:
        status = "failed"
    elif rerecorded_tmp_npz is None:
        status = "missing_output"
    else:
        status = "success"
    if status in {"failed", "missing_output"}:
        status = infer_failure_status_from_log(log_path) or status

    rerecorded_npz = None
    final_reward = None
    max_reward = None
    any_success = None
    if status == "success" and rerecorded_tmp_npz is not None:
        move_tree_contents(temp_output_dir, output_dir)
        remove_dir_if_empty(temp_output_dir, output_root / ".tmp_rerecord")
        rerecorded_npz = (output_dir / rerecorded_tmp_npz.name).resolve()
        final_reward, max_reward, any_success = read_rerecord_metrics(rerecorded_npz)

    with print_lock:
        if status == "success":
            if final_reward is None and max_reward is None:
                print(f"  ok elapsed={elapsed:.1f}s -> {rerecorded_npz}")
            else:
                final_text = "<missing>" if final_reward is None else f"{final_reward:.4f}"
                max_text = "<missing>" if max_reward is None else f"{max_reward:.4f}"
                any_success_text = "<missing>" if any_success is None else str(bool(any_success)).lower()
                print(
                    f"  ok elapsed={elapsed:.1f}s -> {rerecorded_npz}"
                    f" final_reward={final_text}"
                    f" max_reward={max_text}"
                    f" any_success={any_success_text}"
                )
        elif status == "dry_run":
            pass
        elif status == "timeout":
            print(f"  FAILED timeout elapsed={elapsed:.1f}s")
        elif status == "missing_output":
            print(f"  FAILED missing_output elapsed={elapsed:.1f}s temp_output_dir={temp_output_dir}")
        else:
            print(f"  FAILED return_code={return_code} elapsed={elapsed:.1f}s temp_output_dir={temp_output_dir}")

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

    return {
        "status": status,
        "source_file": source_file,
        "return_code": return_code,
        "elapsed": elapsed,
    }


def worker_loop(
    *,
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    total_jobs: int,
    python_bin: str,
    base_env: dict[str, str],
    output_root: Path,
    runtime_config,
    print_lock: threading.Lock,
    stop_event: threading.Event,
    summary_lock: threading.Lock,
    summary_path: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for job in jobs:
        if stop_event.is_set():
            break
        result = execute_job(
            args=args,
            total_jobs=total_jobs,
            job=job,
            python_bin=python_bin,
            base_env=base_env,
            output_root=output_root,
            runtime_config=runtime_config,
            print_lock=print_lock,
            summary_lock=summary_lock,
            summary_path=summary_path,
        )
        results.append(result)
        if result["status"] not in {"success", "dry_run"}:
            stop_event.set()
            break
    return results


def main() -> int:
    args = parse_args()
    if args.parallel_jobs <= 0:
        raise SystemExit("--parallel-jobs must be >= 1")
    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else input_root.with_name(input_root.name + "_multicam_rerecord")
    )
    try:
        output_root.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise SystemExit(
            f"output_root must not be inside input_root:\n"
            f"  input_root={input_root}\n"
            f"  output_root={output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    python_bin = resolve_python_bin(args.python_bin)
    robot_usd_override = resolve_robot_usd_override(args)
    subprocess_env = build_env(robot_usd_override)
    if args.enable_perspective_camera:
        subprocess_env.setdefault("ISAAC_IMAGE_SHM_SIZE_BYTES", str(PERSPECTIVE_RERECORD_SHM_SIZE_BYTES))
    npz_paths = find_npz_files(input_root)
    if args.limit > 0:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        print("No TWIST2 recordings found.")
        return 0
    jobs = build_jobs(args, input_root, output_root, npz_paths)
    worker_count = min(args.parallel_jobs, len(jobs))
    summary_path = output_root / DEFAULT_RERECORD_SUMMARY_FILENAME
    if not args.dry_run:
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
    print(f"input_root={input_root}")
    print(f"output_root={output_root}")
    print(f"jobs={len(jobs)}")
    print(f"parallel_jobs={worker_count}")
    if not args.dry_run:
        print(f"summary_log={summary_path}")

    print_lock = threading.Lock()
    summary_lock = threading.Lock()
    stop_event = threading.Event()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                worker_loop,
                args=args,
                jobs=chunk,
                total_jobs=len(jobs),
                python_bin=python_bin,
                base_env=subprocess_env,
                output_root=output_root,
                runtime_config=runtime_config,
                print_lock=print_lock,
                stop_event=stop_event,
                summary_lock=summary_lock,
                summary_path=summary_path,
            )
            for chunk, runtime_config in zip(job_chunks, worker_configs)
            if chunk
        ]
        for future in futures:
            results.extend(future.result())

    failure_result = next((result for result in results if result["status"] not in {"success", "dry_run"}), None)
    success_count = sum(1 for result in results if result["status"] == "success")
    failed_count = sum(1 for result in results if result["status"] not in {"success", "dry_run"})
    skipped_count = max(0, len(jobs) - len(results))
    print(f"done: success={success_count} failed={failed_count} skipped={skipped_count}")
    if failure_result is not None:
        return int(failure_result.get("return_code") or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
