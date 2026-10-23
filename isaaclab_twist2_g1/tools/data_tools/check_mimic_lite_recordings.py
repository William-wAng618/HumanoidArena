#!/usr/bin/env python3
"""Pre-rerecord integrity check for MimicLite recordings.

Validates raw MimicLite teleoperation episodes before they are replayed through
``rerecord_mimic_lite_recordings_to_multicam.py``. Catches the failure modes that would
otherwise surface only as a silent ``missing_output`` inside a batch job:

* unreadable / wrong-schema npz, missing core or vision arrays
* inconsistent ``num_frames`` across arrays
* non-contiguous ``episode_step`` (breaks Isaac-time alignment downstream)
* ``vision_frame_indices`` that are not a full ``0..N-1`` sequence
* NaN/Inf in numeric arrays
* missing or frame-mismatched RGB video sidecars
* duplicate ``episode_object_seed`` values (breaks deterministic replay restore)
* non-monotonic wall timestamps
* episodes whose root pose ends on the floor (operator fell during collection)

Usage:
    python tools/data_tools/check_mimic_lite_recordings.py <dir> [<dir> ...] \
        [--json-out report.json] [--skip-video-probe]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

MIMIC_LITE_SCHEMAS = {"mimic_lite_episode_v1", "mimic_lite_episode_v2_multicam"}

CORE_ARRAYS = (
    "action_body_29dof",
    "mimic_command",
    "mimic_policy",
    "mimic_policy_action",
    "robot_qpos_before_decimation",
    "robot_root_position",
    "robot_root_orientation",
    "human_raw_joint_pos",
    "action_hand_action_left",
    "action_hand_action_right",
)

# Scalars that are stored without a frame dimension.
SCALAR_KEYS = (
    "schema_version",
    "task",
    "num_frames",
    "meta_control_dt",
    "meta_physics_dt",
    "meta_decimation",
    "episode_object_seed",
    "episode_object_seed_source",
)

FALL_ROOT_Z_THRESHOLD = 0.45  # m; below this the pelvis is on the floor
TILT_WARN_DEG = 30.0


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.size == 0:
        return default
    return value.reshape(-1)[0]


def _probe_video_frame_count(path: Path) -> int | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text or text == "N/A":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def check_episode(npz_path: Path, *, video_probe: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001 - report any load failure verbatim
        return {
            "path": str(npz_path),
            "status": "FAIL",
            "errors": [f"unreadable npz: {exc}"],
            "warnings": [],
        }

    with data:
        schema = str(_scalar(data, "schema_version", ""))
        task = str(_scalar(data, "task", ""))
        if schema not in MIMIC_LITE_SCHEMAS:
            errors.append(f"unexpected schema_version: {schema!r}")

        for key in SCALAR_KEYS:
            if key not in data.files:
                errors.append(f"missing scalar key: {key}")

        num_frames_raw = _scalar(data, "num_frames")
        if num_frames_raw is None:
            return {
                "path": str(npz_path),
                "status": "FAIL",
                "errors": errors + ["num_frames missing; cannot validate lengths"],
                "warnings": warnings,
            }
        num_frames = int(num_frames_raw)
        if num_frames <= 0:
            errors.append(f"num_frames must be positive, got {num_frames}")

        # Length consistency across every frame-indexed array.
        # ``episode_init_env_*`` fields snapshot the reset state once per episode, so they are
        # intentionally not frame-indexed and must be excluded here.
        for key in data.files:
            if key.startswith("episode_init_env_"):
                continue
            value = data[key]
            if value.ndim == 0:
                continue
            if value.shape[0] != num_frames:
                errors.append(
                    f"length mismatch: {key} has {value.shape[0]} rows, num_frames={num_frames}"
                )

        for key in CORE_ARRAYS:
            if key not in data.files:
                errors.append(f"missing core array: {key}")

        # Isaac control-time contiguity (downstream ZOH alignment depends on it).
        if "episode_step" in data.files:
            steps = np.asarray(data["episode_step"], dtype=np.int64).reshape(-1)
            if steps.shape != (num_frames,):
                errors.append(f"episode_step shape {steps.shape} != ({num_frames},)")
            elif steps.size > 1 and not np.all(np.diff(steps) == 1):
                errors.append("episode_step is not contiguous Isaac control time")

        # Vision frame indices must cover 0..N-1 exactly for every camera stream.
        index_keys = [k for k in data.files if k.endswith("_frame_indices") and k.startswith("vision_")]
        for key in index_keys:
            indices = np.asarray(data[key], dtype=np.int64).reshape(-1)
            expected = np.arange(num_frames, dtype=np.int64)
            if indices.shape != expected.shape or not np.array_equal(indices, expected):
                errors.append(f"{key} is not a full 0..{num_frames - 1} sequence")

        # Numeric sanity.
        nan_keys: list[str] = []
        for key in data.files:
            value = data[key]
            if value.dtype.kind != "f":
                continue
            if not np.all(np.isfinite(value)):
                nan_keys.append(key)
        if nan_keys:
            errors.append(f"non-finite values in: {', '.join(sorted(nan_keys)[:6])}")

        # Video sidecars.
        video_keys = [k for k in data.files if k.endswith("_video_path")]
        if not video_keys:
            warnings.append("no vision_*_video_path keys; episode may be depth-only")
        for key in video_keys:
            rel = str(_scalar(data, key, ""))
            if not rel:
                continue
            video_path = npz_path.parent / rel
            if not video_path.is_file():
                errors.append(f"missing video sidecar: {rel}")
                continue
            declared = _scalar(data, key.replace("_video_path", "_video_num_frames"))
            if declared is not None and int(declared) != num_frames:
                errors.append(
                    f"{key}: declared video frames {int(declared)} != num_frames {num_frames}"
                )
            if video_probe:
                actual = _probe_video_frame_count(video_path)
                if actual is not None and actual != num_frames:
                    errors.append(
                        f"{rel}: encoded frames {actual} != num_frames {num_frames}"
                    )

        # Timestamp monotonicity.
        wall = None
        if "timestamp_wall" in data.files:
            wall = np.asarray(data["timestamp_wall"], dtype=np.float64).reshape(-1)
            if wall.size > 1 and not np.all(np.diff(wall) > 0):
                errors.append("timestamp_wall is not strictly increasing")

        # Root pose: detect operator falls captured in the episode.
        root_z_end = None
        tilt_max = None
        if "robot_root_position" in data.files:
            root_pos = np.asarray(data["robot_root_position"], dtype=np.float64)
            if root_pos.ndim == 2 and root_pos.shape[1] >= 3 and root_pos.shape[0]:
                root_z_end = float(root_pos[-1, 2])
        if "robot_root_orientation" in data.files:
            quat = np.asarray(data["robot_root_orientation"], dtype=np.float64)
            if quat.ndim == 2 and quat.shape[1] == 4 and quat.shape[0]:
                w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
                cos_tilt = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
                tilt_max = float(np.degrees(np.arccos(cos_tilt)).max())
        if root_z_end is not None and root_z_end < FALL_ROOT_Z_THRESHOLD:
            warnings.append(
                f"operator fall: final root_z={root_z_end:.3f} < {FALL_ROOT_Z_THRESHOLD}"
            )
        if tilt_max is not None and tilt_max > TILT_WARN_DEG:
            warnings.append(f"large root tilt {tilt_max:.1f} deg (> {TILT_WARN_DEG:.0f})")

        seed = _scalar(data, "episode_object_seed")
        seed_source = str(_scalar(data, "episode_object_seed_source", ""))
        if seed is None or int(seed) < 0:
            warnings.append(f"episode_object_seed not recorded (value={seed})")

        status = "FAIL" if errors else ("WARN" if warnings else "PASS")
        return {
            "path": str(npz_path),
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "schema": schema,
            "task": task,
            "num_frames": num_frames,
            "episode_object_seed": int(seed) if seed is not None else None,
            "episode_object_seed_source": seed_source,
            "root_z_end": root_z_end,
            "tilt_max_deg": tilt_max,
            "duration_s": float(wall[-1] - wall[0]) if wall is not None and wall.size > 1 else None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_roots", nargs="+", help="Directories containing MimicLite .npz episodes.")
    parser.add_argument("--json-out", type=str, default="", help="Write the full report as JSON.")
    parser.add_argument(
        "--skip-video-probe",
        action="store_true",
        help="Skip ffprobe frame-count verification of RGB sidecars (faster).",
    )
    args = parser.parse_args()

    npz_paths: list[Path] = []
    for root_raw in args.source_roots:
        root = Path(root_raw).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"source root not found: {root}")
        npz_paths.extend(sorted(root.rglob("*.npz")))

    npz_paths = [p for p in npz_paths if not p.name.endswith("_temp.npz")]
    if not npz_paths:
        raise SystemExit("no .npz episodes found")

    results = [
        check_episode(path, video_probe=not args.skip_video_probe) for path in npz_paths
    ]

    seed_counts: dict[int, int] = {}
    for result in results:
        seed = result.get("episode_object_seed")
        if seed is not None and seed >= 0:
            seed_counts[seed] = seed_counts.get(seed, 0) + 1
    duplicate_seeds = {seed: count for seed, count in seed_counts.items() if count > 1}
    if duplicate_seeds:
        for result in results:
            seed = result.get("episode_object_seed")
            if seed in duplicate_seeds:
                result["errors"].append(f"duplicate episode_object_seed={seed}")
                if result["status"] == "PASS":
                    result["status"] = "FAIL"

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    total_frames = sum(int(r.get("num_frames") or 0) for r in results)

    print(f"episodes={len(results)} pass={n_pass} warn={n_warn} fail={n_fail}")
    print(f"total_frames={total_frames}")
    if duplicate_seeds:
        print(f"duplicate episode_object_seed values: {duplicate_seeds}")

    for result in results:
        if result["status"] == "PASS":
            continue
        print(f"\n[{result['status']}] {result['path']}")
        for err in result["errors"]:
            print(f"  ERROR {err}")
        for warn in result["warnings"]:
            print(f"  WARN  {warn}")

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "source_roots": [str(Path(r).expanduser().resolve()) for r in args.source_roots],
                    "summary": {
                        "episodes": len(results),
                        "pass": n_pass,
                        "warn": n_warn,
                        "fail": n_fail,
                        "total_frames": total_frames,
                    },
                    "episodes": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nreport written to {out_path}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
