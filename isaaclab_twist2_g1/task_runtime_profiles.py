from __future__ import annotations

import datetime as _dt
import os
import zipfile as _zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as _np


TASK_RUNTIME_PROFILE_AUTO = "auto"
TASK_RUNTIME_PROFILE_INFERENCE = "inference"
TASK_RUNTIME_PROFILE_REPLAY_COMPAT = "replay_compat"

RUNTIME_CONTEXT_LIVE_INFERENCE = "live_inference"
RUNTIME_CONTEXT_DIRECT_REPLAY = "direct_replay"
RUNTIME_CONTEXT_INFERENCE_REPLAY = "inference_replay"
RUNTIME_CONTEXT_RERECORD = "rerecord"

OPEN_DOOR_TASK = "Isaac-Move-Open-Door-G129-Dex3-Wholebody"
OPEN_DOOR_DOOR_ASSET_VARIANT_ENV = "OPEN_DOOR_DOOR_ASSET_VARIANT"
OPEN_DOOR_DOOR_ASSET_VARIANT_INFERENCE_VALI = "inference_vali"
OPEN_DOOR_DOOR_ASSET_VARIANT_RECORDED_COMPAT = "recorded_compat"

# The door asset variant changed on 2026-05-13 (model_door001_vali_gate_welded.usd):
# episodes recorded after this instant were produced with the *inference* variant
# (kinematic door frame + working latch). Replaying such an episode with
# recorded_compat silently swaps the door frame from "gives way" to "nailed down",
# which only diverges once the robot touches the door.
OPEN_DOOR_INFERENCE_VALI_SINCE_US = int(
    _dt.datetime(2026, 5, 13, tzinfo=_dt.timezone.utc).timestamp() * 1_000_000
)

VISION_NAVI_TASK = "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody"
VISION_NAVI_ROOM_ASSET_VARIANT_ENV = "VISION_NAVI_ROOM_ASSET_VARIANT"
VISION_NAVI_ROOM_ASSET_VARIANT_VALIDATION = "validation"
VISION_NAVI_ROOM_ASSET_VARIANT_RECORDED_COMPAT = "recorded_compat"

OPEN_DOOR_TASK_PROFILES: dict[str, dict[str, str]] = {
    TASK_RUNTIME_PROFILE_INFERENCE: {
        "OPEN_DOOR_LATCH_DISABLE": "0",
        "OPEN_DOOR_SCENE_AS_ARTICULATION": "0",
        OPEN_DOOR_DOOR_ASSET_VARIANT_ENV: OPEN_DOOR_DOOR_ASSET_VARIANT_INFERENCE_VALI,
    },
    TASK_RUNTIME_PROFILE_REPLAY_COMPAT: {
        "OPEN_DOOR_LATCH_DISABLE": "1",
        "OPEN_DOOR_SCENE_AS_ARTICULATION": "0",
        OPEN_DOOR_DOOR_ASSET_VARIANT_ENV: OPEN_DOOR_DOOR_ASSET_VARIANT_RECORDED_COMPAT,
    },
}

# Vision-navigation uses two room USDs for different data contracts:
# - validation: optimized inference/validation scene, intended for current model evaluation and future data collection.
# - recorded_compat: original digital-twin scene used by released recordings; keep replay/rerecord on this
#   asset because small geometry differences can make existing demonstrations fail to replay.
VISION_NAVI_TASK_PROFILES: dict[str, dict[str, str]] = {
    TASK_RUNTIME_PROFILE_INFERENCE: {
        VISION_NAVI_ROOM_ASSET_VARIANT_ENV: VISION_NAVI_ROOM_ASSET_VARIANT_VALIDATION,
    },
    TASK_RUNTIME_PROFILE_REPLAY_COMPAT: {
        VISION_NAVI_ROOM_ASSET_VARIANT_ENV: VISION_NAVI_ROOM_ASSET_VARIANT_RECORDED_COMPAT,
    },
}

TASK_RUNTIME_RULES: dict[str, dict[str, Mapping[str, Mapping[str, str]] | Mapping[str, str]]] = {
    OPEN_DOOR_TASK: {
        "profiles": OPEN_DOOR_TASK_PROFILES,
        "default_profile_by_context": {
            RUNTIME_CONTEXT_LIVE_INFERENCE: TASK_RUNTIME_PROFILE_INFERENCE,
            RUNTIME_CONTEXT_DIRECT_REPLAY: TASK_RUNTIME_PROFILE_REPLAY_COMPAT,
            RUNTIME_CONTEXT_INFERENCE_REPLAY: TASK_RUNTIME_PROFILE_REPLAY_COMPAT,
            RUNTIME_CONTEXT_RERECORD: TASK_RUNTIME_PROFILE_REPLAY_COMPAT,
        },
    },
    VISION_NAVI_TASK: {
        "profiles": VISION_NAVI_TASK_PROFILES,
        "default_profile_by_context": {
            RUNTIME_CONTEXT_LIVE_INFERENCE: TASK_RUNTIME_PROFILE_INFERENCE,
            RUNTIME_CONTEXT_DIRECT_REPLAY: TASK_RUNTIME_PROFILE_REPLAY_COMPAT,
            RUNTIME_CONTEXT_INFERENCE_REPLAY: TASK_RUNTIME_PROFILE_REPLAY_COMPAT,
            RUNTIME_CONTEXT_RERECORD: TASK_RUNTIME_PROFILE_REPLAY_COMPAT,
        },
    },
}

OPEN_DOOR_DOOR_ASSET_RELATIVE_PATHS = {
    OPEN_DOOR_DOOR_ASSET_VARIANT_INFERENCE_VALI: (
        "assets/objects/small_warehouse/small_warehouse_opendoor/interaction_obj/"
        "door001/model_door001_vali_gate_welded.usd"
    ),
    OPEN_DOOR_DOOR_ASSET_VARIANT_RECORDED_COMPAT: (
        "assets/objects/small_warehouse/small_warehouse_opendoor/interaction_obj/"
        "door001/model_door001.usd"
    ),
}

VISION_NAVI_ROOM_ASSET_RELATIVE_PATHS = {
    VISION_NAVI_ROOM_ASSET_VARIANT_VALIDATION: (
        "assets/objects/small_warehouse/small_warehouse_vision_navigation/"
        "small_warehouse_digital_twin_validation.usd"
    ),
    VISION_NAVI_ROOM_ASSET_VARIANT_RECORDED_COMPAT: (
        "assets/objects/small_warehouse/small_warehouse_vision_navigation/"
        "small_warehouse_digital_twin.usd"
    ),
}


@dataclass(frozen=True)
class AppliedTaskRuntimeProfile:
    task_name: str
    context: str
    profile: str | None
    overrides: dict[str, str]


def normalize_replay_mode(replay_mode: str | None) -> str:
    token = str(replay_mode or "").strip().lower()
    if token in {"direct", "direct_replay"}:
        return RUNTIME_CONTEXT_DIRECT_REPLAY
    if token in {"inference", "inference_replay"}:
        return RUNTIME_CONTEXT_INFERENCE_REPLAY
    return RUNTIME_CONTEXT_LIVE_INFERENCE


def detect_runtime_context(args_cli) -> str:
    if bool(getattr(args_cli, "record_during_replay", False)):
        return RUNTIME_CONTEXT_RERECORD
    if getattr(args_cli, "input_source", "") == "replay" or getattr(args_cli, "replay_file", ""):
        return normalize_replay_mode(getattr(args_cli, "replay_mode", ""))
    return RUNTIME_CONTEXT_LIVE_INFERENCE


def resolve_task_runtime_profile(
    task_name: str,
    requested_profile: str,
    context: str,
) -> str | None:
    rules = TASK_RUNTIME_RULES.get(task_name)
    normalized_request = str(requested_profile or TASK_RUNTIME_PROFILE_AUTO).strip().lower()
    if rules is None:
        if normalized_request != TASK_RUNTIME_PROFILE_AUTO:
            raise ValueError(
                f"task_runtime_profile={requested_profile!r} is unsupported for task {task_name!r}"
            )
        return None
    profiles = rules["profiles"]
    if normalized_request == TASK_RUNTIME_PROFILE_AUTO:
        normalized_request = str(rules["default_profile_by_context"].get(context, "")).strip().lower()
        if not normalized_request:
            return None
    if normalized_request not in profiles:
        supported = ", ".join(sorted(profiles.keys()))
        raise ValueError(
            f"Unsupported task_runtime_profile={requested_profile!r} for task {task_name!r}; "
            f"supported profiles: {supported}"
        )
    return normalized_request


def resolve_task_runtime_env_overrides(
    task_name: str,
    requested_profile: str,
    context: str,
) -> dict[str, str]:
    resolved_profile = resolve_task_runtime_profile(task_name, requested_profile, context)
    if resolved_profile is None:
        return {}
    rules = TASK_RUNTIME_RULES[task_name]
    return dict(rules["profiles"][resolved_profile])


def apply_task_runtime_profile(
    args_cli,
    *,
    context: str | None = None,
    env: dict[str, str] | None = None,
) -> AppliedTaskRuntimeProfile:
    task_name = str(getattr(args_cli, "task", "") or "").strip()
    resolved_context = context or detect_runtime_context(args_cli)
    requested_profile = str(
        getattr(args_cli, "task_runtime_profile", TASK_RUNTIME_PROFILE_AUTO) or TASK_RUNTIME_PROFILE_AUTO
    )
    overrides = resolve_task_runtime_env_overrides(task_name, requested_profile, resolved_context)
    resolved_profile = resolve_task_runtime_profile(task_name, requested_profile, resolved_context)
    target_env = os.environ if env is None else env
    for key, value in overrides.items():
        target_env[key] = value
    if overrides:
        ordered = ", ".join(f"{key}={value}" for key, value in sorted(overrides.items()))
        print(
            f"[task_runtime_profile] task={task_name} context={resolved_context} "
            f"profile={resolved_profile} overrides={ordered}"
        )
    return AppliedTaskRuntimeProfile(
        task_name=task_name,
        context=resolved_context,
        profile=resolved_profile,
        overrides=overrides,
    )


def resolve_open_door_door_asset_variant(
    variant: str | None = None,
) -> str:
    normalized = str(
        variant
        or os.environ.get(OPEN_DOOR_DOOR_ASSET_VARIANT_ENV, OPEN_DOOR_DOOR_ASSET_VARIANT_INFERENCE_VALI)
    ).strip().lower()
    if normalized not in OPEN_DOOR_DOOR_ASSET_RELATIVE_PATHS:
        supported = ", ".join(sorted(OPEN_DOOR_DOOR_ASSET_RELATIVE_PATHS))
        raise ValueError(
            f"Unsupported {OPEN_DOOR_DOOR_ASSET_VARIANT_ENV}={normalized!r}; supported variants: {supported}"
        )
    return normalized


def resolve_open_door_door_usd_path(
    project_root: str | os.PathLike[str],
    variant: str | None = None,
) -> str:
    resolved_variant = resolve_open_door_door_asset_variant(variant)
    return str(Path(project_root) / OPEN_DOOR_DOOR_ASSET_RELATIVE_PATHS[resolved_variant])

def resolve_vision_navi_room_asset_variant(
    variant: str | None = None,
) -> str:
    normalized = str(
        variant
        or os.environ.get(VISION_NAVI_ROOM_ASSET_VARIANT_ENV, VISION_NAVI_ROOM_ASSET_VARIANT_RECORDED_COMPAT)
    ).strip().lower()
    if normalized not in VISION_NAVI_ROOM_ASSET_RELATIVE_PATHS:
        supported = ", ".join(sorted(VISION_NAVI_ROOM_ASSET_RELATIVE_PATHS))
        raise ValueError(
            f"Unsupported {VISION_NAVI_ROOM_ASSET_VARIANT_ENV}={normalized!r}; supported variants: {supported}"
        )
    return normalized


def resolve_vision_navi_room_usd_path(
    project_root: str | os.PathLike[str],
    variant: str | None = None,
) -> str:
    resolved_variant = resolve_vision_navi_room_asset_variant(variant)
    return str(Path(project_root) / VISION_NAVI_ROOM_ASSET_RELATIVE_PATHS[resolved_variant])


def read_replay_recording_timestamp_us(replay_file):
    """Read save_timestamp_us from a replay npz without loading the arrays."""
    if not replay_file:
        return None
    try:
        with _zipfile.ZipFile(str(replay_file)) as zf:
            if "save_timestamp_us.npy" not in zf.namelist():
                return None
            with zf.open("save_timestamp_us.npy") as fh:
                value = _np.load(fh, allow_pickle=True)
        flat = _np.asarray(value).reshape(-1)
        if flat.size == 0:
            return None
        return int(flat[0])
    except Exception:
        return None


def resolve_recording_aware_profile(task_name, replay_file, requested_profile):
    """Pick a runtime profile from the recording timestamp when the request is auto.

    Only open_door currently has a recording-time dependent asset variant; other tasks
    fall through to the existing auto (context based) behaviour."""
    normalized = str(requested_profile or TASK_RUNTIME_PROFILE_AUTO).strip().lower()
    if normalized != TASK_RUNTIME_PROFILE_AUTO:
        return normalized
    if str(task_name or "").strip() != OPEN_DOOR_TASK:
        return normalized
    timestamp_us = read_replay_recording_timestamp_us(replay_file)
    if timestamp_us is None:
        return normalized
    if timestamp_us >= OPEN_DOOR_INFERENCE_VALI_SINCE_US:
        return TASK_RUNTIME_PROFILE_INFERENCE
    return normalized
