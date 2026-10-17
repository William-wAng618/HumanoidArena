from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:
    from tasks.common_scene.base_scene_pickplace_box import SHELF_INIT_POS
except Exception:
    # Keep this module importable in lightweight unit tests where Isaac Lab is unavailable.
    SHELF_INIT_POS = [-2.7, -0.15, 0.0]

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_BOX_SIZE_M = (0.21, 0.21, 0.21)
_BOX_HALF_EXTENTS_M = tuple(size * 0.5 for size in _BOX_SIZE_M)

_FALLBACK_SUPPORT_HALF_X_M = 0.45
_FALLBACK_SUPPORT_HALF_Y_M = 0.18
_MIN_SUPPORT_SPAN_X_M = _BOX_SIZE_M[0] + 0.02
_MIN_SUPPORT_SPAN_Y_M = _BOX_SIZE_M[1] + 0.02
_SHELF_BASE_SUPPORT_TOP_Z_AT_DEFAULT_M = 0.1090
_SHELF_ACTIVE_LAYER_TOP_Z_AT_DEFAULT_M = 0.6477
_SHELF_ACTIVE_LAYER_Z_FROM_BASE_M = (
    _SHELF_ACTIVE_LAYER_TOP_Z_AT_DEFAULT_M - _SHELF_BASE_SUPPORT_TOP_Z_AT_DEFAULT_M
)
_BOX_BOTTOM_SURFACE_TOLERANCE_M = 0.06


def _candidate_shelf_paths(env_idx: int) -> list[str]:
    base = f"/World/envs/env_{env_idx}/Shelf"
    return [base, f"{base}/PRootNode"]


def _read_prim_world_aabb_from_prim(
    bbox_cache,
    prim,
) -> tuple[float, float, float, float, float, float] | None:
    if prim is None or not prim.IsValid() or not prim.IsActive():
        return None

    try:
        aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    except Exception:
        return None

    mn = aligned_range.GetMin()
    mx = aligned_range.GetMax()
    if any(float(mx[idx] - mn[idx]) <= 0.0 for idx in range(3)):
        return None

    return (
        float(mn[0]),
        float(mx[0]),
        float(mn[1]),
        float(mx[1]),
        float(mn[2]),
        float(mx[2]),
    )


def _load_shelf_support_surfaces_world_from_stage(
    env: "ManagerBasedRLEnv",
) -> list[list[tuple[float, float, float, float, float]]]:
    surfaces_by_env: list[list[tuple[float, float, float, float, float]]] = [[] for _ in range(env.num_envs)]

    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception:
        return surfaces_by_env

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return surfaces_by_env

    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
    except Exception:
        return surfaces_by_env

    for env_idx in range(env.num_envs):
        for path in _candidate_shelf_paths(env_idx):
            shelf_root = stage.GetPrimAtPath(path)
            if shelf_root is None or not shelf_root.IsValid() or not shelf_root.IsActive():
                continue

            try:
                prim_range = Usd.PrimRange(shelf_root)
            except Exception:
                continue

            for prim in prim_range:
                box = _read_prim_world_aabb_from_prim(bbox_cache, prim)
                if box is None:
                    continue
                x_lo, x_hi, y_lo, y_hi, _z_lo, z_hi = box
                span_x = x_hi - x_lo
                span_y = y_hi - y_lo
                if span_x < _MIN_SUPPORT_SPAN_X_M or span_y < _MIN_SUPPORT_SPAN_Y_M:
                    continue
                surfaces_by_env[env_idx].append((x_lo, x_hi, y_lo, y_hi, z_hi))

            if surfaces_by_env[env_idx]:
                break

    return surfaces_by_env


def _scene_env_origins_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        return torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    return env_origins.to(device=env.device, dtype=torch.float32)


def _fallback_shelf_support_surfaces_world(
    env: "ManagerBasedRLEnv",
) -> list[list[tuple[float, float, float, float, float]]]:
    env_origins = _scene_env_origins_w(env)
    support_top_z_offset = _SHELF_BASE_SUPPORT_TOP_Z_AT_DEFAULT_M - float(SHELF_INIT_POS[2])

    surfaces_by_env: list[list[tuple[float, float, float, float, float]]] = []
    for env_idx in range(env.num_envs):
        origin = env_origins[env_idx]
        center_x = float(origin[0].item()) + float(SHELF_INIT_POS[0])
        center_y = float(origin[1].item()) + float(SHELF_INIT_POS[1])
        support_top_z = float(origin[2].item()) + float(SHELF_INIT_POS[2]) + support_top_z_offset
        surfaces_by_env.append(
            [
                (
                    center_x - _FALLBACK_SUPPORT_HALF_X_M,
                    center_x + _FALLBACK_SUPPORT_HALF_X_M,
                    center_y - _FALLBACK_SUPPORT_HALF_Y_M,
                    center_y + _FALLBACK_SUPPORT_HALF_Y_M,
                    support_top_z,
                )
            ]
        )
    return surfaces_by_env


def _get_shelf_support_surfaces_world(
    env: "ManagerBasedRLEnv",
) -> list[list[tuple[float, float, float, float, float]]]:
    stage_surfaces = _load_shelf_support_surfaces_world_from_stage(env)
    fallback_surfaces = _fallback_shelf_support_surfaces_world(env)
    resolved_surfaces = []
    surface_sources = []
    for stage_surface, fallback_surface in zip(stage_surfaces, fallback_surfaces, strict=False):
        if stage_surface:
            resolved_surfaces.append(stage_surface)
            surface_sources.append("stage")
        else:
            resolved_surfaces.append(fallback_surface)
            surface_sources.append("fallback")
    env._ppbox_surface_sources = surface_sources
    return resolved_surfaces


def _resolve_box_asset(env: "ManagerBasedRLEnv"):
    for scene_key in ("box", "Box"):
        try:
            return env.scene[scene_key]
        except Exception:
            continue
    raise KeyError("failed to locate box asset in env.scene using keys 'box' or 'Box'")


def _get_box_centers_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    box_asset = _resolve_box_asset(env)
    data = box_asset.data

    root_com_pos_w = getattr(data, "root_com_pos_w", None)
    if root_com_pos_w is not None:
        return root_com_pos_w

    root_pos_w = getattr(data, "root_pos_w", None)
    if root_pos_w is not None:
        return root_pos_w

    root_state_w = getattr(data, "root_state_w", None)
    if root_state_w is not None:
        return root_state_w[:, 0:3]

    raise AttributeError("box world-position tensor not found")


def _is_box_on_support_surface(
    box_center_w: torch.Tensor,
    surface_w: tuple[float, float, float, float, float],
    *,
    target_support_top_z: float,
) -> bool:
    x_lo, x_hi, y_lo, y_hi, _support_top_z = surface_w

    center_x = float(box_center_w[0].item())
    center_y = float(box_center_w[1].item())
    center_z = float(box_center_w[2].item())
    box_bottom_z = center_z - _BOX_HALF_EXTENTS_M[2]

    inside_xy = (
        (center_x >= x_lo + _BOX_HALF_EXTENTS_M[0])
        and (center_x <= x_hi - _BOX_HALF_EXTENTS_M[0])
        and (center_y >= y_lo + _BOX_HALF_EXTENTS_M[1])
        and (center_y <= y_hi - _BOX_HALF_EXTENTS_M[1])
    )
    aligned_z = abs(box_bottom_z - target_support_top_z) <= _BOX_BOTTOM_SURFACE_TOLERANCE_M
    return bool(inside_xy and aligned_z)


def _surface_debug_metrics(
    box_center_w: torch.Tensor,
    surface_w: tuple[float, float, float, float, float],
    *,
    target_support_top_z: float | None = None,
) -> dict[str, float | bool]:
    x_lo, x_hi, y_lo, y_hi, support_top_z = surface_w

    center_x = float(box_center_w[0].item())
    center_y = float(box_center_w[1].item())
    center_z = float(box_center_w[2].item())
    box_bottom_z = center_z - _BOX_HALF_EXTENTS_M[2]

    inner_x_lo = x_lo + _BOX_HALF_EXTENTS_M[0]
    inner_x_hi = x_hi - _BOX_HALF_EXTENTS_M[0]
    inner_y_lo = y_lo + _BOX_HALF_EXTENTS_M[1]
    inner_y_hi = y_hi - _BOX_HALF_EXTENTS_M[1]
    dx_outside = max(inner_x_lo - center_x, 0.0, center_x - inner_x_hi)
    dy_outside = max(inner_y_lo - center_y, 0.0, center_y - inner_y_hi)
    if target_support_top_z is None:
        target_support_top_z = support_top_z
    z_gap = box_bottom_z - target_support_top_z
    abs_z_gap = abs(z_gap)
    inside_xy = dx_outside == 0.0 and dy_outside == 0.0
    aligned_z = abs_z_gap <= _BOX_BOTTOM_SURFACE_TOLERANCE_M

    return {
        "inside_xy": inside_xy,
        "aligned_z": aligned_z,
        "matched": inside_xy and aligned_z,
        "dx_outside": dx_outside,
        "dy_outside": dy_outside,
        "z_gap": z_gap,
        "abs_z_gap": abs_z_gap,
        "support_top_z": support_top_z,
        "target_support_top_z": target_support_top_z,
        "raw_surface_top_z": support_top_z,
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "y_hi": y_hi,
    }


def _estimate_target_support_top_z(
    env: "ManagerBasedRLEnv",
    env_idx: int,
    box_center_w: torch.Tensor,
    surfaces_w: list[tuple[float, float, float, float, float]],
) -> tuple[float, float | None]:
    base_candidates: list[float] = []
    for surface_w in surfaces_w:
        metrics = _surface_debug_metrics(box_center_w, surface_w)
        if bool(metrics["inside_xy"]):
            base_candidates.append(float(surface_w[4]))

    if base_candidates:
        base_support_top_z = min(base_candidates)
        return (
            base_support_top_z + _SHELF_ACTIVE_LAYER_Z_FROM_BASE_M,
            base_support_top_z,
        )

    env_origins = _scene_env_origins_w(env)
    target_support_top_z = (
        float(env_origins[env_idx, 2].item())
        + float(SHELF_INIT_POS[2])
        + _SHELF_ACTIVE_LAYER_TOP_Z_AT_DEFAULT_M
    )
    return target_support_top_z, None


def _candidate_surface_debug_list(
    box_center_w: torch.Tensor,
    surfaces_w: list[tuple[float, float, float, float, float]],
    *,
    target_support_top_z: float | None = None,
    limit: int = 5,
) -> list[dict[str, float | bool | int]]:
    ranked: list[tuple[tuple[int, int, float, float, float], dict[str, float | bool | int]]] = []
    for idx, surface_w in enumerate(surfaces_w):
        metrics = _surface_debug_metrics(
            box_center_w,
            surface_w,
            target_support_top_z=target_support_top_z,
        )
        score = (
            0 if bool(metrics["inside_xy"]) else 1,
            0 if bool(metrics["aligned_z"]) else 1,
            float(metrics["dx_outside"]) + float(metrics["dy_outside"]),
            float(metrics["abs_z_gap"]),
            float(idx),
        )
        ranked.append((score, {"surface_index": idx, **metrics}))

    ranked.sort(key=lambda item: item[0])
    return [item[1] for item in ranked[:limit]]


def _best_surface_debug(
    box_center_w: torch.Tensor,
    surfaces_w: list[tuple[float, float, float, float, float]],
    *,
    target_support_top_z: float | None = None,
) -> dict[str, float | bool | int] | None:
    if not surfaces_w:
        return None

    best_idx = -1
    best_metrics: dict[str, float | bool] | None = None
    best_score: tuple[int, int, float, float, float] | None = None

    for idx, surface_w in enumerate(surfaces_w):
        metrics = _surface_debug_metrics(
            box_center_w,
            surface_w,
            target_support_top_z=target_support_top_z,
        )
        score = (
            0 if bool(metrics["matched"]) else 1,
            0 if bool(metrics["inside_xy"]) else 1,
            float(metrics["dx_outside"]) + float(metrics["dy_outside"]),
            float(metrics["abs_z_gap"]),
            float(idx),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx
            best_metrics = metrics

    if best_metrics is None:
        return None

    return {"surface_index": best_idx, **best_metrics}


def compute_success_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    box_centers_w = _get_box_centers_world(env)
    shelf_surfaces_w = _get_shelf_support_surfaces_world(env)

    for env_idx in range(env.num_envs):
        target_support_top_z, _base_support_top_z = _estimate_target_support_top_z(
            env,
            env_idx,
            box_centers_w[env_idx],
            shelf_surfaces_w[env_idx],
        )
        success[env_idx] = any(
            _is_box_on_support_surface(
                box_centers_w[env_idx],
                surface_w,
                target_support_top_z=target_support_top_z,
            )
            for surface_w in shelf_surfaces_w[env_idx]
        )

    return success


def compute_reward_pickplace_box(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Binary reward for the pick-place box task.

    Reward is ``1`` when the box is judged to be on the shelf and ``0`` otherwise.
    The check uses the live shelf geometry from the USD stage when available and falls
    back to scene constants only in non-Isaac unit-test environments.
    """

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    success = compute_success_mask(env)
    reward[success] = 1.0

    if env.num_envs >= 1:
        i = 0
        box_centers_w = _get_box_centers_world(env)
        shelf_surfaces_w = _get_shelf_support_surfaces_world(env)
        box_center = box_centers_w[i]
        box_bottom_z = float(box_center[2].item()) - _BOX_HALF_EXTENTS_M[2]
        target_support_top_z, base_support_top_z = _estimate_target_support_top_z(
            env,
            i,
            box_center,
            shelf_surfaces_w[i],
        )
        best_surface = _best_surface_debug(
            box_center,
            shelf_surfaces_w[i],
            target_support_top_z=target_support_top_z,
        )
        candidate_surfaces = _candidate_surface_debug_list(
            box_center,
            shelf_surfaces_w[i],
            target_support_top_z=target_support_top_z,
        )
        surface_sources = getattr(env, "_ppbox_surface_sources", [])
        surface_source = surface_sources[i] if i < len(surface_sources) else "unknown"

        dbg = {
            "task": "pickplace_box",
            "com": (
                float(box_center[0].item()),
                float(box_center[1].item()),
                float(box_center[2].item()),
            ),
            "box_bottom_z": box_bottom_z,
            "box_half_extents": _BOX_HALF_EXTENTS_M,
            "surface_count": len(shelf_surfaces_w[i]),
            "surface_source": surface_source,
            "base_support_top_z": base_support_top_z,
            "target_support_top_z": target_support_top_z,
            "active_layer_top_z_at_default": _SHELF_ACTIVE_LAYER_TOP_Z_AT_DEFAULT_M,
            "active_layer_z_from_base": _SHELF_ACTIVE_LAYER_Z_FROM_BASE_M,
            "placed": bool(success[i].item()),
            "in_goal": bool(success[i].item()),
            "goal_reward": float(reward[i].item()),
            "raw_total": float(reward[i].item()),
            "candidate_surfaces": candidate_surfaces,
        }
        if isinstance(best_surface, dict):
            dbg.update(best_surface)
        env._ppbox_dbg = dbg
        env._ppdd_dbg = dbg

    return reward


__all__ = [
    "compute_reward_pickplace_box",
    "compute_success_mask",
]
