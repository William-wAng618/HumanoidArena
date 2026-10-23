"""Runtime helper for football semantic distractor basketballs.

The semantic goal USD carries the basketball as a child prim of GoalNet
(``/World/envs/env_N/GoalNet/basketball_semantic``). GoalNet is intentionally
spawned as a kinematic scene asset, so that embedded ball is treated as part
of the static goal hierarchy and never falls. Runtime-created rigid bodies are
also NOT simulated by PhysX in this pipeline (verified), so the dynamic
basketball must be registered at scene build time: the football-single scene
cfg (common_scene/base_scene_football_single_cfg_wholebody.py) defines a
scene-level ``RigidObjectCfg semantic_basketball`` which the semantic test yaml
moves to the air spawn point and activates (gravity on).

This helper therefore ONLY deactivates the embedded copy inside the semantic
goal USD, so the scene shows exactly one ball (the dynamic scene-level one).
"""

import os


BASKETBALL_LOCAL_OFFSET = (2.463015057933452, -2.503791749264172, 0.2791186808994101)
DEFAULT_GOAL_ORIGIN = (-2.5, 3.0)
DEFAULT_GOAL_Z = 0.65
DEFAULT_BASKETBALL_MASS = 1.0


def ensure_semantic_basketballs(
    goal_origin: tuple[float, float] = DEFAULT_GOAL_ORIGIN,
    goal_z: float = DEFAULT_GOAL_Z,
    asset_path: str | None = None,
    force: bool = False,
) -> bool:
    """Deactivate the basketball embedded inside the semantic goal USD.

    Callers should pass ``force=True`` when the semantic test config is active
    (e.g. when the goal_net spawn USD path contains ``physics_semantic``); the
    legacy ``TEST_MODE=semantic`` env-var gate is kept for backward
    compatibility.

    Returns True if any embedded child prim was deactivated.
    """
    if not force and os.environ.get("TEST_MODE") != "semantic":
        return False

    try:
        import omni.usd
    except ImportError as exc:
        print(f"[semantic_basketball] skipped: cannot import USD modules: {exc}")
        return False

    stage = omni.usd.get_context().get_stage()
    envs_prim = stage.GetPrimAtPath("/World/envs")
    if envs_prim and envs_prim.IsValid():
        env_paths = [str(child.GetPath()) for child in envs_prim.GetChildren() if child.IsActive()]
    else:
        env_paths = ["/World"]

    deactivated = 0
    for env_path in env_paths:
        for fixed_path in (
            f"{env_path}/GoalNet/basketball_semantic",
            f"{env_path}/GoalNet/football_goal/basketball_semantic",
        ):
            fixed_child = stage.GetPrimAtPath(fixed_path)
            if fixed_child and fixed_child.IsValid() and fixed_child.IsActive():
                fixed_child.SetActive(False)
                deactivated += 1

    if deactivated:
        print(f"[semantic_basketball] deactivated {deactivated} embedded child prim(s)")
    return deactivated > 0