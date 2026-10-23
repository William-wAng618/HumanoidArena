# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Single-goal football scene configuration for G1 wholebody tasks.
Keeps the same object/ground/backdrop layout as the standard football scene,
but only spawns one goal net.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from tasks.common_config import CameraBaseCfg

project_root = os.environ.get("PROJECT_ROOT")

ROBOT_INIT_X = 0.0
ROBOT_INIT_Y = 0.0
ROBOT_INIT_Z = 0.8
GOAL_DISTANCE = 3.0
BALL_DISTANCE = 1.0
GOAL_Z = 0.65
GOAL_CENTER_Y_OFFSET = 2.5
GOAL_COLLISION_ENABLED = True
GOAL_CONTACT_OFFSET = 0.001
GOAL_REST_OFFSET = 0.0

BALL_OFFSET_X = 0.0
BALL_OFFSET_Y = BALL_DISTANCE
GOAL_OFFSET_X = -GOAL_CENTER_Y_OFFSET
GOAL_OFFSET_Y = GOAL_DISTANCE

# Semantic-interference basketball constants.
# The dynamic basketball for the semantic test is registered as a scene-level
# RigidObject (spawned at scene build -> PhysX simulates it normally). For
# non-semantic runs it is parked far below the ground with gravity disabled,
# fully invisible and inert; the semantic test yaml moves it to its air spawn
# point and re-enables gravity.
BASKETBALL_LOCAL_OFFSET = (2.463015057933452, -2.503791749264172, 0.2791186808994101)
SEMANTIC_BASKETBALL_HIDDEN_POS = (0.0, 0.0, -200.0)
SEMANTIC_BASKETBALL_MASS = 1.0
GOAL_NET_1_ORIGIN = (ROBOT_INIT_X + GOAL_OFFSET_X, ROBOT_INIT_Y + GOAL_OFFSET_Y)
GOAL_NET_2_ORIGIN = (ROBOT_INIT_X - GOAL_OFFSET_X, ROBOT_INIT_Y - GOAL_OFFSET_Y)
GOAL_NET_ORIGIN_TO_CENTER_LOCAL = (0.0, 0.0)
GOAL_NET_1_CENTER = (
    GOAL_NET_1_ORIGIN[0] + GOAL_NET_ORIGIN_TO_CENTER_LOCAL[0],
    GOAL_NET_1_ORIGIN[1] + GOAL_NET_ORIGIN_TO_CENTER_LOCAL[1],
)
GOAL_NET_2_CENTER = (
    GOAL_NET_2_ORIGIN[0] - GOAL_NET_ORIGIN_TO_CENTER_LOCAL[0],
    GOAL_NET_2_ORIGIN[1] - GOAL_NET_ORIGIN_TO_CENTER_LOCAL[1],
)
GOAL_BACKDROP_THICKNESS = 0.08
GOAL_BACKDROP_HEIGHT = 2.6
GOAL_BACKDROP_Y_GAP = 3.0
GOAL_BACKDROP_SIDE_MARGIN_X = 4.3
GOAL_BACKDROP_SIDE_X = abs(GOAL_OFFSET_X) + GOAL_BACKDROP_SIDE_MARGIN_X
GOAL_BACKDROP_HALF_Y = GOAL_OFFSET_Y + GOAL_BACKDROP_Y_GAP
GOAL_BACKDROP_WIDTH = GOAL_BACKDROP_SIDE_X * 2.0 + GOAL_BACKDROP_THICKNESS
GOAL_BACKDROP_SIDE_LENGTH = GOAL_BACKDROP_HALF_Y * 2.0 + GOAL_BACKDROP_THICKNESS
GOAL_BACKDROP_COLOR = (0.35, 0.35, 0.35)


@configclass
class TableFootballSceneCfgWH(InteractiveSceneCfg):
    """Football scene with a single goal net."""

    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[BALL_OFFSET_X, BALL_OFFSET_Y, 0.11],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/football_scene/interaction_obj/soccer_ball/soccer_ball_physics.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )

    ground = RigidObjectCfg(
        prim_path="/World/GroundPlane",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, -0.005],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(14.0, 14.0, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.5,
                restitution=0.1,
            ),
        ),
    )

    goal_net = AssetBaseCfg(
        prim_path="/World/envs/env_.*/GoalNet",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[GOAL_NET_1_ORIGIN[0], GOAL_NET_1_ORIGIN[1], GOAL_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/football_scene/interaction_obj/football_goal/football_goal_physics.usd",
            # scale=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=GOAL_COLLISION_ENABLED,
                contact_offset=GOAL_CONTACT_OFFSET,
                rest_offset=GOAL_REST_OFFSET,
            ),
        ),
    )

    # Semantic-interference basketball (dynamic).
    # The semantic goal USD embeds a "basketball_semantic" child at a fixed
    # local pose, but GoalNet is a kinematic asset, so that embedded ball is
    # glued to the goal hierarchy and never falls. Instead of patching bodies
    # at runtime (runtime-created rigid bodies are not simulated by PhysX),
    # the dynamic basketball is registered HERE at scene build: it falls with
    # gravity and interacts like any other rigid object. The embedded child is
    # deactivated by tools/semantic_basketball.ensure_semantic_basketballs()
    # so the semantic scene shows exactly one ball.
    semantic_basketball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SemanticBasketball",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(SEMANTIC_BASKETBALL_HIDDEN_POS),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/football_scene/interaction_obj/basketball/basketball_physics.usda",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=SEMANTIC_BASKETBALL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )

    goal_backdrop_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalBackdrop1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X, ROBOT_INIT_Y + GOAL_BACKDROP_HALF_Y, GOAL_BACKDROP_HEIGHT * 0.5],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(GOAL_BACKDROP_WIDTH, GOAL_BACKDROP_THICKNESS, GOAL_BACKDROP_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_BACKDROP_COLOR),
        ),
    )

    goal_backdrop_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalBackdrop2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X, ROBOT_INIT_Y - GOAL_BACKDROP_HALF_Y, GOAL_BACKDROP_HEIGHT * 0.5],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(GOAL_BACKDROP_WIDTH, GOAL_BACKDROP_THICKNESS, GOAL_BACKDROP_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_BACKDROP_COLOR),
        ),
    )

    goal_backdrop_3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalBackdrop3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X - GOAL_BACKDROP_SIDE_X, ROBOT_INIT_Y, GOAL_BACKDROP_HEIGHT * 0.5],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(GOAL_BACKDROP_THICKNESS, GOAL_BACKDROP_SIDE_LENGTH, GOAL_BACKDROP_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_BACKDROP_COLOR),
        ),
    )

    goal_backdrop_4 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalBackdrop4",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X + GOAL_BACKDROP_SIDE_X, ROBOT_INIT_Y, GOAL_BACKDROP_HEIGHT * 0.5],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(GOAL_BACKDROP_THICKNESS, GOAL_BACKDROP_SIDE_LENGTH, GOAL_BACKDROP_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_BACKDROP_COLOR),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    world_camera = CameraBaseCfg.get_world_camera_config(
        pos_offset=(3.00604, -1.18692, 1.99495),
        rot_offset=(0.70141, 0.51577, 0.29143, 0.39632),
        focal_length=12,
        horizontal_aperture=27,
        convention="opengl",
    )
