import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from common_env_objects import apply_deterministic_object_resets
from tasks.common_config import CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_pickplace_box import (
    PickPlaceBoxSceneCfg,
)

ROBOT_INIT_POS = (-1.6, -5.0, 0.8)
ROBOT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


@configclass
class PickPlaceBoxTaskSceneCfg(PickPlaceBoxSceneCfg):
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
        init_pos=ROBOT_INIT_POS,
        init_rot=ROBOT_INIT_ROT,
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    front_camera = CameraPresets.g1_front_camera()


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    pass


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward_pickplace_box, weight=1.0)


@configclass
class EventCfg:
    pass


@configclass
class MovePickPlaceBoxG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceBoxTaskSceneCfg = PickPlaceBoxTaskSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.object_reset_seed_source = "time"
        # Pose ranges are world-frame meters after asset scaling. Yaw ranges are radians.
        self.deterministic_object_resets = [
            {
                "record_name": "box",
                "scene_keys": ["box", "Box"],
                "pose_range": {
                    "x": [-0.58, -0.40],
                    "y": [-5.63, -3.92],
                    "z": [0.75, 0.75],
                    "yaw": [-0.785398, 0.785398],
                },
                "zero_velocity_on_reset": True,
            },
            {
                "record_name": "shelf",
                "prim_paths": ["/World/envs/env_{env_idx}/Shelf"],
                "pose_range": {
                    "x": [-3.10, -0.90],
                    "y": [-2.40, -1.00],
                    "z": [0.0, 0.0],
                    "yaw": [-0.261799, 0.261799],
                },
                "zero_velocity_on_reset": True,
            },
        ]
        self._replay_initial_env_state_active = False

        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(func=lambda env: self._reset_object_self(env)),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(func=lambda env: self._reset_all_self(env)),
        )

    def initialize_task_scene(self, env, args_cli=None):
        self._replay_initial_env_state_active = (
            bool(getattr(args_cli, "replay_file", "")) if args_cli else False
        )
        self._make_shelf_static_collision(env)

    def _reset_object_self(self, env):
        applied = apply_deterministic_object_resets(
            self,
            env,
            selected_record_names={"box", "shelf"},
        )
        if applied:
            print("[object_reset] " + ", ".join(applied))

    def _reset_all_self(self, env):
        base_mdp.reset_scene_to_default(
            env,
            torch.arange(env.num_envs, device=env.device),
        )
        applied = apply_deterministic_object_resets(self, env)
        if applied:
            print("[object_reset] " + ", ".join(applied))

    def _make_shelf_static_collision(self, env):
        try:
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            removed = []
            for env_idx in range(env.num_envs):
                shelf_root = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Shelf")
                if not shelf_root or not shelf_root.IsValid():
                    continue
                for prim in Usd.PrimRange(shelf_root):
                    for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
                        if prim.HasAPI(api):
                            prim.RemoveAPI(api)
                            api_name = getattr(api, "__name__", str(api))
                            removed.append(f"{prim.GetPath().pathString}:{api_name}")
            if removed:
                print(
                    "[shelf_static_collision] removed dynamic physics APIs: "
                    + ", ".join(removed)
                )
            else:
                print("[shelf_static_collision] no dynamic physics APIs found under /Shelf")
        except Exception as exc:
            print(f"[shelf_static_collision] failed: {exc}")
