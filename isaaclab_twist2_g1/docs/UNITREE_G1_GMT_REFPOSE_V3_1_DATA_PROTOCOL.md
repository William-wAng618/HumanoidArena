# Unitree G1 GMT Reference Pose V3.1 LeRobot Format

This document defines the LeRobot dataset format used for HumanoidArena VLA training across the TWIST2 and SONIC backends.

Implementation references:

- `isaaclab_twist2_g1/tools/data_tools/smpl_lerobot_v3_common.py`
- `isaaclab_twist2_g1/tools/data_tools/twist2lerobot_rotlocal_v3.py`
- `isaaclab_twist2_g1/tools/data_tools/sonic2lerobot_rotlocal_v3.py`
- `isaaclab_twist2_g1/tools/data_tools/mimic_lite2lerobot_64_40.py`
- `isaaclab_twist2_g1/action_provider/vla_robot_current_local_runtime_v3.py`

## Schema

```text
schema: unitree_g1_gmt_refpose_v3_1
robot_type: unitree_g1_refpose_v3_1
state_dim: 64
action_dim: 40
```

The format is a reference-pose protocol. The VLA action is a reference motion target, not a backend tracking residual.

```text
VLA action = reference motion target
VLA action != inverse(Q_robot_current_t) * Q_target_t
```

## LeRobot Features

The converted LeRobot dataset stores:

- `observation.image`: RGB frame from the episode camera stream.
- `observation.state`: 64D robot state.
- `action`: 40D reference-pose action.
- `task`: normalized task name.

The converter writes protocol metadata with the same schema name so downstream code can reject incompatible datasets.

## Observation State Layout

`observation.state` has 64 dimensions:

```text
state[0:6]    state.root_heading_canonical_rot6d
state[6:35]   state.dof_pos.<canonical_g1_joint>
state[35:64]  state.dof_vel.<canonical_g1_joint>
```

The root orientation is heading-canonicalized by the initial robot heading. Joint positions and velocities use the canonical Unitree G1 29-joint order.

## Action Layout

`action` has 40 dimensions:

```text
action[0:2]   action.root_ref_base_local_xy_delta.{x,y}
action[2]     action.root_z
action[3:9]   action.root_ref_rot6d.{0..5}
action[9:38]  action.joint_pos.<canonical_g1_joint>
action[38:40] action.hand_binary.{left,right}
```

Field meanings:

- `root_ref_base_local_xy_delta`: reference root XY displacement from frame `t-1` to `t`, expressed in the current reference root/base frame.
- `root_z`: reference root height at frame `t`.
- `root_ref_rot6d`: 6D rotation representation of the reference root orientation in the episode reference frame.
- `joint_pos`: canonical Unitree G1 29D joint reference target in radians.
- `hand_binary`: left/right binary hand command.

## Episode Reference Convention

The format removes arbitrary global yaw and position from the reference trajectory. Conceptually each episode has a reference frame anchored at the first reference pose:

```text
P_anchor = P_ref_0
Q_anchor = heading(Q_ref_0)
```

Reference rotations and root deltas are expressed relative to that episode convention. The action label must not subtract the simulated robot current root orientation.

## SONIC Conversion

The SONIC converter uses the recorded human/body reference fields:

- `human_body_pos` or fallback `robot_root_position`
- `human_body_quat_w` or fallback `robot_root_orientation`
- `human_joint_pos`, `vla_action_joint_pos_29`, or `vla_action_raw[:, 9:38]`

For each frame it builds:

```text
root_ref_base_local_xy_delta = local displacement of the reference body
root_z                       = body_pos.z
root_ref_rot6d               = rot6d(reference body orientation in episode frame)
joint_pos                    = canonical 29D joint target
hand_binary                  = recorded hand binary
```

## TWIST2 Conversion

The TWIST2 converter reconstructs the reference trajectory from the native TWIST2 command:

```text
robot_action_mimic = [vx_base, vy_base, z, roll, pitch, yaw_vel, joint_pos_29]
```

This is preferred over reconstructing from robot tracking error because `robot_action_mimic` is what the TWIST2 low-level backend consumed during recording.

For each frame it builds:

```text
root_ref_base_local_xy_delta = [vx_base, vy_base] * control_dt
root_z                       = z
root_ref_rot6d               = rot6d(integrated roll/pitch/yaw reference)
joint_pos                    = TWIST2 joints reordered to canonical 29D order
hand_binary                  = recorded hand binary
```

For frame 0, `root_ref_base_local_xy_delta` is `[0, 0]`.

## MimicLite Conversion

The MimicLite converter uses the recorded raw reference fields:

- `human_raw_body_pos`
- `human_raw_body_quat_w`
- `human_raw_joint_pos`
- `human_raw_controller_data_json` for Pico grip binary labels, or compatible recorded hand-binary fields

For each episode it removes the initial reference heading:

```text
Q_anchor = heading(human_raw_body_quat_w[0])
root_ref_rot6d = rot6d(inverse(Q_anchor) * human_raw_body_quat_w[t])
```

Root XY deltas are computed from consecutive `human_raw_body_pos` samples, rotated by `inverse(Q_anchor)`, then expressed in the current reference root/base frame. The converter must not align MimicLite actions to the simulated robot's initial yaw and must not keep raw world-frame yaw.

## Required Source NPZ Fields

Common fields:

- `robot_qpos_before_decimation`: `(T, 29)`
- `robot_qvel_before_decimation`: `(T, 29)`
- `robot_root_orientation`: `(T, 4)`
- `vision_rgb` or `vision_rgb_video_path`
- `vision_frame_indices` when image frames are a subset of Isaac control frames

TWIST2-specific fields:

- `robot_action_mimic` preferred, or `robot_obs_buf` with `observation_semantics` for legacy extraction.

SONIC-specific fields:

- `human_body_pos` / `human_body_quat_w`
- `human_joint_pos` or compatible VLA joint target fields.

MimicLite-specific fields:

- `human_raw_body_pos`
- `human_raw_body_quat_w`
- `human_raw_joint_pos`
- `human_raw_controller_data_json` or compatible hand-binary fields.

## Validation

Use the matching verifier after conversion:

```bash
python isaaclab_twist2_g1/tools/data_tools/verify_lerobot_rotlocal_v3.py   --dataset-root /path/to/lerobot_dataset   --source-root /path/to/source_npz_root
```

The verifier checks the schema, robot type, image frame alignment, action reconstruction, and hand binary values.
