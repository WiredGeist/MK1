# -----------------------------------------------------------------------------
# This file contains rewards functions developed for the Mk1 Humanoid Robot Project.
#
# Author: Wirdegeist
# Website: https://wirdegeist.com
# YouTube: https://www.youtube.com/@WiredGeist
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- Standard Reward Functions/Classes ---

def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()

def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


def action_symmetry_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes asymmetrical actions between left and right limbs for a bipedal robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    actions = env.action_manager.action

    # Define joint pairs that should have mirrored (opposite) actions for symmetry (roll and yaw)
    mirrored_pairs = [
        ("left_hip_roll_joint", "right_hip_roll_joint"),
        ("left_hip_yaw_joint", "right_hip_yaw_joint"),
        ("left_shoulder_roll_joint", "right_shoulder_roll_joint"),
    ]
    
    # Define joint pairs that should have identical actions for symmetry (pitch)
    identical_pairs = [
        ("left_hip_pitch_joint", "right_hip_pitch_joint"),
        ("left_knee_pitch_joint", "right_knee_pitch_joint"),
        ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
        ("left_elbow_pitch_joint", "right_elbow_pitch_joint"),
        ("left_wrist_pitch_joint", "right_wrist_pitch_joint"),
    ]

    total_symmetry_error = torch.zeros(env.num_envs, device=env.device)

    # Calculate penalty for mirrored pairs (error is left_action + right_action)
    for left_joint_name, right_joint_name in mirrored_pairs:
        left_idx = asset.find_joints(left_joint_name)[0][0]
        right_idx = asset.find_joints(right_joint_name)[0][0]
        
        left_action = actions[:, left_idx]
        right_action = actions[:, right_idx]
        
        # For mirrored actions, their sum should be zero
        total_symmetry_error += torch.square(left_action + right_action)

    # Calculate penalty for identical pairs (error is left_action - right_action)
    for left_joint_name, right_joint_name in identical_pairs:
        left_idx = asset.find_joints(left_joint_name)[0][0]
        right_idx = asset.find_joints(right_joint_name)[0][0]
        
        left_action = actions[:, left_idx]
        right_action = actions[:, right_idx]
        
        # For identical actions, their difference should be zero
        total_symmetry_error += torch.square(left_action - right_action)
        
    return total_symmetry_error

class progress_reward(ManagerTermBase):
    """Rewards the robot for making progress towards a target location.
    
    This is a potential-based reward, calculated as the change in distance to the
    target between timesteps. It encourages steady movement towards the goal.
    """
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt
        return self.potentials - self.prev_potentials

class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalizes the robot for moving joints close to their position limits.
    
    The penalty is scaled by the gear ratio of each joint, applying a larger
    penalty to joints with higher gearing, simulating higher stress.
    """
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled
        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)

class power_consumption(ManagerTermBase):
    """Penalizes the robot for high power consumption.
    
    This is calculated as the sum of `|torque * velocity|` for each joint,
    scaled by the gear ratio. It encourages energy-efficient movements.
    """
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        return torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)

def fall_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Applies a large penalty if the robot's torso height falls below a threshold, indicating a fall."""
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] < 0.2).float()

def linear_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for moving forward in the x direction of the world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 0]

def quat_to_euler_rpy(q: torch.Tensor) -> torch.Tensor:
    """Helper function to convert quaternion to roll, pitch, yaw."""
    roll = torch.atan2(2.0 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]), 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    pitch = torch.asin(2.0 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]))
    yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    return torch.stack([roll, pitch, yaw], dim=-1)

def roll_pitch_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes the robot for rolling or pitching."""
    asset: Articulation = env.scene[asset_cfg.name]
    euler_rpy = quat_to_euler_rpy(asset.data.root_quat_w)
    return torch.sum(torch.square(euler_rpy[:, 0:2]), dim=1)

def base_lin_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes the root linear velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)

def base_ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes the root angular velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)

def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes the joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)

def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """A constant reward for every step that the agent is alive."""
    return torch.ones(env.num_envs, device=env.device)


def action_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes the rate of change of actions to encourage smoothness."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def joint_pos_limits_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes being close to joint limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_bounds = torch.abs(asset.data.joint_pos - (asset.data.soft_joint_pos_limits[..., 1] + asset.data.soft_joint_pos_limits[..., 0]) / 2.0)
    out_of_bounds /= (asset.data.soft_joint_pos_limits[..., 1] - asset.data.soft_joint_pos_limits[..., 0]) / 2.0
    return torch.sum(torch.square(out_of_bounds), dim=1)

def base_height_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Rewards the robot for maintaining a high torso height.

    This is a simple reward that is directly proportional to the robot's
    root z-position in the world frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2]

def is_base_tilted(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminates if the base's up-vector projection is below a threshold."""
    up_projection = obs.base_up_proj(env, asset_cfg=asset_cfg)
    return torch.squeeze(up_projection < threshold, dim=-1)

def action_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes the squared magnitude of the actions."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)

def joint_pos_symmetry_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes asymmetrical joint positions between left and right limbs."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos

    mirrored_pairs = [
        ("left_hip_roll_joint", "right_hip_roll_joint"),
        ("left_hip_yaw_joint", "right_hip_yaw_joint"),
        ("left_shoulder_roll_joint", "right_shoulder_roll_joint"),
    ]
    
    identical_pairs = [
        ("left_hip_pitch_joint", "right_hip_pitch_joint"),
        ("left_knee_pitch_joint", "right_knee_pitch_joint"),
        ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
        ("left_elbow_pitch_joint", "right_elbow_pitch_joint"),
        ("left_wrist_pitch_joint", "right_wrist_pitch_joint"),
    ]

    total_symmetry_error = torch.zeros(env.num_envs, device=env.device)
    for left_joint_name, right_joint_name in mirrored_pairs:
        left_idx = asset.find_joints(left_joint_name)[0][0]
        right_idx = asset.find_joints(right_joint_name)[0][0]
        
        left_pos = joint_pos[:, left_idx]
        right_pos = joint_pos[:, right_idx]
        
        total_symmetry_error += torch.abs(left_pos + right_pos)

    for left_joint_name, right_joint_name in identical_pairs:
        left_idx = asset.find_joints(left_joint_name)[0][0]
        right_idx = asset.find_joints(right_joint_name)[0][0]
        
        left_pos = joint_pos[:, left_idx]
        right_pos = joint_pos[:, right_idx]
        
        total_symmetry_error += torch.abs(left_pos - right_pos)
        
    return total_symmetry_error

def joint_pos_target_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes the squared difference from a target joint configuration (i.e., a default pose)."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    target_joint_pos = torch.zeros_like(asset.data.joint_pos)
    
    return torch.sum(torch.square(asset.data.joint_pos - target_joint_pos), dim=1)