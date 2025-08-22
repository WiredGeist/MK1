# ==============================================================================
# Mk1 Humanoid Robot Project - Environment Configuration
#
# Author:  Wirdegeist
# Website: https://wirdegeist.com
# YouTube: https://www.youtube.com/@WiredGeist
# ------------------------------------------------------------------------------
# This file defines the environment configuration for the active balance task.
# ==============================================================================
#
# USAGE:
#
# To Train:
# isaaclab.bat -p scripts/reinforcement_learning/rl_games/train.py --task RobotMK1-ActiveBalance-v0 --num_envs 128 --max_iterations 5000 --headless
#
# To Play a Trained Policy:
# isaaclab.bat -p scripts/reinforcement_learning/rl_games/play.py --task RobotMK1-ActiveBalance-v0 --num_envs 1 --checkpoint <PATH TO THE TRAINED WEIGHT /robotmk1_balance.pth>
#
# ==============================================================================

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from .mdp import *

# [!!!] ACTION REQUIRED [!!!]
# You MUST change the placeholder path below to the absolute path of your robot's USD file.
robot_usd_path = r"PATH TO THE ROBOT USD"
if not os.path.exists(robot_usd_path):
    raise FileNotFoundError(f"Full-body Robot USD file not found at: {robot_usd_path}")

@configclass
class MySceneCfg(InteractiveSceneCfg):
    # Ground plane for the robot to stand on
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    # The robot articulation configuration
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=robot_usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.18),
            # Sets the initial base height of the robot
            # Defines the default starting pose of the robot's joints.
            # Explicitly setting all joints to 0.0 is good practice to avoid ambiguity.
            joint_pos={
                # Left Leg Crouched Pose
                "left_hip_pitch_joint": 0.0,
                "left_knee_pitch_joint": 0.0,
                # Right Leg Crouched Pose
                "right_hip_pitch_joint": 0.0,
                "right_knee_pitch_joint": 0.0,

                # -- Set all other joints explicitly to 0.0 --
                # Hips
                "left_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                # Left Arm
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_elbow_pitch_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # Right Arm
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_elbow_pitch_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
            },
        ),
        # Defines the properties of the robot's motors (actuators).
        # We use an ImplicitActuatorCfg, which simulates a PD controller.
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # Stiffness is the 'P' (Proportional) gain of the PD controller.
                stiffness={
                    ".*hip.*": 300.0,
                    ".*knee.*": 250.0,
                    ".*shoulder.*": 200.0,
                    ".*elbow.*": 100.0,
                    ".*wrist.*": 50.0
                },
                # Damping is the 'D' (Derivative) gain of the PD controller.
                damping={
                    ".*hip.*": 30.0,
                    ".*knee.*": 25.0,
                    ".*shoulder.*": 20.0,
                    ".*elbow.*": 10.0,
                    ".*wrist.*": 5.0
                },
                velocity_limit=20.0
            ),
        },
    )
    # Basic lighting for the scene
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=3000.0))

@configclass
class ActionsCfg:
    """Defines the action space of the environment."""
    # The policy controls the target position of each joint.
    # `use_default_offset=True` means the actions are added to the default joint positions.
    joint_pos = JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Defines the observation space for the policy (what the AI "sees")."""
    @configclass
    class PolicyCfg(ObsGroup):
        """All observations are concatenated into a single vector for the policy."""
        # Base state observations
        base_height = ObsTerm(func=base_pos_z, params={"asset_cfg": SceneEntityCfg("robot")})
        base_lin_vel = ObsTerm(func=base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_up_proj = ObsTerm(func=base_up_proj, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # Joint state observations
        joint_pos_norm = ObsTerm(func=joint_pos_limit_normalized, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
       
        # Last action observation (helps the policy learn smoothness)
        actions = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # The single observation group is named "policy"
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Defines the reward function for the reinforcement learning task."""
    # -- Primary Goal: Stay Upright --
    # Large reward for maintaining an upright posture
    upright = RewTerm(func=upright_posture_bonus, weight=10.0, params={"threshold": 0.9})
    # Large penalty for falling over
    fall_penalty = RewTerm(func=fall_penalty, weight=-10.0)

    # -- Secondary Goal: Encourage a Neutral Resting Pose --
    # Penalize deviation from the default joint positions to encourage stability
    default_pose_penalty = RewTerm(func=joint_pos_target_l2, weight=-0.5) 

    # -- Penalties to Discourage Unnecessary Movement --
    # Penalize linear velocity of the torso
    torso_velocity_penalty = RewTerm(func=base_lin_vel_l2, weight=-1.0)
    # Penalize angular velocity of the torso (wobbling)
    torso_wobble_penalty = RewTerm(func=base_ang_vel_l2, weight=-0.5)
    
    # -- Penalties for Efficient and Smooth Actions --
    # Penalize large changes in action between steps to encourage smooth movements
    action_smoothness_penalty = RewTerm(func=action_rate_l2, weight=-0.01)
    # Penalize large action magnitudes to conserve energy
    action_magnitude_penalty = RewTerm(func=action_l2, weight=-0.01)

    # -- Penalties to Encourage Natural, Symmetrical Poses --
    # Penalize asymmetrical actions between left and right limbs
    action_symmetry = RewTerm(func=action_symmetry_penalty, weight=-1.0)
    # Penalize asymmetrical joint positions between left and right limbs
    pose_symmetry = RewTerm(func=joint_pos_symmetry_penalty, weight=-1.0)

@configclass
class TerminationsCfg:
    """Defines the conditions under which an episode ends."""
    # Terminate after a fixed amount of time
    time_out = DoneTerm(func=time_out, time_out=True)
    # Terminate if the robot falls over (the main failure condition)
    orientation_failure = DoneTerm(
        func=is_base_tilted,
        params={"threshold": 0.6, "asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class EventCfg:
    """Defines events that occur during the simulation, such as resets or randomizations."""
    # On reset, place the robot at the specified starting pose
    reset_base = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.18, 0.18)},
            "velocity_range": {},
        },
    )
    # Periodically apply a random push to the robot's torso.
    # This is the "active" part of active balancing and forces the policy to be more robust.
    push_robot = EventTerm(
        func=push_robot_torso,
        mode="interval",
        interval_range_s=(1.5, 3.5),
        params={
            "min_force": 0.5,
            "max_force": 1.0,
            "horizontal_angle_range": (0.0, 6.28), # Push from any horizontal direction
        },
    )

@configclass
class RobotMk1ActiveBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """The main configuration class that ties all the components together."""
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0)
    # RL settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfy = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # Event settings
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post-initialization checks and settings."""
        # The name of the environment
        self.name = "RobotMk1ActiveBalance"
        # How many simulation steps to run per action from the policy
        self.decimation = 2
        # The total length of each episode in seconds
        self.episode_length_s = 20.0
        # The physics simulation timestep
        self.sim.dt = 1 / 120.0