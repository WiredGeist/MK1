# -----------------------------------------------------------------------------
# This file contains randomizations functions developed for the Mk1 Humanoid Robot Project.
#
# Author: Wirdegeist
# Website: https://wirdegeist.com
# YouTube: https://www.youtube.com/@WiredGeist
# -----------------------------------------------------------------------------

from __future__ import annotations
import torch
import math 
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def push_robot_torso(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    min_force: float,          
    max_force: float,          
    horizontal_angle_range: tuple[float, float] = (0.0, 2 * math.pi) 
):
    """
    Applies a random pushing force to the robot's torso, configured by EventCfg.
    """
    robot: Articulation = env.scene["robot"]
    torso_body_index = robot.find_bodies("torso_link")[0][0]

    # Iterate over each environment that needs a push
    for env_id in env_ids:
        # Generate a random angle from the provided range
        angle = (torch.rand(1, device=env.device) * (horizontal_angle_range[1] - horizontal_angle_range[0])) + horizontal_angle_range[0]
        
        # Calculate the direction vector based on the angle
        force_direction = torch.tensor([torch.cos(angle), torch.sin(angle), 0.0], device=env.device)
        
        # Generate a random force magnitude using the parameters from the config
        force_magnitude = (torch.rand(1, device=env.device) * (max_force - min_force)) + min_force
        
        # Combine to create the final force vector
        force = force_direction * force_magnitude
        
        # Create zero torque
        torques = torch.zeros(1, 3, device=env.device)

        # Apply the force to the single torso of the single environment
        robot.set_external_force_and_torque(
            forces=force.unsqueeze(0), # Add batch dimension
            torques=torques,
            body_ids=torso_body_index,
            env_ids=env_id.unsqueeze(0)
        )