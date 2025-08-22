# -----------------------------------------------------------------------------
# This file contains terminations functions developed for the Mk1 Humanoid Robot Project.
#
# Author: Wirdegeist
# Website: https://wirdegeist.com
# YouTube: https://www.youtube.com/@WiredGeist
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def root_height_below_minimum(env: ManagerBasedEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminates the episode if the root height is below the minimum height."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height