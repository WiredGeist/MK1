# ==============================================================================
# Mk2 Humanoid Robot Project - Gym Environment Registration
#
# Author:  Wirdegeist
# Website: https://wirdegeist.com
# YouTube: https://www.youtube.com/@WiredGeist
# ------------------------------------------------------------------------------
# This file registers the custom Mk2 environment with Gymnasium,
# making it available to be created with `gym.make()`.
# ==============================================================================


import gymnasium as gym
from . import agents

gym.register(
    id="RobotMK1-ActiveBalance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # ----------------------------------------------------------------------
        # [!!!] INSTALLATION & SETUP [!!!]
        #
        # This project is a custom environment for the Isaac Lab framework.
        # You must have Isaac Lab installed first. You can find it here:
        #   https://github.com/isaac-sim/IsaacLab
        #
        # To make this environment visible to Isaac Lab, you must perform two steps:
        #
        # 1. PLACE THIS PROJECT FOLDER IN THE CORRECT DIRECTORY:
        #    Copy the entire 'robotmk1' project folder into the Isaac Lab tasks directory.
        #    The recommended location is:
        #
        #    <path_to_isaaclab>/source/isaaclab_tasks/isaaclab_tasks/manager_based/
        #    └── robotmk1/  <-- YOUR PROJECT FOLDER GOES HERE
        #        ├── __init__.py (this file)
        #        ├── robotmk1_env_active_balance_cfg.py
        #        └── ... (other files)
        #
        # 2. UPDATE THE CONFIGURATION PATH BELOW:
        #    Change the placeholder to the correct Python import path that reflects
        #    the location from Step 1.
        #
        #    Based on the example structure above, the correct path would be:
        #      "isaaclab_tasks.manager_based.robotmk1.robotmk1_env_active_balance_cfg:RobotMk1ActiveBalanceEnvCfg"
        # ----------------------------------------------------------------------
        "env_cfg_entry_point": "CHANGE_THIS_TO_YOUR_ENV_CONFIG_PATH.robotmk1_env_active_balance_cfg:RobotMk1ActiveBalanceEnvCfg",

        # This line points to the RL training hyperparameters (YAML file).
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)