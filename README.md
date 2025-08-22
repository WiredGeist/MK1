# Mk1 Humanoid Robot - Isaac Lab Environment

This repository contains all the necessary files to simulate and train the Mk1 humanoid robot, as featured in my YouTube series. This project provides the simulation assets (URDF, USD, meshes), the complete Isaac Lab environment code, and a pre-trained policy for active balancing.

**[Watch the full story of this robot's development on YouTube](https://www.youtube.com/@WiredGeist)**

---

## Prerequisites

Before you begin, ensure you have **NVIDIA Isaac Lab** installed and working correctly. This project is a custom environment and relies on that framework.

*   **NVIDIA Isaac Lab Repository:** [https://github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)

---

## Installation

Follow these steps carefully to set up the environment.

### 1. Place the Project Folder

Copy the entire `robotmk1` project folder into your Isaac Lab tasks directory. The recommended location is:

`<path_to_isaaclab>/source/isaaclab_tasks/isaaclab_tasks/manager_based/`
.../manager_based/
│
└── robotmk1/ <-- YOUR PROJECT FOLDER GOES HERE
├── init.py
├── robotmk1_env_active_balance_cfg.py
└── ... (other files)
code
Code
### 2. Configure the Environment Path

> [!IMPORTANT]
> This is the most critical step. If this path is wrong, the environment will not be found.

Open the `__init__.py` file inside your new `robotmk1` folder. You **must** edit the `env_cfg_entry_point` variable to match the location from Step 1.

*   **Find this line:**
    ```python
    "env_cfg_entry_point": "CHANGE_THIS_TO_YOUR_ENV_CONFIG_PATH.robotmk1_env_active_balance_cfg:RobotMk1ActiveBalanceEnvCfg",
    ```
*   **Change it to the correct Python import path:**
    ```python
    "env_cfg_entry_point": "isaaclab_tasks.manager_based.robotmk1.robotmk1_env_active_balance_cfg:RobotMk1ActiveBalanceEnvCfg",
    ```

### 3. Configure the Robot Asset Path

Open the `robotmk1_env_active_balance_cfg.py` file. You **must** edit the `robot_usd_path` variable to point to the absolute path of the `mk1.usd` file in this repository.

*   **Find this line:**
    ```python
    robot_usd_path = r"PATH TO THE ROBOT USD"
    ```
*   **Change it to the full path on your system, for example:**
    ```python
    robot_usd_path = r"C:\Users\YourName\Documents\GitHub\robotmk1\mk1.usd"
    ```

---

## Usage

All commands should be run from the root of your Isaac Lab repository.

### Running the Pre-trained Policy

This repository includes a working policy for active balancing. To see it in action, run the following command. Make sure to replace the placeholder with the correct path to the included `.pth` file.

## Project Structure

*   `mk1.urdf`: The Universal Robot Description Format file, defining the robot's physical properties.
*   `mk1.usd`: The Universal Scene Description file, used by Isaac Sim to render the robot.
*   `/meshes`: Contains the `.stl` files for each part of the robot's body.
*   `/robotmk1`: Contains all the Python code for the Isaac Lab environment.
*   `/weights`: Contains the pre-trained balancing policy (`robotmk1_balance.pth`).

---

## License

This project is released into the public domain under **The Unlicense**.

This means you are free to copy, modify, distribute, and use these files for any purpose, without any conditions.

While not required, a link back to my YouTube channel is always appreciated if you find this project helpful!

---

**Created by Wirdegeist**
*   **Website:** [https://wirdegeist.com](https://wirdegeist.com)
*   **YouTube:** [https://www.youtube.com/@WiredGeist](https://www.youtube.com/@WiredGeist)
