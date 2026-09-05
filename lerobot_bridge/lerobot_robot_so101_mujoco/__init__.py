"""LeRobot Robot/Teleoperator pair for the in-process SO-101 MuJoCo digital twin.

Importing this package registers both `--robot.type=so101_mujoco` and
`--teleop.type=so101_mujoco_teleop` against LeRobot's draccus choice registries (the
@RobotConfig.register_subclass / @TeleoperatorConfig.register_subclass decorators in
config_so101_mujoco.py / config_so101_mujoco_teleop.py run at import time). Installed
as a distribution named `lerobot_robot_so101_mujoco` (the `lerobot_robot_` prefix,
underscored not hyphenated) so LeRobot's `register_third_party_plugins()` auto-imports
it before CLI parsing on every `lerobot-*` command -- no explicit import needed in user
scripts. Same pattern as the earlier so101_sim2real project's
lerobot_robot_so101_gazebo package.
"""

from .config_so101_mujoco import SO101MujocoConfig
from .config_so101_mujoco_teleop import SO101MujocoTeleopConfig
from .so101_mujoco import SO101Mujoco
from .so101_mujoco_teleop import SO101MujocoTeleop

__all__ = [
    "SO101MujocoConfig",
    "SO101Mujoco",
    "SO101MujocoTeleopConfig",
    "SO101MujocoTeleop",
]
