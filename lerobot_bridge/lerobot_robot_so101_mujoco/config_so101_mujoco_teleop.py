"""Config for the SO-101 MuJoCo gamepad teleoperator.

Registered as teleoperator type "so101_mujoco_teleop" -- select it with
`--teleop.type=so101_mujoco_teleop`. See so101_mujoco_teleop.py's module docstring for
why this reuses so101_mujoco_env.joint_teleop.GamepadJointTeleop internally rather than
reimplementing gamepad reading here.
"""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_mujoco_teleop")
@dataclass
class SO101MujocoTeleopConfig(TeleoperatorConfig):
    # Defaults to so101_mujoco_env/config/gamepad.config.yaml (resolved relative to
    # this installed package's own location).
    gamepad_config_path: str | None = None

    # If True, bypass pygame/gamepad hardware entirely and drive a small deterministic
    # per-joint sine sweep instead. Exists so the Robot/Teleoperator/dataset plumbing
    # can be smoke-tested (e.g. `lerobot-record --teleop.mock=true`) without a physical
    # gamepad attached -- never enabled implicitly; a real recording session should
    # always leave this False.
    mock: bool = False
    mock_period_s: float = 4.0
    mock_amplitude: float = 0.4  # fraction of the real gamepad's max normalized action
