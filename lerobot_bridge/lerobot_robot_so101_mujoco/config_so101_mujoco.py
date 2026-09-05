"""Config for the in-process SO-101 MuJoCo digital twin, exposed to LeRobot as a Robot.

Registered as robot type "so101_mujoco" -- select it with `--robot.type=so101_mujoco`
on any lerobot-* CLI, the same way the earlier so101_sim2real project's digital twin is
selected with `--robot.type=so101_gazebo`. See so101_mujoco.py's module docstring for
why this one owns MuJoCo directly in-process instead of talking to a separate process
over ZMQ the way SO101Gazebo has to.
"""

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("so101_mujoco")
@dataclass
class SO101MujocoConfig(RobotConfig):
    # Defaults to assets/scenes/pen_pickplace_scene.xml (resolved relative to this
    # installed package's own location -- see so101_mujoco.py's _DEFAULT_SCENE_XML).
    scene_xml: str | None = None

    # Matches so101_mujoco_env/pen_pickplace_env.py's own defaults, so a policy trained
    # against that gym.Env sees the same physics-step cadence when later deployed here.
    control_dt: float = 0.05  # 20 Hz -- matches gamepad.config.yaml's control_rate_hz
    physics_dt: float = 0.002

    # Redrawn on every connect() and on every reset_scene(randomize_can_pose=True) call
    # (the latter is NOT part of the Robot ABC -- see reset_scene()'s own docstring for
    # why a MuJoCo-specific reset hook exists at all).
    randomize_can_pose_on_connect: bool = True

    # (height, width, channels) per camera -- NOT a `cameras: dict[str, CameraConfig]`
    # field. RobotConfig.__post_init__ assumes any field literally named `cameras` holds
    # real CameraConfig objects and calls getattr(config, "width"/"height"/"fps") on each
    # without a default, which raises AttributeError on plain shape tuples. This robot
    # renders its own frames via mujoco.Renderer rather than using LeRobot's
    # CameraConfig/make_cameras_from_configs device abstraction, so the field is
    # deliberately named to avoid tripping that validation -- same trick
    # lerobot_robot_so101_gazebo/config_so101_gazebo.py uses for the same reason.
    #
    # 480x480 matches joint_teleop.py's own --cam-size default -- deliberately, so the
    # camera-preview window during recording looks as sharp as the one already proven
    # during Phase 3 teleop testing. Measured cost of going this high (vs. an earlier,
    # much blurrier 128x128 default): get_observation() ~28ms -> ~30ms per tick (mujoco's
    # offscreen renderer is dominated by fixed per-call overhead, not pixel count, on
    # this hardware) -- negligible next to the ~20Hz control loop's 50ms budget, so there
    # was no real performance reason to have rendered this small in the first place.
    camera_shapes: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {"front": (480, 480, 3), "wrist": (480, 480, 3)}
    )
