import mujoco
import numpy as np

from mink import SE3, SO3, Configuration, FrameTask, solve_ik

# Load robot.
model = mujoco.MjModel.from_xml_path("franka_emika_panda/mjx_scene.xml")
configuration = Configuration(model)
configuration.update_from_keyframe("home")

# Define target pose relative to current end-effector.
ee_pose = configuration.get_transform_frame_to_world("attachment_site", "site")
translation = np.array([0.0, -0.4, -0.2])
rotation = SO3.from_y_radians(-np.pi / 2)
target = SE3.from_translation(translation) @ ee_pose
target = target @ SE3.from_rotation(rotation)

# Create task with gain for smooth 2-second convergence.
duration, fps = 2.0, 60
n_frames = int(duration * fps)
gain = 1.0 - 0.01 ** (1.0 / n_frames)

task = FrameTask(
    frame_name="attachment_site",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=1.0,
    gain=gain,
)
task.set_target(target)

# Run IK loop.
dt = 1.0 / fps
for _ in range(n_frames):
    vel = solve_ik(configuration, [task], dt, "daqp")
    configuration.integrate_inplace(vel, dt)

# Check result.
final = configuration.get_transform_frame_to_world("attachment_site", "site")
print(
    f"Position error: {np.linalg.norm(final.translation() - target.translation()):.2e} m"
)
