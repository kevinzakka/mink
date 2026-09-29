import mujoco
import numpy as np

from mink import (
    SE3,
    Configuration,
    ConfigurationLimit,
    Elastic,
    FrameTask,
    PostureTask,
    VelocityLimit,
    solve_ik,
)

model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
configuration = Configuration(model)
configuration.update_from_keyframe("home")

task = FrameTask(
    frame_name="attachment_site",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=1.0,
)

# Elastic constraint: held exactly while feasible, yields when it is not.
# Orientation has a lower penalty than position, so it gives way first.
elastic = Elastic(task, penalty=[1e3, 1e3, 1e3, 1e1, 1e1, 1e1])

# The posture task pulls toward home, but cannot bias the elastic constraint.
posture_task = PostureTask(model, cost=0.1)
posture_task.set_target_from_configuration(configuration)

tasks = [posture_task]
limits = [
    ConfigurationLimit(model),
    VelocityLimit(model, {model.joint(i).name: np.pi for i in range(model.njnt)}),
]

home_pose = configuration.get_transform_frame_to_world("attachment_site", "site")
dt = 0.01

# Reachable target: converges exactly despite the posture task.
task.set_target(home_pose @ SE3.from_translation(np.array([0.1, -0.05, 0.05])))
for _ in range(50):
    vel = solve_ik(
        configuration, tasks, dt, "daqp", limits=limits, constraints=[elastic]
    )
    configuration.integrate_inplace(vel, dt)
assert np.linalg.norm(task.compute_error(configuration)) < 1e-6

# Unreachable target: the constraint yields instead of raising NoSolutionFound.
task.set_target(home_pose @ SE3.from_translation(np.array([2.0, 0.0, 0.0])))
for _ in range(50):
    vel = solve_ik(
        configuration, tasks, dt, "daqp", limits=limits, constraints=[elastic]
    )
    configuration.integrate_inplace(vel, dt)
