import sys
from pathlib import Path

import numpy as np

from mink import (
    SE3,
    Configuration,
    ConfigurationLimit,
    DofFreezingTask,
    FrameTask,
    PostureTask,
    VelocityLimit,
    solve_ik,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _model_assets import load_example_spec  # noqa: E402

# Load and configure.
model = load_example_spec(
    "franka_emika_panda", Path("franka_emika_panda/mjx_scene.xml")
).compile()
configuration = Configuration(model)
configuration.update_from_keyframe("home")

# Primary task: pose tracking.
task = FrameTask(
    frame_name="attachment_site",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=1.0,
)

# Regularization: bias toward home configuration.
# This handles both singularities (resists extreme velocities near them)
# and nullspace drift (fills unused DOFs with a preference).
posture_task = PostureTask(model, cost=0.1)
posture_task.set_target_from_configuration(configuration)

tasks = [task, posture_task]

# Limits: joint bounds + velocity cap.
velocity_limits = {f"joint{i}": 2.0 for i in range(1, 8)}
limits = [
    ConfigurationLimit(model),
    VelocityLimit(model, velocity_limits),
]

# Constraint: hold the forearm roll (joint5) still. Enforced exactly; the six
# remaining joints suffice to track the circle.
freeze_task = DofFreezingTask(model, dof_indices=[4])

# IK loop: track a target that traces a small circle in front of the arm.
home_pose = configuration.get_transform_frame_to_world("attachment_site", "site")
dt = 0.01
steps = 200
for step in range(steps):
    angle = 2.0 * np.pi * step / steps
    offset = np.array([0.0, 0.1 * np.cos(angle), 0.1 * np.sin(angle)])
    task.set_target(
        SE3.from_translation(offset) @ home_pose
    )  # Update target each step.
    vel = solve_ik(
        configuration, tasks, dt, "daqp", limits=limits, constraints=[freeze_task]
    )
    configuration.integrate_inplace(vel, dt)
