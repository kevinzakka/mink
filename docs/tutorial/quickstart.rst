:github_url: https://github.com/kevinzakka/mink/tree/main/docs/tutorial/quickstart.rst

.. _quickstart:

===============
Getting Started
===============

This page introduces mink's core concepts through a simple example: driving a
robot arm's end-effector to a target pose.

.. raw:: html

   <video width="400" controls style="display: block; margin: 0 auto;">
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/quickstart.mp4" type="video/mp4">
   </video>

Setup
=====

The following example uses the Panda arm from the ``examples/`` directory in the
mink repository. First, load the model and create a configuration:

.. code:: python

   import mujoco
   import numpy as np

   from mink import Configuration

   model = mujoco.MjModel.from_xml_path("franka_emika_panda/mjx_scene.xml")
   configuration = Configuration(model)
   configuration.update_from_keyframe("home")

:class:`~mink.Configuration` wraps a MuJoCo model and provides the kinematic
quantities needed for IK: frame poses, Jacobians, and integration.
See :doc:`mujoco_basics` for details.

Defining a Target Pose
======================

mink represents rigid body transformations using :class:`~mink.SE3` (poses) and
:class:`~mink.SO3` (rotations). These classes support composition via ``@`` and
provide convenient constructors. See :doc:`lie_groups` for details.

A target pose can be defined relative to the current end-effector pose by
applying a transformation:

.. code:: python

   from mink import SE3, SO3

   # Get current end-effector pose.
   ee_pose = configuration.get_transform_frame_to_world("attachment_site", "site")

   # Define target: translate in world frame, then rotate in local frame.
   translation = np.array([0.0, -0.4, -0.2])
   rotation = SO3.from_y_radians(-np.pi / 2)

   target = SE3.from_translation(translation) @ ee_pose  # World frame.
   target = target @ SE3.from_rotation(rotation)         # Local frame.

Left-multiplying applies a transformation in the **world frame**. The translation
moves the pose along world axes. Right-multiplying applies it in the **local
frame**. The rotation is therefore about the gripper's own y-axis, not the
world's.

This produces a target that is offset from the current pose and rotated 90°.
Decoupling translation and rotation makes each component easier to reason about
independently.

Creating a Task
===============

A :class:`~mink.FrameTask` drives a frame toward a target pose:

.. code:: python

   from mink import FrameTask

   task = FrameTask(
       frame_name="attachment_site",
       frame_type="site",
       position_cost=1.0,
       orientation_cost=1.0,
   )
   task.set_target(target)

Setting both ``position_cost`` and ``orientation_cost`` to non-zero values
creates a full 6-DOF pose task. Set ``orientation_cost=0.0`` for a position-only
task where the end-effector can rotate freely.

Controlling Convergence Speed
=============================

The ``gain`` parameter sets the desired rate at which the task error should decay.
With gain :math:`g \in [0, 1]`, the error after :math:`n` steps is approximately:

.. math::

   e_n = e_0 \cdot (1 - g)^n

To reach 1% of the initial error after :math:`n` frames, solve for:

.. math::

   g = 1 - 0.01^{1/n}

For example, to reduce error to 1% of initial over 2 seconds at 60 fps (120
frames):

.. code:: python

   duration = 2.0  # seconds
   fps = 60
   n_frames = int(duration * fps)
   gain = 1.0 - 0.01 ** (1.0 / n_frames)  # ≈ 0.038

   task = FrameTask(
       frame_name="attachment_site",
       frame_type="site",
       position_cost=1.0,
       orientation_cost=1.0,
       gain=gain,
   )

Lower gain values yield smoother, more gradual convergence. The default
``gain=1.0`` produces maximum convergence rate.

.. note::

   This formula assumes first-order error dynamics, which holds when the
   Jacobian is well-conditioned and the target is reachable.

The IK Loop
===========

At each step, :func:`~mink.solve_ik` computes a joint velocity intended to reduce
task error:

.. code:: python

   from mink import solve_ik

   dt = 1.0 / fps
   for _ in range(n_frames):
       vel = solve_ik(configuration, [task], dt)
       configuration.integrate_inplace(vel, dt)

The ``dt`` parameter is the time between successive calls to ``solve_ik``.
Internally, the QP solves for a configuration displacement :math:`\Delta q` and
enforces velocity limits as :math:`\|\Delta q\| \leq v_{\max} \cdot dt`. These
constraints scale correctly with ``dt``, so limit enforcement is independent of
the chosen timestep. The main practical effect of ``dt`` is on step size: larger
values produce larger configuration updates, which can degrade accuracy when the
first-order linearization becomes invalid over that step. Note that ``dt`` need
not equal the robot's servo rate; IK may run at a different frequency, with a
lower-level controller interpolating between updates.

:meth:`~mink.Configuration.integrate_inplace` updates joint positions via
``q ← q + vel * dt``, using proper quaternion integration for ball and free
joints.

Complete Example
================

.. code:: python

   import mujoco
   import numpy as np

   from mink import Configuration, FrameTask, SE3, SO3, solve_ik

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
       vel = solve_ik(configuration, [task], dt)
       configuration.integrate_inplace(vel, dt)

   # Check result.
   final = configuration.get_transform_frame_to_world("attachment_site", "site")
   print(f"Position error: {np.linalg.norm(final.translation() - target.translation()):.2e} m")

Next Steps
==========

This example demonstrated a single pose task with controlled convergence. For
more robust IK formulations incorporating regularization and limits, see
:doc:`tasks_and_limits`.
