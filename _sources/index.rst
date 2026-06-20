:github_url: https://github.com/kevinzakka/mink/tree/main/docs/index.rst

####
mink
####

mink (**M**\ uJoCo **in**\ verse **k**\ inematics) is a library for differential
inverse kinematics built on top of the
`MuJoCo <https://github.com/google-deepmind/mujoco>`_ physics engine.

.. image:: https://github.com/kevinzakka/mink/blob/assets/banner.png?raw=true
   :alt: Banner for mink
   :align: center
   :width: 600px

.. raw:: html

   <br>

It solves the following problem: given a robot's current configuration and a set of
task-space objectives, compute a joint velocity that best satisfies those objectives
while respecting constraints.

Tasks specify what the robot should do (e.g., place a foot on a ledge or reach a
handhold), while limits specify what it must not do (e.g., exceed joint limits or
collide with obstacles). At each control step, mink formulates this as a
`quadratic program <https://en.wikipedia.org/wiki/Quadratic_programming>`_
and solves for a locally optimal joint velocity.

Differential inverse kinematics is widely used in motion planning, legged control,
teleoperation, and motion retargeting.

Key features
------------

- **Composable task abstraction**
  Specify multiple task-space objectives (e.g., foot placement and center-of-mass
  regulation) and combine them in a single solve step.

- **Graceful handling of constraints**
  Objectives and constraints are combined in a single optimization problem, enabling
  principled trade-offs when commands conflict or cannot be fully satisfied.

- **Designed for real-time control**
  A fast, local solver that runs at control rate, suitable for real-time loops.

Minimal example
---------------

Drive an end-effector to a target position:

.. code-block:: python

   import mujoco
   from mink import Configuration, FrameTask, solve_ik

   model = mujoco.MjModel.from_xml_path("robot.xml")
   configuration = Configuration(model)

   task = FrameTask(
       frame_name="end_effector",
       frame_type="site",
       position_cost=1.0,
       orientation_cost=0.0,
   )
   task.set_target_from_position([0.5, 0.0, 0.3])

   for _ in range(max_iters):
       vel = solve_ik(configuration, [task], dt=0.01)
       configuration.integrate_inplace(vel, dt=0.01)

- :doc:`installation`
- :doc:`tutorial/index`
- :doc:`api/index`
- :doc:`background/derivations`
- :doc:`references`

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   tutorial/index
   api/index
   background/derivations
   references
