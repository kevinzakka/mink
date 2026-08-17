:github_url: https://github.com/kevinzakka/mink/tree/main/docs/tutorial/tasks_and_limits.rst

.. _tasks-and-limits:

================
Tasks and Limits
================

The :doc:`quickstart` drove a robot arm to a target pose under ideal conditions:
the target was reachable, joint limits were never approached, and the arm had
enough freedom to satisfy the task. In practice, these assumptions rarely hold.

This page introduces **tasks** and **limits**, mink's mechanism for handling
the trade-offs that arise when they do not. All examples use the same Panda arm
as the quickstart.

.. grid:: 2

   .. grid-item-card:: **Tasks** (soft objectives)

      A *task* expresses something the robot should try to do. Tasks can be
      violated when necessary; when multiple tasks conflict, mink resolves
      them by minimizing a weighted sum of their errors.

      - Move an end-effector to a pose (:class:`~mink.FrameTask`)
      - Keep joints near a nominal configuration (:class:`~mink.PostureTask`)
      - Regulate the center of mass (:class:`~mink.ComTask`)

   .. grid-item-card:: **Limits** (hard constraints)

      A *limit* describes a boundary the robot must not cross. Limits are
      strictly enforced; the solver will never return a solution that
      violates them.

      - Joint position bounds (:class:`~mink.ConfigurationLimit`)
      - Velocity bounds (:class:`~mink.VelocityLimit`)
      - Free-joint velocity bounds (:class:`~mink.FreeJointVelocityLimit`)
      - Collision avoidance (:class:`~mink.CollisionAvoidanceLimit`)

Internally, tasks become quadratic objectives in the QP and limits become
inequality constraints.

.. note::

   :class:`~mink.ConfigurationLimit` is applied automatically when no custom
   ``limits`` list is provided. All other limits must be added explicitly.

What Goes Wrong Without Regularization
=======================================

A single :class:`~mink.FrameTask` is sufficient when the target is well within
the workspace. At the workspace boundary, however, two failure modes emerge.

Singularities
-------------

Consider tracking a target that moves outward, requiring the arm to extend
toward its reach limit:

.. code:: python

   # Move outward toward near-full extension.
   start_pos = np.array([0.5, 0.0, 0.4])
   end_pos = np.array([0.8, 0.0, 0.4])

As the target moves away from the base, the arm must straighten. Near full
extension, the elbow approaches a **singularity**: the Jacobian becomes
ill-conditioned, and small task-space errors demand unbounded joint velocities.

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/without_regularization.mp4" type="video/mp4">
   </video>

*Without regularization: the arm becomes unstable near full extension.
Note the self-collision between the hand and upper arm.*

Nullspace Drift
---------------

The opposite problem occurs when the robot has *more* degrees of freedom than
the task requires. With position-only tracking (``orientation_cost=0.0``), a
7-DOF arm has four extra degrees of freedom. The solver can move the elbow,
rotate the wrist, or reconfigure the arm arbitrarily without affecting the
end-effector position.

This is **nullspace drift**: uncontrolled motion in degrees of freedom that do
not affect the task. Common symptoms include elbow drift, unnecessary
inter-frame joint motion, and non-reproducible poses.

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/nullspace_without_posture.mp4" type="video/mp4">
   </video>

*Position-only tracking: the elbow drifts unpredictably.*

Adding Regularization
=====================

Both failure modes stem from an underconstrained QP. **Regularization** adds
secondary objectives that resolve the ambiguity, trading a small amount of
tracking accuracy for bounded, predictable motion.

mink provides two regularization tasks:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Task
     - Behavior
     - Best for
   * - DampingTask
     - Minimizes per-step motion
     - Numerical stability, when no preferred pose exists
   * - PostureTask
     - Biases toward a reference configuration
     - Redundancy resolution, manipulation with a preferred pose

Either task addresses both singularities and nullspace drift. The following
sections demonstrate one natural pairing, but they are interchangeable.

In addition, ``solve_ik(..., damping=...)`` provides a lightweight numerical
damping term that improves conditioning without adding a task. Per-task
``lm_damping`` adds error-dependent Levenberg-Marquardt damping for more
conservative behavior near infeasible targets. See the :doc:`/api/inverse_kinematics`
reference for details.

Fixing Singularities
--------------------

:class:`~mink.DampingTask` penalizes large values of Δq:

.. code:: python

   from mink import DampingTask

   task = FrameTask(
       frame_name="attachment_site",
       frame_type="site",
       position_cost=1.0,
       orientation_cost=1.0,
   )

   # Regularization: prefer smaller joint velocities.
   damping_task = DampingTask(model, cost=0.5)

   for _ in range(steps):
       vel = solve_ik(configuration, [task, damping_task], dt, "daqp")
       configuration.integrate_inplace(vel, dt)

Near a singularity, satisfying the primary task requires large joint velocities.
The damping task penalizes these, causing the QP to accept some tracking error
in exchange for bounded motion.

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/with_regularization.mp4" type="video/mp4">
   </video>

*With regularization: the arm accepts tracking error to stay stable.*

.. note::

   Regularization does not prevent the arm from reaching singular
   configurations; it bounds the commanded velocities near them by accepting
   reduced tracking accuracy.

Fixing Nullspace Drift
----------------------

:class:`~mink.PostureTask` biases toward a reference configuration:

.. code:: python

   from mink import PostureTask

   posture_task = PostureTask(model, cost=0.1)
   posture_task.set_target_from_configuration(configuration)  # Use current as reference.

The posture task provides a secondary objective that resolves nullspace
ambiguity: redundant degrees of freedom are biased toward the reference.

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/nullspace_with_posture.mp4" type="video/mp4">
   </video>

*With posture task: the arm stays close to its reference configuration.*

Adding Limits
=============

Regularization keeps the solver well-conditioned but does not enforce physical
constraints. Without explicit limits, joints can exceed their mechanical range,
velocities can exceed motor capabilities, and self-collision can occur. Limits
encode these boundaries as inequality constraints in the QP.

Configuration Limits
--------------------

:class:`~mink.ConfigurationLimit` keeps joints within their mechanical range.
When a custom ``limits`` list is provided, this limit must be included
explicitly:

.. code:: python

   from mink import ConfigurationLimit

   limits = [ConfigurationLimit(model)]

   for _ in range(steps):
       vel = solve_ik(configuration, [task], dt, "daqp", limits=limits)
       configuration.integrate_inplace(vel, dt)

Velocity Limits
---------------

:class:`~mink.VelocityLimit` bounds the per-step displacement
(:math:`\|\Delta q\| \leq v_{\max} \cdot dt`), preventing commands that would
require excessive motor torques:

.. code:: python

   from mink import VelocityLimit

   velocity_limits = {
       "joint1": 2.0,  # rad/s
       "joint2": 2.0,
       # ... etc
   }
   velocity_limit = VelocityLimit(model, velocity_limits)

   limits = [ConfigurationLimit(model), velocity_limit]

If the target moves faster than the velocity limits allow, tracking will lag.
Set limits based on the hardware's actual motor capabilities.

Collision Avoidance
-------------------

:class:`~mink.CollisionAvoidanceLimit` prevents specified geometries from
colliding. Pairs of geom groups that should remain separated are specified by
name. The names must match collision geoms defined in the model; replace those
below with the geoms relevant to your robot:

.. code:: python

   from mink import CollisionAvoidanceLimit

   collision_limit = CollisionAvoidanceLimit(
       model,
       geom_pairs=[(["hand_capsule"], ["forearm", "upper_arm"])],
       minimum_distance_from_collisions=0.05,
   )

   limits = [ConfigurationLimit(model), collision_limit]

.. warning::

   Collision avoidance constraints tighten as distances shrink. When geoms are
   close to the minimum distance, this limit can dominate the solution and
   prevent other motion. Start with conservative (larger) minimum distances
   and reduce them as needed.

Recall the self-collision visible in the first video on this page. Without
collision avoidance, nullspace drift and singularity-induced motion can drive
geometries into contact. With it, the solver maintains the minimum separation
between specified geometry pairs:

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/collision_avoidance.mp4" type="video/mp4">
   </video>

*Collision avoidance prevents self-contact between the hand and upper arm.*

Freezing Degrees of Freedom
===========================

Sometimes selected degrees of freedom must remain fixed while the rest of the
robot moves. mink supports this through :class:`~mink.DofFreezingTask`, which
can be passed through the ``constraints`` argument of :func:`~mink.solve_ik`:

.. code:: python

   from mink import DofFreezingTask

   # Freeze the first two degrees of freedom.
   freeze_task = DofFreezingTask(model, dof_indices=[0, 1])

   for _ in range(steps):
       vel = solve_ik(
           configuration,
           [task, posture_task],
           dt,
           "daqp",
           constraints=[freeze_task],
       )
       configuration.integrate_inplace(vel, dt)

When passed through ``constraints``, DOF freezing is a hard equality
constraint: the selected degrees of freedom cannot move. This differs from
passing the same object through ``tasks``, where zero displacement is a soft
objective that can be traded against other tasks.

``dof_indices`` are tangent-space indices, not configuration-space indices.
A ball joint therefore contributes three indices and a free joint six. The
``constraints`` list can change between calls, so degrees of freedom can be
frozen and released at runtime. See
`arm_panda_dof_freezing.py <https://github.com/kevinzakka/mink/blob/main/examples/arm_panda_dof_freezing.py>`__
for an example that toggles the constraint while tracking a trajectory.

.. warning::

   Equality constraints and limits are both hard constraints. If they conflict,
   the QP is infeasible. For example, freezing a joint that is already outside
   its configuration limits prevents the solver from moving it back into the
   feasible region. In this case, :func:`~mink.solve_ik` raises
   :class:`~mink.NoSolutionFound` rather than returning a degraded solution.

.. raw:: html

   <video width="400" controls>
     <source src="https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/dof_freezing.mp4" type="video/mp4">
   </video>

*Red target: the first three degrees of freedom are frozen and the arm cannot
follow. Green target: the constraints are removed and tracking resumes.*

.. note::

   Freezing with an equality constraint gives the same solution for the
   unfrozen degrees of freedom as a reduced QP with the frozen coordinates
   deleted. See the :ref:`derivation <dof-freezing-derivation>` for the
   algebra and for why soft penalties or zeroed Jacobian columns are not
   equivalent to exact freezing.

Practical Guidelines
====================

When to Use What
----------------

**Always include:**

- :class:`~mink.ConfigurationLimit`, to enforce joint bounds.
- Some form of regularization (:class:`~mink.DampingTask` or
  :class:`~mink.PostureTask`), to prevent singularity and nullspace issues.

**Add when needed:**

- :class:`~mink.VelocityLimit`, for smooth, rate-limited motion.
- :class:`~mink.CollisionAvoidanceLimit`, when self-collision or environment
  collision is possible.
- :class:`~mink.DofFreezingTask` (via ``constraints``), when selected joints
  must not move.

Complete Example
================

The following example combines pose tracking with regularization and limits.

.. literalinclude:: ../../examples/docs/tasks_and_limits.py
   :language: python

Next Steps
==========

For more task types, limits, and real-world examples:

- :doc:`example_gallery`: runnable examples organized by use case.
- :doc:`/api/tasks`: all available task types.
- :doc:`/api/limits`: all available limit types.
