:github_url: https://github.com/kevinzakka/mink/tree/main/docs/tutorial/mujoco_basics.rst

.. _mujoco-basics:

==============
MuJoCo Basics
==============

mink uses `MuJoCo <https://github.com/google-deepmind/mujoco>`_ as its physics
backend. This page covers the MuJoCo concepts relevant to inverse kinematics
and explains how mink relies on them.

Model and Data
==============

MuJoCo separates the *model description* from the *state*:

**MjModel** contains quantities that do not change over time: the robot's
kinematic tree, joint definitions, geometry, and physical properties. It is
created once from an XML file.

**MjData** contains the state and quantities derived from it. The state
consists of time (``data.time``), generalized positions (``data.qpos``), and
generalized velocities (``data.qvel``).

.. code:: python

   import mujoco

   model = mujoco.MjModel.from_xml_path("robot.xml")
   data = mujoco.MjData(model)

Calling ``mujoco.mj_forward(model, data)`` runs the full forward pass and
updates derived quantities in ``data``. mink's :meth:`~mink.Configuration.update`
runs only ``mj_kinematics`` and ``mj_comPos``, which suffice for IK. See MuJoCo's
`simulation pipeline <https://mujoco.readthedocs.io/en/stable/computation/index.html#simulation-pipeline>`_
for details.

For a comprehensive introduction, see the official
`tutorial <https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb>`_
and `documentation <https://mujoco.readthedocs.io/en/stable/overview.html>`_.

Frames
======

A **frame** is a coordinate system attached to an element in the model. MuJoCo
has many frame types; mink focuses on three for inverse kinematics:

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. image:: https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/body_frame.jpg
          :alt: Body frame
          :align: center
          :width: 250px
     - .. image:: https://raw.githubusercontent.com/kevinzakka/mink/assets/docs/site_frame.jpg
          :alt: Site frame
          :align: center
          :width: 250px

.. note::

   In this example, the geom and inertial body frames coincide.

**Bodies** (left)
   The kinematic links of the robot. Each body has *two* frames: a frame used
   to define it and position child elements, and an inertial frame centered at
   the body's center of mass and aligned with its principal axes of inertia.
   mink uses the inertial frame (``mjOBJ_BODY``).

**Geoms**
   Collision and visual geometry attached to bodies. Each geom has its own
   frame, which may be offset from the parent body.

**Sites** (right)
   Massless frames attached to bodies. Sites mark points of interest such as
   end-effectors, sensor locations, and grasp points, without adding collision
   geometry. IK tasks are typically defined on sites.

.. code:: xml

   <body name="tool">
     <site name="end_effector" pos="0 0 0.1"/>
   </body>

Joints and Generalized Coordinates
==================================

Joints connect bodies and define how they can move:

- **hinge**: Rotation about one axis (1 DOF)
- **slide**: Translation along one axis (1 DOF)
- **ball**: Spherical joint (3 DOF, but 4 values in qpos due to quaternion overparameterization)
- **free**: Unconstrained 6-DOF motion (6 DOF, but 7 values in qpos: 3 translation + 4 quaternion)

MuJoCo uses scalar-first quaternions (wxyz), where w is the real component.

Joint positions are stored in ``data.qpos`` (generalized coordinates). For
robots with only hinge joints, ``qpos`` is the vector of joint angles.

.. note::

   The dimension of ``qpos`` (``model.nq``) may differ from the number of
   degrees of freedom (``model.nv``) when quaternion-based joints are present.
   mink exposes both via ``configuration.nq`` and ``configuration.nv``.

Coordinate System
=================

MuJoCo uses a right-handed coordinate system. By default:

- **Z is up** (gravity points in -Z)
- **X is forward**
- **Y is left**

Velocity and Jacobian Conventions
=================================

This section covers the frame conventions for velocities and Jacobians. mink
handles these conversions internally, so this material is only needed when
interfacing with other libraries or inspecting internal computations.

MuJoCo's velocity convention
----------------------------

MuJoCo uses a mixed-frame representation for 6D velocities:

- **Linear velocity**: world frame
- **Angular velocity**: body frame

This follows from quaternion integration. Angular velocities live in the
tangent space of the quaternion, which is a local (body) frame.

MuJoCo's Jacobian convention
----------------------------

MuJoCo's ``mj_jacBody``, ``mj_jacSite``, and ``mj_jacGeom`` return Jacobians in
the **local-world-aligned** frame: the origin is at the frame, but axes are
aligned with the world. In Pinocchio terminology, this is ``LOCAL_WORLD_ALIGNED``.

.. code:: python

   # MuJoCo returns a local-world-aligned Jacobian
   jac = np.zeros((6, model.nv))
   mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

mink's convention
-----------------

mink uses **body velocities** (twists expressed in the local frame) and
**body Jacobians** throughout. This is a consistent choice: the Jacobian and
the velocity/error it maps to are in the same frame.

:meth:`~mink.Configuration.get_frame_jacobian` converts MuJoCo's
local-world-aligned Jacobian into a body Jacobian (``LOCAL`` in Pinocchio
terminology):

.. code:: python

   # mink returns a body Jacobian
   jac = configuration.get_frame_jacobian("end_effector", "site")

Similarly, task errors are computed as body twists via the right-minus operation
on SE(3). This ensures the relationship ``error = J @ dq`` is consistent, with
both sides in the body frame.

.. note::

   mink uses right Jacobians (body perturbations), consistent with body
   velocities. For background on left vs right Jacobians on Lie groups, see
   `A micro Lie theory <https://arxiv.org/abs/1812.01537>`_.

Summary: what mink provides
===========================

mink's :class:`~mink.Configuration` wraps MuJoCo's model and data, providing:

- **Poses** in the world frame via
  :meth:`~mink.Configuration.get_transform_frame_to_world`
- **Body Jacobians** in the local frame via
  :meth:`~mink.Configuration.get_frame_jacobian`
- **Inertia matrix** via :meth:`~mink.Configuration.get_inertia_matrix`

All quantities are expressed in the body frame, with conversions from MuJoCo's
conventions handled automatically.
