:github_url: https://github.com/kevinzakka/mink/tree/main/docs/tutorial/lie_groups.rst

.. _lie-groups:

====================
Rotations and Poses
====================

mink represents rotations and rigid transforms using Lie groups:

- :class:`~mink.SO3` for 3D rotations
- :class:`~mink.SE3` for 3D rigid body transforms (rotation + translation)

These classes are used throughout mink to represent frame poses, specify targets,
and compute pose errors consistently.

This tutorial covers construction, composition, and error computation for SO(3) and SE(3).

Conventions
===========

Quaternions
-----------

mink uses MuJoCo's quaternion convention: **scalar-first** quaternions
``(qw, qx, qy, qz)`` (also written ``wxyz``).

Composition and action via ``@``
--------------------------------

mink overloads the ``@`` operator:

- ``T1 @ T2`` composes two transformations.
- ``T @ p`` applies a transformation to a point ``p``.
- Similarly for rotations: ``R1 @ R2`` and ``R @ v``.

This mirrors common robotics notation.

Composition follows standard robotics conventions: ``A @ B`` means "apply B,
then A."

SO3: Rotations
==============

Creating rotations
------------------

Use the exponential map for axis-angle rotations:

.. code-block:: python

   import numpy as np
   import mink

   # Rotation from a tangent vector (axis-angle, radians).
   R = mink.SO3.exp(np.array([0.0, 0.0, np.pi / 2]))

Convenience constructors exist for common axes and roll/pitch/yaw:

.. code-block:: python

   Rz = mink.SO3.from_z_radians(np.pi / 2)
   Rrpy = mink.SO3.from_rpy_radians(roll=0.1, pitch=0.2, yaw=-0.3)

Identity and normalization:

.. code-block:: python

   R = mink.SO3.identity()
   R = R.normalize()

Applying and composing
----------------------

Apply a rotation to a vector:

.. code-block:: python

   v = np.array([1.0, 0.0, 0.0])
   v_rot = Rz @ v

Compose rotations (right-multiplication):

.. code-block:: python

   R = mink.SO3.from_z_radians(0.3) @ mink.SO3.from_x_radians(-0.2)

Log/Exp (tangent space)
-----------------------

``log()`` maps a rotation to a 3D tangent vector (axis-angle),
and ``exp()`` maps a tangent vector back to a rotation:

.. code-block:: python

   omega = R.log()          # shape (3,)
   R_recovered = mink.SO3.exp(omega)

These are useful for expressing rotation errors as tangent vectors and for interpolation.

SE3: Rigid Body Transformations
===============================

Creating transforms
-------------------

An :class:`~mink.SE3` stores rotation + translation. Use:

- :meth:`~mink.SE3.from_rotation_and_translation`
- :meth:`~mink.SE3.from_translation`
- :meth:`~mink.SE3.from_rotation`

.. code-block:: python

   import numpy as np
   import mink

   R = mink.SO3.from_z_radians(np.pi / 4)
   t = np.array([0.2, 0.0, 0.1])

   T = mink.SE3.from_rotation_and_translation(R, t)


Example: building a pose target
-------------------------------

The following example shows how to construct a nearby pose target in the
world frame using SE(3) composition.

.. code-block:: python

   T_current = configuration.get_transform_frame_to_world("ee", "site")
   T_target = mink.SE3.from_translation(np.array([0.0, 0.0, 0.1])) @ T_current
   frame_task.set_target(T_target)

Composition: world vs local updates
-----------------------------------

An increment can be applied in two different frames depending on
multiplication order.

Let ``T`` be a pose. Consider a small translation transform ``Δ``:

.. code-block:: python

   Δ = mink.SE3.from_translation(np.array([0.0, 0.0, 0.1]))

- **Left-multiplying** applies the increment in the **world frame**:

  .. code-block:: python

     T_world = Δ @ T

- **Right-multiplying** applies the increment in the **local frame**:

  .. code-block:: python

     T_local = T @ Δ

Applying a transform
--------------------

.. code-block:: python

   p = np.array([0.0, 0.0, 0.0])
   p_world = T @ p

Access rotation and translation:

.. code-block:: python

   R = T.rotation()
   t = T.translation()

Pose errors (right-minus)
=========================

For :class:`~mink.SE3`, the tangent vector is ordered as
:math:`(v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)`,
i.e., translation first, then rotation.

mink represents pose errors using the **right-minus** operator:

.. math::

   T_1 \ominus T_2 = \log(T_2^{-1} T_1)

The result :math:`\xi` satisfies :math:`T_2 \cdot \exp(\xi) = T_1`: it is the
body-frame twist at :math:`T_2` that reaches :math:`T_1`. In code, this
corresponds to :meth:`~mink.MatrixLieGroup.rminus`:

.. code-block:: python

   # Twist from T_current to T_target, expressed in T_current's body frame.
   e = T_target.rminus(T_current)  # log(T_current^{-1} T_target), shape (6,)

This is exactly what :class:`~mink.FrameTask` computes internally:

.. math::

   e = T_{\text{target}} \ominus T_{\text{current}} = \log(T_{\text{current}}^{-1} \, T_{\text{target}})

The error is expressed in the **current body frame**: translations are along
the frame's local axes and rotations are about its local axes. This is
consistent with mink's body-frame Jacobians, which map joint velocities to
body-frame twists.

Internally, :class:`~mink.FrameTask` also uses :meth:`~mink.MatrixLieGroup.jlog`
to linearize the log-map error; calling it directly is rarely necessary.


Interpolation
=============

Geodesic interpolation on the manifold is available via:

.. code-block:: python

   T_mid = T0.interpolate(T1, alpha=0.5)  # alpha in [0, 1]

This computes :math:`T_0 \cdot \exp(\alpha \cdot \log(T_0^{-1} T_1))`, which
traces the shortest path between two poses on the group manifold. For SO(3),
this is equivalent to SLERP. This avoids the truncation artifacts that arise
from linearly interpolating rotation matrices or Euler angles.

Further Reading
===============

For Lie group fundamentals and Jacobians (left/right Jacobians, plus/minus),
see:

- `A micro Lie theory <https://arxiv.org/abs/1812.01537>`_
