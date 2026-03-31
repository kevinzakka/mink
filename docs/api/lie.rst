:github_url: https://github.com/kevinzakka/mink/tree/main/docs/api/lie.rst

***
Lie
***

This module provides a Lie group interface for rigid-body transforms, delegating to
MuJoCo's quaternion and rotation-matrix routines where possible (e.g.,
:py:meth:`~SO3.from_matrix` calls ``mujoco.mju_mat2Quat``).

This library is heavily ported from `jaxlie <https://github.com/brentyi/jaxlie>`__,
swapping out JAX for Numpy and adding a few additional features.

MatrixLieGroup
==============

.. autoclass:: mink.lie.base.MatrixLieGroup
    :members:

SO3
===

.. autoclass:: mink.lie.so3.SO3
    :show-inheritance:
    :inherited-members:
    :members:

SE3
===

.. autoclass:: mink.lie.se3.SE3
    :show-inheritance:
    :inherited-members:
    :members:
