:github_url: https://github.com/kevinzakka/mink/tree/main/doc/lie.rst

***
Lie
***

MuJoCo does not currently offer a native Lie group interface for rigid body transforms, though it
does have a collection of functions for manipulating quaternions and rotation matrices. The goal
of this library is to provide this unified interface. Whenever possible, the underlying
lie operation leverages the corresponding MuJoCo function. For example,
:py:meth:`~SO3.from_matrix` uses `mujoco.mju_mat2Quat` under the hood.

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
