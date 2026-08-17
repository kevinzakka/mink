:github_url: https://github.com/kevinzakka/mink/tree/main/docs/background/derivations.rst

.. _derivations:

***********
Derivations
***********

.. list-table::
   :header-rows: 1

   * - Variable Name
     - Symbol
   * - Configuration
     - :math:`q`
   * - Configuration displacement
     - :math:`\Delta q`
   * - Integration timestep
     - :math:`dt`
   * - Velocity in tangent space
     - :math:`v = \frac{\Delta q}{dt}`
   * - Configuration limits
     - :math:`q_{\text{min}}, q_{\text{max}}`
   * - Maximum joint velocity magnitude
     - :math:`v_{\text{max}}`
   * - Identity matrix
     - :math:`I`

------
Limits
------

Configuration limit
===================

Applying a first-order Taylor expansion on the configuration yields:

.. math::
    \begin{aligned}
    q_{\text{min}} &\leq q \oplus v \cdot dt \leq q_{\text{max}} \\
    q_{\text{min}} &\leq q \oplus \Delta q \leq q_{\text{max}} \\
    q_{\text{min}} &\ominus q \leq \Delta q \leq q_{\text{max}} \ominus q
    \end{aligned}

Rewriting as :math:`G \Delta q \leq h` and separating the inequalities:

.. math::
    \begin{aligned}
    &+I \cdot \Delta q \leq q_{\text{max}} \ominus q \\
    &-I \cdot \Delta q \leq q \ominus q_{\text{min}}
    \end{aligned}

Stacking these inequalities gives:

.. math::
    \begin{aligned}
    G &= \begin{bmatrix} +I \\ -I \end{bmatrix}, \\
    h &= \begin{bmatrix} q_{\text{max}} \ominus q \\ q \ominus q_{\text{min}} \end{bmatrix}
    \end{aligned}

Velocity limit
==============

Given the maximum joint velocity magnitudes :math:`v_{\text{max}}`, the joint velocity limits can be expressed as:

.. math::
    \begin{aligned}
    -v_{\text{max}} &\leq v \leq v_{\text{max}} \\
    -v_{\text{max}} &\leq \frac{\Delta q}{dt} \leq v_{\text{max}} \\
    -v_{\text{max}} \cdot dt &\leq \Delta q \leq v_{\text{max}} \cdot dt
    \end{aligned}

Rewriting as :math:`G \Delta q \leq h` and separating the inequalities:

.. math::
    \begin{aligned}
    &+I \cdot \Delta q \leq v_{\text{max}} \cdot dt \\
    &-I \cdot \Delta q \leq v_{\text{max}} \cdot dt
    \end{aligned}

Stacking these inequalities gives:

.. math::
    \begin{aligned}
    G \Delta q &\leq h \\
    \begin{bmatrix} +I \\ -I \end{bmatrix} \Delta q &\leq \begin{bmatrix} v_{\text{max}} \cdot dt \\ v_{\text{max}} \cdot dt \end{bmatrix}
    \end{aligned}


-----
Tasks
-----

Posture task
============

A posture task penalizes the deviation of the current configuration from a
preferred posture.  Its error and Jacobian are

.. math::

   e(q) = q^{\star} \ominus q, \qquad
   J(q) = I_{n_v}

First-order task dynamics
-------------------------

Using the generic relation :math:`J(q)\,\Delta q = -\alpha\,e(q)` gives

.. math::

   I_{n_v}\,\Delta q = -\alpha\,e(q)
   \;\;\Longrightarrow\;\;
   \boxed{\; \Delta q = -\alpha\,e(q) \;}

Quadratic-program formulation
-----------------------------

The task enters the QP as the weighted least-squares term

.. math::

   \min_{\Delta q}\; \tfrac12 \left\lVert
      J\,\Delta q + \alpha\,e(q) \right\rVert_{W}^{2},

with a diagonal, positive weight matrix :math:`W = \mathrm{diag}(\lambda_i)`.
Substituting :math:`J = I_{n_v}` and equating the gradient to zero yields

.. math::

   W\bigl(\Delta q + \alpha\,e(q)\bigr) = 0
   \;\;\Longrightarrow\;\;
   \boxed{\; \Delta q = -\alpha\,e(q) \;},

identical to the first-order dynamics and independent of the weights because
:math:`W` is full-rank.

Velocity interpretation
-----------------------

Interpreting the displacement over the solver timestep :math:`dt`
as a velocity command gives

.. math::

   \dot q_{\text{des}} = \frac{\Delta q}{dt}
   = \frac{\alpha}{dt}\bigl(q^{\star} - q\bigr)
   = k_p\bigl(q^{\star} - q\bigr),

where :math:`k_p = \alpha / dt`.  When the implementation uses
:math:`dt = 1`, the proportional gain reduces to :math:`k_p = \alpha`.
Hence the posture task behaves as a joint-space proportional controller that
is always full-rank and therefore provides robust regularization whenever
primary tasks are ill-conditioned.


Damping task
============

The damping task penalizes joint motion itself, i.e.\ the displacement
``Δq``.  It does not target a reference posture: its desired error is
identically zero.

Error and Jacobian
------------------

.. math::

   e(q) = 0, \qquad
   J(q) = I_{n_v}

Quadratic cost
--------------

Inserted in the generic task objective

.. math::

   \min_{\Delta q}\; \tfrac12
      \bigl\lVert J\,\Delta q + \alpha\,e(q) \bigr\rVert_{W}^{2}

with :math:`e(q)=0`, :math:`J = I_{n_v}`, and default gain
:math:`\alpha = 1`, the cost reduces to

.. math::

   \boxed{\;
   \tfrac12\,\Delta q^{\top} W \,\Delta q \;}
   \;=\;
   \tfrac12 \sum_{i=1}^{n_v} \lambda_i\,\Delta q_i^{2},

where :math:`W = \operatorname{diag}(\lambda_i)` comes from the
user-supplied ``cost`` vector.

Optimality conditions
---------------------

Taking the gradient and setting it to zero gives

.. math::

   W \,\Delta q = 0
   \;\;\Longrightarrow\;\;
   \boxed{\; \Delta q = 0 \;}.

Thus, in the absence of other tasks, the damping task enforces :math:`\Delta q = 0`.
When other tasks are present, the QP minimizes joint motion inside their
solution set, yielding the minimum-norm velocity.

Connection to Levenberg-Marquardt / Tikhonov
--------------------------------------------

If a primary task with Hessian :math:`J_{h}^{\top}J_{h}` becomes
rank-deficient, adding the damping task makes the combined Hessian

.. math::

   H = J_{h}^{\top}J_{h} \;+\; W

strictly positive-definite (provided any :math:`\lambda_i>0`).  This is
exactly Tikhonov regularization, preventing large joint speeds near
singularities and selecting a unique solution.

Velocity interpretation
-----------------------

With the library’s convention ``v = Δq / dt`` and
:math:`\Delta q = 0`, the commanded velocity is

.. math::

   \dot q_{\text{des}} = 0.

The damping task therefore commands zero velocity; it acts as a
regularizer that suppresses unnecessary motion when higher-priority tasks
leave residual degrees of freedom.


--------------------
Equality constraints
--------------------

.. _dof-freezing-derivation:

DOF freezing
============

Freezing a set of degrees of freedom imposes the equality constraint

.. math::

    \Delta q_f = 0,

where :math:`\Delta q_f` contains the frozen tangent-space coordinates. For
hinge and slide joints, this fixes the corresponding joint displacement
exactly. For ball and free joints, individual tangent-space components are
frozen only to first order.

Let :math:`\Delta q_r` denote the remaining coordinates. Partitioning the QP
accordingly gives

.. math::
    \begin{aligned}
    \min_{\Delta q_r,\, \Delta q_f} \quad
    & \tfrac12 \Delta q_r^{\top} H_{rr}\Delta q_r
    + \Delta q_r^{\top} H_{rf}\Delta q_f
    + \tfrac12 \Delta q_f^{\top} H_{ff}\Delta q_f
    + c_r^{\top}\Delta q_r
    + c_f^{\top}\Delta q_f \\
    \text{s.t.} \quad
    & G_r\Delta q_r + G_f\Delta q_f \leq h, \\
    & \Delta q_f = 0.
    \end{aligned}

Substituting the equality constraint removes all terms involving
:math:`\Delta q_f`:

.. math::
    \begin{aligned}
    \min_{\Delta q_r} \quad
    & \tfrac12 \Delta q_r^{\top} H_{rr}\Delta q_r
    + c_r^{\top}\Delta q_r \\
    \text{s.t.} \quad
    & G_r\Delta q_r \leq h.
    \end{aligned}

Thus, exact DOF freezing is equivalent to eliminating the frozen decision
variables from the QP: delete the corresponding rows and columns of
:math:`H`, entries of :math:`c`, and columns of :math:`G`. The
equality-constrained formulation produces the same minimizer over the
remaining coordinates while preserving the original QP dimensions.

An inequality involving only frozen coordinates reduces to
:math:`0 \leq h_j`. If :math:`h_j < 0`, the constraint is already violated
and no choice of the remaining coordinates can restore feasibility, since the
coordinates on which the constraint depends are fixed.

Soft freezing
-------------

A large penalty on the frozen coordinates is not equivalent to imposing
:math:`\Delta q_f = 0`. For example, adding

.. math::

    \tfrac12 \Delta q_f^\top C \Delta q_f

to the objective makes motion in those coordinates expensive, but does not
forbid it. The optimizer may still choose :math:`\Delta q_f \neq 0` if doing
so sufficiently reduces another part of the objective.

This matters whenever the frozen and remaining coordinates are coupled through
:math:`H_{rf}`. Ignoring inequality constraints, the optimal value of
:math:`\Delta q_f` for a fixed :math:`\Delta q_r` is

.. math::

    \Delta q_f^\star
    =
    -(H_{ff} + C)^{-1}
    \left(H_{rf}^{\top}\Delta q_r + c_f\right).

Substituting this back into the objective gives the reduced Hessian

.. math::

    H_{rr}
    - H_{rf}(H_{ff} + C)^{-1}H_{rf}^{\top}.

Thus, a finite soft penalty changes the optimization over the remaining
coordinates while still allowing the nominally frozen coordinates to move.
Only in the limit of an infinite penalty does this recover exact freezing.

Zeroing columns of a task Jacobian is not equivalent either. It removes those
coordinates from that task's cost rather than constraining them. The
coordinates remain decision variables and may still move because of other
tasks or active constraints.
