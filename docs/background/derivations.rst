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
