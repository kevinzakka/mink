:github_url: https://github.com/kevinzakka/mink/tree/main/docs/background/derivations.rst

.. _derivations:

***********
Derivations
***********

At each control step, mink solves for a tangent-space displacement
:math:`\Delta q`. Tasks describe what should improve; limits and exact
constraints describe which displacements are admissible:

.. math::

    \begin{aligned}
    \min_{\Delta q} \quad
    & \tfrac12 \Delta q^{\top}
      \color{#0072B2}{H}\Delta q
      + \color{#0072B2}{c}^{\top}\Delta q \\
    \text{subject to} \quad
    & \color{#009E73}{G}\Delta q \leq \color{#009E73}{h}, \\
    & \color{#D55E00}{A}\Delta q = \color{#D55E00}{b}.
    \end{aligned}

The solver returns the velocity :math:`v = \Delta q / dt`. This page develops
the core derivations behind the three parts of the quadratic program: tasks
contribute :math:`\color{#0072B2}{(H,c)}`, limits contribute
:math:`\color{#009E73}{(G,h)}`, and exact task constraints contribute
:math:`\color{#D55E00}{(A,b)}`.

.. list-table:: Notation used throughout
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`q`, :math:`\Delta q`
     - Configuration and tangent-space displacement
   * - :math:`dt`, :math:`v = \Delta q / dt`
     - Integration timestep and velocity
   * - :math:`n_v`
     - Tangent-space dimension
   * - :math:`e(q)`, :math:`J(q)`
     - Task error and its Jacobian
   * - :math:`\alpha \in [0,1]`
     - Task gain
   * - :math:`C = \operatorname{diag}(\kappa_i)`
     - Diagonal matrix of user-supplied task ``cost`` values


-----
Tasks
-----

From task error to quadratic cost
=================================

Every task begins with an error :math:`e(q)` that vanishes at its target. Its
Jacobian is defined by the first-order expansion

.. math::

    e(q \oplus \Delta q)
    = e(q) + J(q)\,\Delta q + o(\lVert \Delta q \rVert).

Asking one step to remove a fraction :math:`\alpha` of the error gives the
first-order task dynamics

.. math::

    J(q)\,\Delta q = -\alpha\,e(q).

Several tasks may request incompatible displacements. mink reconciles them by
minimizing a weighted residual for each task:

.. math::

    \tfrac12 \left\lVert
    C\bigl(J\,\Delta q + \alpha e\bigr)
    \right\rVert_2^2.

After dropping the term that is constant in :math:`\Delta q`, the undamped
Gauss--Newton contribution is

.. math::

    H_{\mathrm{GN}} = J^{\top} C^2 J,
    \qquad
    c = \alpha J^{\top} C^2 e

The public ``lm_damping`` value :math:`\lambda_{\mathrm{LM}}` adds a
state-dependent isotropic term,

.. math::

    \mu
    = \lambda_{\mathrm{LM}}\lVert \alpha C e\rVert_2^2,
    \qquad
    H_{\mathrm{task}} = H_{\mathrm{GN}} + \mu I.

After summing the task contributions, the solver adds its global ``damping``
value :math:`d` once more, so the final Hessian contains
:math:`(d + \sum_i \mu_i)I`. Neither damping term changes :math:`c`.

The public ``cost`` values :math:`\kappa_i` scale the residual before it is
squared; doubling one of them therefore quadruples that coordinate's
quadratic penalty. Zero costs simply remove the corresponding coordinates
from the task residual, although damping may still make the Hessian positive
on those coordinates.


Pose tasks
==========

Conventions
-----------

For a Lie-group element :math:`X`, mink uses the right-plus and right-minus
operators

.. math::

    X \oplus \xi = X\exp(\xi),
    \qquad
    X \ominus Y = \log(Y^{-1}X).

Throughout this section, :math:`0` denotes the world frame, :math:`f` the
controlled frame, and :math:`r` the root frame of a relative task. The
transform :math:`T_{ab}` is the pose of frame :math:`b` expressed in frame
:math:`a`, so :math:`T_{ab}=T_{ac}T_{cb}`. A star marks a target.

Pose error
----------

A pose task drives :math:`T(q) \in SE(3)` toward :math:`T^*`. Its residual is
the transform that carries the current pose to the target,

.. math::

    E(q) = T(q)^{-1}T^*,

so :math:`TE=T^*` and :math:`E=I` exactly at the target. The task error is

.. math::

    e(q) = \log E(q) = T^* \ominus T(q).

This is a body-frame twist: its coordinates are expressed along the current
axes of the controlled frame. Position and orientation costs consequently
travel with the robot rather than remaining aligned with the world. This is
the right-minus convention used in micro Lie theory [MicroLie]_. The spatial,
or left-minus, alternative is discussed in [FrameTaskJacobian]_.

Log-map Jacobian
----------------

Let :math:`J_T` be the body Jacobian of the controlled pose, defined by

.. math::

    T(q \oplus \Delta q)
    \approx T(q)\exp(J_T\Delta q).

With :math:`\xi=J_T\Delta q`, the perturbed residual is
:math:`\exp(-\xi)E`. Using :math:`\log(X^{-1})=-\log(X)` gives

.. math::

    e(q \oplus \Delta q)
    = -\log\bigl(E^{-1}\exp(J_T\Delta q)\bigr).

Define :math:`\operatorname{Jlog}_6` by the differential

.. math::

    \log\bigl(X\exp(\delta)\bigr)
    = \log X + \operatorname{Jlog}_6(X)\,\delta
    + o(\lVert\delta\rVert).

The chain rule now yields [FrameTaskJacobian]_

.. math::

    e(q \oplus \Delta q)
    \approx e(q)
    - \operatorname{Jlog}_6(E^{-1})J_T\Delta q,

and therefore

.. math::

    \boxed{J(q) = -\operatorname{Jlog}_6(E^{-1})J_T.}

At the target, :math:`E=I` and :math:`\operatorname{Jlog}_6(I)=I`, so
:math:`J=-J_T`: moving the current pose forward reduces the
target-minus-current error. Away from the target, the
:math:`\operatorname{Jlog}_6` factor must be retained when several weighted
tasks compete [FrameTaskJacobian]_.

.. admonition:: A revealing edge case

   A zero orientation error does **not** make the log-map Jacobian the
   identity. If :math:`E^{-1}` is a pure translation with tangent
   :math:`[v,0]`, then

   .. math::

      \operatorname{Jlog}_6(E^{-1})
      =
      \begin{bmatrix}
      I & \tfrac{1}{2}[v]_{\times} \\
      0 & I
      \end{bmatrix}.

   The off-diagonal block records how an angular perturbation changes the
   translational coordinates of the log error. It vanishes only when the
   translation error vanishes too. This is why replacing
   :math:`\operatorname{Jlog}_6` with the identity for a small rotation loses
   a real part of the task derivative.

The frame and relative-frame tasks differ only in their controlled pose
:math:`T(q)` and body Jacobian :math:`J_T`.


Frame task
==========

A frame task controls the pose of a robot frame :math:`f` in the world. Its
controlled pose, residual, and body Jacobian are

.. math::

    T(q) = T_{0f},
    \qquad
    E_f = T_{0f}^{-1}T_{0f}^*,
    \qquad
    J_T = {}_f J_{0f}.

Substitution into the generic pose formula gives

.. math::

    e(q) = \log E_f,
    \qquad
    J(q) = -\operatorname{Jlog}_6(E_f^{-1})\,{}_f J_{0f}.


Relative frame task
===================

A relative frame task controls frame :math:`f` with respect to another moving
frame :math:`r`. Its controlled pose and residual are

.. math::

    T(q) = T_{rf} = T_{0r}^{-1}T_{0f},
    \qquad
    E_{rf} = T_{rf}^{-1}T_{rf}^*.

The root is where current and target poses are compared, not where the error
lives: :math:`e=\log E_{rf}` is still expressed in the controlled frame
:math:`f`, so per-axis costs keep their meaning.

Relative pose Jacobian
----------------------

Let :math:`T_{fr}=T_{rf}^{-1}`. Perturbing the frame and root with the body
twists

.. math::

    \xi_f = {}_f J_{0f}\,\Delta q,
    \qquad
    \xi_r = {}_r J_{0r}\,\Delta q

gives

.. math::

    \begin{aligned}
    T_{rf}'
    &= \exp(-\xi_r)T_{rf}\exp(\xi_f) \\
    &\approx T_{rf}\exp\!\left(
       \xi_f - \operatorname{Ad}(T_{fr})\xi_r
       \right),
    \end{aligned}

where the second line discards terms of second order and higher. It follows
that the relative pose has body Jacobian

.. math::

    J_T = {}_f J_{rf}
    = {}_f J_{0f}
    - \operatorname{Ad}(T_{fr})\,{}_r J_{0r}.

The relative-frame task Jacobian is therefore

.. math::

    \boxed{
    J(q) = -\operatorname{Jlog}_6(E_{rf}^{-1})\,{}_f J_{rf}.
    }

If :math:`r` is the world frame, then :math:`{}_r J_{0r}=0` and
:math:`T_{rf}=T_{0f}`. Both the error and Jacobian reduce exactly to those of
the frame task.


From MuJoCo Jacobians to task Jacobians
=======================================

The pose derivations require body Jacobians. MuJoCo's ``mj_jac*`` functions
instead return a Jacobian centered at the frame origin but with axes aligned
with the world. Because this auxiliary frame and the body frame share an
origin, changing coordinates introduces no lever-arm term: it only rotates
the linear and angular blocks.

Define

.. math::

    \mathcal{R}(R)
    = \begin{bmatrix} R^{\top} & 0 \\ 0 & R^{\top} \end{bmatrix}.

The frame body Jacobian is then

.. math::

    {}_f J_{0f}
    = \mathcal{R}(R_{0f})J_f^{\mathrm{MuJoCo}}.

On the native path, a frame task first forms the small map

.. math::

    M = -\operatorname{Jlog}_6(E_f^{-1})\mathcal{R}(R_{0f})
    \in \mathbb{R}^{6\times6},

then evaluates :math:`J(q)=M J_f^{\mathrm{MuJoCo}}`. This avoids a separate
:math:`6\times n_v` multiplication just to construct the body Jacobian. The
fallback path evaluates the same formula after explicitly constructing the
body Jacobian :math:`{}_f J_{0f}`.

The same factorization applies to a relative-frame task. With

.. math::

    L = \operatorname{Jlog}_6(E_{rf}^{-1}),
    \qquad
    M_f = -L\mathcal{R}(R_{0f}),
    \qquad
    M_r = L\operatorname{Ad}(T_{fr})\mathcal{R}(R_{0r}),

the task Jacobian is

.. math::

    J(q)
    = M_f J_f^{\mathrm{MuJoCo}}
    + M_r J_r^{\mathrm{MuJoCo}}.

On the native path, the log-map differential, adjoint, and coordinate
rotations are therefore composed as :math:`6\times6` maps before they touch
either wide MuJoCo Jacobian. The fallback follows the same derivation but
materializes the two body Jacobians first.


Posture task
============

A posture task drives the controlled configuration coordinates toward a
preferred posture :math:`q^*`. Let :math:`P` be the diagonal selector that is
one on controlled coordinates and zero on ignored coordinates. In particular,
the implementation ignores floating-base coordinates. The task is

.. math::

    e(q) = P\bigl(q \ominus q^*\bigr),
    \qquad
    J(q) = P.

For a fixed-base configuration, :math:`P=I_{n_v}`. If every controlled
coordinate has a nonzero cost and the posture task acts alone, its optimality
condition gives

.. math::

    \boxed{\Delta q = -\alpha e(q)}

on that controlled subspace. Unlike the pose tasks, the posture task uses a
current-minus-target error and consequently has a positive Jacobian. Both sign
conventions produce the same corrective law :math:`J\Delta q=-\alpha e`.

The corresponding desired velocity is

.. math::

    v_{\mathrm{des}}
    = -\frac{\alpha}{dt}e(q).

For scalar hinge and slide coordinates, this is the familiar proportional
controller :math:`v_{\mathrm{des}}=(\alpha/dt)(q^*-q)`. The manifold form
above also remains meaningful for rotational coordinates, where ordinary
subtraction does not.


Damping task
============

The damping task penalizes instantaneous motion rather than tracking a
posture. Its error and Jacobian are

.. math::

    e(q) = 0,
    \qquad
    J(q) = P.

The gain has no effect because the error is zero. Its quadratic cost is

.. math::

    \boxed{
    \tfrac12\left\lVert CP\Delta q\right\rVert_2^2
    = \tfrac12\sum_i \kappa_i^2(P\Delta q)_i^2.
    }

With no competing objective, every damped coordinate remains at rest. With
other tasks present, damping trades task residual against weighted joint
motion; among otherwise equivalent minimizers, it selects the one with the
smallest damped velocity norm.

Relation to Tikhonov regularization
-----------------------------------

If the other tasks contribute a Hessian :math:`H_0`, damping gives

.. math::

    H = H_0 + P^{\top}C^2P.

This is Tikhonov regularization on the selected coordinates. The combined
Hessian is positive definite exactly when the added term is positive on every
null direction of :math:`H_0`. For a fixed-base system, positive costs on all
coordinates are sufficient; damping does not provide that guarantee on an
ignored floating base.


------
Limits
------

Configuration limits
====================

Hinge and slide joints
----------------------

Consider first the scalar bounds of limited hinge and slide joints. At the
current configuration, let

.. math::

    d_{\min} = q \ominus q_{\min},
    \qquad
    d_{\max} = q_{\max} \ominus q

be the remaining distances to the lower and upper limits. A configuration
limit with gain :math:`\gamma\in(0,1]` permits at most that fraction of either
distance in one step:

.. math::

    -\gamma d_{\min}
    \leq \Delta q
    \leq \gamma d_{\max}.

Let :math:`P` select the limited coordinates from the full tangent space.
Stacking the upper and lower inequalities gives

.. math::

    G = \begin{bmatrix} P \\ -P \end{bmatrix},
    \qquad
    h = \gamma
    \begin{bmatrix} d_{\max} \\ d_{\min} \end{bmatrix}.

The default :math:`\gamma=0.95` leaves a small margin to reduce overshoot from
the first-order approximation. Free joints and joints without limits do not
contribute rows.

Ball joints
-----------

A limited ball joint bounds its total rotation angle rather than three
independent coordinates. Let :math:`\phi\in\mathbb{R}^3` be its rotation
vector, :math:`\theta=\lVert\phi\rVert`, and
:math:`a=\phi/\theta` its current rotation axis. Linearizing the angle gives
the single inequality

.. math::

    a^{\top}\Delta q_{\mathrm{ball}}
    \leq \gamma(\theta_{\max}-\theta).

At the identity the axis is undefined, so this row is inactive. Motion
orthogonal to :math:`a` changes the angle only at second order; a velocity
limit should therefore accompany this linearization to prevent a large step
from overshooting the bound.


Velocity limits
===============

Velocity bounds become displacement bounds after multiplication by the
timestep:

.. math::

    -v_{\max}
    \leq \frac{P\Delta q}{dt}
    \leq v_{\max}
    \quad\Longleftrightarrow\quad
    -dt\,v_{\max}
    \leq P\Delta q
    \leq dt\,v_{\max},

where :math:`P` selects the velocity-limited coordinates. Thus

.. math::

    G = \begin{bmatrix} P \\ -P \end{bmatrix},
    \qquad
    h = dt
    \begin{bmatrix} v_{\max} \\ v_{\max} \end{bmatrix}.


-----------------
Exact constraints
-----------------

A task passed through the solver's ``constraints`` argument is imposed as
first-order equality rows rather than as a quadratic penalty:

.. math::

    A\Delta q=b,
    \qquad
    A = \begin{bmatrix} J_1 \\ \vdots \\ J_m \end{bmatrix},
    \qquad
    b = \begin{bmatrix} -\alpha_1e_1 \\ \vdots \\ -\alpha_me_m \end{bmatrix}.

The QP backend enforces these rows exactly.

.. note::

   **Naming.** This mechanism is independent of
   :class:`~mink.EqualityConstraintTask`.
   Despite its name, that class wraps MuJoCo model-equality residuals as a
   regular mink task and is ordinarily passed through ``tasks``, contributing
   a soft cost :math:`\color{#0072B2}{(H,c)}`. Passing any task through
   ``constraints`` instead turns its first-order dynamics into hard rows
   :math:`\color{#D55E00}{(A,b)}`.

.. _dof-freezing-derivation:

DOF freezing
============

Freezing a set of degrees of freedom imposes the equality constraint

.. math::

    \Delta q_f = 0,

where :math:`\Delta q_f` contains the frozen tangent-space coordinates. For
hinge and slide joints this fixes the corresponding displacement exactly. For
ball and free joints, individual tangent-space components are frozen to first
order over the current step.

Let :math:`\Delta q_r` denote the remaining coordinates. Partitioning the QP
accordingly gives

.. math::

    \begin{aligned}
    \min_{\Delta q_r,\,\Delta q_f} \quad
    & \tfrac12 \Delta q_r^{\top}H_{rr}\Delta q_r
    + \Delta q_r^{\top}H_{rf}\Delta q_f
    + \tfrac12 \Delta q_f^{\top}H_{ff}\Delta q_f \\
    & + c_r^{\top}\Delta q_r + c_f^{\top}\Delta q_f \\
    \text{subject to} \quad
    & G_r\Delta q_r + G_f\Delta q_f \leq h, \\
    & \Delta q_f = 0.
    \end{aligned}

Substitution removes every term involving :math:`\Delta q_f`:

.. math::

    \begin{aligned}
    \min_{\Delta q_r} \quad
    & \tfrac12 \Delta q_r^{\top}H_{rr}\Delta q_r
    + c_r^{\top}\Delta q_r \\
    \text{subject to} \quad
    & G_r\Delta q_r \leq h.
    \end{aligned}

This reduced problem is algebraically equivalent to deleting the frozen rows
and columns of :math:`H`, entries of :math:`c`, and columns of :math:`G`.
mink retains the original decision-vector dimension and supplies selector rows
in :math:`A`; the elimination argument explains why the optimizer over the
remaining coordinates is unchanged.

One edge case remains. An inequality involving only frozen coordinates reduces
to :math:`0\leq h_j`. If :math:`h_j<0`, the problem is already infeasible and
no motion of the remaining coordinates can restore feasibility.

Soft freezing is different
--------------------------

A large penalty on the frozen coordinates is not equivalent to imposing
:math:`\Delta q_f=0`. Adding

.. math::

    \tfrac12\Delta q_f^{\top}S\Delta q_f,
    \qquad S\succeq0,

makes motion expensive but does not forbid it. Ignoring inequalities and
assuming :math:`H_{ff}+S` is invertible, the best value of
:math:`\Delta q_f` for a fixed :math:`\Delta q_r` is

.. math::

    \Delta q_f^*
    = -(H_{ff}+S)^{-1}
      \left(H_{rf}^{\top}\Delta q_r+c_f\right).

Substitution gives the reduced Hessian

.. math::

    H_{rr}
    - H_{rf}(H_{ff}+S)^{-1}H_{rf}^{\top}.

A finite penalty can therefore change both the nominally frozen coordinates
and the optimization over the remaining coordinates. In general, exact
freezing is recovered only as the penalty tends to infinity.

Zeroing columns of a task Jacobian is not equivalent either. It removes those
coordinates from that task's cost rather than constraining them; other tasks
or active constraints may still move them.
