"""Elastic equality constraints."""

import numpy as np
import numpy.typing as npt

from .configuration import Configuration
from .exceptions import InvalidConstraint
from .tasks import Task


class Elastic:
    r"""Equality constraint that yields instead of making the QP infeasible.

    Passed through ``constraints`` like a plain task, it enforces
    :math:`J \Delta q = -\alpha e` through the exact :math:`\ell_1` penalty

    .. math::

        \rho^T \left| J \Delta q + \alpha e \right|,

    with :math:`|\cdot|` taken componentwise. Let :math:`\lambda` be the
    multiplier the task would have as a hard constraint. If
    :math:`|\lambda_i| < \rho_i` for every component, the solution is that of the
    hard constraint [SoftConstraintsMPC]_. A component that would need more force yields, pulling with
    constant force :math:`\rho_i`, and the QP stays feasible.

    As with hard constraints, the task's ``cost`` and ``lm_damping`` are ignored.
    The penalty is the only per-component scale. A zero penalty leaves that
    component unconstrained.

    By default :func:`~mink.solve_ik` solves the constraint as a hard one while
    it holds, and solves the penalty with slack variables only when it yields.
    Pass ``elastic_strategy="penalty"`` to always solve the penalty directly, in
    a single QP.

    Example:

    .. code-block:: python

        frame_task = FrameTask("attachment_site", "site", 1.0, 1.0)
        solve_ik(
            configuration,
            [posture_task],
            dt,
            "daqp",
            limits=limits,
            constraints=[Elastic(frame_task, penalty=1e3)],
        )

    Attributes:
        task: The constrained task.
        penalty: Per-component :math:`\ell_1` penalty :math:`\rho`, a scalar or
            a vector with the dimension of the task error.
    """

    def __init__(self, task: Task, penalty: npt.ArrayLike = 1e3):
        """Constructor.

        Args:
            task: The task to constrain.
            penalty: Non-negative, finite :math:`\\ell_1` penalty. A scalar, or a
                vector with the dimension of the task error.

        Raises:
            InvalidConstraint: If the penalty is not a finite, non-negative scalar
                or vector.
        """
        penalty = np.array(penalty, dtype=float)
        if penalty.ndim > 1:
            raise InvalidConstraint(
                f"`penalty` must be a scalar or a vector, got shape {penalty.shape}"
            )
        if not np.all(np.isfinite(penalty)) or np.any(penalty < 0.0):
            raise InvalidConstraint("`penalty` must be finite and >= 0")
        self.task = task
        self.penalty = penalty

    def compute_penalized_rows(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute :math:`(J, \alpha e, \rho)` over components with :math:`\rho_i > 0`.

        Raises:
            InvalidConstraint: If the penalty length does not match the task error.
        """
        jacobian = self.task.compute_jacobian(configuration)
        error = self.task.gain * self.task.compute_error(configuration)
        if self.penalty.ndim == 1 and self.penalty.shape != error.shape:
            raise InvalidConstraint(
                f"`penalty` has shape {self.penalty.shape} but the task error has "
                f"shape {error.shape}"
            )
        rho = np.broadcast_to(self.penalty, error.shape)
        if rho.all():
            return jacobian, error, rho
        active = rho > 0.0
        return jacobian[active], error[active], rho[active]
