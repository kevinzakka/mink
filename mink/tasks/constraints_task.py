"""Constraints task implementation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
import mujoco

from ..configuration import Configuration
from .task import Task
from .exceptions import TaskDefinitionError


class ConstraintsTask(Task):
    """Regulate the equality constraints in the MuJoCo model.

    This task aims to satisfy the equality constraints defined in the MuJoCo model.
    It extracts the constraint error and Jacobian directly from the MuJoCo data
    structure.

    Attributes:
        constraint_cost: Cost vector or scalar for the constraint error.
    """

    def __init__(
        self,
        constraint_cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Constructor.

        Args:
            constraint_cost: Cost vector or scalar for the constraint error.
            gain: Task gain alpha in [0, 1] for additional low-pass filtering. Defaults
                to 1.0 (no filtering) for dead-beat control.
            lm_damping: Unitless scale of the Levenberg-Marquardt regularization term,
                which helps when targets are infeasible. Increase this value if the task
                is too jerky under unfeasible targets.
        """
        # Initialize with a placeholder cost, will be updated in compute_error
        super().__init__(cost=np.zeros(1), gain=gain, lm_damping=lm_damping)
        
        self.constraint_cost = np.atleast_1d(constraint_cost)
        
        # The actual cost will be set in compute_error when we know the number of constraints
        self._constraints_mask = None
        self._n_constraints = None

    def _update_constraints_info(self, configuration: Configuration) -> None:
        """Update the constraints mask and count.

        Args:
            configuration: Robot configuration.
        """
        mj_data = configuration.data
        
        # Get the mask for equality constraints
        self._constraints_mask = np.argwhere(
            mj_data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        ).ravel()
        
        # Update the number of constraints
        self._n_constraints = len(self._constraints_mask)
        
        # Update the cost vector if needed
        if self._n_constraints > 0:
            if len(self.constraint_cost) == 1:
                # Use the same cost for all constraints
                self.cost = np.full(self._n_constraints, self.constraint_cost[0])
            elif len(self.constraint_cost) == self._n_constraints:
                # Use the provided cost vector
                self.cost = self.constraint_cost
            else:
                raise TaskDefinitionError(
                    f"{self.__class__.__name__} constraint cost should be a scalar or "
                    f"a vector of length {self._n_constraints} but got "
                    f"{len(self.constraint_cost)}"
                )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the constraint error.

        The error is extracted directly from the MuJoCo data structure, specifically
        from the `efc_pos` field which contains the constraint violations.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Constraint error vector.
        """
        # Update constraints information
        self._update_constraints_info(configuration)
        
        # If there are no constraints of the specified type, return an empty array
        if self._n_constraints == 0:
            return np.zeros(1)
        
        # Extract the constraint error from MuJoCo data
        mj_data = configuration.data
        error = mj_data.efc_pos.copy()[self._constraints_mask]
        
        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the constraint Jacobian.

        The Jacobian is extracted directly from the MuJoCo data structure, specifically
        from the `efc_J` field which contains the constraint Jacobian. For equality
        constraints, this Jacobian relates changes in the configuration to changes in
        the constraint violations:

        .. math::

            \dot{c}(q) = J(q) \dot{q}

        where :math:`c(q)` is the constraint function and :math:`J(q)` is its Jacobian.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Constraint Jacobian matrix :math:`J(q)`.
        """
        # Update constraints information if needed
        if self._constraints_mask is None:
            self._update_constraints_info(configuration)
        
        # If there are no constraints of the specified type, return an empty array
        if self._n_constraints == 0:
            return np.zeros((0, configuration.model.nv))
        
        # Extract the constraint Jacobian from MuJoCo data
        mj_data = configuration.data
        model = configuration.model
        
        # Reshape the Jacobian and select only the rows corresponding to our constraints
        jacobian = np.reshape(
            mj_data.efc_J.copy(), (mj_data.nefc, model.nv)
        )[self._constraints_mask]
        
        return jacobian 