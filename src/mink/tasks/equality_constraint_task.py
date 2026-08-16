"""Equality constraint task implementation."""

from __future__ import annotations

import logging
from typing import Sequence

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..constants import constraint_width
from ..exceptions import InvalidConstraint, TaskDefinitionError
from .task import Task


def _get_dense_constraint_jacobian(
    model: mujoco.MjModel, data: mujoco.MjData
) -> np.ndarray:
    """Return the dense constraint Jacobian for a model."""
    if mujoco.mj_isSparse(model):
        efc_J = np.empty((data.nefc, model.nv))
        mujoco.mju_sparse2dense(
            efc_J,
            data.efc_J,
            data.efc_J_rownnz,
            data.efc_J_rowadr,
            data.efc_J_colind,
        )
        return efc_J
    return data.efc_J.reshape((data.nefc, model.nv)).copy()


class EqualityConstraintTask(Task):
    """Regulate equality constraints in a model.

    Equality constraints are useful, among other things, for modeling "loop joints"
    such as four-bar linkages. In MuJoCo, there are several types of equality
    constraints, including:

    * ``mjEQ_CONNECT``: Connect two bodies at a point (ball joint).
    * ``mjEQ_WELD``: Fix relative pose of two bodies.
    * ``mjEQ_JOINT``: Couple the values of two scalar joints.
    * ``mjEQ_TENDON``: Couple the values of two tendons.

    This task can regulate all equality constraints in the model or a specific subset
    identified by name or ID.

    .. note::

        MuJoCo computes the constraint residual and its Jacobian and stores them in
        ``data.efc_pos`` and ``data.efc_J`` (potentially in sparse format), respectively.
        The :func:`compute_error` and :func:`compute_jacobian` methods simply extract the
        rows corresponding to the active equality constraints specified for this task
        from ``data.efc_pos`` and ``data.efc_J``. More information on MuJoCo's constraint
        model can be found in [MuJoCoEqualityConstraints]_.

    Attributes:
        equalities: ID or name of the equality constraints to regulate. If not provided,
            the task will regulate all equality constraints in the model.
        cost: Cost vector for the equality constraint task. Either a scalar, in which
            case the same cost is applied to all regulated constraints, or a vector
            with one entry per regulated constraint, ``cost[i]`` pairing with
            ``equalities[i]``. Internally the expanded per-row cost always follows
            constraint-id order, which is also the row order of the error and
            Jacobian.

    Raises:
        InvalidConstraint: If a specified equality constraint name or ID is not found,
            or if the constraint is not active at the initial configuration.
        TaskDefinitionError: If no equality constraints are found or if cost parameters
            have invalid shape or values.

    Example:

    .. code-block:: python

        # Regulate all equality constraints with the same cost.
        eq_task = EqualityConstraintTask(model, cost=1.0)

        # Regulate specific equality constraints with different costs.
        eq_task = EqualityConstraintTask(
            model,
            cost=[1.0, 0.5],
            equalities=["connect_right", "connect_left"]
        )
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: npt.ArrayLike,
        equalities: Sequence[int | str] | None = None,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        self._logger = logging.getLogger(__package__)
        eq_ids = self._resolve_equality_ids(model, equalities)
        # Store ids sorted so that the expanded cost vector is always in constraint-id
        # order, which is also the row order of the efc arrays. `_cost_order` permutes
        # a user cost vector (paired with `equalities`) into that order.
        self._cost_order = np.argsort(eq_ids)
        self._eq_ids = eq_ids[self._cost_order]
        self._eq_types = model.eq_type[self._eq_ids].copy()
        self._neq_total = len(self._eq_ids)
        # Maps a MuJoCo equality constraint id to its position in `_eq_ids`, so
        # that per-constraint costs can be looked up from `efc_id` values.
        self._eq_id_to_pos = np.full(model.neq, -1)
        self._eq_id_to_pos[self._eq_ids] = np.arange(self._neq_total)
        self._mask: np.ndarray | None = None

        super().__init__(cost=np.zeros((1,)), gain=gain, lm_damping=lm_damping)
        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        """Set the cost vector for the equality constraint task.

        Args:
            cost: Cost vector for the equality constraint task.
        """
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self._neq_total):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a vector of shape (1,) "
                f"or ({self._neq_total},). Got {cost.shape}."
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")

        # Per constraint cost, permuted into constraint-id order.
        self._cost = (
            np.full((self._neq_total,), cost[0])
            if cost.shape[0] == 1
            else cost[self._cost_order]
        )

        # Expanded per constraint dimension.
        repeats = [constraint_width(eq_type) for eq_type in self._eq_types]
        self.cost = np.repeat(self._cost, repeats)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the task error (constraint residual) :math:`e(q) = r(q)`.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task error vector :math:`e(q)` for the active equality constraints.
        """
        self._update_active_constraints(configuration)
        return configuration.data.efc_pos[self._mask]

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the task Jacobian (constraint Jacobian) :math:`J(q) = J_r(q)`.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task jacobian :math:`J(q)` for the active equality constraints.
        """
        self._update_active_constraints(configuration)
        efc_J = _get_dense_constraint_jacobian(configuration.model, configuration.data)
        return efc_J[self._mask]

    # Helper functions.

    def _update_active_constraints(self, configuration: Configuration) -> None:
        self._mask = (
            configuration.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        ) & np.isin(configuration.data.efc_id, self._eq_ids)
        active_eq_ids = configuration.data.efc_id[self._mask]
        self.cost = self._cost[self._eq_id_to_pos[active_eq_ids]]

    def _resolve_equality_ids(
        self, model: mujoco.MjModel, equalities: Sequence[int | str] | None
    ) -> np.ndarray:
        eq_ids: list[int] = []

        if equalities is not None:
            for eq_id_or_name in equalities:
                eq_id: int
                if isinstance(eq_id_or_name, str):
                    eq_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id_or_name
                    )
                    if eq_id == -1:
                        raise InvalidConstraint(
                            f"Equality constraint '{eq_id_or_name}' not found."
                        )
                else:
                    eq_id = eq_id_or_name
                    if eq_id < 0 or eq_id >= model.neq:
                        raise InvalidConstraint(
                            f"Equality constraint index {eq_id} out of range. "
                            f"Must be in range [0, {model.neq})."
                        )
                if not model.eq_active0[eq_id]:
                    raise InvalidConstraint(
                        f"Equality constraint {eq_id} is not active at initial "
                        "configuration."
                    )
                else:
                    eq_ids.append(eq_id)
            # Check for duplicates.
            if len(eq_ids) != len(set(eq_ids)):
                raise TaskDefinitionError(
                    f"Duplicate equality constraint IDs provided: {eq_ids}."
                )
        else:
            eq_ids = list(range(model.neq))
            self._logger.info("Regulating %d equality constraints", len(eq_ids))

        # Ensure we have at least 1 constraint.
        if len(eq_ids) == 0:
            raise TaskDefinitionError(
                f"{self.__class__.__name__} no equality constraints found in this "
                "model."
            )

        return np.array(eq_ids)
