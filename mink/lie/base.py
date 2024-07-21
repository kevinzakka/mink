import abc
from typing import Self, overload

import numpy as np


class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups.

    Attributes:
        matrix_dim: Dimension of square matrix output.
        parameters_dim:: Dimension of underlying parameters.
        tangent_dim:: Dimension of tangent space.
        space_dim: Dimension of coordinates that can be transformed.
    """

    matrix_dim: int
    parameters_dim: int
    tangent_dim: int
    space_dim: int

    @overload
    def __matmul__(self, other: Self) -> Self: ...

    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray: ...

    def __matmul__(self, other: Self | np.ndarray) -> Self | np.ndarray:
        """Overload of the @ operator."""
        if isinstance(other, np.ndarray):
            return self.apply(target=other)
        assert isinstance(other, MatrixLieGroup)
        return self.multiply(other=other)

    # Factory.

    @classmethod
    @abc.abstractmethod
    def identity(cls) -> Self:
        """Returns identity element."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls, matrix: np.ndarray) -> Self:
        """Get group member from matrix representation."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def sample_uniform(cls) -> Self:
        """Draw a uniform sample from the group."""
        raise NotImplementedError

    # Accessors.

    @abc.abstractmethod
    def as_matrix(self) -> np.ndarray:
        """Get transformation as a matrix."""
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self) -> np.ndarray:
        """Get underlying representation."""
        raise NotImplementedError

    # Operations.

    @abc.abstractmethod
    def apply(self, target: np.ndarray) -> np.ndarray:
        """Applies group action to a point."""
        raise NotImplementedError

    @abc.abstractmethod
    def multiply(self, other: Self) -> Self:
        """Composes this transformation with another."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def exp(cls, tangent: np.ndarray) -> Self:
        """Computes `expm(wedge(tangent))`."""
        raise NotImplementedError

    @abc.abstractmethod
    def log(self) -> np.ndarray:
        """Computes `vee(logm(transformation matrix))`."""
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint(self) -> np.ndarray:
        """Computes the adjoint."""
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self) -> Self:
        """Computes the inverse of the transform."""
        raise NotImplementedError

    @abc.abstractmethod
    def normalize(self) -> Self:
        """Normalize/projects values and returns."""
        raise NotImplementedError
