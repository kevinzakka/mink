"""Exceptions raised by tasks."""

from ..exceptions import MinkError


class TaskDefinitionError(MinkError):
    """Exception raised when a task definition is ill-formed."""


class TargetNotSet(MinkError):
    """Exception raised when attempting to use a task with an unset target."""

    def __init__(self, cls_name: str):
        message = f"No target set for {cls_name}"
        super().__init__(message)


class InvalidTarget(MinkError):
    """Exception raised when the target is invalid."""


class InvalidGain(MinkError):
    """Exception raised when the gain is outside the valid range."""


class InvalidDamping(MinkError):
    """Exception raised when the damping is outside the valid range."""
