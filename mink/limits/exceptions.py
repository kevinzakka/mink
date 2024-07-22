"""Exceptions raised by limits."""

from ..exceptions import MinkError


class LimitDefinitionError(MinkError):
    """Exception raised when a limit definition is ill-formed."""
