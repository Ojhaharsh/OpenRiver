"""
Custom exception classes for the poker solver.
"""


class SolverException(Exception):
    """Base exception for solver errors."""
    pass


class InvalidBoardError(SolverException):
    """Raised when board input is invalid."""
    pass


class InvalidIterationsError(SolverException):
    """Raised when iteration count is invalid."""
    pass


class SolverNotInitializedError(SolverException):
    """Raised when solver hasn't been initialized yet."""
    pass
