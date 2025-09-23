"""
Logging utilities for molecular rendering operations.

Provides context managers and helper functions for consistent operation logging
with timing and error handling.
"""

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def logged_operation(operation_name: str, details: dict[str, str | int] | None = None):
    """
    Context manager for logging operations with timing and error handling.

    Args:
        operation_name: Name of the operation being performed
        details: Optional dictionary of operation details to log

    Yields:
        The start time of the operation

    Example:
        >>> with logged_operation("parse", {"format": "smiles"}):
        ...     # Do parsing work
        ...     pass
    """
    start_time = log_operation_start(operation_name, details)
    try:
        yield start_time
        log_operation_end(operation_name, start_time, True)
    except Exception:
        log_operation_end(operation_name, start_time, False)
        raise


def log_operation_start(
    operation: str, details: dict[str, str | int] | None = None
) -> float:
    """
    Log the start of a rendering operation and return start time.

    Args:
        operation: Name of the operation (e.g., 'render_molecule', 'parse')
        details: Optional dictionary of operation details to log

    Returns:
        Start time for performance measurement
    """
    start_time = time.perf_counter()
    detail_str = ""
    if details:
        detail_str = " | " + " | ".join(f"{k}={v}" for k, v in details.items())
    logger.debug(f"Starting {operation}{detail_str}")
    return start_time


def log_operation_end(operation: str, start_time: float, success: bool = True) -> None:
    """
    Log the end of a rendering operation with timing.

    Args:
        operation: Name of the operation that completed
        start_time: Start time from log_operation_start
        success: Whether the operation succeeded
    """
    duration = time.perf_counter() - start_time
    status = "completed" if success else "failed"
    logger.debug(f"Operation {operation} {status} in {duration:.3f}s")
    if duration > 5.0:  # Log slow operations at info level
        logger.info(f"Slow operation: {operation} took {duration:.3f}s")
