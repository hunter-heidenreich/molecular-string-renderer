"""
Exception classes for molecular string rendering.

Provides a hierarchical exception structure for different types of errors
that can occur during molecular parsing, rendering, and output operations.
"""


class MolecularRenderingError(Exception):
    """Base exception for molecular rendering operations."""

    pass


class ParsingError(MolecularRenderingError):
    """Exception raised when molecular string parsing fails."""

    pass


class RenderingError(MolecularRenderingError):
    """Exception raised when molecule rendering fails."""

    pass


class OutputError(MolecularRenderingError):
    """Exception raised when output generation fails."""

    pass


class ValidationError(MolecularRenderingError):
    """Exception raised when input validation fails."""

    pass


class ConfigurationError(MolecularRenderingError):
    """Exception raised when configuration is invalid."""

    pass


# CLI-specific exceptions
class CLIError(Exception):
    """Base exception for CLI-related errors."""

    def __init__(self, message: str, exit_code: int = 1):
        self.message = message
        self.exit_code = exit_code
        super().__init__(message)


class CLIValidationError(CLIError):
    """Exception raised when CLI input validation fails."""

    pass


class CLIConfigurationError(CLIError):
    """Exception raised when CLI configuration is invalid."""

    pass
