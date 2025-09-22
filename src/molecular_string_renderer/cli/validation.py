"""
CLI validation functions.

This module contains validation functions specific to CLI input arguments
and molecular string validation with user-friendly error messages.
"""

from molecular_string_renderer.core import validate_molecular_string
from molecular_string_renderer.exceptions import CLIValidationError


def validate_input_arguments(args) -> None:
    """Validate command-line arguments for consistency and completeness.

    Args:
        args: Parsed command-line arguments.

    Raises:
        CLIValidationError: If arguments are invalid or inconsistent.
    """
    if args.validate and not args.molecular_string:
        raise CLIValidationError("--validate requires a molecular string argument")

    if not args.grid and not args.molecular_string and not args.list_formats:
        raise CLIValidationError("No action specified. Use --help for usage information.")

    # Validate grid-specific arguments
    if args.grid and args.molecular_string:
        raise CLIValidationError(
            "Cannot specify both --grid and individual molecular string"
        )

    # Validate legends only make sense with grid
    if args.legends and not args.grid:
        raise CLIValidationError("--legends can only be used with --grid")


def validate_molecular_input(molecular_string: str, format_type: str) -> None:
    """Validate and provide helpful feedback for molecular string input.

    Args:
        molecular_string: The molecular string to validate.
        format_type: The format type to validate against.

    Raises:
        CLIValidationError: If the molecular string is invalid with helpful message.
    """
    if not molecular_string or not molecular_string.strip():
        raise CLIValidationError(f"Empty {format_type.upper()} string provided")

    # Basic format-specific validation hints
    format_hints = {
        "smiles": "SMILES should contain atoms (C, N, O, etc.) and bonds. Example: 'CCO' for ethanol",
        "inchi": "InChI should start with 'InChI=' and contain a valid identifier",
        "selfies": "SELFIES should contain bracketed tokens. Example: '[C][C][O]' for ethanol",
        "mol": "MOL format should contain a complete molecule structure block",
    }

    try:
        is_valid = validate_molecular_string(molecular_string.strip(), format_type)
        if not is_valid:
            hint = format_hints.get(format_type.lower(), "")
            raise CLIValidationError(
                f"Invalid {format_type.upper()}: '{molecular_string}'. {hint}"
            )
    except Exception as e:
        hint = format_hints.get(format_type.lower(), "")
        raise CLIValidationError(
            f"Failed to validate {format_type.upper()}: '{molecular_string}'. {hint}\nError: {e}"
        ) from e