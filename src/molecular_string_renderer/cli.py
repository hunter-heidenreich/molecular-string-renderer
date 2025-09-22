"""
Command-line interface for molecular string renderer.

Provides a user-friendly CLI that maintains backward compatibility
with the original SMILES to PNG converter script.

This module has been refactored into submodules for better organization
while maintaining the same public API.
"""

# Import the main CLI function from the new modular structure
from molecular_string_renderer.cli.main import main

# Re-export all functions for backward compatibility
from molecular_string_renderer.cli import (
    create_configs,
    create_parser,
    determine_output_format,
    handle_grid_rendering,
    handle_single_rendering,
    log_rendering_info,
    normalize_format,
    validate_input_arguments,
    validate_molecular_input,
)

# Maintain backward compatibility by providing all the original functions
# in their original location for any code that imports them directly
__all__ = [
    "main",
    "create_parser",
    "create_configs",
    "determine_output_format",
    "handle_grid_rendering", 
    "handle_single_rendering",
    "validate_input_arguments",
    "validate_molecular_input",
    "log_rendering_info",
    "normalize_format",
]


if __name__ == "__main__":
    main()
