"""
CLI module for molecular string renderer.

This module provides a modular command-line interface by splitting CLI functionality
into focused submodules while maintaining backward compatibility with the original
single-file CLI implementation.
"""

# Import main CLI function to maintain backward compatibility
from molecular_string_renderer.cli.main import main

# Re-export all CLI components for internal use
from molecular_string_renderer.cli.config_builder import create_configs, determine_output_format
from molecular_string_renderer.cli.handlers import handle_grid_rendering, handle_single_rendering
from molecular_string_renderer.cli.parser import create_parser
from molecular_string_renderer.cli.utils import log_rendering_info, normalize_format
from molecular_string_renderer.cli.validation import validate_input_arguments, validate_molecular_input

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