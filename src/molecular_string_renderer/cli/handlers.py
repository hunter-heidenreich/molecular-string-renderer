"""
Rendering handlers for different CLI modes.

This module contains the handler functions for different rendering modes:
single molecule rendering and grid rendering.
"""

import logging

from molecular_string_renderer.cli.utils import log_rendering_info
from molecular_string_renderer.core import (
    render_molecule,
    render_molecules_grid,
    validate_molecular_string,
)
from molecular_string_renderer.exceptions import CLIError, CLIValidationError
from molecular_string_renderer.outputs.utils import create_safe_filename

# Set up module-level logger
logger = logging.getLogger(__name__)


def handle_grid_rendering(args, render_config, parser_config, output_config) -> None:
    """Handle grid rendering mode.

    Args:
        args: Parsed command-line arguments.
        render_config: Rendering configuration.
        parser_config: Parser configuration.
        output_config: Output configuration.

    Raises:
        CLIValidationError: If grid input is invalid.
        CLIError: If rendering fails.
    """
    if not args.grid:
        raise CLIValidationError(
            "--grid requires a comma-separated list of molecular strings"
        )

    # Parse grid input
    molecular_strings = [s.strip() for s in args.grid.split(",") if s.strip()]

    if not molecular_strings:
        raise CLIValidationError("No valid molecular strings found in grid input")

    # Parse legends if provided
    legends = None
    if args.legends:
        legends = [s.strip() for s in args.legends.split(",")]
        if len(legends) != len(molecular_strings):
            logger.warning(
                f"Number of legends ({len(legends)}) doesn't match number of molecules ({len(molecular_strings)})"
            )
            legends = None

    if args.verbose:
        log_rendering_info(
            input_string="grid input",
            input_format=args.input_format,
            output_config=output_config,
            molecule_count=len(molecular_strings),
            mols_per_row=args.mols_per_row,
        )

    try:
        render_molecules_grid(
            molecular_strings=molecular_strings,
            format_type=args.input_format,
            output_format=output_config.format,
            output_path=args.output,
            legends=legends,
            mols_per_row=args.mols_per_row,
            mol_size=(args.mol_size, args.mol_size),
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
        )

        if args.output:
            logger.info(f"Grid successfully saved to: {args.output}")
        else:
            logger.info("Grid rendered successfully (no output file specified)")

    except Exception as e:
        raise CLIError(f"Error rendering grid: {e}") from e


def handle_single_rendering(args, render_config, parser_config, output_config) -> None:
    """Handle single molecule rendering.

    Args:
        args: Parsed command-line arguments.
        render_config: Rendering configuration.
        parser_config: Parser configuration.
        output_config: Output configuration.

    Raises:
        CLIValidationError: If molecular string is missing.
        CLIError: If rendering fails.
    """
    if not args.molecular_string:
        raise CLIValidationError(
            "Molecular string is required for single molecule rendering"
        )

    if args.verbose:
        log_rendering_info(
            input_string=args.molecular_string,
            input_format=args.input_format,
            output_config=output_config,
            render_config=render_config,
        )

    # Validate input if requested
    if args.validate:
        is_valid = validate_molecular_string(args.molecular_string, args.input_format)
        if is_valid:
            print(f"✓ Valid {args.input_format.upper()}: {args.molecular_string}")
            return
        else:
            print(f"✗ Invalid {args.input_format.upper()}: {args.molecular_string}")
            raise CLIValidationError(
                f"Invalid {args.input_format.upper()}: {args.molecular_string}"
            )

    try:
        render_molecule(
            molecular_string=args.molecular_string,
            format_type=args.input_format,
            output_format=output_config.format,
            output_path=args.output,
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
            auto_filename=args.auto_filename and not args.output,
        )

        if args.output or args.auto_filename:
            output_file = args.output or create_safe_filename(
                args.molecular_string, f".{output_config.format}"
            )
            logger.info(f"Image successfully saved to: {output_file}")
        else:
            logger.info("Molecule rendered successfully (no output file specified)")

    except Exception as e:
        raise CLIError(f"Error rendering molecule: {e}") from e