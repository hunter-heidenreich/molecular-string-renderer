"""
Main CLI entry point.

This module provides the main CLI function that orchestrates the command-line
interface by parsing arguments and routing to appropriate handlers.
"""

import logging
import sys

from molecular_string_renderer.cli.config_builder import create_configs
from molecular_string_renderer.cli.handlers import (
    handle_grid_rendering,
    handle_single_rendering,
)
from molecular_string_renderer.cli.parser import create_parser
from molecular_string_renderer.cli.validation import (
    validate_input_arguments,
    validate_molecular_input,
)
from molecular_string_renderer.core import get_supported_formats
from molecular_string_renderer.exceptions import (
    CLIConfigurationError,
    CLIError,
    CLIValidationError,
)


def main() -> None:
    """Main CLI entry point.

    Parses command-line arguments, configures logging, and routes to the
    appropriate rendering handler based on the provided arguments.
    """
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging based on verbosity
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    try:
        # Handle utility options first
        if args.list_formats:
            formats = get_supported_formats()
            print("Supported formats:")
            print("\nInput formats:")
            for fmt, desc in formats["input_formats"].items():
                print(f"  {fmt:8} - {desc}")
            print("\nOutput formats:")
            for fmt, desc in formats["output_formats"].items():
                print(f"  {fmt:8} - {desc}")
            return

        # Validate arguments
        validate_input_arguments(args)

        # Create configurations
        render_config, parser_config, output_config = create_configs(args)

        # Route to appropriate handler
        if args.grid:
            handle_grid_rendering(args, render_config, parser_config, output_config)
        else:
            # Early validation for single molecule if not in validate-only mode
            if not args.validate:
                validate_molecular_input(args.molecular_string, args.input_format)
            handle_single_rendering(args, render_config, parser_config, output_config)

    except CLIError as e:
        logging.error(e.message)
        sys.exit(e.exit_code)
    except CLIValidationError as e:
        logging.error(e.message)
        if not args.validate:
            logging.error("Use --help for usage information")
        sys.exit(e.exit_code)
    except CLIConfigurationError as e:
        logging.error(e.message)
        sys.exit(e.exit_code)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
