"""
Command-line interface for molecular string renderer.

Provides a user-friendly CLI that maintains backward compatibility
with the original SMILES to PNG converter script.
"""

import argparse
import logging
import sys
from pathlib import Path

from molecular_string_renderer import __version__
from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.core import (
    get_supported_formats,
    render_molecule,
    render_molecules_grid,
    validate_molecular_string,
)
from molecular_string_renderer.exceptions import (
    CLIConfigurationError,
    CLIError,
    CLIValidationError,
)
from molecular_string_renderer.outputs.utils import create_safe_filename


# Set up logging
logger = logging.getLogger(__name__)


def validate_input_arguments(args) -> None:
    """Validate command-line arguments for consistency and completeness.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValidationError: If arguments are invalid or inconsistent.
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
        ValidationError: If the molecular string is invalid with helpful message.
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


def log_rendering_info(
    input_string: str,
    input_format: str,
    output_config: OutputConfig,
    render_config: RenderConfig | None = None,
    molecule_count: int | None = None,
    mols_per_row: int | None = None,
) -> None:
    """Log rendering configuration information.

    Args:
        input_string: Input molecular string or description.
        input_format: Input format type.
        output_config: Output configuration.
        render_config: Optional render configuration for dimensions.
        molecule_count: Number of molecules (for grid rendering).
        mols_per_row: Molecules per row (for grid rendering).
    """
    if molecule_count:
        logger.info(f"Rendering grid with {molecule_count} molecules")
        if mols_per_row:
            logger.info(f"Molecules per row: {mols_per_row}")
    else:
        logger.info(f"Input: {input_string}")

    logger.info(f"Input format: {input_format}")
    logger.info(f"Output format: {output_config.format}")

    if render_config:
        logger.info(f"Image size: {render_config.width}x{render_config.height}")


def normalize_format(value: str) -> str:
    """Normalize format input to lowercase."""
    normalized = value.lower().strip()
    valid_formats = ["smiles", "smi", "inchi", "mol", "selfies"]
    if normalized not in valid_formats:
        raise argparse.ArgumentTypeError(
            f"invalid choice: '{value}' (choose from {valid_formats})"
        )
    return normalized


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.

    Returns:
        A configured ArgumentParser instance with all CLI options defined.
    """

    parser = argparse.ArgumentParser(
        prog="mol-render",
        description="Convert molecular string representations to publication-quality images.",
        epilog="""
Examples:
  %(prog)s "CCO"                                    # Ethanol (SMILES) to PNG
  %(prog)s "CCO" -o ethanol.png                     # Custom output filename
  %(prog)s "CCO" --format smiles --output-format svg    # SVG output
  %(prog)s "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3" --format inchi    # InChI input
  %(prog)s --grid "CCO,CC(=O)O,C1=CC=CC=C1" --legends "Ethanol,Acetic acid,Benzene"
  %(prog)s "CCO" --size 800 --background-color "#f0f0f0"    # Large image with custom background

Common molecular formats:
  SMILES:   CCO (ethanol), CC(=O)O (acetic acid), C1=CC=CC=C1 (benzene)
  InChI:    InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3 (ethanol)
  SELFIES:  [C][C][O] (ethanol), [C][C][=Branch1][C][=O][O] (acetic acid)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Version
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Input options
    input_group = parser.add_argument_group("Input Options")

    input_group.add_argument(
        "molecular_string",
        nargs="?",
        type=str,
        help="Molecular string representation (SMILES, InChI, etc.)",
    )

    input_group.add_argument(
        "--format",
        "--input-format",
        dest="input_format",
        type=normalize_format,
        default="smiles",
        help="Input molecular format (default: smiles). Case-insensitive.",
    )

    input_group.add_argument(
        "--grid",
        type=str,
        help="Comma-separated list of molecular strings for grid rendering",
    )

    input_group.add_argument(
        "--legends",
        type=str,
        help="Comma-separated legends for grid molecules",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")

    output_group.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="FILE",
        help="Output filename. Extension determines format if --output-format not specified.",
    )

    output_group.add_argument(
        "--output-format",
        type=str,
        choices=["png", "svg", "jpg", "jpeg", "pdf", "webp", "tiff", "tif", "bmp"],
        help="Output image format (default: inferred from filename or png)",
    )

    output_group.add_argument(
        "--auto-filename",
        action="store_true",
        default=True,
        help="Generate safe filename if --output not provided (default: true)",
    )

    # Rendering options
    render_group = parser.add_argument_group("Rendering Options")

    render_group.add_argument(
        "-s",
        "--size",
        type=int,
        default=500,
        metavar="PIXELS",
        help="Square image size in pixels (default: 500)",
    )

    render_group.add_argument(
        "--width",
        type=int,
        metavar="PIXELS",
        help="Image width in pixels (overrides --size)",
    )

    render_group.add_argument(
        "--height",
        type=int,
        metavar="PIXELS",
        help="Image height in pixels (overrides --size)",
    )

    render_group.add_argument(
        "--background-color",
        "--bg-color",
        dest="background_color",
        type=str,
        default="white",
        help="Background color (name or hex code, default: white)",
    )

    render_group.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for high-quality output (default: 150)",
    )

    render_group.add_argument(
        "--show-hydrogen",
        action="store_true",
        help="Show explicit hydrogen atoms",
    )

    render_group.add_argument(
        "--show-carbon",
        action="store_true",
        help="Show carbon atom labels",
    )

    # Grid options
    grid_group = parser.add_argument_group("Grid Options")

    grid_group.add_argument(
        "--mols-per-row",
        type=int,
        default=4,
        help="Number of molecules per row in grid (default: 4)",
    )

    grid_group.add_argument(
        "--mol-size",
        type=int,
        default=200,
        help="Size of each molecule in grid (default: 200)",
    )

    # Quality options
    quality_group = parser.add_argument_group("Quality Options")

    quality_group.add_argument(
        "--quality",
        type=int,
        default=95,
        metavar="1-100",
        help="Output quality 1-100 (default: 95)",
    )

    quality_group.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable output optimization",
    )

    # Utility options
    util_group = parser.add_argument_group("Utility Options")

    util_group.add_argument(
        "--validate",
        action="store_true",
        help="Only validate input, don't render",
    )

    util_group.add_argument(
        "--list-formats",
        action="store_true",
        help="List supported input and output formats",
    )

    util_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def determine_output_format(output_path: str | None, output_format: str | None) -> str:
    """Determine output format from filename or explicit format.

    Args:
        output_path: Optional output file path.
        output_format: Optional explicit output format.

    Returns:
        The determined output format string (e.g., 'png', 'svg').
    """

    if output_format:
        return output_format.lower()

    if output_path:
        suffix = Path(output_path).suffix.lower()
        format_map = {
            ".png": "png",
            ".svg": "svg",
            ".jpg": "jpg",
            ".jpeg": "jpg",
            ".pdf": "pdf",
            ".webp": "webp",
            ".tiff": "tiff",
            ".tif": "tiff",
            ".bmp": "bmp",
        }
        return format_map.get(suffix, "png")

    return "png"


def create_configs(args) -> tuple[RenderConfig, ParserConfig, OutputConfig]:
    """Create configuration objects from command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A tuple containing (render_config, parser_config, output_config).

    Raises:
        ConfigurationError: If configuration values are invalid.
    """
    # Determine image dimensions
    width = args.width or args.size
    height = args.height or args.size

    # Validate dimensions
    if width < 100 or width > 2000:
        raise CLIConfigurationError(f"Width must be between 100-2000 pixels, got {width}")
    if height < 100 or height > 2000:
        raise CLIConfigurationError(
            f"Height must be between 100-2000 pixels, got {height}"
        )

    # Validate DPI
    if args.dpi < 72 or args.dpi > 600:
        raise CLIConfigurationError(f"DPI must be between 72-600, got {args.dpi}")

    # Validate quality
    if args.quality < 1 or args.quality > 100:
        raise CLIConfigurationError(f"Quality must be between 1-100, got {args.quality}")

    try:
        render_config = RenderConfig(
            width=width,
            height=height,
            background_color=args.background_color,
            show_hydrogen=args.show_hydrogen,
            show_carbon=args.show_carbon,
        )

        parser_config = ParserConfig(
            sanitize=True,
            show_hydrogen=args.show_hydrogen,
        )

        output_format = determine_output_format(args.output, args.output_format)

        output_config = OutputConfig(
            format=output_format,
            quality=args.quality,
            optimize=not args.no_optimize,
            dpi=args.dpi,
        )
    except Exception as e:
        raise CLIConfigurationError(f"Failed to create configuration: {e}") from e

    return render_config, parser_config, output_config


def handle_grid_rendering(args, render_config, parser_config, output_config) -> None:
    """Handle grid rendering mode.

    Args:
        args: Parsed command-line arguments.
        render_config: Rendering configuration.
        parser_config: Parser configuration.
        output_config: Output configuration.

    Raises:
        ValidationError: If grid input is invalid.
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
        ValidationError: If molecular string is missing.
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
        logger.error(e.message)
        sys.exit(e.exit_code)
    except CLIValidationError as e:
        logger.error(e.message)
        if not args.validate:
            logger.error("Use --help for usage information")
        sys.exit(e.exit_code)
    except CLIConfigurationError as e:
        logger.error(e.message)
        sys.exit(e.exit_code)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
