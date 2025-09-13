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

# Set up logging
logger = logging.getLogger(__name__)


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

    quality_group.add_argument(
        "--no-antialias",
        action="store_true",
        help="Disable antialiasing",
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
    """

    # Determine image dimensions
    width = args.width or args.size
    height = args.height or args.size

    render_config = RenderConfig(
        width=width,
        height=height,
        background_color=args.background_color,
        dpi=args.dpi,
        antialias=not args.no_antialias,
        show_hydrogen=args.show_hydrogen,
        show_carbon=args.show_carbon,
    )

    parser_config = ParserConfig(
        sanitize=True,
        remove_hs=not args.show_hydrogen,
        strict=False,
    )

    output_format = determine_output_format(args.output, args.output_format)

    output_config = OutputConfig(
        format=output_format,
        quality=args.quality,
        optimize=not args.no_optimize,
    )

    return render_config, parser_config, output_config


def handle_grid_rendering(args, render_config, parser_config, output_config) -> None:
    """Handle grid rendering mode.

    Args:
        args: Parsed command-line arguments.
        render_config: Rendering configuration.
        parser_config: Parser configuration.
        output_config: Output configuration.
    """

    if not args.grid:
        logger.error("--grid requires a comma-separated list of molecular strings")
        sys.exit(1)

    # Parse grid input
    molecular_strings = [s.strip() for s in args.grid.split(",") if s.strip()]

    if not molecular_strings:
        logger.error("No valid molecular strings found in grid input")
        sys.exit(1)

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
        logger.info(f"Rendering grid with {len(molecular_strings)} molecules")
        logger.info(f"Input format: {args.input_format}")
        logger.info(f"Output format: {output_config.format}")
        logger.info(f"Molecules per row: {args.mols_per_row}")

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
        logger.error(f"Error rendering grid: {e}")
        sys.exit(1)


def handle_single_rendering(args, render_config, parser_config, output_config) -> None:
    """Handle single molecule rendering.

    Args:
        args: Parsed command-line arguments.
        render_config: Rendering configuration.
        parser_config: Parser configuration.
        output_config: Output configuration.
    """

    if not args.molecular_string:
        logger.error("Molecular string is required for single molecule rendering")
        logger.error("Use --help for usage information")
        sys.exit(1)

    if args.verbose:
        logger.info(f"Input: {args.molecular_string}")
        logger.info(f"Input format: {args.input_format}")
        logger.info(f"Output format: {output_config.format}")
        logger.info(f"Image size: {render_config.width}x{render_config.height}")

    # Validate input if requested
    if args.validate:
        is_valid = validate_molecular_string(args.molecular_string, args.input_format)
        if is_valid:
            print(f"✓ Valid {args.input_format.upper()}: {args.molecular_string}")
            sys.exit(0)
        else:
            print(f"✗ Invalid {args.input_format.upper()}: {args.molecular_string}")
            sys.exit(1)

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
            output_file = (
                args.output
                or f"molecule_{hash(args.molecular_string) % 100000}.{output_config.format}"
            )
            logger.info(f"Image successfully saved to: {output_file}")
        else:
            logger.info("Molecule rendered successfully (no output file specified)")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


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
        sys.exit(0)

    # Validate arguments
    if not args.grid and not args.molecular_string and not args.validate:
        parser.print_help()
        sys.exit(1)

    # Create configurations
    try:
        render_config, parser_config, output_config = create_configs(args)
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Route to appropriate handler
    if args.grid:
        handle_grid_rendering(args, render_config, parser_config, output_config)
    else:
        handle_single_rendering(args, render_config, parser_config, output_config)


if __name__ == "__main__":
    main()
