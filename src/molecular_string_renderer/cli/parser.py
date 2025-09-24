"""
Argument parser creation for the CLI.

This module handles the creation and configuration of the command-line
argument parser with all available options and help text.
"""

import argparse

from molecular_string_renderer import __version__
from molecular_string_renderer.cli.utils import normalize_format


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
        default=None,
        help="Number of molecules per row in grid (default: auto-fits to molecule count, max 4)",
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
        "--lossless",
        action="store_true",
        help="Use lossless compression (WebP only, default: true)",
    )

    quality_group.add_argument(
        "--no-lossless",
        action="store_true",
        help="Use lossy compression (WebP only)",
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
