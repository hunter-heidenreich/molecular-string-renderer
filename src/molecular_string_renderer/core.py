"""
Core interface for molecular string rendering.

Provides high-level functions that combine parsing, rendering, and output.
"""

import logging
from pathlib import Path

from PIL import Image

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.outputs import create_safe_filename, get_output_handler
from molecular_string_renderer.parsers import get_parser
from molecular_string_renderer.renderers import get_renderer


def render_molecule(
    molecular_string: str,
    format_type: str = "smiles",
    output_format: str = "png",
    output_path: str | Path | None = None,
    render_config: RenderConfig | None = None,
    parser_config: ParserConfig | None = None,
    output_config: OutputConfig | None = None,
    auto_filename: bool = True,
) -> Image.Image:
    """
    High-level function to render a molecular string to an image.

    Args:
        molecular_string: String representation of molecule (SMILES, InChI, etc.)
        format_type: Type of molecular format ('smiles', 'inchi', 'mol')
        output_format: Output image format ('png', 'svg', 'jpg')
        output_path: Path to save image (optional)
        render_config: Configuration for rendering
        parser_config: Configuration for parsing
        output_config: Configuration for output
        auto_filename: Generate safe filename if output_path not provided

    Returns:
        PIL Image object containing the rendered molecule

    Raises:
        ValueError: If molecular string is invalid or format unsupported
    """
    # Initialize configurations with defaults
    render_config = render_config or RenderConfig()
    parser_config = parser_config or ParserConfig()
    output_config = output_config or OutputConfig(format=output_format)

    # Parse the molecular string
    parser = get_parser(format_type, parser_config)
    mol = parser.parse(molecular_string)

    # Render the molecule
    renderer = get_renderer("2d", render_config)
    image = renderer.render(mol)

    # Save if output path is provided or auto_filename is enabled
    if output_path or auto_filename:
        output_handler = get_output_handler(output_format, output_config)

        # For SVG output, provide the molecule for true vector rendering
        if hasattr(output_handler, "set_molecule") and output_format.lower() == "svg":
            output_handler.set_molecule(mol)

        if not output_path and auto_filename:
            # Generate safe filename
            base_name = create_safe_filename(
                molecular_string, output_handler.file_extension
            )
            output_path = Path.cwd() / base_name

        if output_path:
            output_handler.save(image, output_path)

    return image


def render_molecules_grid(
    molecular_strings: list[str],
    format_type: str = "smiles",
    output_format: str = "png",
    output_path: str | Path | None = None,
    legends: list[str] | None = None,
    mols_per_row: int = 4,
    mol_size: tuple[int, int] = (200, 200),
    render_config: RenderConfig | None = None,
    parser_config: ParserConfig | None = None,
    output_config: OutputConfig | None = None,
) -> Image.Image:
    """
    Render multiple molecules in a grid layout.

    Args:
        molecular_strings: List of molecular string representations
        format_type: Type of molecular format ('smiles', 'inchi', 'mol')
        output_format: Output image format ('png', 'svg', 'jpg')
        output_path: Path to save image (optional)
        legends: Optional legends for each molecule
        mols_per_row: Number of molecules per row
        mol_size: Size of each molecule image (width, height)
        render_config: Configuration for rendering
        parser_config: Configuration for parsing
        output_config: Configuration for output

    Returns:
        PIL Image object containing the molecule grid
    """
    if not molecular_strings:
        raise ValueError("Cannot render empty molecule list")

    # Initialize configurations
    render_config = render_config or RenderConfig()
    parser_config = parser_config or ParserConfig()
    output_config = output_config or OutputConfig(format=output_format)

    # Parse all molecules
    parser = get_parser(format_type, parser_config)
    mols = []
    valid_indices = []  # Track which molecules were successfully parsed

    for i, mol_string in enumerate(molecular_strings):
        try:
            mol = parser.parse(mol_string)
            mols.append(mol)
            valid_indices.append(i)
        except Exception as e:
            logging.warning(f"Failed to parse '{mol_string}': {e}")
            # Don't add None - just skip invalid molecules

    # Filter legends to match valid molecules if provided
    if legends and valid_indices:
        if len(legends) != len(molecular_strings):
            logging.warning(
                f"Number of legends ({len(legends)}) doesn't match number of molecules ({len(molecular_strings)})"
            )
        # Filter legends to only include those for valid molecules
        filtered_legends = [legends[i] for i in valid_indices if i < len(legends)]
        # If we still have a mismatch, set legends to None
        if len(filtered_legends) != len(mols):
            logging.warning("Legend count mismatch after filtering, disabling legends")
            legends = None
        else:
            legends = filtered_legends

    if not mols:
        raise ValueError("No valid molecules could be parsed")

    # Create grid renderer
    from molecular_string_renderer.renderers import MoleculeGridRenderer

    grid_renderer = MoleculeGridRenderer(
        config=render_config, mols_per_row=mols_per_row, mol_size=mol_size
    )

    # Render grid
    image = grid_renderer.render_grid(mols, legends)

    # Save if output path provided
    if output_path:
        output_handler = get_output_handler(output_format, output_config)
        output_handler.save(image, output_path)

    return image


def validate_molecular_string(
    molecular_string: str, format_type: str = "smiles"
) -> bool:
    """
    Validate if a molecular string is valid for the given format.

    Args:
        molecular_string: String to validate
        format_type: Format type to validate against

    Returns:
        True if valid, False otherwise
    """
    try:
        parser = get_parser(format_type)
        return parser.validate(molecular_string)
    except Exception:
        return False


def get_supported_formats() -> dict[str, dict[str, str]]:
    """
    Get information about supported input and output formats.

    Returns:
        Dictionary with supported formats and their descriptions
    """
    return {
        "input_formats": {
            "smiles": "Simplified Molecular Input Line Entry System",
            "smi": "SMILES (alternative extension)",
            "inchi": "International Chemical Identifier",
            "mol": "MOL file format",
            "sdf": "Structure Data File format",
            "selfies": "Self-Referencing Embedded Strings",
        },
        "output_formats": {
            "png": "Portable Network Graphics (recommended)",
            "svg": "Scalable Vector Graphics (true vector format)",
            "jpg": "JPEG image format",
            "jpeg": "JPEG image format (alternative extension)",
            "pdf": "Portable Document Format",
            "webp": "WebP image format (modern, efficient compression)",
            "tiff": "Tagged Image File Format (high quality, supports transparency)",
            "tif": "TIFF image format (alternative extension)",
            "bmp": "Bitmap image format (uncompressed)",
        },
        "renderer_types": {
            "2d": "2D molecular structure rendering",
            "grid": "Grid layout for multiple molecules",
        },
    }


class MolecularRenderer:
    """
    High-level class interface for molecular rendering.

    Provides an object-oriented interface that maintains configuration
    across multiple rendering operations.
    """

    def __init__(
        self,
        render_config: RenderConfig | None = None,
        parser_config: ParserConfig | None = None,
        output_config: OutputConfig | None = None,
    ):
        """
        Initialize molecular renderer with configurations.

        Args:
            render_config: Configuration for rendering
            parser_config: Configuration for parsing
            output_config: Configuration for output
        """
        self.render_config = render_config or RenderConfig()
        self.parser_config = parser_config or ParserConfig()
        self.output_config = output_config or OutputConfig()

        # Cache parsers and renderers for efficiency
        self._parsers = {}
        self._renderers = {}
        self._output_handlers = {}

    def render(
        self,
        molecular_string: str,
        format_type: str = "smiles",
        output_format: str = "png",
        output_path: str | Path | None = None,
    ) -> Image.Image:
        """
        Render a molecular string using the configured settings.

        Args:
            molecular_string: Molecular string to render
            format_type: Input format type
            output_format: Output format type
            output_path: Optional output path

        Returns:
            PIL Image object
        """
        return render_molecule(
            molecular_string=molecular_string,
            format_type=format_type,
            output_format=output_format,
            output_path=output_path,
            render_config=self.render_config,
            parser_config=self.parser_config,
            output_config=self.output_config,
            auto_filename=False,
        )

    def render_grid(
        self,
        molecular_strings: list[str],
        format_type: str = "smiles",
        output_format: str = "png",
        output_path: str | Path | None = None,
        legends: list[str] | None = None,
        mols_per_row: int = 4,
    ) -> Image.Image:
        """
        Render multiple molecules in a grid.

        Args:
            molecular_strings: List of molecular strings
            format_type: Input format type
            output_format: Output format type
            output_path: Optional output path
            legends: Optional legends
            mols_per_row: Molecules per row

        Returns:
            PIL Image object
        """
        return render_molecules_grid(
            molecular_strings=molecular_strings,
            format_type=format_type,
            output_format=output_format,
            output_path=output_path,
            legends=legends,
            mols_per_row=mols_per_row,
            render_config=self.render_config,
            parser_config=self.parser_config,
            output_config=self.output_config,
        )

    def update_config(
        self,
        render_config: RenderConfig | None = None,
        parser_config: ParserConfig | None = None,
        output_config: OutputConfig | None = None,
    ) -> None:
        """
        Update renderer configurations.

        Args:
            render_config: New render configuration
            parser_config: New parser configuration
            output_config: New output configuration
        """
        if render_config:
            self.render_config = render_config
        if parser_config:
            self.parser_config = parser_config
        if output_config:
            self.output_config = output_config

        # Clear caches when config changes
        self._parsers.clear()
        self._renderers.clear()
        self._output_handlers.clear()
