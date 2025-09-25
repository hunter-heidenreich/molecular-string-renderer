"""
Core interface for molecular string rendering.

Provides high-level functions that combine parsing, rendering, and output.
"""

import logging
from pathlib import Path

from PIL import Image

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.config_utils import initialize_configurations
from molecular_string_renderer.exceptions import ParsingError, RenderingError
from molecular_string_renderer.logging_utils import logged_operation
from molecular_string_renderer.parsers import get_parser
from molecular_string_renderer.pipeline import RenderingPipeline
from molecular_string_renderer.renderers import MoleculeGridRenderer
from molecular_string_renderer.utils import (
    ERROR_TEMPLATES,
    filter_legends_by_indices,
    format_parsing_errors,
    parse_molecule_list,
    validate_and_normalize_inputs,
)
from molecular_string_renderer.validation import (
    validate_format_type,
    validate_grid_parameters,
)
from molecular_string_renderer.validation import (
    validate_molecular_string as validate_mol_string,
)

# Set up module-level logger
logger = logging.getLogger(__name__)


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

    This is the primary entry point for converting molecular representations
    (SMILES, InChI, etc.) into visual images. Supports all major molecular
    formats and output types with extensive configuration options.

    Args:
        molecular_string: String representation of molecule (max 10,000 chars).
                         Examples: 'CCO' (ethanol), 'InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3'
        format_type: Molecular format type. Supported: 'smiles', 'smi', 'inchi',
                    'mol', 'sdf', 'selfies'. Default: 'smiles'
        output_format: Output image format. Supported: 'png', 'svg', 'jpg', 'jpeg',
                      'pdf', 'webp', 'tiff', 'tif', 'bmp'. Default: 'png'
        output_path: Path to save image. If None and auto_filename=True, generates
                    safe filename. If None and auto_filename=False, no file saved.
                    If provided, auto_filename is automatically set to False
        render_config: Rendering options (size, colors, atom display). If None,
                      uses RenderConfig() defaults (500x500px, white background)
        parser_config: Parsing options (sanitization, hydrogen handling). If None,
                      uses ParserConfig() defaults (sanitize=True, strict=False)
        output_config: Output options (quality, DPI, compression). If None,
                      uses OutputConfig() defaults (quality=95, DPI=150)
        auto_filename: Generate safe filename if output_path=None. Default: True.
                      Automatically set to False if output_path is provided

    Returns:
        PIL Image object containing the rendered molecule. Always RGBA mode for
        PNG/TIFF, RGB for JPEG/WebP/BMP

    Raises:
        ValidationError: If molecular_string is empty, too long, wrong type, or
                        format_type is unsupported
        ParsingError: If molecular string cannot be parsed by RDKit. Common causes:
                     - Invalid SMILES syntax: 'invalid_smiles'
                     - Malformed InChI: 'InChI=invalid'
                     - Empty or corrupted MOL files
        RenderingError: If molecule cannot be rendered to image. Rare, usually
                       indicates RDKit internal issues
        OutputError: If image cannot be saved. Common causes:
                    - Permission denied to output directory
                    - Disk full or filesystem errors
                    - Invalid output path format

    Example:
        Basic usage:
        >>> image = render_molecule('CCO')  # Renders ethanol as PNG
        >>> type(image)
        <class 'PIL.Image.Image'>

        Custom configuration:
        >>> from molecular_string_renderer.config import RenderConfig
        >>> config = RenderConfig(width=800, height=600, show_hydrogen=True)
        >>> image = render_molecule(
        ...     'CCO',
        ...     format_type='smiles',
        ...     output_format='svg',
        ...     output_path='ethanol.svg',
        ...     render_config=config
        ... )

        Batch processing:
        >>> molecules = ['CCO', 'CC(=O)O', 'c1ccccc1']
        >>> images = [render_molecule(mol) for mol in molecules]
    """
    # Start performance timing
    with logged_operation(
        "render_molecule",
        {
            "format": format_type,
            "output": output_format,
            "length": len(molecular_string) if molecular_string else 0,
        },
    ):
        # Initialize configurations with defaults
        render_config, parser_config, output_config = initialize_configurations(
            render_config, parser_config, output_config, output_format
        )

        # Create pipeline for cleaner operation flow
        pipeline = RenderingPipeline(render_config, parser_config, output_config)

        # Validate inputs
        _, format_type, output_path = pipeline.validate_inputs(
            molecular_string, format_type, output_path
        )

        # Override auto_filename if output_path is provided
        if output_path is not None:
            auto_filename = False

        # Parse and render molecule
        mol = pipeline.parse_molecule(molecular_string, format_type)
        image = pipeline.render_molecule(mol)

        # Save output if requested
        pipeline.save_output(
            image=image,
            output_path=output_path,
            output_format=output_format,
            mol=mol,
            auto_filename=auto_filename,
            molecular_string=molecular_string,
        )

        logger.info(
            f"Successfully rendered {format_type.upper()} to {output_format.upper()}"
        )
        return image


def render_molecules_grid(
    molecular_strings: list[str],
    format_type: str = "smiles",
    output_format: str = "png",
    output_path: str | Path | None = None,
    legends: list[str] | None = None,
    mols_per_row: int | None = None,
    mol_size: tuple[int, int] = (200, 200),
    render_config: RenderConfig | None = None,
    parser_config: ParserConfig | None = None,
    output_config: OutputConfig | None = None,
    auto_filename: bool = True,
) -> Image.Image:
    """
    Render multiple molecules in a grid layout.

    Args:
        molecular_strings: List of molecular string representations
        format_type: Type of molecular format ('smiles', 'inchi', 'mol')
        output_format: Output image format ('png', 'svg', 'jpg')
        output_path: Path to save image (optional)
        legends: Optional legends for each molecule
        mols_per_row: Number of molecules per row (default: auto-fits to molecule count, max 4)
        mol_size: Size of each molecule image (width, height)
        render_config: Configuration for rendering
        parser_config: Configuration for parsing
        output_config: Configuration for output
        auto_filename: Generate safe filename if output_path=None. Default: True.
                      Automatically set to False if output_path is provided

    Returns:
        PIL Image object containing the molecule grid

    Raises:
        ValidationError: If input parameters are invalid
        ParsingError: If molecular strings cannot be parsed
        RenderingError: If molecules cannot be rendered
        OutputError: If image cannot be saved
    """
    # Smart default for mols_per_row: auto-fit to molecule count with max of 4
    if mols_per_row is None:
        mols_per_row = min(len(molecular_strings), 4)
        logger.debug(
            f"Using smart default mols_per_row={mols_per_row} for {len(molecular_strings)} molecules"
        )

    # Validate inputs
    validate_grid_parameters(molecular_strings, mols_per_row, mol_size)
    _, format_type, output_path = validate_and_normalize_inputs(
        format_type=format_type, output_path=output_path
    )

    # Override auto_filename if output_path is provided
    if output_path is not None:
        auto_filename = False

    # Validate legends if provided
    if legends is not None:
        if not isinstance(legends, list):
            from molecular_string_renderer.exceptions import ValidationError

            raise ValidationError(
                f"Legends must be a list, got {type(legends).__name__}"
            )
        if len(legends) != len(molecular_strings):
            from molecular_string_renderer.exceptions import ValidationError

            raise ValidationError(
                f"Number of legends ({len(legends)}) must match number of molecules ({len(molecular_strings)})"
            )

    # Initialize configurations
    render_config, parser_config, output_config = initialize_configurations(
        render_config, parser_config, output_config, output_format
    )

    # Parse all molecules
    mols, valid_indices, parsing_errors = parse_molecule_list(
        molecular_strings, format_type, parser_config
    )

    if not mols:
        error_details = format_parsing_errors(parsing_errors)
        raise ParsingError(
            ERROR_TEMPLATES["no_valid_molecules"].format(details=error_details)
        )

    # Filter legends to match valid molecules if provided
    if legends and valid_indices:
        legends = filter_legends_by_indices(
            legends, valid_indices, len(molecular_strings)
        )

    # Create and render grid
    with logged_operation("grid_render"):
        # Create grid renderer
        grid_renderer = MoleculeGridRenderer(
            config=render_config, mols_per_row=mols_per_row, mol_size=mol_size
        )

        # Render grid
        image = grid_renderer.render_grid(mols, legends)
        if image is None:
            raise RenderingError(ERROR_TEMPLATES["grid_renderer_none"])

    # Save if output path provided or auto_filename enabled
    if output_path or auto_filename:
        with logged_operation("grid_save"):
            from molecular_string_renderer.utils import handle_output_saving

            # Generate a representative string for the grid for auto filename
            grid_string = (
                f"grid_{len(molecular_strings)}_molecules" if auto_filename else None
            )

            handle_output_saving(
                image=image,
                output_path=output_path,
                output_format=output_format,
                output_config=output_config,
                auto_filename=auto_filename,
                molecular_string=grid_string,
            )

    return image


def validate_molecular_string(
    molecular_string: str, format_type: str = "smiles"
) -> bool:
    """
    Validate if a molecular string is valid for the given format.

    This function performs lightweight validation without full parsing,
    making it suitable for batch validation or input checking scenarios.
    It combines format validation, basic string checks, and parser validation.

    Args:
        molecular_string: String to validate. Must not be empty or exceed 10,000 characters
        format_type: Format type to validate against. Supported: 'smiles', 'smi',
                    'inchi', 'mol', 'sdf', 'selfies'. Default: 'smiles'

    Returns:
        True if the molecular string is valid for the specified format,
        False if parsing fails or the string is malformed

    Raises:
        ValidationError: If input parameters are invalid (wrong types,
                        unsupported format, empty string, etc.)

    Example:
        Basic validation:
        >>> validate_molecular_string('CCO', 'smiles')  # ethanol
        True
        >>> validate_molecular_string('invalid_smiles', 'smiles')
        False

        Format validation:
        >>> validate_molecular_string('InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3', 'inchi')
        True
        >>> validate_molecular_string('CCO', 'inchi')  # SMILES string as InChI
        False

        Batch validation:
        >>> molecules = ['CCO', 'CC(=O)O', 'invalid', 'c1ccccc1']
        >>> valid_mols = [mol for mol in molecules if validate_molecular_string(mol)]
        >>> len(valid_mols)
        3
    """
    try:
        validate_mol_string(molecular_string, format_type)
        format_type = validate_format_type(
            format_type, {"smiles", "smi", "inchi", "mol", "sdf", "selfies"}
        )
        parser = get_parser(format_type)
        return parser.validate(molecular_string)
    except Exception:
        from molecular_string_renderer.exceptions import ValidationError

        try:
            validate_mol_string(molecular_string, format_type)
            format_type = validate_format_type(
                format_type, {"smiles", "smi", "inchi", "mol", "sdf", "selfies"}
            )
        except ValidationError:
            raise  # Re-raise validation errors
        except Exception:
            return False
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
            "gif": "Graphics Interchange Format (limited color, not recommended)",
        },
        "renderer_types": {
            "2d": "2D molecular structure rendering",
            "grid": "Grid layout for multiple molecules",
        },
    }
