"""
Core interface for molecular string rendering.

Provides high-level functions that combine parsing, rendering, and output.
"""

import logging
import time
from pathlib import Path

from PIL import Image

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.outputs import create_safe_filename, get_output_handler
from molecular_string_renderer.parsers import get_parser
from molecular_string_renderer.renderers import get_renderer

# Set up module-level logger
logger = logging.getLogger(__name__)


class MolecularRenderingError(Exception):
    """Base exception for molecular rendering operations."""

    pass


class ParsingError(MolecularRenderingError):
    """Exception raised when molecular string parsing fails."""

    pass


class RenderingError(MolecularRenderingError):
    """Exception raised when molecule rendering fails."""

    pass


class OutputError(MolecularRenderingError):
    """Exception raised when output generation fails."""

    pass


class ValidationError(MolecularRenderingError):
    """Exception raised when input validation fails."""

    pass


class ConfigurationError(MolecularRenderingError):
    """Exception raised when configuration is invalid."""

    pass


def _log_operation_start(
    operation: str, details: dict[str, str | int] | None = None
) -> float:
    """
    Log the start of a rendering operation and return start time.

    Args:
        operation: Name of the operation (e.g., 'render_molecule', 'parse')
        details: Optional dictionary of operation details to log

    Returns:
        Start time for performance measurement
    """
    start_time = time.perf_counter()
    detail_str = ""
    if details:
        detail_str = " | " + " | ".join(f"{k}={v}" for k, v in details.items())
    logger.debug(f"Starting {operation}{detail_str}")
    return start_time


def _log_operation_end(operation: str, start_time: float, success: bool = True) -> None:
    """
    Log the end of a rendering operation with timing.

    Args:
        operation: Name of the operation that completed
        start_time: Start time from _log_operation_start
        success: Whether the operation succeeded
    """
    duration = time.perf_counter() - start_time
    status = "completed" if success else "failed"
    logger.debug(f"Operation {operation} {status} in {duration:.3f}s")
    if duration > 5.0:  # Log slow operations at info level
        logger.info(f"Slow operation: {operation} took {duration:.3f}s")


def _validate_molecular_string(molecular_string: str, format_type: str) -> None:
    """
    Validate molecular string input parameters.

    Args:
        molecular_string: The molecular string to validate
        format_type: The format type to validate against

    Raises:
        ValidationError: If molecular string is invalid
    """
    if not isinstance(molecular_string, str):
        raise ValidationError(
            f"Molecular string must be a string, got {type(molecular_string).__name__}"
        )

    if not molecular_string or not molecular_string.strip():
        raise ValidationError(f"Empty {format_type.upper()} string provided")

    if len(molecular_string.strip()) > 10000:  # Reasonable limit
        raise ValidationError(
            f"Molecular string too long ({len(molecular_string)} characters). "
            "Maximum supported length is 10,000 characters"
        )


def _validate_format_type(format_type: str, valid_formats: set[str]) -> str:
    """
    Validate and normalize format type.

    Args:
        format_type: The format type to validate
        valid_formats: Set of valid format types

    Returns:
        Normalized format type (lowercase)

    Raises:
        ValidationError: If format type is invalid
    """
    if not isinstance(format_type, str):
        raise ValidationError(
            f"Format type must be a string, got {type(format_type).__name__}"
        )

    normalized = format_type.lower().strip()
    if normalized not in valid_formats:
        raise ValidationError(
            f"Unsupported format type: '{format_type}'. "
            f"Supported formats: {sorted(valid_formats)}"
        )

    return normalized


def _validate_grid_parameters(
    molecular_strings: list[str], mols_per_row: int, mol_size: tuple[int, int]
) -> None:
    """
    Validate grid rendering parameters.

    Args:
        molecular_strings: List of molecular strings
        mols_per_row: Number of molecules per row
        mol_size: Size of each molecule image

    Raises:
        ValidationError: If any parameter is invalid
    """
    if not isinstance(molecular_strings, list):
        raise ValidationError(
            f"Molecular strings must be a list, got {type(molecular_strings).__name__}"
        )

    if not molecular_strings:
        raise ValidationError("Cannot render empty molecule list")

    if len(molecular_strings) > 100:  # Reasonable limit for grid
        raise ValidationError(
            f"Too many molecules for grid ({len(molecular_strings)}). "
            "Maximum supported is 100 molecules"
        )

    if not isinstance(mols_per_row, int) or mols_per_row < 1:
        raise ValidationError(
            f"mols_per_row must be a positive integer, got {mols_per_row}"
        )

    if not isinstance(mol_size, tuple) or len(mol_size) != 2:
        raise ValidationError(
            f"mol_size must be a tuple of (width, height), got {mol_size}"
        )

    width, height = mol_size
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValidationError("mol_size width and height must be integers")

    if width < 50 or height < 50 or width > 1000 or height > 1000:
        raise ValidationError(
            f"mol_size dimensions must be between 50 and 1000 pixels, got {mol_size}"
        )


def _validate_output_path(output_path: str | Path | None) -> Path | None:
    """
    Validate and normalize output path.

    Args:
        output_path: Output path to validate

    Returns:
        Normalized Path object or None

    Raises:
        ValidationError: If path is invalid
    """
    if output_path is None:
        return None

    if isinstance(output_path, str):
        if not output_path.strip():
            raise ValidationError("Output path cannot be empty string")
        output_path = Path(output_path.strip())
    elif not isinstance(output_path, Path):
        raise ValidationError(
            f"Output path must be string or Path, got {type(output_path).__name__}"
        )

    # Check if parent directory exists or can be created
    parent = output_path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValidationError(
                f"Cannot create output directory '{parent}': {e}"
            ) from e

    return output_path


def _validate_configuration_compatibility(
    render_config: RenderConfig,
    parser_config: ParserConfig,
    output_config: OutputConfig,
) -> None:
    """
    Validate that configurations are compatible and make sense together.

    Args:
        render_config: Render configuration to validate
        parser_config: Parser configuration to validate
        output_config: Output configuration to validate

    Raises:
        ConfigurationError: If configurations have conflicts or invalid combinations

    Example:
        >>> render_cfg = RenderConfig(width=100, height=200)
        >>> parser_cfg = ParserConfig()
        >>> output_cfg = OutputConfig(format='svg')
        >>> _validate_configuration_compatibility(render_cfg, parser_cfg, output_cfg)
        # Passes validation
    """
    # Check for dimension conflicts
    if render_config.width < 50 or render_config.height < 50:
        raise ConfigurationError(
            f"Render dimensions too small: {render_config.width}x{render_config.height}. "
            "Minimum is 50x50 pixels for readable output"
        )

    if render_config.width > 5000 or render_config.height > 5000:
        raise ConfigurationError(
            f"Render dimensions too large: {render_config.width}x{render_config.height}. "
            "Maximum is 5000x5000 pixels to prevent memory issues"
        )

    # Check aspect ratio warnings (not errors, just log)
    aspect_ratio = render_config.width / render_config.height
    if aspect_ratio > 10 or aspect_ratio < 0.1:
        logging.warning(
            f"Extreme aspect ratio {aspect_ratio:.2f} may result in distorted molecules"
        )

    # Validate color format
    if render_config.background_color.startswith("#"):
        try:
            int(render_config.background_color[1:], 16)
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid hex color '{render_config.background_color}': {e}"
            ) from e

    # Check output format compatibility
    if (
        output_config.format.lower() == "jpeg"
        and render_config.background_color == "transparent"
    ):
        raise ConfigurationError(
            "JPEG format does not support transparent backgrounds. "
            "Use PNG, WebP, or TIFF for transparency, or choose a solid background color"
        )

    # Validate quality settings for specific formats
    if output_config.format.lower() in ["jpeg", "webp"] and output_config.quality > 100:
        raise ConfigurationError(
            f"Quality {output_config.quality} invalid for {output_config.format.upper()}. "
            "Must be 1-100"
        )

    # Check DPI reasonableness
    if output_config.dpi < 72:
        logging.warning(
            f"Low DPI ({output_config.dpi}) may result in poor print quality"
        )
    elif output_config.dpi > 600:
        logging.warning(f"Very high DPI ({output_config.dpi}) will create large files")

    # SVG-specific validations
    if output_config.format.lower() == "svg":
        if output_config.svg_line_width_mult <= 0:
            raise ConfigurationError(
                f"SVG line width multiplier must be positive, got {output_config.svg_line_width_mult}"
            )

    # Coordination checks
    if render_config.show_hydrogen and not parser_config.show_hydrogen:
        logging.info(
            "Auto-coordinating: render_config.show_hydrogen=True requires parser_config.show_hydrogen=True"
        )


def _initialize_configurations(
    render_config: RenderConfig | None,
    parser_config: ParserConfig | None,
    output_config: OutputConfig | None,
    output_format: str,
) -> tuple[RenderConfig, ParserConfig, OutputConfig]:
    """
    Initialize and coordinate configuration objects with defaults.

    Ensures configuration consistency by auto-coordinating hydrogen display
    settings between render and parser configurations.

    Args:
        render_config: Optional render configuration. If None, uses RenderConfig()
        parser_config: Optional parser configuration. If None, uses ParserConfig()
        output_config: Optional output configuration. If None, uses OutputConfig()
        output_format: Output format for default output config (e.g., 'png', 'svg')

    Returns:
        Tuple of (render_config, parser_config, output_config) with proper coordination

    Raises:
        ConfigurationError: If configurations are incompatible

    Example:
        >>> render_cfg = RenderConfig(show_hydrogen=True)
        >>> render_cfg, parser_cfg, output_cfg = _initialize_configurations(
        ...     render_cfg, None, None, "png"
        ... )
        >>> parser_cfg.show_hydrogen  # Will be True to match render config
        True
    """
    render_config = render_config or RenderConfig()
    parser_config = parser_config or ParserConfig()
    output_config = output_config or OutputConfig(format=output_format)

    # Auto-coordinate hydrogen display settings
    if render_config.show_hydrogen and not parser_config.show_hydrogen:
        parser_config = ParserConfig(
            sanitize=parser_config.sanitize,
            show_hydrogen=True,  # Keep hydrogens for display
            strict=parser_config.strict,
        )

    # Validate compatibility
    _validate_configuration_compatibility(render_config, parser_config, output_config)

    return render_config, parser_config, output_config


def _handle_output_saving(
    image: Image.Image,
    output_path: str | Path | None,
    output_format: str,
    output_config: OutputConfig,
    mol=None,
    auto_filename: bool = False,
    molecular_string: str | None = None,
) -> None:
    """
    Handle saving of rendered images with consistent logic.

    Provides centralized output handling for both single molecules and grids,
    with special handling for SVG vector rendering and automatic filename generation.

    Args:
        image: PIL Image to save. Must be a valid PIL Image object
        output_path: Optional output path. If None, no saving occurs unless auto_filename=True
        output_format: Output format (e.g., 'png', 'svg', 'jpg'). Must be supported format
        output_config: Output configuration with quality, DPI, and format-specific settings
        mol: Optional molecule object for SVG vector rendering. Required for true vector SVG
        auto_filename: Whether to generate automatic filename if output_path is None
        molecular_string: Source string for auto filename generation. Required if auto_filename=True

    Raises:
        OutputError: If saving fails due to file system issues or invalid parameters

    Example:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 100), 'white')
        >>> config = OutputConfig(format='png', quality=95)
        >>> _handle_output_saving(img, 'molecule.png', 'png', config)
        # Saves image to molecule.png with specified quality
    """
    if output_path or auto_filename:
        output_handler = get_output_handler(output_format, output_config)

        # For SVG output, provide the molecule for true vector rendering
        if output_format.lower() == "svg" and mol is not None:
            output_handler.set_molecule(mol)

        if not output_path and auto_filename and molecular_string:
            # Generate safe filename
            base_name = create_safe_filename(
                molecular_string, output_handler.file_extension
            )
            output_path = Path.cwd() / base_name

        if output_path:
            output_handler.save(image, output_path)


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
                    safe filename. If None and auto_filename=False, no file saved
        render_config: Rendering options (size, colors, atom display). If None,
                      uses RenderConfig() defaults (500x500px, white background)
        parser_config: Parsing options (sanitization, hydrogen handling). If None,
                      uses ParserConfig() defaults (sanitize=True, strict=False)
        output_config: Output options (quality, DPI, compression). If None,
                      uses OutputConfig() defaults (quality=95, DPI=150)
        auto_filename: Generate safe filename if output_path=None. Default: True

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
    operation_start = _log_operation_start(
        "render_molecule",
        {
            "format": format_type,
            "output": output_format,
            "length": len(molecular_string) if molecular_string else 0,
        },
    )

    try:
        # Validate inputs
        _validate_molecular_string(molecular_string, format_type)
        format_type = _validate_format_type(
            format_type, {"smiles", "smi", "inchi", "mol", "sdf", "selfies"}
        )
        output_path = _validate_output_path(output_path)

        # Initialize configurations with defaults
        render_config, parser_config, output_config = _initialize_configurations(
            render_config, parser_config, output_config, output_format
        )

        # Parse the molecular string
        parse_start = _log_operation_start("parse", {"format": format_type})
        try:
            parser = get_parser(format_type, parser_config)
            mol = parser.parse(molecular_string)
            if mol is None:
                raise ParsingError(
                    f"Failed to parse {format_type.upper()}: '{molecular_string}'"
                )
            _log_operation_end("parse", parse_start, True)
        except Exception as e:
            _log_operation_end("parse", parse_start, False)
            if isinstance(e, ParsingError):
                raise
            raise ParsingError(
                f"Error parsing {format_type.upper()} '{molecular_string}': {e}"
            ) from e

        # Render the molecule
        render_start = _log_operation_start(
            "render", {"size": f"{render_config.width}x{render_config.height}"}
        )
        try:
            renderer = get_renderer("2d", render_config)
            image = renderer.render(mol)
            if image is None:
                raise RenderingError("Renderer returned None image")
            _log_operation_end("render", render_start, True)
        except Exception as e:
            _log_operation_end("render", render_start, False)
            if isinstance(e, RenderingError):
                raise
            raise RenderingError(f"Error rendering molecule: {e}") from e

        # Save if output path is provided or auto_filename is enabled
        if output_path or auto_filename:
            save_start = _log_operation_start("save", {"format": output_format})
            try:
                _handle_output_saving(
                    image=image,
                    output_path=output_path,
                    output_format=output_format,
                    output_config=output_config,
                    mol=mol,
                    auto_filename=auto_filename,
                    molecular_string=molecular_string,
                )
                _log_operation_end("save", save_start, True)
            except Exception as e:
                _log_operation_end("save", save_start, False)
                if isinstance(e, OutputError):
                    raise
                raise OutputError(f"Error saving output: {e}") from e

        _log_operation_end("render_molecule", operation_start, True)
        logger.info(
            f"Successfully rendered {format_type.upper()} to {output_format.upper()}"
        )
        return image

    except Exception as e:
        _log_operation_end("render_molecule", operation_start, False)
        logger.error(f"Failed to render molecule: {e}")
        raise


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

    Raises:
        ValidationError: If input parameters are invalid
        ParsingError: If molecular strings cannot be parsed
        RenderingError: If molecules cannot be rendered
        OutputError: If image cannot be saved
    """
    # Validate inputs
    _validate_grid_parameters(molecular_strings, mols_per_row, mol_size)
    format_type = _validate_format_type(
        format_type, {"smiles", "smi", "inchi", "mol", "sdf", "selfies"}
    )
    output_path = _validate_output_path(output_path)

    # Validate legends if provided
    if legends is not None:
        if not isinstance(legends, list):
            raise ValidationError(
                f"Legends must be a list, got {type(legends).__name__}"
            )
        if len(legends) != len(molecular_strings):
            raise ValidationError(
                f"Number of legends ({len(legends)}) must match number of molecules ({len(molecular_strings)})"
            )

    # Initialize configurations
    render_config, parser_config, output_config = _initialize_configurations(
        render_config, parser_config, output_config, output_format
    )

    # Parse all molecules
    parser = get_parser(format_type, parser_config)
    mols = []
    valid_indices = []  # Track which molecules were successfully parsed
    parsing_errors = []

    for i, mol_string in enumerate(molecular_strings):
        try:
            _validate_molecular_string(mol_string, format_type)
            mol = parser.parse(mol_string)
            if mol is None:
                parsing_errors.append(f"Index {i}: Failed to parse '{mol_string}'")
                continue
            mols.append(mol)
            valid_indices.append(i)
        except Exception as e:
            parsing_errors.append(f"Index {i}: {e}")
            logging.warning(
                f"Failed to parse molecule at index {i} ('{mol_string}'): {e}"
            )

    if not mols:
        error_details = "; ".join(parsing_errors[:5])  # Show first 5 errors
        if len(parsing_errors) > 5:
            error_details += f" (and {len(parsing_errors) - 5} more errors)"
        raise ParsingError(
            f"No valid molecules could be parsed. Errors: {error_details}"
        )

    # Filter legends to match valid molecules if provided
    if legends and valid_indices:
        filtered_legends = [legends[i] for i in valid_indices if i < len(legends)]
        if len(filtered_legends) != len(mols):
            logging.warning("Legend count mismatch after filtering, disabling legends")
            legends = None
        else:
            legends = filtered_legends

    try:
        # Create grid renderer
        from molecular_string_renderer.renderers import MoleculeGridRenderer

        grid_renderer = MoleculeGridRenderer(
            config=render_config, mols_per_row=mols_per_row, mol_size=mol_size
        )

        # Render grid
        image = grid_renderer.render_grid(mols, legends)
        if image is None:
            raise RenderingError("Grid renderer returned None image")
    except Exception as e:
        if isinstance(e, RenderingError):
            raise
        raise RenderingError(f"Error rendering molecule grid: {e}") from e

    try:
        # Save if output path provided
        _handle_output_saving(
            image=image,
            output_path=output_path,
            output_format=output_format,
            output_config=output_config,
        )
    except Exception as e:
        if isinstance(e, OutputError):
            raise
        raise OutputError(f"Error saving grid output: {e}") from e

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

    Raises:
        ValidationError: If input parameters are invalid
    """
    try:
        _validate_molecular_string(molecular_string, format_type)
        format_type = _validate_format_type(
            format_type, {"smiles", "smi", "inchi", "mol", "sdf", "selfies"}
        )
        parser = get_parser(format_type)
        return parser.validate(molecular_string)
    except ValidationError:
        raise  # Re-raise validation errors
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
    across multiple rendering operations with caching for improved performance.
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

        Raises:
            ValidationError: If configurations are invalid
        """
        self.render_config = render_config or RenderConfig()
        self.parser_config = parser_config or ParserConfig()
        self.output_config = output_config or OutputConfig()

        # Validate configurations
        try:
            # Test that configurations are valid by attempting to convert to dict
            self.render_config.model_dump()
            self.parser_config.model_dump()
            self.output_config.model_dump()
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}") from e

        # Cache for parsers, renderers, and output handlers
        self._parsers: dict[str, object] = {}
        self._renderers: dict[str, object] = {}
        self._output_handlers: dict[str, object] = {}

        # Performance tracking
        self._operation_count = 0
        self._cache_hits = 0

    def _get_cached_parser(self, format_type: str):
        """Get cached parser or create new one."""
        cache_key = f"{format_type}_{hash(str(self.parser_config.model_dump()))}"
        if cache_key not in self._parsers:
            self._parsers[cache_key] = get_parser(format_type, self.parser_config)
        else:
            self._cache_hits += 1
        return self._parsers[cache_key]

    def _get_cached_renderer(self, renderer_type: str):
        """Get cached renderer or create new one."""
        cache_key = f"{renderer_type}_{hash(str(self.render_config.model_dump()))}"
        if cache_key not in self._renderers:
            self._renderers[cache_key] = get_renderer(renderer_type, self.render_config)
        else:
            self._cache_hits += 1
        return self._renderers[cache_key]

    def _get_cached_output_handler(self, format_type: str):
        """Get cached output handler or create new one."""
        cache_key = f"{format_type}_{hash(str(self.output_config.model_dump()))}"
        if cache_key not in self._output_handlers:
            self._output_handlers[cache_key] = get_output_handler(
                format_type, self.output_config
            )
        else:
            self._cache_hits += 1
        return self._output_handlers[cache_key]

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

        Raises:
            ValidationError: If input parameters are invalid
            ParsingError: If molecular string cannot be parsed
            RenderingError: If molecule cannot be rendered
            OutputError: If image cannot be saved
        """
        self._operation_count += 1

        # Validate inputs
        _validate_molecular_string(molecular_string, format_type)
        format_type = _validate_format_type(
            format_type, {"smiles", "smi", "inchi", "mol", "sdf", "selfies"}
        )
        output_path = _validate_output_path(output_path)

        try:
            # Use cached parser
            parser = self._get_cached_parser(format_type)
            mol = parser.parse(molecular_string)
            if mol is None:
                raise ParsingError(
                    f"Failed to parse {format_type.upper()}: '{molecular_string}'"
                )
        except Exception as e:
            if isinstance(e, ParsingError):
                raise
            raise ParsingError(
                f"Error parsing {format_type.upper()} '{molecular_string}': {e}"
            ) from e

        try:
            # Use cached renderer
            renderer = self._get_cached_renderer("2d")
            image = renderer.render(mol)
            if image is None:
                raise RenderingError("Renderer returned None image")
        except Exception as e:
            if isinstance(e, RenderingError):
                raise
            raise RenderingError(f"Error rendering molecule: {e}") from e

        try:
            # Save if output path provided
            if output_path:
                output_handler = self._get_cached_output_handler(output_format)
                if output_format.lower() == "svg" and mol is not None:
                    output_handler.set_molecule(mol)
                output_handler.save(image, output_path)
        except Exception as e:
            if isinstance(e, OutputError):
                raise
            raise OutputError(f"Error saving output: {e}") from e

        return image

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

        Raises:
            ValidationError: If input parameters are invalid
            ParsingError: If molecular strings cannot be parsed
            RenderingError: If molecules cannot be rendered
            OutputError: If image cannot be saved
        """
        self._operation_count += 1

        # Use standard mol_size for cached renderer class
        mol_size = (200, 200)

        # Validate inputs
        _validate_grid_parameters(molecular_strings, mols_per_row, mol_size)
        format_type = _validate_format_type(
            format_type, {"smiles", "smi", "inchi", "mol", "sdf", "selfies"}
        )
        output_path = _validate_output_path(output_path)

        if legends is not None:
            if not isinstance(legends, list):
                raise ValidationError(
                    f"Legends must be a list, got {type(legends).__name__}"
                )
            if len(legends) != len(molecular_strings):
                raise ValidationError(
                    f"Number of legends ({len(legends)}) must match number of molecules ({len(molecular_strings)})"
                )

        # Parse molecules using cached parser
        parser = self._get_cached_parser(format_type)
        mols = []
        valid_indices = []
        parsing_errors = []

        for i, mol_string in enumerate(molecular_strings):
            try:
                _validate_molecular_string(mol_string, format_type)
                mol = parser.parse(mol_string)
                if mol is None:
                    parsing_errors.append(f"Index {i}: Failed to parse '{mol_string}'")
                    continue
                mols.append(mol)
                valid_indices.append(i)
            except Exception as e:
                parsing_errors.append(f"Index {i}: {e}")
                logging.warning(
                    f"Failed to parse molecule at index {i} ('{mol_string}'): {e}"
                )

        if not mols:
            error_details = "; ".join(parsing_errors[:5])
            if len(parsing_errors) > 5:
                error_details += f" (and {len(parsing_errors) - 5} more errors)"
            raise ParsingError(
                f"No valid molecules could be parsed. Errors: {error_details}"
            )

        # Filter legends if needed
        if legends and valid_indices:
            filtered_legends = [legends[i] for i in valid_indices if i < len(legends)]
            if len(filtered_legends) != len(mols):
                logging.warning(
                    "Legend count mismatch after filtering, disabling legends"
                )
                legends = None
            else:
                legends = filtered_legends

        try:
            # Create and use grid renderer (not cached due to varying parameters)
            from molecular_string_renderer.renderers import MoleculeGridRenderer

            grid_renderer = MoleculeGridRenderer(
                config=self.render_config, mols_per_row=mols_per_row, mol_size=mol_size
            )
            image = grid_renderer.render_grid(mols, legends)
            if image is None:
                raise RenderingError("Grid renderer returned None image")
        except Exception as e:
            if isinstance(e, RenderingError):
                raise
            raise RenderingError(f"Error rendering molecule grid: {e}") from e

        try:
            # Save if output path provided
            if output_path:
                output_handler = self._get_cached_output_handler(output_format)
                output_handler.save(image, output_path)
        except Exception as e:
            if isinstance(e, OutputError):
                raise
            raise OutputError(f"Error saving grid output: {e}") from e

        return image

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

        Raises:
            ConfigurationError: If new configurations are invalid
        """
        # Validate new configurations before updating
        if render_config:
            try:
                render_config.model_dump()
            except Exception as e:
                raise ConfigurationError(f"Invalid render configuration: {e}") from e
            self.render_config = render_config

        if parser_config:
            try:
                parser_config.model_dump()
            except Exception as e:
                raise ConfigurationError(f"Invalid parser configuration: {e}") from e
            self.parser_config = parser_config

        if output_config:
            try:
                output_config.model_dump()
            except Exception as e:
                raise ConfigurationError(f"Invalid output configuration: {e}") from e
            self.output_config = output_config

        # Clear caches when config changes to ensure consistency
        self._parsers.clear()
        self._renderers.clear()
        self._output_handlers.clear()

    def get_stats(self) -> dict[str, int]:
        """
        Get performance statistics for this renderer instance.

        Returns:
            Dictionary with operation count and cache statistics
        """
        return {
            "operations": self._operation_count,
            "cache_hits": self._cache_hits,
            "cache_efficiency": (
                round(self._cache_hits / max(1, self._operation_count) * 100, 1)
                if self._operation_count > 0
                else 0.0
            ),
            "cached_parsers": len(self._parsers),
            "cached_renderers": len(self._renderers),
            "cached_output_handlers": len(self._output_handlers),
        }

    def clear_cache(self) -> None:
        """Clear all cached objects to free memory."""
        self._parsers.clear()
        self._renderers.clear()
        self._output_handlers.clear()
