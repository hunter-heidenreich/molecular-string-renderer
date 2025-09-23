"""
Validation utilities for molecular string rendering.

Provides validation functions for molecular strings, format types,
grid parameters, output paths, and configuration compatibility.
"""

import logging
from pathlib import Path

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.exceptions import ConfigurationError, ValidationError


def validate_molecular_string(molecular_string: str, format_type: str) -> None:
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


def validate_format_type(format_type: str, valid_formats: set[str]) -> str:
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


def validate_grid_parameters(
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


def validate_output_path(output_path: str | Path | None) -> Path | None:
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


def validate_configuration_compatibility(
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
        >>> validate_configuration_compatibility(render_cfg, parser_cfg, output_cfg)
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
