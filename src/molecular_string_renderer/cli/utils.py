"""
Utility functions for the CLI.

This module contains various utility functions used throughout the CLI,
including format normalization and rendering information logging.
"""

import argparse
import logging

from molecular_string_renderer.config import OutputConfig, RenderConfig

# Set up module-level logger
logger = logging.getLogger(__name__)


def normalize_format(value: str) -> str:
    """Normalize format input to lowercase.

    Args:
        value: The format string to normalize.

    Returns:
        The normalized format string in lowercase.

    Raises:
        argparse.ArgumentTypeError: If the format is not valid.
    """
    normalized = value.lower().strip()
    valid_formats = ["smiles", "smi", "inchi", "mol", "selfies"]
    if normalized not in valid_formats:
        raise argparse.ArgumentTypeError(
            f"invalid choice: '{value}' (choose from {valid_formats})"
        )
    return normalized


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
