"""
Factory functions for creating output handlers.

Provides a centralized way to create appropriate output handlers for different formats.
"""

import logging

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs.base import OutputHandler
from molecular_string_renderer.outputs.raster import (
    BMPOutput,
    JPEGOutput,
    PNGOutput,
    TIFFOutput,
    WEBPOutput,
)
from molecular_string_renderer.outputs.vector import PDFOutput, SVGOutput

logger = logging.getLogger(__name__)


def get_output_handler(
    format_type: str, config: OutputConfig | None = None
) -> OutputHandler:
    """
    Factory function to get appropriate output handler.

    Args:
        format_type: Output format type ('png', 'svg', 'jpg', 'jpeg', 'pdf', etc.)
        config: Output configuration

    Returns:
        Appropriate output handler instance

    Raises:
        ValueError: If format type is not supported
    """
    format_type = format_type.lower().strip()

    handlers = {
        "png": PNGOutput,
        "svg": SVGOutput,
        "jpg": JPEGOutput,
        "jpeg": JPEGOutput,
        "pdf": PDFOutput,
        "webp": WEBPOutput,
        "tiff": TIFFOutput,
        "tif": TIFFOutput,  # Alternative extension for TIFF
        "bmp": BMPOutput,
    }

    if format_type not in handlers:
        supported = list(handlers.keys())
        logger.error(
            f"Unsupported output format: {format_type}. Supported formats: {supported}"
        )
        raise ValueError(
            f"Unsupported output format: {format_type}. Supported: {supported}"
        )

    logger.debug(f"Creating {format_type} output handler")
    return handlers[format_type](config)
