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

# Handler mapping - centralized for easy maintenance
_OUTPUT_HANDLERS: dict[str, type[OutputHandler]] = {
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


def get_supported_formats() -> list[str]:
    """
    Get list of supported output formats.
    
    Returns:
        List of supported format strings
    """
    return list(_OUTPUT_HANDLERS.keys())


def _validate_format(format_type: str) -> str:
    """
    Validate and normalize format type.
    
    Args:
        format_type: Raw format type string
        
    Returns:
        Normalized format type
        
    Raises:
        ValueError: If format type is not supported
    """
    normalized_format = format_type.lower().strip()
    
    if normalized_format not in _OUTPUT_HANDLERS:
        supported_formats = get_supported_formats()
        error_msg = f"Unsupported output format: {format_type}. Supported: {supported_formats}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return normalized_format


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
    validated_format = _validate_format(format_type)
    handler_class = _OUTPUT_HANDLERS[validated_format]
    
    logger.debug(f"Creating {validated_format} output handler")
    return handler_class(config)
