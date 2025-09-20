"""
Raster image output handlers.

Provides implementations for common raster image formats.
"""

import logging
from typing import Any

from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs.base import RasterOutputHandler
from molecular_string_renderer.outputs.utils import ImageModeUtils

logger = logging.getLogger(__name__)


class PNGOutput(RasterOutputHandler):
    """PNG output handler."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        """Initialize PNG output handler."""
        super().__init__("png", config)


class JPEGOutput(RasterOutputHandler):
    """JPEG output handler."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        """Initialize JPEG output handler."""
        super().__init__("jpeg", config)

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for JPEG saving (convert to JPEG-compatible modes)."""
        return ImageModeUtils.ensure_jpeg_compatible(image)


class WEBPOutput(RasterOutputHandler):
    """WEBP output handler."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        """Initialize WEBP output handler."""
        super().__init__("webp", config)


class TIFFOutput(RasterOutputHandler):
    """TIFF output handler."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        """Initialize TIFF output handler."""
        super().__init__("tiff", config)

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for TIFF saving (preserve transparency for TIFF)."""
        return image

    def _get_save_kwargs(self) -> dict[str, Any]:
        """Get TIFF-specific save kwargs."""
        kwargs = {"format": self.pil_format}

        if self.config.optimize:
            kwargs["compression"] = "tiff_lzw"

        return kwargs


class BMPOutput(RasterOutputHandler):
    """BMP output handler."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        """Initialize BMP output handler."""
        super().__init__("bmp", config)

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for BMP saving (handle unsupported modes)."""
        return ImageModeUtils.ensure_bmp_compatible(image)

    def _get_save_kwargs(self) -> dict[str, Any]:
        """Get BMP-specific save kwargs (no optimization support)."""
        return {"format": self.pil_format}
