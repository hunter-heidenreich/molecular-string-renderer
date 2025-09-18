"""
Raster image output handlers.

Provides implementations for common raster image formats.
"""

import logging
from typing import Any

from molecular_string_renderer.outputs.base import RasterOutputHandler

logger = logging.getLogger(__name__)


class PNGOutput(RasterOutputHandler):
    """PNG output handler."""

    @property
    def file_extension(self) -> str:
        """Get PNG file extension."""
        return ".png"

    @property
    def pil_format(self) -> str:
        """Get PIL format string."""
        return "PNG"

    @property
    def valid_extensions(self) -> list[str]:
        """Get valid PNG extensions."""
        return [".png"]

    @property
    def supports_alpha(self) -> bool:
        """PNG supports alpha channel."""
        return True

    @property
    def supports_quality(self) -> bool:
        """PNG supports quality parameter."""
        return True

    def _has_transparency(self, image) -> bool:
        """Check if image has transparent pixels."""
        if image.mode != "RGBA":
            return False
        # Quick check by examining alpha channel
        alpha = image.split()[-1]
        return alpha.getextrema()[0] < 255

    def _prepare_image(self, image):
        """Prepare PNG image (convert to RGB if no transparency for smaller files)."""
        if image.mode == "RGBA" and not self._has_transparency(image):
            return image.convert("RGB")
        return image


class JPEGOutput(RasterOutputHandler):
    """JPEG output handler."""

    @property
    def file_extension(self) -> str:
        """Get JPEG file extension."""
        return ".jpg"

    @property
    def pil_format(self) -> str:
        """Get PIL format string."""
        return "JPEG"

    @property
    def valid_extensions(self) -> list[str]:
        """Get valid JPEG extensions."""
        return [".jpg", ".jpeg"]

    @property
    def supports_alpha(self) -> bool:
        """JPEG does not support alpha channel."""
        return False

    @property
    def supports_quality(self) -> bool:
        """JPEG supports quality parameter."""
        return True


class WEBPOutput(RasterOutputHandler):
    """WEBP output handler."""

    @property
    def file_extension(self) -> str:
        """Get WEBP file extension."""
        return ".webp"

    @property
    def pil_format(self) -> str:
        """Get PIL format string."""
        return "WEBP"

    @property
    def valid_extensions(self) -> list[str]:
        """Get valid WEBP extensions."""
        return [".webp"]

    @property
    def supports_alpha(self) -> bool:
        """WEBP supports alpha channel."""
        return True

    @property
    def supports_quality(self) -> bool:
        """WEBP supports quality parameter."""
        return True


class TIFFOutput(RasterOutputHandler):
    """TIFF output handler."""

    @property
    def file_extension(self) -> str:
        """Get TIFF file extension."""
        return ".tiff"

    @property
    def pil_format(self) -> str:
        """Get PIL format string."""
        return "TIFF"

    @property
    def valid_extensions(self) -> list[str]:
        """Get valid TIFF extensions."""
        return [".tiff", ".tif"]

    @property
    def supports_alpha(self) -> bool:
        """TIFF supports alpha channel."""
        return True

    @property
    def supports_quality(self) -> bool:
        """TIFF quality only supported with JPEG compression."""
        return False

    def _get_save_kwargs(self) -> dict[str, Any]:
        """Get TIFF-specific save kwargs."""
        kwargs = {"format": self.pil_format}

        if self.config.optimize:
            kwargs["compression"] = "tiff_lzw"
        # Note: Quality only supported with JPEG compression in TIFF

        return kwargs


class BMPOutput(RasterOutputHandler):
    """BMP output handler."""

    @property
    def file_extension(self) -> str:
        """Get BMP file extension."""
        return ".bmp"

    @property
    def pil_format(self) -> str:
        """Get PIL format string."""
        return "BMP"

    @property
    def valid_extensions(self) -> list[str]:
        """Get valid BMP extensions."""
        return [".bmp"]

    @property
    def supports_alpha(self) -> bool:
        """BMP does not support alpha channel."""
        return False

    @property
    def supports_quality(self) -> bool:
        """BMP does not support quality parameter."""
        return False

    def _get_save_kwargs(self) -> dict[str, Any]:
        """Get BMP-specific save kwargs (no optimization support)."""
        return {"format": self.pil_format}
