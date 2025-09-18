"""
Raster image output handlers.

Provides implementations for common raster image formats.
"""

import logging
from typing import Any

from PIL import Image

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

    def _has_transparency(self, image: Image.Image) -> bool:
        """Check if image has transparent pixels."""
        if image.mode not in ("RGBA", "LA"):
            return False
        # Quick check by examining alpha channel
        alpha = image.split()[-1]
        return alpha.getextrema()[0] < 255

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare PNG image (convert to optimal mode if no transparency)."""
        if image.mode == "RGBA" and not self._has_transparency(image):
            return image.convert("RGB")
        elif image.mode == "LA" and not self._has_transparency(image):
            return image.convert("L")
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

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for JPEG saving (convert to JPEG-compatible modes)."""
        # JPEG supports: L (grayscale), RGB, and CMYK modes
        # Convert other modes to RGB for maximum compatibility
        if image.mode in ("RGB", "L", "CMYK"):
            return image
        elif image.mode in ("RGBA", "LA", "PA"):
            # Images with alpha channel - convert to RGB (loses transparency)
            return image.convert("RGB")
        elif image.mode in ("P", "1"):
            # Palette and monochrome images - convert to RGB
            return image.convert("RGB")
        else:
            # Any other modes (e.g., LAB, HSV, etc.) - convert to RGB
            logger.warning(
                f"Converting unusual image mode '{image.mode}' to RGB for JPEG"
            )
            return image.convert("RGB")


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

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for BMP saving (handle unsupported modes)."""
        # BMP supports: 1, L, P, RGB modes
        # BMP can handle RGBA but alpha channel is ignored/flattened
        if image.mode in ("1", "L", "P", "RGB", "RGBA"):
            return image
        elif image.mode == "LA":
            # LA mode not supported by BMP, convert to RGB
            return image.convert("RGB")
        else:
            # Convert any other modes to RGB for maximum compatibility
            logger.warning(f"Converting image mode '{image.mode}' to RGB for BMP")
            return image.convert("RGB")

    def _get_save_kwargs(self) -> dict[str, Any]:
        """Get BMP-specific save kwargs (no optimization support)."""
        return {"format": self.pil_format}
