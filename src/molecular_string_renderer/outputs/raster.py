"""
Raster image output handlers.

Provides implementations for common raster image formats.
"""

import logging
from datetime import datetime
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs.base import RasterOutputHandler
from molecular_string_renderer.outputs.utils import ImageModeUtils

logger = logging.getLogger(__name__)


class PNGOutput(RasterOutputHandler):
    """PNG output handler."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        """Initialize PNG output handler."""
        super().__init__("png", config)

    def _get_save_kwargs(self) -> dict[str, Any]:
        """Get PNG-specific save kwargs including metadata."""
        kwargs = super()._get_save_kwargs()

        # Add PNG metadata if provided
        if self.config.metadata:
            pnginfo = PngInfo()

            # Add default metadata
            pnginfo.add_text("Software", "molecular-string-renderer")
            pnginfo.add_text("Creation Time", datetime.now().isoformat())

            # Add custom metadata
            for key, value in self.config.metadata.items():
                pnginfo.add_text(key, value)

            kwargs["pnginfo"] = pnginfo

        return kwargs


class JPEGOutput(RasterOutputHandler):
    """JPEG output handler."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        """Initialize JPEG output handler."""
        super().__init__("jpeg", config)

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for JPEG saving (convert to JPEG-compatible modes)."""
        return ImageModeUtils.ensure_jpeg_compatible(image)

    def _get_save_kwargs(self) -> dict[str, Any]:
        """Get JPEG-specific save kwargs including metadata."""
        kwargs = super()._get_save_kwargs()

        # Add JPEG EXIF metadata if provided
        if self.config.metadata:
            # JPEG EXIF support is limited in PIL, but we can add basic info
            # Map common metadata to EXIF tags where possible
            if "Description" in self.config.metadata:
                kwargs["description"] = self.config.metadata["Description"]
            if "Comment" in self.config.metadata:
                kwargs["comment"] = self.config.metadata["Comment"]

        return kwargs


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

        # Add DPI information
        kwargs["dpi"] = (self.config.dpi, self.config.dpi)

        # Add TIFF metadata if provided
        if self.config.metadata:
            # TIFF supports various metadata tags
            if "Description" in self.config.metadata:
                kwargs["description"] = self.config.metadata["Description"]
            if "Software" in self.config.metadata:
                kwargs["software"] = self.config.metadata["Software"]
            else:
                kwargs["software"] = "molecular-string-renderer"

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


class GIFOutput(RasterOutputHandler):
    """GIF output handler."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        """Initialize GIF output handler."""
        super().__init__("gif", config)

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for GIF saving (convert to palette mode for optimal quality)."""
        return ImageModeUtils.ensure_gif_compatible(image)

    def _get_save_kwargs(self) -> dict[str, Any]:
        """Get GIF-specific save kwargs."""
        kwargs = {"format": self.pil_format}

        # GIF can be optimized even though it doesn't support quality settings
        if self.config.optimize:
            kwargs["optimize"] = True

        # Add transparency preservation
        # PIL will automatically preserve transparency when saving in P mode
        return kwargs
