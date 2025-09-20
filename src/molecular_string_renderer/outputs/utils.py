"""
Utility functions for output handling.

Provides common utilities for filename generation and format handling.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image

from molecular_string_renderer.config import OutputConfig

logger = logging.getLogger(__name__)

# Constants for image processing
_ALPHA_FULLY_OPAQUE = 255


@dataclass(frozen=True)
class FormatInfo:
    """Information about an output format."""

    extension: str
    pil_format: str
    valid_extensions: list[str]
    supports_alpha: bool
    supports_quality: bool
    mime_type: str = ""

    def __post_init__(self) -> None:
        """Validate format info after initialization."""
        if not self.extension.startswith("."):
            object.__setattr__(self, "extension", f".{self.extension}")

        # Ensure primary extension is in valid_extensions
        if self.extension not in self.valid_extensions:
            valid_exts = [self.extension] + self.valid_extensions
            object.__setattr__(self, "valid_extensions", valid_exts)


class FormatRegistry:
    """Registry for output format information."""

    # Base format definitions - avoid duplication by defining shared formats once
    _base_formats = {
        "png": FormatInfo(
            extension=".png",
            pil_format="PNG",
            valid_extensions=[".png"],
            supports_alpha=True,
            supports_quality=True,
            mime_type="image/png",
        ),
        "jpeg": FormatInfo(
            extension=".jpg",
            pil_format="JPEG",
            valid_extensions=[".jpg", ".jpeg"],
            supports_alpha=False,
            supports_quality=True,
            mime_type="image/jpeg",
        ),
        "webp": FormatInfo(
            extension=".webp",
            pil_format="WEBP",
            valid_extensions=[".webp"],
            supports_alpha=True,
            supports_quality=True,
            mime_type="image/webp",
        ),
        "tiff": FormatInfo(
            extension=".tiff",
            pil_format="TIFF",
            valid_extensions=[".tiff", ".tif"],
            supports_alpha=True,
            supports_quality=False,
            mime_type="image/tiff",
        ),
        "bmp": FormatInfo(
            extension=".bmp",
            pil_format="BMP",
            valid_extensions=[".bmp"],
            supports_alpha=False,
            supports_quality=False,
            mime_type="image/bmp",
        ),
        "svg": FormatInfo(
            extension=".svg",
            pil_format="SVG",
            valid_extensions=[".svg"],
            supports_alpha=True,
            supports_quality=False,
            mime_type="image/svg+xml",
        ),
        "pdf": FormatInfo(
            extension=".pdf",
            pil_format="PDF",
            valid_extensions=[".pdf"],
            supports_alpha=False,
            supports_quality=False,
            mime_type="application/pdf",
        ),
    }

    # Build complete format mapping with aliases
    _formats: dict[str, FormatInfo] = {
        **_base_formats,
        # Aliases for common alternate names
        "jpg": _base_formats["jpeg"],
        "tif": _base_formats["tiff"],
    }

    @classmethod
    def get_format_info(cls, format_name: str) -> FormatInfo:
        """Get format information by name."""
        format_key = format_name.lower().strip()
        if format_key not in cls._formats:
            raise ValueError(f"Unknown format: {format_name}")
        return cls._formats[format_key]

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """Get list of supported format names."""
        return list(cls._formats.keys())

    @classmethod
    def is_supported(cls, format_name: str) -> bool:
        """Check if format is supported."""
        return format_name.lower().strip() in cls._formats


class ImageModeUtils:
    """Utilities for handling PIL image mode conversions."""

    @staticmethod
    def has_transparency(image: Image.Image) -> bool:
        """
        Check if image has transparent pixels.

        Args:
            image: PIL Image to check

        Returns:
            True if image has any transparent pixels
        """
        if image.mode not in ("RGBA", "LA"):
            return False

        alpha = image.split()[-1]
        return alpha.getextrema()[0] < _ALPHA_FULLY_OPAQUE

    @staticmethod
    def prepare_for_no_alpha(image: Image.Image) -> Image.Image:
        """Prepare image for formats that don't support alpha.

        Args:
            image: PIL Image to prepare

        Returns:
            Image with alpha channel removed if present
        """
        if image.mode == "RGBA":
            return image.convert("RGB")
        elif image.mode == "LA":
            return image.convert("L")
        elif image.mode == "PA":
            return image.convert("P")
        return image

    @staticmethod
    def optimize_for_format(image: Image.Image, supports_alpha: bool) -> Image.Image:
        """
        Optimize image mode for specific format capabilities.

        Args:
            image: PIL Image to optimize
            supports_alpha: Whether target format supports alpha channel

        Returns:
            Optimized image
        """
        if not supports_alpha:
            return ImageModeUtils.prepare_for_no_alpha(image)

        image_mode = image.mode
        if image_mode in ("RGBA", "LA") and not ImageModeUtils.has_transparency(image):
            return image.convert("RGB" if image_mode == "RGBA" else "L")

        return image

    @staticmethod
    def _convert_to_rgb_with_warning(image: Image.Image, format_name: str) -> Image.Image:
        """Convert image to RGB with appropriate warning."""
        logger.warning(f"Converting image mode '{image.mode}' to RGB for {format_name}")
        return image.convert("RGB")

    @staticmethod
    def ensure_jpeg_compatible(image: Image.Image) -> Image.Image:
        """Ensure image is compatible with JPEG format.

        Args:
            image: PIL Image to make JPEG-compatible

        Returns:
            JPEG-compatible image
        """
        if image.mode in ("RGB", "L", "CMYK"):
            return image
        elif image.mode in ("RGBA", "LA", "PA", "P", "1"):
            return image.convert("RGB")
        else:
            return ImageModeUtils._convert_to_rgb_with_warning(image, "JPEG")

    @staticmethod
    def ensure_bmp_compatible(image: Image.Image) -> Image.Image:
        """Ensure image is compatible with BMP format.

        Args:
            image: PIL Image to make BMP-compatible

        Returns:
            BMP-compatible image
        """
        if image.mode in ("1", "L", "P", "RGB", "RGBA"):
            return image
        elif image.mode == "LA":
            return image.convert("RGB")
        else:
            return ImageModeUtils._convert_to_rgb_with_warning(image, "BMP")


def create_safe_filename(molecular_string: str, extension: str = ".png") -> str:
    """
    Generate a filesystem-safe filename from a molecular string using MD5 hash.

    Args:
        molecular_string: The input molecular string (SMILES, InChI, etc.)
        extension: File extension to use

    Returns:
        A safe filename with the specified extension
    """
    clean_string = molecular_string.strip()
    hasher = hashlib.md5(clean_string.encode("utf-8"))
    base_name = hasher.hexdigest()

    if not extension.startswith("."):
        extension = f".{extension}"

    return f"{base_name}{extension}"


def build_save_kwargs(format_info: FormatInfo, config: OutputConfig) -> dict[str, Any]:
    """
    Build save kwargs for PIL Image.save() based on format and config.

    Args:
        format_info: Format information
        config: Output configuration

    Returns:
        Dictionary of kwargs for PIL save
    """
    kwargs = {"format": format_info.pil_format}

    if format_info.supports_quality:
        kwargs.update(
            {
                "optimize": config.optimize,
                "quality": config.quality,
            }
        )
    elif config.optimize:
        kwargs["optimize"] = True

    return kwargs
