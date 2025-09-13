"""
Output format abstractions and implementations.

Provides flexible output generation for rendered molecules.
"""

import base64
import hashlib
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

from PIL import Image

from molecular_string_renderer.config import OutputConfig

logger = logging.getLogger(__name__)


class OutputHandler(ABC):
    """Abstract base class for output handlers."""

    def __init__(self, config: OutputConfig | None = None):
        """Initialize output handler with configuration."""
        self.config = config or OutputConfig()

    @abstractmethod
    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """
        Save image to specified path.

        Args:
            image: PIL Image to save
            output_path: Path where image should be saved
        """
        pass

    @abstractmethod
    def get_bytes(self, image: Image.Image) -> bytes:
        """
        Get image as bytes.

        Args:
            image: PIL Image to convert

        Returns:
            Image data as bytes
        """
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Get the file extension for this output format."""
        pass

    def _ensure_output_directory(self, output_path: str | Path) -> Path:
        """Ensure output directory exists."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class PNGOutput(OutputHandler):
    """PNG output handler."""

    @property
    def file_extension(self) -> str:
        """Get PNG file extension."""
        return ".png"

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save image as PNG file."""
        path = self._ensure_output_directory(output_path)

        # Ensure PNG extension
        if not str(path).lower().endswith(".png"):
            path = path.with_suffix(".png")

        try:
            # Convert to RGB if saving as PNG (remove alpha for smaller files if not needed)
            save_image = image
            if image.mode == "RGBA" and not self._has_transparency(image):
                save_image = image.convert("RGB")

            save_image.save(
                path, "PNG", optimize=self.config.optimize, quality=self.config.quality
            )
            logger.info(f"Successfully saved PNG to '{path}'")

        except Exception as e:
            logger.error(f"Failed to save PNG to '{path}': {e}")
            raise IOError(f"Failed to save PNG to '{path}': {e}")

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as PNG bytes."""
        buffer = BytesIO()

        # Convert to RGB if no transparency
        save_image = image
        if image.mode == "RGBA" and not self._has_transparency(image):
            save_image = image.convert("RGB")

        save_image.save(
            buffer, "PNG", optimize=self.config.optimize, quality=self.config.quality
        )

        return buffer.getvalue()

    def _has_transparency(self, image: Image.Image) -> bool:
        """Check if image has transparent pixels."""
        if image.mode != "RGBA":
            return False

        # Quick check by examining alpha channel
        alpha = image.split()[-1]
        return alpha.getextrema()[0] < 255


class SVGOutput(OutputHandler):
    """SVG output handler."""

    @property
    def file_extension(self) -> str:
        """Get SVG file extension."""
        return ".svg"

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save image as SVG file (converts from raster)."""
        # Note: This is a raster-to-SVG conversion
        # For true vector SVG, we'd need to use RDKit's SVG drawing directly
        path = self._ensure_output_directory(output_path)

        if not str(path).lower().endswith(".svg"):
            path = path.with_suffix(".svg")

        # Convert PIL image to SVG by embedding as base64
        # Convert to PNG bytes first
        png_output = PNGOutput(self.config)
        png_bytes = png_output.get_bytes(image)

        # Embed in SVG
        base64_data = base64.b64encode(png_bytes).decode()

        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{image.width}" height="{image.height}" 
     viewBox="0 0 {image.width} {image.height}">
  <image width="{image.width}" height="{image.height}" 
         xlink:href="data:image/png;base64,{base64_data}"/>
</svg>'''

        try:
            path.write_text(svg_content, encoding="utf-8")
            logger.info(f"Successfully saved SVG to '{path}'")
        except Exception as e:
            logger.error(f"Failed to save SVG to '{path}': {e}")
            raise IOError(f"Failed to save SVG to '{path}': {e}")

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as SVG bytes."""
        png_output = PNGOutput(self.config)
        png_bytes = png_output.get_bytes(image)
        base64_data = base64.b64encode(png_bytes).decode()

        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{image.width}" height="{image.height}" 
     viewBox="0 0 {image.width} {image.height}">
  <image width="{image.width}" height="{image.height}" 
         xlink:href="data:image/png;base64,{base64_data}"/>
</svg>'''

        return svg_content.encode("utf-8")


class JPEGOutput(OutputHandler):
    """JPEG output handler."""

    @property
    def file_extension(self) -> str:
        """Get JPEG file extension."""
        return ".jpg"

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save image as JPEG file."""
        path = self._ensure_output_directory(output_path)

        if not str(path).lower().endswith((".jpg", ".jpeg")):
            path = path.with_suffix(".jpg")

        # Convert to RGB (JPEG doesn't support alpha)
        save_image = image.convert("RGB")

        try:
            save_image.save(
                path, "JPEG", optimize=self.config.optimize, quality=self.config.quality
            )
            logger.info(f"Successfully saved JPEG to '{path}'")
        except Exception as e:
            logger.error(f"Failed to save JPEG to '{path}': {e}")
            raise IOError(f"Failed to save JPEG to '{path}': {e}")

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as JPEG bytes."""
        buffer = BytesIO()
        save_image = image.convert("RGB")

        save_image.save(
            buffer, "JPEG", optimize=self.config.optimize, quality=self.config.quality
        )

        return buffer.getvalue()


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


def get_output_handler(
    format_type: str, config: OutputConfig | None = None
) -> OutputHandler:
    """
    Factory function to get appropriate output handler.

    Args:
        format_type: Output format type ('png', 'svg', 'jpg', 'jpeg')
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
