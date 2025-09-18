"""
Base classes for output handlers.

Provides common functionality and abstractions for all output formats.
"""

import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any

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

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Get the PIL format name for this output format."""
        pass

    def _ensure_output_directory(self, output_path: str | Path) -> Path:
        """Ensure output directory exists and return normalized path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _ensure_extension(self, path: Path, extensions: list[str]) -> Path:
        """Ensure path has one of the valid extensions."""
        path_str = str(path).lower()
        if not any(path_str.endswith(ext) for ext in extensions):
            return path.with_suffix(extensions[0])
        return path

    def _handle_save_error(self, path: Path, error: Exception) -> None:
        """Handle and log save errors consistently."""
        logger.error(f"Failed to save {self.format_name} to '{path}': {error}")
        raise IOError(f"Failed to save {self.format_name} to '{path}': {error}")

    def _log_success(self, path: Path) -> None:
        """Log successful save operation."""
        logger.info(f"Successfully saved {self.format_name} to '{path}'")


class RasterOutputHandler(OutputHandler):
    """Base class for raster image output handlers."""

    def __init__(self, config: OutputConfig | None = None):
        """Initialize raster output handler."""
        super().__init__(config)

    @property
    @abstractmethod
    def pil_format(self) -> str:
        """Get the PIL format string for saving."""
        pass

    @property
    @abstractmethod
    def valid_extensions(self) -> list[str]:
        """Get list of valid file extensions for this format."""
        pass

    @property
    @abstractmethod
    def supports_alpha(self) -> bool:
        """Whether this format supports alpha channel."""
        pass

    @property
    @abstractmethod
    def supports_quality(self) -> bool:
        """Whether this format supports quality parameter."""
        pass

    @property
    def format_name(self) -> str:
        """Get the format name (defaults to PIL format)."""
        return self.pil_format

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for saving (handle alpha channel if needed)."""
        if not self.supports_alpha and image.mode in ("RGBA", "LA"):
            return image.convert("RGB")
        return image

    def _get_save_kwargs(self) -> dict[str, Any]:
        """Get keyword arguments for PIL save method."""
        kwargs = {"format": self.pil_format}

        if self.supports_quality:
            kwargs.update(
                {
                    "optimize": self.config.optimize,
                    "quality": self.config.quality,
                }
            )
        elif self.config.optimize:
            # Some formats support optimize but not quality
            kwargs["optimize"] = True

        return kwargs

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save image as raster file."""
        path = self._ensure_output_directory(output_path)
        path = self._ensure_extension(path, self.valid_extensions)

        try:
            prepared_image = self._prepare_image(image)
            save_kwargs = self._get_save_kwargs()
            prepared_image.save(path, **save_kwargs)
            self._log_success(path)
        except Exception as e:
            self._handle_save_error(path, e)

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as raster bytes."""
        buffer = BytesIO()
        prepared_image = self._prepare_image(image)
        save_kwargs = self._get_save_kwargs()
        prepared_image.save(buffer, **save_kwargs)
        return buffer.getvalue()


class VectorOutputHandler(OutputHandler):
    """Base class for vector/document output handlers."""

    def __init__(self, config: OutputConfig | None = None):
        """Initialize vector output handler."""
        super().__init__(config)

    @property
    def format_name(self) -> str:
        """Get the format name (defaults to file extension without dot)."""
        return self.file_extension.lstrip(".")
