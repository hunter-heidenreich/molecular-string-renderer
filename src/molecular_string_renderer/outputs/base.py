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
from molecular_string_renderer.outputs.utils import (
    FormatRegistry,
    ImageModeUtils,
    build_save_kwargs,
)

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


class RegistryBasedOutputHandler(OutputHandler):
    """Base class for output handlers that use the format registry."""

    def __init__(self, format_key: str, config: OutputConfig | None = None):
        """Initialize with format key for registry lookup."""
        super().__init__(config)
        self._format_info = FormatRegistry.get_format_info(format_key)

    @property
    def file_extension(self) -> str:
        """Get file extension from registry."""
        return self._format_info.extension

    @property
    def format_name(self) -> str:
        """Get format name from registry."""
        return self._format_info.pil_format

    @property
    def valid_extensions(self) -> list[str]:
        """Get valid extensions from registry."""
        return self._format_info.valid_extensions

    @property
    def supports_alpha(self) -> bool:
        """Get alpha support from registry."""
        return self._format_info.supports_alpha

    @property
    def supports_quality(self) -> bool:
        """Get quality support from registry."""
        return self._format_info.supports_quality


class RasterOutputHandler(RegistryBasedOutputHandler):
    """Base class for raster image output handlers."""

    def __init__(self, format_key: str, config: OutputConfig | None = None):
        """Initialize raster output handler."""
        super().__init__(format_key, config)

    @property
    def pil_format(self) -> str:
        """Get the PIL format string for saving."""
        return self._format_info.pil_format

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for saving (optimize based on format capabilities)."""
        return ImageModeUtils.optimize_for_format(image, self.supports_alpha)

    def _get_save_kwargs(self) -> dict[str, Any]:
        """Get keyword arguments for PIL save method."""
        return build_save_kwargs(self._format_info, self.config)

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
        with BytesIO() as buffer:
            prepared_image = self._prepare_image(image)
            save_kwargs = self._get_save_kwargs()
            prepared_image.save(buffer, **save_kwargs)
            return buffer.getvalue()


class VectorOutputHandler(RegistryBasedOutputHandler):
    """Base class for vector/document output handlers."""

    def __init__(self, format_key: str, config: OutputConfig | None = None):
        """Initialize vector output handler."""
        super().__init__(format_key, config)

    @property
    def format_name(self) -> str:
        """Get format name (lowercase for vector formats for compatibility)."""
        return self.file_extension.lstrip(".")
