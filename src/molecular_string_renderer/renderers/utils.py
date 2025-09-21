"""
Utility classes and functions for molecular rendering.

Provides common utilities for color handling, formatting, and other
rendering-related helper functions.
"""

import logging

from PIL import Image

logger = logging.getLogger(__name__)


class ColorUtils:
    """Utility class for color parsing and conversion operations."""

    @staticmethod
    def parse_color_to_rgba(color_str: str) -> tuple[float, float, float, float]:
        """
        Parse color string to RGBA tuple (0-1 range) for RDKit.

        Args:
            color_str: Color name or hex string

        Returns:
            RGBA tuple with values in 0-1 range

        Raises:
            ValueError: If color cannot be parsed
        """
        try:
            # Try to create a PIL image with the color to parse it
            test_img = Image.new("RGB", (1, 1), color_str)
            r, g, b = test_img.getpixel((0, 0))
            return (r / 255.0, g / 255.0, b / 255.0, 1.0)
        except Exception as e:
            logger.warning(
                f"Failed to parse color '{color_str}': {e}. Using white as fallback."
            )
            # Fall back to white if color parsing fails
            return (1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def is_white_background(color_str: str) -> bool:
        """
        Check if the given color string represents white background.

        Args:
            color_str: Color string to check

        Returns:
            True if color is white or equivalent
        """
        normalized = color_str.lower().strip()
        return normalized in ("white", "#ffffff", "#fff", "rgb(255,255,255)")
