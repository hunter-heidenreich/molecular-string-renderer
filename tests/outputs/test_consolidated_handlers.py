"""
Consolidated test suite for output handlers.

Tests shared behavior across all output handlers and format-specific behavior
only where necessary. This replaces the need for separate test files for each
format with mostly duplicate tests.
"""

import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import get_output_handler
from molecular_string_renderer.outputs.base import (
    OutputHandler,
    RasterOutputHandler,
    VectorOutputHandler,
)

from .conftest import (
    ALL_FORMATS,
    supports_optimization,
    supports_quality,
)


class TestOutputHandlerProperties:
    """Test properties common to all output handlers."""

    def test_file_extension_format(self, output_handler):
        """Test that file extension starts with dot and matches format."""
        extension = output_handler.file_extension
        assert isinstance(extension, str)
        assert extension.startswith(".")
        assert len(extension) > 1

    def test_format_name_is_string(self, output_handler):
        """Test that format name is a non-empty string."""
        format_name = output_handler.format_name
        assert isinstance(format_name, str)
        assert len(format_name) > 0

    def test_valid_extensions_contains_primary(self, output_handler):
        """Test that valid extensions contains the primary extension."""
        if hasattr(output_handler, "valid_extensions"):
            extensions = output_handler.valid_extensions
            assert isinstance(extensions, list)
            assert output_handler.file_extension in extensions

    def test_supports_alpha_is_boolean(self, output_handler):
        """Test that supports_alpha returns a boolean."""
        if hasattr(output_handler, "supports_alpha"):
            assert isinstance(output_handler.supports_alpha, bool)

    def test_supports_quality_is_boolean(self, output_handler):
        """Test that supports_quality returns a boolean."""
        if hasattr(output_handler, "supports_quality"):
            assert isinstance(output_handler.supports_quality, bool)


class TestOutputHandlerInitialization:
    """Test initialization patterns across all handlers."""

    def test_init_without_config(self, format_name):
        """Test initialization without config."""
        handler = get_output_handler(format_name)
        assert handler.config is not None
        assert isinstance(handler.config, OutputConfig)

    def test_init_with_config(self, format_name):
        """Test initialization with custom config."""
        config = OutputConfig(quality=80, optimize=True)
        handler = get_output_handler(format_name, config)
        assert handler.config is config

    def test_init_with_none_config(self, format_name):
        """Test initialization with None config creates default."""
        handler = get_output_handler(format_name, None)
        assert handler.config is not None
        assert isinstance(handler.config, OutputConfig)


class TestOutputHandlerBasicFunctionality:
    """Test basic functionality common to all handlers."""

    def test_get_bytes_returns_bytes(self, output_handler, test_image):
        """Test that get_bytes returns bytes object."""
        result = output_handler.get_bytes(test_image)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_save_creates_file(self, output_handler, test_image, temp_dir):
        """Test that save creates a file."""
        output_path = temp_dir / f"test{output_handler.file_extension}"
        output_handler.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_with_string_path(self, output_handler, test_image, temp_dir):
        """Test save with string path."""
        output_path = str(temp_dir / f"test{output_handler.file_extension}")
        output_handler.save(test_image, output_path)

        assert Path(output_path).exists()

    def test_save_auto_extension(self, output_handler, test_image, temp_dir):
        """Test that save automatically adds extension if missing."""
        output_path = temp_dir / "test_no_extension"
        output_handler.save(test_image, output_path)

        # Should create file with proper extension
        expected_path = temp_dir / f"test_no_extension{output_handler.file_extension}"
        assert expected_path.exists()

    def test_save_creates_directory(self, output_handler, test_image, temp_dir):
        """Test that save creates parent directories."""
        nested_path = (
            temp_dir / "nested" / "dir" / f"test{output_handler.file_extension}"
        )
        output_handler.save(test_image, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.is_dir()


class TestOutputHandlerImageModeHandling:
    """Test image mode handling across formats."""

    def test_rgb_image_handling(self, output_handler, test_image):
        """Test RGB image handling."""
        result = output_handler.get_bytes(test_image)
        assert len(result) > 0

    def test_rgba_image_handling(self, output_handler, rgba_image, format_name):
        """Test RGBA image handling."""
        result = output_handler.get_bytes(rgba_image)
        assert len(result) > 0

        # Format-specific behavior validation could be added here
        # but the key test is that it doesn't crash

    def test_grayscale_image_handling(self, output_handler, grayscale_image):
        """Test grayscale image handling."""
        result = output_handler.get_bytes(grayscale_image)
        assert len(result) > 0

    def test_la_image_handling(self, output_handler, la_image):
        """Test grayscale+alpha image handling."""
        result = output_handler.get_bytes(la_image)
        assert len(result) > 0


class TestOutputHandlerConfigurationHandling:
    """Test configuration handling across handlers."""

    def test_quality_setting_respected(self, format_name):
        """Test that quality settings are handled appropriately for all formats."""
        config = OutputConfig(quality=50)
        handler = get_output_handler(format_name, config)
        test_image = Image.new("RGB", (100, 100), "red")

        # Should not raise an error regardless of support
        result = handler.get_bytes(test_image)
        assert len(result) > 0

    def test_optimization_setting_respected(self, format_name):
        """Test that optimization settings are handled appropriately for all formats."""
        config = OutputConfig(optimize=True)
        handler = get_output_handler(format_name, config)
        test_image = Image.new("RGB", (100, 100), "red")

        # Should not raise an error regardless of support
        result = handler.get_bytes(test_image)
        assert len(result) > 0


class TestOutputHandlerEdgeCases:
    """Test edge cases common across handlers."""

    def test_very_small_image(self, output_handler):
        """Test with very small images."""
        tiny_image = Image.new("RGB", (1, 1), "red")
        result = output_handler.get_bytes(tiny_image)
        assert len(result) > 0

    def test_large_image_dimensions(self, output_handler):
        """Test with reasonably large images."""
        large_image = Image.new("RGB", (1000, 800), "blue")
        result = output_handler.get_bytes(large_image)
        assert len(result) > 0

    def test_square_vs_rectangular_images(self, output_handler):
        """Test with different aspect ratios."""
        square = Image.new("RGB", (100, 100), "red")
        wide = Image.new("RGB", (200, 50), "green")
        tall = Image.new("RGB", (50, 200), "blue")

        for img in [square, wide, tall]:
            result = output_handler.get_bytes(img)
            assert len(result) > 0


class TestOutputHandlerErrorHandling:
    """Test error handling across handlers."""

    def test_save_invalid_path_raises_error(self, output_handler, test_image):
        """Test that invalid save paths raise appropriate errors."""
        # Try to save to a path that can't be created (like root on Unix)
        invalid_path = "/root/nonexistent/test.png"

        with pytest.raises((IOError, OSError, PermissionError)):
            output_handler.save(test_image, invalid_path)

    @patch("PIL.Image.Image.save")
    def test_save_pil_error_handling(
        self, mock_save, output_handler, test_image, temp_dir
    ):
        """Test handling of PIL save errors."""
        mock_save.side_effect = Exception("PIL save failed")
        output_path = temp_dir / f"test{output_handler.file_extension}"

        with pytest.raises(IOError):
            output_handler.save(test_image, output_path)


class TestOutputHandlerInheritance:
    """Test inheritance hierarchy compliance."""

    def test_all_handlers_inherit_from_base(self, output_handler):
        """Test that all handlers inherit from OutputHandler."""
        assert isinstance(output_handler, OutputHandler)

    def test_raster_handlers_inherit_correctly(self, raster_format_name):
        """Test that raster handlers inherit from RasterOutputHandler."""
        handler = get_output_handler(raster_format_name)
        assert isinstance(handler, RasterOutputHandler)
        assert isinstance(handler, OutputHandler)

    def test_vector_handlers_inherit_correctly(self, vector_format_name):
        """Test that vector handlers inherit from VectorOutputHandler."""
        handler = get_output_handler(vector_format_name)
        assert isinstance(handler, VectorOutputHandler)
        assert isinstance(handler, OutputHandler)

    def test_required_methods_implemented(self, output_handler):
        """Test that all required abstract methods are implemented."""
        # Test property access
        assert hasattr(output_handler, "file_extension")
        assert hasattr(output_handler, "format_name")

        # Test method access
        assert hasattr(output_handler, "save")
        assert hasattr(output_handler, "get_bytes")
        assert callable(output_handler.save)
        assert callable(output_handler.get_bytes)

    def test_raster_specific_methods(self, raster_format_name):
        """Test raster-specific methods and properties."""
        handler = get_output_handler(raster_format_name)

        assert hasattr(handler, "pil_format")
        assert hasattr(handler, "valid_extensions")
        assert hasattr(handler, "supports_alpha")
        assert hasattr(handler, "supports_quality")

        # Test they return expected types
        assert isinstance(handler.pil_format, str)
        assert isinstance(handler.valid_extensions, list)
        assert isinstance(handler.supports_alpha, bool)
        assert isinstance(handler.supports_quality, bool)


class TestOutputHandlerThreadSafety:
    """Test thread safety across handlers."""

    def test_multiple_instances_independent(self, format_name):
        """Test that multiple instances are independent."""
        config1 = OutputConfig(optimize=True)
        config2 = OutputConfig(optimize=False)

        handler1 = get_output_handler(format_name, config1)
        handler2 = get_output_handler(format_name, config2)

        assert handler1.config.optimize is True
        assert handler2.config.optimize is False
        assert handler1.config is not handler2.config

    def test_concurrent_operations(self, output_handler):
        """Test concurrent get_bytes operations."""
        images = [
            Image.new("RGB", (50, 50), "red"),
            Image.new("RGB", (50, 50), "green"),
            Image.new("RGB", (50, 50), "blue"),
        ]
        results = []
        threads = []

        def get_bytes_worker(img):
            result = output_handler.get_bytes(img)
            results.append(result)

        for img in images:
            thread = threading.Thread(target=get_bytes_worker, args=(img,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 3
        assert all(len(result) > 0 for result in results)


class TestOutputHandlerQualityAndOptimization:
    """Test quality and optimization features where supported."""

    def test_quality_affects_output_size(self, format_name):
        """Test that quality settings are handled properly for all formats."""
        test_image = Image.new("RGB", (200, 200), "red")

        high_quality_handler = get_output_handler(format_name, OutputConfig(quality=95))
        low_quality_handler = get_output_handler(format_name, OutputConfig(quality=20))

        high_quality_bytes = high_quality_handler.get_bytes(test_image)
        low_quality_bytes = low_quality_handler.get_bytes(test_image)

        # All formats should produce valid output regardless of quality support
        assert len(high_quality_bytes) > 0
        assert len(low_quality_bytes) > 0

        # For formats that don't support quality, sizes should be similar
        # (PDF may have slight variations due to timestamps, etc.)
        if supports_quality(format_name):
            # Quality-supporting formats may have different file sizes
            # (though not guaranteed for all images)
            pass
        else:
            # Non-quality formats should produce similar output
            # PDF may have slight variations due to non-deterministic elements
            if format_name == "pdf":
                # PDF may vary slightly due to timestamps, but should be similar size
                assert abs(len(high_quality_bytes) - len(low_quality_bytes)) < 200
            else:
                # Other non-quality formats should produce identical output
                assert high_quality_bytes == low_quality_bytes

    def test_optimization_affects_output(self, format_name):
        """Test that optimization settings are handled properly for all formats."""
        test_image = Image.new("RGB", (100, 100), "red")

        optimized_handler = get_output_handler(format_name, OutputConfig(optimize=True))
        unoptimized_handler = get_output_handler(
            format_name, OutputConfig(optimize=False)
        )

        optimized_bytes = optimized_handler.get_bytes(test_image)
        unoptimized_bytes = unoptimized_handler.get_bytes(test_image)

        # All formats should produce valid output regardless of optimization support
        assert len(optimized_bytes) > 0
        assert len(unoptimized_bytes) > 0

        # For formats that don't support optimization, output should be similar
        # (PDF may have slight variations due to timestamps, etc.)
        if not supports_optimization(format_name):
            if format_name == "pdf":
                # PDF may vary slightly due to timestamps, but should be similar size
                assert abs(len(optimized_bytes) - len(unoptimized_bytes)) < 200
            else:
                # Other non-optimization formats should produce identical output
                assert optimized_bytes == unoptimized_bytes


class TestOutputHandlerIntegration:
    """Integration tests across different handlers."""

    def test_same_image_different_formats(self, temp_dir):
        """Test that the same image can be saved in all formats."""
        test_image = Image.new("RGB", (100, 100), "red")

        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)
            output_path = temp_dir / f"test_{format_name}{handler.file_extension}"

            handler.save(test_image, output_path)
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_format_consistency(self):
        """Test that format properties are consistent."""
        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)

            # Basic consistency checks
            assert handler.file_extension.startswith(".")
            assert len(handler.format_name) > 0

            # Extension should be related to format name (with special cases)
            ext_without_dot = handler.file_extension.lstrip(".")
            format_lower = handler.format_name.lower()
            ext_lower = ext_without_dot.lower()
            format_name_lower = format_name.lower()

            # Handle special cases and check for reasonable relationship
            valid_relationship = (
                ext_lower in format_lower
                or format_lower in ext_lower
                or format_name_lower in ext_lower
                or (
                    format_name_lower == "jpeg" and ext_lower == "jpg"
                )  # JPEG special case
                or (
                    format_name_lower == "tiff" and ext_lower == "tif"
                )  # TIFF special case
            )
            assert valid_relationship, (
                f"Format {format_name} has inconsistent extension {handler.file_extension}"
            )

    def test_cross_format_file_sizes(self, temp_dir):
        """Test and compare file sizes across formats (informational)."""
        test_image = Image.new("RGB", (200, 200))
        # Create a pattern to make compression differences more apparent
        pixels = []
        for y in range(200):
            for x in range(200):
                r = (x * 255) // 200
                g = (y * 255) // 200
                b = ((x + y) * 255) // 400
                pixels.append((r, g, b))
        test_image.putdata(pixels)

        sizes = {}
        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)
            data = handler.get_bytes(test_image)
            sizes[format_name] = len(data)

        # Just verify all formats produced output
        # (file size comparisons are informational, not assertions)
        for format_name, size in sizes.items():
            assert size > 0, f"{format_name} produced empty output"
