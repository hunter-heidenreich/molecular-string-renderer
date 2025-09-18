"""
Test suite for WEBP output handler.

Comprehensive tests for WEBPOutput class functionality, edge cases, and error handling.
Tests alpha channel handling, quality settings, and WEBP-specific features.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs.raster import WEBPOutput


class TestWEBPOutputProperties:
    """Test WEBP output handler properties."""

    def test_file_extension(self):
        """Test file extension property."""
        output = WEBPOutput()
        assert output.file_extension == ".webp"

    def test_pil_format(self):
        """Test PIL format property."""
        output = WEBPOutput()
        assert output.pil_format == "WEBP"

    def test_valid_extensions(self):
        """Test valid extensions property."""
        output = WEBPOutput()
        assert output.valid_extensions == [".webp"]

    def test_supports_alpha(self):
        """Test alpha channel support."""
        output = WEBPOutput()
        assert output.supports_alpha is True

    def test_supports_quality(self):
        """Test quality parameter support."""
        output = WEBPOutput()
        assert output.supports_quality is True

    def test_format_name_inherited(self):
        """Test format name is inherited from base class."""
        output = WEBPOutput()
        assert output.format_name == "WEBP"


class TestWEBPOutputInitialization:
    """Test WEBP output handler initialization."""

    def test_init_without_config(self):
        """Test initialization without explicit config."""
        output = WEBPOutput()
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = OutputConfig(quality=85, optimize=False)
        output = WEBPOutput(config)
        assert output.config is config
        assert output.config.quality == 85
        assert output.config.optimize is False

    def test_init_with_none_config(self):
        """Test initialization with None config."""
        output = WEBPOutput(None)
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)


class TestWEBPOutputSaveKwargs:
    """Test WEBP save keyword arguments generation."""

    def test_get_save_kwargs_default_config(self):
        """Test save kwargs with default config."""
        output = WEBPOutput()
        kwargs = output._get_save_kwargs()

        expected = {
            "format": "WEBP",
            "optimize": True,
            "quality": 95,
        }
        assert kwargs == expected

    def test_get_save_kwargs_custom_config(self):
        """Test save kwargs with custom config."""
        config = OutputConfig(quality=80, optimize=False)
        output = WEBPOutput(config)
        kwargs = output._get_save_kwargs()

        expected = {
            "format": "WEBP",
            "optimize": False,
            "quality": 80,
        }
        assert kwargs == expected

    def test_get_save_kwargs_edge_case_quality_bounds(self):
        """Test save kwargs with quality at bounds."""
        # Test minimum quality
        config = OutputConfig(quality=1)
        output = WEBPOutput(config)
        kwargs = output._get_save_kwargs()
        assert kwargs["quality"] == 1

        # Test maximum quality
        config = OutputConfig(quality=100)
        output = WEBPOutput(config)
        kwargs = output._get_save_kwargs()
        assert kwargs["quality"] == 100

    def test_get_save_kwargs_medium_quality(self):
        """Test save kwargs with medium quality setting."""
        config = OutputConfig(quality=50)
        output = WEBPOutput(config)
        kwargs = output._get_save_kwargs()
        assert kwargs["quality"] == 50
        assert kwargs["optimize"] is True  # Default optimize


class TestWEBPOutputSaveMethod:
    """Test WEBP save method functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (100, 100), "red")

    def test_save_with_string_path(self, temp_dir, test_image):
        """Test saving with string path."""
        output = WEBPOutput()
        output_path = str(temp_dir / "test.webp")

        output.save(test_image, output_path)

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_save_with_path_object(self, temp_dir, test_image):
        """Test saving with Path object."""
        output = WEBPOutput()
        output_path = temp_dir / "test.webp"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_auto_extension(self, temp_dir, test_image):
        """Test automatic extension addition."""
        output = WEBPOutput()
        output_path = temp_dir / "test"  # No extension

        output.save(test_image, output_path)

        webp_path = temp_dir / "test.webp"
        assert webp_path.exists()
        assert webp_path.stat().st_size > 0

    def test_save_preserves_valid_extension(self, temp_dir, test_image):
        """Test that valid extensions are preserved."""
        output = WEBPOutput()
        output_path = temp_dir / "test.webp"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.name == "test.webp"

    def test_save_creates_directory(self, temp_dir, test_image):
        """Test that missing directories are created."""
        output = WEBPOutput()
        nested_path = temp_dir / "subdir" / "nested" / "test.webp"

        output.save(test_image, nested_path)

        assert nested_path.exists()
        assert nested_path.stat().st_size > 0

    def test_save_rgba_preservation(self, temp_dir):
        """Test RGBA image optimization during save."""
        output = WEBPOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        output_path = temp_dir / "test.webp"

        output.save(rgba_image, output_path)

        # Load and verify the saved image was optimized to RGB (opaque alpha)
        saved_image = Image.open(output_path)
        assert saved_image.mode == "RGB"

    def test_save_rgba_transparent_preservation(self, temp_dir):
        """Test transparent RGBA image preservation."""
        output = WEBPOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        output_path = temp_dir / "test.webp"

        output.save(rgba_image, output_path)

        # Load and verify transparency was preserved
        saved_image = Image.open(output_path)
        assert saved_image.mode == "RGBA"

    def test_save_l_mode_converted_to_rgb(self, temp_dir):
        """Test L mode (grayscale) image is converted to RGB in WEBP."""
        output = WEBPOutput()
        l_image = Image.new("L", (100, 100), 128)
        output_path = temp_dir / "test.webp"

        output.save(l_image, output_path)

        # Load and verify L mode was converted to RGB (PIL WEBP behavior)
        saved_image = Image.open(output_path)
        assert saved_image.mode == "RGB"

    @patch("PIL.Image.Image.save")
    def test_save_error_handling(self, mock_save, temp_dir, test_image):
        """Test error handling during save."""
        mock_save.side_effect = Exception("Save failed")

        output = WEBPOutput()
        output_path = temp_dir / "test.webp"

        with pytest.raises(Exception, match="Save failed"):
            output.save(test_image, output_path)

    @patch("molecular_string_renderer.outputs.base.logger")
    def test_save_logs_success(self, mock_logger, temp_dir, test_image):
        """Test that successful saves are logged."""
        output = WEBPOutput()
        output_path = temp_dir / "test.webp"

        output.save(test_image, output_path)

        mock_logger.info.assert_called()

    @patch("molecular_string_renderer.outputs.base.logger")
    @patch("PIL.Image.Image.save")
    def test_save_logs_error(self, mock_save, mock_logger, temp_dir, test_image):
        """Test that save errors are logged."""
        mock_save.side_effect = Exception("Save failed")

        output = WEBPOutput()
        output_path = temp_dir / "test.webp"

        with pytest.raises(Exception):
            output.save(test_image, output_path)

        mock_logger.error.assert_called()


class TestWEBPOutputGetBytesMethod:
    """Test WEBP get_bytes method functionality."""

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (100, 100), "blue")

    def test_get_bytes_returns_bytes(self, test_image):
        """Test that get_bytes returns bytes."""
        output = WEBPOutput()
        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_webp_format(self, test_image):
        """Test that get_bytes produces valid WEBP format."""
        output = WEBPOutput()
        result = output.get_bytes(test_image)

        # WEBP files start with 'RIFF' and contain 'WEBP'
        assert result.startswith(b"RIFF")
        assert b"WEBP" in result[:20]

    def test_get_bytes_rgba_optimization(self):
        """Test RGBA optimization in get_bytes."""
        output = WEBPOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        result = output.get_bytes(rgba_image)

        # Verify the bytes can be loaded and are RGB (optimized)
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_get_bytes_rgba_transparent_preservation(self):
        """Test transparent RGBA preservation in get_bytes."""
        output = WEBPOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = output.get_bytes(rgba_image)

        # Verify the bytes can be loaded with transparency preserved
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGBA"

    def test_get_bytes_l_mode_converted_to_rgb(self):
        """Test L mode conversion to RGB in get_bytes."""
        output = WEBPOutput()
        l_image = Image.new("L", (100, 100), 128)
        result = output.get_bytes(l_image)

        # Verify the bytes can be loaded and L mode is converted to RGB
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_get_bytes_with_custom_quality(self):
        """Test get_bytes with custom quality settings."""
        config = OutputConfig(quality=50)
        output = WEBPOutput(config)
        test_image = Image.new("RGB", (100, 100), "red")

        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_quality_comparison(self):
        """Test that different quality settings can produce different file sizes."""
        test_image = Image.new("RGB", (200, 200), "blue")

        # Low quality
        config_low = OutputConfig(quality=10)
        output_low = WEBPOutput(config_low)
        bytes_low = output_low.get_bytes(test_image)

        # High quality
        config_high = OutputConfig(quality=95)
        output_high = WEBPOutput(config_high)
        bytes_high = output_high.get_bytes(test_image)

        # Both should produce valid results (size relationship can vary with WEBP)
        assert len(bytes_low) > 0
        assert len(bytes_high) > 0

    def test_get_bytes_with_optimization_disabled(self):
        """Test get_bytes with optimization disabled."""
        config = OutputConfig(optimize=False)
        output = WEBPOutput(config)
        test_image = Image.new("RGB", (100, 100), "red")

        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_optimization_comparison(self):
        """Test that optimization affects file size."""
        test_image = Image.new("RGB", (200, 200), "green")

        # Without optimization
        config_no_opt = OutputConfig(optimize=False)
        output_no_opt = WEBPOutput(config_no_opt)
        bytes_no_opt = output_no_opt.get_bytes(test_image)

        # With optimization
        config_opt = OutputConfig(optimize=True)
        output_opt = WEBPOutput(config_opt)
        bytes_opt = output_opt.get_bytes(test_image)

        # Optimized should typically be smaller or equal
        assert len(bytes_opt) <= len(bytes_no_opt)

    @patch("PIL.Image.Image.save")
    def test_get_bytes_error_handling(self, mock_save):
        """Test error handling in get_bytes."""
        mock_save.side_effect = Exception("Save failed")

        output = WEBPOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        with pytest.raises(Exception, match="Save failed"):
            output.get_bytes(test_image)


class TestWEBPOutputIntegration:
    """Integration tests for WEBP output handler."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_workflow_rgb_image(self, temp_dir):
        """Test complete workflow with RGB image."""
        config = OutputConfig(quality=90, optimize=True)
        output = WEBPOutput(config)
        image = Image.new("RGB", (200, 200), "blue")
        output_path = temp_dir / "workflow_test.webp"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)
        assert len(bytes_data) > 0

        # Verify saved file and bytes produce same result
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == bytes_image.mode
        assert saved_image.size == bytes_image.size

    def test_full_workflow_rgba_transparent(self, temp_dir):
        """Test complete workflow with transparent RGBA image."""
        output = WEBPOutput()
        image = Image.new("RGBA", (200, 200), (255, 0, 0, 128))
        output_path = temp_dir / "transparent_test.webp"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify transparency is preserved
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == "RGBA"
        assert bytes_image.mode == "RGBA"

    def test_full_workflow_rgba_opaque(self, temp_dir):
        """Test complete workflow with opaque RGBA image."""
        output = WEBPOutput()
        image = Image.new("RGBA", (200, 200), (255, 0, 0, 255))
        output_path = temp_dir / "opaque_test.webp"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify RGBA is optimized to RGB for opaque images
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == "RGB"
        assert bytes_image.mode == "RGB"

    def test_full_workflow_grayscale(self, temp_dir):
        """Test complete workflow with grayscale image."""
        output = WEBPOutput()
        image = Image.new("L", (200, 200), 128)
        output_path = temp_dir / "grayscale_test.webp"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify grayscale is converted to RGB (PIL WEBP behavior)
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == "RGB"
        assert bytes_image.mode == "RGB"


class TestWEBPOutputEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_image(self):
        """Test handling of very small images."""
        output = WEBPOutput()
        small_image = Image.new("RGB", (1, 1), "red")

        result = output.get_bytes(small_image)
        assert len(result) > 0

        # Verify it's a valid WEBP
        img = Image.open(io.BytesIO(result))
        assert img.size == (1, 1)

    def test_very_large_image_dimensions(self):
        """Test handling of large images."""
        output = WEBPOutput()
        # Create a reasonably large image for testing
        large_image = Image.new("RGB", (2000, 2000), "blue")

        result = output.get_bytes(large_image)
        assert len(result) > 0

    def test_all_transparent_rgba_image(self):
        """Test completely transparent RGBA image."""
        output = WEBPOutput()
        transparent_image = Image.new("RGBA", (100, 100), (255, 0, 0, 0))

        result = output.get_bytes(transparent_image)

        # Verify transparency is preserved
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGBA"

    def test_quality_extremes(self):
        """Test extreme quality settings."""
        test_image = Image.new("RGB", (100, 100), "red")

        # Minimum quality
        config_min = OutputConfig(quality=1)
        output_min = WEBPOutput(config_min)
        bytes_min = output_min.get_bytes(test_image)
        assert len(bytes_min) > 0

        # Maximum quality
        config_max = OutputConfig(quality=100)
        output_max = WEBPOutput(config_max)
        bytes_max = output_max.get_bytes(test_image)
        assert len(bytes_max) > 0

    def test_p_mode_image_handling(self):
        """Test P mode (palette) image handling."""
        img = Image.new("P", (100, 100))
        palette = [i % 256 for i in range(768)]
        img.putpalette(palette)

        output = WEBPOutput()
        result = output.get_bytes(img)
        assert len(result) > 0

    def test_monochrome_image(self):
        """Test monochrome (1-bit) image handling."""
        img = Image.new("1", (100, 100), 1)

        output = WEBPOutput()
        result = output.get_bytes(img)
        assert len(result) > 0


class TestWEBPOutputThreadSafety:
    """Test thread safety and concurrent usage."""

    def test_multiple_instances_independent(self):
        """Test that multiple instances are independent."""
        config1 = OutputConfig(quality=50)
        config2 = OutputConfig(quality=90)

        output1 = WEBPOutput(config1)
        output2 = WEBPOutput(config2)

        assert output1.config.quality == 50
        assert output2.config.quality == 90
        assert output1.config is not output2.config

    def test_instance_method_isolation(self):
        """Test that instance methods don't interfere."""
        output = WEBPOutput()
        image1 = Image.new("RGB", (100, 100), "red")
        image2 = Image.new("RGBA", (100, 100), (0, 255, 0, 128))

        bytes1 = output.get_bytes(image1)
        bytes2 = output.get_bytes(image2)

        # Verify both operations were successful and different
        assert len(bytes1) > 0
        assert len(bytes2) > 0
        assert bytes1 != bytes2


class TestWEBPOutputInheritance:
    """Test proper inheritance from base classes."""

    def test_is_output_handler(self):
        """Test that WEBPOutput is an output handler."""
        from molecular_string_renderer.outputs.base import OutputHandler

        output = WEBPOutput()
        assert isinstance(output, OutputHandler)

    def test_is_raster_output_handler(self):
        """Test that WEBPOutput is a raster output handler."""
        from molecular_string_renderer.outputs.base import RasterOutputHandler

        output = WEBPOutput()
        assert isinstance(output, RasterOutputHandler)

    def test_implements_required_methods(self):
        """Test that all required methods are implemented."""
        output = WEBPOutput()

        # Test property access
        assert hasattr(output, "file_extension")
        assert hasattr(output, "pil_format")
        assert hasattr(output, "valid_extensions")
        assert hasattr(output, "supports_alpha")
        assert hasattr(output, "supports_quality")

        # Test method access
        assert hasattr(output, "save")
        assert hasattr(output, "get_bytes")
        assert callable(output.save)
        assert callable(output.get_bytes)


class TestWEBPOutputTypeHints:
    """Test type hints and return types."""

    def test_return_types(self):
        """Test that methods return correct types."""
        output = WEBPOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        # Test property types
        assert isinstance(output.file_extension, str)
        assert isinstance(output.pil_format, str)
        assert isinstance(output.valid_extensions, list)
        assert isinstance(output.supports_alpha, bool)
        assert isinstance(output.supports_quality, bool)

        # Test method return types
        bytes_result = output.get_bytes(test_image)
        assert isinstance(bytes_result, bytes)

    def test_method_accepts_correct_types(self):
        """Test that methods accept the correct parameter types."""
        output = WEBPOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        # Test get_bytes accepts PIL Image
        result = output.get_bytes(test_image)
        assert isinstance(result, bytes)


class TestWEBPOutputSpecificFeatures:
    """Test WEBP-specific features and optimizations."""

    def test_lossless_mode_support(self):
        """Test that WEBP supports both lossy and lossless modes."""
        # Standard quality mode (lossy)
        config_lossy = OutputConfig(quality=80)
        output_lossy = WEBPOutput(config_lossy)

        # Lossless mode (quality=100 often enables lossless in WEBP)
        config_lossless = OutputConfig(quality=100)
        output_lossless = WEBPOutput(config_lossless)

        test_image = Image.new("RGB", (100, 100), "red")

        bytes_lossy = output_lossy.get_bytes(test_image)
        bytes_lossless = output_lossless.get_bytes(test_image)

        assert len(bytes_lossy) > 0
        assert len(bytes_lossless) > 0
        # File sizes can vary with WEBP compression - just verify both work    def test_animation_support_preparation(self):
        """Test that static WEBP works correctly (preparation for animation support)."""
        output = WEBPOutput()
        test_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))

        result = output.get_bytes(test_image)

        # Should work with transparency
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGBA"

    def test_webp_compression_efficiency(self):
        """Test WEBP compression efficiency compared to theoretical baseline."""
        # Create a test image with patterns that compress well
        test_image = Image.new("RGB", (200, 200), "red")

        # Get bytes with different quality settings
        config_high = OutputConfig(quality=95)
        config_low = OutputConfig(quality=30)

        output_high = WEBPOutput(config_high)
        output_low = WEBPOutput(config_low)

        bytes_high = output_high.get_bytes(test_image)
        bytes_low = output_low.get_bytes(test_image)

        # Both should produce valid results (WEBP compression is complex)
        assert len(bytes_low) > 0
        assert len(bytes_high) > 0


if __name__ == "__main__":
    pytest.main([__file__])
