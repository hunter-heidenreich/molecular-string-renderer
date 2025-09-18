"""
Test suite for JPEG output handler.

Comprehensive tests for JPEGOutput class functionality, edge cases, and error handling.
Tests alpha channel handling, quality settings, and JPEG-specific optimizations.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs.raster import JPEGOutput


class TestJPEGOutputProperties:
    """Test JPEG output handler properties."""

    def test_file_extension(self):
        """Test file extension property."""
        output = JPEGOutput()
        assert output.file_extension == ".jpg"

    def test_pil_format(self):
        """Test PIL format property."""
        output = JPEGOutput()
        assert output.pil_format == "JPEG"

    def test_valid_extensions(self):
        """Test valid extensions property."""
        output = JPEGOutput()
        assert output.valid_extensions == [".jpg", ".jpeg"]

    def test_supports_alpha(self):
        """Test alpha channel support (JPEG does not support alpha)."""
        output = JPEGOutput()
        assert output.supports_alpha is False

    def test_supports_quality(self):
        """Test quality parameter support."""
        output = JPEGOutput()
        assert output.supports_quality is True

    def test_format_name_inherited(self):
        """Test format name is inherited from base class."""
        output = JPEGOutput()
        assert output.format_name == "JPEG"


class TestJPEGOutputInitialization:
    """Test JPEG output handler initialization."""

    def test_init_without_config(self):
        """Test initialization without explicit config."""
        output = JPEGOutput()
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = OutputConfig(quality=85, optimize=False)
        output = JPEGOutput(config)
        assert output.config is config
        assert output.config.quality == 85
        assert output.config.optimize is False

    def test_init_with_none_config(self):
        """Test initialization with None config."""
        output = JPEGOutput(None)
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)


class TestJPEGOutputAlphaChannelHandling:
    """Test JPEG alpha channel handling (conversion from RGBA/LA to RGB)."""

    @pytest.fixture
    def rgb_image(self):
        """Create a test RGB image."""
        return Image.new("RGB", (100, 100), "red")

    @pytest.fixture
    def rgba_opaque_image(self):
        """Create an RGBA image with no transparency."""
        return Image.new("RGBA", (100, 100), (255, 0, 0, 255))

    @pytest.fixture
    def rgba_transparent_image(self):
        """Create an RGBA image with transparency."""
        return Image.new("RGBA", (100, 100), (255, 0, 0, 128))

    @pytest.fixture
    def la_opaque_image(self):
        """Create an LA grayscale image with no transparency."""
        return Image.new("LA", (100, 100), (128, 255))

    @pytest.fixture
    def la_transparent_image(self):
        """Create an LA grayscale image with transparency."""
        return Image.new("LA", (100, 100), (128, 200))

    def test_prepare_image_rgb_unchanged(self, rgb_image):
        """Test RGB image preparation (should remain unchanged)."""
        output = JPEGOutput()
        result = output._prepare_image(rgb_image)
        assert result.mode == "RGB"
        assert result is rgb_image  # Should be the same object

    def test_prepare_image_rgba_converts_to_rgb(self, rgba_opaque_image):
        """Test RGBA image converts to RGB (loses alpha channel)."""
        output = JPEGOutput()
        result = output._prepare_image(rgba_opaque_image)
        assert result.mode == "RGB"
        assert result is not rgba_opaque_image  # Should be different object

    def test_prepare_image_rgba_transparent_converts_to_rgb(
        self, rgba_transparent_image
    ):
        """Test transparent RGBA image converts to RGB (transparency lost)."""
        output = JPEGOutput()
        result = output._prepare_image(rgba_transparent_image)
        assert result.mode == "RGB"
        assert result is not rgba_transparent_image  # Should be different object

    def test_prepare_image_la_converts_to_rgb(self, la_opaque_image):
        """Test LA image converts to RGB for JPEG compatibility."""
        output = JPEGOutput()
        result = output._prepare_image(la_opaque_image)
        assert result.mode == "RGB"
        assert result is not la_opaque_image  # Should be different object

    def test_prepare_image_la_transparent_converts_to_rgb(self, la_transparent_image):
        """Test transparent LA image converts to RGB (transparency lost)."""
        output = JPEGOutput()
        result = output._prepare_image(la_transparent_image)
        assert result.mode == "RGB"
        assert result is not la_transparent_image  # Should be different object

    def test_prepare_image_l_mode(self):
        """Test L mode (grayscale) image handling."""
        img = Image.new("L", (100, 100), 128)
        output = JPEGOutput()
        result = output._prepare_image(img)
        # L mode is supported by JPEG, so should remain unchanged
        assert result.mode == "L"
        assert result is img  # Should be the same object

    def test_prepare_image_edge_case_p_mode(self):
        """Test P mode (palette) image preparation."""
        img = Image.new("P", (100, 100))
        # Create a valid palette (values must be 0-255)
        palette = [i % 256 for i in range(768)]  # Create a palette with valid values
        img.putpalette(palette)
        output = JPEGOutput()
        result = output._prepare_image(img)
        # P mode should be converted to RGB for JPEG
        assert result.mode == "RGB"
        assert result is not img  # Should be different object

    def test_prepare_image_cmyk_mode(self):
        """Test CMYK mode image preparation."""
        img = Image.new("CMYK", (100, 100), (0, 100, 100, 0))
        output = JPEGOutput()
        result = output._prepare_image(img)
        # CMYK is supported by JPEG, should remain unchanged
        assert result.mode == "CMYK"
        assert result is img  # Should be the same object


class TestJPEGOutputSaveKwargs:
    """Test JPEG save keyword arguments generation."""

    def test_get_save_kwargs_default_config(self):
        """Test save kwargs with default config."""
        output = JPEGOutput()
        kwargs = output._get_save_kwargs()

        expected = {
            "format": "JPEG",
            "optimize": True,  # Default from OutputConfig
            "quality": 95,  # Default from OutputConfig
        }
        assert kwargs == expected

    def test_get_save_kwargs_custom_config(self):
        """Test save kwargs with custom config."""
        config = OutputConfig(quality=80, optimize=False)
        output = JPEGOutput(config)
        kwargs = output._get_save_kwargs()

        expected = {
            "format": "JPEG",
            "optimize": False,
            "quality": 80,
        }
        assert kwargs == expected

    def test_get_save_kwargs_edge_case_quality_bounds(self):
        """Test save kwargs with quality at bounds."""
        # Test minimum quality
        config = OutputConfig(quality=1)
        output = JPEGOutput(config)
        kwargs = output._get_save_kwargs()
        assert kwargs["quality"] == 1

        # Test maximum quality
        config = OutputConfig(quality=100)
        output = JPEGOutput(config)
        kwargs = output._get_save_kwargs()
        assert kwargs["quality"] == 100

    def test_get_save_kwargs_medium_quality(self):
        """Test save kwargs with medium quality setting."""
        config = OutputConfig(quality=50)
        output = JPEGOutput(config)
        kwargs = output._get_save_kwargs()
        assert kwargs["quality"] == 50
        assert kwargs["optimize"] is True  # Default optimize


class TestJPEGOutputSaveMethod:
    """Test JPEG save method functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def test_image(self):
        """Create a test RGB image."""
        return Image.new("RGB", (100, 100), "red")

    def test_save_with_string_path(self, temp_dir, test_image):
        """Test saving with string path."""
        output = JPEGOutput()
        output_path = str(temp_dir / "test.jpg")

        output.save(test_image, output_path)

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_save_with_path_object(self, temp_dir, test_image):
        """Test saving with Path object."""
        output = JPEGOutput()
        output_path = temp_dir / "test.jpg"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_auto_extension_jpg(self, temp_dir, test_image):
        """Test automatic .jpg extension addition."""
        output = JPEGOutput()
        output_path = temp_dir / "test"  # No extension

        output.save(test_image, output_path)

        jpg_path = temp_dir / "test.jpg"
        assert jpg_path.exists()
        assert jpg_path.stat().st_size > 0

    def test_save_preserves_jpg_extension(self, temp_dir, test_image):
        """Test that .jpg extensions are preserved."""
        output = JPEGOutput()
        output_path = temp_dir / "test.jpg"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.name == "test.jpg"

    def test_save_preserves_jpeg_extension(self, temp_dir, test_image):
        """Test that .jpeg extensions are preserved."""
        output = JPEGOutput()
        output_path = temp_dir / "test.jpeg"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.name == "test.jpeg"

    def test_save_creates_directory(self, temp_dir, test_image):
        """Test that missing directories are created."""
        output = JPEGOutput()
        nested_path = temp_dir / "subdir" / "nested" / "test.jpg"

        output.save(test_image, nested_path)

        assert nested_path.exists()
        assert nested_path.stat().st_size > 0

    def test_save_rgba_conversion(self, temp_dir):
        """Test RGBA image conversion during save."""
        output = JPEGOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        output_path = temp_dir / "test.jpg"

        output.save(rgba_image, output_path)

        # Load and verify the saved image was converted to RGB
        saved_image = Image.open(output_path)
        assert saved_image.mode == "RGB"

    def test_save_rgba_transparent_conversion(self, temp_dir):
        """Test transparent RGBA image conversion (transparency lost)."""
        output = JPEGOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        output_path = temp_dir / "test.jpg"

        output.save(rgba_image, output_path)

        # Load and verify transparency was lost (converted to RGB)
        saved_image = Image.open(output_path)
        assert saved_image.mode == "RGB"

    def test_save_la_conversion(self, temp_dir):
        """Test LA image conversion during save."""
        output = JPEGOutput()
        la_image = Image.new("LA", (100, 100), (128, 255))
        output_path = temp_dir / "test.jpg"

        output.save(la_image, output_path)

        # Load and verify the saved image was converted to RGB
        saved_image = Image.open(output_path)
        assert saved_image.mode == "RGB"

    def test_save_l_mode_preserved(self, temp_dir):
        """Test L mode (grayscale) image is preserved in JPEG."""
        output = JPEGOutput()
        l_image = Image.new("L", (100, 100), 128)
        output_path = temp_dir / "test.jpg"

        output.save(l_image, output_path)

        # Load and verify L mode was preserved
        saved_image = Image.open(output_path)
        assert saved_image.mode == "L"

    @patch("PIL.Image.Image.save")
    def test_save_error_handling(self, mock_save, temp_dir, test_image):
        """Test error handling during save."""
        mock_save.side_effect = IOError("Mock save error")
        output = JPEGOutput()
        output_path = temp_dir / "test.jpg"

        with pytest.raises(IOError, match="Failed to save JPEG"):
            output.save(test_image, output_path)

    @patch("molecular_string_renderer.outputs.base.logger")
    def test_save_logs_success(self, mock_logger, temp_dir, test_image):
        """Test that successful saves are logged."""
        output = JPEGOutput()
        output_path = temp_dir / "test.jpg"

        output.save(test_image, output_path)

        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]
        assert "Successfully saved JPEG" in log_call

    @patch("molecular_string_renderer.outputs.base.logger")
    @patch("PIL.Image.Image.save")
    def test_save_logs_error(self, mock_save, mock_logger, temp_dir, test_image):
        """Test that save errors are logged."""
        mock_save.side_effect = IOError("Mock save error")
        output = JPEGOutput()
        output_path = temp_dir / "test.jpg"

        with pytest.raises(IOError):
            output.save(test_image, output_path)

        mock_logger.error.assert_called_once()
        log_call = mock_logger.error.call_args[0][0]
        assert "Failed to save JPEG" in log_call


class TestJPEGOutputGetBytesMethod:
    """Test JPEG get_bytes method functionality."""

    @pytest.fixture
    def test_image(self):
        """Create a test RGB image."""
        return Image.new("RGB", (100, 100), "red")

    def test_get_bytes_returns_bytes(self, test_image):
        """Test that get_bytes returns bytes."""
        output = JPEGOutput()
        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_jpeg_format(self, test_image):
        """Test that get_bytes produces valid JPEG format."""
        output = JPEGOutput()
        result = output.get_bytes(test_image)

        # JPEG files start with specific magic bytes
        assert result.startswith(b"\xff\xd8\xff")

    def test_get_bytes_rgba_conversion(self):
        """Test RGBA conversion in get_bytes."""
        output = JPEGOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        result = output.get_bytes(rgba_image)

        # Verify the bytes can be loaded and are RGB
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_get_bytes_rgba_transparent_conversion(self):
        """Test transparent RGBA conversion in get_bytes."""
        output = JPEGOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = output.get_bytes(rgba_image)

        # Verify the bytes can be loaded and transparency is lost
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_get_bytes_la_conversion(self):
        """Test LA conversion in get_bytes."""
        output = JPEGOutput()
        la_image = Image.new("LA", (100, 100), (128, 255))
        result = output.get_bytes(la_image)

        # Verify the bytes can be loaded and are RGB
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_get_bytes_l_mode_preserved(self):
        """Test L mode preservation in get_bytes."""
        output = JPEGOutput()
        l_image = Image.new("L", (100, 100), 128)
        result = output.get_bytes(l_image)

        # Verify the bytes can be loaded and L mode is preserved
        img = Image.open(io.BytesIO(result))
        assert img.mode == "L"

    def test_get_bytes_with_custom_quality(self):
        """Test get_bytes with custom quality settings."""
        config = OutputConfig(quality=50)
        output = JPEGOutput(config)
        test_image = Image.new("RGB", (100, 100), "red")

        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_quality_comparison(self):
        """Test that different quality settings produce different file sizes."""
        test_image = Image.new("RGB", (200, 200), "blue")

        # Low quality
        config_low = OutputConfig(quality=10)
        output_low = JPEGOutput(config_low)
        bytes_low = output_low.get_bytes(test_image)

        # High quality
        config_high = OutputConfig(quality=95)
        output_high = JPEGOutput(config_high)
        bytes_high = output_high.get_bytes(test_image)

        # High quality should produce larger files
        assert len(bytes_high) > len(bytes_low)

    def test_get_bytes_with_optimization_disabled(self):
        """Test get_bytes with optimization disabled."""
        config = OutputConfig(optimize=False)
        output = JPEGOutput(config)
        test_image = Image.new("RGB", (100, 100), "red")

        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_optimization_comparison(self):
        """Test that optimization affects file size."""
        test_image = Image.new("RGB", (200, 200), "green")

        # Without optimization
        config_no_opt = OutputConfig(optimize=False)
        output_no_opt = JPEGOutput(config_no_opt)
        bytes_no_opt = output_no_opt.get_bytes(test_image)

        # With optimization
        config_opt = OutputConfig(optimize=True)
        output_opt = JPEGOutput(config_opt)
        bytes_opt = output_opt.get_bytes(test_image)

        # Optimized should typically be smaller or equal
        assert len(bytes_opt) <= len(bytes_no_opt)

    @patch("PIL.Image.Image.save")
    def test_get_bytes_error_handling(self, mock_save):
        """Test error handling in get_bytes."""
        mock_save.side_effect = IOError("Mock save error")
        output = JPEGOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        with pytest.raises(IOError):
            output.get_bytes(test_image)


class TestJPEGOutputIntegration:
    """Integration tests for JPEG output handler."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_full_workflow_rgb_image(self, temp_dir):
        """Test complete workflow with RGB image."""
        config = OutputConfig(quality=90, optimize=True)
        output = JPEGOutput(config)
        image = Image.new("RGB", (200, 200), "blue")
        output_path = temp_dir / "workflow_test.jpg"

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

    def test_full_workflow_rgba_conversion(self, temp_dir):
        """Test complete workflow with RGBA image (alpha channel lost)."""
        output = JPEGOutput()
        image = Image.new("RGBA", (200, 200), (255, 0, 0, 128))
        output_path = temp_dir / "rgba_test.jpg"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify alpha channel was removed
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == "RGB"
        assert bytes_image.mode == "RGB"

    def test_full_workflow_la_conversion(self, temp_dir):
        """Test complete workflow with LA image (converted to RGB)."""
        output = JPEGOutput()
        image = Image.new("LA", (200, 200), (128, 255))
        output_path = temp_dir / "la_test.jpg"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify conversion to RGB
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == "RGB"
        assert bytes_image.mode == "RGB"

    def test_full_workflow_l_mode_preserved(self, temp_dir):
        """Test complete workflow with L mode (grayscale) image."""
        output = JPEGOutput()
        image = Image.new("L", (200, 200), 128)
        output_path = temp_dir / "l_mode_test.jpg"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify L mode is preserved
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == "L"
        assert bytes_image.mode == "L"

    def test_full_workflow_both_extensions(self, temp_dir):
        """Test workflow with both .jpg and .jpeg extensions."""
        output = JPEGOutput()
        image = Image.new("RGB", (100, 100), "purple")

        # Test .jpg extension
        jpg_path = temp_dir / "test.jpg"
        output.save(image, jpg_path)
        assert jpg_path.exists()

        # Test .jpeg extension
        jpeg_path = temp_dir / "test.jpeg"
        output.save(image, jpeg_path)
        assert jpeg_path.exists()

        # Both files should be valid JPEG
        jpg_image = Image.open(jpg_path)
        jpeg_image = Image.open(jpeg_path)
        assert jpg_image.format == "JPEG"
        assert jpeg_image.format == "JPEG"


class TestJPEGOutputEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_image(self):
        """Test with very small image (1x1 pixel)."""
        output = JPEGOutput()
        image = Image.new("RGB", (1, 1), "red")

        bytes_data = output.get_bytes(image)
        assert len(bytes_data) > 0

        # Verify it's still a valid JPEG
        result_image = Image.open(io.BytesIO(bytes_data))
        assert result_image.size == (1, 1)
        assert result_image.format == "JPEG"

    def test_very_large_image_dimensions(self):
        """Test with large image dimensions."""
        output = JPEGOutput()
        # Create a relatively large image (but not too large for CI)
        image = Image.new("RGB", (1000, 1000), "green")

        bytes_data = output.get_bytes(image)
        assert len(bytes_data) > 0

    def test_complex_rgba_pattern_conversion(self):
        """Test complex RGBA patterns are properly converted."""
        output = JPEGOutput()
        image = Image.new("RGBA", (100, 100), (255, 255, 255, 255))

        # Create a complex pattern with alpha
        pixels = image.load()
        for i in range(100):
            for j in range(100):
                alpha = (i + j) % 256
                pixels[i, j] = (i % 256, j % 256, (i * j) % 256, alpha)

        result = output._prepare_image(image)
        assert result.mode == "RGB"

        bytes_data = output.get_bytes(image)
        result_image = Image.open(io.BytesIO(bytes_data))
        assert result_image.mode == "RGB"

    def test_monochrome_image(self):
        """Test with monochrome (1-bit) image."""
        output = JPEGOutput()
        image = Image.new("1", (100, 100), 1)

        # Monochrome should be converted for JPEG compatibility
        result = output._prepare_image(image)
        # Should be converted to a mode JPEG can handle
        assert result.mode in ("RGB", "L")

        bytes_data = output.get_bytes(image)
        result_image = Image.open(io.BytesIO(bytes_data))
        assert result_image.mode in ("RGB", "L")

    def test_quality_extremes_file_size(self):
        """Test file size differences with extreme quality values."""
        image = Image.new("RGB", (200, 200))
        # Create a detailed pattern to see quality differences
        pixels = image.load()
        for i in range(200):
            for j in range(200):
                pixels[i, j] = (i % 256, j % 256, (i + j) % 256)

        # Test minimum quality
        config_min = OutputConfig(quality=1)
        output_min = JPEGOutput(config_min)
        bytes_min = output_min.get_bytes(image)

        # Test maximum quality
        config_max = OutputConfig(quality=100)
        output_max = JPEGOutput(config_max)
        bytes_max = output_max.get_bytes(image)

        # Maximum quality should produce larger files
        assert len(bytes_max) > len(bytes_min)

        # Both should be valid JPEG
        img_min = Image.open(io.BytesIO(bytes_min))
        img_max = Image.open(io.BytesIO(bytes_max))
        assert img_min.format == "JPEG"
        assert img_max.format == "JPEG"

    def test_grayscale_vs_rgb_conversion(self):
        """Test that grayscale images stay grayscale when appropriate."""
        output = JPEGOutput()

        # L mode should stay L
        l_image = Image.new("L", (100, 100), 128)
        l_result = output._prepare_image(l_image)
        assert l_result.mode == "L"

        # LA mode should convert to RGB (not L) due to alpha
        la_image = Image.new("LA", (100, 100), (128, 255))
        la_result = output._prepare_image(la_image)
        assert la_result.mode == "RGB"

    def test_cmyk_image_handling(self):
        """Test CMYK image handling in JPEG."""
        output = JPEGOutput()
        # CMYK is supported by JPEG
        cmyk_image = Image.new("CMYK", (100, 100), (0, 100, 100, 0))

        result = output._prepare_image(cmyk_image)
        assert result.mode == "CMYK"  # Should be preserved

        bytes_data = output.get_bytes(cmyk_image)
        result_image = Image.open(io.BytesIO(bytes_data))
        assert result_image.mode == "CMYK"


class TestJPEGOutputThreadSafety:
    """Test thread safety and concurrent usage."""

    def test_multiple_instances_independent(self):
        """Test that multiple instances are independent."""
        config1 = OutputConfig(quality=50)
        config2 = OutputConfig(quality=90)

        output1 = JPEGOutput(config1)
        output2 = JPEGOutput(config2)

        assert output1.config.quality == 50
        assert output2.config.quality == 90

        # Modifying one shouldn't affect the other
        output1.config.quality = 75
        assert output2.config.quality == 90

    def test_instance_method_isolation(self):
        """Test that instance methods don't interfere."""
        output = JPEGOutput()

        image1 = Image.new("RGB", (50, 50), "red")
        image2 = Image.new("RGBA", (50, 50), (0, 255, 0, 128))

        # These operations should be independent
        result1 = output._prepare_image(image1)
        result2 = output._prepare_image(image2)

        assert result1.mode == "RGB"
        assert result2.mode == "RGB"  # RGBA converted to RGB
        assert result1 is not result2


class TestJPEGOutputInheritance:
    """Test proper inheritance from base classes."""

    def test_is_output_handler(self):
        """Test that JPEGOutput is an OutputHandler."""
        from molecular_string_renderer.outputs.base import OutputHandler

        output = JPEGOutput()
        assert isinstance(output, OutputHandler)

    def test_is_raster_output_handler(self):
        """Test that JPEGOutput is a RasterOutputHandler."""
        from molecular_string_renderer.outputs.base import RasterOutputHandler

        output = JPEGOutput()
        assert isinstance(output, RasterOutputHandler)

    def test_implements_required_methods(self):
        """Test that all required abstract methods are implemented."""
        output = JPEGOutput()

        # Test that all abstract methods can be called
        assert hasattr(output, "save")
        assert hasattr(output, "get_bytes")
        assert hasattr(output, "file_extension")
        assert hasattr(output, "pil_format")
        assert hasattr(output, "valid_extensions")
        assert hasattr(output, "supports_alpha")
        assert hasattr(output, "supports_quality")

        # Test that they return expected types
        assert isinstance(output.file_extension, str)
        assert isinstance(output.pil_format, str)
        assert isinstance(output.valid_extensions, list)
        assert isinstance(output.supports_alpha, bool)
        assert isinstance(output.supports_quality, bool)


class TestJPEGOutputTypeHints:
    """Test type hints and return types."""

    def test_return_types(self):
        """Test that methods return correct types."""
        output = JPEGOutput()
        image = Image.new("RGB", (10, 10), "white")

        # Test property return types
        assert isinstance(output.file_extension, str)
        assert isinstance(output.pil_format, str)
        assert isinstance(output.valid_extensions, list)
        assert isinstance(output.supports_alpha, bool)
        assert isinstance(output.supports_quality, bool)

        # Test method return types
        assert isinstance(output._prepare_image(image), Image.Image)
        assert isinstance(output.get_bytes(image), bytes)
        assert isinstance(output._get_save_kwargs(), dict)

    def test_method_accepts_correct_types(self):
        """Test that methods accept correct input types."""
        output = JPEGOutput()
        image = Image.new("RGB", (10, 10), "white")

        # These should not raise type errors
        output._prepare_image(image)
        output.get_bytes(image)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test both string and Path inputs
            output.save(image, str(Path(tmp_dir) / "test1.jpg"))
            output.save(image, Path(tmp_dir) / "test2.jpg")


class TestJPEGOutputCompressionAndOptimization:
    """Test JPEG-specific compression and optimization features."""

    def test_progressive_jpeg_support(self):
        """Test that JPEG can handle progressive encoding when requested."""
        # Note: This test checks that progressive parameter doesn't break anything
        output = JPEGOutput()
        image = Image.new("RGB", (200, 200), "orange")

        # Get base kwargs and add progressive
        kwargs = output._get_save_kwargs()
        kwargs["progressive"] = True

        # Should not raise an error
        buffer = io.BytesIO()
        image.save(buffer, **kwargs)
        assert buffer.tell() > 0

    def test_subsampling_parameter(self):
        """Test that JPEG handles subsampling parameters correctly."""
        output = JPEGOutput()
        image = Image.new("RGB", (200, 200), "yellow")

        # Test different subsampling values
        for subsampling in [0, 1, 2]:
            kwargs = output._get_save_kwargs()
            kwargs["subsampling"] = subsampling

            buffer = io.BytesIO()
            image.save(buffer, **kwargs)
            assert buffer.tell() > 0

    def test_optimization_with_different_image_types(self):
        """Test optimization behavior with different image types."""
        output = JPEGOutput()

        # Test with RGB
        rgb_image = Image.new("RGB", (100, 100), "red")
        rgb_bytes = output.get_bytes(rgb_image)
        assert len(rgb_bytes) > 0

        # Test with L (grayscale)
        l_image = Image.new("L", (100, 100), 128)
        l_bytes = output.get_bytes(l_image)
        assert len(l_bytes) > 0

        # Grayscale should typically be smaller
        # (though this isn't guaranteed for such simple images)
        assert len(l_bytes) <= len(rgb_bytes) * 1.5  # Allow some variance


class TestJPEGOutputErrorConditions:
    """Test error conditions and edge cases."""

    def test_invalid_image_mode_handling(self):
        """Test handling of unusual image modes."""
        output = JPEGOutput()

        # Test with unsupported modes that need conversion
        modes_to_test = ["P", "RGBA", "LA"]

        for mode in modes_to_test:
            if mode == "P":
                img = Image.new(mode, (50, 50))
                # Create a valid palette (values must be 0-255)
                palette = [i % 256 for i in range(768)]
                img.putpalette(palette)
            elif mode == "RGBA":
                img = Image.new(mode, (50, 50), (255, 128, 64, 200))
            elif mode == "LA":
                img = Image.new(mode, (50, 50), (128, 200))  # Grayscale with alpha
            else:
                img = Image.new(mode, (50, 50))

            # Should not raise an error
            result = output._prepare_image(img)
            assert result.mode in ("RGB", "L", "CMYK")  # Valid JPEG modes

    def test_zero_dimension_image(self):
        """Test handling of zero-dimension images."""
        # In modern PIL versions, zero dimensions might be allowed
        # Let's test that our handler can at least deal with very small images
        output = JPEGOutput()

        # Test 1x1 image (smallest valid size)
        tiny_image = Image.new("RGB", (1, 1), "red")
        result = output._prepare_image(tiny_image)
        assert result.mode == "RGB"

        # Test that we can get bytes from tiny image
        bytes_data = output.get_bytes(tiny_image)
        assert len(bytes_data) > 0


class TestJPEGOutputSpecificBugTests:
    """Test for specific potential bugs found during development."""

    def test_missing_prepare_image_method_bug(self):
        """Test that JPEGOutput properly handles alpha channels.

        This test specifically checks for a bug where JPEGOutput might
        not properly convert RGBA images to RGB, causing save failures.
        """
        output = JPEGOutput()

        # This should not fail - RGBA should be converted to RGB
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))

        # Should not raise an error
        try:
            result = output._prepare_image(rgba_image)
            assert result.mode == "RGB"

            # Should also save without error
            bytes_data = output.get_bytes(rgba_image)
            assert len(bytes_data) > 0

        except Exception as e:
            pytest.fail(f"JPEGOutput failed to handle RGBA image: {e}")

    def test_la_to_rgb_conversion_consistency(self):
        """Test that LA images are consistently converted to RGB, not L.

        This prevents potential bugs where LA images might sometimes
        convert to L (losing color information) and sometimes to RGB.
        """
        output = JPEGOutput()

        # Test multiple LA images
        la_images = [
            Image.new("LA", (50, 50), (0, 255)),  # Black
            Image.new("LA", (50, 50), (128, 255)),  # Gray
            Image.new("LA", (50, 50), (255, 255)),  # White
            Image.new("LA", (50, 50), (100, 200)),  # With transparency
        ]

        for la_image in la_images:
            result = output._prepare_image(la_image)
            # All should consistently convert to RGB
            assert result.mode == "RGB", (
                f"LA image converted to {result.mode} instead of RGB"
            )

    def test_p_mode_palette_preservation(self):
        """Test that P mode images with palettes are properly converted.

        This checks for bugs where palette information might be lost
        or incorrectly handled during conversion.
        """
        output = JPEGOutput()

        # Create P mode image with custom palette
        p_image = Image.new("P", (100, 100))
        # Create a rainbow palette with valid values (0-255)
        palette = []
        for i in range(256):
            palette.extend([i, (255 - i), (i * 2) % 256])
        p_image.putpalette(palette)

        # Fill with some pattern
        pixels = p_image.load()
        for i in range(100):
            for j in range(100):
                pixels[i, j] = (i + j) % 256

        result = output._prepare_image(p_image)
        assert result.mode == "RGB"

        # Verify the conversion preserved the visual information
        bytes_data = output.get_bytes(p_image)
        result_image = Image.open(io.BytesIO(bytes_data))
        assert result_image.mode == "RGB"
        assert result_image.size == (100, 100)


if __name__ == "__main__":
    pytest.main([__file__])
