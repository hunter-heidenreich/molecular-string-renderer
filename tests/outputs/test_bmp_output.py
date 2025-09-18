"""
Test suite for BMP output handler.

Comprehensive tests for BMPOutput class functionality, edge cases, and error handling.
Tests image mode conversion, BMP limitations, and format-specific features.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs.raster import BMPOutput


class TestBMPOutputProperties:
    """Test BMP output handler properties."""

    def test_file_extension(self):
        """Test file extension property."""
        output = BMPOutput()
        assert output.file_extension == ".bmp"

    def test_pil_format(self):
        """Test PIL format property."""
        output = BMPOutput()
        assert output.pil_format == "BMP"

    def test_valid_extensions(self):
        """Test valid extensions property."""
        output = BMPOutput()
        assert output.valid_extensions == [".bmp"]

    def test_supports_alpha(self):
        """Test alpha channel support (BMP does not support alpha)."""
        output = BMPOutput()
        assert output.supports_alpha is False

    def test_supports_quality(self):
        """Test quality parameter support (BMP does not support quality)."""
        output = BMPOutput()
        assert output.supports_quality is False

    def test_format_name_inherited(self):
        """Test format name is inherited from base class."""
        output = BMPOutput()
        assert output.format_name == "BMP"


class TestBMPOutputInitialization:
    """Test BMP output handler initialization."""

    def test_init_without_config(self):
        """Test initialization without explicit config."""
        output = BMPOutput()
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = OutputConfig(quality=85, optimize=False)
        output = BMPOutput(config)
        assert output.config is config
        assert output.config.quality == 85
        assert output.config.optimize is False

    def test_init_with_none_config(self):
        """Test initialization with None config."""
        output = BMPOutput(None)
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)


class TestBMPOutputImagePreparation:
    """Test BMP image preparation functionality."""

    @pytest.fixture
    def rgb_image(self):
        """Create RGB test image."""
        return Image.new("RGB", (100, 100), (255, 0, 0))

    @pytest.fixture
    def rgba_image(self):
        """Create RGBA test image."""
        return Image.new("RGBA", (100, 100), (255, 0, 0, 128))

    @pytest.fixture
    def la_image(self):
        """Create LA test image."""
        return Image.new("LA", (100, 100), (128, 200))

    def test_prepare_image_rgb_unchanged(self, rgb_image):
        """Test RGB image preparation (should remain unchanged)."""
        output = BMPOutput()
        result = output._prepare_image(rgb_image)
        assert result.mode == "RGB"
        assert result is rgb_image  # Should be the same object

    def test_prepare_image_rgba_unchanged(self, rgba_image):
        """Test RGBA image handling (BMP can handle RGBA but flattens alpha)."""
        output = BMPOutput()
        result = output._prepare_image(rgba_image)
        assert result.mode == "RGBA"
        assert result is rgba_image  # Should be the same object

    def test_prepare_image_la_converts_to_rgb(self, la_image):
        """Test LA image converts to RGB for BMP compatibility."""
        output = BMPOutput()
        result = output._prepare_image(la_image)
        assert result.mode == "RGB"
        assert result is not la_image  # Should be different object

    def test_prepare_image_l_mode_unchanged(self):
        """Test L mode (grayscale) image remains unchanged."""
        img = Image.new("L", (100, 100), 128)
        output = BMPOutput()
        result = output._prepare_image(img)
        assert result.mode == "L"
        assert result is img  # Should be the same object

    def test_prepare_image_p_mode_unchanged(self):
        """Test P mode (palette) image remains unchanged."""
        img = Image.new("P", (100, 100))
        palette = [i % 256 for i in range(768)]
        img.putpalette(palette)
        output = BMPOutput()
        result = output._prepare_image(img)
        assert result.mode == "P"
        assert result is img  # Should be the same object

    def test_prepare_image_1_mode_unchanged(self):
        """Test 1 mode (monochrome) image remains unchanged."""
        img = Image.new("1", (100, 100), 1)
        output = BMPOutput()
        result = output._prepare_image(img)
        assert result.mode == "1"
        assert result is img  # Should be the same object

    def test_prepare_image_unusual_mode_converts_to_rgb(self):
        """Test unusual image modes are converted to RGB."""
        # Test with CMYK mode (not directly supported by BMP)
        img = Image.new("CMYK", (100, 100), (0, 100, 100, 0))
        output = BMPOutput()

        with patch("molecular_string_renderer.outputs.raster.logger") as mock_logger:
            result = output._prepare_image(img)

            assert result.mode == "RGB"
            assert result is not img  # Should be different object
            mock_logger.warning.assert_called_once()
            assert (
                "Converting image mode 'CMYK' to RGB for BMP"
                in mock_logger.warning.call_args[0][0]
            )


class TestBMPOutputSaveKwargs:
    """Test BMP save keyword arguments generation."""

    def test_get_save_kwargs_default_config(self):
        """Test save kwargs with default config (no optimization support)."""
        output = BMPOutput()
        kwargs = output._get_save_kwargs()

        expected = {
            "format": "BMP",
        }
        assert kwargs == expected

    def test_get_save_kwargs_custom_config(self):
        """Test save kwargs with custom config (optimization ignored)."""
        config = OutputConfig(quality=80, optimize=True)
        output = BMPOutput(config)
        kwargs = output._get_save_kwargs()

        # BMP doesn't support optimization or quality
        expected = {
            "format": "BMP",
        }
        assert kwargs == expected
        assert "optimize" not in kwargs
        assert "quality" not in kwargs

    def test_get_save_kwargs_all_options_ignored(self):
        """Test that all advanced options are ignored for BMP."""
        config = OutputConfig(quality=50, optimize=True)
        output = BMPOutput(config)
        kwargs = output._get_save_kwargs()

        # Only format should be present
        assert kwargs == {"format": "BMP"}
        assert len(kwargs) == 1


class TestBMPOutputSaveMethod:
    """Test BMP save method functionality."""

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
        output = BMPOutput()
        output_path = str(temp_dir / "test.bmp")

        output.save(test_image, output_path)

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_save_with_path_object(self, temp_dir, test_image):
        """Test saving with Path object."""
        output = BMPOutput()
        output_path = temp_dir / "test.bmp"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_auto_extension(self, temp_dir, test_image):
        """Test automatic extension addition."""
        output = BMPOutput()
        output_path = temp_dir / "test"  # No extension

        output.save(test_image, output_path)

        bmp_path = temp_dir / "test.bmp"
        assert bmp_path.exists()
        assert bmp_path.stat().st_size > 0

    def test_save_preserves_valid_extension(self, temp_dir, test_image):
        """Test that valid extensions are preserved."""
        output = BMPOutput()
        output_path = temp_dir / "test.bmp"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.name == "test.bmp"

    def test_save_creates_directory(self, temp_dir, test_image):
        """Test that missing directories are created."""
        output = BMPOutput()
        nested_path = temp_dir / "subdir" / "nested" / "test.bmp"

        output.save(test_image, nested_path)

        assert nested_path.exists()
        assert nested_path.stat().st_size > 0

    def test_save_rgba_flattens_alpha(self, temp_dir):
        """Test RGBA image saving (alpha channel flattened)."""
        output = BMPOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        output_path = temp_dir / "test.bmp"

        output.save(rgba_image, output_path)

        # BMP can store RGBA but alpha is typically ignored/flattened
        saved_image = Image.open(output_path)
        # PIL may convert to RGB or keep as RGBA depending on implementation
        assert saved_image.mode in ("RGB", "RGBA")

    def test_save_la_conversion(self, temp_dir):
        """Test LA image conversion during save."""
        output = BMPOutput()
        la_image = Image.new("LA", (100, 100), (128, 200))
        output_path = temp_dir / "test.bmp"

        output.save(la_image, output_path)

        # Load and verify the saved image was converted to RGB
        saved_image = Image.open(output_path)
        assert saved_image.mode == "RGB"

    def test_save_l_mode_preserved(self, temp_dir):
        """Test L mode (grayscale) image is preserved in BMP."""
        output = BMPOutput()
        l_image = Image.new("L", (100, 100), 128)
        output_path = temp_dir / "test.bmp"

        output.save(l_image, output_path)

        # Load and verify L mode was preserved
        saved_image = Image.open(output_path)
        assert saved_image.mode == "L"

    def test_save_p_mode_preserved(self, temp_dir):
        """Test P mode (palette) image is preserved in BMP."""
        output = BMPOutput()
        p_image = Image.new("P", (100, 100))
        palette = [i % 256 for i in range(768)]
        p_image.putpalette(palette)
        output_path = temp_dir / "test.bmp"

        output.save(p_image, output_path)

        # Load and verify P mode was preserved
        saved_image = Image.open(output_path)
        assert saved_image.mode == "P"

    def test_save_1_mode_preserved(self, temp_dir):
        """Test 1 mode (monochrome) image is preserved in BMP."""
        output = BMPOutput()
        mono_image = Image.new("1", (100, 100), 1)
        output_path = temp_dir / "test.bmp"

        output.save(mono_image, output_path)

        # Load and verify 1 mode was preserved
        saved_image = Image.open(output_path)
        assert saved_image.mode == "1"

    @patch("PIL.Image.Image.save")
    def test_save_error_handling(self, mock_save, temp_dir, test_image):
        """Test error handling during save."""
        mock_save.side_effect = Exception("Save failed")

        output = BMPOutput()
        output_path = temp_dir / "test.bmp"

        with pytest.raises(Exception, match="Save failed"):
            output.save(test_image, output_path)

    @patch("molecular_string_renderer.outputs.base.logger")
    def test_save_logs_success(self, mock_logger, temp_dir, test_image):
        """Test that successful saves are logged."""
        output = BMPOutput()
        output_path = temp_dir / "test.bmp"

        output.save(test_image, output_path)

        mock_logger.info.assert_called()

    @patch("molecular_string_renderer.outputs.base.logger")
    @patch("PIL.Image.Image.save")
    def test_save_logs_error(self, mock_save, mock_logger, temp_dir, test_image):
        """Test that save errors are logged."""
        mock_save.side_effect = Exception("Save failed")

        output = BMPOutput()
        output_path = temp_dir / "test.bmp"

        with pytest.raises(Exception):
            output.save(test_image, output_path)

        mock_logger.error.assert_called()


class TestBMPOutputGetBytesMethod:
    """Test BMP get_bytes method functionality."""

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (100, 100), "blue")

    def test_get_bytes_returns_bytes(self, test_image):
        """Test that get_bytes returns bytes."""
        output = BMPOutput()
        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_bmp_format(self, test_image):
        """Test that get_bytes produces valid BMP format."""
        output = BMPOutput()
        result = output.get_bytes(test_image)

        # BMP files start with 'BM' magic bytes
        assert result.startswith(b"BM")

    def test_get_bytes_rgba_handling(self):
        """Test RGBA handling in get_bytes (alpha flattened)."""
        output = BMPOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = output.get_bytes(rgba_image)

        # Verify the bytes can be loaded (alpha may be flattened)
        img = Image.open(io.BytesIO(result))
        assert img.mode in ("RGB", "RGBA")

    def test_get_bytes_la_conversion(self):
        """Test LA conversion in get_bytes."""
        output = BMPOutput()
        la_image = Image.new("LA", (100, 100), (128, 200))
        result = output.get_bytes(la_image)

        # Verify the bytes can be loaded and are RGB
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_get_bytes_l_mode_preserved(self):
        """Test L mode preservation in get_bytes."""
        output = BMPOutput()
        l_image = Image.new("L", (100, 100), 128)
        result = output.get_bytes(l_image)

        # Verify the bytes can be loaded and L mode is preserved
        img = Image.open(io.BytesIO(result))
        assert img.mode == "L"

    def test_get_bytes_p_mode_preserved(self):
        """Test P mode preservation in get_bytes."""
        output = BMPOutput()
        p_image = Image.new("P", (100, 100))
        palette = [i % 256 for i in range(768)]
        p_image.putpalette(palette)
        result = output.get_bytes(p_image)

        # Verify the bytes can be loaded and P mode is preserved
        img = Image.open(io.BytesIO(result))
        assert img.mode == "P"

    def test_get_bytes_1_mode_preserved(self):
        """Test 1 mode preservation in get_bytes."""
        output = BMPOutput()
        mono_image = Image.new("1", (100, 100), 1)
        result = output.get_bytes(mono_image)

        # Verify the bytes can be loaded and 1 mode is preserved
        img = Image.open(io.BytesIO(result))
        assert img.mode == "1"

    def test_get_bytes_no_compression_options(self):
        """Test that compression options are ignored in get_bytes."""
        config = OutputConfig(optimize=True, quality=95)
        output = BMPOutput(config)
        test_image = Image.new("RGB", (100, 100), "red")

        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    @patch("PIL.Image.Image.save")
    def test_get_bytes_error_handling(self, mock_save):
        """Test error handling in get_bytes."""
        mock_save.side_effect = Exception("Save failed")

        output = BMPOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        with pytest.raises(Exception, match="Save failed"):
            output.get_bytes(test_image)


class TestBMPOutputIntegration:
    """Integration tests for BMP output handler."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_workflow_rgb_image(self, temp_dir):
        """Test complete workflow with RGB image."""
        output = BMPOutput()
        image = Image.new("RGB", (200, 200), "blue")
        output_path = temp_dir / "workflow_test.bmp"

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

    def test_full_workflow_rgba_image(self, temp_dir):
        """Test complete workflow with RGBA image (alpha flattened)."""
        output = BMPOutput()
        image = Image.new("RGBA", (200, 200), (255, 0, 0, 128))
        output_path = temp_dir / "rgba_test.bmp"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Alpha should be handled (possibly flattened)
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode in ("RGB", "RGBA")
        assert bytes_image.mode in ("RGB", "RGBA")

    def test_full_workflow_grayscale(self, temp_dir):
        """Test complete workflow with grayscale image."""
        output = BMPOutput()
        image = Image.new("L", (200, 200), 128)
        output_path = temp_dir / "grayscale_test.bmp"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify grayscale is preserved
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == "L"
        assert bytes_image.mode == "L"

    def test_full_workflow_monochrome(self, temp_dir):
        """Test complete workflow with monochrome image."""
        output = BMPOutput()
        image = Image.new("1", (200, 200), 1)
        output_path = temp_dir / "mono_test.bmp"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify monochrome is preserved
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == "1"
        assert bytes_image.mode == "1"

    def test_full_workflow_palette(self, temp_dir):
        """Test complete workflow with palette image."""
        output = BMPOutput()
        image = Image.new("P", (200, 200))
        palette = [i % 256 for i in range(768)]
        image.putpalette(palette)
        output_path = temp_dir / "palette_test.bmp"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify palette mode is preserved
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == "P"
        assert bytes_image.mode == "P"


class TestBMPOutputEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_image(self):
        """Test handling of very small images."""
        output = BMPOutput()
        small_image = Image.new("RGB", (1, 1), "red")

        result = output.get_bytes(small_image)
        assert len(result) > 0

        # Verify it's a valid BMP
        img = Image.open(io.BytesIO(result))
        assert img.size == (1, 1)

    def test_very_large_image_dimensions(self):
        """Test handling of large images."""
        output = BMPOutput()
        # Create a reasonably large image for testing
        large_image = Image.new("RGB", (2000, 2000), "blue")

        result = output.get_bytes(large_image)
        assert len(result) > 0

    def test_all_transparent_rgba_image(self):
        """Test completely transparent RGBA image (transparency lost)."""
        output = BMPOutput()
        transparent_image = Image.new("RGBA", (100, 100), (255, 0, 0, 0))

        result = output.get_bytes(transparent_image)

        # Alpha should be handled (transparency lost in BMP)
        img = Image.open(io.BytesIO(result))
        assert img.mode in ("RGB", "RGBA")

    def test_complex_palette_image(self):
        """Test complex palette image handling."""
        output = BMPOutput()
        img = Image.new("P", (100, 100))

        # Create a more complex palette
        palette = []
        for i in range(256):
            palette.extend([i, (255 - i) % 256, (i * 2) % 256])
        img.putpalette(palette)

        # Set some pixel values
        for x in range(10):
            for y in range(10):
                img.putpixel((x, y), x + y)

        result = output.get_bytes(img)
        assert len(result) > 0

    def test_monochrome_patterns(self):
        """Test monochrome image with patterns."""
        output = BMPOutput()
        img = Image.new("1", (100, 100), 0)

        # Create a checkerboard pattern
        for x in range(100):
            for y in range(100):
                if (x + y) % 2 == 0:
                    img.putpixel((x, y), 1)

        result = output.get_bytes(img)
        assert len(result) > 0

        # Verify pattern is preserved
        loaded_img = Image.open(io.BytesIO(result))
        assert loaded_img.mode == "1"

    def test_cmyk_to_rgb_conversion(self):
        """Test CMYK image conversion to RGB."""
        output = BMPOutput()
        cmyk_image = Image.new("CMYK", (100, 100), (0, 100, 100, 0))

        with patch("molecular_string_renderer.outputs.raster.logger") as mock_logger:
            result = output.get_bytes(cmyk_image)

            assert len(result) > 0

            # Verify it was converted to RGB
            img = Image.open(io.BytesIO(result))
            assert img.mode == "RGB"

            # Verify warning was logged
            mock_logger.warning.assert_called_once()


class TestBMPOutputThreadSafety:
    """Test thread safety and concurrent usage."""

    def test_multiple_instances_independent(self):
        """Test that multiple instances are independent."""
        config1 = OutputConfig(optimize=True)
        config2 = OutputConfig(optimize=False)

        output1 = BMPOutput(config1)
        output2 = BMPOutput(config2)

        assert output1.config.optimize is True
        assert output2.config.optimize is False
        assert output1.config is not output2.config

    def test_instance_method_isolation(self):
        """Test that instance methods don't interfere."""
        output = BMPOutput()
        image1 = Image.new("RGB", (100, 100), "red")
        image2 = Image.new("L", (100, 100), 128)

        bytes1 = output.get_bytes(image1)
        bytes2 = output.get_bytes(image2)

        # Verify both operations were successful and different
        assert len(bytes1) > 0
        assert len(bytes2) > 0
        assert bytes1 != bytes2


class TestBMPOutputInheritance:
    """Test proper inheritance from base classes."""

    def test_is_output_handler(self):
        """Test that BMPOutput is an output handler."""
        from molecular_string_renderer.outputs.base import OutputHandler

        output = BMPOutput()
        assert isinstance(output, OutputHandler)

    def test_is_raster_output_handler(self):
        """Test that BMPOutput is a raster output handler."""
        from molecular_string_renderer.outputs.base import RasterOutputHandler

        output = BMPOutput()
        assert isinstance(output, RasterOutputHandler)

    def test_implements_required_methods(self):
        """Test that all required methods are implemented."""
        output = BMPOutput()

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


class TestBMPOutputTypeHints:
    """Test type hints and return types."""

    def test_return_types(self):
        """Test that methods return correct types."""
        output = BMPOutput()
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
        output = BMPOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        # Test get_bytes accepts PIL Image
        result = output.get_bytes(test_image)
        assert isinstance(result, bytes)


class TestBMPOutputSpecificFeatures:
    """Test BMP-specific features and limitations."""

    def test_no_compression_support(self):
        """Test that BMP doesn't support compression options."""
        config_optimized = OutputConfig(optimize=True)
        config_standard = OutputConfig(optimize=False)

        output_opt = BMPOutput(config_optimized)
        output_std = BMPOutput(config_standard)

        test_image = Image.new("RGB", (100, 100), "red")

        bytes_opt = output_opt.get_bytes(test_image)
        bytes_std = output_std.get_bytes(test_image)

        # File sizes should be very similar since BMP doesn't compress
        size_diff = abs(len(bytes_opt) - len(bytes_std))
        # Allow for small header differences but not significant compression
        assert size_diff < 100

    def test_no_quality_support(self):
        """Test that BMP ignores quality settings."""
        config_low = OutputConfig(quality=10)
        config_high = OutputConfig(quality=95)

        output_low = BMPOutput(config_low)
        output_high = BMPOutput(config_high)

        test_image = Image.new("RGB", (100, 100), "red")

        bytes_low = output_low.get_bytes(test_image)
        bytes_high = output_high.get_bytes(test_image)

        # Sizes should be identical since quality is ignored
        assert len(bytes_low) == len(bytes_high)

    def test_alpha_channel_handling(self):
        """Test how BMP handles alpha channels."""
        output = BMPOutput()

        # Fully opaque RGBA
        rgba_opaque = Image.new("RGBA", (50, 50), (255, 0, 0, 255))

        # Semi-transparent RGBA
        rgba_semi = Image.new("RGBA", (50, 50), (255, 0, 0, 128))

        # Fully transparent RGBA
        rgba_transparent = Image.new("RGBA", (50, 50), (255, 0, 0, 0))

        for rgba_img in [rgba_opaque, rgba_semi, rgba_transparent]:
            result = output.get_bytes(rgba_img)
            loaded_img = Image.open(io.BytesIO(result))

            # Alpha information may be lost or preserved depending on BMP variant
            assert loaded_img.mode in ("RGB", "RGBA")

    def test_bmp_file_size_predictability(self):
        """Test that BMP file sizes are predictable (uncompressed)."""
        output = BMPOutput()

        # Create images of known sizes
        small_img = Image.new("RGB", (10, 10), "red")
        large_img = Image.new("RGB", (20, 20), "red")

        small_bytes = output.get_bytes(small_img)
        large_bytes = output.get_bytes(large_img)

        # Larger image should produce larger file
        assert len(large_bytes) > len(small_bytes)

        # File size should scale roughly with pixel count (4x pixels ≈ 3-4x size)
        # Allow for headers and alignment
        ratio = len(large_bytes) / len(small_bytes)
        assert 3.0 < ratio < 5.0  # Should be close to 4x but allow for BMP overhead

    def test_pixel_format_preservation(self):
        """Test that BMP preserves pixel data accurately."""
        output = BMPOutput()

        # Create an image with specific colors
        test_img = Image.new("RGB", (3, 3))
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 128, 128),
            (255, 255, 255),
            (0, 0, 0),
        ]

        for i, color in enumerate(colors):
            x, y = i % 3, i // 3
            test_img.putpixel((x, y), color)

        result = output.get_bytes(test_img)
        loaded_img = Image.open(io.BytesIO(result))

        # Verify colors are preserved
        for i, expected_color in enumerate(colors):
            x, y = i % 3, i // 3
            actual_color = loaded_img.getpixel((x, y))
            if loaded_img.mode == "RGB":
                assert actual_color == expected_color
            # If converted to different mode, just verify it loads correctly
            assert actual_color is not None


class TestBMPOutputErrorConditions:
    """Test error conditions and edge cases."""

    def test_zero_dimension_handling(self):
        """Test handling of zero-dimension images."""
        output = BMPOutput()

        # PIL allows creating zero dimensions but throws errors when saving them
        zero_width = Image.new("RGB", (0, 100), "red")
        with pytest.raises(SystemError, match="tile cannot extend outside image"):
            output.get_bytes(zero_width)

        zero_height = Image.new("RGB", (100, 0), "red")
        with pytest.raises(SystemError, match="tile cannot extend outside image"):
            output.get_bytes(zero_height)

    def test_invalid_palette_handling(self):
        """Test handling of invalid palette data."""
        output = BMPOutput()

        # Create P mode image without palette (should still work)
        img = Image.new("P", (50, 50))
        # Don't set palette - PIL should handle this gracefully

        # Should not raise an error
        result = output.get_bytes(img)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__])
