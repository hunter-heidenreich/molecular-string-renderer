"""
Test suite for PNG output handler.

Comprehensive tests for PNGOutput class functionality, edge cases, and error handling.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs.raster import PNGOutput


class TestPNGOutputProperties:
    """Test PNG output handler properties."""

    def test_file_extension(self):
        """Test file extension property."""
        output = PNGOutput()
        assert output.file_extension == ".png"

    def test_pil_format(self):
        """Test PIL format property."""
        output = PNGOutput()
        assert output.pil_format == "PNG"

    def test_valid_extensions(self):
        """Test valid extensions property."""
        output = PNGOutput()
        assert output.valid_extensions == [".png"]

    def test_supports_alpha(self):
        """Test alpha channel support."""
        output = PNGOutput()
        assert output.supports_alpha is True

    def test_supports_quality(self):
        """Test quality parameter support."""
        output = PNGOutput()
        assert output.supports_quality is True

    def test_format_name_inherited(self):
        """Test format name is inherited from base class."""
        output = PNGOutput()
        assert output.format_name == "PNG"


class TestPNGOutputInitialization:
    """Test PNG output handler initialization."""

    def test_init_without_config(self):
        """Test initialization without explicit config."""
        output = PNGOutput()
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = OutputConfig(quality=85, optimize=False)
        output = PNGOutput(config)
        assert output.config is config
        assert output.config.quality == 85
        assert output.config.optimize is False

    def test_init_with_none_config(self):
        """Test initialization with None config."""
        output = PNGOutput(None)
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)


class TestPNGOutputTransparencyDetection:
    """Test PNG transparency detection functionality."""

    @pytest.fixture
    def rgb_image(self):
        """Create a test RGB image."""
        return Image.new("RGB", (100, 100), "red")

    @pytest.fixture
    def rgba_opaque_image(self):
        """Create an RGBA image with no transparency."""
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        return img

    @pytest.fixture
    def rgba_transparent_image(self):
        """Create an RGBA image with transparency."""
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        return img

    @pytest.fixture
    def rgba_mixed_transparency_image(self):
        """Create an RGBA image with partial transparency."""
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        # Add some transparent pixels
        pixels = img.load()
        for i in range(10):
            for j in range(10):
                pixels[i, j] = (255, 0, 0, 0)  # Fully transparent
        return img

    def test_has_transparency_rgb_image(self, rgb_image):
        """Test transparency detection on RGB image."""
        output = PNGOutput()
        assert output._has_transparency(rgb_image) is False

    def test_has_transparency_rgba_opaque(self, rgba_opaque_image):
        """Test transparency detection on opaque RGBA image."""
        output = PNGOutput()
        assert output._has_transparency(rgba_opaque_image) is False

    def test_has_transparency_rgba_transparent(self, rgba_transparent_image):
        """Test transparency detection on transparent RGBA image."""
        output = PNGOutput()
        assert output._has_transparency(rgba_transparent_image) is True

    def test_has_transparency_rgba_mixed(self, rgba_mixed_transparency_image):
        """Test transparency detection on mixed transparency RGBA image."""
        output = PNGOutput()
        assert output._has_transparency(rgba_mixed_transparency_image) is True

    def test_has_transparency_edge_case_mode_l(self):
        """Test transparency detection on L mode image."""
        img = Image.new("L", (100, 100), 128)
        output = PNGOutput()
        assert output._has_transparency(img) is False

    def test_has_transparency_edge_case_mode_la(self):
        """Test transparency detection on LA mode image."""
        img = Image.new("LA", (100, 100), (128, 255))
        output = PNGOutput()
        assert output._has_transparency(img) is False

    def test_has_transparency_edge_case_mode_la_transparent(self):
        """Test transparency detection on LA mode image with transparency."""
        img = Image.new("LA", (100, 100), (128, 200))
        output = PNGOutput()
        # LA mode should now be properly handled - 200 < 255 means transparency
        assert output._has_transparency(img) is True

    def test_has_transparency_la_fully_opaque(self):
        """Test transparency detection on fully opaque LA mode image."""
        img = Image.new("LA", (100, 100), (128, 255))
        output = PNGOutput()
        # Alpha = 255 means fully opaque, so no transparency
        assert output._has_transparency(img) is False

    def test_has_transparency_la_fully_transparent(self):
        """Test transparency detection on fully transparent LA mode image."""
        img = Image.new("LA", (100, 100), (128, 0))
        output = PNGOutput()
        # Alpha = 0 means fully transparent
        assert output._has_transparency(img) is True


class TestPNGOutputImagePreparation:
    """Test PNG image preparation functionality."""

    @pytest.fixture
    def rgba_opaque_image(self):
        """Create an RGBA image with no transparency."""
        return Image.new("RGBA", (100, 100), (255, 0, 0, 255))

    @pytest.fixture
    def rgba_transparent_image(self):
        """Create an RGBA image with transparency."""
        return Image.new("RGBA", (100, 100), (255, 0, 0, 128))

    @pytest.fixture
    def rgb_image(self):
        """Create an RGB image."""
        return Image.new("RGB", (100, 100), "red")

    def test_prepare_image_rgb_unchanged(self, rgb_image):
        """Test RGB image preparation (should remain unchanged)."""
        output = PNGOutput()
        result = output._prepare_image(rgb_image)
        assert result.mode == "RGB"
        assert result is rgb_image  # Should be the same object

    def test_prepare_image_rgba_opaque_converts_to_rgb(self, rgba_opaque_image):
        """Test RGBA opaque image converts to RGB for smaller files."""
        output = PNGOutput()
        result = output._prepare_image(rgba_opaque_image)
        assert result.mode == "RGB"
        assert result is not rgba_opaque_image  # Should be different object

    def test_prepare_image_rgba_transparent_unchanged(self, rgba_transparent_image):
        """Test RGBA transparent image remains unchanged."""
        output = PNGOutput()
        result = output._prepare_image(rgba_transparent_image)
        assert result.mode == "RGBA"
        assert result is rgba_transparent_image  # Should be the same object

    def test_prepare_image_edge_case_p_mode(self):
        """Test P mode image preparation."""
        img = Image.new("P", (100, 100))
        output = PNGOutput()
        result = output._prepare_image(img)
        assert result is img  # Should be unchanged

    def test_prepare_image_edge_case_l_mode(self):
        """Test L mode image preparation."""
        img = Image.new("L", (100, 100), 128)
        output = PNGOutput()
        result = output._prepare_image(img)
        assert result is img  # Should be unchanged

    def test_prepare_image_la_opaque_converts_to_rgb(self):
        """Test LA opaque image converts to L for optimal grayscale representation."""
        img = Image.new("LA", (100, 100), (128, 255))  # Fully opaque
        output = PNGOutput()
        result = output._prepare_image(img)
        assert result.mode == "L"  # Should convert to L, not RGB
        assert result is not img  # Should be different object

    def test_prepare_image_la_transparent_unchanged(self):
        """Test LA transparent image remains unchanged."""
        img = Image.new("LA", (100, 100), (128, 200))  # Has transparency
        output = PNGOutput()
        result = output._prepare_image(img)
        assert result.mode == "LA"
        assert result is img  # Should be the same object


class TestPNGOutputSaveKwargs:
    """Test PNG save keyword arguments generation."""

    def test_get_save_kwargs_default_config(self):
        """Test save kwargs with default config."""
        output = PNGOutput()
        kwargs = output._get_save_kwargs()

        expected = {
            "format": "PNG",
            "optimize": True,  # Default from OutputConfig
            "quality": 95,  # Default from OutputConfig
        }
        assert kwargs == expected

    def test_get_save_kwargs_custom_config(self):
        """Test save kwargs with custom config."""
        config = OutputConfig(quality=80, optimize=False)
        output = PNGOutput(config)
        kwargs = output._get_save_kwargs()

        expected = {
            "format": "PNG",
            "optimize": False,
            "quality": 80,
        }
        assert kwargs == expected

    def test_get_save_kwargs_edge_case_quality_bounds(self):
        """Test save kwargs with quality at bounds."""
        # Test minimum quality
        config = OutputConfig(quality=1)
        output = PNGOutput(config)
        kwargs = output._get_save_kwargs()
        assert kwargs["quality"] == 1

        # Test maximum quality
        config = OutputConfig(quality=100)
        output = PNGOutput(config)
        kwargs = output._get_save_kwargs()
        assert kwargs["quality"] == 100


class TestPNGOutputSaveMethod:
    """Test PNG save method functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (100, 100), "red")

    def test_save_with_string_path(self, temp_dir, test_image):
        """Test saving with string path."""
        output = PNGOutput()
        output_path = str(temp_dir / "test.png")

        output.save(test_image, output_path)

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_save_with_path_object(self, temp_dir, test_image):
        """Test saving with Path object."""
        output = PNGOutput()
        output_path = temp_dir / "test.png"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_auto_extension(self, temp_dir, test_image):
        """Test automatic extension addition."""
        output = PNGOutput()
        output_path = temp_dir / "test"  # No extension

        output.save(test_image, output_path)

        png_path = temp_dir / "test.png"
        assert png_path.exists()
        assert png_path.stat().st_size > 0

    def test_save_preserves_valid_extension(self, temp_dir, test_image):
        """Test that valid extensions are preserved."""
        output = PNGOutput()
        output_path = temp_dir / "test.png"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.name == "test.png"

    def test_save_creates_directory(self, temp_dir, test_image):
        """Test that missing directories are created."""
        output = PNGOutput()
        nested_path = temp_dir / "subdir" / "nested" / "test.png"

        output.save(test_image, nested_path)

        assert nested_path.exists()
        assert nested_path.stat().st_size > 0

    def test_save_rgba_optimization(self, temp_dir):
        """Test RGBA image optimization during save."""
        output = PNGOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        output_path = temp_dir / "test.png"

        output.save(rgba_image, output_path)

        # Load and verify the saved image was converted to RGB
        saved_image = Image.open(output_path)
        assert saved_image.mode == "RGB"

    def test_save_preserves_transparency(self, temp_dir):
        """Test that transparency is preserved when present."""
        output = PNGOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        output_path = temp_dir / "test.png"

        output.save(rgba_image, output_path)

        # Load and verify transparency was preserved
        saved_image = Image.open(output_path)
        assert saved_image.mode == "RGBA"

    @patch("PIL.Image.Image.save")
    def test_save_error_handling(self, mock_save, temp_dir, test_image):
        """Test error handling during save."""
        mock_save.side_effect = IOError("Mock save error")
        output = PNGOutput()
        output_path = temp_dir / "test.png"

        with pytest.raises(IOError, match="Failed to save PNG"):
            output.save(test_image, output_path)

    @patch("molecular_string_renderer.outputs.base.logger")
    def test_save_logs_success(self, mock_logger, temp_dir, test_image):
        """Test that successful saves are logged."""
        output = PNGOutput()
        output_path = temp_dir / "test.png"

        output.save(test_image, output_path)

        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]
        assert "Successfully saved PNG" in log_call

    @patch("molecular_string_renderer.outputs.base.logger")
    @patch("PIL.Image.Image.save")
    def test_save_logs_error(self, mock_save, mock_logger, temp_dir, test_image):
        """Test that save errors are logged."""
        mock_save.side_effect = IOError("Mock save error")
        output = PNGOutput()
        output_path = temp_dir / "test.png"

        with pytest.raises(IOError):
            output.save(test_image, output_path)

        mock_logger.error.assert_called_once()
        log_call = mock_logger.error.call_args[0][0]
        assert "Failed to save PNG" in log_call


class TestPNGOutputGetBytesMethod:
    """Test PNG get_bytes method functionality."""

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (100, 100), "red")

    def test_get_bytes_returns_bytes(self, test_image):
        """Test that get_bytes returns bytes."""
        output = PNGOutput()
        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_png_format(self, test_image):
        """Test that get_bytes produces valid PNG format."""
        output = PNGOutput()
        result = output.get_bytes(test_image)

        # PNG files start with specific magic bytes
        assert result.startswith(b"\x89PNG\r\n\x1a\n")

    def test_get_bytes_rgba_optimization(self):
        """Test RGBA optimization in get_bytes."""
        output = PNGOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        result = output.get_bytes(rgba_image)

        # Verify the bytes can be loaded and are RGB
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_get_bytes_preserves_transparency(self):
        """Test that transparency is preserved in get_bytes."""
        output = PNGOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = output.get_bytes(rgba_image)

        # Verify the bytes can be loaded with transparency
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGBA"

    def test_get_bytes_with_custom_quality(self):
        """Test get_bytes with custom quality settings."""
        config = OutputConfig(quality=50)
        output = PNGOutput(config)
        test_image = Image.new("RGB", (100, 100), "red")

        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_with_optimization_disabled(self):
        """Test get_bytes with optimization disabled."""
        config = OutputConfig(optimize=False)
        output = PNGOutput(config)
        test_image = Image.new("RGB", (100, 100), "red")

        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    @patch("PIL.Image.Image.save")
    def test_get_bytes_error_handling(self, mock_save):
        """Test error handling in get_bytes."""
        mock_save.side_effect = IOError("Mock save error")
        output = PNGOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        with pytest.raises(IOError):
            output.get_bytes(test_image)


class TestPNGOutputIntegration:
    """Integration tests for PNG output handler."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_full_workflow_rgb_image(self, temp_dir):
        """Test complete workflow with RGB image."""
        config = OutputConfig(quality=90, optimize=True)
        output = PNGOutput(config)
        image = Image.new("RGB", (200, 200), "blue")
        output_path = temp_dir / "workflow_test.png"

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
        output = PNGOutput()
        image = Image.new("RGBA", (200, 200), (255, 0, 0, 128))
        output_path = temp_dir / "transparent_test.png"

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
        output = PNGOutput()
        image = Image.new("RGBA", (200, 200), (255, 0, 0, 255))
        output_path = temp_dir / "opaque_test.png"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify conversion to RGB for optimization
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == "RGB"
        assert bytes_image.mode == "RGB"

    def test_full_workflow_la_transparent(self, temp_dir):
        """Test complete workflow with transparent LA image."""
        output = PNGOutput()
        image = Image.new("LA", (200, 200), (128, 200))  # Semi-transparent
        output_path = temp_dir / "la_transparent_test.png"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify transparency is preserved (LA should be converted to RGBA for PNG)
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        # PNG doesn't directly support LA, so it should be converted to RGBA
        assert saved_image.mode in ("LA", "RGBA")
        assert bytes_image.mode in ("LA", "RGBA")

    def test_full_workflow_la_opaque(self, temp_dir):
        """Test complete workflow with opaque LA image."""
        output = PNGOutput()
        image = Image.new("LA", (200, 200), (128, 255))  # Fully opaque
        output_path = temp_dir / "la_opaque_test.png"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify conversion to L for optimal grayscale representation (no transparency)
        saved_image = Image.open(output_path)
        bytes_image = Image.open(io.BytesIO(bytes_data))

        assert saved_image.mode == "L"
        assert bytes_image.mode == "L"


class TestPNGOutputEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_image(self):
        """Test with very small image (1x1 pixel)."""
        output = PNGOutput()
        image = Image.new("RGB", (1, 1), "red")

        bytes_data = output.get_bytes(image)
        assert len(bytes_data) > 0

        # Verify it's still a valid PNG
        result_image = Image.open(io.BytesIO(bytes_data))
        assert result_image.size == (1, 1)

    def test_very_large_image_dimensions(self):
        """Test with large image dimensions."""
        output = PNGOutput()
        # Create a relatively large image (but not too large for CI)
        image = Image.new("RGB", (1000, 1000), "green")

        bytes_data = output.get_bytes(image)
        assert len(bytes_data) > 0

    def test_all_transparent_rgba_image(self):
        """Test RGBA image with all pixels transparent."""
        output = PNGOutput()
        image = Image.new("RGBA", (100, 100), (255, 0, 0, 0))

        result = output._prepare_image(image)
        assert result.mode == "RGBA"  # Should preserve transparency

        bytes_data = output.get_bytes(image)
        result_image = Image.open(io.BytesIO(bytes_data))
        assert result_image.mode == "RGBA"

    def test_partially_transparent_pixel_pattern(self):
        """Test complex transparency patterns."""
        output = PNGOutput()
        image = Image.new("RGBA", (100, 100), (255, 255, 255, 255))

        # Create a checkerboard pattern of transparency
        pixels = image.load()
        for i in range(100):
            for j in range(100):
                if (i + j) % 2 == 0:
                    pixels[i, j] = (255, 0, 0, 0)  # Transparent
                else:
                    pixels[i, j] = (0, 255, 0, 255)  # Opaque

        assert output._has_transparency(image) is True
        result = output._prepare_image(image)
        assert result.mode == "RGBA"

    def test_partially_transparent_la_pixel_pattern(self):
        """Test complex LA transparency patterns."""
        output = PNGOutput()
        image = Image.new("LA", (100, 100), (128, 255))

        # Create a pattern with some transparent pixels
        pixels = image.load()
        for i in range(10):
            for j in range(10):
                pixels[i, j] = (128, 100)  # Semi-transparent pixels

        assert output._has_transparency(image) is True
        result = output._prepare_image(image)
        assert result.mode == "LA"

    def test_la_to_l_optimization_file_size(self):
        """Test that LA → L conversion produces smaller files than LA → RGB."""
        output = PNGOutput()
        la_opaque = Image.new("LA", (100, 100), (128, 255))  # Fully opaque

        # Get bytes for our optimized version (should be L mode)
        optimized_bytes = output.get_bytes(la_opaque)
        optimized_img = Image.open(io.BytesIO(optimized_bytes))
        assert optimized_img.mode == "L"

        # Compare with direct RGB conversion (the old buggy behavior)
        rgb_converted = la_opaque.convert("RGB")
        output_rgb = PNGOutput()
        rgb_bytes = output_rgb.get_bytes(rgb_converted)

        # L mode should produce smaller files than RGB for grayscale images
        assert len(optimized_bytes) < len(rgb_bytes)

        # Verify the images are visually equivalent
        # Convert both to same mode for comparison
        optimized_as_rgb = optimized_img.convert("RGB")
        rgb_img = Image.open(io.BytesIO(rgb_bytes))
        assert optimized_as_rgb.getextrema() == rgb_img.getextrema()

    def test_quality_extremes(self):
        """Test with extreme quality values."""
        image = Image.new("RGB", (100, 100), "purple")

        # Test minimum quality
        config_min = OutputConfig(quality=1)
        output_min = PNGOutput(config_min)
        bytes_min = output_min.get_bytes(image)
        assert len(bytes_min) > 0

        # Test maximum quality
        config_max = OutputConfig(quality=100)
        output_max = PNGOutput(config_max)
        bytes_max = output_max.get_bytes(image)
        assert len(bytes_max) > 0


class TestPNGOutputThreadSafety:
    """Test thread safety and concurrent usage."""

    def test_multiple_instances_independent(self):
        """Test that multiple instances are independent."""
        config1 = OutputConfig(quality=50)
        config2 = OutputConfig(quality=90)

        output1 = PNGOutput(config1)
        output2 = PNGOutput(config2)

        assert output1.config.quality == 50
        assert output2.config.quality == 90

        # Modifying one shouldn't affect the other
        output1.config.quality = 75
        assert output2.config.quality == 90

    def test_instance_method_isolation(self):
        """Test that instance methods don't interfere."""
        output = PNGOutput()

        image1 = Image.new("RGB", (50, 50), "red")
        image2 = Image.new("RGBA", (50, 50), (0, 255, 0, 128))

        # These operations should be independent
        result1 = output._prepare_image(image1)
        result2 = output._prepare_image(image2)

        assert result1.mode == "RGB"
        assert result2.mode == "RGBA"
        assert result1 is not result2


class TestPNGOutputInheritance:
    """Test proper inheritance from base classes."""

    def test_is_output_handler(self):
        """Test that PNGOutput is an OutputHandler."""
        from molecular_string_renderer.outputs.base import OutputHandler

        output = PNGOutput()
        assert isinstance(output, OutputHandler)

    def test_is_raster_output_handler(self):
        """Test that PNGOutput is a RasterOutputHandler."""
        from molecular_string_renderer.outputs.base import RasterOutputHandler

        output = PNGOutput()
        assert isinstance(output, RasterOutputHandler)

    def test_implements_required_methods(self):
        """Test that all required abstract methods are implemented."""
        output = PNGOutput()

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


class TestPNGOutputTypeHints:
    """Test type hints and return types."""

    def test_return_types(self):
        """Test that methods return correct types."""
        output = PNGOutput()
        image = Image.new("RGB", (10, 10), "white")

        # Test property return types
        assert isinstance(output.file_extension, str)
        assert isinstance(output.pil_format, str)
        assert isinstance(output.valid_extensions, list)
        assert isinstance(output.supports_alpha, bool)
        assert isinstance(output.supports_quality, bool)

        # Test method return types
        assert isinstance(output._has_transparency(image), bool)
        assert isinstance(output._prepare_image(image), Image.Image)
        assert isinstance(output.get_bytes(image), bytes)
        assert isinstance(output._get_save_kwargs(), dict)

    def test_method_accepts_correct_types(self):
        """Test that methods accept correct input types."""
        output = PNGOutput()
        image = Image.new("RGB", (10, 10), "white")

        # These should not raise type errors
        output._has_transparency(image)
        output._prepare_image(image)
        output.get_bytes(image)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test both string and Path inputs
            output.save(image, str(Path(tmp_dir) / "test1.png"))
            output.save(image, Path(tmp_dir) / "test2.png")


if __name__ == "__main__":
    pytest.main([__file__])
