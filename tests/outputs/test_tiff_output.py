"""
Test suite for TIFF output handler.

Comprehensive tests for TIFFOutput class functionality, edge cases, and error handling.
Tests alpha channel handling, compression settings, and TIFF-specific features.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs.raster import TIFFOutput


class TestTIFFOutputProperties:
    """Test TIFF output handler properties."""

    def test_file_extension(self):
        """Test file extension property."""
        output = TIFFOutput()
        assert output.file_extension == ".tiff"

    def test_pil_format(self):
        """Test PIL format property."""
        output = TIFFOutput()
        assert output.pil_format == "TIFF"

    def test_valid_extensions(self):
        """Test valid extensions property."""
        output = TIFFOutput()
        assert output.valid_extensions == [".tiff", ".tif"]

    def test_supports_alpha(self):
        """Test alpha channel support."""
        output = TIFFOutput()
        assert output.supports_alpha is True

    def test_supports_quality(self):
        """Test quality parameter support (limited to JPEG compression)."""
        output = TIFFOutput()
        assert output.supports_quality is False

    def test_format_name_inherited(self):
        """Test format name is inherited from base class."""
        output = TIFFOutput()
        assert output.format_name == "TIFF"


class TestTIFFOutputInitialization:
    """Test TIFF output handler initialization."""

    def test_init_without_config(self):
        """Test initialization without explicit config."""
        output = TIFFOutput()
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = OutputConfig(quality=85, optimize=False)
        output = TIFFOutput(config)
        assert output.config is config
        assert output.config.quality == 85
        assert output.config.optimize is False

    def test_init_with_none_config(self):
        """Test initialization with None config."""
        output = TIFFOutput(None)
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)


class TestTIFFOutputSaveKwargs:
    """Test TIFF save keyword arguments generation."""

    def test_get_save_kwargs_default_config(self):
        """Test save kwargs with default config (with compression)."""
        output = TIFFOutput()
        kwargs = output._get_save_kwargs()

        expected = {
            "format": "TIFF",
            "compression": "tiff_lzw",
        }
        assert kwargs == expected

    def test_get_save_kwargs_optimize_disabled(self):
        """Test save kwargs with optimization disabled (no compression)."""
        config = OutputConfig(optimize=False)
        output = TIFFOutput(config)
        kwargs = output._get_save_kwargs()

        expected = {
            "format": "TIFF",
        }
        assert kwargs == expected

    def test_get_save_kwargs_optimize_enabled(self):
        """Test save kwargs with optimization enabled (LZW compression)."""
        config = OutputConfig(optimize=True)
        output = TIFFOutput(config)
        kwargs = output._get_save_kwargs()

        expected = {
            "format": "TIFF",
            "compression": "tiff_lzw",
        }
        assert kwargs == expected

    def test_get_save_kwargs_quality_ignored(self):
        """Test that quality setting is ignored for TIFF (no JPEG compression)."""
        config = OutputConfig(quality=50, optimize=True)
        output = TIFFOutput(config)
        kwargs = output._get_save_kwargs()

        # Quality should not appear in kwargs (only with JPEG compression)
        expected = {
            "format": "TIFF",
            "compression": "tiff_lzw",
        }
        assert kwargs == expected
        assert "quality" not in kwargs


class TestTIFFOutputSaveMethod:
    """Test TIFF save method functionality."""

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
        output = TIFFOutput()
        output_path = str(temp_dir / "test.tiff")

        output.save(test_image, output_path)

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_save_with_path_object(self, temp_dir, test_image):
        """Test saving with Path object."""
        output = TIFFOutput()
        output_path = temp_dir / "test.tiff"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_auto_extension_tiff(self, temp_dir, test_image):
        """Test automatic .tiff extension addition."""
        output = TIFFOutput()
        output_path = temp_dir / "test"  # No extension

        output.save(test_image, output_path)

        tiff_path = temp_dir / "test.tiff"
        assert tiff_path.exists()
        assert tiff_path.stat().st_size > 0

    def test_save_preserves_tiff_extension(self, temp_dir, test_image):
        """Test that .tiff extensions are preserved."""
        output = TIFFOutput()
        output_path = temp_dir / "test.tiff"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.name == "test.tiff"

    def test_save_preserves_tif_extension(self, temp_dir, test_image):
        """Test that .tif extensions are preserved."""
        output = TIFFOutput()
        output_path = temp_dir / "test.tif"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.name == "test.tif"

    def test_save_creates_directory(self, temp_dir, test_image):
        """Test that missing directories are created."""
        output = TIFFOutput()
        nested_path = temp_dir / "subdir" / "nested" / "test.tiff"

        output.save(test_image, nested_path)

        assert nested_path.exists()
        assert nested_path.stat().st_size > 0

    def test_save_rgba_preservation(self, temp_dir):
        """Test RGBA image preservation during save."""
        output = TIFFOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        output_path = temp_dir / "test.tiff"

        output.save(rgba_image, output_path)

        # Load and verify the saved image preserves RGBA
        saved_image = Image.open(output_path)
        assert saved_image.mode == "RGBA"

    def test_save_rgba_transparent_preservation(self, temp_dir):
        """Test transparent RGBA image preservation."""
        output = TIFFOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        output_path = temp_dir / "test.tiff"

        output.save(rgba_image, output_path)

        # Load and verify transparency was preserved
        saved_image = Image.open(output_path)
        assert saved_image.mode == "RGBA"

    def test_save_l_mode_preserved(self, temp_dir):
        """Test L mode (grayscale) image is preserved in TIFF."""
        output = TIFFOutput()
        l_image = Image.new("L", (100, 100), 128)
        output_path = temp_dir / "test.tiff"

        output.save(l_image, output_path)

        # Load and verify L mode was preserved
        saved_image = Image.open(output_path)
        assert saved_image.mode == "L"

    def test_save_with_compression(self, temp_dir, test_image):
        """Test saving with LZW compression enabled."""
        config = OutputConfig(optimize=True)
        output = TIFFOutput(config)
        output_path = temp_dir / "compressed.tiff"

        output.save(test_image, output_path)

        assert output_path.exists()
        # Verify file was saved successfully
        saved_image = Image.open(output_path)
        assert saved_image.mode == test_image.mode

    def test_save_without_compression(self, temp_dir, test_image):
        """Test saving without compression."""
        config = OutputConfig(optimize=False)
        output = TIFFOutput(config)
        output_path = temp_dir / "uncompressed.tiff"

        output.save(test_image, output_path)

        assert output_path.exists()
        # Verify file was saved successfully
        saved_image = Image.open(output_path)
        assert saved_image.mode == test_image.mode

    def test_save_compression_file_size_difference(self, temp_dir, test_image):
        """Test that compression affects file size."""
        # Save with compression
        config_compressed = OutputConfig(optimize=True)
        output_compressed = TIFFOutput(config_compressed)
        compressed_path = temp_dir / "compressed.tiff"
        output_compressed.save(test_image, compressed_path)

        # Save without compression
        config_uncompressed = OutputConfig(optimize=False)
        output_uncompressed = TIFFOutput(config_uncompressed)
        uncompressed_path = temp_dir / "uncompressed.tiff"
        output_uncompressed.save(test_image, uncompressed_path)

        # Compressed file should typically be smaller
        compressed_size = compressed_path.stat().st_size
        uncompressed_size = uncompressed_path.stat().st_size
        assert compressed_size <= uncompressed_size

    @patch("PIL.Image.Image.save")
    def test_save_error_handling(self, mock_save, temp_dir, test_image):
        """Test error handling during save."""
        mock_save.side_effect = Exception("Save failed")

        output = TIFFOutput()
        output_path = temp_dir / "test.tiff"

        with pytest.raises(Exception, match="Save failed"):
            output.save(test_image, output_path)

    @patch("molecular_string_renderer.outputs.base.logger")
    def test_save_logs_success(self, mock_logger, temp_dir, test_image):
        """Test that successful saves are logged."""
        output = TIFFOutput()
        output_path = temp_dir / "test.tiff"

        output.save(test_image, output_path)

        mock_logger.info.assert_called()

    @patch("molecular_string_renderer.outputs.base.logger")
    @patch("PIL.Image.Image.save")
    def test_save_logs_error(self, mock_save, mock_logger, temp_dir, test_image):
        """Test that save errors are logged."""
        mock_save.side_effect = Exception("Save failed")

        output = TIFFOutput()
        output_path = temp_dir / "test.tiff"

        with pytest.raises(Exception):
            output.save(test_image, output_path)

        mock_logger.error.assert_called()


class TestTIFFOutputGetBytesMethod:
    """Test TIFF get_bytes method functionality."""

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (100, 100), "blue")

    def test_get_bytes_returns_bytes(self, test_image):
        """Test that get_bytes returns bytes."""
        output = TIFFOutput()
        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_tiff_format(self, test_image):
        """Test that get_bytes produces valid TIFF format."""
        output = TIFFOutput()
        result = output.get_bytes(test_image)

        # TIFF files start with specific magic bytes (II* or MM*)
        assert result.startswith(b"II*\x00") or result.startswith(b"MM\x00*")

    def test_get_bytes_rgba_preservation(self):
        """Test RGBA preservation in get_bytes."""
        output = TIFFOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        result = output.get_bytes(rgba_image)

        # Verify the bytes can be loaded and are RGBA
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGBA"

    def test_get_bytes_rgba_transparent_preservation(self):
        """Test transparent RGBA preservation in get_bytes."""
        output = TIFFOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = output.get_bytes(rgba_image)

        # Verify the bytes can be loaded with transparency preserved
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGBA"

    def test_get_bytes_l_mode_preserved(self):
        """Test L mode preservation in get_bytes."""
        output = TIFFOutput()
        l_image = Image.new("L", (100, 100), 128)
        result = output.get_bytes(l_image)

        # Verify the bytes can be loaded and L mode is preserved
        img = Image.open(io.BytesIO(result))
        assert img.mode == "L"

    def test_get_bytes_with_compression(self):
        """Test get_bytes with compression enabled."""
        config = OutputConfig(optimize=True)
        output = TIFFOutput(config)
        test_image = Image.new("RGB", (100, 100), "red")

        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_without_compression(self):
        """Test get_bytes with compression disabled."""
        config = OutputConfig(optimize=False)
        output = TIFFOutput(config)
        test_image = Image.new("RGB", (100, 100), "red")

        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_compression_comparison(self):
        """Test that compression affects file size in get_bytes."""
        test_image = Image.new("RGB", (200, 200), "green")

        # Without compression
        config_no_comp = OutputConfig(optimize=False)
        output_no_comp = TIFFOutput(config_no_comp)
        bytes_no_comp = output_no_comp.get_bytes(test_image)

        # With compression
        config_comp = OutputConfig(optimize=True)
        output_comp = TIFFOutput(config_comp)
        bytes_comp = output_comp.get_bytes(test_image)

        # Compressed should typically be smaller or equal
        assert len(bytes_comp) <= len(bytes_no_comp)

    @patch("PIL.Image.Image.save")
    def test_get_bytes_error_handling(self, mock_save):
        """Test error handling in get_bytes."""
        mock_save.side_effect = Exception("Save failed")

        output = TIFFOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        with pytest.raises(Exception, match="Save failed"):
            output.get_bytes(test_image)


class TestTIFFOutputIntegration:
    """Integration tests for TIFF output handler."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_workflow_rgb_image(self, temp_dir):
        """Test complete workflow with RGB image."""
        config = OutputConfig(optimize=True)
        output = TIFFOutput(config)
        image = Image.new("RGB", (200, 200), "blue")
        output_path = temp_dir / "workflow_test.tiff"

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
        output = TIFFOutput()
        image = Image.new("RGBA", (200, 200), (255, 0, 0, 128))
        output_path = temp_dir / "transparent_test.tiff"

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

    def test_full_workflow_grayscale(self, temp_dir):
        """Test complete workflow with grayscale image."""
        output = TIFFOutput()
        image = Image.new("L", (200, 200), 128)
        output_path = temp_dir / "grayscale_test.tiff"

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

    def test_full_workflow_both_extensions(self, temp_dir):
        """Test workflow with both .tiff and .tif extensions."""
        output = TIFFOutput()
        image = Image.new("RGB", (100, 100), "red")

        # Test .tiff extension
        tiff_path = temp_dir / "test.tiff"
        output.save(image, tiff_path)
        assert tiff_path.exists()

        # Test .tif extension
        tif_path = temp_dir / "test.tif"
        output.save(image, tif_path)
        assert tif_path.exists()

        # Both should be valid TIFF files
        tiff_image = Image.open(tiff_path)
        tif_image = Image.open(tif_path)
        assert tiff_image.mode == tif_image.mode
        assert tiff_image.size == tif_image.size


class TestTIFFOutputEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_image(self):
        """Test handling of very small images."""
        output = TIFFOutput()
        small_image = Image.new("RGB", (1, 1), "red")

        result = output.get_bytes(small_image)
        assert len(result) > 0

        # Verify it's a valid TIFF
        img = Image.open(io.BytesIO(result))
        assert img.size == (1, 1)

    def test_very_large_image_dimensions(self):
        """Test handling of large images."""
        output = TIFFOutput()
        # Create a reasonably large image for testing
        large_image = Image.new("RGB", (2000, 2000), "blue")

        result = output.get_bytes(large_image)
        assert len(result) > 0

    def test_all_transparent_rgba_image(self):
        """Test completely transparent RGBA image."""
        output = TIFFOutput()
        transparent_image = Image.new("RGBA", (100, 100), (255, 0, 0, 0))

        result = output.get_bytes(transparent_image)

        # Verify transparency is preserved
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGBA"

    def test_p_mode_image_handling(self):
        """Test P mode (palette) image handling."""
        img = Image.new("P", (100, 100))
        palette = [i % 256 for i in range(768)]
        img.putpalette(palette)

        output = TIFFOutput()
        result = output.get_bytes(img)
        assert len(result) > 0

    def test_monochrome_image(self):
        """Test monochrome (1-bit) image handling."""
        img = Image.new("1", (100, 100), 1)

        output = TIFFOutput()
        result = output.get_bytes(img)
        assert len(result) > 0

    def test_la_mode_handling(self):
        """Test LA mode (grayscale with alpha) handling."""
        img = Image.new("LA", (100, 100), (128, 200))

        output = TIFFOutput()
        result = output.get_bytes(img)
        assert len(result) > 0

        # Verify LA mode preservation
        saved_img = Image.open(io.BytesIO(result))
        assert saved_img.mode == "LA"

    def test_cmyk_mode_handling(self):
        """Test CMYK mode handling."""
        img = Image.new("CMYK", (100, 100), (0, 100, 100, 0))

        output = TIFFOutput()
        result = output.get_bytes(img)
        assert len(result) > 0


class TestTIFFOutputThreadSafety:
    """Test thread safety and concurrent usage."""

    def test_multiple_instances_independent(self):
        """Test that multiple instances are independent."""
        config1 = OutputConfig(optimize=True)
        config2 = OutputConfig(optimize=False)

        output1 = TIFFOutput(config1)
        output2 = TIFFOutput(config2)

        assert output1.config.optimize is True
        assert output2.config.optimize is False
        assert output1.config is not output2.config

    def test_instance_method_isolation(self):
        """Test that instance methods don't interfere."""
        output = TIFFOutput()
        image1 = Image.new("RGB", (100, 100), "red")
        image2 = Image.new("RGBA", (100, 100), (0, 255, 0, 128))

        bytes1 = output.get_bytes(image1)
        bytes2 = output.get_bytes(image2)

        # Verify both operations were successful and different
        assert len(bytes1) > 0
        assert len(bytes2) > 0
        assert bytes1 != bytes2


class TestTIFFOutputInheritance:
    """Test proper inheritance from base classes."""

    def test_is_output_handler(self):
        """Test that TIFFOutput is an output handler."""
        from molecular_string_renderer.outputs.base import OutputHandler

        output = TIFFOutput()
        assert isinstance(output, OutputHandler)

    def test_is_raster_output_handler(self):
        """Test that TIFFOutput is a raster output handler."""
        from molecular_string_renderer.outputs.base import RasterOutputHandler

        output = TIFFOutput()
        assert isinstance(output, RasterOutputHandler)

    def test_implements_required_methods(self):
        """Test that all required methods are implemented."""
        output = TIFFOutput()

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


class TestTIFFOutputTypeHints:
    """Test type hints and return types."""

    def test_return_types(self):
        """Test that methods return correct types."""
        output = TIFFOutput()
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
        output = TIFFOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        # Test get_bytes accepts PIL Image
        result = output.get_bytes(test_image)
        assert isinstance(result, bytes)


class TestTIFFOutputSpecificFeatures:
    """Test TIFF-specific features and compression options."""

    def test_lzw_compression_effectiveness(self):
        """Test LZW compression effectiveness on different image types."""
        # Create an image with repeating patterns (compresses well)
        pattern_image = Image.new("RGB", (200, 200))
        pixels = []
        for y in range(200):
            for x in range(200):
                # Create a checkerboard pattern
                if (x + y) % 2 == 0:
                    pixels.append((255, 0, 0))
                else:
                    pixels.append((0, 255, 0))
        pattern_image.putdata(pixels)

        # Test compression
        config_compressed = OutputConfig(optimize=True)
        config_uncompressed = OutputConfig(optimize=False)

        output_compressed = TIFFOutput(config_compressed)
        output_uncompressed = TIFFOutput(config_uncompressed)

        bytes_compressed = output_compressed.get_bytes(pattern_image)
        bytes_uncompressed = output_uncompressed.get_bytes(pattern_image)

        # Compressed should be significantly smaller for patterns
        assert len(bytes_compressed) < len(bytes_uncompressed)

    def test_multipage_tiff_preparation(self):
        """Test single-page TIFF works correctly (preparation for multipage support)."""
        output = TIFFOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        result = output.get_bytes(test_image)

        # Should work as single-page TIFF
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"
        assert img.size == (100, 100)

    def test_different_bit_depths(self):
        """Test TIFF support for different bit depths."""
        output = TIFFOutput()

        # Test different modes
        modes_and_colors = [
            ("1", 1),  # 1-bit monochrome
            ("L", 128),  # 8-bit grayscale
            ("RGB", (255, 0, 0)),  # 24-bit RGB
            ("RGBA", (255, 0, 0, 128)),  # 32-bit RGBA
        ]

        for mode, color in modes_and_colors:
            img = Image.new(mode, (50, 50), color)
            result = output.get_bytes(img)
            assert len(result) > 0

            # Verify roundtrip
            loaded_img = Image.open(io.BytesIO(result))
            assert loaded_img.mode == mode

    def test_tiff_metadata_preservation(self):
        """Test that TIFF preserves basic image properties."""
        output = TIFFOutput()
        original_image = Image.new("RGB", (150, 100), "blue")

        result = output.get_bytes(original_image)

        # Load and verify properties are preserved
        loaded_image = Image.open(io.BytesIO(result))
        assert loaded_image.size == original_image.size
        assert loaded_image.mode == original_image.mode


class TestTIFFOutputErrorConditions:
    """Test error conditions and edge cases."""

    def test_zero_dimension_handling(self):
        """Test handling of zero-dimension images."""
        output = TIFFOutput()

        # PIL allows creating zero dimensions but throws errors when saving them
        zero_width = Image.new("RGB", (0, 100), "red")
        with pytest.raises(SystemError, match="tile cannot extend outside image"):
            output.get_bytes(zero_width)

        zero_height = Image.new("RGB", (100, 0), "red")
        with pytest.raises(SystemError, match="tile cannot extend outside image"):
            output.get_bytes(zero_height)

    def test_invalid_image_mode_handling(self):
        """Test handling of unusual image modes."""
        # Most unusual modes should work with TIFF as it's very flexible
        output = TIFFOutput()

        # Test that the handler can process various modes
        modes_to_test = ["1", "L", "P", "RGB", "RGBA", "CMYK", "LA"]

        for mode in modes_to_test:
            if mode == "P":
                img = Image.new(mode, (50, 50))
                palette = [i % 256 for i in range(768)]
                img.putpalette(palette)
            elif mode == "CMYK":
                img = Image.new(mode, (50, 50), (0, 100, 100, 0))
            elif mode == "LA":
                img = Image.new(mode, (50, 50), (128, 255))
            elif mode == "RGBA":
                img = Image.new(mode, (50, 50), (255, 0, 0, 255))
            else:
                img = Image.new(
                    mode, (50, 50), 128 if mode in ("1", "L") else (255, 0, 0)
                )

            result = output.get_bytes(img)
            assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__])
