"""
Additional tests to improve coverage for OutputHandlers.

This file focuses on testing uncovered code paths identified in the coverage report.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image
import tempfile

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import get_output_handler
from molecular_string_renderer.outputs.factory import (
    get_supported_formats,
    _validate_format,
)
from molecular_string_renderer.outputs.utils import (
    FormatInfo,
    FormatRegistry,
    ImageModeUtils,
    create_safe_filename,
    build_save_kwargs,
)
from molecular_string_renderer.outputs.svg_strategies import (
    VectorSVGStrategy,
    HybridSVGStrategy,
)


class TestFactoryErrorHandling:
    """Test error handling and edge cases in factory functions."""

    def test_validate_format_unsupported_format(self):
        """Test _validate_format with unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported output format: unsupported"):
            _validate_format("unsupported")

    def test_validate_format_case_insensitive(self):
        """Test _validate_format normalizes case."""
        assert _validate_format("PNG") == "png"
        assert _validate_format("  JPEG  ") == "jpeg"
        assert _validate_format("WebP") == "webp"

    def test_get_supported_formats(self):
        """Test get_supported_formats returns expected formats."""
        formats = get_supported_formats()
        assert isinstance(formats, list)
        assert "png" in formats
        assert "svg" in formats
        assert "pdf" in formats
        assert "jpeg" in formats
        assert "jpg" in formats  # Alias
        assert "tiff" in formats
        assert "tif" in formats  # Alias
        assert "webp" in formats
        assert "bmp" in formats


class TestFormatInfoValidation:
    """Test FormatInfo validation and edge cases."""

    def test_format_info_post_init_adds_dot(self):
        """Test that __post_init__ adds dot to extension if missing."""
        format_info = FormatInfo(
            extension="png",
            pil_format="PNG",
            valid_extensions=[".png"],
            supports_alpha=True,
            supports_quality=True,
        )
        assert format_info.extension == ".png"

    def test_format_info_post_init_adds_primary_to_valid(self):
        """Test that __post_init__ adds primary extension to valid_extensions."""
        format_info = FormatInfo(
            extension=".test",
            pil_format="TEST",
            valid_extensions=[".other"],
            supports_alpha=True,
            supports_quality=True,
        )
        assert ".test" in format_info.valid_extensions
        assert ".other" in format_info.valid_extensions


class TestFormatRegistry:
    """Test FormatRegistry functionality and error cases."""

    def test_get_format_info_unsupported(self):
        """Test get_format_info with unsupported format."""
        with pytest.raises(ValueError, match="Unknown format: unsupported"):
            FormatRegistry.get_format_info("unsupported")

    def test_get_format_info_case_insensitive(self):
        """Test get_format_info is case insensitive."""
        info = FormatRegistry.get_format_info("PNG")
        assert info.extension == ".png"
        assert info.pil_format == "PNG"

    def test_get_supported_formats(self):
        """Test get_supported_formats returns all known formats."""
        formats = FormatRegistry.get_supported_formats()
        assert "png" in formats
        assert "jpg" in formats
        assert "jpeg" in formats
        assert "svg" in formats
        assert "pdf" in formats

    def test_is_supported(self):
        """Test is_supported method."""
        assert FormatRegistry.is_supported("png")
        assert FormatRegistry.is_supported("PNG")
        assert FormatRegistry.is_supported("  jpeg  ")
        assert not FormatRegistry.is_supported("unsupported")


class TestImageModeUtilsEdgeCases:
    """Test ImageModeUtils edge cases and error handling."""

    def test_has_transparency_non_alpha_modes(self):
        """Test has_transparency with non-alpha image modes."""
        rgb_image = Image.new("RGB", (10, 10), "red")
        assert not ImageModeUtils.has_transparency(rgb_image)

        l_image = Image.new("L", (10, 10), 128)
        assert not ImageModeUtils.has_transparency(l_image)

        p_image = Image.new("P", (10, 10), 0)
        assert not ImageModeUtils.has_transparency(p_image)

    def test_has_transparency_fully_opaque_alpha(self):
        """Test has_transparency with fully opaque alpha images."""
        rgba_opaque = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        assert not ImageModeUtils.has_transparency(rgba_opaque)

        la_opaque = Image.new("LA", (10, 10), (128, 255))
        assert not ImageModeUtils.has_transparency(la_opaque)

    def test_has_transparency_with_transparency(self):
        """Test has_transparency with transparent images."""
        rgba_transparent = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        assert ImageModeUtils.has_transparency(rgba_transparent)

        la_transparent = Image.new("LA", (10, 10), (128, 200))
        assert ImageModeUtils.has_transparency(la_transparent)

    def test_prepare_for_no_alpha_edge_cases(self):
        """Test prepare_for_no_alpha with various image modes."""
        # PA mode (palette with alpha)
        pa_image = Image.new("PA", (10, 10))
        result = ImageModeUtils.prepare_for_no_alpha(pa_image)
        assert result.mode == "P"

        # Mode that doesn't need conversion
        rgb_image = Image.new("RGB", (10, 10), "red")
        result = ImageModeUtils.prepare_for_no_alpha(rgb_image)
        assert result.mode == "RGB"
        assert result is rgb_image  # Should return same object

    def test_optimize_for_format_fully_opaque_conversion(self):
        """Test optimize_for_format with fully opaque alpha images."""
        # RGBA image that's fully opaque should convert to RGB
        rgba_opaque = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        result = ImageModeUtils.optimize_for_format(rgba_opaque, supports_alpha=True)
        assert result.mode == "RGB"

        # LA image that's fully opaque should convert to L
        la_opaque = Image.new("LA", (10, 10), (128, 255))
        result = ImageModeUtils.optimize_for_format(la_opaque, supports_alpha=True)
        assert result.mode == "L"

    def test_ensure_jpeg_compatible_edge_cases(self):
        """Test ensure_jpeg_compatible with various modes."""
        # Mode "1" (bitmap)
        bitmap_image = Image.new("1", (10, 10))
        result = ImageModeUtils.ensure_jpeg_compatible(bitmap_image)
        assert result.mode == "RGB"

        # CMYK mode (should remain CMYK)
        cmyk_image = Image.new("CMYK", (10, 10))
        result = ImageModeUtils.ensure_jpeg_compatible(cmyk_image)
        assert result.mode == "CMYK"
        assert result is cmyk_image

        # P mode (palette)
        p_image = Image.new("P", (10, 10))
        result = ImageModeUtils.ensure_jpeg_compatible(p_image)
        assert result.mode == "RGB"

    @patch("molecular_string_renderer.outputs.utils.logger")
    def test_ensure_jpeg_compatible_unknown_mode_warning(self, mock_logger):
        """Test ensure_jpeg_compatible logs warning for unknown modes."""
        # Mock an image with an unusual mode
        mock_image = MagicMock()
        mock_image.mode = "UNKNOWN_MODE"
        mock_image.convert.return_value = Image.new("RGB", (10, 10))

        ImageModeUtils.ensure_jpeg_compatible(mock_image)

        # Should log warning
        mock_logger.warning.assert_called_once()
        assert "Converting image mode 'UNKNOWN_MODE' to RGB for JPEG" in str(
            mock_logger.warning.call_args
        )

    def test_ensure_bmp_compatible_edge_cases(self):
        """Test ensure_bmp_compatible with various modes."""
        # Supported modes should pass through
        for mode in ["1", "L", "P", "RGB", "RGBA"]:
            test_image = Image.new(mode, (10, 10))
            result = ImageModeUtils.ensure_bmp_compatible(test_image)
            assert result.mode == mode
            assert result is test_image

        # LA mode should convert to RGB
        la_image = Image.new("LA", (10, 10))
        result = ImageModeUtils.ensure_bmp_compatible(la_image)
        assert result.mode == "RGB"

    @patch("molecular_string_renderer.outputs.utils.logger")
    def test_ensure_bmp_compatible_unknown_mode_warning(self, mock_logger):
        """Test ensure_bmp_compatible logs warning for unknown modes."""
        mock_image = MagicMock()
        mock_image.mode = "UNKNOWN_MODE"
        mock_image.convert.return_value = Image.new("RGB", (10, 10))

        ImageModeUtils.ensure_bmp_compatible(mock_image)

        mock_logger.warning.assert_called_once()
        assert "Converting image mode 'UNKNOWN_MODE' to RGB for BMP" in str(
            mock_logger.warning.call_args
        )


class TestUtilityFunctions:
    """Test utility functions and edge cases."""

    def test_create_safe_filename_various_inputs(self):
        """Test create_safe_filename with various inputs."""
        # Basic test
        filename = create_safe_filename("CCO", ".png")
        assert filename.endswith(".png")
        assert len(filename) == 36  # 32 char hash + 4 char extension

        # Test with extension without dot
        filename = create_safe_filename("CCO", "jpg")
        assert filename.endswith(".jpg")

        # Test default extension
        filename = create_safe_filename("CCO")
        assert filename.endswith(".png")

        # Test with whitespace
        filename1 = create_safe_filename("  CCO  ")
        filename2 = create_safe_filename("CCO")
        assert filename1 == filename2  # Should strip whitespace

    def test_build_save_kwargs_quality_support(self):
        """Test build_save_kwargs with quality-supporting format."""
        format_info = FormatInfo(
            extension=".jpg",
            pil_format="JPEG",
            valid_extensions=[".jpg"],
            supports_alpha=False,
            supports_quality=True,
        )
        config = OutputConfig(quality=80, optimize=True)

        kwargs = build_save_kwargs(format_info, config)

        assert kwargs["format"] == "JPEG"
        assert kwargs["quality"] == 80
        assert kwargs["optimize"] is True

    def test_build_save_kwargs_no_quality_support_with_optimize(self):
        """Test build_save_kwargs without quality support but with optimize."""
        format_info = FormatInfo(
            extension=".png",
            pil_format="PNG",
            valid_extensions=[".png"],
            supports_alpha=True,
            supports_quality=False,
        )
        config = OutputConfig(quality=80, optimize=True)

        kwargs = build_save_kwargs(format_info, config)

        assert kwargs["format"] == "PNG"
        assert "quality" not in kwargs
        assert kwargs["optimize"] is True

    def test_build_save_kwargs_no_optimize(self):
        """Test build_save_kwargs without optimization."""
        format_info = FormatInfo(
            extension=".bmp",
            pil_format="BMP",
            valid_extensions=[".bmp"],
            supports_alpha=False,
            supports_quality=False,
        )
        config = OutputConfig(optimize=False)

        kwargs = build_save_kwargs(format_info, config)

        assert kwargs["format"] == "BMP"
        assert "quality" not in kwargs
        assert "optimize" not in kwargs


class TestBaseClassErrorHandling:
    """Test error handling in base classes."""

    def test_raster_output_handler_get_bytes_error(self):
        """Test RasterOutputHandler.get_bytes error handling."""
        handler = get_output_handler("png")

        # Mock a failing save operation
        with patch.object(handler, "_save_to_destination") as mock_save:
            mock_save.side_effect = OSError("Mock save error")

            test_image = Image.new("RGB", (10, 10), "red")

            with pytest.raises(IOError, match="Failed to convert PNG to bytes"):
                handler.get_bytes(test_image)

    def test_raster_output_handler_save_error_handling(self):
        """Test RasterOutputHandler.save error handling."""
        handler = get_output_handler("png")

        with patch.object(handler, "_save_to_destination") as mock_save:
            mock_save.side_effect = ValueError("Mock save error")

            test_image = Image.new("RGB", (10, 10), "red")

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "test.png"

                with pytest.raises(IOError, match="Failed to save PNG"):
                    handler.save(test_image, output_path)

    @patch("molecular_string_renderer.outputs.base.logger")
    def test_error_logging(self, mock_logger):
        """Test that error handling logs appropriately."""
        handler = get_output_handler("png")

        with patch.object(handler, "_save_to_destination") as mock_save:
            mock_save.side_effect = ValueError("Mock error")

            test_image = Image.new("RGB", (10, 10), "red")

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "test.png"

                with pytest.raises(IOError):
                    handler.save(test_image, output_path)

                # Should log error
                mock_logger.error.assert_called_once()

    @patch("molecular_string_renderer.outputs.base.logger")
    def test_success_logging(self, mock_logger):
        """Test that successful saves log appropriately."""
        handler = get_output_handler("png")
        test_image = Image.new("RGB", (10, 10), "red")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"

            handler.save(test_image, output_path)

            # Should log success
            mock_logger.info.assert_called_once()
            assert "Successfully saved PNG" in str(mock_logger.info.call_args)


class TestSVGStrategyErrorHandling:
    """Test SVG strategy error handling and edge cases."""

    def test_vector_svg_strategy_no_molecule_error(self):
        """Test VectorSVGStrategy raises error when no molecule is set."""
        strategy = VectorSVGStrategy()
        test_image = Image.new("RGB", (100, 100), "red")
        config = OutputConfig()

        with pytest.raises(
            ValueError, match="No molecule set for vector SVG generation"
        ):
            strategy.generate_svg(test_image, config)

    @patch("molecular_string_renderer.outputs.svg_strategies.Draw.MolToSVG")
    def test_vector_svg_strategy_rdkit_error(self, mock_mol_to_svg):
        """Test VectorSVGStrategy handles RDKit errors."""
        strategy = VectorSVGStrategy()
        mock_mol = MagicMock()
        strategy.set_molecule(mock_mol)

        # Mock RDKit to raise an error
        mock_mol_to_svg.side_effect = Exception("RDKit error")

        test_image = Image.new("RGB", (100, 100), "red")
        config = OutputConfig()

        with pytest.raises(ValueError, match="Failed to generate vector SVG"):
            strategy.generate_svg(test_image, config)

    @patch("molecular_string_renderer.outputs.svg_strategies.Draw.MolToSVG")
    def test_vector_svg_strategy_optimization(self, mock_mol_to_svg):
        """Test VectorSVGStrategy optimization feature."""
        strategy = VectorSVGStrategy()
        mock_mol = MagicMock()
        strategy.set_molecule(mock_mol)

        # Mock SVG with comments and whitespace
        mock_svg = """<?xml version="1.0" encoding="UTF-8"?>
<!-- Comment -->
<svg>
  <!-- Another comment -->
  <circle r="10"/>
</svg>"""
        mock_mol_to_svg.return_value = mock_svg

        test_image = Image.new("RGB", (100, 100), "red")
        config = OutputConfig(optimize=True)

        result = strategy.generate_svg(test_image, config)

        # Should remove comments and extra whitespace
        assert "<!-- Comment -->" not in result
        assert "<!-- Another comment -->" not in result

    def test_hybrid_svg_strategy_vector_disabled(self):
        """Test HybridSVGStrategy with vector disabled."""
        strategy = HybridSVGStrategy()
        test_image = Image.new("RGB", (100, 100), "red")
        config = OutputConfig(svg_use_vector=False)

        result = strategy.generate_svg(test_image, config)

        # Should generate raster SVG
        assert "<?xml" in result
        assert "data:image/png;base64," in result

    @patch("molecular_string_renderer.outputs.svg_strategies.logger")
    def test_hybrid_svg_strategy_vector_fallback(self, mock_logger):
        """Test HybridSVGStrategy fallback to raster when vector fails."""
        strategy = HybridSVGStrategy()
        mock_mol = MagicMock()
        strategy.set_molecule(mock_mol)

        test_image = Image.new("RGB", (100, 100), "red")
        config = OutputConfig(svg_use_vector=True)

        # Mock vector strategy to fail
        with patch.object(strategy._vector_strategy, "generate_svg") as mock_vector:
            mock_vector.side_effect = ValueError("Vector failed")

            result = strategy.generate_svg(test_image, config)

            # Should fall back to raster
            assert "data:image/png;base64," in result

            # Should log the fallback
            mock_logger.debug.assert_called()
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            assert any("falling back to raster" in call for call in debug_calls)


class TestVectorOutputHandlerFormatName:
    """Test vector output handler format name behavior."""

    def test_vector_format_name_lowercase(self):
        """Test that vector handlers return lowercase format names."""
        svg_handler = get_output_handler("svg")
        pdf_handler = get_output_handler("pdf")

        assert svg_handler.format_name == "svg"
        assert pdf_handler.format_name == "pdf"


class TestRegistryBasedHandlerProperties:
    """Test RegistryBasedOutputHandler property access."""

    def test_all_properties_accessible(self):
        """Test that all registry-based properties are accessible."""
        handler = get_output_handler("png")

        # These should all work without errors
        assert isinstance(handler.file_extension, str)
        assert isinstance(handler.format_name, str)
        assert isinstance(handler.valid_extensions, list)
        assert isinstance(handler.supports_alpha, bool)
        assert isinstance(handler.supports_quality, bool)

        # Test specific values for PNG
        assert handler.file_extension == ".png"
        assert handler.format_name == "PNG"
        assert ".png" in handler.valid_extensions
        assert handler.supports_alpha is True
        assert handler.supports_quality is True
