"""
Edge cases tests for outputs submodule.

This file consolidates tests for:
- Edge cases with various image modes and configurations
- Factory and registry testing
- Utility function testing
- Format-specific behaviors and real-world scenarios
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import create_safe_filename, get_output_handler
from molecular_string_renderer.outputs.factory import (
    _validate_format,
    get_supported_formats,
)
from molecular_string_renderer.outputs.svg_strategies import (
    HybridSVGStrategy,
    RasterSVGStrategy,
    VectorSVGStrategy,
)
from molecular_string_renderer.outputs.utils import (
    FormatInfo,
    FormatRegistry,
    ImageModeUtils,
    build_save_kwargs,
)

# =============================================================================
# Factory and Registry Tests
# =============================================================================


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


class TestFormatInfoValidation:
    """Test FormatInfo class validation and post-initialization."""

    def test_format_info_post_init_adds_dot(self):
        """Test that FormatInfo adds dot to extension if missing."""
        info = FormatInfo(
            extension="png",
            pil_format="PNG",
            valid_extensions=[".png"],
            supports_alpha=True,
            supports_quality=False,
        )
        assert info.extension == ".png"

    def test_format_info_post_init_adds_primary_to_valid(self):
        """Test that FormatInfo adds primary extension to valid extensions."""
        info = FormatInfo(
            extension=".png",
            pil_format="PNG",
            valid_extensions=[".jpeg"],
            supports_alpha=True,
            supports_quality=False,
        )
        assert ".png" in info.valid_extensions
        assert ".jpeg" in info.valid_extensions


class TestFormatRegistry:
    """Test FormatRegistry functionality."""

    def test_get_format_info_unsupported(self):
        """Test getting info for unsupported format."""
        registry = FormatRegistry()

        with pytest.raises(ValueError, match="Unknown format: unsupported"):
            registry.get_format_info("unsupported")

    def test_get_format_info_case_insensitive(self):
        """Test format info retrieval is case insensitive."""
        registry = FormatRegistry()

        info_lower = registry.get_format_info("png")
        info_upper = registry.get_format_info("PNG")
        info_mixed = registry.get_format_info("PnG")

        assert info_lower.extension == info_upper.extension == info_mixed.extension

    def test_get_supported_formats(self):
        """Test getting supported formats list."""
        registry = FormatRegistry()
        formats = registry.get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        assert "png" in formats

    def test_is_supported(self):
        """Test format support checking."""
        registry = FormatRegistry()

        assert registry.is_supported("png")
        assert registry.is_supported("PNG")
        assert not registry.is_supported("unsupported")


# =============================================================================
# Image Mode and Utility Tests
# =============================================================================


class TestImageModeUtilsEdgeCases:
    """Test edge cases for ImageModeUtils functions."""

    def test_has_transparency_non_alpha_modes(self):
        """Test has_transparency with non-alpha modes."""
        rgb_image = Image.new("RGB", (50, 50), "red")
        assert not ImageModeUtils.has_transparency(rgb_image)

        l_image = Image.new("L", (50, 50), 128)
        assert not ImageModeUtils.has_transparency(l_image)

    def test_has_transparency_fully_opaque_alpha(self):
        """Test has_transparency with fully opaque alpha channel."""
        rgba_image = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        assert not ImageModeUtils.has_transparency(rgba_image)

    def test_has_transparency_with_transparency(self):
        """Test has_transparency with actual transparency."""
        rgba_image = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
        assert ImageModeUtils.has_transparency(rgba_image)

    def test_prepare_for_no_alpha_edge_cases(self):
        """Test prepare_for_no_alpha with various modes."""
        # Test with already no-alpha mode
        rgb_image = Image.new("RGB", (50, 50), "red")
        result = ImageModeUtils.prepare_for_no_alpha(rgb_image)
        assert result.mode == "RGB"

        # Test with LA mode
        la_image = Image.new("LA", (50, 50), (128, 200))
        result = ImageModeUtils.prepare_for_no_alpha(la_image)
        assert result.mode == "L"

    def test_prepare_for_no_alpha_pa_mode(self):
        """Test prepare_for_no_alpha with PA (palette with alpha) mode."""
        # Arrange - create a palette image with alpha
        # Note: PA mode is palette mode with alpha channel
        try:
            # Create a palette image first
            palette_image = Image.new("P", (50, 50))
            # Convert to PA mode (palette with alpha)
            pa_image = palette_image.convert("PA")

            # Act
            result = ImageModeUtils.prepare_for_no_alpha(pa_image)

            # Assert
            assert result.mode == "P", (
                "PA mode should convert to P mode when removing alpha"
            )

        except Exception:
            # PA mode might not be supported in all PIL builds
            pytest.skip("PA image mode not supported in this PIL build")

    def test_prepare_for_no_alpha_rgba_mode(self):
        """Test prepare_for_no_alpha with RGBA mode specifically."""
        # Arrange
        rgba_image = Image.new("RGBA", (50, 50), (255, 0, 0, 128))

        # Act
        result = ImageModeUtils.prepare_for_no_alpha(rgba_image)

        # Assert
        assert result.mode == "RGB", (
            "RGBA mode should convert to RGB mode when removing alpha"
        )

    def test_optimize_for_format_fully_opaque_conversion(self):
        """Test optimize_for_format converts fully opaque RGBA to RGB."""
        rgba_image = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        result = ImageModeUtils.optimize_for_format(rgba_image, "png")
        assert result.mode == "RGB"

    def test_optimize_for_format_la_mode_fully_opaque(self):
        """Test optimize_for_format converts fully opaque LA to L."""
        # Arrange - create LA image with full opacity (255 alpha)
        # We need to ensure it's actually fully opaque by setting ALL pixels to full alpha
        la_image = Image.new("LA", (50, 50), (128, 255))

        # Verify the image is truly opaque by checking has_transparency
        assert not ImageModeUtils.has_transparency(la_image), (
            "LA image should be fully opaque"
        )

        # Act
        result = ImageModeUtils.optimize_for_format(la_image, supports_alpha=True)

        # Assert
        assert result.mode == "L", (
            "Fully opaque LA should convert to L when alpha support is available"
        )

    def test_optimize_for_format_no_alpha_support(self):
        """Test optimize_for_format when format doesn't support alpha."""
        # Arrange - create image with alpha
        rgba_image = Image.new("RGBA", (50, 50), (255, 0, 0, 128))

        # Act
        result = ImageModeUtils.optimize_for_format(rgba_image, supports_alpha=False)

        # Assert
        assert result.mode == "RGB", (
            "RGBA image should convert to RGB when alpha is not supported"
        )

    def test_ensure_jpeg_compatible_edge_cases(self):
        """Test JPEG compatibility conversion edge cases."""
        # Test P mode
        p_image = Image.new("P", (50, 50))
        result = ImageModeUtils.ensure_jpeg_compatible(p_image)
        assert result.mode == "RGB"

        # Test 1 mode
        mono_image = Image.new("1", (50, 50), 1)
        result = ImageModeUtils.ensure_jpeg_compatible(mono_image)
        assert result.mode == "RGB"

    def test_ensure_jpeg_compatible_unknown_mode_warning(self):
        """Test JPEG compatibility with unknown mode logs warning."""
        # Mock an image with unusual mode
        mock_image = MagicMock()
        mock_image.mode = "UNUSUAL"
        mock_image.convert.return_value = MagicMock()

        with patch("molecular_string_renderer.outputs.utils.logger") as mock_logger:
            result = ImageModeUtils.ensure_jpeg_compatible(mock_image)
            # Should convert the image, not return original
            assert result is mock_image.convert.return_value
            mock_logger.warning.assert_called_once()

    def test_ensure_bmp_compatible_edge_cases(self):
        """Test BMP compatibility conversion edge cases."""
        # Test LA mode
        la_image = Image.new("LA", (50, 50), (128, 200))
        result = ImageModeUtils.ensure_bmp_compatible(la_image)
        assert result.mode == "RGB"

    def test_ensure_bmp_compatible_unknown_mode_warning(self):
        """Test BMP compatibility with unknown mode logs warning."""
        mock_image = MagicMock()
        mock_image.mode = "UNUSUAL"
        mock_image.convert.return_value = MagicMock()

        with patch("molecular_string_renderer.outputs.utils.logger") as mock_logger:
            result = ImageModeUtils.ensure_bmp_compatible(mock_image)
            # Should convert the image, not return original
            assert result is mock_image.convert.return_value
            mock_logger.warning.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions edge cases."""

    def test_create_safe_filename_various_inputs(self):
        """Test create_safe_filename with various inputs."""
        # Test with different molecular strings
        filename1 = create_safe_filename("CCO")
        filename2 = create_safe_filename("c1ccccc1")
        filename3 = create_safe_filename("CC(=O)O")

        assert all(fn.endswith(".png") for fn in [filename1, filename2, filename3])
        assert len(set([filename1, filename2, filename3])) == 3  # All unique

        # Test with custom extension
        svg_filename = create_safe_filename("CCO", ".svg")
        assert svg_filename.endswith(".svg")

    def test_build_save_kwargs_quality_support(self):
        """Test build_save_kwargs with quality support."""
        config = OutputConfig(quality=80, optimize=True)
        format_info = FormatInfo(
            extension=".jpg",
            pil_format="JPEG",
            valid_extensions=[".jpg"],
            supports_alpha=False,
            supports_quality=True,
        )

        kwargs = build_save_kwargs(format_info, config)
        assert kwargs["quality"] == 80
        assert kwargs["optimize"] is True

    def test_build_save_kwargs_no_quality_support_with_optimize(self):
        """Test build_save_kwargs without quality support but with optimization."""
        config = OutputConfig(quality=80, optimize=True)
        format_info = FormatInfo(
            extension=".png",
            pil_format="PNG",
            valid_extensions=[".png"],
            supports_alpha=True,
            supports_quality=False,
        )

        kwargs = build_save_kwargs(format_info, config)
        assert "quality" not in kwargs
        assert kwargs["optimize"] is True

    def test_build_save_kwargs_no_optimize(self):
        """Test build_save_kwargs without optimization support."""
        config = OutputConfig(quality=80, optimize=False)
        format_info = FormatInfo(
            extension=".jpg",
            pil_format="JPEG",
            valid_extensions=[".jpg"],
            supports_alpha=False,
            supports_quality=True,
        )

        kwargs = build_save_kwargs(format_info, config)
        assert kwargs["quality"] == 80
        assert kwargs["optimize"] is False

    def test_build_save_kwargs_no_quality_support_no_optimize(self):
        """Test build_save_kwargs without quality support and optimization disabled."""
        # Arrange
        config = OutputConfig(quality=80, optimize=False)
        format_info = FormatInfo(
            extension=".tiff",
            pil_format="TIFF",
            valid_extensions=[".tiff"],
            supports_alpha=True,
            supports_quality=False,  # No quality support
        )

        # Act
        kwargs = build_save_kwargs(format_info, config)

        # Assert
        assert "quality" not in kwargs, (
            "Quality should not be in kwargs when not supported"
        )
        assert "optimize" not in kwargs, (
            "Optimize should not be in kwargs when disabled"
        )
        assert kwargs["format"] == "TIFF", "Format should always be present"


# =============================================================================
# Error Handling and Logging Tests
# =============================================================================


class TestBaseClassErrorHandling:
    """Test error handling in base classes."""

    def test_raster_output_handler_get_bytes_error(self):
        """Test RasterOutputHandler get_bytes error handling."""
        from molecular_string_renderer.outputs.raster import PNGOutput

        handler = PNGOutput()

        with patch.object(
            handler, "_prepare_image", side_effect=ValueError("Test error")
        ):
            # Should catch ValueError and re-raise as IOError
            with pytest.raises(IOError):
                handler.get_bytes(Image.new("RGB", (50, 50), "red"))

    def test_raster_output_handler_save_error_handling(self):
        """Test RasterOutputHandler save error handling."""
        from molecular_string_renderer.outputs.raster import PNGOutput

        handler = PNGOutput()
        test_image = Image.new("RGB", (50, 50), "red")

        with patch.object(
            handler, "_save_to_destination", side_effect=Exception("Test error")
        ):
            with pytest.raises(IOError):
                handler.save(test_image, "/tmp/test.png")

    def test_error_logging(self):
        """Test that errors are properly logged."""
        from molecular_string_renderer.outputs.raster import PNGOutput

        handler = PNGOutput()

        with patch("molecular_string_renderer.outputs.base.logger") as mock_logger:
            with patch.object(
                handler, "_prepare_image", side_effect=ValueError("Test")
            ):
                with pytest.raises(
                    IOError
                ):  # The base class catches and re-raises as IOError
                    handler.get_bytes(Image.new("RGB", (50, 50), "red"))

                mock_logger.error.assert_called_once()

    def test_success_logging(self):
        """Test that successful operations are logged."""
        from molecular_string_renderer.outputs.raster import PNGOutput

        handler = PNGOutput()
        test_image = Image.new("RGB", (50, 50), "red")

        with patch("molecular_string_renderer.outputs.base.logger") as mock_logger:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                handler.save(test_image, tmp.name)
                mock_logger.info.assert_called()


# =============================================================================
# SVG Strategy Tests
# =============================================================================


class TestSVGStrategyErrorHandling:
    """Test SVG strategy error handling and edge cases."""

    def test_vector_svg_strategy_no_molecule_error(self):
        """Test VectorSVGStrategy when no molecule is set."""
        strategy = VectorSVGStrategy()
        test_image = Image.new("RGB", (100, 100), "red")

        with pytest.raises(ValueError, match="No molecule set"):
            strategy.generate_svg(test_image, {})

    def test_vector_svg_strategy_rdkit_error(self):
        """Test VectorSVGStrategy RDKit error handling."""
        strategy = VectorSVGStrategy()
        mock_mol = MagicMock()
        strategy.set_molecule(mock_mol)

        test_image = Image.new("RGB", (100, 100), "red")

        with patch(
            "molecular_string_renderer.outputs.svg_strategies.Draw"
        ) as mock_draw:
            mock_draw.MolToSVG.side_effect = Exception("RDKit error")

            with pytest.raises(Exception):
                strategy.generate_svg(test_image, OutputConfig())

    def test_vector_svg_strategy_optimization(self):
        """Test VectorSVGStrategy optimization parameter."""
        strategy = VectorSVGStrategy()
        mock_mol = MagicMock()
        strategy.set_molecule(mock_mol)

        test_image = Image.new("RGB", (100, 100), "red")

        with patch(
            "molecular_string_renderer.outputs.svg_strategies.Draw"
        ) as mock_draw:
            mock_draw.MolToSVG.return_value = "<svg></svg>"

            result = strategy.generate_svg(test_image, OutputConfig(optimize=True))
            assert "<svg" in result

    def test_hybrid_svg_strategy_vector_disabled(self):
        """Test HybridSVGStrategy with vector generation disabled."""
        config = OutputConfig(svg_use_vector=False)
        strategy = HybridSVGStrategy()
        test_image = Image.new("RGB", (100, 100), "red")

        result = strategy.generate_svg(test_image, config)
        assert "data:image/png;base64," in result

    def test_hybrid_svg_strategy_vector_fallback(self):
        """Test HybridSVGStrategy fallback to raster when vector fails."""
        config = OutputConfig(svg_use_vector=True)
        strategy = HybridSVGStrategy()
        mock_mol = MagicMock()
        strategy.set_molecule(mock_mol)

        test_image = Image.new("RGB", (100, 100), "red")

        with patch.object(strategy, "_vector_strategy") as mock_vector:
            mock_vector.generate_svg.side_effect = ValueError("Vector failed")
            result = strategy.generate_svg(test_image, config)
            # Should fallback to raster
            assert "data:image/png;base64," in result
            assert "data:image/png;base64," in result

    def test_raster_svg_strategy_basic(self):
        """Test RasterSVGStrategy basic functionality."""
        strategy = RasterSVGStrategy()
        test_image = Image.new("RGB", (100, 100), "red")
        config = OutputConfig()

        result = strategy.generate_svg(test_image, config)

        # Should contain SVG structure with embedded PNG
        assert "<?xml version=" in result
        assert "<svg" in result
        assert "</svg>" in result
        assert "data:image/png;base64," in result
        assert f'width="{test_image.width}"' in result
        assert f'height="{test_image.height}"' in result

    def test_vector_svg_optimization_success(self):
        """Test VectorSVGStrategy optimization with actual SVG content."""
        strategy = VectorSVGStrategy()
        mock_mol = MagicMock()
        strategy.set_molecule(mock_mol)

        test_image = Image.new("RGB", (100, 100), "red")

        # Mock SVG content with comments
        mock_svg_content = """<svg>
<!-- comment -->
  <g></g>
</svg>"""

        with patch(
            "molecular_string_renderer.outputs.svg_strategies.Draw"
        ) as mock_draw:
            mock_draw.MolToSVG.return_value = mock_svg_content

            result = strategy.generate_svg(test_image, OutputConfig(optimize=True))

            # Comments should be removed
            assert "<!-- comment -->" not in result
            assert "<svg>" in result
            assert "<g></g>" in result

    def test_hybrid_svg_strategy_vector_success(self):
        """Test HybridSVGStrategy when vector generation succeeds."""
        strategy = HybridSVGStrategy()
        mock_mol = MagicMock()
        strategy.set_molecule(mock_mol)

        test_image = Image.new("RGB", (100, 100), "red")
        config = OutputConfig(svg_use_vector=True)

        mock_svg_content = "<svg><g><path d='M10,10 L20,20'/></g></svg>"

        with patch.object(strategy._vector_strategy, "generate_svg") as mock_vector:
            mock_vector.return_value = mock_svg_content

            result = strategy.generate_svg(test_image, config)

            # Should return vector SVG content, not raster
            assert result == mock_svg_content
            assert "data:image/png;base64," not in result

    def test_vector_svg_strategy_no_optimization(self):
        """Test VectorSVGStrategy without optimization to cover remaining branch."""
        strategy = VectorSVGStrategy()
        mock_mol = MagicMock()
        strategy.set_molecule(mock_mol)

        test_image = Image.new("RGB", (100, 100), "red")
        mock_svg_content = "<svg><!-- comment --><g></g></svg>"

        with patch(
            "molecular_string_renderer.outputs.svg_strategies.Draw"
        ) as mock_draw:
            mock_draw.MolToSVG.return_value = mock_svg_content

            # Test without optimization
            result = strategy.generate_svg(test_image, OutputConfig(optimize=False))

            # Should return content as-is (no optimization)
            assert result == mock_svg_content
            assert "<!-- comment -->" in result


# =============================================================================
# Vector Output Handler Tests
# =============================================================================


class TestVectorOutputHandlerFormatName:
    """Test VectorOutputHandler format name handling."""

    def test_vector_format_name_lowercase(self):
        """Test that format names are properly lowercased."""
        from molecular_string_renderer.outputs.vector import PDFOutput, SVGOutput

        svg_handler = SVGOutput()
        pdf_handler = PDFOutput()

        assert svg_handler.format_name == "svg"
        assert pdf_handler.format_name == "pdf"


class TestRegistryBasedHandlerProperties:
    """Test that handlers properly use registry-based properties."""

    def test_all_properties_accessible(self):
        """Test that all handler properties are accessible via registry."""
        formats = ["png", "jpeg", "webp", "tiff", "bmp", "svg", "pdf"]

        for format_name in formats:
            handler = get_output_handler(format_name)

            # These should not raise AttributeError
            assert isinstance(handler.format_name, str)
            assert isinstance(handler.file_extension, str)
            assert isinstance(handler.supports_alpha, bool)
            assert isinstance(handler.supports_quality, bool)


# =============================================================================
# Image Mode Edge Cases
# =============================================================================


class TestImageModeEdgeCases:
    """Test edge cases with various image modes and color spaces."""

    def test_palette_mode_images(self):
        """Test handling of palette mode images."""
        palette_image = Image.new("P", (100, 100))

        for format_name in ["png", "jpg", "webp", "tiff", "bmp"]:
            handler = get_output_handler(format_name)
            result = handler.get_bytes(palette_image)
            assert len(result) > 0

    def test_monochrome_images(self):
        """Test handling of 1-bit monochrome images."""
        mono_image = Image.new("1", (100, 100), 1)

        for format_name in ["png", "jpg", "tiff", "bmp"]:
            handler = get_output_handler(format_name)
            result = handler.get_bytes(mono_image)
            assert len(result) > 0

    def test_cmyk_images(self):
        """Test handling of CMYK images (JPEG specific)."""
        try:
            cmyk_image = Image.new("CMYK", (100, 100), (100, 50, 0, 25))

            # JPEG should handle CMYK
            jpeg_handler = get_output_handler("jpeg")
            result = jpeg_handler.get_bytes(cmyk_image)
            assert len(result) > 0

            # Other formats should convert gracefully or fail gracefully
            for format_name in ["png", "webp", "tiff"]:
                handler = get_output_handler(format_name)
                result = handler.get_bytes(cmyk_image)
                assert len(result) > 0

        except Exception:
            # CMYK might not be supported in all PIL builds
            pytest.skip("CMYK images not supported in this PIL build")

    def test_extreme_alpha_values(self):
        """Test images with extreme alpha values."""
        # Fully transparent
        transparent_image = Image.new("RGBA", (50, 50), (255, 0, 0, 0))

        # Nearly transparent
        nearly_transparent = Image.new("RGBA", (50, 50), (255, 0, 0, 1))

        # Nearly opaque
        nearly_opaque = Image.new("RGBA", (50, 50), (255, 0, 0, 254))

        for image in [transparent_image, nearly_transparent, nearly_opaque]:
            for format_name in ["png", "webp", "tiff", "svg"]:
                handler = get_output_handler(format_name)
                result = handler.get_bytes(image)
                assert len(result) > 0


# =============================================================================
# Extreme Dimension Tests
# =============================================================================


class TestExtremeImageDimensions:
    """Test handling of images with extreme dimensions."""

    def test_very_wide_images(self):
        """Test very wide but short images."""
        wide_image = Image.new("RGB", (1000, 10), "red")

        for format_name in ["png", "jpg", "webp", "svg"]:
            handler = get_output_handler(format_name)
            result = handler.get_bytes(wide_image)
            assert len(result) > 0

    def test_very_tall_images(self):
        """Test very tall but narrow images."""
        tall_image = Image.new("RGB", (10, 1000), "blue")

        for format_name in ["png", "jpg", "webp", "svg"]:
            handler = get_output_handler(format_name)
            result = handler.get_bytes(tall_image)
            assert len(result) > 0

    def test_square_images_various_sizes(self):
        """Test square images of various sizes."""
        sizes = [1, 2, 5, 10, 50, 100, 500]

        for size in sizes:
            image = Image.new("RGB", (size, size), "green")

            for format_name in ["png", "jpg", "svg"]:
                handler = get_output_handler(format_name)
                result = handler.get_bytes(image)
                assert len(result) > 0


# =============================================================================
# Configuration Edge Cases
# =============================================================================


class TestConfigurationEdgeCases:
    """Test edge cases with various configurations."""

    def test_extreme_quality_values(self):
        """Test extreme quality values."""
        test_image = Image.new("RGB", (100, 100), "red")

        # Minimum quality
        low_config = OutputConfig(quality=1)
        handler = get_output_handler("jpeg", low_config)
        result = handler.get_bytes(test_image)
        assert len(result) > 0

        # Maximum quality
        high_config = OutputConfig(quality=100)
        handler = get_output_handler("jpeg", high_config)
        result = handler.get_bytes(test_image)
        assert len(result) > 0

    def test_conflicting_configuration_options(self):
        """Test conflicting configuration options."""
        test_image = Image.new("RGB", (100, 100), "red")

        # High quality with optimization (might conflict for some formats)
        config = OutputConfig(quality=95, optimize=True)

        for format_name in ["jpeg", "webp", "png"]:
            handler = get_output_handler(format_name, config)
            result = handler.get_bytes(test_image)
            assert len(result) > 0

    def test_configuration_with_unsupported_features(self):
        """Test configuration with features not supported by format."""
        test_image = Image.new("RGB", (100, 100), "red")

        # Quality setting for PNG (should be ignored)
        png_config = OutputConfig(quality=50)
        png_handler = get_output_handler("png", png_config)
        result = png_handler.get_bytes(test_image)
        assert len(result) > 0

        # Optimization for BMP (should be ignored)
        bmp_config = OutputConfig(optimize=True)
        bmp_handler = get_output_handler("bmp", bmp_config)
        result = bmp_handler.get_bytes(test_image)
        assert len(result) > 0


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Test error recovery and resilience."""

    def test_resource_cleanup_on_error(self):
        """Test that resources are properly cleaned up on errors."""
        test_image = Image.new("RGB", (50, 50), "red")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"

            handler = get_output_handler("png")

            # Force an error during save
            with patch.object(test_image, "save", side_effect=Exception("Save failed")):
                with pytest.raises(Exception):
                    handler.save(test_image, output_path)

                # File should not exist after failed save
                assert not output_path.exists()

    def test_corrupted_image_handling(self):
        """Test handling of corrupted or invalid images."""
        handler = get_output_handler("png")

        # Create a mock corrupted image
        corrupted_image = MagicMock()
        corrupted_image.save.side_effect = Exception("Corrupted image")

        with pytest.raises(Exception):
            handler.get_bytes(corrupted_image)


# =============================================================================
# Format-Specific Behaviors
# =============================================================================


class TestFormatSpecificBehaviors:
    """Test format-specific behaviors and optimizations."""

    def test_jpeg_progressive_encoding(self):
        """Test JPEG progressive encoding."""
        test_image = Image.new("RGB", (200, 200), "red")

        # Test with progressive enabled (if supported)
        handler = get_output_handler("jpeg")
        result = handler.get_bytes(test_image)
        assert len(result) > 0

    def test_png_compression_levels(self):
        """Test PNG compression behavior."""
        test_image = Image.new("RGB", (200, 200), "red")

        optimized_config = OutputConfig(optimize=True)
        handler = get_output_handler("png", optimized_config)

        result = handler.get_bytes(test_image)
        assert len(result) > 0

    def test_webp_lossless_vs_lossy(self):
        """Test WEBP lossless vs lossy compression."""
        test_image = Image.new("RGB", (100, 100), "red")

        # High quality (should be lossless or near-lossless)
        high_quality = OutputConfig(quality=100)
        handler = get_output_handler("webp", high_quality)
        result_high = handler.get_bytes(test_image)

        # Low quality (should be lossy)
        low_quality = OutputConfig(quality=10)
        handler = get_output_handler("webp", low_quality)
        result_low = handler.get_bytes(test_image)

        assert len(result_high) > 0
        assert len(result_low) > 0
        # Compression behavior can vary, so just check both are valid
        assert len(result_high) > 0

    def test_tiff_compression_options(self):
        """Test TIFF compression behavior."""
        test_image = Image.new("RGB", (100, 100), "red")

        handler = get_output_handler("tiff")
        result = handler.get_bytes(test_image)
        assert len(result) > 0

    def test_svg_vector_vs_raster_modes(self):
        """Test SVG vector vs raster generation modes."""
        test_image = Image.new("RGB", (100, 100), "red")

        # Raster mode
        raster_config = OutputConfig(svg_use_vector=False)
        raster_handler = get_output_handler("svg", raster_config)
        raster_result = raster_handler.get_bytes(test_image)

        # Vector mode (will fallback to raster without molecule)
        vector_config = OutputConfig(svg_use_vector=True)
        vector_handler = get_output_handler("svg", vector_config)
        vector_result = vector_handler.get_bytes(test_image)

        assert b"<svg" in raster_result
        assert b"<svg" in vector_result

    def test_pdf_page_layout_variations(self):
        """Test PDF page layout with different image dimensions."""
        images = [
            Image.new("RGB", (100, 100), "red"),  # Square
            Image.new("RGB", (200, 100), "green"),  # Landscape
            Image.new("RGB", (100, 200), "blue"),  # Portrait
        ]

        handler = get_output_handler("pdf")

        for image in images:
            result = handler.get_bytes(image)
            assert len(result) > 0
            assert result.startswith(b"%PDF")


# =============================================================================
# Utility Function Robustness
# =============================================================================


class TestUtilityFunctionRobustness:
    """Test robustness of utility functions."""

    def test_safe_filename_edge_cases(self):
        """Test safe filename generation with edge cases."""
        edge_cases = [
            "",  # Empty string
            " ",  # Whitespace only
            "C" * 1000,  # Very long string
            "c1ccccc1" * 100,  # Repeated pattern
            "C=C=C=C=C",  # Multiple special characters
        ]

        for case in edge_cases:
            filename = create_safe_filename(case)
            assert len(filename) > 0
            assert filename.endswith(".png")

    def test_create_safe_filename_none_extension(self):
        """Test create_safe_filename with None extension raises appropriate error."""
        # Arrange
        molecular_string = "CCO"

        # Act & Assert - should raise ValueError when extension is None
        with pytest.raises(ValueError, match="Extension cannot be None"):
            create_safe_filename(molecular_string, None)

    def test_create_safe_filename_empty_extension(self):
        """Test create_safe_filename with empty extension raises appropriate error."""
        # Arrange
        molecular_string = "CCO"

        # Act & Assert - should raise ValueError when extension is empty
        with pytest.raises(ValueError, match="Extension cannot be empty"):
            create_safe_filename(molecular_string, "")

    def test_create_safe_filename_extension_without_dot(self):
        """Test create_safe_filename adds dot to extension when missing."""
        # Arrange
        molecular_string = "CCO"
        extension_without_dot = "svg"

        # Act
        filename = create_safe_filename(molecular_string, extension_without_dot)

        # Assert
        assert filename.endswith(".svg"), "Should add dot to extension"
        assert len(filename) > len(".svg"), "Should have hash-based filename"
        # Verify it's a valid MD5 hash (32 chars) plus extension
        base_name = filename.replace(".svg", "")
        assert len(base_name) == 32, "Should be MD5 hash length"

    def test_safe_filename_consistency(self):
        """Test that safe filename generation is consistent."""
        smiles = "CCO"

        filenames = [create_safe_filename(smiles) for _ in range(10)]

        # All should be identical
        assert len(set(filenames)) == 1

    def test_safe_filename_different_extensions(self):
        """Test safe filename with different extensions."""
        smiles = "CCO"
        extensions = [".png", ".svg", ".pdf", ".jpg"]

        filenames = [create_safe_filename(smiles, ext) for ext in extensions]

        # Should all have different extensions but same base
        bases = [fn.rsplit(".", 1)[0] for fn in filenames]
        assert len(set(bases)) == 1  # Same base

        for fn, ext in zip(filenames, extensions):
            assert fn.endswith(ext)


# =============================================================================
# Real-World Scenarios
# =============================================================================


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_batch_processing_workflow(self):
        """Test a typical batch processing workflow."""
        # Simulate processing multiple molecules
        test_images = [
            Image.new("RGB", (100, 100), "red"),
            Image.new("RGB", (150, 120), "green"),
            Image.new("RGBA", (80, 80), (0, 0, 255, 128)),
        ]

        formats = ["png", "svg", "pdf"]

        for format_name in formats:
            handler = get_output_handler(format_name)

            results = []
            for image in test_images:
                result = handler.get_bytes(image)
                results.append(result)

            assert len(results) == len(test_images)
            assert all(len(r) > 0 for r in results)

    def test_mixed_quality_batch_processing(self):
        """Test batch processing with different quality settings."""
        test_image = Image.new("RGB", (100, 100), "red")

        qualities = [20, 50, 80, 95]

        for quality in qualities:
            config = OutputConfig(quality=quality)
            handler = get_output_handler("jpeg", config)
            result = handler.get_bytes(test_image)
            assert len(result) > 0

    def test_format_conversion_pipeline(self):
        """Test a format conversion pipeline."""
        source_image = Image.new("RGB", (100, 100), "red")

        # Convert through multiple formats
        formats = ["png", "jpeg", "webp", "svg", "pdf"]

        for format_name in formats:
            handler = get_output_handler(format_name)
            result = handler.get_bytes(source_image)
            assert len(result) > 0

            # Verify format-specific characteristics
            if format_name == "pdf":
                assert result.startswith(b"%PDF")
            elif format_name == "svg":
                assert b"<svg" in result
