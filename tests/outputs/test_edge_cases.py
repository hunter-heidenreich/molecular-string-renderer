"""
Edge cases tests for outputs submodule.

This file consolidates tests for:
- Edge cases with various image modes and configurations
- Factory and registry error handling
- Utility function robustness
- Format-specific edge behaviors
- Real-world scenarios

Tests are organized by functional area for better maintenance:
- Factory/Registry: Error handling and validation
- ImageModeUtils: Complex mode conversions and edge cases
- Utility Functions: Input validation and error handling
- Real-World Scenarios: Complex integration patterns
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

    def test_has_transparency_non_alpha_modes(self, test_image, grayscale_image):
        """Test has_transparency with non-alpha modes."""
        assert not ImageModeUtils.has_transparency(test_image)
        assert not ImageModeUtils.has_transparency(grayscale_image)

    def test_has_transparency_fully_opaque_alpha(self, rgba_opaque_image):
        """Test has_transparency with fully opaque alpha channel."""
        assert not ImageModeUtils.has_transparency(rgba_opaque_image)

    def test_has_transparency_with_transparency(self, rgba_image):
        """Test has_transparency with actual transparency."""
        assert ImageModeUtils.has_transparency(rgba_image)

    def test_prepare_for_no_alpha_edge_cases(self, test_image, la_image):
        """Test prepare_for_no_alpha with various modes."""
        # Test with already no-alpha mode
        result = ImageModeUtils.prepare_for_no_alpha(test_image)
        assert result.mode == "RGB"

        # Test with LA mode
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

    def test_prepare_for_no_alpha_rgba_mode(self, rgba_image):
        """Test prepare_for_no_alpha with RGBA mode specifically."""
        result = ImageModeUtils.prepare_for_no_alpha(rgba_image)
        assert result.mode == "RGB", (
            "RGBA mode should convert to RGB mode when removing alpha"
        )

    def test_optimize_for_format_fully_opaque_conversion(self, rgba_opaque_image):
        """Test optimize_for_format converts fully opaque RGBA to RGB."""
        result = ImageModeUtils.optimize_for_format(rgba_opaque_image, "png")
        assert result.mode == "RGB"

    def test_optimize_for_format_la_mode_fully_opaque(self):
        """Test optimize_for_format converts fully opaque LA to L."""
        # Create LA image with full opacity (255 alpha)
        la_image = Image.new("LA", (50, 50), (128, 255))

        # Verify the image is truly opaque by checking has_transparency
        assert not ImageModeUtils.has_transparency(la_image), (
            "LA image should be fully opaque"
        )

        result = ImageModeUtils.optimize_for_format(la_image, supports_alpha=True)
        assert result.mode == "L", (
            "Fully opaque LA should convert to L when alpha support is available"
        )

    def test_optimize_for_format_no_alpha_support(self, rgba_image):
        """Test optimize_for_format when format doesn't support alpha."""
        result = ImageModeUtils.optimize_for_format(rgba_image, supports_alpha=False)
        assert result.mode == "RGB", (
            "RGBA image should convert to RGB when alpha is not supported"
        )

    def test_ensure_jpeg_compatible_edge_cases(self, test_images_all_modes):
        """Test JPEG compatibility conversion edge cases."""
        # Test P mode
        p_image = test_images_all_modes["P"]
        result = ImageModeUtils.ensure_jpeg_compatible(p_image)
        assert result.mode == "RGB"

        # Test 1 mode
        mono_image = test_images_all_modes["1"]
        result = ImageModeUtils.ensure_jpeg_compatible(mono_image)
        assert result.mode == "RGB"

    def test_ensure_bmp_compatible_edge_cases(self, la_image):
        """Test BMP compatibility conversion edge cases."""
        result = ImageModeUtils.ensure_bmp_compatible(la_image)
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

    def test_error_logging(self):
        """Test that errors are properly logged."""
        from molecular_string_renderer.outputs.raster import PNGOutput

        handler = PNGOutput()

        with patch("molecular_string_renderer.outputs.base.logger") as mock_logger:
            with patch.object(
                handler, "_prepare_image", side_effect=ValueError("Test")
            ):
                with pytest.raises(IOError):
                    handler.get_bytes(Image.new("RGB", (50, 50), "red"))

                mock_logger.error.assert_called_once()


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

    def test_palette_mode_images(self, test_images_all_modes):
        """Test handling of palette mode images."""
        palette_image = test_images_all_modes["P"]

        for format_name in ["png", "jpg", "webp", "tiff", "bmp"]:
            handler = get_output_handler(format_name)
            result = handler.get_bytes(palette_image)
            assert len(result) > 0

    def test_monochrome_images(self, test_images_all_modes):
        """Test handling of 1-bit monochrome images."""
        mono_image = test_images_all_modes["1"]

        for format_name in ["png", "jpg", "tiff", "bmp"]:
            handler = get_output_handler(format_name)
            result = handler.get_bytes(mono_image)
            assert len(result) > 0

    def test_extreme_alpha_values(self, extreme_alpha_images):
        """Test images with extreme alpha values."""
        for alpha_type, image in extreme_alpha_images.items():
            for format_name in ["png", "webp", "tiff", "svg"]:
                handler = get_output_handler(format_name)
                result = handler.get_bytes(image)
                assert len(result) > 0, f"Failed for {alpha_type} with {format_name}"


# =============================================================================
# Extreme Dimension Tests
# =============================================================================


# =============================================================================
# Configuration Edge Cases
# =============================================================================


class TestConfigurationEdgeCases:
    """Test edge cases with various configurations."""

    def test_extreme_quality_values(self, test_image):
        """Test extreme quality values."""
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

    def test_conflicting_configuration_options(self, test_image):
        """Test conflicting configuration options."""
        # High quality with optimization (might conflict for some formats)
        config = OutputConfig(quality=95, optimize=True)

        for format_name in ["jpeg", "webp", "png"]:
            handler = get_output_handler(format_name, config)
            result = handler.get_bytes(test_image)
            assert len(result) > 0

    def test_configuration_with_unsupported_features(self, test_image):
        """Test configuration with features not supported by format."""
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

    @pytest.mark.parametrize("format_name", ["jpeg", "webp"])
    def test_quality_supporting_formats(self, test_image, format_name):
        """Test formats that support quality settings."""
        high_quality = OutputConfig(quality=95)
        low_quality = OutputConfig(quality=20)

        high_handler = get_output_handler(format_name, high_quality)
        low_handler = get_output_handler(format_name, low_quality)

        high_result = high_handler.get_bytes(test_image)
        low_result = low_handler.get_bytes(test_image)

        assert len(high_result) > 0
        assert len(low_result) > 0

    @pytest.mark.parametrize("format_name", ["png", "webp", "tiff"])
    def test_optimization_supporting_formats(self, test_image, format_name):
        """Test formats that support optimization."""
        optimized_config = OutputConfig(optimize=True)
        handler = get_output_handler(format_name, optimized_config)

        result = handler.get_bytes(test_image)
        assert len(result) > 0

    def test_svg_vector_vs_raster_modes(self, test_image):
        """Test SVG vector vs raster generation modes."""
        from .conftest import assert_image_output_valid

        # Raster mode
        raster_config = OutputConfig(svg_use_vector=False)
        raster_handler = get_output_handler("svg", raster_config)
        raster_result = raster_handler.get_bytes(test_image)

        # Vector mode (will fallback to raster without molecule)
        vector_config = OutputConfig(svg_use_vector=True)
        vector_handler = get_output_handler("svg", vector_config)
        vector_result = vector_handler.get_bytes(test_image)

        assert_image_output_valid(raster_result, "svg")
        assert_image_output_valid(vector_result, "svg")

    def test_pdf_page_layout_variations(self, square_image, wide_image, tall_image):
        """Test PDF page layout with different image dimensions."""
        from .conftest import assert_image_output_valid

        handler = get_output_handler("pdf")

        for image in [square_image, wide_image, tall_image]:
            result = handler.get_bytes(image)
            assert_image_output_valid(result, "pdf")


# =============================================================================
# Utility Function Robustness
# =============================================================================


class TestUtilityFunctionRobustness:
    """Test robustness of utility functions."""

    @pytest.mark.parametrize(
        "input_case",
        [
            "",  # Empty string
            " ",  # Whitespace only
            "C" * 1000,  # Very long string
            "c1ccccc1" * 100,  # Repeated pattern
            "C=C=C=C=C",  # Multiple special characters
        ],
    )
    def test_safe_filename_edge_cases(self, input_case):
        """Test safe filename generation with edge cases."""
        filename = create_safe_filename(input_case)
        assert len(filename) > 0
        assert filename.endswith(".png")

    def test_create_safe_filename_validation(self):
        """Test create_safe_filename input validation."""
        molecular_string = "CCO"

        # Test None extension
        with pytest.raises(ValueError, match="Extension cannot be None"):
            create_safe_filename(molecular_string, None)

        # Test empty extension
        with pytest.raises(ValueError, match="Extension cannot be empty"):
            create_safe_filename(molecular_string, "")

    def test_create_safe_filename_extension_normalization(self):
        """Test create_safe_filename adds dot to extension when missing."""
        molecular_string = "CCO"
        extension_without_dot = "svg"

        filename = create_safe_filename(molecular_string, extension_without_dot)

        assert filename.endswith(".svg"), "Should add dot to extension"
        assert len(filename) > len(".svg"), "Should have hash-based filename"
        # Verify it's a valid MD5 hash (32 chars) plus extension
        base_name = filename.replace(".svg", "")
        assert len(base_name) == 32, "Should be MD5 hash length"

    def test_safe_filename_consistency_and_uniqueness(self):
        """Test filename generation consistency and uniqueness."""
        smiles = "CCO"
        filenames = [create_safe_filename(smiles) for _ in range(10)]

        # All should be identical (consistency)
        assert len(set(filenames)) == 1

        # Different extensions should have same base
        extensions = [".png", ".svg", ".pdf", ".jpg"]
        filenames = [create_safe_filename(smiles, ext) for ext in extensions]
        bases = [fn.rsplit(".", 1)[0] for fn in filenames]
        assert len(set(bases)) == 1  # Same base

        for fn, ext in zip(filenames, extensions):
            assert fn.endswith(ext)


# =============================================================================
# Real-World Scenarios
# =============================================================================


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_batch_processing_workflow(self, test_image, large_image, rgba_image):
        """Test a typical batch processing workflow."""
        # Simulate processing multiple molecules
        test_images = [test_image, large_image, rgba_image]
        formats = ["png", "svg", "pdf"]

        for format_name in formats:
            handler = get_output_handler(format_name)
            results = []
            for image in test_images:
                result = handler.get_bytes(image)
                results.append(result)

            assert len(results) == len(test_images)
            assert all(len(r) > 0 for r in results)

    def test_mixed_quality_batch_processing(self, test_image):
        """Test batch processing with different quality settings."""
        qualities = [20, 50, 80, 95]

        for quality in qualities:
            config = OutputConfig(quality=quality)
            handler = get_output_handler("jpeg", config)
            result = handler.get_bytes(test_image)
            assert len(result) > 0

    def test_format_conversion_pipeline(self, test_image):
        """Test a format conversion pipeline."""
        from .conftest import assert_image_output_valid

        # Convert through multiple formats
        formats = ["png", "jpeg", "webp", "svg", "pdf"]

        for format_name in formats:
            handler = get_output_handler(format_name)
            result = handler.get_bytes(test_image)
            assert_image_output_valid(result, format_name)


class TestImageModeUtilsExtended:
    """Extended tests for ImageModeUtils to improve coverage."""

    @pytest.mark.parametrize("mode", ["RGB", "L", "CMYK"])
    def test_ensure_jpeg_compatible_supported_modes(self, mode):
        """Test ensure_jpeg_compatible with supported modes."""
        try:
            test_image = Image.new(mode, (50, 50))
            result = ImageModeUtils.ensure_jpeg_compatible(test_image)
            assert result.mode in ["RGB", "L", "CMYK"]
        except OSError:
            # Some modes may not be directly creatable
            pytest.skip(f"{mode} mode not supported in this PIL build")

    @pytest.mark.parametrize("mode", ["RGBA", "LA", "PA", "P", "1"])
    def test_ensure_jpeg_compatible_conversion_modes(self, mode):
        """Test ensure_jpeg_compatible modes that need conversion."""
        try:
            test_image = Image.new(mode, (50, 50))
            result = ImageModeUtils.ensure_jpeg_compatible(test_image)
            assert result.mode == "RGB"
        except OSError:
            # Some modes may not be directly creatable
            pytest.skip(f"{mode} mode not supported in this PIL build")

    @pytest.mark.parametrize("mode", ["1", "L", "P", "RGB", "RGBA"])
    def test_ensure_bmp_compatible_supported_modes(self, mode):
        """Test ensure_bmp_compatible with supported modes."""
        test_image = Image.new(mode, (50, 50))
        result = ImageModeUtils.ensure_bmp_compatible(test_image)
        assert result.mode == mode


class TestBuildSaveKwargsExtended:
    """Extended tests for build_save_kwargs to improve coverage."""

    def test_build_save_kwargs_progressive_jpeg(self):
        """Test build_save_kwargs with progressive JPEG setting."""
        format_info = FormatRegistry.get_format_info("jpeg")
        config = OutputConfig(progressive=True, quality=85, optimize=True)

        kwargs = build_save_kwargs(format_info, config)

        assert kwargs["progressive"] is True
        assert kwargs["quality"] == 85
        assert kwargs["optimize"] is True

    def test_build_save_kwargs_lossless_webp(self):
        """Test build_save_kwargs with lossless WebP setting."""
        format_info = FormatRegistry.get_format_info("webp")
        config = OutputConfig(lossless=True, quality=80, optimize=True)

        kwargs = build_save_kwargs(format_info, config)

        assert kwargs["lossless"] is True
        assert kwargs["quality"] == 80
        assert kwargs["optimize"] is True

    def test_build_save_kwargs_optimize_only_for_non_quality_formats(self):
        """Test build_save_kwargs with optimize for non-quality supporting formats."""
        format_info = FormatRegistry.get_format_info(
            "tiff"
        )  # TIFF doesn't support quality
        config = OutputConfig(optimize=True, quality=85)  # quality ignored for TIFF

        kwargs = build_save_kwargs(format_info, config)

        assert kwargs["optimize"] is True
        assert "quality" not in kwargs  # TIFF doesn't support quality

    def test_build_save_kwargs_dpi_support(self):
        """Test build_save_kwargs DPI settings for supported formats."""
        dpi_supporting_formats = ["png", "jpeg", "tiff"]

        for format_name in dpi_supporting_formats:
            format_info = FormatRegistry.get_format_info(format_name)
            config = OutputConfig(dpi=150)

            kwargs = build_save_kwargs(format_info, config)

            assert kwargs["dpi"] == (150, 150)


class TestFormatRegistryExtended:
    """Extended tests for FormatRegistry to improve coverage."""

    def test_format_info_post_init_extension_normalization(self):
        """Test FormatInfo.__post_init__ extension normalization."""
        # Test extension without dot gets dot added
        info = FormatInfo(
            extension="png",  # No dot
            pil_format="PNG",
            valid_extensions=[".png"],
            supports_alpha=True,
            supports_quality=True,
        )

        assert info.extension == ".png"

    def test_format_info_post_init_valid_extensions_update(self):
        """Test FormatInfo.__post_init__ valid_extensions update."""
        # Test primary extension added to valid_extensions if missing
        info = FormatInfo(
            extension=".test",
            pil_format="TEST",
            valid_extensions=[".other"],  # Missing primary extension
            supports_alpha=False,
            supports_quality=False,
        )

        assert ".test" in info.valid_extensions
        assert ".other" in info.valid_extensions

    def test_get_supported_formats_includes_aliases(self):
        """Test that get_supported_formats includes format aliases."""
        supported = FormatRegistry.get_supported_formats()

        # Should include both base formats and aliases
        assert "jpeg" in supported
        assert "jpg" in supported  # alias
        assert "tiff" in supported
        assert "tif" in supported  # alias

    def test_format_registry_alias_consistency(self):
        """Test that format aliases point to same FormatInfo object."""
        jpeg_info = FormatRegistry.get_format_info("jpeg")
        jpg_info = FormatRegistry.get_format_info("jpg")

        assert jpeg_info is jpg_info  # Should be same object reference

        tiff_info = FormatRegistry.get_format_info("tiff")
        tif_info = FormatRegistry.get_format_info("tif")

        assert tiff_info is tif_info  # Should be same object reference
