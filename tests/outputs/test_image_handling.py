"""
Image handling tests for output handlers.

This module focuses specifically on how output handlers process different
image types, modes, and edge cases. It complements the core functionality
tests by providing detailed coverage of image-specific behaviors.

Test organization:
- ImageModeHandling: Tests for RGB, RGBA, grayscale, palette modes
- EdgeCases: Small/large images, extreme aspect ratios, unusual dimensions

All tests use conftest fixtures for consistency and maintainability.
"""

from .conftest import (
    FormatCapabilities,
    TestValidators,
    create_test_image_with_mode,
)


class TestOutputHandlerImageModeHandling:
    """Test image mode handling across formats."""

    def test_rgb_image_handling(self, output_handler, test_image):
        """Test RGB image handling produces valid output."""
        # Arrange - RGB image provided by fixture (most compatible mode)
        assert test_image.mode == "RGB", "Test image fixture should be RGB mode"

        # Act
        result = output_handler.get_bytes(test_image)

        # Assert
        TestValidators.assert_valid_bytes_output(result, "RGB image processing")

    def test_rgba_image_handling(self, output_handler, rgba_image, format_name):
        """Test RGBA image handling with transparency."""
        # Arrange
        assert rgba_image.mode == "RGBA", "RGBA image fixture should have RGBA mode"
        assert rgba_image.getchannel("A").getextrema()[0] < 255, (
            "RGBA image should have transparency"
        )

        # Act
        result = output_handler.get_bytes(rgba_image)

        # Assert
        TestValidators.assert_valid_bytes_output(result, "RGBA image processing")

        # Verify format capabilities are respected
        capabilities = FormatCapabilities.get(format_name)
        if not capabilities.get("supports_alpha", False):
            # Format doesn't support transparency - conversion should occur
            assert len(result) > 0, "Non-alpha format should still produce valid output"

    def test_grayscale_image_handling(self, output_handler, grayscale_image):
        """Test grayscale image handling."""
        # Arrange
        assert grayscale_image.mode == "L", "Grayscale image fixture should be L mode"

        # Act
        result = output_handler.get_bytes(grayscale_image)

        # Assert
        TestValidators.assert_valid_bytes_output(result, "Grayscale image processing")

    def test_la_image_handling(self, output_handler, la_image):
        """Test grayscale+alpha image handling."""
        # Arrange
        assert la_image.mode == "LA", "LA image fixture should be LA mode"

        # Act
        result = output_handler.get_bytes(la_image)

        # Assert
        TestValidators.assert_valid_bytes_output(result, "LA image processing")

    def test_palette_mode_handling(self, output_handler):
        """Test handling of palette mode images."""
        # Arrange - use conftest helper for consistent P mode creation
        palette_image = create_test_image_with_mode("P", (50, 50))

        # Act
        result = output_handler.get_bytes(palette_image)

        # Assert
        TestValidators.assert_valid_bytes_output(
            result, "Palette mode image processing"
        )

    def test_parametrized_image_modes_from_conftest(self, output_handler, varied_image):
        """Test with various image modes using conftest parametrized fixture."""
        # Act
        result = output_handler.get_bytes(varied_image)

        # Assert
        TestValidators.assert_valid_bytes_output(
            result, f"{varied_image.mode} mode processing"
        )


class TestOutputHandlerEdgeCases:
    """Test edge cases common across handlers."""

    def test_very_small_image(self, output_handler, small_image):
        """Test with very small images using conftest fixture."""
        # Act
        result = output_handler.get_bytes(small_image)

        # Assert
        TestValidators.assert_valid_bytes_output(result, "Small image processing")

    def test_large_image_dimensions(self, output_handler, large_image, format_name):
        """Test with large images using conftest fixture."""
        # Act
        result = output_handler.get_bytes(large_image)

        # Assert
        TestValidators.assert_valid_bytes_output(result, "Large image processing")

        # Use reasonable size expectations for large images
        min_expected_size = 40 if format_name.lower() == "webp" else 500
        assert len(result) > min_expected_size, (
            f"Large image should produce output >= {min_expected_size} bytes "
            f"for {format_name} format, got {len(result)} bytes"
        )

    def test_extreme_aspect_ratios(self, output_handler):
        """Test with extreme aspect ratios to verify robustness."""
        # Arrange - use conftest helper for consistent creation
        very_wide = create_test_image_with_mode("RGB", (1000, 1))  # 1000:1 ratio
        very_tall = create_test_image_with_mode("RGB", (1, 1000))  # 1:1000 ratio

        # Act
        wide_result = output_handler.get_bytes(very_wide)
        tall_result = output_handler.get_bytes(very_tall)

        # Assert
        TestValidators.assert_valid_bytes_output(
            wide_result, "Very wide image processing"
        )
        TestValidators.assert_valid_bytes_output(
            tall_result, "Very tall image processing"
        )

    def test_parametrized_dimensions(self, output_handler, image_dimensions):
        """Test various image dimensions using conftest parametrized fixture."""
        # Arrange
        test_image = create_test_image_with_mode("RGB", image_dimensions)

        # Act
        result = output_handler.get_bytes(test_image)

        # Assert
        TestValidators.assert_valid_bytes_output(
            result, f"Image {image_dimensions} processing"
        )

    def test_edge_case_dimensions(self, output_handler, edge_case_dimensions):
        """Test edge case dimensions using conftest parametrized fixture."""
        test_image = create_test_image_with_mode("RGB", edge_case_dimensions)
        result = output_handler.get_bytes(test_image)
        TestValidators.assert_valid_bytes_output(
            result, f"Edge case {edge_case_dimensions} processing"
        )

    def test_aspect_ratio_fixtures(
        self, output_handler, square_image, wide_image, tall_image
    ):
        """Test aspect ratios using dedicated conftest fixtures."""
        # Act & Assert for each fixture
        for name, img in [
            ("square", square_image),
            ("wide", wide_image),
            ("tall", tall_image),
        ]:
            result = output_handler.get_bytes(img)
            TestValidators.assert_valid_bytes_output(
                result, f"{name} image fixture processing"
            )
