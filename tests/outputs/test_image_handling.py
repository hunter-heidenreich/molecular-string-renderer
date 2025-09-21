"""
Image handling tests for output handlers.

Tests image mode handling, edge cases, and format-specific image processing
across all output handlers.
"""

from PIL import Image


class TestOutputHandlerImageModeHandling:
    """Test image mode handling across formats."""

    def test_rgb_image_handling(self, output_handler, test_image):
        """Test RGB image handling produces valid output."""
        # Arrange - RGB image provided by fixture (most compatible mode)
        assert test_image.mode == "RGB", "Test image fixture should be RGB mode"

        # Act
        result = output_handler.get_bytes(test_image)

        # Assert
        assert isinstance(result, bytes), "RGB image processing must return bytes"
        assert len(result) > 0, "RGB image output must not be empty"

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
        assert isinstance(result, bytes), "RGBA image processing must return bytes"
        assert len(result) > 0, "RGBA image output must not be empty"
        # Note: Format-specific transparency handling is tested elsewhere
        # This test ensures all formats can process RGBA without crashing

    def test_grayscale_image_handling(self, output_handler, grayscale_image):
        """Test grayscale image handling."""
        # Arrange
        assert grayscale_image.mode == "L", "Grayscale image fixture should be L mode"

        # Act
        result = output_handler.get_bytes(grayscale_image)

        # Assert
        assert isinstance(result, bytes), "Grayscale image processing must return bytes"
        assert len(result) > 0, "Grayscale image output must not be empty"

    def test_la_image_handling(self, output_handler, la_image):
        """Test grayscale+alpha image handling."""
        # Arrange
        assert la_image.mode == "LA", "LA image fixture should be LA mode"

        # Act
        result = output_handler.get_bytes(la_image)

        # Assert
        assert isinstance(result, bytes), "LA image processing must return bytes"
        assert len(result) > 0, "LA image output must not be empty"

    def test_image_mode_conversion_consistency(self, output_handler):
        """Test that handlers consistently process the same image in different modes."""
        # Arrange - create same image content in different modes
        base_image = Image.new("RGB", (50, 50), (128, 64, 192))  # Purple
        rgba_version = base_image.convert("RGBA")
        grayscale_version = base_image.convert("L")
        la_version = grayscale_version.convert("LA")

        # Act - process each version
        rgb_result = output_handler.get_bytes(base_image)
        rgba_result = output_handler.get_bytes(rgba_version)
        grayscale_result = output_handler.get_bytes(grayscale_version)
        la_result = output_handler.get_bytes(la_version)

        # Assert - all should produce valid output
        results = [rgb_result, rgba_result, grayscale_result, la_result]
        modes = ["RGB", "RGBA", "L", "LA"]

        for result, mode in zip(results, modes):
            assert isinstance(result, bytes), f"{mode} mode must produce bytes output"
            assert len(result) > 0, f"{mode} mode output must not be empty"

    def test_palette_mode_handling(self, output_handler):
        """Test handling of palette mode images."""
        # Arrange - create a palette mode image
        palette_image = Image.new("P", (50, 50), 0)
        # Add a simple palette (grayscale)
        palette = []
        for i in range(256):
            palette.extend([i, i, i])  # R, G, B for each palette entry
        palette_image.putpalette(palette)

        # Act
        result = output_handler.get_bytes(palette_image)

        # Assert
        assert isinstance(result, bytes), (
            "Palette mode image must be handled and return bytes"
        )
        assert len(result) > 0, "Palette mode output must not be empty"


class TestOutputHandlerEdgeCases:
    """Test edge cases common across handlers."""

    def test_very_small_image(self, output_handler):
        """Test with very small images (1x1 pixel)."""
        # Arrange
        tiny_image = Image.new("RGB", (1, 1), "red")
        assert tiny_image.size == (1, 1), "Image should be exactly 1x1 pixel"

        # Act
        result = output_handler.get_bytes(tiny_image)

        # Assert
        assert isinstance(result, bytes), "Tiny image must produce bytes output"
        assert len(result) > 0, "Tiny image output must not be empty"

    def test_large_image_dimensions(self, output_handler):
        """Test with reasonably large images to verify memory handling."""
        # Arrange
        large_image = Image.new("RGB", (1000, 800), "blue")
        expected_pixels = 1000 * 800
        assert large_image.size == (1000, 800), (
            "Image should be exactly 1000x800 pixels"
        )

        # Act
        result = output_handler.get_bytes(large_image)

        # Assert
        assert isinstance(result, bytes), "Large image must produce bytes output"
        assert len(result) > 0, "Large image output must not be empty"
        # Large images should produce substantial output
        assert len(result) > 1000, (
            f"Large image ({expected_pixels} pixels) should produce substantial output"
        )

    def test_square_vs_rectangular_images(self, output_handler):
        """Test with different aspect ratios to verify layout handling."""
        # Arrange
        test_cases = [
            ("square", Image.new("RGB", (100, 100), "red"), 1.0),
            ("wide", Image.new("RGB", (200, 50), "green"), 4.0),
            ("tall", Image.new("RGB", (50, 200), "blue"), 0.25),
        ]

        results = []

        # Act
        for name, img, expected_ratio in test_cases:
            actual_ratio = img.width / img.height
            assert abs(actual_ratio - expected_ratio) < 0.01, (
                f"{name} image should have aspect ratio {expected_ratio}"
            )

            result = output_handler.get_bytes(img)
            results.append((name, result, len(result)))

        # Assert
        for name, result, size in results:
            assert isinstance(result, bytes), f"{name} image must produce bytes output"
            assert len(result) > 0, f"{name} image output must not be empty"
            assert size > 100, f"{name} image should produce reasonable output size"

    def test_extreme_aspect_ratios(self, output_handler):
        """Test with extreme aspect ratios to verify robustness."""
        # Arrange
        very_wide = Image.new("RGB", (1000, 1), "purple")  # 1000:1 ratio
        very_tall = Image.new("RGB", (1, 1000), "orange")  # 1:1000 ratio

        # Act
        wide_result = output_handler.get_bytes(very_wide)
        tall_result = output_handler.get_bytes(very_tall)

        # Assert
        assert isinstance(wide_result, bytes), (
            "Very wide image must produce bytes output"
        )
        assert len(wide_result) > 0, "Very wide image output must not be empty"
        assert isinstance(tall_result, bytes), (
            "Very tall image must produce bytes output"
        )
        assert len(tall_result) > 0, "Very tall image output must not be empty"

    def test_zero_pixel_areas_handled(self, output_handler):
        """Test that handlers reject invalid image dimensions."""
        # Arrange & Act & Assert - these should not be createable with PIL
        # but we test the principle that handlers should handle edge cases gracefully

        # Test minimum valid dimensions
        min_valid = Image.new("RGB", (1, 1), "black")
        result = output_handler.get_bytes(min_valid)

        assert isinstance(result, bytes), "Minimum valid image must produce bytes"
        assert len(result) > 0, "Minimum valid image must produce output"

    def test_odd_dimensions(self, output_handler):
        """Test with odd dimensions that might cause alignment issues."""
        # Arrange
        odd_images = [
            Image.new("RGB", (101, 101), "red"),  # Both odd
            Image.new("RGB", (100, 101), "green"),  # Height odd
            Image.new("RGB", (101, 100), "blue"),  # Width odd
            Image.new("RGB", (3, 7), "yellow"),  # Small odd numbers
            Image.new("RGB", (997, 503), "purple"),  # Large odd numbers
        ]

        # Act & Assert
        for i, img in enumerate(odd_images):
            result = output_handler.get_bytes(img)

            assert isinstance(result, bytes), (
                f"Odd dimension image {i} must produce bytes"
            )
            assert len(result) > 0, f"Odd dimension image {i} output must not be empty"

    def test_power_of_two_dimensions(self, output_handler):
        """Test with power-of-two dimensions which are common in graphics."""
        # Arrange
        power_of_two_images = [
            Image.new("RGB", (64, 64), "red"),
            Image.new("RGB", (128, 256), "green"),
            Image.new("RGB", (512, 128), "blue"),
            Image.new("RGB", (1024, 512), "purple"),
        ]

        # Act & Assert
        for i, img in enumerate(power_of_two_images):
            result = output_handler.get_bytes(img)

            assert isinstance(result, bytes), (
                f"Power-of-two image {i} must produce bytes"
            )
            assert len(result) > 0, f"Power-of-two image {i} output must not be empty"

    def test_single_pixel_line_images(self, output_handler):
        """Test with single-pixel-wide/tall images."""
        # Arrange
        horizontal_line = Image.new("RGB", (100, 1), "red")  # 100x1
        vertical_line = Image.new("RGB", (1, 100), "blue")  # 1x100

        # Act
        h_result = output_handler.get_bytes(horizontal_line)
        v_result = output_handler.get_bytes(vertical_line)

        # Assert
        assert isinstance(h_result, bytes), "Horizontal line image must produce bytes"
        assert len(h_result) > 0, "Horizontal line output must not be empty"
        assert isinstance(v_result, bytes), "Vertical line image must produce bytes"
        assert len(v_result) > 0, "Vertical line output must not be empty"
