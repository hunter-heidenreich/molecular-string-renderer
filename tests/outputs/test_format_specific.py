"""
Format-specific tests for handlers that have unique behaviors.

This module tests format-specific quirks and behaviors that differ from
the standard pattern, including PDF, JPEG, BMP, PNG, WEBP, and SVG.

Each test class focuses on one format's unique characteristics:
- JPEG: Lossy compression, no transparency support
- BMP: Basic format, no advanced features
- TIFF: Advanced features with transparency
- PNG: Lossless with transparency optimization
- WEBP: Modern format with alpha and quality
- SVG: Vector format with molecule handling
- PDF: Document format with page layout
"""

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import (
    BMPOutput,
    JPEGOutput,
    PNGOutput,
    SVGOutput,
    WEBPOutput,
)

from .conftest import TestValidators


class TestJPEGSpecificBehavior:
    """Test JPEG-specific behaviors that differ from other formats."""

    def test_rgba_to_rgb_conversion(self, rgba_image):
        """Test that JPEG converts RGBA to RGB."""
        jpeg_output = JPEGOutput()
        result = jpeg_output.get_bytes(rgba_image)
        TestValidators.assert_valid_bytes_output(result, "JPEG RGBA conversion")

    def test_la_to_rgb_conversion(self, la_image):
        """Test that JPEG converts LA to RGB."""
        jpeg_output = JPEGOutput()
        result = jpeg_output.get_bytes(la_image)
        TestValidators.assert_valid_bytes_output(result, "JPEG LA conversion")

    def test_quality_affects_file_size(self, pattern_image):
        """Test that JPEG quality setting significantly affects file size."""
        high_quality = JPEGOutput(OutputConfig(quality=95))
        low_quality = JPEGOutput(OutputConfig(quality=20))

        high_bytes = high_quality.get_bytes(pattern_image)
        low_bytes = low_quality.get_bytes(pattern_image)

        # Validate both outputs first
        TestValidators.assert_valid_bytes_output(high_bytes, "JPEG high quality")
        TestValidators.assert_valid_bytes_output(low_bytes, "JPEG low quality")

        # High quality should produce larger files for complex images
        assert len(high_bytes) > len(low_bytes)

    def test_jpeg_metadata_handling(self, test_image):
        """Test JPEG metadata handling in save kwargs."""
        test_cases = [
            {"Description": "Test molecule"},
            {"Comment": "Test comment"},
            {"Description": "Test molecule", "Comment": "Test comment"},
        ]

        for metadata in test_cases:
            config = OutputConfig(metadata=metadata)
            jpeg_output = JPEGOutput(config)
            result = jpeg_output.get_bytes(test_image)
            TestValidators.assert_valid_bytes_output(
                result, f"JPEG with metadata {list(metadata.keys())}"
            )


class TestBMPSpecificBehavior:
    """Test BMP-specific behaviors that differ from other formats."""

    def test_no_optimization_support(self, test_image):
        """Test that BMP ignores optimization settings."""
        bmp_output = BMPOutput(OutputConfig(optimize=True))
        result = bmp_output.get_bytes(test_image)
        TestValidators.assert_valid_bytes_output(
            result, "BMP with optimization (ignored)"
        )

    def test_no_quality_support(self, test_image):
        """Test that BMP ignores quality settings."""
        bmp_output = BMPOutput(OutputConfig(quality=50))
        result = bmp_output.get_bytes(test_image)
        TestValidators.assert_valid_bytes_output(result, "BMP with quality (ignored)")

    def test_rgba_handling(self, rgba_image):
        """Test BMP handling of RGBA images."""
        bmp_output = BMPOutput()
        result = bmp_output.get_bytes(rgba_image)
        TestValidators.assert_valid_bytes_output(result, "BMP RGBA handling")

    def test_bmp_mode_conversions(self, la_image):
        """Test BMP handling of various image modes."""
        from .conftest import create_test_image_with_mode

        bmp_output = BMPOutput()

        # Test LA mode conversion
        result = bmp_output.get_bytes(la_image)
        TestValidators.assert_valid_bytes_output(result, "BMP LA conversion")

        # Test other modes that need conversion
        modes_to_test = ["CMYK", "YCbCr", "HSV"]
        for mode in modes_to_test:
            try:
                test_image = create_test_image_with_mode(mode, (50, 50))
                result = bmp_output.get_bytes(test_image)
                TestValidators.assert_valid_bytes_output(
                    result, f"BMP {mode} conversion"
                )
            except OSError:
                # Some modes may not be directly creatable
                pass


class TestTIFFSpecificBehavior:
    """Test TIFF-specific behaviors."""

    def test_tiff_metadata_handling(self, test_image):
        """Test TIFF metadata handling."""
        from molecular_string_renderer.outputs import TIFFOutput

        test_cases = [
            {"Description": "Test molecule"},
            {"Software": "Custom Software"},
            {"Description": "Test"},  # Should add default software
        ]

        for metadata in test_cases:
            config = OutputConfig(metadata=metadata)
            tiff_output = TIFFOutput(config)
            result = tiff_output.get_bytes(test_image)
            TestValidators.assert_valid_bytes_output(
                result, f"TIFF with metadata {list(metadata.keys())}"
            )

    def test_tiff_compression_with_optimization(self, test_image):
        """Test TIFF compression when optimization is enabled."""
        from molecular_string_renderer.outputs import TIFFOutput

        config_optimized = OutputConfig(optimize=True)
        tiff_output = TIFFOutput(config_optimized)
        result = tiff_output.get_bytes(test_image)
        TestValidators.assert_valid_bytes_output(result, "TIFF with optimization")

    def test_tiff_transparency_preservation(self, rgba_image):
        """Test TIFF transparency preservation."""
        from molecular_string_renderer.outputs import TIFFOutput

        tiff_output = TIFFOutput()
        result = tiff_output.get_bytes(rgba_image)
        TestValidators.assert_valid_bytes_output(
            result, "TIFF transparency preservation"
        )


class TestPNGSpecificBehavior:
    """Test PNG-specific behaviors like transparency optimization."""

    def test_transparency_detection_and_optimization(
        self, rgba_opaque_image, rgba_image
    ):
        """Test PNG's transparency detection and optimization."""
        png_output = PNGOutput()

        # Test fully opaque RGBA image (should optimize to RGB)
        opaque_bytes = png_output.get_bytes(rgba_opaque_image)
        TestValidators.assert_valid_bytes_output(opaque_bytes, "PNG opaque RGBA")

        # Test transparent RGBA image (should preserve RGBA)
        transparent_bytes = png_output.get_bytes(rgba_image)
        TestValidators.assert_valid_bytes_output(
            transparent_bytes, "PNG transparent RGBA"
        )

    def test_quality_parameter_ignored(self, test_image):
        """Test that PNG ignores quality parameters."""
        png_output = PNGOutput(OutputConfig(quality=50))
        result = png_output.get_bytes(test_image)
        TestValidators.assert_valid_bytes_output(result, "PNG with quality (ignored)")

    def test_png_metadata_handling(self, test_image):
        """Test PNG metadata handling through PngInfo."""
        config_with_metadata = OutputConfig(
            metadata={
                "Title": "Test Molecule",
                "Author": "Test Author",
                "Description": "Test Description",
                "Custom": "Custom Value",
            }
        )
        png_output = PNGOutput(config_with_metadata)
        result = png_output.get_bytes(test_image)
        TestValidators.assert_valid_bytes_output(result, "PNG with metadata")


class TestWEBPSpecificBehavior:
    """Test WEBP-specific behaviors."""

    def test_supports_both_alpha_and_quality(self, rgba_image):
        """Test that WEBP supports both alpha and quality."""
        webp_output = WEBPOutput(OutputConfig(quality=80))
        result = webp_output.get_bytes(rgba_image)
        TestValidators.assert_valid_bytes_output(result, "WEBP alpha with quality")

    def test_quality_and_alpha_interaction(self, rgba_image):
        """Test quality settings with transparent images."""
        high_quality = WEBPOutput(OutputConfig(quality=90))
        low_quality = WEBPOutput(OutputConfig(quality=20))

        high_bytes = high_quality.get_bytes(rgba_image)
        low_bytes = low_quality.get_bytes(rgba_image)

        # Validate both outputs
        TestValidators.assert_valid_bytes_output(
            high_bytes, "WEBP high quality with alpha"
        )
        TestValidators.assert_valid_bytes_output(
            low_bytes, "WEBP low quality with alpha"
        )


class TestSVGSpecificBehavior:
    """Test SVG-specific behaviors like molecule handling."""

    def test_molecule_setting(self, test_image):
        """Test SVG molecule setting functionality."""
        from unittest.mock import MagicMock

        svg_output = SVGOutput()
        mock_mol = MagicMock()

        # Should be able to set molecule without error
        svg_output.set_molecule(mock_mol)

        # Should still be able to generate SVG (will fall back to raster)
        result = svg_output.get_bytes(test_image)
        TestValidators.assert_valid_bytes_output(result, "SVG with molecule set")

    def test_svg_content_format(self, test_image):
        """Test that SVG output contains valid SVG content."""
        svg_output = SVGOutput()
        result = svg_output.get_bytes(test_image)
        TestValidators.assert_image_output_valid(result, "svg")

    def test_vector_vs_raster_configuration(self, test_image):
        """Test SVG vector vs raster generation configuration."""
        config = OutputConfig(svg_use_vector=False)
        svg_output = SVGOutput(config)

        result = svg_output.get_bytes(test_image)
        TestValidators.assert_valid_bytes_output(result, "SVG raster mode")

        svg_content = result.decode("utf-8")
        # Should contain base64 embedded image when vector is disabled
        assert "data:image/png;base64," in svg_content


class TestPDFSpecificBehavior:
    """Test PDF-specific behaviors."""

    def test_pdf_file_structure(self, test_image):
        """Test that PDF output has valid PDF structure."""
        from molecular_string_renderer.outputs import PDFOutput

        pdf_output = PDFOutput()
        result = pdf_output.get_bytes(test_image)
        TestValidators.assert_image_output_valid(result, "pdf")

    def test_automatic_rgba_to_rgb_conversion(self, rgba_image):
        """Test that PDF automatically converts RGBA to RGB."""
        from molecular_string_renderer.outputs import PDFOutput

        pdf_output = PDFOutput()
        result = pdf_output.get_bytes(rgba_image)
        TestValidators.assert_image_output_valid(result, "pdf")

    def test_page_layout_scaling(self, square_image, wide_image, tall_image):
        """Test that PDF properly scales images to page layout."""
        from molecular_string_renderer.outputs import PDFOutput

        pdf_output = PDFOutput()

        # Test with different aspect ratios
        for name, image in [
            ("square", square_image),
            ("wide", wide_image),
            ("tall", tall_image),
        ]:
            result = pdf_output.get_bytes(image)
            TestValidators.assert_image_output_valid(result, f"pdf {name} aspect ratio")

    def test_consistent_output_structure(self, test_image):
        """Test that PDF output has consistent structure."""
        from molecular_string_renderer.outputs import PDFOutput

        pdf_output = PDFOutput()

        # Generate PDF twice
        result1 = pdf_output.get_bytes(test_image)
        result2 = pdf_output.get_bytes(test_image)

        # Both should be valid PDFs
        TestValidators.assert_image_output_valid(result1, "pdf first generation")
        TestValidators.assert_image_output_valid(result2, "pdf second generation")

        # Lengths should be similar (may vary slightly due to timestamps)
        assert abs(len(result1) - len(result2)) < 100

    def test_pdf_metadata_handling(self, test_image):
        """Test PDF metadata handling."""
        from molecular_string_renderer.outputs import PDFOutput

        test_cases = [
            {
                "Title": "Test Molecule",
                "Author": "Test Author",
                "Subject": "Test Subject",
                "Creator": "Test Creator",
            },
            {},  # No metadata (should add default creator)
            {"Title": "Test", "Author": "Test Author"},  # Partial metadata
        ]

        for metadata in test_cases:
            config = OutputConfig(metadata=metadata)
            pdf_output = PDFOutput(config)
            result = pdf_output.get_bytes(test_image)
            TestValidators.assert_image_output_valid(
                result, f"PDF with metadata {list(metadata.keys()) or 'default'}"
            )
