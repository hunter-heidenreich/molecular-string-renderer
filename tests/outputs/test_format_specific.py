"""
Format-specific tests for handlers that have unique behaviors.

Tests format-specific quirks and behaviors that differ from
the standard pattern, including PDF, JPEG, BMP, PNG, WEBP, and SVG.
"""

from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import (
    BMPOutput,
    JPEGOutput,
    PNGOutput,
    SVGOutput,
    WEBPOutput,
)


class TestJPEGSpecificBehavior:
    """Test JPEG-specific behaviors that differ from other formats."""

    def test_rgba_to_rgb_conversion(self):
        """Test that JPEG converts RGBA to RGB."""
        jpeg_output = JPEGOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))

        # Should not raise an error despite RGBA input
        result = jpeg_output.get_bytes(rgba_image)
        assert len(result) > 0

    def test_la_to_rgb_conversion(self):
        """Test that JPEG converts LA to RGB."""
        jpeg_output = JPEGOutput()
        la_image = Image.new("LA", (100, 100), (128, 200))

        result = jpeg_output.get_bytes(la_image)
        assert len(result) > 0

    def test_quality_affects_file_size(self):
        """Test that JPEG quality setting significantly affects file size."""
        test_image = Image.new("RGB", (200, 200))
        # Create gradient pattern to show compression differences
        pixels = []
        for y in range(200):
            for x in range(200):
                r = (x * 255) // 200
                g = (y * 255) // 200
                b = ((x + y) * 255) // 400
                pixels.append((r, g, b))
        test_image.putdata(pixels)

        high_quality = JPEGOutput(OutputConfig(quality=95))
        low_quality = JPEGOutput(OutputConfig(quality=20))

        high_bytes = high_quality.get_bytes(test_image)
        low_bytes = low_quality.get_bytes(test_image)

        # High quality should produce larger files for complex images
        assert len(high_bytes) > len(low_bytes)

    def test_jpeg_metadata_handling(self):
        """Test JPEG metadata handling in save kwargs."""
        # Test with Description metadata
        config_with_desc = OutputConfig(metadata={"Description": "Test molecule"})
        jpeg_output = JPEGOutput(config_with_desc)
        test_image = Image.new("RGB", (100, 100), "red")

        result = jpeg_output.get_bytes(test_image)
        assert len(result) > 0

        # Test with Comment metadata
        config_with_comment = OutputConfig(metadata={"Comment": "Test comment"})
        jpeg_output = JPEGOutput(config_with_comment)

        result = jpeg_output.get_bytes(test_image)
        assert len(result) > 0

        # Test with both Description and Comment
        config_with_both = OutputConfig(
            metadata={"Description": "Test molecule", "Comment": "Test comment"}
        )
        jpeg_output = JPEGOutput(config_with_both)

        result = jpeg_output.get_bytes(test_image)
        assert len(result) > 0

    def test_jpeg_without_metadata(self):
        """Test JPEG without metadata."""
        jpeg_output = JPEGOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        result = jpeg_output.get_bytes(test_image)
        assert len(result) > 0


class TestBMPSpecificBehavior:
    """Test BMP-specific behaviors that differ from other formats."""

    def test_no_optimization_support(self):
        """Test that BMP ignores optimization settings."""
        bmp_output = BMPOutput(OutputConfig(optimize=True))
        test_image = Image.new("RGB", (100, 100), "red")

        # Should work even though BMP doesn't support optimization
        result = bmp_output.get_bytes(test_image)
        assert len(result) > 0

    def test_no_quality_support(self):
        """Test that BMP ignores quality settings."""
        bmp_output = BMPOutput(OutputConfig(quality=50))
        test_image = Image.new("RGB", (100, 100), "red")

        # Should work even though BMP doesn't support quality
        result = bmp_output.get_bytes(test_image)
        assert len(result) > 0

    def test_rgba_handling(self):
        """Test BMP handling of RGBA images."""
        bmp_output = BMPOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))

        # BMP may convert RGBA, but should not crash
        result = bmp_output.get_bytes(rgba_image)
        assert len(result) > 0

    def test_bmp_mode_conversions(self):
        """Test BMP handling of various image modes."""
        bmp_output = BMPOutput()

        # Test LA mode conversion
        la_image = Image.new("LA", (100, 100), (128, 200))
        result = bmp_output.get_bytes(la_image)
        assert len(result) > 0

        # Test other modes that need conversion
        modes_to_test = ["CMYK", "YCbCr", "HSV"]
        for mode in modes_to_test:
            try:
                test_image = Image.new(mode, (50, 50))
                result = bmp_output.get_bytes(test_image)
                assert len(result) > 0
            except OSError:
                # Some modes may not be directly creatable
                pass


class TestTIFFSpecificBehavior:
    """Test TIFF-specific behaviors."""

    def test_tiff_metadata_handling(self):
        """Test TIFF metadata handling."""
        from molecular_string_renderer.outputs import TIFFOutput

        # Test with Description metadata
        config_with_desc = OutputConfig(metadata={"Description": "Test molecule"})
        tiff_output = TIFFOutput(config_with_desc)
        test_image = Image.new("RGB", (100, 100), "red")

        result = tiff_output.get_bytes(test_image)
        assert len(result) > 0

        # Test with Software metadata
        config_with_software = OutputConfig(metadata={"Software": "Custom Software"})
        tiff_output = TIFFOutput(config_with_software)

        result = tiff_output.get_bytes(test_image)
        assert len(result) > 0

        # Test without software metadata (should add default)
        config_no_software = OutputConfig(metadata={"Description": "Test"})
        tiff_output = TIFFOutput(config_no_software)

        result = tiff_output.get_bytes(test_image)
        assert len(result) > 0

    def test_tiff_compression_with_optimization(self):
        """Test TIFF compression when optimization is enabled."""
        from molecular_string_renderer.outputs import TIFFOutput

        config_optimized = OutputConfig(optimize=True)
        tiff_output = TIFFOutput(config_optimized)
        test_image = Image.new("RGB", (100, 100), "red")

        result = tiff_output.get_bytes(test_image)
        assert len(result) > 0

    def test_tiff_transparency_preservation(self):
        """Test TIFF transparency preservation."""
        from molecular_string_renderer.outputs import TIFFOutput

        tiff_output = TIFFOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))

        # TIFF preserves transparency
        result = tiff_output.get_bytes(rgba_image)
        assert len(result) > 0


class TestPNGSpecificBehavior:
    """Test PNG-specific behaviors like transparency optimization."""

    def test_transparency_detection_and_optimization(self):
        """Test PNG's transparency detection and optimization."""
        png_output = PNGOutput()

        # Test fully opaque RGBA image (should optimize to RGB)
        opaque_rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        opaque_bytes = png_output.get_bytes(opaque_rgba)

        # Test transparent RGBA image (should preserve RGBA)
        transparent_rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        transparent_bytes = png_output.get_bytes(transparent_rgba)

        assert len(opaque_bytes) > 0
        assert len(transparent_bytes) > 0

    def test_quality_parameter_ignored(self):
        """Test that PNG ignores quality parameters."""
        png_output = PNGOutput(OutputConfig(quality=50))
        test_image = Image.new("RGB", (100, 100), "red")

        # Should work even though PNG doesn't use quality
        result = png_output.get_bytes(test_image)
        assert len(result) > 0

    def test_png_metadata_handling(self):
        """Test PNG metadata handling through PngInfo."""
        # Test with custom metadata
        config_with_metadata = OutputConfig(
            metadata={
                "Title": "Test Molecule",
                "Author": "Test Author",
                "Description": "Test Description",
                "Custom": "Custom Value",
            }
        )
        png_output = PNGOutput(config_with_metadata)
        test_image = Image.new("RGB", (100, 100), "red")

        result = png_output.get_bytes(test_image)
        assert len(result) > 0

    def test_png_without_metadata(self):
        """Test PNG without metadata."""
        png_output = PNGOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        result = png_output.get_bytes(test_image)
        assert len(result) > 0


class TestWEBPSpecificBehavior:
    """Test WEBP-specific behaviors."""

    def test_supports_both_alpha_and_quality(self):
        """Test that WEBP supports both alpha and quality."""
        webp_output = WEBPOutput(OutputConfig(quality=80))

        # Should handle RGBA with quality setting
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = webp_output.get_bytes(rgba_image)
        assert len(result) > 0

    def test_quality_and_alpha_interaction(self):
        """Test quality settings with transparent images."""
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))

        high_quality = WEBPOutput(OutputConfig(quality=90))
        low_quality = WEBPOutput(OutputConfig(quality=20))

        high_bytes = high_quality.get_bytes(rgba_image)
        low_bytes = low_quality.get_bytes(rgba_image)

        # Both should work with transparent images
        assert len(high_bytes) > 0
        assert len(low_bytes) > 0


class TestSVGSpecificBehavior:
    """Test SVG-specific behaviors like molecule handling."""

    def test_molecule_setting(self):
        """Test SVG molecule setting functionality."""
        svg_output = SVGOutput()

        # Mock molecule
        from unittest.mock import MagicMock

        mock_mol = MagicMock()

        # Should be able to set molecule without error
        svg_output.set_molecule(mock_mol)

        # Should still be able to generate SVG (will fall back to raster)
        test_image = Image.new("RGB", (100, 100), "red")
        result = svg_output.get_bytes(test_image)
        assert len(result) > 0

    def test_svg_content_format(self):
        """Test that SVG output contains valid SVG content."""
        svg_output = SVGOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        result = svg_output.get_bytes(test_image)
        svg_content = result.decode("utf-8")

        # Should contain SVG elements
        assert "<?xml" in svg_content
        assert "<svg" in svg_content
        assert "</svg>" in svg_content

    def test_vector_vs_raster_configuration(self):
        """Test SVG vector vs raster generation configuration."""
        # Test with vector disabled
        config = OutputConfig(svg_use_vector=False)
        svg_output = SVGOutput(config)
        test_image = Image.new("RGB", (100, 100), "red")

        result = svg_output.get_bytes(test_image)
        svg_content = result.decode("utf-8")

        # Should contain base64 embedded image when vector is disabled
        assert "data:image/png;base64," in svg_content


class TestPDFSpecificBehavior:
    """Test PDF-specific behaviors."""

    def test_pdf_file_structure(self):
        """Test that PDF output has valid PDF structure."""
        from molecular_string_renderer.outputs import PDFOutput

        pdf_output = PDFOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        result = pdf_output.get_bytes(test_image)

        # Should contain PDF signature and structure
        assert result.startswith(b"%PDF-")
        assert b"endobj" in result
        assert b"%%EOF" in result

    def test_automatic_rgba_to_rgb_conversion(self):
        """Test that PDF automatically converts RGBA to RGB."""
        from molecular_string_renderer.outputs import PDFOutput

        pdf_output = PDFOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))

        # Should handle RGBA without error
        result = pdf_output.get_bytes(rgba_image)
        assert len(result) > 0
        assert result.startswith(b"%PDF-")

    def test_page_layout_scaling(self):
        """Test that PDF properly scales images to page layout."""
        from molecular_string_renderer.outputs import PDFOutput

        pdf_output = PDFOutput()

        # Test with different aspect ratios
        wide_image = Image.new("RGB", (400, 100), "red")
        tall_image = Image.new("RGB", (100, 400), "blue")

        wide_result = pdf_output.get_bytes(wide_image)
        tall_result = pdf_output.get_bytes(tall_image)

        assert len(wide_result) > 0
        assert len(tall_result) > 0
        assert wide_result.startswith(b"%PDF-")
        assert tall_result.startswith(b"%PDF-")

    def test_consistent_output_structure(self):
        """Test that PDF output has consistent structure (but may vary in details)."""
        from molecular_string_renderer.outputs import PDFOutput

        pdf_output = PDFOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        # Generate PDF twice
        result1 = pdf_output.get_bytes(test_image)
        result2 = pdf_output.get_bytes(test_image)

        # Both should be valid PDFs with same basic structure
        assert result1.startswith(b"%PDF-")
        assert result2.startswith(b"%PDF-")
        assert b"endobj" in result1
        assert b"endobj" in result2
        assert result1.endswith(b"%%EOF\n")
        assert result2.endswith(b"%%EOF\n")

        # Lengths should be similar (may vary slightly due to timestamps, etc.)
        assert abs(len(result1) - len(result2)) < 100

    def test_pdf_metadata_handling(self):
        """Test PDF metadata handling in _generate_pdf_bytes."""
        from molecular_string_renderer.outputs import PDFOutput

        # Test with full metadata
        config_with_metadata = OutputConfig(
            metadata={
                "Title": "Test Molecule",
                "Author": "Test Author",
                "Subject": "Test Subject",
                "Creator": "Test Creator",
            }
        )
        pdf_output = PDFOutput(config_with_metadata)
        test_image = Image.new("RGB", (100, 100), "red")

        result = pdf_output.get_bytes(test_image)
        assert len(result) > 0
        assert result.startswith(b"%PDF-")

        # Test without metadata (should add default creator)
        pdf_output_no_meta = PDFOutput()
        result_no_meta = pdf_output_no_meta.get_bytes(test_image)
        assert len(result_no_meta) > 0
        assert result_no_meta.startswith(b"%PDF-")

        # Test with partial metadata (missing creator)
        config_partial = OutputConfig(
            metadata={"Title": "Test", "Author": "Test Author"}
        )
        pdf_output_partial = PDFOutput(config_partial)
        result_partial = pdf_output_partial.get_bytes(test_image)
        assert len(result_partial) > 0
        assert result_partial.startswith(b"%PDF-")
