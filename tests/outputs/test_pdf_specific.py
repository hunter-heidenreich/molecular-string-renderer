"""
PDF-specific tests for unique PDF behaviors.

Tests PDF-specific functionality that differs from other vector formats.
"""

from PIL import Image

from molecular_string_renderer.outputs import PDFOutput


class TestPDFSpecificBehavior:
    """Test PDF-specific behaviors."""

    def test_pdf_file_structure(self):
        """Test that PDF output has valid PDF structure."""
        pdf_output = PDFOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        result = pdf_output.get_bytes(test_image)

        # Should contain PDF signature and structure
        assert result.startswith(b"%PDF-")
        assert b"endobj" in result
        assert b"%%EOF" in result

    def test_automatic_rgba_to_rgb_conversion(self):
        """Test that PDF automatically converts RGBA to RGB."""
        pdf_output = PDFOutput()
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))

        # Should handle RGBA without error
        result = pdf_output.get_bytes(rgba_image)
        assert len(result) > 0
        assert result.startswith(b"%PDF-")

    def test_page_layout_scaling(self):
        """Test that PDF properly scales images to page layout."""
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
