"""
Test PDF output functionality.

This module provides comprehensive testing for PDF output generation,
focusing on actual functionality rather than mocking to discover real bugs.
"""

import tempfile
from pathlib import Path
from PIL import Image
import pytest

from molecular_string_renderer.outputs.vector import PDFOutput
from molecular_string_renderer.config import OutputConfig


class TestPDFOutputProperties:
    """Test basic properties and inheritance."""

    def test_file_extension(self):
        """Test file extension property."""
        output = PDFOutput()
        assert output.file_extension == ".pdf"

    def test_format_name_inherited(self):
        """Test format name inheritance."""
        output = PDFOutput()
        assert output.format_name == "pdf"


class TestPDFOutputInitialization:
    """Test initialization patterns."""

    def test_init_without_config(self):
        """Test initialization without config."""
        output = PDFOutput()
        assert output.config is not None
        assert output.config.format == "png"  # Default config

    def test_init_with_config(self):
        """Test initialization with config."""
        config = OutputConfig(format="pdf", quality=80)
        output = PDFOutput(config)
        assert output.config is config
        assert output.config.format == "pdf"
        assert output.config.quality == 80


class TestPDFOutputFunctionality:
    """Test core PDF generation functionality."""

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (100, 100), "red")

    @pytest.fixture
    def large_image(self):
        """Create a large test image."""
        return Image.new("RGB", (1000, 800), "blue")

    @pytest.fixture
    def small_image(self):
        """Create a very small test image."""
        return Image.new("RGB", (10, 10), "green")

    def test_get_bytes_basic(self, test_image):
        """Test basic PDF byte generation."""
        output = PDFOutput()
        pdf_bytes = output.get_bytes(test_image)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF-")  # PDF signature

    def test_get_bytes_large_image(self, large_image):
        """Test PDF generation with large image."""
        output = PDFOutput()
        pdf_bytes = output.get_bytes(large_image)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF-")

    def test_get_bytes_small_image(self, small_image):
        """Test PDF generation with very small image."""
        output = PDFOutput()
        pdf_bytes = output.get_bytes(small_image)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF-")

    def test_save_to_file(self, test_image):
        """Test saving PDF to file."""
        output = PDFOutput()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            try:
                output.save(test_image, tmp.name)

                path = Path(tmp.name)
                assert path.exists()
                assert path.stat().st_size > 0

                # Verify it's a valid PDF
                with open(path, "rb") as f:
                    content = f.read()
                    assert content.startswith(b"%PDF-")

            finally:
                Path(tmp.name).unlink(missing_ok=True)

    def test_rgba_conversion(self):
        """Test automatic RGBA to RGB conversion."""
        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        output = PDFOutput()

        pdf_bytes = output.get_bytes(rgba_image)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF-")

    def test_different_image_modes(self):
        """Test with different image modes."""
        modes = ["L", "P", "RGB", "RGBA"]
        output = PDFOutput()

        for mode in modes:
            if mode == "P":
                # Palette mode needs special handling
                image = Image.new("RGB", (50, 50), "red").convert("P")
            else:
                image = Image.new(
                    mode, (50, 50), "red" if mode in ["RGB", "RGBA"] else 128
                )

            pdf_bytes = output.get_bytes(image)
            assert isinstance(pdf_bytes, bytes)
            assert len(pdf_bytes) > 0
            assert pdf_bytes.startswith(b"%PDF-")

    def test_extreme_aspect_ratios(self):
        """Test with extreme aspect ratios."""
        # Very wide image
        wide_image = Image.new("RGB", (1000, 10), "red")
        output = PDFOutput()
        pdf_bytes = output.get_bytes(wide_image)
        assert pdf_bytes.startswith(b"%PDF-")

        # Very tall image
        tall_image = Image.new("RGB", (10, 1000), "blue")
        pdf_bytes = output.get_bytes(tall_image)
        assert pdf_bytes.startswith(b"%PDF-")

    def test_pdf_file_structure(self, test_image):
        """Test that generated PDF has expected structure."""
        output = PDFOutput()
        pdf_bytes = output.get_bytes(test_image)

        # Convert to string for content analysis
        content = pdf_bytes.decode("latin-1")

        # Check for required PDF structure elements
        assert "%PDF-" in content
        assert "endobj" in content
        assert "xref" in content
        assert "%%EOF" in content

    def test_consistent_output(self, test_image):
        """Test that same input produces consistent output."""
        output = PDFOutput()

        # Generate PDF twice
        pdf_bytes1 = output.get_bytes(test_image)
        pdf_bytes2 = output.get_bytes(test_image)

        # They might not be exactly identical due to timestamps, but should be similar size
        assert len(pdf_bytes1) > 0
        assert len(pdf_bytes2) > 0
        assert abs(len(pdf_bytes1) - len(pdf_bytes2)) < 100  # Allow small differences


class TestPDFOutputErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_image(self):
        """Test handling of invalid image."""
        output = PDFOutput()

        with pytest.raises((AttributeError, TypeError, OSError, IOError)):
            output.get_bytes(None)

    def test_save_invalid_path(self):
        """Test saving to invalid path."""
        output = PDFOutput()
        image = Image.new("RGB", (100, 100), "red")

        with pytest.raises((OSError, IOError, PermissionError)):
            output.save(image, "/invalid/path/that/doesnt/exist.pdf")


class TestPDFOutputInheritance:
    """Test inheritance and interface compliance."""

    def test_is_vector_output(self):
        """Test that PDFOutput is recognized as vector output."""
        from molecular_string_renderer.outputs.base import VectorOutputHandler

        output = PDFOutput()
        assert isinstance(output, VectorOutputHandler)

    def test_implements_required_methods(self):
        """Test that all required methods are implemented."""
        output = PDFOutput()

        # Check that essential methods exist and are callable
        assert hasattr(output, "get_bytes")
        assert callable(output.get_bytes)
        assert hasattr(output, "save")
        assert callable(output.save)
        assert hasattr(output, "file_extension")
        assert hasattr(output, "format_name")


class TestPDFOutputSpecificBugTests:
    """Test specific scenarios that might reveal bugs."""

    def test_memory_usage_large_images(self):
        """Test memory usage with large images doesn't cause issues."""
        # Create a reasonably large image (not too large to avoid CI issues)
        large_image = Image.new("RGB", (2000, 1500), "red")
        output = PDFOutput()

        # This should not cause memory errors
        pdf_bytes = output.get_bytes(large_image)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF-")

    def test_threading_safety(self):
        """Test that multiple PDF outputs can be generated concurrently."""
        import threading

        results = []
        errors = []

        def generate_pdf():
            try:
                image = Image.new("RGB", (100, 100), "red")
                output = PDFOutput()
                pdf_bytes = output.get_bytes(image)
                results.append(len(pdf_bytes))
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_pdf)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(size > 0 for size in results)

    def test_unicode_handling(self):
        """Test that Unicode filenames are handled correctly."""
        output = PDFOutput()
        image = Image.new("RGB", (100, 100), "red")

        with tempfile.NamedTemporaryFile(suffix="_测试.pdf", delete=False) as tmp:
            try:
                output.save(image, tmp.name)

                path = Path(tmp.name)
                assert path.exists()
                assert path.stat().st_size > 0

            finally:
                Path(tmp.name).unlink(missing_ok=True)

    def test_repeated_operations(self):
        """Test repeated operations don't cause issues."""
        output = PDFOutput()
        image = Image.new("RGB", (100, 100), "red")

        # Generate many PDFs to test for memory leaks or accumulation issues
        for i in range(10):
            pdf_bytes = output.get_bytes(image)
            assert len(pdf_bytes) > 0
            assert pdf_bytes.startswith(b"%PDF-")
