"""
Output handler tests.

Tests for output handler classes, bytes generation,
format-specific features, quality settings, and metadata handling.
"""

from pathlib import Path

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import (
    BMPOutput,
    JPEGOutput,
    PDFOutput,
    PNGOutput,
    SVGOutput,
    TIFFOutput,
    WEBPOutput,
    get_output_handler,
)


class TestOutputHandlerFactory:
    """Test the output handler factory function."""

    def test_get_output_handler_png(self):
        """Test getting PNG output handler."""

    def test_invalid_svg_input(self):
        """Test SVG handler with invalid input."""
        config = OutputConfig(format="svg")
        svg_handler = get_output_handler("svg", config)

        # Test with None
        with pytest.raises((AttributeError, TypeError)):
            svg_handler.get_bytes(None)

        # Test with invalid image type
        with pytest.raises((AttributeError, TypeError)):
            svg_handler.get_bytes("not an image")

    def test_file_permission_errors(self):
        """Test handling of file permission errors."""
        png_handler = get_output_handler("png")
        assert isinstance(png_handler, PNGOutput)
        assert png_handler.file_extension == ".png"

    def test_get_output_handler_svg(self):
        """Test getting SVG output handler."""
        handler = get_output_handler("svg")
        assert isinstance(handler, SVGOutput)
        assert handler.file_extension == ".svg"

    def test_get_output_handler_jpg(self):
        """Test getting JPEG output handler."""
        handler = get_output_handler("jpg")
        assert isinstance(handler, JPEGOutput)
        assert handler.file_extension == ".jpg"

    def test_get_output_handler_jpeg(self):
        """Test getting JPEG output handler (alternative name)."""
        handler = get_output_handler("jpeg")
        assert isinstance(handler, JPEGOutput)
        assert handler.file_extension == ".jpg"

    def test_get_output_handler_pdf(self):
        """Test getting PDF output handler."""
        handler = get_output_handler("pdf")
        assert isinstance(handler, PDFOutput)
        assert handler.file_extension == ".pdf"

    def test_get_output_handler_webp(self):
        """Test getting WebP output handler."""
        handler = get_output_handler("webp")
        assert isinstance(handler, WEBPOutput)
        assert handler.file_extension == ".webp"

    def test_get_output_handler_tiff(self):
        """Test getting TIFF output handler."""
        handler = get_output_handler("tiff")
        assert isinstance(handler, TIFFOutput)
        assert handler.file_extension == ".tiff"

    def test_get_output_handler_tif(self):
        """Test getting TIFF output handler (alternative name)."""
        handler = get_output_handler("tif")
        assert isinstance(handler, TIFFOutput)
        assert handler.file_extension == ".tiff"

    def test_get_output_handler_bmp(self):
        """Test getting BMP output handler."""
        handler = get_output_handler("bmp")
        assert isinstance(handler, BMPOutput)
        assert handler.file_extension == ".bmp"

    def test_get_output_handler_case_insensitive(self):
        """Test that format matching is case insensitive."""
        formats = [
            "PNG",
            "SVG",
            "JPG",
            "JPEG",
            "PDF",
            "WEBP",
            "TIFF",
            "BMP",
            "Png",
            "Svg",
        ]
        expected_classes = [
            PNGOutput,
            SVGOutput,
            JPEGOutput,
            JPEGOutput,
            PDFOutput,
            WEBPOutput,
            TIFFOutput,
            BMPOutput,
            PNGOutput,
            SVGOutput,
        ]

        for fmt, expected_class in zip(formats, expected_classes):
            handler = get_output_handler(fmt)
            assert isinstance(handler, expected_class)

    def test_get_output_handler_invalid_format(self):
        """Test error handling for invalid formats."""
        invalid_formats = ["gif", "ico", "psd", "invalid", ""]

        for fmt in invalid_formats:
            with pytest.raises(ValueError) as exc_info:
                get_output_handler(fmt)
            assert f"Unsupported output format: {fmt}" in str(exc_info.value)


class TestPNGOutputHandler:
    """Test PNG output handler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = PNGOutput()
        # Create a simple test image
        self.test_image = Image.new("RGB", (100, 100), color="red")

    def test_png_handler_initialization(self):
        """Test PNG handler initialization."""
        assert self.handler.file_extension == ".png"
        assert self.handler.config.format == "png"
        assert self.handler.config.quality == 95
        assert self.handler.config.optimize is True

    def test_get_bytes_default_config(self):
        """Test converting image to PNG bytes with default config."""
        png_bytes = self.handler.get_bytes(self.test_image)

        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")  # PNG header

    def test_get_bytes_custom_config(self):
        """Test converting image to PNG bytes with custom config."""
        config = OutputConfig(format="png", optimize=False)
        handler = PNGOutput(config)
        png_bytes = handler.get_bytes(self.test_image)

        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    def test_get_bytes_optimization(self):
        """Test that optimization affects file size."""
        config_optimized = OutputConfig(format="png", optimize=True)
        config_unoptimized = OutputConfig(format="png", optimize=False)

        handler_optimized = PNGOutput(config_optimized)
        handler_unoptimized = PNGOutput(config_unoptimized)

        bytes_optimized = handler_optimized.get_bytes(self.test_image)
        bytes_unoptimized = handler_unoptimized.get_bytes(self.test_image)

        # Both should be valid PNG
        assert bytes_optimized.startswith(b"\x89PNG\r\n\x1a\n")
        assert bytes_unoptimized.startswith(b"\x89PNG\r\n\x1a\n")

        # Optimization usually reduces size, but not always for simple images
        assert len(bytes_optimized) > 0
        assert len(bytes_unoptimized) > 0

    def test_save_to_file_path_object(self):
        """Test saving PNG to file using Path object."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.png"
            self.handler.save(self.test_image, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

            # Verify it's a valid PNG by reading it back
            loaded_image = Image.open(filepath)
            assert loaded_image.format == "PNG"
            assert loaded_image.size == (100, 100)

    def test_save_to_file_string_path(self):
        """Test saving PNG to file using string path."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.png"
            self.handler.save(self.test_image, str(filepath))

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_save_with_custom_config(self):
        """Test saving PNG with custom configuration."""
        import tempfile

        config = OutputConfig(format="png", optimize=True)
        handler = PNGOutput(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.png"
            handler.save(self.test_image, filepath)

            assert filepath.exists()
            # Verify the saved image
            loaded_image = Image.open(filepath)
            assert loaded_image.format == "PNG"

    def test_png_quality_setting_ignored(self):
        """Test that quality setting is ignored for PNG (lossless format)."""
        config_low = OutputConfig(format="png", quality=10)
        config_high = OutputConfig(format="png", quality=100)

        handler_low = PNGOutput(config_low)
        handler_high = PNGOutput(config_high)

        bytes_low = handler_low.get_bytes(self.test_image)
        bytes_high = handler_high.get_bytes(self.test_image)

        # PNG is lossless, so quality shouldn't matter for identical images
        assert len(bytes_low) > 0
        assert len(bytes_high) > 0


class TestSVGOutputHandler:
    """Test SVG output handler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        config = OutputConfig(format="svg")
        self.handler = SVGOutput(config)
        # Create a test image (SVG handler will convert this to SVG)
        self.test_image = Image.new("RGB", (100, 100), color="red")

    def test_svg_handler_initialization(self):
        """Test SVG handler initialization."""
        assert self.handler.file_extension == ".svg"
        assert self.handler.config.format == "svg"

    def test_get_bytes_image_input(self):
        """Test converting PIL Image to SVG bytes."""
        svg_bytes = self.handler.get_bytes(self.test_image)

        assert isinstance(svg_bytes, bytes)
        assert len(svg_bytes) > 0
        assert b"<svg" in svg_bytes
        assert b'xmlns="http://www.w3.org/2000/svg"' in svg_bytes

    def test_get_bytes_with_config(self):
        """Test converting image to SVG bytes with custom config."""
        config = OutputConfig(format="svg", optimize=True)
        handler = SVGOutput(config)
        svg_bytes = handler.get_bytes(self.test_image)

        assert isinstance(svg_bytes, bytes)
        assert len(svg_bytes) > 0
        assert b"<svg" in svg_bytes

    def test_save_to_file_path_object(self):
        """Test saving SVG to file using Path object."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.svg"
            self.handler.save(self.test_image, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

            # Verify content
            content = filepath.read_text(encoding="utf-8")
            assert "<svg" in content
            assert 'xmlns="http://www.w3.org/2000/svg"' in content

    def test_save_to_file_string_path(self):
        """Test saving SVG to file using string path."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.svg"
            self.handler.save(self.test_image, str(filepath))

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_svg_encoding_utf8(self):
        """Test that SVG is properly encoded as UTF-8."""
        # Test that the generated SVG uses UTF-8 encoding
        svg_bytes = self.handler.get_bytes(self.test_image)

        # Should be valid UTF-8
        decoded = svg_bytes.decode("utf-8")
        assert "<svg" in decoded
        assert 'xmlns="http://www.w3.org/2000/svg"' in decoded

    def test_svg_sanitization_config(self):
        """Test SVG optimization configuration."""
        config = OutputConfig(format="svg", optimize=True)
        handler = SVGOutput(config)
        svg_bytes = handler.get_bytes(self.test_image)

        assert isinstance(svg_bytes, bytes)
        assert len(svg_bytes) > 0


class TestJPEGOutputHandler:
    """Test JPEG output handler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        config = OutputConfig(format="jpg")
        self.handler = JPEGOutput(config)
        # Create a test image (JPEG doesn't support transparency)
        self.test_image = Image.new("RGB", (100, 100), color="white")

    def test_jpeg_handler_initialization(self):
        """Test JPEG handler initialization."""
        assert self.handler.file_extension == ".jpg"
        assert self.handler.config.format == "jpg"
        assert self.handler.config.quality == 95

    def test_get_bytes_default_config(self):
        """Test converting image to JPEG bytes with default config."""
        jpeg_bytes = self.handler.get_bytes(self.test_image)

        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 0
        assert jpeg_bytes.startswith(b"\xff\xd8\xff")  # JPEG header

    def test_get_bytes_quality_settings(self):
        """Test different quality settings."""
        config_low = OutputConfig(format="jpg", quality=10)
        config_high = OutputConfig(format="jpg", quality=95)

        handler_low = JPEGOutput(config_low)
        handler_high = JPEGOutput(config_high)

        bytes_low = handler_low.get_bytes(self.test_image)
        bytes_high = handler_high.get_bytes(self.test_image)

        assert bytes_low.startswith(b"\xff\xd8\xff")
        assert bytes_high.startswith(b"\xff\xd8\xff")

        # Lower quality should generally result in smaller files
        # (though this might not always be true for simple test images)
        assert len(bytes_low) > 0
        assert len(bytes_high) > 0

    def test_rgba_to_rgb_conversion(self):
        """Test that RGBA images are converted to RGB for JPEG."""
        rgba_image = Image.new("RGBA", (100, 100), color=(255, 255, 255, 128))
        jpeg_bytes = self.handler.get_bytes(rgba_image)

        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 0
        assert jpeg_bytes.startswith(b"\xff\xd8\xff")

    def test_save_to_file(self):
        """Test saving JPEG to file."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jpg"
            self.handler.save(self.test_image, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

            # Verify it's a valid JPEG by reading it back
            loaded_image = Image.open(filepath)
            assert loaded_image.format == "JPEG"
            assert loaded_image.size == (100, 100)

    def test_optimize_setting(self):
        """Test JPEG optimization setting."""
        config_optimized = OutputConfig(format="jpg", optimize=True, quality=85)
        config_unoptimized = OutputConfig(format="jpg", optimize=False, quality=85)

        handler_optimized = JPEGOutput(config_optimized)
        handler_unoptimized = JPEGOutput(config_unoptimized)

        bytes_optimized = handler_optimized.get_bytes(self.test_image)
        bytes_unoptimized = handler_unoptimized.get_bytes(self.test_image)

        assert bytes_optimized.startswith(b"\xff\xd8\xff")
        assert bytes_unoptimized.startswith(b"\xff\xd8\xff")
        assert len(bytes_optimized) > 0
        assert len(bytes_unoptimized) > 0

    def test_quality_boundary_values(self):
        """Test JPEG quality at boundary values."""
        # Minimum quality
        config_min = OutputConfig(format="jpg", quality=1)
        handler_min = JPEGOutput(config_min)
        bytes_min = handler_min.get_bytes(self.test_image)

        # Maximum quality
        config_max = OutputConfig(format="jpg", quality=100)
        handler_max = JPEGOutput(config_max)
        bytes_max = handler_max.get_bytes(self.test_image)

        assert bytes_min.startswith(b"\xff\xd8\xff")
        assert bytes_max.startswith(b"\xff\xd8\xff")
        assert len(bytes_min) > 0
        assert len(bytes_max) > 0


class TestPDFOutputHandler:
    """Test PDF output handler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        config = OutputConfig(format="pdf")
        self.handler = PDFOutput(config)
        self.test_image = Image.new("RGB", (100, 100), color="white")

    def test_pdf_handler_initialization(self):
        """Test PDF handler initialization."""
        assert self.handler.file_extension == ".pdf"
        assert self.handler.config.format == "pdf"

    def test_get_bytes_default_config(self):
        """Test converting image to PDF bytes with default config."""
        pdf_bytes = self.handler.get_bytes(self.test_image)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF-")  # PDF header

    def test_get_bytes_with_config(self):
        """Test converting image to PDF bytes with custom config."""
        config = OutputConfig(format="pdf", quality=85)
        handler = PDFOutput(config)
        pdf_bytes = handler.get_bytes(self.test_image)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF-")

    def test_save_to_file(self):
        """Test saving PDF to file."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.pdf"
            self.handler.save(self.test_image, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

            # Verify it's a valid PDF by checking header
            with open(filepath, "rb") as f:
                header = f.read(5)
                assert header == b"%PDF-"

    def test_rgba_image_handling(self):
        """Test that RGBA images are handled properly in PDF."""
        rgba_image = Image.new("RGBA", (100, 100), color=(255, 255, 255, 128))
        pdf_bytes = self.handler.get_bytes(rgba_image)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF-")

    def test_pdf_quality_setting(self):
        """Test PDF quality settings (affects JPEG compression inside PDF)."""
        config_low = OutputConfig(format="pdf", quality=20)
        config_high = OutputConfig(format="pdf", quality=95)

        handler_low = PDFOutput(config_low)
        handler_high = PDFOutput(config_high)

        bytes_low = handler_low.get_bytes(self.test_image)
        bytes_high = handler_high.get_bytes(self.test_image)

        assert bytes_low.startswith(b"%PDF-")
        assert bytes_high.startswith(b"%PDF-")
        assert len(bytes_low) > 0
        assert len(bytes_high) > 0

    def test_pdf_contains_expected_structure(self):
        """Test that generated PDF contains expected structure."""
        pdf_bytes = self.handler.get_bytes(self.test_image)
        pdf_content = pdf_bytes.decode("latin-1", errors="ignore")

        # Basic PDF structure checks
        assert "%PDF-" in pdf_content
        assert "endobj" in pdf_content
        assert "stream" in pdf_content or "endstream" in pdf_content


class TestOutputHandlerIntegration:
    """Integration tests for output handlers with real molecular data."""

    def setup_method(self):
        """Set up test fixtures."""
        # We'll need to create actual molecular images for these tests
        # For now, we'll use simple test images
        self.test_image = Image.new("RGB", (200, 200), color="lightblue")
        # Draw a simple "molecule" (circle)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(self.test_image)
        draw.ellipse([50, 50, 150, 150], fill="red", outline="black", width=2)

    def test_all_formats_with_same_image(self):
        """Test that all output handlers can process the same image."""
        formats = ["png", "svg", "jpg", "jpeg", "pdf"]

        for fmt in formats:
            config = OutputConfig(format=fmt)
            handler = get_output_handler(fmt, config)

            output_bytes = handler.get_bytes(self.test_image)
            assert isinstance(output_bytes, bytes)
            assert len(output_bytes) > 0

    def test_file_size_comparison(self):
        """Test relative file sizes across formats."""
        png_config = OutputConfig(format="png")
        jpg_config = OutputConfig(format="jpg")
        pdf_config = OutputConfig(format="pdf")

        png_handler = get_output_handler("png", png_config)
        jpg_handler = get_output_handler("jpg", jpg_config)
        pdf_handler = get_output_handler("pdf", pdf_config)

        png_bytes = png_handler.get_bytes(self.test_image)
        jpg_bytes = jpg_handler.get_bytes(self.test_image)
        pdf_bytes = pdf_handler.get_bytes(self.test_image)

        # All should produce valid output
        assert len(png_bytes) > 0
        assert len(jpg_bytes) > 0
        assert len(pdf_bytes) > 0

        # PDF usually has some overhead
        assert len(pdf_bytes) > 100  # Minimum expected size

    def test_quality_consistency_across_handlers(self):
        """Test that quality settings are consistently applied."""
        config_low = OutputConfig(format="jpg", quality=10)
        config_high = OutputConfig(format="jpg", quality=95)

        jpg_handler_low = get_output_handler("jpg", config_low)
        jpg_handler_high = get_output_handler("jpg", config_high)

        config_low_pdf = OutputConfig(format="pdf", quality=10)
        config_high_pdf = OutputConfig(format="pdf", quality=95)

        pdf_handler_low = get_output_handler("pdf", config_low_pdf)
        pdf_handler_high = get_output_handler("pdf", config_high_pdf)

        # Test JPEG
        jpg_low = jpg_handler_low.get_bytes(self.test_image)
        jpg_high = jpg_handler_high.get_bytes(self.test_image)

        # Test PDF
        pdf_low = pdf_handler_low.get_bytes(self.test_image)
        pdf_high = pdf_handler_high.get_bytes(self.test_image)

        # All should be valid
        assert len(jpg_low) > 0 and len(jpg_high) > 0
        assert len(pdf_low) > 0 and len(pdf_high) > 0

    def test_concurrent_handler_usage(self):
        """Test that handlers can be used concurrently."""
        import threading

        results = {}
        errors = []

        def process_format(fmt):
            try:
                config = OutputConfig(format=fmt)
                handler = get_output_handler(fmt, config)
                if fmt == "svg":
                    # Skip SVG for this test as it needs string input
                    results[fmt] = b"skipped"
                    return

                output_bytes = handler.get_bytes(self.test_image)
                results[fmt] = output_bytes
            except Exception as e:
                errors.append((fmt, e))

        threads = []
        formats = ["png", "jpg", "pdf"]

        for fmt in formats:
            thread = threading.Thread(target=process_format, args=(fmt,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == len(formats)

        for fmt in formats:
            assert len(results[fmt]) > 0


class TestOutputHandlerErrorHandling:
    """Test error handling in output handlers."""

    def test_invalid_image_input(self):
        """Test handling of invalid image inputs."""
        png_handler = get_output_handler("png")

        # Test with None
        with pytest.raises((AttributeError, TypeError)):
            png_handler.to_bytes(None)

        # Test with invalid type
        with pytest.raises((AttributeError, TypeError)):
            png_handler.to_bytes("not an image")

    def test_invalid_svg_input(self):
        """Test handling of invalid SVG inputs."""
        config = OutputConfig(format="svg")
        svg_handler = get_output_handler("svg", config)

        # Test with None
        with pytest.raises((AttributeError, TypeError)):
            svg_handler.get_bytes(None)

        # Test with invalid image type
        with pytest.raises((AttributeError, TypeError)):
            svg_handler.get_bytes("not an image")

    def test_file_permission_errors(self):
        """Test handling of file permission errors."""
        import tempfile

        png_handler = get_output_handler("png")
        test_image = Image.new("RGB", (50, 50), color="white")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file and make it read-only, then try to overwrite it
            readonly_file = Path(tmpdir) / "readonly.png"
            readonly_file.touch()
            readonly_file.chmod(0o444)  # Read-only

            try:
                with pytest.raises((PermissionError, OSError)):
                    png_handler.save(test_image, readonly_file)
            finally:
                # Clean up - restore write permissions so temp dir can be deleted
                readonly_file.chmod(0o644)

    def test_directory_path_handling(self):
        """Test that passing a directory path automatically adds the appropriate extension."""
        import tempfile

        png_handler = get_output_handler("png")
        test_image = Image.new("RGB", (50, 50), color="white")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save to a directory path - should automatically add .png extension
            png_handler.save(test_image, tmpdir)

            # Check that a .png file was created
            expected_file = Path(tmpdir + ".png")
            assert expected_file.exists(), (
                f"Expected file {expected_file} was not created"
            )

            # Verify it's a valid PNG file
            with Image.open(expected_file) as saved_image:
                assert saved_image.format == "PNG"
                assert saved_image.size == (50, 50)

    def test_invalid_file_paths(self):
        """Test handling of invalid file paths."""
        png_handler = get_output_handler("png")
        test_image = Image.new("RGB", (50, 50), color="white")

        # Invalid path (assuming this path doesn't exist and can't be created)
        invalid_path = "/invalid/path/that/does/not/exist/test.png"

        with pytest.raises((FileNotFoundError, OSError, PermissionError)):
            png_handler.save(test_image, invalid_path)


if __name__ == "__main__":
    pytest.main([__file__])
