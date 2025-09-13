"""
Comprehensive tests for all supported output formats.

Tests every output format that's mentioned in the codebase to ensure
they can actually be produced.
"""

import tempfile
from pathlib import Path

import pytest

from molecular_string_renderer import render_molecule
from molecular_string_renderer.core import get_supported_formats
from molecular_string_renderer.outputs import get_output_handler
from tests.conftest import SAMPLE_MOLECULES


class TestAllSupportedOutputFormats:
    """Test all output formats mentioned in the codebase."""

    def test_cli_supported_formats(self):
        """Test all formats supported by CLI can be rendered."""
        # These are the formats from cli.py choices
        cli_formats = ["png", "svg", "jpg", "jpeg"]
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in cli_formats:
                output_path = Path(temp_dir) / f"molecule.{fmt}"

                image = render_molecule(
                    molecular_string=smiles,
                    format_type="smiles",
                    output_format=fmt,
                    output_path=output_path,
                )

                assert image is not None, f"No image returned for {fmt}"
                assert output_path.exists(), f"Output file not created for {fmt}"
                assert output_path.stat().st_size > 0, f"Empty output file for {fmt}"

    def test_config_supported_formats(self):
        """Test all formats from OutputConfig validation."""
        # These are from config.py OutputConfig.validate_format()
        # Only test the ones that have implementations
        implemented_formats = ["png", "svg", "jpg", "jpeg"]
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in implemented_formats:
                output_path = Path(temp_dir) / f"molecule.{fmt}"

                image = render_molecule(
                    molecular_string=smiles,
                    format_type="smiles",
                    output_format=fmt,
                    output_path=output_path,
                )

                assert image is not None, f"No image returned for {fmt}"
                assert output_path.exists(), f"Output file not created for {fmt}"
                assert output_path.stat().st_size > 0, f"Empty output file for {fmt}"

        # Test that PDF is NOT implemented (should fail)
        with pytest.raises(ValueError, match="Unsupported output format"):
            get_output_handler("pdf")

    def test_get_supported_formats_output_formats(self):
        """Test all formats returned by get_supported_formats() can be rendered."""
        formats = get_supported_formats()
        output_formats = list(formats["output_formats"].keys())

        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in output_formats:
                output_path = Path(temp_dir) / f"molecule.{fmt}"

                image = render_molecule(
                    molecular_string=smiles,
                    format_type="smiles",
                    output_format=fmt,
                    output_path=output_path,
                )

                assert image is not None, f"No image returned for {fmt}"
                assert output_path.exists(), f"Output file not created for {fmt}"
                assert output_path.stat().st_size > 0, f"Empty output file for {fmt}"

    def test_jpg_vs_jpeg_equivalence(self):
        """Test that 'jpg' and 'jpeg' produce equivalent results."""
        smiles = SAMPLE_MOLECULES["smiles"]["benzene"]

        with tempfile.TemporaryDirectory() as temp_dir:
            jpg_path = Path(temp_dir) / "molecule.jpg"
            jpeg_path = Path(temp_dir) / "molecule.jpeg"

            # Render with jpg format
            jpg_image = render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="jpg",
                output_path=jpg_path,
            )

            # Render with jpeg format
            jpeg_image = render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="jpeg",
                output_path=jpeg_path,
            )

            # Both should succeed
            assert jpg_image is not None
            assert jpeg_image is not None
            assert jpg_path.exists()
            assert jpeg_path.exists()

            # Both should be the same size (same image dimensions)
            assert jpg_image.size == jpeg_image.size

            # File sizes should be similar (within 10% - accounting for compression differences)
            jpg_size = jpg_path.stat().st_size
            jpeg_size = jpeg_path.stat().st_size
            size_diff_ratio = abs(jpg_size - jpeg_size) / max(jpg_size, jpeg_size)
            assert size_diff_ratio < 0.1, (
                f"File sizes too different: {jpg_size} vs {jpeg_size}"
            )

    def test_format_file_extensions(self):
        """Test that output handlers have correct file extensions."""
        expected_extensions = {
            "png": ".png",
            "svg": ".svg",
            "jpg": ".jpg",
            "jpeg": ".jpg",  # JPEG handler uses .jpg extension
        }

        for fmt, expected_ext in expected_extensions.items():
            handler = get_output_handler(fmt)
            assert handler.file_extension == expected_ext, (
                f"Wrong extension for {fmt}: got {handler.file_extension}, expected {expected_ext}"
            )

    def test_format_auto_extension_correction(self):
        """Test that output handlers auto-correct file extensions."""
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test PNG with wrong extension
            png_path = Path(temp_dir) / "molecule.wrongext"
            render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="png",
                output_path=png_path,
            )
            # Should create file with .png extension
            expected_png = png_path.with_suffix(".png")
            assert expected_png.exists()

            # Test JPEG with .jpeg extension when using jpg format
            jpeg_path = Path(temp_dir) / "molecule.jpeg"
            render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="jpg",
                output_path=jpeg_path,
            )
            # Should accept .jpeg extension for jpg format
            assert jpeg_path.exists()

    def test_memory_only_rendering_all_formats(self):
        """Test in-memory rendering for all formats (no file output)."""
        formats = ["png", "svg", "jpg", "jpeg"]
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        for fmt in formats:
            image = render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format=fmt,
                auto_filename=False,  # No file output
            )

            assert image is not None, f"No image returned for in-memory {fmt}"
            assert hasattr(image, "size"), f"Invalid image object for {fmt}"
            assert image.size[0] > 0 and image.size[1] > 0, f"Zero-size image for {fmt}"

    def test_vector_svg_generation(self):
        """Test that SVG output is true vector SVG, not raster-embedded."""
        smiles = SAMPLE_MOLECULES["smiles"]["benzene"]

        with tempfile.TemporaryDirectory() as temp_dir:
            svg_path = Path(temp_dir) / "vector_test.svg"

            render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="svg",
                output_path=svg_path,
            )

            assert svg_path.exists(), "SVG file was not created"

            # Read SVG content
            svg_content = svg_path.read_text()

            # Verify it's true vector SVG
            assert "base64" not in svg_content, (
                "SVG should not contain embedded raster data"
            )
            assert "xmlns:rdkit" in svg_content, (
                "SVG should contain RDKit namespace for vector format"
            )
            assert "<path" in svg_content, "SVG should contain vector path elements"

            # Verify proper SVG structure
            assert svg_content.startswith("<?xml version="), (
                "SVG should start with XML declaration"
            )
            assert "width=" in svg_content, "SVG should specify width"
            assert "height=" in svg_content, "SVG should specify height"


class TestUnsupportedFormats:
    """Test handling of unsupported formats."""

    def test_unsupported_output_format_error(self):
        """Test that unsupported output formats raise appropriate errors."""
        unsupported_formats = ["bmp", "tiff", "webp", "gif"]
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        for fmt in unsupported_formats:
            with pytest.raises(Exception, match="Unsupported format"):
                render_molecule(
                    molecular_string=smiles,
                    format_type="smiles",
                    output_format=fmt,
                )

        # PDF is listed in config validation but not implemented in output handlers
        with pytest.raises(ValueError, match="Unsupported output format"):
            render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="pdf",
            )

    def test_output_handler_factory_errors(self):
        """Test that output handler factory rejects unsupported formats."""
        unsupported_formats = ["bmp", "tiff", "webp", "gif", "pdf", "invalid"]

        for fmt in unsupported_formats:
            with pytest.raises(ValueError, match="Unsupported output format"):
                get_output_handler(fmt)

    def test_case_insensitive_format_handling(self):
        """Test that format names are case-insensitive."""
        formats_to_test = [
            ("PNG", "png"),
            ("SVG", "svg"),
            ("JPG", "jpg"),
            ("JPEG", "jpeg"),
            ("Png", "png"),
            ("Svg", "svg"),
        ]

        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for upper_fmt, lower_fmt in formats_to_test:
                output_path = Path(temp_dir) / f"molecule_{upper_fmt}.{lower_fmt}"

                image = render_molecule(
                    molecular_string=smiles,
                    format_type="smiles",
                    output_format=upper_fmt,
                    output_path=output_path,
                )

                assert image is not None, f"No image returned for {upper_fmt}"
                assert output_path.exists(), f"Output file not created for {upper_fmt}"


class TestFormatQualityAndOptions:
    """Test format-specific quality and options."""

    def test_jpeg_quality_settings(self):
        """Test JPEG quality settings actually affect output."""
        smiles = SAMPLE_MOLECULES["smiles"]["benzene"]

        with tempfile.TemporaryDirectory() as temp_dir:
            high_quality_path = Path(temp_dir) / "high_quality.jpg"
            low_quality_path = Path(temp_dir) / "low_quality.jpg"

            # High quality
            from molecular_string_renderer.config import OutputConfig

            high_config = OutputConfig(format="jpg", quality=95)
            render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="jpg",
                output_path=high_quality_path,
                output_config=high_config,
            )

            # Low quality
            low_config = OutputConfig(format="jpg", quality=20)
            render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="jpg",
                output_path=low_quality_path,
                output_config=low_config,
            )

            # High quality should generally produce larger files
            high_size = high_quality_path.stat().st_size
            low_size = low_quality_path.stat().st_size

            assert high_size > low_size, (
                f"High quality ({high_size}) should be larger than low quality ({low_size})"
            )

    def test_png_optimization_settings(self):
        """Test PNG optimization settings."""
        smiles = SAMPLE_MOLECULES["smiles"]["benzene"]

        with tempfile.TemporaryDirectory() as temp_dir:
            optimized_path = Path(temp_dir) / "optimized.png"
            unoptimized_path = Path(temp_dir) / "unoptimized.png"

            # Optimized
            from molecular_string_renderer.config import OutputConfig

            opt_config = OutputConfig(format="png", optimize=True)
            render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="png",
                output_path=optimized_path,
                output_config=opt_config,
            )

            # Unoptimized
            unopt_config = OutputConfig(format="png", optimize=False)
            render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="png",
                output_path=unoptimized_path,
                output_config=unopt_config,
            )

            # Both should exist
            assert optimized_path.exists()
            assert unoptimized_path.exists()

            # Files should have reasonable sizes
            opt_size = optimized_path.stat().st_size
            unopt_size = unoptimized_path.stat().st_size

            assert opt_size > 0
            assert unopt_size > 0


if __name__ == "__main__":
    pytest.main([__file__])
