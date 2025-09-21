"""
Integration tests for output handlers.

Tests inheritance hierarchy, cross-format integration, factory consistency,
and overall system integration across all output handlers.
"""

from unittest.mock import patch

import pytest

from molecular_string_renderer.outputs import get_output_handler
from molecular_string_renderer.outputs.base import (
    OutputHandler,
    RasterOutputHandler,
    VectorOutputHandler,
)

from .conftest import (
    ALL_FORMATS,
    RASTER_FORMATS,
    VECTOR_FORMATS,
    TestValidators,
)


class TestOutputHandlerInheritance:
    """Test inheritance hierarchy compliance across all handlers."""

    def test_all_handlers_inherit_from_base(self, output_handler):
        """Test that all handlers inherit from OutputHandler."""
        assert isinstance(output_handler, OutputHandler), (
            f"Handler {type(output_handler).__name__} must inherit from OutputHandler"
        )

    def test_raster_handlers_inherit_correctly(self, raster_output_handler):
        """Test that raster handlers inherit from RasterOutputHandler."""
        assert isinstance(raster_output_handler, RasterOutputHandler), (
            "Handler must be a RasterOutputHandler"
        )
        assert isinstance(raster_output_handler, OutputHandler), (
            "Handler must also be an OutputHandler"
        )

    def test_vector_handlers_inherit_correctly(self, vector_output_handler):
        """Test that vector handlers inherit from VectorOutputHandler."""
        assert isinstance(vector_output_handler, VectorOutputHandler), (
            "Handler must be a VectorOutputHandler"
        )
        assert isinstance(vector_output_handler, OutputHandler), (
            "Handler must also be an OutputHandler"
        )


class TestCrossFormatIntegration:
    """Integration tests across different handlers and formats."""

    def test_all_formats_save_successfully(self, temp_dir, test_image):
        """Test that the same image can be saved successfully in all supported formats."""
        created_files = {}

        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)
            output_path = (
                temp_dir / f"integration_test_{format_name}{handler.file_extension}"
            )

            handler.save(test_image, output_path)
            created_files[format_name] = output_path

        # Validate all files were created properly
        assert len(created_files) == len(ALL_FORMATS)

        for format_name, file_path in created_files.items():
            handler = get_output_handler(format_name)
            TestValidators.assert_file_created_properly(
                file_path, handler.file_extension, f"{format_name} file"
            )

    def test_all_formats_get_bytes_successfully(self, test_image):
        """Test that all formats can generate bytes output successfully."""
        results = {}

        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)
            result = handler.get_bytes(test_image)
            results[format_name] = result

        # Validate all formats produced valid output
        assert len(results) == len(ALL_FORMATS)

        for format_name, result in results.items():
            TestValidators.assert_image_output_valid(result, format_name)

    def test_format_consistency_across_methods(self, temp_dir, test_image):
        """Test that save() and get_bytes() produce consistent results."""
        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)

            # Get bytes output
            bytes_output = handler.get_bytes(test_image)

            # Save to file and read back
            file_path = (
                temp_dir / f"consistency_test_{format_name}{handler.file_extension}"
            )
            handler.save(test_image, file_path)

            with open(file_path, "rb") as f:
                file_output = f.read()

            # For most formats, these should be identical or very similar
            # Some formats (like PDF) may have timestamps that cause minor differences
            size_diff_ratio = abs(len(bytes_output) - len(file_output)) / max(
                len(bytes_output), len(file_output)
            )
            assert size_diff_ratio < 0.1, (
                f"Significant size difference for {format_name}: "
                f"get_bytes={len(bytes_output)}, save={len(file_output)}"
            )

    def test_handler_metadata_consistency(self):
        """Test that format properties are consistent and logical."""
        expected_raster_formats = set(RASTER_FORMATS.keys())
        expected_vector_formats = set(VECTOR_FORMATS.keys())

        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)

            # Basic property validation
            assert isinstance(handler.file_extension, str)
            assert handler.file_extension.startswith(".")
            assert len(handler.file_extension) > 1

            assert isinstance(handler.format_name, str)
            assert len(handler.format_name) > 0

            # Type-specific validation
            if format_name in expected_raster_formats:
                assert isinstance(handler, RasterOutputHandler)
                assert hasattr(handler, "supports_alpha")
                assert hasattr(handler, "supports_quality")
            elif format_name in expected_vector_formats:
                assert isinstance(handler, VectorOutputHandler)


class TestFactoryConsistency:
    """Test handler factory consistency and reliability."""

    def test_factory_produces_consistent_handlers(self):
        """Test that factory produces consistent handler types and properties."""
        for format_name in ALL_FORMATS:
            # Create multiple instances
            handler1 = get_output_handler(format_name)
            handler2 = get_output_handler(format_name)

            # Should produce same type
            assert type(handler1) is type(handler2)

            # Should have same properties
            assert handler1.file_extension == handler2.file_extension
            assert handler1.format_name == handler2.format_name

            # But should be independent instances
            assert handler1 is not handler2
            assert handler1.config is not handler2.config

    def test_factory_handles_case_variations(self):
        """Test that factory handles different case variations."""
        test_cases = [
            ("png", "PNG", "PnG"),
            ("svg", "SVG", "SvG"),
            ("jpeg", "JPEG", "JpEg"),
        ]

        for variations in test_cases:
            if variations[0] in ALL_FORMATS:  # Only test if format is supported
                handlers = [get_output_handler(variant) for variant in variations]

                # All variations should produce same handler type
                handler_types = [type(h) for h in handlers]
                assert len(set(handler_types)) == 1, (
                    f"Case variations should produce same handler type for {variations[0]}"
                )

    def test_format_coverage_completeness(self):
        """Test that all expected formats are available."""
        expected_formats = set(RASTER_FORMATS.keys()) | set(VECTOR_FORMATS.keys())
        available_formats = set(ALL_FORMATS.keys())

        # Should have all expected formats
        missing_formats = expected_formats - available_formats
        assert len(missing_formats) == 0, f"Missing expected formats: {missing_formats}"

        # Each format should be instantiable
        for format_name in available_formats:
            handler = get_output_handler(format_name)
            TestValidators.assert_handler_interface_complete(handler, format_name)


class TestErrorHandlingIntegration:
    """Test error handling patterns across output handlers."""

    def test_invalid_format_handling(self):
        """Test that factory handles invalid formats gracefully."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            get_output_handler("invalid_format")

    def test_svg_error_handling(self, temp_dir, test_image):
        """Test SVG error handling in save method."""
        from molecular_string_renderer.outputs import SVGOutput

        svg_output = SVGOutput()
        output_path = temp_dir / "error_test.svg"

        # Mock strategy to raise an exception
        with patch.object(
            svg_output._strategy, "generate_svg", side_effect=Exception("Test error")
        ):
            # Should handle error gracefully
            with pytest.raises(Exception):
                svg_output.save(test_image, output_path)

    def test_pdf_error_handling(self, test_image):
        """Test PDF error handling when generation fails."""
        from molecular_string_renderer.outputs import PDFOutput

        pdf_output = PDFOutput()

        # Mock PDF generation to raise an exception
        with patch.object(
            pdf_output, "_generate_pdf_bytes", side_effect=Exception("Test error")
        ):
            with pytest.raises(IOError, match="Failed to generate PDF bytes"):
                pdf_output.get_bytes(test_image)

    def test_raster_format_error_recovery(self, test_image):
        """Test that raster formats handle PIL errors gracefully."""
        from molecular_string_renderer.outputs import PNGOutput

        png_output = PNGOutput()

        # Mock PIL save to raise an exception
        with patch("PIL.Image.Image.save", side_effect=Exception("PIL error")):
            with pytest.raises(Exception):
                png_output.get_bytes(test_image)


class TestHandlerBehaviorConsistency:
    """Test consistent behavior patterns across handlers."""

    def test_all_handlers_respect_configuration(self, test_image):
        """Test that all handlers properly handle configuration objects."""
        from molecular_string_renderer.config import OutputConfig

        # Test with various configurations
        configs = [
            OutputConfig(),
            OutputConfig(quality=75),
            OutputConfig(optimize=True),
            OutputConfig(quality=90, optimize=True),
        ]

        for config in configs:
            for format_name in ALL_FORMATS:
                handler = get_output_handler(format_name, config)

                # Handler should accept the config
                assert handler.config is config

                # Should be able to generate output
                result = handler.get_bytes(test_image)
                TestValidators.assert_valid_bytes_output(
                    result, f"{format_name} with config"
                )

    def test_image_mode_handling_consistency(self, varied_image):
        """Test that all handlers handle different image modes consistently."""
        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)

            # Should be able to handle the image (may convert internally)
            result = handler.get_bytes(varied_image)
            TestValidators.assert_valid_bytes_output(
                result, f"{format_name} with {varied_image.mode}"
            )

    def test_dimension_handling_consistency(self, image_dimensions):
        """Test that all handlers handle various image dimensions consistently."""
        from .conftest import create_test_image_with_mode

        test_image_sized = create_test_image_with_mode("RGB", image_dimensions)

        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)

            # Should handle any reasonable dimensions
            result = handler.get_bytes(test_image_sized)
            TestValidators.assert_valid_bytes_output(
                result, f"{format_name} with dimensions {image_dimensions}"
            )
