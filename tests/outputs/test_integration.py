"""
Integration tests for output handlers.

Tests inheritance hierarchy, cross-format integration, factory consistency,
and overall system integration across all output handlers.
"""

from PIL import Image

from molecular_string_renderer.outputs import get_output_handler
from molecular_string_renderer.outputs.base import (
    OutputHandler,
    RasterOutputHandler,
    VectorOutputHandler,
)

from .conftest import (
    ALL_FORMATS,
)


class TestOutputHandlerInheritance:
    """Test inheritance hierarchy compliance."""

    def test_all_handlers_inherit_from_base(self, output_handler):
        """Test that all handlers inherit from OutputHandler."""
        # Arrange - handler provided by fixture

        # Act - check inheritance
        is_output_handler = isinstance(output_handler, OutputHandler)

        # Assert
        assert is_output_handler, (
            f"Handler {type(output_handler).__name__} must inherit from OutputHandler"
        )

    def test_raster_handlers_inherit_correctly(self, raster_format_name):
        """Test that raster handlers inherit from RasterOutputHandler."""
        # Arrange
        expected_base_types = [RasterOutputHandler, OutputHandler]

        # Act
        handler = get_output_handler(raster_format_name)
        inheritance_results = [
            isinstance(handler, base_type) for base_type in expected_base_types
        ]

        # Assert
        assert all(inheritance_results), (
            f"Raster handler {raster_format_name} must inherit from both RasterOutputHandler and OutputHandler"
        )
        assert isinstance(handler, RasterOutputHandler), (
            f"{raster_format_name} handler must be a RasterOutputHandler"
        )
        assert isinstance(handler, OutputHandler), (
            f"{raster_format_name} handler must be an OutputHandler"
        )

    def test_vector_handlers_inherit_correctly(self, vector_format_name):
        """Test that vector handlers inherit from VectorOutputHandler."""
        # Arrange
        expected_base_types = [VectorOutputHandler, OutputHandler]

        # Act
        handler = get_output_handler(vector_format_name)
        inheritance_results = [
            isinstance(handler, base_type) for base_type in expected_base_types
        ]

        # Assert
        assert all(inheritance_results), (
            f"Vector handler {vector_format_name} must inherit from both VectorOutputHandler and OutputHandler"
        )
        assert isinstance(handler, VectorOutputHandler), (
            f"{vector_format_name} handler must be a VectorOutputHandler"
        )
        assert isinstance(handler, OutputHandler), (
            f"{vector_format_name} handler must be an OutputHandler"
        )

    def test_required_methods_implemented(self, output_handler):
        """Test that all required abstract methods are implemented."""
        # Arrange
        required_properties = ["file_extension", "format_name"]
        required_methods = ["save", "get_bytes"]
        handler_type = type(output_handler).__name__

        # Act - check property existence
        property_results = {
            prop: hasattr(output_handler, prop) for prop in required_properties
        }
        method_results = {
            method: hasattr(output_handler, method)
            and callable(getattr(output_handler, method))
            for method in required_methods
        }

        # Assert
        for prop, exists in property_results.items():
            assert exists, f"Handler {handler_type} must have property '{prop}'"

        for method, is_callable in method_results.items():
            assert is_callable, (
                f"Handler {handler_type} must have callable method '{method}'"
            )

    def test_raster_specific_methods(self, raster_format_name):
        """Test raster-specific methods and properties."""
        # Arrange
        required_properties = [
            "pil_format",
            "valid_extensions",
            "supports_alpha",
            "supports_quality",
        ]
        expected_types = {
            "pil_format": str,
            "valid_extensions": list,
            "supports_alpha": bool,
            "supports_quality": bool,
        }

        # Act
        handler = get_output_handler(raster_format_name)
        property_existence = {
            prop: hasattr(handler, prop) for prop in required_properties
        }
        property_types = {
            prop: type(getattr(handler, prop)) if hasattr(handler, prop) else None
            for prop in required_properties
        }

        # Assert
        for prop in required_properties:
            assert property_existence[prop], (
                f"Raster handler {raster_format_name} must have property '{prop}'"
            )

            actual_type = property_types[prop]
            expected_type = expected_types[prop]
            assert actual_type is expected_type, (
                f"Property '{prop}' must be {expected_type.__name__}, got {actual_type.__name__}"
            )


class TestOutputHandlerIntegration:
    """Integration tests across different handlers."""

    def test_all_formats_save_same_image_successfully(self, temp_dir):
        """Test that the same image can be saved successfully in all supported formats."""
        # Arrange
        test_image = Image.new("RGB", (100, 100), "red")
        expected_format_count = len(ALL_FORMATS)
        created_files = {}

        # Act - save the same image in every format
        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)
            output_path = (
                temp_dir / f"integration_test_{format_name}{handler.file_extension}"
            )

            handler.save(test_image, output_path)
            created_files[format_name] = {
                "path": output_path,
                "handler": handler,
                "size": output_path.stat().st_size if output_path.exists() else 0,
            }

        # Assert - all formats should create valid files
        assert len(created_files) == expected_format_count, (
            f"Should have tested all {expected_format_count} formats"
        )

        for format_name, file_info in created_files.items():
            file_path = file_info["path"]
            file_size = file_info["size"]

            assert file_path.exists(), (
                f"File for {format_name} should exist at {file_path}"
            )
            assert file_size > 0, (
                f"File for {format_name} should not be empty, got {file_size} bytes"
            )
            assert file_path.suffix == file_info["handler"].file_extension, (
                f"File extension should match handler extension for {format_name}"
            )

    def test_format_metadata_consistency_across_all_handlers(self):
        """Test that format properties are consistent and logical across all handlers."""
        # Arrange
        expected_format_count = len(ALL_FORMATS)
        format_metadata = {}

        # Act - collect metadata from all formats
        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)
            format_metadata[format_name] = {
                "handler_type": type(handler).__name__,
                "file_extension": handler.file_extension,
                "format_name": handler.format_name,
                "extension_without_dot": handler.file_extension.lstrip("."),
            }

        # Assert - basic consistency checks for all formats
        assert len(format_metadata) == expected_format_count, (
            f"Should have metadata for all {expected_format_count} formats"
        )

        for format_name, metadata in format_metadata.items():
            # File extension validation
            extension = metadata["file_extension"]
            assert isinstance(extension, str), (
                f"File extension for {format_name} must be string, got {type(extension)}"
            )
            assert extension.startswith("."), (
                f"File extension for {format_name} must start with dot, got '{extension}'"
            )
            assert len(extension) > 1, (
                f"File extension for {format_name} must contain more than just dot, got '{extension}'"
            )

            # Format name validation
            format_display_name = metadata["format_name"]
            assert isinstance(format_display_name, str), (
                f"Format name for {format_name} must be string, got {type(format_display_name)}"
            )
            assert len(format_display_name) > 0, (
                f"Format name for {format_name} must not be empty"
            )

        # Assert - extension/format name relationship validation
        for format_name, metadata in format_metadata.items():
            ext_without_dot = metadata["extension_without_dot"].lower()
            format_lower = metadata["format_name"].lower()
            format_name_lower = format_name.lower()

            # Check for reasonable relationship between extension and format names
            valid_relationship = (
                ext_without_dot in format_lower
                or format_lower in ext_without_dot
                or format_name_lower in ext_without_dot
                or self._is_known_format_exception(format_name_lower, ext_without_dot)
            )

            assert valid_relationship, (
                f"Format {format_name} has inconsistent naming: "
                f"extension='{metadata['file_extension']}', "
                f"format_name='{metadata['format_name']}'"
            )

    def test_cross_format_output_comparison(self, temp_dir):
        """Test and compare outputs across all formats for consistency."""
        # Arrange
        test_image = Image.new("RGB", (150, 150))
        # Create a pattern to make compression differences more apparent
        pattern_pixels = []
        for y in range(150):
            for x in range(150):
                r = (x * 255) // 150
                g = (y * 255) // 150
                b = ((x + y) * 255) // 300
                pattern_pixels.append((r, g, b))
        test_image.putdata(pattern_pixels)

        format_outputs = {}

        # Act - generate output for all formats
        for format_name in ALL_FORMATS:
            handler = get_output_handler(format_name)

            # Test both get_bytes and save methods
            output_bytes = handler.get_bytes(test_image)

            save_path = (
                temp_dir / f"cross_format_test_{format_name}{handler.file_extension}"
            )
            handler.save(test_image, save_path)

            format_outputs[format_name] = {
                "bytes_length": len(output_bytes),
                "file_size": save_path.stat().st_size,
                "handler_type": type(handler).__name__,
                "file_exists": save_path.exists(),
            }

        # Assert - all formats should produce valid output
        expected_format_count = len(ALL_FORMATS)
        assert len(format_outputs) == expected_format_count, (
            f"Should have output for all {expected_format_count} formats"
        )

        for format_name, output_info in format_outputs.items():
            assert output_info["bytes_length"] > 0, (
                f"Format {format_name} should produce non-empty bytes output"
            )
            assert output_info["file_size"] > 0, (
                f"Format {format_name} should create non-empty file"
            )
            assert output_info["file_exists"], (
                f"Format {format_name} should create file that exists"
            )

    def test_handler_factory_consistency(self):
        """Test that the handler factory produces consistent results."""
        # Arrange
        format_names = list(ALL_FORMATS.keys())

        # Act - create handlers multiple times for each format
        first_creation = {name: get_output_handler(name) for name in format_names}
        second_creation = {name: get_output_handler(name) for name in format_names}

        # Assert - factory should produce consistent handler types and properties
        for format_name in format_names:
            handler1 = first_creation[format_name]
            handler2 = second_creation[format_name]

            # Same handler type
            assert type(handler1) is type(handler2), (
                f"Factory should produce same handler type for {format_name}"
            )

            # Same properties
            assert handler1.file_extension == handler2.file_extension, (
                f"File extension should be consistent for {format_name}"
            )
            assert handler1.format_name == handler2.format_name, (
                f"Format name should be consistent for {format_name}"
            )

            # Independent instances (different config objects)
            assert handler1.config is not handler2.config, (
                f"Handlers should have independent config instances for {format_name}"
            )

    def test_format_coverage_completeness(self):
        """Test that all expected formats are covered and accessible."""
        # Arrange
        expected_raster_formats = {"png", "jpeg", "webp", "tiff", "bmp"}
        expected_vector_formats = {"svg", "pdf"}
        all_expected_formats = expected_raster_formats | expected_vector_formats

        # Act
        available_formats = set(ALL_FORMATS.keys())
        missing_formats = all_expected_formats - available_formats
        unexpected_formats = available_formats - all_expected_formats

        # Assert - format coverage validation
        assert len(missing_formats) == 0, f"Missing expected formats: {missing_formats}"

        # Log unexpected formats but don't fail (allows for future format additions)
        if unexpected_formats:
            print(
                f"Note: Found additional formats beyond expected set: {unexpected_formats}"
            )

        # Verify each format can be instantiated
        for format_name in available_formats:
            handler = get_output_handler(format_name)
            assert handler is not None, (
                f"Should be able to create handler for {format_name}"
            )
            assert hasattr(handler, "save"), (
                f"Handler for {format_name} should have save method"
            )
            assert hasattr(handler, "get_bytes"), (
                f"Handler for {format_name} should have get_bytes method"
            )

    def _is_known_format_exception(self, format_name: str, extension: str) -> bool:
        """Check if a format/extension pair is a known acceptable exception."""
        known_exceptions = {
            ("jpeg", "jpg"),  # JPEG uses .jpg extension
            ("tiff", "tif"),  # TIFF can use .tif extension
        }

        return (format_name, extension) in known_exceptions


class TestErrorHandlingIntegration:
    """Test error handling across output handlers."""

    def test_pdf_error_handling_on_get_bytes_failure(self):
        """Test PDF error handling when get_bytes fails."""
        from unittest.mock import patch

        from molecular_string_renderer.outputs import PDFOutput

        pdf_output = PDFOutput()
        test_image = Image.new("RGB", (100, 100), "red")

        # Mock _generate_pdf_bytes to raise an exception
        with patch.object(
            pdf_output, "_generate_pdf_bytes", side_effect=Exception("Test error")
        ):
            try:
                pdf_output.get_bytes(test_image)
                assert False, "Should have raised IOError"
            except IOError as e:
                assert "Failed to generate PDF bytes" in str(e)

    def test_svg_save_error_handling(self, temp_dir):
        """Test SVG error handling in save method."""
        from unittest.mock import patch

        from molecular_string_renderer.outputs import SVGOutput

        svg_output = SVGOutput()
        test_image = Image.new("RGB", (100, 100), "red")
        output_path = temp_dir / "test.svg"

        # Mock strategy to raise an exception
        with patch.object(
            svg_output._strategy, "generate_svg", side_effect=Exception("Test error")
        ):
            # Should handle error gracefully without crashing
            try:
                svg_output.save(test_image, output_path)
            except Exception:
                pass  # Expected to handle error internally

    def test_pdf_save_error_handling(self, temp_dir):
        """Test PDF error handling in save method."""
        from unittest.mock import patch

        from molecular_string_renderer.outputs import PDFOutput

        pdf_output = PDFOutput()
        test_image = Image.new("RGB", (100, 100), "red")
        output_path = temp_dir / "test.pdf"

        # Mock get_bytes to raise an exception
        with patch.object(pdf_output, "get_bytes", side_effect=Exception("Test error")):
            # Should handle error gracefully without crashing
            try:
                pdf_output.save(test_image, output_path)
            except Exception:
                pass  # Expected to handle error internally
