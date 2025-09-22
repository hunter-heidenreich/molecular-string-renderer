"""
Core functionality tests for output handlers.

Tests basic properties, initialization, fundamental operations,
and configuration handling shared across all output handlers.

This module focuses on testing the core contract that all output handlers
must fulfill, ensuring consistency across formats and proper error handling.
"""

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import get_output_handler


class TestOutputHandlerInterface:
    """Test that all handlers implement the required interface correctly."""

    def test_handler_interface_complete(self, output_handler, format_name):
        """Test that handler implements complete interface."""
        from .conftest import TestValidators

        TestValidators.assert_handler_interface_complete(output_handler, format_name)

    def test_file_extension_format(self, output_handler):
        """Test that file extension starts with dot and is valid."""
        extension = output_handler.file_extension
        assert isinstance(extension, str), "File extension must be a string"
        assert extension.startswith("."), "File extension must start with a dot"
        assert len(extension) > 1, "File extension must contain more than just the dot"

    def test_raster_handler_properties(self, raster_format_name):
        """Test that raster handlers have all required properties."""
        from .conftest import assert_raster_handler_properties

        handler = get_output_handler(raster_format_name)
        assert_raster_handler_properties(handler, raster_format_name)

    def test_valid_extensions_include_primary(self, raster_format_name):
        """Test that valid extensions include the primary extension."""
        handler = get_output_handler(raster_format_name)
        assert handler.file_extension in handler.valid_extensions, (
            f"Primary extension {handler.file_extension} must be in valid_extensions"
        )


class TestOutputHandlerInitialization:
    """Test initialization patterns and config handling."""

    @pytest.mark.parametrize(
        "config_input,expected_type",
        [
            (None, OutputConfig),
            (OutputConfig(), OutputConfig),
            (OutputConfig(quality=80), OutputConfig),
        ],
    )
    def test_config_initialization(self, format_name, config_input, expected_type):
        """Test various config initialization scenarios."""
        handler = get_output_handler(format_name, config_input)

        assert handler.config is not None, "Handler must have a config"
        assert isinstance(handler.config, expected_type), (
            "Config must be OutputConfig instance"
        )

        if config_input is not None and hasattr(config_input, "quality"):
            assert handler.config.quality == config_input.quality, (
                "Custom config must be preserved"
            )

    def test_config_independence(self, format_name):
        """Test that handler instances have independent configs."""
        config1 = OutputConfig(quality=90)
        config2 = OutputConfig(quality=10)

        handler1 = get_output_handler(format_name, config1)
        handler2 = get_output_handler(format_name, config2)
        handler3 = get_output_handler(format_name)

        # Configs should be independent
        assert handler1.config is not handler2.config, (
            "Different handlers must have independent configs"
        )
        assert handler1.config is not handler3.config, (
            "Default and custom configs must be independent"
        )
        assert handler2.config is not handler3.config, "All configs must be independent"

        # Values should be preserved
        assert handler1.config.quality == 90, "Handler1 must retain its config"
        assert handler2.config.quality == 10, "Handler2 must retain its config"


class TestOutputHandlerBasicOperations:
    """Test fundamental save and get_bytes operations."""

    def test_get_bytes_basic_operation(self, output_handler, test_image):
        """Test that get_bytes returns valid bytes."""
        from .conftest import TestValidators

        result = output_handler.get_bytes(test_image)
        TestValidators.assert_valid_bytes_output(result, f"{output_handler.format_name} get_bytes")

    def test_save_basic_operation(self, output_handler, test_image, temp_dir):
        """Test that save creates valid files."""
        from .conftest import TestValidators

        file_path = temp_dir / f"test{output_handler.file_extension}"
        output_handler.save(test_image, file_path)
        TestValidators.assert_file_created_properly(file_path, output_handler.file_extension, "test output")

    @pytest.mark.parametrize("path_type", ["string", "pathlib"])
    def test_save_accepts_both_path_types(
        self, output_handler, test_image, temp_dir, path_type
    ):
        """Test save accepts both string and Path objects."""
        from .conftest import assert_file_created_properly

        base_path = temp_dir / f"test_{path_type}{output_handler.file_extension}"
        path_input = str(base_path) if path_type == "string" else base_path

        output_handler.save(test_image, path_input)
        assert_file_created_properly(base_path, output_handler.file_extension)

    def test_save_auto_extension(self, output_handler, test_image, temp_dir):
        """Test that save automatically adds extension when missing."""
        from .conftest import assert_file_created_properly

        base_name = "test_no_extension"
        output_path = temp_dir / base_name
        expected_path = temp_dir / f"{base_name}{output_handler.file_extension}"

        output_handler.save(test_image, output_path)
        assert_file_created_properly(expected_path, output_handler.file_extension)
        assert not output_path.exists(), (
            "Original path without extension should not exist"
        )

    def test_save_creates_directories(self, output_handler, test_image, temp_dir):
        """Test that save creates parent directories."""
        from .conftest import assert_file_created_properly

        nested_path = (
            temp_dir / "nested" / "dirs" / f"test{output_handler.file_extension}"
        )
        assert not nested_path.parent.exists(), "Parent should not exist initially"

        output_handler.save(test_image, nested_path)
        assert_file_created_properly(nested_path, output_handler.file_extension)
        assert nested_path.parent.is_dir(), "Parent directories should be created"


class TestOutputHandlerImageModeSupport:
    """Test basic image mode support - detailed mode testing is in test_image_handling.py."""

    def test_transparency_handling(self, output_handler, rgba_image):
        """Test handling of images with transparency."""
        from .conftest import assert_valid_bytes_output

        result = output_handler.get_bytes(rgba_image)
        assert_valid_bytes_output(
            result, f"{output_handler.format_name} with transparency"
        )


class TestOutputHandlerErrorHandling:
    """Test comprehensive error handling for various invalid inputs and edge cases."""

    @pytest.mark.parametrize("invalid_input", [None, "not an image", 123, []])
    def test_invalid_image_inputs(self, output_handler, invalid_input):
        """Test handling of invalid image inputs."""
        with pytest.raises((TypeError, AttributeError, OSError, IOError)):
            output_handler.get_bytes(invalid_input)

    def test_zero_size_image(self, output_handler):
        """Test handling of zero-size images."""
        with pytest.raises((ValueError, OSError, SystemError, MemoryError)):
            zero_image = Image.new("RGB", (0, 0))
            output_handler.get_bytes(zero_image)

    def test_corrupted_image_handling(self, output_handler):
        """Test handling of corrupted image data."""
        corrupted_image = Image.new("RGB", (10, 10))
        corrupted_image.close()  # Simulate corruption

        try:
            output_handler.get_bytes(corrupted_image)
            # Some handlers may still work with closed images
        except (ValueError, OSError):
            pass  # Expected for most formats

    def test_save_invalid_path_raises_error(self, output_handler, test_image):
        """Test that invalid save paths raise appropriate errors."""
        invalid_path = "/root/nonexistent/test.png"

        with pytest.raises((IOError, OSError, PermissionError)) as exc_info:
            output_handler.save(test_image, invalid_path)

        error_message = str(exc_info.value).lower()
        assert any(
            keyword in error_message
            for keyword in [
                "permission",
                "access",
                "denied",
                "not found",
                "no such",
                "read-only",
                "file system",
            ]
        ), f"Error message should indicate the nature of the problem: {exc_info.value}"

    def test_save_with_none_image_raises_error(self, output_handler, temp_dir):
        """Test that passing None as image raises appropriate error."""
        output_path = temp_dir / f"test{output_handler.file_extension}"

        with pytest.raises((IOError, OSError)) as exc_info:
            output_handler.save(None, output_path)

        error_message = str(exc_info.value).lower()
        assert (
            "failed to save" in error_message
            and output_handler.format_name.lower() in error_message
        ), f"Error should indicate save failure: {exc_info.value}"

    def test_save_with_invalid_image_object_raises_error(
        self, output_handler, temp_dir
    ):
        """Test that passing invalid image object raises appropriate error."""
        output_path = temp_dir / f"test{output_handler.file_extension}"
        invalid_image = "not an image"

        with pytest.raises((IOError, OSError)) as exc_info:
            output_handler.save(invalid_image, output_path)

        error_message = str(exc_info.value).lower()
        assert (
            "failed to save" in error_message
            and output_handler.format_name.lower() in error_message
        ), f"Error should indicate save failure: {exc_info.value}"

    def test_get_bytes_with_none_image_raises_error(self, output_handler):
        """Test that get_bytes with None image raises appropriate error."""
        with pytest.raises((IOError, OSError, AttributeError, TypeError)):
            output_handler.get_bytes(None)

    def test_save_readonly_directory_error(self, output_handler, test_image, temp_dir):
        """Test saving to a read-only directory raises appropriate error."""
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()

        try:
            readonly_dir.chmod(0o555)  # Make directory read-only
            output_path = readonly_dir / f"test{output_handler.file_extension}"

            with pytest.raises((IOError, OSError, PermissionError)) as exc_info:
                output_handler.save(test_image, output_path)

            error_message = str(exc_info.value).lower()
            assert any(
                keyword in error_message
                for keyword in ["permission", "denied", "access"]
            ), f"Error should indicate permission problem: {exc_info.value}"

        finally:
            try:
                readonly_dir.chmod(0o755)  # Restore permissions for cleanup
            except (OSError, PermissionError):
                pass

    def test_error_messages_are_informative(self, output_handler, test_image):
        """Test that error messages provide useful debugging information."""
        deeply_nested_invalid_path = (
            "/root/very/deeply/nested/nonexistent/path/test.png"
        )

        with pytest.raises((IOError, OSError, PermissionError)) as exc_info:
            output_handler.save(test_image, deeply_nested_invalid_path)

        error_message = str(exc_info.value)
        assert len(error_message) > 10, (
            "Error message should be informative, not just a generic message"
        )


class TestOutputHandlerConfiguration:
    """Test configuration handling and option support."""

    @pytest.mark.parametrize("quality", [1, 50, 95, 100])
    def test_quality_settings(self, format_name, test_image, quality):
        """Test that quality settings are handled appropriately."""
        from .conftest import assert_config_preserved, assert_valid_bytes_output

        config = OutputConfig(quality=quality)
        handler = get_output_handler(format_name, config)

        result = handler.get_bytes(test_image)
        assert_valid_bytes_output(result, f"{format_name} quality {quality}")
        assert_config_preserved(handler, config, f"Quality {quality}")

    @pytest.mark.parametrize("optimize", [True, False])
    def test_optimization_settings(self, format_name, test_image, optimize):
        """Test that optimization settings are handled appropriately."""
        from .conftest import assert_config_preserved, assert_valid_bytes_output

        config = OutputConfig(optimize=optimize)
        handler = get_output_handler(format_name, config)

        result = handler.get_bytes(test_image)
        assert_valid_bytes_output(result, f"{format_name} optimize {optimize}")
        assert_config_preserved(handler, config, f"Optimize {optimize}")

    def test_complex_config_combinations(self, format_name, test_image):
        """Test complex configuration combinations."""
        from .conftest import assert_config_preserved, assert_valid_bytes_output

        complex_config = OutputConfig(
            quality=75, optimize=True, svg_use_vector=False, dpi=200
        )
        handler = get_output_handler(format_name, complex_config)

        result = handler.get_bytes(test_image)
        assert_valid_bytes_output(result, f"{format_name} complex config")
        assert_config_preserved(handler, complex_config, "Complex config")

    def test_config_isolation_between_handlers(self, format_name):
        """Test that config changes don't affect other handlers."""
        config1 = OutputConfig(quality=90)
        config2 = OutputConfig(quality=10)

        handler1 = get_output_handler(format_name, config1)
        handler2 = get_output_handler(format_name, config2)

        # Modify config1 after handler creation
        config1.quality = 50

        assert handler1.config.quality == 50, "Handler1 should see config changes"
        assert handler2.config.quality == 10, "Handler2 should not be affected"


class TestOutputHandlerPerformance:
    """Test performance characteristics and resource management."""

    def test_large_image_handling(self, output_handler):
        """Test handling of large images."""
        from .conftest import LARGE_IMAGE_DIMENSION, assert_valid_bytes_output

        large_image = Image.new(
            "RGB", (LARGE_IMAGE_DIMENSION, LARGE_IMAGE_DIMENSION), "blue"
        )

        try:
            result = output_handler.get_bytes(large_image)
            assert_valid_bytes_output(
                result, f"Large image {output_handler.format_name}"
            )
        finally:
            large_image.close()

    def test_multiple_operations_memory_stability(self, output_handler, test_image):
        """Test that multiple operations don't cause memory issues."""
        from .conftest import MEMORY_TEST_ITERATIONS

        for i in range(MEMORY_TEST_ITERATIONS):
            result = output_handler.get_bytes(test_image)
            assert len(result) > 0, f"Operation {i + 1} must produce valid output"

    def test_output_size_reasonableness(self, output_handler, small_image):
        """Test that output sizes are reasonable for small images."""
        from .conftest import MIN_FILE_SIZE, assert_valid_bytes_output

        result = output_handler.get_bytes(small_image)
        assert_valid_bytes_output(result, f"Small image {output_handler.format_name}")
        assert len(result) >= MIN_FILE_SIZE, f"Output too small: {len(result)} bytes"


class TestOutputHandlerConsistency:
    """Test consistency between different handler operations."""

    def test_get_bytes_save_output_similarity(
        self, output_handler, test_image, temp_dir
    ):
        """Test that get_bytes and save produce similar results."""
        # Get bytes result
        bytes_result = output_handler.get_bytes(test_image)

        # Save to file
        output_path = temp_dir / f"consistency_test{output_handler.file_extension}"
        output_handler.save(test_image, output_path)
        saved_bytes = output_path.read_bytes()

        # Both should produce non-empty data
        assert len(bytes_result) > 0, "get_bytes must return non-empty data"
        assert len(saved_bytes) > 0, "Saved file must contain non-empty data"

        # For most formats, they should be identical or very similar
        # PDF may have timestamps that cause minor differences
        if output_handler.format_name.lower() != "pdf":
            size_difference = abs(len(bytes_result) - len(saved_bytes))
            max_allowed_difference = max(
                100, len(bytes_result) * 0.01
            )  # 1% or 100 bytes
            assert size_difference <= max_allowed_difference, (
                f"Size difference too large: {size_difference} bytes"
            )

    def test_repeated_operations_consistency(self, output_handler, test_image):
        """Test that repeated operations on same image produce consistent results."""
        results = [output_handler.get_bytes(test_image) for _ in range(3)]

        # All results should be non-empty
        for i, result in enumerate(results):
            assert len(result) > 0, f"Result {i + 1} must be non-empty"

        # For deterministic formats, results should be identical
        # PDF may vary due to timestamps
        if output_handler.format_name.lower() != "pdf":
            first_result = results[0]
            for i, result in enumerate(results[1:], 2):
                assert result == first_result, f"Result {i} differs from first result"
