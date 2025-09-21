"""
Core functionality tests for output handlers.

Tests basic properties, initialization, and fundamental operations
shared across all output handlers.
"""

from pathlib import Path

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import get_output_handler


class TestOutputHandlerProperties:
    """Test properties common to all output handlers."""

    def test_file_extension_format(self, output_handler):
        """Test that file extension starts with dot and matches format."""
        # Arrange - handler is provided by fixture

        # Act
        extension = output_handler.file_extension

        # Assert
        assert isinstance(extension, str), "File extension must be a string"
        assert extension.startswith("."), "File extension must start with a dot"
        assert len(extension) > 1, "File extension must contain more than just the dot"

    def test_format_name_is_string(self, output_handler):
        """Test that format name is a non-empty string."""
        # Arrange - handler is provided by fixture

        # Act
        format_name = output_handler.format_name

        # Assert
        assert isinstance(format_name, str), "Format name must be a string"
        assert len(format_name) > 0, "Format name must not be empty"

    def test_raster_handler_has_required_properties(self, raster_format_name):
        """Test that raster handlers have all required properties."""
        # Arrange
        handler = get_output_handler(raster_format_name)

        # Act & Assert - test all required raster properties exist and have correct types
        assert hasattr(handler, "valid_extensions"), (
            "Raster handlers must have valid_extensions property"
        )
        assert hasattr(handler, "supports_alpha"), (
            "Raster handlers must have supports_alpha property"
        )
        assert hasattr(handler, "supports_quality"), (
            "Raster handlers must have supports_quality property"
        )
        assert hasattr(handler, "pil_format"), (
            "Raster handlers must have pil_format property"
        )

        # Verify property types
        assert isinstance(handler.valid_extensions, list), (
            "valid_extensions must be a list"
        )
        assert isinstance(handler.supports_alpha, bool), (
            "supports_alpha must be a boolean"
        )
        assert isinstance(handler.supports_quality, bool), (
            "supports_quality must be a boolean"
        )
        assert isinstance(handler.pil_format, str), "pil_format must be a string"

    def test_valid_extensions_contains_primary(self, raster_format_name):
        """Test that valid extensions contains the primary extension."""
        # Arrange
        handler = get_output_handler(raster_format_name)

        # Act
        extensions = handler.valid_extensions
        primary_extension = handler.file_extension

        # Assert
        assert primary_extension in extensions, (
            f"Primary extension {primary_extension} must be in valid_extensions {extensions}"
        )

    def test_vector_handler_properties(self, vector_format_name):
        """Test that vector handlers have appropriate properties."""
        # Arrange
        handler = get_output_handler(vector_format_name)

        # Act & Assert - vector handlers should have basic properties
        assert hasattr(handler, "file_extension"), (
            "Vector handlers must have file_extension property"
        )
        assert hasattr(handler, "format_name"), (
            "Vector handlers must have format_name property"
        )

        # Vector handlers may not have raster-specific properties
        file_extension = handler.file_extension
        format_name = handler.format_name

        assert isinstance(file_extension, str), "file_extension must be a string"
        assert isinstance(format_name, str), "format_name must be a string"


class TestOutputHandlerInitialization:
    """Test initialization patterns across all handlers."""

    def test_init_without_config(self, format_name):
        """Test initialization without config creates default config."""
        # Arrange - format_name provided by fixture

        # Act
        handler = get_output_handler(format_name)

        # Assert
        assert handler.config is not None, (
            "Handler must have a config when initialized without arguments"
        )
        assert isinstance(handler.config, OutputConfig), (
            "Default config must be an OutputConfig instance"
        )

    def test_init_with_config(self, format_name):
        """Test initialization with custom config retains the provided config."""
        # Arrange
        custom_config = OutputConfig(quality=80, optimize=True)

        # Act
        handler = get_output_handler(format_name, custom_config)

        # Assert
        assert handler.config is custom_config, (
            "Handler must retain the exact config instance provided"
        )
        assert handler.config.quality == 80, (
            "Custom config quality setting must be preserved"
        )
        assert handler.config.optimize is True, (
            "Custom config optimize setting must be preserved"
        )

    def test_init_with_none_config(self, format_name):
        """Test initialization with None config creates default config."""
        # Arrange
        none_config = None

        # Act
        handler = get_output_handler(format_name, none_config)

        # Assert
        assert handler.config is not None, (
            "Handler must create default config when None is provided"
        )
        assert isinstance(handler.config, OutputConfig), (
            "Default config must be an OutputConfig instance"
        )

    def test_config_independence_across_instances(self, format_name):
        """Test that different handler instances have independent configs."""
        # Arrange
        config1 = OutputConfig(quality=90)
        config2 = OutputConfig(quality=10)

        # Act
        handler1 = get_output_handler(format_name, config1)
        handler2 = get_output_handler(format_name, config2)
        handler3 = get_output_handler(format_name)  # Default config

        # Assert
        assert handler1.config is not handler2.config, (
            "Different handlers must have independent config instances"
        )
        assert handler1.config is not handler3.config, (
            "Default and custom configs must be independent"
        )
        assert handler2.config is not handler3.config, (
            "Different custom configs must be independent"
        )
        assert handler1.config.quality == 90, "Handler1 must retain its config settings"
        assert handler2.config.quality == 10, "Handler2 must retain its config settings"


class TestOutputHandlerBasicFunctionality:
    """Test basic functionality common to all handlers."""

    def test_get_bytes_returns_bytes(self, output_handler, test_image):
        """Test that get_bytes returns valid bytes object."""
        # Arrange - fixtures provide handler and test image

        # Act
        result = output_handler.get_bytes(test_image)

        # Assert
        assert isinstance(result, bytes), "get_bytes must return a bytes object"
        assert len(result) > 0, "get_bytes must return non-empty bytes"

    def test_save_creates_file(self, output_handler, test_image, temp_dir):
        """Test that save creates a file with correct extension."""
        # Arrange
        output_path = temp_dir / f"test{output_handler.file_extension}"

        # Act
        output_handler.save(test_image, output_path)

        # Assert
        assert output_path.exists(), f"File should be created at {output_path}"
        assert output_path.stat().st_size > 0, "Created file must not be empty"
        assert output_path.suffix == output_handler.file_extension, (
            "File must have correct extension"
        )

    def test_save_with_string_path(self, output_handler, test_image, temp_dir):
        """Test save accepts string paths in addition to Path objects."""
        # Arrange
        output_path_str = str(temp_dir / f"test{output_handler.file_extension}")

        # Act
        output_handler.save(test_image, output_path_str)

        # Assert
        output_path = Path(output_path_str)
        assert output_path.exists(), (
            f"File should be created at string path {output_path_str}"
        )
        assert output_path.stat().st_size > 0, (
            "File created from string path must not be empty"
        )

    def test_save_auto_extension(self, output_handler, test_image, temp_dir):
        """Test that save automatically adds extension when missing."""
        # Arrange
        base_name = "test_no_extension"
        output_path = temp_dir / base_name
        expected_path = temp_dir / f"{base_name}{output_handler.file_extension}"

        # Act
        output_handler.save(test_image, output_path)

        # Assert
        assert expected_path.exists(), (
            f"File should be created with auto-added extension at {expected_path}"
        )
        assert not output_path.exists(), (
            f"Original path without extension should not exist: {output_path}"
        )
        assert expected_path.stat().st_size > 0, "Auto-extension file must not be empty"

    def test_save_creates_directory(self, output_handler, test_image, temp_dir):
        """Test that save creates parent directories when they don't exist."""
        # Arrange
        nested_path = (
            temp_dir / "nested" / "dir" / f"test{output_handler.file_extension}"
        )
        parent_dir = nested_path.parent

        # Verify parent doesn't exist initially
        assert not parent_dir.exists(), (
            f"Parent directory should not exist initially: {parent_dir}"
        )

        # Act
        output_handler.save(test_image, nested_path)

        # Assert
        assert nested_path.exists(), (
            f"File should be created at nested path: {nested_path}"
        )
        assert parent_dir.is_dir(), (
            f"Parent directories should be created: {parent_dir}"
        )
        assert nested_path.stat().st_size > 0, (
            "File in nested directory must not be empty"
        )

    def test_get_bytes_and_save_consistency(self, output_handler, test_image, temp_dir):
        """Test that get_bytes and save produce consistent results."""
        # Arrange
        output_path = temp_dir / f"consistency_test{output_handler.file_extension}"

        # Act
        bytes_result = output_handler.get_bytes(test_image)
        output_handler.save(test_image, output_path)
        saved_bytes = output_path.read_bytes()

        # Assert
        assert len(bytes_result) > 0, "get_bytes must return non-empty data"
        assert len(saved_bytes) > 0, "Saved file must contain non-empty data"
        # Note: We don't assert equality because some formats (like PDF) may include
        # timestamps or other non-deterministic data that differs between calls
