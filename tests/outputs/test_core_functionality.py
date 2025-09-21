"""
Core functionality tests for output handlers.

Tests basic properties, initialization, fundamental operations,
and configuration handling shared across all output handlers.
"""

from pathlib import Path

from PIL import Image

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


# =============================================================================
# Configuration and Quality Tests
# =============================================================================


class TestOutputHandlerConfigurationHandling:
    """Test configuration handling across handlers."""

    def test_quality_setting_respected(self, format_name):
        """Test that quality settings are handled appropriately for all formats."""
        # Arrange
        quality_config = OutputConfig(quality=50)
        handler = get_output_handler(format_name, quality_config)
        test_image = Image.new("RGB", (100, 100), "red")

        # Act
        result = handler.get_bytes(test_image)

        # Assert
        assert isinstance(result, bytes), (
            "Quality config should produce valid bytes output"
        )
        assert len(result) > 0, "Quality config output must not be empty"
        assert handler.config.quality == 50, (
            "Handler must retain the configured quality setting"
        )

    def test_optimization_setting_respected(self, format_name):
        """Test that optimization settings are handled appropriately for all formats."""
        # Arrange
        optimize_config = OutputConfig(optimize=True)
        handler = get_output_handler(format_name, optimize_config)
        test_image = Image.new("RGB", (100, 100), "red")

        # Act
        result = handler.get_bytes(test_image)

        # Assert
        assert isinstance(result, bytes), (
            "Optimize config should produce valid bytes output"
        )
        assert len(result) > 0, "Optimize config output must not be empty"
        assert handler.config.optimize is True, (
            "Handler must retain the configured optimization setting"
        )

    def test_config_settings_are_preserved(self, format_name):
        """Test that all config settings are preserved in handler."""
        # Arrange
        complex_config = OutputConfig(quality=75, optimize=True, svg_use_vector=False)

        # Act
        handler = get_output_handler(format_name, complex_config)

        # Assert
        assert handler.config is complex_config, (
            "Handler must use the exact config instance provided"
        )
        assert handler.config.quality == 75, "Quality setting must be preserved"
        assert handler.config.optimize is True, "Optimize setting must be preserved"
        assert handler.config.svg_use_vector is False, (
            "SVG vector setting must be preserved"
        )

    def test_config_changes_dont_affect_other_handlers(self, format_name):
        """Test that config changes don't affect other handler instances."""
        # Arrange
        config1 = OutputConfig(quality=90, optimize=True)
        config2 = OutputConfig(quality=10, optimize=False)

        # Act
        handler1 = get_output_handler(format_name, config1)
        handler2 = get_output_handler(format_name, config2)

        # Modify config1 after handler creation
        config1.quality = 50

        # Assert
        assert handler1.config.quality == 50, (
            "Handler1 should see config changes to its config object"
        )
        assert handler2.config.quality == 10, (
            "Handler2 should not be affected by changes to other configs"
        )
        assert handler1.config is not handler2.config, (
            "Handlers must have independent config instances"
        )

    def test_unsupported_config_options_handled_gracefully(self, format_name):
        """Test that formats handle unsupported config options without errors."""
        # Arrange
        config_with_all_options = OutputConfig(
            quality=85,
            optimize=True,
            svg_use_vector=True,  # Only relevant for SVG
        )
        handler = get_output_handler(format_name, config_with_all_options)
        test_image = Image.new("RGB", (100, 100), "red")

        # Act - should not raise errors even for unsupported options
        result = handler.get_bytes(test_image)

        # Assert
        assert isinstance(result, bytes), (
            "Handler must produce valid output even with unsupported config options"
        )
        assert len(result) > 0, "Output must not be empty regardless of config support"

        # Config should still be preserved even if not all options are used
        assert handler.config.quality == 85, "Quality setting should be preserved"
        assert handler.config.optimize is True, "Optimize setting should be preserved"
        assert handler.config.svg_use_vector is True, "SVG setting should be preserved"

    def test_extreme_quality_values_handled(self, format_name):
        """Test that extreme quality values are handled appropriately."""
        # Arrange
        test_image = Image.new("RGB", (50, 50), "red")

        # Test minimum quality
        min_handler = get_output_handler(format_name, OutputConfig(quality=1))
        # Test maximum quality
        max_handler = get_output_handler(format_name, OutputConfig(quality=100))

        # Act
        min_result = min_handler.get_bytes(test_image)
        max_result = max_handler.get_bytes(test_image)

        # Assert
        assert isinstance(min_result, bytes), "Minimum quality must produce valid bytes"
        assert len(min_result) > 0, "Minimum quality output must not be empty"
        assert isinstance(max_result, bytes), "Maximum quality must produce valid bytes"
        assert len(max_result) > 0, "Maximum quality output must not be empty"


class TestOutputHandlerQualityAndOptimization:
    """Test quality and optimization features where supported."""

    def test_quality_settings_handled_appropriately(self, format_name):
        """Test that quality settings are handled appropriately for all formats."""
        from .conftest import supports_quality

        # Arrange
        test_image = Image.new("RGB", (200, 200), "red")
        high_quality_config = OutputConfig(quality=95)
        low_quality_config = OutputConfig(quality=20)
        format_supports_quality = supports_quality(format_name)

        # Act
        high_quality_handler = get_output_handler(format_name, high_quality_config)
        low_quality_handler = get_output_handler(format_name, low_quality_config)

        high_quality_bytes = high_quality_handler.get_bytes(test_image)
        low_quality_bytes = low_quality_handler.get_bytes(test_image)

        # Assert - all formats should produce valid output
        assert isinstance(high_quality_bytes, bytes), (
            f"High quality {format_name} must produce bytes"
        )
        assert len(high_quality_bytes) > 0, (
            f"High quality {format_name} must produce non-empty output"
        )
        assert isinstance(low_quality_bytes, bytes), (
            f"Low quality {format_name} must produce bytes"
        )
        assert len(low_quality_bytes) > 0, (
            f"Low quality {format_name} must produce non-empty output"
        )

        # Assert - behavior should match format capabilities
        if format_supports_quality:
            # Quality-supporting formats should accept different quality settings
            # (output may or may not differ depending on image content and format)
            assert high_quality_handler.config.quality == 95, (
                "High quality handler must preserve quality setting"
            )
            assert low_quality_handler.config.quality == 20, (
                "Low quality handler must preserve quality setting"
            )
        else:
            # Non-quality formats should produce consistent output regardless of quality setting
            if format_name == "pdf":
                # PDF may vary slightly due to timestamps or metadata
                size_difference = abs(len(high_quality_bytes) - len(low_quality_bytes))
                assert size_difference < 200, (
                    f"PDF quality setting should not significantly affect output size, difference: {size_difference}"
                )
            else:
                # Other non-quality formats should produce identical output
                assert high_quality_bytes == low_quality_bytes, (
                    f"Non-quality format {format_name} should produce identical output regardless of quality setting"
                )

    def test_optimization_settings_handled_appropriately(self, format_name):
        """Test that optimization settings are handled appropriately for all formats."""
        from .conftest import supports_optimization

        # Arrange
        test_image = Image.new("RGB", (100, 100), "red")
        optimized_config = OutputConfig(optimize=True)
        unoptimized_config = OutputConfig(optimize=False)
        format_supports_optimization = supports_optimization(format_name)

        # Act
        optimized_handler = get_output_handler(format_name, optimized_config)
        unoptimized_handler = get_output_handler(format_name, unoptimized_config)

        optimized_bytes = optimized_handler.get_bytes(test_image)
        unoptimized_bytes = unoptimized_handler.get_bytes(test_image)

        # Assert - all formats should produce valid output
        assert isinstance(optimized_bytes, bytes), (
            f"Optimized {format_name} must produce bytes"
        )
        assert len(optimized_bytes) > 0, (
            f"Optimized {format_name} must produce non-empty output"
        )
        assert isinstance(unoptimized_bytes, bytes), (
            f"Unoptimized {format_name} must produce bytes"
        )
        assert len(unoptimized_bytes) > 0, (
            f"Unoptimized {format_name} must produce non-empty output"
        )

        # Assert - behavior should match format capabilities
        if format_supports_optimization:
            # Optimization-supporting formats should accept different optimization settings
            # (optimized output is typically smaller, but not guaranteed for all images)
            assert optimized_handler.config.optimize is True, (
                "Optimized handler must preserve optimization setting"
            )
            assert unoptimized_handler.config.optimize is False, (
                "Unoptimized handler must preserve optimization setting"
            )
        else:
            # Non-optimization formats should produce consistent output regardless of optimization setting
            if format_name == "pdf":
                # PDF may vary slightly due to timestamps or metadata
                size_difference = abs(len(optimized_bytes) - len(unoptimized_bytes))
                assert size_difference < 200, (
                    f"PDF optimization setting should not significantly affect output size, difference: {size_difference}"
                )
            else:
                # Other non-optimization formats should produce identical output
                assert optimized_bytes == unoptimized_bytes, (
                    f"Non-optimization format {format_name} should produce identical output regardless of optimization setting"
                )

    def test_quality_and_optimization_combinations(self, format_name):
        """Test combinations of quality and optimization settings."""
        # Arrange
        test_image = Image.new("RGB", (150, 150), "blue")
        config_combinations = [
            ("high_qual_optimized", OutputConfig(quality=90, optimize=True)),
            ("high_qual_unoptimized", OutputConfig(quality=90, optimize=False)),
            ("low_qual_optimized", OutputConfig(quality=30, optimize=True)),
            ("low_qual_unoptimized", OutputConfig(quality=30, optimize=False)),
        ]

        # Act - generate output for each combination
        results = {}
        for config_name, config in config_combinations:
            handler = get_output_handler(format_name, config)
            output_bytes = handler.get_bytes(test_image)
            results[config_name] = {
                "bytes": output_bytes,
                "size": len(output_bytes),
                "config": config,
            }

        # Assert - all combinations should produce valid output
        for config_name, result in results.items():
            assert isinstance(result["bytes"], bytes), (
                f"Config {config_name} for {format_name} must produce bytes"
            )
            assert result["size"] > 0, (
                f"Config {config_name} for {format_name} must produce non-empty output"
            )

            # Verify config preservation
            expected_quality = result["config"].quality
            expected_optimize = result["config"].optimize
            handler = get_output_handler(format_name, result["config"])
            assert handler.config.quality == expected_quality, (
                f"Quality setting must be preserved for {config_name}"
            )
            assert handler.config.optimize == expected_optimize, (
                f"Optimize setting must be preserved for {config_name}"
            )

    def test_extreme_quality_values_handled_gracefully(self, format_name):
        """Test that extreme quality values are handled without errors."""
        # Arrange
        test_image = Image.new("RGB", (50, 50), "green")
        extreme_configs = [
            ("minimum_quality", OutputConfig(quality=1)),
            ("maximum_quality", OutputConfig(quality=100)),
            ("default_quality", OutputConfig()),  # Default quality for comparison
        ]

        # Act - test each extreme configuration
        results = {}
        for config_name, config in extreme_configs:
            handler = get_output_handler(format_name, config)
            output_bytes = handler.get_bytes(test_image)
            results[config_name] = {
                "bytes": output_bytes,
                "size": len(output_bytes),
                "quality": config.quality,
            }

        # Assert - all extreme values should be handled gracefully
        for config_name, result in results.items():
            assert isinstance(result["bytes"], bytes), (
                f"Extreme quality config {config_name} for {format_name} must produce bytes"
            )
            assert result["size"] > 0, (
                f"Extreme quality config {config_name} for {format_name} must produce non-empty output"
            )
            assert result["size"] < 1000000, (
                f"Output size should be reasonable for {config_name}, got {result['size']} bytes"
            )

        # Assert - quality settings should be preserved even if not used
        min_quality_handler = get_output_handler(format_name, extreme_configs[0][1])
        max_quality_handler = get_output_handler(format_name, extreme_configs[1][1])

        assert min_quality_handler.config.quality == 1, (
            "Minimum quality setting must be preserved"
        )
        assert max_quality_handler.config.quality == 100, (
            "Maximum quality setting must be preserved"
        )
