"""
Configuration and quality tests for output handlers.

Tests configuration handling, quality settings, optimization settings,
and their effects across all output handlers.
"""

from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import get_output_handler

from .conftest import (
    supports_optimization,
    supports_quality,
)


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
