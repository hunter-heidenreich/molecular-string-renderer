"""
Tests for the config_utils module.

Tests configuration utility functions including initialization and coordination.
"""

from unittest.mock import patch

import pytest

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.config_utils import initialize_configurations
from molecular_string_renderer.exceptions import ConfigurationError


class TestInitializeConfigurations:
    """Test the initialize_configurations function."""

    def test_all_none_inputs_creates_defaults(self):
        """Test that None inputs create default configurations."""
        render_config, parser_config, output_config = initialize_configurations(
            render_config=None,
            parser_config=None,
            output_config=None,
            output_format="png",
        )

        # Check that default configs are created
        assert isinstance(render_config, RenderConfig)
        assert isinstance(parser_config, ParserConfig)
        assert isinstance(output_config, OutputConfig)

        # Check default values
        assert render_config.width == 500
        assert render_config.height == 500
        assert render_config.background_color == "white"
        assert render_config.show_hydrogen is False

        assert parser_config.sanitize is True
        assert parser_config.show_hydrogen is False
        assert parser_config.strict is False

        assert output_config.format == "png"
        assert output_config.quality == 95
        assert output_config.dpi == 150

    def test_existing_configs_preserved(self):
        """Test that existing configurations are preserved."""
        render_config = RenderConfig(width=300, height=400, show_hydrogen=True)
        parser_config = ParserConfig(sanitize=False, strict=True)
        output_config = OutputConfig(format="svg", quality=80)

        result_render, result_parser, result_output = initialize_configurations(
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
            output_format="png",  # Should be ignored when output_config provided
        )

        # Check that original configs are preserved
        assert result_render.width == 300
        assert result_render.height == 400
        assert result_render.show_hydrogen is True

        assert result_parser.sanitize is False
        assert result_parser.strict is True

        assert result_output.format == "svg"
        assert result_output.quality == 80

    def test_output_format_used_for_default_output_config(self):
        """Test that output_format is used when output_config is None."""
        formats_to_test = ["svg", "jpeg", "pdf", "webp"]

        for fmt in formats_to_test:
            _, _, output_config = initialize_configurations(
                render_config=None,
                parser_config=None,
                output_config=None,
                output_format=fmt,
            )
            assert output_config.format == fmt

    def test_hydrogen_coordination_from_render_to_parser(self):
        """Test auto-coordination of hydrogen settings from render to parser."""
        # Case: render shows hydrogen, parser doesn't - should coordinate
        render_config = RenderConfig(show_hydrogen=True)
        parser_config = ParserConfig(show_hydrogen=False, sanitize=True, strict=False)

        result_render, result_parser, _ = initialize_configurations(
            render_config=render_config,
            parser_config=parser_config,
            output_config=None,
            output_format="png",
        )

        # Render config should be unchanged
        assert result_render.show_hydrogen is True

        # Parser config should be coordinated (new instance with hydrogen=True)
        assert result_parser.show_hydrogen is True
        assert result_parser.sanitize is True  # Other values preserved
        assert result_parser.strict is False

    def test_no_hydrogen_coordination_when_both_false(self):
        """Test no coordination when both configs have hydrogen=False."""
        render_config = RenderConfig(show_hydrogen=False)
        parser_config = ParserConfig(show_hydrogen=False, sanitize=False, strict=True)

        result_render, result_parser, _ = initialize_configurations(
            render_config=render_config,
            parser_config=parser_config,
            output_config=None,
            output_format="png",
        )

        # No coordination should occur
        assert result_render.show_hydrogen is False
        assert result_parser.show_hydrogen is False
        assert result_parser.sanitize is False
        assert result_parser.strict is True

    def test_no_hydrogen_coordination_when_both_true(self):
        """Test no coordination when both configs have hydrogen=True."""
        render_config = RenderConfig(show_hydrogen=True)
        parser_config = ParserConfig(show_hydrogen=True, sanitize=False, strict=True)

        result_render, result_parser, _ = initialize_configurations(
            render_config=render_config,
            parser_config=parser_config,
            output_config=None,
            output_format="png",
        )

        # No coordination needed
        assert result_render.show_hydrogen is True
        assert result_parser.show_hydrogen is True
        assert result_parser.sanitize is False
        assert result_parser.strict is True

    def test_no_hydrogen_coordination_when_render_false_parser_true(self):
        """Test no coordination when render=False and parser=True."""
        render_config = RenderConfig(show_hydrogen=False)
        parser_config = ParserConfig(show_hydrogen=True, sanitize=True, strict=False)

        result_render, result_parser, _ = initialize_configurations(
            render_config=render_config,
            parser_config=parser_config,
            output_config=None,
            output_format="png",
        )

        # No coordination should occur (render doesn't need hydrogens)
        assert result_render.show_hydrogen is False
        assert result_parser.show_hydrogen is True
        assert result_parser.sanitize is True
        assert result_parser.strict is False

    def test_mixed_none_and_existing_configs(self):
        """Test mixing None and existing configurations."""
        # Only render_config provided
        render_config = RenderConfig(width=600, show_hydrogen=True)

        result_render, result_parser, result_output = initialize_configurations(
            render_config=render_config,
            parser_config=None,
            output_config=None,
            output_format="svg",
        )

        assert result_render.width == 600
        assert result_render.show_hydrogen is True

        # Default parser should be coordinated for hydrogen
        assert result_parser.show_hydrogen is True
        assert result_parser.sanitize is True  # Default
        assert result_parser.strict is False  # Default

        # Default output with specified format
        assert result_output.format == "svg"
        assert result_output.quality == 95  # Default

    def test_parser_config_recreation_preserves_all_fields(self):
        """Test that parser config recreation during coordination preserves all fields."""
        original_parser = ParserConfig(sanitize=False, show_hydrogen=False, strict=True)
        render_config = RenderConfig(show_hydrogen=True)

        _, result_parser, _ = initialize_configurations(
            render_config=render_config,
            parser_config=original_parser,
            output_config=None,
            output_format="png",
        )

        # Should create new parser config with hydrogen=True but preserve other fields
        assert result_parser.show_hydrogen is True  # Coordinated
        assert result_parser.sanitize is False  # Preserved
        assert result_parser.strict is True  # Preserved

    @patch(
        "molecular_string_renderer.config_utils.validate_configuration_compatibility"
    )
    def test_validation_called_with_final_configs(self, mock_validate):
        """Test that validation is called with the final coordinated configurations."""
        render_config = RenderConfig(show_hydrogen=True)
        parser_config = ParserConfig(show_hydrogen=False)

        result_render, result_parser, result_output = initialize_configurations(
            render_config=render_config,
            parser_config=parser_config,
            output_config=None,
            output_format="png",
        )

        # Validation should be called once with the final configs
        mock_validate.assert_called_once_with(
            result_render, result_parser, result_output
        )

        # The parser config passed to validation should have hydrogen=True
        called_args = mock_validate.call_args[0]
        assert called_args[1].show_hydrogen is True  # Coordinated parser config

    @patch(
        "molecular_string_renderer.config_utils.validate_configuration_compatibility"
    )
    def test_validation_error_propagated(self, mock_validate):
        """Test that validation errors are propagated correctly."""
        mock_validate.side_effect = ConfigurationError("Test validation error")

        with pytest.raises(ConfigurationError, match="Test validation error"):
            initialize_configurations(
                render_config=None,
                parser_config=None,
                output_config=None,
                output_format="png",
            )

    def test_return_type_is_tuple(self):
        """Test that the function returns a tuple of exactly 3 items."""
        result = initialize_configurations(
            render_config=None,
            parser_config=None,
            output_config=None,
            output_format="png",
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], RenderConfig)
        assert isinstance(result[1], ParserConfig)
        assert isinstance(result[2], OutputConfig)

    def test_complex_coordination_scenario(self):
        """Test a complex scenario with multiple coordination needs."""
        # Custom render config that needs hydrogen coordination
        render_config = RenderConfig(
            width=800,
            height=600,
            background_color="#f0f0f0",
            show_hydrogen=True,
            show_carbon=True,
            highlight_atoms=[0, 1, 2],
        )

        # Custom parser config that needs coordination
        parser_config = ParserConfig(
            sanitize=False,
            show_hydrogen=False,
            strict=True,  # Will be coordinated
        )

        # Custom output config
        output_config = OutputConfig(
            format="svg", quality=90, dpi=300, svg_line_width_mult=1.5
        )

        result_render, result_parser, result_output = initialize_configurations(
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
            output_format="png",  # Should be ignored
        )

        # Render config should be unchanged
        assert result_render.width == 800
        assert result_render.height == 600
        assert result_render.background_color == "#f0f0f0"
        assert result_render.show_hydrogen is True
        assert result_render.show_carbon is True
        assert result_render.highlight_atoms == [0, 1, 2]

        # Parser config should be coordinated
        assert result_parser.sanitize is False  # Preserved
        assert result_parser.show_hydrogen is True  # Coordinated
        assert result_parser.strict is True  # Preserved

        # Output config should be unchanged
        assert result_output.format == "svg"  # Not "png"
        assert result_output.quality == 90
        assert result_output.dpi == 300
        assert result_output.svg_line_width_mult == 1.5

    def test_edge_case_empty_string_format(self):
        """Test behavior with edge case inputs."""
        # This should work as OutputConfig will validate the format
        with pytest.raises(Exception):  # OutputConfig should validate this
            initialize_configurations(
                render_config=None,
                parser_config=None,
                output_config=None,
                output_format="",
            )

    def test_coordination_creates_new_parser_instance(self):
        """Test that coordination creates a new ParserConfig instance."""
        original_parser = ParserConfig(show_hydrogen=False, sanitize=True, strict=False)
        render_config = RenderConfig(show_hydrogen=True)

        _, result_parser, _ = initialize_configurations(
            render_config=render_config,
            parser_config=original_parser,
            output_config=None,
            output_format="png",
        )

        # Should be a different instance
        assert result_parser is not original_parser
        assert result_parser.show_hydrogen is True
        assert original_parser.show_hydrogen is False  # Original unchanged


class TestConfigUtilsIntegration:
    """Integration tests for config_utils with real validation."""

    def test_successful_integration_with_real_validation(self):
        """Test successful integration with actual validation function."""
        # This should work without mocking validation
        render_config, parser_config, output_config = initialize_configurations(
            render_config=RenderConfig(width=300, height=300),
            parser_config=ParserConfig(sanitize=True),
            output_config=OutputConfig(format="png", quality=85),
            output_format="svg",  # Should be ignored
        )

        assert render_config.width == 300
        assert parser_config.sanitize is True
        assert output_config.format == "png"

    def test_validation_failure_integration(self):
        """Test that real validation failures are properly handled."""
        # Create configs that should fail validation
        render_config = RenderConfig(background_color="transparent")
        output_config = OutputConfig(format="jpeg")  # JPEG + transparent = invalid

        with pytest.raises(ConfigurationError, match="JPEG.*transparent"):
            initialize_configurations(
                render_config=render_config,
                parser_config=None,
                output_config=output_config,
                output_format="png",
            )

    def test_coordination_with_validation_integration(self):
        """Test that coordination works correctly with real validation."""
        # Test case where coordination is needed and validation should pass
        render_config = RenderConfig(show_hydrogen=True, width=400, height=400)
        parser_config = ParserConfig(show_hydrogen=False, sanitize=True)

        result_render, result_parser, result_output = initialize_configurations(
            render_config=render_config,
            parser_config=parser_config,
            output_config=None,
            output_format="svg",
        )

        # Coordination should have occurred
        assert result_render.show_hydrogen is True
        assert result_parser.show_hydrogen is True
        assert result_parser.sanitize is True

        # Validation should have passed (no exception raised)
        assert result_output.format == "svg"
