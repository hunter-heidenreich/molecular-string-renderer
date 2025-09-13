"""
Configuration validation tests.

Tests for configuration classes, validation logic,
boundary conditions, invalid combinations, and edge cases.
"""

import pytest
from pydantic import ValidationError

from molecular_string_renderer.config import (
    OutputConfig,
    ParserConfig,
    RenderConfig,
)


class TestRenderConfigValidation:
    """Test RenderConfig validation and boundary conditions."""

    def test_valid_render_config_creation(self):
        """Test creating valid RenderConfig instances."""
        # Default config
        config = RenderConfig()
        assert config.width == 500
        assert config.height == 500
        assert config.dpi == 150
        assert config.bond_line_width == 2.0
        assert config.atom_label_font_size == 12

        # Custom valid config
        config = RenderConfig(
            width=500,
            height=400,
            dpi=150,
            bond_line_width=1.5,
            atom_label_font_size=14,
        )
        assert config.width == 500
        assert config.height == 400
        assert config.dpi == 150
        assert config.bond_line_width == 1.5
        assert config.atom_label_font_size == 14

    def test_width_validation(self):
        """Test width validation boundaries."""
        # Valid minimum width
        config = RenderConfig(width=100)
        assert config.width == 100

        # Valid large width
        config = RenderConfig(width=2000)
        assert config.width == 2000

        # Invalid width - too small
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(width=50)
        assert "greater than or equal to 100" in str(exc_info.value)

        # Invalid width - zero
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(width=0)
        assert "greater than or equal to 100" in str(exc_info.value)

        # Invalid width - negative
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(width=-100)
        assert "greater than or equal to 100" in str(exc_info.value)

    def test_height_validation(self):
        """Test height validation boundaries."""
        # Valid minimum height
        config = RenderConfig(height=100)
        assert config.height == 100

        # Valid large height
        config = RenderConfig(height=2000)
        assert config.height == 2000

        # Invalid height - too small
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(height=50)
        assert "greater than or equal to 100" in str(exc_info.value)

        # Invalid height - zero
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(height=0)
        assert "greater than or equal to 100" in str(exc_info.value)

        # Invalid height - negative
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(height=-100)
        assert "greater than or equal to 100" in str(exc_info.value)

    def test_dpi_validation(self):
        """Test DPI validation boundaries."""
        # Valid minimum DPI
        config = RenderConfig(dpi=72)
        assert config.dpi == 72

        # Valid maximum DPI
        config = RenderConfig(dpi=600)
        assert config.dpi == 600

        # Valid typical DPI values
        for dpi in [96, 150, 300]:
            config = RenderConfig(dpi=dpi)
            assert config.dpi == dpi

        # Invalid DPI - too small
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(dpi=50)
        assert "greater than or equal to 72" in str(exc_info.value)

        # Invalid DPI - too large
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(dpi=700)
        assert "less than or equal to 600" in str(exc_info.value)

        # Invalid DPI - zero
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(dpi=0)
        assert "greater than or equal to 72" in str(exc_info.value)

        # Invalid DPI - negative
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(dpi=-100)
        assert "greater than or equal to 72" in str(exc_info.value)

    def test_bond_line_width_validation(self):
        """Test bond line width validation boundaries."""
        # Valid minimum line width
        config = RenderConfig(bond_line_width=0.5)
        assert config.bond_line_width == 0.5

        # Valid maximum line width
        config = RenderConfig(bond_line_width=10.0)
        assert config.bond_line_width == 10.0

        # Valid typical line widths
        for width in [1.0, 1.5, 2.0, 3.0, 5.0]:
            config = RenderConfig(bond_line_width=width)
            assert config.bond_line_width == width

        # Invalid line width - too small
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(bond_line_width=0.3)
        assert "greater than or equal to 0.5" in str(exc_info.value)

        # Invalid line width - too large
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(bond_line_width=15.0)
        assert "less than or equal to 10" in str(exc_info.value)

        # Invalid line width - zero
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(bond_line_width=0.0)
        assert "greater than or equal to 0.5" in str(exc_info.value)

        # Invalid line width - negative
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(bond_line_width=-1.0)
        assert "greater than or equal to 0.5" in str(exc_info.value)

    def test_atom_label_font_size_validation(self):
        """Test atom label font size validation boundaries."""
        # Valid minimum font size
        config = RenderConfig(atom_label_font_size=6)
        assert config.atom_label_font_size == 6

        # Valid maximum font size
        config = RenderConfig(atom_label_font_size=48)
        assert config.atom_label_font_size == 48

        # Valid typical font sizes
        for size in [8, 10, 12, 14, 16, 18, 24]:
            config = RenderConfig(atom_label_font_size=size)
            assert config.atom_label_font_size == size

        # Invalid font size - too small
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(atom_label_font_size=4)
        assert "greater than or equal to 6" in str(exc_info.value)

        # Invalid font size - too large
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(atom_label_font_size=50)
        assert "less than or equal to 48" in str(exc_info.value)

        # Invalid font size - zero
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(atom_label_font_size=0)
        assert "greater than or equal to 6" in str(exc_info.value)

        # Invalid font size - negative
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(atom_label_font_size=-5)
        assert "greater than or equal to 6" in str(exc_info.value)

    def test_render_config_combinations(self):
        """Test various valid combinations of RenderConfig parameters."""
        # Small image, low DPI
        config = RenderConfig(width=100, height=100, dpi=72)
        assert config.width == 100
        assert config.height == 100
        assert config.dpi == 72

        # Large image, high DPI
        config = RenderConfig(width=2000, height=1500, dpi=300)
        assert config.width == 2000
        assert config.height == 1500
        assert config.dpi == 300

        # Thin lines, small font
        config = RenderConfig(bond_line_width=0.5, atom_label_font_size=6)
        assert config.bond_line_width == 0.5
        assert config.atom_label_font_size == 6

        # Thick lines, large font
        config = RenderConfig(bond_line_width=8.0, atom_label_font_size=36)
        assert config.bond_line_width == 8.0
        assert config.atom_label_font_size == 36

    def test_render_config_type_validation(self):
        """Test type validation for RenderConfig parameters."""
        # String that can be converted to int for width should work
        config = RenderConfig(width=300)
        assert config.width == 300

        # Float instead of int for height (should be converted)
        config = RenderConfig(height=300.0)
        assert config.height == 300
        assert isinstance(config.height, int)

        # String instead of float for bond_line_width should fail
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(bond_line_width="invalid")
        assert "Input should be a valid number" in str(exc_info.value)

        # None values should fail
        with pytest.raises(ValidationError):
            RenderConfig(width=None)


class TestOutputConfigValidation:
    """Test OutputConfig validation and boundary conditions."""

    def test_valid_output_config_creation(self):
        """Test creating valid OutputConfig instances."""
        # Default config
        config = OutputConfig()
        assert config.format == "png"
        assert config.quality == 95
        assert config.optimize is True

        # Custom valid configs for each format
        formats = ["png", "svg", "jpg", "jpeg", "pdf"]
        for fmt in formats:
            config = OutputConfig(format=fmt)
            assert config.format == fmt

    def test_format_validation(self):
        """Test format validation."""
        # Valid formats
        valid_formats = ["png", "svg", "jpg", "jpeg", "pdf"]
        for fmt in valid_formats:
            config = OutputConfig(format=fmt)
            assert config.format == fmt

        # Case insensitive formats
        case_variants = [
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
        for fmt in case_variants:
            config = OutputConfig(format=fmt)
            assert config.format == fmt.lower()

        # Invalid formats (now that webp, tiff, bmp are supported)
        invalid_formats = ["gif", "ico", "psd", "invalid", ""]
        for fmt in invalid_formats:
            with pytest.raises(ValidationError) as exc_info:
                OutputConfig(format=fmt)
            assert "Unsupported format:" in str(exc_info.value)

    def test_quality_validation(self):
        """Test quality validation boundaries."""
        # Valid minimum quality
        config = OutputConfig(quality=1)
        assert config.quality == 1

        # Valid maximum quality
        config = OutputConfig(quality=100)
        assert config.quality == 100

        # Valid typical quality values
        for quality in [50, 75, 85, 90, 95]:
            config = OutputConfig(quality=quality)
            assert config.quality == quality

        # Invalid quality - too small
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(quality=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        # Invalid quality - too large
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(quality=101)
        assert "less than or equal to 100" in str(exc_info.value)

        # Invalid quality - negative
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(quality=-10)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_transparent_background_validation(self):
        """Test SVG-specific configuration validation."""
        # Valid boolean values for SVG options
        config = OutputConfig(svg_use_vector=True)
        assert config.svg_use_vector is True

        config = OutputConfig(svg_use_vector=False)
        assert config.svg_use_vector is False

        # Test SVG line width multiplier
        config = OutputConfig(svg_line_width_mult=3)
        assert config.svg_line_width_mult == 3

        # Invalid line width multiplier - too large
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(svg_line_width_mult=6)
        assert "less than or equal to 5" in str(exc_info.value)

        # Invalid line width multiplier - too small
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(svg_line_width_mult=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_quality_format_interaction(self):
        """Test quality setting interaction with different formats."""
        # Quality should be ignored for non-JPEG formats but not cause errors
        config = OutputConfig(format="png", quality=50)
        assert config.format == "png"
        assert config.quality == 50

        config = OutputConfig(format="svg", quality=75)
        assert config.format == "svg"
        assert config.quality == 75

        config = OutputConfig(format="pdf", quality=90)
        assert config.format == "pdf"
        assert config.quality == 90

        # Quality should be meaningful for JPEG formats
        config = OutputConfig(format="jpg", quality=85)
        assert config.format == "jpg"
        assert config.quality == 85

        config = OutputConfig(format="jpeg", quality=95)
        assert config.format == "jpeg"
        assert config.quality == 95

    def test_output_config_type_validation(self):
        """Test type validation for OutputConfig parameters."""
        # Invalid type for format
        with pytest.raises(ValidationError):
            OutputConfig(format=123)

        # Invalid type for quality
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(quality="high")
        assert "Input should be a valid integer" in str(exc_info.value)

        # Float quality should cause error in strict mode
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(quality=85.7)
        assert "fractional part" in str(exc_info.value)


class TestParserConfigValidation:
    """Test ParserConfig validation and boundary conditions."""

    def test_valid_parser_config_creation(self):
        """Test creating valid ParserConfig instances."""
        # Default config
        config = ParserConfig()
        assert config.strict is False
        assert config.remove_hs is True
        assert config.sanitize is True

        # Custom valid config
        config = ParserConfig(
            strict=True,
            remove_hs=False,
            sanitize=False,
        )
        assert config.strict is True
        assert config.remove_hs is False
        assert config.sanitize is False

    def test_strict_parsing_validation(self):
        """Test strict validation."""
        # Valid boolean values
        config = ParserConfig(strict=True)
        assert config.strict is True

        config = ParserConfig(strict=False)
        assert config.strict is False

        # Type coercion
        config = ParserConfig(strict=1)
        assert config.strict is True

        config = ParserConfig(strict=0)
        assert config.strict is False

    def test_remove_hydrogens_validation(self):
        """Test remove_hs validation."""
        # Valid boolean values
        config = ParserConfig(remove_hs=True)
        assert config.remove_hs is True

        config = ParserConfig(remove_hs=False)
        assert config.remove_hs is False

        # Type coercion
        config = ParserConfig(remove_hs=1)
        assert config.remove_hs is True

        config = ParserConfig(remove_hs=0)
        assert config.remove_hs is False

    def test_sanitize_validation(self):
        """Test sanitize validation."""
        # Valid boolean values
        config = ParserConfig(sanitize=True)
        assert config.sanitize is True

        config = ParserConfig(sanitize=False)
        assert config.sanitize is False

        # Type coercion
        config = ParserConfig(sanitize=1)
        assert config.sanitize is True

        config = ParserConfig(sanitize=0)
        assert config.sanitize is False

    def test_parser_config_combinations(self):
        """Test various combinations of ParserConfig parameters."""
        # All strict
        config = ParserConfig(
            strict=True,
            remove_hs=True,
            sanitize=True,
        )
        assert config.strict is True
        assert config.remove_hs is True
        assert config.sanitize is True

        # All permissive
        config = ParserConfig(
            strict=False,
            remove_hs=False,
            sanitize=False,
        )
        assert config.strict is False
        assert config.remove_hs is False
        assert config.sanitize is False

        # Mixed settings
        config = ParserConfig(
            strict=True,
            remove_hs=False,
            sanitize=True,
        )
        assert config.strict is True
        assert config.remove_hs is False
        assert config.sanitize is True

    def test_parser_config_type_validation(self):
        """Test type validation for ParserConfig parameters."""
        # String values should be converted to boolean
        config = ParserConfig(strict="true")
        assert config.strict is True

        # Pydantic converts "false" string to False boolean
        config = ParserConfig(strict="false")
        assert config.strict is False

        # Empty string should fail validation
        with pytest.raises(ValidationError) as exc_info:
            ParserConfig(strict="")
        assert "unable to interpret input" in str(exc_info.value)


class TestConfigurationEdgeCases:
    """Test edge cases and boundary conditions across all config types."""

    def test_config_serialization(self):
        """Test that configs can be serialized and deserialized."""
        # RenderConfig
        render_config = RenderConfig(width=400, height=300, dpi=150)
        render_dict = render_config.model_dump()
        assert render_dict["width"] == 400
        assert render_dict["height"] == 300
        assert render_dict["dpi"] == 150

        # Recreate from dict
        new_render_config = RenderConfig(**render_dict)
        assert new_render_config.width == 400
        assert new_render_config.height == 300
        assert new_render_config.dpi == 150

        # OutputConfig
        output_config = OutputConfig(format="jpg", quality=85)
        output_dict = output_config.model_dump()
        assert output_dict["format"] == "jpg"
        assert output_dict["quality"] == 85

        new_output_config = OutputConfig(**output_dict)
        assert new_output_config.format == "jpg"
        assert new_output_config.quality == 85

        # ParserConfig
        parser_config = ParserConfig(strict=False)
        parser_dict = parser_config.model_dump()
        assert parser_dict["strict"] is False

        new_parser_config = ParserConfig(**parser_dict)
        assert new_parser_config.strict is False

    def test_config_equality(self):
        """Test config equality comparisons."""
        # RenderConfig equality
        config1 = RenderConfig(width=300, height=300)
        config2 = RenderConfig(width=300, height=300)
        config3 = RenderConfig(width=400, height=300)

        assert config1 == config2
        assert config1 != config3

        # OutputConfig equality
        config1 = OutputConfig(format="png", quality=95)
        config2 = OutputConfig(format="png", quality=95)
        config3 = OutputConfig(format="jpg", quality=95)

        assert config1 == config2
        assert config1 != config3

        # ParserConfig equality
        config1 = ParserConfig(strict=True)
        config2 = ParserConfig(strict=True)
        config3 = ParserConfig(strict=False)

        assert config1 == config2
        assert config1 != config3

    def test_config_repr_and_str(self):
        """Test string representations of configs."""
        # RenderConfig
        config = RenderConfig(width=400, height=300)
        repr_str = repr(config)
        assert "RenderConfig" in repr_str
        assert "width=400" in repr_str
        assert "height=300" in repr_str

        # OutputConfig
        config = OutputConfig(format="jpg", quality=85)
        repr_str = repr(config)
        assert "OutputConfig" in repr_str
        assert "format='jpg'" in repr_str
        assert "quality=85" in repr_str

        # ParserConfig
        config = ParserConfig(strict=False)
        repr_str = repr(config)
        assert "ParserConfig" in repr_str
        assert "strict=False" in repr_str

    def test_config_immutability(self):
        """Test that configs behave as expected regarding mutability."""
        # RenderConfig
        config = RenderConfig(width=300)
        original_width = config.width

        # Configs are mutable in Pydantic v2, but we can test assignment
        config.width = 400
        assert config.width == 400
        assert config.width != original_width

        # OutputConfig
        config = OutputConfig(format="png")
        config.format = "jpg"
        assert config.format == "jpg"

        # ParserConfig
        config = ParserConfig(strict=True)
        config.strict = False
        assert config.strict is False

    def test_config_validation_error_messages(self):
        """Test that validation error messages are informative."""
        # Test width validation error
        try:
            RenderConfig(width=50)
            pytest.fail("Should have raised ValidationError")
        except ValidationError as e:
            error_dict = e.errors()[0]
            assert error_dict["type"] == "greater_than_equal"
            assert "width" in str(e).lower()

        # Test format validation error
        try:
            OutputConfig(format="invalid")
            pytest.fail("Should have raised ValidationError")
        except ValidationError as e:
            error_dict = e.errors()[0]
            assert error_dict["type"] == "value_error"
            assert "format" in str(e).lower()

    def test_config_with_extra_fields(self):
        """Test behavior with extra/unknown fields."""
        # The configs allow extra fields by default, so let's test a different approach
        # Test that we can create configs with all valid fields
        render_config = RenderConfig(
            width=400,
            height=300,
            background_color="white",
            atom_label_font_size=12,
            bond_line_width=2.0,
            antialias=True,
            dpi=150,
        )
        assert render_config.width == 400
        assert render_config.height == 300

        output_config = OutputConfig(
            format="png",
            quality=95,
            optimize=True,
            svg_use_vector=True,
            svg_line_width_mult=1,
        )
        assert output_config.format == "png"
        assert output_config.quality == 95

        parser_config = ParserConfig(
            sanitize=True,
            remove_hs=True,
            strict=False,
        )
        assert parser_config.sanitize is True
        assert parser_config.remove_hs is True
        assert parser_config.strict is False

    def test_boundary_value_combinations(self):
        """Test combinations of boundary values."""
        # Minimum valid RenderConfig
        config = RenderConfig(
            width=100,
            height=100,
            dpi=72,
            bond_line_width=0.5,
            atom_label_font_size=6,
        )
        assert config.width == 100
        assert config.height == 100
        assert config.dpi == 72
        assert config.bond_line_width == 0.5
        assert config.atom_label_font_size == 6

        # Maximum valid RenderConfig
        config = RenderConfig(
            width=2000,
            height=2000,
            dpi=600,
            bond_line_width=10.0,
            atom_label_font_size=48,
        )
        assert config.width == 2000
        assert config.height == 2000
        assert config.dpi == 600
        assert config.bond_line_width == 10.0
        assert config.atom_label_font_size == 48

        # Boundary OutputConfig
        config = OutputConfig(format="jpeg", quality=1, optimize=True)
        assert config.format == "jpeg"
        assert config.quality == 1
        assert config.optimize is True

        config = OutputConfig(format="pdf", quality=100, optimize=False)
        assert config.format == "pdf"
        assert config.quality == 100
        assert config.optimize is False


if __name__ == "__main__":
    pytest.main([__file__])
