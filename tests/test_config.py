"""
Tests for the config module.

Tests configuration classes and their validation methods.
"""

import pytest
from pydantic import ValidationError

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig


class TestRenderConfig:
    """Test RenderConfig class and its methods."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = RenderConfig()

        assert config.width == 500
        assert config.height == 500
        assert config.background_color == "white"
        assert config.show_hydrogen is False
        assert config.show_carbon is False
        assert config.highlight_atoms is None
        assert config.highlight_bonds is None

    def test_custom_values(self):
        """Test initialization with custom values."""
        config = RenderConfig(
            width=300,
            height=400,
            background_color="#ff0000",
            show_hydrogen=True,
            show_carbon=True,
            highlight_atoms=[0, 1, 2],
            highlight_bonds=[0, 1],
        )

        assert config.width == 300
        assert config.height == 400
        assert config.background_color == "#ff0000"
        assert config.show_hydrogen is True
        assert config.show_carbon is True
        assert config.highlight_atoms == [0, 1, 2]
        assert config.highlight_bonds == [0, 1]

    def test_width_validation(self):
        """Test width field validation."""
        # Valid width
        config = RenderConfig(width=1000)
        assert config.width == 1000

        # Invalid width - too small
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(width=50)
        assert "greater than or equal to 100" in str(exc_info.value)

        # Invalid width - too large
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(width=3000)
        assert "less than or equal to 2000" in str(exc_info.value)

    def test_height_validation(self):
        """Test height field validation."""
        # Valid height
        config = RenderConfig(height=800)
        assert config.height == 800

        # Invalid height - too small
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(height=50)
        assert "greater than or equal to 100" in str(exc_info.value)

        # Invalid height - too large
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(height=3000)
        assert "less than or equal to 2000" in str(exc_info.value)

    def test_validate_color_valid_names(self):
        """Test validate_color with valid color names."""
        # Valid color names should pass through unchanged
        valid_colors = ["white", "black", "red", "blue", "green", "transparent"]

        for color in valid_colors:
            config = RenderConfig(background_color=color)
            assert config.background_color == color

    def test_validate_color_valid_hex(self):
        """Test validate_color with valid hex colors."""
        # Valid hex colors - short form
        config = RenderConfig(background_color="#fff")
        assert config.background_color == "#fff"

        config = RenderConfig(background_color="#000")
        assert config.background_color == "#000"

        # Valid hex colors - long form
        config = RenderConfig(background_color="#ffffff")
        assert config.background_color == "#ffffff"

        config = RenderConfig(background_color="#123456")
        assert config.background_color == "#123456"

        # Mixed case should work
        config = RenderConfig(background_color="#AbCdEf")
        assert config.background_color == "#AbCdEf"

    def test_validate_color_invalid_hex(self):
        """Test validate_color with invalid hex colors."""
        # Invalid hex - wrong length
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(background_color="#ff")
        assert "Invalid hex color" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(background_color="#fffff")
        assert "Invalid hex color" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(background_color="#fffffff")
        assert "Invalid hex color" in str(exc_info.value)

        # Invalid hex - invalid characters
        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(background_color="#gggggg")
        assert "Invalid hex color" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RenderConfig(background_color="#12345z")
        assert "Invalid hex color" in str(exc_info.value)

    def test_size_property(self):
        """Test the size property returns correct tuple."""
        config = RenderConfig(width=300, height=400)
        assert config.size == (300, 400)

        config = RenderConfig()  # Default values
        assert config.size == (500, 500)

    def test_to_rdkit_options_defaults(self):
        """Test to_rdkit_options with default values."""
        config = RenderConfig()
        options = config.to_rdkit_options()

        expected = {
            "addAtomIndices": False,
            "addBondIndices": False,
            "highlightAtoms": [],
            "highlightBonds": [],
            "explicitMethyl": False,
            "bondLineWidth": 2.0,
            "addStereoAnnotation": False,
            "includeRadicals": True,
            "rotate": 0.0,
        }

        assert options == expected

    def test_to_rdkit_options_custom_values(self):
        """Test to_rdkit_options with custom values."""
        config = RenderConfig(
            show_carbon=True, highlight_atoms=[0, 1, 2], highlight_bonds=[0, 1]
        )
        options = config.to_rdkit_options()

        expected = {
            "addAtomIndices": False,
            "addBondIndices": False,
            "highlightAtoms": [0, 1, 2],
            "highlightBonds": [0, 1],
            "explicitMethyl": True,
            "bondLineWidth": 2.0,
            "addStereoAnnotation": False,
            "includeRadicals": True,
            "rotate": 0.0,
        }

        assert options == expected

    def test_to_rdkit_options_none_highlights(self):
        """Test to_rdkit_options handles None highlight lists."""
        config = RenderConfig(highlight_atoms=None, highlight_bonds=None)
        options = config.to_rdkit_options()

        assert options["highlightAtoms"] == []
        assert options["highlightBonds"] == []


class TestParserConfig:
    """Test ParserConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ParserConfig()

        assert config.sanitize is True
        assert config.show_hydrogen is False
        assert config.strict is False

    def test_custom_values(self):
        """Test initialization with custom values."""
        config = ParserConfig(sanitize=False, show_hydrogen=True, strict=True)

        assert config.sanitize is False
        assert config.show_hydrogen is True
        assert config.strict is True


class TestOutputConfig:
    """Test OutputConfig class and its methods."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = OutputConfig()

        assert config.format == "png"
        assert config.quality == 95
        assert config.optimize is True
        assert config.dpi == 150
        assert config.progressive is False
        assert config.lossless is True
        assert config.metadata is None
        assert config.svg_sanitize is True
        assert config.svg_use_vector is True
        assert config.svg_line_width_mult == 1.0

    def test_custom_values(self):
        """Test initialization with custom values."""
        metadata = {"title": "Test Image", "author": "Test Author"}
        config = OutputConfig(
            format="svg",
            quality=80,
            optimize=False,
            dpi=300,
            progressive=True,
            lossless=False,
            metadata=metadata,
            svg_sanitize=False,
            svg_use_vector=False,
            svg_line_width_mult=2.5,
        )

        assert config.format == "svg"
        assert config.quality == 80
        assert config.optimize is False
        assert config.dpi == 300
        assert config.progressive is True
        assert config.lossless is False
        assert config.metadata == metadata
        assert config.svg_sanitize is False
        assert config.svg_use_vector is False
        assert config.svg_line_width_mult == 2.5

    def test_quality_validation(self):
        """Test quality field validation."""
        # Valid quality
        config = OutputConfig(quality=50)
        assert config.quality == 50

        # Invalid quality - too low
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(quality=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        # Invalid quality - too high
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(quality=101)
        assert "less than or equal to 100" in str(exc_info.value)

    def test_dpi_validation(self):
        """Test DPI field validation."""
        # Valid DPI
        config = OutputConfig(dpi=300)
        assert config.dpi == 300

        # Invalid DPI - too low
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(dpi=50)
        assert "greater than or equal to 72" in str(exc_info.value)

        # Invalid DPI - too high
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(dpi=700)
        assert "less than or equal to 600" in str(exc_info.value)

    def test_svg_line_width_mult_validation(self):
        """Test SVG line width multiplier validation."""
        # Valid multiplier
        config = OutputConfig(svg_line_width_mult=2.0)
        assert config.svg_line_width_mult == 2.0

        # Invalid multiplier - too low
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(svg_line_width_mult=0.05)
        assert "greater than or equal to 0.1" in str(exc_info.value)

        # Invalid multiplier - too high
        with pytest.raises(ValidationError) as exc_info:
            OutputConfig(svg_line_width_mult=6.0)
        assert "less than or equal to 5" in str(exc_info.value)

    def test_validate_format_supported_formats(self):
        """Test validate_format with supported formats."""
        supported_formats = [
            "png",
            "svg",
            "jpg",
            "jpeg",
            "pdf",
            "webp",
            "tiff",
            "tif",
            "bmp",
            "gif",
        ]

        for fmt in supported_formats:
            config = OutputConfig(format=fmt)
            assert config.format == fmt.lower()

    def test_validate_format_case_insensitive(self):
        """Test validate_format normalizes case."""
        config = OutputConfig(format="PNG")
        assert config.format == "png"

        config = OutputConfig(format="JPEG")
        assert config.format == "jpeg"

        config = OutputConfig(format="WebP")
        assert config.format == "webp"

    def test_validate_format_unsupported(self):
        """Test validate_format with unsupported formats."""
        unsupported_formats = ["ico", "raw", "invalid", "xyz"]

        for fmt in unsupported_formats:
            with pytest.raises(ValidationError) as exc_info:
                OutputConfig(format=fmt)
            assert f"Unsupported format: {fmt}" in str(exc_info.value)
            assert "Supported:" in str(exc_info.value)


class TestConfigIntegration:
    """Test integration between config classes."""

    def test_all_configs_can_be_created_with_defaults(self):
        """Test that all config classes can be instantiated with defaults."""
        render_config = RenderConfig()
        parser_config = ParserConfig()
        output_config = OutputConfig()

        # Should not raise any exceptions
        assert isinstance(render_config, RenderConfig)
        assert isinstance(parser_config, ParserConfig)
        assert isinstance(output_config, OutputConfig)

    def test_config_serialization(self):
        """Test that configs can be serialized and deserialized."""
        # Test RenderConfig
        render_config = RenderConfig(width=300, height=400, background_color="#ff0000")
        render_dict = render_config.model_dump()
        new_render_config = RenderConfig(**render_dict)
        assert new_render_config.width == 300
        assert new_render_config.height == 400
        assert new_render_config.background_color == "#ff0000"

        # Test OutputConfig
        output_config = OutputConfig(format="svg", quality=80, dpi=300)
        output_dict = output_config.model_dump()
        new_output_config = OutputConfig(**output_dict)
        assert new_output_config.format == "svg"
        assert new_output_config.quality == 80
        assert new_output_config.dpi == 300

        # Test ParserConfig
        parser_config = ParserConfig(sanitize=False, strict=True)
        parser_dict = parser_config.model_dump()
        new_parser_config = ParserConfig(**parser_dict)
        assert new_parser_config.sanitize is False
        assert new_parser_config.strict is True
