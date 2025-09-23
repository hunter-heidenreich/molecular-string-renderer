"""
Tests for the validation module.

Tests all validation functions including molecular string validation,
format type validation, grid parameters, output paths, and configuration
compatibility validation.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.exceptions import ConfigurationError, ValidationError
from molecular_string_renderer.validation import (
    validate_configuration_compatibility,
    validate_format_type,
    validate_grid_parameters,
    validate_molecular_string,
    validate_output_path,
)


class TestValidateMolecularString:
    """Test validate_molecular_string function."""

    def test_valid_molecular_strings(self):
        """Test validation with valid molecular strings."""
        valid_cases = [
            ("CCO", "smiles"),
            ("c1ccccc1", "smiles"),
            ("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "inchi"),
            ("[C][C][O]", "selfies"),
            ("C" * 100, "smiles"),  # Long but reasonable
        ]

        for molecular_string, format_type in valid_cases:
            # Should not raise any exception
            validate_molecular_string(molecular_string, format_type)

    def test_empty_molecular_strings(self):
        """Test validation with empty molecular strings."""
        empty_cases = [
            ("", "smiles"),
            ("   ", "smiles"),
            ("\t\n  ", "inchi"),
        ]

        for molecular_string, format_type in empty_cases:
            with pytest.raises(
                ValidationError, match=f"Empty {format_type.upper()} string"
            ):
                validate_molecular_string(molecular_string, format_type)

    def test_non_string_types(self):
        """Test validation with non-string molecular strings."""
        invalid_types = [
            (None, "smiles"),
            (123, "smiles"),
            ([], "inchi"),
            ({}, "selfies"),
            (True, "smiles"),
        ]

        for molecular_string, format_type in invalid_types:
            with pytest.raises(ValidationError, match="must be a string"):
                validate_molecular_string(molecular_string, format_type)

    def test_excessively_long_strings(self):
        """Test validation with excessively long molecular strings."""
        long_string = "C" * 10001  # Over the 10,000 character limit

        with pytest.raises(
            ValidationError, match="too long.*Maximum supported length is 10,000"
        ):
            validate_molecular_string(long_string, "smiles")

    def test_boundary_length_strings(self):
        """Test validation at boundary conditions."""
        # Exactly at the limit should pass
        boundary_string = "C" * 10000
        validate_molecular_string(boundary_string, "smiles")

        # Just over the limit should fail
        over_limit_string = "C" * 10001
        with pytest.raises(ValidationError, match="too long"):
            validate_molecular_string(over_limit_string, "smiles")


class TestValidateFormatType:
    """Test validate_format_type function."""

    def test_valid_formats(self):
        """Test validation with valid format types."""
        valid_formats = {"smiles", "inchi", "selfies", "mol"}

        test_cases = [
            ("smiles", "smiles"),
            ("SMILES", "smiles"),  # Case normalization
            ("InChI", "inchi"),
            ("SELFIES", "selfies"),
            ("  mol  ", "mol"),  # Whitespace handling
        ]

        for input_format, expected in test_cases:
            result = validate_format_type(input_format, valid_formats)
            assert result == expected

    def test_invalid_formats(self):
        """Test validation with invalid format types."""
        valid_formats = {"smiles", "inchi", "selfies"}
        invalid_formats = ["xyz", "pdb", "cif", "invalid"]

        for invalid_format in invalid_formats:
            with pytest.raises(
                ValidationError, match=f"Unsupported format type: '{invalid_format}'"
            ):
                validate_format_type(invalid_format, valid_formats)

    def test_non_string_format_types(self):
        """Test validation with non-string format types."""
        valid_formats = {"smiles", "inchi"}
        invalid_types = [None, 123, [], {}, True]

        for invalid_type in invalid_types:
            with pytest.raises(ValidationError, match="must be a string"):
                validate_format_type(invalid_type, valid_formats)

    def test_case_normalization(self):
        """Test that format types are properly normalized to lowercase."""
        valid_formats = {"smiles", "inchi"}

        test_cases = [
            "SMILES",
            "SmIlEs",
            "INCHI",
            "InChI",
        ]

        for format_type in test_cases:
            result = validate_format_type(format_type, valid_formats)
            assert result == format_type.lower()

    def test_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        valid_formats = {"smiles"}

        result = validate_format_type("  smiles  ", valid_formats)
        assert result == "smiles"

    def test_error_message_includes_valid_formats(self):
        """Test that error messages include list of valid formats."""
        valid_formats = {"smiles", "inchi", "selfies"}

        with pytest.raises(
            ValidationError,
            match="Supported formats: \\['inchi', 'selfies', 'smiles'\\]",
        ):
            validate_format_type("invalid", valid_formats)


class TestValidateGridParameters:
    """Test validate_grid_parameters function."""

    def test_valid_grid_parameters(self):
        """Test validation with valid grid parameters."""
        valid_cases = [
            (["CCO", "c1ccccc1"], 2, (200, 200)),
            (["C"], 1, (100, 100)),
            (["CCO"] * 10, 5, (150, 150)),
            (["c1ccccc1"] * 50, 10, (300, 300)),
        ]

        for molecular_strings, mols_per_row, mol_size in valid_cases:
            # Should not raise any exception
            validate_grid_parameters(molecular_strings, mols_per_row, mol_size)

    def test_empty_molecular_strings_list(self):
        """Test validation with empty molecular strings list."""
        with pytest.raises(ValidationError, match="Cannot render empty molecule list"):
            validate_grid_parameters([], 2, (200, 200))

    def test_non_list_molecular_strings(self):
        """Test validation with non-list molecular strings."""
        invalid_types = ["not_a_list", 123, None, {}]

        for invalid_type in invalid_types:
            with pytest.raises(ValidationError, match="must be a list"):
                validate_grid_parameters(invalid_type, 2, (200, 200))

    def test_too_many_molecules(self):
        """Test validation with too many molecules."""
        too_many_molecules = ["CCO"] * 101  # Over the 100 molecule limit

        with pytest.raises(
            ValidationError, match="Too many molecules.*Maximum supported is 100"
        ):
            validate_grid_parameters(too_many_molecules, 10, (200, 200))

    def test_boundary_molecule_count(self):
        """Test validation at molecule count boundaries."""
        # Exactly 100 should pass
        exactly_100 = ["CCO"] * 100
        validate_grid_parameters(exactly_100, 10, (200, 200))

        # 101 should fail
        over_100 = ["CCO"] * 101
        with pytest.raises(ValidationError, match="Too many molecules"):
            validate_grid_parameters(over_100, 10, (200, 200))

    def test_invalid_mols_per_row(self):
        """Test validation with invalid mols_per_row values."""
        invalid_values = [0, -1, -10, 1.5, "2", None]

        for invalid_value in invalid_values:
            with pytest.raises(ValidationError, match="must be a positive integer"):
                validate_grid_parameters(["CCO"], invalid_value, (200, 200))

    def test_valid_mols_per_row(self):
        """Test validation with valid mols_per_row values."""
        valid_values = [1, 2, 5, 10, 100]

        for valid_value in valid_values:
            # Should not raise any exception
            validate_grid_parameters(["CCO"], valid_value, (200, 200))

    def test_invalid_mol_size_structure(self):
        """Test validation with invalid mol_size structure."""
        invalid_sizes = [
            (200,),  # Too few elements
            (200, 200, 200),  # Too many elements
            [200, 200],  # List instead of tuple
            "200x200",  # String
            None,
            200,  # Single integer
        ]

        for invalid_size in invalid_sizes:
            with pytest.raises(
                ValidationError, match="must be a tuple of \\(width, height\\)"
            ):
                validate_grid_parameters(["CCO"], 2, invalid_size)

    def test_invalid_mol_size_values(self):
        """Test validation with invalid mol_size values."""
        invalid_sizes = [
            (49, 200),  # Width too small
            (200, 49),  # Height too small
            (1001, 200),  # Width too large
            (200, 1001),  # Height too large
            (200.5, 200),  # Non-integer width
            (200, "200"),  # Non-integer height
        ]

        for invalid_size in invalid_sizes:
            with pytest.raises(ValidationError):
                validate_grid_parameters(["CCO"], 2, invalid_size)

    def test_boundary_mol_sizes(self):
        """Test validation at mol_size boundaries."""
        # Minimum valid size
        validate_grid_parameters(["CCO"], 2, (50, 50))

        # Maximum valid size
        validate_grid_parameters(["CCO"], 2, (1000, 1000))

        # Just below minimum
        with pytest.raises(ValidationError, match="must be between 50 and 1000 pixels"):
            validate_grid_parameters(["CCO"], 2, (49, 50))

        # Just above maximum
        with pytest.raises(ValidationError, match="must be between 50 and 1000 pixels"):
            validate_grid_parameters(["CCO"], 2, (1001, 1000))


class TestValidateOutputPath:
    """Test validate_output_path function."""

    def test_none_output_path(self):
        """Test validation with None output path."""
        result = validate_output_path(None)
        assert result is None

    def test_valid_string_paths(self):
        """Test validation with valid string paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_paths = [
                str(Path(temp_dir) / "output.png"),
                str(Path(temp_dir) / "subdir" / "output.png"),  # Non-existent subdir
                str(Path(temp_dir) / "test.svg"),
            ]

            for path_str in valid_paths:
                result = validate_output_path(path_str)
                assert isinstance(result, Path)
                assert str(result) == path_str

    def test_valid_path_objects(self):
        """Test validation with Path objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_obj = Path(temp_dir) / "output.png"
            result = validate_output_path(path_obj)
            assert isinstance(result, Path)
            assert result == path_obj

    def test_empty_string_path(self):
        """Test validation with empty string path."""
        with pytest.raises(ValidationError, match="cannot be empty string"):
            validate_output_path("")

        with pytest.raises(ValidationError, match="cannot be empty string"):
            validate_output_path("   ")

    def test_invalid_path_types(self):
        """Test validation with invalid path types."""
        invalid_types = [123, [], {}, True, 45.6]

        for invalid_type in invalid_types:
            with pytest.raises(ValidationError, match="must be string or Path"):
                validate_output_path(invalid_type)

    def test_directory_creation(self):
        """Test that parent directories are created when needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "level1" / "level2" / "output.png"

            # Directory should not exist initially
            assert not nested_path.parent.exists()

            result = validate_output_path(nested_path)

            # Directory should be created
            assert nested_path.parent.exists()
            assert result == nested_path

    @patch("pathlib.Path.mkdir")
    def test_directory_creation_permission_error(self, mock_mkdir):
        """Test handling of permission errors during directory creation."""
        mock_mkdir.side_effect = PermissionError("Permission denied")

        with pytest.raises(
            ValidationError, match="Cannot create output directory.*Permission denied"
        ):
            validate_output_path("/root/restricted/output.png")

    @patch("pathlib.Path.mkdir")
    def test_directory_creation_os_error(self, mock_mkdir):
        """Test handling of OS errors during directory creation."""
        mock_mkdir.side_effect = OSError("Disk full")

        with pytest.raises(
            ValidationError, match="Cannot create output directory.*Disk full"
        ):
            validate_output_path("/path/to/output.png")


class TestValidateConfigurationCompatibility:
    """Test validate_configuration_compatibility function."""

    def test_compatible_configurations(self):
        """Test validation with compatible configurations."""
        render_config = RenderConfig(width=400, height=300, background_color="white")
        parser_config = ParserConfig()
        output_config = OutputConfig(format="png")

        # Should not raise any exception
        validate_configuration_compatibility(
            render_config, parser_config, output_config
        )

    def test_dimensions_too_small(self):
        """Test validation with dimensions that are too small."""
        # Create valid config and then manually set dimensions too small for our validation logic
        render_config = RenderConfig(width=100, height=100)  # Valid by Pydantic
        parser_config = ParserConfig()
        output_config = OutputConfig()

        # Manually set a value too small for our validation logic
        render_config.width = 40  # Bypass Pydantic after creation

        with pytest.raises(
            ConfigurationError, match="Render dimensions too small.*Minimum is 50x50"
        ):
            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )

    def test_dimensions_too_large(self):
        """Test validation with dimensions that are too large."""
        # Create valid config and then manually set oversized dimensions
        render_config = RenderConfig(width=2000, height=2000)  # Valid by Pydantic
        parser_config = ParserConfig()
        output_config = OutputConfig()

        # Manually set a value too large for our validation logic
        render_config.width = 6000  # Bypass Pydantic validation

        with pytest.raises(
            ConfigurationError,
            match="Render dimensions too large.*Maximum is 5000x5000",
        ):
            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )

    def test_aspect_ratio_warnings(self, caplog):
        """Test that extreme aspect ratios generate warnings."""
        with caplog.at_level(logging.WARNING):
            # Very wide aspect ratio
            render_config = RenderConfig(width=2000, height=100)  # 20:1 ratio
            parser_config = ParserConfig()
            output_config = OutputConfig()

            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )

            assert "Extreme aspect ratio" in caplog.text
            assert "distorted molecules" in caplog.text

    def test_invalid_hex_color(self):
        """Test validation with invalid hex colors."""
        # Create valid config and then manually set invalid hex color
        render_config = RenderConfig(background_color="#FFFFFF")  # Valid hex
        parser_config = ParserConfig()
        output_config = OutputConfig()

        # Manually set invalid hex to bypass Pydantic validation
        render_config.background_color = "#GGGGGG"

        with pytest.raises(ConfigurationError, match="Invalid hex color"):
            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )

    def test_valid_hex_colors(self):
        """Test validation with valid hex colors."""
        valid_colors = ["#FFFFFF", "#000000", "#FF5733", "#123ABC"]

        for color in valid_colors:
            render_config = RenderConfig(background_color=color)
            parser_config = ParserConfig()
            output_config = OutputConfig()

            # Should not raise any exception
            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )

    def test_jpeg_transparency_conflict(self):
        """Test JPEG format with transparent background raises error."""
        render_config = RenderConfig(background_color="transparent")
        parser_config = ParserConfig()
        output_config = OutputConfig(format="jpeg")

        with pytest.raises(
            ConfigurationError,
            match="JPEG format does not support transparent backgrounds",
        ):
            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )

    def test_supported_transparency_formats(self):
        """Test that transparency works with supported formats."""
        render_config = RenderConfig(background_color="transparent")
        parser_config = ParserConfig()

        supported_formats = ["png", "webp", "tiff"]

        for fmt in supported_formats:
            output_config = OutputConfig(format=fmt)
            # Should not raise any exception
            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )

    def test_quality_validation_for_jpeg_webp(self):
        """Test quality validation for JPEG and WebP formats."""
        render_config = RenderConfig()
        parser_config = ParserConfig()

        # Valid quality should pass
        for fmt in ["jpeg", "webp"]:
            output_config = OutputConfig(format=fmt, quality=85)
            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )

        # Test invalid quality by manually setting it after creation
        for fmt in ["jpeg", "webp"]:
            output_config = OutputConfig(format=fmt, quality=90)  # Valid initially
            output_config.quality = 101  # Manually set invalid value
            with pytest.raises(
                ConfigurationError, match="Quality.*invalid.*Must be 1-100"
            ):
                validate_configuration_compatibility(
                    render_config, parser_config, output_config
                )

    def test_dpi_warnings(self, caplog):
        """Test DPI warnings for low and high values."""
        render_config = RenderConfig()
        parser_config = ParserConfig()

        # Low DPI warning - create valid config and manually set low DPI
        with caplog.at_level(logging.WARNING):
            output_config = OutputConfig(dpi=72)  # Valid minimum
            output_config.dpi = 50  # Manually set below minimum for warning
            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )
            assert "Low DPI" in caplog.text
            assert "poor print quality" in caplog.text

        caplog.clear()

        # High DPI warning - create valid config and manually set high DPI
        with caplog.at_level(logging.WARNING):
            output_config = OutputConfig(dpi=600)  # Valid maximum
            output_config.dpi = 650  # Manually set above maximum for warning
            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )
            assert "Very high DPI" in caplog.text
            assert "large files" in caplog.text

    def test_svg_line_width_validation(self):
        """Test SVG line width multiplier validation."""
        render_config = RenderConfig()
        parser_config = ParserConfig()

        # Valid line width
        output_config = OutputConfig(format="svg", svg_line_width_mult=1.5)
        validate_configuration_compatibility(
            render_config, parser_config, output_config
        )

        # Test invalid line width by manually setting after creation
        output_config = OutputConfig(
            format="svg", svg_line_width_mult=0.5
        )  # Valid initially
        output_config.svg_line_width_mult = 0  # Manually set invalid value
        with pytest.raises(
            ConfigurationError, match="SVG line width multiplier must be positive"
        ):
            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )

        # Test negative line width
        output_config = OutputConfig(
            format="svg", svg_line_width_mult=1.0
        )  # Valid initially
        output_config.svg_line_width_mult = -1.0  # Manually set negative value
        with pytest.raises(
            ConfigurationError, match="SVG line width multiplier must be positive"
        ):
            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )

    def test_hydrogen_coordination_info(self, caplog):
        """Test hydrogen display coordination logging."""
        with caplog.at_level(logging.INFO):
            render_config = RenderConfig(show_hydrogen=True)
            parser_config = ParserConfig(show_hydrogen=False)  # Conflict
            output_config = OutputConfig()

            validate_configuration_compatibility(
                render_config, parser_config, output_config
            )

            assert "Auto-coordinating" in caplog.text
            assert (
                "render_config.show_hydrogen=True requires parser_config.show_hydrogen=True"
                in caplog.text
            )

    def test_non_hex_color_formats(self):
        """Test that non-hex color formats don't trigger hex validation."""
        render_config = RenderConfig(background_color="white")  # Named color
        parser_config = ParserConfig()
        output_config = OutputConfig()

        # Should not raise any exception for named colors
        validate_configuration_compatibility(
            render_config, parser_config, output_config
        )
