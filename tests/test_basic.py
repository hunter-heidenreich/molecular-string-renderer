"""
Test suite for molecular string renderer.

Basic tests to ensure core functionality works correctly.
"""

import tempfile
from pathlib import Path

import pytest

from molecular_string_renderer import RenderConfig, render_molecule
from molecular_string_renderer.core import (
    get_supported_formats,
    validate_molecular_string,
)
from molecular_string_renderer.parsers import SMILESParser, SELFIESParser, get_parser


class TestMolecularRendering:
    """Test core molecular rendering functionality."""

    def test_simple_smiles_rendering(self):
        """Test rendering a simple SMILES string."""
        smiles = "CCO"  # Ethanol

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "ethanol.png"

            image = render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="png",
                output_path=output_path,
            )

            # Check that image was created
            assert image is not None
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES strings."""
        invalid_smiles = "INVALID_SMILES_STRING"

        with pytest.raises(ValueError, match="Invalid SMILES"):
            render_molecule(
                molecular_string=invalid_smiles,
                format_type="smiles",
                output_format="png",
            )

    def test_custom_config(self):
        """Test rendering with custom configuration."""
        smiles = "C1=CC=CC=C1"  # Benzene

        config = RenderConfig(
            width=300, height=300, background_color="lightblue", show_hydrogen=True
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "benzene.png"

            image = render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="png",
                output_path=output_path,
                render_config=config,
            )

            assert image is not None
            assert image.size == (300, 300)
            assert output_path.exists()

    def test_different_output_formats(self):
        """Test different output formats."""
        smiles = "CC(=O)O"  # Acetic acid

        formats = ["png", "svg", "jpg"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in formats:
                output_path = Path(temp_dir) / f"molecule.{fmt}"

                image = render_molecule(
                    molecular_string=smiles,
                    format_type="smiles",
                    output_format=fmt,
                    output_path=output_path,
                )

                assert image is not None
                assert output_path.exists()


class TestParsers:
    """Test molecular parsers."""

    def test_smiles_parser(self):
        """Test SMILES parser functionality."""
        parser = SMILESParser()

        # Valid SMILES
        valid_smiles = ["CCO", "C1=CC=CC=C1", "CC(=O)O"]
        for smiles in valid_smiles:
            mol = parser.parse(smiles)
            assert mol is not None
            assert parser.validate(smiles) is True

        # Invalid SMILES
        invalid_smiles = ["INVALID", "C1=CC=CC=C"]
        for smiles in invalid_smiles:
            with pytest.raises(ValueError):
                parser.parse(smiles)
            assert parser.validate(smiles) is False

        # Empty string is a special case - should be invalid
        assert parser.validate("") is False

    def test_selfies_parser(self):
        """Test SELFIES parser functionality."""
        parser = SELFIESParser()

        # Valid SELFIES
        valid_selfies = [
            "[C][C][O]",  # Ethanol
            "[C][Branch1][C][C][C][O]",  # Propanol
            "[C][=C][C][=C][C][=C][Ring1][=Branch1]",  # Benzene
        ]
        for selfies in valid_selfies:
            mol = parser.parse(selfies)
            assert mol is not None
            assert parser.validate(selfies) is True

        # Invalid SELFIES (malformed syntax)
        invalid_selfies = [
            "[C][C",
            "[X][Y]",
            "[Z][Z][Z]",
        ]  # malformed bracket, invalid symbols
        for selfies in invalid_selfies:
            with pytest.raises(ValueError):
                parser.parse(selfies)
            assert parser.validate(selfies) is False

        # Empty string is a special case - should be invalid
        assert parser.validate("") is False

    def test_selfies_rendering(self):
        """Test rendering SELFIES strings."""
        selfies = "[C][C][O]"  # Ethanol in SELFIES format

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "ethanol_selfies.png"

            image = render_molecule(
                molecular_string=selfies,
                format_type="selfies",
                output_format="png",
                output_path=output_path,
            )

            # Check that image was created
            assert image is not None
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_parser_factory(self):
        """Test parser factory function."""
        # Test supported formats
        supported_formats = ["smiles", "smi", "selfies"]

        for fmt in supported_formats:
            parser = get_parser(fmt)
            assert parser is not None

        # Test unsupported format
        with pytest.raises(ValueError, match="Unsupported format"):
            get_parser("unsupported_format")


class TestValidation:
    """Test validation functionality."""

    def test_molecular_string_validation(self):
        """Test molecular string validation."""
        # Valid SMILES
        assert validate_molecular_string("CCO", "smiles") is True
        assert validate_molecular_string("C1=CC=CC=C1", "smiles") is True

        # Invalid SMILES
        assert validate_molecular_string("INVALID", "smiles") is False
        assert validate_molecular_string("", "smiles") is False

        # Valid SELFIES
        assert validate_molecular_string("[C][C][O]", "selfies") is True
        assert validate_molecular_string("[C][Branch1][C][C][C][O]", "selfies") is True

        # Invalid SELFIES
        assert validate_molecular_string("[C][C", "selfies") is False
        assert validate_molecular_string("[X][Y]", "selfies") is False
        assert validate_molecular_string("", "selfies") is False

    def test_supported_formats(self):
        """Test getting supported formats."""
        formats = get_supported_formats()

        assert "input_formats" in formats
        assert "output_formats" in formats
        assert "renderer_types" in formats

        # Check some expected formats
        assert "smiles" in formats["input_formats"]
        assert "selfies" in formats["input_formats"]
        assert "png" in formats["output_formats"]
        assert "2d" in formats["renderer_types"]


class TestConfiguration:
    """Test configuration system."""

    def test_render_config_defaults(self):
        """Test RenderConfig default values."""
        config = RenderConfig()

        assert config.width == 500
        assert config.height == 500
        assert config.background_color == "white"
        assert config.antialias is True
        assert config.show_hydrogen is False

    def test_render_config_validation(self):
        """Test RenderConfig validation."""
        # Valid config
        config = RenderConfig(width=300, height=400, background_color="#ff0000")
        assert config.width == 300
        assert config.height == 400
        assert config.background_color == "#ff0000"

        # Invalid size (too small)
        with pytest.raises(ValueError):
            RenderConfig(width=50)  # Below minimum of 100

    def test_config_to_rdkit_options(self):
        """Test conversion to RDKit options."""
        config = RenderConfig(
            bond_line_width=3.0, show_carbon=True, highlight_atoms=[0, 1, 2]
        )

        rdkit_options = config.to_rdkit_options()

        assert rdkit_options["bondLineWidth"] == 3.0
        assert rdkit_options["explicitMethyl"] is True
        assert rdkit_options["highlightAtoms"] == [0, 1, 2]


# Integration tests that don't require external dependencies
class TestIntegration:
    """Integration tests that work without dependencies."""

    def test_import_structure(self):
        """Test that the package structure can be imported."""
        try:
            import molecular_string_renderer

            assert hasattr(molecular_string_renderer, "__version__")
        except ImportError:
            pytest.skip("Package not installed in development mode")

    def test_cli_help(self):
        """Test that CLI help works."""
        try:
            from molecular_string_renderer.cli import create_parser

            parser = create_parser()

            # Should not raise an exception
            help_text = parser.format_help()
            assert "molecular string" in help_text.lower()

        except ImportError:
            pytest.skip("CLI module not available")


if __name__ == "__main__":
    pytest.main([__file__])
