"""
End-to-end tests for all supported molecular formats.

Tests the complete rendering pipeline for each format through the main API.
"""

import tempfile
from pathlib import Path

import pytest

from molecular_string_renderer import render_molecule, RenderConfig
from molecular_string_renderer.core import (
    validate_molecular_string,
    get_supported_formats,
)
from tests.conftest import SAMPLE_MOLECULES


class TestEndToEndRendering:
    """Test complete rendering pipeline for all supported formats."""

    def test_smiles_rendering(self):
        """Test end-to-end SMILES rendering."""
        for name, smiles in SAMPLE_MOLECULES["smiles"].items():
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"{name}.png"

                image = render_molecule(
                    molecular_string=smiles,
                    format_type="smiles",
                    output_format="png",
                    output_path=output_path,
                )

                # Verify image was created
                assert image is not None, f"No image returned for {name}"
                assert output_path.exists(), f"Output file not created for {name}"
                assert output_path.stat().st_size > 0, f"Empty output file for {name}"

    def test_inchi_rendering(self):
        """Test end-to-end InChI rendering."""
        for name, inchi in SAMPLE_MOLECULES["inchi"].items():
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"{name}_inchi.png"

                image = render_molecule(
                    molecular_string=inchi,
                    format_type="inchi",
                    output_format="png",
                    output_path=output_path,
                )

                # Verify image was created
                assert image is not None, f"No image returned for {name}"
                assert output_path.exists(), f"Output file not created for {name}"
                assert output_path.stat().st_size > 0, f"Empty output file for {name}"

    def test_selfies_rendering(self):
        """Test end-to-end SELFIES rendering."""
        for name, selfies in SAMPLE_MOLECULES["selfies"].items():
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"{name}_selfies.png"

                image = render_molecule(
                    molecular_string=selfies,
                    format_type="selfies",
                    output_format="png",
                    output_path=output_path,
                )

                # Verify image was created
                assert image is not None, f"No image returned for {name}"
                assert output_path.exists(), f"Output file not created for {name}"
                assert output_path.stat().st_size > 0, f"Empty output file for {name}"

    def test_mol_block_rendering(self):
        """Test end-to-end MOL block rendering."""
        for name, mol_block in SAMPLE_MOLECULES["mol_block"].items():
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"{name}_mol.png"

                image = render_molecule(
                    molecular_string=mol_block,
                    format_type="mol",
                    output_format="png",
                    output_path=output_path,
                )

                # Verify image was created
                assert image is not None, f"No image returned for {name}"
                assert output_path.exists(), f"Output file not created for {name}"
                assert output_path.stat().st_size > 0, f"Empty output file for {name}"

    def test_multiple_output_formats(self):
        """Test rendering to different output formats for each molecular format."""
        output_formats = ["png", "svg", "jpg"]

        test_cases = [
            ("smiles", SAMPLE_MOLECULES["smiles"]["ethanol"]),
            ("inchi", SAMPLE_MOLECULES["inchi"]["ethanol"]),
            ("selfies", SAMPLE_MOLECULES["selfies"]["ethanol"]),
            ("mol", SAMPLE_MOLECULES["mol_block"]["ethanol"]),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for format_type, molecular_string in test_cases:
                for output_format in output_formats:
                    output_path = (
                        Path(temp_dir) / f"ethanol_{format_type}.{output_format}"
                    )

                    image = render_molecule(
                        molecular_string=molecular_string,
                        format_type=format_type,
                        output_format=output_format,
                        output_path=output_path,
                    )

                    assert image is not None, (
                        f"No image for {format_type} -> {output_format}"
                    )
                    assert output_path.exists(), (
                        f"No file for {format_type} -> {output_format}"
                    )
                    assert output_path.stat().st_size > 0, (
                        f"Empty file for {format_type} -> {output_format}"
                    )

    def test_custom_render_config(self):
        """Test rendering with custom configuration for all formats."""
        config = RenderConfig(
            width=300,
            height=300,
            background_color="lightblue",
            show_hydrogen=True,
        )

        test_cases = [
            ("smiles", SAMPLE_MOLECULES["smiles"]["benzene"]),
            ("inchi", SAMPLE_MOLECULES["inchi"]["benzene"]),
            ("selfies", SAMPLE_MOLECULES["selfies"]["benzene"]),
            (
                "mol",
                SAMPLE_MOLECULES["mol_block"]["ethanol"],
            ),  # Use ethanol for MOL as benzene not available
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for format_type, molecular_string in test_cases:
                output_path = Path(temp_dir) / f"custom_{format_type}.png"

                image = render_molecule(
                    molecular_string=molecular_string,
                    format_type=format_type,
                    output_format="png",
                    output_path=output_path,
                    render_config=config,
                )

                assert image is not None
                assert image.size == (300, 300), f"Wrong image size for {format_type}"
                assert output_path.exists()

    def test_format_aliases(self):
        """Test that format aliases work correctly."""
        alias_tests = [
            ("smi", SAMPLE_MOLECULES["smiles"]["ethanol"]),
            ("sdf", SAMPLE_MOLECULES["mol_block"]["ethanol"]),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for format_alias, molecular_string in alias_tests:
                output_path = Path(temp_dir) / f"alias_{format_alias}.png"

                image = render_molecule(
                    molecular_string=molecular_string,
                    format_type=format_alias,
                    output_format="png",
                    output_path=output_path,
                )

                assert image is not None
                assert output_path.exists()

    def test_rendering_without_output_path(self):
        """Test rendering without specifying output path (memory only)."""
        test_cases = [
            ("smiles", SAMPLE_MOLECULES["smiles"]["ethanol"]),
            ("inchi", SAMPLE_MOLECULES["inchi"]["ethanol"]),
            ("selfies", SAMPLE_MOLECULES["selfies"]["ethanol"]),
            ("mol", SAMPLE_MOLECULES["mol_block"]["ethanol"]),
        ]

        for format_type, molecular_string in test_cases:
            image = render_molecule(
                molecular_string=molecular_string,
                format_type=format_type,
                output_format="png",
                auto_filename=False,  # Don't auto-save
            )

            assert image is not None, f"No image returned for {format_type}"
            assert hasattr(image, "size"), f"Invalid image object for {format_type}"


class TestMolecularValidation:
    """Test validation functionality for all supported formats."""

    def test_valid_molecular_strings(self):
        """Test validation of valid molecular strings for all formats."""
        test_cases = [
            ("smiles", SAMPLE_MOLECULES["smiles"]),
            ("inchi", SAMPLE_MOLECULES["inchi"]),
            ("selfies", SAMPLE_MOLECULES["selfies"]),
            ("mol", SAMPLE_MOLECULES["mol_block"]),
        ]

        for format_type, molecules in test_cases:
            for name, molecular_string in molecules.items():
                is_valid = validate_molecular_string(molecular_string, format_type)
                assert is_valid, (
                    f"Failed to validate {format_type} {name}: {molecular_string[:50]}..."
                )

    def test_invalid_molecular_strings(self):
        """Test validation of invalid molecular strings for all formats."""
        test_cases = [
            ("smiles", SAMPLE_MOLECULES["invalid_smiles"]),
            ("inchi", SAMPLE_MOLECULES["invalid_inchi"]),
            ("selfies", SAMPLE_MOLECULES["invalid_selfies"]),
            ("mol", SAMPLE_MOLECULES["invalid_mol"]),
        ]

        for format_type, invalid_molecules in test_cases:
            for invalid_string in invalid_molecules:
                is_valid = validate_molecular_string(invalid_string, format_type)
                assert not is_valid, (
                    f"Incorrectly validated invalid {format_type}: {invalid_string}"
                )

    def test_validation_with_aliases(self):
        """Test validation works with format aliases."""
        # Test SMILES aliases
        assert validate_molecular_string(SAMPLE_MOLECULES["smiles"]["ethanol"], "smi")
        assert validate_molecular_string(
            SAMPLE_MOLECULES["smiles"]["ethanol"], "SMILES"
        )

        # Test MOL aliases
        assert validate_molecular_string(
            SAMPLE_MOLECULES["mol_block"]["ethanol"], "sdf"
        )
        assert validate_molecular_string(
            SAMPLE_MOLECULES["mol_block"]["ethanol"], "MOL"
        )

    def test_validation_error_handling(self):
        """Test validation handles errors gracefully."""
        # Test with unsupported format
        is_valid = validate_molecular_string("CCO", "unsupported_format")
        assert not is_valid

        # Test with None or empty inputs
        is_valid = validate_molecular_string("", "smiles")
        assert not is_valid


class TestSupportedFormats:
    """Test the get_supported_formats function."""

    def test_supported_formats_structure(self):
        """Test that get_supported_formats returns expected structure."""
        formats = get_supported_formats()

        # Check required keys exist
        assert "input_formats" in formats
        assert "output_formats" in formats
        assert "renderer_types" in formats

        # Check that each section is a dictionary
        assert isinstance(formats["input_formats"], dict)
        assert isinstance(formats["output_formats"], dict)
        assert isinstance(formats["renderer_types"], dict)

    def test_input_formats_completeness(self):
        """Test that all supported input formats are listed."""
        formats = get_supported_formats()
        input_formats = formats["input_formats"]

        # Check that all formats from parsers are included
        expected_formats = ["smiles", "smi", "inchi", "mol", "sdf", "selfies"]

        for fmt in expected_formats:
            assert fmt in input_formats, f"Missing input format: {fmt}"

    def test_output_formats_completeness(self):
        """Test that all supported output formats are listed."""
        formats = get_supported_formats()
        output_formats = formats["output_formats"]

        # Check basic output formats
        expected_formats = ["png", "svg", "jpg"]

        for fmt in expected_formats:
            assert fmt in output_formats, f"Missing output format: {fmt}"

    def test_format_descriptions(self):
        """Test that format descriptions are informative."""
        formats = get_supported_formats()

        # Check that descriptions are non-empty strings
        for section in ["input_formats", "output_formats", "renderer_types"]:
            for fmt, description in formats[section].items():
                assert isinstance(description, str), f"Non-string description for {fmt}"
                assert len(description) > 0, f"Empty description for {fmt}"
                assert len(description) > 10, (
                    f"Too short description for {fmt}: {description}"
                )


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_format_rendering(self):
        """Test that invalid format types raise appropriate errors."""
        with pytest.raises(ValueError, match="Unsupported format"):
            render_molecule("CCO", format_type="unsupported", output_format="png")

    def test_invalid_molecular_string_rendering(self):
        """Test that invalid molecular strings raise appropriate errors."""
        test_cases = [
            ("smiles", "INVALID_SMILES"),
            ("inchi", "NotAnInChI"),
            ("selfies", "[INVALID"),
            ("mol", "Not a MOL block"),
        ]

        for format_type, invalid_string in test_cases:
            with pytest.raises(ValueError):
                render_molecule(
                    invalid_string, format_type=format_type, output_format="png"
                )

    def test_empty_string_handling(self):
        """Test that empty strings are handled appropriately."""
        formats_to_test = ["smiles", "inchi", "selfies", "mol"]

        for format_type in formats_to_test:
            with pytest.raises(ValueError, match="cannot be empty"):
                render_molecule("", format_type=format_type, output_format="png")


if __name__ == "__main__":
    pytest.main([__file__])
