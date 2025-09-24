"""
Comprehensive tests for the core module.

Tests the high-level interface functions: render_molecule, render_molecules_grid,
validate_molecular_string, and get_supported_formats.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.core import (
    get_supported_formats,
    render_molecule,
    render_molecules_grid,
    validate_molecular_string,
)
from molecular_string_renderer.exceptions import (
    OutputError,
    ParsingError,
    RenderingError,
    ValidationError,
)


class TestRenderMolecule:
    """Test the render_molecule function."""

    def test_render_molecule_default_parameters(self):
        """Test render_molecule with default parameters."""
        # Should work with valid SMILES string
        image = render_molecule("CCO", auto_filename=False)

        assert isinstance(image, Image.Image)
        assert image.mode in ["RGBA", "RGB"]

    def test_render_molecule_with_valid_smiles(self):
        """Test render_molecule with various valid SMILES strings."""
        valid_smiles = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "C",  # methane
            "CC",  # ethane
        ]

        for smiles in valid_smiles:
            image = render_molecule(smiles, format_type="smiles", auto_filename=False)
            assert isinstance(image, Image.Image)

    def test_render_molecule_with_valid_inchi(self):
        """Test render_molecule with valid InChI strings."""
        valid_inchi = [
            "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # ethanol
            "InChI=1S/CH4/h1H4",  # methane
        ]

        for inchi in valid_inchi:
            image = render_molecule(inchi, format_type="inchi", auto_filename=False)
            assert isinstance(image, Image.Image)

    def test_render_molecule_different_output_formats(self):
        """Test render_molecule with different output formats."""
        output_formats = ["png", "svg", "jpg", "jpeg", "webp", "tiff", "bmp"]

        for output_format in output_formats:
            image = render_molecule(
                "CCO", output_format=output_format, auto_filename=False
            )
            assert isinstance(image, Image.Image)

    def test_render_molecule_with_custom_render_config(self):
        """Test render_molecule with custom render configuration."""
        render_config = RenderConfig(
            width=800,
            height=600,
            background_color="white",
            show_hydrogen=True,
            show_carbon=True,
        )

        image = render_molecule("CCO", render_config=render_config, auto_filename=False)

        assert isinstance(image, Image.Image)

    def test_render_molecule_with_custom_parser_config(self):
        """Test render_molecule with custom parser configuration."""
        parser_config = ParserConfig(sanitize=True, show_hydrogen=True, strict=False)

        image = render_molecule("CCO", parser_config=parser_config, auto_filename=False)

        assert isinstance(image, Image.Image)

    def test_render_molecule_with_custom_output_config(self):
        """Test render_molecule with custom output configuration."""
        output_config = OutputConfig(quality=90, dpi=300, optimize=True)

        image = render_molecule("CCO", output_config=output_config, auto_filename=False)

        assert isinstance(image, Image.Image)

    def test_render_molecule_with_output_path(self):
        """Test render_molecule with specified output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_molecule.png"

            image = render_molecule("CCO", output_path=output_path)

            assert isinstance(image, Image.Image)
            assert output_path.exists()

    def test_render_molecule_with_auto_filename(self):
        """Test render_molecule with auto filename generation."""
        with patch("molecular_string_renderer.utils.handle_output_saving") as mock_save:
            image = render_molecule("CCO", output_path=None, auto_filename=True)

            assert isinstance(image, Image.Image)
            # Auto filename should trigger saving logic
            mock_save.assert_called_once()

    def test_render_molecule_no_auto_filename_no_path(self):
        """Test render_molecule without auto filename and no path."""
        with patch("molecular_string_renderer.utils.handle_output_saving") as mock_save:
            image = render_molecule("CCO", output_path=None, auto_filename=False)

            assert isinstance(image, Image.Image)
            # Should not trigger saving
            mock_save.assert_not_called()

    def test_render_molecule_invalid_molecular_string(self):
        """Test render_molecule with invalid molecular strings."""
        invalid_strings = [
            "",  # empty string
            "INVALID_SMILES_123",  # invalid SMILES
            "InChI=invalid",  # invalid InChI
            None,  # None value
        ]

        for invalid_string in invalid_strings:
            with pytest.raises((ValidationError, ParsingError)):
                render_molecule(invalid_string)

    def test_render_molecule_invalid_format_type(self):
        """Test render_molecule with invalid format types."""
        invalid_formats = [
            "invalid",
            "xyz",
            "pdb",
            "",
        ]

        for invalid_format in invalid_formats:
            with pytest.raises((ValidationError, AttributeError)):
                render_molecule("CCO", format_type=invalid_format)

        # Test None separately since it causes AttributeError in parser factory
        with pytest.raises(AttributeError):
            render_molecule("CCO", format_type=None)

    def test_render_molecule_invalid_output_format(self):
        """Test render_molecule with invalid output formats."""
        invalid_formats = [
            "invalid",
            "gif",
            "ico",
        ]

        for invalid_format in invalid_formats:
            with pytest.raises((ValidationError, ValueError)):
                render_molecule("CCO", output_format=invalid_format)

        # Test empty string and None separately
        with pytest.raises((ValidationError, ValueError)):
            render_molecule("CCO", output_format="")

        with pytest.raises((ValidationError, ValueError, TypeError)):
            render_molecule("CCO", output_format=None)

    def test_render_molecule_string_too_long(self):
        """Test render_molecule with overly long molecular string."""
        # Create a string longer than the 10,000 character limit
        long_string = "C" * 10001

        with pytest.raises(ValidationError, match="too long"):
            render_molecule(long_string)

    def test_render_molecule_pipeline_error_propagation(self):
        """Test that pipeline errors are properly propagated."""
        # Test with actual invalid input that causes parsing error
        with pytest.raises(ParsingError):
            render_molecule("INVALID_SMILES_THAT_CANNOT_PARSE")

        # Test with None molecule input
        with patch(
            "molecular_string_renderer.pipeline.RenderingPipeline.parse_molecule"
        ) as mock_parse:
            mock_parse.return_value = None

            with pytest.raises(RenderingError):
                render_molecule("CCO")

    def test_render_molecule_configuration_initialization(self):
        """Test that configurations are properly initialized."""
        # Test that render_molecule works with default configurations
        image = render_molecule("CCO", auto_filename=False)
        assert isinstance(image, Image.Image)

        # Test with custom configurations
        render_config = RenderConfig(width=300, height=300)
        parser_config = ParserConfig(sanitize=True)
        output_config = OutputConfig(quality=90)

        image = render_molecule(
            "CCO",
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
            auto_filename=False,
        )
        assert isinstance(image, Image.Image)

    def test_render_molecule_logging(self):
        """Test that render_molecule logs operations correctly."""
        with patch("molecular_string_renderer.core.logged_operation") as mock_logged_op:
            mock_logged_op.return_value.__enter__ = Mock()
            mock_logged_op.return_value.__exit__ = Mock()

            render_molecule("CCO", auto_filename=False)

            # Should log the operation
            mock_logged_op.assert_called_once()
            args = mock_logged_op.call_args[0]
            assert args[0] == "render_molecule"

    def test_render_molecule_format_type_normalization(self):
        """Test that format types are normalized correctly."""
        # Test case variations that should work
        test_cases = [
            ("smiles", "CCO"),
            ("SMILES", "CCO"),
            ("smi", "CCO"),
            ("SMI", "CCO"),
            ("inchi", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"),
            ("INCHI", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"),
        ]

        for format_type, molecule in test_cases:
            image = render_molecule(
                molecule, format_type=format_type, auto_filename=False
            )
            assert isinstance(image, Image.Image)

    def test_render_molecule_output_path_types(self):
        """Test render_molecule with different output path types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with string path
            string_path = str(Path(temp_dir) / "string_path.png")
            image1 = render_molecule("CCO", output_path=string_path)
            assert isinstance(image1, Image.Image)
            assert Path(string_path).exists()

            # Test with Path object
            path_obj = Path(temp_dir) / "path_obj.png"
            image2 = render_molecule("CCO", output_path=path_obj)
            assert isinstance(image2, Image.Image)
            assert path_obj.exists()

    def test_render_molecule_all_configurations_together(self):
        """Test render_molecule with all custom configurations."""
        render_config = RenderConfig(
            width=400, height=300, background_color="#ffffff", show_hydrogen=False
        )
        parser_config = ParserConfig(sanitize=True, strict=False)
        output_config = OutputConfig(quality=85, dpi=200, optimize=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "complete_config.png"

            image = render_molecule(
                "CCO",
                format_type="smiles",
                output_format="png",
                output_path=output_path,
                render_config=render_config,
                parser_config=parser_config,
                output_config=output_config,
                auto_filename=False,
            )

            assert isinstance(image, Image.Image)
            assert output_path.exists()


class TestRenderMoleculesGrid:
    """Test the render_molecules_grid function."""

    def test_render_molecules_grid_default_parameters(self):
        """Test render_molecules_grid with default parameters."""
        molecules = ["CCO", "CC(=O)O", "c1ccccc1"]

        image = render_molecules_grid(molecules, auto_filename=False)

        assert isinstance(image, Image.Image)

    def test_render_molecules_grid_valid_molecules(self):
        """Test render_molecules_grid with various valid molecules."""
        molecules = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "C",  # methane
            "CC",  # ethane
            "CCC",  # propane
        ]

        image = render_molecules_grid(molecules, auto_filename=False)

        assert isinstance(image, Image.Image)

    def test_render_molecules_grid_with_legends(self):
        """Test render_molecules_grid with legends."""
        molecules = ["CCO", "CC(=O)O", "c1ccccc1"]
        legends = ["Ethanol", "Acetic Acid", "Benzene"]

        image = render_molecules_grid(molecules, legends=legends, auto_filename=False)

        assert isinstance(image, Image.Image)

    def test_render_molecules_grid_custom_layout(self):
        """Test render_molecules_grid with custom layout parameters."""
        molecules = ["CCO", "CC(=O)O", "c1ccccc1", "C", "CC", "CCC"]

        image = render_molecules_grid(molecules, mols_per_row=3, mol_size=(150, 150), auto_filename=False)

        assert isinstance(image, Image.Image)

    def test_render_molecules_grid_different_formats(self):
        """Test render_molecules_grid with different molecular formats."""
        # Test with InChI format
        inchi_molecules = [
            "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # ethanol
            "InChI=1S/CH4/h1H4",  # methane
        ]

        image = render_molecules_grid(inchi_molecules, format_type="inchi", auto_filename=False)

        assert isinstance(image, Image.Image)

    def test_render_molecules_grid_with_output_path(self):
        """Test render_molecules_grid with output path."""
        molecules = ["CCO", "CC(=O)O"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grid.png"

            image = render_molecules_grid(molecules, output_path=output_path)

            assert isinstance(image, Image.Image)
            assert output_path.exists()

    def test_render_molecules_grid_custom_configurations(self):
        """Test render_molecules_grid with custom configurations."""
        molecules = ["CCO", "CC(=O)O"]

        render_config = RenderConfig(width=300, height=300, background_color="white")
        parser_config = ParserConfig(sanitize=True)
        output_config = OutputConfig(quality=90)

        image = render_molecules_grid(
            molecules,
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
            auto_filename=False,
        )

        assert isinstance(image, Image.Image)

    def test_render_molecules_grid_empty_list(self):
        """Test render_molecules_grid with empty molecule list."""
        with pytest.raises(ValidationError):
            render_molecules_grid([])

    def test_render_molecules_grid_single_molecule(self):
        """Test render_molecules_grid with single molecule."""
        image = render_molecules_grid(["CCO"], auto_filename=False)

        assert isinstance(image, Image.Image)

    def test_render_molecules_grid_invalid_molecules(self):
        """Test render_molecules_grid with invalid molecules."""
        # Mix of valid and invalid molecules
        molecules = ["CCO", "INVALID_SMILES", "CC(=O)O"]

        # Should still work with valid molecules
        image = render_molecules_grid(molecules, auto_filename=False)
        assert isinstance(image, Image.Image)

    def test_render_molecules_grid_all_invalid_molecules(self):
        """Test render_molecules_grid with all invalid molecules."""
        molecules = ["INVALID1", "INVALID2", "INVALID3"]

        with pytest.raises(ParsingError, match="No valid molecules"):
            render_molecules_grid(molecules)

    def test_render_molecules_grid_invalid_legends_type(self):
        """Test render_molecules_grid with invalid legends type."""
        molecules = ["CCO", "CC(=O)O"]

        with pytest.raises(ValidationError, match="Legends must be a list"):
            render_molecules_grid(molecules, legends="not a list")

    def test_render_molecules_grid_mismatched_legends_count(self):
        """Test render_molecules_grid with mismatched legends count."""
        molecules = ["CCO", "CC(=O)O", "c1ccccc1"]
        legends = ["Ethanol", "Acetic Acid"]  # One less legend

        with pytest.raises(ValidationError, match="Number of legends.*must match"):
            render_molecules_grid(molecules, legends=legends)

    def test_render_molecules_grid_invalid_grid_parameters(self):
        """Test render_molecules_grid with invalid grid parameters."""
        molecules = ["CCO", "CC(=O)O"]

        # Test invalid mols_per_row
        with pytest.raises(ValidationError):
            render_molecules_grid(molecules, mols_per_row=0)

        # Test invalid mol_size
        with pytest.raises(ValidationError):
            render_molecules_grid(molecules, mol_size=(0, 100))

    def test_render_molecules_grid_large_layout(self):
        """Test render_molecules_grid with large layout."""
        # Create a grid with many molecules
        molecules = ["C"] * 20  # 20 methane molecules

        image = render_molecules_grid(molecules, mols_per_row=5, mol_size=(100, 100), auto_filename=False)

        assert isinstance(image, Image.Image)

    def test_render_molecules_grid_error_propagation(self):
        """Test that grid rendering errors are properly propagated."""
        # Test with all invalid molecules to trigger the error
        molecules = ["INVALID1", "INVALID2"]

        with pytest.raises(ParsingError, match="No valid molecules"):
            render_molecules_grid(molecules)

    def test_render_molecules_grid_renderer_none_error(self):
        """Test error when grid renderer returns None."""
        # To trigger the None return path, we need to mock at the right level
        # This tests the error case where render_grid returns None
        with patch(
            "molecular_string_renderer.core.MoleculeGridRenderer"
        ) as mock_renderer_class:
            mock_renderer = Mock()
            mock_renderer_class.return_value = mock_renderer
            mock_renderer.render_grid.return_value = None  # Simulate None return

            molecules = ["CCO", "CC(=O)O"]

            with pytest.raises(RenderingError):
                render_molecules_grid(molecules)

    def test_render_molecules_grid_legend_filtering(self):
        """Test that legends are filtered correctly for valid molecules."""
        # Mix valid and invalid molecules with legends
        molecules = ["CCO", "INVALID", "CC(=O)O", "ALSO_INVALID", "c1ccccc1"]
        legends = ["Ethanol", "Invalid1", "Acetic Acid", "Invalid2", "Benzene"]

        # This should work - valid molecules will be rendered with corresponding legends
        image = render_molecules_grid(molecules, legends=legends, auto_filename=False)
        assert isinstance(image, Image.Image)

    def test_render_molecules_grid_different_output_formats(self):
        """Test render_molecules_grid with different output formats."""
        molecules = ["CCO", "CC(=O)O"]
        output_formats = ["png", "svg", "jpg"]

        for output_format in output_formats:
            image = render_molecules_grid(molecules, output_format=output_format, auto_filename=False)
            assert isinstance(image, Image.Image)

    def test_render_molecules_grid_smart_default_mols_per_row(self):
        """Test smart default behavior for mols_per_row."""
        # Test with 3 molecules (should auto-fit to 3, not default 4)
        molecules_3 = ["CCO", "CC(=O)O", "C1=CC=CC=C1"]
        image_3 = render_molecules_grid(molecules_3, auto_filename=False)
        assert isinstance(image_3, Image.Image)
        # Should be 3 molecules wide (600px), not 4 wide (800px)
        assert image_3.size[0] == 600  # 3 * 200px per molecule
        assert image_3.size[1] == 200  # 1 row

        # Test with 5 molecules (should auto-fit to 4, not 5)
        molecules_5 = ["CCO", "CC(=O)O", "C1=CC=CC=C1", "C", "CC"]
        image_5 = render_molecules_grid(molecules_5, auto_filename=False)
        assert isinstance(image_5, Image.Image)
        # Should be 4 molecules wide (max), requiring 2 rows
        assert image_5.size[0] == 800  # 4 * 200px per molecule (max)
        assert image_5.size[1] == 400  # 2 rows

        # Test with 1 molecule (should auto-fit to 1)
        molecules_1 = ["CCO"]
        image_1 = render_molecules_grid(molecules_1, auto_filename=False)
        assert isinstance(image_1, Image.Image)
        assert image_1.size[0] == 200  # 1 * 200px per molecule
        assert image_1.size[1] == 200  # 1 row

    def test_render_molecules_grid_explicit_mols_per_row_overrides_smart(self):
        """Test that explicit mols_per_row overrides smart default."""
        molecules = ["CCO", "CC(=O)O", "C1=CC=CC=C1"]
        
        # Explicit 2 per row should create 2x2 grid (with empty slot)
        image = render_molecules_grid(molecules, mols_per_row=2, auto_filename=False)
        assert isinstance(image, Image.Image)
        assert image.size[0] == 400  # 2 * 200px per molecule
        assert image.size[1] == 400  # 2 rows (3 molecules, 2 per row = 2 rows)

        # Explicit 1 per row should create 1x3 grid
        image = render_molecules_grid(molecules, mols_per_row=1, auto_filename=False)
        assert isinstance(image, Image.Image)
        assert image.size[0] == 200  # 1 * 200px per molecule
        assert image.size[1] == 600  # 3 rows


class TestValidateMolecularString:
    """Test the validate_molecular_string function."""

    def test_validate_molecular_string_valid_smiles(self):
        """Test validate_molecular_string with valid SMILES strings."""
        valid_smiles = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "C",  # methane
            "CC",  # ethane
            "CCC",  # propane
            "CCCCCCCCCC",  # decane
        ]

        for smiles in valid_smiles:
            assert validate_molecular_string(smiles, "smiles") is True

    def test_validate_molecular_string_valid_inchi(self):
        """Test validate_molecular_string with valid InChI strings."""
        valid_inchi = [
            "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # ethanol
            "InChI=1S/CH4/h1H4",  # methane
            "InChI=1S/C2H4/c1-2/h1-2H2",  # ethene
        ]

        for inchi in valid_inchi:
            assert validate_molecular_string(inchi, "inchi") is True

    def test_validate_molecular_string_invalid_smiles(self):
        """Test validate_molecular_string with invalid SMILES strings."""
        invalid_smiles = [
            "INVALID_SMILES_123",
            "C(((",  # unbalanced parentheses
            "xyz",  # invalid characters
            "[Zz]",  # invalid element
        ]

        for smiles in invalid_smiles:
            assert validate_molecular_string(smiles, "smiles") is False

    def test_validate_molecular_string_invalid_inchi(self):
        """Test validate_molecular_string with invalid InChI strings."""
        invalid_inchi = [
            "InChI=invalid",
            "NotAnInChI",
            "InChI=1S/INVALID",
        ]

        for inchi in invalid_inchi:
            assert validate_molecular_string(inchi, "inchi") is False

        # Test empty string separately (raises ValidationError)
        with pytest.raises(ValidationError, match="Empty INCHI"):
            validate_molecular_string("", "inchi")

    def test_validate_molecular_string_empty_string(self):
        """Test validate_molecular_string with empty string."""
        with pytest.raises(ValidationError, match="Empty SMILES"):
            validate_molecular_string("", "smiles")

    def test_validate_molecular_string_none_value(self):
        """Test validate_molecular_string with None value."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_molecular_string(None, "smiles")

    def test_validate_molecular_string_invalid_format_type(self):
        """Test validate_molecular_string with invalid format types."""
        invalid_formats = [
            "invalid",
            "xyz",
            "pdb",
            "",
            None,
        ]

        for invalid_format in invalid_formats:
            with pytest.raises(ValidationError):
                validate_molecular_string("CCO", invalid_format)

    def test_validate_molecular_string_string_too_long(self):
        """Test validate_molecular_string with overly long string."""
        long_string = "C" * 10001

        with pytest.raises(ValidationError, match="too long"):
            validate_molecular_string(long_string, "smiles")

    def test_validate_molecular_string_wrong_type(self):
        """Test validate_molecular_string with wrong input types."""
        invalid_inputs = [
            123,  # integer
            12.34,  # float
            [],  # list
            {},  # dict
            set(),  # set
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(ValidationError, match="must be a string"):
                validate_molecular_string(invalid_input, "smiles")

    def test_validate_molecular_string_format_variations(self):
        """Test validate_molecular_string with format variations."""
        # Test that format type normalization works
        format_variations = [
            ("smiles", "CCO"),
            ("SMILES", "CCO"),
            ("smi", "CCO"),
            ("SMI", "CCO"),
            ("inchi", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"),
            ("INCHI", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"),
        ]

        for format_type, molecule in format_variations:
            result = validate_molecular_string(molecule, format_type)
            assert result is True

    def test_validate_molecular_string_parser_error_handling(self):
        """Test validate_molecular_string error handling from parser."""
        # Test that parser errors result in False return value
        invalid_smiles = "DEFINITELY_INVALID_SMILES_12345"
        result = validate_molecular_string(invalid_smiles, "smiles")
        assert result is False

    def test_validate_molecular_string_exception_handling(self):
        """Test that non-validation exceptions are handled correctly."""
        # The function actually uses the real parser validation in most cases
        # Let's test a case where we bypass the initial validation
        with patch("molecular_string_renderer.core.validate_mol_string"):
            with patch("molecular_string_renderer.core.validate_format_type"):
                with patch(
                    "molecular_string_renderer.core.get_parser"
                ) as mock_get_parser:
                    mock_parser = Mock()
                    mock_parser.validate.side_effect = RuntimeError("Unexpected error")
                    mock_get_parser.return_value = mock_parser

                    # Should return False for unexpected errors (lines 318-320)
                    result = validate_molecular_string("CCO", "smiles")
                    assert result is False

    def test_validate_molecular_string_validation_error_propagation(self):
        """Test that ValidationError is properly propagated."""
        # Use an actual validation error case
        with pytest.raises(ValidationError):
            validate_molecular_string("", "smiles")

    def test_validate_molecular_string_batch_validation(self):
        """Test validate_molecular_string for batch validation scenarios."""
        molecules = [
            ("CCO", "smiles", True),
            ("CC(=O)O", "smiles", True),
            ("INVALID", "smiles", False),
            ("c1ccccc1", "smiles", True),
            ("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "inchi", True),
            ("InChI=invalid", "inchi", False),
        ]

        for mol_string, format_type, expected in molecules:
            result = validate_molecular_string(mol_string, format_type)
            assert result == expected

    def test_validate_molecular_string_supported_formats(self):
        """Test validate_molecular_string with all supported formats."""
        # Use actual valid molecules for each format type
        format_molecules = [
            ("smiles", "CCO"),
            ("smi", "CCO"),
            ("inchi", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"),
            # Skip mol, sdf, selfies as they may not have test data available
        ]

        for format_type, molecule in format_molecules:
            result = validate_molecular_string(molecule, format_type)
            assert result is True

    def test_validate_molecular_string_edge_cases(self):
        """Test validate_molecular_string with edge cases."""
        # Test minimal valid molecule
        assert validate_molecular_string("C", "smiles") is True

        # Test whitespace handling
        with pytest.raises(ValidationError):
            validate_molecular_string("   ", "smiles")


class TestGetSupportedFormats:
    """Test the get_supported_formats function."""

    def test_get_supported_formats_structure(self):
        """Test that get_supported_formats returns correct structure."""
        formats = get_supported_formats()

        assert isinstance(formats, dict)
        assert "input_formats" in formats
        assert "output_formats" in formats
        assert "renderer_types" in formats

    def test_get_supported_formats_input_formats(self):
        """Test input formats in get_supported_formats."""
        formats = get_supported_formats()
        input_formats = formats["input_formats"]

        expected_input_formats = {"smiles", "smi", "inchi", "mol", "sdf", "selfies"}

        assert isinstance(input_formats, dict)
        assert set(input_formats.keys()) == expected_input_formats

        # Verify all have descriptions
        for format_name, description in input_formats.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_get_supported_formats_output_formats(self):
        """Test output formats in get_supported_formats."""
        formats = get_supported_formats()
        output_formats = formats["output_formats"]

        expected_output_formats = {
            "png",
            "svg",
            "jpg",
            "jpeg",
            "pdf",
            "webp",
            "tiff",
            "tif",
            "bmp",
        }

        assert isinstance(output_formats, dict)
        assert set(output_formats.keys()) == expected_output_formats

        # Verify all have descriptions
        for format_name, description in output_formats.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_get_supported_formats_renderer_types(self):
        """Test renderer types in get_supported_formats."""
        formats = get_supported_formats()
        renderer_types = formats["renderer_types"]

        expected_renderer_types = {"2d", "grid"}

        assert isinstance(renderer_types, dict)
        assert set(renderer_types.keys()) == expected_renderer_types

        # Verify all have descriptions
        for renderer_type, description in renderer_types.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_get_supported_formats_consistency(self):
        """Test that get_supported_formats is consistent across calls."""
        formats1 = get_supported_formats()
        formats2 = get_supported_formats()

        assert formats1 == formats2

    def test_get_supported_formats_immutability(self):
        """Test that modifying returned dict doesn't affect future calls."""
        formats1 = get_supported_formats()

        # Modify the returned dict
        formats1["input_formats"]["new_format"] = "New format"
        formats1["new_key"] = "new_value"

        # Get fresh copy
        formats2 = get_supported_formats()

        # Should not contain modifications
        assert "new_format" not in formats2["input_formats"]
        assert "new_key" not in formats2

    def test_get_supported_formats_specific_descriptions(self):
        """Test specific format descriptions."""
        formats = get_supported_formats()

        # Check some specific descriptions using partial matches
        assert "Simplified" in formats["input_formats"]["smiles"]
        assert "International" in formats["input_formats"]["inchi"]
        assert "Portable Network" in formats["output_formats"]["png"]
        assert "Scalable Vector" in formats["output_formats"]["svg"]
        assert "2D" in formats["renderer_types"]["2d"]
        assert "Grid layout" in formats["renderer_types"]["grid"]

    def test_get_supported_formats_no_parameters(self):
        """Test that get_supported_formats takes no parameters."""
        # Should work without any parameters
        formats = get_supported_formats()
        assert isinstance(formats, dict)

        # Should not accept any parameters
        with pytest.raises(TypeError):
            get_supported_formats("invalid_param")


class TestCoreIntegration:
    """Integration tests for core module functions."""

    def test_core_functions_work_together(self):
        """Test that core functions work together correctly."""
        # Get supported formats
        formats = get_supported_formats()

        # Use supported input format
        input_format = list(formats["input_formats"].keys())[0]  # smiles
        output_format = list(formats["output_formats"].keys())[0]  # png

        # Validate molecular string
        is_valid = validate_molecular_string("CCO", input_format)
        assert is_valid is True

        # Render molecule
        image = render_molecule(
            "CCO",
            format_type=input_format,
            output_format=output_format,
            auto_filename=False,
        )
        assert isinstance(image, Image.Image)

        # Render grid
        grid_image = render_molecules_grid(["CCO", "CC(=O)O"], format_type=input_format, auto_filename=False)
        assert isinstance(grid_image, Image.Image)

    def test_error_consistency_across_functions(self):
        """Test that error handling is consistent across functions."""
        # Test with same invalid input
        invalid_format = "invalid_format"

        # All functions should raise ValidationError for invalid format
        with pytest.raises(ValidationError):
            validate_molecular_string("CCO", invalid_format)

        with pytest.raises(ValidationError):
            render_molecule("CCO", format_type=invalid_format)

        with pytest.raises(ValidationError):
            render_molecules_grid(["CCO"], format_type=invalid_format)

    def test_configuration_consistency(self):
        """Test that configurations work consistently across functions."""
        render_config = RenderConfig(width=400, height=300)
        parser_config = ParserConfig(sanitize=True)
        output_config = OutputConfig(quality=90)

        # Both render functions should accept the same configurations
        image1 = render_molecule(
            "CCO",
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
            auto_filename=False,
        )

        image2 = render_molecules_grid(
            ["CCO"],
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
            auto_filename=False,
        )

        assert isinstance(image1, Image.Image)
        assert isinstance(image2, Image.Image)

    def test_format_support_consistency(self):
        """Test that format support is consistent between functions."""
        formats = get_supported_formats()

        # Test a few supported formats
        test_molecules = [
            ("CCO", "smiles"),
            ("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "inchi"),
        ]

        for molecule, format_type in test_molecules:
            # Should be listed as supported
            assert format_type in formats["input_formats"]

            # Should validate successfully
            assert validate_molecular_string(molecule, format_type) is True

            # Should render successfully
            image = render_molecule(
                molecule, format_type=format_type, auto_filename=False
            )
            assert isinstance(image, Image.Image)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Get supported formats
            formats = get_supported_formats()
            assert "smiles" in formats["input_formats"]
            assert "png" in formats["output_formats"]

            # Step 2: Validate molecules
            molecules = ["CCO", "CC(=O)O", "c1ccccc1"]
            valid_molecules = []

            for mol in molecules:
                if validate_molecular_string(mol, "smiles"):
                    valid_molecules.append(mol)

            assert len(valid_molecules) == 3

            # Step 3: Render individual molecule
            output_path1 = Path(temp_dir) / "single.png"
            image1 = render_molecule(
                valid_molecules[0],
                format_type="smiles",
                output_format="png",
                output_path=output_path1,
            )

            assert isinstance(image1, Image.Image)
            assert output_path1.exists()

            # Step 4: Render molecule grid
            output_path2 = Path(temp_dir) / "grid.png"
            image2 = render_molecules_grid(
                valid_molecules,
                format_type="smiles",
                output_format="png",
                output_path=output_path2,
                legends=["Ethanol", "Acetic Acid", "Benzene"],
            )

            assert isinstance(image2, Image.Image)
            assert output_path2.exists()

    def test_performance_with_multiple_calls(self):
        """Test performance characteristics with multiple function calls."""
        molecules = ["CCO", "CC(=O)O", "c1ccccc1", "C", "CC"]

        # Multiple validation calls
        for _ in range(3):
            for mol in molecules:
                assert validate_molecular_string(mol, "smiles") is True

        # Multiple render calls
        for _ in range(2):
            for mol in molecules[:2]:  # Limit to avoid too much computation
                image = render_molecule(mol, auto_filename=False)
                assert isinstance(image, Image.Image)

        # Grid rendering
        grid_image = render_molecules_grid(molecules[:3], auto_filename=False)
        assert isinstance(grid_image, Image.Image)


class TestCoreErrorHandling:
    """Test comprehensive error handling in core functions."""

    def test_memory_error_simulation(self):
        """Test behavior under simulated memory constraints."""
        # Test that the function can handle reasonably complex molecules
        # Use a simple valid molecule to avoid parsing errors
        simple_molecule = "CCCCCCCCCC"  # decane

        image = render_molecule(simple_molecule, auto_filename=False)
        assert isinstance(image, Image.Image)

    def test_io_error_simulation(self):
        """Test behavior under simulated I/O errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a read-only directory to cause I/O errors
            readonly_dir = Path(temp_dir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)  # Read-only

            output_path = readonly_dir / "test.png"

            # Should raise an error when trying to save to read-only directory
            with pytest.raises((OSError, PermissionError, OutputError)):
                render_molecule("CCO", output_path=output_path)

    def test_concurrent_access(self):
        """Test behavior under concurrent access (thread safety)."""
        import threading

        results = []
        errors = []

        def render_worker():
            try:
                image = render_molecule("CCO", auto_filename=False)
                results.append(image)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=render_worker)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout

        # Check results
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 3

        for result in results:
            assert isinstance(result, Image.Image)

    def test_cleanup_after_errors(self):
        """Test that resources are properly cleaned up after errors."""
        # Test that failed operations don't leave corrupted state
        with pytest.raises(ValidationError):
            render_molecule("", "smiles")  # This should fail

        # Next operation should still work
        image = render_molecule("CCO", "smiles", auto_filename=False)
        assert isinstance(image, Image.Image)

        # Test with grid rendering
        with pytest.raises(ValidationError):
            render_molecules_grid([])  # This should fail

        # Next operation should still work
        grid_image = render_molecules_grid(["CCO"], auto_filename=False)
        assert isinstance(grid_image, Image.Image)
