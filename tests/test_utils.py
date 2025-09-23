"""
Tests for the utils module.

Tests utility functions for molecular rendering operations including parsing,
validation, output handling, and error management.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig, ParserConfig
from molecular_string_renderer.exceptions import ValidationError
from molecular_string_renderer.utils import (
    ERROR_TEMPLATES,
    MAX_GRID_MOLECULES,
    MAX_MOLECULAR_STRING_LENGTH,
    SUPPORTED_INPUT_FORMATS,
    filter_legends_by_indices,
    format_parsing_errors,
    handle_operation_error,
    handle_output_saving,
    parse_molecule_list,
    validate_and_normalize_inputs,
)


class TestConstants:
    """Test module constants."""

    def test_supported_input_formats_is_frozenset(self):
        """Test that SUPPORTED_INPUT_FORMATS is a frozenset."""
        assert isinstance(SUPPORTED_INPUT_FORMATS, frozenset)
        assert len(SUPPORTED_INPUT_FORMATS) > 0

    def test_supported_input_formats_contains_expected_formats(self):
        """Test that common molecular formats are supported."""
        expected_formats = {"smiles", "smi", "inchi", "mol", "sdf", "selfies"}
        assert expected_formats.issubset(SUPPORTED_INPUT_FORMATS)

    def test_max_constants_are_reasonable(self):
        """Test that max constants have reasonable values."""
        assert MAX_MOLECULAR_STRING_LENGTH > 0
        assert MAX_GRID_MOLECULES > 0
        assert MAX_MOLECULAR_STRING_LENGTH >= 1000  # Should handle reasonable molecules
        assert MAX_GRID_MOLECULES >= 10  # Should handle reasonable grids

    def test_error_templates_is_dict(self):
        """Test that ERROR_TEMPLATES is a dict with expected keys."""
        assert isinstance(ERROR_TEMPLATES, dict)
        expected_keys = {
            "parse_failed",
            "parsing_error",
            "no_valid_molecules",
            "renderer_none",
            "grid_renderer_none",
        }
        assert expected_keys.issubset(set(ERROR_TEMPLATES.keys()))


class TestParseMoleculeList:
    """Test parse_molecule_list function."""

    @patch("molecular_string_renderer.utils.get_parser")
    @patch("molecular_string_renderer.utils.validate_molecular_string")
    def test_successful_parsing_all_molecules(self, mock_validate, mock_get_parser):
        """Test successful parsing of all molecules."""
        # Setup mocks
        mock_parser = Mock()
        mock_mol1 = Mock()
        mock_mol2 = Mock()
        mock_parser.parse.side_effect = [mock_mol1, mock_mol2]
        mock_get_parser.return_value = mock_parser

        molecular_strings = ["CCO", "c1ccccc1"]
        parser_config = ParserConfig()

        mols, valid_indices, parsing_errors = parse_molecule_list(
            molecular_strings, "smiles", parser_config
        )

        assert len(mols) == 2
        assert mols == [mock_mol1, mock_mol2]
        assert valid_indices == [0, 1]
        assert parsing_errors == []

        # Verify calls
        mock_get_parser.assert_called_once_with("smiles", parser_config)
        assert mock_validate.call_count == 2
        assert mock_parser.parse.call_count == 2

    @patch("molecular_string_renderer.utils.get_parser")
    @patch("molecular_string_renderer.utils.validate_molecular_string")
    def test_parsing_with_failures(self, mock_validate, mock_get_parser):
        """Test parsing with some failed molecules."""
        mock_parser = Mock()
        mock_mol = Mock()
        mock_parser.parse.side_effect = [mock_mol, None, Exception("Parse error")]
        mock_get_parser.return_value = mock_parser

        molecular_strings = ["CCO", "invalid", "bad_mol"]
        parser_config = ParserConfig()

        mols, valid_indices, parsing_errors = parse_molecule_list(
            molecular_strings, "smiles", parser_config
        )

        assert len(mols) == 1
        assert mols == [mock_mol]
        assert valid_indices == [0]
        assert len(parsing_errors) == 2
        assert "Index 1" in parsing_errors[0]
        assert "Index 2" in parsing_errors[1]

    @patch("molecular_string_renderer.utils.validate_molecular_string")
    def test_parsing_with_validation_error(self, mock_validate):
        """Test parsing when validation fails."""
        mock_validate.side_effect = [None, ValidationError("Invalid string"), None]
        mock_parser = Mock()
        mock_mol = Mock()
        mock_parser.parse.side_effect = [mock_mol, mock_mol]

        molecular_strings = ["CCO", "invalid", "c1ccccc1"]
        parser_config = ParserConfig()

        mols, valid_indices, parsing_errors = parse_molecule_list(
            molecular_strings, "smiles", parser_config, parser=mock_parser
        )

        assert len(mols) == 2
        assert valid_indices == [0, 2]
        assert len(parsing_errors) == 1
        assert "Index 1" in parsing_errors[0]

    def test_parsing_with_provided_parser(self):
        """Test parsing with pre-configured parser."""
        mock_parser = Mock()
        mock_mol = Mock()
        mock_parser.parse.return_value = mock_mol

        molecular_strings = ["CCO"]
        parser_config = ParserConfig()

        with patch("molecular_string_renderer.utils.validate_molecular_string"):
            mols, valid_indices, parsing_errors = parse_molecule_list(
                molecular_strings, "smiles", parser_config, parser=mock_parser
            )

        assert len(mols) == 1
        assert valid_indices == [0]
        assert parsing_errors == []

    def test_parsing_empty_list(self):
        """Test parsing empty molecule list."""
        molecular_strings = []
        parser_config = ParserConfig()

        mols, valid_indices, parsing_errors = parse_molecule_list(
            molecular_strings, "smiles", parser_config
        )

        assert mols == []
        assert valid_indices == []
        assert parsing_errors == []

    @patch("molecular_string_renderer.utils.logger")
    def test_parsing_logs_warnings_on_failure(self, mock_logger):
        """Test that parsing failures are logged as warnings."""
        mock_parser = Mock()
        mock_parser.parse.side_effect = Exception("Parse error")

        molecular_strings = ["invalid"]
        parser_config = ParserConfig()

        with patch("molecular_string_renderer.utils.validate_molecular_string"):
            parse_molecule_list(
                molecular_strings, "smiles", parser_config, parser=mock_parser
            )

        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Failed to parse molecule at index 0" in warning_call


class TestFormatParsingErrors:
    """Test format_parsing_errors function."""

    def test_format_few_errors(self):
        """Test formatting when error count is below max."""
        errors = ["Error 1", "Error 2", "Error 3"]
        result = format_parsing_errors(errors, max_errors=5)

        assert result == "Error 1; Error 2; Error 3"

    def test_format_many_errors_with_truncation(self):
        """Test formatting with truncation when errors exceed max."""
        errors = ["Error 1", "Error 2", "Error 3", "Error 4", "Error 5", "Error 6"]
        result = format_parsing_errors(errors, max_errors=3)

        expected = "Error 1; Error 2; Error 3 (and 3 more errors)"
        assert result == expected

    def test_format_exact_max_errors(self):
        """Test formatting when error count equals max."""
        errors = ["Error 1", "Error 2", "Error 3"]
        result = format_parsing_errors(errors, max_errors=3)

        assert result == "Error 1; Error 2; Error 3"

    def test_format_empty_errors(self):
        """Test formatting empty error list."""
        errors = []
        result = format_parsing_errors(errors)

        assert result == ""

    def test_format_single_error(self):
        """Test formatting single error."""
        errors = ["Single error"]
        result = format_parsing_errors(errors)

        assert result == "Single error"

    def test_format_default_max_errors(self):
        """Test default max_errors parameter."""
        errors = [f"Error {i}" for i in range(10)]
        result = format_parsing_errors(errors)  # Default max_errors=5

        assert (
            "Error 0; Error 1; Error 2; Error 3; Error 4 (and 5 more errors)" == result
        )


class TestFilterLegendsByIndices:
    """Test filter_legends_by_indices function."""

    def test_filter_with_valid_indices(self):
        """Test filtering legends with valid indices."""
        legends = ["Molecule A", "Molecule B", "Molecule C", "Molecule D"]
        valid_indices = [0, 2, 3]
        total_molecules = 4

        result = filter_legends_by_indices(legends, valid_indices, total_molecules)

        assert result == ["Molecule A", "Molecule C", "Molecule D"]

    def test_filter_with_none_legends(self):
        """Test filtering when legends is None."""
        result = filter_legends_by_indices(None, [0, 1], 2)
        assert result is None

    def test_filter_with_empty_legends(self):
        """Test filtering with empty legends list."""
        result = filter_legends_by_indices([], [0, 1], 2)
        assert result == []

    def test_filter_with_empty_valid_indices(self):
        """Test filtering with empty valid indices."""
        legends = ["A", "B", "C"]
        result = filter_legends_by_indices(legends, [], 3)
        assert result == legends

    def test_filter_with_index_out_of_bounds(self):
        """Test filtering when index exceeds legends length."""
        legends = ["A", "B"]
        valid_indices = [0, 1, 2]  # Index 2 is out of bounds

        with patch("molecular_string_renderer.utils.logger") as mock_logger:
            result = filter_legends_by_indices(legends, valid_indices, 3)

        assert result is None
        mock_logger.warning.assert_called_once_with(
            "Legend count mismatch after filtering, disabling legends"
        )

    def test_filter_partial_indices(self):
        """Test filtering with partial valid indices."""
        legends = ["A", "B", "C", "D", "E"]
        valid_indices = [1, 3]

        result = filter_legends_by_indices(legends, valid_indices, 5)
        assert result == ["B", "D"]

    def test_filter_maintains_order(self):
        """Test that filtering maintains the order of valid indices."""
        legends = ["First", "Second", "Third", "Fourth"]
        valid_indices = [3, 1, 0]  # Out of order

        result = filter_legends_by_indices(legends, valid_indices, 4)
        assert result == ["Fourth", "Second", "First"]


class TestValidateAndNormalizeInputs:
    """Test validate_and_normalize_inputs function."""

    @patch("molecular_string_renderer.utils.validate_format_type")
    @patch("molecular_string_renderer.utils.validate_output_path")
    @patch("molecular_string_renderer.utils.validate_molecular_string")
    def test_validate_all_inputs(
        self, mock_validate_mol, mock_validate_path, mock_validate_format
    ):
        """Test validation of all input parameters."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = Path("/test/path")

        mol_string = "CCO"
        format_type = "SMILES"
        output_path = "/test/path"

        result = validate_and_normalize_inputs(mol_string, format_type, output_path)

        assert result == (mol_string, "smiles", Path("/test/path"))
        mock_validate_format.assert_called_once_with("SMILES", SUPPORTED_INPUT_FORMATS)
        mock_validate_path.assert_called_once_with("/test/path")
        mock_validate_mol.assert_called_once_with("CCO", "smiles")

    @patch("molecular_string_renderer.utils.validate_format_type")
    def test_validate_only_format(self, mock_validate_format):
        """Test validation of only format type."""
        mock_validate_format.return_value = "inchi"

        result = validate_and_normalize_inputs(
            molecular_string=None, format_type="InChI", output_path=None
        )

        assert result == (None, "inchi", None)
        mock_validate_format.assert_called_once_with("InChI", SUPPORTED_INPUT_FORMATS)

    @patch("molecular_string_renderer.utils.validate_output_path")
    def test_validate_only_output_path(self, mock_validate_path):
        """Test validation of only output path."""
        mock_validate_path.return_value = Path("/output")

        result = validate_and_normalize_inputs(
            molecular_string=None, format_type=None, output_path="/output"
        )

        assert result == (None, None, Path("/output"))
        mock_validate_path.assert_called_once_with("/output")

    def test_validate_no_inputs(self):
        """Test validation when all inputs are None."""
        result = validate_and_normalize_inputs()
        assert result == (None, None, None)

    def test_validate_molecular_string_without_format(self):
        """Test that molecular string is not validated without format."""
        result = validate_and_normalize_inputs(molecular_string="CCO")
        assert result == ("CCO", None, None)
        # Should not raise any validation errors


class TestHandleOperationError:
    """Test handle_operation_error function."""

    def test_reraise_original_exception_type(self):
        """Test that original exception is re-raised if it's the expected type."""
        original_error = ValueError("Original error")

        with pytest.raises(ValueError, match="Original error"):
            handle_operation_error(
                "test_operation", original_error, ValueError, "Fallback message"
            )

    def test_wrap_different_exception_type(self):
        """Test that different exception types are wrapped."""
        original_error = RuntimeError("Runtime error")

        with pytest.raises(ValueError) as exc_info:
            handle_operation_error(
                "test_operation", original_error, ValueError, "Fallback message"
            )

        assert "Fallback message: Runtime error" in str(exc_info.value)
        assert exc_info.value.__cause__ is original_error

    @patch("molecular_string_renderer.utils.logger")
    def test_logs_error(self, mock_logger):
        """Test that error is logged."""
        original_error = Exception("Test error")

        with pytest.raises(ValueError):
            handle_operation_error(
                "test_operation", original_error, ValueError, "Fallback"
            )

        mock_logger.error.assert_called_once_with("Error in test_operation: Test error")

    def test_preserve_exception_chain(self):
        """Test that exception chaining is preserved."""
        original_error = RuntimeError("Original")

        with pytest.raises(ValueError) as exc_info:
            handle_operation_error("operation", original_error, ValueError, "Wrapped")

        assert exc_info.value.__cause__ is original_error


class TestHandleOutputSaving:
    """Test handle_output_saving function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_image = Image.new("RGB", (100, 100), "white")
        self.output_config = OutputConfig(format="png", quality=95)

    @patch("molecular_string_renderer.utils.get_output_handler")
    def test_save_with_output_path(self, mock_get_handler):
        """Test saving with specified output path."""
        mock_handler = Mock()
        mock_get_handler.return_value = mock_handler

        image = Image.new("RGB", (100, 100), "white")
        output_path = Path("/test/output.png")

        handle_output_saving(image, output_path, "png", self.output_config)

        mock_get_handler.assert_called_once_with("png", self.output_config)
        mock_handler.save.assert_called_once_with(image, output_path)

    @patch("molecular_string_renderer.utils.get_output_handler")
    def test_save_svg_with_molecule(self, mock_get_handler):
        """Test SVG saving with molecule object for vector rendering."""
        mock_handler = Mock()
        mock_get_handler.return_value = mock_handler
        mock_mol = Mock()

        image = Image.new("RGB", (100, 100), "white")
        output_path = Path("/test/output.svg")

        handle_output_saving(
            image, output_path, "svg", self.output_config, mol=mock_mol
        )

        mock_handler.set_molecule.assert_called_once_with(mock_mol)
        mock_handler.save.assert_called_once_with(image, output_path)

    @patch("molecular_string_renderer.utils.create_safe_filename")
    @patch("molecular_string_renderer.utils.get_output_handler")
    def test_auto_filename_generation(self, mock_get_handler, mock_safe_filename):
        """Test automatic filename generation."""
        mock_handler = Mock()
        mock_handler.file_extension = ".png"
        mock_get_handler.return_value = mock_handler
        mock_safe_filename.return_value = "molecule.png"

        image = Image.new("RGB", (100, 100), "white")

        handle_output_saving(
            image,
            output_path=None,
            output_format="png",
            output_config=self.output_config,
            auto_filename=True,
            molecular_string="CCO",
        )

        mock_safe_filename.assert_called_once_with("CCO", ".png")
        expected_path = Path.cwd() / "molecule.png"
        mock_handler.save.assert_called_once_with(image, expected_path)

    @patch("molecular_string_renderer.utils.get_output_handler")
    def test_no_saving_when_no_path_no_auto(self, mock_get_handler):
        """Test that no saving occurs when no path and auto_filename=False."""
        image = Image.new("RGB", (100, 100), "white")

        handle_output_saving(
            image,
            output_path=None,
            output_format="png",
            output_config=self.output_config,
            auto_filename=False,
        )

        mock_get_handler.assert_not_called()

    @patch("molecular_string_renderer.utils.get_output_handler")
    def test_svg_without_molecule_object(self, mock_get_handler):
        """Test SVG saving without molecule object."""
        mock_handler = Mock()
        mock_get_handler.return_value = mock_handler

        image = Image.new("RGB", (100, 100), "white")
        output_path = Path("/test/output.svg")

        handle_output_saving(image, output_path, "svg", self.output_config, mol=None)

        mock_handler.set_molecule.assert_not_called()
        mock_handler.save.assert_called_once_with(image, output_path)

    @patch("molecular_string_renderer.utils.get_output_handler")
    def test_auto_filename_without_molecular_string(self, mock_get_handler):
        """Test auto filename when molecular_string is None."""
        mock_handler = Mock()
        mock_get_handler.return_value = mock_handler

        image = Image.new("RGB", (100, 100), "white")

        handle_output_saving(
            image,
            output_path=None,
            output_format="png",
            output_config=self.output_config,
            auto_filename=True,
            molecular_string=None,
        )

        # Should not attempt to save when molecular_string is None
        mock_handler.save.assert_not_called()


class TestUtilsIntegration:
    """Integration tests for utils module."""

    def test_constants_are_used_correctly(self):
        """Test that module constants have reasonable values for integration."""
        # Test that constants are suitable for real usage
        assert "smiles" in SUPPORTED_INPUT_FORMATS
        assert "inchi" in SUPPORTED_INPUT_FORMATS
        assert MAX_MOLECULAR_STRING_LENGTH >= 1000
        assert MAX_GRID_MOLECULES >= 10

    def test_error_templates_can_be_formatted(self):
        """Test that error templates can be used for string formatting."""
        # Test that error templates work with expected parameters
        formatted = ERROR_TEMPLATES["parse_failed"].format(
            format="SMILES", string="invalid"
        )
        assert "SMILES" in formatted
        assert "invalid" in formatted

        formatted = ERROR_TEMPLATES["parsing_error"].format(
            format="InChI", string="bad", error="syntax error"
        )
        assert "InChI" in formatted
        assert "bad" in formatted
        assert "syntax error" in formatted

    @patch("molecular_string_renderer.utils.validate_molecular_string")
    @patch("molecular_string_renderer.utils.validate_format_type")
    @patch("molecular_string_renderer.utils.validate_output_path")
    def test_full_validation_workflow(self, mock_path, mock_format, mock_mol):
        """Test the complete validation workflow."""
        mock_format.return_value = "smiles"
        mock_path.return_value = Path("/output")

        result = validate_and_normalize_inputs("CCO", "SMILES", "/output")

        assert result[0] == "CCO"
        assert result[1] == "smiles"
        assert isinstance(result[2], Path)

        # All validations should be called
        mock_format.assert_called_once()
        mock_path.assert_called_once()
        mock_mol.assert_called_once()
