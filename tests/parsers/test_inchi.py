"""
Tests for the InChI parser implementation.
"""

from unittest.mock import patch

import pytest
from rdkit import Chem

from molecular_string_renderer.config import ParserConfig
from molecular_string_renderer.parsers.inchi import InChIParser


class TestInChIParser:
    """Test the InChIParser class."""

    def test_initialization_default_config(self):
        """Test parser initializes with default config."""
        parser = InChIParser()
        assert parser.config is not None
        assert isinstance(parser.config, ParserConfig)

    def test_initialization_custom_config(self):
        """Test parser initializes with custom config."""
        config = ParserConfig(sanitize=False)
        parser = InChIParser(config)
        assert parser.config is config


class TestInChIParsingValid:
    """Test InChI parsing with valid inputs."""

    def test_parse_simple_molecules(self):
        """Test parsing simple InChI strings."""
        parser = InChIParser()

        test_cases = [
            ("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "ethanol"),
            ("InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H", "benzene"),
            ("InChI=1S/CH4/h1H4", "methane"),
            ("InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)", "acetic acid"),
        ]

        for inchi, description in test_cases:
            mol = parser.parse(inchi)
            assert mol is not None, f"Failed to parse {description}: {inchi}"
            assert isinstance(mol, Chem.Mol)

    def test_parse_with_whitespace(self):
        """Test parsing InChI with leading/trailing whitespace."""
        parser = InChIParser()

        inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        test_cases = [
            f"  {inchi}  ",
            f"\t{inchi}\t",
            f"\n{inchi}\n",
            f" {inchi} ",
        ]

        for inchi_with_ws in test_cases:
            mol = parser.parse(inchi_with_ws)
            assert mol is not None, f"Failed to parse: '{inchi_with_ws}'"

    def test_parse_complex_molecules(self):
        """Test parsing more complex InChI strings."""
        parser = InChIParser()

        # Caffeine InChI
        caffeine_inchi = (
            "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"
        )

        mol = parser.parse(caffeine_inchi)
        assert mol is not None, "Failed to parse caffeine InChI"
        assert isinstance(mol, Chem.Mol)


class TestInChIParsingInvalid:
    """Test InChI parsing with invalid inputs."""

    def test_parse_empty_string(self):
        """Test parsing empty InChI string raises error."""
        parser = InChIParser()

        with pytest.raises(ValueError, match="InChI string cannot be empty"):
            parser.parse("")

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only string raises error."""
        parser = InChIParser()

        test_cases = ["   ", "\t", "\n", "\r\n"]

        for inchi in test_cases:
            with pytest.raises(ValueError, match="InChI string cannot be empty"):
                parser.parse(inchi)

    def test_parse_none(self):
        """Test parsing None raises error."""
        parser = InChIParser()

        with pytest.raises(ValueError, match="InChI string cannot be empty"):
            parser.parse(None)

    def test_parse_missing_inchi_prefix(self):
        """Test parsing string without InChI= prefix raises error."""
        parser = InChIParser()

        invalid_strings = [
            "1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # Missing InChI=
            "INCHI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # Wrong case
            "inchi=1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # Wrong case
            "InChI1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # Missing =
        ]

        for invalid_string in invalid_strings:
            with pytest.raises(
                ValueError, match="InChI string must start with 'InChI='"
            ):
                parser.parse(invalid_string)

    def test_parse_invalid_inchi_format(self):
        """Test parsing invalid InChI format raises error."""
        parser = InChIParser()

        invalid_inchis = [
            "InChI=",  # Empty after prefix
            "InChI=invalid",  # Invalid format
            "InChI=1S/",  # Incomplete
            "InChI=1S/X/c1/h1H",  # Invalid atom symbol
        ]

        for inchi in invalid_inchis:
            with pytest.raises(ValueError, match="Invalid InChI string"):
                parser.parse(inchi)


class TestInChIValidation:
    """Test InChI validation methods."""

    def test_validate_valid_inchis(self):
        """Test validation with valid InChI strings."""
        parser = InChIParser()

        valid_inchis = [
            "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",
            "InChI=1S/CH4/h1H4",
        ]

        for inchi in valid_inchis:
            assert parser.validate(inchi) is True, f"Should validate: {inchi}"

    def test_validate_invalid_inchis(self):
        """Test validation with invalid InChI strings."""
        parser = InChIParser()

        invalid_inchis = [
            "",
            "   ",
            None,
            "1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # Missing prefix
            "InChI=",
            "InChI=invalid",
        ]

        for inchi in invalid_inchis:
            assert parser.validate(inchi) is False, f"Should not validate: {inchi}"

    def test_validate_with_whitespace(self):
        """Test validation handles whitespace correctly."""
        parser = InChIParser()

        inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        test_cases = [
            (f"  {inchi}  ", True),
            (f"\t{inchi}\t", True),
            ("  ", False),
        ]

        for inchi_test, expected in test_cases:
            result = parser.validate(inchi_test)
            assert result is expected, f"Validation failed for: '{inchi_test}'"

    def test_validate_prefix_checking(self):
        """Test validation properly checks for InChI= prefix."""
        parser = InChIParser()

        test_cases = [
            ("InChI=1S/CH4/h1H4", True),
            ("INCHI=1S/CH4/h1H4", False),  # Wrong case
            ("inchi=1S/CH4/h1H4", False),  # Wrong case
            ("1S/CH4/h1H4", False),  # Missing prefix
        ]

        for inchi, expected in test_cases:
            result = parser.validate(inchi)
            assert result is expected, f"Prefix validation failed for: {inchi}"


class TestInChIPostProcessing:
    """Test InChI parsing with different post-processing configurations."""

    def test_parse_with_hydrogen_removal(self):
        """Test parsing with hydrogen removal enabled."""
        config = ParserConfig(show_hydrogen=False)
        parser = InChIParser(config)

        mol = parser.parse("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")  # Ethanol
        assert mol is not None

        # Check that no explicit hydrogens are present
        has_explicit_h = any(atom.GetSymbol() == "H" for atom in mol.GetAtoms())
        assert not has_explicit_h

    def test_parse_with_hydrogen_addition(self):
        """Test parsing with hydrogen addition enabled."""
        config = ParserConfig(show_hydrogen=True)
        parser = InChIParser(config)

        mol = parser.parse("InChI=1S/CH4/h1H4")  # Methane
        assert mol is not None

        # Should have explicit hydrogens added
        total_atoms = mol.GetNumAtoms()
        assert total_atoms > 1  # More than just carbon

    def test_parse_with_sanitization_disabled(self):
        """Test parsing with sanitization disabled."""
        config = ParserConfig(sanitize=False)
        parser = InChIParser(config)

        mol = parser.parse("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
        assert mol is not None
        assert isinstance(mol, Chem.Mol)

    def test_parse_error_message_formatting(self):
        """Test that error messages contain the problematic InChI."""
        parser = InChIParser()

        invalid_inchi = "InChI=invalid"

        with pytest.raises(ValueError) as exc_info:
            parser.parse(invalid_inchi)

        error_message = str(exc_info.value)
        assert invalid_inchi in error_message


class TestInChIExceptionHandling:
    """Test InChI parser exception handling edge cases."""

    def test_rdkit_exception_reraising(self):
        """Test that RDKit exceptions are properly re-raised."""
        parser = InChIParser()

        # Mock Chem.MolFromInchi to raise an exception with "Invalid InChI" message
        with patch(
            "molecular_string_renderer.parsers.inchi.Chem.MolFromInchi"
        ) as mock_mol_from_inchi:
            mock_mol_from_inchi.side_effect = ValueError("Invalid InChI: test error")

            with pytest.raises(ValueError, match="Invalid InChI: test error"):
                parser.parse("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")

    def test_validation_exception_handling(self):
        """Test InChI validation with exception conditions."""
        parser = InChIParser()

        # Mock MolFromInchi to raise an exception during validation
        with patch(
            "molecular_string_renderer.parsers.inchi.Chem.MolFromInchi"
        ) as mock_mol_from_inchi:
            mock_mol_from_inchi.side_effect = RuntimeError("Validation exception")

            # Should return False instead of raising
            result = parser.validate("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
            assert result is False
