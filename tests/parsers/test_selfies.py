"""
Tests for the SELFIES parser implementation.
"""

import pytest
from rdkit import Chem
from unittest.mock import patch

from molecular_string_renderer.config import ParserConfig
from molecular_string_renderer.parsers.selfies import SELFIESParser


class TestSELFIESParser:
    """Test the SELFIESParser class."""

    def test_initialization_default_config(self):
        """Test parser initializes with default config."""
        parser = SELFIESParser()
        assert parser.config is not None
        assert isinstance(parser.config, ParserConfig)

    def test_initialization_custom_config(self):
        """Test parser initializes with custom config."""
        config = ParserConfig(sanitize=False)
        parser = SELFIESParser(config)
        assert parser.config is config


class TestSELFIESParsingValid:
    """Test SELFIES parsing with valid inputs."""

    def test_parse_simple_molecules(self):
        """Test parsing simple SELFIES strings."""
        parser = SELFIESParser()

        # Test cases with known SELFIES representations
        test_cases = [
            ("[C][C][O]", "ethanol"),
            ("[C][#C]", "acetylene"),
            ("[C]", "methane"),
        ]

        for selfies, description in test_cases:
            mol = parser.parse(selfies)
            assert mol is not None, f"Failed to parse {description}: {selfies}"
            assert isinstance(mol, Chem.Mol)

    def test_parse_with_whitespace(self):
        """Test parsing SELFIES with leading/trailing whitespace."""
        parser = SELFIESParser()

        selfies = "[C][C][O]"
        test_cases = [
            f"  {selfies}  ",
            f"\t{selfies}\t",
            f"\n{selfies}\n",
            f" {selfies} ",
        ]

        for selfies_with_ws in test_cases:
            mol = parser.parse(selfies_with_ws)
            assert mol is not None, f"Failed to parse: '{selfies_with_ws}'"


class TestSELFIESParsingInvalid:
    """Test SELFIES parsing with invalid inputs."""

    def test_parse_empty_string(self):
        """Test parsing empty SELFIES string raises error."""
        parser = SELFIESParser()

        with pytest.raises(ValueError, match="SELFIES string cannot be empty"):
            parser.parse("")

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only string raises error."""
        parser = SELFIESParser()

        test_cases = ["   ", "\t", "\n", "\r\n"]

        for selfies in test_cases:
            with pytest.raises(ValueError, match="SELFIES string cannot be empty"):
                parser.parse(selfies)

    def test_parse_none(self):
        """Test parsing None raises error."""
        parser = SELFIESParser()

        with pytest.raises(ValueError, match="SELFIES string cannot be empty"):
            parser.parse(None)

    def test_parse_invalid_selfies(self):
        """Test parsing invalid SELFIES strings raises error."""
        parser = SELFIESParser()

        # Test with a string that should definitely fail
        with pytest.raises(ValueError, match="Failed to parse SELFIES"):
            parser.parse("[INVALID_ATOM_SYMBOL_THAT_DOES_NOT_EXIST]")


class TestSELFIESValidation:
    """Test SELFIES validation methods."""

    def test_validate_valid_selfies(self):
        """Test validation with valid SELFIES strings."""
        parser = SELFIESParser()

        valid_selfies = [
            "[C][C][O]",
            "[C][#C]",
            "[C]",
            "[C][C][C][C]",
        ]

        for selfies in valid_selfies:
            assert parser.validate(selfies) is True, f"Should validate: {selfies}"

    def test_validate_invalid_selfies(self):
        """Test validation with invalid SELFIES strings."""
        parser = SELFIESParser()

        invalid_selfies = [
            "",
            "   ",
            None,
            "[INVALID_ATOM_SYMBOL_THAT_DOES_NOT_EXIST]",  # Clearly invalid atom
        ]

        for selfies in invalid_selfies:
            assert parser.validate(selfies) is False, f"Should not validate: {selfies}"

    def test_validate_with_whitespace(self):
        """Test validation handles whitespace correctly."""
        parser = SELFIESParser()

        selfies = "[C][C][O]"
        test_cases = [
            (f"  {selfies}  ", True),
            (f"\t{selfies}\t", True),
            ("  ", False),
        ]

        for selfies_test, expected in test_cases:
            result = parser.validate(selfies_test)
            assert result is expected, f"Validation failed for: '{selfies_test}'"


class TestSELFIESPostProcessing:
    """Test SELFIES parsing with different post-processing configurations."""

    def test_parse_with_hydrogen_removal(self):
        """Test parsing with hydrogen removal enabled."""
        config = ParserConfig(remove_hs=True)
        parser = SELFIESParser(config)

        mol = parser.parse("[C][C][O]")  # Ethanol
        assert mol is not None

        # Check that no explicit hydrogens are present
        has_explicit_h = any(atom.GetSymbol() == "H" for atom in mol.GetAtoms())
        assert not has_explicit_h

    def test_parse_with_hydrogen_addition(self):
        """Test parsing with hydrogen addition enabled."""
        config = ParserConfig(remove_hs=False)
        parser = SELFIESParser(config)

        mol = parser.parse("[C]")  # Methane
        assert mol is not None

        # Should have explicit hydrogens added
        total_atoms = mol.GetNumAtoms()
        assert total_atoms > 1  # More than just carbon

    def test_parse_with_sanitization_disabled(self):
        """Test parsing with sanitization disabled."""
        config = ParserConfig(sanitize=False)
        parser = SELFIESParser(config)

        mol = parser.parse("[C][C][O]")
        assert mol is not None
        assert isinstance(mol, Chem.Mol)

    def test_parse_error_message_formatting(self):
        """Test that error messages contain the problematic SELFIES."""
        parser = SELFIESParser()

        invalid_selfies = "[C"

        with pytest.raises(ValueError) as exc_info:
            parser.parse(invalid_selfies)

        error_message = str(exc_info.value)
        assert invalid_selfies in error_message


class TestSELFIESExceptionHandling:
    """Test SELFIES parser exception handling edge cases."""

    def test_rdkit_exception_reraising(self):
        """Test that RDKit exceptions are properly re-raised."""
        parser = SELFIESParser()

        # Mock sf.decoder to return a valid SMILES, but MolFromSmiles to fail
        with patch(
            "molecular_string_renderer.parsers.selfies.sf.decoder"
        ) as mock_decoder:
            mock_decoder.return_value = "CCO"

            with patch(
                "molecular_string_renderer.parsers.selfies.Chem.MolFromSmiles"
            ) as mock_mol_from_smiles:
                mock_mol_from_smiles.side_effect = ValueError(
                    "Invalid SELFIES: test error"
                )

                with pytest.raises(ValueError, match="Invalid SELFIES: test error"):
                    parser.parse("[C][C][O]")

    def test_decoder_exception(self):
        """Test SELFIES parser handles decoder exceptions."""
        parser = SELFIESParser()

        # Mock sf.decoder to raise an exception
        with patch(
            "molecular_string_renderer.parsers.selfies.sf.decoder"
        ) as mock_decoder:
            mock_decoder.side_effect = RuntimeError("SELFIES decoder error")

            with pytest.raises(
                ValueError,
                match="Failed to parse SELFIES '\\[C\\]': SELFIES decoder error",
            ):
                parser.parse("[C]")

    def test_validation_exception_handling(self):
        """Test SELFIES validation with exception conditions."""
        parser = SELFIESParser()

        # Mock decoder to raise an exception during validation
        with patch(
            "molecular_string_renderer.parsers.selfies.sf.decoder"
        ) as mock_decoder:
            mock_decoder.side_effect = RuntimeError("Decoder exception")

            # Should return False instead of raising
            result = parser.validate("[C][C][O]")
            assert result is False
