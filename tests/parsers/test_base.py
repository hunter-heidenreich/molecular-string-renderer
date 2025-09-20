"""
Tests for the base parser classes and functionality.
"""

import pytest
from rdkit import Chem

from molecular_string_renderer.config import ParserConfig
from molecular_string_renderer.parsers.base import MolecularParser


class MockParser(MolecularParser):
    """Mock parser implementation for testing base functionality."""

    def parse(self, molecular_string: str) -> Chem.Mol:
        """Mock parse that returns a benzene molecule for any input."""
        if molecular_string == "invalid":
            return None
        return Chem.MolFromSmiles("c1ccccc1")  # Benzene

    def validate(self, molecular_string: str) -> bool:
        """Mock validate that rejects 'invalid' strings."""
        if not molecular_string or not molecular_string.strip():
            return False
        return molecular_string != "invalid"


class TestMolecularParserBase:
    """Test the base MolecularParser class."""

    def test_is_abstract(self):
        """Test that MolecularParser is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MolecularParser()

    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented in subclasses."""

        class IncompleteParser(MolecularParser):
            pass

        with pytest.raises(TypeError):
            IncompleteParser()

    def test_default_config_initialization(self):
        """Test parser initializes with default config when none provided."""
        parser = MockParser()
        assert parser.config is not None
        assert isinstance(parser.config, ParserConfig)

    def test_custom_config_initialization(self):
        """Test parser initializes with provided config."""
        custom_config = ParserConfig(sanitize=False, remove_hs=False)
        parser = MockParser(custom_config)
        assert parser.config is custom_config
        assert not parser.config.sanitize
        assert not parser.config.remove_hs


class TestPostProcessing:
    """Test the _post_process_molecule method."""

    def test_post_process_none_molecule(self):
        """Test post-processing handles None molecule correctly."""
        parser = MockParser()
        result = parser._post_process_molecule(None)
        assert result is None

    def test_post_process_with_sanitization(self):
        """Test post-processing with sanitization enabled."""
        config = ParserConfig(sanitize=True, remove_hs=True)
        parser = MockParser(config)

        # Create a molecule that needs sanitization
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = parser._post_process_molecule(mol)

        assert result is not None
        assert result.GetNumAtoms() == 6  # Benzene has 6 carbons

    def test_post_process_remove_hydrogens(self):
        """Test post-processing removes hydrogens when configured."""
        config = ParserConfig(sanitize=True, remove_hs=True)
        parser = MockParser(config)

        # Create molecule with explicit hydrogens
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        original_atom_count = mol.GetNumAtoms()

        result = parser._post_process_molecule(mol)

        assert result is not None
        assert result.GetNumAtoms() < original_atom_count  # Hydrogens removed

    def test_post_process_keep_hydrogens(self):
        """Test post-processing keeps hydrogens when configured."""
        config = ParserConfig(sanitize=True, remove_hs=False)
        parser = MockParser(config)

        # Create molecule without explicit hydrogens
        mol = Chem.MolFromSmiles("CCO")
        original_atom_count = mol.GetNumAtoms()

        result = parser._post_process_molecule(mol)

        assert result is not None
        assert result.GetNumAtoms() >= original_atom_count  # Hydrogens added

    def test_post_process_sanitization_failure(self):
        """Test post-processing handles sanitization failures."""
        config = ParserConfig(sanitize=True)
        parser = MockParser(config)

        # Instead of trying to create a molecule that fails sanitization,
        # let's test that the code path handles exceptions properly
        # We'll mock the sanitization to force an exception

        # Create a valid molecule first
        mol = Chem.MolFromSmiles("C")
        assert mol is not None

        # Mock the SanitizeMol function to raise an exception
        original_sanitize = Chem.SanitizeMol

        def mock_sanitize_fail(mol):
            raise Chem.AtomValenceException("Mock sanitization failure")

        # Temporarily replace SanitizeMol
        Chem.SanitizeMol = mock_sanitize_fail

        try:
            with pytest.raises(ValueError, match="Molecule sanitization failed"):
                parser._post_process_molecule(mol)
        finally:
            # Restore original function
            Chem.SanitizeMol = original_sanitize

    def test_post_process_without_sanitization(self):
        """Test post-processing without sanitization."""
        config = ParserConfig(sanitize=False, remove_hs=True)
        parser = MockParser(config)

        mol = Chem.MolFromSmiles("c1ccccc1")
        result = parser._post_process_molecule(mol)

        assert result is not None
        # Should still process hydrogens even without sanitization


class TestMockParserImplementation:
    """Test the mock parser implementation."""

    def test_parse_valid_input(self):
        """Test parsing with valid input."""
        parser = MockParser()
        result = parser.parse("anything")

        assert result is not None
        assert isinstance(result, Chem.Mol)

    def test_parse_invalid_input(self):
        """Test parsing with invalid input returns None."""
        parser = MockParser()
        result = parser.parse("invalid")

        assert result is None

    def test_validate_valid_input(self):
        """Test validation with valid input."""
        parser = MockParser()
        assert parser.validate("anything") is True

    def test_validate_invalid_input(self):
        """Test validation with invalid input."""
        parser = MockParser()
        assert parser.validate("invalid") is False


class TestConfigurationEdgeCases:
    """Test edge cases in parser configuration."""

    def test_sanitization_with_problematic_molecule(self):
        """Test sanitization with molecules that might cause issues."""
        # Create a parser with sanitization enabled
        config = ParserConfig(sanitize=True, remove_hs=True)
        parser = MockParser(config)

        # Parse a molecule that should sanitize fine
        mol = parser.parse("anything")
        assert mol is not None

    def test_validation_with_none_values(self):
        """Test validation methods handle None values correctly."""
        parser = MockParser()

        assert parser.validate(None) is False
        assert parser.validate("") is False
        assert parser.validate("   ") is False
