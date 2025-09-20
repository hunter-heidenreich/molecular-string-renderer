"""
Tests for the SMILES parser implementation.
"""

import pytest
from rdkit import Chem

from molecular_string_renderer.config import ParserConfig
from molecular_string_renderer.parsers.smiles import SMILESParser


class TestSMILESParser:
    """Test the SMILESParser class."""
    
    def test_initialization_default_config(self):
        """Test parser initializes with default config."""
        parser = SMILESParser()
        assert parser.config is not None
        assert isinstance(parser.config, ParserConfig)
    
    def test_initialization_custom_config(self):
        """Test parser initializes with custom config."""
        config = ParserConfig(sanitize=False)
        parser = SMILESParser(config)
        assert parser.config is config


class TestSMILESParsingValid:
    """Test SMILES parsing with valid inputs."""
    
    def test_parse_simple_molecules(self):
        """Test parsing simple SMILES strings."""
        parser = SMILESParser()
        
        test_cases = [
            ("CCO", "ethanol"),
            ("c1ccccc1", "benzene"),
            ("CC(=O)O", "acetic acid"),
            ("CCN", "ethylamine"),
            ("C", "methane"),
        ]
        
        for smiles, description in test_cases:
            mol = parser.parse(smiles)
            assert mol is not None, f"Failed to parse {description}: {smiles}"
            assert isinstance(mol, Chem.Mol)
    
    def test_parse_with_whitespace(self):
        """Test parsing SMILES with leading/trailing whitespace."""
        parser = SMILESParser()
        
        test_cases = [
            "  CCO  ",
            "\tC\t",
            "\nCC(=O)O\n",
            " c1ccccc1 ",
        ]
        
        for smiles in test_cases:
            mol = parser.parse(smiles)
            assert mol is not None, f"Failed to parse: '{smiles}'"
    
    def test_parse_complex_molecules(self):
        """Test parsing more complex SMILES strings."""
        parser = SMILESParser()
        
        test_cases = [
            "CC(C)C(=O)N[C@@H](CC1=CNC2=CC=CC=C21)C(=O)O",  # Tryptophan
            "CC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)N(C)C",  # Crystal violet precursor
            "C1=CC=C2C(=C1)C=CC=C2",  # Naphthalene
        ]
        
        for smiles in test_cases:
            mol = parser.parse(smiles)
            assert mol is not None, f"Failed to parse complex molecule: {smiles}"
    
    def test_parse_stereochemistry(self):
        """Test parsing SMILES with stereochemistry."""
        parser = SMILESParser()
        
        test_cases = [
            "C[C@H](N)C(=O)O",  # L-alanine
            "C[C@@H](N)C(=O)O",  # D-alanine
            "C/C=C/C",  # Trans-2-butene
            "C/C=C\\C",  # Cis-2-butene
        ]
        
        for smiles in test_cases:
            mol = parser.parse(smiles)
            assert mol is not None, f"Failed to parse stereochemistry: {smiles}"


class TestSMILESParsingInvalid:
    """Test SMILES parsing with invalid inputs."""
    
    def test_parse_empty_string(self):
        """Test parsing empty SMILES string raises error."""
        parser = SMILESParser()
        
        with pytest.raises(ValueError, match="SMILES string cannot be empty"):
            parser.parse("")
    
    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only string raises error."""
        parser = SMILESParser()
        
        test_cases = ["   ", "\t", "\n", "\r\n"]
        
        for smiles in test_cases:
            with pytest.raises(ValueError, match="SMILES string cannot be empty"):
                parser.parse(smiles)
    
    def test_parse_none(self):
        """Test parsing None raises error."""
        parser = SMILESParser()
        
        with pytest.raises(ValueError, match="SMILES string cannot be empty"):
            parser.parse(None)
    
    def test_parse_invalid_smiles(self):
        """Test parsing invalid SMILES strings raises error."""
        parser = SMILESParser()
        
        invalid_smiles = [
            "C[",  # Unclosed bracket
            "C(",  # Unclosed parenthesis
            "[X][X][X]",  # Invalid atom symbols
            "C1CC1C1CC1CC1",  # Invalid ring closure pattern
        ]
        
        for smiles in invalid_smiles:
            with pytest.raises(ValueError, match="Invalid SMILES string"):
                parser.parse(smiles)


class TestSMILESValidation:
    """Test SMILES validation methods."""
    
    def test_validate_valid_smiles(self):
        """Test validation with valid SMILES strings."""
        parser = SMILESParser()
        
        valid_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "C",
            "CC(C)C",
        ]
        
        for smiles in valid_smiles:
            assert parser.validate(smiles) is True, f"Should validate: {smiles}"
    
    def test_validate_invalid_smiles(self):
        """Test validation with invalid SMILES strings."""
        parser = SMILESParser()
        
        invalid_smiles = [
            "",
            "   ",
            None,
            "C[",
            "C(",
            "X",
        ]
        
        for smiles in invalid_smiles:
            assert parser.validate(smiles) is False, f"Should not validate: {smiles}"
    
    def test_validate_with_whitespace(self):
        """Test validation handles whitespace correctly."""
        parser = SMILESParser()
        
        test_cases = [
            ("  CCO  ", True),
            ("\tC\t", True),
            ("  ", False),
        ]
        
        for smiles, expected in test_cases:
            result = parser.validate(smiles)
            assert result is expected, f"Validation failed for: '{smiles}'"


class TestSMILESPostProcessing:
    """Test SMILES parsing with different post-processing configurations."""
    
    def test_parse_with_hydrogen_removal(self):
        """Test parsing with hydrogen removal enabled."""
        config = ParserConfig(remove_hs=True)
        parser = SMILESParser(config)
        
        mol = parser.parse("CCO")
        assert mol is not None
        
        # Check that no explicit hydrogens are present
        has_explicit_h = any(atom.GetSymbol() == 'H' for atom in mol.GetAtoms())
        assert not has_explicit_h
    
    def test_parse_with_hydrogen_addition(self):
        """Test parsing with hydrogen addition enabled."""
        config = ParserConfig(remove_hs=False)
        parser = SMILESParser(config)
        
        mol = parser.parse("C")  # Methane
        assert mol is not None
        
        # Should have explicit hydrogens added
        total_atoms = mol.GetNumAtoms()
        assert total_atoms > 1  # More than just carbon
    
    def test_parse_with_sanitization_disabled(self):
        """Test parsing with sanitization disabled."""
        config = ParserConfig(sanitize=False)
        parser = SMILESParser(config)
        
        mol = parser.parse("CCO")
        assert mol is not None
        assert isinstance(mol, Chem.Mol)
    
    def test_parse_error_message_formatting(self):
        """Test that error messages contain the problematic SMILES."""
        parser = SMILESParser()
        
        invalid_smiles = "C["
        
        with pytest.raises(ValueError) as exc_info:
            parser.parse(invalid_smiles)
        
        error_message = str(exc_info.value)
        assert invalid_smiles in error_message
