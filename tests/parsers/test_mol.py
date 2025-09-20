"""
Tests for the MOL file parser implementation.
"""

import pytest
from pathlib import Path
from rdkit import Chem
import tempfile
import os

from molecular_string_renderer.config import ParserConfig
from molecular_string_renderer.parsers.mol import MOLFileParser


class TestMOLFileParser:
    """Test the MOLFileParser class."""
    
    def test_initialization_default_config(self):
        """Test parser initializes with default config."""
        parser = MOLFileParser()
        assert parser.config is not None
        assert isinstance(parser.config, ParserConfig)
    
    def test_initialization_custom_config(self):
        """Test parser initializes with custom config."""
        config = ParserConfig(sanitize=False)
        parser = MOLFileParser(config)
        assert parser.config is config


class TestMOLFileParsingFromString:
    """Test MOL file parsing from string content."""
    
    def test_parse_simple_mol_block(self):
        """Test parsing a simple MOL block string."""
        parser = MOLFileParser()
        
        # Simple methane MOL block
        mol_block = """
  Mrv2311 12092414142D          

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
"""
        
        mol = parser.parse(mol_block)
        assert mol is not None
        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumAtoms() == 1
        assert mol.GetAtomWithIdx(0).GetSymbol() == 'C'
    
    def test_parse_multi_atom_mol_block(self):
        """Test parsing a multi-atom MOL block."""
        parser = MOLFileParser()
        
        # Ethanol MOL block
        mol_block = """
  Mrv2311 12092414142D          

  3  2  0  0  0  0            999 V2000
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5000    0.8660    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
"""
        
        mol = parser.parse(mol_block)
        assert mol is not None
        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumAtoms() == 3
        assert mol.GetNumBonds() == 2


class TestMOLFileParsingFromPath:
    """Test MOL file parsing from file paths."""
    
    def test_parse_from_path_object(self):
        """Test parsing MOL file using Path object."""
        parser = MOLFileParser()
        
        mol_content = """
  Mrv2311 12092414142D          

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as f:
            f.write(mol_content)
            temp_path = Path(f.name)
        
        try:
            mol = parser.parse(temp_path)
            assert mol is not None
            assert isinstance(mol, Chem.Mol)
        finally:
            os.unlink(temp_path)
    
    def test_parse_from_string_path(self):
        """Test parsing MOL file using string path."""
        parser = MOLFileParser()
        
        mol_content = """
  Mrv2311 12092414142D          

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as f:
            f.write(mol_content)
            temp_path = f.name
        
        try:
            mol = parser.parse(temp_path)
            assert mol is not None
            assert isinstance(mol, Chem.Mol)
        finally:
            os.unlink(temp_path)
    
    def test_parse_nonexistent_path_object(self):
        """Test parsing nonexistent file with Path object raises error."""
        parser = MOLFileParser()
        
        nonexistent_path = Path("/nonexistent/file.mol")
        
        with pytest.raises(ValueError, match="MOL file does not exist"):
            parser.parse(nonexistent_path)
    
    def test_parse_nonexistent_string_path(self):
        """Test parsing nonexistent file with string path."""
        parser = MOLFileParser()
        
        # This should be treated as MOL content, not a path, and fail differently
        nonexistent_path = "/this/path/does/not/exist.mol"
        
        with pytest.raises(ValueError, match="Invalid MOL data"):
            parser.parse(nonexistent_path)


class TestMOLFileParsingInvalid:
    """Test MOL file parsing with invalid inputs."""
    
    def test_parse_empty_string(self):
        """Test parsing empty MOL string raises error."""
        parser = MOLFileParser()
        
        with pytest.raises(ValueError, match="MOL data cannot be empty"):
            parser.parse("")
    
    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only string raises error."""
        parser = MOLFileParser()
        
        test_cases = ["   ", "\t", "\n", "\r\n"]
        
        for mol_data in test_cases:
            with pytest.raises(ValueError, match="MOL data cannot be empty"):
                parser.parse(mol_data)
    
    def test_parse_none(self):
        """Test parsing None raises error."""
        parser = MOLFileParser()
        
        with pytest.raises(ValueError, match="MOL data cannot be empty"):
            parser.parse(None)
    
    def test_parse_invalid_mol_format(self):
        """Test parsing invalid MOL format raises error."""
        parser = MOLFileParser()
        
        invalid_mol_blocks = [
            "invalid mol data",
            "not a mol block at all",
            "12345",
        ]
        
        for mol_data in invalid_mol_blocks:
            with pytest.raises(ValueError, match="Invalid MOL data"):
                parser.parse(mol_data)


class TestMOLFileValidation:
    """Test MOL file validation methods."""
    
    def test_validate_valid_mol_blocks(self):
        """Test validation with valid MOL blocks."""
        parser = MOLFileParser()
        
        valid_mol_block = """
  Mrv2311 12092414142D          

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
"""
        
        assert parser.validate(valid_mol_block) is True
    
    def test_validate_invalid_mol_blocks(self):
        """Test validation with invalid MOL blocks."""
        parser = MOLFileParser()
        
        invalid_mol_blocks = [
            "",
            "   ",
            None,
            "invalid data",
            "not a mol block",
        ]
        
        for mol_data in invalid_mol_blocks:
            assert parser.validate(mol_data) is False, f"Should not validate: {mol_data}"
    
    def test_validate_with_path_objects(self):
        """Test validation with Path objects."""
        parser = MOLFileParser()
        
        mol_content = """
  Mrv2311 12092414142D          

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
"""
        
        # Test valid file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as f:
            f.write(mol_content)
            temp_path = Path(f.name)
        
        try:
            assert parser.validate(temp_path) is True
        finally:
            os.unlink(temp_path)
        
        # Test nonexistent file
        nonexistent_path = Path("/nonexistent/file.mol")
        assert parser.validate(nonexistent_path) is False


class TestMOLFilePathDetection:
    """Test MOL file path detection logic."""
    
    def test_string_recognized_as_path(self):
        """Test that valid file paths are recognized and loaded."""
        parser = MOLFileParser()
        
        mol_content = """
  Mrv2311 12092414142D          

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as f:
            f.write(mol_content)
            temp_path = f.name
        
        try:
            # Should work when passed as string path
            mol = parser.parse(temp_path)
            assert mol is not None
            assert isinstance(mol, Chem.Mol)
        finally:
            os.unlink(temp_path)
    
    def test_string_with_mol_keywords_not_treated_as_path(self):
        """Test that strings containing MOL keywords are not treated as paths."""
        parser = MOLFileParser()
        
        # These should be treated as MOL content, not file paths
        mol_like_strings = [
            "something V2000 something",
            "data with V3000 in it",
            "contains M  END marker",
        ]
        
        for mol_string in mol_like_strings:
            # Should treat as MOL content and fail because it's invalid
            with pytest.raises(ValueError, match="Invalid MOL data"):
                parser.parse(mol_string)


class TestMOLFilePostProcessing:
    """Test MOL file parsing with different post-processing configurations."""
    
    def test_parse_with_hydrogen_removal(self):
        """Test parsing with hydrogen removal enabled."""
        config = ParserConfig(remove_hs=True)
        parser = MOLFileParser(config)
        
        mol_block = """
  Mrv2311 12092414142D          

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
"""
        
        mol = parser.parse(mol_block)
        assert mol is not None
        
        # Check that no explicit hydrogens are present
        has_explicit_h = any(atom.GetSymbol() == 'H' for atom in mol.GetAtoms())
        assert not has_explicit_h
    
    def test_parse_with_hydrogen_addition(self):
        """Test parsing with hydrogen addition enabled."""
        config = ParserConfig(remove_hs=False)
        parser = MOLFileParser(config)
        
        mol_block = """
  Mrv2311 12092414142D          

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
"""
        
        mol = parser.parse(mol_block)
        assert mol is not None
        
        # Should have explicit hydrogens added
        total_atoms = mol.GetNumAtoms()
        assert total_atoms > 1  # More than just carbon
    
    def test_parse_with_sanitization_disabled(self):
        """Test parsing with sanitization disabled."""
        config = ParserConfig(sanitize=False)
        parser = MOLFileParser(config)
        
        mol_block = """
  Mrv2311 12092414142D          

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
"""
        
        mol = parser.parse(mol_block)
        assert mol is not None
        assert isinstance(mol, Chem.Mol)
