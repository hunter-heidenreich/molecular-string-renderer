"""
Comprehensive tests for all molecular parsers.

Tests all supported formats: SMILES, InChI, SELFIES, and MOL files.
"""

import tempfile
from pathlib import Path

import pytest

from molecular_string_renderer.config import ParserConfig
from molecular_string_renderer.parsers import (
    InChIParser,
    MOLFileParser,
    SELFIESParser,
    SMILESParser,
    get_parser,
)
from tests.conftest import SAMPLE_MOLECULES


class TestSMILESParser:
    """Test SMILES parser functionality."""

    def test_valid_smiles_parsing(self):
        """Test parsing valid SMILES strings."""
        parser = SMILESParser()

        for name, smiles in SAMPLE_MOLECULES["smiles"].items():
            mol = parser.parse(smiles)
            assert mol is not None, f"Failed to parse {name}: {smiles}"
            assert mol.GetNumAtoms() > 0, f"No atoms in parsed molecule {name}"

    def test_valid_smiles_validation(self):
        """Test validation of valid SMILES strings."""
        parser = SMILESParser()

        for name, smiles in SAMPLE_MOLECULES["smiles"].items():
            assert parser.validate(smiles), f"Failed to validate {name}: {smiles}"

    def test_invalid_smiles_parsing(self):
        """Test parsing invalid SMILES strings raises ValueError."""
        parser = SMILESParser()

        for invalid_smiles in SAMPLE_MOLECULES["invalid_smiles"]:
            with pytest.raises(
                ValueError, match="(Invalid SMILES|cannot be empty|Failed to parse)"
            ):
                parser.parse(invalid_smiles)

    def test_invalid_smiles_validation(self):
        """Test validation of invalid SMILES strings returns False."""
        parser = SMILESParser()

        for invalid_smiles in SAMPLE_MOLECULES["invalid_smiles"]:
            assert not parser.validate(invalid_smiles), (
                f"Incorrectly validated invalid SMILES: {invalid_smiles}"
            )

    def test_smiles_with_config(self):
        """Test SMILES parsing with different configurations."""
        # Test with strict mode
        strict_config = ParserConfig(strict=True, sanitize=True)
        parser = SMILESParser(strict_config)

        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])
        assert mol is not None

        # Test with non-sanitizing config
        no_sanitize_config = ParserConfig(sanitize=False)
        parser = SMILESParser(no_sanitize_config)

        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])
        assert mol is not None

    def test_smiles_whitespace_handling(self):
        """Test SMILES parser handles whitespace correctly."""
        parser = SMILESParser()

        # Test with leading/trailing whitespace
        smiles_with_whitespace = "  CCO  "
        mol = parser.parse(smiles_with_whitespace)
        assert mol is not None
        assert parser.validate(smiles_with_whitespace)


class TestInChIParser:
    """Test InChI parser functionality."""

    def test_valid_inchi_parsing(self):
        """Test parsing valid InChI strings."""
        parser = InChIParser()

        for name, inchi in SAMPLE_MOLECULES["inchi"].items():
            mol = parser.parse(inchi)
            assert mol is not None, f"Failed to parse {name}: {inchi}"
            assert mol.GetNumAtoms() > 0, f"No atoms in parsed molecule {name}"

    def test_valid_inchi_validation(self):
        """Test validation of valid InChI strings."""
        parser = InChIParser()

        for name, inchi in SAMPLE_MOLECULES["inchi"].items():
            assert parser.validate(inchi), f"Failed to validate {name}: {inchi}"

    def test_invalid_inchi_parsing(self):
        """Test parsing invalid InChI strings raises ValueError."""
        parser = InChIParser()

        for invalid_inchi in SAMPLE_MOLECULES["invalid_inchi"]:
            with pytest.raises(
                ValueError,
                match="(Invalid InChI|cannot be empty|must start with|Failed to parse)",
            ):
                parser.parse(invalid_inchi)

    def test_invalid_inchi_validation(self):
        """Test validation of invalid InChI strings returns False."""
        parser = InChIParser()

        for invalid_inchi in SAMPLE_MOLECULES["invalid_inchi"]:
            assert not parser.validate(invalid_inchi), (
                f"Incorrectly validated invalid InChI: {invalid_inchi}"
            )

    def test_inchi_prefix_requirement(self):
        """Test that InChI strings must start with 'InChI=' prefix."""
        parser = InChIParser()

        # Valid InChI without prefix should fail
        invalid_no_prefix = "1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        with pytest.raises(ValueError, match="must start with"):
            parser.parse(invalid_no_prefix)

        assert not parser.validate(invalid_no_prefix)

    def test_inchi_whitespace_handling(self):
        """Test InChI parser handles whitespace correctly."""
        parser = InChIParser()

        # Test with leading/trailing whitespace
        inchi_with_whitespace = "  " + SAMPLE_MOLECULES["inchi"]["ethanol"] + "  "
        mol = parser.parse(inchi_with_whitespace)
        assert mol is not None
        assert parser.validate(inchi_with_whitespace)


class TestSELFIESParser:
    """Test SELFIES parser functionality."""

    def test_valid_selfies_parsing(self):
        """Test parsing valid SELFIES strings."""
        parser = SELFIESParser()

        for name, selfies in SAMPLE_MOLECULES["selfies"].items():
            mol = parser.parse(selfies)
            assert mol is not None, f"Failed to parse {name}: {selfies}"
            assert mol.GetNumAtoms() > 0, f"No atoms in parsed molecule {name}"

    def test_valid_selfies_validation(self):
        """Test validation of valid SELFIES strings."""
        parser = SELFIESParser()

        for name, selfies in SAMPLE_MOLECULES["selfies"].items():
            assert parser.validate(selfies), f"Failed to validate {name}: {selfies}"

    def test_invalid_selfies_parsing(self):
        """Test parsing invalid SELFIES strings raises ValueError."""
        parser = SELFIESParser()

        for invalid_selfies in SAMPLE_MOLECULES["invalid_selfies"]:
            with pytest.raises(
                ValueError, match="(Invalid SELFIES|cannot be empty|Failed to parse)"
            ):
                parser.parse(invalid_selfies)

    def test_invalid_selfies_validation(self):
        """Test validation of invalid SELFIES strings returns False."""
        parser = SELFIESParser()

        for invalid_selfies in SAMPLE_MOLECULES["invalid_selfies"]:
            assert not parser.validate(invalid_selfies), (
                f"Incorrectly validated invalid SELFIES: {invalid_selfies}"
            )

    def test_selfies_whitespace_handling(self):
        """Test SELFIES parser handles whitespace correctly."""
        parser = SELFIESParser()

        # Test with leading/trailing whitespace
        selfies_with_whitespace = "  " + SAMPLE_MOLECULES["selfies"]["ethanol"] + "  "
        mol = parser.parse(selfies_with_whitespace)
        assert mol is not None
        assert parser.validate(selfies_with_whitespace)


class TestMOLFileParser:
    """Test MOL file parser functionality."""

    def test_valid_mol_block_parsing(self):
        """Test parsing valid MOL block strings."""
        parser = MOLFileParser()

        for name, mol_block in SAMPLE_MOLECULES["mol_block"].items():
            mol = parser.parse(mol_block)
            assert mol is not None, f"Failed to parse {name}: {mol_block[:50]}..."
            assert mol.GetNumAtoms() > 0, f"No atoms in parsed molecule {name}"

    def test_valid_mol_block_validation(self):
        """Test validation of valid MOL block strings."""
        parser = MOLFileParser()

        for name, mol_block in SAMPLE_MOLECULES["mol_block"].items():
            assert parser.validate(mol_block), (
                f"Failed to validate {name}: {mol_block[:50]}..."
            )

    def test_invalid_mol_parsing(self):
        """Test parsing invalid MOL data raises ValueError."""
        parser = MOLFileParser()

        for invalid_mol in SAMPLE_MOLECULES["invalid_mol"]:
            with pytest.raises(
                ValueError, match="(Invalid MOL|cannot be empty|Failed to parse)"
            ):
                parser.parse(invalid_mol)

    def test_invalid_mol_validation(self):
        """Test validation of invalid MOL data returns False."""
        parser = MOLFileParser()

        for invalid_mol in SAMPLE_MOLECULES["invalid_mol"]:
            assert not parser.validate(invalid_mol), (
                f"Incorrectly validated invalid MOL: {invalid_mol}"
            )

    def test_mol_file_path_parsing(self):
        """Test parsing MOL data from file paths."""
        parser = MOLFileParser()

        # Create temporary MOL file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mol", delete=False
        ) as temp_file:
            temp_file.write(SAMPLE_MOLECULES["mol_block"]["ethanol"])
            temp_file_path = temp_file.name

        try:
            # Test parsing from file path
            mol = parser.parse(temp_file_path)
            assert mol is not None
            assert mol.GetNumAtoms() > 0

            # Test validation from file path
            assert parser.validate(temp_file_path)

        finally:
            # Clean up temporary file
            Path(temp_file_path).unlink()

    def test_mol_pathlib_support(self):
        """Test that MOL parser supports pathlib.Path objects."""
        parser = MOLFileParser()

        # Create temporary MOL file using pathlib
        with tempfile.TemporaryDirectory() as temp_dir:
            mol_file = Path(temp_dir) / "test.mol"
            mol_file.write_text(SAMPLE_MOLECULES["mol_block"]["ethanol"])

            # Test parsing with Path object
            mol = parser.parse(mol_file)
            assert mol is not None
            assert mol.GetNumAtoms() > 0

            # Test validation with Path object
            assert parser.validate(mol_file)


class TestParserFactory:
    """Test the parser factory function."""

    def test_supported_format_aliases(self):
        """Test all supported format aliases return correct parsers."""
        format_to_parser = {
            "smiles": SMILESParser,
            "smi": SMILESParser,
            "inchi": InChIParser,
            "mol": MOLFileParser,
            "sdf": MOLFileParser,
            "selfies": SELFIESParser,
        }

        for fmt, expected_parser_class in format_to_parser.items():
            parser = get_parser(fmt)
            assert isinstance(parser, expected_parser_class), (
                f"Format {fmt} returned wrong parser type"
            )

    def test_case_insensitive_format_handling(self):
        """Test that format types are case-insensitive."""
        test_cases = [
            ("SMILES", SMILESParser),
            ("Smiles", SMILESParser),
            ("INCHI", InChIParser),
            ("InChI", InChIParser),
            ("SELFIES", SELFIESParser),
            ("MOL", MOLFileParser),
        ]

        for fmt, expected_parser_class in test_cases:
            parser = get_parser(fmt)
            assert isinstance(parser, expected_parser_class), (
                f"Format {fmt} returned wrong parser type"
            )

    def test_whitespace_handling_in_format(self):
        """Test that format types handle whitespace correctly."""
        test_cases = [
            ("  smiles  ", SMILESParser),
            (" inchi ", InChIParser),
            ("selfies   ", SELFIESParser),
        ]

        for fmt, expected_parser_class in test_cases:
            parser = get_parser(fmt)
            assert isinstance(parser, expected_parser_class), (
                f"Format {fmt} returned wrong parser type"
            )

    def test_unsupported_format_raises_error(self):
        """Test that unsupported format types raise ValueError."""
        unsupported_formats = [
            "xyz",
            "pdb",
            "cml",
            "unsupported",
            "",
            "invalid_format",
        ]

        for fmt in unsupported_formats:
            with pytest.raises(ValueError, match="Unsupported format"):
                get_parser(fmt)

    def test_parser_config_propagation(self):
        """Test that parser config is properly passed to created parsers."""
        config = ParserConfig(strict=True, sanitize=False, remove_hs=False)

        parser = get_parser("smiles", config)
        assert parser.config.strict is True
        assert parser.config.sanitize is False
        assert parser.config.remove_hs is False


class TestParserConfiguration:
    """Test parser configuration options."""

    def test_default_config(self):
        """Test parsers work with default configuration."""
        parsers = [
            SMILESParser(),
            InChIParser(),
            SELFIESParser(),
            MOLFileParser(),
        ]

        test_data = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["inchi"]["ethanol"],
            SAMPLE_MOLECULES["selfies"]["ethanol"],
            SAMPLE_MOLECULES["mol_block"]["ethanol"],
        ]

        for parser, data in zip(parsers, test_data):
            mol = parser.parse(data)
            assert mol is not None

    def test_strict_config(self):
        """Test parsers with strict configuration."""
        strict_config = ParserConfig(strict=True)

        parsers = [
            SMILESParser(strict_config),
            InChIParser(strict_config),
            SELFIESParser(strict_config),
            MOLFileParser(strict_config),
        ]

        test_data = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["inchi"]["ethanol"],
            SAMPLE_MOLECULES["selfies"]["ethanol"],
            SAMPLE_MOLECULES["mol_block"]["ethanol"],
        ]

        for parser, data in zip(parsers, test_data):
            mol = parser.parse(data)
            assert mol is not None

    def test_no_sanitize_config(self):
        """Test parsers with sanitization disabled."""
        no_sanitize_config = ParserConfig(sanitize=False)

        parsers = [
            SMILESParser(no_sanitize_config),
            InChIParser(no_sanitize_config),
            SELFIESParser(no_sanitize_config),
            MOLFileParser(no_sanitize_config),
        ]

        test_data = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["inchi"]["ethanol"],
            SAMPLE_MOLECULES["selfies"]["ethanol"],
            SAMPLE_MOLECULES["mol_block"]["ethanol"],
        ]

        for parser, data in zip(parsers, test_data):
            mol = parser.parse(data)
            assert mol is not None

    def test_keep_hydrogens_config(self):
        """Test parsers with hydrogen removal disabled."""
        keep_hs_config = ParserConfig(remove_hs=False)

        parsers = [
            SMILESParser(keep_hs_config),
            InChIParser(keep_hs_config),
            SELFIESParser(keep_hs_config),
            MOLFileParser(keep_hs_config),
        ]

        test_data = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["inchi"]["ethanol"],
            SAMPLE_MOLECULES["selfies"]["ethanol"],
            SAMPLE_MOLECULES["mol_block"]["ethanol"],
        ]

        for parser, data in zip(parsers, test_data):
            mol = parser.parse(data)
            assert mol is not None


if __name__ == "__main__":
    pytest.main([__file__])
