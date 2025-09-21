"""
Tests for the parser factory functionality.
"""

import pytest

from molecular_string_renderer.config import ParserConfig
from molecular_string_renderer.parsers.base import MolecularParser
from molecular_string_renderer.parsers.factory import get_parser
from molecular_string_renderer.parsers.inchi import InChIParser
from molecular_string_renderer.parsers.mol import MOLFileParser
from molecular_string_renderer.parsers.selfies import SELFIESParser
from molecular_string_renderer.parsers.smiles import SMILESParser


class TestGetParser:
    """Test the get_parser factory function."""

    def test_get_smiles_parser(self):
        """Test getting SMILES parser."""
        parser = get_parser("smiles")

        assert isinstance(parser, SMILESParser)
        assert isinstance(parser, MolecularParser)

    def test_get_smi_parser(self):
        """Test getting SMILES parser with 'smi' alias."""
        parser = get_parser("smi")

        assert isinstance(parser, SMILESParser)
        assert isinstance(parser, MolecularParser)

    def test_get_inchi_parser(self):
        """Test getting InChI parser."""
        parser = get_parser("inchi")

        assert isinstance(parser, InChIParser)
        assert isinstance(parser, MolecularParser)

    def test_get_mol_parser(self):
        """Test getting MOL file parser."""
        parser = get_parser("mol")

        assert isinstance(parser, MOLFileParser)
        assert isinstance(parser, MolecularParser)

    def test_get_sdf_parser(self):
        """Test getting MOL file parser with 'sdf' alias."""
        parser = get_parser("sdf")

        assert isinstance(parser, MOLFileParser)
        assert isinstance(parser, MolecularParser)

    def test_get_selfies_parser(self):
        """Test getting SELFIES parser."""
        parser = get_parser("selfies")

        assert isinstance(parser, SELFIESParser)
        assert isinstance(parser, MolecularParser)

    def test_case_insensitive_format_type(self):
        """Test that format type is case insensitive."""
        test_cases = [
            ("SMILES", SMILESParser),
            ("Smiles", SMILESParser),
            ("sMiLeS", SMILESParser),
            ("INCHI", InChIParser),
            ("InChI", InChIParser),
            ("MOL", MOLFileParser),
            ("Mol", MOLFileParser),
            ("SELFIES", SELFIESParser),
            ("Selfies", SELFIESParser),
        ]

        for format_type, expected_class in test_cases:
            parser = get_parser(format_type)
            assert isinstance(parser, expected_class), (
                f"Failed for format: {format_type}"
            )

    def test_whitespace_handling(self):
        """Test that format type handles whitespace correctly."""
        test_cases = [
            "  smiles  ",
            "\tsmiles\t",
            "\nsmiles\n",
            " smiles ",
        ]

        for format_type in test_cases:
            parser = get_parser(format_type)
            assert isinstance(parser, SMILESParser), (
                f"Failed for format: '{format_type}'"
            )

    def test_unsupported_format_raises_error(self):
        """Test that unsupported format types raise ValueError."""
        unsupported_formats = [
            "xyz",
            "pdb",
            "unknown",
            "invalid",
            "",
        ]

        for format_type in unsupported_formats:
            with pytest.raises(ValueError, match="Unsupported format"):
                get_parser(format_type)

    def test_error_message_includes_supported_formats(self):
        """Test that error message includes list of supported formats."""
        with pytest.raises(ValueError) as exc_info:
            get_parser("unsupported")

        error_message = str(exc_info.value)
        assert "Supported:" in error_message
        assert "smiles" in error_message
        assert "inchi" in error_message
        assert "mol" in error_message
        assert "selfies" in error_message


class TestGetParserWithConfig:
    """Test get_parser with custom configuration."""

    def test_get_parser_with_default_config(self):
        """Test getting parser without providing config."""
        parser = get_parser("smiles")

        assert parser.config is not None
        assert isinstance(parser.config, ParserConfig)
        # Should use default values
        assert parser.config.sanitize is True
        assert parser.config.remove_hs is True

    def test_get_parser_with_custom_config(self):
        """Test getting parser with custom config."""
        custom_config = ParserConfig(sanitize=False, remove_hs=False)
        parser = get_parser("smiles", custom_config)

        assert parser.config is custom_config
        assert parser.config.sanitize is False
        assert parser.config.remove_hs is False

    def test_get_parser_with_none_config(self):
        """Test getting parser with None config (should use default)."""
        parser = get_parser("smiles", None)

        assert parser.config is not None
        assert isinstance(parser.config, ParserConfig)

    def test_config_passed_to_all_parser_types(self):
        """Test that custom config is passed to all parser types."""
        custom_config = ParserConfig(sanitize=False)

        parser_formats = ["smiles", "inchi", "mol", "selfies"]

        for format_type in parser_formats:
            parser = get_parser(format_type, custom_config)
            assert parser.config is custom_config, (
                f"Config not passed for {format_type}"
            )
            assert parser.config.sanitize is False, (
                f"Config values not preserved for {format_type}"
            )


class TestParserMapping:
    """Test the internal parser mapping."""

    def test_all_aliases_mapped(self):
        """Test that all expected aliases are properly mapped."""
        expected_mappings = {
            "smiles": SMILESParser,
            "smi": SMILESParser,
            "inchi": InChIParser,
            "mol": MOLFileParser,
            "sdf": MOLFileParser,
            "selfies": SELFIESParser,
        }

        for format_type, expected_class in expected_mappings.items():
            parser = get_parser(format_type)
            assert isinstance(parser, expected_class), (
                f"Mapping failed for {format_type}"
            )

    def test_no_duplicate_instances(self):
        """Test that each call to get_parser returns a new instance."""
        parser1 = get_parser("smiles")
        parser2 = get_parser("smiles")

        assert parser1 is not parser2, "get_parser should return new instances"
        assert type(parser1) is type(parser2), "Parsers should be same type"

    def test_different_configs_create_different_instances(self):
        """Test that different configs create properly configured instances."""
        config1 = ParserConfig(sanitize=True)
        config2 = ParserConfig(sanitize=False)

        parser1 = get_parser("smiles", config1)
        parser2 = get_parser("smiles", config2)

        assert parser1.config.sanitize is True
        assert parser2.config.sanitize is False
        assert parser1 is not parser2
