"""
Integration tests for the parsers sub-module.

Tests that the new sub-module structure integrates correctly with the rest of the codebase.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from molecular_string_renderer.parsers import (
    MolecularParser,
    SMILESParser,
    InChIParser,
    MOLFileParser,
    SELFIESParser,
    get_parser,
)
from molecular_string_renderer.config import ParserConfig


class TestSubModuleImports:
    """Test that all imports work correctly from the sub-module."""

    def test_import_base_parser(self):
        """Test importing base parser class."""
        assert MolecularParser is not None
        assert hasattr(MolecularParser, "parse")
        assert hasattr(MolecularParser, "validate")

    def test_import_all_parser_classes(self):
        """Test importing all parser implementation classes."""
        parser_classes = [SMILESParser, InChIParser, MOLFileParser, SELFIESParser]

        for parser_class in parser_classes:
            assert parser_class is not None
            assert issubclass(parser_class, MolecularParser)

    def test_import_factory_function(self):
        """Test importing factory function."""
        assert get_parser is not None
        assert callable(get_parser)


class TestSubModuleBackwardCompatibility:
    """Test that the sub-module maintains backward compatibility."""

    def test_legacy_imports_still_work(self):
        """Test that old import patterns still work."""
        # These imports should work exactly as they did before
        from molecular_string_renderer.parsers import SMILESParser as LegacySMILESParser
        from molecular_string_renderer.parsers import get_parser as legacy_get_parser

        # Should be the same classes
        assert LegacySMILESParser is SMILESParser
        assert legacy_get_parser is get_parser

    def test_factory_function_compatibility(self):
        """Test that factory function maintains same API."""
        # Should work exactly as before
        parser = get_parser("smiles")
        assert isinstance(parser, SMILESParser)

        # With config
        config = ParserConfig(sanitize=False)
        parser_with_config = get_parser("smiles", config)
        assert isinstance(parser_with_config, SMILESParser)
        assert parser_with_config.config is config

    def test_core_module_integration(self):
        """Test that core module can still import and use parsers."""
        # Import as core module would
        from molecular_string_renderer.parsers import get_parser

        # Use as core module would
        parser = get_parser("smiles")
        mol = parser.parse("CCO")
        assert mol is not None
        assert isinstance(mol, Chem.Mol)


class TestSubModuleFunctionality:
    """Test that all functionality works correctly after refactoring."""

    def test_all_parsers_work_independently(self):
        """Test that each parser works correctly in isolation."""
        test_cases = [
            (SMILESParser(), "CCO"),
            (InChIParser(), "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"),
            (SELFIESParser(), "[C][C][O]"),
        ]

        for parser, test_input in test_cases:
            mol = parser.parse(test_input)
            assert mol is not None, f"Parser {type(parser).__name__} failed"
            assert isinstance(mol, Chem.Mol)
            assert parser.validate(test_input) is True

    def test_factory_creates_correct_parsers(self):
        """Test that factory creates the correct parser types."""
        test_cases = [
            ("smiles", SMILESParser),
            ("smi", SMILESParser),
            ("inchi", InChIParser),
            ("mol", MOLFileParser),
            ("sdf", MOLFileParser),
            ("selfies", SELFIESParser),
        ]

        for format_type, expected_class in test_cases:
            parser = get_parser(format_type)
            assert isinstance(parser, expected_class)
            assert isinstance(parser, MolecularParser)

    def test_configuration_propagation(self):
        """Test that configuration is properly propagated through sub-module."""
        custom_config = ParserConfig(sanitize=False, remove_hs=False)

        # Test with factory
        parser = get_parser("smiles", custom_config)
        assert parser.config is custom_config
        assert parser.config.sanitize is False
        assert parser.config.remove_hs is False

        # Test direct instantiation
        direct_parser = SMILESParser(custom_config)
        assert direct_parser.config is custom_config

    def test_error_handling_consistency(self):
        """Test that error handling is consistent across all parsers."""
        parsers = [
            (SMILESParser(), ""),
            (InChIParser(), ""),
            (SELFIESParser(), ""),
            (MOLFileParser(), ""),
        ]

        for parser, invalid_input in parsers:
            with pytest.raises(ValueError):
                parser.parse(invalid_input)

            assert parser.validate(invalid_input) is False


class TestSubModulePerformance:
    """Test that the sub-module refactoring doesn't impact performance."""

    def test_import_speed(self):
        """Test that imports are reasonably fast."""
        import time

        start_time = time.time()

        # Re-import to test fresh import time
        _ = __import__("molecular_string_renderer.parsers", fromlist=[""])

        import_time = time.time() - start_time

        # Should import in reasonable time (less than 1 second)
        assert import_time < 1.0, f"Import took too long: {import_time:.3f} seconds"

    def test_parser_creation_speed(self):
        """Test that parser creation is reasonably fast."""
        import time

        start_time = time.time()

        # Create multiple parsers
        for _ in range(100):
            get_parser("smiles")

        creation_time = time.time() - start_time

        # Should create parsers quickly
        assert creation_time < 1.0, (
            f"Parser creation took too long: {creation_time:.3f} seconds"
        )


class TestSubModuleStructure:
    """Test the internal structure of the sub-module."""

    def test_module_has_correct_exports(self):
        """Test that the sub-module exports the correct symbols."""
        import molecular_string_renderer.parsers as parsers_module

        expected_exports = [
            "MolecularParser",
            "SMILESParser",
            "InChIParser",
            "MOLFileParser",
            "SELFIESParser",
            "get_parser",
        ]

        for export in expected_exports:
            assert hasattr(parsers_module, export), f"Missing export: {export}"

        # Check __all__ if it exists
        if hasattr(parsers_module, "__all__"):
            for export in expected_exports:
                assert export in parsers_module.__all__, (
                    f"Export {export} not in __all__"
                )

    def test_individual_modules_importable(self):
        """Test that individual sub-modules can be imported directly."""
        # These should all work without errors
        from molecular_string_renderer.parsers.base import (
            MolecularParser as BaseMolecularParser,
        )
        from molecular_string_renderer.parsers.smiles import (
            SMILESParser as BaseSMILESParser,
        )
        from molecular_string_renderer.parsers.factory import (
            get_parser as base_get_parser,
        )

        # Verify these are the same objects as the main imports
        from molecular_string_renderer.parsers import (
            MolecularParser as MainMolecularParser,
            SMILESParser as MainSMILESParser,
            get_parser as main_get_parser,
        )

        assert BaseMolecularParser is MainMolecularParser
        assert BaseSMILESParser is MainSMILESParser
        assert base_get_parser is main_get_parser


class TestDocumentationAndMetadata:
    """Test that documentation and metadata are preserved."""

    def test_parser_classes_have_docstrings(self):
        """Test that all parser classes have proper docstrings."""
        parser_classes = [SMILESParser, InChIParser, MOLFileParser, SELFIESParser]

        for parser_class in parser_classes:
            assert parser_class.__doc__ is not None, (
                f"{parser_class.__name__} missing docstring"
            )
            assert len(parser_class.__doc__.strip()) > 0, (
                f"{parser_class.__name__} has empty docstring"
            )

    def test_methods_have_docstrings(self):
        """Test that key methods have proper docstrings."""
        parser = SMILESParser()

        methods_to_check = ["parse", "validate"]

        for method_name in methods_to_check:
            method = getattr(parser, method_name)
            assert method.__doc__ is not None, f"Method {method_name} missing docstring"
            assert len(method.__doc__.strip()) > 0, (
                f"Method {method_name} has empty docstring"
            )

    def test_factory_function_has_docstring(self):
        """Test that factory function has proper docstring."""
        assert get_parser.__doc__ is not None
        assert len(get_parser.__doc__.strip()) > 0
        assert (
            "factory" in get_parser.__doc__.lower() or "Factory" in get_parser.__doc__
        )


class TestCrossFormatValidation:
    """Test consistency between different molecular formats."""

    def test_smiles_to_inchi_consistency(self):
        """Test that SMILES can be converted to InChI and back."""
        smiles_parser = SMILESParser()
        inchi_parser = InChIParser()

        # Simple molecule
        original_smiles = "CCO"
        mol1 = smiles_parser.parse(original_smiles)

        # Convert to InChI
        inchi = Chem.MolToInchi(mol1)

        # Parse the InChI
        mol2 = inchi_parser.parse(inchi)

        # Both should represent the same molecule
        assert mol1 is not None
        assert mol2 is not None

        # Compare molecular formulas
        formula1 = rdMolDescriptors.CalcMolFormula(mol1)
        formula2 = rdMolDescriptors.CalcMolFormula(mol2)
        assert formula1 == formula2

    def test_selfies_to_smiles_consistency(self):
        """Test that SELFIES can be converted to SMILES consistently."""
        selfies_parser = SELFIESParser()
        smiles_parser = SMILESParser()

        # Use SELFIES that we know should work
        selfies_string = "[C][C][O]"
        mol1 = selfies_parser.parse(selfies_string)

        # Convert back to SMILES
        canonical_smiles = Chem.MolToSmiles(mol1)
        mol2 = smiles_parser.parse(canonical_smiles)

        # Both should represent the same molecule
        assert mol1 is not None
        assert mol2 is not None

        # Compare molecular formulas
        formula1 = rdMolDescriptors.CalcMolFormula(mol1)
        formula2 = rdMolDescriptors.CalcMolFormula(mol2)
        assert formula1 == formula2


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""

    def test_many_small_molecules_parsing(self):
        """Test parsing many small molecules doesn't leak memory."""
        parser = SMILESParser()

        # Parse many small molecules
        small_molecules = ["C", "CC", "CCC", "CCCC", "CCO", "CCC(=O)O"] * 100

        parsed_mols = []
        for smiles in small_molecules:
            mol = parser.parse(smiles)
            assert mol is not None
            parsed_mols.append(mol)

        # Should have parsed all molecules successfully
        assert len(parsed_mols) == 600
