"""
Parser factory functions.

Provides factory functions for creating appropriate parser instances.
"""

from molecular_string_renderer.config import ParserConfig
from molecular_string_renderer.parsers.base import MolecularParser
from molecular_string_renderer.parsers.inchi import InChIParser
from molecular_string_renderer.parsers.mol import MOLFileParser
from molecular_string_renderer.parsers.selfies import SELFIESParser
from molecular_string_renderer.parsers.smiles import SMILESParser


def get_parser(format_type: str, config: ParserConfig | None = None) -> MolecularParser:
    """
    Factory function to get appropriate parser for format type.

    Args:
        format_type: Type of molecular format ('smiles', 'inchi', 'mol')
        config: Parser configuration

    Returns:
        Appropriate parser instance

    Raises:
        ValueError: If format type is not supported
    """
    format_type = format_type.lower().strip()

    parsers = {
        "smiles": SMILESParser,
        "smi": SMILESParser,
        "inchi": InChIParser,
        "mol": MOLFileParser,
        "sdf": MOLFileParser,
        "selfies": SELFIESParser,
    }

    if format_type not in parsers:
        supported = list(parsers.keys())
        raise ValueError(f"Unsupported format: {format_type}. Supported: {supported}")

    return parsers[format_type](config)
