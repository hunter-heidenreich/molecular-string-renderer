"""
Molecular String Renderer
=========================

A flexible library for rendering molecular structures from various string representations.

This library provides a modular architecture for:
- Parsing molecular representations (SMILES, InChI, etc.)
- Rendering molecules as 2D images
- Flexible output formats and configurations

Example usage:
    >>> from molecular_string_renderer import render_molecule
    >>> image = render_molecule("CCO", format_type="smiles", output_format="png")
    >>> image.save("ethanol.png")
"""

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.core import (
    get_supported_formats,
    render_molecule,
    render_molecules_grid,
    validate_molecular_string,
)
from molecular_string_renderer.exceptions import (
    ConfigurationError,
    MolecularRenderingError,
    OutputError,
    ParsingError,
    RenderingError,
    ValidationError,
)
from molecular_string_renderer.outputs import PNGOutput, SVGOutput
from molecular_string_renderer.parsers import SMILESParser
from molecular_string_renderer.renderer import MolecularRenderer
from molecular_string_renderer.renderers import Molecule2DRenderer

__version__ = "0.2.0"
__author__ = "Hunter Heidenreich"

__all__ = [
    "render_molecule",
    "render_molecules_grid",
    "validate_molecular_string",
    "get_supported_formats",
    "MolecularRenderer",
    "SMILESParser",
    "Molecule2DRenderer",
    "PNGOutput",
    "SVGOutput",
    "RenderConfig",
    "ParserConfig",
    "OutputConfig",
    # Exception types
    "MolecularRenderingError",
    "ValidationError",
    "ParsingError",
    "RenderingError",
    "OutputError",
    "ConfigurationError",
]
