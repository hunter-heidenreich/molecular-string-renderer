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

from molecular_string_renderer.config import RenderConfig, ParserConfig, OutputConfig
from molecular_string_renderer.core import (
    render_molecule,
    render_molecules_grid,
    validate_molecular_string,
    MolecularRenderer,
)
from molecular_string_renderer.outputs import PNGOutput, SVGOutput
from molecular_string_renderer.parsers import SMILESParser
from molecular_string_renderer.renderers import Molecule2DRenderer

__version__ = "0.1.0"
__author__ = "Hunter Heidenreich"

__all__ = [
    "render_molecule",
    "render_molecules_grid",
    "validate_molecular_string",
    "MolecularRenderer",
    "SMILESParser",
    "Molecule2DRenderer",
    "PNGOutput",
    "SVGOutput",
    "RenderConfig",
    "ParserConfig",
    "OutputConfig",
]
