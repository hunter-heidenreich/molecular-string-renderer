"""
Molecular renderer module for 2D molecular structure visualization.

This module provides a comprehensive set of classes and utilities for rendering
molecular structures from RDKit Mol objects to images in various formats.

Classes:
    MolecularRenderer: Abstract base class for all renderers
    Molecule2DRenderer: 2D molecular structure renderer
    MoleculeGridRenderer: Grid layout renderer for multiple molecules
    ColorUtils: Utility class for color operations
    DrawerConfigurationManager: RDKit drawer configuration manager

Functions:
    get_renderer: Factory function for creating renderer instances

Example:
    >>> from molecular_string_renderer.renderers import get_renderer, RenderConfig
    >>> config = RenderConfig(width=600, height=400)
    >>> renderer = get_renderer("2d", config)
    >>> # Use renderer to render molecules...
"""

# Base classes and abstractions
from .base import MolecularRenderer

# Utilities and configuration
from .config_manager import DrawerConfigurationManager

# Factory functions
from .factory import get_renderer

# Concrete renderer implementations
from .grid import MoleculeGridRenderer
from .two_dimensional import Molecule2DRenderer
from .utils import ColorUtils

__all__ = [
    # Base classes
    "MolecularRenderer",
    # Concrete renderers
    "Molecule2DRenderer",
    "MoleculeGridRenderer",
    # Utilities
    "ColorUtils",
    "DrawerConfigurationManager",
    # Factory functions
    "get_renderer",
]
