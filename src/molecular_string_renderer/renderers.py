"""
Molecular renderer abstractions and implementations.

Provides flexible rendering of molecules to various formats.
"""

import logging
from abc import ABC, abstractmethod

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from molecular_string_renderer.config import RenderConfig

logger = logging.getLogger(__name__)

Mol = Chem.Mol


class MolecularRenderer(ABC):
    """Abstract base class for molecular renderers."""

    def __init__(self, config: RenderConfig | None = None):
        """Initialize renderer with configuration."""
        self.config = config or RenderConfig()

    @abstractmethod
    def render(self, mol: Mol) -> Image.Image:
        """
        Render a molecule to an image.

        Args:
            mol: RDKit Mol object to render

        Returns:
            PIL Image object
        """
        pass

    def _prepare_molecule(self, mol: Mol) -> Mol:
        """Prepare molecule for rendering (compute coordinates, etc.)."""
        if mol is None:
            raise ValueError("Cannot render None molecule")

        # Ensure 2D coordinates are computed
        try:
            rdDepictor.Compute2DCoords(mol)
        except Exception as e:
            raise ValueError(f"Failed to compute 2D coordinates: {e}")

        return mol


class Molecule2DRenderer(MolecularRenderer):
    """Renderer for 2D molecular structures."""

    def render(self, mol: Mol) -> Image.Image:
        """
        Render molecule as 2D structure.

        Args:
            mol: RDKit Mol object to render

        Returns:
            PIL Image object containing the rendered molecule
        """
        mol = self._prepare_molecule(mol)

        # Convert config to RDKit drawing options
        rdkit_options = self.config.to_rdkit_options()

        try:
            # Render using RDKit
            img = Draw.MolToImage(mol, size=self.config.size, **rdkit_options)

            # Ensure RGBA mode for consistency
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # Apply background color if not white
            if self.config.background_color.lower() != "white":
                background = Image.new("RGBA", img.size, self.config.background_color)
                img = Image.alpha_composite(background, img)

            return img

        except Exception as e:
            raise RuntimeError(f"Failed to render molecule: {e}")

    def render_with_highlights(
        self,
        mol: Mol,
        highlight_atoms: list | None = None,
        highlight_bonds: list | None = None,
        highlight_colors: dict | None = None,
    ) -> Image.Image:
        """
        Render molecule with specific atoms/bonds highlighted.

        Args:
            mol: RDKit Mol object to render
            highlight_atoms: List of atom indices to highlight
            highlight_bonds: List of bond indices to highlight
            highlight_colors: Dict mapping indices to colors

        Returns:
            PIL Image object with highlights
        """
        mol = self._prepare_molecule(mol)

        # Override config highlights temporarily
        original_atoms = self.config.highlight_atoms
        original_bonds = self.config.highlight_bonds

        self.config.highlight_atoms = highlight_atoms or []
        self.config.highlight_bonds = highlight_bonds or []

        try:
            rdkit_options = self.config.to_rdkit_options()

            # Add highlight colors if provided
            if highlight_colors:
                rdkit_options["highlightAtomColors"] = highlight_colors
                rdkit_options["highlightBondColors"] = highlight_colors

            img = Draw.MolToImage(mol, size=self.config.size, **rdkit_options)

            if img.mode != "RGBA":
                img = img.convert("RGBA")

            return img

        finally:
            # Restore original config
            self.config.highlight_atoms = original_atoms
            self.config.highlight_bonds = original_bonds


class MoleculeGridRenderer(MolecularRenderer):
    """Renderer for grids of multiple molecules."""

    def __init__(
        self,
        config: RenderConfig | None = None,
        mols_per_row: int = 4,
        mol_size: tuple[int, int] = (200, 200),
    ):
        """
        Initialize grid renderer.

        Args:
            config: Render configuration
            mols_per_row: Number of molecules per row in grid
            mol_size: Size of each individual molecule image
        """
        super().__init__(config)
        self.mols_per_row = mols_per_row
        self.mol_size = mol_size

    def render(self, mol: Mol) -> Image.Image:
        """Single molecule render - delegates to Molecule2DRenderer."""
        renderer = Molecule2DRenderer(self.config)
        return renderer.render(mol)

    def render_grid(self, mols: list, legends: list | None = None) -> Image.Image:
        """
        Render multiple molecules in a grid layout.

        Args:
            mols: List of RDKit Mol objects
            legends: Optional list of legend strings for each molecule

        Returns:
            PIL Image containing the molecule grid
        """
        if not mols:
            raise ValueError("Cannot render empty molecule list")

        try:
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=self.mols_per_row,
                subImgSize=self.mol_size,
                legends=legends,
            )

            if img.mode != "RGBA":
                img = img.convert("RGBA")

            return img

        except Exception as e:
            raise RuntimeError(f"Failed to render molecule grid: {e}")


def get_renderer(
    renderer_type: str = "2d", config: RenderConfig | None = None
) -> MolecularRenderer:
    """
    Factory function to get appropriate renderer.

    Args:
        renderer_type: Type of renderer ('2d', 'grid')
        config: Render configuration

    Returns:
        Appropriate renderer instance

    Raises:
        ValueError: If renderer type is not supported
    """
    renderer_type = renderer_type.lower().strip()

    renderers = {
        "2d": Molecule2DRenderer,
        "grid": MoleculeGridRenderer,
    }

    if renderer_type not in renderers:
        supported = list(renderers.keys())
        raise ValueError(
            f"Unsupported renderer: {renderer_type}. Supported: {supported}"
        )

    return renderers[renderer_type](config)
