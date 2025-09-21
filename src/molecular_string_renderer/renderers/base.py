"""
Base classes and abstractions for molecular renderers.

Provides the foundational abstract base class and common functionality
that all molecular renderers inherit from.
"""

import logging
from abc import ABC, abstractmethod

from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor

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
        """
        Prepare molecule for rendering (compute coordinates, etc.).

        Args:
            mol: RDKit Mol object to prepare

        Returns:
            Prepared molecule ready for rendering

        Raises:
            ValueError: If molecule is None or invalid
        """
        if mol is None:
            raise ValueError("Cannot render None molecule")

        # Create a copy to avoid modifying the original molecule
        try:
            mol = Chem.Mol(mol)
        except Exception as e:
            raise ValueError(f"Failed to create molecule copy: {e}")

        # Ensure 2D coordinates are computed
        try:
            rdDepictor.Compute2DCoords(mol)
            logger.debug("Computed 2D coordinates for molecule")
        except Exception as e:
            logger.error(f"Failed to compute 2D coordinates: {e}")
            raise ValueError(f"Failed to compute 2D coordinates: {e}")

        # Add carbon labels if requested
        if self.config.show_carbon:
            try:
                carbon_count = 0
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == "C":
                        atom.SetProp("atomLabel", "C")
                        carbon_count += 1
                logger.debug(f"Added carbon labels to {carbon_count} atoms")
            except Exception as e:
                logger.warning(f"Failed to add carbon labels: {e}")
                # Non-critical error, continue without carbon labels

        return mol

    def _get_molecule_dimensions(self, mol: Mol) -> tuple[float, float]:
        """
        Calculate molecule dimensions from 2D coordinates.

        Args:
            mol: RDKit Mol object with 2D coordinates

        Returns:
            tuple: (width, height) in RDKit coordinate units

        Raises:
            ValueError: If molecule has no valid coordinates
        """
        if mol is None:
            raise ValueError("Cannot calculate dimensions for None molecule")

        if mol.GetNumConformers() == 0:
            logger.warning("Molecule has no conformers, using default dimensions")
            return 1.0, 1.0

        try:
            # Get the conformer (2D coordinates)
            conf = mol.GetConformer()

            # Get atom positions
            positions = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                positions.append((pos.x, pos.y))

            if not positions:
                logger.warning("No atom positions found, using default dimensions")
                return 1.0, 1.0

            # For single atom molecules, use a default size
            if len(positions) == 1:
                logger.debug("Single atom molecule, using default dimensions")
                return 1.0, 1.0

            # Calculate bounding box
            min_x = min(pos[0] for pos in positions)
            max_x = max(pos[0] for pos in positions)
            min_y = min(pos[1] for pos in positions)
            max_y = max(pos[1] for pos in positions)

            width = max_x - min_x
            height = max_y - min_y

            # Return dimensions with minimum values to avoid zero
            final_width = max(width, 0.5)
            final_height = max(height, 0.5)

            logger.debug(
                f"Calculated molecule dimensions: {final_width:.2f} x {final_height:.2f}"
            )
            return final_width, final_height

        except Exception as e:
            logger.error(f"Failed to calculate molecule dimensions: {e}")
            raise ValueError(f"Failed to calculate molecule dimensions: {e}")
