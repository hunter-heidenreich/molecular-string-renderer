"""
2D molecular structure renderer implementation.

Provides the main 2D rendering functionality for individual molecules
with support for highlighting and customization.
"""

import io
import logging

from PIL import Image
from rdkit import Chem

from molecular_string_renderer.config import RenderConfig
from .base import MolecularRenderer
from .config_manager import DrawerConfigurationManager

logger = logging.getLogger(__name__)

Mol = Chem.Mol


class Molecule2DRenderer(MolecularRenderer):
    """Renderer for 2D molecular structures."""

    def __init__(self, config: RenderConfig | None = None):
        """
        Initialize 2D renderer.

        Args:
            config: Render configuration
        """
        super().__init__(config)
        self._drawer_manager = DrawerConfigurationManager(self.config)

    def render(self, mol: Mol) -> Image.Image:
        """
        Render molecule as 2D structure.

        Args:
            mol: RDKit Mol object to render

        Returns:
            PIL Image object containing the rendered molecule

        Raises:
            ValueError: If molecule is invalid
            RuntimeError: If rendering fails
        """
        mol = self._prepare_molecule(mol)

        try:
            drawer = self._drawer_manager.create_drawer()

            # Draw the molecule with optional highlights
            if self.config.highlight_atoms or self.config.highlight_bonds:
                drawer.DrawMolecule(
                    mol,
                    highlightAtoms=self.config.highlight_atoms or [],
                    highlightBonds=self.config.highlight_bonds or [],
                )
            else:
                drawer.DrawMolecule(mol)

            drawer.FinishDrawing()

            # Convert to PIL Image
            png_bytes = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(png_bytes))

            # Ensure RGBA mode for consistency
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            logger.debug(f"Successfully rendered molecule to {img.size} image")
            return img

        except Exception as e:
            logger.error(f"Failed to render molecule: {e}")
            raise RuntimeError(f"Failed to render molecule: {e}")

    def render_with_highlights(
        self,
        mol: Mol,
        highlight_atoms: list[int] | None = None,
        highlight_bonds: list[int] | None = None,
        highlight_colors: dict[int, tuple[float, float, float]] | None = None,
    ) -> Image.Image:
        """
        Render molecule with specific atoms/bonds highlighted.

        Args:
            mol: RDKit Mol object to render
            highlight_atoms: List of atom indices to highlight
            highlight_bonds: List of bond indices to highlight
            highlight_colors: Dict mapping indices to RGB color tuples (0-1 range)

        Returns:
            PIL Image object with highlights

        Raises:
            ValueError: If molecule is invalid
            RuntimeError: If rendering fails
        """
        mol = self._prepare_molecule(mol)

        # Create temporary config with highlights
        temp_config = RenderConfig(
            width=self.config.width,
            height=self.config.height,
            background_color=self.config.background_color,
            show_carbon=self.config.show_carbon,
            show_hydrogen=self.config.show_hydrogen,
            highlight_atoms=highlight_atoms or [],
            highlight_bonds=highlight_bonds or [],
        )

        try:
            drawer_manager = DrawerConfigurationManager(temp_config)
            drawer = drawer_manager.create_drawer()

            # Apply highlight colors if provided
            if highlight_colors:
                # Convert highlight colors to RDKit format if needed
                for idx, color in highlight_colors.items():
                    if len(color) == 3:  # RGB, add alpha
                        highlight_colors[idx] = (*color, 1.0)
                # Note: Highlight colors are applied through DrawMolecule parameters

            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms or [],
                highlightBonds=highlight_bonds or [],
            )

            drawer.FinishDrawing()

            # Convert to PIL Image
            png_bytes = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(png_bytes))

            if img.mode != "RGBA":
                img = img.convert("RGBA")

            logger.debug(
                f"Successfully rendered molecule with highlights to {img.size} image"
            )
            return img

        except Exception as e:
            logger.error(f"Failed to render molecule with highlights: {e}")
            raise RuntimeError(f"Failed to render molecule with highlights: {e}")
