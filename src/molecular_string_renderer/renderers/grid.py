"""
Grid layout renderer for multiple molecules.

Provides functionality to render multiple molecules in a grid layout
with support for legends and highlighting.
"""

import logging

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from molecular_string_renderer.config import RenderConfig

from .base import MolecularRenderer
from .config_manager import DrawerConfigurationManager

logger = logging.getLogger(__name__)

Mol = Chem.Mol


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

        Raises:
            ValueError: If mols_per_row is less than 1 or mol_size is invalid
        """
        super().__init__(config)

        if mols_per_row < 1:
            raise ValueError("mols_per_row must be at least 1")
        if not mol_size or len(mol_size) != 2 or any(s <= 0 for s in mol_size):
            raise ValueError(
                "mol_size must be a tuple of positive integers (width, height)"
            )

        self.mols_per_row = mols_per_row
        self.mol_size = mol_size

    def _create_draw_options(self) -> Draw.MolDrawOptions:
        """
        Create RDKit MolDrawOptions from RenderConfig using DrawerConfigurationManager.

        Returns:
            MolDrawOptions configured based on self.config settings

        Note:
            This uses the same configuration approach as Molecule2DRenderer for consistency.
        """
        # Create a temporary drawer to get properly configured options
        config_manager = DrawerConfigurationManager(self.config)
        drawer = config_manager.create_drawer()
        
        # Get the configured options from the drawer
        draw_opts = drawer.drawOptions()
        
        logger.debug("Created MolDrawOptions using DrawerConfigurationManager")
        return draw_opts

    def render(self, mol: Mol) -> Image.Image:
        """
        Render single molecule using grid renderer (fallback to single-cell grid).

        Args:
            mol: RDKit Mol object to render

        Returns:
            PIL Image object containing the rendered molecule

        Note:
            For single molecules, consider using Molecule2DRenderer directly for better performance.
        """
        logger.debug("Rendering single molecule as 1x1 grid")
        return self.render_grid([mol])

    def render_grid(
        self, mols: list[Mol], legends: list[str] | None = None
    ) -> Image.Image:
        """
        Render multiple molecules in a grid layout.

        Args:
            mols: List of RDKit Mol objects
            legends: Optional list of legend strings for each molecule

        Returns:
            PIL Image containing the molecule grid

        Raises:
            ValueError: If molecule list is empty or legends count doesn't match
            RuntimeError: If grid rendering fails
        """
        if not mols:
            raise ValueError("Cannot render empty molecule list")

        # Validate molecules
        valid_mols = []
        for i, mol in enumerate(mols):
            if mol is None:
                logger.warning(f"Skipping None molecule at index {i}")
                continue
            try:
                # Prepare molecule to ensure it's valid
                prepared_mol = self._prepare_molecule(mol)
                valid_mols.append(prepared_mol)
            except Exception as e:
                logger.warning(f"Skipping invalid molecule at index {i}: {e}")
                continue

        if not valid_mols:
            raise ValueError("No valid molecules found in input list")

        # Handle legends
        if legends is not None:
            if len(legends) != len(mols):
                logger.warning(
                    f"Legend count ({len(legends)}) doesn't match molecule count ({len(mols)}). "
                    "Legends will be truncated or ignored."
                )
            # Only keep legends for valid molecules
            if len(valid_mols) != len(mols):
                legends = None  # Disable legends if molecules were filtered
                logger.warning("Disabling legends due to filtered molecules")

        try:
            logger.debug(
                f"Rendering grid of {len(valid_mols)} molecules, {self.mols_per_row} per row"
            )

            # Create draw options from configuration
            draw_opts = self._create_draw_options()

            img = Draw.MolsToGridImage(
                valid_mols,
                molsPerRow=self.mols_per_row,
                subImgSize=self.mol_size,
                legends=legends,
                drawOptions=draw_opts,
            )

            # Handle both PIL Image and IPython Display Image objects
            if isinstance(img, Image.Image):
                # Standard PIL Image - use directly
                pil_img = img
            else:
                # Check if it's an IPython Display Image (common in Jupyter notebooks)
                try:
                    # Try to access the data attribute of IPython.core.display.Image
                    if hasattr(img, 'data'):
                        from io import BytesIO
                        # Convert bytes data to PIL Image
                        pil_img = Image.open(BytesIO(img.data))
                        logger.debug("Converted IPython Display Image to PIL Image")
                    else:
                        raise RuntimeError(f"RDKit returned unexpected object type: {type(img)}")
                except Exception as e:
                    logger.error(f"Failed to convert RDKit output to PIL Image: {e}")
                    raise RuntimeError(f"RDKit failed to render molecule grid to image: {type(img)}")

            if pil_img.mode != "RGBA":
                pil_img = pil_img.convert("RGBA")

            logger.debug(f"Successfully rendered molecule grid to {pil_img.size} image")
            return pil_img

        except Exception as e:
            logger.error(f"Failed to render molecule grid: {e}")
            raise RuntimeError(f"Failed to render molecule grid: {e}")

    def render_grid_with_highlights(
        self,
        mols: list[Mol],
        highlight_atoms_list: list[list[int]] | None = None,
        highlight_bonds_list: list[list[int]] | None = None,
        legends: list[str] | None = None,
    ) -> Image.Image:
        """
        Render molecule grid with per-molecule highlights.

        Args:
            mols: List of RDKit Mol objects
            highlight_atoms_list: List of atom highlight lists for each molecule
            highlight_bonds_list: List of bond highlight lists for each molecule
            legends: Optional legends for each molecule

        Returns:
            PIL Image containing the highlighted molecule grid

        Raises:
            ValueError: If input lists have mismatched lengths
            RuntimeError: If rendering fails
        """
        if not mols:
            raise ValueError("Cannot render empty molecule list")

        # Validate highlight list lengths
        if highlight_atoms_list and len(highlight_atoms_list) != len(mols):
            raise ValueError(
                "highlight_atoms_list length must match molecules list length"
            )
        if highlight_bonds_list and len(highlight_bonds_list) != len(mols):
            raise ValueError(
                "highlight_bonds_list length must match molecules list length"
            )

        try:
            # Prepare molecules with highlights
            prepared_mols = []
            for i, mol in enumerate(mols):
                if mol is None:
                    logger.warning(f"Skipping None molecule at index {i}")
                    continue

                prepared_mol = self._prepare_molecule(mol)

                # Apply highlights if provided
                if highlight_atoms_list and i < len(highlight_atoms_list):
                    highlight_atoms = highlight_atoms_list[i]
                    if highlight_atoms:
                        # Store highlights as molecule properties for Draw.MolsToGridImage
                        for atom_idx in highlight_atoms:
                            if atom_idx < prepared_mol.GetNumAtoms():
                                atom = prepared_mol.GetAtomWithIdx(atom_idx)
                                atom.SetProp("atomNote", "highlight")

                prepared_mols.append(prepared_mol)

            return self.render_grid(prepared_mols, legends)

        except Exception as e:
            logger.error(f"Failed to render molecule grid with highlights: {e}")
            raise RuntimeError(f"Failed to render molecule grid with highlights: {e}")
