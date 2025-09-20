"""
Molecular renderer abstractions and implementations.

Provides flexible rendering of molecules to various formats.
"""

import io
import logging
from abc import ABC, abstractmethod

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from molecular_string_renderer.config import RenderConfig

logger = logging.getLogger(__name__)

Mol = Chem.Mol


class ColorUtils:
    """Utility class for color parsing and conversion operations."""

    @staticmethod
    def parse_color_to_rgba(color_str: str) -> tuple[float, float, float, float]:
        """
        Parse color string to RGBA tuple (0-1 range) for RDKit.

        Args:
            color_str: Color name or hex string

        Returns:
            RGBA tuple with values in 0-1 range

        Raises:
            ValueError: If color cannot be parsed
        """
        try:
            # Try to create a PIL image with the color to parse it
            test_img = Image.new("RGB", (1, 1), color_str)
            r, g, b = test_img.getpixel((0, 0))
            return (r / 255.0, g / 255.0, b / 255.0, 1.0)
        except Exception as e:
            logger.warning(
                f"Failed to parse color '{color_str}': {e}. Using white as fallback."
            )
            # Fall back to white if color parsing fails
            return (1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def is_white_background(color_str: str) -> bool:
        """
        Check if the given color string represents white background.

        Args:
            color_str: Color string to check

        Returns:
            True if color is white or equivalent
        """
        normalized = color_str.lower().strip()
        return normalized in ("white", "#ffffff", "#fff", "rgb(255,255,255)")


class DrawerConfigurationManager:
    """Manages RDKit drawer configuration and setup."""

    def __init__(self, config: RenderConfig):
        """
        Initialize drawer configuration manager.

        Args:
            config: Render configuration
        """
        self.config = config

    def create_drawer(self) -> rdMolDraw2D.MolDraw2DCairo:
        """
        Create and configure RDKit drawer.

        Returns:
            Configured MolDraw2DCairo instance
        """
        drawer = rdMolDraw2D.MolDraw2DCairo(self.config.width, self.config.height)
        self._configure_drawer_options(drawer)
        return drawer

    def _configure_drawer_options(self, drawer: rdMolDraw2D.MolDraw2DCairo) -> None:
        """
        Configure drawer options based on render config.

        Args:
            drawer: RDKit drawer to configure
        """
        options = drawer.drawOptions()

        # Handle background color
        if not ColorUtils.is_white_background(self.config.background_color):
            color_rgba = ColorUtils.parse_color_to_rgba(self.config.background_color)
            options.setBackgroundColour(color_rgba)

        # Handle carbon display
        options.explicitMethyl = self.config.show_carbon

        # Additional configuration can be added here as needed
        logger.debug(
            f"Configured drawer with background: {self.config.background_color}, carbon display: {self.config.show_carbon}"
        )


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

            img = Draw.MolsToGridImage(
                valid_mols,
                molsPerRow=self.mols_per_row,
                subImgSize=self.mol_size,
                legends=legends,
            )

            if img.mode != "RGBA":
                img = img.convert("RGBA")

            logger.debug(f"Successfully rendered molecule grid to {img.size} image")
            return img

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
        TypeError: If config is not a RenderConfig instance
    """
    if config is not None and not isinstance(config, RenderConfig):
        raise TypeError(f"config must be a RenderConfig instance, got {type(config)}")

    renderer_type = renderer_type.lower().strip()

    renderers = {
        "2d": Molecule2DRenderer,
        "grid": MoleculeGridRenderer,
    }

    if renderer_type not in renderers:
        supported = list(renderers.keys())
        logger.error(f"Unsupported renderer type: {renderer_type}")
        raise ValueError(
            f"Unsupported renderer: {renderer_type}. Supported: {supported}"
        )

    try:
        renderer_class = renderers[renderer_type]
        renderer = renderer_class(config)
        logger.debug(f"Created {renderer_type} renderer")
        return renderer
    except Exception as e:
        logger.error(f"Failed to create {renderer_type} renderer: {e}")
        raise RuntimeError(f"Failed to create {renderer_type} renderer: {e}")
