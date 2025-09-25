"""
Configuration management for RDKit molecular drawing.

Handles the setup and configuration of RDKit drawer objects
with appropriate options based on render configuration.
"""

import logging

from rdkit.Chem.Draw import rdMolDraw2D

from molecular_string_renderer.config import RenderConfig

from .utils import ColorUtils

logger = logging.getLogger(__name__)


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
        
        # Handle new visual options
        options.bondLineWidth = self.config.bond_line_width
        options.addStereoAnnotation = self.config.show_stereo_labels
        options.includeRadicals = self.config.show_formal_charges
        options.rotate = self.config.rotation_angle
        
        # Additional configuration can be added here as needed
        logger.debug(
            f"Configured drawer with background: {self.config.background_color}, "
            f"carbon display: {self.config.show_carbon}, "
            f"bond width: {self.config.bond_line_width}, "
            f"stereo labels: {self.config.show_stereo_labels}, "
            f"formal charges: {self.config.show_formal_charges}, "
            f"rotation: {self.config.rotation_angle}°"
        )
