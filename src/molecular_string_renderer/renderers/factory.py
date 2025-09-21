"""
Factory functions for creating renderer instances.

Provides a convenient interface for obtaining the appropriate
renderer type based on user requirements.
"""

import logging

from molecular_string_renderer.config import RenderConfig
from .base import MolecularRenderer
from .grid import MoleculeGridRenderer
from .two_dimensional import Molecule2DRenderer

logger = logging.getLogger(__name__)


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
