"""
SVG generation strategies.

Provides different strategies for generating SVG content (vector vs raster).
"""

import base64
import logging
from abc import ABC, abstractmethod

from PIL import Image
from rdkit.Chem import Draw

from molecular_string_renderer.outputs.raster import PNGOutput

logger = logging.getLogger(__name__)


class SVGGenerationStrategy(ABC):
    """Abstract base class for SVG generation strategies."""

    @abstractmethod
    def generate_svg(self, image: Image.Image, config: object) -> str:
        """Generate SVG content from an image.

        Args:
            image: PIL Image to convert to SVG
            config: Output configuration object

        Returns:
            SVG content as string
        """
        pass


class VectorSVGStrategy(SVGGenerationStrategy):
    """Strategy for generating true vector SVG using RDKit."""

    def __init__(self):
        """Initialize vector SVG strategy."""
        self._molecule = None

    def set_molecule(self, mol: object) -> None:
        """Set the molecule for vector SVG generation.

        Args:
            mol: RDKit Mol object to be used for vector SVG generation
        """
        self._molecule = mol

    def generate_svg(self, image: Image.Image, config: object) -> str:
        """Generate true vector SVG from molecule.

        Args:
            image: PIL Image (used for size reference)
            config: Output configuration object

        Returns:
            SVG content as string

        Raises:
            ValueError: If no molecule is set
        """
        if self._molecule is None:
            raise ValueError("No molecule set for vector SVG generation")

        try:
            # Use RDKit's native SVG generation
            svg_content = Draw.MolToSVG(
                self._molecule,
                width=image.width,
                height=image.height,
                kekulize=True,
                lineWidthMult=config.svg_line_width_mult,
                includeAtomCircles=True,
            )

            # Clean up the SVG if optimization is enabled
            if config.optimize:
                svg_content = self._optimize_svg(svg_content)

            logger.debug("Generated true vector SVG using RDKit")
            return svg_content

        except Exception as e:
            logger.warning(f"Failed to generate vector SVG: {e}")
            raise ValueError(f"Failed to generate vector SVG: {e}")

    def _optimize_svg(self, svg_content: str) -> str:
        """Optimize SVG content for smaller file size.

        Args:
            svg_content: Original SVG content

        Returns:
            Optimized SVG content
        """
        lines = svg_content.split("\n")
        optimized_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("<!--"):
                optimized_lines.append(line)

        return "\n".join(optimized_lines)


class RasterSVGStrategy(SVGGenerationStrategy):
    """Strategy for generating SVG by embedding raster images."""

    def generate_svg(self, image: Image.Image, config: object) -> str:
        """Generate SVG by embedding raster image.

        Args:
            image: PIL Image to embed
            config: Output configuration object

        Returns:
            SVG content with embedded raster image
        """
        # Convert to PNG bytes for embedding
        png_output = PNGOutput(config)
        png_bytes = png_output.get_bytes(image)
        base64_data = base64.b64encode(png_bytes).decode()

        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{image.width}" height="{image.height}" 
     viewBox="0 0 {image.width} {image.height}">
  <image width="{image.width}" height="{image.height}" 
         xlink:href="data:image/png;base64,{base64_data}"/>
</svg>'''

        logger.debug("Generated raster-embedded SVG")
        return svg_content


class HybridSVGStrategy(SVGGenerationStrategy):
    """Strategy that attempts vector SVG first, then falls back to raster."""

    def __init__(self):
        """Initialize hybrid SVG strategy."""
        self._vector_strategy = VectorSVGStrategy()
        self._raster_strategy = RasterSVGStrategy()

    def set_molecule(self, mol: object) -> None:
        """Set the molecule for vector SVG generation.

        Args:
            mol: RDKit Mol object to be used for vector SVG generation
        """
        self._vector_strategy.set_molecule(mol)

    def generate_svg(self, image: Image.Image, config: object) -> str:
        """Generate SVG using vector strategy first, fallback to raster.

        Args:
            image: PIL Image to convert to SVG
            config: Output configuration object

        Returns:
            SVG content as string
        """
        # Check if vector SVG is enabled
        if not config.svg_use_vector:
            logger.debug("Vector SVG disabled, using raster strategy")
            return self._raster_strategy.generate_svg(image, config)

        # Try vector strategy first
        try:
            return self._vector_strategy.generate_svg(image, config)
        except ValueError as e:
            logger.debug(f"Vector SVG failed, falling back to raster: {e}")
            return self._raster_strategy.generate_svg(image, config)
