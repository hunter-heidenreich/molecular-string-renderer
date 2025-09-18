"""
Vector and document output handlers.

Provides implementations for SVG and PDF formats.
"""

import base64
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image

from molecular_string_renderer.outputs.base import VectorOutputHandler
from molecular_string_renderer.outputs.raster import PNGOutput

logger = logging.getLogger(__name__)


class SVGOutput(VectorOutputHandler):
    """SVG output handler with true vector SVG rendering."""

    def __init__(self, config=None):
        """Initialize SVG output handler."""
        super().__init__(config)
        self._molecule = None  # Store molecule for direct SVG rendering

    @property
    def file_extension(self) -> str:
        """Get SVG file extension."""
        return ".svg"

    def set_molecule(self, mol) -> None:
        """Set the molecule for direct SVG rendering.

        Args:
            mol: RDKit Mol object to be used for vector SVG generation
        """
        self._molecule = mol

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save as true vector SVG file."""
        path = self._ensure_output_directory(output_path)

        if not str(path).lower().endswith(".svg"):
            path = path.with_suffix(".svg")

        try:
            svg_content = self._generate_vector_svg(image)
            path.write_text(svg_content, encoding="utf-8")
            self._log_success(path)
        except Exception as e:
            self._handle_save_error(path, e)

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as true vector SVG bytes."""
        svg_content = self._generate_vector_svg(image)
        return svg_content.encode("utf-8")

    def _generate_vector_svg(self, image: Image.Image) -> str:
        """Generate true vector SVG from molecule.

        Args:
            image: PIL Image (used for size reference if no molecule available)

        Returns:
            SVG content as string
        """
        # Check if vector SVG is enabled (default to True if not specified)
        svg_use_vector = getattr(self.config, "svg_use_vector", True)
        if not svg_use_vector:
            logger.debug("Vector SVG disabled, using raster fallback")
            return self._generate_raster_svg(image)

        # If we have the original molecule, use RDKit's direct SVG rendering
        if self._molecule is not None:
            try:
                from rdkit.Chem import Draw

                # Get line width multiplier if available
                line_width_mult = getattr(self.config, "svg_line_width_mult", 1)

                # Use RDKit's native SVG generation
                svg_content = Draw.MolToSVG(
                    self._molecule,
                    width=image.width,
                    height=image.height,
                    kekulize=True,
                    lineWidthMult=line_width_mult,
                    includeAtomCircles=True,
                )

                # Clean up the SVG if needed (RDKit's SVG is already well-formed)
                if self.config.optimize:
                    svg_content = self._optimize_svg(svg_content)

                logger.debug("Generated true vector SVG using RDKit")
                return svg_content

            except Exception as e:
                logger.warning(
                    f"Failed to generate vector SVG, falling back to raster: {e}"
                )

        # Fallback to raster-embedded SVG if molecule not available
        logger.debug("Using raster fallback for SVG generation")
        return self._generate_raster_svg(image)

    def _generate_raster_svg(self, image: Image.Image) -> str:
        """Generate SVG by embedding raster image (fallback method).

        Args:
            image: PIL Image to embed

        Returns:
            SVG content with embedded raster image
        """
        # Convert to PNG bytes for embedding
        png_output = PNGOutput(self.config)
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
        return svg_content

    def _optimize_svg(self, svg_content: str) -> str:
        """Optimize SVG content for smaller file size.

        Args:
            svg_content: Original SVG content

        Returns:
            Optimized SVG content
        """
        if not self.config.optimize:
            return svg_content

        # Basic SVG optimizations
        # Remove unnecessary whitespace and comments
        lines = svg_content.split("\n")
        optimized_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("<!--"):
                optimized_lines.append(line)

        return "\n".join(optimized_lines)


class PDFOutput(VectorOutputHandler):
    """PDF output handler using ReportLab."""

    @property
    def file_extension(self) -> str:
        """Get PDF file extension."""
        return ".pdf"

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save image as PDF file."""
        path = self._ensure_output_directory(output_path)

        if not str(path).lower().endswith(".pdf"):
            path = path.with_suffix(".pdf")

        try:
            pdf_bytes = self.get_bytes(image)
            path.write_bytes(pdf_bytes)
            self._log_success(path)
        except Exception as e:
            self._handle_save_error(path, e)

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as PDF bytes."""
        try:
            return self._generate_pdf_bytes(image)
        except ImportError as e:
            logger.error(f"ReportLab not available for PDF generation: {e}")
            raise IOError(f"ReportLab not available for PDF generation: {e}")
        except Exception as e:
            logger.error(f"Failed to generate PDF bytes: {e}")
            raise IOError(f"Failed to generate PDF bytes: {e}")

    def _generate_pdf_bytes(self, image: Image.Image) -> bytes:
        """Generate PDF bytes from image using ReportLab.

        Args:
            image: PIL Image to convert to PDF

        Returns:
            PDF data as bytes
        """
        # Import reportlab modules
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas

        # Create a canvas for the PDF in memory
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)

        # Get page dimensions
        page_width, page_height = letter

        # Calculate image dimensions to fit on page with some margin
        margin = 0.5 * inch
        max_width = page_width - 2 * margin
        max_height = page_height - 2 * margin

        # Convert PIL image to RGB if it isn't already
        rgb_image = image.convert("RGB")

        # Calculate scale factor to fit image on page
        img_width, img_height = rgb_image.size
        scale_x = max_width / img_width
        scale_y = max_height / img_height
        scale = min(scale_x, scale_y)

        # Calculate actual dimensions
        actual_width = img_width * scale
        actual_height = img_height * scale

        # Calculate position to center the image
        x = (page_width - actual_width) / 2
        y = (page_height - actual_height) / 2

        # Save image to temporary buffer for embedding
        img_buffer = BytesIO()
        rgb_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # Draw the image on the PDF using the buffer directly
        img_reader = ImageReader(img_buffer)
        c.drawImage(
            img_reader,
            x,
            y,
            width=actual_width,
            height=actual_height,
            preserveAspectRatio=True,
        )

        # Save the PDF and get bytes
        c.save()
        return buffer.getvalue()
