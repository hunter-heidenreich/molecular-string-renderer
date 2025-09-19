"""
Vector and document output handlers.

Provides implementations for SVG and PDF formats.
"""

import logging
from io import BytesIO
from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from rdkit import Chem

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs.base import VectorOutputHandler
from molecular_string_renderer.outputs.svg_strategies import HybridSVGStrategy

logger = logging.getLogger(__name__)

# Type alias for RDKit Mol objects
Mol = Chem.Mol


class SVGOutput(VectorOutputHandler):
    """SVG output handler with configurable generation strategy."""

    def __init__(self, config: OutputConfig | None = None):
        """Initialize SVG output handler.
        
        Args:
            config: Output configuration object
        """
        super().__init__("svg", config)
        self._strategy = HybridSVGStrategy()

    def set_molecule(self, mol: Mol) -> None:
        """Set the molecule for vector SVG generation.

        Args:
            mol: RDKit Mol object to be used for vector SVG generation
        """
        self._strategy.set_molecule(mol)

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save as SVG file."""
        path = self._ensure_output_directory(output_path)

        if not str(path).lower().endswith(".svg"):
            path = path.with_suffix(".svg")

        try:
            svg_content = self._strategy.generate_svg(image, self.config)
            path.write_text(svg_content, encoding="utf-8")
            self._log_success(path)
        except Exception as e:
            self._handle_save_error(path, e)

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as SVG bytes."""
        svg_content = self._strategy.generate_svg(image, self.config)
        return svg_content.encode("utf-8")


class PDFOutput(VectorOutputHandler):
    """PDF output handler using ReportLab."""

    def __init__(self, config: OutputConfig | None = None):
        """Initialize PDF output handler.
        
        Args:
            config: Output configuration object
        """
        super().__init__("pdf", config)

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
        with BytesIO() as buffer:
            c = canvas.Canvas(buffer, pagesize=letter)

            # Get page dimensions
            page_width, page_height = letter

            margin = 0.5 * inch
            max_width = page_width - 2 * margin
            max_height = page_height - 2 * margin

            rgb_image = image.convert("RGB")

            img_width, img_height = rgb_image.size
            scale_x = max_width / img_width
            scale_y = max_height / img_height
            scale = min(scale_x, scale_y)

            # Calculate actual dimensions
            actual_width = img_width * scale
            actual_height = img_height * scale

            x = (page_width - actual_width) / 2
            y = (page_height - actual_height) / 2

            with BytesIO() as img_buffer:
                rgb_image.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                img_reader = ImageReader(img_buffer)
                c.drawImage(
                    img_reader,
                    x,
                    y,
                    width=actual_width,
                    height=actual_height,
                    preserveAspectRatio=True,
                )

            c.save()
            return buffer.getvalue()
