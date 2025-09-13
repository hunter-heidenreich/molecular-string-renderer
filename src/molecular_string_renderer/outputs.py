"""
Output format abstractions and implementations.

Provides flexible output generation for rendered molecules.
"""

import base64
import hashlib
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

from PIL import Image

from molecular_string_renderer.config import OutputConfig

logger = logging.getLogger(__name__)


class OutputHandler(ABC):
    """Abstract base class for output handlers."""

    def __init__(self, config: OutputConfig | None = None):
        """Initialize output handler with configuration."""
        self.config = config or OutputConfig()

    @abstractmethod
    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """
        Save image to specified path.

        Args:
            image: PIL Image to save
            output_path: Path where image should be saved
        """
        pass

    @abstractmethod
    def get_bytes(self, image: Image.Image) -> bytes:
        """
        Get image as bytes.

        Args:
            image: PIL Image to convert

        Returns:
            Image data as bytes
        """
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Get the file extension for this output format."""
        pass

    def _ensure_output_directory(self, output_path: str | Path) -> Path:
        """Ensure output directory exists."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class PNGOutput(OutputHandler):
    """PNG output handler."""

    @property
    def file_extension(self) -> str:
        """Get PNG file extension."""
        return ".png"

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save image as PNG file."""
        path = self._ensure_output_directory(output_path)

        # Ensure PNG extension
        if not str(path).lower().endswith(".png"):
            path = path.with_suffix(".png")

        try:
            # Convert to RGB if saving as PNG (remove alpha for smaller files if not needed)
            save_image = image
            if image.mode == "RGBA" and not self._has_transparency(image):
                save_image = image.convert("RGB")

            save_image.save(
                path, "PNG", optimize=self.config.optimize, quality=self.config.quality
            )
            logger.info(f"Successfully saved PNG to '{path}'")

        except Exception as e:
            logger.error(f"Failed to save PNG to '{path}': {e}")
            raise IOError(f"Failed to save PNG to '{path}': {e}")

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as PNG bytes."""
        buffer = BytesIO()

        # Convert to RGB if no transparency
        save_image = image
        if image.mode == "RGBA" and not self._has_transparency(image):
            save_image = image.convert("RGB")

        save_image.save(
            buffer, "PNG", optimize=self.config.optimize, quality=self.config.quality
        )

        return buffer.getvalue()

    def _has_transparency(self, image: Image.Image) -> bool:
        """Check if image has transparent pixels."""
        if image.mode != "RGBA":
            return False

        # Quick check by examining alpha channel
        alpha = image.split()[-1]
        return alpha.getextrema()[0] < 255


class SVGOutput(OutputHandler):
    """SVG output handler with true vector SVG rendering."""

    def __init__(self, config: OutputConfig | None = None):
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
            logger.info(f"Successfully saved vector SVG to '{path}'")
        except Exception as e:
            logger.error(f"Failed to save SVG to '{path}': {e}")
            raise IOError(f"Failed to save SVG to '{path}': {e}")

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


class JPEGOutput(OutputHandler):
    """JPEG output handler."""

    @property
    def file_extension(self) -> str:
        """Get JPEG file extension."""
        return ".jpg"

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save image as JPEG file."""
        path = self._ensure_output_directory(output_path)

        if not str(path).lower().endswith((".jpg", ".jpeg")):
            path = path.with_suffix(".jpg")

        # Convert to RGB (JPEG doesn't support alpha)
        save_image = image.convert("RGB")

        try:
            save_image.save(
                path, "JPEG", optimize=self.config.optimize, quality=self.config.quality
            )
            logger.info(f"Successfully saved JPEG to '{path}'")
        except Exception as e:
            logger.error(f"Failed to save JPEG to '{path}': {e}")
            raise IOError(f"Failed to save JPEG to '{path}': {e}")

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as JPEG bytes."""
        buffer = BytesIO()
        save_image = image.convert("RGB")

        save_image.save(
            buffer, "JPEG", optimize=self.config.optimize, quality=self.config.quality
        )

        return buffer.getvalue()


class WEBPOutput(OutputHandler):
    """WEBP output handler."""

    @property
    def file_extension(self) -> str:
        """Get WEBP file extension."""
        return ".webp"

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save image as WEBP file."""
        path = self._ensure_output_directory(output_path)

        # Ensure WEBP extension
        if not str(path).lower().endswith(".webp"):
            path = path.with_suffix(".webp")

        try:
            image.save(
                path, "WEBP", optimize=self.config.optimize, quality=self.config.quality
            )
            logger.info(f"Successfully saved WEBP to '{path}'")

        except Exception as e:
            logger.error(f"Failed to save WEBP to '{path}': {e}")
            raise IOError(f"Failed to save WEBP to '{path}': {e}")

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as WEBP bytes."""
        buffer = BytesIO()
        image.save(
            buffer, "WEBP", optimize=self.config.optimize, quality=self.config.quality
        )
        return buffer.getvalue()


class TIFFOutput(OutputHandler):
    """TIFF output handler."""

    @property
    def file_extension(self) -> str:
        """Get TIFF file extension."""
        return ".tiff"

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save image as TIFF file."""
        path = self._ensure_output_directory(output_path)

        # Ensure TIFF extension
        if not str(path).lower().endswith((".tiff", ".tif")):
            path = path.with_suffix(".tiff")

        try:
            # TIFF supports transparency, so keep original mode
            compression = "tiff_lzw" if self.config.optimize else None

            # Quality is only supported with JPEG compression in TIFF
            save_kwargs = {"format": "TIFF"}
            if compression:
                save_kwargs["compression"] = compression
            # Don't pass quality for TIFF unless using JPEG compression

            image.save(path, **save_kwargs)
            logger.info(f"Successfully saved TIFF to '{path}'")

        except Exception as e:
            logger.error(f"Failed to save TIFF to '{path}': {e}")
            raise IOError(f"Failed to save TIFF to '{path}': {e}")

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as TIFF bytes."""
        buffer = BytesIO()
        compression = "tiff_lzw" if self.config.optimize else None

        # Quality is only supported with JPEG compression in TIFF
        save_kwargs = {"format": "TIFF"}
        if compression:
            save_kwargs["compression"] = compression
        # Don't pass quality for TIFF unless using JPEG compression

        image.save(buffer, **save_kwargs)
        return buffer.getvalue()


class BMPOutput(OutputHandler):
    """BMP output handler."""

    @property
    def file_extension(self) -> str:
        """Get BMP file extension."""
        return ".bmp"

    def save(self, image: Image.Image, output_path: str | Path) -> None:
        """Save image as BMP file."""
        path = self._ensure_output_directory(output_path)

        # Ensure BMP extension
        if not str(path).lower().endswith(".bmp"):
            path = path.with_suffix(".bmp")

        try:
            # Convert to RGB (BMP doesn't support alpha channel)
            save_image = image.convert("RGB")

            save_image.save(path, "BMP")
            logger.info(f"Successfully saved BMP to '{path}'")

        except Exception as e:
            logger.error(f"Failed to save BMP to '{path}': {e}")
            raise IOError(f"Failed to save BMP to '{path}': {e}")

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as BMP bytes."""
        buffer = BytesIO()
        save_image = image.convert("RGB")
        save_image.save(buffer, "BMP")
        return buffer.getvalue()


class PDFOutput(OutputHandler):
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
            # Import reportlab modules
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.units import inch
            from reportlab.pdfgen import canvas

            # Create a canvas for the PDF
            c = canvas.Canvas(str(path), pagesize=letter)

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
            from reportlab.lib.utils import ImageReader

            img_reader = ImageReader(img_buffer)
            c.drawImage(
                img_reader,
                x,
                y,
                width=actual_width,
                height=actual_height,
                preserveAspectRatio=True,
            )

            # Save the PDF
            c.save()
            logger.info(f"Successfully saved PDF to '{path}'")

        except ImportError as e:
            logger.error(f"ReportLab not available for PDF generation: {e}")
            raise IOError(f"ReportLab not available for PDF generation: {e}")
        except Exception as e:
            logger.error(f"Failed to save PDF to '{path}': {e}")
            raise IOError(f"Failed to save PDF to '{path}': {e}")

    def get_bytes(self, image: Image.Image) -> bytes:
        """Get image as PDF bytes."""
        try:
            # Import reportlab modules
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.units import inch
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
            from reportlab.lib.utils import ImageReader

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

        except ImportError as e:
            logger.error(f"ReportLab not available for PDF generation: {e}")
            raise IOError(f"ReportLab not available for PDF generation: {e}")
        except Exception as e:
            logger.error(f"Failed to generate PDF bytes: {e}")
            raise IOError(f"Failed to generate PDF bytes: {e}")


def create_safe_filename(molecular_string: str, extension: str = ".png") -> str:
    """
    Generate a filesystem-safe filename from a molecular string using MD5 hash.

    Args:
        molecular_string: The input molecular string (SMILES, InChI, etc.)
        extension: File extension to use

    Returns:
        A safe filename with the specified extension
    """
    clean_string = molecular_string.strip()
    hasher = hashlib.md5(clean_string.encode("utf-8"))
    base_name = hasher.hexdigest()

    if not extension.startswith("."):
        extension = f".{extension}"

    return f"{base_name}{extension}"


def get_output_handler(
    format_type: str, config: OutputConfig | None = None
) -> OutputHandler:
    """
    Factory function to get appropriate output handler.

    Args:
        format_type: Output format type ('png', 'svg', 'jpg', 'jpeg', 'pdf')
        config: Output configuration

    Returns:
        Appropriate output handler instance

    Raises:
        ValueError: If format type is not supported
    """
    format_type = format_type.lower().strip()

    handlers = {
        "png": PNGOutput,
        "svg": SVGOutput,
        "jpg": JPEGOutput,
        "jpeg": JPEGOutput,
        "pdf": PDFOutput,
        "webp": WEBPOutput,
        "tiff": TIFFOutput,
        "tif": TIFFOutput,  # Alternative extension for TIFF
        "bmp": BMPOutput,
    }

    if format_type not in handlers:
        supported = list(handlers.keys())
        logger.error(
            f"Unsupported output format: {format_type}. Supported formats: {supported}"
        )
        raise ValueError(
            f"Unsupported output format: {format_type}. Supported: {supported}"
        )

    logger.debug(f"Creating {format_type} output handler")
    return handlers[format_type](config)
