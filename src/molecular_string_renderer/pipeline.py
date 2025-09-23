"""
Pipeline for molecular rendering operations.

Provides a composable way to chain validation, parsing, rendering, and output
operations while maintaining clear separation of concerns.
"""

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.exceptions import ParsingError, RenderingError
from molecular_string_renderer.parsers import get_parser
from molecular_string_renderer.renderers import get_renderer

logger = logging.getLogger(__name__)


class RenderingPipeline:
    """
    A simple pipeline for molecular rendering operations.

    Provides a composable way to chain validation, parsing, rendering, and output
    operations while maintaining clear separation of concerns.
    """

    def __init__(
        self,
        render_config: RenderConfig,
        parser_config: ParserConfig,
        output_config: OutputConfig,
    ):
        """
        Initialize the rendering pipeline.

        Args:
            render_config: Configuration for rendering
            parser_config: Configuration for parsing
            output_config: Configuration for output
        """
        self.render_config = render_config
        self.parser_config = parser_config
        self.output_config = output_config

    def validate_inputs(
        self,
        molecular_string: str,
        format_type: str,
        output_path: str | Path | None = None,
    ) -> tuple[str, str, Path | None]:
        """
        Validate and normalize inputs.

        Args:
            molecular_string: Molecular string to validate
            format_type: Format type to validate
            output_path: Output path to validate

        Returns:
            Tuple of (molecular_string, format_type, normalized_output_path)
        """
        from molecular_string_renderer.utils import validate_and_normalize_inputs

        return validate_and_normalize_inputs(molecular_string, format_type, output_path)

    def parse_molecule(self, molecular_string: str, format_type: str) -> Any:
        """
        Parse a single molecular string.

        Args:
            molecular_string: String to parse
            format_type: Format type for parsing

        Returns:
            Parsed molecule object

        Raises:
            ParsingError: If parsing fails
        """
        from molecular_string_renderer.logging_utils import logged_operation

        with logged_operation("parse", {"format": format_type}):
            parser = get_parser(format_type, self.parser_config)
            mol = parser.parse(molecular_string)
            if mol is None:
                raise ParsingError(
                    f"Failed to parse {format_type.upper()}: '{molecular_string}'"
                )
            return mol

    def render_molecule(self, mol: Any) -> Image.Image:
        """
        Render a molecule to an image.

        Args:
            mol: Molecule object to render

        Returns:
            PIL Image object

        Raises:
            RenderingError: If rendering fails
        """
        from molecular_string_renderer.logging_utils import logged_operation

        with logged_operation(
            "render",
            {"size": f"{self.render_config.width}x{self.render_config.height}"},
        ):
            renderer = get_renderer("2d", self.render_config)
            image = renderer.render(mol)
            if image is None:
                raise RenderingError("Renderer returned None image")
            return image

    def save_output(
        self,
        image: Image.Image,
        output_path: str | Path | None,
        output_format: str,
        mol: Any = None,
        auto_filename: bool = False,
        molecular_string: str | None = None,
    ) -> None:
        """
        Save rendered image to output.

        Args:
            image: Image to save
            output_path: Path to save to
            output_format: Output format
            mol: Optional molecule for vector formats
            auto_filename: Whether to generate filename automatically
            molecular_string: Source string for auto filename
        """
        if output_path or auto_filename:
            from molecular_string_renderer.logging_utils import logged_operation
            from molecular_string_renderer.utils import handle_output_saving

            with logged_operation("save", {"format": output_format}):
                handle_output_saving(
                    image=image,
                    output_path=output_path,
                    output_format=output_format,
                    output_config=self.output_config,
                    mol=mol,
                    auto_filename=auto_filename,
                    molecular_string=molecular_string,
                )
