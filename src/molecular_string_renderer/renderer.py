"""
High-level class interface for molecular rendering.

Provides an object-oriented interface that maintains configuration
across multiple rendering operations with caching for improved performance.
"""

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.exceptions import (
    ConfigurationError,
    OutputError,
    ParsingError,
    RenderingError,
    ValidationError,
)
from molecular_string_renderer.outputs import get_output_handler
from molecular_string_renderer.parsers import get_parser
from molecular_string_renderer.renderers import get_renderer
from molecular_string_renderer.utils import parse_molecule_list
from molecular_string_renderer.validation import (
    validate_format_type,
    validate_grid_parameters,
    validate_molecular_string,
    validate_output_path,
)

logger = logging.getLogger(__name__)


class MolecularRenderer:
    """
    High-level class interface for molecular rendering.

    Provides an object-oriented interface that maintains configuration
    across multiple rendering operations with caching for improved performance.
    """

    def __init__(
        self,
        render_config: RenderConfig | None = None,
        parser_config: ParserConfig | None = None,
        output_config: OutputConfig | None = None,
    ):
        """
        Initialize molecular renderer with configurations.

        Args:
            render_config: Configuration for rendering
            parser_config: Configuration for parsing
            output_config: Configuration for output

        Raises:
            ValidationError: If configurations are invalid
        """
        self.render_config = render_config or RenderConfig()
        self.parser_config = parser_config or ParserConfig()
        self.output_config = output_config or OutputConfig()

        # Validate configurations
        try:
            # Test that configurations are valid by attempting to convert to dict
            self.render_config.model_dump()
            self.parser_config.model_dump()
            self.output_config.model_dump()
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}") from e

        # Cache for parsers, renderers, and output handlers
        self._parsers: dict[str, Any] = {}
        self._renderers: dict[str, Any] = {}
        self._output_handlers: dict[str, Any] = {}

        # Performance tracking
        self._operation_count = 0
        self._cache_hits = 0

        # Cache configuration hashes to avoid repeated serialization
        self._config_hashes: dict[str, int] = {}

    def _get_config_hash(self, config_type: str) -> int:
        """Get cached configuration hash or compute and cache it."""
        if config_type not in self._config_hashes:
            if config_type == "parser":
                self._config_hashes[config_type] = hash(
                    str(self.parser_config.model_dump())
                )
            elif config_type == "render":
                self._config_hashes[config_type] = hash(
                    str(self.render_config.model_dump())
                )
            elif config_type == "output":
                self._config_hashes[config_type] = hash(
                    str(self.output_config.model_dump())
                )
        return self._config_hashes[config_type]

    def _get_cached_parser(self, format_type: str):
        """Get cached parser or create new one."""
        cache_key = f"{format_type}_{self._get_config_hash('parser')}"
        if cache_key not in self._parsers:
            self._parsers[cache_key] = get_parser(format_type, self.parser_config)
        else:
            self._cache_hits += 1
        return self._parsers[cache_key]

    def _get_cached_renderer(self, renderer_type: str):
        """Get cached renderer or create new one."""
        cache_key = f"{renderer_type}_{self._get_config_hash('render')}"
        if cache_key not in self._renderers:
            self._renderers[cache_key] = get_renderer(renderer_type, self.render_config)
        else:
            self._cache_hits += 1
        return self._renderers[cache_key]

    def _get_cached_output_handler(self, format_type: str):
        """Get cached output handler or create new one."""
        cache_key = f"{format_type}_{self._get_config_hash('output')}"
        if cache_key not in self._output_handlers:
            self._output_handlers[cache_key] = get_output_handler(
                format_type, self.output_config
            )
        else:
            self._cache_hits += 1
        return self._output_handlers[cache_key]

    def render(
        self,
        molecular_string: str,
        format_type: str = "smiles",
        output_format: str = "png",
        output_path: str | Path | None = None,
    ) -> Image.Image:
        """
        Render a molecular string using the configured settings.

        Args:
            molecular_string: Molecular string to render
            format_type: Input format type
            output_format: Output format type
            output_path: Optional output path

        Returns:
            PIL Image object

        Raises:
            ValidationError: If input parameters are invalid
            ParsingError: If molecular string cannot be parsed
            RenderingError: If molecule cannot be rendered
            OutputError: If image cannot be saved
        """
        self._operation_count += 1

        # Validate inputs
        validate_molecular_string(molecular_string, format_type)
        format_type = validate_format_type(
            format_type, {"smiles", "smi", "inchi", "mol", "sdf", "selfies"}
        )
        output_path = validate_output_path(output_path)

        try:
            # Use cached parser
            parser = self._get_cached_parser(format_type)
            mol = parser.parse(molecular_string)
            if mol is None:
                raise ParsingError(
                    f"Failed to parse {format_type.upper()}: '{molecular_string}'"
                )
        except Exception as e:
            if isinstance(e, ParsingError):
                raise
            raise ParsingError(
                f"Error parsing {format_type.upper()} '{molecular_string}': {e}"
            ) from e

        try:
            # Use cached renderer
            renderer = self._get_cached_renderer("2d")
            image = renderer.render(mol)
            if image is None:
                raise RenderingError("Renderer returned None image")
        except Exception as e:
            if isinstance(e, RenderingError):
                raise
            raise RenderingError(f"Error rendering molecule: {e}") from e

        try:
            # Save if output path provided
            if output_path:
                output_handler = self._get_cached_output_handler(output_format)
                if output_format.lower() == "svg" and mol is not None:
                    output_handler.set_molecule(mol)
                output_handler.save(image, output_path)
        except Exception as e:
            if isinstance(e, OutputError):
                raise
            raise OutputError(f"Error saving output: {e}") from e

        return image

    def render_grid(
        self,
        molecular_strings: list[str],
        format_type: str = "smiles",
        output_format: str = "png",
        output_path: str | Path | None = None,
        legends: list[str] | None = None,
        mols_per_row: int = 4,
    ) -> Image.Image:
        """
        Render multiple molecules in a grid.

        Args:
            molecular_strings: List of molecular strings
            format_type: Input format type
            output_format: Output format type
            output_path: Optional output path
            legends: Optional legends
            mols_per_row: Molecules per row

        Returns:
            PIL Image object

        Raises:
            ValidationError: If input parameters are invalid
            ParsingError: If molecular strings cannot be parsed
            RenderingError: If molecules cannot be rendered
            OutputError: If image cannot be saved
        """
        self._operation_count += 1

        # Use standard mol_size for cached renderer class
        mol_size = (200, 200)

        # Validate inputs
        validate_grid_parameters(molecular_strings, mols_per_row, mol_size)
        format_type = validate_format_type(
            format_type, {"smiles", "smi", "inchi", "mol", "sdf", "selfies"}
        )
        output_path = validate_output_path(output_path)

        if legends is not None:
            if not isinstance(legends, list):
                raise ValidationError(
                    f"Legends must be a list, got {type(legends).__name__}"
                )
            if len(legends) != len(molecular_strings):
                raise ValidationError(
                    f"Number of legends ({len(legends)}) must match number of molecules ({len(molecular_strings)})"
                )

        # Parse molecules using cached parser
        cached_parser = self._get_cached_parser(format_type)
        mols, valid_indices, parsing_errors = parse_molecule_list(
            molecular_strings, format_type, self.parser_config, cached_parser
        )

        if not mols:
            from molecular_string_renderer.utils import format_parsing_errors

            error_details = format_parsing_errors(parsing_errors)
            raise ParsingError(
                f"No valid molecules could be parsed. Errors: {error_details}"
            )

        # Filter legends if needed
        if legends and valid_indices:
            filtered_legends = [legends[i] for i in valid_indices if i < len(legends)]
            if len(filtered_legends) != len(mols):
                logger.warning(
                    "Legend count mismatch after filtering, disabling legends"
                )
                legends = None
            else:
                legends = filtered_legends

        try:
            # Create and use grid renderer (not cached due to varying parameters)
            from molecular_string_renderer.renderers import MoleculeGridRenderer

            grid_renderer = MoleculeGridRenderer(
                config=self.render_config, mols_per_row=mols_per_row, mol_size=mol_size
            )
            image = grid_renderer.render_grid(mols, legends)
            if image is None:
                raise RenderingError("Grid renderer returned None image")
        except Exception as e:
            if isinstance(e, RenderingError):
                raise
            raise RenderingError(f"Error rendering molecule grid: {e}") from e

        try:
            # Save if output path provided
            if output_path:
                output_handler = self._get_cached_output_handler(output_format)
                output_handler.save(image, output_path)
        except Exception as e:
            if isinstance(e, OutputError):
                raise
            raise OutputError(f"Error saving grid output: {e}") from e

        return image

    def update_config(
        self,
        render_config: RenderConfig | None = None,
        parser_config: ParserConfig | None = None,
        output_config: OutputConfig | None = None,
    ) -> None:
        """
        Update renderer configurations.

        Args:
            render_config: New render configuration
            parser_config: New parser configuration
            output_config: New output configuration

        Raises:
            ConfigurationError: If new configurations are invalid
        """
        # Validate new configurations before updating
        if render_config:
            try:
                render_config.model_dump()
            except Exception as e:
                raise ConfigurationError(f"Invalid render configuration: {e}") from e
            self.render_config = render_config

        if parser_config:
            try:
                parser_config.model_dump()
            except Exception as e:
                raise ConfigurationError(f"Invalid parser configuration: {e}") from e
            self.parser_config = parser_config

        if output_config:
            try:
                output_config.model_dump()
            except Exception as e:
                raise ConfigurationError(f"Invalid output configuration: {e}") from e
            self.output_config = output_config

        # Clear caches when config changes to ensure consistency
        self._parsers.clear()
        self._renderers.clear()
        self._output_handlers.clear()
        self._config_hashes.clear()  # Clear cached hashes too

    def get_stats(self) -> dict[str, int | float]:
        """
        Get performance statistics for this renderer instance.

        Returns:
            Dictionary with operation count and cache statistics
        """
        return {
            "operations": self._operation_count,
            "cache_hits": self._cache_hits,
            "cache_efficiency": (
                round(self._cache_hits / max(1, self._operation_count) * 100, 1)
                if self._operation_count > 0
                else 0.0
            ),
            "cached_parsers": len(self._parsers),
            "cached_renderers": len(self._renderers),
            "cached_output_handlers": len(self._output_handlers),
        }

    def clear_cache(self) -> None:
        """Clear all cached objects to free memory."""
        self._parsers.clear()
        self._renderers.clear()
        self._output_handlers.clear()
        self._config_hashes.clear()
