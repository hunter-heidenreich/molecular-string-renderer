"""
Utility functions for molecular rendering operations.

Contains helper functions for parsing, validation, output handling, and other
common operations used across the molecular rendering pipeline.
"""

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from molecular_string_renderer.config import OutputConfig, ParserConfig
from molecular_string_renderer.outputs import create_safe_filename, get_output_handler
from molecular_string_renderer.parsers import get_parser
from molecular_string_renderer.validation import (
    validate_format_type,
    validate_molecular_string,
    validate_output_path,
)

logger = logging.getLogger(__name__)

# Constants for performance optimization
SUPPORTED_INPUT_FORMATS = frozenset({"smiles", "smi", "inchi", "mol", "sdf", "selfies"})
MAX_MOLECULAR_STRING_LENGTH = 10000
MAX_GRID_MOLECULES = 100

# Error message templates
ERROR_TEMPLATES = {
    "parse_failed": "Failed to parse {format}: '{string}'",
    "parsing_error": "Error parsing {format} '{string}': {error}",
    "no_valid_molecules": "No valid molecules could be parsed. Errors: {details}",
    "renderer_none": "Renderer returned None image",
    "grid_renderer_none": "Grid renderer returned None image",
}


def parse_molecule_list(
    molecular_strings: list[str],
    format_type: str,
    parser_config: ParserConfig,
    parser: Any = None,
) -> tuple[list[Any], list[int], list[str]]:
    """
    Parse a list of molecular strings, tracking successes and failures.

    Args:
        molecular_strings: List of molecular strings to parse
        format_type: Format type for parsing
        parser_config: Configuration for the parser
        parser: Optional pre-configured parser to use instead of creating new one

    Returns:
        Tuple of (parsed_molecules, valid_indices, parsing_errors)

    Example:
        >>> molecules = ['CCO', 'invalid', 'c1ccccc1']
        >>> mols, indices, errors = parse_molecule_list(molecules, 'smiles', config)
        >>> len(mols)  # Number of successfully parsed molecules
        2
        >>> indices    # Indices of valid molecules
        [0, 2]
    """
    if parser is None:
        parser = get_parser(format_type, parser_config)

    mols = []
    valid_indices = []
    parsing_errors = []

    for i, mol_string in enumerate(molecular_strings):
        try:
            validate_molecular_string(mol_string, format_type)
            mol = parser.parse(mol_string)
            if mol is None:
                parsing_errors.append(f"Index {i}: Failed to parse '{mol_string}'")
                continue
            mols.append(mol)
            valid_indices.append(i)
        except Exception as e:
            parsing_errors.append(f"Index {i}: {e}")
            logger.warning(
                f"Failed to parse molecule at index {i} ('{mol_string}'): {e}"
            )

    return mols, valid_indices, parsing_errors


def format_parsing_errors(parsing_errors: list[str], max_errors: int = 5) -> str:
    """
    Format parsing errors for display with truncation.

    Args:
        parsing_errors: List of error messages
        max_errors: Maximum number of errors to show

    Returns:
        Formatted error string
    """
    error_details = "; ".join(parsing_errors[:max_errors])
    if len(parsing_errors) > max_errors:
        error_details += f" (and {len(parsing_errors) - max_errors} more errors)"
    return error_details


def filter_legends_by_indices(
    legends: list[str] | None, valid_indices: list[int], total_molecules: int
) -> list[str] | None:
    """
    Filter legends to match successfully parsed molecules.

    Args:
        legends: Original legends list
        valid_indices: Indices of successfully parsed molecules
        total_molecules: Total number of molecules that were parsed

    Returns:
        Filtered legends or None if filtering failed
    """
    if not legends or not valid_indices:
        return legends

    filtered_legends = [legends[i] for i in valid_indices if i < len(legends)]
    if len(filtered_legends) != len(valid_indices):
        logger.warning("Legend count mismatch after filtering, disabling legends")
        return None
    return filtered_legends


def validate_and_normalize_inputs(
    molecular_string: str | None = None,
    format_type: str | None = None,
    output_path: str | Path | None = None,
) -> tuple[str | None, str | None, Path | None]:
    """
    Validate and normalize common input parameters.

    Args:
        molecular_string: Optional molecular string to validate
        format_type: Optional format type to validate and normalize
        output_path: Optional output path to validate and normalize

    Returns:
        Tuple of (molecular_string, normalized_format_type, normalized_output_path)

    Raises:
        ValidationError: If any parameter is invalid
    """
    normalized_format = None
    if format_type is not None:
        normalized_format = validate_format_type(format_type, SUPPORTED_INPUT_FORMATS)

    normalized_path = None
    if output_path is not None:
        normalized_path = validate_output_path(output_path)

    if molecular_string is not None and format_type is not None:
        validate_molecular_string(molecular_string, normalized_format)

    return molecular_string, normalized_format, normalized_path


def handle_operation_error(
    operation: str,
    exception: Exception,
    original_error_type: type[Exception],
    fallback_message: str,
) -> None:
    """
    Handle errors consistently across operations.

    Args:
        operation: Name of the operation that failed
        exception: The exception that occurred
        original_error_type: Expected exception type to pass through unchanged
        fallback_message: Message to use if wrapping in new exception type

    Raises:
        The original exception if it's already the expected type,
        otherwise wraps it in the expected type with proper chaining
    """
    logger.error(f"Error in {operation}: {exception}")
    if isinstance(exception, original_error_type):
        raise exception
    raise original_error_type(f"{fallback_message}: {exception}") from exception


def handle_output_saving(
    image: Image.Image,
    output_path: str | Path | None,
    output_format: str,
    output_config: OutputConfig,
    mol=None,
    auto_filename: bool = False,
    molecular_string: str | None = None,
) -> None:
    """
    Handle saving of rendered images with consistent logic.

    Provides centralized output handling for both single molecules and grids,
    with special handling for SVG vector rendering and automatic filename generation.

    Args:
        image: PIL Image to save. Must be a valid PIL Image object
        output_path: Optional output path. If None, no saving occurs unless auto_filename=True
        output_format: Output format (e.g., 'png', 'svg', 'jpg'). Must be supported format
        output_config: Output configuration with quality, DPI, and format-specific settings
        mol: Optional molecule object for SVG vector rendering. Required for true vector SVG
        auto_filename: Whether to generate automatic filename if output_path is None
        molecular_string: Source string for auto filename generation. Required if auto_filename=True

    Raises:
        OutputError: If saving fails due to file system issues or invalid parameters

    Example:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 100), 'white')
        >>> config = OutputConfig(format='png', quality=95)
        >>> handle_output_saving(img, 'molecule.png', 'png', config)
        # Saves image to molecule.png with specified quality
    """
    if output_path or auto_filename:
        output_handler = get_output_handler(output_format, output_config)

        # For SVG output, provide the molecule for true vector rendering
        if output_format.lower() == "svg" and mol is not None:
            output_handler.set_molecule(mol)

        if not output_path and auto_filename and molecular_string:
            # Generate safe filename
            base_name = create_safe_filename(
                molecular_string, output_handler.file_extension
            )
            output_path = Path.cwd() / base_name

        if output_path:
            output_handler.save(image, output_path)
