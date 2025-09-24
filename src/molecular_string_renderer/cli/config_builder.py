"""
Configuration builder for CLI arguments.

This module handles the creation of configuration objects from
parsed command-line arguments and format determination.
"""

from pathlib import Path

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.exceptions import CLIConfigurationError


def determine_output_format(output_path: str | None, output_format: str | None) -> str:
    """Determine output format from filename or explicit format.

    Args:
        output_path: Optional output file path.
        output_format: Optional explicit output format.

    Returns:
        The determined output format string (e.g., 'png', 'svg').
    """
    if output_format:
        return output_format.lower()

    if output_path:
        suffix = Path(output_path).suffix.lower()
        format_map = {
            ".png": "png",
            ".svg": "svg",
            ".jpg": "jpg",
            ".jpeg": "jpg",
            ".pdf": "pdf",
            ".webp": "webp",
            ".tiff": "tiff",
            ".tif": "tiff",
            ".bmp": "bmp",
            ".gif": "gif",
        }
        return format_map.get(suffix, "png")

    return "png"


def create_configs(args) -> tuple[RenderConfig, ParserConfig, OutputConfig]:
    """Create configuration objects from command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A tuple containing (render_config, parser_config, output_config).

    Raises:
        CLIConfigurationError: If configuration values are invalid.
    """
    # Determine image dimensions
    width = args.width or args.size
    height = args.height or args.size

    # Validate dimensions
    if width < 100 or width > 2000:
        raise CLIConfigurationError(
            f"Width must be between 100-2000 pixels, got {width}"
        )
    if height < 100 or height > 2000:
        raise CLIConfigurationError(
            f"Height must be between 100-2000 pixels, got {height}"
        )

    # Validate DPI
    if args.dpi < 72 or args.dpi > 600:
        raise CLIConfigurationError(f"DPI must be between 72-600, got {args.dpi}")

    # Validate quality
    if args.quality < 1 or args.quality > 100:
        raise CLIConfigurationError(
            f"Quality must be between 1-100, got {args.quality}"
        )

    # Determine lossless setting
    if hasattr(args, 'lossless') and hasattr(args, 'no_lossless') and args.lossless and args.no_lossless:
        raise CLIConfigurationError("Cannot specify both --lossless and --no-lossless")
    
    # Default to lossless=True, unless --no-lossless is specified
    lossless = True
    if hasattr(args, 'no_lossless') and args.no_lossless:
        lossless = False
    elif hasattr(args, 'lossless') and args.lossless:
        lossless = True

    try:
        render_config = RenderConfig(
            width=width,
            height=height,
            background_color=args.background_color,
            show_hydrogen=args.show_hydrogen,
            show_carbon=args.show_carbon,
        )

        parser_config = ParserConfig(
            sanitize=True,
            show_hydrogen=args.show_hydrogen,
        )

        output_format = determine_output_format(args.output, args.output_format)

        output_config = OutputConfig(
            format=output_format,
            quality=args.quality,
            optimize=not args.no_optimize,
            dpi=args.dpi,
            lossless=lossless,
        )
    except Exception as e:
        raise CLIConfigurationError(f"Failed to create configuration: {e}") from e

    return render_config, parser_config, output_config
