"""
Configuration utilities for molecular rendering.

Provides functions for initializing, coordinating, and validating configurations
across the molecular rendering pipeline.
"""

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.validation import validate_configuration_compatibility


def initialize_configurations(
    render_config: RenderConfig | None,
    parser_config: ParserConfig | None,
    output_config: OutputConfig | None,
    output_format: str,
) -> tuple[RenderConfig, ParserConfig, OutputConfig]:
    """
    Initialize and coordinate configuration objects with defaults.

    Ensures configuration consistency by auto-coordinating hydrogen display
    settings between render and parser configurations. This prevents common
    issues where rendering expects hydrogen atoms that weren't parsed.

    Args:
        render_config: Optional render configuration. If None, uses RenderConfig()
            with defaults (500x500px, white background, no hydrogens shown)
        parser_config: Optional parser configuration. If None, uses ParserConfig()
            with defaults (sanitize=True, strict=False, no hydrogens shown)
        output_config: Optional output configuration. If None, uses OutputConfig()
            with defaults (quality=95, DPI=150, optimize=True)
        output_format: Output format for default output config (e.g., 'png', 'svg').
            Must be one of the supported formats

    Returns:
        Tuple of (render_config, parser_config, output_config) with proper coordination.
        If render_config shows hydrogens, parser_config will be adjusted to include them

    Raises:
        ConfigurationError: If configurations are incompatible (e.g., JPEG with transparency)

    Example:
        Auto-coordination of hydrogen settings:
        >>> render_cfg = RenderConfig(show_hydrogen=True)
        >>> render_cfg, parser_cfg, output_cfg = initialize_configurations(
        ...     render_cfg, None, None, "png"
        ... )
        >>> parser_cfg.show_hydrogen  # Will be True to match render config
        True

        Default initialization:
        >>> configs = initialize_configurations(None, None, None, "svg")
        >>> len(configs)  # Returns (RenderConfig, ParserConfig, OutputConfig)
        3
    """
    render_config = render_config or RenderConfig()
    parser_config = parser_config or ParserConfig()
    output_config = output_config or OutputConfig(format=output_format)

    # Auto-coordinate hydrogen display settings
    if render_config.show_hydrogen and not parser_config.show_hydrogen:
        parser_config = ParserConfig(
            sanitize=parser_config.sanitize,
            show_hydrogen=True,  # Keep hydrogens for display
            strict=parser_config.strict,
        )

    # Validate compatibility
    validate_configuration_compatibility(render_config, parser_config, output_config)

    return render_config, parser_config, output_config
