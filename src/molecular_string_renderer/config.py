"""
Configuration module for molecular string renderer.

Provides flexible configuration options for rendering behavior.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class RenderConfig(BaseModel):
    """Configuration for molecular rendering options.

    This class defines the essential rendering parameters that control how molecular
    structures are visualized.

    Attributes:
        width: Image width in pixels (100-2000).
        height: Image height in pixels (100-2000).
        background_color: Background color (name or hex code).
        dpi: DPI for high-quality output (72-600).
        show_hydrogen: Show explicit hydrogen atoms.
        show_carbon: Show carbon atom labels.
        highlight_atoms: List of atom indices to highlight.
        highlight_bonds: List of bond indices to highlight.
    """

    # Image dimensions
    width: int = Field(
        default=500, ge=100, le=2000, description="Image width in pixels"
    )
    height: int = Field(
        default=500, ge=100, le=2000, description="Image height in pixels"
    )

    # Rendering options
    background_color: str = Field(
        default="white", description="Background color (name or hex)"
    )

    # Quality settings
    dpi: int = Field(
        default=150, ge=72, le=600, description="DPI for high-quality output"
    )

    # Molecular display options
    show_hydrogen: bool = Field(
        default=False, description="Show explicit hydrogen atoms"
    )
    show_carbon: bool = Field(default=False, description="Show carbon atom labels")
    highlight_atoms: list[int] | None = Field(
        default=None, description="List of atom indices to highlight"
    )
    highlight_bonds: list[int] | None = Field(
        default=None, description="List of bond indices to highlight"
    )

    @field_validator("background_color")
    @classmethod
    def validate_color(cls, v):
        """Validate color is either a valid name or hex code.

        Args:
            v: The color value to validate.

        Returns:
            The validated color value.

        Raises:
            ValueError: If the hex color format is invalid.
        """
        if v.startswith("#"):
            if len(v) not in [4, 7] or not all(
                c in "0123456789abcdefABCDEF" for c in v[1:]
            ):
                raise ValueError(f"Invalid hex color: {v}")
        return v

    @property
    def size(self) -> tuple[int, int]:
        """Get image size as tuple.

        Returns:
            A tuple containing (width, height) in pixels.
        """
        return (self.width, self.height)

    def to_rdkit_options(self) -> dict[str, Any]:
        """Convert config to RDKit drawing options.

        Returns:
            Dictionary of RDKit drawing options compatible with the molecule rendering.
        """
        return {
            "addAtomIndices": False,
            "addBondIndices": False,
            "highlightAtoms": self.highlight_atoms or [],
            "highlightBonds": self.highlight_bonds or [],
            "explicitMethyl": self.show_carbon,
        }


class ParserConfig(BaseModel):
    """Configuration for molecular parsing options.

    This class defines parameters that control how molecular strings
    (SMILES, InChI, etc.) are parsed and processed.

    Attributes:
        sanitize: Sanitize molecule after parsing to fix common issues.
        show_hydrogen: Show explicit hydrogen atoms (determines if hydrogens are removed).
    """

    # Sanitization options
    sanitize: bool = Field(default=True, description="Sanitize molecule after parsing")
    show_hydrogen: bool = Field(
        default=False, description="Show explicit hydrogen atoms"
    )

    @property
    def remove_hs(self) -> bool:
        """Remove explicit hydrogens (inverse of show_hydrogen).

        Returns:
            True if hydrogens should be removed, False otherwise.
        """
        return not self.show_hydrogen


class OutputConfig(BaseModel):
    """Configuration for output generation.

    This class defines parameters that control how the rendered molecular
    images are saved and optimized.

    Attributes:
        format: Output format (png, svg, jpg, jpeg, pdf).
        quality: Output quality from 1-100.
        optimize: Optimize output file size.
        svg_sanitize: Sanitize SVG output for security.
    """

    format: str = Field(default="png", description="Output format (png, svg, etc.)")
    quality: int = Field(default=95, ge=1, le=100, description="Output quality (1-100)")
    optimize: bool = Field(default=True, description="Optimize output file size")
    svg_sanitize: bool = Field(default=True, description="Sanitize SVG output for security")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        """Validate output format is supported.

        Args:
            v: The format value to validate.

        Returns:
            The validated format value in lowercase.

        Raises:
            ValueError: If the format is not supported.
        """
        supported = ["png", "svg", "jpg", "jpeg", "pdf", "webp", "tiff", "tif", "bmp"]
        if v.lower() not in supported:
            raise ValueError(f"Unsupported format: {v}. Supported: {supported}")
        return v.lower()
