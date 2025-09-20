"""
Shared test configuration and fixtures for output handler tests.

Provides common fixtures and utilities used across all output handler tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import (
    BMPOutput,
    JPEGOutput,
    PDFOutput,
    PNGOutput,
    SVGOutput,
    TIFFOutput,
    WEBPOutput,
    get_output_handler,
)


# Format configurations for testing
RASTER_FORMATS = {
    "png": PNGOutput,
    "jpeg": JPEGOutput,
    "webp": WEBPOutput,
    "tiff": TIFFOutput,
    "bmp": BMPOutput,
}

VECTOR_FORMATS = {
    "svg": SVGOutput,
    "pdf": PDFOutput,
}

ALL_FORMATS = {**RASTER_FORMATS, **VECTOR_FORMATS}

# Format-specific capabilities and limitations
FORMAT_CAPABILITIES = {
    "png": {
        "supports_alpha": True,
        "supports_quality": False,
        "supports_optimization": True,
    },
    "jpeg": {
        "supports_alpha": False,
        "supports_quality": True,
        "supports_optimization": True,
    },
    "webp": {
        "supports_alpha": True,
        "supports_quality": True,
        "supports_optimization": True,
    },
    "tiff": {
        "supports_alpha": True,
        "supports_quality": False,
        "supports_optimization": True,
    },
    "bmp": {
        "supports_alpha": False,
        "supports_quality": False,
        "supports_optimization": False,
    },
    "svg": {
        "supports_alpha": True,
        "supports_quality": False,
        "supports_optimization": True,
    },
    "pdf": {
        "supports_alpha": True,
        "supports_quality": False,
        "supports_optimization": False,
    },
}

# Expected format-specific behavior for different image modes
FORMAT_MODE_BEHAVIOR = {
    "jpeg": {
        "RGBA": "RGB",  # JPEG converts RGBA to RGB
        "LA": "RGB",  # JPEG converts LA to RGB
    },
    "bmp": {
        "RGBA": "RGB",  # BMP may not support RGBA in all cases
        "LA": "RGB",  # BMP converts LA to RGB
    },
    # PNG, WEBP, TIFF preserve transparency by default
    # SVG and PDF handle transparency through their own mechanisms
}


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_image():
    """Create a basic RGB test image."""
    return Image.new("RGB", (100, 100), "red")


@pytest.fixture
def rgba_image():
    """Create an RGBA test image with transparency."""
    return Image.new("RGBA", (100, 100), (255, 0, 0, 128))


@pytest.fixture
def rgba_opaque_image():
    """Create an RGBA test image without transparency."""
    return Image.new("RGBA", (100, 100), (255, 0, 0, 255))


@pytest.fixture
def la_image():
    """Create a grayscale+alpha test image."""
    return Image.new("LA", (100, 100), (128, 200))


@pytest.fixture
def grayscale_image():
    """Create a grayscale test image."""
    return Image.new("L", (50, 50), 128)


@pytest.fixture
def small_image():
    """Create a very small test image."""
    return Image.new("RGB", (10, 10), "blue")


@pytest.fixture
def large_image():
    """Create a larger test image."""
    return Image.new("RGB", (500, 300), "green")


@pytest.fixture
def mock_molecule():
    """Create a mock RDKit molecule for testing."""
    mol = MagicMock()
    mol.GetNumAtoms.return_value = 10
    mol.GetNumBonds.return_value = 9
    return mol


@pytest.fixture
def basic_config():
    """Create a basic output configuration."""
    return OutputConfig()


@pytest.fixture
def optimized_config():
    """Create an optimized output configuration."""
    return OutputConfig(optimize=True, quality=90)


@pytest.fixture
def high_quality_config():
    """Create a high-quality output configuration."""
    return OutputConfig(quality=95, optimize=True)


@pytest.fixture
def low_quality_config():
    """Create a low-quality output configuration."""
    return OutputConfig(quality=20, optimize=False)


@pytest.fixture(params=list(ALL_FORMATS.keys()))
def format_name(request):
    """Parametrized fixture providing all supported format names."""
    return request.param


@pytest.fixture(params=list(RASTER_FORMATS.keys()))
def raster_format_name(request):
    """Parametrized fixture providing all raster format names."""
    return request.param


@pytest.fixture(params=list(VECTOR_FORMATS.keys()))
def vector_format_name(request):
    """Parametrized fixture providing all vector format names."""
    return request.param


@pytest.fixture
def output_handler(format_name):
    """Create an output handler for the given format."""
    return get_output_handler(format_name)


@pytest.fixture
def raster_output_handler(raster_format_name):
    """Create a raster output handler for the given format."""
    return get_output_handler(raster_format_name)


@pytest.fixture
def vector_output_handler(vector_format_name):
    """Create a vector output handler for the given format."""
    return get_output_handler(vector_format_name)


def get_format_capabilities(format_name: str) -> dict:
    """Get capabilities for a specific format."""
    return FORMAT_CAPABILITIES.get(format_name, {})


def get_expected_mode_conversion(format_name: str, original_mode: str) -> str:
    """Get expected image mode after conversion for a format."""
    format_behavior = FORMAT_MODE_BEHAVIOR.get(format_name, {})
    return format_behavior.get(original_mode, original_mode)


def supports_transparency(format_name: str) -> bool:
    """Check if format supports transparency."""
    return get_format_capabilities(format_name).get("supports_alpha", False)


def supports_quality(format_name: str) -> bool:
    """Check if format supports quality settings."""
    return get_format_capabilities(format_name).get("supports_quality", False)


def supports_optimization(format_name: str) -> bool:
    """Check if format supports optimization."""
    return get_format_capabilities(format_name).get("supports_optimization", False)
