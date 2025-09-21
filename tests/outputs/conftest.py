"""
Shared test configuration and fixtures for output handler tests.

This module provides:
- Format configurations and capabilities
- Common test fixtures for images and configurations
- Utility functions for format testing
- Parametrized fixtures for comprehensive testing

The fixtures are organized into logical groups:
- Format definitions and capabilities
- Image fixtures (various types and sizes)
- Configuration fixtures
- Parametrized format fixtures
- Utility functions
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

# =============================================================================
# Format Definitions and Capabilities
# =============================================================================

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


# =============================================================================
# Directory and Path Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Image Fixtures
# =============================================================================


@pytest.fixture
def test_image():
    """Create a basic RGB test image (100x100 red)."""
    return Image.new("RGB", (100, 100), "red")


@pytest.fixture
def rgba_image():
    """Create an RGBA test image with transparency (100x100 semi-transparent red)."""
    return Image.new("RGBA", (100, 100), (255, 0, 0, 128))


@pytest.fixture
def rgba_opaque_image():
    """Create an RGBA test image without transparency (100x100 opaque red)."""
    return Image.new("RGBA", (100, 100), (255, 0, 0, 255))


@pytest.fixture
def la_image():
    """Create a grayscale+alpha test image (100x100)."""
    return Image.new("LA", (100, 100), (128, 200))


@pytest.fixture
def grayscale_image():
    """Create a grayscale test image (50x50)."""
    return Image.new("L", (50, 50), 128)


@pytest.fixture
def small_image():
    """Create a very small test image (10x10 blue)."""
    return Image.new("RGB", (10, 10), "blue")


@pytest.fixture
def large_image():
    """Create a larger test image (500x300 green)."""
    return Image.new("RGB", (500, 300), "green")


@pytest.fixture
def square_image():
    """Create a square test image (200x200 purple)."""
    return Image.new("RGB", (200, 200), "purple")


@pytest.fixture
def wide_image():
    """Create a wide test image (400x100 orange)."""
    return Image.new("RGB", (400, 100), "orange")


@pytest.fixture
def tall_image():
    """Create a tall test image (100x400 cyan)."""
    return Image.new("RGB", (100, 400), "cyan")


# =============================================================================
# Mock and Test Object Fixtures
# =============================================================================


@pytest.fixture
def mock_molecule():
    """Create a mock RDKit molecule for testing."""
    mol = MagicMock()
    mol.GetNumAtoms.return_value = 10
    mol.GetNumBonds.return_value = 9
    return mol


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def basic_config():
    """Create a basic output configuration with default settings."""
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


@pytest.fixture
def svg_vector_config():
    """Create a configuration for SVG vector output."""
    return OutputConfig(svg_use_vector=True)


@pytest.fixture
def svg_raster_config():
    """Create a configuration for SVG raster output."""
    return OutputConfig(svg_use_vector=False)


# =============================================================================
# Parametrized Format Fixtures
# =============================================================================


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


# =============================================================================
# Quality and Optimization Testing Fixtures
# =============================================================================


@pytest.fixture(
    params=[
        OutputConfig(quality=10),
        OutputConfig(quality=50),
        OutputConfig(quality=90),
        OutputConfig(quality=100),
    ]
)
def quality_config(request):
    """Parametrized fixture for different quality settings."""
    return request.param


@pytest.fixture(
    params=[
        OutputConfig(optimize=False),
        OutputConfig(optimize=True),
    ]
)
def optimization_config(request):
    """Parametrized fixture for optimization settings."""
    return request.param


# =============================================================================
# Image Variety Fixtures
# =============================================================================


@pytest.fixture(
    params=[
        lambda: Image.new("RGB", (100, 100), "red"),
        lambda: Image.new("RGBA", (100, 100), (0, 255, 0, 128)),
        lambda: Image.new("L", (100, 100), 128),
        lambda: Image.new("LA", (100, 100), (128, 200)),
    ]
)
def varied_image(request):
    """Parametrized fixture providing various image types."""
    return request.param()


@pytest.fixture(
    params=[
        (10, 10),  # Very small
        (50, 50),  # Small
        (100, 100),  # Medium
        (200, 150),  # Medium rectangular
        (400, 300),  # Large
    ]
)
def image_dimensions(request):
    """Parametrized fixture providing various image dimensions."""
    return request.param


# =============================================================================
# Utility Functions
# =============================================================================


def get_format_capabilities(format_name: str) -> dict:
    """
    Get capabilities for a specific format.

    Args:
        format_name: Name of the format (e.g., 'png', 'jpeg')

    Returns:
        Dictionary of format capabilities
    """
    return FORMAT_CAPABILITIES.get(format_name, {})


def get_expected_mode_conversion(format_name: str, original_mode: str) -> str:
    """
    Get expected image mode after conversion for a format.

    Args:
        format_name: Name of the format
        original_mode: Original image mode

    Returns:
        Expected mode after conversion
    """
    format_behavior = FORMAT_MODE_BEHAVIOR.get(format_name, {})
    return format_behavior.get(original_mode, original_mode)


def supports_transparency(format_name: str) -> bool:
    """
    Check if format supports transparency.

    Args:
        format_name: Name of the format

    Returns:
        True if format supports transparency
    """
    return get_format_capabilities(format_name).get("supports_alpha", False)


def supports_quality(format_name: str) -> bool:
    """
    Check if format supports quality settings.

    Args:
        format_name: Name of the format

    Returns:
        True if format supports quality settings
    """
    return get_format_capabilities(format_name).get("supports_quality", False)


def supports_optimization(format_name: str) -> bool:
    """
    Check if format supports optimization.

    Args:
        format_name: Name of the format

    Returns:
        True if format supports optimization
    """
    return get_format_capabilities(format_name).get("supports_optimization", False)


def is_raster_format(format_name: str) -> bool:
    """
    Check if format is a raster format.

    Args:
        format_name: Name of the format

    Returns:
        True if format is raster
    """
    return format_name in RASTER_FORMATS


def is_vector_format(format_name: str) -> bool:
    """
    Check if format is a vector format.

    Args:
        format_name: Name of the format

    Returns:
        True if format is vector
    """
    return format_name in VECTOR_FORMATS


def create_test_image_with_mode(mode: str, size: tuple = (100, 100)) -> Image.Image:
    """
    Create a test image with specific mode.

    Args:
        mode: PIL image mode (e.g., 'RGB', 'RGBA', 'L')
        size: Image dimensions as (width, height)

    Returns:
        PIL Image object
    """
    if mode == "RGB":
        return Image.new(mode, size, "red")
    elif mode == "RGBA":
        return Image.new(mode, size, (255, 0, 0, 128))
    elif mode == "L":
        return Image.new(mode, size, 128)
    elif mode == "LA":
        return Image.new(mode, size, (128, 200))
    elif mode == "P":
        return Image.new(mode, size, 0)
    elif mode == "1":
        return Image.new(mode, size, 1)
    else:
        return Image.new("RGB", size, "red")


def get_format_test_matrix():
    """
    Get a comprehensive test matrix for format testing.

    Returns:
        List of tuples (format_name, config, image_mode, expected_success)
    """
    test_cases = []

    for format_name in ALL_FORMATS.keys():
        for mode in ["RGB", "RGBA", "L", "LA"]:
            for config in [
                OutputConfig(),
                OutputConfig(quality=50),
                OutputConfig(optimize=True),
            ]:
                # All combinations should work, handlers should convert as needed
                test_cases.append((format_name, config, mode, True))

    return test_cases


# =============================================================================
# Test Constants
# =============================================================================

# Size limits for various test scenarios
MIN_FILE_SIZE = (
    30  # Minimum expected file size in bytes (some formats can be very small)
)
MAX_REASONABLE_FILE_SIZE = (
    15_000_000  # Maximum reasonable file size for tests (BMP can be large)
)
LARGE_IMAGE_DIMENSION = 2000  # Dimension for large image tests
SMALL_IMAGE_DIMENSION = 10  # Dimension for small image tests

# Quality test values
QUALITY_LOW = 20
QUALITY_HIGH = 95
QUALITY_MIN = 1
QUALITY_MAX = 100

# Test iteration counts
MEMORY_TEST_ITERATIONS = 10
STRESS_TEST_ITERATIONS = 5


# =============================================================================
# Test Helper Functions
# =============================================================================


def assert_valid_bytes_output(result: bytes, context: str = "operation") -> None:
    """
    Assert that a result is valid bytes output.

    Args:
        result: The result to validate
        context: Description of the operation for error messages
    """
    assert isinstance(result, bytes), f"{context} must return a bytes object"
    assert len(result) > 0, f"{context} must return non-empty bytes"
    assert len(result) < MAX_REASONABLE_FILE_SIZE, (
        f"{context} produced unreasonably large output: {len(result)} bytes"
    )


def assert_file_created_properly(
    file_path, expected_extension: str, context: str = "file"
) -> None:
    """
    Assert that a file was created properly with correct properties.

    Args:
        file_path: Path to the file to check
        expected_extension: Expected file extension (with dot)
        context: Description for error messages
    """
    assert file_path.exists(), f"{context} should be created at {file_path}"
    file_size = file_path.stat().st_size
    assert file_size > 0, f"Created {context} must not be empty"
    assert file_size < MAX_REASONABLE_FILE_SIZE, (
        f"Created {context} is unreasonably large: {file_size} bytes"
    )
    assert file_path.suffix == expected_extension, (
        f"{context} must have correct extension {expected_extension}"
    )


def assert_handler_interface_complete(handler, format_name: str) -> None:
    """
    Assert that a handler implements the complete required interface.

    Args:
        handler: Handler instance to validate
        format_name: Expected format name for error messages
    """
    # Required properties
    required_attrs = ["file_extension", "format_name", "config"]
    for attr in required_attrs:
        assert hasattr(handler, attr), f"Handler for {format_name} must have {attr}"

    # Required methods
    required_methods = ["save", "get_bytes"]
    for method in required_methods:
        assert hasattr(handler, method) and callable(getattr(handler, method)), (
            f"Handler for {format_name} must have callable {method} method"
        )

    # Property types
    assert isinstance(handler.file_extension, str), (
        f"file_extension must be string for {format_name}"
    )
    assert isinstance(handler.format_name, str), (
        f"format_name must be string for {format_name}"
    )
    assert isinstance(handler.config, OutputConfig), (
        f"config must be OutputConfig for {format_name}"
    )


def assert_config_preserved(
    handler, expected_config: OutputConfig, context: str = ""
) -> None:
    """
    Assert that handler config is properly preserved.

    Args:
        handler: Handler instance to check
        expected_config: Expected config object
        context: Additional context for error messages
    """
    prefix = f"{context}: " if context else ""
    assert handler.config is expected_config, (
        f"{prefix}Handler must use exact config instance"
    )
    assert handler.config.quality == expected_config.quality, (
        f"{prefix}Quality setting must be preserved"
    )
    assert handler.config.optimize == expected_config.optimize, (
        f"{prefix}Optimize setting must be preserved"
    )


def assert_raster_handler_properties(handler, format_name: str) -> None:
    """
    Assert that raster handlers have all required properties.

    Args:
        handler: Raster handler instance to validate
        format_name: Format name for error messages
    """
    required_properties = {
        "valid_extensions": list,
        "supports_alpha": bool,
        "supports_quality": bool,
        "pil_format": str,
    }

    for prop_name, expected_type in required_properties.items():
        assert hasattr(handler, prop_name), (
            f"Raster handlers must have {prop_name} property"
        )
        prop_value = getattr(handler, prop_name)
        assert isinstance(prop_value, expected_type), (
            f"{prop_name} must be a {expected_type.__name__}, got {type(prop_value)}"
        )


def validate_image_modes_handled(handler, format_name: str) -> None:
    """
    Validate that handler can process different image modes appropriately.

    Args:
        handler: Handler to test
        format_name: Format name for error messages
    """
    test_modes = ["RGB", "RGBA", "L", "LA"]
    for mode in test_modes:
        try:
            test_image = create_test_image_with_mode(mode, (50, 50))
            result = handler.get_bytes(test_image)
            assert_valid_bytes_output(result, f"{format_name} with {mode} mode")
        except Exception as e:
            # Some mode conversions are expected to fail or be converted
            # The key is that the handler shouldn't crash unexpectedly
            assert isinstance(e, (ValueError, OSError, IOError)), (
                f"Unexpected error type for {format_name} with {mode}: {type(e)}"
            )


def test_error_scenarios(handler, format_name: str) -> None:
    """
    Test common error scenarios for a handler.

    Args:
        handler: Handler to test
        format_name: Format name for error messages
    """
    # Test invalid inputs
    invalid_inputs = [None, "not an image", 123, []]
    for invalid_input in invalid_inputs:
        try:
            handler.get_bytes(invalid_input)
            assert False, (
                f"{format_name} should reject invalid input: {type(invalid_input)}"
            )
        except (TypeError, AttributeError, OSError, IOError):
            pass  # Expected

    # Test zero-size image
    try:
        zero_image = Image.new("RGB", (0, 0))
        handler.get_bytes(zero_image)
        assert False, f"{format_name} should reject zero-size image"
    except (ValueError, OSError, SystemError, MemoryError):
        pass  # Expected


def assert_quality_behavior_correct(
    format_name: str, high_bytes: bytes, low_bytes: bytes
) -> None:
    """
    Assert correct quality behavior based on format capabilities.

    Args:
        format_name: Name of the format being tested
        high_bytes: Output from high quality setting
        low_bytes: Output from low quality setting
    """
    if not supports_quality(format_name):
        if format_name == "pdf":
            # PDF may vary slightly due to timestamps or metadata
            size_difference = abs(len(high_bytes) - len(low_bytes))
            assert size_difference < 200, (
                f"PDF quality setting should not significantly affect output size, difference: {size_difference}"
            )
        else:
            # Other non-quality formats should produce identical output
            assert high_bytes == low_bytes, (
                f"Non-quality format {format_name} should produce identical output regardless of quality setting"
            )


def assert_optimization_behavior_correct(
    format_name: str, optimized_bytes: bytes, unoptimized_bytes: bytes
) -> None:
    """
    Assert correct optimization behavior based on format capabilities.

    Args:
        format_name: Name of the format being tested
        optimized_bytes: Output from optimization enabled
        unoptimized_bytes: Output from optimization disabled
    """
    if not supports_optimization(format_name):
        if format_name == "pdf":
            # PDF may vary slightly due to timestamps or metadata
            size_difference = abs(len(optimized_bytes) - len(unoptimized_bytes))
            assert size_difference < 200, (
                f"PDF optimization setting should not significantly affect output size, difference: {size_difference}"
            )
        else:
            # Other non-optimization formats should produce identical output
            assert optimized_bytes == unoptimized_bytes, (
                f"Non-optimization format {format_name} should produce identical output regardless of optimization setting"
            )


def create_test_pattern_image(width: int = 150, height: int = 150) -> Image.Image:
    """
    Create a test image with a color pattern for better compression testing.

    Args:
        width: Image width
        height: Image height

    Returns:
        PIL Image with color pattern
    """
    test_image = Image.new("RGB", (width, height))
    pattern_pixels = []
    for y in range(height):
        for x in range(width):
            r = (x * 255) // width
            g = (y * 255) // height
            b = ((x + y) * 255) // (width + height)
            pattern_pixels.append((r, g, b))
    test_image.putdata(pattern_pixels)
    return test_image


def validate_handler_interface(handler, format_name: str) -> None:
    """
    Validate that a handler implements the required interface.

    Args:
        handler: Handler instance to validate
        format_name: Expected format name for error messages
    """
    # Required properties
    assert hasattr(handler, "file_extension"), (
        f"Handler for {format_name} must have file_extension"
    )
    assert hasattr(handler, "format_name"), (
        f"Handler for {format_name} must have format_name"
    )
    assert hasattr(handler, "config"), f"Handler for {format_name} must have config"

    # Required methods
    assert hasattr(handler, "save") and callable(handler.save), (
        f"Handler for {format_name} must have save method"
    )
    assert hasattr(handler, "get_bytes") and callable(handler.get_bytes), (
        f"Handler for {format_name} must have get_bytes method"
    )

    # Property types
    assert isinstance(handler.file_extension, str), (
        f"file_extension must be string for {format_name}"
    )
    assert isinstance(handler.format_name, str), (
        f"format_name must be string for {format_name}"
    )
    assert isinstance(handler.config, OutputConfig), (
        f"config must be OutputConfig for {format_name}"
    )
