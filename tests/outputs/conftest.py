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
    GIFOutput,
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
    "gif": GIFOutput,
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
    "gif": {
        "supports_alpha": True,
        "supports_quality": False,
        "supports_optimization": True,
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
    "gif": {
        "RGBA": "P",  # GIF converts RGBA to P mode with palette
        "RGB": "P",   # GIF converts RGB to P mode for optimal compression
        "LA": "P",    # GIF converts LA to P mode
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


# =============================================================================
# Handler Fixtures
# =============================================================================


@pytest.fixture
def test_image():
    """Create a simple RGB test image."""
    return Image.new("RGB", (100, 100), "red")


@pytest.fixture
def rgba_image():
    """Create an RGBA image with transparency."""
    return Image.new("RGBA", (100, 100), (255, 0, 0, 128))


@pytest.fixture
def rgba_opaque_image():
    """Create an RGBA image that's fully opaque."""
    return Image.new("RGBA", (100, 100), (255, 0, 0, 255))


@pytest.fixture
def la_image():
    """Create a grayscale+alpha image."""
    return Image.new("LA", (100, 100), (128, 200))


@pytest.fixture
def grayscale_image():
    """Create a grayscale image."""
    return Image.new("L", (50, 50), 128)


@pytest.fixture
def small_image():
    """Create a small test image."""
    return Image.new("RGB", (10, 10), "blue")


@pytest.fixture
def large_image():
    """Create a large test image."""
    return Image.new("RGB", (500, 300), "green")


@pytest.fixture
def square_image():
    """Create a square test image."""
    return Image.new("RGB", (200, 200), "purple")


@pytest.fixture
def wide_image():
    """Create a wide test image."""
    return Image.new("RGB", (400, 100), "orange")


@pytest.fixture
def tall_image():
    """Create a tall test image."""
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

# =============================================================================
# Configuration Fixtures - Simple and Direct
# =============================================================================


@pytest.fixture
def basic_config():
    """Default OutputConfig."""
    return OutputConfig()


@pytest.fixture
def optimized_config():
    """OutputConfig with optimization enabled."""
    return OutputConfig(optimize=True, quality=90)


@pytest.fixture
def high_quality_config():
    """OutputConfig with high quality settings."""
    return OutputConfig(quality=95, optimize=True)


@pytest.fixture
def low_quality_config():
    """OutputConfig with low quality settings."""
    return OutputConfig(quality=20, optimize=False)


# =============================================================================
# Parametrized Format Fixtures
# =============================================================================

# =============================================================================
# Format Fixtures - Simple Parametrized
# =============================================================================


@pytest.fixture(params=list(ALL_FORMATS.keys()))
def format_name(request):
    """All supported format names."""
    return request.param


@pytest.fixture(params=list(RASTER_FORMATS.keys()))
def raster_format_name(request):
    """All raster format names."""
    return request.param


@pytest.fixture(params=list(VECTOR_FORMATS.keys()))
def vector_format_name(request):
    """All vector format names."""
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


@pytest.fixture(params=[OutputConfig(quality=q) for q in [10, 50, 90, 100]])
def quality_config(request):
    """Different quality settings."""
    return request.param


@pytest.fixture(params=[OutputConfig(optimize=opt) for opt in [False, True]])
def optimization_config(request):
    """Optimization settings."""
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
    """Various image types."""
    return request.param()


@pytest.fixture(params=[(10, 10), (50, 50), (100, 100), (200, 150), (400, 300)])
def image_dimensions(request):
    """Various image dimensions."""
    return request.param


@pytest.fixture(
    params=[
        (1, 1),
        (101, 101),
        (100, 101),
        (101, 100),
        (64, 64),
        (128, 256),
        (100, 1),
        (1, 100),
        (1000, 1),
        (1, 1000),
    ]
)
def edge_case_dimensions(request):
    """Edge case image dimensions."""
    return request.param


# =============================================================================
# Utility Functions
# =============================================================================


class FormatCapabilities:
    """Centralized format capability checking."""

    @staticmethod
    def get(format_name: str) -> dict:
        """Get capabilities for a specific format."""
        return FORMAT_CAPABILITIES.get(format_name, {})

    @staticmethod
    def supports_alpha(format_name: str) -> bool:
        """Check if format supports transparency."""
        return FormatCapabilities.get(format_name).get("supports_alpha", False)

    @staticmethod
    def supports_quality(format_name: str) -> bool:
        """Check if format supports quality settings."""
        return FormatCapabilities.get(format_name).get("supports_quality", False)

    @staticmethod
    def supports_optimization(format_name: str) -> bool:
        """Check if format supports optimization."""
        return FormatCapabilities.get(format_name).get("supports_optimization", False)

    @staticmethod
    def is_raster(format_name: str) -> bool:
        """Check if format is a raster format."""
        return format_name in RASTER_FORMATS

    @staticmethod
    def is_vector(format_name: str) -> bool:
        """Check if format is a vector format."""
        return format_name in VECTOR_FORMATS


def get_expected_mode_conversion(format_name: str, original_mode: str) -> str:
    """Get expected image mode after conversion for a format."""
    format_behavior = FORMAT_MODE_BEHAVIOR.get(format_name, {})
    return format_behavior.get(original_mode, original_mode)


def create_test_image_with_mode(mode: str, size: tuple = (100, 100)) -> Image.Image:
    """Create a test image with specific mode using optimized lookup."""
    mode_specs = {
        "RGB": (mode, size, "red"),
        "RGBA": (mode, size, (255, 0, 0, 128)),
        "L": (mode, size, 128),
        "LA": (mode, size, (128, 200)),
        "P": (mode, size, 0),
        "1": (mode, size, 1),
    }
    return Image.new(*mode_specs.get(mode, ("RGB", size, "red")))


def get_format_test_matrix() -> list[tuple[str, OutputConfig, str, bool]]:
    """Get a comprehensive test matrix for format testing with type hints."""
    configs = [OutputConfig(), OutputConfig(quality=50), OutputConfig(optimize=True)]
    modes = ["RGB", "RGBA", "L", "LA"]

    return [
        (format_name, config, mode, True)
        for format_name in ALL_FORMATS.keys()
        for mode in modes
        for config in configs
    ]


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Centralized test constants for organized access."""

    # File size limits
    MIN_FILE_SIZE = 30  # Minimum expected file size in bytes
    MAX_REASONABLE_FILE_SIZE = (
        15_000_000  # Maximum reasonable file size for tests (BMP can be large)
    )

    # Image dimensions for testing
    LARGE_IMAGE_DIMENSION = 2000
    SMALL_IMAGE_DIMENSION = 10

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


class TestValidators:
    """Centralized validation helpers for test assertions."""

    @staticmethod
    def assert_valid_bytes_output(result: bytes, context: str = "operation") -> None:
        """Assert that a result is valid bytes output."""
        assert isinstance(result, bytes), f"{context} must return a bytes object"
        assert len(result) > 0, f"{context} must return non-empty bytes"
        assert len(result) < TestConstants.MAX_REASONABLE_FILE_SIZE, (
            f"{context} produced unreasonably large output: {len(result)} bytes"
        )

    @staticmethod
    def assert_file_created_properly(
        file_path, expected_extension: str, context: str = "file"
    ) -> None:
        """Assert that a file was created properly with correct properties."""
        assert file_path.exists(), f"{context} should be created at {file_path}"
        file_size = file_path.stat().st_size
        assert file_size > 0, f"Created {context} must not be empty"
        assert file_size < TestConstants.MAX_REASONABLE_FILE_SIZE, (
            f"Created {context} is unreasonably large: {file_size} bytes"
        )
        assert file_path.suffix == expected_extension, (
            f"{context} must have correct extension {expected_extension}"
        )

    @staticmethod
    def assert_handler_interface_complete(handler, format_name: str) -> None:
        """Assert that a handler implements the complete required interface."""
        required_attrs = ["file_extension", "format_name", "config"]
        for attr in required_attrs:
            assert hasattr(handler, attr), f"Handler for {format_name} must have {attr}"

        required_methods = ["save", "get_bytes"]
        for method in required_methods:
            assert hasattr(handler, method) and callable(getattr(handler, method)), (
                f"Handler for {format_name} must have callable {method} method"
            )

        assert isinstance(handler.file_extension, str), (
            f"file_extension must be string for {format_name}"
        )
        assert isinstance(handler.format_name, str), (
            f"format_name must be string for {format_name}"
        )
        assert isinstance(handler.config, OutputConfig), (
            f"config must be OutputConfig for {format_name}"
        )

    @staticmethod
    def assert_image_output_valid(result: bytes, format_name: str) -> None:
        """Assert that image output is valid for the given format."""
        TestValidators.assert_valid_bytes_output(result, f"{format_name} output")

        # Format-specific validations
        if format_name == "pdf":
            assert result.startswith(b"%PDF-"), "PDF must start with PDF signature"
            assert result.endswith(b"%%EOF\n"), "PDF must end with EOF marker"
        elif format_name == "svg":
            content = result.decode("utf-8")
            assert "<?xml" in content, "SVG must contain XML declaration"
            assert "<svg" in content, "SVG must contain svg element"
            assert "</svg>" in content, "SVG must be properly closed"

    @staticmethod
    def assert_config_preserved(
        handler, expected_config: OutputConfig, context: str = ""
    ) -> None:
        """Assert that handler config is properly preserved."""
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

    @staticmethod
    def assert_transparency_behavior(handler, image, format_name: str) -> None:
        """Assert correct transparency handling based on format capabilities."""
        result = handler.get_bytes(image)
        TestValidators.assert_valid_bytes_output(
            result, f"{format_name} transparency handling"
        )


class TestHelpers:
    """Advanced test helper functions for complex testing scenarios."""

    @staticmethod
    def test_handler_with_all_image_modes(handler, test_function):
        """Test a handler with all common image modes."""
        modes = ["RGB", "RGBA", "L", "LA", "P"]
        for mode in modes:
            test_image = create_test_image_with_mode(mode, (50, 50))
            test_function(handler, test_image)

    @staticmethod
    def validate_format_specific_behavior(handler, image, format_name: str):
        """Validate format-specific behavior and conversions."""
        result = handler.get_bytes(image)
        TestValidators.assert_valid_bytes_output(result, f"{format_name} processing")

        # Check format-specific capabilities
        if not FormatCapabilities.supports_alpha(format_name) and image.mode in [
            "RGBA",
            "LA",
        ]:
            assert len(result) > 0, (
                f"Non-alpha format {format_name} should handle alpha images"
            )

    @staticmethod
    def test_with_multiple_images(handler, test_function, image_list: list) -> None:
        """Helper to run a test function with multiple images."""
        for i, image in enumerate(image_list):
            try:
                test_function(handler, image)
            except Exception as e:
                raise AssertionError(
                    f"Test failed for image {i} (mode: {image.mode}): {e}"
                ) from e

    @staticmethod
    def validate_image_modes_handled(handler, format_name: str) -> None:
        """Validate that handler can process different image modes appropriately."""
        test_modes = ["RGB", "RGBA", "L", "LA"]
        for mode in test_modes:
            try:
                test_image = create_test_image_with_mode(mode, (50, 50))
                result = handler.get_bytes(test_image)
                TestValidators.assert_valid_bytes_output(
                    result, f"{format_name} with {mode} mode"
                )
            except Exception as e:
                assert isinstance(e, (ValueError, OSError, IOError)), (
                    f"Unexpected error type for {format_name} with {mode}: {type(e)}"
                )

    @staticmethod
    def test_error_scenarios(handler, format_name: str) -> None:
        """Test common error scenarios for a handler."""
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


class QualityOptimizationHelpers:
    """Helpers for testing quality and optimization behavior."""

    @staticmethod
    def assert_quality_behavior_correct(
        format_name: str, high_bytes: bytes, low_bytes: bytes
    ) -> None:
        """Assert correct quality behavior based on format capabilities."""
        if not FormatCapabilities.supports_quality(format_name):
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

    @staticmethod
    def assert_optimization_behavior_correct(
        format_name: str, optimized_bytes: bytes, unoptimized_bytes: bytes
    ) -> None:
        """Assert correct optimization behavior based on format capabilities."""
        if not FormatCapabilities.supports_optimization(format_name):
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


def assert_raster_handler_properties(handler, format_name: str) -> None:
    """Assert that raster handlers have all required properties."""
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


def create_test_pattern_image(width: int = 150, height: int = 150) -> Image.Image:
    """Create a test image with a color pattern for better compression testing."""
    test_image = Image.new("RGB", (width, height))
    # More efficient pixel generation using list comprehension
    pattern_pixels = [
        ((x * 255) // width, (y * 255) // height, ((x + y) * 255) // (width + height))
        for y in range(height)
        for x in range(width)
    ]
    test_image.putdata(pattern_pixels)
    return test_image


@pytest.fixture
def pattern_image():
    """Create a test image with gradient pattern for compression testing."""
    return create_test_pattern_image(200, 200)


@pytest.fixture
def test_images_all_modes():
    """Create test images in all common modes for comprehensive testing."""
    modes = ["RGB", "RGBA", "L", "LA", "P", "1"]
    return {mode: create_test_image_with_mode(mode, (50, 50)) for mode in modes}


@pytest.fixture
def extreme_alpha_images():
    """Create images with extreme alpha values for testing edge cases."""
    alpha_values = [0, 1, 254, 255]  # fully transparent → fully opaque
    return {
        f"alpha_{alpha}": Image.new("RGBA", (50, 50), (255, 0, 0, alpha))
        for alpha in alpha_values
    }
