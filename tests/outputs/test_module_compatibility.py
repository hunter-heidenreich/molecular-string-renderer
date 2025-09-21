"""
Module compatibility and import tests for outputs submodule.

Tests that the outputs submodule maintains proper import structure
and backward compatibility.
"""

from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import (
    BMPOutput,
    JPEGOutput,
    OutputHandler,
    PDFOutput,
    PNGOutput,
    SVGOutput,
    TIFFOutput,
    WEBPOutput,
    get_output_handler,
)


class TestSubModuleImports:
    """Test that all imports work correctly from the sub-module."""

    def test_basic_imports(self):
        """Test that basic imports work without errors."""
        # These should not raise ImportError
        from molecular_string_renderer.outputs import (
            OutputHandler,
            create_safe_filename,
            get_output_handler,
        )

        assert OutputHandler is not None
        assert get_output_handler is not None
        assert create_safe_filename is not None

    def test_all_handler_imports(self):
        """Test that all handler classes can be imported."""
        handlers = [
            PNGOutput,
            JPEGOutput,
            WEBPOutput,
            TIFFOutput,
            BMPOutput,
            SVGOutput,
            PDFOutput,
        ]

        for handler_class in handlers:
            assert handler_class is not None
            assert issubclass(handler_class, OutputHandler)

    def test_factory_import_and_usage(self):
        """Test that factory function imports and works correctly."""
        handler = get_output_handler("png")
        assert isinstance(handler, PNGOutput)
        assert isinstance(handler, OutputHandler)

    def test_individual_modules_importable(self):
        """Test that individual sub-modules can be imported directly."""
        # These should not raise ImportError
        from molecular_string_renderer.outputs import (
            base,
            factory,
            raster,
            utils,
            vector,
        )

        assert base is not None
        assert factory is not None
        assert raster is not None
        assert vector is not None
        assert utils is not None


class TestSubModuleBackwardCompatibility:
    """Test that the sub-module maintains backward compatibility."""

    def test_legacy_imports_still_work(self):
        """Test that old import patterns still work."""
        # These imports should work exactly as they did before
        from molecular_string_renderer.outputs import PNGOutput as LegacyPNGOutput
        from molecular_string_renderer.outputs import (
            get_output_handler as legacy_get_handler,
        )

        # Should be the same classes
        assert LegacyPNGOutput is PNGOutput
        assert legacy_get_handler is get_output_handler

    def test_factory_function_compatibility(self):
        """Test that factory function maintains same API."""
        # Should work exactly as before
        handler = get_output_handler("png")
        assert isinstance(handler, PNGOutput)

        # With config
        config = OutputConfig(quality=80)
        handler_with_config = get_output_handler("png", config)
        assert isinstance(handler_with_config, PNGOutput)
        assert handler_with_config.config is config

    def test_core_module_integration(self):
        """Test that core module can still import and use outputs."""
        # Import as core module would
        from molecular_string_renderer.outputs import get_output_handler

        # Use as core module would
        handler = get_output_handler("png")
        test_image = Image.new("RGB", (100, 100), "red")
        result = handler.get_bytes(test_image)
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestSubModuleStructure:
    """Test the internal structure of the sub-module."""

    def test_module_has_correct_exports(self):
        """Test that the sub-module exports the correct symbols."""
        import molecular_string_renderer.outputs as outputs_module

        expected_exports = [
            "OutputHandler",
            "get_output_handler",
            "create_safe_filename",
            "PNGOutput",
            "JPEGOutput",
            "WEBPOutput",
            "TIFFOutput",
            "BMPOutput",
            "SVGOutput",
            "PDFOutput",
        ]

        for export in expected_exports:
            assert hasattr(outputs_module, export), f"Missing export: {export}"

        # Check __all__ if it exists
        if hasattr(outputs_module, "__all__"):
            for export in expected_exports:
                assert export in outputs_module.__all__, (
                    f"Export {export} not in __all__"
                )


class TestDocumentationAndMetadata:
    """Test that documentation and metadata are preserved."""

    def test_handler_classes_have_docstrings(self):
        """Test that all handler classes have proper docstrings."""
        handler_classes = [
            PNGOutput,
            JPEGOutput,
            WEBPOutput,
            TIFFOutput,
            BMPOutput,
            SVGOutput,
            PDFOutput,
        ]

        for handler_class in handler_classes:
            assert handler_class.__doc__ is not None
            assert len(handler_class.__doc__.strip()) > 0

    def test_module_has_docstring(self):
        """Test that the main module has a docstring."""
        import molecular_string_renderer.outputs as outputs_module

        assert outputs_module.__doc__ is not None
        assert len(outputs_module.__doc__.strip()) > 0

    def test_factory_function_has_docstring(self):
        """Test that factory function has proper docstring."""
        assert get_output_handler.__doc__ is not None
        assert "Args:" in get_output_handler.__doc__
        assert "Returns:" in get_output_handler.__doc__


class TestCrossFormatConsistency:
    """Test consistency across different output formats."""

    def test_all_formats_handle_same_image_types(self):
        """Test that all formats can handle the same basic image types."""
        image_types = [
            Image.new("RGB", (50, 50), "red"),
            Image.new("RGBA", (50, 50), (255, 0, 0, 128)),
            Image.new("L", (50, 50), 128),
        ]

        format_names = ["png", "jpg", "webp", "tiff", "bmp", "svg", "pdf"]

        for format_name in format_names:
            handler = get_output_handler(format_name)
            for img in image_types:
                # Should not raise an error
                result = handler.get_bytes(img)
                assert len(result) > 0

    def test_config_parameter_consistency(self):
        """Test that config parameters are handled consistently."""
        configs = [
            OutputConfig(),
            OutputConfig(quality=50),
            OutputConfig(optimize=True),
            OutputConfig(quality=90, optimize=True),
        ]

        test_image = Image.new("RGB", (50, 50), "red")
        format_names = ["png", "jpg", "webp", "tiff", "bmp", "svg", "pdf"]

        for config in configs:
            for format_name in format_names:
                handler = get_output_handler(format_name, config)
                # Should not raise an error regardless of config support
                result = handler.get_bytes(test_image)
                assert len(result) > 0

    def test_file_extension_consistency(self):
        """Test that file extensions are consistent with format names."""
        format_expectations = {
            "png": ".png",
            "jpg": ".jpg",
            "jpeg": ".jpg",  # JPEG uses .jpg extension
            "webp": ".webp",
            "tiff": ".tiff",
            "tif": ".tiff",  # TIF uses .tiff extension
            "bmp": ".bmp",
            "svg": ".svg",
            "pdf": ".pdf",
        }

        for format_name, expected_ext in format_expectations.items():
            handler = get_output_handler(format_name)
            assert handler.file_extension == expected_ext
