"""
Integration tests for the renderers sub-module.

Tests that the new sub-module structure integrates correctly with the rest of the codebase.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor

from molecular_string_renderer.config import RenderConfig
from molecular_string_renderer.renderers import (
    ColorUtils,
    DrawerConfigurationManager,
    MolecularRenderer,
    Molecule2DRenderer,
    MoleculeGridRenderer,
    get_renderer,
)


class TestSubModuleImports:
    """Test that all imports work correctly from the sub-module."""

    def test_import_base_renderer(self):
        """Test importing base renderer class."""
        assert MolecularRenderer is not None
        assert hasattr(MolecularRenderer, "render")
        assert hasattr(MolecularRenderer, "_prepare_molecule")

    def test_import_all_renderer_classes(self):
        """Test importing all renderer implementation classes."""
        renderer_classes = [Molecule2DRenderer, MoleculeGridRenderer]

        for renderer_class in renderer_classes:
            assert renderer_class is not None
            assert issubclass(renderer_class, MolecularRenderer)

    def test_import_utility_classes(self):
        """Test importing utility classes."""
        assert ColorUtils is not None
        assert DrawerConfigurationManager is not None
        assert hasattr(ColorUtils, "parse_color_to_rgba")
        assert hasattr(DrawerConfigurationManager, "create_drawer")

    def test_import_factory_function(self):
        """Test importing factory function."""
        assert get_renderer is not None
        assert callable(get_renderer)


class TestSubModuleBackwardCompatibility:
    """Test that the sub-module maintains backward compatibility."""

    def test_factory_function_compatibility(self):
        """Test that factory function maintains same API."""
        # Should work exactly as before
        renderer = get_renderer("2d")
        assert isinstance(renderer, Molecule2DRenderer)

        # With config
        config = RenderConfig(width=600, height=400)
        renderer_with_config = get_renderer("2d", config)
        assert isinstance(renderer_with_config, Molecule2DRenderer)
        assert renderer_with_config.config is config

    def test_core_module_integration(self):
        """Test that core module can still import and use renderers."""
        # Import as core module would
        from molecular_string_renderer.renderers import get_renderer

        # Use as core module would
        renderer = get_renderer("2d")
        mol = Chem.MolFromSmiles("CCO")
        rdDepictor.Compute2DCoords(mol)
        result = renderer.render(mol)
        assert result is not None


class TestSubModuleFunctionality:
    """Test that all functionality works correctly after refactoring."""

    def test_all_renderers_work_independently(self):
        """Test that each renderer works correctly in isolation."""
        mol = Chem.MolFromSmiles("CCO")
        rdDepictor.Compute2DCoords(mol)

        test_cases = [
            (Molecule2DRenderer(), mol),
            (MoleculeGridRenderer(), mol),
        ]

        for renderer, test_mol in test_cases:
            result = renderer.render(test_mol)
            assert result is not None, f"Renderer {type(renderer).__name__} failed"
            assert hasattr(result, "size")
            assert hasattr(result, "mode")

    def test_factory_creates_correct_renderers(self):
        """Test that factory creates the correct renderer types."""
        test_cases = [
            ("2d", Molecule2DRenderer),
            ("2D", Molecule2DRenderer),
            ("grid", MoleculeGridRenderer),
            ("GRID", MoleculeGridRenderer),
        ]

        for renderer_type, expected_class in test_cases:
            renderer = get_renderer(renderer_type)
            assert isinstance(renderer, expected_class)
            assert isinstance(renderer, MolecularRenderer)

    def test_configuration_propagation(self):
        """Test that configuration is properly propagated through sub-module."""
        custom_config = RenderConfig(width=600, height=400, show_carbon=True)

        # Test with factory
        renderer = get_renderer("2d", custom_config)
        assert renderer.config is custom_config
        assert renderer.config.width == 600
        assert renderer.config.show_carbon is True

        # Test direct instantiation
        direct_renderer = Molecule2DRenderer(custom_config)
        assert direct_renderer.config is custom_config

    def test_error_handling_consistency(self):
        """Test that error handling is consistent across all renderers."""
        renderers = [
            Molecule2DRenderer(),
            MoleculeGridRenderer(),
        ]

        for renderer in renderers:
            with pytest.raises(ValueError):
                renderer.render(None)

    def test_utilities_work_correctly(self):
        """Test that utility classes work correctly."""
        # Test ColorUtils
        rgba = ColorUtils.parse_color_to_rgba("red")
        assert rgba == (1.0, 0.0, 0.0, 1.0)

        assert ColorUtils.is_white_background("white") is True
        assert ColorUtils.is_white_background("red") is False

        # Test DrawerConfigurationManager
        config = RenderConfig()
        manager = DrawerConfigurationManager(config)
        assert manager.config is config


class TestSubModulePerformance:
    """Test that the sub-module refactoring doesn't impact performance."""

    def test_import_speed(self):
        """Test that imports are reasonably fast."""
        import time

        start_time = time.time()

        # Re-import to test fresh import time
        _ = __import__("molecular_string_renderer.renderers", fromlist=[""])

        import_time = time.time() - start_time

        # Should import in reasonable time (less than 2 seconds for complex module)
        assert import_time < 2.0, f"Import took too long: {import_time:.3f} seconds"

    def test_renderer_creation_speed(self):
        """Test that renderer creation is reasonably fast."""
        import time

        start_time = time.time()

        # Create multiple renderers
        for _ in range(50):
            get_renderer("2d")
            get_renderer("grid")

        creation_time = time.time() - start_time

        # Should create renderers quickly
        assert creation_time < 2.0, (
            f"Renderer creation took too long: {creation_time:.3f} seconds"
        )


class TestSubModuleStructure:
    """Test the internal structure of the sub-module."""

    def test_module_has_correct_exports(self):
        """Test that the sub-module exports the correct symbols."""
        import molecular_string_renderer.renderers as renderers_module

        expected_exports = [
            "MolecularRenderer",
            "Molecule2DRenderer",
            "MoleculeGridRenderer",
            "ColorUtils",
            "DrawerConfigurationManager",
            "get_renderer",
        ]

        for export in expected_exports:
            assert hasattr(renderers_module, export), f"Missing export: {export}"

        # Check __all__ if it exists
        if hasattr(renderers_module, "__all__"):
            for export in expected_exports:
                assert export in renderers_module.__all__, (
                    f"Export {export} not in __all__"
                )

    def test_individual_modules_importable(self):
        """Test that individual sub-modules can be imported directly."""
        # These should all work without errors
        # Verify these are the same objects as the main imports
        from molecular_string_renderer.renderers import (
            MolecularRenderer as MainMolecularRenderer,
        )
        from molecular_string_renderer.renderers import (
            Molecule2DRenderer as Main2DRenderer,
        )
        from molecular_string_renderer.renderers import (
            get_renderer as main_get_renderer,
        )
        from molecular_string_renderer.renderers.base import (
            MolecularRenderer as BaseMolecularRenderer,
        )
        from molecular_string_renderer.renderers.factory import (
            get_renderer as base_get_renderer,
        )
        from molecular_string_renderer.renderers.two_dimensional import (
            Molecule2DRenderer as Base2DRenderer,
        )

        assert BaseMolecularRenderer is MainMolecularRenderer
        assert Base2DRenderer is Main2DRenderer
        assert base_get_renderer is main_get_renderer


class TestDocumentationAndMetadata:
    """Test that documentation and metadata are preserved."""

    def test_renderer_classes_have_docstrings(self):
        """Test that all renderer classes have proper docstrings."""
        renderer_classes = [Molecule2DRenderer, MoleculeGridRenderer]

        for renderer_class in renderer_classes:
            assert renderer_class.__doc__ is not None, (
                f"{renderer_class.__name__} missing docstring"
            )
            assert len(renderer_class.__doc__.strip()) > 0, (
                f"{renderer_class.__name__} has empty docstring"
            )

    def test_methods_have_docstrings(self):
        """Test that key methods have proper docstrings."""
        renderer = Molecule2DRenderer()

        methods_to_check = ["render", "_prepare_molecule"]

        for method_name in methods_to_check:
            method = getattr(renderer, method_name)
            assert method.__doc__ is not None, f"Method {method_name} missing docstring"
            assert len(method.__doc__.strip()) > 0, (
                f"Method {method_name} has empty docstring"
            )

    def test_factory_function_has_docstring(self):
        """Test that factory function has proper docstring."""
        assert get_renderer.__doc__ is not None
        assert len(get_renderer.__doc__.strip()) > 0
        assert (
            "factory" in get_renderer.__doc__.lower()
            or "Factory" in get_renderer.__doc__
        )

    def test_utility_classes_have_docstrings(self):
        """Test that utility classes have proper docstrings."""
        utility_classes = [ColorUtils, DrawerConfigurationManager]

        for utility_class in utility_classes:
            assert utility_class.__doc__ is not None, (
                f"{utility_class.__name__} missing docstring"
            )
            assert len(utility_class.__doc__.strip()) > 0, (
                f"{utility_class.__name__} has empty docstring"
            )


class TestCrossRendererConsistency:
    """Test consistency between different renderer types."""

    def test_same_molecule_different_renderers(self):
        """Test that the same molecule can be rendered by different renderers."""
        mol = Chem.MolFromSmiles("CCO")
        rdDepictor.Compute2DCoords(mol)

        renderers = [
            get_renderer("2d"),
            get_renderer("grid"),
        ]

        results = []
        for renderer in renderers:
            result = renderer.render(mol)
            results.append(result)
            assert result is not None
            assert hasattr(result, "size")
            assert hasattr(result, "mode")

        # All should produce valid results
        assert len(results) == len(renderers)

    def test_config_consistency_across_renderers(self):
        """Test that config handling is consistent across renderers."""
        config = RenderConfig(
            width=600,
            height=400,
            background_color="lightblue",
            show_carbon=True,
        )

        renderer_types = ["2d", "grid"]

        for renderer_type in renderer_types:
            renderer = get_renderer(renderer_type, config)

            assert renderer.config.width == 600
            assert renderer.config.height == 400
            assert renderer.config.background_color == "lightblue"
            assert renderer.config.show_carbon is True

    def test_highlight_support_across_renderers(self):
        """Test that highlight support is consistent across renderers."""
        mol = Chem.MolFromSmiles("CCO")
        rdDepictor.Compute2DCoords(mol)

        config = RenderConfig(
            highlight_atoms=[0, 1],
            highlight_bonds=[0],
        )

        renderer_types = ["2d", "grid"]

        for renderer_type in renderer_types:
            renderer = get_renderer(renderer_type, config)

            # Should not crash with highlights
            result = renderer.render(mol)
            assert result is not None


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""

    def test_many_small_renders(self):
        """Test rendering many small molecules doesn't leak memory."""
        mol = Chem.MolFromSmiles("C")
        rdDepictor.Compute2DCoords(mol)

        renderer = get_renderer("2d")

        # Render many times
        results = []
        for _ in range(100):
            result = renderer.render(mol)
            assert result is not None
            results.append(result)

        # Should have rendered all successfully
        assert len(results) == 100

    def test_renderer_reuse_performance(self):
        """Test that renderer reuse is efficient."""
        mol = Chem.MolFromSmiles("CCO")
        rdDepictor.Compute2DCoords(mol)

        renderer = get_renderer("2d")

        import time

        start_time = time.time()

        # Reuse the same renderer multiple times
        for _ in range(20):
            renderer.render(mol)

        render_time = time.time() - start_time

        # Should be reasonably fast with reuse
        assert render_time < 10.0, (
            f"Reuse rendering took too long: {render_time:.3f} seconds"
        )

    def test_concurrent_renderer_usage(self):
        """Test that multiple renderers can be used concurrently."""
        import threading

        mol = Chem.MolFromSmiles("CCO")
        rdDepictor.Compute2DCoords(mol)

        results = []
        threads = []

        def render_worker(renderer_type):
            renderer = get_renderer(renderer_type)
            result = renderer.render(mol)
            results.append(result)

        # Create threads for different renderer types
        for renderer_type in ["2d", "grid"]:
            for _ in range(3):  # 3 threads per type
                thread = threading.Thread(target=render_worker, args=(renderer_type,))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        # Should have 6 successful renders (2 types * 3 threads each)
        assert len(results) == 6
        for result in results:
            assert result is not None
