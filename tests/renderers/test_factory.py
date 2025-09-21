"""
Tests for the renderer factory functionality.
"""

import pytest

from molecular_string_renderer.config import RenderConfig
from molecular_string_renderer.renderers.base import MolecularRenderer
from molecular_string_renderer.renderers.factory import get_renderer
from molecular_string_renderer.renderers.grid import MoleculeGridRenderer
from molecular_string_renderer.renderers.two_dimensional import Molecule2DRenderer


class TestGetRenderer:
    """Test the get_renderer factory function."""

    def test_get_2d_renderer(self):
        """Test getting 2D renderer."""
        renderer = get_renderer("2d")

        assert isinstance(renderer, Molecule2DRenderer)
        assert isinstance(renderer, MolecularRenderer)

    def test_get_grid_renderer(self):
        """Test getting grid renderer."""
        renderer = get_renderer("grid")

        assert isinstance(renderer, MoleculeGridRenderer)
        assert isinstance(renderer, MolecularRenderer)

    def test_case_insensitive_renderer_type(self):
        """Test that renderer type is case insensitive."""
        test_cases = [
            ("2D", Molecule2DRenderer),
            ("2d", Molecule2DRenderer),
            ("Grid", MoleculeGridRenderer),
            ("GRID", MoleculeGridRenderer),
            ("grid", MoleculeGridRenderer),
        ]

        for renderer_type, expected_class in test_cases:
            renderer = get_renderer(renderer_type)
            assert isinstance(renderer, expected_class), (
                f"Failed for renderer type: {renderer_type}"
            )

    def test_whitespace_handling(self):
        """Test that renderer type handles whitespace correctly."""
        test_cases = [
            "  2d  ",
            "\t2d\t",
            "\n2d\n",
            " 2d ",
            "  grid  ",
            " grid ",
        ]

        for renderer_type in test_cases:
            renderer = get_renderer(renderer_type)
            # Should succeed without error
            assert isinstance(renderer, MolecularRenderer)

    def test_unsupported_renderer_raises_error(self):
        """Test that unsupported renderer types raise ValueError."""
        unsupported_renderers = [
            "3d",
            "raster",
            "unknown",
            "invalid",
            "",
            "svg",
            "pdf",
        ]

        for renderer_type in unsupported_renderers:
            with pytest.raises(ValueError, match="Unsupported renderer"):
                get_renderer(renderer_type)

    def test_error_message_includes_supported_renderers(self):
        """Test that error message includes list of supported renderers."""
        with pytest.raises(ValueError) as exc_info:
            get_renderer("unsupported")

        error_message = str(exc_info.value)
        assert "Supported:" in error_message
        assert "2d" in error_message
        assert "grid" in error_message

    def test_none_renderer_type_raises_error(self):
        """Test that None renderer type raises appropriate error."""
        with pytest.raises(AttributeError):
            get_renderer(None)

    def test_empty_string_renderer_type_raises_error(self):
        """Test that empty string renderer type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported renderer"):
            get_renderer("")


class TestGetRendererWithConfig:
    """Test get_renderer with custom configuration."""

    def test_get_renderer_with_default_config(self):
        """Test getting renderer without providing config."""
        renderer = get_renderer("2d")

        assert renderer.config is not None
        assert isinstance(renderer.config, RenderConfig)
        # Should use default values
        assert renderer.config.width == 500
        assert renderer.config.height == 500

    def test_get_renderer_with_custom_config(self):
        """Test getting renderer with custom config."""
        custom_config = RenderConfig(width=600, height=400, show_carbon=True)
        renderer = get_renderer("2d", custom_config)

        assert renderer.config is custom_config
        assert renderer.config.width == 600
        assert renderer.config.height == 400
        assert renderer.config.show_carbon is True

    def test_get_renderer_with_none_config(self):
        """Test getting renderer with None config (should use default)."""
        renderer = get_renderer("2d", None)

        assert renderer.config is not None
        assert isinstance(renderer.config, RenderConfig)

    def test_config_passed_to_all_renderer_types(self):
        """Test that custom config is passed to all renderer types."""
        custom_config = RenderConfig(width=800, height=600)

        renderer_types = ["2d", "grid"]

        for renderer_type in renderer_types:
            renderer = get_renderer(renderer_type, custom_config)
            assert renderer.config is custom_config, (
                f"Config not passed for {renderer_type}"
            )
            assert renderer.config.width == 800, (
                f"Config values not preserved for {renderer_type}"
            )
            assert renderer.config.height == 600, (
                f"Config values not preserved for {renderer_type}"
            )

    def test_invalid_config_type_raises_error(self):
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="config must be a RenderConfig instance"):
            get_renderer("2d", "invalid_config")

        with pytest.raises(TypeError, match="config must be a RenderConfig instance"):
            get_renderer("2d", {"width": 500})

        with pytest.raises(TypeError, match="config must be a RenderConfig instance"):
            get_renderer("2d", 123)


class TestRendererMapping:
    """Test the internal renderer mapping."""

    def test_all_expected_renderers_mapped(self):
        """Test that all expected renderer types are properly mapped."""
        expected_mappings = {
            "2d": Molecule2DRenderer,
            "grid": MoleculeGridRenderer,
        }

        for renderer_type, expected_class in expected_mappings.items():
            renderer = get_renderer(renderer_type)
            assert isinstance(renderer, expected_class), (
                f"Mapping failed for {renderer_type}"
            )

    def test_no_duplicate_instances(self):
        """Test that each call to get_renderer returns a new instance."""
        renderer1 = get_renderer("2d")
        renderer2 = get_renderer("2d")

        assert renderer1 is not renderer2, "get_renderer should return new instances"
        assert type(renderer1) is type(renderer2), "Renderers should be same type"

    def test_different_configs_create_different_instances(self):
        """Test that different configs create properly configured instances."""
        config1 = RenderConfig(width=300, height=300)
        config2 = RenderConfig(width=500, height=400)

        renderer1 = get_renderer("2d", config1)
        renderer2 = get_renderer("2d", config2)

        assert renderer1.config.width == 300
        assert renderer2.config.width == 500
        assert renderer1 is not renderer2

    def test_renderer_creation_failure_handling(self):
        """Test handling of renderer creation failures."""
        from unittest.mock import patch

        # Mock a renderer class to raise an exception during instantiation
        with patch(
            "molecular_string_renderer.renderers.factory.Molecule2DRenderer"
        ) as mock_renderer:
            mock_renderer.side_effect = Exception("Renderer creation failed")

            with pytest.raises(RuntimeError, match="Failed to create 2d renderer"):
                get_renderer("2d")


class TestRendererFactoryPerformance:
    """Test performance characteristics of the factory."""

    def test_factory_performance(self):
        """Test that factory function is reasonably fast."""
        import time

        start_time = time.time()

        # Create many renderers
        for _ in range(100):
            get_renderer("2d")
            get_renderer("grid")

        creation_time = time.time() - start_time

        # Should create renderers quickly
        assert creation_time < 2.0, (
            f"Factory function took too long: {creation_time:.3f} seconds"
        )

    def test_factory_memory_efficiency(self):
        """Test that factory doesn't leak memory."""
        # Create many renderers with different configs
        renderers = []

        for i in range(50):
            config = RenderConfig(width=300 + i, height=300 + i)
            renderer = get_renderer("2d", config)
            renderers.append(renderer)

        # Each should have unique config
        for i, renderer in enumerate(renderers):
            assert renderer.config.width == 300 + i
            assert renderer.config.height == 300 + i

        # Clear references
        renderers.clear()


class TestRendererFactoryIntegration:
    """Test factory integration with other components."""

    def test_factory_creates_renderers_with_working_render_method(self):
        """Test that factory-created renderers can actually render."""
        from rdkit import Chem
        from rdkit.Chem import rdDepictor

        # Create a simple test molecule
        mol = Chem.MolFromSmiles("CCO")
        rdDepictor.Compute2DCoords(mol)

        for renderer_type in ["2d", "grid"]:
            renderer = get_renderer(renderer_type)

            # Should be able to render without error
            result = renderer.render(mol)
            assert result is not None
            assert hasattr(result, "size")
            assert hasattr(result, "mode")

    def test_factory_config_inheritance(self):
        """Test that config settings are properly inherited."""
        config = RenderConfig(
            width=600,
            height=400,
            background_color="lightblue",
            show_carbon=True,
            show_hydrogen=True,
        )

        for renderer_type in ["2d", "grid"]:
            renderer = get_renderer(renderer_type, config)

            # All config properties should be preserved
            assert renderer.config.width == 600
            assert renderer.config.height == 400
            assert renderer.config.background_color == "lightblue"
            assert renderer.config.show_carbon is True
            assert renderer.config.show_hydrogen is True

    def test_factory_with_highlight_config(self):
        """Test factory with highlight configuration."""
        config = RenderConfig(
            highlight_atoms=[0, 1, 2],
            highlight_bonds=[0, 1],
        )

        for renderer_type in ["2d", "grid"]:
            renderer = get_renderer(renderer_type, config)

            assert renderer.config.highlight_atoms == [0, 1, 2]
            assert renderer.config.highlight_bonds == [0, 1]


class TestRendererFactoryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_factory_with_extreme_config_values(self):
        """Test factory with extreme configuration values."""
        # Minimum dimensions
        small_config = RenderConfig(width=100, height=100)
        small_renderer = get_renderer("2d", small_config)
        assert small_renderer.config.width == 100
        assert small_renderer.config.height == 100

        # Maximum dimensions
        large_config = RenderConfig(width=2000, height=2000)
        large_renderer = get_renderer("2d", large_config)
        assert large_renderer.config.width == 2000
        assert large_renderer.config.height == 2000

    def test_factory_with_unusual_color_values(self):
        """Test factory with unusual color configuration."""
        unusual_colors = [
            "#FF00FF",  # Magenta
            "rgb(255,128,0)",  # Orange
            "transparent",  # Transparent
            "lightsteelblue",  # Named color
        ]

        for color in unusual_colors:
            config = RenderConfig(background_color=color)
            renderer = get_renderer("2d", config)
            assert renderer.config.background_color == color

    def test_factory_thread_safety(self):
        """Test that factory is thread-safe."""
        import threading

        results = []

        def create_renderer_worker():
            config = RenderConfig(width=300, height=300)
            renderer = get_renderer("2d", config)
            results.append(renderer)

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_renderer_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have created 10 independent renderers
        assert len(results) == 10

        # All should be different instances
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                assert results[i] is not results[j]
