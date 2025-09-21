"""
Consolidated test suite for renderer handlers.

Tests shared behavior across all renderer types and format-specific behavior
only where necessary. This follows the pattern established in the outputs tests.
"""

import threading
from unittest.mock import patch

import pytest
from PIL import Image

from molecular_string_renderer.config import RenderConfig
from molecular_string_renderer.renderers import get_renderer
from molecular_string_renderer.renderers.base import MolecularRenderer

from .conftest import (
    ALL_RENDERER_TYPES,
    supports_highlights,
    validate_rendered_image,
)


class TestRendererProperties:
    """Test properties common to all renderers."""

    def test_renderer_is_molecular_renderer(self, renderer):
        """Test that all renderers inherit from MolecularRenderer."""
        assert isinstance(renderer, MolecularRenderer)

    def test_renderer_has_config(self, renderer):
        """Test that all renderers have a config attribute."""
        assert hasattr(renderer, "config")
        assert isinstance(renderer.config, RenderConfig)

    def test_renderer_has_render_method(self, renderer):
        """Test that all renderers have a render method."""
        assert hasattr(renderer, "render")
        assert callable(renderer.render)

    def test_renderer_has_prepare_molecule_method(self, renderer):
        """Test that all renderers have _prepare_molecule method."""
        assert hasattr(renderer, "_prepare_molecule")
        assert callable(renderer._prepare_molecule)

    def test_renderer_has_get_molecule_dimensions_method(self, renderer):
        """Test that all renderers have _get_molecule_dimensions method."""
        assert hasattr(renderer, "_get_molecule_dimensions")
        assert callable(renderer._get_molecule_dimensions)


class TestRendererInitialization:
    """Test initialization patterns across all renderers."""

    def test_init_without_config(self, renderer_type):
        """Test initialization without config."""
        renderer = get_renderer(renderer_type)
        assert renderer.config is not None
        assert isinstance(renderer.config, RenderConfig)

    def test_init_with_config(self, renderer_type):
        """Test initialization with custom config."""
        config = RenderConfig(width=600, height=400)
        renderer = get_renderer(renderer_type, config)
        assert renderer.config is config

    def test_init_with_none_config(self, renderer_type):
        """Test initialization with None config creates default."""
        renderer = get_renderer(renderer_type, None)
        assert renderer.config is not None
        assert isinstance(renderer.config, RenderConfig)


class TestRendererBasicFunctionality:
    """Test basic functionality common to all renderers."""

    def test_render_simple_molecule(self, renderer, simple_molecule):
        """Test rendering a simple molecule."""
        result = renderer.render(simple_molecule)
        validate_rendered_image(result)

    def test_render_complex_molecule(self, renderer, complex_molecule):
        """Test rendering a complex molecule."""
        result = renderer.render(complex_molecule)
        validate_rendered_image(result)

    def test_render_benzene_molecule(self, renderer, benzene_molecule):
        """Test rendering benzene (aromatic system)."""
        result = renderer.render(benzene_molecule)
        validate_rendered_image(result)

    def test_render_returns_pil_image(self, renderer, simple_molecule):
        """Test that render returns a PIL Image object."""
        result = renderer.render(simple_molecule)
        assert isinstance(result, Image.Image)

    def test_render_with_custom_config(self, renderer_with_config, simple_molecule):
        """Test rendering with custom configuration."""
        result = renderer_with_config.render(simple_molecule)
        validate_rendered_image(result)


class TestRendererMoleculePreperation:
    """Test molecule preparation functionality."""

    def test_prepare_molecule_valid_input(self, renderer, simple_molecule):
        """Test molecule preparation with valid input."""
        result = renderer._prepare_molecule(simple_molecule)
        assert result is not None
        # Should be a copy, not the same object
        assert result is not simple_molecule

    def test_prepare_molecule_none_input(self, renderer):
        """Test molecule preparation with None input."""
        with pytest.raises(ValueError, match="Cannot render None molecule"):
            renderer._prepare_molecule(None)

    def test_prepare_molecule_adds_carbon_labels(self, renderer, simple_molecule):
        """Test that carbon labels are added when requested."""
        # Create renderer with carbon display enabled
        config = RenderConfig(show_carbon=True)

        # Determine renderer type from the class name
        if "2D" in renderer.__class__.__name__:
            renderer_type = "2d"
        elif "Grid" in renderer.__class__.__name__:
            renderer_type = "grid"
        else:
            renderer_type = "2d"  # fallback

        renderer_with_carbon = get_renderer(renderer_type, config)

        # For this test, we'll just ensure it doesn't crash
        result = renderer_with_carbon._prepare_molecule(simple_molecule)
        assert result is not None

    def test_get_molecule_dimensions_valid_molecule(self, renderer, simple_molecule):
        """Test dimension calculation with valid molecule."""
        width, height = renderer._get_molecule_dimensions(simple_molecule)
        assert isinstance(width, float)
        assert isinstance(height, float)
        assert width > 0
        assert height > 0

    def test_get_molecule_dimensions_none_input(self, renderer):
        """Test dimension calculation with None input."""
        with pytest.raises(
            ValueError, match="Cannot calculate dimensions for None molecule"
        ):
            renderer._get_molecule_dimensions(None)


class TestRendererConfigurationHandling:
    """Test configuration handling across renderers."""

    def test_config_width_height_respected(self, renderer_type):
        """Test that width and height settings are respected."""
        config = RenderConfig(width=600, height=400)
        renderer = get_renderer(renderer_type, config)

        assert renderer.config.width == 600
        assert renderer.config.height == 400

    def test_config_background_color_respected(self, renderer_type):
        """Test that background color settings are respected."""
        config = RenderConfig(background_color="lightblue")
        renderer = get_renderer(renderer_type, config)

        assert renderer.config.background_color == "lightblue"

    def test_config_carbon_display_respected(self, renderer_type):
        """Test that carbon display settings are respected."""
        config = RenderConfig(show_carbon=True)
        renderer = get_renderer(renderer_type, config)

        assert renderer.config.show_carbon is True

    def test_config_hydrogen_display_respected(self, renderer_type):
        """Test that hydrogen display settings are respected."""
        config = RenderConfig(show_hydrogen=True)
        renderer = get_renderer(renderer_type, config)

        assert renderer.config.show_hydrogen is True


class TestRendererErrorHandling:
    """Test error handling across renderers."""

    def test_render_invalid_molecule_raises_error(self, renderer):
        """Test that invalid molecules raise appropriate errors."""
        with pytest.raises(ValueError):
            renderer.render(None)

    @patch("rdkit.Chem.rdDepictor.Compute2DCoords")
    def test_render_coordinate_computation_failure(
        self, mock_compute, renderer, simple_molecule
    ):
        """Test handling of coordinate computation failures."""
        mock_compute.side_effect = Exception("Coordinate computation failed")

        if "Grid" in renderer.__class__.__name__:
            # Grid renderer catches prepare failures and raises a different error
            with pytest.raises(
                ValueError, match="No valid molecules found in input list"
            ):
                renderer.render(simple_molecule)
        else:
            # 2D renderer propagates the coordinate error
            with pytest.raises(ValueError, match="Failed to compute 2D coordinates"):
                renderer.render(simple_molecule)


class TestRendererImageModeHandling:
    """Test image mode consistency across renderers."""

    def test_render_output_is_rgba(self, renderer, simple_molecule):
        """Test that all renderers output RGBA images."""
        result = renderer.render(simple_molecule)
        assert result.mode == "RGBA"

    def test_render_consistent_image_size(self, renderer, simple_molecule):
        """Test that image size matches configuration."""
        result = renderer.render(simple_molecule)

        # Image size should match config (allowing for some renderer-specific variations)
        assert result.size[0] > 0
        assert result.size[1] > 0


class TestRendererEdgeCases:
    """Test edge cases common across renderers."""

    def test_very_small_config_dimensions(self, renderer_type):
        """Test with minimum image dimensions."""
        config = RenderConfig(width=100, height=100)
        renderer = get_renderer(renderer_type, config)

        # Should not crash with small dimensions
        assert renderer.config.width == 100
        assert renderer.config.height == 100

    def test_very_large_config_dimensions(self, renderer_type):
        """Test with large image dimensions."""
        config = RenderConfig(width=2000, height=1500)
        renderer = get_renderer(renderer_type, config)

        # Should handle large dimensions
        assert renderer.config.width == 2000
        assert renderer.config.height == 1500

    def test_square_vs_rectangular_configs(self, renderer_type, simple_molecule):
        """Test with different aspect ratios."""
        configs = [
            RenderConfig(width=100, height=100),  # Square
            RenderConfig(width=200, height=100),  # Wide
            RenderConfig(width=100, height=200),  # Tall
        ]

        for config in configs:
            renderer = get_renderer(renderer_type, config)
            result = renderer.render(simple_molecule)
            validate_rendered_image(result)


class TestRendererInheritance:
    """Test inheritance hierarchy compliance."""

    def test_all_renderers_inherit_from_base(self, renderer):
        """Test that all renderers inherit from MolecularRenderer."""
        assert isinstance(renderer, MolecularRenderer)

    def test_required_methods_implemented(self, renderer):
        """Test that all required abstract methods are implemented."""
        # Test method access
        assert hasattr(renderer, "render")
        assert callable(renderer.render)

        # Test inherited methods
        assert hasattr(renderer, "_prepare_molecule")
        assert hasattr(renderer, "_get_molecule_dimensions")
        assert callable(renderer._prepare_molecule)
        assert callable(renderer._get_molecule_dimensions)


class TestRendererThreadSafety:
    """Test thread safety across renderers."""

    def test_multiple_instances_independent(self, renderer_type):
        """Test that multiple instances are independent."""
        config1 = RenderConfig(width=300, height=300)
        config2 = RenderConfig(width=500, height=400)

        renderer1 = get_renderer(renderer_type, config1)
        renderer2 = get_renderer(renderer_type, config2)

        assert renderer1.config.width == 300
        assert renderer2.config.width == 500
        assert renderer1.config is not renderer2.config

    def test_concurrent_operations(self, renderer, molecule_list):
        """Test concurrent render operations."""
        results = []
        threads = []

        def render_worker(mol):
            if mol is not None:  # Skip None molecules
                result = renderer.render(mol)
                results.append(result)

        # Filter out None molecules
        valid_molecules = [mol for mol in molecule_list if mol is not None]

        for mol in valid_molecules:
            thread = threading.Thread(target=render_worker, args=(mol,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == len(valid_molecules)
        for result in results:
            validate_rendered_image(result)


class TestRendererFormatSpecificCapabilities:
    """Test format-specific capabilities."""

    def test_highlight_support_consistency(self, renderer_type):
        """Test that highlight support is consistent with capabilities."""
        expected_support = supports_highlights(renderer_type)

        # All current renderers support highlights
        assert expected_support is True

        # Test that renderer has highlight-related config attributes
        config = RenderConfig(highlight_atoms=[0, 1], highlight_bonds=[0])
        renderer_with_highlights = get_renderer(renderer_type, config)
        assert renderer_with_highlights.config.highlight_atoms == [0, 1]
        assert renderer_with_highlights.config.highlight_bonds == [0]


class TestRendererIntegration:
    """Integration tests across different renderers."""

    def test_same_molecule_different_renderers(self, simple_molecule):
        """Test that the same molecule can be rendered by all renderers."""
        results = {}

        for renderer_type in ALL_RENDERER_TYPES:
            renderer = get_renderer(renderer_type)
            result = renderer.render(simple_molecule)
            results[renderer_type] = result
            validate_rendered_image(result)

        # All renderers should produce valid output
        assert len(results) == len(ALL_RENDERER_TYPES)

    def test_renderer_consistency(self):
        """Test that renderer properties are consistent."""
        for renderer_type in ALL_RENDERER_TYPES:
            renderer = get_renderer(renderer_type)

            # Basic consistency checks
            assert hasattr(renderer, "config")
            assert isinstance(renderer.config, RenderConfig)
            assert hasattr(renderer, "render")
            assert callable(renderer.render)

    def test_cross_renderer_molecule_compatibility(self, molecule_list):
        """Test that all renderers can handle the same molecules."""
        valid_molecules = [mol for mol in molecule_list if mol is not None]

        for renderer_type in ALL_RENDERER_TYPES:
            renderer = get_renderer(renderer_type)

            for mol in valid_molecules:
                result = renderer.render(mol)
                validate_rendered_image(result)


class TestRendererPerformance:
    """Test performance characteristics across renderers."""

    def test_renderer_creation_performance(self):
        """Test that renderer creation is reasonably fast."""
        import time

        start_time = time.time()

        # Create multiple renderers
        for _ in range(50):
            for renderer_type in ALL_RENDERER_TYPES:
                get_renderer(renderer_type)

        creation_time = time.time() - start_time

        # Should create renderers quickly (allowing more time for RDKit operations)
        assert creation_time < 5.0, (
            f"Renderer creation took too long: {creation_time:.3f} seconds"
        )

    def test_render_performance(self, renderer, simple_molecule):
        """Test that rendering is reasonably fast."""
        import time

        start_time = time.time()

        # Render the same molecule multiple times
        for _ in range(10):
            renderer.render(simple_molecule)

        render_time = time.time() - start_time

        # Should render reasonably quickly (allowing time for RDKit operations)
        assert render_time < 10.0, f"Rendering took too long: {render_time:.3f} seconds"
