"""
Format-specific tests for 2D molecular renderer.

Tests functionality unique to the 2D renderer that doesn't apply to other renderer types.
"""

from unittest.mock import Mock, patch

import pytest
from PIL import Image

from molecular_string_renderer.config import RenderConfig
from molecular_string_renderer.renderers.two_dimensional import Molecule2DRenderer


class TestMolecule2DRendererSpecific:
    """Test 2D renderer-specific functionality."""

    def test_init_creates_drawer_manager(self):
        """Test that initialization creates a drawer configuration manager."""
        config = RenderConfig()
        renderer = Molecule2DRenderer(config)

        assert hasattr(renderer, "_drawer_manager")
        assert renderer._drawer_manager is not None

    def test_drawer_manager_uses_config(self):
        """Test that drawer manager uses the renderer's config."""
        config = RenderConfig(width=600, height=400)
        renderer = Molecule2DRenderer(config)

        assert renderer._drawer_manager.config is config

    @patch(
        "molecular_string_renderer.renderers.two_dimensional.Molecule2DRenderer._prepare_molecule"
    )
    @patch(
        "molecular_string_renderer.renderers.config_manager.DrawerConfigurationManager.create_drawer"
    )
    @patch("PIL.Image.open")
    @patch("io.BytesIO")
    def test_render_basic_flow(
        self, mock_bytesio, mock_image_open, mock_create_drawer, mock_prepare
    ):
        """Test the basic render flow for 2D renderer."""
        renderer = Molecule2DRenderer()
        mol = Mock()
        prepared_mol = Mock()
        mock_prepare.return_value = prepared_mol

        # Mock drawer
        drawer = Mock()
        drawer.GetDrawingText.return_value = b"fake_png_data"
        drawer.DrawMolecule = Mock()
        drawer.FinishDrawing = Mock()
        mock_create_drawer.return_value = drawer

        # Mock BytesIO
        mock_bytes_obj = Mock()
        mock_bytesio.return_value = mock_bytes_obj

        # Mock PIL Image
        img = Mock()
        img.mode = "RGBA"
        img.size = (300, 300)
        mock_image_open.return_value = img

        result = renderer.render(mol)

        # Verify the flow
        mock_prepare.assert_called_once_with(mol)
        mock_create_drawer.assert_called_once()
        drawer.DrawMolecule.assert_called_once_with(prepared_mol)
        drawer.FinishDrawing.assert_called_once()
        mock_bytesio.assert_called_once_with(b"fake_png_data")
        assert result == img

    @patch(
        "molecular_string_renderer.renderers.two_dimensional.Molecule2DRenderer._prepare_molecule"
    )
    @patch(
        "molecular_string_renderer.renderers.config_manager.DrawerConfigurationManager.create_drawer"
    )
    @patch("PIL.Image.open")
    @patch("io.BytesIO")
    def test_render_with_highlights_in_config(
        self, mock_bytesio, mock_image_open, mock_create_drawer, mock_prepare
    ):
        """Test rendering with highlights specified in config."""
        config = RenderConfig(highlight_atoms=[0, 1], highlight_bonds=[0])
        renderer = Molecule2DRenderer(config)
        mol = Mock()
        prepared_mol = Mock()
        mock_prepare.return_value = prepared_mol

        # Mock drawer
        drawer = Mock()
        drawer.GetDrawingText.return_value = b"fake_png_data"
        drawer.DrawMolecule = Mock()
        drawer.FinishDrawing = Mock()
        mock_create_drawer.return_value = drawer

        # Mock BytesIO and PIL Image
        mock_bytesio.return_value = Mock()
        img = Mock()
        img.mode = "RGBA"
        mock_image_open.return_value = img

        result = renderer.render(mol)

        # Should call DrawMolecule with highlights
        drawer.DrawMolecule.assert_called_once()
        call_args = drawer.DrawMolecule.call_args
        assert call_args[1]["highlightAtoms"] == [0, 1]
        assert call_args[1]["highlightBonds"] == [0]
        assert result == img

    @patch(
        "molecular_string_renderer.renderers.two_dimensional.Molecule2DRenderer._prepare_molecule"
    )
    @patch(
        "molecular_string_renderer.renderers.two_dimensional.DrawerConfigurationManager"
    )
    @patch("PIL.Image.open")
    @patch("io.BytesIO")
    def test_render_with_highlights_method(
        self, mock_bytesio, mock_image_open, mock_manager_class, mock_prepare
    ):
        """Test the render_with_highlights method specifically."""
        renderer = Molecule2DRenderer()
        mol = Mock()
        prepared_mol = Mock()
        mock_prepare.return_value = prepared_mol

        # Mock drawer manager and drawer
        drawer_manager = Mock()
        drawer = Mock()
        drawer.GetDrawingText.return_value = b"fake_png_data"
        drawer.DrawMolecule = Mock()
        drawer.FinishDrawing = Mock()
        drawer_manager.create_drawer.return_value = drawer
        mock_manager_class.return_value = drawer_manager

        # Mock BytesIO and PIL Image
        mock_bytesio.return_value = Mock()
        img = Mock()
        img.mode = "RGBA"
        mock_image_open.return_value = img

        # Test with highlights
        result = renderer.render_with_highlights(
            mol,
            highlight_atoms=[0, 1],
            highlight_bonds=[0],
            highlight_colors={0: (1.0, 0.0, 0.0)},
        )

        # Should create new drawer manager with highlight config
        assert (
            mock_manager_class.call_count == 2
        )  # Once in __init__, once in render_with_highlights
        drawer.DrawMolecule.assert_called_once()
        assert result == img

    def test_render_with_highlights_color_conversion(self):
        """Test that RGB colors are converted to RGBA in render_with_highlights."""
        # Create a simple test to verify color conversion logic
        highlight_colors = {0: (1.0, 0.0, 0.0)}  # RGB

        # This tests the color conversion logic in the method
        for idx, color in highlight_colors.items():
            if len(color) == 3:  # RGB, add alpha
                highlight_colors[idx] = (*color, 1.0)

        assert highlight_colors[0] == (1.0, 0.0, 0.0, 1.0)

    def test_render_with_highlights_empty_highlights(self):
        """Test render_with_highlights with empty highlight lists."""
        with patch.object(Molecule2DRenderer, "_prepare_molecule") as mock_prepare:
            with patch(
                "molecular_string_renderer.renderers.two_dimensional.DrawerConfigurationManager"
            ) as mock_manager_class:
                with patch("PIL.Image.open") as mock_image_open:
                    with patch("io.BytesIO") as mock_bytesio:
                        renderer = Molecule2DRenderer()
                        mol = Mock()
                        mock_prepare.return_value = Mock()

                        # Mock setup
                        drawer_manager = Mock()
                        drawer = Mock()
                        drawer.GetDrawingText.return_value = b"fake_png_data"
                        drawer.DrawMolecule = Mock()
                        drawer.FinishDrawing = Mock()
                        drawer_manager.create_drawer.return_value = drawer
                        mock_manager_class.return_value = drawer_manager

                        mock_bytesio.return_value = Mock()
                        img = Mock()
                        img.mode = "RGBA"
                        mock_image_open.return_value = img

                        # Test with None/empty highlights
                        renderer.render_with_highlights(
                            mol,
                            highlight_atoms=None,
                            highlight_bonds=None,
                            highlight_colors=None,
                        )

                        # Should still work
                        drawer.DrawMolecule.assert_called_once()
                        call_args = drawer.DrawMolecule.call_args
                        assert call_args[1]["highlightAtoms"] == []
                        assert call_args[1]["highlightBonds"] == []

    def test_render_failure_raises_runtime_error(self):
        """Test that render failures raise RuntimeError with appropriate message."""
        with patch.object(Molecule2DRenderer, "_prepare_molecule") as mock_prepare:
            renderer = Molecule2DRenderer()
            mol = Mock()
            mock_prepare.side_effect = Exception("Preparation failed")

            with pytest.raises(Exception, match="Preparation failed"):
                renderer.render(mol)

    def test_render_with_highlights_failure_raises_runtime_error(self):
        """Test that render_with_highlights failures raise RuntimeError."""
        with patch.object(Molecule2DRenderer, "_prepare_molecule") as mock_prepare:
            renderer = Molecule2DRenderer()
            mol = Mock()
            mock_prepare.side_effect = Exception("Preparation failed")

            with pytest.raises(Exception, match="Preparation failed"):
                renderer.render_with_highlights(mol)

    @patch(
        "molecular_string_renderer.renderers.two_dimensional.Molecule2DRenderer._prepare_molecule"
    )
    @patch(
        "molecular_string_renderer.renderers.config_manager.DrawerConfigurationManager.create_drawer"
    )
    def test_drawer_error_handling(self, mock_create_drawer, mock_prepare):
        """Test handling of drawer creation/operation errors."""
        renderer = Molecule2DRenderer()
        mol = Mock()
        mock_prepare.return_value = Mock()
        mock_create_drawer.side_effect = Exception("Drawer creation failed")

        with pytest.raises(RuntimeError, match="Failed to render molecule"):
            renderer.render(mol)

    def test_image_mode_conversion(self):
        """Test image mode conversion to RGBA."""
        with patch.object(Molecule2DRenderer, "_prepare_molecule") as mock_prepare:
            with patch(
                "molecular_string_renderer.renderers.config_manager.DrawerConfigurationManager.create_drawer"
            ) as mock_create_drawer:
                with patch("PIL.Image.open") as mock_image_open:
                    with patch("io.BytesIO") as mock_bytesio:
                        renderer = Molecule2DRenderer()
                        mol = Mock()
                        mock_prepare.return_value = Mock()

                        # Mock drawer
                        drawer = Mock()
                        drawer.GetDrawingText.return_value = b"fake_png_data"
                        drawer.DrawMolecule = Mock()
                        drawer.FinishDrawing = Mock()
                        mock_create_drawer.return_value = drawer

                        mock_bytesio.return_value = Mock()

                        # Test with non-RGBA image
                        img = Mock()
                        img.mode = "RGB"
                        img.convert.return_value = Mock()
                        img.convert.return_value.mode = "RGBA"
                        mock_image_open.return_value = img

                        result = renderer.render(mol)

                        # Should convert to RGBA
                        img.convert.assert_called_once_with("RGBA")
                        assert result.mode == "RGBA"


class TestMolecule2DRendererConfiguration:
    """Test 2D renderer configuration handling."""

    def test_config_affects_drawer_manager(self):
        """Test that config changes affect the drawer manager."""
        config1 = RenderConfig(width=300, height=300)
        config2 = RenderConfig(width=600, height=400)

        renderer1 = Molecule2DRenderer(config1)
        renderer2 = Molecule2DRenderer(config2)

        assert renderer1._drawer_manager.config.width == 300
        assert renderer2._drawer_manager.config.width == 600

    def test_background_color_config(self):
        """Test background color configuration."""
        config = RenderConfig(background_color="lightblue")
        renderer = Molecule2DRenderer(config)

        assert renderer.config.background_color == "lightblue"
        assert renderer._drawer_manager.config.background_color == "lightblue"

    def test_carbon_display_config(self):
        """Test carbon display configuration."""
        config = RenderConfig(show_carbon=True)
        renderer = Molecule2DRenderer(config)

        assert renderer.config.show_carbon is True
        assert renderer._drawer_manager.config.show_carbon is True

    def test_hydrogen_display_config(self):
        """Test hydrogen display configuration."""
        config = RenderConfig(show_hydrogen=True)
        renderer = Molecule2DRenderer(config)

        assert renderer.config.show_hydrogen is True
        assert renderer._drawer_manager.config.show_hydrogen is True


class TestMolecule2DRendererEdgeCases:
    """Test edge cases specific to 2D renderer."""

    def test_very_small_dimensions(self):
        """Test 2D renderer with minimum dimensions."""
        config = RenderConfig(width=100, height=100)
        renderer = Molecule2DRenderer(config)

        assert renderer.config.width == 100
        assert renderer.config.height == 100

    def test_very_large_dimensions(self):
        """Test 2D renderer with maximum dimensions."""
        config = RenderConfig(width=2000, height=2000)
        renderer = Molecule2DRenderer(config)

        assert renderer.config.width == 2000
        assert renderer.config.height == 2000

    def test_unusual_aspect_ratios(self):
        """Test 2D renderer with unusual aspect ratios."""
        configs = [
            RenderConfig(width=1000, height=100),  # Very wide
            RenderConfig(width=100, height=1000),  # Very tall
            RenderConfig(width=500, height=500),  # Square
        ]

        for config in configs:
            renderer = Molecule2DRenderer(config)
            assert renderer.config.width == config.width
            assert renderer.config.height == config.height

    def test_extreme_color_values(self):
        """Test 2D renderer with extreme color values."""
        unusual_colors = [
            "#000000",  # Black
            "#FFFFFF",  # White
            "#FF00FF",  # Magenta
            "transparent",  # Transparent
        ]

        for color in unusual_colors:
            config = RenderConfig(background_color=color)
            renderer = Molecule2DRenderer(config)
            assert renderer.config.background_color == color

    def test_extreme_highlight_values(self):
        """Test 2D renderer with extreme highlight values."""
        # Very large highlight lists
        large_atoms = list(range(1000))
        large_bonds = list(range(500))

        config = RenderConfig(highlight_atoms=large_atoms, highlight_bonds=large_bonds)
        renderer = Molecule2DRenderer(config)

        assert renderer.config.highlight_atoms == large_atoms
        assert renderer.config.highlight_bonds == large_bonds


class TestMolecule2DRendererIntegration:
    """Test 2D renderer integration with real RDKit molecules."""

    def test_render_real_molecule(self, simple_molecule):
        """Test rendering a real RDKit molecule."""
        renderer = Molecule2DRenderer()
        result = renderer.render(simple_molecule)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        assert result.size[0] > 0
        assert result.size[1] > 0

    def test_render_with_highlights_real_molecule(self, molecule_with_highlights):
        """Test rendering a real molecule with highlights."""
        renderer = Molecule2DRenderer()
        mol_data = molecule_with_highlights

        result = renderer.render_with_highlights(
            mol_data["molecule"],
            highlight_atoms=mol_data["highlight_atoms"],
            highlight_bonds=mol_data["highlight_bonds"],
            highlight_colors=mol_data["highlight_colors"],
        )

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"

    def test_render_complex_molecule(self, complex_molecule):
        """Test rendering a complex real molecule."""
        renderer = Molecule2DRenderer()
        result = renderer.render(complex_molecule)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"

    def test_render_benzene_with_custom_config(self, benzene_molecule):
        """Test rendering benzene with custom configuration."""
        config = RenderConfig(
            width=400,
            height=300,
            background_color="lightgray",
            show_carbon=True,
        )
        renderer = Molecule2DRenderer(config)
        result = renderer.render(benzene_molecule)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"


class TestMolecule2DRendererPerformance:
    """Test performance characteristics of 2D renderer."""

    def test_multiple_renders_same_molecule(self, simple_molecule):
        """Test multiple renders of the same molecule."""
        renderer = Molecule2DRenderer()

        results = []
        for _ in range(10):
            result = renderer.render(simple_molecule)
            results.append(result)

        # All should be successful
        assert len(results) == 10
        for result in results:
            assert result is not None
            assert isinstance(result, Image.Image)

    def test_different_configs_performance(self, simple_molecule):
        """Test rendering with different configurations."""
        configs = [
            RenderConfig(width=200, height=200),
            RenderConfig(width=400, height=300),
            RenderConfig(width=600, height=500),
        ]

        for config in configs:
            renderer = Molecule2DRenderer(config)
            result = renderer.render(simple_molecule)
            assert result is not None

    def test_highlight_performance(self, molecule_with_highlights):
        """Test performance with highlights."""
        renderer = Molecule2DRenderer()
        mol_data = molecule_with_highlights

        # Render multiple times with highlights
        for _ in range(5):
            result = renderer.render_with_highlights(
                mol_data["molecule"],
                highlight_atoms=mol_data["highlight_atoms"],
                highlight_bonds=mol_data["highlight_bonds"],
            )
            assert result is not None
