"""
Tests for molecular renderers module.

Comprehensive test coverage for all renderer classes and utility functions.
"""

import pytest
from unittest.mock import Mock, patch

from molecular_string_renderer.config import RenderConfig
from molecular_string_renderer.renderers import (
    ColorUtils,
    DrawerConfigurationManager,
    Molecule2DRenderer,
    MoleculeGridRenderer,
    get_renderer,
)


class TestColorUtils:
    """Test color utility functions."""

    def test_parse_color_to_rgba_valid_color_name(self):
        """Test parsing valid color names."""
        rgba = ColorUtils.parse_color_to_rgba("red")
        assert rgba == (1.0, 0.0, 0.0, 1.0)

    def test_parse_color_to_rgba_valid_hex(self):
        """Test parsing valid hex colors."""
        rgba = ColorUtils.parse_color_to_rgba("#FF0000")
        assert rgba == (1.0, 0.0, 0.0, 1.0)

    def test_parse_color_to_rgba_invalid_color(self):
        """Test parsing invalid color falls back to white."""
        with patch("molecular_string_renderer.renderers.logger") as mock_logger:
            rgba = ColorUtils.parse_color_to_rgba("invalid_color")
            assert rgba == (1.0, 1.0, 1.0, 1.0)
            mock_logger.warning.assert_called_once()

    def test_is_white_background_true_cases(self):
        """Test white background detection for various white representations."""
        white_values = [
            "white",
            "WHITE",
            " white ",
            "#ffffff",
            "#fff",
            "rgb(255,255,255)",
        ]
        for value in white_values:
            assert ColorUtils.is_white_background(value)

    def test_is_white_background_false_cases(self):
        """Test white background detection for non-white colors."""
        non_white_values = ["red", "#ff0000", "rgb(255,0,0)", "blue"]
        for value in non_white_values:
            assert not ColorUtils.is_white_background(value)


class TestDrawerConfigurationManager:
    """Test drawer configuration manager."""

    def test_init(self):
        """Test initialization."""
        config = RenderConfig()
        manager = DrawerConfigurationManager(config)
        assert manager.config == config

    @patch("molecular_string_renderer.renderers.rdMolDraw2D.MolDraw2DCairo")
    def test_create_drawer(self, mock_drawer_class):
        """Test drawer creation."""
        config = RenderConfig(width=400, height=300)
        manager = DrawerConfigurationManager(config)

        mock_drawer = Mock()
        mock_drawer_class.return_value = mock_drawer

        drawer = manager.create_drawer()

        mock_drawer_class.assert_called_once_with(400, 300)
        assert drawer == mock_drawer

    @patch("molecular_string_renderer.renderers.rdMolDraw2D.MolDraw2DCairo")
    def test_configure_drawer_options_white_background(self, mock_drawer_class):
        """Test drawer configuration with white background."""
        config = RenderConfig(background_color="white", show_carbon=True)
        manager = DrawerConfigurationManager(config)

        mock_drawer = Mock()
        mock_options = Mock()
        mock_drawer.drawOptions.return_value = mock_options
        mock_drawer_class.return_value = mock_drawer

        manager.create_drawer()

        # White background should not call setBackgroundColour
        mock_options.setBackgroundColour.assert_not_called()
        assert mock_options.explicitMethyl

    @patch("molecular_string_renderer.renderers.rdMolDraw2D.MolDraw2DCairo")
    @patch("molecular_string_renderer.renderers.ColorUtils.parse_color_to_rgba")
    def test_configure_drawer_options_colored_background(
        self, mock_parse_color, mock_drawer_class
    ):
        """Test drawer configuration with colored background."""
        config = RenderConfig(background_color="red", show_carbon=False)
        manager = DrawerConfigurationManager(config)

        mock_drawer = Mock()
        mock_options = Mock()
        mock_drawer.drawOptions.return_value = mock_options
        mock_drawer_class.return_value = mock_drawer
        mock_parse_color.return_value = (1.0, 0.0, 0.0, 1.0)

        manager.create_drawer()

        mock_parse_color.assert_called_once_with("red")
        mock_options.setBackgroundColour.assert_called_once_with((1.0, 0.0, 0.0, 1.0))
        assert not mock_options.explicitMethyl


class TestMolecularRenderer:
    """Test abstract molecular renderer base class."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = RenderConfig(width=500)

        # We can't instantiate abstract class directly, so we'll test through subclass
        renderer = Molecule2DRenderer(config)
        assert renderer.config == config

    def test_init_without_config(self):
        """Test initialization without config uses default."""
        renderer = Molecule2DRenderer()
        assert isinstance(renderer.config, RenderConfig)

    def test_prepare_molecule_none_input(self):
        """Test molecule preparation with None input."""
        renderer = Molecule2DRenderer()
        with pytest.raises(ValueError, match="Cannot render None molecule"):
            renderer._prepare_molecule(None)

    @patch("molecular_string_renderer.renderers.Chem.Mol")
    @patch("molecular_string_renderer.renderers.rdDepictor.Compute2DCoords")
    def test_prepare_molecule_valid_input(self, mock_compute_coords, mock_mol):
        """Test molecule preparation with valid input."""
        renderer = Molecule2DRenderer()
        mol = Mock()

        # Mock molecule creation
        prepared_mol = Mock()
        mock_mol.return_value = prepared_mol
        prepared_mol.GetAtoms.return_value = []

        result = renderer._prepare_molecule(mol)

        mock_mol.assert_called_once_with(mol)
        mock_compute_coords.assert_called_once_with(prepared_mol)
        assert result == prepared_mol

    def test_prepare_molecule_compute_coords_failure(self):
        """Test molecule preparation when coordinate computation fails."""
        renderer = Molecule2DRenderer()
        mol = Mock()

        with patch("molecular_string_renderer.renderers.Chem.Mol") as mock_mol:
            with patch(
                "molecular_string_renderer.renderers.rdDepictor.Compute2DCoords"
            ) as mock_compute:
                mock_mol.return_value = Mock()
                mock_compute.side_effect = Exception("Compute failed")

                with pytest.raises(
                    ValueError, match="Failed to compute 2D coordinates"
                ):
                    renderer._prepare_molecule(mol)

    def test_get_molecule_dimensions_none_input(self):
        """Test dimension calculation with None input."""
        renderer = Molecule2DRenderer()
        with pytest.raises(
            ValueError, match="Cannot calculate dimensions for None molecule"
        ):
            renderer._get_molecule_dimensions(None)

    def test_get_molecule_dimensions_no_conformers(self):
        """Test dimension calculation with no conformers."""
        renderer = Molecule2DRenderer()
        mol = Mock()
        mol.GetNumConformers.return_value = 0

        width, height = renderer._get_molecule_dimensions(mol)
        assert width == 1.0
        assert height == 1.0

    def test_get_molecule_dimensions_single_atom(self):
        """Test dimension calculation for single atom molecule."""
        renderer = Molecule2DRenderer()
        mol = Mock()
        mol.GetNumConformers.return_value = 1
        mol.GetNumAtoms.return_value = 1

        conformer = Mock()
        position = Mock()
        position.x, position.y = 0.0, 0.0
        conformer.GetAtomPosition.return_value = position
        mol.GetConformer.return_value = conformer

        width, height = renderer._get_molecule_dimensions(mol)
        assert width == 1.0
        assert height == 1.0

    def test_get_molecule_dimensions_multi_atom(self):
        """Test dimension calculation for multi-atom molecule."""
        renderer = Molecule2DRenderer()
        mol = Mock()
        mol.GetNumConformers.return_value = 1
        mol.GetNumAtoms.return_value = 2

        conformer = Mock()
        positions = [Mock(), Mock()]
        positions[0].x, positions[0].y = 0.0, 0.0
        positions[1].x, positions[1].y = 2.0, 1.5
        conformer.GetAtomPosition.side_effect = positions
        mol.GetConformer.return_value = conformer

        width, height = renderer._get_molecule_dimensions(mol)
        assert width == 2.0
        assert height == 1.5


class TestMolecule2DRenderer:
    """Test 2D molecule renderer."""

    def test_init(self):
        """Test 2D renderer initialization."""
        config = RenderConfig()
        renderer = Molecule2DRenderer(config)
        assert renderer.config == config
        assert isinstance(renderer._drawer_manager, DrawerConfigurationManager)

    @patch("molecular_string_renderer.renderers.Molecule2DRenderer._prepare_molecule")
    @patch(
        "molecular_string_renderer.renderers.DrawerConfigurationManager.create_drawer"
    )
    @patch("molecular_string_renderer.renderers.Image.open")
    def test_render_success(self, mock_image_open, mock_create_drawer, mock_prepare):
        """Test successful molecule rendering."""
        renderer = Molecule2DRenderer()
        mol = Mock()
        prepared_mol = Mock()
        mock_prepare.return_value = prepared_mol

        # Mock drawer
        drawer = Mock()
        drawer.GetDrawingText.return_value = b"fake_png_data"
        mock_create_drawer.return_value = drawer

        # Mock PIL Image
        img = Mock()
        img.mode = "RGBA"
        img.size = (300, 300)
        mock_image_open.return_value = img

        result = renderer.render(mol)

        mock_prepare.assert_called_once_with(mol)
        mock_create_drawer.assert_called_once()
        drawer.DrawMolecule.assert_called_once_with(prepared_mol)
        drawer.FinishDrawing.assert_called_once()
        assert result == img

    @patch("molecular_string_renderer.renderers.Molecule2DRenderer._prepare_molecule")
    def test_render_prepare_failure(self, mock_prepare):
        """Test render failure during molecule preparation."""
        renderer = Molecule2DRenderer()
        mol = Mock()
        mock_prepare.side_effect = ValueError("Preparation failed")

        with pytest.raises(ValueError, match="Preparation failed"):
            renderer.render(mol)

    @patch("molecular_string_renderer.renderers.Molecule2DRenderer._prepare_molecule")
    @patch(
        "molecular_string_renderer.renderers.DrawerConfigurationManager.create_drawer"
    )
    def test_render_drawer_failure(self, mock_create_drawer, mock_prepare):
        """Test render failure during drawing."""
        renderer = Molecule2DRenderer()
        mol = Mock()
        mock_prepare.return_value = Mock()
        mock_create_drawer.side_effect = Exception("Drawer failed")

        with pytest.raises(RuntimeError, match="Failed to render molecule"):
            renderer.render(mol)

    def test_render_with_highlights_invalid_molecule(self):
        """Test highlighted rendering with invalid molecule."""
        renderer = Molecule2DRenderer()
        with pytest.raises(ValueError):
            renderer.render_with_highlights(None)

    @patch("molecular_string_renderer.renderers.Molecule2DRenderer._prepare_molecule")
    @patch("molecular_string_renderer.renderers.DrawerConfigurationManager")
    @patch("molecular_string_renderer.renderers.Image.open")
    def test_render_with_highlights_success(
        self, mock_image_open, mock_manager_class, mock_prepare
    ):
        """Test successful highlighted rendering."""
        renderer = Molecule2DRenderer()
        mol = Mock()
        prepared_mol = Mock()
        mock_prepare.return_value = prepared_mol

        # Mock drawer manager and drawer
        drawer_manager = Mock()
        drawer = Mock()
        drawer.GetDrawingText.return_value = b"fake_png_data"
        drawer_manager.create_drawer.return_value = drawer
        mock_manager_class.return_value = drawer_manager

        # Mock PIL Image
        img = Mock()
        img.mode = "RGBA"
        img.size = (300, 300)
        mock_image_open.return_value = img

        result = renderer.render_with_highlights(
            mol,
            highlight_atoms=[0, 1],
            highlight_bonds=[0],
            highlight_colors={0: (1.0, 0.0, 0.0)},
        )

        mock_prepare.assert_called_once_with(mol)
        drawer.DrawMolecule.assert_called_once()
        assert result == img


class TestMoleculeGridRenderer:
    """Test molecule grid renderer."""

    def test_init_valid_params(self):
        """Test grid renderer initialization with valid parameters."""
        config = RenderConfig()
        renderer = MoleculeGridRenderer(config, mols_per_row=3, mol_size=(150, 150))
        assert renderer.config == config
        assert renderer.mols_per_row == 3
        assert renderer.mol_size == (150, 150)

    def test_init_invalid_mols_per_row(self):
        """Test initialization with invalid mols_per_row."""
        with pytest.raises(ValueError, match="mols_per_row must be at least 1"):
            MoleculeGridRenderer(mols_per_row=0)

    def test_init_invalid_mol_size(self):
        """Test initialization with invalid mol_size."""
        with pytest.raises(ValueError, match="mol_size must be a tuple"):
            MoleculeGridRenderer(mol_size=(0, 100))

    @patch("molecular_string_renderer.renderers.MoleculeGridRenderer.render_grid")
    def test_render_single_molecule(self, mock_render_grid):
        """Test rendering single molecule (delegates to grid)."""
        renderer = MoleculeGridRenderer()
        mol = Mock()
        expected_result = Mock()
        mock_render_grid.return_value = expected_result

        result = renderer.render(mol)

        mock_render_grid.assert_called_once_with([mol])
        assert result == expected_result

    def test_render_grid_empty_list(self):
        """Test grid rendering with empty molecule list."""
        renderer = MoleculeGridRenderer()
        with pytest.raises(ValueError, match="Cannot render empty molecule list"):
            renderer.render_grid([])

    @patch("molecular_string_renderer.renderers.MoleculeGridRenderer._prepare_molecule")
    @patch("molecular_string_renderer.renderers.Draw.MolsToGridImage")
    def test_render_grid_success(self, mock_grid_image, mock_prepare):
        """Test successful grid rendering."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock()]
        prepared_mols = [Mock(), Mock()]
        mock_prepare.side_effect = prepared_mols

        # Mock grid image
        img = Mock()
        img.mode = "RGBA"
        img.size = (600, 400)
        mock_grid_image.return_value = img

        result = renderer.render_grid(mols)

        assert mock_prepare.call_count == 2
        mock_grid_image.assert_called_once_with(
            prepared_mols, molsPerRow=4, subImgSize=(200, 200), legends=None
        )
        assert result == img

    @patch("molecular_string_renderer.renderers.MoleculeGridRenderer._prepare_molecule")
    def test_render_grid_with_invalid_molecules(self, mock_prepare):
        """Test grid rendering with some invalid molecules."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock(), Mock()]

        # First molecule is valid, second fails, third is valid
        valid_mol1, valid_mol2 = Mock(), Mock()
        mock_prepare.side_effect = [valid_mol1, Exception("Invalid"), valid_mol2]

        with patch(
            "molecular_string_renderer.renderers.Draw.MolsToGridImage"
        ) as mock_grid:
            img = Mock()
            img.mode = "RGBA"
            mock_grid.return_value = img

            renderer.render_grid(mols)

            # Should only render valid molecules
            mock_grid.assert_called_once_with(
                [valid_mol1, valid_mol2],
                molsPerRow=4,
                subImgSize=(200, 200),
                legends=None,
            )

    @patch("molecular_string_renderer.renderers.MoleculeGridRenderer._prepare_molecule")
    def test_render_grid_all_invalid_molecules(self, mock_prepare):
        """Test grid rendering when all molecules are invalid."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock()]
        mock_prepare.side_effect = [Exception("Invalid1"), Exception("Invalid2")]

        with pytest.raises(ValueError, match="No valid molecules found in input list"):
            renderer.render_grid(mols)

    @patch("molecular_string_renderer.renderers.MoleculeGridRenderer._prepare_molecule")
    @patch("molecular_string_renderer.renderers.Draw.MolsToGridImage")
    def test_render_grid_with_legends_mismatch(self, mock_grid_image, mock_prepare):
        """Test grid rendering with mismatched legend count."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock()]
        legends = ["Legend1"]  # Only one legend for two molecules

        mock_prepare.side_effect = [Mock(), Mock()]

        img = Mock()
        img.mode = "RGBA"
        mock_grid_image.return_value = img

        with patch("molecular_string_renderer.renderers.logger") as mock_logger:
            renderer.render_grid(mols, legends)

            # Should warn about legend count mismatch but still pass the legends
            mock_grid_image.assert_called_once()
            call_args = mock_grid_image.call_args
            assert call_args[1]["legends"] == ["Legend1"]
            mock_logger.warning.assert_called()

    def test_render_grid_with_highlights_mismatched_lengths(self):
        """Test grid rendering with highlights having mismatched lengths."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock()]
        highlight_atoms = [[0]]  # Only one highlight list for two molecules

        with pytest.raises(ValueError, match="highlight_atoms_list length must match"):
            renderer.render_grid_with_highlights(
                mols, highlight_atoms_list=highlight_atoms
            )


class TestGetRenderer:
    """Test renderer factory function."""

    def test_get_renderer_2d(self):
        """Test getting 2D renderer."""
        renderer = get_renderer("2d")
        assert isinstance(renderer, Molecule2DRenderer)

    def test_get_renderer_grid(self):
        """Test getting grid renderer."""
        renderer = get_renderer("grid")
        assert isinstance(renderer, MoleculeGridRenderer)

    def test_get_renderer_with_config(self):
        """Test getting renderer with config."""
        config = RenderConfig(width=500)
        renderer = get_renderer("2d", config)
        assert isinstance(renderer, Molecule2DRenderer)
        assert renderer.config == config

    def test_get_renderer_invalid_type(self):
        """Test getting renderer with invalid type."""
        with pytest.raises(ValueError, match="Unsupported renderer"):
            get_renderer("invalid")

    def test_get_renderer_invalid_config_type(self):
        """Test getting renderer with invalid config type."""
        with pytest.raises(TypeError, match="config must be a RenderConfig instance"):
            get_renderer("2d", "invalid_config")

    def test_get_renderer_case_insensitive(self):
        """Test renderer type is case insensitive."""
        renderer1 = get_renderer("2D")
        renderer2 = get_renderer("GRID")
        assert isinstance(renderer1, Molecule2DRenderer)
        assert isinstance(renderer2, MoleculeGridRenderer)

    @patch("molecular_string_renderer.renderers.Molecule2DRenderer")
    def test_get_renderer_creation_failure(self, mock_renderer_class):
        """Test handling of renderer creation failure."""
        mock_renderer_class.side_effect = Exception("Creation failed")

        with pytest.raises(RuntimeError, match="Failed to create 2d renderer"):
            get_renderer("2d")
