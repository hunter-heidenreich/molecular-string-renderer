"""
Format-specific tests for grid molecular renderer.

Tests functionality unique to the grid renderer that doesn't apply to other renderer types.
"""

from unittest.mock import Mock, patch

import pytest
from PIL import Image
from rdkit import Chem

from molecular_string_renderer.config import RenderConfig
from molecular_string_renderer.renderers.grid import MoleculeGridRenderer


class TestMoleculeGridRendererSpecific:
    """Test grid renderer-specific functionality."""

    def test_init_with_custom_parameters(self):
        """Test grid renderer initialization with custom parameters."""
        config = RenderConfig()
        renderer = MoleculeGridRenderer(config, mols_per_row=3, mol_size=(150, 150))

        assert renderer.config is config
        assert renderer.mols_per_row == 3
        assert renderer.mol_size == (150, 150)

    def test_init_with_default_parameters(self):
        """Test grid renderer initialization with default parameters."""
        renderer = MoleculeGridRenderer()

        assert renderer.mols_per_row == 4  # Default value
        assert renderer.mol_size == (200, 200)  # Default value

    def test_init_invalid_mols_per_row(self):
        """Test grid renderer initialization with invalid mols_per_row."""
        with pytest.raises(ValueError, match="mols_per_row must be at least 1"):
            MoleculeGridRenderer(mols_per_row=0)

        with pytest.raises(ValueError, match="mols_per_row must be at least 1"):
            MoleculeGridRenderer(mols_per_row=-1)

    def test_init_invalid_mol_size(self):
        """Test grid renderer initialization with invalid mol_size."""
        invalid_sizes = [
            (0, 100),  # Zero width
            (100, 0),  # Zero height
            (-100, 100),  # Negative width
            (100, -100),  # Negative height
            (100,),  # Wrong tuple length
            None,  # None
        ]

        for invalid_size in invalid_sizes:
            with pytest.raises(ValueError, match="mol_size must be a tuple"):
                MoleculeGridRenderer(mol_size=invalid_size)

    def test_render_single_molecule_delegates_to_grid(self):
        """Test that render method delegates to render_grid for single molecule."""
        with patch.object(MoleculeGridRenderer, "render_grid") as mock_render_grid:
            renderer = MoleculeGridRenderer()
            mol = Mock()
            expected_result = Mock()
            mock_render_grid.return_value = expected_result

            result = renderer.render(mol)

            mock_render_grid.assert_called_once_with([mol])
            assert result == expected_result

    @patch(
        "molecular_string_renderer.renderers.grid.MoleculeGridRenderer._prepare_molecule"
    )
    @patch("rdkit.Chem.Draw.MolsToGridImage")
    def test_render_grid_basic_flow(self, mock_grid_image, mock_prepare):
        """Test the basic render_grid flow."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock()]
        prepared_mols = [Mock(), Mock()]
        mock_prepare.side_effect = prepared_mols

        # Mock grid image - create a proper PIL Image mock
        from PIL import Image
        img = Mock(spec=Image.Image)
        img.mode = "RGBA"
        img.size = (800, 400)
        mock_grid_image.return_value = img

        result = renderer.render_grid(mols)

        # Verify the flow
        assert mock_prepare.call_count == 2
        mock_grid_image.assert_called_once_with(
            prepared_mols,
            molsPerRow=4,  # Default value
            subImgSize=(200, 200),  # Default value
            legends=None,
            drawOptions=mock_grid_image.call_args.kwargs['drawOptions'],
            highlightAtomLists=None,
            highlightAtomColors=None,
        )
        assert result == img

    def test_render_grid_empty_list_raises_error(self):
        """Test that render_grid with empty list raises ValueError."""
        renderer = MoleculeGridRenderer()

        with pytest.raises(ValueError, match="Cannot render empty molecule list"):
            renderer.render_grid([])

    @patch(
        "molecular_string_renderer.renderers.grid.MoleculeGridRenderer._prepare_molecule"
    )
    @patch("rdkit.Chem.Draw.MolsToGridImage")
    def test_render_grid_with_legends(self, mock_grid_image, mock_prepare):
        """Test render_grid with legends."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock()]
        legends = ["Mol1", "Mol2"]
        prepared_mols = [Mock(), Mock()]
        mock_prepare.side_effect = prepared_mols

        from PIL import Image
        img = Mock(spec=Image.Image)
        img.mode = "RGBA"
        mock_grid_image.return_value = img

        result = renderer.render_grid(mols, legends)

        mock_grid_image.assert_called_once_with(
            prepared_mols,
            molsPerRow=4,
            subImgSize=(200, 200),
            legends=legends,
            drawOptions=mock_grid_image.call_args.kwargs['drawOptions'],
            highlightAtomLists=None,
            highlightAtomColors=None,
        )
        assert result == img

    @patch(
        "molecular_string_renderer.renderers.grid.MoleculeGridRenderer._prepare_molecule"
    )
    @patch("rdkit.Chem.Draw.MolsToGridImage")
    @patch("molecular_string_renderer.renderers.grid.logger")
    def test_render_grid_legend_count_mismatch(
        self, mock_logger, mock_grid_image, mock_prepare
    ):
        """Test render_grid with mismatched legend count."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock()]
        legends = ["Mol1"]  # Only one legend for two molecules
        prepared_mols = [Mock(), Mock()]
        mock_prepare.side_effect = prepared_mols

        from PIL import Image
        img = Mock(spec=Image.Image)
        img.mode = "RGBA"
        mock_grid_image.return_value = img

        renderer.render_grid(mols, legends)

        # Should warn about mismatch
        mock_logger.warning.assert_called()
        mock_grid_image.assert_called_once_with(
            prepared_mols,
            molsPerRow=4,
            subImgSize=(200, 200),
            legends=legends,
            drawOptions=mock_grid_image.call_args.kwargs['drawOptions'],
            highlightAtomLists=None,
            highlightAtomColors=None,
        )

    @patch(
        "molecular_string_renderer.renderers.grid.MoleculeGridRenderer._prepare_molecule"
    )
    def test_render_grid_with_invalid_molecules(self, mock_prepare):
        """Test render_grid filtering out invalid molecules."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock(), Mock()]

        # First and third molecules are valid, second fails
        valid_mol1, valid_mol2 = Mock(), Mock()
        mock_prepare.side_effect = [valid_mol1, Exception("Invalid"), valid_mol2]

        with patch("rdkit.Chem.Draw.MolsToGridImage") as mock_grid:
            from PIL import Image
            img = Mock(spec=Image.Image)
            img.mode = "RGBA"
            mock_grid.return_value = img

            renderer.render_grid(mols)

            # Should only render valid molecules
            mock_grid.assert_called_once_with(
                [valid_mol1, valid_mol2],
                molsPerRow=4,
                subImgSize=(200, 200),
                legends=None,
                drawOptions=mock_grid.call_args.kwargs['drawOptions'],
                highlightAtomLists=None,
                highlightAtomColors=None,
            )

    @patch(
        "molecular_string_renderer.renderers.grid.MoleculeGridRenderer._prepare_molecule"
    )
    def test_render_grid_all_invalid_molecules(self, mock_prepare):
        """Test render_grid when all molecules are invalid."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock()]
        mock_prepare.side_effect = [Exception("Invalid1"), Exception("Invalid2")]

        with pytest.raises(ValueError, match="No valid molecules found in input list"):
            renderer.render_grid(mols)

    @patch(
        "molecular_string_renderer.renderers.grid.MoleculeGridRenderer._prepare_molecule"
    )
    @patch("rdkit.Chem.Draw.MolsToGridImage")
    @patch("molecular_string_renderer.renderers.grid.logger")
    def test_render_grid_filtered_molecules_disables_legends(
        self, mock_logger, mock_grid_image, mock_prepare
    ):
        """Test that legends are disabled when molecules are filtered."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock()]
        legends = ["Mol1", "Mol2"]

        # Only first molecule is valid
        valid_mol = Mock()
        mock_prepare.side_effect = [valid_mol, Exception("Invalid")]

        from PIL import Image
        img = Mock(spec=Image.Image)
        img.mode = "RGBA"
        mock_grid_image.return_value = img

        renderer.render_grid(mols, legends)

        # Should warn about disabling legends
        mock_logger.warning.assert_called()

        # Should call with legends=None
        mock_grid_image.assert_called_once_with(
            [valid_mol],
            molsPerRow=4,
            subImgSize=(200, 200),
            legends=None,
            drawOptions=mock_grid_image.call_args.kwargs['drawOptions'],
            highlightAtomLists=None,
            highlightAtomColors=None,
        )

    def test_render_grid_with_highlights_basic(self):
        """Test render_grid_with_highlights basic functionality."""
        renderer = MoleculeGridRenderer()
        # Use real molecules instead of Mock for RDKit compatibility
        benzene = Chem.MolFromSmiles("c1ccccc1")
        mols = [benzene, benzene]
        highlight_atoms_list = [[0, 1], [2, 3]]
        highlight_bonds_list = [[0], [1, 2]]
        legends = ["Mol1", "Mol2"]

        with patch.object(renderer, "render_grid") as mock_render_grid:
            mock_render_grid.return_value = Mock()

            renderer.render_grid_with_highlights(
                mols,
                highlight_atoms_list=highlight_atoms_list,
                highlight_bonds_list=highlight_bonds_list,
                legends=legends,
            )  # Should delegate to render_grid
            mock_render_grid.assert_called_once()

    def test_render_grid_with_highlights_empty_list_raises_error(self):
        """Test render_grid_with_highlights with empty molecule list."""
        renderer = MoleculeGridRenderer()

        with pytest.raises(ValueError, match="Cannot render empty molecule list"):
            renderer.render_grid_with_highlights([])

    def test_render_grid_with_highlights_mismatched_atoms_list(self):
        """Test render_grid_with_highlights with mismatched highlight_atoms_list length."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock()]
        highlight_atoms_list = [[0, 1]]  # Only one list for two molecules

        with pytest.raises(ValueError, match="highlight_atoms_list length must match"):
            renderer.render_grid_with_highlights(
                mols, highlight_atoms_list=highlight_atoms_list
            )

    def test_render_grid_with_highlights_mismatched_bonds_list(self):
        """Test render_grid_with_highlights with mismatched highlight_bonds_list length."""
        renderer = MoleculeGridRenderer()
        mols = [Mock(), Mock()]
        highlight_bonds_list = [[0]]  # Only one list for two molecules

        with pytest.raises(ValueError, match="highlight_bonds_list length must match"):
            renderer.render_grid_with_highlights(
                mols, highlight_bonds_list=highlight_bonds_list
            )


class TestMoleculeGridRendererConfiguration:
    """Test grid renderer configuration handling."""

    def test_custom_mols_per_row(self):
        """Test custom molecules per row configuration."""
        for mols_per_row in [1, 2, 3, 5, 10]:
            renderer = MoleculeGridRenderer(mols_per_row=mols_per_row)
            assert renderer.mols_per_row == mols_per_row

    def test_custom_mol_size(self):
        """Test custom molecule size configuration."""
        sizes = [
            (50, 50),
            (100, 150),
            (300, 200),
            (500, 400),
        ]

        for size in sizes:
            renderer = MoleculeGridRenderer(mol_size=size)
            assert renderer.mol_size == size

    def test_config_with_custom_parameters(self):
        """Test configuration combined with custom parameters."""
        config = RenderConfig(width=800, height=600, background_color="lightblue")
        renderer = MoleculeGridRenderer(config, mols_per_row=3, mol_size=(150, 150))

        assert renderer.config is config
        assert renderer.config.width == 800
        assert renderer.config.background_color == "lightblue"
        assert renderer.mols_per_row == 3
        assert renderer.mol_size == (150, 150)


class TestMoleculeGridRendererEdgeCases:
    """Test edge cases specific to grid renderer."""

    def test_extreme_mols_per_row_values(self):
        """Test grid renderer with extreme molecules per row values."""
        # Very small
        renderer_small = MoleculeGridRenderer(mols_per_row=1)
        assert renderer_small.mols_per_row == 1

        # Very large
        renderer_large = MoleculeGridRenderer(mols_per_row=100)
        assert renderer_large.mols_per_row == 100

    def test_extreme_mol_size_values(self):
        """Test grid renderer with extreme molecule size values."""
        # Very small
        renderer_small = MoleculeGridRenderer(mol_size=(1, 1))
        assert renderer_small.mol_size == (1, 1)

        # Very large
        renderer_large = MoleculeGridRenderer(mol_size=(2000, 1500))
        assert renderer_large.mol_size == (2000, 1500)

        # Asymmetric
        renderer_asymmetric = MoleculeGridRenderer(mol_size=(50, 500))
        assert renderer_asymmetric.mol_size == (50, 500)

    def test_very_long_molecule_lists(self):
        """Test grid renderer with very long molecule lists."""
        renderer = MoleculeGridRenderer()

        # Create a list of 100 mock molecules
        mols = [Mock() for _ in range(100)]

        with patch.object(renderer, "_prepare_molecule") as mock_prepare:
            with patch("rdkit.Chem.Draw.MolsToGridImage") as mock_grid_image:
                # Mock successful preparation for all molecules
                mock_prepare.side_effect = [Mock() for _ in range(100)]

                from PIL import Image
                img = Mock(spec=Image.Image)
                img.mode = "RGBA"
                mock_grid_image.return_value = img

                result = renderer.render_grid(mols)

                # Should handle all molecules
                assert mock_prepare.call_count == 100
                assert result == img

    def test_single_molecule_in_grid(self):
        """Test grid renderer with single molecule."""
        renderer = MoleculeGridRenderer(mols_per_row=3)
        mols = [Mock()]

        with patch.object(renderer, "_prepare_molecule") as mock_prepare:
            with patch("rdkit.Chem.Draw.MolsToGridImage") as mock_grid_image:
                mock_prepare.return_value = Mock()

                from PIL import Image
                img = Mock(spec=Image.Image)
                img.mode = "RGBA"
                mock_grid_image.return_value = img

                result = renderer.render_grid(mols)

                # Should work with single molecule
                assert mock_prepare.call_count == 1
                mock_grid_image.assert_called_once_with(
                    [mock_prepare.return_value],
                    molsPerRow=3,
                    subImgSize=(200, 200),
                    legends=None,
                    drawOptions=mock_grid_image.call_args.kwargs['drawOptions'],
                    highlightAtomLists=None,
                    highlightAtomColors=None,
                )
                assert result == img


class TestMoleculeGridRendererIntegration:
    """Test grid renderer integration with real RDKit molecules."""

    def test_render_real_molecule_list(self, molecule_list):
        """Test rendering a list of real RDKit molecules."""
        renderer = MoleculeGridRenderer()
        # Filter out None molecules
        valid_mols = [mol for mol in molecule_list if mol is not None]

        if valid_mols:  # Only test if we have valid molecules
            result = renderer.render_grid(valid_mols)

            assert result is not None
            assert isinstance(result, Image.Image)
            assert result.mode == "RGBA"
            assert result.size[0] > 0
            assert result.size[1] > 0

    def test_render_with_legends_real_molecules(self, molecule_list, molecule_legends):
        """Test rendering real molecules with legends."""
        renderer = MoleculeGridRenderer()
        # Filter out None molecules
        valid_mols = [mol for mol in molecule_list if mol is not None]

        if valid_mols:  # Only test if we have valid molecules
            # Adjust legends to match valid molecules
            valid_legends = molecule_legends[: len(valid_mols)]

            result = renderer.render_grid(valid_mols, valid_legends)

            assert result is not None
            assert isinstance(result, Image.Image)
            assert result.mode == "RGBA"

    def test_render_single_real_molecule(self, simple_molecule):
        """Test rendering a single real molecule using grid renderer."""
        renderer = MoleculeGridRenderer()
        result = renderer.render(simple_molecule)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"

    def test_render_with_custom_grid_config(self, molecule_list):
        """Test rendering with custom grid configuration."""
        renderer = MoleculeGridRenderer(mols_per_row=2, mol_size=(150, 150))
        # Filter out None molecules
        valid_mols = [mol for mol in molecule_list if mol is not None]

        if valid_mols:  # Only test if we have valid molecules
            result = renderer.render_grid(valid_mols)

            assert result is not None
            assert isinstance(result, Image.Image)


class TestMoleculeGridRendererPerformance:
    """Test performance characteristics of grid renderer."""

    def test_multiple_renders_same_molecules(self, molecule_list):
        """Test multiple renders of the same molecule list."""
        renderer = MoleculeGridRenderer()
        # Filter out None molecules
        valid_mols = [mol for mol in molecule_list if mol is not None]

        if valid_mols:  # Only test if we have valid molecules
            results = []
            for _ in range(5):
                result = renderer.render_grid(valid_mols)
                results.append(result)

            # All should be successful
            assert len(results) == 5
            for result in results:
                assert result is not None
                assert isinstance(result, Image.Image)

    def test_different_grid_configs_performance(self, simple_molecule):
        """Test rendering with different grid configurations."""
        configs = [
            {"mols_per_row": 2, "mol_size": (100, 100)},
            {"mols_per_row": 4, "mol_size": (200, 200)},
            {"mols_per_row": 6, "mol_size": (150, 150)},
        ]

        for config in configs:
            renderer = MoleculeGridRenderer(**config)
            result = renderer.render(simple_molecule)
            assert result is not None

    def test_large_grid_performance(self, simple_molecule):
        """Test performance with larger grids."""
        renderer = MoleculeGridRenderer(mols_per_row=5)

        # Create list with same molecule repeated
        large_list = [simple_molecule] * 15  # 3 rows of 5 molecules

        result = renderer.render_grid(large_list)
        assert result is not None
        assert isinstance(result, Image.Image)
