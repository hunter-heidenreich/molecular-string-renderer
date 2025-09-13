"""
Direct tests for renderer module classes.

Tests Molecule2DRenderer, MoleculeGridRenderer classes and their specific
features like highlighting, initialization, and direct rendering capabilities.
"""

import pytest

from molecular_string_renderer.config import RenderConfig
from molecular_string_renderer.parsers import SMILESParser
from molecular_string_renderer.renderers import (
    Molecule2DRenderer,
    MoleculeGridRenderer,
    get_renderer,
)
from tests.conftest import SAMPLE_MOLECULES


class TestMolecule2DRenderer:
    """Test Molecule2DRenderer class directly."""

    def test_renderer_initialization(self):
        """Test Molecule2DRenderer initialization."""
        # Default initialization
        renderer = Molecule2DRenderer()
        assert renderer.config is not None
        assert renderer.config.width == 500
        assert renderer.config.height == 500

        # Custom configuration
        config = RenderConfig(
            width=300,
            height=400,
            background_color="red",
            show_hydrogen=True,
        )
        renderer = Molecule2DRenderer(config)
        assert renderer.config.width == 300
        assert renderer.config.height == 400
        assert renderer.config.background_color == "red"
        assert renderer.config.show_hydrogen is True

    def test_renderer_basic_rendering(self):
        """Test basic molecule rendering."""
        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])

        renderer = Molecule2DRenderer()
        image = renderer.render(mol)

        assert image is not None
        assert hasattr(image, "size")
        assert image.size == (500, 500)  # Default size
        assert image.mode == "RGBA"

    def test_renderer_custom_config_rendering(self):
        """Test rendering with custom configuration."""
        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["benzene"])

        config = RenderConfig(
            width=200,
            height=300,
            background_color="lightblue",
            show_hydrogen=True,
            dpi=200,
        )

        renderer = Molecule2DRenderer(config)
        image = renderer.render(mol)

        assert image is not None
        assert image.size == (200, 300)
        assert image.mode == "RGBA"

    def test_renderer_different_molecules(self):
        """Test rendering different types of molecules."""
        parser = SMILESParser()
        renderer = Molecule2DRenderer()

        test_molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
            SAMPLE_MOLECULES["smiles"]["acetic_acid"],
            SAMPLE_MOLECULES["smiles"]["aspirin"],
            SAMPLE_MOLECULES["smiles"]["caffeine"],
            SAMPLE_MOLECULES["smiles"]["methane"],
            SAMPLE_MOLECULES["smiles"]["water"],
        ]

        for smiles in test_molecules:
            mol = parser.parse(smiles)
            image = renderer.render(mol)

            assert image is not None, f"Failed to render {smiles}"
            assert image.size[0] > 0 and image.size[1] > 0, f"Zero size for {smiles}"

    def test_renderer_error_handling(self):
        """Test error handling in renderer."""
        renderer = Molecule2DRenderer()

        # Test with None molecule
        with pytest.raises(ValueError, match="Cannot render None molecule"):
            renderer.render(None)

    def test_renderer_prepare_molecule(self):
        """Test molecule preparation functionality."""
        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])

        renderer = Molecule2DRenderer()
        prepared_mol = renderer._prepare_molecule(mol)

        assert prepared_mol is not None
        assert prepared_mol.GetNumAtoms() > 0

        # Test with None
        with pytest.raises(ValueError, match="Cannot render None molecule"):
            renderer._prepare_molecule(None)


class TestMolecule2DRendererHighlighting:
    """Test highlighting functionality of Molecule2DRenderer."""

    def test_render_with_highlights_atoms(self):
        """Test rendering with atom highlighting."""
        parser = SMILESParser()
        mol = parser.parse(
            SAMPLE_MOLECULES["smiles"]["benzene"]
        )  # Benzene has 6 carbons

        renderer = Molecule2DRenderer()

        # Highlight first 3 atoms
        highlight_atoms = [0, 1, 2]
        image = renderer.render_with_highlights(
            mol,
            highlight_atoms=highlight_atoms,
        )

        assert image is not None
        assert image.size[0] > 0 and image.size[1] > 0

    def test_render_with_highlights_bonds(self):
        """Test rendering with bond highlighting."""
        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["benzene"])

        renderer = Molecule2DRenderer()

        # Highlight first few bonds
        highlight_bonds = [0, 1, 2]
        image = renderer.render_with_highlights(
            mol,
            highlight_bonds=highlight_bonds,
        )

        assert image is not None
        assert image.size[0] > 0 and image.size[1] > 0

    def test_render_with_highlights_colors(self):
        """Test rendering with custom highlight colors."""
        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])

        renderer = Molecule2DRenderer()

        highlight_atoms = [0, 1, 2]
        highlight_colors = {0: "red", 1: "green", 2: "blue"}

        image = renderer.render_with_highlights(
            mol,
            highlight_atoms=highlight_atoms,
            highlight_colors=highlight_colors,
        )

        assert image is not None
        assert image.size[0] > 0 and image.size[1] > 0

    def test_render_with_highlights_combined(self):
        """Test rendering with both atom and bond highlights."""
        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["acetic_acid"])

        renderer = Molecule2DRenderer()

        highlight_atoms = [0, 1]
        highlight_bonds = [0]
        highlight_colors = {0: "yellow", 1: "orange"}

        image = renderer.render_with_highlights(
            mol,
            highlight_atoms=highlight_atoms,
            highlight_bonds=highlight_bonds,
            highlight_colors=highlight_colors,
        )

        assert image is not None
        assert image.size[0] > 0 and image.size[1] > 0

    def test_render_with_highlights_config_preservation(self):
        """Test that original config is preserved after highlighting."""
        config = RenderConfig(
            width=300,
            highlight_atoms=[5, 6],  # Original highlights
            highlight_bonds=[3, 4],
        )

        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["caffeine"])

        renderer = Molecule2DRenderer(config)

        # Use different highlights temporarily
        image = renderer.render_with_highlights(
            mol,
            highlight_atoms=[0, 1],
            highlight_bonds=[0, 1],
        )

        assert image is not None

        # Check that original config is restored
        assert renderer.config.highlight_atoms == [5, 6]
        assert renderer.config.highlight_bonds == [3, 4]

    def test_render_with_highlights_empty_lists(self):
        """Test rendering with empty highlight lists."""
        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])

        renderer = Molecule2DRenderer()

        image = renderer.render_with_highlights(
            mol,
            highlight_atoms=[],
            highlight_bonds=[],
        )

        assert image is not None
        assert image.size[0] > 0 and image.size[1] > 0

    def test_render_with_highlights_none_values(self):
        """Test rendering with None highlight values."""
        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])

        renderer = Molecule2DRenderer()

        image = renderer.render_with_highlights(
            mol,
            highlight_atoms=None,
            highlight_bonds=None,
            highlight_colors=None,
        )

        assert image is not None
        assert image.size[0] > 0 and image.size[1] > 0


class TestMoleculeGridRendererDirect:
    """Test MoleculeGridRenderer class directly."""

    def test_grid_renderer_initialization(self):
        """Test MoleculeGridRenderer initialization."""
        # Default initialization
        renderer = MoleculeGridRenderer()
        assert renderer.config is not None
        assert renderer.mols_per_row == 4
        assert renderer.mol_size == (200, 200)

        # Custom initialization
        config = RenderConfig(width=300, background_color="yellow")
        renderer = MoleculeGridRenderer(
            config=config,
            mols_per_row=3,
            mol_size=(150, 150),
        )
        assert renderer.config.width == 300
        assert renderer.config.background_color == "yellow"
        assert renderer.mols_per_row == 3
        assert renderer.mol_size == (150, 150)

    def test_grid_renderer_single_molecule_render(self):
        """Test that grid renderer can render single molecules."""
        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])

        renderer = MoleculeGridRenderer()
        image = renderer.render(mol)

        assert image is not None
        assert image.size[0] > 0 and image.size[1] > 0

    def test_grid_renderer_grid_rendering(self):
        """Test grid rendering functionality."""
        parser = SMILESParser()
        mols = [
            parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"]),
            parser.parse(SAMPLE_MOLECULES["smiles"]["benzene"]),
            parser.parse(SAMPLE_MOLECULES["smiles"]["acetic_acid"]),
        ]

        renderer = MoleculeGridRenderer(mols_per_row=2, mol_size=(100, 100))
        image = renderer.render_grid(mols)

        assert image is not None
        assert image.mode == "RGBA"

        # Grid should be roughly 2*100 wide (2 mols per row) and 2*100 tall (2 rows for 3 mols)
        assert image.size[0] >= 180  # Allow some tolerance for spacing
        assert image.size[1] >= 180

    def test_grid_renderer_with_legends(self):
        """Test grid rendering with legends."""
        parser = SMILESParser()
        mols = [
            parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"]),
            parser.parse(SAMPLE_MOLECULES["smiles"]["benzene"]),
        ]
        legends = ["Ethanol", "Benzene"]

        renderer = MoleculeGridRenderer()
        image = renderer.render_grid(mols, legends)

        assert image is not None
        assert image.size[0] > 0 and image.size[1] > 0

    def test_grid_renderer_different_layouts(self):
        """Test grid renderer with different layout configurations."""
        parser = SMILESParser()
        mols = [
            parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"]),
            parser.parse(SAMPLE_MOLECULES["smiles"]["benzene"]),
            parser.parse(SAMPLE_MOLECULES["smiles"]["acetic_acid"]),
            parser.parse(SAMPLE_MOLECULES["smiles"]["methane"]),
            parser.parse(SAMPLE_MOLECULES["smiles"]["water"]),
            parser.parse(SAMPLE_MOLECULES["smiles"]["propanol"]),
        ]

        # Test different mols_per_row values
        for mols_per_row in [1, 2, 3, 6]:
            renderer = MoleculeGridRenderer(
                mols_per_row=mols_per_row,
                mol_size=(80, 80),
            )
            image = renderer.render_grid(mols)

            assert image is not None, f"Failed for {mols_per_row} mols per row"
            assert image.size[0] > 0 and image.size[1] > 0

            # Check approximate dimensions
            expected_cols = min(mols_per_row, len(mols))
            expected_rows = (len(mols) + mols_per_row - 1) // mols_per_row

            # Allow some tolerance for spacing and borders
            assert image.size[0] >= expected_cols * 80 * 0.8
            assert image.size[1] >= expected_rows * 80 * 0.8

    def test_grid_renderer_different_sizes(self):
        """Test grid renderer with different molecule sizes."""
        parser = SMILESParser()
        mols = [
            parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"]),
            parser.parse(SAMPLE_MOLECULES["smiles"]["benzene"]),
        ]

        mol_sizes = [(50, 50), (100, 100), (200, 200), (150, 300)]

        for mol_size in mol_sizes:
            renderer = MoleculeGridRenderer(
                mols_per_row=2,
                mol_size=mol_size,
            )
            image = renderer.render_grid(mols)

            assert image is not None, f"Failed for mol_size {mol_size}"
            assert image.size[0] > 0 and image.size[1] > 0

            # Check that image size is related to mol_size
            # For 2 molecules in 1 row, width should be roughly 2 * mol_size[0]
            assert image.size[0] >= mol_size[0] * 2 * 0.8
            assert image.size[1] >= mol_size[1] * 0.8

    def test_grid_renderer_error_handling(self):
        """Test error handling in grid renderer."""
        renderer = MoleculeGridRenderer()

        # Empty molecule list
        with pytest.raises(ValueError, match="Cannot render empty molecule list"):
            renderer.render_grid([])

        # None molecule in render method
        with pytest.raises(ValueError, match="Cannot render None molecule"):
            renderer.render(None)

    def test_grid_renderer_single_molecule_in_grid(self):
        """Test grid rendering with just one molecule."""
        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])

        renderer = MoleculeGridRenderer(mols_per_row=4, mol_size=(150, 150))
        image = renderer.render_grid([mol])

        assert image is not None
        assert image.size[0] > 0 and image.size[1] > 0

    def test_grid_renderer_large_grid(self):
        """Test grid rendering with a larger number of molecules."""
        parser = SMILESParser()

        # Create 8 molecules (using available samples, repeating some)
        smiles_list = list(SAMPLE_MOLECULES["smiles"].values())
        mols = [parser.parse(smiles) for smiles in smiles_list]

        renderer = MoleculeGridRenderer(mols_per_row=3, mol_size=(80, 80))
        image = renderer.render_grid(mols)

        assert image is not None
        assert image.size[0] > 0 and image.size[1] > 0

        # For 8 molecules with 3 per row: 3 rows (3+3+2)
        num_rows = (len(mols) + 2) // 3
        assert image.size[0] >= 3 * 80 * 0.8  # 3 columns
        assert image.size[1] >= num_rows * 80 * 0.8


class TestRendererFactory:
    """Test the renderer factory function."""

    def test_get_renderer_2d(self):
        """Test getting 2D renderer from factory."""
        renderer = get_renderer("2d")
        assert isinstance(renderer, Molecule2DRenderer)
        assert renderer.config is not None

        # With custom config
        config = RenderConfig(width=300, height=300)
        renderer = get_renderer("2d", config)
        assert isinstance(renderer, Molecule2DRenderer)
        assert renderer.config.width == 300

    def test_get_renderer_grid(self):
        """Test getting grid renderer from factory."""
        renderer = get_renderer("grid")
        assert isinstance(renderer, MoleculeGridRenderer)
        assert renderer.config is not None
        assert renderer.mols_per_row == 4

        # With custom config
        config = RenderConfig(background_color="red")
        renderer = get_renderer("grid", config)
        assert isinstance(renderer, MoleculeGridRenderer)
        assert renderer.config.background_color == "red"

    def test_get_renderer_case_insensitive(self):
        """Test that renderer factory is case-insensitive."""
        test_cases = [
            ("2D", Molecule2DRenderer),
            ("2d", Molecule2DRenderer),
            ("GRID", MoleculeGridRenderer),
            ("Grid", MoleculeGridRenderer),
            ("grid", MoleculeGridRenderer),
        ]

        for renderer_type, expected_class in test_cases:
            renderer = get_renderer(renderer_type)
            assert isinstance(renderer, expected_class), (
                f"Wrong renderer type for {renderer_type}"
            )

    def test_get_renderer_whitespace_handling(self):
        """Test that renderer factory handles whitespace."""
        test_cases = [
            ("  2d  ", Molecule2DRenderer),
            (" grid ", MoleculeGridRenderer),
        ]

        for renderer_type, expected_class in test_cases:
            renderer = get_renderer(renderer_type)
            assert isinstance(renderer, expected_class), (
                f"Wrong renderer type for '{renderer_type}'"
            )

    def test_get_renderer_unsupported_type(self):
        """Test error handling for unsupported renderer types."""
        unsupported_types = [
            "3d",
            "unsupported",
            "invalid",
            "",
            "xyz",
        ]

        for renderer_type in unsupported_types:
            with pytest.raises(ValueError, match="Unsupported renderer"):
                get_renderer(renderer_type)


class TestRendererIntegration:
    """Integration tests for renderers with other components."""

    def test_renderer_with_different_parsers(self):
        """Test renderers with molecules from different parsers."""
        from molecular_string_renderer.parsers import InChIParser, SELFIESParser

        # Parse same molecule with different parsers
        smiles_parser = SMILESParser()
        inchi_parser = InChIParser()
        selfies_parser = SELFIESParser()

        smiles_mol = smiles_parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])
        inchi_mol = inchi_parser.parse(SAMPLE_MOLECULES["inchi"]["ethanol"])
        selfies_mol = selfies_parser.parse(SAMPLE_MOLECULES["selfies"]["ethanol"])

        renderer = Molecule2DRenderer()

        # All should render successfully
        for mol in [smiles_mol, inchi_mol, selfies_mol]:
            image = renderer.render(mol)
            assert image is not None
            assert image.size[0] > 0 and image.size[1] > 0

    def test_renderer_output_consistency(self):
        """Test that renderers produce consistent output sizes."""
        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["benzene"])

        config = RenderConfig(width=250, height=250)

        # Single molecule renderer
        single_renderer = Molecule2DRenderer(config)
        single_image = single_renderer.render(mol)

        # Grid renderer with single molecule
        grid_renderer = MoleculeGridRenderer(config, mol_size=(250, 250))
        grid_image = grid_renderer.render_grid([mol])

        assert single_image.size == (250, 250)
        # Grid image might be slightly different due to grid layout
        # but should be roughly similar in total area
        assert grid_image.size[0] >= 200  # At least close to expected size
        assert grid_image.size[1] >= 200

    def test_renderer_performance_comparison(self):
        """Test relative performance of different renderers."""
        import time

        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])

        # Time single molecule rendering
        single_renderer = Molecule2DRenderer()
        start_time = time.time()
        single_image = single_renderer.render(mol)
        single_time = time.time() - start_time

        # Time grid rendering with single molecule
        grid_renderer = MoleculeGridRenderer()
        start_time = time.time()
        grid_image = grid_renderer.render_grid([mol])
        grid_time = time.time() - start_time

        assert single_image is not None
        assert grid_image is not None

        # Both should complete in reasonable time
        assert single_time < 5.0, f"Single rendering too slow: {single_time:.2f}s"
        assert grid_time < 5.0, f"Grid rendering too slow: {grid_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__])
