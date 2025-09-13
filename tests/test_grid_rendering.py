"""
Comprehensive tests for grid rendering functionality.

Tests the render_molecules_grid function and MoleculeGridRenderer class
with various configurations, error conditions, and edge cases.
"""

import tempfile
from pathlib import Path

import pytest

from molecular_string_renderer import (
    OutputConfig,
    ParserConfig,
    RenderConfig,
    render_molecules_grid,
)
from molecular_string_renderer.renderers import MoleculeGridRenderer
from tests.conftest import SAMPLE_MOLECULES


class TestGridRenderingCore:
    """Test core grid rendering functionality."""

    def test_basic_grid_rendering(self):
        """Test basic grid rendering with default settings."""
        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
            SAMPLE_MOLECULES["smiles"]["acetic_acid"],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grid.png"

            image = render_molecules_grid(
                molecular_strings=molecules,
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Grid should be wider than individual molecules
            assert image.size[0] > 200  # Default mol_size is (200, 200)
            assert image.size[1] > 0

    def test_grid_with_legends(self):
        """Test grid rendering with legends."""
        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
            SAMPLE_MOLECULES["smiles"]["acetic_acid"],
        ]
        legends = ["Ethanol", "Benzene", "Acetic Acid"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grid_with_legends.png"

            image = render_molecules_grid(
                molecular_strings=molecules,
                legends=legends,
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()
            assert image.size[0] > 0
            assert image.size[1] > 0

    def test_grid_layout_configurations(self):
        """Test different grid layout configurations."""
        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
            SAMPLE_MOLECULES["smiles"]["acetic_acid"],
            SAMPLE_MOLECULES["smiles"]["aspirin"],
            SAMPLE_MOLECULES["smiles"]["caffeine"],
            SAMPLE_MOLECULES["smiles"]["methane"],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different mols_per_row configurations
            for mols_per_row in [2, 3, 4, 6]:
                output_path = Path(temp_dir) / f"grid_{mols_per_row}_per_row.png"

                image = render_molecules_grid(
                    molecular_strings=molecules,
                    mols_per_row=mols_per_row,
                    output_path=output_path,
                )

                assert image is not None
                assert output_path.exists()

                # Calculate expected dimensions roughly
                num_rows = (len(molecules) + mols_per_row - 1) // mols_per_row
                expected_min_width = min(mols_per_row, len(molecules)) * 200
                expected_min_height = num_rows * 200

                assert image.size[0] >= expected_min_width * 0.8  # Allow some tolerance
                assert image.size[1] >= expected_min_height * 0.8

    def test_grid_molecule_size_configurations(self):
        """Test different molecule size configurations in grid."""
        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
        ]

        mol_sizes = [(150, 150), (300, 300), (200, 400), (400, 200)]

        with tempfile.TemporaryDirectory() as temp_dir:
            for mol_size in mol_sizes:
                output_path = Path(temp_dir) / f"grid_{mol_size[0]}x{mol_size[1]}.png"

                image = render_molecules_grid(
                    molecular_strings=molecules,
                    mol_size=mol_size,
                    mols_per_row=2,
                    output_path=output_path,
                )

                assert image is not None
                assert output_path.exists()

                # Grid should be roughly 2 * mol_size wide and 1 * mol_size tall
                expected_width = 2 * mol_size[0]
                expected_height = mol_size[1]

                # Allow some tolerance for spacing and borders
                assert image.size[0] >= expected_width * 0.8
                assert image.size[1] >= expected_height * 0.8

    def test_grid_different_output_formats(self):
        """Test grid rendering to different output formats."""
        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
        ]

        output_formats = ["png", "svg", "jpg", "pdf"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in output_formats:
                output_path = Path(temp_dir) / f"grid.{fmt}"

                image = render_molecules_grid(
                    molecular_strings=molecules,
                    output_format=fmt,
                    output_path=output_path,
                )

                assert image is not None, f"No image returned for {fmt}"
                assert output_path.exists(), f"Output file not created for {fmt}"
                assert output_path.stat().st_size > 0, f"Empty output file for {fmt}"

    def test_grid_with_custom_render_config(self):
        """Test grid rendering with custom render configuration."""
        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
        ]

        custom_config = RenderConfig(
            width=300,
            height=300,
            background_color="lightblue",
            show_hydrogen=True,
            dpi=200,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "custom_grid.png"

            image = render_molecules_grid(
                molecular_strings=molecules,
                render_config=custom_config,
                mol_size=(300, 300),
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()

    def test_grid_mixed_molecular_formats(self):
        """Test grid with molecules from different input formats."""
        # Mix SMILES, InChI, and SELFIES
        molecules_smiles = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
        ]

        molecules_inchi = [
            SAMPLE_MOLECULES["inchi"]["ethanol"],
            SAMPLE_MOLECULES["inchi"]["benzene"],
        ]

        molecules_selfies = [
            SAMPLE_MOLECULES["selfies"]["ethanol"],
            SAMPLE_MOLECULES["selfies"]["benzene"],
        ]

        format_tests = [
            ("smiles", molecules_smiles),
            ("inchi", molecules_inchi),
            ("selfies", molecules_selfies),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for format_type, molecules in format_tests:
                output_path = Path(temp_dir) / f"grid_{format_type}.png"

                image = render_molecules_grid(
                    molecular_strings=molecules,
                    format_type=format_type,
                    output_path=output_path,
                )

                assert image is not None, f"No image for {format_type}"
                assert output_path.exists(), f"No file for {format_type}"

    def test_grid_memory_only_rendering(self):
        """Test grid rendering without saving to file."""
        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
            SAMPLE_MOLECULES["smiles"]["acetic_acid"],
        ]

        image = render_molecules_grid(
            molecular_strings=molecules,
            # No output_path specified
        )

        assert image is not None
        assert hasattr(image, "size")
        assert image.size[0] > 0
        assert image.size[1] > 0


class TestGridErrorHandling:
    """Test error handling in grid rendering."""

    def test_empty_molecule_list(self):
        """Test that empty molecule list raises appropriate error."""
        with pytest.raises(ValueError, match="Cannot render empty molecule list"):
            render_molecules_grid(molecular_strings=[])

    def test_grid_with_invalid_molecules(self):
        """Test grid rendering with some invalid molecules."""
        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],  # Valid
            "INVALID_SMILES",  # Invalid
            SAMPLE_MOLECULES["smiles"]["benzene"],  # Valid
        ]

        # Should handle invalid molecules gracefully by filtering them out
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "partial_grid.png"

            # Should succeed with valid molecules only
            image = render_molecules_grid(
                molecular_strings=molecules,
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()

    def test_grid_all_invalid_molecules(self):
        """Test grid rendering when all molecules are invalid."""
        invalid_molecules = ["INVALID1", "INVALID2", "INVALID3"]

        with pytest.raises(ValueError, match="No valid molecules could be parsed"):
            render_molecules_grid(molecular_strings=invalid_molecules)

    def test_grid_legends_mismatch_warning(self):
        """Test that mismatched legends count is handled gracefully."""
        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
            SAMPLE_MOLECULES["smiles"]["acetic_acid"],
        ]

        # Too few legends
        short_legends = ["Ethanol", "Benzene"]  # Missing one

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mismatch_grid.png"

            # Should still render without crashing (legends should be disabled due to mismatch)
            image = render_molecules_grid(
                molecular_strings=molecules,
                legends=short_legends,
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()

            # Test too many legends as well
            long_legends = ["Ethanol", "Benzene", "Acetic Acid", "Extra"]
            output_path2 = Path(temp_dir) / "mismatch_grid2.png"

            image2 = render_molecules_grid(
                molecular_strings=molecules,
                legends=long_legends,
                output_path=output_path2,
            )

            assert image2 is not None
            assert output_path2.exists()

    def test_grid_invalid_configuration(self):
        """Test grid rendering with invalid configuration parameters."""
        molecules = [SAMPLE_MOLECULES["smiles"]["ethanol"]]

        # Test invalid mols_per_row
        with pytest.raises(Exception):  # Should fail during rendering
            render_molecules_grid(
                molecular_strings=molecules,
                mols_per_row=0,  # Invalid
            )

        # Test invalid mol_size
        with pytest.raises(Exception):  # Should fail during rendering
            render_molecules_grid(
                molecular_strings=molecules,
                mol_size=(-100, 200),  # Invalid negative size
            )


class TestMoleculeGridRenderer:
    """Test MoleculeGridRenderer class directly."""

    def test_grid_renderer_initialization(self):
        """Test MoleculeGridRenderer initialization with different parameters."""
        # Default initialization
        renderer = MoleculeGridRenderer()
        assert renderer.mols_per_row == 4
        assert renderer.mol_size == (200, 200)

        # Custom initialization
        custom_config = RenderConfig(width=300, height=300)
        renderer = MoleculeGridRenderer(
            config=custom_config,
            mols_per_row=3,
            mol_size=(250, 250),
        )
        assert renderer.mols_per_row == 3
        assert renderer.mol_size == (250, 250)
        assert renderer.config.width == 300

    def test_grid_renderer_single_molecule(self):
        """Test that grid renderer can render single molecules."""
        from molecular_string_renderer.parsers import SMILESParser

        parser = SMILESParser()
        mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])

        renderer = MoleculeGridRenderer()
        image = renderer.render(mol)

        assert image is not None
        assert image.size[0] > 0
        assert image.size[1] > 0

    def test_grid_renderer_molecule_grid(self):
        """Test MoleculeGridRenderer.render_grid method."""
        from molecular_string_renderer.parsers import SMILESParser

        parser = SMILESParser()
        mols = [
            parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"]),
            parser.parse(SAMPLE_MOLECULES["smiles"]["benzene"]),
            parser.parse(SAMPLE_MOLECULES["smiles"]["acetic_acid"]),
        ]

        renderer = MoleculeGridRenderer(mols_per_row=2, mol_size=(150, 150))

        # Test without legends
        image = renderer.render_grid(mols)
        assert image is not None
        assert image.size[0] > 0
        assert image.size[1] > 0

        # Test with legends
        legends = ["Ethanol", "Benzene", "Acetic Acid"]
        image_with_legends = renderer.render_grid(mols, legends)
        assert image_with_legends is not None
        assert image_with_legends.size[0] > 0
        assert image_with_legends.size[1] > 0

    def test_grid_renderer_error_handling(self):
        """Test error handling in MoleculeGridRenderer."""
        renderer = MoleculeGridRenderer()

        # Test empty molecule list
        with pytest.raises(ValueError, match="Cannot render empty molecule list"):
            renderer.render_grid([])

        # Test with None molecules (should be filtered out by caller)
        # This test verifies renderer behavior with already-parsed molecules
        from molecular_string_renderer.parsers import SMILESParser

        parser = SMILESParser()
        valid_mol = parser.parse(SAMPLE_MOLECULES["smiles"]["ethanol"])

        # Pass valid molecules only (as the filtering should happen before this)
        image = renderer.render_grid([valid_mol])
        assert image is not None


class TestGridRenderingIntegration:
    """Integration tests for grid rendering with other components."""

    def test_grid_rendering_with_all_configurations(self):
        """Test grid rendering with comprehensive configuration combinations."""
        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
        ]

        render_config = RenderConfig(
            width=250,
            height=250,
            background_color="lightgray",
            show_hydrogen=True,
        )

        parser_config = ParserConfig(
            sanitize=True,
            remove_hs=False,  # Keep hydrogens for show_hydrogen
            strict=False,
        )

        output_config = OutputConfig(
            format="png",
            quality=90,
            optimize=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "full_config_grid.png"

            image = render_molecules_grid(
                molecular_strings=molecules,
                format_type="smiles",
                output_format="png",
                output_path=output_path,
                mols_per_row=2,
                mol_size=(250, 250),
                render_config=render_config,
                parser_config=parser_config,
                output_config=output_config,
            )

            assert image is not None
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_grid_rendering_large_grid(self):
        """Test rendering a larger grid of molecules."""
        # Use all available SMILES molecules for a larger grid
        molecules = list(SAMPLE_MOLECULES["smiles"].values())
        legends = list(SAMPLE_MOLECULES["smiles"].keys())

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "large_grid.png"

            image = render_molecules_grid(
                molecular_strings=molecules,
                legends=legends,
                mols_per_row=3,
                mol_size=(180, 180),
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()

            # Verify grid dimensions are reasonable
            num_molecules = len(molecules)
            num_rows = (num_molecules + 2) // 3  # 3 mols per row
            expected_min_width = min(3, num_molecules) * 180
            expected_min_height = num_rows * 180

            assert image.size[0] >= expected_min_width * 0.8
            assert image.size[1] >= expected_min_height * 0.8

    def test_grid_rendering_single_molecule_edge_case(self):
        """Test grid rendering with just one molecule."""
        molecules = [SAMPLE_MOLECULES["smiles"]["ethanol"]]
        legends = ["Ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "single_mol_grid.png"

            image = render_molecules_grid(
                molecular_strings=molecules,
                legends=legends,
                mols_per_row=4,  # More than number of molecules
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()

    def test_grid_rendering_performance_reasonable(self):
        """Test that grid rendering completes in reasonable time."""
        import time

        molecules = [SAMPLE_MOLECULES["smiles"]["ethanol"]] * 12  # 12 copies

        start_time = time.time()

        image = render_molecules_grid(
            molecular_strings=molecules,
            mols_per_row=4,
            mol_size=(100, 100),  # Smaller to speed up
        )

        elapsed_time = time.time() - start_time

        assert image is not None
        # Should complete in reasonable time (less than 30 seconds for 12 molecules)
        assert elapsed_time < 30.0, f"Grid rendering took too long: {elapsed_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__])
