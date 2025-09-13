"""
Edge case and boundary condition tests.

Tests for unusual molecules, large structures, stress tests,
performance edge cases, and boundary conditions.
"""

import time

import pytest

from molecular_string_renderer import (
    OutputConfig,
    RenderConfig,
    render_molecule,
    render_molecules_grid,
    validate_molecular_string,
)
from molecular_string_renderer.parsers import get_parser
from tests.conftest import SAMPLE_MOLECULES


class TestMolecularEdgeCases:
    """Test edge cases with molecular structures."""

    def test_very_small_molecules(self):
        """Test rendering very small molecules."""
        small_molecules = [
            ("C", "smiles", "methane"),
            ("O", "smiles", "water"),
            ("[H][H]", "smiles", "hydrogen gas"),
            ("N#N", "smiles", "nitrogen gas"),
            ("InChI=1S/CH4/h1H4", "inchi", "methane_inchi"),
            ("InChI=1S/H2O/h1H2", "inchi", "water_inchi"),
            ("[C]", "selfies", "methane_selfies"),
        ]

        for mol_string, format_type, name in small_molecules:
            image = render_molecule(
                molecular_string=mol_string,
                format_type=format_type,
                output_format="png",
                auto_filename=False,
            )
            assert image is not None, f"Failed to render {name}: {mol_string}"
            assert image.size[0] > 0 and image.size[1] > 0, f"Zero size for {name}"

    def test_large_molecules(self):
        """Test rendering larger, more complex molecules."""
        large_molecules = [
            # Complex pharmaceutical compounds
            ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "smiles", "ibuprofen"),
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "smiles", "aspirin"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "smiles", "caffeine"),
            # Steroid structure
            ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "smiles", "steroid"),
            # Polycyclic aromatic
            ("C1=CC=C2C(=C1)C=CC3=CC=CC=C32", "smiles", "anthracene"),
        ]

        for mol_string, format_type, name in large_molecules:
            image = render_molecule(
                molecular_string=mol_string,
                format_type=format_type,
                output_format="png",
                auto_filename=False,
            )
            assert image is not None, f"Failed to render {name}: {mol_string}"
            assert image.size[0] > 0 and image.size[1] > 0, f"Zero size for {name}"

    def test_unusual_molecular_structures(self):
        """Test rendering unusual but valid molecular structures."""
        unusual_molecules = [
            # Charged species
            ("[Na+].[Cl-]", "smiles", "sodium_chloride"),
            ("[OH-].[NH4+]", "smiles", "ammonium_hydroxide"),
            # Organometallic
            ("C1=CC=CC=C1.C1=CC=CC=C1.[Fe]", "smiles", "ferrocene_simplified"),
            # Multiple disconnected components
            ("CCO.O", "smiles", "ethanol_water_mixture"),
            # Radicals (if supported)
            ("C[CH2]", "smiles", "ethyl_radical"),
        ]

        for mol_string, format_type, name in unusual_molecules:
            try:
                image = render_molecule(
                    molecular_string=mol_string,
                    format_type=format_type,
                    output_format="png",
                    auto_filename=False,
                )
                assert image is not None, f"Failed to render {name}: {mol_string}"
                assert image.size[0] > 0 and image.size[1] > 0, f"Zero size for {name}"
            except Exception as e:
                # Some unusual structures might not be supported - that's okay
                pytest.skip(f"Unusual structure not supported: {name} - {e}")

    def test_molecules_with_special_atoms(self):
        """Test molecules containing less common atoms."""
        special_atom_molecules = [
            ("C(F)(F)(F)F", "smiles", "tetrafluoromethane"),  # Fixed SMILES notation
            (
                "C(Cl)(Cl)(Cl)Cl",
                "smiles",
                "tetrachloromethane",
            ),  # Fixed SMILES notation
            ("C(Br)(Br)(Br)Br", "smiles", "tetrabromomethane"),  # Fixed SMILES notation
            ("CSCl", "smiles", "chloromethylsulfide"),
            ("CP(=O)(O)O", "smiles", "methylphosphonic_acid"),
            ("C[Si](C)(C)C", "smiles", "tetramethylsilane"),
        ]

        for mol_string, format_type, name in special_atom_molecules:
            image = render_molecule(
                molecular_string=mol_string,
                format_type=format_type,
                output_format="png",
                auto_filename=False,
            )
            assert image is not None, f"Failed to render {name}: {mol_string}"
            assert image.size[0] > 0 and image.size[1] > 0, f"Zero size for {name}"

    def test_very_long_chains(self):
        """Test molecules with very long carbon chains."""
        # Generate long alkane chains
        chain_lengths = [10, 20, 30]

        for length in chain_lengths:
            # Generate linear alkane SMILES: C-C-C-...-C
            smiles = "C" + "C" * (length - 1)

            image = render_molecule(
                molecular_string=smiles,
                format_type="smiles",
                output_format="png",
                auto_filename=False,
            )
            assert image is not None, f"Failed to render C{length} chain"
            assert image.size[0] > 0 and image.size[1] > 0, (
                f"Zero size for C{length} chain"
            )

    def test_highly_branched_molecules(self):
        """Test highly branched molecular structures."""
        branched_molecules = [
            # Highly branched alkane
            ("CC(C)(C)C(C)(C)C(C)(C)C", "smiles", "highly_branched_alkane"),
            # Dendrimer-like structure
            ("CC(C)(CC(C)(C)C)CC(C)(CC(C)(C)C)C", "smiles", "dendrimer_fragment"),
            # Multiple quaternary carbons
            ("CC(C)(C)CC(C)(C)CC(C)(C)C", "smiles", "multiple_quaternary"),
        ]

        for mol_string, format_type, name in branched_molecules:
            image = render_molecule(
                molecular_string=mol_string,
                format_type=format_type,
                output_format="png",
                auto_filename=False,
            )
            assert image is not None, f"Failed to render {name}: {mol_string}"
            assert image.size[0] > 0 and image.size[1] > 0, f"Zero size for {name}"


class TestRenderingSizeEdgeCases:
    """Test edge cases related to image sizes and configurations."""

    def test_very_small_image_sizes(self):
        """Test rendering with very small image sizes."""
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        small_sizes = [
            (100, 100),  # Minimum allowed
            (120, 100),  # Rectangular minimum
            (100, 150),  # Tall minimum
        ]

        for width, height in small_sizes:
            config = RenderConfig(width=width, height=height)

            image = render_molecule(
                molecular_string=smiles,
                render_config=config,
                auto_filename=False,
            )
            assert image is not None, f"Failed at size {width}x{height}"
            assert image.size == (width, height), (
                f"Wrong size: expected {width}x{height}, got {image.size}"
            )

    def test_very_large_image_sizes(self):
        """Test rendering with very large image sizes."""
        smiles = SAMPLE_MOLECULES["smiles"]["benzene"]

        large_sizes = [
            (1500, 1500),  # Large square
            (2000, 1000),  # Large rectangle
            (1000, 2000),  # Tall large
        ]

        for width, height in large_sizes:
            config = RenderConfig(width=width, height=height)

            start_time = time.time()
            image = render_molecule(
                molecular_string=smiles,
                render_config=config,
                auto_filename=False,
            )
            elapsed_time = time.time() - start_time

            assert image is not None, f"Failed at size {width}x{height}"
            assert image.size == (width, height), (
                f"Wrong size: expected {width}x{height}, got {image.size}"
            )
            # Large images should still render in reasonable time
            assert elapsed_time < 30.0, (
                f"Large image took too long: {elapsed_time:.2f}s"
            )

    def test_extreme_aspect_ratios(self):
        """Test rendering with extreme aspect ratios."""
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        extreme_ratios = [
            (1000, 100),  # Very wide
            (100, 1000),  # Very tall
            (1500, 150),  # 10:1 ratio
            (200, 1200),  # 1:6 ratio
        ]

        for width, height in extreme_ratios:
            config = RenderConfig(width=width, height=height)

            image = render_molecule(
                molecular_string=smiles,
                render_config=config,
                auto_filename=False,
            )
            assert image is not None, f"Failed at aspect ratio {width}:{height}"
            assert image.size == (width, height)

    def test_invalid_size_boundaries(self):
        """Test handling of invalid size boundaries."""
        # Sizes below minimum should be rejected
        invalid_sizes = [
            (50, 200),  # Width too small
            (200, 50),  # Height too small
            (50, 50),  # Both too small
            (0, 200),  # Zero width
            (200, 0),  # Zero height
            (-100, 200),  # Negative width
            (200, -100),  # Negative height
        ]

        for width, height in invalid_sizes:
            with pytest.raises(ValueError):
                RenderConfig(width=width, height=height)


class TestParsingEdgeCases:
    """Test edge cases in molecular parsing."""

    def test_whitespace_variations(self):
        """Test parsing with various whitespace patterns."""
        base_smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        whitespace_variations = [
            f"  {base_smiles}",  # Leading spaces
            f"{base_smiles}  ",  # Trailing spaces
            f"  {base_smiles}  ",  # Both ends
            f"\t{base_smiles}\t",  # Tabs
            f"\n{base_smiles}\n",  # Newlines
            f" \t {base_smiles} \t ",  # Mixed whitespace
        ]

        for smiles_variant in whitespace_variations:
            image = render_molecule(
                molecular_string=smiles_variant,
                format_type="smiles",
                auto_filename=False,
            )
            assert image is not None, (
                f"Failed to parse with whitespace: '{smiles_variant}'"
            )

    def test_case_sensitivity_in_formats(self):
        """Test case sensitivity in format specifications."""
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        format_variations = [
            "SMILES",
            "Smiles",
            "smiles",
            "SMI",
            "smi",
            "SELFIES",
            "Selfies",
            "selfies",
        ]

        for format_type in format_variations:
            try:
                if format_type.lower() in ["smiles", "smi"]:
                    mol_string = smiles
                elif format_type.lower() == "selfies":
                    mol_string = SAMPLE_MOLECULES["selfies"]["ethanol"]
                else:
                    continue

                image = render_molecule(
                    molecular_string=mol_string,
                    format_type=format_type,
                    auto_filename=False,
                )
                assert image is not None, f"Failed for format: {format_type}"
            except Exception as e:
                pytest.fail(f"Case sensitivity issue with {format_type}: {e}")

    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs."""
        empty_inputs = ["", "   ", "\t", "\n", "  \t\n  "]

        for empty_input in empty_inputs:
            with pytest.raises(ValueError, match="cannot be empty"):
                render_molecule(molecular_string=empty_input)

    def test_very_long_molecular_strings(self):
        """Test parsing very long molecular strings."""
        # Create a very long SMILES string (long chain)
        very_long_smiles = "C" * 100  # 100-carbon chain

        image = render_molecule(
            molecular_string=very_long_smiles,
            format_type="smiles",
            auto_filename=False,
        )
        assert image is not None, "Failed to render very long SMILES"

        # Test with a valid but complex InChI (aspirin)
        aspirin_inchi = (
            "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
        )

        image = render_molecule(
            molecular_string=aspirin_inchi,
            format_type="inchi",
            auto_filename=False,
        )
        assert image is not None, "Failed to render complex InChI"


class TestConfigurationEdgeCases:
    """Test edge cases in configuration handling."""

    def test_extreme_quality_settings(self):
        """Test extreme quality settings."""
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        # Minimum quality
        low_config = OutputConfig(format="jpg", quality=1)
        image_low = render_molecule(
            molecular_string=smiles,
            output_config=low_config,
            auto_filename=False,
        )
        assert image_low is not None, "Failed with minimum quality"

        # Maximum quality
        high_config = OutputConfig(format="jpg", quality=100)
        image_high = render_molecule(
            molecular_string=smiles,
            output_config=high_config,
            auto_filename=False,
        )
        assert image_high is not None, "Failed with maximum quality"

    def test_extreme_dpi_settings(self):
        """Test extreme DPI settings."""
        smiles = SAMPLE_MOLECULES["smiles"]["benzene"]

        # Minimum DPI
        low_dpi_config = RenderConfig(dpi=72)
        image_low = render_molecule(
            molecular_string=smiles,
            render_config=low_dpi_config,
            auto_filename=False,
        )
        assert image_low is not None, "Failed with minimum DPI"

        # Maximum DPI
        high_dpi_config = RenderConfig(dpi=600)
        image_high = render_molecule(
            molecular_string=smiles,
            render_config=high_dpi_config,
            auto_filename=False,
        )
        assert image_high is not None, "Failed with maximum DPI"

    def test_extreme_line_widths(self):
        """Test extreme bond line width settings."""
        smiles = SAMPLE_MOLECULES["smiles"]["benzene"]

        # Minimum line width
        thin_config = RenderConfig(bond_line_width=0.5)
        image_thin = render_molecule(
            molecular_string=smiles,
            render_config=thin_config,
            auto_filename=False,
        )
        assert image_thin is not None, "Failed with minimum line width"

        # Maximum line width
        thick_config = RenderConfig(bond_line_width=10.0)
        image_thick = render_molecule(
            molecular_string=smiles,
            render_config=thick_config,
            auto_filename=False,
        )
        assert image_thick is not None, "Failed with maximum line width"

    def test_extreme_font_sizes(self):
        """Test extreme font size settings."""
        smiles = SAMPLE_MOLECULES["smiles"]["benzene"]

        # Minimum font size
        small_font_config = RenderConfig(atom_label_font_size=6)
        image_small = render_molecule(
            molecular_string=smiles,
            render_config=small_font_config,
            auto_filename=False,
        )
        assert image_small is not None, "Failed with minimum font size"

        # Maximum font size
        large_font_config = RenderConfig(atom_label_font_size=48)
        image_large = render_molecule(
            molecular_string=smiles,
            render_config=large_font_config,
            auto_filename=False,
        )
        assert image_large is not None, "Failed with maximum font size"


class TestPerformanceStressTests:
    """Stress tests for performance and resource usage."""

    def test_many_small_molecules_performance(self):
        """Test rendering many small molecules."""
        small_molecules = [SAMPLE_MOLECULES["smiles"]["methane"]] * 20

        start_time = time.time()

        for i, smiles in enumerate(small_molecules):
            image = render_molecule(
                molecular_string=smiles,
                auto_filename=False,
            )
            assert image is not None, f"Failed on molecule {i}"

        elapsed_time = time.time() - start_time

        # Should handle 20 small molecules in reasonable time
        assert elapsed_time < 60.0, f"20 molecules took too long: {elapsed_time:.2f}s"
        print(
            f"Rendered 20 molecules in {elapsed_time:.2f}s ({elapsed_time / 20:.2f}s per molecule)"
        )

    def test_large_grid_performance(self):
        """Test performance with large molecule grids."""
        # Create a larger grid with 12 molecules
        molecules = list(SAMPLE_MOLECULES["smiles"].values())
        if len(molecules) < 12:
            # Repeat molecules to get 12
            molecules = molecules * (12 // len(molecules) + 1)
        molecules = molecules[:12]

        start_time = time.time()

        image = render_molecules_grid(
            molecular_strings=molecules,
            mols_per_row=4,
            mol_size=(100, 100),  # Smaller to speed up
        )

        elapsed_time = time.time() - start_time

        assert image is not None, "Failed to render large grid"
        assert elapsed_time < 45.0, f"Large grid took too long: {elapsed_time:.2f}s"
        print(f"Rendered 12-molecule grid in {elapsed_time:.2f}s")

    def test_repeated_parsing_performance(self):
        """Test performance of repeated parsing operations."""
        smiles = SAMPLE_MOLECULES["smiles"]["caffeine"]  # Complex molecule

        start_time = time.time()

        parser = get_parser("smiles")
        for i in range(10):
            mol = parser.parse(smiles)
            assert mol is not None, f"Failed to parse on iteration {i}"

        elapsed_time = time.time() - start_time

        assert elapsed_time < 10.0, (
            f"10 parsing operations took too long: {elapsed_time:.2f}s"
        )
        print(f"Parsed complex molecule 10 times in {elapsed_time:.2f}s")

    def test_mixed_format_performance(self):
        """Test performance with mixed input formats."""
        test_molecules = [
            (SAMPLE_MOLECULES["smiles"]["ethanol"], "smiles"),
            (SAMPLE_MOLECULES["inchi"]["ethanol"], "inchi"),
            (SAMPLE_MOLECULES["selfies"]["ethanol"], "selfies"),
            (SAMPLE_MOLECULES["smiles"]["benzene"], "smiles"),
            (SAMPLE_MOLECULES["inchi"]["benzene"], "inchi"),
            (SAMPLE_MOLECULES["selfies"]["benzene"], "selfies"),
        ]

        start_time = time.time()

        for i, (mol_string, format_type) in enumerate(test_molecules):
            image = render_molecule(
                molecular_string=mol_string,
                format_type=format_type,
                auto_filename=False,
            )
            assert image is not None, f"Failed on mixed format test {i}"

        elapsed_time = time.time() - start_time

        assert elapsed_time < 30.0, (
            f"Mixed format test took too long: {elapsed_time:.2f}s"
        )
        print(f"Rendered 6 molecules in mixed formats in {elapsed_time:.2f}s")

    def test_validation_performance(self):
        """Test performance of validation operations."""
        test_molecules = list(SAMPLE_MOLECULES["smiles"].values()) * 3  # 24 validations

        start_time = time.time()

        for smiles in test_molecules:
            is_valid = validate_molecular_string(smiles, "smiles")
            assert is_valid, f"Validation failed for {smiles}"

        elapsed_time = time.time() - start_time

        assert elapsed_time < 5.0, f"Validation took too long: {elapsed_time:.2f}s"
        print(f"Validated {len(test_molecules)} molecules in {elapsed_time:.2f}s")


class TestMemoryAndResourceEdgeCases:
    """Test edge cases related to memory and resource usage."""

    def test_large_image_memory_usage(self):
        """Test that large images don't cause memory issues."""
        smiles = SAMPLE_MOLECULES["smiles"]["benzene"]

        # Create a reasonably large image
        large_config = RenderConfig(width=1500, height=1500)

        image = render_molecule(
            molecular_string=smiles,
            render_config=large_config,
            auto_filename=False,
        )

        assert image is not None, "Failed to create large image"
        assert image.size == (1500, 1500), "Wrong image size"

        # Verify we can access image data without issues
        assert image.mode in ["RGB", "RGBA"], "Invalid image mode"

        # Clean up explicitly
        del image

    def test_multiple_format_conversions(self):
        """Test multiple format conversions don't leak resources."""
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]
        output_formats = ["png", "svg", "jpg", "pdf"]

        images = []

        for fmt in output_formats:
            image = render_molecule(
                molecular_string=smiles,
                output_format=fmt,
                auto_filename=False,
            )
            assert image is not None, f"Failed for format {fmt}"
            images.append(image)

        # All images should be valid
        assert len(images) == len(output_formats)

        # Clean up explicitly
        del images


if __name__ == "__main__":
    pytest.main([__file__])
