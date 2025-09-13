"""
Comprehensive tests for the MolecularRenderer class OOP interface.

Tests the object-oriented interface including configuration management,
caching behavior, method combinations, and state management.
"""

import tempfile
from pathlib import Path

import pytest

from molecular_string_renderer import (
    MolecularRenderer,
    OutputConfig,
    ParserConfig,
    RenderConfig,
)
from tests.conftest import SAMPLE_MOLECULES


class TestMolecularRendererInitialization:
    """Test MolecularRenderer initialization and configuration."""

    def test_default_initialization(self):
        """Test MolecularRenderer with default configurations."""
        renderer = MolecularRenderer()

        # Check that default configs are created
        assert renderer.render_config is not None
        assert renderer.parser_config is not None
        assert renderer.output_config is not None

        # Check default values
        assert renderer.render_config.width == 500
        assert renderer.render_config.height == 500
        assert renderer.parser_config.sanitize is True
        assert renderer.output_config.format == "png"

        # Check that caches are initialized
        assert renderer._parsers == {}
        assert renderer._renderers == {}
        assert renderer._output_handlers == {}

    def test_custom_initialization(self):
        """Test MolecularRenderer with custom configurations."""
        render_config = RenderConfig(
            width=300,
            height=400,
            background_color="lightblue",
            show_hydrogen=True,
        )

        parser_config = ParserConfig(
            sanitize=False,
            remove_hs=False,
            strict=True,
        )

        output_config = OutputConfig(
            format="svg",
            quality=90,
            optimize=False,
        )

        renderer = MolecularRenderer(
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
        )

        # Check that custom configs are used
        assert renderer.render_config.width == 300
        assert renderer.render_config.height == 400
        assert renderer.render_config.background_color == "lightblue"
        assert renderer.render_config.show_hydrogen is True

        assert renderer.parser_config.sanitize is False
        assert renderer.parser_config.remove_hs is False
        assert renderer.parser_config.strict is True

        assert renderer.output_config.format == "svg"
        assert renderer.output_config.quality == 90
        assert renderer.output_config.optimize is False

    def test_partial_configuration(self):
        """Test MolecularRenderer with partial custom configuration."""
        # Only provide render config, others should use defaults
        render_config = RenderConfig(width=600, height=600)

        renderer = MolecularRenderer(render_config=render_config)

        assert renderer.render_config.width == 600
        assert renderer.render_config.height == 600
        # Other configs should be defaults
        assert renderer.parser_config.sanitize is True
        assert renderer.output_config.format == "png"


class TestMolecularRendererBasicRendering:
    """Test basic rendering functionality of MolecularRenderer."""

    def test_render_single_molecule(self):
        """Test rendering a single molecule."""
        renderer = MolecularRenderer()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "molecule.png"

            image = renderer.render(
                molecular_string=SAMPLE_MOLECULES["smiles"]["ethanol"],
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_render_different_formats(self):
        """Test rendering with different input formats."""
        renderer = MolecularRenderer()

        test_cases = [
            ("smiles", SAMPLE_MOLECULES["smiles"]["ethanol"]),
            ("inchi", SAMPLE_MOLECULES["inchi"]["ethanol"]),
            ("selfies", SAMPLE_MOLECULES["selfies"]["ethanol"]),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for format_type, molecule in test_cases:
                output_path = Path(temp_dir) / f"molecule_{format_type}.png"

                image = renderer.render(
                    molecular_string=molecule,
                    format_type=format_type,
                    output_path=output_path,
                )

                assert image is not None, f"No image for {format_type}"
                assert output_path.exists(), f"No file for {format_type}"

    def test_render_different_output_formats(self):
        """Test rendering to different output formats."""
        renderer = MolecularRenderer()
        output_formats = ["png", "svg", "jpg", "pdf"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in output_formats:
                output_path = Path(temp_dir) / f"molecule.{fmt}"

                image = renderer.render(
                    molecular_string=SAMPLE_MOLECULES["smiles"]["ethanol"],
                    output_format=fmt,
                    output_path=output_path,
                )

                assert image is not None, f"No image for {fmt}"
                assert output_path.exists(), f"No file for {fmt}"

    def test_render_without_output_path(self):
        """Test rendering without saving to file."""
        renderer = MolecularRenderer()

        image = renderer.render(molecular_string=SAMPLE_MOLECULES["smiles"]["ethanol"])

        assert image is not None
        assert hasattr(image, "size")
        assert image.size[0] > 0 and image.size[1] > 0


class TestMolecularRendererGridRendering:
    """Test grid rendering functionality of MolecularRenderer."""

    def test_render_grid_basic(self):
        """Test basic grid rendering."""
        renderer = MolecularRenderer()

        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
            SAMPLE_MOLECULES["smiles"]["acetic_acid"],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grid.png"

            image = renderer.render_grid(
                molecular_strings=molecules,
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_render_grid_with_legends(self):
        """Test grid rendering with legends."""
        renderer = MolecularRenderer()

        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
        ]
        legends = ["Ethanol", "Benzene"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grid_legends.png"

            image = renderer.render_grid(
                molecular_strings=molecules,
                legends=legends,
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()

    def test_render_grid_layout_options(self):
        """Test grid rendering with layout options."""
        renderer = MolecularRenderer()

        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
            SAMPLE_MOLECULES["smiles"]["acetic_acid"],
            SAMPLE_MOLECULES["smiles"]["methane"],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grid_layout.png"

            image = renderer.render_grid(
                molecular_strings=molecules,
                mols_per_row=2,
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()

    def test_render_grid_different_formats(self):
        """Test grid rendering with different input and output formats."""
        renderer = MolecularRenderer()

        molecules = [
            SAMPLE_MOLECULES["inchi"]["ethanol"],
            SAMPLE_MOLECULES["inchi"]["benzene"],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grid_inchi.svg"

            image = renderer.render_grid(
                molecular_strings=molecules,
                format_type="inchi",
                output_format="svg",
                output_path=output_path,
            )

            assert image is not None
            assert output_path.exists()


class TestMolecularRendererConfigurationManagement:
    """Test configuration management and updates."""

    def test_update_config_render(self):
        """Test updating render configuration."""
        renderer = MolecularRenderer()

        # Initial config
        assert renderer.render_config.width == 500

        # Update render config
        new_render_config = RenderConfig(
            width=800,
            height=600,
            background_color="yellow",
        )

        renderer.update_config(render_config=new_render_config)

        assert renderer.render_config.width == 800
        assert renderer.render_config.height == 600
        assert renderer.render_config.background_color == "yellow"

        # Verify caches are cleared
        assert renderer._parsers == {}
        assert renderer._renderers == {}
        assert renderer._output_handlers == {}

    def test_update_config_parser(self):
        """Test updating parser configuration."""
        renderer = MolecularRenderer()

        # Initial config
        assert renderer.parser_config.sanitize is True

        # Update parser config
        new_parser_config = ParserConfig(
            sanitize=False,
            remove_hs=False,
            strict=True,
        )

        renderer.update_config(parser_config=new_parser_config)

        assert renderer.parser_config.sanitize is False
        assert renderer.parser_config.remove_hs is False
        assert renderer.parser_config.strict is True

    def test_update_config_output(self):
        """Test updating output configuration."""
        renderer = MolecularRenderer()

        # Initial config
        assert renderer.output_config.format == "png"

        # Update output config
        new_output_config = OutputConfig(
            format="svg",
            quality=85,
            optimize=False,
        )

        renderer.update_config(output_config=new_output_config)

        assert renderer.output_config.format == "svg"
        assert renderer.output_config.quality == 85
        assert renderer.output_config.optimize is False

    def test_update_config_multiple(self):
        """Test updating multiple configurations at once."""
        renderer = MolecularRenderer()

        new_render_config = RenderConfig(width=300, height=300)
        new_parser_config = ParserConfig(sanitize=False)
        new_output_config = OutputConfig(format="jpg")

        renderer.update_config(
            render_config=new_render_config,
            parser_config=new_parser_config,
            output_config=new_output_config,
        )

        assert renderer.render_config.width == 300
        assert renderer.parser_config.sanitize is False
        assert renderer.output_config.format == "jpg"

    def test_update_config_partial(self):
        """Test updating only some configurations."""
        renderer = MolecularRenderer()

        original_parser_config = renderer.parser_config
        original_output_config = renderer.output_config

        # Update only render config
        new_render_config = RenderConfig(width=400)
        renderer.update_config(render_config=new_render_config)

        assert renderer.render_config.width == 400
        # Other configs should remain the same object
        assert renderer.parser_config is original_parser_config
        assert renderer.output_config is original_output_config

    def test_config_affects_rendering(self):
        """Test that configuration changes affect subsequent rendering."""
        renderer = MolecularRenderer()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Render with default config
            output_path1 = Path(temp_dir) / "default.png"
            image1 = renderer.render(
                molecular_string=SAMPLE_MOLECULES["smiles"]["ethanol"],
                output_path=output_path1,
            )

            # Update config
            new_render_config = RenderConfig(
                width=300,
                height=300,
                background_color="red",
            )
            renderer.update_config(render_config=new_render_config)

            # Render with new config
            output_path2 = Path(temp_dir) / "custom.png"
            image2 = renderer.render(
                molecular_string=SAMPLE_MOLECULES["smiles"]["ethanol"],
                output_path=output_path2,
            )

            # Images should have different sizes
            assert image1.size != image2.size
            assert image2.size == (300, 300)


class TestMolecularRendererStatePersistence:
    """Test state persistence and reuse across multiple operations."""

    def test_multiple_renders_same_config(self):
        """Test multiple renders with the same configuration."""
        config = RenderConfig(
            width=250,
            height=250,
            background_color="lightgreen",
        )

        renderer = MolecularRenderer(render_config=config)

        molecules = [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            SAMPLE_MOLECULES["smiles"]["benzene"],
            SAMPLE_MOLECULES["smiles"]["acetic_acid"],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for i, molecule in enumerate(molecules):
                output_path = Path(temp_dir) / f"molecule_{i}.png"

                image = renderer.render(
                    molecular_string=molecule,
                    output_path=output_path,
                )

                assert image is not None
                assert image.size == (250, 250)
                assert output_path.exists()

    def test_mixed_operations(self):
        """Test mixing single molecule and grid rendering."""
        renderer = MolecularRenderer()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Single molecule render
            single_path = Path(temp_dir) / "single.png"
            single_image = renderer.render(
                molecular_string=SAMPLE_MOLECULES["smiles"]["ethanol"],
                output_path=single_path,
            )

            # Grid render
            molecules = [
                SAMPLE_MOLECULES["smiles"]["benzene"],
                SAMPLE_MOLECULES["smiles"]["acetic_acid"],
            ]
            grid_path = Path(temp_dir) / "grid.png"
            grid_image = renderer.render_grid(
                molecular_strings=molecules,
                output_path=grid_path,
            )

            # Another single molecule render
            single_path2 = Path(temp_dir) / "single2.png"
            single_image2 = renderer.render(
                molecular_string=SAMPLE_MOLECULES["smiles"]["methane"],
                output_path=single_path2,
            )

            assert single_image is not None
            assert grid_image is not None
            assert single_image2 is not None
            assert single_path.exists()
            assert grid_path.exists()
            assert single_path2.exists()

    def test_configuration_consistency(self):
        """Test that configuration remains consistent across operations."""
        custom_config = RenderConfig(
            width=350,
            height=350,
            background_color="purple",
            show_hydrogen=True,
        )

        renderer = MolecularRenderer(render_config=custom_config)

        # Perform multiple operations
        molecules = [SAMPLE_MOLECULES["smiles"]["ethanol"]]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Single render
            single_path = Path(temp_dir) / "single.png"
            single_image = renderer.render(
                molecular_string=molecules[0],
                output_path=single_path,
            )

            # Grid render
            grid_path = Path(temp_dir) / "grid.png"
            renderer.render_grid(
                molecular_strings=molecules,
                output_path=grid_path,
            )

            # Check that configuration is still the same
            assert renderer.render_config.width == 350
            assert renderer.render_config.height == 350
            assert renderer.render_config.background_color == "purple"
            assert renderer.render_config.show_hydrogen is True

            # Check that images respect the configuration
            assert single_image.size == (350, 350)
            # Grid image will be larger due to grid layout, but individual molecules should respect config


class TestMolecularRendererErrorHandling:
    """Test error handling in MolecularRenderer."""

    def test_invalid_molecular_string(self):
        """Test error handling with invalid molecular strings."""
        renderer = MolecularRenderer()

        with pytest.raises(ValueError, match="Invalid SMILES"):
            renderer.render(molecular_string="INVALID_SMILES")

    def test_unsupported_format(self):
        """Test error handling with unsupported format."""
        renderer = MolecularRenderer()

        with pytest.raises(ValueError, match="Unsupported format"):
            renderer.render(
                molecular_string=SAMPLE_MOLECULES["smiles"]["ethanol"],
                format_type="unsupported_format",
            )

    def test_grid_empty_list(self):
        """Test error handling with empty molecule list in grid."""
        renderer = MolecularRenderer()

        with pytest.raises(ValueError, match="Cannot render empty molecule list"):
            renderer.render_grid(molecular_strings=[])

    def test_grid_all_invalid_molecules(self):
        """Test error handling when all molecules in grid are invalid."""
        renderer = MolecularRenderer()

        invalid_molecules = ["INVALID1", "INVALID2", "INVALID3"]

        with pytest.raises(ValueError, match="No valid molecules could be parsed"):
            renderer.render_grid(molecular_strings=invalid_molecules)

    def test_invalid_output_path(self):
        """Test error handling with invalid output path."""
        renderer = MolecularRenderer()

        # This should not raise an error during render() call
        # but might fail during file writing
        # The exact behavior depends on the OS and permissions
        try:
            image = renderer.render(
                molecular_string=SAMPLE_MOLECULES["smiles"]["ethanol"],
                output_path="/nonexistent/path/file.png",
            )
            # If it succeeds, we should at least get an image object
            assert image is not None
        except (OSError, IOError, PermissionError):
            # This is expected for invalid paths
            pass


class TestMolecularRendererPerformance:
    """Test performance-related aspects of MolecularRenderer."""

    def test_repeated_operations_performance(self):
        """Test that repeated operations are reasonably fast."""
        import time

        renderer = MolecularRenderer()

        molecules = [SAMPLE_MOLECULES["smiles"]["ethanol"]] * 5

        start_time = time.time()

        # Render multiple molecules
        for i, molecule in enumerate(molecules):
            image = renderer.render(molecular_string=molecule)
            assert image is not None

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (less than 15 seconds for 5 molecules)
        assert elapsed_time < 15.0, (
            f"Repeated operations took too long: {elapsed_time:.2f}s"
        )

    def test_config_update_efficiency(self):
        """Test that configuration updates are efficient."""
        import time

        renderer = MolecularRenderer()

        # Time config updates
        start_time = time.time()

        for i in range(10):
            new_config = RenderConfig(width=200 + i * 10, height=200 + i * 10)
            renderer.update_config(render_config=new_config)

        elapsed_time = time.time() - start_time

        # Config updates should be very fast
        assert elapsed_time < 1.0, f"Config updates took too long: {elapsed_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__])
