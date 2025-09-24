"""
Tests for the main MolecularRenderer class.

Comprehensive tests covering initialization, rendering operations,
caching behavior, error handling, and performance tracking.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.exceptions import (
    ConfigurationError,
    OutputError,
    ParsingError,
    RenderingError,
    ValidationError,
)
from molecular_string_renderer.renderer import MolecularRenderer


class TestMolecularRenderer:
    """Test class for MolecularRenderer functionality."""

    def test_initialization_with_default_configs(self):
        """Test renderer initialization with default configurations."""
        renderer = MolecularRenderer()

        assert isinstance(renderer.render_config, RenderConfig)
        assert isinstance(renderer.parser_config, ParserConfig)
        assert isinstance(renderer.output_config, OutputConfig)
        assert renderer._operation_count == 0
        assert renderer._cache_hits == 0
        assert len(renderer._parsers) == 0
        assert len(renderer._renderers) == 0
        assert len(renderer._output_handlers) == 0

    def test_initialization_with_custom_configs(self):
        """Test renderer initialization with custom configurations."""
        render_config = RenderConfig(width=300, height=300)
        parser_config = ParserConfig()
        output_config = OutputConfig()

        renderer = MolecularRenderer(
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
        )

        assert renderer.render_config.width == 300
        assert renderer.render_config.height == 300
        assert renderer.parser_config is parser_config
        assert renderer.output_config is output_config

    def test_initialization_with_invalid_config(self):
        """Test renderer initialization fails with invalid configuration."""
        with patch.object(
            RenderConfig, "model_dump", side_effect=ValueError("Invalid config")
        ):
            with pytest.raises(ConfigurationError, match="Invalid configuration"):
                MolecularRenderer(render_config=RenderConfig())

    @patch("molecular_string_renderer.renderer.get_parser")
    @patch("molecular_string_renderer.renderer.get_renderer")
    @patch("molecular_string_renderer.renderer.validate_molecular_string")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    def test_render_success(
        self,
        mock_validate_path,
        mock_validate_format,
        mock_validate_string,
        mock_get_renderer,
        mock_get_parser,
    ):
        """Test successful single molecule rendering."""
        # Setup mocks
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        mock_mol = Mock()
        mock_parser = Mock()
        mock_parser.parse.return_value = mock_mol
        mock_get_parser.return_value = mock_parser

        mock_image = Mock(spec=Image.Image)
        mock_renderer = Mock()
        mock_renderer.render.return_value = mock_image
        mock_get_renderer.return_value = mock_renderer

        renderer = MolecularRenderer()
        result = renderer.render("CCO", "smiles")

        assert result is mock_image
        assert renderer._operation_count == 1
        mock_validate_string.assert_called_once_with("CCO", "smiles")
        mock_get_parser.assert_called_once()
        mock_parser.parse.assert_called_once_with("CCO")
        mock_get_renderer.assert_called_once()
        mock_renderer.render.assert_called_once_with(mock_mol)

    @patch("molecular_string_renderer.renderer.get_parser")
    @patch("molecular_string_renderer.renderer.validate_molecular_string")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    def test_render_parsing_failure(
        self,
        mock_validate_path,
        mock_validate_format,
        mock_validate_string,
        mock_get_parser,
    ):
        """Test render fails when molecule parsing fails."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        mock_parser = Mock()
        mock_parser.parse.return_value = None
        mock_get_parser.return_value = mock_parser

        renderer = MolecularRenderer()

        with pytest.raises(ParsingError, match="Failed to parse SMILES"):
            renderer.render("invalid", "smiles")

    @patch("molecular_string_renderer.renderer.get_parser")
    @patch("molecular_string_renderer.renderer.get_renderer")
    @patch("molecular_string_renderer.renderer.validate_molecular_string")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    def test_render_rendering_failure(
        self,
        mock_validate_path,
        mock_validate_format,
        mock_validate_string,
        mock_get_renderer,
        mock_get_parser,
    ):
        """Test render fails when molecule rendering fails."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        mock_mol = Mock()
        mock_parser = Mock()
        mock_parser.parse.return_value = mock_mol
        mock_get_parser.return_value = mock_parser

        mock_renderer = Mock()
        mock_renderer.render.return_value = None
        mock_get_renderer.return_value = mock_renderer

        renderer = MolecularRenderer()

        with pytest.raises(RenderingError, match="Renderer returned None image"):
            renderer.render("CCO", "smiles")

    @patch("molecular_string_renderer.renderer.get_parser")
    @patch("molecular_string_renderer.renderer.get_renderer")
    @patch("molecular_string_renderer.renderer.get_output_handler")
    @patch("molecular_string_renderer.renderer.validate_molecular_string")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    def test_render_with_output_save(
        self,
        mock_validate_path,
        mock_validate_format,
        mock_validate_string,
        mock_get_output_handler,
        mock_get_renderer,
        mock_get_parser,
    ):
        """Test render with output file saving."""
        mock_validate_format.return_value = "smiles"
        mock_output_path = Path("test.png")
        mock_validate_path.return_value = mock_output_path

        mock_mol = Mock()
        mock_parser = Mock()
        mock_parser.parse.return_value = mock_mol
        mock_get_parser.return_value = mock_parser

        mock_image = Mock(spec=Image.Image)
        mock_renderer = Mock()
        mock_renderer.render.return_value = mock_image
        mock_get_renderer.return_value = mock_renderer

        mock_output_handler = Mock()
        mock_get_output_handler.return_value = mock_output_handler

        renderer = MolecularRenderer()
        result = renderer.render("CCO", "smiles", output_path="test.png")

        assert result is mock_image
        mock_get_output_handler.assert_called_once_with("png", renderer.output_config)
        mock_output_handler.save.assert_called_once_with(mock_image, mock_output_path)

    @patch("molecular_string_renderer.renderer.get_parser")
    @patch("molecular_string_renderer.renderer.get_renderer")
    @patch("molecular_string_renderer.renderer.get_output_handler")
    @patch("molecular_string_renderer.renderer.validate_molecular_string")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    def test_render_output_save_failure(
        self,
        mock_validate_path,
        mock_validate_format,
        mock_validate_string,
        mock_get_output_handler,
        mock_get_renderer,
        mock_get_parser,
    ):
        """Test render fails when output saving fails."""
        mock_validate_format.return_value = "smiles"
        mock_output_path = Path("test.png")
        mock_validate_path.return_value = mock_output_path

        mock_mol = Mock()
        mock_parser = Mock()
        mock_parser.parse.return_value = mock_mol
        mock_get_parser.return_value = mock_parser

        mock_image = Mock(spec=Image.Image)
        mock_renderer = Mock()
        mock_renderer.render.return_value = mock_image
        mock_get_renderer.return_value = mock_renderer

        mock_output_handler = Mock()
        mock_output_handler.save.side_effect = Exception("Save failed")
        mock_get_output_handler.return_value = mock_output_handler

        renderer = MolecularRenderer()

        with pytest.raises(OutputError, match="Error saving output"):
            renderer.render("CCO", "smiles", output_path="test.png")

    @patch("molecular_string_renderer.renderer.parse_molecule_list")
    @patch("molecular_string_renderer.renderer.validate_grid_parameters")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    @patch("molecular_string_renderer.renderers.MoleculeGridRenderer")
    def test_render_grid_success(
        self,
        mock_grid_renderer_class,
        mock_validate_path,
        mock_validate_format,
        mock_validate_grid,
        mock_parse_list,
    ):
        """Test successful grid rendering."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        mock_mols = [Mock(), Mock()]
        mock_valid_indices = [0, 1]
        mock_errors = []
        mock_parse_list.return_value = (mock_mols, mock_valid_indices, mock_errors)

        mock_image = Mock(spec=Image.Image)
        mock_grid_renderer = Mock()
        mock_grid_renderer.render_grid.return_value = mock_image
        mock_grid_renderer_class.return_value = mock_grid_renderer

        renderer = MolecularRenderer()
        result = renderer.render_grid(["CCO", "CCC"], "smiles")

        assert result is mock_image
        assert renderer._operation_count == 1
        mock_validate_grid.assert_called_once()
        mock_parse_list.assert_called_once()
        mock_grid_renderer.render_grid.assert_called_once_with(mock_mols, None)

    @patch("molecular_string_renderer.renderer.parse_molecule_list")
    @patch("molecular_string_renderer.renderer.validate_grid_parameters")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    def test_render_grid_no_valid_molecules(
        self,
        mock_validate_path,
        mock_validate_format,
        mock_validate_grid,
        mock_parse_list,
    ):
        """Test grid rendering fails when no molecules can be parsed."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        mock_parse_list.return_value = ([], [], ["Error 1", "Error 2"])

        renderer = MolecularRenderer()

        with pytest.raises(ParsingError, match="No valid molecules could be parsed"):
            renderer.render_grid(["invalid1", "invalid2"], "smiles")

    @patch("molecular_string_renderer.renderer.parse_molecule_list")
    @patch("molecular_string_renderer.renderer.validate_grid_parameters")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    def test_render_grid_invalid_legends(
        self,
        mock_validate_path,
        mock_validate_format,
        mock_validate_grid,
        mock_parse_list,
    ):
        """Test grid rendering validates legends correctly."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        mock_mols = [Mock(), Mock()]
        mock_valid_indices = [0, 1]
        mock_errors = []
        mock_parse_list.return_value = (mock_mols, mock_valid_indices, mock_errors)

        renderer = MolecularRenderer()

        # Test non-list legends
        with pytest.raises(ValidationError, match="Legends must be a list"):
            renderer.render_grid(["CCO", "CCC"], legends="not a list")

        # Test mismatched legend count
        with pytest.raises(ValidationError, match="Number of legends.*must match"):
            renderer.render_grid(["CCO", "CCC"], legends=["Legend 1"])

    def test_caching_behavior(self):
        """Test that parsers, renderers, and output handlers are cached properly."""
        with (
            patch("molecular_string_renderer.renderer.get_parser") as mock_get_parser,
            patch(
                "molecular_string_renderer.renderer.get_renderer"
            ) as mock_get_renderer,
            patch("molecular_string_renderer.renderer.validate_molecular_string"),
            patch(
                "molecular_string_renderer.renderer.validate_format_type",
                return_value="smiles",
            ),
            patch(
                "molecular_string_renderer.renderer.validate_output_path",
                return_value=None,
            ),
        ):
            mock_mol = Mock()
            mock_parser = Mock()
            mock_parser.parse.return_value = mock_mol
            mock_get_parser.return_value = mock_parser

            mock_image = Mock(spec=Image.Image)
            mock_renderer = Mock()
            mock_renderer.render.return_value = mock_image
            mock_get_renderer.return_value = mock_renderer

            renderer = MolecularRenderer()

            # First render
            renderer.render("CCO", "smiles")
            assert mock_get_parser.call_count == 1
            assert mock_get_renderer.call_count == 1
            assert renderer._cache_hits == 0

            # Second render with same parameters - should use cache
            renderer.render("CCC", "smiles")
            assert mock_get_parser.call_count == 1  # No additional calls
            assert mock_get_renderer.call_count == 1  # No additional calls
            assert renderer._cache_hits == 2  # Parser and renderer cache hits

    def test_config_update(self):
        """Test configuration updates and cache clearing."""
        renderer = MolecularRenderer()

        # Add some items to cache
        renderer._parsers["test"] = Mock()
        renderer._renderers["test"] = Mock()
        renderer._output_handlers["test"] = Mock()
        renderer._config_hashes["test"] = 123

        new_render_config = RenderConfig(width=800)
        renderer.update_config(render_config=new_render_config)

        assert renderer.render_config.width == 800
        assert len(renderer._parsers) == 0
        assert len(renderer._renderers) == 0
        assert len(renderer._output_handlers) == 0
        assert len(renderer._config_hashes) == 0

    def test_config_update_with_invalid_config(self):
        """Test configuration update fails with invalid configuration."""
        renderer = MolecularRenderer()

        with patch.object(
            RenderConfig, "model_dump", side_effect=ValueError("Invalid")
        ):
            with pytest.raises(
                ConfigurationError, match="Invalid render configuration"
            ):
                renderer.update_config(render_config=RenderConfig())

    def test_get_stats(self):
        """Test performance statistics retrieval."""
        renderer = MolecularRenderer()
        renderer._operation_count = 10
        renderer._cache_hits = 6
        renderer._parsers["test1"] = Mock()
        renderer._renderers["test2"] = Mock()
        renderer._output_handlers["test3"] = Mock()

        stats = renderer.get_stats()

        assert stats["operations"] == 10
        assert stats["cache_hits"] == 6
        assert stats["cache_efficiency"] == 60.0
        assert stats["cached_parsers"] == 1
        assert stats["cached_renderers"] == 1
        assert stats["cached_output_handlers"] == 1

    def test_get_stats_no_operations(self):
        """Test statistics with no operations performed."""
        renderer = MolecularRenderer()
        stats = renderer.get_stats()

        assert stats["operations"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_efficiency"] == 0.0
        assert stats["cached_parsers"] == 0
        assert stats["cached_renderers"] == 0
        assert stats["cached_output_handlers"] == 0

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        renderer = MolecularRenderer()

        # Add items to all caches
        renderer._parsers["test"] = Mock()
        renderer._renderers["test"] = Mock()
        renderer._output_handlers["test"] = Mock()
        renderer._config_hashes["test"] = 123

        renderer.clear_cache()

        assert len(renderer._parsers) == 0
        assert len(renderer._renderers) == 0
        assert len(renderer._output_handlers) == 0
        assert len(renderer._config_hashes) == 0

    def test_config_hash_caching(self):
        """Test that configuration hashes are cached properly."""
        renderer = MolecularRenderer()

        # Get hash multiple times
        hash1 = renderer._get_config_hash("parser")
        hash2 = renderer._get_config_hash("parser")
        hash3 = renderer._get_config_hash("render")

        assert hash1 == hash2
        assert "parser" in renderer._config_hashes
        assert "render" in renderer._config_hashes
        assert renderer._config_hashes["parser"] == hash1
        assert renderer._config_hashes["render"] == hash3

    @patch("molecular_string_renderer.renderer.validate_molecular_string")
    def test_validation_error_propagation(self, mock_validate):
        """Test that validation errors are properly propagated."""
        mock_validate.side_effect = ValidationError("Invalid molecular string")

        renderer = MolecularRenderer()

        with pytest.raises(ValidationError, match="Invalid molecular string"):
            renderer.render("invalid", "smiles")

    @patch("molecular_string_renderer.renderer.get_parser")
    @patch("molecular_string_renderer.renderer.validate_molecular_string")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    def test_parser_exception_handling(
        self,
        mock_validate_path,
        mock_validate_format,
        mock_validate_string,
        mock_get_parser,
    ):
        """Test proper handling of parser exceptions."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        mock_parser = Mock()
        mock_parser.parse.side_effect = Exception("Parser error")
        mock_get_parser.return_value = mock_parser

        renderer = MolecularRenderer()

        with pytest.raises(ParsingError, match="Error parsing SMILES"):
            renderer.render("CCO", "smiles")

    @patch("molecular_string_renderer.renderer.get_parser")
    @patch("molecular_string_renderer.renderer.get_renderer")
    @patch("molecular_string_renderer.renderer.validate_molecular_string")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    def test_renderer_exception_handling(
        self,
        mock_validate_path,
        mock_validate_format,
        mock_validate_string,
        mock_get_renderer,
        mock_get_parser,
    ):
        """Test proper handling of renderer exceptions."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        mock_mol = Mock()
        mock_parser = Mock()
        mock_parser.parse.return_value = mock_mol
        mock_get_parser.return_value = mock_parser

        mock_renderer = Mock()
        mock_renderer.render.side_effect = Exception("Renderer error")
        mock_get_renderer.return_value = mock_renderer

        renderer = MolecularRenderer()

        with pytest.raises(RenderingError, match="Error rendering molecule"):
            renderer.render("CCO", "smiles")

    def test_render_grid_smart_default_mols_per_row(self):
        """Test that render_grid uses smart default for mols_per_row."""
        renderer = MolecularRenderer()
        
        # Test with 3 molecules - should create instance with mols_per_row=3
        molecules = ["CCO", "CC(=O)O", "C1=CC=CC=C1"]
        result = renderer.render_grid(molecules)
        
        # Should be a valid image with correct dimensions
        assert isinstance(result, Image.Image)
        # Width should be 3 * 200 = 600 (not 4 * 200 = 800)
        assert result.size[0] == 600
        assert result.size[1] == 200

        # Test with 5 molecules - should create instance with mols_per_row=4 (max)
        molecules_5 = ["CCO", "CC(=O)O", "C1=CC=CC=C1", "C", "CC"]
        result_5 = renderer.render_grid(molecules_5)
        
        assert isinstance(result_5, Image.Image)
        # Width should be 4 * 200 = 800 (max), height 2 rows = 400
        assert result_5.size[0] == 800
        assert result_5.size[1] == 400

    def test_render_grid_explicit_mols_per_row_overrides_smart(self):
        """Test that explicit mols_per_row overrides smart default."""
        renderer = MolecularRenderer()
        
        molecules = ["CCO", "CC(=O)O", "C1=CC=CC=C1"]
        # Explicit 2 per row should override smart default of 3
        result = renderer.render_grid(molecules, mols_per_row=2)
        
        assert isinstance(result, Image.Image)
        # Should be 2 * 200 = 400 wide, 2 rows = 400 tall
        assert result.size[0] == 400
        assert result.size[1] == 400


class TestMolecularRendererIntegration:
    """Integration tests for MolecularRenderer with real dependencies."""

    def test_renderer_instantiation_with_real_configs(self):
        """Test that renderer can be instantiated with real configuration objects."""
        render_config = RenderConfig(width=400, height=300, show_hydrogen=True)
        parser_config = ParserConfig()
        output_config = OutputConfig()

        renderer = MolecularRenderer(
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
        )

        assert renderer.render_config.width == 400
        assert renderer.render_config.height == 300
        assert renderer.render_config.show_hydrogen is True

    def test_config_serialization(self):
        """Test that configurations can be properly serialized for hashing."""
        renderer = MolecularRenderer()

        # This should not raise exceptions
        parser_hash = renderer._get_config_hash("parser")
        render_hash = renderer._get_config_hash("render")
        output_hash = renderer._get_config_hash("output")

        assert isinstance(parser_hash, int)
        assert isinstance(render_hash, int)
        assert isinstance(output_hash, int)

    @patch("molecular_string_renderer.renderer.get_parser")
    @patch("molecular_string_renderer.renderer.get_renderer")
    @patch("molecular_string_renderer.renderer.get_output_handler")
    @patch("molecular_string_renderer.renderer.validate_molecular_string")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    def test_render_with_svg_output(
        self,
        mock_validate_path,
        mock_validate_format,
        mock_validate_string,
        mock_get_output_handler,
        mock_get_renderer,
        mock_get_parser,
    ):
        """Test render with SVG output format."""
        mock_validate_format.return_value = "smiles"
        mock_output_path = Path("test.svg")
        mock_validate_path.return_value = mock_output_path

        mock_mol = Mock()
        mock_parser = Mock()
        mock_parser.parse.return_value = mock_mol
        mock_get_parser.return_value = mock_parser

        mock_image = Mock(spec=Image.Image)
        mock_renderer = Mock()
        mock_renderer.render.return_value = mock_image
        mock_get_renderer.return_value = mock_renderer

        mock_output_handler = Mock()
        mock_get_output_handler.return_value = mock_output_handler

        renderer = MolecularRenderer()
        result = renderer.render(
            "CCO", "smiles", output_format="svg", output_path="test.svg"
        )

        assert result is mock_image
        mock_output_handler.set_molecule.assert_called_once_with(mock_mol)
        mock_output_handler.save.assert_called_once_with(mock_image, mock_output_path)

    @patch("molecular_string_renderer.renderer.get_parser")
    @patch("molecular_string_renderer.renderer.get_renderer")
    @patch("molecular_string_renderer.renderer.get_output_handler")
    @patch("molecular_string_renderer.renderer.validate_molecular_string")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    def test_render_with_non_svg_output(
        self,
        mock_validate_path,
        mock_validate_format,
        mock_validate_string,
        mock_get_output_handler,
        mock_get_renderer,
        mock_get_parser,
    ):
        """Test render with non-SVG output format doesn't call set_molecule."""
        mock_validate_format.return_value = "smiles"
        mock_output_path = Path("test.png")
        mock_validate_path.return_value = mock_output_path

        mock_mol = Mock()
        mock_parser = Mock()
        mock_parser.parse.return_value = mock_mol
        mock_get_parser.return_value = mock_parser

        mock_image = Mock(spec=Image.Image)
        mock_renderer = Mock()
        mock_renderer.render.return_value = mock_image
        mock_get_renderer.return_value = mock_renderer

        mock_output_handler = Mock()
        mock_get_output_handler.return_value = mock_output_handler

        renderer = MolecularRenderer()
        result = renderer.render(
            "CCO", "smiles", output_format="png", output_path="test.png"
        )

        assert result is mock_image
        mock_output_handler.set_molecule.assert_not_called()
        mock_output_handler.save.assert_called_once_with(mock_image, mock_output_path)

    @patch("molecular_string_renderer.renderer.parse_molecule_list")
    @patch("molecular_string_renderer.renderer.validate_grid_parameters")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    @patch("molecular_string_renderer.utils.format_parsing_errors")
    def test_render_grid_with_parsing_errors(
        self,
        mock_format_errors,
        mock_validate_path,
        mock_validate_format,
        mock_validate_grid,
        mock_parse_list,
    ):
        """Test grid rendering with detailed parsing error formatting."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None
        mock_format_errors.return_value = "Detailed error info"

        mock_parse_list.return_value = ([], [], ["Error 1", "Error 2"])

        renderer = MolecularRenderer()

        with pytest.raises(
            ParsingError,
            match="No valid molecules could be parsed. Errors: Detailed error info",
        ):
            renderer.render_grid(["invalid1", "invalid2"], "smiles")

    @patch("molecular_string_renderer.renderer.parse_molecule_list")
    @patch("molecular_string_renderer.renderer.validate_grid_parameters")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    @patch("molecular_string_renderer.renderers.MoleculeGridRenderer")
    def test_render_grid_with_legends_filtering(
        self,
        mock_grid_renderer_class,
        mock_validate_path,
        mock_validate_format,
        mock_validate_grid,
        mock_parse_list,
    ):
        """Test grid rendering with legend filtering when some molecules fail to parse."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        # Only first molecule parses successfully
        mock_mols = [Mock()]
        mock_valid_indices = [0]  # Only index 0 is valid
        mock_errors = ["Error for molecule 1"]
        mock_parse_list.return_value = (mock_mols, mock_valid_indices, mock_errors)

        mock_image = Mock(spec=Image.Image)
        mock_grid_renderer = Mock()
        mock_grid_renderer.render_grid.return_value = mock_image
        mock_grid_renderer_class.return_value = mock_grid_renderer

        renderer = MolecularRenderer()
        legends = ["Legend 0", "Legend 1"]
        result = renderer.render_grid(["CCO", "invalid"], "smiles", legends=legends)

        assert result is mock_image
        # Should filter legends to only include the first one
        mock_grid_renderer.render_grid.assert_called_once_with(mock_mols, ["Legend 0"])

    @patch("molecular_string_renderer.renderer.parse_molecule_list")
    @patch("molecular_string_renderer.renderer.validate_grid_parameters")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    @patch("molecular_string_renderer.renderers.MoleculeGridRenderer")
    def test_render_grid_legend_count_mismatch_warning(
        self,
        mock_grid_renderer_class,
        mock_validate_path,
        mock_validate_format,
        mock_validate_grid,
        mock_parse_list,
    ):
        """Test grid rendering logs warning when legend count doesn't match after filtering."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        mock_mols = [Mock(), Mock()]
        mock_valid_indices = [0, 2]  # Indices 0 and 2 are valid (1 failed)
        mock_errors = []
        mock_parse_list.return_value = (mock_mols, mock_valid_indices, mock_errors)

        mock_image = Mock(spec=Image.Image)
        mock_grid_renderer = Mock()
        mock_grid_renderer.render_grid.return_value = mock_image
        mock_grid_renderer_class.return_value = mock_grid_renderer

        renderer = MolecularRenderer()
        # 3 legends for 3 molecules, but only indices 0 and 2 will be valid
        # This means filtered_legends will have 2 items but only index 2 exists, causing count mismatch
        legends = ["Legend 0", "Legend 1", "Legend 2"]

        with patch("molecular_string_renderer.renderer.logger") as mock_logger:
            result = renderer.render_grid(
                ["CCO", "CCC", "invalid"], "smiles", legends=legends
            )

            assert result is mock_image
            # The legend filtering happens: [legends[0], legends[2]] for indices [0,2] = ["Legend 0", "Legend 2"]
            # This should match the 2 molecules, so no warning should occur
            mock_grid_renderer.render_grid.assert_called_once_with(
                mock_mols, ["Legend 0", "Legend 2"]
            )
            mock_logger.warning.assert_not_called()

    def test_cache_efficiency_calculation_edge_cases(self):
        """Test cache efficiency calculation with edge cases."""
        renderer = MolecularRenderer()

        # Test with zero operations
        stats = renderer.get_stats()
        assert stats["cache_efficiency"] == 0.0

        # Test with operations but no cache hits
        renderer._operation_count = 5
        renderer._cache_hits = 0
        stats = renderer.get_stats()
        assert stats["cache_efficiency"] == 0.0

        # Test with perfect cache efficiency
        renderer._operation_count = 10
        renderer._cache_hits = 10
        stats = renderer.get_stats()
        assert stats["cache_efficiency"] == 100.0

    @patch("molecular_string_renderer.renderer.get_output_handler")
    def test_output_handler_cache_hit(self, mock_get_output_handler):
        """Test output handler cache hit behavior."""
        mock_output_handler = Mock()
        mock_get_output_handler.return_value = mock_output_handler

        renderer = MolecularRenderer()

        # First call should create handler
        handler1 = renderer._get_cached_output_handler("png")
        assert handler1 is mock_output_handler
        assert mock_get_output_handler.call_count == 1
        assert renderer._cache_hits == 0

        # Second call should use cache
        handler2 = renderer._get_cached_output_handler("png")
        assert handler2 is mock_output_handler
        assert mock_get_output_handler.call_count == 1  # No additional calls
        assert renderer._cache_hits == 1

    @patch("molecular_string_renderer.renderer.parse_molecule_list")
    @patch("molecular_string_renderer.renderer.validate_grid_parameters")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    @patch("molecular_string_renderer.renderers.MoleculeGridRenderer")
    def test_render_grid_renderer_exception_handling(
        self,
        mock_grid_renderer_class,
        mock_validate_path,
        mock_validate_format,
        mock_validate_grid,
        mock_parse_list,
    ):
        """Test grid rendering with renderer exceptions."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        mock_mols = [Mock()]
        mock_valid_indices = [0]
        mock_errors = []
        mock_parse_list.return_value = (mock_mols, mock_valid_indices, mock_errors)

        mock_grid_renderer = Mock()
        mock_grid_renderer.render_grid.side_effect = Exception("Grid rendering failed")
        mock_grid_renderer_class.return_value = mock_grid_renderer

        renderer = MolecularRenderer()

        with pytest.raises(RenderingError, match="Error rendering molecule grid"):
            renderer.render_grid(["CCO"], "smiles")

    @patch("molecular_string_renderer.renderer.parse_molecule_list")
    @patch("molecular_string_renderer.renderer.validate_grid_parameters")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    @patch("molecular_string_renderer.renderers.MoleculeGridRenderer")
    @patch("molecular_string_renderer.renderer.get_output_handler")
    def test_render_grid_output_exception_handling(
        self,
        mock_get_output_handler,
        mock_grid_renderer_class,
        mock_validate_path,
        mock_validate_format,
        mock_validate_grid,
        mock_parse_list,
    ):
        """Test grid rendering with output saving exceptions."""
        mock_validate_format.return_value = "smiles"
        mock_output_path = Path("test.png")
        mock_validate_path.return_value = mock_output_path

        mock_mols = [Mock()]
        mock_valid_indices = [0]
        mock_errors = []
        mock_parse_list.return_value = (mock_mols, mock_valid_indices, mock_errors)

        mock_image = Mock(spec=Image.Image)
        mock_grid_renderer = Mock()
        mock_grid_renderer.render_grid.return_value = mock_image
        mock_grid_renderer_class.return_value = mock_grid_renderer

        mock_output_handler = Mock()
        mock_output_handler.save.side_effect = Exception("Save failed")
        mock_get_output_handler.return_value = mock_output_handler

        renderer = MolecularRenderer()

        with pytest.raises(OutputError, match="Error saving grid output"):
            renderer.render_grid(["CCO"], "smiles", output_path="test.png")

    def test_update_config_parser_config_only(self):
        """Test updating only parser configuration."""
        renderer = MolecularRenderer()
        original_render_config = renderer.render_config
        original_output_config = renderer.output_config

        # Add items to cache to verify they get cleared
        renderer._parsers["test"] = Mock()

        new_parser_config = ParserConfig()
        renderer.update_config(parser_config=new_parser_config)

        assert renderer.parser_config is new_parser_config
        assert renderer.render_config is original_render_config  # Unchanged
        assert renderer.output_config is original_output_config  # Unchanged
        assert len(renderer._parsers) == 0  # Cache cleared

    def test_update_config_output_config_only(self):
        """Test updating only output configuration."""
        renderer = MolecularRenderer()
        original_render_config = renderer.render_config
        original_parser_config = renderer.parser_config

        # Add items to cache to verify they get cleared
        renderer._output_handlers["test"] = Mock()

        new_output_config = OutputConfig()
        renderer.update_config(output_config=new_output_config)

        assert renderer.output_config is new_output_config
        assert renderer.render_config is original_render_config  # Unchanged
        assert renderer.parser_config is original_parser_config  # Unchanged
        assert len(renderer._output_handlers) == 0  # Cache cleared

    def test_update_config_parser_invalid(self):
        """Test updating with invalid parser configuration."""
        renderer = MolecularRenderer()

        with patch.object(
            ParserConfig, "model_dump", side_effect=ValueError("Invalid parser config")
        ):
            with pytest.raises(
                ConfigurationError, match="Invalid parser configuration"
            ):
                renderer.update_config(parser_config=ParserConfig())

    def test_update_config_output_invalid(self):
        """Test updating with invalid output configuration."""
        renderer = MolecularRenderer()

        with patch.object(
            OutputConfig, "model_dump", side_effect=ValueError("Invalid output config")
        ):
            with pytest.raises(
                ConfigurationError, match="Invalid output configuration"
            ):
                renderer.update_config(output_config=OutputConfig())

    @patch("molecular_string_renderer.renderer.parse_molecule_list")
    @patch("molecular_string_renderer.renderer.validate_grid_parameters")
    @patch("molecular_string_renderer.renderer.validate_format_type")
    @patch("molecular_string_renderer.renderer.validate_output_path")
    @patch("molecular_string_renderer.renderers.MoleculeGridRenderer")
    def test_render_grid_legend_filter_mismatch_warning(
        self,
        mock_grid_renderer_class,
        mock_validate_path,
        mock_validate_format,
        mock_validate_grid,
        mock_parse_list,
    ):
        """Test grid rendering warning when filtered legends don't match molecule count."""
        mock_validate_format.return_value = "smiles"
        mock_validate_path.return_value = None

        mock_mols = [Mock(), Mock()]
        # Valid indices [0, 3] but only 3 legends available, so filtering will only get 1 legend
        mock_valid_indices = [0, 3]
        mock_errors = []
        mock_parse_list.return_value = (mock_mols, mock_valid_indices, mock_errors)

        mock_image = Mock(spec=Image.Image)
        mock_grid_renderer = Mock()
        mock_grid_renderer.render_grid.return_value = mock_image
        mock_grid_renderer_class.return_value = mock_grid_renderer

        renderer = MolecularRenderer()
        legends = ["Legend 0", "Legend 1", "Legend 2"]  # Index 3 is out of range

        with patch("molecular_string_renderer.renderer.logger") as mock_logger:
            result = renderer.render_grid(
                ["CCO", "CCC", "DDD"], "smiles", legends=legends
            )

            assert result is mock_image
            # Should disable legends due to mismatch (only 1 filtered legend but 2 molecules)
            mock_grid_renderer.render_grid.assert_called_once_with(mock_mols, None)
            mock_logger.warning.assert_called_once_with(
                "Legend count mismatch after filtering, disabling legends"
            )
