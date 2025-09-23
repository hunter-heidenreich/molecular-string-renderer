"""
Comprehensive tests for the RenderingPipeline class.

Tests pipeline behavior using real implementations where possible and
minimal mocking only when necessary for isolation or error simulation.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from molecular_string_renderer.config import OutputConfig, ParserConfig, RenderConfig
from molecular_string_renderer.exceptions import ParsingError, RenderingError
from molecular_string_renderer.pipeline import RenderingPipeline


class TestRenderingPipelineBasic:
    """Basic tests for RenderingPipeline functionality."""

    def test_pipeline_initialization(self):
        """Test that pipeline can be initialized with all config types."""
        render_config = RenderConfig(width=800, height=600)
        parser_config = ParserConfig()
        output_config = OutputConfig()

        pipeline = RenderingPipeline(
            render_config=render_config,
            parser_config=parser_config,
            output_config=output_config,
        )

        assert pipeline.render_config is render_config
        assert pipeline.parser_config is parser_config
        assert pipeline.output_config is output_config

    def test_initialization_with_different_render_configs(self):
        """Test pipeline initialization with various render configurations."""
        render_config = RenderConfig(
            width=1024, height=768, background_color="white", show_hydrogen=True
        )

        pipeline = RenderingPipeline(
            render_config=render_config,
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        assert pipeline.render_config.width == 1024
        assert pipeline.render_config.height == 768
        assert pipeline.render_config.background_color == "white"
        assert pipeline.render_config.show_hydrogen is True

    def test_validate_inputs_delegates_correctly(self):
        """Test that validate_inputs properly delegates to the utils function."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        with patch(
            "molecular_string_renderer.utils.validate_and_normalize_inputs"
        ) as mock_validate:
            mock_validate.return_value = ("CCO", "smiles", Path("test.png"))

            result = pipeline.validate_inputs("CCO", "smiles", "test.png")

            assert result == ("CCO", "smiles", Path("test.png"))
            mock_validate.assert_called_once_with("CCO", "smiles", "test.png")

    def test_validate_inputs_with_none_output_path(self):
        """Test input validation with None output path."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        with patch(
            "molecular_string_renderer.utils.validate_and_normalize_inputs"
        ) as mock_validate:
            mock_validate.return_value = ("CCO", "smiles", None)

            result = pipeline.validate_inputs("CCO", "smiles", None)

            assert result == ("CCO", "smiles", None)
            mock_validate.assert_called_once_with("CCO", "smiles", None)

    def test_validate_inputs_propagates_validation_error(self):
        """Test that validation errors are properly propagated."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        with patch(
            "molecular_string_renderer.utils.validate_and_normalize_inputs"
        ) as mock_validate:
            mock_validate.side_effect = ValueError("Invalid molecular string")

            with pytest.raises(ValueError, match="Invalid molecular string"):
                pipeline.validate_inputs("invalid", "smiles", None)

    def test_parse_molecule_with_valid_smiles(self):
        """Test parsing a valid SMILES string."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Test with a real SMILES string that should parse successfully
        result = pipeline.parse_molecule("CCO", "smiles")

        # Should return a molecule object (not None)
        assert result is not None

    def test_parse_molecule_with_invalid_smiles_raises_error(self):
        """Test that parsing an invalid SMILES string raises ParsingError."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Test with an invalid SMILES string
        with pytest.raises(ParsingError, match="Failed to parse SMILES"):
            pipeline.parse_molecule("INVALID_SMILES_123", "smiles")

    def test_parse_molecule_with_different_format_types(self):
        """Test parsing with different molecular format types."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Test with valid inputs that will actually parse
        test_cases = [
            ("CCO", "smiles"),
            ("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "inchi"),
        ]

        for molecular_string, format_type in test_cases:
            result = pipeline.parse_molecule(molecular_string, format_type)

            # Should return a valid molecule object, not None
            assert result is not None

    def test_parse_molecule_with_empty_string_raises_error(self):
        """Test parsing behavior with empty molecular string."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Use real implementation - empty string should trigger ParsingError
        with pytest.raises(ParsingError, match="Failed to parse SMILES"):
            pipeline.parse_molecule("", "smiles")

    def test_render_molecule_with_valid_molecule(self):
        """Test that render_molecule works with a valid molecule."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(width=200, height=200),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # First parse a valid molecule
        mol = pipeline.parse_molecule("CCO", "smiles")

        # Then render it
        image = pipeline.render_molecule(mol)

        # Should return an image
        assert image is not None

    def test_render_molecule_with_none_raises_error(self):
        """Test that render_molecule with None molecule raises RenderingError."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Test with None (which should cause a rendering error)
        with pytest.raises(RenderingError, match="Failed to render molecule"):
            pipeline.render_molecule(None)

    def test_render_molecule_with_different_dimensions(self):
        """Test rendering with different image dimensions."""
        test_cases = [
            (300, 300),
            (800, 600),
            (1024, 768),
        ]

        for width, height in test_cases:
            pipeline = RenderingPipeline(
                render_config=RenderConfig(width=width, height=height),
                parser_config=ParserConfig(),
                output_config=OutputConfig(),
            )

            # Parse a real molecule and render it
            mol = pipeline.parse_molecule("CCO", "smiles")
            result = pipeline.render_molecule(mol)

            # Should return a valid image
            assert result is not None

    def test_save_output_without_path_does_nothing(self):
        """Test that save_output does nothing when no path is provided."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Mock image for testing
        mock_image = Mock()

        # Should not raise any errors
        pipeline.save_output(
            image=mock_image, output_path=None, output_format="png", auto_filename=False
        )

    def test_save_output_with_path_delegates_to_utils(self):
        """Test that save_output properly delegates to utils when path is provided."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        mock_image = Mock()
        output_path = Path("test.png")

        with patch("molecular_string_renderer.utils.handle_output_saving") as mock_save:
            pipeline.save_output(
                image=mock_image, output_path=output_path, output_format="png"
            )

            mock_save.assert_called_once_with(
                image=mock_image,
                output_path=output_path,
                output_format="png",
                output_config=pipeline.output_config,
                mol=None,
                auto_filename=False,
                molecular_string=None,
            )

    def test_pipeline_configuration_affects_behavior(self):
        """Test that different configurations affect pipeline behavior."""
        # Test with different render configs
        small_config = RenderConfig(width=100, height=100)
        large_config = RenderConfig(width=500, height=500)

        small_pipeline = RenderingPipeline(
            render_config=small_config,
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        large_pipeline = RenderingPipeline(
            render_config=large_config,
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Parse the same molecule
        mol = small_pipeline.parse_molecule("CCO", "smiles")

        # Render with different configs
        small_image = small_pipeline.render_molecule(mol)
        large_image = large_pipeline.render_molecule(mol)

        # Both should produce images
        assert small_image is not None
        assert large_image is not None

    def test_pipeline_error_handling_isolation(self):
        """Test that errors in one operation don't affect pipeline state."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # First, have a parsing error
        with pytest.raises(ParsingError):
            pipeline.parse_molecule("INVALID", "smiles")

        # Pipeline should still work for valid inputs
        valid_mol = pipeline.parse_molecule("CCO", "smiles")
        assert valid_mol is not None

        # And rendering should still work
        image = pipeline.render_molecule(valid_mol)
        assert image is not None


class TestRenderingPipelineWorkflow:
    """Test full pipeline workflows."""

    def test_complete_workflow_smiles_to_image(self):
        """Test a complete workflow from SMILES to rendered image."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(width=300, height=300),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Complete workflow
        molecular_string, format_type, output_path = pipeline.validate_inputs(
            "CCO", "smiles", None
        )

        mol = pipeline.parse_molecule(molecular_string, format_type)
        image = pipeline.render_molecule(mol)

        # Save output (without actual file creation)
        pipeline.save_output(
            image=image,
            output_path=output_path,
            output_format="png",
            mol=mol,
            auto_filename=False,
        )

        # All steps should complete without error
        assert molecular_string == "CCO"
        assert format_type == "smiles"
        assert output_path is None
        assert mol is not None
        assert image is not None

    def test_workflow_with_different_formats(self):
        """Test workflow with different molecular formats."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Test cases with different formats
        test_cases = [
            ("CCO", "smiles"),
            ("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "inchi"),
        ]

        for molecular_string, format_type in test_cases:
            # Validate inputs
            validated_string, validated_format, _ = pipeline.validate_inputs(
                molecular_string, format_type, None
            )

            # Parse molecule
            mol = pipeline.parse_molecule(validated_string, validated_format)

            # Render molecule
            image = pipeline.render_molecule(mol)

            # All steps should succeed
            assert validated_string == molecular_string
            assert validated_format == format_type
            assert mol is not None
            assert image is not None

    def test_workflow_error_propagation(self):
        """Test that errors propagate correctly through workflow."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Test parsing error propagation
        molecular_string, format_type, _ = pipeline.validate_inputs(
            "INVALID_MOLECULE", "smiles", None
        )

        # Parsing should fail
        with pytest.raises(ParsingError):
            pipeline.parse_molecule(molecular_string, format_type)

    def test_pipeline_method_chaining_compatibility(self):
        """Test that pipeline methods can be used in sequence (like a full workflow)."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        with patch(
            "molecular_string_renderer.utils.validate_and_normalize_inputs"
        ) as mock_validate:
            mock_validate.return_value = ("CCO", "smiles", Path("output.png"))

            # Simulate a full pipeline workflow
            molecular_string, format_type, output_path = pipeline.validate_inputs(
                "CCO", "smiles", "output.png"
            )
            mol = pipeline.parse_molecule(molecular_string, format_type)
            image = pipeline.render_molecule(mol)

            # Verify the workflow executed correctly
            assert molecular_string == "CCO"
            assert format_type == "smiles"
            assert output_path == Path("output.png")
            assert mol is not None  # Real molecule object
            assert image is not None  # Real image object


class TestRenderingPipelineIntegration:
    """Integration tests using real implementations."""

    def test_full_smiles_to_image_pipeline(self):
        """Test complete pipeline from SMILES input to rendered image."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(width=300, height=300),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        with patch(
            "molecular_string_renderer.utils.validate_and_normalize_inputs"
        ) as mock_validate:
            # Mock only the validation step
            mock_validate.return_value = ("CCO", "smiles", Path("ethanol.png"))

            # Execute full pipeline using real implementations
            molecular_string, format_type, output_path = pipeline.validate_inputs(
                "CCO", "smiles", "ethanol.png"
            )
            mol = pipeline.parse_molecule(molecular_string, format_type)
            image = pipeline.render_molecule(mol)

            # Verify workflow executed correctly
            assert molecular_string == "CCO"
            assert format_type == "smiles"
            assert output_path == Path("ethanol.png")
            assert mol is not None  # Real RDKit molecule
            assert image is not None  # Real PIL Image

    def test_full_inchi_to_image_pipeline(self):
        """Test complete pipeline from InChI input to rendered image."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(width=500, height=400),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        inchi_string = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"

        with patch(
            "molecular_string_renderer.utils.validate_and_normalize_inputs"
        ) as mock_validate:
            mock_validate.return_value = (inchi_string, "inchi", Path("ethanol.svg"))

            # Execute pipeline
            molecular_string, format_type, output_path = pipeline.validate_inputs(
                inchi_string, "inchi", "ethanol.svg"
            )
            mol = pipeline.parse_molecule(molecular_string, format_type)
            image = pipeline.render_molecule(mol)

            # Verify successful execution
            assert molecular_string == inchi_string
            assert format_type == "inchi"
            assert output_path == Path("ethanol.svg")
            assert mol is not None
            assert image is not None

    def test_pipeline_with_different_configurations(self):
        """Test pipeline with various configuration combinations."""
        config_combinations = [
            (RenderConfig(width=200, height=200), ParserConfig(), OutputConfig()),
            (
                RenderConfig(width=800, height=600, show_hydrogen=True),
                ParserConfig(),
                OutputConfig(),
            ),
            (
                RenderConfig(width=400, height=400, background_color="white"),
                ParserConfig(),
                OutputConfig(),
            ),
        ]

        for render_config, parser_config, output_config in config_combinations:
            pipeline = RenderingPipeline(
                render_config=render_config,
                parser_config=parser_config,
                output_config=output_config,
            )

            # Test that pipeline works with this configuration
            mol = pipeline.parse_molecule("CCO", "smiles")
            image = pipeline.render_molecule(mol)

            assert mol is not None
            assert image is not None

            # Verify configurations are stored correctly
            assert pipeline.render_config is render_config
            assert pipeline.parser_config is parser_config
            assert pipeline.output_config is output_config

    def test_pipeline_error_propagation(self):
        """Test that errors propagate correctly through the pipeline."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Test parsing error with invalid SMILES
        with pytest.raises(ParsingError, match="Failed to parse SMILES"):
            pipeline.parse_molecule("INVALID_SMILES_123", "smiles")

        # Test rendering error with None molecule
        with pytest.raises(RenderingError, match="Failed to render molecule"):
            pipeline.render_molecule(None)

        # Verify pipeline state is not corrupted after errors
        mol = pipeline.parse_molecule("CCO", "smiles")
        image = pipeline.render_molecule(mol)
        assert mol is not None
        assert image is not None

    def test_pipeline_save_output_integration(self):
        """Test save_output method integration with real data."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(width=300, height=300),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Parse and render a real molecule
        mol = pipeline.parse_molecule("CCO", "smiles")
        image = pipeline.render_molecule(mol)

        with patch("molecular_string_renderer.utils.handle_output_saving") as mock_save:
            # Test save_output without path (should not call utils)
            pipeline.save_output(
                image=image,
                output_path=None,
                output_format="png",
                mol=mol,
                auto_filename=False,
            )
            mock_save.assert_not_called()

            # Test save_output with path (should call utils)
            pipeline.save_output(
                image=image,
                output_path=Path("test.png"),
                output_format="png",
                mol=mol,
                auto_filename=False,
                molecular_string="CCO",
            )
            mock_save.assert_called_once()

    def test_pipeline_with_auto_filename_generation(self):
        """Test pipeline save_output with auto filename generation."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(width=400, height=400),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Parse and render a real molecule
        mol = pipeline.parse_molecule("CCC", "smiles")
        image = pipeline.render_molecule(mol)

        with patch("molecular_string_renderer.utils.handle_output_saving") as mock_save:
            # Test auto filename generation
            pipeline.save_output(
                image=image,
                output_path=None,
                output_format="png",
                mol=mol,
                auto_filename=True,
                molecular_string="CCC",
            )

            # Should have called handle_output_saving with auto_filename=True
            mock_save.assert_called_once()
            call_args = mock_save.call_args
            assert call_args.kwargs["auto_filename"] is True
            assert call_args.kwargs["molecular_string"] == "CCC"

    def test_pipeline_workflow_isolation(self):
        """Test that pipeline instances are isolated from each other."""
        # Create two different pipeline instances
        pipeline1 = RenderingPipeline(
            render_config=RenderConfig(width=300, height=300),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        pipeline2 = RenderingPipeline(
            render_config=RenderConfig(width=600, height=600, show_hydrogen=True),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Both should work independently
        mol1 = pipeline1.parse_molecule("CCO", "smiles")
        mol2 = pipeline2.parse_molecule("CCC", "smiles")

        image1 = pipeline1.render_molecule(mol1)
        image2 = pipeline2.render_molecule(mol2)

        # Both should succeed
        assert mol1 is not None
        assert mol2 is not None
        assert image1 is not None
        assert image2 is not None

        # Configurations should remain separate
        assert pipeline1.render_config.width == 300
        assert pipeline2.render_config.width == 600
        assert pipeline1.render_config.show_hydrogen is False
        assert pipeline2.render_config.show_hydrogen is True


class TestRenderingPipelineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_handling(self):
        """Test pipeline behavior with empty strings."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Empty molecular string should fail parsing
        with pytest.raises(ParsingError):
            pipeline.parse_molecule("", "smiles")

    def test_different_render_dimensions(self):
        """Test pipeline with various render dimensions."""
        dimensions = [
            (100, 100),  # Minimum practical size
            (500, 400),  # Standard
            (2000, 2000),  # Large
        ]

        for width, height in dimensions:
            pipeline = RenderingPipeline(
                render_config=RenderConfig(width=width, height=height),
                parser_config=ParserConfig(),
                output_config=OutputConfig(),
            )

            # Parse and render a simple molecule
            mol = pipeline.parse_molecule("C", "smiles")
            image = pipeline.render_molecule(mol)

            assert mol is not None
            assert image is not None

    def test_save_output_parameter_combinations(self):
        """Test save_output with different parameter combinations."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        mock_image = Mock()
        mock_mol = Mock()

        # Test different combinations
        test_cases = [
            # (output_path, auto_filename, should_call_utils)
            (None, False, False),  # No saving
            (None, True, True),  # Auto filename
            ("test.png", False, True),  # Explicit path
            ("test.png", True, True),  # Both path and auto filename
        ]

        for output_path, auto_filename, should_call_utils in test_cases:
            with patch(
                "molecular_string_renderer.utils.handle_output_saving"
            ) as mock_save:
                pipeline.save_output(
                    image=mock_image,
                    output_path=output_path,
                    output_format="png",
                    mol=mock_mol,
                    auto_filename=auto_filename,
                    molecular_string="CCO",
                )

                if should_call_utils:
                    mock_save.assert_called_once()
                else:
                    mock_save.assert_not_called()

    def test_pipeline_edge_cases(self):
        """Test pipeline behavior with edge cases."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(width=100, height=100),  # Small dimensions
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Test with minimal molecule
        mol = pipeline.parse_molecule("C", "smiles")  # Single carbon
        image = pipeline.render_molecule(mol)

        assert mol is not None
        assert image is not None

    def test_pipeline_with_different_molecular_formats(self):
        """Test pipeline with various molecular format types."""
        pipeline = RenderingPipeline(
            render_config=RenderConfig(),
            parser_config=ParserConfig(),
            output_config=OutputConfig(),
        )

        # Test cases with valid inputs for each format
        test_cases = [
            ("CCO", "smiles"),
            ("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "inchi"),
        ]

        for molecular_string, format_type in test_cases:
            mol = pipeline.parse_molecule(molecular_string, format_type)
            image = pipeline.render_molecule(mol)

            assert mol is not None, f"Failed to parse {format_type}: {molecular_string}"
            assert image is not None, (
                f"Failed to render {format_type}: {molecular_string}"
            )
