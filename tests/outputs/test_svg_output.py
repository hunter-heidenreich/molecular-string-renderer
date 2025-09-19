"""
Test suite for SVG output handler.

Comprehensive tests for SVGOutput class functionality, edge cases, and error handling.
Tests vector SVG generation, raster fallback, molecule handling, and SVG optimization.
"""

import base64
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs.vector import SVGOutput


class TestSVGOutputProperties:
    """Test SVG output handler properties."""

    def test_file_extension(self):
        """Test file extension property."""
        output = SVGOutput()
        assert output.file_extension == ".svg"

    def test_format_name_inherited(self):
        """Test format name is inherited from base class."""
        output = SVGOutput()
        assert output.format_name == "svg"


class TestSVGOutputInitialization:
    """Test SVG output handler initialization."""

    def test_init_without_config(self):
        """Test initialization without explicit config."""
        output = SVGOutput()
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)
        assert output._strategy is not None

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = OutputConfig(quality=85, optimize=False)
        output = SVGOutput(config)
        assert output.config is config
        assert output.config.quality == 85
        assert output.config.optimize is False
        assert output._strategy is not None

    def test_init_with_none_config(self):
        """Test initialization with None config."""
        output = SVGOutput(None)
        assert output.config is not None
        assert isinstance(output.config, OutputConfig)
        assert output._strategy is not None


class TestSVGOutputMoleculeHandling:
    """Test SVG molecule handling functionality."""

    @pytest.fixture
    def mock_molecule(self):
        """Create a mock RDKit molecule."""
        mol = MagicMock()
        mol.GetNumAtoms.return_value = 5
        mol.GetNumBonds.return_value = 4
        return mol

    def test_set_molecule(self, mock_molecule):
        """Test setting molecule for vector SVG generation."""
        output = SVGOutput()
        output.set_molecule(mock_molecule)
        assert output._strategy._vector_strategy._molecule is mock_molecule

    def test_set_molecule_none(self):
        """Test setting molecule to None."""
        output = SVGOutput()
        output.set_molecule(None)
        assert output._strategy._vector_strategy._molecule is None

    def test_set_molecule_replaces_previous(self, mock_molecule):
        """Test that setting molecule replaces previous molecule."""
        output = SVGOutput()

        # Set first molecule
        first_mol = MagicMock()
        output.set_molecule(first_mol)
        assert output._strategy._vector_strategy._molecule is first_mol

        # Set second molecule
        output.set_molecule(mock_molecule)
        assert output._strategy._vector_strategy._molecule is mock_molecule
        assert output._strategy._vector_strategy._molecule is not first_mol


class TestSVGOutputVectorGeneration:
    """Test SVG vector generation functionality."""

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (200, 200), "red")

    @pytest.fixture
    def mock_molecule(self):
        """Create a mock RDKit molecule."""
        mol = MagicMock()
        mol.GetNumAtoms.return_value = 5
        return mol

    def test_generate_vector_svg_with_molecule(self, test_image, mock_molecule):
        """Test vector SVG generation with molecule."""
        # Mock RDKit Draw module
        mock_svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="10"/></svg>'

        # Patch the Draw module in the svg_strategies module where it's imported
        with patch("molecular_string_renderer.outputs.svg_strategies.Draw") as mock_draw:
            mock_draw.MolToSVG.return_value = mock_svg

            output = SVGOutput()
            output.set_molecule(mock_molecule)

            result = output.get_bytes(test_image).decode('utf-8')

            assert result == mock_svg
            mock_draw.MolToSVG.assert_called_once_with(
                mock_molecule,
                width=200,
                height=200,
                kekulize=True,
                lineWidthMult=1,
                includeAtomCircles=True,
            )

    def test_generate_vector_svg_with_molecule_custom_line_width(
        self, test_image, mock_molecule
    ):
        """Test vector SVG generation with custom line width multiplier."""
        mock_svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="10"/></svg>'

        with patch("molecular_string_renderer.outputs.svg_strategies.Draw") as mock_draw:
            mock_draw.MolToSVG.return_value = mock_svg

            # Create config and output
            config = OutputConfig()
            # Add attribute directly to the object's __dict__ to bypass Pydantic validation
            config.__dict__["svg_line_width_mult"] = 2.5
            output = SVGOutput(config)
            output.set_molecule(mock_molecule)

            result = output.get_bytes(test_image).decode('utf-8')

            assert result == mock_svg
            mock_draw.MolToSVG.assert_called_once_with(
                mock_molecule,
                width=200,
                height=200,
                kekulize=True,
                lineWidthMult=2.5,
                includeAtomCircles=True,
            )

    def test_generate_vector_svg_with_optimization(self, test_image, mock_molecule):
        """Test vector SVG generation with optimization enabled."""
        mock_svg = """<?xml version="1.0"?>
<!-- Comment line -->
<svg xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="10"/>
</svg>"""

        with patch("molecular_string_renderer.outputs.svg_strategies.Draw") as mock_draw:
            mock_draw.MolToSVG.return_value = mock_svg

            config = OutputConfig(optimize=True)
            output = SVGOutput(config)
            output.set_molecule(mock_molecule)

            result = output.get_bytes(test_image).decode('utf-8')

            # Should remove comments and extra whitespace
            assert "<!-- Comment line -->" not in result
            assert result.count("\n") < mock_svg.count("\n")

    def test_generate_vector_svg_vector_disabled(self, test_image):
        """Test vector SVG generation when vector mode is disabled."""
        config = OutputConfig()
        # Add attribute directly to the object's __dict__ to bypass Pydantic validation
        config.__dict__["svg_use_vector"] = False
        output = SVGOutput(config)

        result = output.get_bytes(test_image).decode('utf-8')

        # Should use raster fallback
        assert "data:image/png;base64," in result
        assert result.startswith('<?xml version="1.0" encoding="UTF-8"?>')

    def test_generate_vector_svg_rdkit_error_fallback(self, test_image, mock_molecule):
        """Test fallback to raster when RDKit fails."""
        with patch("molecular_string_renderer.outputs.svg_strategies.Draw") as mock_draw:
            # Mock RDKit Draw.MolToSVG to raise an exception
            mock_draw.MolToSVG.side_effect = Exception("RDKit error")

            output = SVGOutput()
            output.set_molecule(mock_molecule)

            result = output.get_bytes(test_image).decode('utf-8')

            # Should fall back to raster SVG
            assert "data:image/png;base64," in result
            assert result.startswith('<?xml version="1.0" encoding="UTF-8"?>')

    def test_generate_vector_svg_no_molecule_fallback(self, test_image):
        """Test fallback to raster when no molecule is set."""
        output = SVGOutput()
        # Don't set molecule

        result = output.get_bytes(test_image).decode('utf-8')

        # Should fall back to raster SVG
        assert "data:image/png;base64," in result
        assert result.startswith('<?xml version="1.0" encoding="UTF-8"?>')


class TestSVGOutputRasterFallback:
    """Test SVG raster fallback functionality."""

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (100, 100), "blue")

    def test_generate_raster_svg(self, test_image):
        """Test raster SVG generation."""
        output = SVGOutput()
        result = output._strategy._raster_strategy.generate_svg(test_image, output.config)

        # Should be valid XML
        root = ET.fromstring(result)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"

        # Should have correct dimensions
        assert root.get("width") == "100"
        assert root.get("height") == "100"
        assert root.get("viewBox") == "0 0 100 100"

        # Should have embedded image
        image_elem = root.find(".//{http://www.w3.org/2000/svg}image")
        assert image_elem is not None
        href = image_elem.get("{http://www.w3.org/1999/xlink}href")
        assert href.startswith("data:image/png;base64,")

    def test_generate_raster_svg_valid_base64(self, test_image):
        """Test that raster SVG contains valid base64 data."""
        output = SVGOutput()
        result = output._strategy._raster_strategy.generate_svg(test_image, output.config)

        # Extract base64 data
        root = ET.fromstring(result)
        image_elem = root.find(".//{http://www.w3.org/2000/svg}image")
        href = image_elem.get("{http://www.w3.org/1999/xlink}href")
        base64_data = href.split("data:image/png;base64,")[1]

        # Should be valid base64
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0

        # Should be valid PNG
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")

    def test_generate_raster_svg_different_sizes(self):
        """Test raster SVG with different image sizes."""
        output = SVGOutput()

        sizes = [(50, 50), (200, 100), (100, 300)]
        for width, height in sizes:
            image = Image.new("RGB", (width, height), "green")
            result = output._strategy._raster_strategy.generate_svg(image, output.config)

            root = ET.fromstring(result)
            assert root.get("width") == str(width)
            assert root.get("height") == str(height)
            assert root.get("viewBox") == f"0 0 {width} {height}"


class TestSVGOutputOptimization:
    """Test SVG optimization functionality."""

    def test_optimize_svg_enabled(self):
        """Test SVG optimization when enabled."""
        config = OutputConfig(optimize=True)
        output = SVGOutput(config)

        svg_content = """<?xml version="1.0"?>
<!-- This is a comment -->
<svg xmlns="http://www.w3.org/2000/svg">
  <!-- Another comment -->
  <circle cx="50" cy="50" r="10"/>

  <rect x="10" y="10" width="20" height="20"/>
</svg>"""

        result = output._strategy._vector_strategy._optimize_svg(svg_content)

        # Should remove comments
        assert "<!-- This is a comment -->" not in result
        assert "<!-- Another comment -->" not in result

        # Should remove empty lines
        assert "\n\n" not in result

        # Should preserve essential content
        assert "<circle" in result
        assert "<rect" in result

    def test_optimize_svg_disabled(self):
        """Test SVG optimization when disabled."""
        config = OutputConfig(optimize=False)
        output = SVGOutput(config)

        svg_content = """<?xml version="1.0"?>
<!-- This is a comment -->
<svg xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="10"/>
</svg>"""

        # When optimization is disabled, we should test the actual behavior
        # The _optimize_svg method itself always optimizes, but it's only called when config.optimize is True
        # So let's test that the end result preserves comments when optimization is disabled
        with patch("molecular_string_renderer.outputs.svg_strategies.Draw") as mock_draw:
            mock_draw.MolToSVG.return_value = svg_content
            
            mock_molecule = MagicMock()
            output.set_molecule(mock_molecule)
            image = Image.new("RGB", (100, 100), "red")
            
            result = output.get_bytes(image).decode('utf-8')
            
            # Since optimization is disabled, comments should be preserved
            assert "<!-- This is a comment -->" in result

    def test_optimize_svg_preserve_structure(self):
        """Test that optimization preserves SVG structure."""
        config = OutputConfig(optimize=True)
        output = SVGOutput(config)

        svg_content = """<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <g id="group1">
    <circle cx="50" cy="50" r="10"/>
  </g>
</svg>"""

        result = output._strategy._vector_strategy._optimize_svg(svg_content)

        # Should still be valid XML
        root = ET.fromstring(result)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"

        # Should preserve structure
        group = root.find(".//{http://www.w3.org/2000/svg}g")
        assert group is not None
        assert group.get("id") == "group1"


class TestSVGOutputSaveMethod:
    """Test SVG save method functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (100, 100), "red")

    def test_save_with_string_path(self, temp_dir, test_image):
        """Test saving with string path."""
        output = SVGOutput()
        output_path = str(temp_dir / "test.svg")

        output.save(test_image, output_path)

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_save_with_path_object(self, temp_dir, test_image):
        """Test saving with Path object."""
        output = SVGOutput()
        output_path = temp_dir / "test.svg"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_auto_extension(self, temp_dir, test_image):
        """Test automatic extension addition."""
        output = SVGOutput()
        output_path = temp_dir / "test"  # No extension

        output.save(test_image, output_path)

        svg_path = temp_dir / "test.svg"
        assert svg_path.exists()
        assert svg_path.stat().st_size > 0

    def test_save_preserves_svg_extension(self, temp_dir, test_image):
        """Test that .svg extensions are preserved."""
        output = SVGOutput()
        output_path = temp_dir / "test.svg"

        output.save(test_image, output_path)

        assert output_path.exists()
        assert output_path.name == "test.svg"

    def test_save_creates_directory(self, temp_dir, test_image):
        """Test that missing directories are created."""
        output = SVGOutput()
        nested_path = temp_dir / "subdir" / "nested" / "test.svg"

        output.save(test_image, nested_path)

        assert nested_path.exists()
        assert nested_path.stat().st_size > 0

    def test_save_overwrites_existing(self, temp_dir, test_image):
        """Test that existing files are overwritten."""
        output = SVGOutput()
        output_path = temp_dir / "test.svg"

        # Create initial file
        output_path.write_text("initial content")
        assert output_path.read_text() == "initial content"

        # Save should overwrite
        output.save(test_image, output_path)
        content = output_path.read_text()
        assert content != "initial content"
        assert "svg" in content.lower()

    def test_save_produces_valid_svg(self, temp_dir, test_image):
        """Test that saved file is valid SVG."""
        output = SVGOutput()
        output_path = temp_dir / "test.svg"

        output.save(test_image, output_path)

        # Should be valid XML
        content = output_path.read_text()
        root = ET.fromstring(content)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"

    @patch("molecular_string_renderer.outputs.svg_strategies.HybridSVGStrategy.generate_svg")
    def test_save_error_handling(self, mock_generate, temp_dir, test_image):
        """Test error handling during save."""
        mock_generate.side_effect = Exception("Mock generation error")
        output = SVGOutput()
        output_path = temp_dir / "test.svg"

        with pytest.raises(IOError, match="Failed to save svg"):
            output.save(test_image, output_path)

    @patch("molecular_string_renderer.outputs.base.logger")
    def test_save_logs_success(self, mock_logger, temp_dir, test_image):
        """Test that successful saves are logged."""
        output = SVGOutput()
        output_path = temp_dir / "test.svg"

        output.save(test_image, output_path)

        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]
        assert "Successfully saved svg" in log_call

    @patch("molecular_string_renderer.outputs.base.logger")
    @patch("molecular_string_renderer.outputs.svg_strategies.HybridSVGStrategy.generate_svg")
    def test_save_logs_error(self, mock_generate, mock_logger, temp_dir, test_image):
        """Test that save errors are logged."""
        mock_generate.side_effect = Exception("Mock generation error")
        output = SVGOutput()
        output_path = temp_dir / "test.svg"

        with pytest.raises(IOError):
            output.save(test_image, output_path)

        mock_logger.error.assert_called_once()
        log_call = mock_logger.error.call_args[0][0]
        assert "Failed to save svg" in log_call


class TestSVGOutputGetBytesMethod:
    """Test SVG get_bytes method functionality."""

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (100, 100), "red")

    def test_get_bytes_returns_bytes(self, test_image):
        """Test that get_bytes returns bytes."""
        output = SVGOutput()
        result = output.get_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_get_bytes_valid_svg(self, test_image):
        """Test that get_bytes produces valid SVG."""
        output = SVGOutput()
        result = output.get_bytes(test_image)

        # Should be valid XML
        content = result.decode("utf-8")
        root = ET.fromstring(content)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"

    def test_get_bytes_utf8_encoding(self, test_image):
        """Test that get_bytes uses UTF-8 encoding."""
        output = SVGOutput()
        result = output.get_bytes(test_image)

        # Should decode properly as UTF-8
        content = result.decode("utf-8")
        assert isinstance(content, str)
        assert len(content) > 0

    def test_get_bytes_with_molecule(self, test_image):
        """Test get_bytes with molecule for vector SVG."""
        mock_svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="10"/></svg>'

        with patch("molecular_string_renderer.outputs.svg_strategies.Draw") as mock_draw:
            mock_draw.MolToSVG.return_value = mock_svg

            mock_molecule = MagicMock()
            output = SVGOutput()
            output.set_molecule(mock_molecule)

            result = output.get_bytes(test_image)
            content = result.decode("utf-8")

            assert content == mock_svg

    def test_get_bytes_raster_fallback(self, test_image):
        """Test get_bytes with raster fallback."""
        output = SVGOutput()
        # Don't set molecule, should fall back to raster

        result = output.get_bytes(test_image)
        content = result.decode("utf-8")

        # Should contain base64 embedded image
        assert "data:image/png;base64," in content

    @patch("molecular_string_renderer.outputs.svg_strategies.HybridSVGStrategy.generate_svg")
    def test_get_bytes_error_handling(self, mock_generate, test_image):
        """Test error handling in get_bytes."""
        mock_generate.side_effect = Exception("Mock generation error")
        output = SVGOutput()

        with pytest.raises(Exception):
            output.get_bytes(test_image)


class TestSVGOutputIntegration:
    """Integration tests for SVG output handler."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_full_workflow_vector_svg(self, temp_dir):
        """Test complete workflow with vector SVG."""
        mock_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
  <circle cx="100" cy="100" r="50" fill="red"/>
</svg>"""

        with patch("molecular_string_renderer.outputs.svg_strategies.Draw") as mock_draw:
            mock_draw.MolToSVG.return_value = mock_svg

            config = OutputConfig(optimize=True)
            output = SVGOutput(config)
            image = Image.new("RGB", (200, 200), "blue")
            mock_molecule = MagicMock()
            output.set_molecule(mock_molecule)
            output_path = temp_dir / "workflow_test.svg"

            # Test save
            output.save(image, output_path)
            assert output_path.exists()

            # Test get_bytes
            bytes_data = output.get_bytes(image)
            assert len(bytes_data) > 0

            # Verify both produce valid SVG
            saved_content = output_path.read_text()
            bytes_content = bytes_data.decode("utf-8")

            saved_root = ET.fromstring(saved_content)
            bytes_root = ET.fromstring(bytes_content)

            assert saved_root.tag == "{http://www.w3.org/2000/svg}svg"
            assert bytes_root.tag == "{http://www.w3.org/2000/svg}svg"

    def test_full_workflow_raster_fallback(self, temp_dir):
        """Test complete workflow with raster fallback."""
        output = SVGOutput()
        image = Image.new("RGB", (200, 200), "green")
        # Don't set molecule - should use raster fallback
        output_path = temp_dir / "raster_test.svg"

        # Test save
        output.save(image, output_path)
        assert output_path.exists()

        # Test get_bytes
        bytes_data = output.get_bytes(image)

        # Verify both produce valid SVG with embedded raster
        saved_content = output_path.read_text()
        bytes_content = bytes_data.decode("utf-8")

        assert "data:image/png;base64," in saved_content
        assert "data:image/png;base64," in bytes_content

        # Both should be valid XML
        saved_root = ET.fromstring(saved_content)
        bytes_root = ET.fromstring(bytes_content)

        assert saved_root.tag == "{http://www.w3.org/2000/svg}svg"
        assert bytes_root.tag == "{http://www.w3.org/2000/svg}svg"

    def test_full_workflow_optimization(self, temp_dir):
        """Test workflow with optimization enabled/disabled."""
        image = Image.new("RGB", (100, 100), "purple")

        # With optimization
        config_opt = OutputConfig(optimize=True)
        output_opt = SVGOutput(config_opt)
        path_opt = temp_dir / "optimized.svg"
        output_opt.save(image, path_opt)

        # Without optimization
        config_no_opt = OutputConfig(optimize=False)
        output_no_opt = SVGOutput(config_no_opt)
        path_no_opt = temp_dir / "unoptimized.svg"
        output_no_opt.save(image, path_no_opt)

        # Both should be valid
        assert path_opt.exists()
        assert path_no_opt.exists()

        # Optimized might be smaller (though for simple raster fallback, might be same)
        opt_size = path_opt.stat().st_size
        no_opt_size = path_no_opt.stat().st_size
        assert opt_size <= no_opt_size * 1.1  # Allow some variance


class TestSVGOutputEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_image(self):
        """Test with very small image (1x1 pixel)."""
        output = SVGOutput()
        image = Image.new("RGB", (1, 1), "red")

        bytes_data = output.get_bytes(image)
        assert len(bytes_data) > 0

        # Should be valid SVG
        content = bytes_data.decode("utf-8")
        root = ET.fromstring(content)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert root.get("width") == "1"
        assert root.get("height") == "1"

    def test_very_large_image_dimensions(self):
        """Test with large image dimensions."""
        output = SVGOutput()
        image = Image.new("RGB", (1000, 1000), "green")

        bytes_data = output.get_bytes(image)
        assert len(bytes_data) > 0

        # Should handle large dimensions
        content = bytes_data.decode("utf-8")
        root = ET.fromstring(content)
        assert root.get("width") == "1000"
        assert root.get("height") == "1000"

    def test_non_rgb_image_modes(self):
        """Test with different image modes."""
        output = SVGOutput()

        modes_to_test = [
            ("RGBA", (255, 0, 0, 128)),
            ("L", 128),
            ("LA", (128, 200)),
        ]

        for mode, color in modes_to_test:
            image = Image.new(mode, (50, 50), color)
            bytes_data = output.get_bytes(image)

            # Should handle any mode
            assert len(bytes_data) > 0
            content = bytes_data.decode("utf-8")
            root = ET.fromstring(content)
            assert root.tag == "{http://www.w3.org/2000/svg}svg"

    def test_empty_svg_optimization(self):
        """Test optimization with minimal SVG content."""
        config = OutputConfig(optimize=True)
        output = SVGOutput(config)

        minimal_svg = """<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg">
</svg>"""

        result = output._strategy._vector_strategy._optimize_svg(minimal_svg)

        # Should still be valid even when minimal
        root = ET.fromstring(result)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"

    def test_rdkit_import_error_fallback(self):
        """Test fallback when RDKit is not available."""
        output = SVGOutput()
        mock_molecule = MagicMock()
        output.set_molecule(mock_molecule)
        image = Image.new("RGB", (100, 100), "red")

        # Directly test the exception handling by patching the try block
        with patch(
            "rdkit.Chem.Draw.MolToSVG", 
            side_effect=ImportError("RDKit not available")
        ):
            # Should fall back to raster without error
            result = output.get_bytes(image).decode('utf-8')
            assert "data:image/png;base64," in result

    def test_molecule_with_rdkit_unavailable(self):
        """Test molecule handling when RDKit module is unavailable."""
        output = SVGOutput()
        image = Image.new("RGB", (100, 100), "red")

        # Set a mock molecule but patch Draw to be unavailable
        mock_molecule = MagicMock()
        output.set_molecule(mock_molecule)

        with patch.dict("sys.modules", {"rdkit.Chem": None}):
            # Should fall back gracefully
            result = output.get_bytes(image).decode('utf-8')
            assert "data:image/png;base64," in result


class TestSVGOutputThreadSafety:
    """Test thread safety and concurrent usage."""

    def test_multiple_instances_independent(self):
        """Test that multiple instances are independent."""
        config1 = OutputConfig(optimize=True)
        config2 = OutputConfig(optimize=False)

        output1 = SVGOutput(config1)
        output2 = SVGOutput(config2)

        assert output1.config.optimize is True
        assert output2.config.optimize is False

        # Set different molecules
        mol1 = MagicMock()
        mol2 = MagicMock()
        output1.set_molecule(mol1)
        output2.set_molecule(mol2)

        assert output1._strategy._vector_strategy._molecule is mol1
        assert output2._strategy._vector_strategy._molecule is mol2

    def test_instance_method_isolation(self):
        """Test that instance methods don't interfere."""
        output = SVGOutput()

        image1 = Image.new("RGB", (50, 50), "red")
        image2 = Image.new("RGB", (100, 100), "blue")

        # These operations should be independent
        result1 = output._strategy._raster_strategy.generate_svg(image1, output.config)
        result2 = output._strategy._raster_strategy.generate_svg(image2, output.config)

        # Both should be valid but different
        root1 = ET.fromstring(result1)
        root2 = ET.fromstring(result2)

        assert root1.get("width") == "50"
        assert root2.get("width") == "100"


class TestSVGOutputInheritance:
    """Test proper inheritance from base classes."""

    def test_is_output_handler(self):
        """Test that SVGOutput is an OutputHandler."""
        from molecular_string_renderer.outputs.base import OutputHandler

        output = SVGOutput()
        assert isinstance(output, OutputHandler)

    def test_is_vector_output_handler(self):
        """Test that SVGOutput is a VectorOutputHandler."""
        from molecular_string_renderer.outputs.base import VectorOutputHandler

        output = SVGOutput()
        assert isinstance(output, VectorOutputHandler)

    def test_implements_required_methods(self):
        """Test that all required abstract methods are implemented."""
        output = SVGOutput()

        # Test that all abstract methods can be called
        assert hasattr(output, "save")
        assert hasattr(output, "get_bytes")
        assert hasattr(output, "file_extension")
        assert hasattr(output, "format_name")
        assert hasattr(output, "set_molecule")

        # Test that they return expected types
        assert isinstance(output.file_extension, str)
        assert isinstance(output.format_name, str)


class TestSVGOutputTypeHints:
    """Test type hints and return types."""

    def test_return_types(self):
        """Test that methods return correct types."""
        output = SVGOutput()
        image = Image.new("RGB", (10, 10), "white")

        # Test property return types
        assert isinstance(output.file_extension, str)
        assert isinstance(output.format_name, str)

        # Test method return types
        assert isinstance(output.get_bytes(image), bytes)
        assert isinstance(output._strategy._raster_strategy.generate_svg(image, output.config), str)
        assert isinstance(output._strategy._vector_strategy._optimize_svg("<svg></svg>"), str)

    def test_method_accepts_correct_types(self):
        """Test that methods accept correct input types."""
        output = SVGOutput()
        image = Image.new("RGB", (10, 10), "white")

        # These should not raise type errors
        output.get_bytes(image)
        output._strategy._raster_strategy.generate_svg(image, output.config)
        output.set_molecule(None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test both string and Path inputs
            output.save(image, str(Path(tmp_dir) / "test1.svg"))
            output.save(image, Path(tmp_dir) / "test2.svg")


class TestSVGOutputSpecificBugTests:
    """Test for specific potential bugs found during development."""

    def test_base64_encoding_consistency(self):
        """Test that base64 encoding is consistent and valid."""
        output = SVGOutput()
        image = Image.new("RGB", (50, 50), "red")

        # Generate multiple times
        result1 = output._strategy._raster_strategy.generate_svg(image, output.config)
        result2 = output._strategy._raster_strategy.generate_svg(image, output.config)

        # Should be identical (deterministic)
        assert result1 == result2

        # Extract base64 data
        root = ET.fromstring(result1)
        image_elem = root.find(".//{http://www.w3.org/2000/svg}image")
        href = image_elem.get("{http://www.w3.org/1999/xlink}href")
        base64_data = href.split("data:image/png;base64,")[1]

        # Should decode without error
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0

    def test_svg_namespace_handling(self):
        """Test proper SVG namespace handling."""
        output = SVGOutput()
        image = Image.new("RGB", (50, 50), "blue")

        result = output._strategy._raster_strategy.generate_svg(image, output.config)
        root = ET.fromstring(result)

        # Should have proper namespaces
        assert root.tag == "{http://www.w3.org/2000/svg}svg"

        # Should have xlink namespace for href
        image_elem = root.find(".//{http://www.w3.org/2000/svg}image")
        href = image_elem.get("{http://www.w3.org/1999/xlink}href")
        assert href is not None
        assert href.startswith("data:image/png;base64,")

    def test_optimization_preserves_functionality(self):
        """Test that optimization doesn't break functionality."""
        mock_svg = """<?xml version="1.0"?>
<!-- Comment to remove -->
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <!-- Another comment -->
  <circle cx="50" cy="50" r="25" fill="red"/>
</svg>"""

        with patch("molecular_string_renderer.outputs.svg_strategies.Draw") as mock_draw:
            mock_draw.MolToSVG.return_value = mock_svg

            config = OutputConfig(optimize=True)
            output = SVGOutput(config)
            image = Image.new("RGB", (100, 100), "red")
            mock_molecule = MagicMock()
            output.set_molecule(mock_molecule)

            result = output.get_bytes(image).decode('utf-8')

            # Should remove comments but preserve functionality
            assert "<!-- Comment to remove -->" not in result
            assert "<!-- Another comment -->" not in result

            # Should still be valid XML with essential elements
            root = ET.fromstring(result)
            assert root.tag == "{http://www.w3.org/2000/svg}svg"
            circle = root.find(".//{http://www.w3.org/2000/svg}circle")
            assert circle is not None
            assert circle.get("fill") == "red"

    def test_unicode_handling_in_svg(self):
        """Test proper Unicode handling in SVG content."""
        output = SVGOutput()

        # Test with unicode characters in optimization
        svg_with_unicode = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <text>Benzène résumé naïve</text>
</svg>"""

        result = output._strategy._vector_strategy._optimize_svg(svg_with_unicode)

        # Should preserve unicode content
        assert "Benzène" in result
        assert "résumé" in result
        assert "naïve" in result

        # Should encode properly to bytes
        bytes_result = result.encode("utf-8")
        decoded = bytes_result.decode("utf-8")
        assert decoded == result


if __name__ == "__main__":
    pytest.main([__file__])
