"""
Shared test configuration and fixtures for renderer tests.

Provides common fixtures and utilities used across all renderer tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from rdkit import Chem
from rdkit.Chem import rdDepictor

from molecular_string_renderer.config import RenderConfig
from molecular_string_renderer.renderers import (
    Molecule2DRenderer,
    MoleculeGridRenderer,
    get_renderer,
)


# Renderer types and their corresponding classes
RENDERER_TYPES = {
    "2d": Molecule2DRenderer,
    "grid": MoleculeGridRenderer,
}

ALL_RENDERER_TYPES = list(RENDERER_TYPES.keys())


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_molecule():
    """Create a simple test molecule (ethanol)."""
    mol = Chem.MolFromSmiles("CCO")
    if mol is not None:
        rdDepictor.Compute2DCoords(mol)
    return mol


@pytest.fixture
def complex_molecule():
    """Create a more complex test molecule (caffeine)."""
    mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    if mol is not None:
        rdDepictor.Compute2DCoords(mol)
    return mol


@pytest.fixture
def benzene_molecule():
    """Create a benzene molecule for testing aromatic systems."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    if mol is not None:
        rdDepictor.Compute2DCoords(mol)
    return mol


@pytest.fixture
def molecule_with_highlights():
    """Create a molecule with predefined highlight information."""
    mol = Chem.MolFromSmiles("CCO")
    if mol is not None:
        rdDepictor.Compute2DCoords(mol)
    # Return molecule with highlight data
    return {
        "molecule": mol,
        "highlight_atoms": [0, 1],
        "highlight_bonds": [0],
        "highlight_colors": {0: (1.0, 0.0, 0.0)},
    }


@pytest.fixture
def molecule_list(simple_molecule, complex_molecule, benzene_molecule):
    """Create a list of test molecules for grid rendering."""
    return [simple_molecule, complex_molecule, benzene_molecule]


@pytest.fixture
def invalid_molecule():
    """Create an invalid molecule (None) for error testing."""
    return None


@pytest.fixture
def mock_molecule():
    """Create a mock molecule for unit testing."""
    mol = Mock()
    mol.GetNumAtoms.return_value = 3
    mol.GetNumBonds.return_value = 2
    mol.GetNumConformers.return_value = 1

    # Mock conformer
    conformer = Mock()
    positions = [Mock(), Mock(), Mock()]
    positions[0].x, positions[0].y = 0.0, 0.0
    positions[1].x, positions[1].y = 1.0, 0.5
    positions[2].x, positions[2].y = 2.0, 1.0
    conformer.GetAtomPosition.side_effect = positions
    mol.GetConformer.return_value = conformer

    return mol


@pytest.fixture
def basic_config():
    """Create a basic render configuration."""
    return RenderConfig()


@pytest.fixture
def custom_config():
    """Create a custom render configuration."""
    return RenderConfig(
        width=600,
        height=400,
        background_color="lightblue",
        show_carbon=True,
        show_hydrogen=True,
    )


@pytest.fixture
def highlight_config():
    """Create a configuration with highlights."""
    return RenderConfig(
        width=400,
        height=300,
        highlight_atoms=[0, 1],
        highlight_bonds=[0],
    )


@pytest.fixture
def small_config():
    """Create a small-sized configuration."""
    return RenderConfig(width=100, height=100)


@pytest.fixture
def large_config():
    """Create a large-sized configuration."""
    return RenderConfig(width=1000, height=800)


@pytest.fixture(params=ALL_RENDERER_TYPES)
def renderer_type(request):
    """Parametrized fixture providing all supported renderer types."""
    return request.param


@pytest.fixture
def renderer(renderer_type):
    """Create a renderer instance for the given type."""
    return get_renderer(renderer_type)


@pytest.fixture
def renderer_with_config(renderer_type, custom_config):
    """Create a renderer instance with custom configuration."""
    return get_renderer(renderer_type, custom_config)


@pytest.fixture
def mock_drawer():
    """Create a mock RDKit drawer for unit testing."""
    drawer = Mock()
    drawer.GetDrawingText.return_value = b"fake_png_data"
    drawer.DrawMolecule = Mock()
    drawer.FinishDrawing = Mock()
    return drawer


@pytest.fixture
def mock_drawer_manager(mock_drawer):
    """Create a mock drawer configuration manager."""
    manager = Mock()
    manager.create_drawer.return_value = mock_drawer
    return manager


@pytest.fixture
def mock_pil_image():
    """Create a mock PIL Image for unit testing."""
    img = Mock()
    img.mode = "RGBA"
    img.size = (300, 300)
    img.convert.return_value = img
    return img


# Renderer-specific fixtures
@pytest.fixture
def two_d_renderer():
    """Create a 2D renderer instance."""
    return Molecule2DRenderer()


@pytest.fixture
def grid_renderer():
    """Create a grid renderer instance."""
    return MoleculeGridRenderer()


@pytest.fixture
def grid_renderer_custom():
    """Create a grid renderer with custom parameters."""
    return MoleculeGridRenderer(mols_per_row=3, mol_size=(150, 150))


# Test data fixtures
@pytest.fixture
def valid_renderer_types():
    """List of valid renderer type strings."""
    return ["2d", "grid", "2D", "GRID", " 2d ", " grid "]


@pytest.fixture
def invalid_renderer_types():
    """List of invalid renderer type strings."""
    return ["3d", "invalid", "", "unknown", "raster"]


@pytest.fixture
def molecule_legends():
    """Sample legends for molecule grid rendering."""
    return ["Ethanol", "Caffeine", "Benzene"]


# Utility functions for tests
def get_renderer_capabilities(renderer_type: str) -> dict:
    """Get capabilities for a specific renderer type."""
    capabilities = {
        "2d": {
            "supports_highlights": True,
            "supports_multiple_molecules": False,
            "supports_legends": False,
        },
        "grid": {
            "supports_highlights": True,
            "supports_multiple_molecules": True,
            "supports_legends": True,
        },
    }
    return capabilities.get(renderer_type.lower(), {})


def supports_highlights(renderer_type: str) -> bool:
    """Check if renderer type supports highlighting."""
    return get_renderer_capabilities(renderer_type).get("supports_highlights", False)


def supports_multiple_molecules(renderer_type: str) -> bool:
    """Check if renderer type supports multiple molecules."""
    return get_renderer_capabilities(renderer_type).get(
        "supports_multiple_molecules", False
    )


def supports_legends(renderer_type: str) -> bool:
    """Check if renderer type supports legends."""
    return get_renderer_capabilities(renderer_type).get("supports_legends", False)


def create_test_molecule(smiles: str):
    """Create a test molecule from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        rdDepictor.Compute2DCoords(mol)
    return mol


def validate_rendered_image(image, expected_mode="RGBA", min_size=(10, 10)):
    """Validate that a rendered image meets basic requirements."""
    assert image is not None, "Rendered image should not be None"
    assert hasattr(image, "mode"), "Image should have a mode attribute"
    assert hasattr(image, "size"), "Image should have a size attribute"

    if expected_mode:
        assert image.mode == expected_mode, (
            f"Expected mode {expected_mode}, got {image.mode}"
        )

    if min_size:
        assert image.size[0] >= min_size[0], f"Image width {image.size[0]} too small"
        assert image.size[1] >= min_size[1], f"Image height {image.size[1]} too small"
