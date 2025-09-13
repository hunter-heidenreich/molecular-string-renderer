"""Test configuration and utilities."""

import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test data
SAMPLE_MOLECULES = {
    "smiles": {
        "ethanol": "CCO",
        "benzene": "C1=CC=CC=C1",
        "acetic_acid": "CC(=O)O",
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    },
    "invalid_smiles": [
        "INVALID_SMILES",
        "C1=CC=CC=C",  # Incomplete ring
        "",  # Empty string
        "XYZ123",  # Invalid atoms
    ],
}

# Common test configurations
TEST_CONFIGS = {
    "small": {"width": 200, "height": 200},
    "medium": {"width": 500, "height": 500},
    "large": {"width": 800, "height": 800},
    "rectangular": {"width": 400, "height": 600},
}
