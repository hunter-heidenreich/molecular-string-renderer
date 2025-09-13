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
        "methane": "C",
        "water": "O",
        "propanol": "CCC[OH]",
    },
    "inchi": {
        "ethanol": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        "benzene": "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",
        "acetic_acid": "InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)",
        "aspirin": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
        "caffeine": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
        "methane": "InChI=1S/CH4/h1H4",
        "water": "InChI=1S/H2O/h1H2",
    },
    "selfies": {
        "ethanol": "[C][C][O]",
        "benzene": "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
        "acetic_acid": "[C][C][Branch1][C][O][=O]",
        "methane": "[C]",
        "propanol": "[C][C][C][O]",
        "branched": "[C][Branch1][C][C][C][O]",  # Isopropanol
    },
    "mol_block": {
        "ethanol": "\n     RDKit          2D\n\n  3  2  0  0  0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.2990    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.5981   -0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n  1  2  1  0\n  2  3  1  0\nM  END\n",
        "methane": "\n     RDKit          2D\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n",
    },
    "invalid_smiles": [
        "INVALID_SMILES",
        "C1=CC=CC=C",  # Incomplete ring
        "",  # Empty string
        "XYZ123",  # Invalid atoms
    ],
    "invalid_inchi": [
        "NotAnInChI",  # Doesn't start with InChI=
        "InChI=INVALID",  # Invalid InChI
        "",  # Empty string
        "InChI=1S/INVALID",  # Malformed InChI
    ],
    "invalid_selfies": [
        "[C][C",  # Incomplete brackets
        "[X][Y]",  # Invalid SELFIES tokens
        "",  # Empty string
        "[Z][Z][Z]",  # Invalid symbols
    ],
    "invalid_mol": [
        "",  # Empty string
        "Not a MOL block",  # Invalid format
        "INVALID\nDATA",  # Malformed MOL block
    ],
}

# Common test configurations
TEST_CONFIGS = {
    "small": {"width": 200, "height": 200},
    "medium": {"width": 500, "height": 500},
    "large": {"width": 800, "height": 800},
    "rectangular": {"width": 400, "height": 600},
}
