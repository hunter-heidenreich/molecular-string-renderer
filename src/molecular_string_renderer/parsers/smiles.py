"""
SMILES parser implementation.

Provides parsing functionality for SMILES (Simplified Molecular Input Line Entry System) strings.
"""

from rdkit import Chem

from molecular_string_renderer.parsers.base import MolecularParser


class SMILESParser(MolecularParser):
    """Parser for SMILES (Simplified Molecular Input Line Entry System) strings."""

    def parse(self, smiles_string: str) -> Chem.Mol:
        """
        Parse a SMILES string into an RDKit Mol object.

        Args:
            smiles_string: SMILES representation of the molecule

        Returns:
            RDKit Mol object

        Raises:
            ValueError: If SMILES string is invalid
        """
        if not smiles_string or not smiles_string.strip():
            raise ValueError("SMILES string cannot be empty")

        smiles_string = smiles_string.strip()

        try:
            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: '{smiles_string}'")

            return self._post_process_molecule(mol)

        except Exception as e:
            if "Invalid SMILES" in str(e):
                raise
            raise ValueError(f"Failed to parse SMILES '{smiles_string}': {e}")

    def validate(self, smiles_string: str) -> bool:
        """Check if string is a valid SMILES."""
        if not smiles_string or not smiles_string.strip():
            return False

        try:
            mol = Chem.MolFromSmiles(smiles_string.strip())
            return mol is not None
        except Exception:
            return False
