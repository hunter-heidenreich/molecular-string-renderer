"""
SELFIES parser implementation.

Provides parsing functionality for SELFIES (Self-Referencing Embedded Strings) format.
"""

import selfies as sf
from rdkit import Chem

from molecular_string_renderer.parsers.base import MolecularParser


class SELFIESParser(MolecularParser):
    """Parser for SELFIES (Self-Referencing Embedded Strings) format."""

    def parse(self, selfies_string: str) -> Chem.Mol:
        """
        Parse a SELFIES string into an RDKit Mol object.

        Args:
            selfies_string: SELFIES representation of the molecule

        Returns:
            RDKit Mol object

        Raises:
            ValueError: If SELFIES string is invalid
        """
        if not selfies_string or not selfies_string.strip():
            raise ValueError("SELFIES string cannot be empty")

        selfies_string = selfies_string.strip()

        try:
            # Convert SELFIES to SMILES first
            smiles = sf.decoder(selfies_string)

            # Then parse the SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SELFIES string: '{selfies_string}'")

            return self._post_process_molecule(mol)

        except Exception as e:
            if "Invalid SELFIES" in str(e):
                raise
            raise ValueError(f"Failed to parse SELFIES '{selfies_string}': {e}")

    def validate(self, selfies_string: str) -> bool:
        """Check if string is a valid SELFIES."""
        if not selfies_string or not selfies_string.strip():
            return False

        try:
            selfies_string = selfies_string.strip()
            # Try to decode SELFIES to SMILES
            smiles = sf.decoder(selfies_string)
            # Then validate the resulting SMILES
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False
