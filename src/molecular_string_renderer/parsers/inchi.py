"""
InChI parser implementation.

Provides parsing functionality for InChI (International Chemical Identifier) strings.
"""

from rdkit import Chem

from molecular_string_renderer.parsers.base import MolecularParser


class InChIParser(MolecularParser):
    """Parser for InChI (International Chemical Identifier) strings."""

    def parse(self, inchi_string: str) -> Chem.Mol:
        """
        Parse an InChI string into an RDKit Mol object.

        Args:
            inchi_string: InChI representation of the molecule

        Returns:
            RDKit Mol object

        Raises:
            ValueError: If InChI string is invalid
        """
        if not inchi_string or not inchi_string.strip():
            raise ValueError("InChI string cannot be empty")

        inchi_string = inchi_string.strip()

        if not inchi_string.startswith("InChI="):
            raise ValueError("InChI string must start with 'InChI='")

        try:
            mol = Chem.MolFromInchi(inchi_string)
            if mol is None:
                raise ValueError(f"Invalid InChI string: '{inchi_string}'")

            return self._post_process_molecule(mol)

        except Exception as e:
            if "Invalid InChI" in str(e):
                raise
            raise ValueError(f"Failed to parse InChI '{inchi_string}': {e}")

    def validate(self, inchi_string: str) -> bool:
        """Check if string is a valid InChI."""
        try:
            inchi_string = inchi_string.strip()
            if not inchi_string.startswith("InChI="):
                return False
            mol = Chem.MolFromInchi(inchi_string)
            return mol is not None
        except Exception:
            return False
