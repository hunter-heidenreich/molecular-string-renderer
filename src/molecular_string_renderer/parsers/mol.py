"""
MOL file parser implementation.

Provides parsing functionality for MOL file format.
"""

from pathlib import Path

from rdkit import Chem

from molecular_string_renderer.parsers.base import MolecularParser


class MOLFileParser(MolecularParser):
    """Parser for MOL file format."""

    def parse(self, mol_data: str | Path) -> Chem.Mol:
        """
        Parse MOL file data into an RDKit Mol object.

        Args:
            mol_data: Either MOL file content as string or path to MOL file

        Returns:
            RDKit Mol object

        Raises:
            ValueError: If MOL data is invalid
        """
        # Handle Path objects directly
        if isinstance(mol_data, Path):
            if not mol_data.exists():
                raise ValueError(f"MOL file does not exist: {mol_data}")
            mol_data = mol_data.read_text()
        elif isinstance(mol_data, str):
            # Check if it's a file path - must be a single line without newlines
            # and not contain typical MOL block content
            if (
                "\n" not in mol_data
                and "\r" not in mol_data
                and len(mol_data) < 200  # Reasonable path length
                and not any(
                    mol_keyword in mol_data
                    for mol_keyword in ["V2000", "V3000", "M  END"]
                )
            ):
                # Might be a file path
                potential_path = Path(mol_data)
                if potential_path.exists() and potential_path.is_file():
                    mol_data = potential_path.read_text()

        if not mol_data or not mol_data.strip():
            raise ValueError("MOL data cannot be empty")

        try:
            mol = Chem.MolFromMolBlock(mol_data)
            if mol is None:
                raise ValueError("Invalid MOL data")

            return self._post_process_molecule(mol)

        except Exception as e:
            raise ValueError(f"Failed to parse MOL data: {e}")

    def validate(self, mol_data: str | Path) -> bool:
        """Check if data is valid MOL format."""
        try:
            # Handle Path objects directly
            if isinstance(mol_data, Path):
                if not mol_data.exists():
                    return False
                mol_data = mol_data.read_text()
            elif isinstance(mol_data, str):
                # Check if it's a file path - must be a single line without newlines
                # and not contain typical MOL block content
                if (
                    "\n" not in mol_data
                    and "\r" not in mol_data
                    and len(mol_data) < 200  # Reasonable path length
                    and not any(
                        mol_keyword in mol_data
                        for mol_keyword in ["V2000", "V3000", "M  END"]
                    )
                ):
                    # Might be a file path
                    potential_path = Path(mol_data)
                    if potential_path.exists() and potential_path.is_file():
                        mol_data = potential_path.read_text()

            mol = Chem.MolFromMolBlock(mol_data)
            return mol is not None
        except Exception:
            return False
