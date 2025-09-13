"""
Molecular parser abstractions and implementations.

Provides a flexible system for parsing different molecular string representations.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from rdkit import Chem
import selfies as sf

from molecular_string_renderer.config import ParserConfig


class MolecularParser(ABC):
    """Abstract base class for molecular parsers."""

    def __init__(self, config: ParserConfig | None = None):
        """Initialize parser with configuration."""
        self.config = config or ParserConfig()

    @abstractmethod
    def parse(self, molecular_string: str) -> Chem.Mol:
        """
        Parse a molecular string representation into an RDKit Mol object.

        Args:
            molecular_string: String representation of the molecule

        Returns:
            RDKit Mol object

        Raises:
            ValueError: If the string cannot be parsed
        """
        pass

    @abstractmethod
    def validate(self, molecular_string: str) -> bool:
        """
        Validate if a string is a valid representation for this parser.

        Args:
            molecular_string: String to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    def _post_process_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """Apply post-processing based on configuration."""
        if mol is None:
            return None

        if self.config.sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                if self.config.strict:
                    raise ValueError(f"Molecule sanitization failed: {e}")

        if self.config.remove_hs:
            mol = Chem.RemoveHs(mol)

        return mol


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


def get_parser(format_type: str, config: ParserConfig | None = None) -> MolecularParser:
    """
    Factory function to get appropriate parser for format type.

    Args:
        format_type: Type of molecular format ('smiles', 'inchi', 'mol')
        config: Parser configuration

    Returns:
        Appropriate parser instance

    Raises:
        ValueError: If format type is not supported
    """
    format_type = format_type.lower().strip()

    parsers = {
        "smiles": SMILESParser,
        "smi": SMILESParser,
        "inchi": InChIParser,
        "mol": MOLFileParser,
        "sdf": MOLFileParser,
        "selfies": SELFIESParser,
    }

    if format_type not in parsers:
        supported = list(parsers.keys())
        raise ValueError(f"Unsupported format: {format_type}. Supported: {supported}")

    return parsers[format_type](config)
