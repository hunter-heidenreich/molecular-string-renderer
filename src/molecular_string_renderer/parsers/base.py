"""
Base classes for molecular parsers.

Provides abstract base class and common functionality for all molecular parsers.
"""

from abc import ABC, abstractmethod

from rdkit import Chem

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
                # Always be strict
                raise ValueError(f"Molecule sanitization failed: {e}")

        if self.config.remove_hs:
            mol = Chem.RemoveHs(mol)
        else:
            # If we're keeping hydrogens, ensure they are explicit
            # This is necessary for hydrogen display to work properly
            mol = Chem.AddHs(mol)

        return mol
