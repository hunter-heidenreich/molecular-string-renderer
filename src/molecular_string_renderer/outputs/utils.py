"""
Utility functions for output handling.

Provides common utilities for filename generation and format handling.
"""

import hashlib


def create_safe_filename(molecular_string: str, extension: str = ".png") -> str:
    """
    Generate a filesystem-safe filename from a molecular string using MD5 hash.

    Args:
        molecular_string: The input molecular string (SMILES, InChI, etc.)
        extension: File extension to use

    Returns:
        A safe filename with the specified extension
    """
    clean_string = molecular_string.strip()
    hasher = hashlib.md5(clean_string.encode("utf-8"))
    base_name = hasher.hexdigest()

    if not extension.startswith("."):
        extension = f".{extension}"

    return f"{base_name}{extension}"
