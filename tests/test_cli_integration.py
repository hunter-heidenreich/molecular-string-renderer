"""
End-to-end CLI tests using subprocess to test actual command-line usage.

Tests the mol-render CLI command with various options and scenarios
to ensure it works correctly in real usage.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.conftest import SAMPLE_MOLECULES


def get_cli_command():
    """Get the CLI command to run."""
    # Try to find the mol-render script or use python -m
    venv_bin = Path(__file__).parent.parent / ".venv" / "bin"
    mol_render_script = venv_bin / "mol-render"

    if mol_render_script.exists():
        return [str(mol_render_script)]
    else:
        # Fall back to python -m
        python_exe = venv_bin / "python"
        if python_exe.exists():
            return [str(python_exe), "-m", "molecular_string_renderer.cli"]
        else:
            # Use system python as last resort
            return ["python", "-m", "molecular_string_renderer.cli"]


class TestCLIBasicCommands:
    """Test basic CLI commands and options."""

    def test_cli_help(self):
        """Test that CLI help command works."""
        cmd = get_cli_command() + ["--help"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert "molecular string" in result.stdout.lower()
        assert "usage:" in result.stdout.lower()

    def test_cli_version(self):
        """Test that CLI version command works."""
        cmd = get_cli_command() + ["--version"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_cli_list_formats(self):
        """Test that CLI list formats command works."""
        cmd = get_cli_command() + ["--list-formats"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert "supported formats:" in result.stdout.lower()
        assert "smiles" in result.stdout.lower()
        assert "png" in result.stdout.lower()

    def test_cli_simple_smiles_rendering(self):
        """Test basic SMILES rendering via CLI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "ethanol.png"

            cmd = get_cli_command() + [
                SAMPLE_MOLECULES["smiles"]["ethanol"],
                "-o",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
            assert output_path.exists(), "Output file was not created"
            assert output_path.stat().st_size > 0, "Output file is empty"

    def test_cli_auto_filename(self):
        """Test CLI with auto-generated filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory so auto-generated file goes there
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                cmd = get_cli_command() + [SAMPLE_MOLECULES["smiles"]["ethanol"]]

                result = subprocess.run(cmd, capture_output=True, text=True)

                assert result.returncode == 0, (
                    f"CLI failed with stderr: {result.stderr}"
                )

                # Check that some .png file was created
                png_files = list(Path(temp_dir).glob("*.png"))
                assert len(png_files) > 0, "No PNG file was created with auto-filename"
                assert png_files[0].stat().st_size > 0, "Auto-generated file is empty"

            finally:
                os.chdir(original_cwd)


class TestCLIOutputFormats:
    """Test CLI with different output formats."""

    def test_cli_different_output_formats(self):
        """Test CLI rendering to different output formats."""
        formats = ["png", "svg", "jpg", "pdf"]
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in formats:
                output_path = Path(temp_dir) / f"molecule.{fmt}"

                cmd = get_cli_command() + [
                    smiles,
                    "--output-format",
                    fmt,
                    "-o",
                    str(output_path),
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                assert result.returncode == 0, (
                    f"CLI failed for {fmt} with stderr: {result.stderr}"
                )
                assert output_path.exists(), f"Output file not created for {fmt}"
                assert output_path.stat().st_size > 0, f"Output file is empty for {fmt}"

    def test_cli_format_detection_from_extension(self):
        """Test that CLI detects format from file extension."""
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test various extensions
            extensions = ["png", "svg", "jpg", "pdf"]

            for ext in extensions:
                output_path = Path(temp_dir) / f"molecule.{ext}"

                cmd = get_cli_command() + [smiles, "-o", str(output_path)]

                result = subprocess.run(cmd, capture_output=True, text=True)

                assert result.returncode == 0, (
                    f"CLI failed for .{ext} extension with stderr: {result.stderr}"
                )
                assert output_path.exists(), f"Output file not created for .{ext}"


class TestCLIInputFormats:
    """Test CLI with different input formats."""

    def test_cli_different_input_formats(self):
        """Test CLI with different molecular input formats."""
        test_cases = [
            ("smiles", SAMPLE_MOLECULES["smiles"]["ethanol"]),
            ("inchi", SAMPLE_MOLECULES["inchi"]["ethanol"]),
            ("selfies", SAMPLE_MOLECULES["selfies"]["ethanol"]),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for format_type, molecule in test_cases:
                output_path = Path(temp_dir) / f"molecule_{format_type}.png"

                cmd = get_cli_command() + [
                    molecule,
                    "--format",
                    format_type,
                    "-o",
                    str(output_path),
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                assert result.returncode == 0, (
                    f"CLI failed for {format_type} with stderr: {result.stderr}"
                )
                assert output_path.exists(), (
                    f"Output file not created for {format_type}"
                )

    def test_cli_format_aliases(self):
        """Test CLI with format aliases."""
        test_cases = [
            ("smi", SAMPLE_MOLECULES["smiles"]["ethanol"]),
            ("SMILES", SAMPLE_MOLECULES["smiles"]["ethanol"]),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for format_alias, molecule in test_cases:
                output_path = Path(temp_dir) / f"molecule_{format_alias}.png"

                cmd = get_cli_command() + [
                    molecule,
                    "--format",
                    format_alias,
                    "-o",
                    str(output_path),
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                assert result.returncode == 0, (
                    f"CLI failed for {format_alias} with stderr: {result.stderr}"
                )
                assert output_path.exists(), (
                    f"Output file not created for {format_alias}"
                )


class TestCLIRenderingOptions:
    """Test CLI rendering options and configurations."""

    def test_cli_size_options(self):
        """Test CLI size configuration options."""
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test square size
            output_path1 = Path(temp_dir) / "molecule_square.png"

            cmd = get_cli_command() + [smiles, "--size", "300", "-o", str(output_path1)]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
            assert output_path1.exists()

            # Test width/height override
            output_path2 = Path(temp_dir) / "molecule_rect.png"

            cmd = get_cli_command() + [
                smiles,
                "--width",
                "400",
                "--height",
                "600",
                "-o",
                str(output_path2),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
            assert output_path2.exists()

    def test_cli_background_color(self):
        """Test CLI background color option."""
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "molecule_colored.png"

            cmd = get_cli_command() + [
                smiles,
                "--background-color",
                "lightblue",
                "-o",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
            assert output_path.exists()

    def test_cli_hydrogen_options(self):
        """Test CLI hydrogen display options."""
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "molecule_hydrogens.png"

            cmd = get_cli_command() + [
                smiles,
                "--show-hydrogen",
                "--show-carbon",
                "-o",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
            assert output_path.exists()

    def test_cli_quality_options(self):
        """Test CLI quality and optimization options."""
        smiles = SAMPLE_MOLECULES["smiles"]["ethanol"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "molecule_quality.jpg"

            cmd = get_cli_command() + [
                smiles,
                "--output-format",
                "jpg",
                "--quality",
                "85",
                "--no-optimize",
                "--no-antialias",
                "-o",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
            assert output_path.exists()


class TestCLIGridRendering:
    """Test CLI grid rendering functionality."""

    def test_cli_basic_grid(self):
        """Test basic CLI grid rendering."""
        molecules = ",".join(
            [
                SAMPLE_MOLECULES["smiles"]["ethanol"],
                SAMPLE_MOLECULES["smiles"]["benzene"],
                SAMPLE_MOLECULES["smiles"]["acetic_acid"],
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grid.png"

            cmd = get_cli_command() + ["--grid", molecules, "-o", str(output_path)]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_cli_grid_with_legends(self):
        """Test CLI grid rendering with legends."""
        molecules = ",".join(
            [
                SAMPLE_MOLECULES["smiles"]["ethanol"],
                SAMPLE_MOLECULES["smiles"]["benzene"],
            ]
        )
        legends = "Ethanol,Benzene"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grid_legends.png"

            cmd = get_cli_command() + [
                "--grid",
                molecules,
                "--legends",
                legends,
                "-o",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
            assert output_path.exists()

    def test_cli_grid_layout_options(self):
        """Test CLI grid layout options."""
        molecules = ",".join(
            [
                SAMPLE_MOLECULES["smiles"]["ethanol"],
                SAMPLE_MOLECULES["smiles"]["benzene"],
                SAMPLE_MOLECULES["smiles"]["acetic_acid"],
                SAMPLE_MOLECULES["smiles"]["methane"],
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grid_layout.png"

            cmd = get_cli_command() + [
                "--grid",
                molecules,
                "--mols-per-row",
                "2",
                "--mol-size",
                "150",
                "-o",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
            assert output_path.exists()


class TestCLIValidation:
    """Test CLI validation functionality."""

    def test_cli_validate_valid_molecule(self):
        """Test CLI validation with valid molecule."""
        cmd = get_cli_command() + [SAMPLE_MOLECULES["smiles"]["ethanol"], "--validate"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert "valid" in result.stdout.lower() or "✓" in result.stdout

    def test_cli_validate_invalid_molecule(self):
        """Test CLI validation with invalid molecule."""
        cmd = get_cli_command() + ["INVALID_SMILES", "--validate"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode != 0  # Should exit with error code
        assert "invalid" in result.stderr.lower() or "✗" in result.stderr

    def test_cli_validate_different_formats(self):
        """Test CLI validation with different formats."""
        test_cases = [
            ("smiles", SAMPLE_MOLECULES["smiles"]["ethanol"]),
            ("inchi", SAMPLE_MOLECULES["inchi"]["ethanol"]),
            ("selfies", SAMPLE_MOLECULES["selfies"]["ethanol"]),
        ]

        for format_type, molecule in test_cases:
            cmd = get_cli_command() + [molecule, "--format", format_type, "--validate"]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, (
                f"Validation failed for valid {format_type}: {result.stderr}"
            )


class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""

    def test_cli_invalid_smiles_error(self):
        """Test CLI error handling with invalid SMILES."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "invalid.png"

            cmd = get_cli_command() + ["INVALID_SMILES", "-o", str(output_path)]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode != 0  # Should fail
            assert not output_path.exists()  # No file should be created

    def test_cli_unsupported_format_error(self):
        """Test CLI error handling with unsupported format."""
        cmd = get_cli_command() + [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            "--format",
            "unsupported_format",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode != 0  # Should fail
        assert (
            "unsupported" in result.stderr.lower()
            or "invalid choice" in result.stderr.lower()
        )

    def test_cli_missing_required_args(self):
        """Test CLI error handling with missing required arguments."""
        # No molecular string provided
        cmd = get_cli_command()

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode != 0  # Should fail
        # Should show help or error message

    def test_cli_invalid_grid_format(self):
        """Test CLI error handling with invalid grid input."""
        # Empty grid
        cmd = get_cli_command() + ["--grid", ""]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode != 0  # Should fail

    def test_cli_permission_error_handling(self):
        """Test CLI handling of permission errors."""
        # Try to write to a directory that likely doesn't exist or isn't writable
        cmd = get_cli_command() + [
            SAMPLE_MOLECULES["smiles"]["ethanol"],
            "-o",
            "/nonexistent/directory/file.png",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode != 0  # Should fail gracefully


class TestCLIVerboseMode:
    """Test CLI verbose mode functionality."""

    def test_cli_verbose_output(self):
        """Test that verbose mode produces more output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "verbose.png"

            # Test without verbose
            cmd_quiet = get_cli_command() + [
                SAMPLE_MOLECULES["smiles"]["ethanol"],
                "-o",
                str(output_path),
            ]

            result_quiet = subprocess.run(cmd_quiet, capture_output=True, text=True)

            # Test with verbose
            output_path_verbose = Path(temp_dir) / "verbose2.png"
            cmd_verbose = get_cli_command() + [
                SAMPLE_MOLECULES["smiles"]["ethanol"],
                "-v",
                "-o",
                str(output_path_verbose),
            ]

            result_verbose = subprocess.run(cmd_verbose, capture_output=True, text=True)

            assert result_quiet.returncode == 0
            assert result_verbose.returncode == 0

            # Verbose should produce more output (either stdout or stderr)
            verbose_output_len = len(result_verbose.stdout) + len(result_verbose.stderr)
            quiet_output_len = len(result_quiet.stdout) + len(result_quiet.stderr)

            assert verbose_output_len >= quiet_output_len


if __name__ == "__main__":
    pytest.main([__file__])
