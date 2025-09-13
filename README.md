# Molecular String Renderer

A flexible Python library for rendering molecular structures from various string representations (SMILES, InChI, etc.) to high-quality images.

## Features

- **Multiple Input Formats**: Support for SMILES, InChI, and MOL files
- **Flexible Output**: PNG, SVG, and JPEG output formats with customizable quality
- **Modular Architecture**: Extensible design for adding new parsers and renderers
- **High-Quality Rendering**: Publication-ready 2D molecular structure images
- **Grid Layouts**: Render multiple molecules in organized grids
- **Command-Line Interface**: Easy-to-use CLI for batch processing
- **Python API**: Programmatic access with comprehensive configuration options

## Installation

```bash
# Install from PyPI (when published)
pip install molecular-string-renderer

# Or install from source
git clone https://github.com/hunter-heidenreich/molecular-string-renderer.git
cd molecular-string-renderer
pip install .

# For development
pip install -e ".[dev]"
```

## Quick Start

### Command Line Usage

```bash
# Basic SMILES rendering
mol-render "CCO"  # Renders ethanol to auto-generated filename

# Custom output filename
mol-render "CCO" -o ethanol.png

# Different formats
mol-render "CCO" --output-format svg
mol-render "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3" --format inchi

# Grid of molecules
mol-render --grid "CCO,CC(=O)O,C1=CC=CC=C1" --legends "Ethanol,Acetic acid,Benzene"

# Custom styling
mol-render "CCO" --size 800 --background-color "#f0f0f0" --show-hydrogen
```

### Python API

```python
```python
from molecular_string_renderer import render_molecule, render_molecules_grid, RenderConfig

# Basic usage
image = render_molecule("CCO", format_type="smiles", output_format="png")
image.save("ethanol.png")

# With custom configuration
config = RenderConfig(
    width=800,
    height=600,
    background_color="lightblue",
    show_hydrogen=True,
    dpi=300
)

image = render_molecule(
    "C1=CC=CC=C1", 
    format_type="smiles",
    render_config=config
)

# Grid rendering
molecules = ["CCO", "CC(=O)O", "C1=CC=CC=C1"]
legends = ["Ethanol", "Acetic Acid", "Benzene"]

grid_image = render_molecules_grid(
    molecular_strings=molecules,
    legends=legends,
    mols_per_row=3,
    output_path="molecules_grid.png"
)
```
```

### Object-Oriented Interface

```python
from molecular_string_renderer import MolecularRenderer, RenderConfig

# Create renderer with custom config
config = RenderConfig(width=600, height=600, background_color="white")
renderer = MolecularRenderer(render_config=config)

# Render multiple molecules with same config
molecules = ["CCO", "CC(=O)O", "C1=CC=CC=C1"]
for i, mol in enumerate(molecules):
    image = renderer.render(mol, output_path=f"molecule_{i}.png")

# Render grid
grid = renderer.render_grid(molecules, legends=["Ethanol", "Acetic Acid", "Benzene"])
```

## Supported Formats

### Input Formats

- **SMILES** (`smiles`, `smi`): Simplified Molecular Input Line Entry System
- **InChI** (`inchi`): International Chemical Identifier
- **MOL** (`mol`): MOL file format

### Output Formats

- **PNG** (`png`): Portable Network Graphics (recommended for most uses)
- **SVG** (`svg`): Scalable Vector Graphics
- **JPEG** (`jpg`, `jpeg`): JPEG format (no transparency support)

## Configuration Options

### Render Configuration

```python
from molecular_string_renderer import RenderConfig

config = RenderConfig(
    width=500,                    # Image width in pixels
    height=500,                   # Image height in pixels
    background_color="white",     # Background color (name or hex)
    atom_label_font_size=12,      # Font size for atom labels
    bond_line_width=2.0,          # Bond line width
    antialias=True,               # Enable antialiasing
    dpi=150,                      # DPI for high-quality output
    show_hydrogen=False,          # Show explicit hydrogens
    show_carbon=False,            # Show carbon labels
    highlight_atoms=None,         # Atoms to highlight
    highlight_bonds=None,         # Bonds to highlight
)
```

### Parser Configuration

```python
from molecular_string_renderer import ParserConfig

parser_config = ParserConfig(
    sanitize=True,                # Sanitize molecules after parsing
    remove_hs=True,               # Remove explicit hydrogens
    strict=False,                 # Use strict parsing (fail on warnings)
)
```

### Output Configuration

```python
from molecular_string_renderer import OutputConfig

output_config = OutputConfig(
    format="png",                 # Output format
    quality=95,                   # Quality (1-100)
    optimize=True,                # Optimize file size
)
```

## Advanced Usage

### Custom Highlighting

```python
from molecular_string_renderer.renderers import Molecule2DRenderer
from molecular_string_renderer.parsers import SMILESParser

parser = SMILESParser()
mol = parser.parse("C1=CC=CC=C1")  # Benzene

renderer = Molecule2DRenderer()
image = renderer.render_with_highlights(
    mol,
    highlight_atoms=[0, 1, 2],  # Highlight first 3 carbons
    highlight_colors={0: "red", 1: "green", 2: "blue"}
)
```

### Validation

```python
from molecular_string_renderer import validate_molecular_string

# Validate SMILES
is_valid = validate_molecular_string("CCO", "smiles")  # True
is_invalid = validate_molecular_string("INVALID", "smiles")  # False

# Validate InChI
inchi_valid = validate_molecular_string("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "inchi")
```

### Batch Processing

```python
from molecular_string_renderer import MolecularRenderer
from pathlib import Path

renderer = MolecularRenderer()
molecules = ["CCO", "CC(=O)O", "C1=CC=CC=C1", "CC(C)C"]

output_dir = Path("molecules")
output_dir.mkdir(exist_ok=True)

for i, mol_string in enumerate(molecules):
    try:
        image = renderer.render(
            mol_string,
            output_path=output_dir / f"molecule_{i:03d}.png"
        )
        print(f"✓ Rendered molecule {i}: {mol_string}")
    except Exception as e:
        print(f"✗ Failed to render {mol_string}: {e}")
```

## Architecture

The library is built with a modular architecture:

- **Parsers** (`molecular_string_renderer.parsers`): Parse molecular strings into RDKit Mol objects
- **Renderers** (`molecular_string_renderer.renderers`): Render Mol objects to images
- **Outputs** (`molecular_string_renderer.outputs`): Handle different output formats and file operations
- **Config** (`molecular_string_renderer.config`): Configuration management with validation
- **Core** (`molecular_string_renderer.core`): High-level API combining all components
- **CLI** (`molecular_string_renderer.cli`): Command-line interface

This design makes it easy to:
- Add support for new molecular formats
- Implement new rendering engines
- Add new output formats
- Customize behavior through configuration

## Development

```bash
# Clone repository
git clone https://github.com/hunter-heidenreich/molecular-string-renderer.git
cd molecular-string-renderer

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
ruff format src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

## Dependencies

- **RDKit**: Molecular informatics toolkit for parsing and coordinate generation
- **Pillow**: Python Imaging Library for image processing
- **Pydantic**: Data validation and configuration management

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Changelog

### Version 0.1.0
- Initial release
- Support for SMILES, InChI, and MOL format parsing
- PNG, SVG, and JPEG output formats
- Command-line interface for easy use
- Modular architecture for extensibility
- Comprehensive configuration system
- Grid rendering for multiple molecules

## Acknowledgments

- Built on the excellent [RDKit](https://www.rdkit.org/) cheminformatics toolkit
- Inspired by the need for flexible, publication-quality molecular visualization tools
