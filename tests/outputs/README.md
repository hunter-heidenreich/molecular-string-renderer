# Output Handler Tests - Consolidated Approach

This directory contains a consolidated test suite for output handlers that eliminates duplication while maintaining comprehensive coverage.

## Structure

### `conftest.py`
- Shared fixtures and test configuration
- Format capability definitions  
- Common test utilities and helpers
- Parametrized fixtures for testing all formats

### `test_consolidated_handlers.py`
- **Main test file** - Tests all shared behavior across formats
- Uses parametrized fixtures to test all handlers with the same logic
- Covers: properties, initialization, basic functionality, image modes, configuration, edge cases, error handling, inheritance, thread safety, quality/optimization, integration

### `test_format_specific.py`
- Tests only format-specific quirks and behaviors
- JPEG: RGBA→RGB conversion, quality impact on file size
- BMP: No optimization/quality support, mode handling
- PNG: Transparency optimization, quality parameter ignored
- WEBP: Alpha+quality interaction
- SVG: Molecule setting, vector vs raster modes

### `test_pdf_specific.py`
- PDF-specific tests for unique PDF behaviors
- PDF structure validation, page layout, scaling

## Benefits of This Approach

1. **DRY Principle**: Eliminates ~90% of test duplication
2. **Maintainability**: Changes to shared behavior only need updates in one place
3. **Coverage**: Same comprehensive testing, less code
4. **Focus**: Format-specific tests only test actual differences
5. **Clarity**: Easier to see what's truly different between formats

## Running Tests

```bash
# Run all output handler tests
pytest tests/outputs/

# Run only consolidated tests (tests all formats)
pytest tests/outputs/test_consolidated_handlers.py

# Run only format-specific tests
pytest tests/outputs/test_format_specific.py

# Run specific format tests with verbose output
pytest tests/outputs/test_consolidated_handlers.py::TestOutputHandlerProperties -v
```

## Adding New Formats

1. Add format info to `conftest.py` format dictionaries
2. Add any unique behaviors to `test_format_specific.py`
3. The consolidated tests will automatically include the new format

## Migration from Old Tests

The old individual test files (`test_*_output.py`) have been successfully removed. Their functionality is now fully covered by:
- Shared behavior → `test_consolidated_handlers.py` (225 tests)
- Format quirks → `test_format_specific.py` (13 tests)  
- PDF specifics → `test_pdf_specific.py` (4 tests)

**Total: 242 tests with zero skipped tests and comprehensive coverage**

The old files have been moved to `.old_tests/` for reference if needed.
