"""
Performance and robustness tests for outputs submodule.

Tests performance characteristics, memory usage, thread safety,
and error handling under various conditions.
"""

import gc
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import get_output_handler


class TestSubModulePerformance:
    """Test performance characteristics of the outputs submodule."""

    def test_import_speed(self):
        """Test that imports are reasonably fast."""
        start_time = time.time()

        # Re-import to test fresh import time
        _ = __import__("molecular_string_renderer.outputs", fromlist=[""])

        import_time = time.time() - start_time

        # Should import in reasonable time (less than 1 second)
        assert import_time < 1.0, f"Import took too long: {import_time:.3f} seconds"

    def test_handler_creation_speed(self):
        """Test that handler creation is reasonably fast."""
        start_time = time.time()

        # Create multiple handlers
        for _ in range(100):
            get_output_handler("png")
            get_output_handler("svg")

        creation_time = time.time() - start_time

        # Should create handlers quickly
        assert creation_time < 1.0, (
            f"Handler creation took too long: {creation_time:.3f} seconds"
        )

    def test_concurrent_handler_usage(self):
        """Test concurrent usage doesn't cause issues."""
        test_image = Image.new("RGB", (50, 50), "red")
        results = []
        errors = []

        def worker():
            try:
                handler = get_output_handler("png")
                result = handler.get_bytes(test_image)
                results.append(len(result))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 10
        assert all(r > 0 for r in results)


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""

    @pytest.mark.slow
    def test_memory_usage_reasonable(self):
        """Test that handlers don't use excessive memory."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create many handlers and use them
        handlers = []
        test_image = Image.new("RGB", (200, 200), "red")

        for _ in range(50):
            for fmt in ["png", "jpg", "svg"]:
                handler = get_output_handler(fmt)
                handlers.append(handler)
                _ = handler.get_bytes(test_image)

        # Force garbage collection
        del handlers
        gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should not use more than 50MB additional memory
        assert memory_increase < 50 * 1024 * 1024, (
            f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"
        )

    def test_large_batch_processing(self):
        """Test handling of large batches of files."""
        handler = get_output_handler("png")
        test_image = Image.new("RGB", (100, 100), "red")

        start_time = time.time()

        # Process many images
        for _ in range(100):
            result = handler.get_bytes(test_image)
            assert len(result) > 0

        processing_time = time.time() - start_time

        # Should process 100 images in reasonable time
        assert processing_time < 5.0, (
            f"Batch processing too slow: {processing_time:.2f} seconds"
        )

    def test_handler_reuse_efficiency(self):
        """Test that reusing handlers is efficient."""
        handler = get_output_handler("png")
        test_image = Image.new("RGB", (50, 50), "red")

        # Time single-handler reuse
        start_time = time.time()
        for _ in range(50):
            _ = handler.get_bytes(test_image)
        reuse_time = time.time() - start_time

        # Time creating new handlers each time
        start_time = time.time()
        for _ in range(50):
            new_handler = get_output_handler("png")
            _ = new_handler.get_bytes(test_image)
        creation_time = time.time() - start_time

        # Reuse should be faster (or at least not much slower)
        assert reuse_time <= creation_time * 1.5, (
            f"Handler reuse not efficient: {reuse_time:.3f}s vs {creation_time:.3f}s"
        )


class TestRobustness:
    """Test robustness and edge case handling."""

    def test_concurrent_file_operations(self, temp_dir):
        """Test concurrent file save operations."""
        handler = get_output_handler("png")
        test_image = Image.new("RGB", (50, 50), "red")
        errors = []

        def save_worker(index):
            try:
                output_path = temp_dir / f"concurrent_{index}.png"
                handler.save(test_image, output_path)
                assert output_path.exists()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_worker, args=(i,)) for i in range(20)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Concurrent save errors: {errors}"

    def test_format_specific_edge_cases(self):
        """Test format-specific edge cases are handled gracefully."""
        # Test JPEG with transparency (should convert)
        rgba_image = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
        jpeg_handler = get_output_handler("jpeg")
        result = jpeg_handler.get_bytes(rgba_image)
        assert len(result) > 0

        # Test BMP with unusual modes
        la_image = Image.new("LA", (50, 50), (128, 200))
        bmp_handler = get_output_handler("bmp")
        result = bmp_handler.get_bytes(la_image)
        assert len(result) > 0

        # Test PDF with very small images
        tiny_image = Image.new("RGB", (1, 1), "red")
        pdf_handler = get_output_handler("pdf")
        result = pdf_handler.get_bytes(tiny_image)
        assert len(result) > 0

    def test_invalid_configuration_handling(self):
        """Test that invalid configurations are handled gracefully."""
        # Test with boundary quality values
        configs = [
            OutputConfig(quality=1),  # Minimum quality
            OutputConfig(quality=100),  # Maximum quality
        ]

        test_image = Image.new("RGB", (50, 50), "red")

        for config in configs:
            # Should not crash, even with boundary config values
            handler = get_output_handler("jpeg", config)
            result = handler.get_bytes(test_image)
            assert len(result) > 0

    def test_error_handling_consistency(self):
        """Test that error handling is consistent across all handlers."""
        test_image = Image.new("RGB", (100, 100), "red")

        handlers = [
            get_output_handler("png"),
            get_output_handler("jpeg"),
            get_output_handler("webp"),
            get_output_handler("tiff"),
            get_output_handler("bmp"),
            get_output_handler("svg"),
            get_output_handler("pdf"),
        ]

        for handler in handlers:
            # Test invalid path handling
            with pytest.raises((IOError, OSError, PermissionError)):
                handler.save(test_image, "/root/nonexistent/test.png")


class TestSubModuleFunctionality:
    """Test that all functionality works correctly after refactoring."""

    def test_all_handlers_work_independently(self):
        """Test that each handler works correctly in isolation."""
        from molecular_string_renderer.outputs import (
            BMPOutput,
            JPEGOutput,
            PDFOutput,
            PNGOutput,
            SVGOutput,
            TIFFOutput,
            WEBPOutput,
        )

        test_image = Image.new("RGB", (100, 100), "red")

        handler_classes = [
            PNGOutput,
            JPEGOutput,
            WEBPOutput,
            TIFFOutput,
            BMPOutput,
            SVGOutput,
            PDFOutput,
        ]

        for handler_class in handler_classes:
            handler = handler_class()
            result = handler.get_bytes(test_image)
            assert isinstance(result, bytes), f"Handler {handler_class.__name__} failed"
            assert len(result) > 0

    def test_factory_creates_correct_handlers(self):
        """Test that factory creates the correct handler types."""
        from molecular_string_renderer.outputs import (
            BMPOutput,
            JPEGOutput,
            PDFOutput,
            PNGOutput,
            SVGOutput,
            TIFFOutput,
            WEBPOutput,
        )

        expected_types = {
            "png": PNGOutput,
            "jpg": JPEGOutput,
            "jpeg": JPEGOutput,
            "webp": WEBPOutput,
            "tiff": TIFFOutput,
            "tif": TIFFOutput,
            "bmp": BMPOutput,
            "svg": SVGOutput,
            "pdf": PDFOutput,
        }

        for format_name, expected_type in expected_types.items():
            handler = get_output_handler(format_name)
            assert isinstance(handler, expected_type)

    def test_configuration_propagation(self):
        """Test that configuration is properly propagated through sub-module."""
        from molecular_string_renderer.outputs import JPEGOutput

        custom_config = OutputConfig(quality=75, optimize=True)

        # Test with factory
        handler = get_output_handler("jpeg", custom_config)
        assert handler.config is custom_config
        assert handler.config.quality == 75
        assert handler.config.optimize is True

        # Test direct instantiation
        direct_handler = JPEGOutput(custom_config)
        assert direct_handler.config is custom_config

    def test_utilities_work_correctly(self):
        """Test that utility functions work correctly."""
        from molecular_string_renderer.outputs import create_safe_filename

        # Test filename creation
        filename1 = create_safe_filename("CCO")
        filename2 = create_safe_filename("CCO", ".svg")

        assert filename1.endswith(".png")
        assert filename2.endswith(".svg")
        assert len(filename1) > 10  # Should be a reasonable hash


class TestErrorRecovery:
    """Test error recovery scenarios."""

    def test_partial_failure_recovery(self):
        """Test recovery from partial failures in batch operations."""
        test_image = Image.new("RGB", (50, 50), "red")
        handler = get_output_handler("png")

        results = []
        errors = []

        # Simulate some failures in a batch
        for i in range(10):
            try:
                if i == 5:  # Simulate one failure
                    raise Exception("Simulated failure")
                result = handler.get_bytes(test_image)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Should have 9 successes and 1 failure
        assert len(results) == 9
        assert len(errors) == 1

    def test_resource_cleanup_on_error(self):
        """Test that resources are properly cleaned up on errors."""
        import tempfile

        test_image = Image.new("RGB", (50, 50), "red")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"

            handler = get_output_handler("png")

            # Force an error during save
            with patch.object(test_image, "save", side_effect=Exception("Save failed")):
                with pytest.raises(Exception):
                    handler.save(test_image, output_path)

                # File should not exist after failed save
                assert not output_path.exists()

    def test_corrupted_image_handling(self):
        """Test handling of corrupted or invalid images."""
        handler = get_output_handler("png")

        # Create a mock corrupted image
        corrupted_image = MagicMock()
        corrupted_image.save.side_effect = Exception("Corrupted image")

        with pytest.raises(Exception):
            handler.get_bytes(corrupted_image)
