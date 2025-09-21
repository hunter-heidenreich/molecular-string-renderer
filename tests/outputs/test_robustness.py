"""
Robustness tests for output handlers.

Tests error handling, thread safety, and concurrent operations
across all output handlers.
"""

import threading
from unittest.mock import patch

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import get_output_handler


class TestOutputHandlerErrorHandling:
    """Test error handling across handlers."""

    def test_save_invalid_path_raises_error(self, output_handler, test_image):
        """Test that invalid save paths raise appropriate errors."""
        # Arrange - use a path that cannot be created (like root on Unix systems)
        invalid_path = "/root/nonexistent/test.png"

        # Act & Assert - should raise an appropriate exception
        with pytest.raises((IOError, OSError, PermissionError)) as exc_info:
            output_handler.save(test_image, invalid_path)

        # Verify exception contains meaningful information
        error_message = str(exc_info.value).lower()
        assert any(
            keyword in error_message
            for keyword in [
                "permission",
                "access",
                "denied",
                "not found",
                "no such",
                "read-only",
                "file system",
            ]
        ), f"Error message should indicate the nature of the problem: {exc_info.value}"

    @patch("PIL.Image.Image.save")
    def test_save_pil_error_handling(
        self, mock_save, output_handler, test_image, temp_dir
    ):
        """Test handling of PIL save errors."""
        # Arrange
        mock_save.side_effect = Exception("PIL save failed")
        output_path = temp_dir / f"test{output_handler.file_extension}"

        # Act & Assert - PIL errors should be caught and re-raised as IOError
        with pytest.raises(IOError) as exc_info:
            output_handler.save(test_image, output_path)

        # Verify error handling wraps the original error appropriately
        error_message = str(exc_info.value)
        assert "pil save failed" in error_message.lower(), (
            "IOError should include original PIL error message"
        )

    def test_get_bytes_error_handling(self, output_handler):
        """Test error handling in get_bytes method."""
        # Arrange - create an invalid image scenario by mocking save operation to fail
        with patch("PIL.Image.Image.save") as mock_save:
            mock_save.side_effect = OSError(
                "Corrupted image data"
            )  # Use OSError which is caught by get_bytes
            test_image = Image.new("RGB", (50, 50), "red")

            # Act & Assert - should raise IOError for get_bytes failures
            with pytest.raises(IOError) as exc_info:
                output_handler.get_bytes(test_image)

            # Verify error message is informative
            error_message = str(exc_info.value).lower()
            assert (
                "failed to" in error_message or "corrupted image data" in error_message
            ), f"Error should indicate conversion failure: {exc_info.value}"

    def test_save_readonly_directory_error(self, output_handler, test_image, temp_dir):
        """Test saving to a read-only directory raises appropriate error."""
        # Arrange - create a read-only directory
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()

        try:
            # Make directory read-only (permissions: r-xr-xr-x)
            readonly_dir.chmod(0o555)
            output_path = readonly_dir / f"test{output_handler.file_extension}"

            # Act & Assert - should raise permission error
            with pytest.raises((IOError, OSError, PermissionError)) as exc_info:
                output_handler.save(test_image, output_path)

            # Verify error indicates permission issue
            error_message = str(exc_info.value).lower()
            assert any(
                keyword in error_message
                for keyword in ["permission", "denied", "access"]
            ), f"Error should indicate permission problem: {exc_info.value}"

        finally:
            # Cleanup - restore write permissions for cleanup
            try:
                readonly_dir.chmod(0o755)
            except (OSError, PermissionError):
                pass  # May fail on some systems, but that's okay for cleanup

    def test_save_with_none_image_raises_error(self, output_handler, temp_dir):
        """Test that passing None as image raises appropriate error."""
        # Arrange
        output_path = temp_dir / f"test{output_handler.file_extension}"

        # Act & Assert - None image should be wrapped in IOError by handlers
        with pytest.raises((IOError, OSError)) as exc_info:
            output_handler.save(None, output_path)

        # Verify exception contains meaningful information about the failure
        error_message = str(exc_info.value).lower()
        assert (
            "failed to save" in error_message
            and output_handler.format_name.lower() in error_message
        ), f"Error should indicate save failure: {exc_info.value}"

    def test_save_with_invalid_image_object_raises_error(
        self, output_handler, temp_dir
    ):
        """Test that passing invalid image object raises appropriate error."""
        # Arrange
        output_path = temp_dir / f"test{output_handler.file_extension}"
        invalid_image = "not an image"  # String instead of PIL Image

        # Act & Assert - invalid image should be wrapped in IOError by handlers
        with pytest.raises((IOError, OSError)) as exc_info:
            output_handler.save(invalid_image, output_path)

        # Verify error message indicates the failure
        error_message = str(exc_info.value).lower()
        assert (
            "failed to save" in error_message
            and output_handler.format_name.lower() in error_message
        ), f"Error should indicate save failure: {exc_info.value}"

    def test_get_bytes_with_none_image_raises_error(self, output_handler):
        """Test that get_bytes with None image raises appropriate error."""
        # Arrange - None image

        # Act & Assert - None image should raise IOError for vector formats, others may vary
        with pytest.raises((IOError, OSError, AttributeError, TypeError)):
            output_handler.get_bytes(None)

    def test_error_messages_are_informative(self, output_handler, test_image):
        """Test that error messages provide useful debugging information."""
        # Arrange
        deeply_nested_invalid_path = (
            "/root/very/deeply/nested/nonexistent/path/test.png"
        )

        # Act & Assert
        with pytest.raises((IOError, OSError, PermissionError)) as exc_info:
            output_handler.save(test_image, deeply_nested_invalid_path)

        # Verify error message contains path information
        error_message = str(exc_info.value)
        assert len(error_message) > 10, (
            "Error message should be informative, not just a generic message"
        )
        # Don't require specific text since error messages vary by OS and format


class TestOutputHandlerThreadSafety:
    """Test thread safety and concurrent operations across handlers."""

    def test_concurrent_get_bytes_operations(self, output_handler):
        """Test that concurrent get_bytes operations are thread-safe."""
        # Arrange
        test_images = [
            Image.new("RGB", (50, 50), "red"),
            Image.new("RGB", (50, 50), "green"),
            Image.new("RGB", (50, 50), "blue"),
        ]
        results = []
        errors = []
        threads = []

        def get_bytes_worker(img, image_index):
            try:
                result = output_handler.get_bytes(img)
                results.append((image_index, result))
            except Exception as e:
                errors.append((image_index, e))

        # Act - start concurrent operations
        for i, img in enumerate(test_images):
            thread = threading.Thread(target=get_bytes_worker, args=(img, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Assert
        assert len(errors) == 0, (
            f"Concurrent operations should not produce errors: {errors}"
        )
        assert len(results) == len(test_images), (
            f"All {len(test_images)} operations should complete successfully"
        )

        # Verify all results are valid
        for image_index, result in results:
            assert isinstance(result, bytes), (
                f"Result for image {image_index} must be bytes"
            )
            assert len(result) > 0, f"Result for image {image_index} must not be empty"

    def test_concurrent_save_operations(self, output_handler, temp_dir):
        """Test that concurrent save operations are thread-safe."""
        # Arrange
        test_images = [
            Image.new("RGB", (50, 50), "red"),
            Image.new("RGB", (50, 50), "green"),
            Image.new("RGB", (50, 50), "blue"),
        ]
        created_files = []
        errors = []
        threads = []

        def save_worker(img, image_index):
            try:
                output_path = (
                    temp_dir
                    / f"concurrent_test_{image_index}{output_handler.file_extension}"
                )
                output_handler.save(img, output_path)
                created_files.append((image_index, output_path))
            except Exception as e:
                errors.append((image_index, e))

        # Act - start concurrent save operations
        for i, img in enumerate(test_images):
            thread = threading.Thread(target=save_worker, args=(img, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Assert
        assert len(errors) == 0, (
            f"Concurrent save operations should not produce errors: {errors}"
        )
        assert len(created_files) == len(test_images), (
            f"All {len(test_images)} save operations should complete successfully"
        )

        # Verify all files were created correctly
        for image_index, file_path in created_files:
            assert file_path.exists(), (
                f"File for image {image_index} should exist at {file_path}"
            )
            assert file_path.stat().st_size > 0, (
                f"File for image {image_index} should not be empty"
            )

    def test_handler_instance_isolation(self, format_name):
        """Test that handler instances don't interfere with each other during concurrent use."""
        # Arrange
        config1 = OutputConfig(quality=90, optimize=True)
        config2 = OutputConfig(quality=10, optimize=False)

        handler1 = get_output_handler(format_name, config1)
        handler2 = get_output_handler(format_name, config2)

        test_image = Image.new("RGB", (100, 100), "red")
        results = []
        errors = []
        threads = []

        def handler_worker(handler, handler_id, expected_quality):
            try:
                # Verify config hasn't been affected by other instances
                actual_quality = handler.config.quality
                result = handler.get_bytes(test_image)
                results.append(
                    (handler_id, actual_quality, expected_quality, len(result))
                )
            except Exception as e:
                errors.append((handler_id, e))

        # Act - use both handlers concurrently
        thread1 = threading.Thread(
            target=handler_worker, args=(handler1, "handler1", 90)
        )
        thread2 = threading.Thread(
            target=handler_worker, args=(handler2, "handler2", 10)
        )

        threads = [thread1, thread2]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Assert
        assert len(errors) == 0, (
            f"Concurrent handler operations should not produce errors: {errors}"
        )
        assert len(results) == 2, "Both handlers should complete successfully"

        # Verify each handler maintained its configuration
        for handler_id, actual_quality, expected_quality, result_size in results:
            assert actual_quality == expected_quality, (
                f"{handler_id} should maintain its quality setting ({expected_quality}), got {actual_quality}"
            )
            assert result_size > 0, f"{handler_id} should produce valid output"
