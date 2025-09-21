"""
Benchmark tests for outputs submodule.

These tests monitor performance characteristics and help identify
performance regressions across different scenarios.
"""

import gc
import time

import pytest
from PIL import Image

from molecular_string_renderer.config import OutputConfig
from molecular_string_renderer.outputs import get_output_handler


class BenchmarkConstants:
    """Constants for benchmark tests to avoid magic numbers."""

    # Performance thresholds
    MAX_HANDLER_CREATION_TIME = 0.001  # 1ms average
    MAX_IMAGE_PROCESSING_TIME = 0.1  # 100ms average
    MAX_MEMORY_INCREASE_MB = 10  # 10MB memory increase
    MAX_CONCURRENT_OPS_TIME = 0.05  # 50ms for concurrent ops
    MIN_OPERATIONS_PER_SECOND = 50  # Minimum ops/sec for concurrent processing
    MIN_BATCH_OPS_PER_SECOND = 100  # Minimum ops/sec for batch processing

    # Test iterations and sizes
    DEFAULT_ITERATIONS = 100
    PROCESSING_ITERATIONS = 50
    SCALING_ITERATIONS = 10
    MEMORY_TEST_ITERATIONS = 100
    OPTIMIZATION_ITERATIONS = 20
    EFFICIENCY_ITERATIONS = 50
    BATCH_SIZE = 20

    # Image dimensions
    DEFAULT_IMAGE_SIZE = (100, 100)
    SMALL_IMAGE_SIZE = (50, 50)
    LARGE_IMAGE_SIZE = (200, 200)
    SCALING_SIZES = [50, 100, 200, 400]
    LARGE_SCALING_SIZES = [500, 1000, 1500]
    FORMAT_SCALING_SIZES = [100, 200, 400]

    # Threading
    NUM_THREADS = 10
    OPERATIONS_PER_THREAD = 10

    # Quality settings
    QUALITY_LEVELS = [20, 50, 80, 95]

    # Format lists
    ALL_FORMATS = ["png", "jpg", "webp", "svg", "pdf"]
    RASTER_FORMATS = ["png", "jpg", "webp"]
    COMPARISON_FORMATS = ["png", "jpg", "webp"]

    # Time precision for formatting
    TIME_PRECISION = 6  # Number of decimal places for time formatting


@pytest.fixture
def benchmark_image():
    """Create a standard benchmark image."""
    return Image.new("RGB", BenchmarkConstants.DEFAULT_IMAGE_SIZE, "red")


@pytest.fixture
def complex_benchmark_image():
    """Create a complex image for compression testing."""
    width, height = BenchmarkConstants.LARGE_IMAGE_SIZE
    test_image = Image.new("RGB", (width, height))
    pixels = []
    for y in range(height):
        for x in range(width):
            r = (x * 255) // width
            g = (y * 255) // height
            b = ((x + y) * 255) // (width + height)
            pixels.append((r, g, b))
    test_image.putdata(pixels)
    return test_image


@pytest.fixture
def pattern_benchmark_image():
    """Create an image with a repeating pattern for optimization testing."""
    width, height = BenchmarkConstants.LARGE_IMAGE_SIZE
    test_image = Image.new("RGB", (width, height))
    pixels = []
    for y in range(height):
        for x in range(width):
            # Repeating pattern that compresses well
            if (x // 10 + y // 10) % 2:
                pixels.append((255, 255, 255))
            else:
                pixels.append((0, 0, 0))
    test_image.putdata(pixels)
    return test_image


@pytest.fixture(autouse=True)
def cleanup_benchmark():
    """Ensure clean state for benchmarks."""
    gc.collect()
    yield
    gc.collect()


class BenchmarkHelper:
    """Helper class for benchmark test operations."""

    @staticmethod
    def measure_handler_creation_time(formats: list[str], iterations: int) -> float:
        """Measure time to create handlers.

        Args:
            formats: List of format names to test
            iterations: Number of iterations to perform

        Returns:
            Total time taken for all handler creations
        """
        start_time = time.time()
        for _ in range(iterations):
            for format_name in formats:
                handler = get_output_handler(format_name)
                assert handler is not None
        return time.time() - start_time

    @staticmethod
    def measure_image_processing_time(
        image: Image.Image, format_name: str, iterations: int
    ) -> float:
        """Measure time to process an image.

        Args:
            image: PIL Image to process
            format_name: Output format name
            iterations: Number of iterations to perform

        Returns:
            Total time taken for all processing operations
        """
        handler = get_output_handler(format_name)
        start_time = time.time()
        for _ in range(iterations):
            result = handler.get_bytes(image)
            assert len(result) > 0
        return time.time() - start_time

    @staticmethod
    def create_test_image(size: tuple[int, int], color: str = "red") -> Image.Image:
        """Create a test image with specified size and color.

        Args:
            size: Image dimensions as (width, height)
            color: Color name or hex code

        Returns:
            PIL Image object
        """
        return Image.new("RGB", size, color)

    @staticmethod
    def format_time_message(operation: str, time_value: float, threshold: float) -> str:
        """Format a time-based error message consistently.

        Args:
            operation: Description of the operation
            time_value: Actual time taken
            threshold: Expected time threshold

        Returns:
            Formatted error message
        """
        precision = BenchmarkConstants.TIME_PRECISION
        return f"{operation} too slow: {time_value:.{precision}f}s (threshold: {threshold:.{precision}f}s)"


class TestPerformanceBenchmarks:
    """Benchmark tests for performance monitoring."""

    @pytest.mark.benchmark
    def test_handler_creation_benchmark(self):
        """Benchmark handler creation speed."""
        # Arrange
        formats = BenchmarkConstants.ALL_FORMATS
        iterations = BenchmarkConstants.DEFAULT_ITERATIONS

        # Act
        total_time = BenchmarkHelper.measure_handler_creation_time(formats, iterations)

        # Assert
        avg_time_per_handler = total_time / (iterations * len(formats))
        assert avg_time_per_handler < BenchmarkConstants.MAX_HANDLER_CREATION_TIME, (
            BenchmarkHelper.format_time_message(
                "Handler creation",
                avg_time_per_handler,
                BenchmarkConstants.MAX_HANDLER_CREATION_TIME,
            )
        )

    @pytest.mark.benchmark
    def test_image_processing_benchmark(self, benchmark_image):
        """Benchmark image processing speed across formats."""
        # Arrange
        formats = BenchmarkConstants.RASTER_FORMATS
        iterations = BenchmarkConstants.PROCESSING_ITERATIONS

        # Act
        results = {}
        for format_name in formats:
            processing_time = BenchmarkHelper.measure_image_processing_time(
                benchmark_image, format_name, iterations
            )
            results[format_name] = processing_time / iterations

        # Assert
        for format_name, avg_time in results.items():
            assert avg_time < BenchmarkConstants.MAX_IMAGE_PROCESSING_TIME, (
                BenchmarkHelper.format_time_message(
                    f"{format_name} processing",
                    avg_time,
                    BenchmarkConstants.MAX_IMAGE_PROCESSING_TIME,
                )
            )

        print(f"\\nBenchmark results: {results}")

    @pytest.mark.benchmark
    def test_image_size_scaling_benchmark(self):
        """Benchmark how processing time scales with image size."""
        # Arrange
        sizes = BenchmarkConstants.SCALING_SIZES
        format_name = "png"
        iterations = BenchmarkConstants.SCALING_ITERATIONS

        # Act
        results = {}
        for size in sizes:
            test_image = BenchmarkHelper.create_test_image((size, size), "blue")
            processing_time = BenchmarkHelper.measure_image_processing_time(
                test_image, format_name, iterations
            )
            results[size] = processing_time / iterations

        # Assert
        self._assert_scaling_performance(results, sizes)
        print(f"\\nSize scaling results: {results}")

    def _assert_scaling_performance(self, results: dict[int, float], sizes: list[int]):
        """Assert that processing time scales reasonably with image size."""
        base_size = sizes[0]
        for size, avg_time in results.items():
            # Allow for some non-linearity but shouldn't be excessive
            max_expected_time = (size / base_size) ** 1.5 * 0.01
            assert avg_time < max_expected_time, BenchmarkHelper.format_time_message(
                f"Size {size}x{size}", avg_time, max_expected_time
            )

    @pytest.mark.benchmark
    def test_quality_impact_benchmark(self, complex_benchmark_image):
        """Benchmark impact of quality settings on processing time."""
        # Arrange
        quality_levels = BenchmarkConstants.QUALITY_LEVELS
        format_name = "jpeg"
        iterations = BenchmarkConstants.OPTIMIZATION_ITERATIONS

        # Act
        results = {}
        for quality in quality_levels:
            config = OutputConfig(quality=quality)
            handler = get_output_handler(format_name, config)

            start_time = time.time()
            for _ in range(iterations):
                result = handler.get_bytes(complex_benchmark_image)
                assert len(result) > 0

            total_time = time.time() - start_time
            results[quality] = total_time / iterations

        # Assert
        for quality, avg_time in results.items():
            assert avg_time < BenchmarkConstants.MAX_CONCURRENT_OPS_TIME, (
                BenchmarkHelper.format_time_message(
                    f"Quality {quality}",
                    avg_time,
                    BenchmarkConstants.MAX_CONCURRENT_OPS_TIME,
                )
            )

        print(f"\\nQuality impact results: {results}")

    @pytest.mark.benchmark
    def test_concurrent_processing_benchmark(self, benchmark_image):
        """Benchmark concurrent processing performance."""
        # Arrange
        import threading

        format_name = "png"
        num_threads = BenchmarkConstants.NUM_THREADS
        operations_per_thread = BenchmarkConstants.OPERATIONS_PER_THREAD

        results = []
        errors = []

        def worker():
            """Worker function for concurrent processing test."""
            try:
                handler = get_output_handler(format_name)
                thread_start = time.time()

                for _ in range(operations_per_thread):
                    result = handler.get_bytes(benchmark_image)
                    assert len(result) > 0

                thread_time = time.time() - thread_start
                results.append(thread_time)
            except Exception as e:
                errors.append(e)

        # Act
        overall_start = time.time()
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        overall_time = time.time() - overall_start

        # Assert
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == num_threads

        total_operations = num_threads * operations_per_thread
        operations_per_second = total_operations / overall_time

        assert operations_per_second > BenchmarkConstants.MIN_OPERATIONS_PER_SECOND, (
            f"Concurrent processing too slow: {operations_per_second:.1f} ops/sec "
            f"(minimum: {BenchmarkConstants.MIN_OPERATIONS_PER_SECOND})"
        )

        print(f"\\nConcurrent processing: {operations_per_second:.1f} ops/sec")

    @pytest.mark.benchmark
    def test_memory_usage_benchmark(self, benchmark_image):
        """Benchmark memory usage during processing."""
        # Arrange
        try:
            import os

            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        process = psutil.Process(os.getpid())
        handler = get_output_handler("png")
        iterations = BenchmarkConstants.MEMORY_TEST_ITERATIONS

        # Act
        initial_memory = process.memory_info().rss

        for _ in range(iterations):
            result = handler.get_bytes(benchmark_image)
            assert len(result) > 0

        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss

        # Assert
        memory_increase = final_memory - initial_memory
        max_memory_bytes = BenchmarkConstants.MAX_MEMORY_INCREASE_MB * 1024 * 1024

        assert memory_increase < max_memory_bytes, (
            f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB "
            f"(max: {BenchmarkConstants.MAX_MEMORY_INCREASE_MB}MB)"
        )

        print(f"\\nMemory increase: {memory_increase / 1024 / 1024:.1f}MB")

    @pytest.mark.benchmark
    def test_normal_processing_performance(self, pattern_benchmark_image):
        """Test performance without optimization."""
        # Arrange
        format_name = "png"
        iterations = BenchmarkConstants.OPTIMIZATION_ITERATIONS
        config = OutputConfig(optimize=False)

        # Act
        handler = get_output_handler(format_name, config)
        processing_time = self._measure_optimization_processing_time(
            handler, pattern_benchmark_image, iterations
        )

        # Assert
        max_time = 2.0  # seconds
        assert processing_time < max_time, BenchmarkHelper.format_time_message(
            "Normal processing", processing_time, max_time
        )

    @pytest.mark.benchmark
    def test_optimized_processing_performance(self, pattern_benchmark_image):
        """Test performance with optimization."""
        # Arrange
        format_name = "png"
        iterations = BenchmarkConstants.OPTIMIZATION_ITERATIONS
        config = OutputConfig(optimize=True)

        # Act
        handler = get_output_handler(format_name, config)
        processing_time = self._measure_optimization_processing_time(
            handler, pattern_benchmark_image, iterations
        )

        # Assert
        max_time = 5.0  # seconds (optimization can be slower)
        assert processing_time < max_time, BenchmarkHelper.format_time_message(
            "Optimized processing", processing_time, max_time
        )

    @pytest.mark.benchmark
    def test_optimization_comparison_benchmark(self, pattern_benchmark_image):
        """Compare optimization vs normal processing performance."""
        # Arrange
        format_name = "png"
        iterations = BenchmarkConstants.OPTIMIZATION_ITERATIONS

        # Act
        normal_config = OutputConfig(optimize=False)
        optimized_config = OutputConfig(optimize=True)

        normal_handler = get_output_handler(format_name, normal_config)
        optimized_handler = get_output_handler(format_name, optimized_config)

        normal_time = self._measure_optimization_processing_time(
            normal_handler, pattern_benchmark_image, iterations
        )
        optimized_time = self._measure_optimization_processing_time(
            optimized_handler, pattern_benchmark_image, iterations
        )

        # Assert
        # Both should complete but allow optimization to be slower
        assert normal_time < 2.0, f"Normal processing too slow: {normal_time:.3f}s"
        assert optimized_time < 5.0, (
            f"Optimized processing too slow: {optimized_time:.3f}s"
        )

        print(
            f"\\nOptimization comparison - Normal: {normal_time:.3f}s, Optimized: {optimized_time:.3f}s"
        )

    def _measure_optimization_processing_time(
        self, handler, image: Image.Image, iterations: int
    ) -> float:
        """Measure processing time for optimization tests."""
        start_time = time.time()
        for _ in range(iterations):
            result = handler.get_bytes(image)
            assert len(result) > 0
        return time.time() - start_time


class TestResourceUtilization:
    """Test resource utilization patterns."""

    def test_handler_reuse_efficiency(self, benchmark_image):
        """Test efficiency of reusing handlers vs creating new ones."""
        # Arrange
        iterations = BenchmarkConstants.EFFICIENCY_ITERATIONS
        format_name = "png"

        # Act
        reuse_time = self._measure_handler_reuse_time(
            benchmark_image, format_name, iterations
        )
        recreation_time = self._measure_handler_recreation_time(
            benchmark_image, format_name, iterations
        )

        # Assert
        efficiency_ratio = recreation_time / reuse_time if reuse_time > 0 else 1.0
        min_efficiency = 0.8  # Recreation should be at least 80% as efficient as reuse

        assert efficiency_ratio >= min_efficiency, (
            f"Handler reuse not efficient: {efficiency_ratio:.2f}x speedup "
            f"(minimum: {min_efficiency:.2f}x)"
        )

        print(f"\\nHandler reuse efficiency: {efficiency_ratio:.2f}x faster")

    def _measure_handler_reuse_time(
        self, image: Image.Image, format_name: str, iterations: int
    ) -> float:
        """Measure time when reusing the same handler."""
        handler = get_output_handler(format_name)
        start_time = time.time()
        for _ in range(iterations):
            result = handler.get_bytes(image)
            assert len(result) > 0
        return time.time() - start_time

    def _measure_handler_recreation_time(
        self, image: Image.Image, format_name: str, iterations: int
    ) -> float:
        """Measure time when recreating handlers each time."""
        start_time = time.time()
        for _ in range(iterations):
            new_handler = get_output_handler(format_name)
            result = new_handler.get_bytes(image)
            assert len(result) > 0
        return time.time() - start_time

    def test_batch_processing_efficiency(self):
        """Test efficiency patterns for batch processing."""
        # Arrange
        images = [
            BenchmarkHelper.create_test_image((50, 50), "red"),
            BenchmarkHelper.create_test_image((75, 75), "green"),
            BenchmarkHelper.create_test_image(
                BenchmarkConstants.DEFAULT_IMAGE_SIZE, "blue"
            ),
        ]
        format_name = "png"
        batch_size = BenchmarkConstants.BATCH_SIZE

        # Act
        handler = get_output_handler(format_name)
        start_time = time.time()

        for _ in range(batch_size):
            for img in images:
                result = handler.get_bytes(img)
                assert len(result) > 0

        total_time = time.time() - start_time

        # Assert
        operations = batch_size * len(images)
        ops_per_second = operations / total_time if total_time > 0 else 0

        assert ops_per_second > BenchmarkConstants.MIN_BATCH_OPS_PER_SECOND, (
            f"Batch processing too slow: {ops_per_second:.1f} ops/sec "
            f"(minimum: {BenchmarkConstants.MIN_BATCH_OPS_PER_SECOND})"
        )

    def test_garbage_collection_impact(self, benchmark_image):
        """Test impact of garbage collection on processing."""
        # Arrange
        handler = get_output_handler("png")
        iterations = BenchmarkConstants.MEMORY_TEST_ITERATIONS
        gc_interval = 10

        # Act
        time_without_gc = self._measure_processing_without_gc(
            handler, benchmark_image, iterations
        )
        time_with_gc = self._measure_processing_with_gc(
            handler, benchmark_image, iterations, gc_interval
        )

        # Assert
        max_time_without_gc = 5.0
        max_time_with_gc = 10.0

        assert time_without_gc < max_time_without_gc, (
            BenchmarkHelper.format_time_message(
                "Processing without GC", time_without_gc, max_time_without_gc
            )
        )
        assert time_with_gc < max_time_with_gc, BenchmarkHelper.format_time_message(
            "Processing with GC", time_with_gc, max_time_with_gc
        )

        # GC overhead should be reasonable (but can vary significantly)
        if time_without_gc > 0:
            gc_overhead = (time_with_gc - time_without_gc) / time_without_gc
            max_gc_overhead = 5.0  # 500% overhead maximum
            assert gc_overhead < max_gc_overhead, (
                f"GC overhead extremely high: {gc_overhead:.2%} "
                f"(maximum: {max_gc_overhead:.2%})"
            )

    def _measure_processing_without_gc(
        self, handler, image: Image.Image, iterations: int
    ) -> float:
        """Measure processing time without explicit garbage collection."""
        start_time = time.time()
        for _ in range(iterations):
            result = handler.get_bytes(image)
            assert len(result) > 0
        return time.time() - start_time

    def _measure_processing_with_gc(
        self, handler, image: Image.Image, iterations: int, gc_interval: int
    ) -> float:
        """Measure processing time with periodic garbage collection."""
        start_time = time.time()
        for i in range(iterations):
            result = handler.get_bytes(image)
            assert len(result) > 0
            if i % gc_interval == 0:
                gc.collect()
        return time.time() - start_time


class TestScalabilityPatterns:
    """Test scalability patterns and limits."""

    def test_large_image_handling(self):
        """Test handling of large images."""
        # Arrange
        sizes = BenchmarkConstants.LARGE_SCALING_SIZES
        format_name = "png"

        # Act & Assert
        for size in sizes:
            test_image = BenchmarkHelper.create_test_image((size, size), "purple")
            handler = get_output_handler(format_name)

            processing_time = self._measure_single_image_processing(handler, test_image)

            # Processing time should scale sub-quadratically
            max_expected_time = self._calculate_max_expected_time(size)
            assert processing_time < max_expected_time, (
                BenchmarkHelper.format_time_message(
                    f"Large image {size}x{size}", processing_time, max_expected_time
                )
            )

    def _measure_single_image_processing(self, handler, image: Image.Image) -> float:
        """Measure time to process a single image."""
        start_time = time.time()
        result = handler.get_bytes(image)
        processing_time = time.time() - start_time

        assert len(result) > 0
        return processing_time

    def _calculate_max_expected_time(self, size: int) -> float:
        """Calculate maximum expected processing time for image size."""
        base_size = BenchmarkConstants.LARGE_SCALING_SIZES[0]
        return (size / base_size) ** 1.8 * 0.5

    def test_format_comparison_scaling(self):
        """Compare how different formats scale with image size."""
        # Arrange
        sizes = BenchmarkConstants.FORMAT_SCALING_SIZES
        formats = BenchmarkConstants.COMPARISON_FORMATS

        # Act
        scaling_results = {}
        for format_name in formats:
            scaling_results[format_name] = self._measure_format_scaling(
                format_name, sizes
            )

        # Assert
        self._assert_format_scaling_performance(scaling_results, sizes)
        print(f"\\nScaling comparison: {scaling_results}")

    def _measure_format_scaling(
        self, format_name: str, sizes: list[int]
    ) -> list[float]:
        """Measure how a format scales across different image sizes."""
        handler = get_output_handler(format_name)
        format_times = []

        for size in sizes:
            test_image = BenchmarkHelper.create_test_image((size, size), "orange")
            processing_time = self._measure_single_image_processing(handler, test_image)
            format_times.append(processing_time)

        return format_times

    def _assert_format_scaling_performance(
        self, scaling_results: dict, sizes: list[int]
    ):
        """Assert that all formats scale reasonably with image size."""
        base_size = sizes[0]

        for format_name, times in scaling_results.items():
            for i, time_val in enumerate(times):
                size = sizes[i]
                # Quadratic scaling baseline with some tolerance
                max_time = (size / base_size) ** 2 * 0.01
                assert time_val < max_time, BenchmarkHelper.format_time_message(
                    f"{format_name} at {size}x{size}", time_val, max_time
                )
