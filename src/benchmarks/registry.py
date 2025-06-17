import time
import functools
from typing import Callable, Dict, Any

# Global registry for benchmarks
BENCHMARK_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_benchmark(name: str | None = None):
    """
    Decorator to register a function as a benchmark.

    Args:
        name: Optional name for the benchmark. If None, the function name is used.
    """

    def decorator(func: Callable[..., Any]):
        benchmark_name = name if name is not None else func.__name__
        if benchmark_name in BENCHMARK_REGISTRY:
            raise ValueError(
                f"Benchmark with name '{benchmark_name}' already registered."
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"--- Running benchmark: {benchmark_name} ---")
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            print(
                f"--- Finished benchmark: {benchmark_name} (Duration: {duration:.4f}s) ---"
            )
            # Optionally return results or store them
            return result  # Or potentially store {'name': benchmark_name, 'duration': duration, 'result': result}

        BENCHMARK_REGISTRY[benchmark_name] = wrapper
        # Return the original function, not the wrapper,
        # so it can be called directly if needed without the timing overhead.
        # The registry holds the wrapped version.
        return func

    return decorator


def run_all_benchmarks():
    """Runs all registered benchmarks."""
    if not BENCHMARK_REGISTRY:
        print("No benchmarks registered.")
        return

    print(f"Found {len(BENCHMARK_REGISTRY)} benchmarks. Running all...")
    results = {}
    for name, benchmark_func in BENCHMARK_REGISTRY.items():
        try:
            # Call the wrapped function from the registry
            result = benchmark_func()
            results[name] = result  # Store results if needed
        except Exception as e:
            print(f"!!! Benchmark '{name}' failed: {e} !!!")
            results[name] = f"Failed: {e}"  # Mark as failed

    print("\n--- Benchmark Summary ---")
    # Summary logic can be added here, e.g., printing durations stored during execution
    # For now, just confirms completion.
    print("All benchmarks finished.")
    return results


# Example Usage (within a benchmark file like denoising.py):
# from .registry import register_benchmark
#
# @register_benchmark("simple_denoise_test")
# def my_denoising_benchmark():
#     # ... benchmark code ...
#     time.sleep(0.5) # Simulate work
#     print("Denoising complete.")
#     return {"accuracy": 0.95}

# Example Runner Script (e.g., run_benchmarks.py):
# import srf.benchmarks.denoising # Import to trigger registration
# import srf.benchmarks.hypothesis # Import others...
# from srf.benchmarks.registry import run_all_benchmarks
#
# if __name__ == "__main__":
#     run_all_benchmarks()
