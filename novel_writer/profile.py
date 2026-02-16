import cProfile
import pstats
import io
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Any

from loguru import logger

@contextmanager
def profile_function(name: str, output_file: str = None):
    """
    Profile function execution.

    Args:
        name: Name of the profile
        output_file: Optional file to save stats
    """
    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    yield pr
    end_time = time.time()

    pr.disable()

    # Print summary
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions

    logger.info(f"\n=== Profile: {name} ===")
    logger.info(f"Total time: {end_time - start_time:.2f}s")
    logger.info(s.getvalue())

    # Save to file if requested
    if output_file:
        ps.dump_stats(output_file)
        logger.info(f"Profile saved to {output_file}")

def profile_pipeline(
    clean_func: Callable,
    format_func: Callable,
    config: Any
):
    """Profile the entire data pipeline."""
    logger.info("Starting pipeline profiling...")

    with profile_function("clean_data", "profile_clean.prof"):
        clean_func(config)

    with profile_function("format_data", "profile_format.prof"):
        format_func(config)

    logger.info("Pipeline profiling complete")

def profile_generation(
    generator,
    prompt: str,
    num_calls: int = 10
):
    """Profile text generation."""
    logger.info(f"Profiling {num_calls} generations...")

    times = []

    with profile_function("generation", "profile_gen.prof"):
        for i in range(num_calls):
            start = time.time()
            _ = generator.generate(prompt, max_new_tokens=500)
            end = time.time()
            times.append(end - start)

            if (i + 1) % 5 == 0:
                logger.info(f"Completed {i + 1}/{num_calls}")

    # Stats
    avg_time = sum(times) / len(times)
    logger.info(f"\nGeneration Stats:")
    logger.info(f"  Average: {avg_time:.2f}s")
    logger.info(f"  Min: {min(times):.2f}s")
    logger.info(f"  Max: {max(times):.2f}s")
    logger.info(f"  Total: {sum(times):.2f}s")

    # Tokens per second
    estimated_tokens = num_calls * 500  # Assuming 500 tokens per call
    tps = estimated_tokens / sum(times)
    logger.info(f"  Tokens/sec: {tps:.2f}")
