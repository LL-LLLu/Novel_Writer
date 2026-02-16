"""Benchmark runner for evaluating novel generation quality."""

from pathlib import Path
from typing import Optional
import json
import time

from loguru import logger
from pydantic import BaseModel

from .metrics import evaluate_text, EvaluationResult
from .prompts import BENCHMARK_PROMPTS


class BenchmarkResult(BaseModel):
    """Result from a single benchmark prompt."""
    prompt_id: str
    prompt_name: str
    category: str
    generated_text: str
    evaluation: EvaluationResult
    generation_time_seconds: float


class BenchmarkReport(BaseModel):
    """Full benchmark report."""
    model_name: str
    results: list[BenchmarkResult]
    avg_overall_score: float
    avg_generation_time: float

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Benchmark Report: {self.model_name}",
            f"{'='*50}",
            f"Prompts evaluated: {len(self.results)}",
            f"Average overall score: {self.avg_overall_score:.3f}",
            f"Average generation time: {self.avg_generation_time:.2f}s",
            "",
        ]
        for r in self.results:
            lines.append(f"  [{r.category}] {r.prompt_name}: {r.evaluation.overall_score:.3f}")
        return "\n".join(lines)


def run_benchmark(
    generator_fn,  # callable(prompt: str) -> str
    model_name: str = "unknown",
    prompts: Optional[list[dict]] = None,
    output_path: Optional[Path] = None,
) -> BenchmarkReport:
    """
    Run benchmark evaluation on a text generator.

    Args:
        generator_fn: Callable that takes a prompt string and returns generated text.
        model_name: Name of the model being evaluated.
        prompts: Optional custom prompts. Uses BENCHMARK_PROMPTS if None.
        output_path: Optional path to save JSON results.

    Returns:
        BenchmarkReport with results for each prompt.
    """
    if prompts is None:
        prompts = BENCHMARK_PROMPTS

    results = []

    for prompt_data in prompts:
        logger.info(f"Running benchmark: {prompt_data['name']}")

        start_time = time.time()
        generated = generator_fn(prompt_data["prompt"])
        elapsed = time.time() - start_time

        evaluation = evaluate_text(generated)

        result = BenchmarkResult(
            prompt_id=prompt_data["id"],
            prompt_name=prompt_data["name"],
            category=prompt_data["category"],
            generated_text=generated,
            evaluation=evaluation,
            generation_time_seconds=round(elapsed, 3),
        )
        results.append(result)
        logger.info(f"  Score: {evaluation.overall_score:.3f} ({elapsed:.2f}s)")

    avg_score = sum(r.evaluation.overall_score for r in results) / len(results) if results else 0.0
    avg_time = sum(r.generation_time_seconds for r in results) / len(results) if results else 0.0

    report = BenchmarkReport(
        model_name=model_name,
        results=results,
        avg_overall_score=round(avg_score, 4),
        avg_generation_time=round(avg_time, 3),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report.model_dump_json(indent=2))
        logger.info(f"Benchmark results saved to {output_path}")

    return report
