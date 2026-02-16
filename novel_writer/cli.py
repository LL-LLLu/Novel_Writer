import click
from pathlib import Path

from .config import Config
from .processing import clean_data, format_data, segment_directory, deduplicate_dataset, filter_dataset
from .utils.logger import setup_logger

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), default='config.yaml',
              help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool):
    """Novel Writer - Fine-tune LLMs on custom web novels."""
    ctx.ensure_object(dict)

    # Load config
    config_path = Path(config)
    if config_path.exists():
        ctx.obj['config'] = Config.from_yaml(config_path)
    else:
        # Don't warn if just helping
        if ctx.invoked_subcommand != 'dashboard':
             # click.echo(f"Config file not found: {config_path}. Using defaults.")
             pass
        ctx.obj['config'] = Config()

    # Setup logger
    log_level = "DEBUG" if verbose else ctx.obj['config'].log_level
    logger = setup_logger(log_level)
    ctx.obj['logger'] = logger

    logger.info(f"Novel Writer v0.1.0")
    if config_path.exists():
        logger.info(f"Config loaded from: {config_path}")

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), help='Override input directory')
@click.option('--output', '-o', type=click.Path(), help='Override output directory')
@click.pass_context
def clean(ctx: click.Context, input: str, output: str):
    """Clean raw novel data (PDF/TXT)."""
    config = ctx.obj['config']

    # Override config if provided
    if input:
        config.data.input_dir = Path(input)
    if output:
        config.data.temp_dir = Path(output)

    logger = ctx.obj['logger']
    logger.info(f"Starting clean process...")
    logger.info(f"Input: {config.data.input_dir}")
    logger.info(f"Output: {config.data.temp_dir}")

    try:
        results = clean_data(config)
        logger.success(f"Cleaned {len(results)} files successfully")
    except Exception as e:
        logger.error(f"Clean process failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), help='Override input directory')
@click.option('--output', '-o', type=click.Path(), help='Override output file')
@click.pass_context
def format(ctx: click.Context, input: str, output: str):
    """Format cleaned data into training JSONL."""
    config = ctx.obj['config']

    # Override config if provided
    if input:
        config.data.temp_dir = Path(input)
    if output:
        config.data.output_dir = Path(output).parent

    logger = ctx.obj['logger']
    logger.info(f"Starting format process...")

    try:
        num_entries = format_data(config)
        logger.success(f"Formatted {num_entries} training entries")
    except Exception as e:
        logger.error(f"Format process failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), help='Input directory')
@click.option('--min-length', type=int, default=1000, help='Min chapter length')
@click.pass_context
def segment(ctx: click.Context, input: str, min_length: int):
    """Segment novels into chapters."""
    config = ctx.obj['config']
    input_dir = Path(input) if input else config.data.temp_dir
    logger = ctx.obj['logger']

    logger.info(f"Segmenting files in {input_dir}...")
    try:
        chapters = segment_directory(input_dir, min_chapter_length=min_length)
        logger.success(f"Created {len(chapters)} chapter files")
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), help='Input JSONL file')
@click.option('--threshold', type=float, default=0.85, help='Similarity threshold')
@click.pass_context
def deduplicate(ctx: click.Context, input: str, threshold: float):
    """Remove duplicate entries from dataset."""
    logger = ctx.obj['logger']
    input_file = Path(input)

    logger.info(f"Deduplicating {input_file.name}...")
    try:
        num_unique = deduplicate_dataset(input_file, threshold=threshold)
        logger.success(f"Kept {num_unique} unique entries")
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), help='Input JSONL file')
@click.option('--keep-ratio', type=float, default=0.8, help='Fraction to keep')
@click.pass_context
def filter(ctx: click.Context, input: str, keep_ratio: float):
    """Filter dataset by quality score."""
    logger = ctx.obj['logger']
    input_file = Path(input)

    logger.info(f"Filtering {input_file.name}...")
    try:
        num_kept = filter_dataset(input_file, keep_ratio=keep_ratio)
        logger.success(f"Kept {num_kept} high-quality entries")
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), help='Input JSONL file')
@click.option('--output', '-o', type=click.Path(), help='Output JSONL file')
@click.pass_context
def instruct(ctx: click.Context, input: str, output: str):
    """Generate instruction dataset."""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    input_file = Path(input) if input else config.data.output_dir / "train.jsonl"
    output_file = Path(output) if output else config.data.output_dir / "train_instruct.jsonl"
    
    logger.info(f"Generating instructions from {input_file}...")
    try:
        from .processing.instruct import generate_instruct_dataset
        num = generate_instruct_dataset(input_file, output_file, config)
        logger.success(f"Generated {num} instruction entries")
    except Exception as e:
        logger.error(f"Instruction generation failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input directory with novel files')
@click.option('--output', '-o', type=click.Path(), help='Output directory for extracted text')
@click.option('--extensions', '-e', multiple=True, help='File extensions to process (e.g. .epub .html)')
@click.pass_context
def ingest(ctx: click.Context, input: str, output: str, extensions: tuple):
    """Ingest novels from multiple formats (EPUB, HTML, Markdown, MOBI)."""
    config = ctx.obj['config']
    logger = ctx.obj['logger']

    input_dir = Path(input)
    output_dir = Path(output) if output else config.data.temp_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ext_list = list(extensions) if extensions else None

    logger.info(f"Ingesting files from {input_dir}...")

    try:
        from .processing.ingest import ingest_directory

        results = ingest_directory(input_dir, extensions=ext_list)

        for file_path, content in results:
            if content:
                out_file = output_dir / f"{file_path.stem}_ingested.txt"
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"Saved: {out_file.name}")

        logger.success(f"Ingested {len(results)} files to {output_dir}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--prompt', '-p', required=True, help='Text prompt')
@click.option('--model', '-m', default='lora_model', help='LoRA model path')
@click.option('--max-tokens', type=int, default=500, help='Max tokens to generate')
@click.option('--temperature', type=float, default=0.8, help='Sampling temperature')
@click.pass_context
def generate(ctx: click.Context, prompt: str, model: str, max_tokens: int, temperature: float):
    """Generate text with trained model."""
    logger = ctx.obj['logger']

    try:
        from .inference import NovelGenerator

        generator = NovelGenerator(
            base_model_path="unsloth/llama-3-8b-bnb-4bit",
            lora_path=Path(model) if Path(model).exists() else None
        )

        generated = generator.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

        click.echo(f"\nGenerated Text:\n{'-' * 50}")
        click.echo(generated)
        click.echo(f"{'-' * 50}\nLength: {len(generated)} chars")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--loras', '-l', multiple=True, required=True, help='LoRA paths')
@click.option('--weights', '-w', type=float, multiple=True, help='Weights')
@click.option('--output', '-o', required=True, help='Output path')
@click.pass_context
def mix(ctx: click.Context, loras: tuple, weights: tuple, output: str):
    """Merge multiple LoRA styles."""
    logger = ctx.obj['logger']

    if weights and len(weights) != len(loras):
        raise click.ClickException("Weights must match number of LoRAs")

    if not weights:
        # Equal weights
        weights = [1.0 / len(loras)] * len(loras)

    try:
        from .processing.mix import mix_styles

        mix_styles(
            base_model="unsloth/llama-3-8b-bnb-4bit",
            lora_paths=list(loras),
            weights=list(weights),
            output_path=output
        )

        logger.success(f"Mixed model saved to {output}")

    except Exception as e:
        logger.error(f"Mixing failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
def dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    subprocess.run(["streamlit", "run", "novel_writer/dashboard.py"])

@cli.command()
@click.option('--ingest', 'run_ingest', is_flag=True, help='Ingest multi-format files first')
@click.option('--clean', 'run_clean', is_flag=True, help='Clean then format')
@click.option('--segment', 'run_segment', is_flag=True, help='Segment into chapters')
@click.option('--deduplicate', 'run_dedup', is_flag=True, help='Remove duplicates')
@click.option('--filter', 'run_filter', is_flag=True, help='Filter by quality')
@click.pass_context
def pipeline(ctx: click.Context, run_ingest: bool, run_clean: bool, run_segment: bool, run_dedup: bool, run_filter: bool):
    """Run complete data preparation pipeline."""
    logger = ctx.obj['logger']
    config = ctx.obj['config']
    logger.info(f"Starting full pipeline...")

    if run_ingest:
        ctx.invoke(ingest, input=str(config.data.input_dir), output=str(config.data.temp_dir))

    if run_clean:
        ctx.invoke(clean)

    if run_segment:
        ctx.invoke(segment)

    ctx.invoke(format)

    train_jsonl = config.data.output_dir / "train.jsonl"

    if run_dedup:
        ctx.invoke(deduplicate, input=str(train_jsonl))
        train_jsonl = train_jsonl.parent / f"{train_jsonl.stem}_dedup.jsonl"

    if run_filter:
        ctx.invoke(filter, input=str(train_jsonl))

    logger.success(f"Pipeline completed successfully")


@cli.command()
@click.option('--base-model', '-b', required=True, help='Base model path')
@click.option('--dataset', '-d', required=True, help='Dataset JSONL path')
@click.option('--trials', '-n', default=20, help='Number of trials')
@click.option('--output', '-o', default='best_hyperparams.json', help='Output file')
@click.pass_context
def tune(ctx: click.Context, base_model: str, dataset: str, trials: int, output: str):
    """Run hyperparameter tuning."""
    logger = ctx.obj['logger']

    try:
        from .tuning import run_hyperparameter_tuning

        logger.info(f"Starting hyperparameter tuning: {trials} trials")
        run_hyperparameter_tuning(base_model, dataset, output, trials)

        logger.success(f"Tuning complete. Best params saved to {output}")

    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--format', '-f', type=click.Choice(['full', 'gguf', 'vllm', 'onnx']), required=True, help='Export format')
@click.option('--base-model', '-b', default='unsloth/llama-3-8b-bnb-4bit', help='Base model')
@click.option('--lora', '-l', help='LoRA adapter path')
@click.option('--output', '-o', default='exported_model', help='Output path')
@click.option('--quantization', '-q', default='q4_k_m', help='Quantization level')
@click.pass_context
def export(ctx: click.Context, format: str, base_model: str, lora: str, output: str, quantization: str):
    """Export model to various formats."""
    logger = ctx.obj['logger']

    try:
        from .export import export_model

        logger.info(f"Exporting model to {format} format...")
        export_model(format, base_model, lora, output, quantization)

        logger.success(f"Model exported to {output}")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--model', '-m', default='lora_model', help='LoRA model path')
@click.option('--output', '-o', default='benchmark_results.json', help='Output JSON file')
@click.option('--max-tokens', type=int, default=500, help='Max tokens per generation')
@click.pass_context
def evaluate(ctx: click.Context, model: str, output: str, max_tokens: int):
    """Run evaluation benchmarks on a model."""
    logger = ctx.obj['logger']

    try:
        from .evaluation.benchmark import run_benchmark
        from .inference import NovelGenerator

        logger.info(f"Loading model for evaluation...")
        generator = NovelGenerator(
            base_model_path="unsloth/llama-3-8b-bnb-4bit",
            lora_path=Path(model) if Path(model).exists() else None
        )

        def generator_fn(prompt: str) -> str:
            return generator.generate(prompt=prompt, max_new_tokens=max_tokens)

        report = run_benchmark(
            generator_fn=generator_fn,
            model_name=model,
            output_path=Path(output),
        )

        click.echo(report.summary())
        logger.success(f"Evaluation complete. Results saved to {output}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input JSONL dataset')
@click.option('--output', '-o', type=click.Path(), help='Output JSONL file')
@click.option('--min-diff', type=float, default=0.1, help='Minimum score difference')
@click.option('--max-pairs', type=int, default=None, help='Maximum pairs to generate')
@click.pass_context
def preference(ctx: click.Context, input: str, output: str, min_diff: float, max_pairs: int):
    """Generate DPO preference pairs from dataset."""
    config = ctx.obj['config']
    logger = ctx.obj['logger']

    input_path = Path(input)
    output_path = Path(output) if output else config.data.output_dir / "preference_pairs.jsonl"

    logger.info(f"Generating preference pairs from {input_path}...")

    try:
        from .processing.preference import generate_preference_pairs

        num_pairs = generate_preference_pairs(
            input_path=input_path,
            output_path=output_path,
            min_score_diff=min_diff,
            max_pairs=max_pairs,
        )

        logger.success(f"Generated {num_pairs} preference pairs -> {output_path}")
    except Exception as e:
        logger.error(f"Preference pair generation failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input JSONL dataset')
@click.option('--output', '-o', type=click.Path(), help='Output sorted JSONL file')
@click.option('--reverse', is_flag=True, help='Sort hard-to-easy (anti-curriculum)')
@click.option('--buckets', type=int, default=3, help='Number of difficulty buckets')
@click.pass_context
def curriculum(ctx: click.Context, input: str, output: str, reverse: bool, buckets: int):
    """Sort training data by difficulty for curriculum learning."""
    config = ctx.obj['config']
    logger = ctx.obj['logger']

    input_path = Path(input)
    output_path = Path(output) if output else config.data.output_dir / "train_curriculum.jsonl"

    logger.info(f"Sorting data by difficulty...")

    try:
        from .processing.curriculum import sort_by_curriculum

        stats = sort_by_curriculum(
            input_path=input_path,
            output_path=output_path,
            reverse=reverse,
            num_buckets=buckets,
        )

        click.echo(f"Sorted {stats['total']} entries ({stats['sorted_order']})")
        click.echo(f"Difficulty: avg={stats['avg_difficulty']}, range=[{stats['min_difficulty']}, {stats['max_difficulty']}]")

    except Exception as e:
        logger.error(f"Curriculum sorting failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--pipeline', is_flag=True, help='Profile data pipeline')
@click.option('--generation', is_flag=True, help='Profile text generation')
@click.option('--num-calls', type=int, default=10, help='Number of generation calls')
@click.pass_context
def profile(ctx: click.Context, pipeline: bool, generation: bool, num_calls: int):
    """Profile performance."""
    logger = ctx.obj['logger']

    try:
        from .profile import profile_pipeline, profile_generation
        from .config import Config

        if pipeline:
            config = ctx.obj['config']
            from .processing import clean_data, format_data
            profile_pipeline(clean_data, format_data, config)

        if generation:
            from .inference import NovelGenerator

            generator = NovelGenerator(
                base_model_path="unsloth/llama-3-8b-bnb-4bit",
                lora_path=Path("lora_model") if Path("lora_model").exists() else None
            )

            profile_generation(generator, "Continue the story...", num_calls)

    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        raise click.ClickException(str(e))

def main():
    cli()

if __name__ == '__main__':
    main()
