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
@click.option('--clean', 'run_clean', is_flag=True, help='Clean then format')
@click.option('--segment', 'run_segment', is_flag=True, help='Segment into chapters')
@click.option('--deduplicate', 'run_dedup', is_flag=True, help='Remove duplicates')
@click.option('--filter', 'run_filter', is_flag=True, help='Filter by quality')
@click.pass_context
def pipeline(ctx: click.Context, run_clean: bool, run_segment: bool, run_dedup: bool, run_filter: bool):
    """Run complete data preparation pipeline."""
    logger = ctx.obj['logger']
    logger.info(f"Starting full pipeline...")

    if run_clean:
        ctx.invoke(clean)

    if run_segment:
        ctx.invoke(segment)

    ctx.invoke(format)

    if run_dedup:
        ctx.invoke(deduplicate)

    if run_filter:
        ctx.invoke(filter)

    logger.success(f"Pipeline completed successfully")

def main():
    cli()

if __name__ == '__main__':
    main()
