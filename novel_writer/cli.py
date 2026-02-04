import click
from pathlib import Path

from .config import Config
from .processing import clean_data, format_data
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
        click.echo(f"Config file not found: {config_path}. Using defaults.")
        ctx.obj['config'] = Config()

    # Setup logger
    log_level = "DEBUG" if verbose else ctx.obj['config'].log_level
    logger = setup_logger(log_level)
    ctx.obj['logger'] = logger

    logger.info(f"Novel Writer v0.1.0")
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
@click.option('--clean', is_flag=True, help='Clean then format')
@click.pass_context
def pipeline(ctx: click.Context, clean: bool):
    """Run complete data preparation pipeline."""
    logger = ctx.obj['logger']
    logger.info(f"Starting full pipeline...")

    if clean:
        ctx.invoke(clean)

    ctx.invoke(format)

    logger.success(f"Pipeline completed successfully")

def main():
    cli()

if __name__ == '__main__':
    main()
