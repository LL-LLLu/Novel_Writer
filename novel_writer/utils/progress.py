from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from typing import Iterable, Callable, Optional, Any, List

def create_progress(description: str = "Processing...", total: Optional[int] = None) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

def process_with_progress(
    items: Iterable,
    process_func: Callable,
    description: str = "Processing...",
    total: Optional[int] = None,
) -> List:
    results = []

    with create_progress(description, total=total) as progress:
        task_id = progress.add_task(description, total=total)

        for item in items:
            try:
                result = process_func(item)
                results.append(result)
                progress.update(task_id, advance=1)
            except Exception as e:
                progress.console.print(f"[red]Error processing {item}: {e}[/red]")
                progress.update(task_id, advance=1)
                continue

    return results
