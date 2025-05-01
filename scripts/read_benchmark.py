from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def read_benchmark(
    path: Path = typer.Argument(..., help="Path to the MibiFile"),
    num_chunks: int = typer.Option(16, help="Number of chunks to read"),
):
    """Benchmark the read performance of a MibiFile."""
    from mbt.core.mibifile import MibiFile

    mibi_file = MibiFile(path, num_chunks=num_chunks)
    mibi_file.data  # noqa: B018


if __name__ == "__main__":
    app()
