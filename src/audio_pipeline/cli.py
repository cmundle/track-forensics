"""Typer CLI entry point.

    track-forensics all input.wav              # default path
    track-forensics separate input.wav
    track-forensics analyze input.wav
    track-forensics export-strudel-hints input.wav
    track-forensics doctor
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="track-forensics",
    help="Local-only stem separation and audio analysis. Offline, no cloud.",
    no_args_is_help=True,
)

DEFAULT_OUTPUT_ROOT = Path("output")


@app.command()
def separate(
    input_file: Path = typer.Argument(..., exists=True, readable=True),
    output_root: Path = typer.Option(DEFAULT_OUTPUT_ROOT, "--output", "-o"),
    model: str = typer.Option("htdemucs", "--model"),
    device: str | None = typer.Option(None, "--device", help="mps | cpu (auto-detected)"),
    force: bool = typer.Option(False, "--force", help="Re-run even if stems exist"),
) -> None:
    """Separate an input track into drums, bass, vocals and other."""
    raise NotImplementedError


@app.command()
def analyze(
    input_file: Path = typer.Argument(..., exists=True, readable=True),
    output_root: Path = typer.Option(DEFAULT_OUTPUT_ROOT, "--output", "-o"),
    backend: str | None = typer.Option(None, "--backend", help="essentia | librosa"),
) -> None:
    """Analyze the mix and any existing stems, writing analysis/*.json."""
    raise NotImplementedError


@app.command(name="export-strudel-hints")
def export_strudel_hints(
    input_file: Path = typer.Argument(..., exists=True, readable=True),
    output_root: Path = typer.Option(DEFAULT_OUTPUT_ROOT, "--output", "-o"),
) -> None:
    """Write strudel_hints.json from an existing track_summary.json."""
    raise NotImplementedError


@app.command(name="all")
def run_all(
    input_file: Path = typer.Argument(..., exists=True, readable=True),
    output_root: Path = typer.Option(DEFAULT_OUTPUT_ROOT, "--output", "-o"),
    model: str = typer.Option("htdemucs", "--model"),
    device: str | None = typer.Option(None, "--device"),
    backend: str | None = typer.Option(None, "--backend"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Separate, analyze, and export hints in one pass. The usual entry point."""
    raise NotImplementedError


@app.command()
def doctor() -> None:
    """Report Python version, FFmpeg, Demucs device, and active analysis backend."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
