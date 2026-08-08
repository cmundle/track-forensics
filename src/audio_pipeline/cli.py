"""Typer CLI entry point.

    track-forensics all input.wav              # default path
    track-forensics separate input.wav
    track-forensics analyze input.wav
    track-forensics export-strudel-hints input.wav
    track-forensics doctor

Exit codes, honoured by every command:

    0  success
    1  user error (bad input suffix, missing file, unknown --backend name)
    2  environment error (no analysis backend installed, Demucs/torch
       unavailable, `BackendUnavailableError`)

Nothing here imports `essentia`, `librosa`, `torch`, or `demucs` at module top
level -- see
`tests/test_backends.py::test_importing_the_package_does_not_pull_in_the_analysis_libraries`.
Every heavyweight import happens lazily inside `separate.py` / `backends/`,
which this module only calls into. That is what lets `doctor` run -- and
report the absence usefully -- even when all four are missing or broken.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from . import SCHEMA_VERSION, SUPPORTED_INPUT_SUFFIXES
from . import analyze as analyze_module
from . import separate as separate_module
from . import strudel_hints as strudel_hints_module
from .backends import BackendName, BackendUnavailableError
from .schemas import SourceAnalysis, TrackSummary, rehydrate_stripped_lists

app = typer.Typer(
    name="track-forensics",
    help="Local-only stem separation and audio analysis. Offline, no cloud.",
    no_args_is_help=True,
)

DEFAULT_OUTPUT_ROOT = Path("output")

_VALID_BACKEND_NAMES: tuple[BackendName, ...] = ("essentia", "librosa")

_EXIT_USER_ERROR = 1
_EXIT_ENV_ERROR = 2


# --- shared validation / formatting helpers ---------------------------------


def _fail(message: str, *, code: int) -> NoReturn:
    """Print `message` to stderr in red and exit with `code`. Never returns."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def _warn(message: str) -> None:
    typer.secho(message, fg=typer.colors.YELLOW, err=True)


def _check_input_file(input_file: Path) -> None:
    """Validate `input_file` is a supported, existing file, or exit 1.

    Suffix is checked before existence: it needs no I/O and gives a more
    specific message when both are wrong (e.g. a typo'd path with a `.txt`
    extension). `typer.Argument(..., exists=True)` is deliberately not used
    here -- click's own `exists=True` check exits 2, which would collide with
    this CLI's own meaning for exit code 2 (environment errors).
    """
    suffix = input_file.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        accepted = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        _fail(
            f"Unsupported input file type {suffix!r}. Accepted suffixes: {accepted}.",
            code=_EXIT_USER_ERROR,
        )
    if not input_file.is_file():
        _fail(f"Input file not found: {input_file}", code=_EXIT_USER_ERROR)


def _resolve_model(model: str | None, fast: bool) -> str:
    """Resolve the Demucs model from `--model` / `--fast` / the environment.

    `--model` always wins when both are given -- an explicit model name is a
    more specific instruction than the `--fast` shortcut, so silently letting
    `--fast` override it would surprise a user who typed `--model` on
    purpose. The conflict is not silent, though: a warning names which one
    was used.
    """
    if model is not None and fast:
        _warn(
            f"Both --model {model!r} and --fast were given; --model takes "
            f"precedence and --fast is ignored (would have resolved to "
            f"{separate_module.FAST_MODEL!r})."
        )
        return model
    if model is not None:
        return model
    if fast:
        return separate_module.FAST_MODEL
    return separate_module.default_model()


def _validate_backend_option(backend: str | None) -> BackendName | None:
    if backend is None:
        return None
    if backend not in _VALID_BACKEND_NAMES:
        _fail(
            f"Unknown backend {backend!r}. Choose one of: "
            f"{', '.join(_VALID_BACKEND_NAMES)}.",
            code=_EXIT_USER_ERROR,
        )
    return backend


def _announce_separation_plan(model: str, device: str) -> None:
    """Print the model/device before a Demucs run starts.

    The default `htdemucs_ft` model is roughly 4x slower than `htdemucs`, so a
    run that just sits there for several minutes with no output reads as a
    hang rather than as normal, expected behaviour. Printing this up front,
    before any heavyweight import or model load happens, is the fix.
    """
    typer.echo(f"Separation: model={model!r} device={device!r}")
    if model == separate_module.DEFAULT_MODEL:
        typer.echo(
            f"  {model!r} is the quality-first default and roughly 4x slower than "
            f"{separate_module.FAST_MODEL!r}. Expect minutes, not seconds, on a full "
            "track -- this is not a hang. Pass --fast for a quicker, rougher run."
        )


def _build_summary(
    input_file: Path,
    track_name: str,
    analysis: analyze_module.TrackAnalysis | dict[str, SourceAnalysis],
    backend_name: str,
    separation_model: str | None,
    separation_device: str | None,
) -> TrackSummary:
    """Wrap one run's analysis in the identity and provenance only the CLI knows.

    Accepts a bare source mapping as well as a `TrackAnalysis`, because that is
    what `analyze_track` returned before schema v5 put one tempo and one
    structure at track level. The track-level blocks then fall back to their
    empty defaults, which is the honest result for a caller that never measured
    them.
    """
    sources = analysis.sources if isinstance(analysis, analyze_module.TrackAnalysis) else analysis
    mix = sources.get("mix")
    duration_seconds = mix.duration_seconds if mix is not None else 0.0
    summary = TrackSummary(
        track_name=track_name,
        input_path=str(input_file),
        duration_seconds=duration_seconds,
        backend=backend_name,
        separation_model=separation_model,
        separation_device=separation_device,
        sources=sources,
    )
    if isinstance(analysis, analyze_module.TrackAnalysis):
        summary.tempo = analysis.tempo
        summary.downbeat = analysis.downbeat
        summary.arrangement = analysis.arrangement
    return summary


def _load_track_summary_with_timing(output_root: Path, track_name: str) -> TrackSummary:
    """Load `track_summary.json`, rehydrating every list `summary_payload()` strips.

    `TrackSummary.summary_payload()` strips `rhythm.beat_times`/`onset_times`,
    `drum_decomposition.hits` and `bass_line.notes` down to their counts
    before writing -- see `schemas._SUMMARY_LIST_FIELDS`. That is the right
    call for the file meant to be read by hand, but several downstream
    functions need the full lists:
    `strudel_hints.infer_subdivision_feel()` needs the drums stem's actual
    `rhythm.onset_times`, and `strudel_vocab.suggest_drum_sounds()` needs
    `drum_decomposition.hits` (per-hit `decay_ratio` is what decides `hh`
    versus `oh`; the folded `patterns` list does not carry it). A
    `TrackSummary` loaded straight from the stripped file has none of these:
    pydantic quietly defaults each to `[]` (extra keys like `beat_count` are
    ignored, the missing list falls back to its `default_factory=list`), so
    `suggest_drum_sounds()` would not raise -- it would just silently return
    `[]`, indistinguishable from "no kick/snare/hat hits were ever found."
    `export-strudel-hints` run standalone would then quietly produce a
    weaker result than `all` does, which holds the summary in memory with
    every list still intact.

    None of these lists are actually lost, though -- they stay complete in
    each source's own `analysis/<source>.json`. So this rehydrates all of
    them via `schemas.rehydrate_stripped_lists()`, the one table-driven
    helper that strips and rehydrate both read, before handing the summary
    to `strudel_hints.build()` -- rather than leaving each caller to
    separately notice which lists it needs back. A source whose analysis
    file is missing (partial run, manually deleted) is left with every list
    empty and reported via `missing_analysis_sources`, rather than raising --
    hints should still degrade gracefully.
    """
    summary_path = analyze_module.track_summary_path(output_root, track_name)
    summary = TrackSummary.model_validate_json(summary_path.read_text())

    missing_analysis_sources: list[str] = []
    for source_name, analysis in summary.sources.items():
        source_path = analyze_module.source_analysis_path(output_root, track_name, source_name)
        if not source_path.is_file():
            missing_analysis_sources.append(source_name)
            continue
        full = SourceAnalysis.model_validate_json(source_path.read_text())
        rehydrate_stripped_lists(analysis, full)

    if missing_analysis_sources:
        _warn(
            "Per-source analysis files missing for: "
            f"{', '.join(sorted(missing_analysis_sources))}. Beat/onset times, "
            "drum hits and bass notes for these sources could not be "
            "rehydrated, so hints derived from them (e.g. subdivision feel "
            "from a missing drums.json, or sound suggestions from a missing "
            "drums.json/bass.json) may come back None or empty."
        )
    return summary


# --- commands ----------------------------------------------------------------


@app.command()
def separate(
    input_file: Annotated[
        Path, typer.Argument(help="Input audio file to separate into stems.")
    ],
    output_root: Annotated[
        Path, typer.Option("--output", "-o", help="Output root directory.")
    ] = DEFAULT_OUTPUT_ROOT,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Demucs model name. Defaults via TRACK_FORENSICS_MODEL."),
    ] = None,
    fast: Annotated[
        bool,
        typer.Option(
            "--fast", help=f"Shortcut for --model {separate_module.FAST_MODEL}."
        ),
    ] = False,
    device: Annotated[
        str | None, typer.Option("--device", help="mps | cpu (auto-detected).")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-run even if stems already exist.")
    ] = False,
) -> None:
    """Separate an input track into drums, bass, vocals and other."""
    _check_input_file(input_file)
    resolved_model = _resolve_model(model, fast)
    resolved_device = device or separate_module.pick_device()
    _announce_separation_plan(resolved_model, resolved_device)

    try:
        result = separate_module.separate(
            input_file,
            output_root,
            model=resolved_model,
            device=resolved_device,
            force=force,
        )
    except ImportError as exc:
        _fail(
            f"Demucs separation requires torch and demucs, which are not usable: {exc}",
            code=_EXIT_ENV_ERROR,
        )

    if result.skipped:
        typer.echo(
            f"Skipped: stems already exist for {input_file.name!r} "
            f"(model/device shown above are what was requested, not necessarily "
            "what produced the existing files -- pass --force to re-run)."
        )
    else:
        typer.echo(f"Separated using model={result.model!r} device={result.device!r}")
    for stem_name, stem_path in result.stems.items():
        typer.echo(f"  {stem_name}: {stem_path}")


@app.command()
def analyze(
    input_file: Annotated[
        Path, typer.Argument(help="Input audio file to analyze (mix plus any existing stems).")
    ],
    output_root: Annotated[
        Path, typer.Option("--output", "-o", help="Output root directory.")
    ] = DEFAULT_OUTPUT_ROOT,
    backend: Annotated[
        str | None, typer.Option("--backend", help="essentia | librosa (auto-detected).")
    ] = None,
) -> None:
    """Analyze the mix and any existing stems, writing analysis/*.json."""
    _check_input_file(input_file)
    validated_backend = _validate_backend_option(backend)

    try:
        resolved_backend = analyze_module.get_backend(validated_backend)
    except BackendUnavailableError as exc:
        _fail(str(exc), code=_EXIT_ENV_ERROR)

    typer.echo(f"Analysis backend: {resolved_backend.name!r}")

    track_name = separate_module.slugify(input_file.stem)
    stems = separate_module.stem_paths(output_root, track_name)
    started = time.perf_counter()
    analysis = analyze_module.analyze_track(input_file, stems, backend=resolved_backend)
    elapsed = time.perf_counter() - started
    summary = _build_summary(
        input_file,
        track_name,
        analysis,
        resolved_backend.name,
        separation_model=None,
        separation_device=None,
    )
    written = analyze_module.write_analysis_outputs(summary, output_root)
    typer.echo(
        f"Analysis took {elapsed:.1f}s (includes drum decomposition and bass pitch tracking)."
    )
    for name, path in written.items():
        typer.echo(f"Wrote {name}: {path}")


@app.command(name="export-strudel-hints")
def export_strudel_hints(
    input_file: Annotated[
        Path, typer.Argument(help="Input audio file whose track_summary.json to read.")
    ],
    output_root: Annotated[
        Path, typer.Option("--output", "-o", help="Output root directory.")
    ] = DEFAULT_OUTPUT_ROOT,
) -> None:
    """Write strudel_hints.json from an existing track_summary.json."""
    _check_input_file(input_file)
    track_name = separate_module.slugify(input_file.stem)
    summary_path = analyze_module.track_summary_path(output_root, track_name)

    if not summary_path.is_file():
        _fail(
            f"No track_summary.json found at {summary_path}. Run "
            f"`track-forensics analyze {input_file}` or "
            f"`track-forensics all {input_file}` first.",
            code=_EXIT_USER_ERROR,
        )

    summary = _load_track_summary_with_timing(output_root, track_name)
    if summary.schema_version != SCHEMA_VERSION:
        _warn(
            f"{summary_path} was written with schema_version="
            f"{summary.schema_version}, this build expects {SCHEMA_VERSION}. "
            "Proceeding on a best-effort basis -- fields this version added or "
            "renamed may come back None."
        )

    hints = strudel_hints_module.build(summary)
    hints_path = output_root / track_name / "strudel_hints.json"
    hints_path.parent.mkdir(parents=True, exist_ok=True)
    hints_path.write_text(json.dumps(hints.model_dump(mode="json"), indent=2) + "\n")

    typer.echo(f"Wrote strudel hints: {hints_path}")
    for note in hints.notes:
        typer.echo(f"  note: {note}")


@app.command(name="all")
def run_all(
    input_file: Annotated[
        Path, typer.Argument(help="Input audio file to process end-to-end.")
    ],
    output_root: Annotated[
        Path, typer.Option("--output", "-o", help="Output root directory.")
    ] = DEFAULT_OUTPUT_ROOT,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Demucs model name. Defaults via TRACK_FORENSICS_MODEL."),
    ] = None,
    fast: Annotated[
        bool,
        typer.Option(
            "--fast", help=f"Shortcut for --model {separate_module.FAST_MODEL}."
        ),
    ] = False,
    device: Annotated[
        str | None, typer.Option("--device", help="mps | cpu (auto-detected).")
    ] = None,
    backend: Annotated[
        str | None, typer.Option("--backend", help="essentia | librosa (auto-detected).")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-run separation even if stems already exist.")
    ] = False,
) -> None:
    """Separate, analyze, and export hints in one pass. The usual entry point."""
    _check_input_file(input_file)
    resolved_model = _resolve_model(model, fast)
    resolved_device = device or separate_module.pick_device()
    validated_backend = _validate_backend_option(backend)

    # Resolve the analysis backend up front, before the (much slower)
    # separation phase runs, so an environment error (no backend installed)
    # is reported in seconds rather than after a multi-minute Demucs run.
    try:
        resolved_backend = analyze_module.get_backend(validated_backend)
    except BackendUnavailableError as exc:
        _fail(str(exc), code=_EXIT_ENV_ERROR)

    typer.echo("== track-forensics all ==")
    _announce_separation_plan(resolved_model, resolved_device)
    typer.echo(f"Analysis backend: {resolved_backend.name!r}")

    typer.echo("\n[1/3] Separating stems...")
    try:
        separation = separate_module.separate(
            input_file,
            output_root,
            model=resolved_model,
            device=resolved_device,
            force=force,
        )
    except ImportError as exc:
        _fail(
            f"Demucs separation requires torch and demucs, which are not usable: {exc}",
            code=_EXIT_ENV_ERROR,
        )
    if separation.skipped:
        typer.echo("  Skipped: stems already exist (pass --force to re-run).")
    else:
        typer.echo(f"  Done: model={separation.model!r} device={separation.device!r}")

    typer.echo(
        "\n[2/3] Analyzing mix and stems "
        "(includes drum decomposition and bass pitch tracking)..."
    )
    track_name = separate_module.slugify(input_file.stem)
    analyze_started = time.perf_counter()
    analysis = analyze_module.analyze_track(
        input_file, separation.stems, backend=resolved_backend
    )
    analyze_elapsed = time.perf_counter() - analyze_started
    summary = _build_summary(
        input_file,
        track_name,
        analysis,
        resolved_backend.name,
        separation_model=separation.model,
        separation_device=separation.device,
    )
    written = analyze_module.write_analysis_outputs(summary, output_root)
    typer.echo(f"  Wrote {len(written)} analysis file(s) in {analyze_elapsed:.1f}s.")

    typer.echo("\n[3/3] Building Strudel hints...")
    hints = strudel_hints_module.build(summary)
    hints_path = output_root / track_name / "strudel_hints.json"
    hints_path.parent.mkdir(parents=True, exist_ok=True)
    hints_path.write_text(json.dumps(hints.model_dump(mode="json"), indent=2) + "\n")
    typer.echo(f"  Wrote {hints_path}")
    for note in hints.notes:
        typer.echo(f"  note: {note}")

    typer.echo("\n== Summary ==")
    for name, path in written.items():
        typer.echo(f"  {name}: {path}")
    typer.echo(f"  strudel_hints: {hints_path}")


@app.command()
def doctor() -> None:
    """Report Python version, FFmpeg, Demucs device, and active analysis backend.

    This is the first thing to run when something breaks, so every probe here
    is defensive: a missing or broken dependency is reported as absent, never
    raised. Always exits 0 -- diagnosing a broken environment is the whole
    point, so the command itself must survive one.
    """
    typer.echo(f"Python: {sys.version.split()[0]} ({sys.executable})")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        typer.echo(f"FFmpeg: found ({ffmpeg_path})")
    else:
        _warn("FFmpeg: NOT found on PATH -- non-wav input will fail to decode.")

    typer.echo("")
    typer.echo("-- Demucs / separation --")
    try:
        # Lazy import, matching the module-wide policy: torch must not load
        # merely because `doctor` wants to report its version.
        import torch

        typer.echo(f"torch: {torch.__version__}")
        try:
            mps_available = bool(torch.backends.mps.is_available())
        except Exception as exc:  # noqa: BLE001 - any probe failure means "not usable"
            mps_available = False
            typer.echo(f"  MPS availability check failed ({type(exc).__name__}: {exc})")
        typer.echo(f"  MPS available: {mps_available}")
    except ImportError:
        _warn("torch: NOT installed -- `separate`/`all` cannot run.")

    try:
        resolved_device = separate_module.pick_device()
    except Exception as exc:  # noqa: BLE001 - doctor must never raise
        resolved_device = "cpu"
        _warn(f"Device resolution raised ({type(exc).__name__}: {exc}); reporting 'cpu'.")
    typer.echo(f"Resolved Demucs device: {resolved_device!r}")

    try:
        from_env = separate_module.MODEL_ENV_VAR in os.environ
        resolved_model = separate_module.default_model()
    except Exception as exc:  # noqa: BLE001 - doctor must never raise
        from_env = False
        resolved_model = separate_module.DEFAULT_MODEL
        _warn(f"Model resolution raised ({type(exc).__name__}: {exc}); reporting the default.")
    source = f"{separate_module.MODEL_ENV_VAR} environment variable" if from_env else "default"
    typer.echo(f"Resolved Demucs model: {resolved_model!r} (from {source})")

    typer.echo("")
    typer.echo("-- Analysis backend --")
    try:
        installed = analyze_module.available_backends()
    except Exception as exc:  # noqa: BLE001 - doctor must never raise
        installed = []
        _warn(f"Backend discovery raised ({type(exc).__name__}: {exc}).")
    typer.echo(f"Installed backends: {', '.join(installed) if installed else 'none'}")

    try:
        active = analyze_module.get_backend()
        typer.echo(f"Active backend: {active.name!r}")
    except BackendUnavailableError as exc:
        _warn(f"Active backend: none available -- {exc}")
    except Exception as exc:  # noqa: BLE001 - doctor must never raise
        _warn(f"Active backend resolution raised unexpectedly ({type(exc).__name__}: {exc}).")


if __name__ == "__main__":
    app()
