"""Analysis orchestration: resolve a backend, loop over sources, write JSON.

Feature extraction itself lives in `backends/`; decoding lives in `audio_io`.
This module only sequences them, assembles `SourceAnalysis`/`TrackSummary`
models, and writes the JSON artefacts under `<output_root>/<track_name>/`.

Failure isolation is the load-bearing behaviour here, at two levels:

* Within one source, each of the four backend feature methods
  (`rhythm`/`tonal`/`spectral`/`dynamics`) is called independently. One
  raising does not take the other three down with it -- see
  `_call_backend_method`. Heuristic labelling is isolated the same way.
* Across sources, `analyze_track` catches anything `analyze_source` did not
  itself absorb (e.g. a corrupt stem file that fails to decode at all) and
  records a fully-`None` placeholder for that source rather than losing the
  mix and the other three stems. A five-minute Demucs run followed by a crash
  on stem four that discards everything is the failure mode this exists to
  prevent.

`AnalysisBackend`, `BackendName`, `BackendUnavailableError`,
`available_backends`, and `get_backend` are re-exported here for callers that
predate the `backends` package split (the CLI in particular).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from . import ANALYSIS_SAMPLE_RATE, STEM_NAMES
from . import heuristics as heuristics_module
from .audio_io import AudioArray, load_audio, to_mono
from .backends import (
    AnalysisBackend,
    BackendName,
    BackendUnavailableError,
    available_backends,
    get_backend,
)
from .schemas import (
    DynamicsFeatures,
    RhythmFeatures,
    SourceAnalysis,
    SpectralFeatures,
    TonalFeatures,
    TrackSummary,
)

__all__ = [
    "AnalysisBackend",
    "BackendName",
    "BackendUnavailableError",
    "analysis_output_dir",
    "analyze_source",
    "analyze_track",
    "available_backends",
    "get_backend",
    "round_floats",
    "source_analysis_path",
    "track_summary_path",
    "write_analysis_outputs",
    "write_source_analysis",
    "write_track_summary",
]

logger = logging.getLogger(__name__)

_FeatureModel = TypeVar("_FeatureModel", bound=BaseModel)

#: Decimal places every float is rounded to before a JSON artefact is written,
#: so two runs over the same audio diff cleanly instead of wobbling in the 9th
#: decimal place of a float32-derived value. Not a nicety -- see `round_floats`.
_JSON_FLOAT_NDIGITS = 6


def _call_backend_method(
    method: Callable[[], _FeatureModel],
    *,
    category: str,
    source: str,
    backend_name: str,
    default: _FeatureModel,
) -> _FeatureModel:
    """Call one backend feature method, degrading to `default` on any exception.

    A single failed descriptor category must never take the rest of the
    source down with it. The exception is logged, not silently swallowed;
    the resulting all-`None` model then feeds into `_collect_unavailable`
    exactly like a working backend's partial `None`s would -- the two cases
    (a raise vs. a backend that legitimately returns `None`) end up as the
    same entries in `unavailable_features`. They are told apart only in the
    log: a raise logs a warning naming the exception, a `None` from a working
    backend does not, since returning `None` for something a backend cannot
    compute is expected behaviour, not something worth a warning.
    """
    try:
        return method()
    except Exception as exc:  # noqa: BLE001 - any backend failure must degrade, not crash
        logger.warning(
            "%s backend failed to compute %r features for source %r (%s: %s); "
            "recording as unavailable and continuing.",
            backend_name,
            category,
            source,
            type(exc).__name__,
            exc,
        )
        return default


def _collect_unavailable(prefix: str, data: dict[str, object]) -> list[str]:
    """Dotted field paths inside `data` whose value is `None` or an empty list.

    Operates on a plain `model_dump()` dict rather than the model itself so
    nested models (e.g. `SpectralFeatures.band_energy_ratios`) are walked the
    same way as everything else, recursively.
    """
    missing: list[str] = []
    for key, value in data.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            missing.extend(_collect_unavailable(path, value))
        elif isinstance(value, list):
            if not value:
                missing.append(path)
        elif value is None:
            missing.append(path)
    return missing


def _unavailable_features(
    rhythm: RhythmFeatures,
    tonal: TonalFeatures,
    spectral: SpectralFeatures,
    dynamics: DynamicsFeatures,
) -> list[str]:
    combined = {
        *_collect_unavailable("rhythm", rhythm.model_dump()),
        *_collect_unavailable("tonal", tonal.model_dump()),
        *_collect_unavailable("spectral", spectral.model_dump()),
        *_collect_unavailable("dynamics", dynamics.model_dump()),
    }
    return sorted(combined)


def analyze_source(
    path: Path,
    source: str,
    backend: AnalysisBackend | None = None,
) -> SourceAnalysis:
    """Analyze one audio file and attach heuristic labels.

    Loads the file once via `audio_io.load_audio(..., mono=False)` and
    derives the mono signal locally with `audio_io.to_mono` rather than
    decoding twice: `rhythm`/`tonal`/`spectral` want mono, `dynamics` (LUFS)
    needs the original channel count, and a second full decode+resample pass
    over the same file would roughly double this function's cost for no
    benefit. `to_mono` is a cheap in-memory average, not a re-decode.

    Each of the four backend methods is called independently and may fail
    without losing the other three -- see `_call_backend_method`. Heuristic
    labelling is likewise isolated: `heuristics.apply` may still be an
    unfinished stub raising `NotImplementedError` while W1D is in flight, and
    a raise there degrades to an empty label list rather than losing the
    rest of the analysis. `heuristics_module.apply` is called through the
    module object (not imported by name) specifically so tests can
    monkeypatch `audio_pipeline.heuristics.apply` without needing W1D done.

    Backend resolution defaults to `get_backend()`'s normal preference order
    when `backend` is not given, which is convenient for one-off calls; when
    called from `analyze_track`, the *same* resolved backend instance is
    passed in explicitly so the mix and every stem in one run are analyzed
    by the same backend.

    Raises whatever `audio_io.load_audio` or backend resolution raises --
    e.g. a missing file, an undecodable format, or `BackendUnavailableError`.
    This function analyzes a single, already-known-to-exist source; the "one
    bad stem must not lose the others" isolation lives one level up, in
    `analyze_track`, which is where "the file did not even load" failures
    are caught and turned into a placeholder instead of propagating.
    """
    resolved_backend = backend if backend is not None else get_backend()

    stereo, sample_rate = load_audio(path, mono=False)
    mono: AudioArray = to_mono(stereo)
    duration_seconds = float(mono.shape[0]) / float(sample_rate) if sample_rate else 0.0

    rhythm = _call_backend_method(
        lambda: resolved_backend.rhythm(mono, sample_rate),
        category="rhythm",
        source=source,
        backend_name=resolved_backend.name,
        default=RhythmFeatures(),
    )
    tonal = _call_backend_method(
        lambda: resolved_backend.tonal(mono, sample_rate),
        category="tonal",
        source=source,
        backend_name=resolved_backend.name,
        default=TonalFeatures(),
    )
    spectral = _call_backend_method(
        lambda: resolved_backend.spectral(mono, sample_rate),
        category="spectral",
        source=source,
        backend_name=resolved_backend.name,
        default=SpectralFeatures(),
    )
    dynamics = _call_backend_method(
        lambda: resolved_backend.dynamics(stereo, sample_rate),
        category="dynamics",
        source=source,
        backend_name=resolved_backend.name,
        default=DynamicsFeatures(),
    )

    analysis = SourceAnalysis(
        source=source,
        audio_path=str(path),
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        backend=resolved_backend.name,
        rhythm=rhythm,
        tonal=tonal,
        spectral=spectral,
        dynamics=dynamics,
        unavailable_features=_unavailable_features(rhythm, tonal, spectral, dynamics),
    )

    try:
        analysis.labels = heuristics_module.apply(analysis)
    except Exception as exc:  # noqa: BLE001 - a bad label must not lose the analysis
        logger.warning(
            "Heuristic labelling failed for source %r (%s: %s); continuing with no labels.",
            source,
            type(exc).__name__,
            exc,
        )
        analysis.labels = []

    return analysis


def _placeholder_analysis(
    path: Path, source: str, backend_name: str, error: Exception
) -> SourceAnalysis:
    """An all-`None` `SourceAnalysis` recording a total per-source failure.

    Used by `analyze_track` when `analyze_source` itself raises (a corrupt
    file, a decode failure -- something no per-descriptor try/except inside
    `analyze_source` could have caught). Keeps the source present in the
    result dict, rather than absent, so callers can see it was attempted and
    why it failed instead of it silently vanishing.
    """
    rhythm, tonal, spectral, dynamics = (
        RhythmFeatures(),
        TonalFeatures(),
        SpectralFeatures(),
        DynamicsFeatures(),
    )
    unavailable = _unavailable_features(rhythm, tonal, spectral, dynamics)
    unavailable.append(f"source (analysis failed: {type(error).__name__}: {error})")
    return SourceAnalysis(
        source=source,
        audio_path=str(path),
        duration_seconds=0.0,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        backend=backend_name,
        rhythm=rhythm,
        tonal=tonal,
        spectral=spectral,
        dynamics=dynamics,
        labels=[],
        unavailable_features=unavailable,
    )


def _analyze_or_placeholder(path: Path, source: str, backend: AnalysisBackend) -> SourceAnalysis:
    try:
        return analyze_source(path, source, backend)
    except Exception as exc:  # noqa: BLE001 - one source's failure must not lose the others
        logger.error(
            "Analysis failed entirely for source %r (%s: %s); continuing with the "
            "remaining sources.",
            source,
            type(exc).__name__,
            exc,
        )
        return _placeholder_analysis(path, source, backend.name, exc)


def analyze_track(
    input_path: Path,
    stems: dict[str, Path],
    backend: AnalysisBackend | None = None,
) -> dict[str, SourceAnalysis]:
    """Analyze the mix plus every stem present. Keys: mix, drums, bass, vocals, other.

    The backend is resolved once (or accepted as given) and reused for every
    source, so one run's mix and stems are always analyzed by the same
    backend -- silently mixing librosa and essentia numbers within a single
    run would make the sources incomparable with each other.
    `BackendUnavailableError` from resolution is *not* caught here: it is an
    environment problem (nothing usable is installed), not a per-source one,
    and the CLI is expected to treat it as a distinct exit code from "one
    stem's file was unreadable".

    A stem name absent from `stems`, or present but pointing at a path that
    does not exist on disk (e.g. `separate` has not been run yet), is skipped
    with a logged warning rather than an error -- `analyze` must work on a
    bare mix alone, before separation has ever run.

    Any other per-source failure is caught in `_analyze_or_placeholder` and
    recorded as a fully-unavailable placeholder for that source, so one bad
    stem never costs the mix or the other stems their results.
    """
    resolved_backend = backend if backend is not None else get_backend()

    results: dict[str, SourceAnalysis] = {
        "mix": _analyze_or_placeholder(input_path, "mix", resolved_backend)
    }

    for stem_name in STEM_NAMES:
        stem_path = stems.get(stem_name)
        if stem_path is None or not Path(stem_path).is_file():
            logger.warning(
                "Skipping stem %r: no stem file found at %s (run `separate` first).",
                stem_name,
                stem_path,
            )
            continue
        results[stem_name] = _analyze_or_placeholder(stem_path, stem_name, resolved_backend)

    return results


def round_floats(value: object, ndigits: int = _JSON_FLOAT_NDIGITS) -> object:
    """Recursively round every float inside `value` to `ndigits` decimals.

    Meant to run on an already-`model_dump(mode="json")`-ed payload (plain
    dict/list/scalar, no pydantic models left) right before serialisation, so
    two runs over the same audio produce output that diffs cleanly instead of
    wobbling in the low decimal places of a float32-derived value. This is a
    real requirement, not cosmetic -- see the module and package docstrings.

    `bool` is deliberately left alone: it is a subclass of `int` in Python,
    never of `float`, so it never matches the `isinstance(value, float)`
    branch and needs no special case.
    """
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {key: round_floats(item, ndigits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, ndigits) for item in value]
    return value


def analysis_output_dir(output_root: Path, track_name: str) -> Path:
    """Where per-source analysis files live: `<output_root>/<track_name>/analysis/`.

    `output_root` is the same *global* root passed to `separate()` (e.g.
    `Path("output")`), not the per-track directory -- mirroring
    `separate.stem_paths`, which builds `output_root/track_name/stems/...`
    the same way.
    """
    return output_root / track_name / "analysis"


def source_analysis_path(output_root: Path, track_name: str, source: str) -> Path:
    """Path for one source's analysis file: `.../analysis/<source>.json`."""
    return analysis_output_dir(output_root, track_name) / f"{source}.json"


def track_summary_path(output_root: Path, track_name: str) -> Path:
    """Path for the track summary: `<output_root>/<track_name>/track_summary.json`."""
    return output_root / track_name / "track_summary.json"


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rounded = round_floats(payload)
    path.write_text(json.dumps(rounded, indent=2, sort_keys=False) + "\n")
    return path


def write_source_analysis(analysis: SourceAnalysis, output_root: Path, track_name: str) -> Path:
    """Write one `analysis/<source>.json`, beat times included, floats rounded.

    Key order matches `SourceAnalysis`'s field declaration order --
    `model_dump(mode="json")` preserves it and `sort_keys=False` here keeps
    it -- which, combined with the float rounding, is what makes two runs
    over the same audio diff cleanly instead of noisily.
    """
    path = source_analysis_path(output_root, track_name, analysis.source)
    return _write_json(path, analysis.model_dump(mode="json"))


def write_track_summary(summary: TrackSummary, output_root: Path) -> Path:
    """Write `track_summary.json` via `TrackSummary.summary_payload()`.

    `summary_payload()` already strips `beat_times` in favour of `beat_count`;
    this function adds float rounding and stable-order serialisation on top,
    the same treatment `write_source_analysis` gives the per-source files.
    """
    path = track_summary_path(output_root, summary.track_name)
    return _write_json(path, summary.summary_payload())


def write_analysis_outputs(summary: TrackSummary, output_root: Path) -> dict[str, Path]:
    """Write every source's `analysis/<source>.json` plus `track_summary.json`.

    Returns the written paths keyed by source name (`mix`, `drums`, `bass`,
    `vocals`, `other`, whichever are present in `summary.sources`) plus
    `"track_summary"` for the summary file, so a caller (the CLI) can report
    exactly what landed on disk without re-deriving the paths itself.
    """
    written: dict[str, Path] = {
        name: write_source_analysis(analysis, output_root, summary.track_name)
        for name, analysis in summary.sources.items()
    }
    written["track_summary"] = write_track_summary(summary, output_root)
    return written
