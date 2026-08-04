"""Analysis orchestration: resolve a backend, loop over sources, write JSON.

Feature extraction itself lives in `backends/`; decoding lives in `audio_io`.
This module only sequences them, assembles `SourceAnalysis`/`TrackSummary`
models, and writes the JSON artefacts under `<output_root>/<track_name>/`.

Failure isolation is the load-bearing behaviour here, at two levels:

* Within one source, each of the four backend feature methods
  (`rhythm`/`tonal`/`spectral`/`dynamics`) is called independently. One
  raising does not take the other three down with it -- see
  `_call_backend_method`. Heuristic labelling is isolated the same way, and
  so are the Wave 4 blocks: `drum_elements.decompose()` (drums only) and
  `backend.pitch()` + `note_track.segment_notes()` (bass only) -- see
  `_drum_decomposition_for_source` and `_bass_line_for_source`.
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

from . import ANALYSIS_SAMPLE_RATE, STEM_NAMES, drum_elements, note_track
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
    BassLine,
    DrumDecomposition,
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
    drum_decomposition: DrumDecomposition,
    bass_line: BassLine,
) -> list[str]:
    """Dotted paths for every missing descriptor, plus the two Wave 4 blocks.

    `drum_decomposition` and `bass_line` are deliberately **not** routed
    through `_collect_unavailable`: that helper treats an empty list as
    unavailable and does not recurse into a populated one, which is backwards
    for these two. A `DrumDecomposition` with ten real hits and one caveat
    would be read as fully present by field-emptiness alone (its `hits` list
    is non-empty, nothing else about it is inspected), and a `BassLine` on
    every source that is not `bass` -- the overwhelming common case, since it
    is `not_attempted` there by policy -- would be read as fully *missing*,
    flooding every mix/drums/vocals/other analysis with a spurious
    `bass_line.notes` entry.

    `status` already says everything worth saying here: `ok` and
    `not_attempted` are the two states with nothing to report; `no_grid`,
    `too_few_hits`, `unvoiced` and `failed` are exactly what this list exists
    to surface. This keeps every pre-Wave-4 `unavailable_features == []`
    assertion passing untouched, since a source where neither block was
    attempted contributes nothing.
    """
    combined = {
        *_collect_unavailable("rhythm", rhythm.model_dump()),
        *_collect_unavailable("tonal", tonal.model_dump()),
        *_collect_unavailable("spectral", spectral.model_dump()),
        *_collect_unavailable("dynamics", dynamics.model_dump()),
    }
    if drum_decomposition.status not in ("ok", "not_attempted"):
        combined.add(f"drum_decomposition (status={drum_decomposition.status})")
    if bass_line.status not in ("ok", "not_attempted"):
        combined.add(f"bass_line (status={bass_line.status})")
    return sorted(combined)


def _bass_grid(
    decomposition: DrumDecomposition | None,
) -> tuple[float | None, float | None, int | None]:
    """`(grid_anchor_seconds, step_seconds, steps_per_cycle)` for `segment_notes()`.

    Reads the drums stem's own `DrumDecomposition` so bass notes land on the
    exact grid the drums define -- `note_track.segment_notes()`'s
    `step_seconds` is the drums grid's `cycle_seconds / steps_per_cycle`.
    `None` in every field -- no drums stem was analysed (or was never even
    reached yet), or its decomposition never settled on a usable grid
    (`no_grid`/`too_few_hits`/`failed`, any of which leave `steps_per_cycle`
    or `cycle_seconds` unset) -- degrades to "no grid", which
    `segment_notes()` already treats as "report `step=None` on every note"
    rather than raising. A wrong grid is worse than none, the same bias
    `drum_elements` and `strudel_hints` already take.
    """
    if decomposition is None:
        return None, None, None
    if decomposition.steps_per_cycle is None or not decomposition.cycle_seconds:
        return None, None, None
    return (
        decomposition.grid_anchor_seconds,
        decomposition.cycle_seconds / decomposition.steps_per_cycle,
        decomposition.steps_per_cycle,
    )


def _drum_decomposition_for_source(
    source: str,
    mono: AudioArray,
    sample_rate: int,
    rhythm: RhythmFeatures,
) -> DrumDecomposition:
    """`drum_elements.decompose()`, called only for the `drums` source.

    A policy of `analyze.py`, not of `drum_elements`: every other source
    keeps the default `status="not_attempted"`. Uses the drums source's own
    already-computed `rhythm.bpm` / `rhythm.beat_times` -- not the mix's --
    so the grid is anchored on what was actually measured for this stem.

    `decompose()` already never raises (it wraps its own body and returns
    `status="failed"` on any internal error -- see its docstring). This is
    defence in depth in the same spirit as `_call_backend_method`: a future
    change to that guarantee must not be able to take the rest of this
    source's analysis down with it.
    """
    if source != "drums":
        return DrumDecomposition()
    try:
        return drum_elements.decompose(
            mono, sample_rate, bpm=rhythm.bpm, beat_times=rhythm.beat_times
        )
    except Exception as exc:  # noqa: BLE001 - a drum-decomposition bug must not lose the analysis
        logger.warning(
            "Drum decomposition failed for source %r (%s: %s); recording as failed.",
            source,
            type(exc).__name__,
            exc,
        )
        return DrumDecomposition(
            status="failed",
            caveats=[f"drum decomposition raised {type(exc).__name__}: {exc}"],
        )


def _bass_line_for_source(
    source: str,
    backend: AnalysisBackend,
    mono: AudioArray,
    sample_rate: int,
    drum_decomposition: DrumDecomposition | None,
) -> BassLine:
    """`backend.pitch()` then `note_track.segment_notes()`, only for `bass`.

    A policy of `analyze.py`, not of the `AnalysisBackend` Protocol -- see
    that Protocol's own docstring for the full reasoning. It matters for
    performance, not just symmetry: `librosa.pyin` runs at roughly 0.12x real
    time (~35 s on a five-minute track), by far the most expensive single
    call this module makes, so calling it for all five sources would roughly
    quintuple that cost for four sources that gain nothing from it.

    `backend.pitch()` can raise (a real backend calling into `librosa.pyin`
    or Essentia's YIN family); `note_track.segment_notes()` cannot -- it
    wraps its own body and returns `status="failed"` on any internal error.
    Both are covered here so a break in either guarantee cannot take the rest
    of the source's analysis down with it.

    `drum_decomposition` is the drums stem's own block (or `None`), threaded
    through by `analyze_track` so notes land on the same grid the drums
    define -- see `_bass_grid`.
    """
    if source != "bass":
        return BassLine()

    try:
        pitch_track = backend.pitch(mono, sample_rate)
    except Exception as exc:  # noqa: BLE001 - a bad pitch tracker must not lose the analysis
        logger.warning(
            "%s backend failed to compute pitch for source %r (%s: %s); "
            "recording bass_line as failed.",
            backend.name,
            source,
            type(exc).__name__,
            exc,
        )
        return BassLine(
            status="failed",
            caveats=[f"pitch() raised {type(exc).__name__}: {exc}"],
        )

    grid_anchor_seconds, step_seconds, steps_per_cycle = _bass_grid(drum_decomposition)
    return note_track.segment_notes(
        pitch_track,
        grid_anchor_seconds=grid_anchor_seconds,
        step_seconds=step_seconds,
        steps_per_cycle=steps_per_cycle,
    )


def analyze_source(
    path: Path,
    source: str,
    backend: AnalysisBackend | None = None,
    *,
    drum_decomposition: DrumDecomposition | None = None,
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

    `drum_decomposition` and `bass_line` (Wave 4) follow the same one-source,
    one-computation, isolated-failure shape, but are gated on `source`
    rather than called for everyone -- see `_drum_decomposition_for_source`
    and `_bass_line_for_source` for the policy and the reasoning.
    `drum_decomposition` here is an *input*: the drums stem's own already-
    computed block, threaded through by `analyze_track` so a bass note's
    `step` lands on the same grid as a drum hit's. It has nothing to do with
    the `drum_decomposition` this function *returns* on `SourceAnalysis`
    except when `source == "drums"`, where they are the same object.

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
    drums_block = _drum_decomposition_for_source(source, mono, sample_rate, rhythm)
    bass_block = _bass_line_for_source(
        source, resolved_backend, mono, sample_rate, drum_decomposition
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
        drum_decomposition=drums_block,
        bass_line=bass_block,
        unavailable_features=_unavailable_features(
            rhythm, tonal, spectral, dynamics, drums_block, bass_block
        ),
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
    drums_block, bass_block = DrumDecomposition(), BassLine()
    unavailable = _unavailable_features(
        rhythm, tonal, spectral, dynamics, drums_block, bass_block
    )
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
        drum_decomposition=drums_block,
        bass_line=bass_block,
        labels=[],
        unavailable_features=unavailable,
    )


def _analyze_or_placeholder(
    path: Path,
    source: str,
    backend: AnalysisBackend,
    *,
    drum_decomposition: DrumDecomposition | None = None,
) -> SourceAnalysis:
    try:
        return analyze_source(path, source, backend, drum_decomposition=drum_decomposition)
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

    **How the drums grid reaches the bass source.** `STEM_NAMES` is
    `("drums", "bass", "vocals", "other")`, so this loop always analyzes
    drums before bass. The drums stem's own `DrumDecomposition` -- whatever
    its `status` -- is kept in `drums_decomposition` and handed to
    `analyze_source` when `stem_name == "bass"`, so `note_track.segment_notes`
    can quantise bass notes onto the same grid `DrumPattern.steps` uses (see
    `_bass_grid`). When there is no drums stem at all (skipped above, or
    absent from `stems` entirely), `drums_decomposition` stays `None` and
    every bass note's `step` comes back `None` -- no grid, not a guessed one.
    The same is true when the drums stem was analyzed but never reached
    `status="ok"` (`no_grid`/`too_few_hits`/`failed`): its `steps_per_cycle`
    is unset, so `_bass_grid` reads that the same way as "no drums stem".
    """
    resolved_backend = backend if backend is not None else get_backend()

    results: dict[str, SourceAnalysis] = {
        "mix": _analyze_or_placeholder(input_path, "mix", resolved_backend)
    }

    drums_decomposition: DrumDecomposition | None = None
    for stem_name in STEM_NAMES:
        stem_path = stems.get(stem_name)
        if stem_path is None or not Path(stem_path).is_file():
            logger.warning(
                "Skipping stem %r: no stem file found at %s (run `separate` first).",
                stem_name,
                stem_path,
            )
            continue
        analysis = _analyze_or_placeholder(
            stem_path,
            stem_name,
            resolved_backend,
            drum_decomposition=drums_decomposition if stem_name == "bass" else None,
        )
        results[stem_name] = analysis
        if stem_name == "drums":
            drums_decomposition = analysis.drum_decomposition

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
