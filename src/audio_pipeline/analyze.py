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

Schema v5 added the other thing orchestration is for: **there is one tempo and
one structure per record, and this module is where they are resolved.** In v4
every source refined its own grid from its own backend estimate, the five
disagreed (131.855 / 132.040 / 131.815 / 130.359 / 131.992 on the calibration
track), and whichever a downstream module happened to read it built a grid that
drifted apart from the audio over four minutes. `_resolve_grid` now runs
`tempo.refine_bpm` and `tempo.find_downbeat` exactly once, and the same period
and downbeat go into `drum_elements`, `note_track` and `arrangement`.

Two decisions live here rather than in those modules, both because this is the
only place that can see across them:

* **Which octave.** `tempo.py` measures the x0.5/x1/x2 candidates and refuses to
  choose, having proved that autocorrelation cannot — the doubled lag scores
  higher on nearly all material. This module owns the drum grid fitter, so it
  arbitrates by fitting each live candidate and comparing grid quality. See
  `_arbitrate_octave`.
* **Which sources are real.** A stem below `SILENCE_RMS_FLOOR` is separation
  residue, and everything derived from it is skipped rather than computed from a
  noise floor. See `_is_silent`.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, ItemsView, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel

from . import ANALYSIS_SAMPLE_RATE, STEM_NAMES, arrangement, drum_elements, note_track, tempo
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
    SILENCE_RMS_FLOOR,
    Arrangement,
    BassLine,
    DownbeatFit,
    DrumDecomposition,
    DynamicsFeatures,
    OctaveGridFit,
    RhythmFeatures,
    SourceAnalysis,
    SpectralFeatures,
    TempoFit,
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
    "resolve_track_grid",
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


# ---------------------------------------------------------------------------
# Schema v5: one tempo, one downbeat, one structure
# ---------------------------------------------------------------------------

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _as_model(model: type[_ModelT], value: Any) -> _ModelT:
    """A frozen dataclass from `tempo`/`arrangement` as its pydantic mirror.

    `schemas.py` declares those models field for field and name for name, so
    the conversion is `asdict()` into `model_validate()` and there is no
    hand-written mapping to fall out of step. Field-name drift would be silent
    here — pydantic would default the field it did not find and drop the one it
    did not know — so `tests/test_schemas_summary.py` asserts the two sets of
    names are equal rather than trusting this.
    """
    return model.model_validate(asdict(value))


def _mean_frame_rms(mono: AudioArray) -> float | None:
    """Mean of the per-frame RMS, the statistic `dynamics.rms_mean` reports.

    Deliberately the same quantity, computed through `arrangement.frame_rms`,
    so `_is_silent` means one thing whether it is applied to a backend's
    measurement or to audio this module has in hand before any backend has seen
    it. A global RMS would not be the same number.
    """
    if mono.size == 0:
        return None
    value = float(np.mean(arrangement.frame_rms(mono)))
    return value if math.isfinite(value) else None


def _is_silent(rms_mean: float | None) -> bool:
    """Is this source separation residue rather than a stem?

    `None` is **not** silent: an unmeasured level is unknown, and refusing to
    analyse a source because a backend failed to report its loudness would turn
    one missing descriptor into a missing analysis. See `SILENCE_RMS_FLOOR` for
    the number and the 27.7x gap it sits in.
    """
    return rms_mean is not None and math.isfinite(rms_mean) and rms_mean < SILENCE_RMS_FLOOR


def _usable_coarse_bpm(rhythm: RhythmFeatures) -> float | None:
    """The backend's tempo estimate, or `None` when it is not worth refining.

    Rejects a **negative** `bpm_confidence`. Essentia's confidence is its own
    unbounded scale, and on quiet stems it goes below zero — measured at -0.187,
    -0.043, -0.030 and -0.004 across four of the five corpus tracks, always on a
    stem with nothing in it. A negative reading is not a weak measurement, it is
    not a measurement.

    **Exactly 0.0 is passed through**, which is a deliberate departure from the
    "treat <= 0 as unusable" this package was briefed with. All four measured
    values behind that instruction are strictly negative, whereas 0.0 appears on
    all five sources of one example track at once — the signature of an
    extractor that did not report a confidence, not of five worthless estimates.
    Rejecting it would delete that track's tempo on no evidence.
    """
    bpm = rhythm.bpm
    if bpm is None or not math.isfinite(bpm) or bpm <= 0.0:
        return None
    confidence = rhythm.bpm_confidence
    if confidence is not None and math.isfinite(confidence) and confidence < 0.0:
        return None
    return float(bpm)


def _octave_grid_fit(
    mono: AudioArray, sample_rate: int, ratio: float, candidate_bpm: float
) -> OctaveGridFit:
    """Fit a drum grid at one candidate tempo and report how well it landed.

    Each candidate gets its own downbeat: a grid at half the tempo has bars
    twice as long and its phase is a different question, so reusing one
    downbeat across octaves would compare a fitted grid against a rotated one.
    """
    period = 60.0 / candidate_bpm
    downbeat = tempo.find_downbeat(mono, sample_rate, period)
    decomposition = drum_elements.decompose(
        mono,
        sample_rate,
        bpm=candidate_bpm,
        beat_times=(),
        beat_period_seconds=period,
        downbeat_seconds=downbeat.offset_seconds,
    )
    error_steps = decomposition.quantisation_error_steps
    error_seconds: float | None = None
    if error_steps is not None and decomposition.steps_per_cycle and decomposition.cycle_seconds:
        error_seconds = (
            error_steps * decomposition.cycle_seconds / decomposition.steps_per_cycle
        )
    return OctaveGridFit(
        ratio=ratio,
        bpm=candidate_bpm,
        grid_status=decomposition.status,
        quantisation_error_steps=error_steps,
        quantisation_error_seconds=error_seconds,
    )


def _beats_incumbent(candidate: OctaveGridFit, incumbent: OctaveGridFit) -> bool:
    """Does this octave fit the drums better than the coarse octave does?

    **Both error measures must agree, and that is the whole rule.** Neither is
    scale-free, and they are biased in opposite directions:

    * `quantisation_error_steps` shrinks when the tempo halves, because the step
      it is measured in doubles while the playing does not get tidier.
    * `quantisation_error_seconds` shrinks when the tempo doubles, because the
      steps get closer together and any hit is nearer to one.

    Measured on the corpus drums stems, error in steps then in milliseconds:

        madonna  x0.5 0.1430 / 32.50   x1 0.0333 / 3.79   x2 0.0730 / 4.15
        badu     x0.5 0.0689 / 15.29   x1 0.1115 / 12.36  x2 no_grid
        roni     x0.5 no_grid          x1 0.1218 / 21.49  x2 0.1046 / 9.23

    The swung hip-hop track is the case that settles it: at *half* its true
    tempo it scores 0.0689 steps against the truth's 0.1115, so "lowest
    quantisation error wins" halves a tempo that is already correct. It loses on
    milliseconds, and the drum-and-bass track — the one that genuinely needs
    doubling — wins on both. Requiring both is requiring a candidate to win
    despite the bias running against it, whichever way it is trying to move.

    A comparison, not a threshold. There is nothing here to tune.
    """
    if candidate.grid_status != "ok":
        return False
    if incumbent.grid_status != "ok":
        return True
    if (
        candidate.quantisation_error_steps is None
        or incumbent.quantisation_error_steps is None
        or candidate.quantisation_error_seconds is None
        or incumbent.quantisation_error_seconds is None
    ):
        return False
    return (
        candidate.quantisation_error_steps < incumbent.quantisation_error_steps
        and candidate.quantisation_error_seconds < incumbent.quantisation_error_seconds
    )


def _arbitrate_octave(
    fit: tempo.TempoFit, mono: AudioArray, sample_rate: int
) -> tuple[list[OctaveGridFit], OctaveGridFit | None]:
    """Choose between the live octave candidates on drum-grid quality.

    `tempo.py` surfaces the candidates and never moves `bpm`, because the
    statistic it has cannot separate them: the one corpus track that must double
    scores 1.21 on `r(x2)/r(x1)` while three that must not score 1.00, 1.12 and
    1.76 — the must-move value sits *inside* the must-stay range. Grid quality
    is a different measurement and it does separate them.

    Only attempted on a `refined` fit with more than one live candidate and the
    coarse octave among them. A refusal to refine is not an invitation to go
    octave-hunting: on the live-band corpus row the only live candidate is x2 at
    r = 0.159, which would move a 143 BPM track to 285.

    Returns `(rows, winner)`. `winner` is `None` when the coarse octave held,
    which is the answer on four of the five corpus tracks.
    """
    live = [candidate for candidate in fit.octave_candidates if candidate.status == "live"]
    if fit.status != "refined" or len(live) < 2:
        return [], None
    if not any(candidate.ratio == 1.0 for candidate in live):
        return [], None

    rows = [
        _octave_grid_fit(mono, sample_rate, candidate.ratio, candidate.bpm) for candidate in live
    ]
    incumbent = next(row for row in rows if row.ratio == 1.0)
    winner: OctaveGridFit | None = None
    for row in rows:
        if row.ratio == 1.0 or not _beats_incumbent(row, incumbent):
            continue
        # Both octaves beating the incumbent has never been observed and would
        # need each to win on the measure biased against it. Ordering by the
        # steps measure keeps the outcome deterministic if it ever happens.
        if winner is None or (row.quantisation_error_steps or 0.0) < (
            winner.quantisation_error_steps or 0.0
        ):
            winner = row
    (winner or incumbent).chosen = True
    return rows, winner


def _octave_caveats(rows: list[OctaveGridFit], winner: OctaveGridFit | None) -> list[str]:
    """Say in words what the arbitration did. It never happens silently."""
    if not rows:
        return []
    if winner is not None:
        return [
            f"tempo octave corrected x{winner.ratio:g} to {winner.bpm:.3f} BPM: the drum grid "
            f"fits better there than at the backend's octave on both quantisation-error "
            f"measures ({winner.quantisation_error_steps:.4f} steps and "
            f"{(winner.quantisation_error_seconds or 0.0) * 1000:.1f} ms). The backend's "
            f"estimate is unchanged in every source's rhythm.bpm"
        ]
    if not any(row.grid_status == "ok" for row in rows):
        return [
            "no octave of this estimate produced a usable drum grid, so the tempo was left "
            "where the backend put it; the octave may be wrong and nothing here can tell"
        ]
    return [
        "the octave was arbitrated against the drum grid and the backend's own octave won; "
        f"the alternatives are in octave_arbitration ({len(rows)} fitted)"
    ]


def resolve_track_grid(
    mono: AudioArray,
    sample_rate: int,
    coarse_bpm: float | None,
    *,
    arbitrate_octave: bool = False,
) -> tuple[TempoFit, DownbeatFit]:
    """The track's one tempo and one downbeat, from one source's audio.

    Args:
        mono: The source the grid is measured from — the drums stem when there
            is a real one, since `tempo.TEMPO_BAND_HZ` is 20-110 Hz and that is
            where a kick lives, otherwise the mix.
        sample_rate: `ANALYSIS_SAMPLE_RATE`.
        coarse_bpm: A backend estimate to refine, already filtered by
            `_usable_coarse_bpm`. `None` yields `status="unavailable"`.
        arbitrate_octave: Fit competing octaves with `drum_elements.decompose`.
            Only meaningful when `mono` is a drums stem — the fitter classifies
            kicks, snares and hats, and asking it to score a vocal is asking a
            question it cannot answer. Costs roughly one second per candidate.

    Returns:
        `(TempoFit, DownbeatFit)` as the schema's models. Never raises: both
        underlying functions absorb their own failures and report a status.
    """
    fit = tempo.refine_bpm(mono, sample_rate, coarse_bpm)
    rows: list[OctaveGridFit] = []
    winner: OctaveGridFit | None = None
    if arbitrate_octave:
        rows, winner = _arbitrate_octave(fit, mono, sample_rate)
        if winner is not None:
            # Re-refine at the chosen octave rather than keeping the candidate's
            # own BPM. A candidate is measured at one beat multiple only, where
            # the full fit cross-checks two; on the drum-and-bass row that is
            # 170.0693 against 170.0748, and F1 is the record of what six
            # thousandths of a BPM does to a grid over four minutes.
            fit = tempo.refine_bpm(mono, sample_rate, winner.bpm)

    model = _as_model(TempoFit, fit)
    model.octave_arbitration = rows
    model.caveats = [*model.caveats, *_octave_caveats(rows, winner)]
    downbeat = tempo.find_downbeat(mono, sample_rate, fit.period_seconds)
    return model, _as_model(DownbeatFit, downbeat)


def _arrangement_period(fit: TempoFit) -> float | None:
    """The bar length to fold an arrangement onto, or `None` to refuse one.

    `TempoFit.bpm` is populated even at `status="coarse"` -- it passes the
    backend's estimate through -- so it cannot be used directly here. Handing a
    bar length to `arrangement` when nothing measured one produces a plausible
    section list built on a number nobody stands behind: on the ambient corpus
    row that is **52 sections across a 17-minute record with no pulse**, which
    is precisely the "the tool correctly refused" outcome that row exists to
    test, inverted.

    A coarse tempo is still enough when the source is *periodic* -- the
    live-band row genuinely drifts, so its refinement is refused, and its
    arrangement (a solo-drum intro isolated at bars 0-3) is real and useful.
    What separates the two is not the status but whether any lag in the source
    correlates at all:

        eno    every octave candidate `ruled_out`, best r 0.0103  -> no grid
        levee  x2 `live` at r 0.1588                              -> grid, caveated

    `live` is `tempo.MIN_AUTOCORRELATION_R` (0.15), a threshold the corpus
    already validated end to end (0.804 / 0.735 / 0.476 against Levee's 0.108).
    This reuses it rather than introducing a second one.
    """
    if fit.period_seconds is None:
        return None
    if fit.status == "refined":
        return fit.period_seconds
    if any(candidate.status == "live" for candidate in fit.octave_candidates):
        return fit.period_seconds
    return None


def _grid_confidence(fit: TempoFit) -> str:
    """One label for `arrangement`, which wants to know whether bars are exact.

    `arrangement` treats `low`, `coarse`, `unavailable` and `unknown` as a weak
    grid and adds an approximate-boundaries caveat. A refused refinement is the
    stronger warning of the two, so it wins over the confidence band.
    """
    return fit.status if fit.status != "refined" else fit.confidence_label


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
    grid: TempoFit | None = None,
    downbeat: DownbeatFit | None = None,
    *,
    silent: bool = False,
) -> DrumDecomposition:
    """`drum_elements.decompose()`, called only for a non-silent `drums` source.

    A policy of `analyze.py`, not of `drum_elements`: every other source keeps
    the default `status="not_attempted"`.

    **The grid comes from the track, not from this stem.** `grid.period_seconds`
    is the one refined period `_resolve_grid` measured and `downbeat.
    offset_seconds` is where bar one actually starts. Both take precedence over
    `rhythm.bpm`/`rhythm.beat_times`, which stay as the fallback for a
    `decompose` called outside a full run. `beat_times[0]` in particular is
    **not** a downbeat: on the calibration track it sits at 0.348 s and lands the
    kick on steps 3/7/11/15, a rotation that looks entirely plausible in the
    output.

    `decompose()` already never raises (it wraps its own body and returns
    `status="failed"` on any internal error -- see its docstring). This is
    defence in depth in the same spirit as `_call_backend_method`: a future
    change to that guarantee must not be able to take the rest of this
    source's analysis down with it.
    """
    if source != "drums":
        return DrumDecomposition()
    if silent:
        return DrumDecomposition(status="silent", caveats=[_SILENT_CAVEAT])
    try:
        return drum_elements.decompose(
            mono,
            sample_rate,
            bpm=rhythm.bpm,
            beat_times=rhythm.beat_times,
            beat_period_seconds=grid.period_seconds if grid is not None else None,
            downbeat_seconds=downbeat.offset_seconds if downbeat is not None else None,
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
    grid: TempoFit | None = None,
    downbeat: DownbeatFit | None = None,
    *,
    silent: bool = False,
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
    define -- see `_bass_grid`. It is preferred over the raw `grid`/`downbeat`
    when it settled on a usable grid, because `decompose` refines the supplied
    downbeat's sub-step phase against the hits it actually found: on the
    calibration track the raw downbeat sits 0.267 steps from where the hits are,
    and quantising bass notes to the unrefined anchor would put the two
    instruments on step numbers that agree with each other and with nothing
    audible. `grid`/`downbeat` are the fallback for a record with no drums stem,
    or one whose drums never produced a grid.
    """
    if source != "bass":
        return BassLine()
    if silent:
        return BassLine(status="silent", caveats=[_SILENT_CAVEAT])

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
    if step_seconds is None:
        return note_track.segment_notes(
            pitch_track,
            steps_per_cycle=note_track.DEFAULT_STEPS_PER_CYCLE,
            beat_period_seconds=grid.period_seconds if grid is not None else None,
            downbeat_seconds=downbeat.offset_seconds if downbeat is not None else None,
        )
    return note_track.segment_notes(
        pitch_track,
        grid_anchor_seconds=grid_anchor_seconds,
        step_seconds=step_seconds,
        steps_per_cycle=steps_per_cycle,
    )


#: Attached to every block skipped because its source is below
#: `SILENCE_RMS_FLOOR`, so the reason travels with the result rather than
#: living only in `unavailable_features`.
_SILENT_CAVEAT = (
    "source is below the silence floor and was not analysed: at this level a "
    "stem is separation residue, and a pitch tracker or onset detector pointed "
    "at a noise floor returns confident nonsense"
)


def analyze_source(
    path: Path,
    source: str,
    backend: AnalysisBackend | None = None,
    *,
    drum_decomposition: DrumDecomposition | None = None,
    grid: TempoFit | None = None,
    downbeat: DownbeatFit | None = None,
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
    # The four descriptor blocks are computed for every source including a
    # silent one -- they are cheap, they are what *shows* the source is silent,
    # and `dynamics.rms_mean` is the measurement the gate reads. Only the
    # derived blocks, which interpret rather than measure, are skipped.
    silent = _is_silent(dynamics.rms_mean)
    drums_block = _drum_decomposition_for_source(
        source, mono, sample_rate, rhythm, grid, downbeat, silent=silent
    )
    bass_block = _bass_line_for_source(
        source,
        resolved_backend,
        mono,
        sample_rate,
        drum_decomposition,
        grid,
        downbeat,
        silent=silent,
    )

    unavailable = _unavailable_features(
        rhythm, tonal, spectral, dynamics, drums_block, bass_block
    )
    if silent:
        unavailable.append(
            f"source (silent: rms_mean {dynamics.rms_mean:.2g} is below the "
            f"{SILENCE_RMS_FLOOR:g} silence floor)"
        )
        unavailable.sort()

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
        unavailable_features=unavailable,
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
    grid: TempoFit | None = None,
    downbeat: DownbeatFit | None = None,
) -> SourceAnalysis:
    try:
        return analyze_source(
            path,
            source,
            backend,
            drum_decomposition=drum_decomposition,
            grid=grid,
            downbeat=downbeat,
        )
    except Exception as exc:  # noqa: BLE001 - one source's failure must not lose the others
        logger.error(
            "Analysis failed entirely for source %r (%s: %s); continuing with the "
            "remaining sources.",
            source,
            type(exc).__name__,
            exc,
        )
        return _placeholder_analysis(path, source, backend.name, exc)


@dataclass(frozen=True)
class TrackAnalysis:
    """What one run measured: the five sources, plus the three track-level blocks.

    `analyze_track` returned a bare `dict[str, SourceAnalysis]` through v4,
    when every source carried its own tempo. v5 resolves one tempo, one
    downbeat and one structure for the whole record, and they belong to the
    track rather than to any source, so they need somewhere to travel. Mapping
    behaviour is preserved (`result["mix"]`, `for name in result`) so callers
    that only want the sources read exactly as they did.
    """

    sources: dict[str, SourceAnalysis]
    tempo: TempoFit
    downbeat: DownbeatFit
    arrangement: Arrangement

    def __getitem__(self, key: str) -> SourceAnalysis:
        return self.sources[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def items(self) -> ItemsView[str, SourceAnalysis]:
        return self.sources.items()

    def get(self, key: str, default: SourceAnalysis | None = None) -> SourceAnalysis | None:
        return self.sources.get(key, default)


def _load_mono_stems(stems: Mapping[str, Path]) -> tuple[dict[str, AudioArray], int]:
    """Every stem that exists on disk, as mono audio, plus the sample rate.

    Loaded once and used twice -- to measure the track's tempo from the drums
    stem, and to fold every stem's per-bar energy into an arrangement -- then
    dropped before the per-source loop starts, so the two passes never hold
    the audio at the same time. A stem that fails to decode is skipped with a
    warning: an unreadable vocals stem must not cost the record its tempo.
    """
    loaded: dict[str, AudioArray] = {}
    sample_rate = ANALYSIS_SAMPLE_RATE
    for name in STEM_NAMES:
        path = stems.get(name)
        if path is None or not Path(path).is_file():
            continue
        try:
            stereo, sample_rate = load_audio(path, mono=False)
            loaded[name] = to_mono(stereo)
        except Exception as exc:  # noqa: BLE001 - one bad stem must not lose the grid
            logger.warning(
                "Could not load stem %r for the track grid (%s: %s); continuing without it.",
                name,
                type(exc).__name__,
                exc,
            )
    return loaded, sample_rate


def _grid_source(
    stem_audio: Mapping[str, AudioArray], input_path: Path
) -> tuple[AudioArray | None, int, bool]:
    """The audio the track's tempo is measured from: `(mono, sample_rate, is_drums)`.

    The drums stem when there is a real one. `tempo.TEMPO_BAND_HZ` is 20-110 Hz,
    chosen so a kick fundamental sits inside it and a bass fundamental does not,
    and the octave arbitration needs a source the drum grid fitter can score.

    Falls back to the mix when the drums stem is missing or is below
    `SILENCE_RMS_FLOOR` -- not hypothetical: the ambient corpus row's drums stem
    is separation residue at 4.8e-05 RMS, and refining a tempo from it would be
    refining a tempo from nothing.
    """
    drums = stem_audio.get("drums")
    if drums is not None and not _is_silent(_mean_frame_rms(drums)):
        return drums, ANALYSIS_SAMPLE_RATE, True
    try:
        stereo, sample_rate = load_audio(input_path, mono=False)
    except Exception as exc:  # noqa: BLE001 - no grid is better than no analysis
        logger.warning(
            "Could not load %s to measure the track tempo (%s: %s).",
            input_path,
            type(exc).__name__,
            exc,
        )
        return None, ANALYSIS_SAMPLE_RATE, False
    return to_mono(stereo), sample_rate, False


def analyze_track(
    input_path: Path,
    stems: dict[str, Path],
    backend: AnalysisBackend | None = None,
) -> TrackAnalysis:
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

    **The order of work changed in v5.** The mix is analyzed first, for a coarse
    tempo estimate to refine. Every stem is then loaded once to measure the
    track's one tempo and downbeat (`_grid_source`, `resolve_track_grid`) and to
    fold the arrangement, and that audio is released before the per-source loop
    begins, so the grid pass and the analysis pass never hold stems at the same
    time. Only then are the stems analyzed, each handed the same period and the
    same downbeat.
    """
    resolved_backend = backend if backend is not None else get_backend()

    results: dict[str, SourceAnalysis] = {
        "mix": _analyze_or_placeholder(input_path, "mix", resolved_backend)
    }

    stem_audio, sample_rate = _load_mono_stems(stems)
    grid_audio, grid_sample_rate, grid_is_drums = _grid_source(stem_audio, input_path)
    coarse_bpm = _usable_coarse_bpm(results["mix"].rhythm)
    if grid_audio is None:
        track_tempo, track_downbeat = TempoFit(), DownbeatFit()
    else:
        track_tempo, track_downbeat = resolve_track_grid(
            grid_audio,
            grid_sample_rate,
            coarse_bpm,
            arbitrate_octave=grid_is_drums,
        )
    if not grid_is_drums and track_tempo.status != "unavailable":
        track_tempo.caveats.append(
            "tempo measured from the mix, not from a drums stem: the 20-110 Hz band "
            "the fit reads carries bass and other low material here, and no octave "
            "arbitration was possible without a drum grid to score"
        )

    # `arrangement.arrangement` derives the kick track from the drums stem
    # itself. Its `arrangement_from_frames` sibling takes the kick as a
    # keyword-only argument and silently ignores unknown keys in `levels`, so
    # passing the kick as a mapping entry there yields a plausible arrangement
    # with no kick track and therefore no `breakdown` or `drop` anywhere. This
    # entry point cannot be called that way, which is why it is the one used.
    track_arrangement = _as_model(
        Arrangement,
        arrangement.arrangement(
            stem_audio,
            sample_rate,
            _arrangement_period(track_tempo),
            track_downbeat.offset_seconds,
            grid_confidence=_grid_confidence(track_tempo),
        ),
    )
    stem_audio.clear()

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
            grid=track_tempo,
            downbeat=track_downbeat,
        )
        results[stem_name] = analysis
        if stem_name == "drums":
            drums_decomposition = analysis.drum_decomposition

    return TrackAnalysis(
        sources=results,
        tempo=track_tempo,
        downbeat=track_downbeat,
        arrangement=track_arrangement,
    )


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
