"""Condense a full analysis into a compact, hand-readable hints file.

v1 emits descriptive hints only. It does NOT generate Strudel code — the point
is to give you something to read while writing patterns yourself.

Everything here is a pure function over `TrackSummary`: no audio, no I/O, no
backend imports. Every descriptor in the summary may be `None` and any stem may
be absent from `summary.sources`, so every path degrades to `None` plus a prose
note rather than raising.

Subdivision inference reads the drums stem's `rhythm.onset_times` — what was
actually played — and measures it against the beat period implied by `bpm` —
what it was played against. The ratio between the two is what names a
subdivision; neither series can do it alone. `beat_times` is deliberately not
used: it is evenly spaced by construction, so it cannot carry swing.

The bias throughout is towards silence over a guess. A wrong grid is worse than
no grid: it sends you down the wrong path rebuilding the track by hand, and by
the time you notice, you have written a bar of patterns against the wrong feel.
Whenever a field comes back `None`, `StrudelHints.notes` says why.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean, median

from .heuristics import THRESHOLDS
from .schemas import SourceAnalysis, StrudelHints, TrackSummary

# One cycle = one bar of 4/4. Strudel's cycle is the unit you write patterns
# against, and near enough all the material this tool is pointed at is in 4/4,
# so a bar-per-cycle mapping keeps `"bd sd"`-shaped thinking aligned with the
# suggested cycle length. Overridable per call for 3/4 or 7/8 material.
BEATS_PER_CYCLE = 4

# --- Subdivision inference tuning -------------------------------------------
#
# These govern `infer_subdivision_feel()`. They are deliberately strict: the
# function is allowed to say nothing, so the cost of a tight threshold is a
# missing hint, while the cost of a loose one is a confidently wrong grid.
#
# They are tuned for *observed onsets*, not for an idealised grid. Real onset
# data differs from a synthetic click train in three ways that matter here:
# events carry performance jitter, quiet hits go undetected (leaving a gap of
# two or three grid units where one was played), and bleed or flams add events
# that were never a grid position at all.

# Minimum usable inter-onset intervals. Below this, a "distribution" is a
# handful of numbers and any modality you find in it is noise. Raised from 8 for
# real onset input: a drums stem yields onsets in the hundreds, so demanding 16
# costs nothing on real material and keeps the cluster and alternation
# statistics from being decided by a couple of events.
MIN_IOIS = 16

# Intervals further than this factor either side of the median are dropped
# before analysis: they are section gaps, dropouts, or double-triggered flams,
# not grid events. Filtering happens on the sequence, so a dropped gap joins its
# neighbours — that can only weaken the alternation evidence swing needs, never
# manufacture it.
OUTLIER_IOI_FACTOR = 4.0

# --- "straight" criterion ---
# An interval counts as on-grid when it is within this relative distance of a
# whole-number multiple of the base unit. Raised from 0.15, which was tuned
# against grid-perfect click trains: at 120 BPM a sixteenth is 125 ms, so 0.15
# allowed only +/-19 ms — barely above a typical onset detector's own hop
# resolution (~12 ms at hop 512 / 44.1 kHz) before any human timing is counted.
# 0.22 gives +/-27 ms, which covers detector resolution plus ordinary
# performance jitter. Loosening this is only safe because the anti-lilt veto
# below, not the tolerance, is what keeps swung material out of "straight".
STRAIGHT_TOLERANCE = 0.22
# ...and the grid is called straight only if at least this fraction of intervals
# land on it. Relaxed from 0.85 now that whole-number multiples count as on-grid
# (see MAX_GRID_MULTIPLE): missed hits no longer register as evidence against
# the grid, so what remains to tolerate is spurious onsets from bleed and flams.
STRAIGHT_MIN_FRACTION = 0.80
# An undetected hit turns two grid units into one interval, so an interval that
# is a whole-number multiple of the base unit is evidence *for* the grid, not
# against it. Multiples are matched up to this bound; beyond it the interval is
# a rest rather than a subdivision, and OUTLIER_IOI_FACTOR has usually dropped
# it already.
MAX_GRID_MULTIPLE = 4
# The base unit must genuinely be the pulse rather than an artefact of a median
# sitting between two modes, so this fraction of intervals must be a *single*
# unit rather than a multiple of one.
STRAIGHT_MIN_BASE_FRACTION = 0.5
# Veto: a light lilt (say 1.25:1) sits inside STRAIGHT_TOLERANCE either side of
# the median and would otherwise be reported as dead straight. A *systematic*
# long-short alternation is positive evidence against evenness whatever its
# amplitude, so any alternating pair above this ratio disqualifies "straight".
# Between this and SWING_RATIO_RANGE[0] the function deliberately says nothing:
# the material is neither straight nor clearly swung, and guessing either way
# would misplace every offbeat.
STRAIGHT_MAX_LILT_RATIO = 1.1

# --- "swung" criterion ---
# Long/short ratio band. A dead-straight pair is 1.0, triplet swing is 2.0,
# hard-quantised dotted swing is 3.0. This band brackets the usual musical
# range of roughly 60-70% swing while excluding near-straight material.
SWING_RATIO_RANGE = (1.5, 2.5)
# Each of the two clusters must be internally tight, measured as median absolute
# deviation over the median. This replaces a (max - min) / mean range statistic,
# which was fine for synthetic input but is the wrong tool for observed onsets:
# a range is decided entirely by its two most extreme members, so it grows with
# sample size and a single mis-detected onset in a hundred could veto an
# otherwise clean swing. A relative MAD ignores the tails and stays put.
SWING_CLUSTER_MAD_MAX = 0.12
# Each cluster must hold at least this share of the intervals, so that one
# outlying long note cannot pose as half a bimodal distribution.
SWING_MIN_CLUSTER_FRACTION = 0.35
# Swing is a long-short *alternation*, not merely two clusters. This is the
# fraction of adjacent interval pairs that must cross the cluster split. It is
# the criterion that separates real swing from an arbitrary bimodal mess.
# Relaxed from 0.85: a drums stem's onsets are the union of kick, snare and hat,
# so a stray extra hit breaks the alternation locally without changing the feel.
# Of every constant here this is the one most likely to want W3C calibration —
# it is the one that decides whether real swung material is found at all.
SWING_MIN_ALTERNATION = 0.75
# Cluster splitting uses this percentile in from each end rather than the raw
# min and max, so one stray short or long interval cannot drag the split point.
SPLIT_PERCENTILE = 0.10

# How close an inferred unit must sit to an exact subdivision of the beat before
# it is named. Relative, so it scales with the subdivision.
GRID_MATCH_TOLERANCE = 0.15

# Named straight subdivisions, in beats. Deliberately excludes 1.0: intervals
# one beat long mean the event series resolved only to the beat, which says
# nothing at all about subdivision and must not be reported as if it did.
STRAIGHT_UNITS_IN_BEATS: tuple[tuple[float, str], ...] = (
    (0.5, "straight 8ths"),
    (0.25, "straight 16ths"),
    (0.125, "straight 32nds"),
)

# Named swung feels, keyed by the duration of one long+short *pair* in beats.
SWUNG_PAIRS_IN_BEATS: tuple[tuple[float, str], ...] = (
    (1.0, "swung 8ths"),
    (0.5, "swung 16ths"),
)

# Minimum key confidence before a tonal centre is worth printing. Deliberately
# local rather than taken from `heuristics.THRESHOLDS`: that module's
# `tonally_stable_confidence` answers a different question (is this source
# harmonically *stable*), and naming a centre to start from is a lower bar than
# asserting stability. Density thresholds, which do answer the same question in
# both modules, are imported rather than restated.
TONAL_CENTRE_MIN_CONFIDENCE = 0.5

# Every string these functions can emit into a `StrudelHints` field. v1 is
# descriptive only, so the vocabulary is closed and contains no pattern syntax.
DENSITY_TERMS: frozenset[str] = frozenset({"sparse", "moderate", "busy"})
SUBDIVISION_TERMS: frozenset[str] = frozenset(
    label for _, label in (*STRAIGHT_UNITS_IN_BEATS, *SWUNG_PAIRS_IN_BEATS)
)


# --- small helpers ----------------------------------------------------------


def _source(summary: TrackSummary, name: str) -> SourceAnalysis | None:
    """Return one analysed source, or None when that stem was never analysed."""
    return summary.sources.get(name)


def _positive(value: float | None) -> float | None:
    """Return `value` when it is a usable positive number, else None."""
    if value is None:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def _relative_mad(values: Sequence[float]) -> float:
    """Median absolute deviation over the median.

    0.0 for identical values, ~0.04 for +/-8% jitter, and — unlike a range-based
    spread — unmoved by one stray member, which is what observed onset data
    supplies in quantity.
    """
    centre = median(values)
    if centre <= 0.0:
        return math.inf
    return median([abs(v - centre) for v in values]) / centre


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted sequence."""
    last = len(values) - 1
    return values[min(last, max(0, round(fraction * last)))]


def _split_point(iois: Sequence[float]) -> float:
    """Threshold separating short intervals from long ones.

    The midpoint between the 10th and 90th percentile rather than between the
    min and max: on observed onsets the extremes are exactly the values least
    worth trusting, and a split dragged by one of them mis-sorts every interval
    behind it.
    """
    ordered = sorted(iois)
    low = _percentile(ordered, SPLIT_PERCENTILE)
    high = _percentile(ordered, 1.0 - SPLIT_PERCENTILE)
    return (low + high) / 2.0


def _clean_iois(times: Sequence[float]) -> list[float]:
    """Inter-onset intervals from an event-time series, outliers removed.

    Times are sorted defensively; non-positive intervals (duplicate timestamps)
    are dropped, then anything beyond `OUTLIER_IOI_FACTOR` either side of the
    median goes too.
    """
    ordered = sorted(float(t) for t in times)
    raw = [b - a for a, b in zip(ordered, ordered[1:], strict=False) if b > a]
    if not raw:
        return []
    mid = median(raw)
    if mid <= 0.0:
        return []
    lo, hi = mid / OUTLIER_IOI_FACTOR, mid * OUTLIER_IOI_FACTOR
    return [d for d in raw if lo <= d <= hi]


def _match_grid(value_in_beats: float, table: Sequence[tuple[float, str]]) -> str | None:
    """Name a duration expressed in beats, or None if it matches nothing."""
    best: str | None = None
    best_error = math.inf
    for beats, label in table:
        error = abs(value_in_beats - beats) / beats
        if error < best_error:
            best_error, best = error, label
    if best is None or best_error > GRID_MATCH_TOLERANCE:
        return None
    return best


def _alternating_pair(iois: Sequence[float]) -> tuple[list[float], list[float]] | None:
    """Split intervals into short/long clusters, if they systematically alternate.

    The split comes from `_split_point`, not the median: a median split forces
    two equal halves onto even unimodal data, and with an odd number of
    intervals the median *is* one cluster's value, which empties the other.
    Returns `(short, long)` only when both clusters are populated
    (`SWING_MIN_CLUSTER_FRACTION`) and adjacent intervals cross the split at
    least `SWING_MIN_ALTERNATION` of the time — that is, only when a long-short
    alternation genuinely runs through the sequence rather than two clusters
    merely existing somewhere in it.
    """
    split = _split_point(iois)
    is_long = [d >= split for d in iois]
    long = [d for d, flag in zip(iois, is_long, strict=True) if flag]
    short = [d for d, flag in zip(iois, is_long, strict=True) if not flag]

    if min(len(long), len(short)) / len(iois) < SWING_MIN_CLUSTER_FRACTION:
        return None
    crossings = sum(1 for a, b in zip(is_long, is_long[1:], strict=False) if a != b)
    if crossings / (len(is_long) - 1) < SWING_MIN_ALTERNATION:
        return None
    return short, long


def _straight_label(iois: Sequence[float], beat_seconds: float) -> tuple[str | None, str | None]:
    """Test the straight hypothesis.

    Intervals are matched against whole-number multiples of the base unit, so a
    hit the onset detector missed reads as one interval of two units rather than
    as evidence against the grid. The base unit must still account for
    `STRAIGHT_MIN_BASE_FRACTION` of the intervals on its own.

    Returns `(label, reason)`. `reason` is only set when the distribution *was*
    straight-shaped but could not be named — that is a distinct outcome from
    "not straight", which returns `(None, None)` so the swing test can run.
    """
    unit = median(iois)
    if unit <= 0.0:
        return None, None

    multiples = [min(MAX_GRID_MULTIPLE, max(1, round(d / unit))) for d in iois]
    on_grid = sum(
        1
        for d, k in zip(iois, multiples, strict=True)
        if abs(d - k * unit) <= STRAIGHT_TOLERANCE * unit
    )
    if on_grid / len(iois) < STRAIGHT_MIN_FRACTION:
        return None, None
    if multiples.count(1) / len(iois) < STRAIGHT_MIN_BASE_FRACTION:
        return None, None

    # Anti-lilt veto: tolerance alone would swallow a light swing whole.
    pair = _alternating_pair(iois)
    if pair is not None and fmean(pair[1]) / fmean(pair[0]) > STRAIGHT_MAX_LILT_RATIO:
        return None, None

    label = _match_grid(unit / beat_seconds, STRAIGHT_UNITS_IN_BEATS)
    if label is None:
        return None, (
            "drum grid is even but does not resolve to a subdivision of the beat "
            "(it may sit on the beat itself, or the tempo may be wrong), "
            "subdivision not inferred"
        )
    return label, None


def _swung_label(iois: Sequence[float], beat_seconds: float) -> tuple[str | None, str | None]:
    """Test the swung hypothesis.

    Requires all four of: two populated clusters that genuinely alternate (via
    `_alternating_pair`), each cluster internally tight, a long/short ratio
    inside `SWING_RATIO_RANGE`, and a long+short pair that lines up with a
    subdivision of the beat. Returns `(label, reason)` with the same convention
    as `_straight_label`.
    """
    pair = _alternating_pair(iois)
    if pair is None:
        return None, None
    short, long = pair

    if _relative_mad(long) > SWING_CLUSTER_MAD_MAX:
        return None, None
    if _relative_mad(short) > SWING_CLUSTER_MAD_MAX:
        return None, None

    long_mean, short_mean = fmean(long), fmean(short)
    ratio = long_mean / short_mean
    if not SWING_RATIO_RANGE[0] <= ratio <= SWING_RATIO_RANGE[1]:
        return None, None

    label = _match_grid((long_mean + short_mean) / beat_seconds, SWUNG_PAIRS_IN_BEATS)
    if label is None:
        return None, (
            "drum grid looks swung but the long-short pair does not line up with "
            "the tempo, subdivision not inferred"
        )
    return label, None


def _grid_bpm(summary: TrackSummary) -> float | None:
    """Tempo to measure the drum grid against: drums stem first, then the mix."""
    drums = _source(summary, "drums")
    if drums is not None:
        bpm = _positive(drums.rhythm.bpm)
        if bpm is not None:
            return bpm
    mix = _source(summary, "mix")
    if mix is not None:
        return _positive(mix.rhythm.bpm)
    return None


def _subdivision_feel(summary: TrackSummary) -> tuple[str | None, str | None]:
    """`infer_subdivision_feel()` plus the note explaining any None.

    Confidence criterion, stated in full. The label is returned only when:

    1. a drums stem is present and carries observed onsets
       (`rhythm.onset_times`);
    2. at least `MIN_IOIS` intervals survive outlier filtering;
    3. a positive tempo is available to measure the intervals against;
    4. **exactly one** of the straight and swung hypotheses passes its own test
       (see `_straight_label` / `_swung_label`); and
    5. the resulting unit lands within `GRID_MATCH_TOLERANCE` of a named
       subdivision of the beat.

    The two hypotheses do not cover the whole space, by design. A systematic
    long-short lilt between `STRAIGHT_MAX_LILT_RATIO` and `SWING_RATIO_RANGE[0]`
    fails both and reports nothing, as does an even grid that resolves only to
    the beat: neither tells you where to put an offbeat.

    Failing any of those returns `(None, reason)`. Two hypotheses passing at
    once is treated as contradictory evidence and also returns None — it should
    be unreachable, since alternation and unimodality exclude each other, but a
    tuning change should surface as silence rather than as a coin toss.

    Note on the data this needs: the event series is the drums stem's
    `rhythm.onset_times`, and the reference is the beat period from `bpm`.
    `TrackSummary.summary_payload()` strips both `onset_times` and `beat_times`,
    so a summary reloaded from `track_summary.json` has no timing to work from
    and this returns None with a note. Call it on the in-memory summary produced
    by the same run as the analysis.
    """
    drums = _source(summary, "drums")
    if drums is None:
        return None, "no drums stem, subdivision not inferred"

    times = drums.rhythm.onset_times
    if not times and drums.rhythm.beat_times:
        return None, (
            "drums stem has beat times but no onset times, subdivision not inferred — "
            "the beat grid is evenly spaced by construction and cannot show swing"
        )
    if len(times) <= MIN_IOIS:
        return None, (
            "too few drum onsets to judge the grid, subdivision not inferred "
            "(a summary reloaded from track_summary.json has no onset times: "
            "summary_payload strips them to an onset_count)"
        )

    iois = _clean_iois(times)
    if len(iois) < MIN_IOIS:
        return None, (
            "drum onset timing is too irregular to judge the grid, subdivision not inferred"
        )

    bpm = _grid_bpm(summary)
    if bpm is None:
        return None, "tempo unknown, subdivision not inferred"
    beat_seconds = 60.0 / bpm

    straight, straight_reason = _straight_label(iois, beat_seconds)
    swung, swung_reason = _swung_label(iois, beat_seconds)

    if straight is not None and swung is not None:
        return None, "grid evidence is contradictory, subdivision not inferred"
    if straight is not None:
        return straight, None
    if swung is not None:
        return swung, None
    reason = straight_reason or swung_reason
    if reason is not None:
        return None, reason
    return None, (
        "drum inter-onset intervals fit neither an even nor a swung grid, "
        "subdivision not inferred — verify by ear"
    )


def _format_tonal_centre(analysis: SourceAnalysis) -> str | None:
    """'A minor' / 'A' from a source's tonal features, if confident enough."""
    tonal = analysis.tonal
    if not tonal.key:
        return None
    confidence = tonal.key_confidence
    if confidence is None or not math.isfinite(confidence):
        return None
    if confidence < TONAL_CENTRE_MIN_CONFIDENCE:
        return None
    return f"{tonal.key} {tonal.scale}".strip() if tonal.scale else tonal.key


def _tonal_centre(summary: TrackSummary) -> tuple[str | None, str | None]:
    """Mix key when confident, else the bass stem's, else None plus a caveat."""
    mix = _source(summary, "mix")
    if mix is not None:
        centre = _format_tonal_centre(mix)
        if centre is not None:
            return centre, None

    bass = _source(summary, "bass")
    if bass is not None:
        centre = _format_tonal_centre(bass)
        if centre is not None:
            return centre, (
                "tonal centre taken from the bass stem; mix key confidence low or unavailable"
            )

    return None, "key confidence low, tonal centre not reported — verify by ear"


def _density_with_note(
    summary: TrackSummary, stem: str, field_label: str
) -> tuple[str | None, str | None]:
    """Classify one stem's onset density, with a note when it cannot be done."""
    analysis = _source(summary, stem)
    if analysis is None:
        return None, f"no {stem} stem, {field_label} not reported"
    density = classify_density(analysis.rhythm.onset_density)
    if density is None:
        return None, f"{stem} onset density unavailable, {field_label} not reported"
    return density, None


# --- public API -------------------------------------------------------------


def infer_subdivision_feel(summary: TrackSummary) -> str | None:
    """Guess the rhythmic grid, e.g. 'straight 16ths' or 'swung 8ths'.

    Derived from the inter-onset interval distribution of the drums stem
    relative to the beat grid. Returns None whenever confidence is low — a wrong
    guess is worse than no guess. See `_subdivision_feel` for the exact
    criterion and for which fields of the summary are required.
    """
    label, _ = _subdivision_feel(summary)
    return label


def classify_density(onset_density: float | None) -> str | None:
    """Map onsets/sec to 'sparse' | 'moderate' | 'busy'.

    Bands are half-open, lower bound inclusive, using the thresholds owned by
    `heuristics.THRESHOLDS`:

        [0, sparse)        -> 'sparse'
        [sparse, busy)     -> 'moderate'
        [busy, inf)        -> 'busy'

    So a value sitting exactly on the sparse threshold reads 'moderate', and one
    exactly on the busy threshold reads 'busy'. None, non-finite, and negative
    inputs are unmeasurable rather than quiet, so they return None.
    """
    if onset_density is None:
        return None
    if not math.isfinite(onset_density) or onset_density < 0.0:
        return None
    if onset_density < THRESHOLDS["sparse_onsets_per_sec"]:
        return "sparse"
    if onset_density < THRESHOLDS["busy_drums_onsets_per_sec"]:
        return "moderate"
    return "busy"


def suggest_cycle_seconds(
    bpm: float | None, beats_per_cycle: int = BEATS_PER_CYCLE
) -> float | None:
    """Strudel cycle length in seconds for the detected tempo.

    `beats_per_cycle * 60 / bpm`, so 120 BPM at the default 4 beats per cycle is
    a 2.0 s cycle. Returns None for an unknown, non-finite, zero, or negative
    tempo, and for a non-positive `beats_per_cycle` — never divides by zero.
    """
    tempo = _positive(bpm)
    if tempo is None or beats_per_cycle <= 0:
        return None
    return round(beats_per_cycle * 60.0 / tempo, 6)


def build(summary: TrackSummary, beats_per_cycle: int = BEATS_PER_CYCLE) -> StrudelHints:
    """Assemble the hints object from a completed track summary.

    Every field degrades to None with an explanatory note rather than raising,
    including for a summary whose `sources` dict is empty.
    """
    notes: list[str] = []

    def note(text: str | None) -> None:
        if text and text not in notes:
            notes.append(text)

    if not summary.sources:
        note("summary contains no analysed sources; nothing could be inferred")

    mix = _source(summary, "mix")
    bpm = _positive(mix.rhythm.bpm) if mix is not None else None
    if bpm is None:
        bpm = _grid_bpm(summary)
        if bpm is not None:
            note("tempo taken from the drums stem; mix tempo unavailable")
    if bpm is None:
        note("tempo unknown, cycle length not suggested")

    feel, feel_note = _subdivision_feel(summary)
    note(feel_note)

    drum_density, drum_note = _density_with_note(summary, "drums", "drum density")
    note(drum_note)

    bass_activity, bass_note = _density_with_note(summary, "bass", "bass activity")
    note(bass_note)

    tonal_centre, tonal_note = _tonal_centre(summary)
    note(tonal_note)

    return StrudelHints(
        track_name=summary.track_name,
        bpm=bpm,
        suggested_cycle_seconds=suggest_cycle_seconds(bpm, beats_per_cycle),
        subdivision_feel=feel,
        drum_density=drum_density,
        bass_activity=bass_activity,
        tonal_centre=tonal_centre,
        notes=notes,
    )
