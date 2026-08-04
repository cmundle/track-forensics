"""Turn a raw framewise F0 track into a note sequence. Pure numpy.

This module is one half of a deliberate seam:

* A backend's `pitch()` returns a `PitchTrack` — per-frame F0 in Hz plus a
  voicing decision. That is *physics*, and running YIN is an algorithm this
  project should not own, so it lives behind the `AnalysisBackend` Protocol.
* :func:`segment_notes` turns that F0 into notes. That is *musical
  interpretation* — where a note starts, whether a 30 ms wobble is a new note
  or vibrato, which octave a half-ambiguous reading belongs in — and it is
  shared, so both backends produce **byte-identical** `BassLine`s from
  identical F0. The only thing left that can diverge across backends is the F0
  estimate itself, which unlike `tonal_stability` has a right answer.

**No librosa, no essentia, at any level.** Not at module top level, not inside
a function body. `tests/test_note_track.py` pins this by importing the module in
a subprocess with both libraries blocked. numpy only, so this half of the seam
cannot acquire a dependency that would make it un-shared.

Nothing here raises. Every failure path — no frames, no voiced frames, a
malformed track, an unexpected exception — returns a `BassLine` whose `status`
and `caveats` say what happened. A bass stem that defeats the tracker must not
take the rest of the analysis down with it.

Pipeline, in order (each step's constant carries its own measured
justification):

1. Voicing gate — trust the backend's own `voiced` flags, which each backend
   sets against its own confidence scale (see `PITCH_MIN_CONFIDENCE` in
   `essentia_backend` and `PYIN_MIN_VOICED_PROBABILITY` in `librosa_backend`).
2. Continuous MIDI, ``69 + 12 * log2(f0 / 440)``.
3. `MEDIAN_FILTER_FRAMES`-frame median filter, to kill single-frame flickers.
4. **Exact-octave snap** toward a `OCTAVE_MEDIAN_FRAMES`-frame running median,
   counting every correction.
5. Round to the nearest semitone.
6. Segment on a note change or a voicing gap.
7. Merge sub-`MIN_NOTE_SECONDS` segments into a neighbour no more than
   `MERGE_MAX_SEMITONES` away — glides and vibrato — and drop what will not
   merge.
8. Quantise to the drum grid, when the caller has one.

Conventions this module fixes for the whole project:

* **Strudel spells MIDI 60 as `c4`**, and note names are lowercase with sharps
  (`a1`, `c2`, `d#3`). See `NOTE_NAMES` and :func:`note_name`.
* `BassNote.median_f0_hz` is the median of the **raw, pre-filter** F0 over the
  note's frames, so the reported Hz stays a measurement rather than a
  reconstruction of the note the segmenter decided on. `cents_offset` is
  therefore a real residual and not always zero.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

from .schemas import BassLine, BassNote, PitchTrack

__all__ = [
    "A4_HZ",
    "A4_MIDI",
    "HIGH_REGISTER_CAVEAT_MIDI",
    "LOW_VOICING_CAVEAT_FRACTION",
    "MEDIAN_FILTER_FRAMES",
    "MERGE_MAX_SEMITONES",
    "MIN_NOTE_SECONDS",
    "NOTE_NAMES",
    "OCTAVE_CORRECTION_CAVEAT_RATE",
    "OCTAVE_MEDIAN_FRAMES",
    "TUNING_OFFSET_CAVEAT_CENTS",
    "hz_from_midi",
    "midi_from_hz",
    "note_name",
    "segment_notes",
]

#: Concert pitch, and the MIDI number it sits on. The whole Hz <-> MIDI
#: conversion is defined by this pair and nothing else.
A4_HZ = 440.0
A4_MIDI = 69

#: Pitch-class spellings, index 0 = C. **Lowercase, sharps not flats**, because
#: that is what Strudel's mini-notation accepts — `c#2`, not `Db2` and not
#: `C#2`. Combined with `midi // 12 - 1` for the octave this puts MIDI 60 at
#: `c4`, which is Strudel's convention (and not, e.g., Yamaha's `c3`).
NOTE_NAMES: tuple[str, ...] = (
    "c",
    "c#",
    "d",
    "d#",
    "e",
    "f",
    "f#",
    "g",
    "g#",
    "a",
    "a#",
    "b",
)

#: Width, in frames, of the median filter applied to continuous MIDI before
#: anything is decided. At the 512-sample hop both backends use (11.6 ms at
#: 44.1 kHz) this is ~58 ms, so it removes one- and two-frame flickers while
#: leaving a real 60 ms note — the shortest this module will emit at all — with
#: a majority of its own frames. Odd, so the median is an actual sample.
MEDIAN_FILTER_FRAMES = 5

#: Width, in frames, of the running median the octave guard compares against:
#: ~244 ms at the shared hop, i.e. long enough to span a note but short enough
#: to follow a bass line that moves. Deliberately much wider than
#: `MEDIAN_FILTER_FRAMES` — a guard that tracked the flicker it is meant to
#: correct would agree with every error.
OCTAVE_MEDIAN_FRAMES = 21

#: Correction rate above which the octave guard is reported as a caveat rather
#: than trusted silently. Isolated octave errors are what the guard is for; a
#: line where more than 5% of voiced frames needed moving is a line whose
#: fundamental is weak everywhere, and the guard cannot fix that (see
#: `HIGH_REGISTER_CAVEAT_MIDI`).
OCTAVE_CORRECTION_CAVEAT_RATE = 0.05

#: Shortest note this module will emit. 60 ms is a 32nd note at 125 BPM — below
#: it, on a bass, you are looking at a glide, a vibrato excursion or a tracker
#: artefact rather than a note anyone played. `bass_line_with_glide` pins this:
#: its 100 ms ramp across 9 semitones gives each intermediate semitone ~11 ms,
#: and without a floor the segmenter emits that as chromatic junk.
MIN_NOTE_SECONDS = 0.06

#: How far apart, in semitones, two segments may be and still merge when one of
#: them is too short to stand alone. 1 semitone absorbs glides and vibrato; it
#: will not merge a genuine leap, and a short segment with no near neighbour is
#: dropped rather than absorbed into something it does not belong to.
MERGE_MAX_SEMITONES = 1

#: Voiced-frame share below which the line is reported with a low-confidence
#: caveat. Measured on `bass_line_a_minor`, both backends sit at 0.77-0.84 with
#: their own voicing gates applied; a real stem full of bleed and rests will sit
#: lower, and the reader should know.
LOW_VOICING_CAVEAT_FRACTION = 0.5

#: Above this median MIDI note the line gets an octave caveat regardless of how
#: few corrections the guard made. MIDI 48 is c3, an octave above where a bass
#: normally sits. This is the disclosure for the failure the octave guard
#: structurally *cannot* catch: a stem whose fundamental was rolled off tracks an
#: octave high **consistently**, so the running median agrees with every frame
#: and the guard correctly leaves it all alone. `median_cents_offset` cannot see
#: it either — an octave is exactly 0 cents.
HIGH_REGISTER_CAVEAT_MIDI = 48

#: Median cents offset beyond which the line is flagged as deliberately detuned
#: rather than mistracked. 25 cents is a quarter of a semitone; a 432 Hz master
#: reads -31.8 cents, and a pitched-up remix reads whatever it was pitched by.
TUNING_OFFSET_CAVEAT_CENTS = 25.0

#: Value used where a frame has no pitch. NaN rather than 0.0 so it can never be
#: mistaken for a frequency and so `np.nanmedian` skips it for free.
_UNVOICED = float("nan")


def midi_from_hz(frequency_hz: float) -> float:
    """Continuous MIDI number for `frequency_hz`, or NaN when undefined.

    ``69 + 12 * log2(f / 440)``. Deliberately continuous — the fractional part
    is what `cents_offset` is measured from and what the octave guard reasons
    about. Non-positive and non-finite inputs give NaN rather than raising.
    """
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return _UNVOICED
    return A4_MIDI + 12.0 * float(np.log2(frequency_hz / A4_HZ))


def hz_from_midi(midi_note: float) -> float:
    """Equal-tempered frequency of `midi_note` against A4 = 440 Hz."""
    return A4_HZ * float(2.0 ** ((float(midi_note) - A4_MIDI) / 12.0))


def note_name(midi_note: int) -> str:
    """Strudel spelling of `midi_note`: lowercase, sharps, MIDI 60 = ``c4``.

    ``midi // 12 - 1`` is the octave, so 33 -> ``a1``, 36 -> ``c2``, 60 ->
    ``c4``. Negative MIDI numbers are spelled the same way (they cannot arise
    from `BASS_F0_MIN_HZ`, but the function should not lie if they do).
    """
    index = int(midi_note)
    return f"{NOTE_NAMES[index % 12]}{index // 12 - 1}"


def _nan_median_filter(values: npt.NDArray[np.float64], width: int) -> npt.NDArray[np.float64]:
    """Centred running median over `width` frames, ignoring NaN.

    Edges are padded with NaN rather than with the edge value, so a note at the
    very start of the track is not widened by frames that were never measured.
    An all-NaN window yields NaN, which is the correct answer and not an error —
    numpy warns about it, so the warning is suppressed here rather than in the
    caller.
    """
    if width <= 1 or values.size == 0:
        return values.astype(np.float64, copy=True)

    half = width // 2
    padded = np.pad(values.astype(np.float64), half, constant_values=_UNVOICED)
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        smoothed = np.nanmedian(windows, axis=-1)
    return np.asarray(smoothed, dtype=np.float64)


def _snap_octaves(
    midi: npt.NDArray[np.float64],
    voiced: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], int]:
    """Move frames by whole octaves toward a `OCTAVE_MEDIAN_FRAMES` running median.

    Pitch trackers fail by octaves, not by semitones: YIN halves or doubles a
    period, so an error is almost always exactly 12 or 24 semitones. That makes
    the fix safe in a way a general smoother would not be — only **exact
    multiples of 12** are ever applied, so a frame that is genuinely a semitone
    off is never dragged onto its neighbours' note.

    A frame moves only when the move brings it strictly closer to the running
    median. Returns the corrected track and the number of frames moved; the
    caller turns a high rate into a caveat rather than hiding it.
    """
    running = _nan_median_filter(midi, OCTAVE_MEDIAN_FRAMES)
    usable = voiced & np.isfinite(midi) & np.isfinite(running)
    if not bool(np.any(usable)):
        return midi.astype(np.float64, copy=True), 0

    with np.errstate(invalid="ignore"):
        octaves = np.where(usable, np.round((running - midi) / 12.0), 0.0)
    candidate = midi + 12.0 * octaves
    with np.errstate(invalid="ignore"):
        closer = np.abs(candidate - running) < np.abs(midi - running)
    snap = usable & (octaves != 0.0) & closer

    corrected = np.where(snap, candidate, midi)
    return np.asarray(corrected, dtype=np.float64), int(np.count_nonzero(snap))


class _Segment:
    """A run of consecutive frames the segmenter currently calls one note."""

    __slots__ = ("end", "midi", "start")

    def __init__(self, start: int, end: int, midi: int) -> None:
        self.start = start
        self.end = end  # exclusive
        self.midi = midi

    @property
    def frames(self) -> int:
        return self.end - self.start


def _initial_segments(
    rounded: npt.NDArray[np.float64],
    voiced: npt.NDArray[np.bool_],
) -> list[_Segment]:
    """Split voiced frames wherever the rounded semitone changes.

    A voicing gap of any length also splits, which is what separates the two
    consecutive `a1` notes in `bass_line_a_minor` — they are the same pitch, so
    nothing else could. Both backends' voicing gates were chosen against that
    fixture precisely so the 40 ms silence between its notes shows up here (see
    each backend's voicing-threshold constant for the measured run lengths).
    """
    segments: list[_Segment] = []
    current: _Segment | None = None
    for index in range(rounded.size):
        if not bool(voiced[index]) or not np.isfinite(rounded[index]):
            current = None
            continue
        value = int(rounded[index])
        if current is not None and current.end == index and current.midi == value:
            current.end = index + 1
            continue
        current = _Segment(index, index + 1, value)
        segments.append(current)
    return segments


def _merge_short_segments(segments: list[_Segment], hop_seconds: float) -> list[_Segment]:
    """Absorb sub-`MIN_NOTE_SECONDS` segments into a near neighbour, else drop them.

    Repeatedly takes the **shortest** offending segment and merges it into
    whichever frame-adjacent neighbour is longer, provided that neighbour is
    within `MERGE_MAX_SEMITONES`. The merged run takes the *longer* segment's
    pitch, which is what makes a glide collapse outward into the notes at its
    two ends rather than settling on some chromatic value in the middle: each
    merge makes the ends longer and therefore more attractive to the next one.

    Neighbours must be frame-contiguous. Two segments either side of a voicing
    gap are two notes, however close in pitch, so a gap is never merged across.

    A short segment with no eligible neighbour is dropped. That is the rule that
    keeps `bass_unvoiced` honest: F0 on noise jumps by far more than a semitone
    frame to frame, so nothing merges, nothing survives the floor, and nothing
    is invented.

    Terminates because every iteration removes exactly one segment from the
    list.
    """
    minimum_frames = MIN_NOTE_SECONDS / hop_seconds
    working = list(segments)

    while True:
        short = [i for i, segment in enumerate(working) if segment.frames < minimum_frames]
        if not short:
            return working

        index = min(short, key=lambda i: working[i].frames)
        segment = working[index]

        best: int | None = None
        for neighbour_index in (index - 1, index + 1):
            if not 0 <= neighbour_index < len(working):
                continue
            neighbour = working[neighbour_index]
            contiguous = (
                neighbour.end == segment.start
                if neighbour_index < index
                else segment.end == neighbour.start
            )
            if not contiguous:
                continue
            if abs(neighbour.midi - segment.midi) > MERGE_MAX_SEMITONES:
                continue
            if best is None or working[neighbour_index].frames > working[best].frames:
                best = neighbour_index

        if best is None:
            working.pop(index)
            continue

        target = working[best]
        target.start = min(target.start, segment.start)
        target.end = max(target.end, segment.end)
        working.pop(index)


def _grid_step(
    start_seconds: float,
    grid_anchor_seconds: float | None,
    step_seconds: float | None,
    steps_per_cycle: int | None,
) -> int | None:
    """Which drum-grid step `start_seconds` falls on, or None with no grid.

    Folded into one cycle when `steps_per_cycle` is known, so the step number
    means the same thing it does in `DrumPattern.steps`.
    """
    if step_seconds is None or not np.isfinite(step_seconds) or step_seconds <= 0.0:
        return None
    anchor = 0.0 if grid_anchor_seconds is None else float(grid_anchor_seconds)
    step = int(round((start_seconds - anchor) / step_seconds))
    if steps_per_cycle is not None and steps_per_cycle > 0:
        return step % steps_per_cycle
    return step


def _voiced_mask(track: PitchTrack, frame_count: int) -> npt.NDArray[np.bool_]:
    """The backend's voicing decision, defaulting to "voiced" where absent.

    `voiced` is the backend's own call, made against its own confidence scale —
    the same arrangement `bpm_confidence` already has, and for the same reason:
    the two libraries' confidence numbers are not comparable, so a shared
    threshold here would mean two different things. A track that omits `voiced`
    entirely (a hand-built one, or a future backend) falls back to "every frame
    with a usable F0 is voiced", which the F0-range gate below still filters.
    """
    voiced = np.asarray(track.voiced, dtype=bool)
    if voiced.size != frame_count:
        return np.ones(frame_count, dtype=bool)
    return voiced


def segment_notes(
    track: PitchTrack,
    *,
    grid_anchor_seconds: float | None = None,
    step_seconds: float | None = None,
    steps_per_cycle: int | None = None,
) -> BassLine:
    """Turn a backend's raw F0 track into a `BassLine`. Never raises.

    Args:
        track: A `PitchTrack` from any backend's `pitch()`. Only `f0_hz`,
            `voiced`, `voiced_probability` and `frame_hop_seconds` are read, so
            a hand-built track works exactly as well as a measured one — which
            is what makes "identical F0 gives identical notes on both backends"
            testable rather than aspirational.
        grid_anchor_seconds: Time of grid step 0, from
            `DrumDecomposition.grid_anchor_seconds`. Defaults to 0.0 when a
            `step_seconds` is given without one.
        step_seconds: Length of one grid step. When omitted, every note's `step`
            is `None` — a wrong grid is worse than no grid, the same bias
            `strudel_hints` already takes.
        steps_per_cycle: Steps in one cycle, used to fold `step` into a single
            cycle so it means what `DrumPattern.steps` means.

    Returns:
        A `BassLine` with `status`:

        * ``ok`` — at least one note survived.
        * ``unvoiced`` — no frames, no voiced frames, or voiced frames that
          never held one pitch for `MIN_NOTE_SECONDS`. All three are the same
          fact for a reader: this source has no trackable pitch. The `caveats`
          say which one it was.
        * ``failed`` — a malformed track or an unexpected exception, described
          in `caveats`.

        **`start_seconds` is always the measured onset**, never the quantised
        one. Quantisation is reported in `step`, which is the field the schema
        provides for it; overwriting a measurement with a grid position would
        destroy the evidence for the grid being right.
    """
    try:
        return _segment_notes(
            track,
            grid_anchor_seconds=grid_anchor_seconds,
            step_seconds=step_seconds,
            steps_per_cycle=steps_per_cycle,
        )
    except Exception as error:  # pragma: no cover - defence in depth
        return BassLine(
            status="failed",
            caveats=[f"Note segmentation raised {type(error).__name__}: {error}"],
        )


def _segment_notes(
    track: PitchTrack,
    *,
    grid_anchor_seconds: float | None,
    step_seconds: float | None,
    steps_per_cycle: int | None,
) -> BassLine:
    """Body of :func:`segment_notes`, wrapped there so nothing escapes."""
    hop = track.frame_hop_seconds
    raw_f0 = np.asarray(track.f0_hz, dtype=np.float64)

    if raw_f0.size == 0:
        return BassLine(
            status="unvoiced",
            voiced_fraction=0.0,
            caveats=["The pitch tracker returned no frames, so there is nothing to segment."],
        )
    if hop is None or not np.isfinite(hop) or hop <= 0.0:
        return BassLine(
            status="failed",
            caveats=[f"PitchTrack.frame_hop_seconds is {hop!r}; frame times are unknowable."],
        )

    voiced = _voiced_mask(track, raw_f0.size)
    probability = np.asarray(track.voiced_probability, dtype=np.float64)
    if probability.size != raw_f0.size:
        probability = np.full(raw_f0.size, _UNVOICED)

    # A frame is only usable if the backend called it voiced *and* the F0 is a
    # real, positive number. Backends write 0.0 into unvoiced frames, which
    # would otherwise become MIDI -inf.
    usable = voiced & np.isfinite(raw_f0) & (raw_f0 > 0.0)
    voiced_fraction = float(np.count_nonzero(usable)) / float(raw_f0.size)

    if not bool(np.any(usable)):
        return BassLine(
            status="unvoiced",
            voiced_fraction=0.0,
            caveats=[
                "The pitch tracker found no voiced frames: this source has no pitch to track."
            ],
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        continuous = np.where(usable, A4_MIDI + 12.0 * np.log2(raw_f0 / A4_HZ), _UNVOICED)
    continuous = np.asarray(continuous, dtype=np.float64)

    smoothed = _nan_median_filter(continuous, MEDIAN_FILTER_FRAMES)
    smoothed = np.where(usable, smoothed, _UNVOICED)

    corrected, octave_corrections = _snap_octaves(smoothed, usable)
    rounded = np.where(usable & np.isfinite(corrected), np.round(corrected), _UNVOICED)

    segments = _merge_short_segments(_initial_segments(rounded, usable), float(hop))

    notes = [
        _build_note(
            segment,
            raw_f0=raw_f0,
            probability=probability,
            usable=usable,
            hop=float(hop),
            grid_anchor_seconds=grid_anchor_seconds,
            step_seconds=step_seconds,
            steps_per_cycle=steps_per_cycle,
        )
        for segment in segments
    ]

    if not notes:
        return BassLine(
            status="unvoiced",
            voiced_fraction=voiced_fraction,
            octave_corrections=octave_corrections,
            caveats=[
                f"{int(np.count_nonzero(usable))} frames were called voiced, but none held a "
                f"single pitch for {MIN_NOTE_SECONDS * 1000:.0f} ms, so no note was emitted. "
                "Unpitched low material — rumble, or a kick bleeding into a bass stem — reads "
                "exactly like this."
            ],
        )

    return _finish_line(
        notes,
        rounded=rounded,
        usable=usable,
        voiced_fraction=voiced_fraction,
        octave_corrections=octave_corrections,
    )


def _build_note(
    segment: _Segment,
    *,
    raw_f0: npt.NDArray[np.float64],
    probability: npt.NDArray[np.float64],
    usable: npt.NDArray[np.bool_],
    hop: float,
    grid_anchor_seconds: float | None,
    step_seconds: float | None,
    steps_per_cycle: int | None,
) -> BassNote:
    """One `BassNote` from one settled segment.

    `median_f0_hz` is the median of the **raw** F0 over the segment's own voiced
    frames — before the median filter and before the octave snap — so the Hz
    figure remains something that was measured. `cents_offset` is that
    measurement's distance from equal temperament, which is therefore a real
    residual: a line played against a 432 Hz reference shows a consistent -32
    cents here rather than a page of wrong note names.
    """
    frames = slice(segment.start, segment.end)
    in_note = usable[frames]

    note_f0 = raw_f0[frames][in_note]
    median_f0 = float(np.median(note_f0)) if note_f0.size else None

    cents: float | None = None
    if median_f0 is not None and median_f0 > 0.0:
        cents = 1200.0 * float(np.log2(median_f0 / hz_from_midi(segment.midi)))

    note_probability = probability[frames][in_note]
    finite_probability = note_probability[np.isfinite(note_probability)]
    confidence = float(np.mean(finite_probability)) if finite_probability.size else None

    start_seconds = segment.start * hop
    return BassNote(
        start_seconds=start_seconds,
        duration_seconds=segment.frames * hop,
        midi_note=segment.midi,
        note_name=note_name(segment.midi),
        median_f0_hz=median_f0,
        cents_offset=cents,
        confidence=confidence,
        step=_grid_step(start_seconds, grid_anchor_seconds, step_seconds, steps_per_cycle),
    )


def _finish_line(
    notes: list[BassNote],
    *,
    rounded: npt.NDArray[np.float64],
    usable: npt.NDArray[np.bool_],
    voiced_fraction: float,
    octave_corrections: int,
) -> BassLine:
    """Line-level statistics and the caveats they imply."""
    settled = rounded[usable & np.isfinite(rounded)]
    median_midi = int(round(float(np.median(settled)))) if settled.size else None

    offsets = [note.cents_offset for note in notes if note.cents_offset is not None]
    median_cents = float(np.median(offsets)) if offsets else None

    caveats: list[str] = []

    voiced_frames = int(np.count_nonzero(usable))
    correction_rate = octave_corrections / voiced_frames if voiced_frames else 0.0
    if correction_rate > OCTAVE_CORRECTION_CAVEAT_RATE:
        caveats.append(
            f"The octave guard moved {correction_rate:.1%} of voiced frames "
            f"({octave_corrections} of {voiced_frames}). Above "
            f"{OCTAVE_CORRECTION_CAVEAT_RATE:.0%} that usually means a weak fundamental "
            "throughout, not isolated slips — treat the octave of this line as uncertain."
        )

    if median_midi is not None and median_midi > HIGH_REGISTER_CAVEAT_MIDI:
        caveats.append(
            f"Median note is {note_name(median_midi)} (MIDI {median_midi}), above "
            f"{note_name(HIGH_REGISTER_CAVEAT_MIDI)} — high for a bass. A stem whose "
            "fundamental was rolled off tracks an octave high consistently, and neither the "
            "octave guard nor median_cents_offset can see that (an octave is 0 cents). "
            "Check the octave by ear before typing this in."
        )

    if voiced_fraction < LOW_VOICING_CAVEAT_FRACTION:
        caveats.append(
            f"Only {voiced_fraction:.0%} of frames were voiced, so this line is built from "
            "less than half the stem. Expect missed notes."
        )

    if median_cents is not None and abs(median_cents) > TUNING_OFFSET_CAVEAT_CENTS:
        caveats.append(
            f"The whole line sits {median_cents:+.0f} cents from equal temperament. That is a "
            "tuning reference or a pitched remix, not a line of wrong notes — the note names "
            "are still the right relative spelling."
        )

    return BassLine(
        status="ok",
        notes=notes,
        median_midi_note=median_midi,
        median_cents_offset=median_cents,
        voiced_fraction=voiced_fraction,
        octave_corrections=octave_corrections,
        caveats=caveats,
    )
