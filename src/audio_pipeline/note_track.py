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
   Below `MIN_VOICED_FRACTION` of the stem the whole line is refused rather
   than reported — see that constant.
2. Continuous MIDI, ``69 + 12 * log2(f0 / 440)``.
3. `MEDIAN_FILTER_FRAMES`-frame median filter, to kill single-frame flickers.
4. **Exact-octave snap** toward a `OCTAVE_MEDIAN_FRAMES`-frame running median,
   counting every correction.
5. Round to the nearest semitone.
6. Segment on a note change or a voicing gap.
7. Merge sub-`MIN_NOTE_SECONDS` segments into a neighbour no more than
   `MERGE_MAX_SEMITONES` away — glides and vibrato — and drop what will not
   merge.
8. **Backdate each onset** across the frames the voicing gate rejected while
   the tracker's confidence was still climbing — see :func:`_backdate_onsets`.
9. Quantise to the drum grid, when the caller has one.

Onset timing, and where the lag actually was
--------------------------------------------

Calibration (F3 in `V2-PLAN.md`) measured every one of the 709 bass notes on
the Madonna fixture sitting **0.29 sixteenth-steps — 33 ms — late** against a
verified 132.000 BPM grid, and attributed it to "the median filter plus the
minimum-frames-to-confirm logic" in this module. Re-measured here against
`tests/fixtures/real/madonna__bass_f0.npz`, that attribution is wrong, and it
is worth writing down why so nobody re-derives it:

* A **centred median filter of odd width has exactly zero group delay at a step
  edge.** For a window of width ``W`` centred on frame ``i``, the new value wins
  the median as soon as ``(W + 1) / 2`` of the window's frames are past the
  edge, which first happens at ``i`` equal to the edge itself. Sweeping
  `MEDIAN_FILTER_FRAMES` over 1, 3, 5, 7, 9 and 11 on the fixture moves the
  measured offset by 0.0006 steps — i.e. not at all. A ``(W - 1) / 2 * hop``
  correction would have been a 23 ms over-correction dressed up as a
  derivation.
* There is no minimum-frames-to-confirm threshold in this module. The nearest
  thing, `MIN_NOTE_SECONDS`, gates *length*, not onset: setting it to 0 leaves
  the offset at 0.2965 steps against 0.2916.
* The **raw voicing-run onsets carry the whole 33 ms** before this module sees
  them: the first frame each backend flags `voiced` already sits at 0.2925
  steps. 700 of the 709 notes begin on one.

So the lag is the pitch tracker's *voicing confidence ramp*, not segmentation.
A YIN-family estimator needs several periods of the new fundamental inside its
analysis window before periodicity clears the confidence threshold, and the
lower the note the longer that takes — measured per pitch on the fixture, a2
(110 Hz) lags 23.6 ms and c2 (65 Hz) lags 31.6 ms.

What makes this recoverable rather than merely explainable: **the F0 estimate
is already correct before the confidence catches up.** Backends write an F0 for
every frame, voiced or not, and on the frames immediately before a voicing
onset that F0 is typically already within a few cents of the note that is
about to be declared, with `voiced_probability` visibly climbing toward the
gate. :func:`_backdate_onsets` walks back down that ramp. It is a per-note
measurement, not a constant: on the fixture it moves notes by 0-5 frames
(mean 2.46), takes the circular mean offset from **0.2916 steps to 0.0749**,
mean absolute quantisation error from **0.2814 steps to 0.1116**, and — the
part a constant offset structurally cannot do — raises the concentration
``R`` of the offset distribution from **0.548 to 0.724**. Subtracting a
constant leaves ``R`` exactly where it was; this removes real variance.

The 0.075 steps that remain are 8.5 ms, below the 11.6 ms frame hop, so they
are the frame grid itself and not something further to chase.

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
    "BEATS_PER_CYCLE",
    "DEFAULT_STEPS_PER_CYCLE",
    "HIGH_REGISTER_CAVEAT_MIDI",
    "LOW_VOICING_CAVEAT_FRACTION",
    "MEDIAN_FILTER_FRAMES",
    "MERGE_MAX_SEMITONES",
    "MIN_NOTE_SECONDS",
    "MIN_VOICED_FRACTION",
    "NOTE_NAMES",
    "OCTAVE_CORRECTION_CAVEAT_RATE",
    "OCTAVE_MEDIAN_FRAMES",
    "ONSET_BACKDATE_MAX_SECONDS",
    "ONSET_BACKDATE_MAX_SEMITONES",
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

#: Voiced-frame share below which **no line is reported at all**: the result is
#: `status="unvoiced"` with no notes, rather than `ok` with pitches tracked out
#: of a noise floor. This is half of F5 in `V2-PLAN.md`.
#:
#: Measured, from the three v4 calibration tracks in `calibration/v4/`:
#:
#: | stem                                   | rms_mean | voiced_fraction | is it a bass? |
#: |----------------------------------------|----------|-----------------|---------------|
#: | madonna, real separated bass           | 0.104    | **0.454**       | yes           |
#: | ancient-heavy-tech-donjon, silent stem | 7.7e-05  | **0.089**       | no            |
#: | showers-of-gold, silent stem           | 8.4e-05  | **0.012**       | no            |
#:
#: Both silent stems reported `status: "ok"` under v4, with 2 and 1 notes and a
#: median of `e4` — a pitch tracked from a stem sitting at about -82 dBFS. The
#: gap between 0.089 and 0.454 is a factor of five, so the floor is not
#: delicately placed: 0.15 is 1.7x above the worst false positive and 3x below
#: the one real bass in the corpus. Started at the 0.15 the plan suggested and
#: left there because the measurements gave no reason to move it — but it is
#: calibrated against exactly one real bass, so it is provisional until W8B's
#: five-track corpus exists. The ambient/rubato track in that corpus is the one
#: that should decide it.
#:
#: This is a *coverage* floor, not a level floor. A stem can be loud and still
#: unpitched (a kick bleeding into a bass stem); W6 adds the separate
#: `rms_mean` silence gate that catches the other half of F5.
MIN_VOICED_FRACTION = 0.15

#: Voiced-frame share below which the line is *reported* but carries a coverage
#: caveat. Lowered from 0.5 to 0.3 in v5 — F6.
#:
#: At 0.5 this fired on the Madonna bass at 0.454 and said "this line is built
#: from less than half the stem", which reads as a defect and is not one: a bass
#: line with rests is unvoiced roughly half the time **by construction**, and
#: the 709 notes it produced quantise to the grid at 0.11 steps. A caveat that
#: fires on the one correct result in the corpus trains the reader to skip the
#: caveat list, which costs more than the caveat was ever worth.
#:
#: Note the deliberate consequence of pairing this with `MIN_VOICED_FRACTION`:
#: the caveat can now only ever fire in the 0.15-0.30 window, because below 0.15
#: there is no line to attach it to. That is the intent. Nothing in the current
#: three-track corpus lands in that window — the measured values are 0.012,
#: 0.089 and 0.454 — so this threshold is untested against real material and is
#: reasoned, not measured. **Do not widen either number to make the other
#: reachable**; a caveat that never fires is a better failure than one that
#: cries wolf.
LOW_VOICING_CAVEAT_FRACTION = 0.3

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

#: How far, in semitones, a pre-voicing frame's raw F0 may sit from the note the
#: segmenter settled on and still be counted as part of that note's attack.
#:
#: Deliberately **the same number as `MERGE_MAX_SEMITONES`**, and defined in
#: terms of it rather than repeated, because it is the same judgement: one
#: semitone is already this module's definition of "close enough in pitch to be
#: the same note". Widening it to 2 semitones buys 0.005 steps on the Madonna
#: fixture (0.1116 -> 0.1065) and starts being able to swallow a neighbouring
#: note a whole tone away, which is a bad trade.
ONSET_BACKDATE_MAX_SEMITONES = MERGE_MAX_SEMITONES

#: Furthest back an onset may be moved. **A note may not be backdated by more
#: than the shortest note this module is willing to emit** — if the correction
#: could be longer than a whole note, it is not a correction any more.
#:
#: Defined in terms of `MIN_NOTE_SECONDS` for that reason, and converted to
#: frames at call time against the track's own hop, so the bound follows the
#: backend's frame rate rather than assuming one. At the 512-sample hop both
#: backends use this is 5 frames.
#:
#: Measured on the Madonna fixture, the correction saturates well inside it:
#: raising the bound to 8 or 12 frames changes the result by 0.003 steps, and
#: the realised shifts are 0-5 frames with a mean of 2.46. It is a guard rail,
#: not a tuning knob.
ONSET_BACKDATE_MAX_SECONDS = MIN_NOTE_SECONDS

#: Beats in one Strudel cycle, and steps in one cycle, when the caller supplies
#: a *beat period* (from `tempo.TempoFit`) instead of a step length. 4 beats of
#: 4 sixteenths — the same 4/4 convention `drum_elements` uses, where
#: `cycle_seconds = beats_per_cycle * 60 / bpm` and `GRID_STEP_CANDIDATES`
#: leads with 16. Kept as two numbers rather than one "steps per beat" so a
#: caller working in triplets can pass `steps_per_cycle=12` and get
#: `beat_period * 4 / 12` — a triplet eighth — without a second convention.
BEATS_PER_CYCLE = 4
DEFAULT_STEPS_PER_CYCLE = 16

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


def _backdate_onsets(
    segments: list[_Segment],
    *,
    raw_f0: npt.NDArray[np.float64],
    probability: npt.NDArray[np.float64],
    usable: npt.NDArray[np.bool_],
    hop: float,
) -> None:
    """Move each onset back across the pre-voicing attack, in place.

    Every backend flags a note voiced some frames *after* it starts, because a
    YIN-family estimator needs several periods of the new fundamental inside its
    analysis window before periodicity clears the confidence threshold. See this
    module's docstring for the measurement, including the proof that this
    module's own median filter contributes none of it.

    The correction is possible because backends write an F0 for **every** frame,
    including the ones their own voicing gate rejects — so the frames just
    before an onset usually already carry the right pitch at a confidence that
    is climbing toward the gate but has not reached it. This walks back down
    that ramp, one frame at a time, and stops at the first frame that fails any
    of four conditions:

    * it belongs to the previous note (or is before frame 0). Notes may not
      overlap, and an onset may never be dragged into the note in front of it.
    * the voicing gate **accepted** it. This correction only ever recovers
      frames the gate rejected; a frame it accepted and the segmenter did not
      give to this note belongs to something else — a dropped short segment,
      or a gap the merge rule left behind.
    * its raw F0 is missing, or further than `ONSET_BACKDATE_MAX_SEMITONES`
      from the note the segmenter settled on. This is what keeps the walk
      inside the attack of *this* note rather than running back through the
      release of the last one.
    * `voiced_probability` is *higher* there than at the frame in front of it.
      Confidence rising toward the gate is the signature of an attack;
      confidence falling away from it is the tail of something else. A track
      with no usable probabilities (a hand-built one, or a backend that omits
      them) simply skips this condition and relies on the other three.

    Everything it needs is already in the `PitchTrack`, so both backends still
    produce byte-identical notes from identical F0 — the property this whole
    module exists to hold.

    `median_f0_hz`, `cents_offset` and `confidence` are unaffected: `_build_note`
    computes them over the segment's `usable` frames only, and every frame added
    here is by construction not usable. The reported Hz stays a measurement over
    frames the backend was confident about, while the reported *time* stops
    being 33 ms late.
    """
    max_frames = int(ONSET_BACKDATE_MAX_SECONDS / hop)
    if max_frames < 1 or not segments:
        return

    with np.errstate(divide="ignore", invalid="ignore"):
        pre_voicing_midi = np.where(
            np.isfinite(raw_f0) & (raw_f0 > 0.0),
            A4_MIDI + 12.0 * np.log2(np.where(raw_f0 > 0.0, raw_f0, 1.0) / A4_HZ),
            _UNVOICED,
        )

    floor = 0
    for segment in segments:
        frame = segment.start
        while segment.start - frame < max_frames:
            candidate = frame - 1
            if candidate < floor:
                break
            if bool(usable[candidate]):
                break
            if not np.isfinite(pre_voicing_midi[candidate]):
                break
            if abs(float(pre_voicing_midi[candidate]) - segment.midi) > (
                ONSET_BACKDATE_MAX_SEMITONES
            ):
                break
            here, ahead = probability[candidate], probability[frame]
            if np.isfinite(here) and np.isfinite(ahead) and here > ahead:
                break
            frame = candidate
        segment.start = frame
        floor = segment.end


def _resolve_grid(
    *,
    grid_anchor_seconds: float | None,
    step_seconds: float | None,
    steps_per_cycle: int | None,
    beat_period_seconds: float | None,
    downbeat_seconds: float | None,
) -> tuple[float | None, float | None, int | None]:
    """Reduce the two ways of describing a grid to `(anchor, step, steps)`.

    A caller may describe the grid either in **steps** — the shape
    `drum_elements.DrumDecomposition` already reports, and what `analyze.py`
    passes today — or in **beats**, which is the shape `tempo.TempoFit` will
    report once W4A lands: one refined beat period and one downbeat, in seconds.
    The beat form is converted here rather than at each call site so the 4/4
    convention lives in one documented place (`BEATS_PER_CYCLE`,
    `DEFAULT_STEPS_PER_CYCLE`).

    An explicit `step_seconds` **wins**, because it is the more specific
    statement: a caller that has measured a step length is not asking to have
    one inferred from a beat period. Missing or unusable inputs collapse to
    "no grid", which every caller downstream already reads as `step=None` on
    every note.
    """
    if step_seconds is not None:
        return grid_anchor_seconds, step_seconds, steps_per_cycle

    if beat_period_seconds is None or not np.isfinite(beat_period_seconds):
        return grid_anchor_seconds, None, steps_per_cycle
    if beat_period_seconds <= 0.0:
        return grid_anchor_seconds, None, steps_per_cycle

    steps = DEFAULT_STEPS_PER_CYCLE if steps_per_cycle is None else steps_per_cycle
    if steps <= 0:
        return grid_anchor_seconds, None, None

    anchor = grid_anchor_seconds if downbeat_seconds is None else downbeat_seconds
    return anchor, float(beat_period_seconds) * BEATS_PER_CYCLE / steps, steps


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
    beat_period_seconds: float | None = None,
    downbeat_seconds: float | None = None,
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
            cycle so it means what `DrumPattern.steps` means. Defaults to
            `DEFAULT_STEPS_PER_CYCLE` when the grid is given as a beat period.
        beat_period_seconds: One refined beat period, in seconds — the second,
            equivalent way to state the grid, and the shape `tempo.TempoFit`
            reports. Converted via `BEATS_PER_CYCLE` and `steps_per_cycle`; an
            explicit `step_seconds` wins over it. Optional so this module stays
            independently testable before W4A and W6 exist to supply one.
        downbeat_seconds: Time of the downbeat, in seconds — the anchor that
            goes with `beat_period_seconds`. Falls back to
            `grid_anchor_seconds`, then to 0.0.

    Returns:
        A `BassLine` with `status`:

        * ``ok`` — at least one note survived.
        * ``unvoiced`` — no frames, no voiced frames, fewer than
          `MIN_VOICED_FRACTION` of frames voiced, or voiced frames that never
          held one pitch for `MIN_NOTE_SECONDS`. All four are the same fact for
          a reader: this source has no trackable pitch. The `caveats` say which
          one it was.
        * ``failed`` — a malformed track or an unexpected exception, described
          in `caveats`.

        **`start_seconds` is always a measured onset**, never a quantised one.
        It is the frame where this note's attack starts, which since v5 is
        recovered from the pre-voicing frames (see :func:`_backdate_onsets`)
        rather than taken from the frame the backend's confidence happened to
        cross its own threshold. Quantisation is reported in `step`, which is
        the field the schema provides for it; overwriting a measurement with a
        grid position would destroy the evidence for the grid being right.
    """
    try:
        return _segment_notes(
            track,
            grid_anchor_seconds=grid_anchor_seconds,
            step_seconds=step_seconds,
            steps_per_cycle=steps_per_cycle,
            beat_period_seconds=beat_period_seconds,
            downbeat_seconds=downbeat_seconds,
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
    beat_period_seconds: float | None = None,
    downbeat_seconds: float | None = None,
) -> BassLine:
    """Body of :func:`segment_notes`, wrapped there so nothing escapes."""
    grid_anchor_seconds, step_seconds, steps_per_cycle = _resolve_grid(
        grid_anchor_seconds=grid_anchor_seconds,
        step_seconds=step_seconds,
        steps_per_cycle=steps_per_cycle,
        beat_period_seconds=beat_period_seconds,
        downbeat_seconds=downbeat_seconds,
    )
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

    # F5: a stem Demucs left at the noise floor still yields scattered voiced
    # frames, and v4 turned those into notes with `status="ok"`. Refuse the
    # whole line rather than pitch-track a noise floor — see
    # `MIN_VOICED_FRACTION` for the three measured stems this floor sits
    # between. Placed before any segmentation because there is no result worth
    # computing on the far side of it.
    if voiced_fraction < MIN_VOICED_FRACTION:
        return BassLine(
            status="unvoiced",
            voiced_fraction=voiced_fraction,
            caveats=[
                f"Only {voiced_fraction:.1%} of frames carried a pitch, below the "
                f"{MIN_VOICED_FRACTION:.0%} floor this module will report a line from. A stem "
                "left near silence by separation reads exactly like this — it tracks a noise "
                "floor and produces a handful of confident-looking notes that are not there. "
                "No notes are reported rather than notes nobody played."
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
    # After merging, deliberately: backdating must not rescue a segment that was
    # too short to stand on its own, and a merge that has already extended a
    # segment backwards leaves nothing for this to find.
    _backdate_onsets(
        segments,
        raw_f0=raw_f0,
        probability=probability,
        usable=usable,
        hop=float(hop),
    )

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
            f"These notes cover {voiced_fraction:.0%} of the stem — the rest of it carried no "
            "trackable pitch. A bass that rests is silent for much of a track by design, so "
            "low coverage is not by itself a fault, but below "
            f"{LOW_VOICING_CAVEAT_FRACTION:.0%} there is little material behind the line and "
            "quiet or short notes are more likely to be missing than resting."
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
