"""Tests for `note_track.segment_notes` — F0 in, notes out, pure numpy.

Every test here builds a `PitchTrack` **by hand**, from literal frequencies. No
backend runs, no audio is decoded, and neither librosa nor essentia is imported.
That is the point of the seam: note segmentation is shared code with one right
answer, so it can be pinned exactly rather than to a tolerance, and a change in
either backend's F0 estimate cannot silently move these expectations.

The backend-driven half — does `pitch()` actually recover 55.000 Hz, do the two
backends agree — lives in `test_librosa_backend.py` and `test_essentia_backend.py`
where the libraries are already available.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    BASS_GLIDE_HOLD_SECONDS,
    BASS_GLIDE_MIDI,
    BASS_GLIDE_NOTE_NAMES,
    BASS_GLIDE_SECONDS,
    BASS_LINE_FREQS_HZ,
    BASS_LINE_MIDI,
    BASS_LINE_NOTE_NAMES,
)

from audio_pipeline import ANALYSIS_SAMPLE_RATE, note_track
from audio_pipeline.note_track import (
    A4_HZ,
    A4_MIDI,
    HIGH_REGISTER_CAVEAT_MIDI,
    LOW_VOICING_CAVEAT_FRACTION,
    MEDIAN_FILTER_FRAMES,
    MERGE_MAX_SEMITONES,
    MIN_NOTE_SECONDS,
    MIN_VOICED_FRACTION,
    NOTE_NAMES,
    OCTAVE_CORRECTION_CAVEAT_RATE,
    ONSET_BACKDATE_MAX_SECONDS,
    TUNING_OFFSET_CAVEAT_CENTS,
    hz_from_midi,
    midi_from_hz,
    note_name,
    segment_notes,
)
from audio_pipeline.schemas import BassLine, PitchTrack

#: The hop both backends produce, so hand-built tracks share their time grid.
HOP_SECONDS = 512 / ANALYSIS_SAMPLE_RATE

#: The committed raw F0 track for the Madonna bass stem, and the verified grid
#: it sits on. See `tests/fixtures/real/PROVENANCE.md`: 132.000 ± 0.01 BPM
#: measured by autocorrelation of the 20-110 Hz flux, so a bar is 1.818182 s
#: and a sixteenth-step is 0.113636 s. The downbeat is F1's measured 0.228 s.
#:
#: This is deliberately the *input* to segmentation and not a committed note
#: list: a committed note list would already carry the 33 ms lag F3 found, and
#: could never show it had been removed.
MADONNA_F0_NPZ = Path(__file__).parent / "fixtures/real/madonna__bass_f0.npz"
MADONNA_BPM = 132.0
MADONNA_STEP_SECONDS = 60.0 / MADONNA_BPM / 4.0
MADONNA_DOWNBEAT_SECONDS = 0.228


def _track(
    segments: list[tuple[float | None, float]],
    *,
    hop_seconds: float = HOP_SECONDS,
    probability: float = 0.9,
) -> PitchTrack:
    """Build a `PitchTrack` from `(frequency_hz_or_None, seconds)` runs.

    `None` means an unvoiced run — a rest, or the silence between two notes of
    the same pitch, which is the only thing that can separate them.
    """
    f0: list[float] = []
    voiced: list[bool] = []
    for frequency, seconds in segments:
        frames = int(round(seconds / hop_seconds))
        f0 += [0.0 if frequency is None else float(frequency)] * frames
        voiced += [frequency is not None] * frames
    return PitchTrack(
        f0_hz=f0,
        voiced=voiced,
        voiced_probability=[probability if flag else 0.0 for flag in voiced],
        frame_hop_seconds=hop_seconds,
        method="hand-built",
    )


# --------------------------------------------------------------------------- #
# Module hygiene — the property that makes this half of the seam shared
# --------------------------------------------------------------------------- #


def test_note_track_never_imports_an_analysis_library_at_any_level() -> None:
    """Not at module top level, and not inside a function body either.

    `librosa_backend` and `essentia_backend` are allowed their lazy in-function
    imports because they *are* the library adapters. `note_track` is not: it is
    the shared half of the seam, and the moment it can reach a backend library
    the guarantee that both backends produce identical notes from identical F0
    stops being structural.
    """
    source = Path(__file__).parent.parent / "src/audio_pipeline/note_track.py"
    tree = ast.parse(source.read_text())

    forbidden = {"librosa", "essentia"}
    offenders: list[str] = []
    for node in ast.walk(tree):  # every level, not just module top level
        if isinstance(node, ast.Import):
            offenders += [
                alias.name for alias in node.names if alias.name.split(".")[0] in forbidden
            ]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in forbidden:
                offenders.append(node.module)

    assert offenders == [], f"note_track must be numpy-only; found {offenders}"


def test_note_track_runs_with_neither_backend_importable() -> None:
    """Import and segment in a subprocess where both libraries raise on import.

    The AST check above proves nothing is *written*; this proves nothing is
    reached transitively either. Run out of process because this session has
    already imported both libraries for real.
    """
    program = """
import sys

class _Blocked:
    def find_module(self, name, path=None):
        if name.split(".")[0] in {"librosa", "essentia"}:
            raise ImportError(f"{name} is blocked for this test")
        return None

sys.meta_path.insert(0, _Blocked())

from audio_pipeline.note_track import segment_notes
from audio_pipeline.schemas import PitchTrack

track = PitchTrack(
    f0_hz=[55.0] * 40,
    voiced=[True] * 40,
    voiced_probability=[0.9] * 40,
    frame_hop_seconds=512 / 44100,
)
line = segment_notes(track)
print(line.status, len(line.notes), line.notes[0].note_name)
print(sorted(m for m in sys.modules if m.split(".")[0] in {"essentia", "librosa"}))
"""
    src = Path(__file__).resolve().parent.parent / "src"
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "ok 1 a1", result.stdout
    assert lines[1] == "[]", result.stdout


# --------------------------------------------------------------------------- #
# Hz <-> MIDI <-> Strudel note names
# --------------------------------------------------------------------------- #


def test_midi_60_is_c4_which_is_strudels_convention() -> None:
    """Strudel spells middle C as `c4`, not `c3` and not `C4`.

    Load-bearing: this is what the whole `note_name` octave arithmetic hangs
    off, and getting it wrong shifts every note in every output file by an
    octave while leaving the pitch classes right — the kind of error that looks
    fine until you play it.
    """
    assert note_name(60) == "c4"
    assert note_name(69) == "a4"
    assert midi_from_hz(A4_HZ) == pytest.approx(float(A4_MIDI))


def test_note_names_are_lowercase_with_sharps() -> None:
    assert NOTE_NAMES[0] == "c"
    assert set("".join(NOTE_NAMES)) <= set("abcdefg#")
    assert "b" not in "".join(name[1:] for name in NOTE_NAMES)  # no flats


def test_the_fixture_frequencies_round_trip_to_their_documented_names() -> None:
    """`conftest` states A1 = 55.0 Hz -> MIDI 33 -> `a1`. Prove all three agree."""
    for frequency, midi, name in zip(
        BASS_LINE_FREQS_HZ, BASS_LINE_MIDI[:4], BASS_LINE_NOTE_NAMES[:4], strict=True
    ):
        assert round(midi_from_hz(frequency)) == midi
        assert note_name(midi) == name
        assert hz_from_midi(midi) == pytest.approx(frequency, rel=1e-5)


def test_midi_from_hz_is_nan_rather_than_an_exception_on_bad_input() -> None:
    for value in (0.0, -55.0, float("nan"), float("inf")):
        assert np.isnan(midi_from_hz(value))


# --------------------------------------------------------------------------- #
# The bass-line fixtures, as F0
# --------------------------------------------------------------------------- #


def test_the_a_minor_line_segments_into_its_sixteen_documented_notes() -> None:
    """(A1, A1, C2, E2) x 4 at the literal synthesis frequencies.

    Includes the case nothing else in this file covers: **two consecutive
    notes of the same pitch**. Only the silence between them can separate
    `a1` from `a1`, which is why both backends' voicing gates were calibrated
    against exactly this fixture.
    """
    segments: list[tuple[float | None, float]] = []
    for index in range(len(BASS_LINE_MIDI)):
        segments.append((BASS_LINE_FREQS_HZ[index % 4], 0.46))
        segments.append((None, 0.04))

    line = segment_notes(_track(segments))

    assert line.status == "ok"
    assert [note.note_name for note in line.notes] == list(BASS_LINE_NOTE_NAMES)
    assert [note.midi_note for note in line.notes] == list(BASS_LINE_MIDI)
    assert line.caveats == []


def test_reported_hz_is_the_raw_median_not_a_reconstruction() -> None:
    """A note 20 cents sharp keeps its measured Hz and reports the offset.

    If `median_f0_hz` were recomputed from the chosen MIDI number it would read
    exactly 55.0 here and `cents_offset` would be 0.0 — the output would look
    like a measurement while being a restatement of the segmenter's own
    decision.
    """
    detuned = 55.0 * 2 ** (20.0 / 1200.0)
    line = segment_notes(_track([(detuned, 0.5)]))

    assert line.notes[0].midi_note == 33
    assert line.notes[0].median_f0_hz == pytest.approx(detuned)
    assert line.notes[0].cents_offset == pytest.approx(20.0, abs=0.01)


def test_a_consistent_tuning_offset_is_reported_as_one_offset_not_wrong_notes() -> None:
    """A 432 Hz master reads -31.8 cents on every note, with the right names.

    This is the failure `median_cents_offset` exists to prevent: an entire line
    reported a semitone out because the reference pitch was not 440.
    """
    scale = 432.0 / 440.0
    segments: list[tuple[float | None, float]] = []
    for index in range(8):
        segments.append((BASS_LINE_FREQS_HZ[index % 4] * scale, 0.46))
        segments.append((None, 0.04))
    line = segment_notes(_track(segments))

    assert [note.note_name for note in line.notes] == list(BASS_LINE_NOTE_NAMES[:8])
    assert line.median_cents_offset == pytest.approx(-31.77, abs=0.1)
    assert abs(line.median_cents_offset or 0.0) > TUNING_OFFSET_CAVEAT_CENTS
    assert any("cents" in caveat for caveat in line.caveats)


# --------------------------------------------------------------------------- #
# Glides: the reason there is a minimum note length at all
# --------------------------------------------------------------------------- #


def test_a_glide_yields_two_notes_not_a_chromatic_run() -> None:
    """A1 held, a 100 ms ramp across 9 semitones, E2 held — exactly 2 notes.

    Without the minimum-length floor and the merge rule this emits the eight
    intermediate semitones the ramp passes through, each lasting ~11 ms. That
    is the single most obvious way a frame-to-note segmenter produces junk.
    """
    ramp_frames = int(round(BASS_GLIDE_SECONDS / HOP_SECONDS))
    low, high = hz_from_midi(BASS_GLIDE_MIDI[0]), hz_from_midi(BASS_GLIDE_MIDI[1])

    segments: list[tuple[float | None, float]] = []
    for _ in range(4):
        segments.append((low, BASS_GLIDE_HOLD_SECONDS))
        segments += [
            (float(value), HOP_SECONDS)
            for value in np.linspace(low, high, ramp_frames, endpoint=False)
        ]
        segments.append((high, BASS_GLIDE_HOLD_SECONDS))
        segments.append((None, 0.06))

    line = segment_notes(_track(segments))

    assert [note.note_name for note in line.notes] == list(BASS_GLIDE_NOTE_NAMES) * 4
    assert len(line.notes) == 8


def test_a_two_frame_excursion_never_reaches_the_segmenter_at_all() -> None:
    """The median filter is the first line of defence, before any merge rule.

    A 20 ms blip is two frames at the shared hop, so a 5-frame median erases it:
    the segmenter never sees a pitch change and emits one continuous note. Worth
    pinning separately from the merge rule below, because the two look like the
    same behaviour from the outside and fail for completely different reasons.
    """
    line = segment_notes(
        _track(
            [
                (hz_from_midi(33), 0.4),
                (hz_from_midi(39), 0.02),  # two frames — below the median filter's reach
                (hz_from_midi(33), 0.4),
            ]
        )
    )

    assert [note.midi_note for note in line.notes] == [33]
    assert line.notes[0].duration_seconds == pytest.approx(0.82, abs=0.02)


def test_a_short_segment_more_than_one_semitone_away_is_dropped_not_absorbed() -> None:
    """The merge rule is a glide/vibrato rule, not a "swallow anything" rule.

    A 45 ms blip a tritone from its neighbours survives the median filter but is
    still under `MIN_NOTE_SECONDS`. It belongs to neither neighbour: merging it
    would move a real note's pitch, emitting it would invent a note nobody
    played. So it is dropped, and the notes either side are left alone — which
    also means they stay two notes, because the gap it leaves behind separates
    them.
    """
    line = segment_notes(
        _track(
            [
                (hz_from_midi(33), 0.4),
                (hz_from_midi(39), 0.045),  # a tritone up, far beyond MERGE_MAX_SEMITONES
                (hz_from_midi(33), 0.4),
            ]
        )
    )

    assert [note.midi_note for note in line.notes] == [33, 33]
    assert all(note.duration_seconds >= MIN_NOTE_SECONDS for note in line.notes)


def test_vibrato_within_the_merge_tolerance_stays_one_note() -> None:
    """A semitone wobble is one note; MERGE_MAX_SEMITONES says so."""
    wobble = [(hz_from_midi(33 + (index % 2) * MERGE_MAX_SEMITONES), 0.03) for index in range(12)]
    line = segment_notes(_track([(hz_from_midi(33), 0.3), *wobble]))

    assert len(line.notes) == 1
    assert line.notes[0].midi_note == 33


# --------------------------------------------------------------------------- #
# Octave handling
# --------------------------------------------------------------------------- #


def test_isolated_octave_slips_are_snapped_back_and_counted() -> None:
    """Three frames jumping to the octave are pulled back, and reported.

    Trackers fail by octaves, so the guard only ever moves by exact multiples
    of 12 — and it counts what it moved rather than doing it silently.
    """
    frames = [(55.0, HOP_SECONDS)] * 40
    frames[20] = frames[21] = frames[22] = (110.0, HOP_SECONDS)

    line = segment_notes(_track(frames))

    assert [note.midi_note for note in line.notes] == [33]
    assert line.octave_corrections >= 3


def test_a_high_correction_rate_becomes_a_caveat_rather_than_silent_confidence() -> None:
    """Alternate frames an octave up: the guard fixes them and says so."""
    frames = [(110.0 if index % 2 else 55.0, HOP_SECONDS) for index in range(60)]
    line = segment_notes(_track(frames))

    rate = line.octave_corrections / max(sum(1 for value in line.notes), 1)
    assert line.octave_corrections > 0
    assert rate > 0
    assert any("octave guard" in caveat for caveat in line.caveats)
    assert line.octave_corrections / 60 > OCTAVE_CORRECTION_CAVEAT_RATE


def test_a_consistently_high_line_is_disclosed_because_the_guard_cannot_see_it() -> None:
    """The failure the octave guard structurally cannot catch.

    A stem whose fundamental was rolled off tracks an octave high on *every*
    frame. The running median agrees with every frame, so the guard correctly
    makes no correction, and `median_cents_offset` is 0.0 because an octave is
    exactly 0 cents. The only honest response is to say the register is
    suspicious for a bass.
    """
    line = segment_notes(_track([(hz_from_midi(33 + 24), 0.5) for _ in range(6)]))

    assert line.octave_corrections == 0
    assert line.median_cents_offset == pytest.approx(0.0, abs=0.01)
    assert (line.median_midi_note or 0) > HIGH_REGISTER_CAVEAT_MIDI
    assert any("octave" in caveat for caveat in line.caveats)


# --------------------------------------------------------------------------- #
# Nothing to track, and nothing invented
# --------------------------------------------------------------------------- #


def test_an_empty_track_is_unvoiced_and_explains_itself() -> None:
    """What every backend returns for digital silence."""
    line = segment_notes(PitchTrack())

    assert line.status == "unvoiced"
    assert line.notes == []
    assert line.voiced_fraction == 0.0
    assert line.caveats and "no frames" in line.caveats[0]


def test_frames_with_no_voiced_flag_set_are_unvoiced_and_explain_themselves() -> None:
    line = segment_notes(_track([(None, 2.0)]))

    assert line.status == "unvoiced"
    assert line.notes == []
    assert line.caveats and "no pitch to track" in line.caveats[0]


def test_sparsely_voiced_noise_invents_nothing() -> None:
    """The real shape of `bass_unvoiced` after a backend's voicing gate.

    Measured on that fixture, the gates leave 0% of frames voiced (librosa) and
    7% (Essentia), scattered rather than in runs. Two independent defences now
    reject it: `MIN_VOICED_FRACTION` catches it first, and the
    `MIN_NOTE_SECONDS` floor would still catch it if the coverage gate were
    removed (pinned separately below). Either way nothing is invented.
    """
    rng = np.random.default_rng(3)
    frames: list[tuple[float | None, float]] = []
    for _ in range(200):
        voiced_here = rng.random() < 0.07
        frames.append((float(rng.uniform(40.0, 200.0)) if voiced_here else None, HOP_SECONDS))

    line = segment_notes(_track(frames))

    assert line.status == "unvoiced"
    assert line.notes == []
    assert 0.0 < (line.voiced_fraction or 0.0) < MIN_VOICED_FRACTION
    assert line.caveats and "floor" in line.caveats[0]


def test_voiced_frames_that_never_settle_emit_nothing_even_above_the_coverage_floor() -> None:
    """The `MIN_NOTE_SECONDS` defence, isolated from the coverage floor.

    Half the frames are voiced, so `MIN_VOICED_FRACTION` is comfortably cleared
    and cannot be what rejects this. Every voiced frame is isolated between two
    unvoiced ones and lands a random interval from its neighbours, so no
    segment is frame-contiguous with another, nothing merges, nothing reaches
    60 ms, and nothing is emitted. Before v5 this case and the sparse-noise
    case above were the same test; they now fail for different reasons and are
    worth pinning separately.
    """
    rng = np.random.default_rng(11)
    frames: list[tuple[float | None, float]] = []
    for _ in range(200):
        frames.append((float(rng.uniform(40.0, 200.0)), HOP_SECONDS))
        frames.append((None, HOP_SECONDS))

    line = segment_notes(_track(frames))

    assert line.status == "unvoiced"
    assert line.notes == []
    assert (line.voiced_fraction or 0.0) > MIN_VOICED_FRACTION
    assert line.caveats and "60 ms" in line.caveats[0]


def test_densely_voiced_random_f0_can_still_produce_short_spurious_notes() -> None:
    """A known limitation, pinned rather than asserted away.

    If a tracker calls *every* frame of unpitched material voiced, the merge
    rule works against us: random neighbours land within `MERGE_MAX_SEMITONES`
    often enough to chain into runs that clear `MIN_NOTE_SECONDS`. Measured
    here: 200 frames of uniform 40-200 Hz noise, all flagged voiced, yield
    around nine notes of 80-130 ms.

    That chaining is not a bug to remove — it is exactly what absorbs a glide
    (see the glide test above), and the two cases are indistinguishable from F0
    alone. **The voicing gate is the defence, not this module**, which is what
    `conftest.bass_unvoiced` says too: "the voicing gate, not the tracker, is
    what has to hold here." Both backends' gates do hold on the real fixture;
    this test exists so that if one ever stops, the consequence is already
    written down.
    """
    rng = np.random.default_rng(3)
    frames = [(float(rng.uniform(40.0, 200.0)), HOP_SECONDS) for _ in range(200)]

    line = segment_notes(_track(frames))

    assert line.status == "ok"
    assert 1 <= len(line.notes) <= 15
    assert all(note.duration_seconds < 0.2 for note in line.notes)


@pytest.mark.parametrize(
    "track",
    [
        PitchTrack(f0_hz=[55.0] * 10, voiced=[True] * 10, frame_hop_seconds=None),
        PitchTrack(f0_hz=[55.0] * 10, voiced=[True] * 10, frame_hop_seconds=0.0),
        PitchTrack(f0_hz=[55.0] * 10, voiced=[True] * 10, frame_hop_seconds=-1.0),
    ],
)
def test_a_track_with_no_usable_hop_fails_loudly_rather_than_guessing(track: PitchTrack) -> None:
    """Without a hop there are no frame times, so every duration would be a guess."""
    line = segment_notes(track)

    assert line.status == "failed"
    assert line.notes == []
    assert line.caveats


def test_zero_hz_frames_never_become_notes_even_when_flagged_voiced() -> None:
    """Backends write 0.0 into unvoiced frames; 0.0 Hz is not a pitch.

    A mismatched `voiced` list would otherwise put `log2(0)` into the MIDI
    conversion and produce a note at negative infinity.
    """
    line = segment_notes(
        PitchTrack(
            f0_hz=[0.0] * 60,
            voiced=[True] * 60,
            voiced_probability=[0.9] * 60,
            frame_hop_seconds=HOP_SECONDS,
        )
    )

    assert line.status == "unvoiced"
    assert line.notes == []


def test_frequencies_outside_the_bass_range_are_still_segmented_if_a_backend_sends_them() -> None:
    """`note_track` does not re-apply the backends' range gate.

    The range is enforced where it belongs — in `pitch()`, where it also acts as
    the octave guard. Silently dropping out-of-range frames here would hide a
    backend bug instead of surfacing it, and would make a hand-built track
    behave differently from a measured one.
    """
    line = segment_notes(_track([(880.0, 0.5)]))

    assert line.status == "ok"
    assert line.notes[0].note_name == "a5"


# --------------------------------------------------------------------------- #
# Grid quantisation
# --------------------------------------------------------------------------- #


def test_notes_carry_a_grid_step_when_the_caller_has_a_grid() -> None:
    """Steps fold into one cycle, so they mean what `DrumPattern.steps` means."""
    segments: list[tuple[float | None, float]] = []
    for index in range(8):
        segments.append((BASS_LINE_FREQS_HZ[index % 4], 0.46))
        segments.append((None, 0.04))
    line = segment_notes(
        _track(segments),
        grid_anchor_seconds=0.0,
        step_seconds=0.125,
        steps_per_cycle=16,
    )

    assert [note.step for note in line.notes] == [0, 4, 8, 12, 0, 4, 8, 12]


def test_start_seconds_stays_the_measurement_and_step_carries_the_quantisation() -> None:
    """Overwriting a measured onset with its grid position destroys the evidence.

    The note below starts 30 ms late. It still quantises to step 0 — and
    `start_seconds` still says 0.03, so a reader can see how far off the grid
    the performance actually was.
    """
    line = segment_notes(
        _track([(None, 0.03), (55.0, 0.5)]),
        grid_anchor_seconds=0.0,
        step_seconds=0.125,
        steps_per_cycle=16,
    )

    assert line.notes[0].step == 0
    assert line.notes[0].start_seconds == pytest.approx(0.03, abs=HOP_SECONDS)


def test_no_grid_means_no_step_rather_than_an_invented_one() -> None:
    """A wrong grid is worse than none — the bias `strudel_hints` already takes."""
    line = segment_notes(_track([(55.0, 0.5)]))

    assert line.notes[0].step is None


def test_a_beat_period_and_downbeat_describe_the_same_grid_as_a_step_length() -> None:
    """The second, equivalent way to state the grid — what `TempoFit` will report.

    W4A measures a refined beat period and a downbeat; W6 hands them here. The
    4/4 convention that turns one into the other (`BEATS_PER_CYCLE` and
    `DEFAULT_STEPS_PER_CYCLE`) lives in this module rather than at each call
    site, so a caller cannot get it subtly wrong in one place and right in
    another. Both spellings must produce identical steps.
    """
    segments: list[tuple[float | None, float]] = []
    for index in range(8):
        segments.append((BASS_LINE_FREQS_HZ[index % 4], 0.46))
        segments.append((None, 0.04))
    track = _track(segments)

    as_steps = segment_notes(track, grid_anchor_seconds=0.0, step_seconds=0.125, steps_per_cycle=16)
    as_beats = segment_notes(track, beat_period_seconds=0.5, downbeat_seconds=0.0)

    assert [note.step for note in as_beats.notes] == [note.step for note in as_steps.notes]
    assert [note.step for note in as_beats.notes] == [0, 4, 8, 12, 0, 4, 8, 12]


def test_a_triplet_cycle_divides_the_same_beat_period_into_twelve() -> None:
    """`steps_per_cycle=12` is a triplet eighth, not a shorter sixteenth.

    Pinned because the conversion is `beat_period * BEATS_PER_CYCLE /
    steps_per_cycle` rather than a hardcoded quarter of a beat, and the whole
    reason for writing it that way is that 12 has to keep working.
    """
    line = segment_notes(
        _track([(55.0, 0.5)]),
        beat_period_seconds=0.6,
        downbeat_seconds=0.0,
        steps_per_cycle=12,
    )

    # 0.6 s per beat, 4 beats per cycle, 12 steps -> 0.2 s per step.
    assert line.notes[0].step == 0
    assert (
        segment_notes(
            _track([(None, 0.2), (55.0, 0.5)]),
            beat_period_seconds=0.6,
            downbeat_seconds=0.0,
            steps_per_cycle=12,
        )
        .notes[0]
        .step
        == 1
    )


def test_an_explicit_step_length_wins_over_a_beat_period() -> None:
    """The more specific statement wins; a measured step is not re-derived."""
    line = segment_notes(
        _track([(None, 0.25), (55.0, 0.5)]),
        grid_anchor_seconds=0.0,
        step_seconds=0.125,
        steps_per_cycle=16,
        beat_period_seconds=2.0,  # would put this note on step 0
        downbeat_seconds=0.0,
    )

    assert line.notes[0].step == 2


@pytest.mark.parametrize("period", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_beat_period_degrades_to_no_grid_rather_than_raising(
    period: float | None,
) -> None:
    """W4A returns low-confidence and nonsense periods on material it cannot fit."""
    line = segment_notes(_track([(55.0, 0.5)]), beat_period_seconds=period)

    assert line.status == "ok"
    assert line.notes[0].step is None


# --------------------------------------------------------------------------- #
# Onset timing — F3, and where the lag actually was
# --------------------------------------------------------------------------- #


def _madonna_track() -> PitchTrack:
    """The committed raw F0 for the Madonna bass stem as a `PitchTrack`.

    No audio: this is one float, one bool and one confidence per 11.6 ms frame,
    an irreversible reduction of the stem. See `fixtures/real/PROVENANCE.md`.
    """
    if not MADONNA_F0_NPZ.exists():  # pragma: no cover - fixture is committed
        pytest.skip(f"missing real-material fixture {MADONNA_F0_NPZ}")
    data = np.load(MADONNA_F0_NPZ, allow_pickle=False)
    return PitchTrack(
        f0_hz=[float(value) for value in data["f0_hz"]],
        voiced=[bool(value) for value in data["voiced"]],
        voiced_probability=[float(value) for value in data["voiced_probability"]],
        frame_hop_seconds=float(data["frame_hop_seconds"]),
        method="yinfft",
    )


def _fractional_steps(line: BassLine, step_seconds: float = MADONNA_STEP_SECONDS) -> np.ndarray:
    """Each onset's position within its step, in [0, 1)."""
    starts = np.array([note.start_seconds for note in line.notes], dtype=np.float64)
    return (starts / step_seconds) % 1.0


def _circular_mean_steps(fractions: np.ndarray) -> tuple[float, float]:
    """`(mean position, concentration R)` of fractional step positions.

    Circular because 0.99 and 0.01 are 0.02 apart, not 0.98. `R` is the length
    of the mean unit vector: 0 for a uniform scatter, 1 for every note landing
    at the same fractional position. **`R` is what separates a real correction
    from a cosmetic one** — subtracting any constant offset leaves `R`
    untouched, so a rise in `R` can only come from per-note variance actually
    being removed.
    """
    angles = 2.0 * np.pi * fractions
    resultant = np.mean(np.exp(1j * angles))
    return float(np.angle(resultant) / (2.0 * np.pi) % 1.0), float(np.abs(resultant))


def _mean_quantisation_error(fractions: np.ndarray) -> float:
    """Mean absolute distance to the nearest step, in steps. No offset removed."""
    return float(np.mean(np.abs(((fractions + 0.5) % 1.0) - 0.5)))


def test_the_real_bass_fixture_quantises_without_a_constant_offset_being_removed() -> None:
    """F3, closed. The whole point of the package, on real material.

    v4 measured 0.2814 steps of mean quantisation error here and needed a 0.282
    step (32 ms) constant subtracted to reach 0.137. That constant was charged
    against the music. Backdating each onset across its own pre-voicing attack
    reaches 0.112 with **nothing subtracted**, which is what this asserts.

    The bounds are loose on purpose — they are a regression fence around a
    measured 0.1116, not a re-statement of it to four decimal places, so an
    unrelated change that moves the number slightly does not fail here while a
    return of the lag does.
    """
    line = segment_notes(_madonna_track())
    fractions = _fractional_steps(line)

    assert line.status == "ok"
    assert len(line.notes) == 709
    assert _mean_quantisation_error(fractions) < 0.15


def test_the_real_bass_fixtures_onsets_no_longer_sit_a_third_of_a_step_late() -> None:
    """Task 2: the circular mean must be near zero, not 0.28.

    Measured: 0.2916 steps before, 0.0749 after. The residual is 8.5 ms, under
    the 11.6 ms frame hop — it is the frame grid, not a remaining lag, and
    there is nothing left in the F0 track to recover it from.
    """
    line = segment_notes(_madonna_track())
    centre, _ = _circular_mean_steps(_fractional_steps(line))

    assert centre < 0.12 or centre > 0.88, f"onsets still offset by {centre:.4f} steps"
    assert min(centre, 1.0 - centre) * MADONNA_STEP_SECONDS < HOP_SECONDS


def test_backdating_tightens_the_onset_distribution_which_an_offset_cannot() -> None:
    """The evidence that this is a correction and not a fudge.

    Subtracting a constant — the thing F3 did to prove the lag existed, and the
    thing this package was told not to ship — moves the circular *mean* and
    leaves the concentration `R` exactly where it was. Measured here, `R` rises
    from 0.548 to 0.724: real per-note variance was removed, because the amount
    each note was late by was measured for that note rather than assumed.
    """
    _, concentration = _circular_mean_steps(_fractional_steps(segment_notes(_madonna_track())))

    assert concentration > 0.65


def test_the_real_bass_sits_on_the_offbeat_eighths_of_the_verified_grid() -> None:
    """F7's third point: 103 notes on each of steps 2, 6, 10 and 14.

    Steps are folded into a 16-step cycle anchored on F1's measured 0.228 s
    downbeat, so step 2 is the second sixteenth of the bar — the offbeat
    eighth. This is what `BassNote.step` exists to say, and what was `null` on
    every note of the v4 output because the grid it needed did not survive the
    tempo error F1 describes.
    """
    line = segment_notes(
        _madonna_track(),
        step_seconds=MADONNA_STEP_SECONDS,
        steps_per_cycle=16,
        grid_anchor_seconds=MADONNA_DOWNBEAT_SECONDS,
    )
    counts = np.bincount([note.step for note in line.notes if note.step is not None], minlength=16)

    assert [int(counts[step]) for step in (2, 6, 10, 14)] == [103, 103, 103, 103]
    assert counts[2] > counts.sum() / 16 * 2, "the offbeat eighths must dominate, not merely appear"


def test_the_median_filter_width_does_not_move_onsets_at_all() -> None:
    """F3 blamed the median filter. It is not guilty, and this is the proof.

    A centred median of odd width `W` has **zero group delay at a step edge**:
    the new value takes the median as soon as `(W + 1) / 2` of the window is
    past the edge, which first happens with the window centred exactly on the
    edge. So `(W - 1) / 2 * hop` — the "obvious" derived latency, 23 ms at the
    default width — would have been a pure over-correction.

    The width is swept for real rather than reasoned about: a note preceded by
    silence, and a note preceded by a different pitch, both start on their true
    frame at every width from 1 to 11. On the real fixture the same sweep moves
    the measured offset by 0.0006 steps.
    """
    onset_frames = 30
    onset = onset_frames * HOP_SECONDS
    for width in (1, 3, 5, 7, 9, 11):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(note_track, "MEDIAN_FILTER_FRAMES", width)
            after_silence = segment_notes(_track([(None, onset), (55.0, 0.5)]))
            after_a_note = segment_notes(_track([(hz_from_midi(45), onset), (55.0, 0.5)]))

        assert after_silence.notes[0].start_seconds == pytest.approx(onset, abs=1e-9), (
            f"width {width}"
        )
        assert after_a_note.notes[-1].start_seconds == pytest.approx(onset, abs=1e-9), (
            f"width {width}"
        )
    assert MEDIAN_FILTER_FRAMES in (1, 3, 5, 7, 9, 11), "the swept range must cover the default"


def _attack_ramp_track(
    *,
    ramp_f0: list[float],
    ramp_probability: list[float],
    settled_hz: float = 55.0,
    settled_frames: int = 40,
    lead_frames: int = 20,
) -> PitchTrack:
    """A hand-built track shaped like a real one at a note onset.

    Backends write an F0 for every frame, including the ones their own voicing
    gate rejects — `_track()` above writes 0.0 there instead, because it models
    rests rather than attacks. This models the attack: `ramp_f0` frames carry a
    pitch the gate has not yet accepted, and `ramp_probability` is the
    confidence climbing toward it.
    """
    f0 = [0.0] * lead_frames + list(ramp_f0) + [settled_hz] * settled_frames
    voiced = [False] * (lead_frames + len(ramp_f0)) + [True] * settled_frames
    probability = [0.0] * lead_frames + list(ramp_probability) + [0.9] * settled_frames
    return PitchTrack(
        f0_hz=f0,
        voiced=voiced,
        voiced_probability=probability,
        frame_hop_seconds=HOP_SECONDS,
        method="hand-built",
    )


def test_an_onset_is_backdated_across_the_frames_the_voicing_gate_rejected() -> None:
    """The mechanism, on the smallest track that shows it.

    Three frames already at 55 Hz with confidence climbing 0.43 -> 0.60 -> 0.69
    sit in front of the frame the gate accepts. That shape is taken from the
    fixture (frames 2997-3000 of the Madonna bass). The note starts at the
    first of them, not the fourth.
    """
    line = segment_notes(
        _attack_ramp_track(ramp_f0=[54.87, 54.86, 54.88], ramp_probability=[0.43, 0.60, 0.69])
    )

    assert len(line.notes) == 1
    assert line.notes[0].start_seconds == pytest.approx(20 * HOP_SECONDS, abs=1e-9)
    assert line.notes[0].duration_seconds == pytest.approx(43 * HOP_SECONDS, abs=1e-9)


def test_backdating_leaves_the_reported_hz_a_measurement_over_confident_frames() -> None:
    """The recovered frames move the *time*, never the pitch or the confidence.

    `median_f0_hz`, `cents_offset` and `confidence` are computed over frames the
    backend called voiced. A pre-voicing frame reading a wild 41 Hz must not
    drag the reported pitch down with it — it is evidence about *when*, not
    about *what*.
    """
    line = segment_notes(
        _attack_ramp_track(ramp_f0=[54.0, 55.4, 55.0], ramp_probability=[0.4, 0.5, 0.6])
    )

    assert line.notes[0].median_f0_hz == pytest.approx(55.0)
    assert line.notes[0].confidence == pytest.approx(0.9)


def test_backdating_stops_where_the_pre_voicing_pitch_stops_agreeing() -> None:
    """Two frames of the note's own pitch behind two frames of something else."""
    line = segment_notes(
        _attack_ramp_track(
            ramp_f0=[hz_from_midi(45), hz_from_midi(45), 55.0, 55.0],
            ramp_probability=[0.3, 0.4, 0.5, 0.6],
        )
    )

    assert line.notes[0].start_seconds == pytest.approx(22 * HOP_SECONDS, abs=1e-9)


def test_backdating_stops_where_the_confidence_stops_climbing() -> None:
    """Confidence rising toward the gate is an attack; confidence falling is not.

    The middle frame is *more* confident than the one in front of it, so the
    walk stops there: what is behind it is the tail of something else, not this
    note's attack.
    """
    line = segment_notes(
        _attack_ramp_track(ramp_f0=[55.0, 55.0, 55.0], ramp_probability=[0.6, 0.8, 0.7])
    )

    assert line.notes[0].start_seconds == pytest.approx(22 * HOP_SECONDS, abs=1e-9)


def test_an_onset_is_never_backdated_further_than_the_shortest_note() -> None:
    """`ONSET_BACKDATE_MAX_SECONDS`: a correction longer than a note is not one.

    Twenty frames of a perfectly agreeing, perfectly ramping pre-voicing pitch —
    far more than any real attack — still moves the onset by at most
    `int(ONSET_BACKDATE_MAX_SECONDS / hop)` frames.
    """
    ramp = 20
    line = segment_notes(
        _attack_ramp_track(
            ramp_f0=[55.0] * ramp,
            ramp_probability=[0.3 + 0.02 * index for index in range(ramp)],
            lead_frames=10,
        )
    )

    bound = int(ONSET_BACKDATE_MAX_SECONDS / HOP_SECONDS)
    moved = (10 + ramp) - line.notes[0].start_seconds / HOP_SECONDS
    assert 0 < moved <= bound
    assert line.notes[0].start_seconds > 10 * HOP_SECONDS


def test_backdating_never_reaches_into_the_note_in_front_of_it() -> None:
    """Notes may not overlap, and a semitone neighbour is the tempting case.

    Two `MERGE_MAX_SEMITONES`-apart notes back to back with no rest between
    them: the second is within tolerance of the first at every frame, so only
    the "do not cross the previous note" rule stops the walk.
    """
    line = segment_notes(
        _track(
            [
                (hz_from_midi(33), 0.4),
                (hz_from_midi(33 + MERGE_MAX_SEMITONES), 0.4),
            ]
        )
    )

    assert len(line.notes) == 2
    first, second = line.notes
    assert second.start_seconds >= first.start_seconds + first.duration_seconds


def test_backdating_is_a_no_op_on_a_track_with_no_pre_voicing_frames() -> None:
    """Hand-built tracks write 0.0 Hz into rests, so there is nothing to recover.

    Load-bearing for every other test in this file: they were all written
    against onsets that are exact by construction, and this correction must not
    quietly move any of them.
    """
    onset = 30 * HOP_SECONDS
    line = segment_notes(_track([(None, onset), (55.0, 0.5)]))

    assert line.notes[0].start_seconds == pytest.approx(onset, abs=1e-9)


# --------------------------------------------------------------------------- #
# Voiced-fraction gating — F5's half, and F6
# --------------------------------------------------------------------------- #


def _coverage_track(fraction: float, frames: int = 600) -> PitchTrack:
    """A steady 55 Hz note occupying `fraction` of the track, the rest silent.

    One long voiced run rather than a scatter, so `MIN_NOTE_SECONDS` cannot be
    what rejects it and the coverage gate is the only thing under test.
    """
    voiced_frames = int(round(frames * fraction))
    return _track(
        [(55.0, voiced_frames * HOP_SECONDS), (None, (frames - voiced_frames) * HOP_SECONDS)]
    )


@pytest.mark.parametrize("fraction", [0.012, 0.089])
def test_a_stem_left_at_the_noise_floor_reports_unvoiced_rather_than_notes(
    fraction: float,
) -> None:
    """F5's first half, at the two coverage values real silent stems produced.

    0.012 and 0.089 are `showers-of-gold` and `ancient-heavy-tech-donjon` in
    `calibration/v4/`: stems Demucs left at about -82 dBFS, which v4 reported
    as `status: "ok"` with a median of `e4`. A pitch tracked from a noise floor
    is not a bass, and the caveat it earned — "check the octave by ear" — sent
    the reader off to verify something that does not exist.
    """
    line = segment_notes(_coverage_track(fraction))

    assert line.status == "unvoiced"
    assert line.notes == []
    assert line.median_midi_note is None
    assert (line.voiced_fraction or 0.0) == pytest.approx(fraction, abs=0.01)
    assert line.caveats and "floor" in line.caveats[0]
    assert not any("by ear" in caveat for caveat in line.caveats)


def test_the_real_bass_stem_clears_the_coverage_floor_by_a_wide_margin() -> None:
    """The floor must not be anywhere near the one real bass in the corpus.

    Measured 0.4535 against a 0.15 floor — a factor of three. The floor is
    placed in the gap between that and the 0.089 of the worst silent stem, not
    tuned against either end of it.
    """
    line = segment_notes(_madonna_track())

    assert line.status == "ok"
    assert (line.voiced_fraction or 0.0) == pytest.approx(0.4535, abs=0.001)
    assert (line.voiced_fraction or 0.0) > MIN_VOICED_FRACTION * 2


def test_a_bass_that_rests_half_the_track_earns_no_coverage_caveat() -> None:
    """F6. 45% voiced is what a bass line with rests looks like, not a defect.

    The v4 caveat — "Only 45% of frames were voiced, so this line is built from
    less than half the stem" — fired on the one correct result in the corpus,
    against 709 notes that quantise to the grid at 0.11 steps. Noise in the
    caveat list trains the reader to skip caveats, which costs more than this
    caveat was ever worth.
    """
    line = segment_notes(_madonna_track())

    assert line.caveats == []


def test_the_coverage_caveat_still_fires_between_the_floor_and_the_caveat_threshold() -> None:
    """The 0.15-0.30 window is the only place it can fire, and that is intended.

    Below `MIN_VOICED_FRACTION` there is no line to attach a caveat to; above
    `LOW_VOICING_CAVEAT_FRACTION` there is nothing to warn about. Pinned so
    that a future change which widens either threshold to "make the caveat
    reachable" has to argue with a test first.
    """
    midpoint = (MIN_VOICED_FRACTION + LOW_VOICING_CAVEAT_FRACTION) / 2
    line = segment_notes(_coverage_track(midpoint))

    assert line.status == "ok"
    assert line.notes
    assert any("cover" in caveat for caveat in line.caveats)


def test_the_coverage_caveat_describes_coverage_rather_than_implying_failure() -> None:
    """F6's wording half. "Expect missed notes" reads as a broken measurement."""
    line = segment_notes(_coverage_track(0.2))
    caveat = next(text for text in line.caveats if "cover" in text)

    assert "less than half" not in caveat
    assert "rests" in caveat
    assert "not by itself a fault" in caveat


# --------------------------------------------------------------------------- #
# Purity: the property the whole seam rests on
# --------------------------------------------------------------------------- #


def test_segment_notes_is_a_pure_function_of_the_track() -> None:
    """Same F0 in, byte-identical `BassLine` out — that is the seam's guarantee.

    If this ever stops holding (hidden state, a time-dependent default, a
    mutated input), the claim that both backends produce identical notes from
    identical F0 quietly becomes false.
    """
    track = _track([(BASS_LINE_FREQS_HZ[index % 4], 0.46) for index in range(8)])
    before = track.model_dump()

    first = segment_notes(track)
    second = segment_notes(track)

    assert first.model_dump() == second.model_dump()
    assert track.model_dump() == before, "segment_notes must not mutate its input"


def test_a_pitch_track_from_either_backend_shape_produces_the_same_line() -> None:
    """The two backends differ in `method` and in confidence scale, not in notes.

    Two tracks carrying the same F0 and voicing but each backend's own metadata
    must segment identically — nothing downstream of `pitch()` may branch on
    which library produced the numbers.
    """
    frequencies = [(BASS_LINE_FREQS_HZ[index % 4], 0.46) for index in range(8)]
    as_pyin = _track(frequencies, probability=0.43)  # pYIN's measured median
    as_yinfft = _track(frequencies, probability=0.77)  # YinFFT's measured median
    as_pyin.method, as_yinfft.method = "pyin", "yinfft"

    left, right = segment_notes(as_pyin), segment_notes(as_yinfft)

    assert [note.note_name for note in left.notes] == [note.note_name for note in right.notes]
    assert [note.start_seconds for note in left.notes] == [
        note.start_seconds for note in right.notes
    ]


def test_segment_notes_returns_a_bass_line_and_never_a_pitch_track() -> None:
    """`PitchTrack` is transport only: ~25,800 floats for a five-minute track.

    `tests/test_schemas_summary.py` pins that no field of `SourceAnalysis` can
    reach it. This pins the other end — the function that consumes one hands
    back the model that *is* written.
    """
    line = segment_notes(_track([(55.0, 0.5)]))

    assert isinstance(line, BassLine)
    assert not isinstance(line, PitchTrack)
