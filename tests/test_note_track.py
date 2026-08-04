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

from audio_pipeline import ANALYSIS_SAMPLE_RATE
from audio_pipeline.note_track import (
    A4_HZ,
    A4_MIDI,
    HIGH_REGISTER_CAVEAT_MIDI,
    MERGE_MAX_SEMITONES,
    MIN_NOTE_SECONDS,
    NOTE_NAMES,
    OCTAVE_CORRECTION_CAVEAT_RATE,
    TUNING_OFFSET_CAVEAT_CENTS,
    hz_from_midi,
    midi_from_hz,
    note_name,
    segment_notes,
)
from audio_pipeline.schemas import BassLine, PitchTrack

#: The hop both backends produce, so hand-built tracks share their time grid.
HOP_SECONDS = 512 / ANALYSIS_SAMPLE_RATE


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
    7% (Essentia), scattered rather than in runs. Nothing survives
    `MIN_NOTE_SECONDS`, so nothing is emitted and the status says why.
    """
    rng = np.random.default_rng(3)
    frames: list[tuple[float | None, float]] = []
    for _ in range(200):
        voiced_here = rng.random() < 0.07
        frames.append((float(rng.uniform(40.0, 200.0)) if voiced_here else None, HOP_SECONDS))

    line = segment_notes(_track(frames))

    assert line.status == "unvoiced"
    assert line.notes == []
    assert 0.0 < (line.voiced_fraction or 0.0) < 0.2
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
