"""Tests for `strudel_hints`: pure functions over hand-built TrackSummary objects.

No audio, no fixtures from disk, no backend. Every summary here is constructed
inline so the expected output can be reasoned about from the numbers.

The single most important test in this module is
`test_ambiguous_iois_return_none` — a wrong grid is worse than no grid, so the
inference must stay silent on material it cannot read.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Sequence

import pytest

from audio_pipeline import SCHEMA_VERSION, strudel_vocab
from audio_pipeline.heuristics import THRESHOLDS
from audio_pipeline.schemas import (
    BassLine,
    BassNote,
    DrumDecomposition,
    DrumHit,
    DrumPattern,
    RhythmFeatures,
    SourceAnalysis,
    StrudelHints,
    TonalFeatures,
    TrackSummary,
)
from audio_pipeline.strudel_hints import (
    BASS_LINE_NOTE_SEQUENCE_CAP,
    BEATS_PER_CYCLE,
    DENSITY_TERMS,
    SUBDIVISION_TERMS,
    build,
    classify_density,
    infer_subdivision_feel,
    suggest_cycle_seconds,
)

SPARSE = THRESHOLDS["sparse_onsets_per_sec"]
BUSY = THRESHOLDS["busy_drums_onsets_per_sec"]


# --- builders ---------------------------------------------------------------


def times_from_iois(iois: Iterable[float], start: float = 0.0) -> list[float]:
    """Turn a sequence of intervals into the event times that produce them."""
    times = [start]
    for gap in iois:
        times.append(times[-1] + gap)
    return times


def make_source(
    name: str,
    *,
    bpm: float | None = None,
    onset_times: Sequence[float] | None = None,
    beat_times: Sequence[float] | None = None,
    onset_density: float | None = None,
    key: str | None = None,
    scale: str | None = None,
    key_confidence: float | None = None,
    drum_decomposition: DrumDecomposition | None = None,
    bass_line: BassLine | None = None,
) -> SourceAnalysis:
    """A SourceAnalysis with only the descriptors a given test cares about."""
    return SourceAnalysis(
        source=name,
        audio_path=f"/tmp/{name}.wav",
        duration_seconds=30.0,
        sample_rate=44100,
        backend="fake",
        rhythm=RhythmFeatures(
            bpm=bpm,
            beat_times=list(beat_times or []),
            onset_times=list(onset_times or []),
            onset_density=onset_density,
        ),
        tonal=TonalFeatures(key=key, scale=scale, key_confidence=key_confidence),
        drum_decomposition=drum_decomposition or DrumDecomposition(),
        bass_line=bass_line or BassLine(),
    )


def make_drum_decomposition(*, unclassified_count: int = 0) -> DrumDecomposition:
    """A small, self-consistent `status="ok"` `DrumDecomposition`.

    One kick on step 0, one hat on step 2, one cycle of a 16-step grid --
    enough for both `_drum_grid_hint` (which reads `patterns`) and
    `strudel_vocab.suggest_drum_sounds` (which reads `hits`) to have real
    material to work from.
    """
    hits = [
        DrumHit(
            time_seconds=0.0,
            drum="kick",
            confidence=0.9,
            step=0,
            kick_ratio=0.95,
            body_ratio=0.02,
            noise_ratio=0.02,
            air_ratio=0.01,
            decay_ratio=2.0,
            flatness=1e-5,
        ),
        DrumHit(
            time_seconds=0.25,
            drum="hat",
            confidence=0.8,
            step=2,
            kick_ratio=0.0,
            body_ratio=0.0,
            noise_ratio=0.01,
            air_ratio=0.99,
            decay_ratio=10.0,
            flatness=0.04,
        ),
    ]
    return DrumDecomposition(
        status="ok",
        steps_per_cycle=16,
        cycle_seconds=2.0,
        grid_anchor_seconds=0.0,
        grid_anchor_source="beats",
        quantisation_error_steps=0.02,
        patterns=[
            DrumPattern(drum="kick", steps=[0], step_occupancy=[1.0], hit_count=1),
            DrumPattern(drum="hat", steps=[2], step_occupancy=[1.0], hit_count=1),
        ],
        hits=hits,
        unclassified_count=unclassified_count,
    )


def make_bass_line(note_count: int = 1) -> BassLine:
    """A small, self-consistent `status="ok"` `BassLine` of held A1s."""
    notes = [
        BassNote(
            start_seconds=index * 0.5,
            duration_seconds=0.46,
            midi_note=33,
            note_name="a1",
            median_f0_hz=55.0,
            cents_offset=0.0,
            confidence=0.9,
            step=(4 * index) % 16,
        )
        for index in range(note_count)
    ]
    return BassLine(
        status="ok",
        notes=notes,
        median_midi_note=33,
        median_cents_offset=0.0,
        voiced_fraction=0.9,
        octave_corrections=0,
    )


def jitter(iois: Sequence[float], amount: float, seed: int) -> list[float]:
    """Scatter each interval by up to +/-`amount`, as human timing would."""
    rng = random.Random(seed)
    return [d * (1.0 + rng.uniform(-amount, amount)) for d in iois]


def make_summary(*sources: SourceAnalysis, track_name: str = "test-track") -> TrackSummary:
    return TrackSummary(
        track_name=track_name,
        input_path="/tmp/test-track.wav",
        duration_seconds=30.0,
        backend="fake",
        sources={source.source: source for source in sources},
    )


def drums_with_iois(iois: Sequence[float], bpm: float | None = 120.0) -> TrackSummary:
    """A summary whose only source is a drums stem with the given onset intervals."""
    return make_summary(make_source("drums", bpm=bpm, onset_times=times_from_iois(iois)))


def repeat_to(pattern: Sequence[float], count: int) -> list[float]:
    return list(itertools.islice(itertools.cycle(pattern), count))


# --- suggest_cycle_seconds --------------------------------------------------


def test_120bpm_four_beats_is_two_seconds() -> None:
    assert suggest_cycle_seconds(120.0, 4) == 2.0


def test_default_beats_per_cycle_is_four_beats() -> None:
    assert BEATS_PER_CYCLE == 4
    assert suggest_cycle_seconds(120.0) == suggest_cycle_seconds(120.0, 4)


@pytest.mark.parametrize(
    ("bpm", "beats", "expected"),
    [
        (60.0, 4, 4.0),
        (120.0, 8, 4.0),
        (140.0, 4, round(4 * 60.0 / 140.0, 6)),
        (90.0, 3, 2.0),
    ],
)
def test_cycle_arithmetic(bpm: float, beats: int, expected: float) -> None:
    assert suggest_cycle_seconds(bpm, beats) == pytest.approx(expected)


@pytest.mark.parametrize("bpm", [None, 0.0, -120.0, math.nan, math.inf])
def test_unusable_bpm_returns_none_without_dividing_by_zero(bpm: float | None) -> None:
    assert suggest_cycle_seconds(bpm) is None


@pytest.mark.parametrize("beats", [0, -4])
def test_non_positive_beats_per_cycle_returns_none(beats: int) -> None:
    assert suggest_cycle_seconds(120.0, beats) is None


# --- classify_density -------------------------------------------------------
#
# Convention under test: bands are half-open with the lower bound inclusive.
#   [0, SPARSE) -> sparse ; [SPARSE, BUSY) -> moderate ; [BUSY, inf) -> busy


def test_density_below_sparse_threshold_is_sparse() -> None:
    assert classify_density(SPARSE - 0.01) == "sparse"
    assert classify_density(0.0) == "sparse"


def test_density_exactly_on_sparse_threshold_is_moderate() -> None:
    assert classify_density(SPARSE) == "moderate"


def test_density_between_thresholds_is_moderate() -> None:
    assert classify_density((SPARSE + BUSY) / 2.0) == "moderate"
    assert classify_density(BUSY - 0.01) == "moderate"


def test_density_exactly_on_busy_threshold_is_busy() -> None:
    assert classify_density(BUSY) == "busy"


def test_density_above_busy_threshold_is_busy() -> None:
    assert classify_density(BUSY + 5.0) == "busy"


@pytest.mark.parametrize("value", [None, -1.0, math.nan, math.inf])
def test_unmeasurable_density_returns_none(value: float | None) -> None:
    assert classify_density(value) is None


# --- infer_subdivision_feel: the confident cases ----------------------------


def test_clean_straight_16ths() -> None:
    # 120 BPM -> 0.5 s beat -> 0.125 s sixteenth.
    summary = drums_with_iois([0.125] * 32)
    assert infer_subdivision_feel(summary) == "straight 16ths"


def test_clean_straight_8ths() -> None:
    summary = drums_with_iois([0.25] * 32)
    assert infer_subdivision_feel(summary) == "straight 8ths"


def test_clean_swung_8ths() -> None:
    # Triplet swing at 120 BPM: long 2/3 beat, short 1/3 beat, pair = 1 beat.
    beat = 0.5
    summary = drums_with_iois(repeat_to([beat * 2 / 3, beat / 3], 32))
    assert infer_subdivision_feel(summary) == "swung 8ths"


def test_swung_8ths_at_the_conftest_fixture_timings() -> None:
    """W0's `swung_click_8ths` ground truth: 0.4/0.2 s onsets against 100 BPM.

    Uses the fixture's documented numbers rather than its audio, since this
    module never touches a backend — but it must agree with what a backend
    reading that fixture would hand over.
    """
    summary = drums_with_iois(repeat_to([0.4, 0.2], 26), bpm=100.0)
    assert infer_subdivision_feel(summary) == "swung 8ths"


def test_swung_16ths_pair_names_the_finer_grid() -> None:
    # Pair sums to half a beat -> swung 16ths rather than swung 8ths.
    beat = 0.5
    summary = drums_with_iois(repeat_to([beat / 3, beat / 6], 32))
    assert infer_subdivision_feel(summary) == "swung 16ths"


# --- infer_subdivision_feel: real-onset conditions --------------------------
#
# Observed onsets are never a clean click train. These exercise the three ways
# real data differs: performance jitter, undetected hits, and spurious extras.


def test_jittered_but_genuinely_straight_16ths_is_found() -> None:
    """+/-10% human timing on every onset must not hide an even grid."""
    iois = jitter([0.125] * 48, amount=0.10, seed=7)
    assert infer_subdivision_feel(drums_with_iois(iois)) == "straight 16ths"


def test_jittered_but_genuinely_swung_8ths_is_found() -> None:
    """+/-8% timing on a real swing feel must still read as swung, not straight."""
    beat = 0.5
    iois = jitter(repeat_to([beat * 2 / 3, beat / 3], 40), amount=0.08, seed=11)
    assert infer_subdivision_feel(drums_with_iois(iois)) == "swung 8ths"


def test_undetected_hits_do_not_break_a_straight_grid() -> None:
    """A missed onset merges two 16ths into one 0.25 s interval.

    That interval is a whole multiple of the unit, so it is evidence for the
    grid rather than against it.
    """
    iois = repeat_to([0.125, 0.125, 0.125, 0.25], 40)  # one hit in five undetected
    assert infer_subdivision_feel(drums_with_iois(iois)) == "straight 16ths"


def test_undetected_hits_and_jitter_together_still_read_straight() -> None:
    iois = jitter(repeat_to([0.125, 0.125, 0.25], 45), amount=0.08, seed=5)
    assert infer_subdivision_feel(drums_with_iois(iois)) == "straight 16ths"


def test_a_few_spurious_onsets_do_not_break_a_straight_grid() -> None:
    """Bleed and flams add onsets at no grid position at all."""
    rng = random.Random(19)
    iois = [0.125] * 40
    for _ in range(4):  # split 4 intervals into off-grid pairs
        index = rng.randrange(len(iois))
        offset = rng.uniform(0.03, 0.06)
        iois[index : index + 1] = [offset, 0.125 - offset]
    assert infer_subdivision_feel(drums_with_iois(iois)) == "straight 16ths"


def test_one_stray_onset_does_not_veto_a_clean_swing() -> None:
    """The cluster tightness test must not be decided by a single outlier."""
    beat = 0.5
    iois = repeat_to([beat * 2 / 3, beat / 3], 40)
    iois[17] = beat * 0.95  # one badly mis-detected long
    assert infer_subdivision_feel(drums_with_iois(iois)) == "swung 8ths"


# --- infer_subdivision_feel: the silences -----------------------------------


def test_ambiguous_iois_return_none() -> None:
    """The most important test here: random intervals must not become a grid.

    Uniformly random IOIs in [0.1, 0.2] s have a real, dense distribution — this
    is not the trivially-empty case. It is neither even enough to be straight
    nor bimodal-and-alternating enough to be swung, so the answer is silence.
    """
    rng = random.Random(42)
    iois = [rng.uniform(0.1, 0.2) for _ in range(64)]
    assert infer_subdivision_feel(drums_with_iois(iois)) is None


def test_near_uniform_but_badly_smeared_iois_return_none() -> None:
    """A grid smeared by +/-40% is no longer readable. Say nothing.

    The tolerance for real onsets is deliberately wide enough to absorb +/-10%
    (see `test_jittered_but_genuinely_straight_16ths_is_found`); this is the
    other side of that line, where nothing trustworthy is left to report.
    """
    iois = jitter([0.125] * 64, amount=0.40, seed=3)
    assert infer_subdivision_feel(drums_with_iois(iois)) is None


def test_bimodal_but_not_alternating_returns_none() -> None:
    """Two tight clusters at 2:1 that do not interleave are not swing."""
    beat = 0.5
    iois = [beat * 2 / 3] * 16 + [beat / 3] * 16
    assert infer_subdivision_feel(drums_with_iois(iois)) is None


def test_alternating_but_wrong_ratio_returns_none() -> None:
    """A 1.25:1 lilt is neither straight nor swung, so it gets no label.

    It sits inside STRAIGHT_TOLERANCE of the median, so only the explicit
    anti-lilt veto stops it being reported as dead straight.
    """
    iois = repeat_to([0.125 * 1.25, 0.125], 32)
    summary = drums_with_iois(iois)
    assert infer_subdivision_feel(summary) is None
    assert any("neither an even nor a swung grid" in note for note in build(summary).notes)


def test_a_ratio_below_the_lilt_veto_still_reads_as_straight() -> None:
    """The veto must not fire on ordinary timing wobble: 1.05:1 stays straight."""
    iois = repeat_to([0.125 * 1.05, 0.125], 32)
    assert infer_subdivision_feel(drums_with_iois(iois)) == "straight 16ths"


def test_even_grid_that_only_resolves_to_the_beat_returns_none() -> None:
    """One event per beat carries no subdivision information at all."""
    summary = drums_with_iois([0.5] * 32)
    assert infer_subdivision_feel(summary) is None


def test_even_grid_off_the_tempo_returns_none() -> None:
    """An even grid that lines up with no subdivision of the stated tempo."""
    summary = drums_with_iois([0.19] * 32)
    assert infer_subdivision_feel(summary) is None


def test_too_few_onsets_return_none_with_a_note() -> None:
    summary = drums_with_iois([0.125] * 5)
    assert infer_subdivision_feel(summary) is None
    assert any("too few drum onsets" in note for note in build(summary).notes)


def test_summary_reloaded_from_json_has_no_onset_times() -> None:
    """summary_payload() strips both event lists, so a round-trip yields None."""
    summary = drums_with_iois([0.125] * 32)
    assert infer_subdivision_feel(summary) == "straight 16ths"

    sources = summary.summary_payload()["sources"]
    assert isinstance(sources, dict)
    rhythm = sources["drums"]["rhythm"]
    assert "onset_times" not in rhythm and "beat_times" not in rhythm
    assert rhythm["onset_count"] == 33

    # Reading that payload back gives a summary with both lists empty, which is
    # the shape `export-strudel-hints` will be handed.
    reloaded = make_summary(make_source("drums", bpm=120.0))
    assert infer_subdivision_feel(reloaded) is None
    assert any("no onset times" in note for note in build(reloaded).notes)


def test_beat_times_alone_are_refused_with_an_explanatory_note() -> None:
    """The evenly-spaced pulse cannot show swing, so it is not a substitute."""
    summary = make_summary(
        make_source("drums", bpm=120.0, beat_times=times_from_iois([0.5] * 32))
    )
    assert infer_subdivision_feel(summary) is None
    assert any("cannot show swing" in note for note in build(summary).notes)


def test_no_event_times_at_all_returns_none() -> None:
    summary = make_summary(make_source("drums", bpm=120.0, onset_density=6.0))
    assert infer_subdivision_feel(summary) is None


def test_missing_drums_stem_returns_none_with_a_note() -> None:
    summary = make_summary(make_source("mix", bpm=120.0))
    assert infer_subdivision_feel(summary) is None
    assert any("no drums stem" in note for note in build(summary).notes)


def test_unknown_tempo_returns_none_with_a_note() -> None:
    summary = drums_with_iois([0.125] * 32, bpm=None)
    assert infer_subdivision_feel(summary) is None
    assert any("tempo unknown" in note for note in build(summary).notes)


def test_grid_measured_against_mix_tempo_when_drums_bpm_missing() -> None:
    drums = make_source("drums", bpm=None, onset_times=times_from_iois([0.125] * 32))
    summary = make_summary(make_source("mix", bpm=120.0), drums)
    assert infer_subdivision_feel(summary) == "straight 16ths"


def test_section_gap_does_not_break_a_straight_grid() -> None:
    """One long gap between sections is an outlier, not a grid event."""
    iois = [0.125] * 16 + [4.0] + [0.125] * 16
    assert infer_subdivision_feel(drums_with_iois(iois)) == "straight 16ths"


# --- build ------------------------------------------------------------------


def full_summary() -> TrackSummary:
    return make_summary(
        make_source("mix", bpm=120.0, key="A", scale="minor", key_confidence=0.82),
        make_source(
            "drums",
            bpm=120.0,
            onset_times=times_from_iois([0.125] * 32),
            onset_density=BUSY + 1.0,
            drum_decomposition=make_drum_decomposition(),
        ),
        make_source(
            "bass",
            onset_density=SPARSE / 2.0,
            key="A",
            scale="minor",
            key_confidence=0.9,
            bass_line=make_bass_line(),
        ),
        make_source("vocals", onset_density=(SPARSE + BUSY) / 2.0),
    )


def test_build_populates_every_field_when_the_data_is_there() -> None:
    hints = build(full_summary())
    assert hints.track_name == "test-track"
    assert hints.bpm == 120.0
    assert hints.suggested_cycle_seconds == 2.0
    assert hints.subdivision_feel == "straight 16ths"
    assert hints.drum_density == "busy"
    assert hints.bass_activity == "sparse"
    assert hints.tonal_centre == "A minor"
    assert hints.notes == []
    assert hints.schema_version == SCHEMA_VERSION

    # Wave 4 fields.
    assert hints.drum_grid.status == "ok"
    assert hints.drum_grid.steps_per_cycle == 16
    assert hints.drum_grid.kick_steps == [0]
    assert hints.drum_grid.hat_steps == [2]
    assert hints.drum_grid.snare_steps == []

    assert hints.bass_line.status == "ok"
    assert hints.bass_line.note_sequence == ["a1"]
    assert hints.bass_line.truncated_from is None
    assert hints.bass_line.median_midi_note == 33

    roles = {suggestion.role for suggestion in hints.sound_suggestions}
    assert roles == {"kick", "hat", "bass"}

    assert hints.strudel_vocabulary_read == strudel_vocab.STRUDEL_DOCS_READ


def test_build_returns_a_strudel_hints_model() -> None:
    assert isinstance(build(full_summary()), StrudelHints)


def test_empty_sources_produce_valid_empty_hints_without_raising() -> None:
    summary = make_summary()
    hints = build(summary)
    assert hints.track_name == "test-track"
    assert hints.bpm is None
    assert hints.suggested_cycle_seconds is None
    assert hints.subdivision_feel is None
    assert hints.drum_density is None
    assert hints.bass_activity is None
    assert hints.tonal_centre is None
    assert any("no analysed sources" in note for note in hints.notes)


def test_all_none_descriptors_degrade_to_none_fields() -> None:
    summary = make_summary(
        make_source("mix"),
        make_source("drums"),
        make_source("bass"),
    )
    hints = build(summary)
    assert hints.bpm is None
    assert hints.suggested_cycle_seconds is None
    assert hints.subdivision_feel is None
    assert hints.drum_density is None
    assert hints.bass_activity is None
    assert hints.tonal_centre is None
    assert len(hints.notes) >= 4


def test_missing_stems_are_noted_rather_than_fatal() -> None:
    summary = make_summary(make_source("mix", bpm=100.0, key="C", key_confidence=0.9))
    hints = build(summary)
    assert hints.bpm == 100.0
    assert hints.drum_density is None
    assert hints.bass_activity is None
    assert any("no drums stem" in note for note in hints.notes)
    assert any("no bass stem" in note for note in hints.notes)


def test_bpm_falls_back_to_drums_when_the_mix_has_none() -> None:
    summary = make_summary(
        make_source("mix", bpm=None),
        make_source("drums", bpm=128.0, onset_density=2.0),
    )
    hints = build(summary)
    assert hints.bpm == 128.0
    assert hints.suggested_cycle_seconds == pytest.approx(4 * 60.0 / 128.0)
    assert any("tempo taken from the drums stem" in note for note in hints.notes)


def test_tonal_centre_falls_back_to_bass_when_mix_confidence_is_low() -> None:
    summary = make_summary(
        make_source("mix", bpm=120.0, key="F#", scale="major", key_confidence=0.1),
        make_source("bass", key="D", scale="minor", key_confidence=0.88),
    )
    hints = build(summary)
    assert hints.tonal_centre == "D minor"
    assert any("bass stem" in note for note in hints.notes)


def test_tonal_centre_none_when_neither_source_is_confident() -> None:
    summary = make_summary(
        make_source("mix", key="F#", scale="major", key_confidence=0.1),
        make_source("bass", key="D", scale="minor", key_confidence=0.2),
    )
    hints = build(summary)
    assert hints.tonal_centre is None
    assert any("key confidence low" in note for note in hints.notes)


def test_tonal_centre_none_when_key_confidence_is_missing() -> None:
    summary = make_summary(make_source("mix", key="G", scale="major", key_confidence=None))
    hints = build(summary)
    assert hints.tonal_centre is None
    assert any("key confidence low" in note for note in hints.notes)


def test_tonal_centre_without_a_scale_prints_the_key_alone() -> None:
    summary = make_summary(make_source("mix", key="E", scale=None, key_confidence=0.9))
    assert build(summary).tonal_centre == "E"


def test_build_honours_a_non_default_beats_per_cycle() -> None:
    summary = make_summary(make_source("mix", bpm=120.0))
    assert build(summary, beats_per_cycle=8).suggested_cycle_seconds == 4.0


def test_notes_are_deduplicated_and_non_empty_strings() -> None:
    hints = build(make_summary(make_source("mix")))
    assert len(hints.notes) == len(set(hints.notes))
    assert all(note.strip() for note in hints.notes)


# --- drum_grid / bass_line / sound_suggestions (Wave 4) ---------------------


def test_drum_grid_absent_when_no_drums_stem() -> None:
    summary = make_summary(make_source("mix", bpm=120.0))
    hints = build(summary)
    assert hints.drum_grid.status == "not_attempted"
    assert hints.drum_grid.kick_steps == []
    assert any("no drums stem" in note for note in hints.notes)


def test_drum_grid_notes_when_decomposition_not_attempted() -> None:
    summary = make_summary(make_source("mix", bpm=120.0), make_source("drums", bpm=120.0))
    hints = build(summary)
    assert hints.drum_grid.status == "not_attempted"
    assert any("drum decomposition was not attempted" in note for note in hints.notes)


def test_drum_grid_surfaces_a_non_ok_status_with_a_note() -> None:
    decomposition = DrumDecomposition(status="no_grid", caveats=["no usable tempo estimate"])
    summary = make_summary(make_source("drums", bpm=None, drum_decomposition=decomposition))
    hints = build(summary)
    assert hints.drum_grid.status == "no_grid"
    assert hints.drum_grid.caveats == ["no usable tempo estimate"]
    assert any("drum grid status is 'no_grid'" in note for note in hints.notes)


def test_drum_grid_reads_patterns_not_hits() -> None:
    """Steps come from `DrumPattern.steps`, the field the plan requires --
    not from folding `hits` again here, which would duplicate `drum_elements`'
    own cycle-fitting logic.
    """
    decomposition = make_drum_decomposition()
    summary = make_summary(make_source("drums", bpm=120.0, drum_decomposition=decomposition))
    hints = build(summary)
    assert hints.drum_grid.kick_steps == [pattern.steps for pattern in decomposition.patterns
                                           if pattern.drum == "kick"][0]


def test_bass_line_absent_when_no_bass_stem() -> None:
    summary = make_summary(make_source("mix", bpm=120.0))
    hints = build(summary)
    assert hints.bass_line.status == "not_attempted"
    assert hints.bass_line.note_sequence == []
    assert any("no bass stem" in note for note in hints.notes)


def test_bass_line_notes_when_not_attempted() -> None:
    summary = make_summary(make_source("mix", bpm=120.0), make_source("bass"))
    hints = build(summary)
    assert hints.bass_line.status == "not_attempted"
    assert any("bass pitch tracking was not attempted" in note for note in hints.notes)


def test_bass_line_surfaces_a_non_ok_status_with_a_note() -> None:
    unvoiced = BassLine(status="unvoiced", voiced_fraction=0.0, caveats=["no pitch to track"])
    summary = make_summary(make_source("bass", bass_line=unvoiced))
    hints = build(summary)
    assert hints.bass_line.status == "unvoiced"
    assert hints.bass_line.note_sequence == []
    assert hints.bass_line.caveats == ["no pitch to track"]
    assert any("bass line status is 'unvoiced'" in note for note in hints.notes)


def test_bass_line_note_sequence_is_capped_with_truncated_from() -> None:
    """`strudel_hints.json` is documented as small and hand-readable -- a
    several-hundred-note line must not blow that up.
    """
    long_line = make_bass_line(note_count=BASS_LINE_NOTE_SEQUENCE_CAP + 10)
    summary = make_summary(make_source("bass", bass_line=long_line))
    hints = build(summary)
    assert len(hints.bass_line.note_sequence) == BASS_LINE_NOTE_SEQUENCE_CAP
    assert hints.bass_line.truncated_from == BASS_LINE_NOTE_SEQUENCE_CAP + 10
    assert hints.bass_line.note_sequence == ["a1"] * BASS_LINE_NOTE_SEQUENCE_CAP


def test_bass_line_not_truncated_when_under_the_cap() -> None:
    short_line = make_bass_line(note_count=3)
    summary = make_summary(make_source("bass", bass_line=short_line))
    hints = build(summary)
    assert hints.bass_line.truncated_from is None
    assert len(hints.bass_line.note_sequence) == 3


def test_sound_suggestions_empty_when_neither_drums_nor_bass_present() -> None:
    summary = make_summary(make_source("mix", bpm=120.0))
    assert build(summary).sound_suggestions == []


def test_sound_suggestions_from_drums_only() -> None:
    decomposition = make_drum_decomposition()
    summary = make_summary(make_source("drums", bpm=120.0, drum_decomposition=decomposition))
    hints = build(summary)
    roles = {s.role for s in hints.sound_suggestions}
    assert roles == {"kick", "hat"}
    assert "bass" not in roles


def test_sound_suggestions_from_bass_only_is_always_exactly_one() -> None:
    """`suggest_bass_sound` always returns one suggestion for a present bass
    stem, per its own contract -- even with no spectral evidence at all.
    """
    summary = make_summary(make_source("bass", bass_line=make_bass_line()))
    hints = build(summary)
    assert len(hints.sound_suggestions) == 1
    assert hints.sound_suggestions[0].role == "bass"


def test_strudel_vocabulary_read_is_always_populated() -> None:
    assert build(make_summary()).strudel_vocabulary_read == strudel_vocab.STRUDEL_DOCS_READ


def test_drum_grid_and_sound_suggestions_use_hits_not_the_stripped_summary_shape() -> None:
    """The Task 3 trap: `summary_payload()` strips `drum_decomposition.hits`
    to `total_hit_count`, and `suggest_drum_sounds()` reads `hits`. Calling
    `build()` on the stripped, un-rehydrated shape must not silently produce
    fewer suggestions than the same decomposition would with `hits` intact --
    this pins that `build()` itself does the right thing when handed a
    populated summary; `test_cli.py` pins that `export-strudel-hints`
    actually rehydrates before calling `build()`.
    """
    decomposition = make_drum_decomposition()
    full_hints = build(
        make_summary(make_source("drums", bpm=120.0, drum_decomposition=decomposition))
    )
    stripped = decomposition.model_copy(update={"hits": []})
    stripped_hints = build(
        make_summary(make_source("drums", bpm=120.0, drum_decomposition=stripped))
    )
    assert full_hints.sound_suggestions != []
    assert stripped_hints.sound_suggestions == []
    # ...but the folded pattern survives regardless, since `patterns` is never
    # stripped from `track_summary.json`.
    assert full_hints.drum_grid.kick_steps == stripped_hints.drum_grid.kick_steps == [0]


# --- v1 scope guard ---------------------------------------------------------


def test_hints_emit_descriptions_not_strudel_code() -> None:
    """v1 is descriptive only: no pattern strings, no mini-notation."""
    hints = build(full_summary())
    assert hints.subdivision_feel in SUBDIVISION_TERMS
    assert hints.drum_density in DENSITY_TERMS
    assert hints.bass_activity in DENSITY_TERMS
    fields = [hints.subdivision_feel, hints.drum_density, hints.bass_activity]
    for value in fields:
        assert value is not None
        assert not any(token in value for token in ("~", "<", ">", "*", "(", "s(", "sound"))
