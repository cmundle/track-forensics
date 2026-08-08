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
    Arrangement,
    BassLine,
    BassNote,
    DrumDecomposition,
    DrumHit,
    DrumPattern,
    DynamicsFeatures,
    OctaveCandidate,
    OctaveGridFit,
    RhythmFeatures,
    Section,
    SourceAnalysis,
    StrudelHints,
    TempoFit,
    TonalFeatures,
    TrackSummary,
)
from audio_pipeline.strudel_hints import (
    ARRANGEMENT_SECTION_CAP,
    BASS_LINE_NOTE_SEQUENCE_CAP,
    BEATS_PER_CYCLE,
    DENSITY_TERMS,
    SUBDIVISION_TERMS,
    _octave_note,
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
    rms_mean: float = 0.05,
) -> SourceAnalysis:
    """A SourceAnalysis with only the descriptors a given test cares about.

    `rms_mean` defaults well clear of `SILENCE_RMS_FLOOR`, so a source is a real
    stem unless a test deliberately makes it residue.
    """
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
        dynamics=DynamicsFeatures(rms_mean=rms_mean),
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


def make_summary(
    *sources: SourceAnalysis,
    track_name: str = "test-track",
    tempo: TempoFit | None = None,
    arrangement: Arrangement | None = None,
) -> TrackSummary:
    """A summary with no track-level tempo unless a test asks for one.

    Deliberately defaulted to absent rather than populated: most tests here
    exercise the per-source fallbacks, and a summary written before schema v5
    genuinely has no `TempoFit`. `full_summary()` supplies one, because "full"
    has to mean full.
    """
    return TrackSummary(
        track_name=track_name,
        input_path="/tmp/test-track.wav",
        duration_seconds=30.0,
        backend="fake",
        tempo=tempo or TempoFit(),
        arrangement=arrangement or Arrangement(),
        sources={source.source: source for source in sources},
    )


def make_tempo_fit(
    bpm: float = 120.0, status: str = "refined", confidence_label: str = "high"
) -> TempoFit:
    return TempoFit(
        bpm=bpm,
        period_seconds=60.0 / bpm,
        coarse_bpm=bpm,
        confidence=0.75,
        confidence_label=confidence_label,
        status=status,
    )


def make_arrangement(*, sections: int = 2, absent: list[str] | None = None) -> Arrangement:
    return Arrangement(
        sections=[
            Section(
                start_bar=index * 8,
                length_bars=8,
                start_seconds=index * 16.0,
                active=["drums", "bass", "kick"],
                label="full" if index else "intro",
                label_reason="everything is playing",
            )
            for index in range(sections)
        ],
        bar_count=sections * 8,
        bar_seconds=2.0,
        downbeat_seconds=0.0,
        absent_tracks=absent or [],
        status="ok",
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
        tempo=make_tempo_fit(),
        arrangement=make_arrangement(),
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


# --- schema v5: confidence has to reach the reader ---------------------------


def test_a_refined_tempo_reaches_the_hints_and_says_so() -> None:
    hints = build(full_summary())
    assert hints.bpm == 120.0
    assert hints.tempo_status == "refined"
    assert hints.tempo_confidence == "high"


def test_a_refused_tempo_is_printed_with_its_refusal() -> None:
    """v4 printed a bare `bpm: 143.25` where refinement had explicitly declined.

    The number is kept — a reader who can see it is unrefined can still start
    from it — but it can no longer be mistaken for a measurement.
    """
    summary = make_summary(
        make_source("mix", bpm=143.25),
        tempo=make_tempo_fit(bpm=143.25, status="coarse", confidence_label="low"),
    )
    hints = build(summary)
    assert hints.bpm == 143.25
    assert hints.tempo_status == "coarse"
    assert hints.tempo_confidence == "low"
    assert any("refinement was refused" in note for note in hints.notes)


def test_the_refined_tempo_wins_over_every_per_source_estimate() -> None:
    """The root of F1: five sources, five tempos, and no agreement on which."""
    summary = make_summary(
        make_source("mix", bpm=131.855),
        make_source("drums", bpm=132.040),
        tempo=make_tempo_fit(bpm=131.999957946896),
    )
    assert build(summary).bpm == 131.999957946896


def test_without_a_refined_tempo_the_v4_fallbacks_still_work() -> None:
    """A summary written before v5 has no `TempoFit`, and must still build."""
    summary = make_summary(make_source("mix", bpm=128.0))
    hints = build(summary)
    assert hints.bpm == 128.0
    assert any("no refined tempo" in note for note in hints.notes)


# --- schema v5: F5, the tonal centre a silent stem used to win ---------------


def test_a_silent_bass_stem_cannot_supply_the_tonal_centre() -> None:
    """Measured: a -70 LUFS bass stem reported "E minor" at 0.688 confidence,
    and beat the mix's own F major because the mix read 0.445."""
    summary = make_summary(
        make_source("mix", key="F", scale="major", key_confidence=0.445),
        make_source("bass", key="E", scale="minor", key_confidence=0.688, rms_mean=7.7e-05),
    )
    hints = build(summary)
    assert hints.tonal_centre is None
    assert any("below the silence floor" in note for note in hints.notes)


def test_a_real_bass_stem_still_supplies_the_tonal_centre() -> None:
    """The fallback is not deleted, only stopped from reaching into residue."""
    summary = make_summary(
        make_source("mix", key="F", scale="major", key_confidence=0.445),
        make_source("bass", key="E", scale="minor", key_confidence=0.688, rms_mean=0.1),
    )
    assert build(summary).tonal_centre == "E minor"


# --- schema v5: arrangement and bass step placement --------------------------


def test_the_arrangement_reaches_the_hints_one_line_per_section() -> None:
    hints = build(full_summary())
    assert hints.arrangement.status == "ok"
    assert hints.arrangement.bar_count == 16
    assert hints.arrangement.sections == [
        "bar 0 x8 intro: drums, bass, kick",
        "bar 8 x8 full: drums, bass, kick",
    ]
    assert hints.arrangement.truncated_from is None


def test_a_long_arrangement_is_capped_and_says_by_how_much() -> None:
    """The drum-and-bass corpus row reports 44 sections; this file stays small."""
    summary = make_summary(
        make_source("mix", bpm=170.0), arrangement=make_arrangement(sections=44)
    )
    hints = build(summary)
    assert len(hints.arrangement.sections) == ARRANGEMENT_SECTION_CAP
    assert hints.arrangement.truncated_from == 44


def test_absent_tracks_are_surfaced_as_do_not_write_parts_for_these() -> None:
    """The ambient corpus row has no drums, no vocals and no kick in it."""
    summary = make_summary(
        make_source("mix", bpm=83.0),
        arrangement=make_arrangement(absent=["drums", "vocals", "kick"]),
    )
    hints = build(summary)
    assert hints.arrangement.absent_tracks == ["drums", "vocals", "kick"]
    assert any("not in this record at all" in note for note in hints.notes)


def test_no_arrangement_leaves_the_block_empty_and_quiet() -> None:
    """A v4 summary has no arrangement; that is not something to warn about."""
    hints = build(make_summary(make_source("mix", bpm=120.0)))
    assert hints.arrangement.sections == []
    assert not any("arrangement" in note for note in hints.notes)


def test_the_bass_line_carries_its_step_placement() -> None:
    """F7's prediction: on the calibration track the bass sits on 2/6/10/14."""
    line = BassLine(
        status="ok",
        notes=[
            BassNote(
                start_seconds=float(step),
                duration_seconds=0.2,
                midi_note=33,
                note_name="a1",
                step=step,
            )
            for step in (2, 6, 10, 14, 2, 6)
        ],
    )
    summary = make_summary(make_source("bass", bass_line=line))
    assert build(summary).bass_line.steps == [2, 6, 10, 14]


def test_a_bass_line_with_no_grid_reports_no_steps() -> None:
    line = BassLine(
        status="ok",
        notes=[
            BassNote(start_seconds=0.0, duration_seconds=0.2, midi_note=33, note_name="a1"),
        ],
    )
    summary = make_summary(make_source("bass", bass_line=line))
    assert build(summary).bass_line.steps == []


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


# ---------------------------------------------------------------------------
# W8B: the swung branch, and whether anything can reach it
# ---------------------------------------------------------------------------
# The branch had returned nothing on every track in the project's history,
# including two chosen specifically to exercise it. The tests above show it
# firing on clean alternating interval sequences; what none of them show is a
# whole kit, which is what a drums stem's `onset_times` actually is — the union
# of kick, snare and hat, with the kick and snare landing on beats that the
# shuffled hats already occupy and every hit carrying its own jitter.
#
# Measured on the eight-track corpus, folding each drums stem's real onsets:
#
#   track       n IOIs   long/short   alternation   verdict
#   badu          1039        1.191         0.748   straight 8ths
#   madonna       1415            -         0.245   neither
#   roni          1547            -         0.356   neither
#   chameleon     5082            -         0.333   neither
#   levee         1095            -         0.303   neither
#   eno            274            -         0.418   neither
#
# `SWING_MIN_ALTERNATION` is 0.75 and Badu reaches 0.748, so its clusters are
# not even split into a pair — and if they were, the ratio is 1.191, well under
# `SWING_RATIO_RANGE[0]` at 1.5. That is Dilla's loose micro-timing on a
# straight-8th hat pattern, not a shuffle, and the tool naming it `straight
# 8ths` is the right answer. **The branch is reachable; the corpus has no
# shuffle in it.**


def kit_shuffle_times(
    bpm: float = 120.0, bars: int = 16, jitter: float = 0.0, seed: int = 0
) -> list[float]:
    """A triplet-feel shuffle played by a whole kit, as onset times.

    Hats on shuffled 8ths (offbeat at 2/3 of the beat, not 1/2), kick on every
    beat, snare on 2 and 4, every hit jittered independently. Coincident hits
    collapse the way an onset detector's would.
    """
    rng = random.Random(seed)
    beat = 60.0 / bpm
    times: list[float] = []
    for index in range(bars * 4):
        start = index * beat
        times.append(start)  # hat downbeat + kick
        times.append(start + beat * 2 / 3)  # shuffled offbeat hat
        if index % 2 == 1:
            times.append(start)  # snare on 2 and 4, coincident with the kick
    jittered = sorted(value + rng.gauss(0.0, jitter) for value in times)
    deduped = [jittered[0]]
    for value in jittered[1:]:
        if value - deduped[-1] > 0.005:
            deduped.append(value)
    return deduped


def test_a_whole_kit_playing_a_shuffle_reaches_the_swung_branch() -> None:
    """The reachability answer. Not a bare alternating IOI list — kick, snare
    and shuffled hats together, which is the shape a real drums stem hands
    over."""
    summary = make_summary(
        make_source("drums", bpm=120.0, onset_times=kit_shuffle_times()),
        tempo=make_tempo_fit(120.0),
    )
    assert infer_subdivision_feel(summary) == "swung 8ths"


@pytest.mark.parametrize("jitter", [0.004, 0.008, 0.015])
def test_the_swung_branch_survives_human_timing(jitter: float) -> None:
    """+/-15 ms is well past what a drummer contributes and past the ~12 ms
    onset-detector hop, and the branch still fires. It is not a knife-edge
    synthetic-only path."""
    summary = make_summary(
        make_source("drums", bpm=120.0, onset_times=kit_shuffle_times(jitter=jitter, seed=7)),
        tempo=make_tempo_fit(120.0),
    )
    assert infer_subdivision_feel(summary) == "swung 8ths"


def test_a_straight_kit_at_the_same_tempo_is_not_called_swung() -> None:
    """The control: the same kit, offbeat hats on the half rather than 2/3."""
    beat = 0.5
    times: list[float] = []
    for index in range(64):
        times.append(index * beat)
        times.append(index * beat + beat / 2)
    summary = make_summary(
        make_source("drums", bpm=120.0, onset_times=times), tempo=make_tempo_fit(120.0)
    )
    assert infer_subdivision_feel(summary) == "straight 8ths"


def test_badu_style_loose_straight_8ths_is_not_mistaken_for_swing() -> None:
    """The corpus's near miss, reconstructed from its measured statistics: a
    1.19 long/short ratio, under `SWING_RATIO_RANGE[0]`. Loose is not swung."""
    beat = 0.5
    long, short = beat / 2 * 1.087, beat / 2 * 0.913  # ratio 1.191, pair = 1 beat
    summary = drums_with_iois(repeat_to([long, short], 64))
    assert infer_subdivision_feel(summary) not in {"swung 8ths", "swung 16ths"}


# ---------------------------------------------------------------------------
# W8B: one account of the tempo octave
# ---------------------------------------------------------------------------


def make_octave_tempo_fit(
    *, arbitration: list[tuple[float, float, str, bool]], live: list[tuple[float, float]]
) -> TempoFit:
    return TempoFit(
        bpm=next(bpm for _, bpm, _, chosen in arbitration if chosen),
        period_seconds=1.0,
        coarse_bpm=next(bpm for _, bpm, _, chosen in arbitration if chosen),
        confidence=0.48,
        confidence_label="medium",
        status="refined",
        octave_status="ambiguous",
        octave_candidates=[
            OctaveCandidate(ratio=ratio, bpm=bpm, r=0.5, status="live") for ratio, bpm in live
        ],
        octave_arbitration=[
            OctaveGridFit(
                ratio=ratio,
                bpm=bpm,
                grid_status=grid,
                quantisation_error_steps=0.1,
                quantisation_error_seconds=0.01,
                chosen=chosen,
            )
            for ratio, bpm, grid, chosen in arbitration
        ],
        caveats=[
            "the octave is not settled by correlation alone: the reported tempo was not shifted",
            "tempo octave corrected x2 to 170.069 BPM",
        ],
    )


def test_a_corrected_octave_produces_one_note_that_says_the_tempo_moved() -> None:
    """Roni Size. `tempo.py` says it did not shift the tempo and `analyze.py`
    says it corrected it x2, both into the same caveat list. Read together they
    contradict; the hints file must give the reader one account, and it must be
    the one that describes the printed `bpm`."""
    tempo = make_octave_tempo_fit(
        arbitration=[
            (0.5, 42.52, "no_grid", False),
            (1.0, 85.04, "ok", False),
            (2.0, 170.07, "ok", True),
        ],
        live=[(0.5, 85.04), (1.0, 170.07), (2.0, 340.16)],
    )
    notes = build(make_summary(make_source("drums", bpm=85.0), tempo=tempo)).notes
    octave = [note for note in notes if note.startswith("tempo octave:")]
    assert len(octave) == 1
    assert "moved x2" in octave[0]
    # The pre-move reading, not `coarse_bpm` — which is re-refined at the new
    # octave and would print the corrected number twice.
    assert "85.04" in octave[0]
    assert "170.07" not in octave[0]
    assert not any("was not shifted" in note for note in notes)


def test_an_unmoved_but_ambiguous_octave_says_the_grid_confirmed_it() -> None:
    """Madonna and Badu: correlation could not settle it, the grid was fitted at
    every candidate, and the backend's own octave won. That is a different fact
    from 'nothing could arbitrate' and must not read as a warning."""
    tempo = make_octave_tempo_fit(
        arbitration=[
            (0.5, 66.0, "ok", False),
            (1.0, 132.0, "ok", True),
            (2.0, 264.0, "ok", False),
        ],
        live=[(0.5, 66.0), (1.0, 132.0), (2.0, 264.0)],
    )
    note = _octave_note(make_summary(make_source("drums"), tempo=tempo))
    assert note is not None
    assert "the backend's own octave won" in note
    assert "verify by ear" not in note


def test_an_ambiguous_octave_with_no_grid_warns_instead() -> None:
    """Eno and Levee Breaks: nothing to arbitrate with, so the octave is a real
    open question and the note says so."""
    tempo = TempoFit(
        bpm=143.25,
        period_seconds=60.0 / 143.25,
        coarse_bpm=143.25,
        status="coarse",
        confidence_label="low",
        octave_status="ambiguous",
        octave_candidates=[
            OctaveCandidate(ratio=1.0, bpm=143.25, r=0.11, status="live"),
            OctaveCandidate(ratio=2.0, bpm=286.5, r=0.10, status="live"),
        ],
    )
    note = _octave_note(make_summary(make_source("drums"), tempo=tempo))
    assert note is not None
    assert "no drum grid to arbitrate with" in note
    assert "286.50" in note


def test_a_settled_octave_says_nothing_at_all() -> None:
    """Chameleon and showers-of-gold: `octave_status="single"`. Silence is the
    correct output — this file does not narrate the absence of a problem."""
    tempo = make_tempo_fit(120.0)
    assert tempo.octave_status == "unavailable"
    assert _octave_note(make_summary(make_source("drums"), tempo=tempo)) is None

    settled = make_tempo_fit(120.0)
    settled.octave_status = "single"
    assert _octave_note(make_summary(make_source("drums"), tempo=settled)) is None


def test_a_silent_stem_gets_no_density() -> None:
    """F5's shape, found again in W8B. Three of the eight corpus tracks printed
    a density for separation residue: donjon and showers-of-gold both reported
    `bass_activity: moderate` for bass stems at rms 8.2e-05 and 6.4e-05, and
    Brian Eno's empty drums stem reported `drum_density: sparse`."""
    hints = build(
        make_summary(
            make_source("drums", onset_density=0.35, rms_mean=4.8e-05),
            make_source("bass", onset_density=3.03, rms_mean=8.2e-05),
        )
    )
    assert hints.drum_density is None
    assert hints.bass_activity is None
    # Three notes, not two: the tonal-centre fallback declines the same bass
    # stem for the same reason, which is the half of F5 that W6 already closed.
    assert sum("below the silence floor" in note for note in hints.notes) == 3
    assert any("drum density is not reported" in note for note in hints.notes)
    assert any("bass activity is not reported" in note for note in hints.notes)


def test_a_quiet_but_real_stem_still_gets_a_density() -> None:
    """The other side: the gate reads `SILENCE_RMS_FLOOR`, not 'quiet'."""
    hints = build(make_summary(make_source("bass", onset_density=3.03, rms_mean=0.0038)))
    assert hints.bass_activity == "moderate"
