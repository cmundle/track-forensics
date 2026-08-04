"""Tests for `drum_elements`: per-band detection, classification and grid.

Three things this file exists to prove, in order of how much they would cost to
get wrong:

1. **Per-band detection really does find coincident hits.**
   `drum_pattern_120bpm` has 48 hits at 32 distinct instants, and the whole
   design rests on recovering all 48. A design that classified one global onset
   list would find at most 32 and would delete the hat on every downbeat.
   `test_all_thirty_two_hats_survive_coincidence` is the load-bearing assertion.
2. **Nothing is hallucinated.** `drum_pattern_kick_only` must yield 16 kicks and
   literally nothing else, and the noise, silence and pure-tone signals must
   yield nothing at all. In practice a classifier inventing a hat pattern out of
   a kick's harmonics is worse than one that reports less.
3. **It runs with neither librosa nor essentia importable**, which is what makes
   drum output identical whichever backend `analyze` resolved.

Everything here is tuned against **all four drum fixtures at once**. Retuning a
threshold against one of them in isolation trades one fixture for another
silently; the parametrised `test_fixture_pattern_is_exactly_as_synthesised`
exists so that trade fails loudly.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from conftest import (
    DRUM_PATTERN_AMBIGUOUS_CLICK_STEPS,
    DRUM_PATTERN_AMBIGUOUS_KICK_STEPS,
    DRUM_PATTERN_ANCHOR_SECONDS,
    DRUM_PATTERN_BEATS_PER_CYCLE,
    DRUM_PATTERN_BPM,
    DRUM_PATTERN_CYCLE_SECONDS,
    DRUM_PATTERN_CYCLES,
    DRUM_PATTERN_HAT_STEPS,
    DRUM_PATTERN_KICK_ONLY_STEPS,
    DRUM_PATTERN_KICK_STEPS,
    DRUM_PATTERN_SNARE_STEPS,
    DRUM_PATTERN_STEP_SECONDS,
    DRUM_PATTERN_STEPS_PER_CYCLE,
    PATTERN_FIXTURE_DURATION_SECONDS,
    _click,
    _hat_closed,
    _hat_open,
    _hit_train,
    _kick,
    _snare,
)

from audio_pipeline import ANALYSIS_SAMPLE_RATE, BAND_EDGES_HZ, drum_elements
from audio_pipeline.drum_elements import (
    DETECTION_BANDS,
    DETECTOR_CLASS_AFFINITY,
    GRID_STEP_CANDIDATES,
    MAX_DECAY_RATIO,
    STFT_HOP_LENGTH,
    STFT_N_FFT,
    THRESHOLDS,
    decompose,
)
from audio_pipeline.schemas import (
    BLOCK_STATUSES,
    DRUM_CLASSES,
    GRID_ANCHOR_SOURCES,
    DrumDecomposition,
    RhythmFeatures,
)

#: Beat positions matching the drum fixtures: 120 BPM from the fixture anchor.
BEAT_TIMES: tuple[float, ...] = tuple(
    DRUM_PATTERN_ANCHOR_SECONDS + index * (60.0 / DRUM_PATTERN_BPM) for index in range(17)
)

#: Hits per step per class, for the fixtures that repeat over four cycles.
_PER_STEP = DRUM_PATTERN_CYCLES


def _run(
    audio: np.ndarray,
    *,
    bpm: float | None = DRUM_PATTERN_BPM,
    beat_times: Sequence[float] = BEAT_TIMES,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
) -> DrumDecomposition:
    """`decompose` with the drum fixtures' own tempo and anchor."""
    return decompose(
        audio,
        sample_rate,
        bpm=bpm,
        beat_times=beat_times,
        beats_per_cycle=DRUM_PATTERN_BEATS_PER_CYCLE,
    )


def _by_class(result: DrumDecomposition) -> dict[str, list]:
    """Hits grouped by class, every class in `DRUM_CLASSES` present as a key."""
    grouped: dict[str, list] = {name: [] for name in sorted(DRUM_CLASSES)}
    for hit in result.hits:
        grouped[hit.drum].append(hit)
    return grouped


def _one_shot_alone(one_shot: np.ndarray) -> np.ndarray:
    """One one-shot at the fixture anchor in an otherwise silent buffer.

    The measurement convention the module docstring's table is quoted in. The
    anchor matters: a one-shot starting at t=0 has no preceding frame and
    produces zero flux, so it is never detected.
    """
    return _hit_train(
        [(DRUM_PATTERN_ANCHOR_SECONDS, one_shot)], PATTERN_FIXTURE_DURATION_SECONDS
    )


# ---------------------------------------------------------------------------
# The four drum fixtures, asserted exactly and simultaneously
# ---------------------------------------------------------------------------

#: `fixture name -> {class: (steps within a cycle, total hits)}`, exact.
_EXPECTED: dict[str, dict[str, tuple[tuple[int, ...], int]]] = {
    "drum_pattern_120bpm": {
        "kick": (DRUM_PATTERN_KICK_STEPS, len(DRUM_PATTERN_KICK_STEPS) * _PER_STEP),
        "snare": (DRUM_PATTERN_SNARE_STEPS, len(DRUM_PATTERN_SNARE_STEPS) * _PER_STEP),
        "hat": (DRUM_PATTERN_HAT_STEPS, len(DRUM_PATTERN_HAT_STEPS) * _PER_STEP),
        "unclassified": ((), 0),
    },
    "drum_pattern_kick_only": {
        "kick": (DRUM_PATTERN_KICK_ONLY_STEPS, len(DRUM_PATTERN_KICK_ONLY_STEPS) * _PER_STEP),
        "snare": ((), 0),
        "hat": ((), 0),
        "unclassified": ((), 0),
    },
    "drum_pattern_open_hats": {
        "kick": (DRUM_PATTERN_KICK_STEPS, len(DRUM_PATTERN_KICK_STEPS) * _PER_STEP),
        "snare": (DRUM_PATTERN_SNARE_STEPS, len(DRUM_PATTERN_SNARE_STEPS) * _PER_STEP),
        "hat": (DRUM_PATTERN_HAT_STEPS, len(DRUM_PATTERN_HAT_STEPS) * _PER_STEP),
        "unclassified": ((), 0),
    },
    "drum_pattern_ambiguous": {
        "kick": (
            DRUM_PATTERN_AMBIGUOUS_KICK_STEPS,
            len(DRUM_PATTERN_AMBIGUOUS_KICK_STEPS) * _PER_STEP,
        ),
        "snare": ((), 0),
        "hat": ((), 0),
        "unclassified": (
            DRUM_PATTERN_AMBIGUOUS_CLICK_STEPS,
            len(DRUM_PATTERN_AMBIGUOUS_CLICK_STEPS) * _PER_STEP,
        ),
    },
}


@pytest.mark.parametrize("fixture_name", sorted(_EXPECTED))
@pytest.mark.parametrize("drum", ["kick", "snare", "hat", "unclassified"])
def test_fixture_pattern_is_exactly_as_synthesised(
    fixture_name: str, drum: str, request: pytest.FixtureRequest
) -> None:
    """Exact step set and exact hit count, per class, on every drum fixture.

    Parametrised over fixture *and* class deliberately: a threshold retuned
    against one fixture in isolation will trade another one away, and this way
    the trade shows up as several named failures rather than one.
    """
    steps, count = _EXPECTED[fixture_name][drum]
    hits = _by_class(_run(request.getfixturevalue(fixture_name)))[drum]
    assert {hit.step for hit in hits} == set(steps)
    assert len(hits) == count


def test_all_thirty_two_hats_survive_coincidence(drum_pattern_120bpm: np.ndarray) -> None:
    """**The load-bearing test.** All 48 hits, at only 32 distinct instants.

    Kick and hat sound together on steps 0 and 8, snare and hat on 4 and 12, so
    16 of the 48 hits share an instant with another hit. A design that
    classified a single global onset list would assign one class per instant and
    could not exceed 32 hits in total, losing 8 hats outright. Per-band
    detection finds two hits at those instants because two detectors fired.
    """
    result = _run(drum_pattern_120bpm)
    grouped = _by_class(result)

    assert len(grouped["hat"]) == 32
    assert len(result.hits) == 48

    coincident_steps = set(DRUM_PATTERN_KICK_STEPS) | set(DRUM_PATTERN_SNARE_STEPS)
    hats_on_coincident_steps = [hit for hit in grouped["hat"] if hit.step in coincident_steps]
    assert len(hats_on_coincident_steps) == len(coincident_steps) * DRUM_PATTERN_CYCLES == 16

    # And each of those really does share its instant with a non-hat hit.
    for hat in hats_on_coincident_steps:
        assert any(
            other.drum != "hat"
            and abs(other.time_seconds - hat.time_seconds)
            < THRESHOLDS["min_hit_separation_seconds"]
            for other in result.hits
        )


def test_kick_only_hallucinates_nothing(drum_pattern_kick_only: np.ndarray) -> None:
    """16 kicks and literally nothing else. The failure mode that matters most.

    A kick has harmonics in the body band and float residue everywhere above it.
    Reporting "no hats" is far more useful than reporting a hat pattern invented
    from a kick's own spectrum, so silence in a class has to be reachable.
    """
    result = _run(drum_pattern_kick_only)
    grouped = _by_class(result)
    assert len(grouped["kick"]) == 16
    assert grouped["snare"] == []
    assert grouped["hat"] == []
    assert grouped["unclassified"] == []
    assert result.unclassified_count == 0
    assert {pattern.drum for pattern in result.patterns} == {"kick"}


def test_ambiguous_clicks_are_unclassified_with_timing_preserved(
    drum_pattern_ambiguous: np.ndarray,
) -> None:
    """The clicks land in `unclassified`, keep their times, and are not forced into a class.

    A `_click` is noisier than a snare and has more air than one, so a rule
    keyed on `noise_ratio` calls all sixteen of them snares. Dropping them
    instead would lose two thirds of the source's hits.
    """
    result = _run(drum_pattern_ambiguous)
    clicks = _by_class(result)["unclassified"]
    assert len(clicks) == 16

    expected_times = sorted(
        DRUM_PATTERN_ANCHOR_SECONDS
        + cycle * DRUM_PATTERN_CYCLE_SECONDS
        + step * DRUM_PATTERN_STEP_SECONDS
        for cycle in range(DRUM_PATTERN_CYCLES)
        for step in DRUM_PATTERN_AMBIGUOUS_CLICK_STEPS
    )
    for hit, expected in zip(
        sorted(clicks, key=lambda item: item.time_seconds), expected_times, strict=True
    ):
        assert hit.time_seconds == pytest.approx(expected, abs=0.03)
    # Timing survived, and so did the measurements behind the non-decision.
    assert all(hit.body_ratio is not None and hit.decay_ratio is not None for hit in clicks)


def test_ambiguous_caveats_say_the_bucket_is_large(drum_pattern_ambiguous: np.ndarray) -> None:
    """Two thirds unclassified is correct here, and the output has to say so."""
    result = _run(drum_pattern_ambiguous)
    assert any("unclassified" in caveat for caveat in result.caveats)
    assert any("partial transcription" in caveat for caveat in result.caveats)


# ---------------------------------------------------------------------------
# Nothing found, nothing crashed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name", ["bass_unvoiced", "silence", "white_noise", "sine_a440", "stereo_pink_noise"]
)
def test_non_percussive_sources_find_nothing(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """A rumble, silence, noise and a pure tone are not drum patterns.

    `bass_unvoiced` is the sharp case: 20-200 Hz band-limited noise is loud and
    sits squarely in the kick band, and an adaptive threshold against its own
    fluctuation reported 128 hits before `_is_percussive` existed. Silence and a
    sine are the other two shapes of nothing — one with no energy, one with
    energy but no transient.
    """
    result = _run(request.getfixturevalue(fixture_name))
    assert result.hits == []
    assert result.status == "too_few_hits"
    assert result.patterns == []
    assert result.unclassified_count == 0


@pytest.mark.parametrize(
    ("name", "audio"),
    [
        ("empty", np.zeros(0, dtype=np.float32)),
        ("one sample", np.ones(1, dtype=np.float32)),
        ("shorter than a frame", np.ones(STFT_N_FFT // 2, dtype=np.float32)),
        ("nan", np.full(ANALYSIS_SAMPLE_RATE, np.nan, dtype=np.float32)),
        ("inf", np.full(ANALYSIS_SAMPLE_RATE, np.inf, dtype=np.float32)),
        ("huge", np.full(ANALYSIS_SAMPLE_RATE, 1e30, dtype=np.float32)),
        ("dc", np.ones(ANALYSIS_SAMPLE_RATE, dtype=np.float32)),
    ],
)
def test_degenerate_input_never_raises(name: str, audio: np.ndarray) -> None:
    """Whatever comes in, a valid `DrumDecomposition` comes out.

    A drum classifier is not worth losing the rest of an analysis over, so
    `decompose` swallows everything and reports `status="failed"` at worst.
    """
    result = _run(audio)
    assert isinstance(result, DrumDecomposition)
    assert result.status in BLOCK_STATUSES
    assert all(np.isfinite(hit.time_seconds) for hit in result.hits)


def test_internal_failure_becomes_failed_status(
    drum_pattern_120bpm: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception anywhere inside is reported, not raised — with its name."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic")

    monkeypatch.setattr(drum_elements, "_band_envelopes", _boom)
    result = _run(drum_pattern_120bpm)
    assert result.status == "failed"
    assert result.caveats == ["drum decomposition failed with RuntimeError"]
    assert result.hits == []


def test_stereo_input_is_decomposed_as_mono(drum_pattern_120bpm: np.ndarray) -> None:
    """Nothing here is stereo-aware, so two identical channels change nothing."""
    stereo = np.stack([drum_pattern_120bpm, drum_pattern_120bpm], axis=1)
    assert _run(stereo).model_dump() == _run(drum_pattern_120bpm).model_dump()


def test_off_rate_audio_is_analysed_but_caveated(drum_pattern_120bpm: np.ndarray) -> None:
    """Every threshold here was calibrated at 44.1 kHz; say so rather than refuse."""
    result = _run(drum_pattern_120bpm, sample_rate=48000)
    assert any("calibrated" in caveat for caveat in result.caveats)


# ---------------------------------------------------------------------------
# Backend independence
# ---------------------------------------------------------------------------


def test_module_source_names_no_audio_library() -> None:
    """A static guard: neither library may appear anywhere in the source.

    Cheap, and it catches the case the runtime test cannot — a lazy import added
    inside a branch that the fixtures never take.
    """
    source = (
        importlib.resources.files("audio_pipeline")
        .joinpath("drum_elements.py")
        .read_text(encoding="utf-8")
    )
    imports = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from ")) or "__import__" in line
    ]
    assert imports, "the guard found no import statements at all, so it proves nothing"
    for statement in imports:
        assert "librosa" not in statement, statement
        assert "essentia" not in statement, statement


def test_runs_with_neither_librosa_nor_essentia_importable(
    drum_pattern_120bpm: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Import the module fresh with both libraries blocked, and run it.

    This is the architectural claim of the whole module, so it is tested by
    actually breaking the imports rather than by inspection: `__import__` raises
    for either name, the module is reloaded from source under that regime, and
    the reloaded module must reproduce `drum_pattern_120bpm` hit for hit.
    """
    blocked = ("librosa", "essentia")
    real_import = builtins.__import__

    def _guarded(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] in blocked:
            raise ImportError(f"{name} is blocked for this test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    for name in list(sys.modules):
        if name.split(".")[0] in blocked:
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(builtins, "__import__", _guarded)

    reloaded = importlib.reload(drum_elements)
    try:
        with pytest.raises(ImportError):
            _guarded("librosa")
        result = reloaded.decompose(
            drum_pattern_120bpm,
            ANALYSIS_SAMPLE_RATE,
            bpm=DRUM_PATTERN_BPM,
            beat_times=BEAT_TIMES,
            beats_per_cycle=DRUM_PATTERN_BEATS_PER_CYCLE,
        )
        assert result.model_dump() == _run(drum_pattern_120bpm).model_dump()
    finally:
        monkeypatch.undo()
        importlib.reload(drum_elements)


def test_band_energy_matches_the_projects_canonical_definition(
    drum_pattern_120bpm: np.ndarray,
) -> None:
    """`_band_energy` reproduces `band_energy_ratios()` exactly on its own bounds.

    The two helpers must agree bin for bin — same centre-frequency assignment,
    same half-open intervals with a closed top, same `magnitude ** 2` summed
    over frames — or a `DrumHit`'s ratios and a `SourceAnalysis`'s band ratios
    would mean subtly different things.

    `band_energy_ratios` is imported **here and nowhere else**: it lives in
    `backends.librosa_backend`, and `drum_elements` importing it would defeat
    the point of the module.
    """
    from audio_pipeline.backends.librosa_backend import band_energy_ratios

    magnitude, freqs = drum_elements._stft_magnitude(drum_pattern_120bpm, ANALYSIS_SAMPLE_RATE)
    ceiling = max(high for _, high in BAND_EDGES_HZ.values())
    mine = {
        name: drum_elements._band_energy(
            magnitude, freqs, low, high, include_high=high >= ceiling
        )
        for name, (low, high) in BAND_EDGES_HZ.items()
    }
    total = sum(mine.values())
    expected = band_energy_ratios(magnitude, freqs)

    assert set(mine) == set(expected)
    for name, value in mine.items():
        assert expected[name] is not None
        assert value / total == pytest.approx(expected[name], rel=1e-12)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_rhythm_onsets_cannot_influence_the_decomposition(
    drum_pattern_120bpm: np.ndarray,
) -> None:
    """Two wildly different `RhythmFeatures` produce an identical decomposition.

    The two backends' onset detectors diverge hard — 0.125 onsets/s against
    8.125 on the same white noise — so any dependence on `onset_times` would
    make the drum pattern a function of which wheel happened to install.
    `decompose` takes `bpm` and `beat_times` and nothing else from the rhythm
    block, and this pins that.
    """
    sparse = RhythmFeatures(
        bpm=DRUM_PATTERN_BPM,
        beat_times=list(BEAT_TIMES),
        onset_times=[0.1],
        onset_density=0.125,
        transient_sharpness=1.0,
    )
    dense = RhythmFeatures(
        bpm=DRUM_PATTERN_BPM,
        beat_times=list(BEAT_TIMES),
        onset_times=list(np.linspace(0.0, 8.5, 700)),
        onset_density=82.4,
        transient_sharpness=99.0,
    )
    results = [
        decompose(
            drum_pattern_120bpm,
            ANALYSIS_SAMPLE_RATE,
            bpm=rhythm.bpm,
            beat_times=rhythm.beat_times,
            beats_per_cycle=DRUM_PATTERN_BEATS_PER_CYCLE,
        )
        for rhythm in (sparse, dense)
    ]
    assert results[0].model_dump() == results[1].model_dump()


def test_hits_are_unchanged_when_only_the_grid_input_changes(
    drum_pattern_120bpm: np.ndarray,
) -> None:
    """Detection is grid-free: changing `beat_times` moves steps, never hits."""
    anchored = _run(drum_pattern_120bpm)
    shifted = _run(drum_pattern_120bpm, beat_times=[value + 0.0625 for value in BEAT_TIMES])
    assert [(hit.time_seconds, hit.drum) for hit in anchored.hits] == [
        (hit.time_seconds, hit.drum) for hit in shifted.hits
    ]


def test_repeated_calls_are_identical(drum_pattern_open_hats: np.ndarray) -> None:
    """No RNG, no dict-ordering dependence, no iteration-order ties."""
    assert _run(drum_pattern_open_hats).model_dump() == _run(drum_pattern_open_hats).model_dump()


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def test_grid_is_sixteen_steps_anchored_on_the_beats(drum_pattern_120bpm: np.ndarray) -> None:
    result = _run(drum_pattern_120bpm)
    assert result.status == "ok"
    assert result.steps_per_cycle == DRUM_PATTERN_STEPS_PER_CYCLE
    assert result.cycle_seconds == pytest.approx(DRUM_PATTERN_CYCLE_SECONDS)
    assert result.grid_anchor_source == "beats"
    assert result.grid_anchor_seconds == pytest.approx(DRUM_PATTERN_ANCHOR_SECONDS)
    assert result.quantisation_error_steps is not None
    assert result.quantisation_error_steps < THRESHOLDS["max_quantisation_error_steps"]


def test_quarter_notes_do_not_become_a_triplet_grid(
    drum_pattern_kick_only: np.ndarray,
) -> None:
    """A quarter-note pattern fits 16 and 12 equally; 16 must win.

    Comparing quantisation error *in steps* silently prefers the coarser grid,
    because the same timing jitter is a smaller fraction of a longer step. This
    fixture measured 12 steps per cycle, with its kicks on 0/3/6/9, until the
    comparison moved into seconds.
    """
    result = _run(drum_pattern_kick_only)
    assert result.steps_per_cycle == 16
    assert {hit.step for hit in result.hits} == set(DRUM_PATTERN_KICK_ONLY_STEPS)


def test_no_bpm_means_no_grid_but_still_hits(drum_pattern_120bpm: np.ndarray) -> None:
    """A wrong grid is worse than none, and the hits are the useful part anyway."""
    result = _run(drum_pattern_120bpm, bpm=None, beat_times=[])
    assert result.status == "no_grid"
    assert len(result.hits) == 48
    assert result.steps_per_cycle is None
    assert all(hit.step is None for hit in result.hits)
    assert any("tempo" in caveat for caveat in result.caveats)


def test_a_wrong_tempo_is_rejected_rather_than_fitted(drum_pattern_120bpm: np.ndarray) -> None:
    """97.3 BPM on a 120 BPM pattern must not produce a grid.

    It scores 0.246 steps of mean error, which is essentially the 0.25 that
    uniformly random hits score — which is exactly why the limit is not 0.25.
    """
    result = _run(drum_pattern_120bpm, bpm=97.3, beat_times=[])
    assert result.status == "no_grid"
    assert result.steps_per_cycle is None
    assert result.quantisation_error_steps is not None
    assert result.quantisation_error_steps > THRESHOLDS["max_quantisation_error_steps"]


def test_grid_falls_back_to_the_first_hit_without_beats(drum_pattern_120bpm: np.ndarray) -> None:
    result = _run(drum_pattern_120bpm, beat_times=[])
    assert result.grid_anchor_source == "first_hit"
    assert result.grid_anchor_seconds == pytest.approx(DRUM_PATTERN_ANCHOR_SECONDS, abs=0.03)
    assert result.status == "ok"


def test_step_occupancy_is_full_on_a_repeating_pattern(
    drum_pattern_120bpm: np.ndarray,
) -> None:
    """Every step of every class is hit in all four cycles, so occupancy is 1.0."""
    for pattern in _run(drum_pattern_120bpm).patterns:
        assert pattern.step_occupancy == pytest.approx([1.0] * len(pattern.steps))
        assert len(pattern.steps) == len(pattern.step_occupancy)


def test_step_occupancy_reports_a_ghost_hit_as_a_fraction() -> None:
    """One extra kick in one cycle of four reads as 0.25, not as membership."""
    def _at(cycle: int, step: int) -> float:
        return (
            DRUM_PATTERN_ANCHOR_SECONDS
            + cycle * DRUM_PATTERN_CYCLE_SECONDS
            + step * DRUM_PATTERN_STEP_SECONDS
        )

    times = [
        _at(cycle, step)
        for cycle in range(DRUM_PATTERN_CYCLES)
        for step in DRUM_PATTERN_KICK_STEPS
    ]
    times.append(_at(2, 6))
    audio = _hit_train(
        [(time_s, _kick()) for time_s in sorted(times)], PATTERN_FIXTURE_DURATION_SECONDS
    )
    kicks = next(pattern for pattern in _run(audio).patterns if pattern.drum == "kick")
    assert kicks.steps == [0, 6, 8]
    occupancy = dict(zip(kicks.steps, kicks.step_occupancy, strict=True))
    assert occupancy[0] == pytest.approx(1.0)
    assert occupancy[8] == pytest.approx(1.0)
    assert occupancy[6] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Schema conformance and the WP-C contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", sorted(_EXPECTED))
def test_output_only_uses_the_schemas_vocabularies(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """`drum`, `status` and `grid_anchor_source` are plain `str` in the schema.

    They are validated by these frozensets rather than by `Literal`, because a
    `Literal` breaks the fake-backend filler in `tests/test_analyze.py`. Nothing
    else enforces them, so this does.
    """
    result = _run(request.getfixturevalue(fixture_name))
    assert result.status in BLOCK_STATUSES
    assert result.grid_anchor_source in GRID_ANCHOR_SOURCES
    assert {hit.drum for hit in result.hits} <= DRUM_CLASSES
    assert {pattern.drum for pattern in result.patterns} <= DRUM_CLASSES
    assert result.unclassified_count == sum(
        1 for hit in result.hits if hit.drum == "unclassified"
    )
    assert [hit.time_seconds for hit in result.hits] == sorted(
        hit.time_seconds for hit in result.hits
    )


@pytest.mark.parametrize("fixture_name", sorted(_EXPECTED))
def test_every_hit_carries_the_evidence_that_classified_it(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Ratios sum to 1 and every measurement is finite and in range.

    A hit that cannot be audited from its own record is a label, not a
    measurement — the same rule `HeuristicLabel.evidence` follows.
    """
    for hit in _run(request.getfixturevalue(fixture_name)).hits:
        ratios = [hit.kick_ratio, hit.body_ratio, hit.noise_ratio, hit.air_ratio]
        assert all(value is not None and 0.0 <= value <= 1.0 for value in ratios)
        assert sum(value for value in ratios if value is not None) == pytest.approx(1.0)
        assert hit.decay_ratio is not None and 0.0 < hit.decay_ratio <= MAX_DECAY_RATIO
        assert hit.flatness is not None and 0.0 <= hit.flatness <= 1.0
        assert 0.0 <= hit.confidence <= 1.0


def test_patterns_agree_with_the_hits_they_fold(drum_pattern_120bpm: np.ndarray) -> None:
    result = _run(drum_pattern_120bpm)
    grouped = _by_class(result)
    for pattern in result.patterns:
        assert pattern.hit_count == len(grouped[pattern.drum])
        assert pattern.steps == sorted({hit.step for hit in grouped[pattern.drum]})


def test_decay_ratio_separates_closed_from_open_hats(
    drum_pattern_120bpm: np.ndarray, drum_pattern_open_hats: np.ndarray
) -> None:
    """The `hh` versus `oh` call `strudel_vocab` makes lands on the right side.

    `strudel_vocab.HAT_CLOSED_DECAY_RATIO_MIN` compares the **median**
    `decay_ratio` across a source's hat hits against 4.0, so `decay_ratio` is a
    load-bearing output field rather than a diagnostic one. Measured here:
    closed hats 5.40-12.49, open hats 1.46-1.72.
    """
    closed = [hit.decay_ratio for hit in _by_class(_run(drum_pattern_120bpm))["hat"]]
    opened = [hit.decay_ratio for hit in _by_class(_run(drum_pattern_open_hats))["hat"]]
    assert len(closed) == len(opened) == 32
    assert float(np.median(closed)) > 4.0 > float(np.median(opened))
    # Not merely on the right side of the line — separated with room to spare.
    assert min(closed) > max(opened) * 2.0


# ---------------------------------------------------------------------------
# The measurement conventions, pinned
# ---------------------------------------------------------------------------

#: The module docstring's table, as `one-shot -> (band, kick, body, noise, air,
#: decay_ratio, flatness)`. Measured with each one-shot alone at 0.25 s in an
#: 8.5 s buffer. WP-CAL recalibrates against exactly these numbers, so they are
#: asserted rather than merely documented — a docstring cannot fail.
_OneShotRow = tuple[Callable[[], np.ndarray], str, tuple[float, ...], float, float]
_ONE_SHOT_TABLE: dict[str, _OneShotRow] = {
    "kick": (_kick, "kick", (0.9928, 0.0072, 0.0000, 0.0000), 2.01, 9.0e-07),
    "snare": (_snare, "body", (0.0118, 0.5939, 0.3931, 0.0012), 2.59, 2.5e-03),
    "hat_closed": (_hat_closed, "air", (0.0000, 0.0000, 0.0012, 0.9988), 10.36, 4.9e-02),
    "hat_open": (_hat_open, "air", (0.0000, 0.0000, 0.0009, 0.9991), 1.47, 4.1e-02),
    "click": (_click, "noise", (0.0152, 0.0508, 0.5468, 0.3872), 100.00, 5.2e-01),
}


@pytest.mark.parametrize("name", sorted(_ONE_SHOT_TABLE))
def test_one_shot_measurements_match_the_documented_convention(name: str) -> None:
    """Every number in the module docstring's table, re-measured.

    Reaches into the module's own helpers rather than going through
    `decompose`, because the table is quoted per *detecting band* and
    `decompose` reports only the surviving hit.
    """
    make, band, ratios, decay, flatness = _ONE_SHOT_TABLE[name]
    audio = _one_shot_alone(make())

    magnitude, freqs = drum_elements._stft_magnitude(audio, ANALYSIS_SAMPLE_RATE)
    envelopes = drum_elements._band_envelopes(magnitude, freqs)
    fluxes = {
        key: drum_elements._spectral_flux(envelope) for key, envelope in envelopes.items()
    }
    active, _dormant = drum_elements._active_bands(envelopes, fluxes)
    assert band in active

    candidates = [
        drum_elements._Candidate(int(frame), key)
        for key in active
        for frame in drum_elements._pick_peaks(fluxes[key], ANALYSIS_SAMPLE_RATE)
    ]
    candidates.sort(key=lambda item: (item.frame, item.band))
    drum_elements._measure(candidates, magnitude, freqs, envelopes, ANALYSIS_SAMPLE_RATE)

    measured = next(item for item in candidates if item.band == band)
    assert [measured.ratios[key] for key in DETECTION_BANDS] == pytest.approx(ratios, abs=5e-4)
    assert measured.decay_ratio == pytest.approx(decay, rel=0.01)
    assert measured.flatness == pytest.approx(flatness, rel=0.05)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("kick", "kick"), ("snare", "snare"), ("hat_closed", "hat"), ("hat_open", "hat")],
)
def test_an_isolated_one_shot_is_classified_as_itself(name: str, expected: str) -> None:
    """Each one-shot alone, through the real entry point, reads as what it is.

    The snare is the interesting one: its air share of 0.0012 clears the
    band-activity floor and its air-band decay of 1.94 is indistinguishable from
    an open hat's, so it reported a phantom hat on top of itself until
    `hat_air_over_noise` existed. None of the four pattern fixtures can catch
    that — they have a hat on every snare, or no snares at all.
    """
    audio = _one_shot_alone(_ONE_SHOT_TABLE[name][0]())
    result = _run(audio)
    assert {hit.drum for hit in result.hits} == {expected}


def test_an_isolated_click_is_unclassified() -> None:
    """`_click` is not a kick, a snare or a hat, and saying so is the right answer."""
    result = _run(_one_shot_alone(_click()))
    assert [hit.drum for hit in result.hits] == ["unclassified"]
    assert result.hits[0].confidence == 0.0


def test_the_stft_grid_matches_both_backends() -> None:
    """2048/512 here, in `librosa_backend` and in `essentia_backend`, or the
    band edges land on different bins in different modules."""
    from audio_pipeline.backends import librosa_backend

    assert (STFT_N_FFT, STFT_HOP_LENGTH) == (
        librosa_backend.STFT_N_FFT,
        librosa_backend.STFT_HOP_LENGTH,
    )


def test_detection_bands_are_not_the_shared_band_edges() -> None:
    """Deliberately different, and the difference is the point.

    `BAND_EDGES_HZ` puts a kick's fundamental and a snare's 200 Hz shell tone in
    the same `low` band, so no threshold on it can separate the two.
    """
    assert set(DETECTION_BANDS.values()) != set(BAND_EDGES_HZ.values())
    kick_low, kick_high = DETECTION_BANDS["kick"]
    body_low, body_high = DETECTION_BANDS["body"]
    assert kick_high == body_low
    low_low, low_high = BAND_EDGES_HZ["low"]
    assert low_low <= kick_low and low_high > body_low  # one band spans both


# ---------------------------------------------------------------------------
# Threshold and affinity meta-tests
# ---------------------------------------------------------------------------

#: Intended ramp direction of every threshold/saturation pair in `THRESHOLDS`.
#: `heuristics._ramp` infers direction from its arguments, so retuning a
#: threshold past its saturation silently *inverts* the rule instead of
#: failing. This mirrors `tests/test_heuristics.py::RAMP_DIRECTIONS` for this
#: module's own dict.
_RAMP_DIRECTIONS: dict[str, str] = {
    "kick_low_ratio": "up",
    "snare_body_ratio": "up",
    "hat_decay_ratio": "down",
    "hat_air_over_noise": "up",
}


@pytest.mark.parametrize("key", sorted(_RAMP_DIRECTIONS))
def test_every_threshold_pair_ramps_in_its_intended_direction(key: str) -> None:
    threshold = THRESHOLDS[key]
    saturation = THRESHOLDS[f"{key}_saturation"]
    assert threshold != saturation
    if _RAMP_DIRECTIONS[key] == "up":
        assert saturation > threshold
    else:
        assert saturation < threshold


def test_every_saturation_has_a_threshold_and_a_declared_direction() -> None:
    """No orphan `*_saturation` key, and none missing from `_RAMP_DIRECTIONS`."""
    saturations = {key[: -len("_saturation")] for key in THRESHOLDS if key.endswith("_saturation")}
    assert saturations <= set(THRESHOLDS)
    assert saturations == set(_RAMP_DIRECTIONS)


def test_affinity_covers_every_band_and_every_scored_class() -> None:
    """A band with no affinity row would silently score zero for everything."""
    assert set(DETECTOR_CLASS_AFFINITY) == set(DETECTION_BANDS)
    for band, row in DETECTOR_CLASS_AFFINITY.items():
        assert set(row) == set(DRUM_CLASSES) - {"unclassified"}, band
        assert all(0.0 <= value <= 1.0 for value in row.values()), band
        assert max(row.values()) == 1.0, band


def test_air_cannot_be_a_kick_and_kick_cannot_be_a_hat() -> None:
    """The two hard vetoes, which are physics rather than tuning.

    A kick has no measurable energy above 6 kHz, and a hat has none below
    150 Hz, so those two affinities are zero and WP-CAL should not retune them.
    """
    assert DETECTOR_CLASS_AFFINITY["air"]["kick"] == 0.0
    assert DETECTOR_CLASS_AFFINITY["kick"]["hat"] == 0.0
    assert DETECTOR_CLASS_AFFINITY["body"]["hat"] == 0.0
    assert DETECTOR_CLASS_AFFINITY["noise"]["kick"] == 0.0


def test_grid_candidates_prefer_the_straight_grid() -> None:
    assert GRID_STEP_CANDIDATES[0] == 16
    assert set(GRID_STEP_CANDIDATES) == {16, 12}


# ---------------------------------------------------------------------------
# Unit tests on the primitives
# ---------------------------------------------------------------------------


def test_decide_requires_both_a_floor_and_a_margin() -> None:
    """argmax alone is a coin toss; the margin is what makes it a decision."""
    floor = THRESHOLDS["decision_floor"]
    margin = THRESHOLDS["decision_margin"]

    clear = {"kick": 0.9, "snare": 0.1, "hat": 0.0}
    assert drum_elements._decide(clear) == ("kick", 0.9)

    below_floor = {"kick": floor / 2, "snare": 0.0, "hat": 0.0}
    drum, confidence = drum_elements._decide(below_floor)
    assert drum == "unclassified"
    assert confidence == pytest.approx(floor / 2)

    too_close = {"kick": 0.9, "snare": 0.9 - margin / 2, "hat": 0.0}
    drum, confidence = drum_elements._decide(too_close)
    assert drum == "unclassified"
    # The honest winner confidence survives the non-decision.
    assert confidence == pytest.approx(0.9)

    exactly_on_both_lines = {"kick": floor, "snare": floor - margin, "hat": 0.0}
    assert drum_elements._decide(exactly_on_both_lines) == ("kick", floor)


def test_decide_breaks_exact_ties_alphabetically() -> None:
    """A tie is not a decision, but it must at least be the same non-decision twice."""
    tied = {"kick": 0.8, "snare": 0.8, "hat": 0.0}
    assert drum_elements._decide(tied) == ("unclassified", 0.8)


def test_spectral_flux_is_rectified_and_starts_at_zero() -> None:
    envelope = np.array([0.0, 10.0, 5.0, 5.0, 100.0], dtype=np.float64)
    flux = drum_elements._spectral_flux(envelope)
    assert flux[0] == 0.0
    assert np.all(flux >= 0.0)
    assert flux[2] == 0.0  # a fall contributes nothing
    assert flux[4] > flux[1]


def test_rolling_median_shrinks_at_the_edges_rather_than_padding() -> None:
    """Edge padding would bias the first half second towards frame 0's value.

    Every fixture in this repo puts its first hit inside that region, so the
    difference is not academic.
    """
    values = np.array([9.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    median = drum_elements._rolling_median(values, 1)
    assert median[0] == pytest.approx(4.5)  # median of [9, 0], not of [9, 9, 0]
    assert median[-1] == 0.0


def test_band_envelope_assigns_bins_by_centre_frequency() -> None:
    """Half-open `[low, high)`, closed only when asked."""
    freqs = np.array([0.0, 100.0, 150.0, 200.0], dtype=np.float64)
    magnitude = np.ones((4, 1), dtype=np.float64)
    assert drum_elements._band_energy(magnitude, freqs, 100.0, 150.0) == pytest.approx(1.0)
    assert drum_elements._band_energy(
        magnitude, freqs, 100.0, 150.0, include_high=True
    ) == pytest.approx(2.0)


def test_band_envelope_rejects_mismatched_inputs() -> None:
    magnitude = np.ones((4, 1), dtype=np.float64)
    assert drum_elements._band_energy(magnitude, np.zeros(3), 0.0, 1.0) == 0.0


def test_flatness_is_scale_invariant() -> None:
    """A quiet hat and a loud kick must be measured on the same scale."""
    rng = np.random.default_rng(3)
    window = np.abs(rng.standard_normal((100, 5)))
    in_range = np.ones(100, dtype=bool)
    assert drum_elements._flatness(window, in_range) == pytest.approx(
        drum_elements._flatness(window * 1e6, in_range)
    )


def test_decay_ratio_clamps_a_silent_tail() -> None:
    envelope = np.array([100.0, 50.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert drum_elements._decay_ratio(envelope, 0, 6, 3) == MAX_DECAY_RATIO
    assert drum_elements._decay_ratio(envelope, 0, 1, 3) is None
