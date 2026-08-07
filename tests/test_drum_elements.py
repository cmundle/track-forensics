"""Tests for `drum_elements`: per-band detection, classification and grid.

Five things this file exists to prove, in order of how much they would cost to
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
4. **A kick found twice is one kick, and a hat over a kick is still two hits.**
   Those pull in opposite directions and the v5 bleed rule has to satisfy both;
   `test_a_bright_kick_alone_is_sixteen_kicks_and_nothing_else` and
   `test_a_hat_over_a_bright_kick_is_still_two_hits` are the pair, and neither
   is meaningful without the other.
5. **Real material does what the calibration says it does.** The Madonna drum
   fixture is four envelope arrays, not audio, and it is the only thing here
   that can fail because a threshold is right in the abstract and wrong on a
   record — which is how three of `V2-PLAN.md`'s eight findings went wrong.

Everything here is tuned against **all four synthetic drum fixtures and the
real one at once**. Retuning a threshold against one of them in isolation
trades one fixture for another silently; the parametrised
`test_fixture_pattern_is_exactly_as_synthesised` exists so that trade fails
loudly.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

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
    KICK_PEAK,
    PATTERN_FIXTURE_DURATION_SECONDS,
    _band_limited_noise,
    _click,
    _decay,
    _drum_pattern,
    _hat_closed,
    _hat_open,
    _hit_train,
    _kick,
    _normalised,
    _snare,
    _step_times,
)

from audio_pipeline import ANALYSIS_SAMPLE_RATE, BAND_EDGES_HZ, drum_elements
from audio_pipeline.drum_elements import (
    BLEED_SOURCE_BAND,
    BLEED_TARGET_BANDS,
    DETECTION_BANDS,
    DETECTOR_CLASS_AFFINITY,
    GRID_ON_GRID_SHARE_MIN,
    GRID_OVERSAMPLE,
    GRID_STEP_CANDIDATES,
    KICK_BLEED_AIR_OVER_NOISE,
    KICK_BLEED_DOMINANCE,
    MAX_DECAY_RATIO,
    MIN_FOLD_CYCLES,
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

#: `GRID_ANCHOR_SOURCES` in the frozen `schemas.py` holds `beats` and
#: `first_hit`. v5 adds a third: an anchor taken from `tempo.DownbeatFit`,
#: which is neither of those and deserves to be distinguishable — a grid
#: anchored on a measured downbeat and one anchored on whatever happened to be
#: loudest first are not equally trustworthy, which is the whole reason the
#: field exists.
#:
#: **`schemas.py` is frozen and only W6 may extend that frozenset.** Until it
#: does, `decompose` emits the value (pydantic does not validate it) and this
#: constant is the record of what W6 owes. Delete it and use the frozenset
#: directly once `supplied` is in there.
SUPPLIED_ANCHOR_SOURCE: frozenset[str] = frozenset({"supplied"})

#: Beat positions matching the drum fixtures: 120 BPM from the fixture anchor.
BEAT_TIMES: tuple[float, ...] = tuple(
    DRUM_PATTERN_ANCHOR_SECONDS + index * (60.0 / DRUM_PATTERN_BPM) for index in range(17)
)

#: Hits per step per class, for the fixtures that repeat over four cycles.
_PER_STEP = DRUM_PATTERN_CYCLES

#: Seconds between STFT frames, and therefore the time resolution of every hit
#: this module reports. Several assertions below are stated in halves and
#: multiples of it rather than in bare seconds, because that is what the
#: tolerance actually is.
HOP_SECONDS: float = STFT_HOP_LENGTH / ANALYSIS_SAMPLE_RATE

#: Half an STFT window, 23.2 ms. The real uncertainty on a reported hit time:
#: a frame's window spans `STFT_N_FFT` samples, so a frame centred *before* an
#: onset still sees it and still fluxes, and the peak-picked frame can precede
#: the onset by up to this much. Anything asserting where a hit or an anchor
#: landed in absolute time is stated against this rather than against the hop.
WINDOW_HALF_SECONDS: float = 0.5 * STFT_N_FFT / ANALYSIS_SAMPLE_RATE


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


# ---------------------------------------------------------------------------
# A kick that reproduces the real failure
# ---------------------------------------------------------------------------
#
# `conftest._kick` is a 60 Hz sine with a pitch sweep and nothing above 150 Hz,
# which is the right fixture for the classifier's kick rule and is **incapable
# of showing the bug this module's v5 bleed rule exists for**: its noise and
# air bands never clear `BAND_ACTIVITY_FLOOR`, so the upper detectors never
# fire on it and there is no second detection to suppress.
#
# A real kick is not like that. It has a beater click at 1-6 kHz with a 6-16
# kHz shoulder, and on the Madonna drums stem that click fires the noise and
# air detectors on every single kick. `_bright_kick` is `_kick()` plus exactly
# that, tuned to the band shares measured on the record rather than to
# whatever looked plausible.

#: Weight on the 6-16 kHz half of the beater click. Chosen so the hit window's
#: `air / (air + noise)` lands at 0.195, against a median of 0.20 measured over
#: the 979 kick-coincident upper-band detections on the Madonna drums fixture.
BEATER_AIR_WEIGHT = 0.52

#: Amplitude of the click relative to the kick before normalisation. Chosen so
#: the hit window measures `kick_ratio` 0.811, inside the 0.72-0.84 measured on
#: the Madonna bleed.
BEATER_SCALE = 2.0

#: Click length and decay constant. 20 ms is the number that matters: shorter
#: and the click reads as a `_click` (air-band `decay_ratio` above
#: `hat_decay_ratio`, so it lands in `unclassified` and the *existing* leakage
#: rule already removes it — the bug never appears); longer and the click
#: dominates its own window and the kick stops measuring as a kick at all
#: (`kick_ratio` 0.63 at 50 ms). At 20 ms the air-band `decay_ratio` is 6.46,
#: which reads as a perfectly good closed hat, which is the failure.
BEATER_SECONDS = 0.12
BEATER_DECAY_SECONDS = 0.02

#: Peak of `_bright_kick`, below `KICK_PEAK` so a loud coincident hat can be
#: summed on top of it without `_hit_train` clipping. Band *shares* are
#: scale-invariant, so lowering it changes no measurement in this file.
BRIGHT_KICK_PEAK = 0.5

#: A hat loud enough that the air band holds as much energy as the noise band
#: over a `_bright_kick` window, and one that is not. The pair brackets
#: `KICK_BLEED_AIR_OVER_NOISE` and makes the rule's cost explicit instead of
#: leaving it to be discovered on a record: measured window
#: `air / (air + noise)` is 0.523 at the first and 0.313 at the second.
COINCIDENT_HAT_PEAK = 0.4
QUIET_COINCIDENT_HAT_PEAK = 0.2


def _bright_kick(seed: int = 21) -> np.ndarray:
    """`conftest._kick` with a real beater click on top of it.

    Measured in its own hit window, alone at 16th-note quarters:
    `kick_ratio` 0.811, `body_ratio` 0.006, `noise_ratio` 0.147,
    `air_ratio` 0.036, `air / (air + noise)` 0.195, air-band `decay_ratio`
    6.46 — a closed hat's decay, on a hit that is 81% kick.
    """
    rng = np.random.default_rng(seed)
    length = int(round(BEATER_SECONDS * ANALYSIS_SAMPLE_RATE))
    click = (
        _decay(length, BEATER_DECAY_SECONDS)
        * (
            _band_limited_noise(rng, length, 1000.0, 6000.0)
            + BEATER_AIR_WEIGHT * _band_limited_noise(rng, length, 6000.0, 16000.0)
        )
        * BEATER_SCALE
    )
    kick = _kick().astype(np.float64)
    kick[: click.size] += click
    return _normalised(kick, BRIGHT_KICK_PEAK)


def _bright_kick_pattern(hat_peak: float | None = None) -> np.ndarray:
    """`_bright_kick` on `DRUM_PATTERN_KICK_ONLY_STEPS`, optionally with a hat on top."""
    placements = [(time_s, _bright_kick()) for time_s in _step_times(DRUM_PATTERN_KICK_ONLY_STEPS)]
    if hat_peak is not None:
        hat = _normalised(_hat_closed().astype(np.float64), hat_peak)
        placements += [(time_s, hat) for time_s in _step_times(DRUM_PATTERN_KICK_ONLY_STEPS)]
    placements.sort(key=lambda item: item[0])
    return _hit_train(placements, PATTERN_FIXTURE_DURATION_SECONDS)


def _classified_candidates(audio: np.ndarray) -> tuple[list, list]:
    """Detection and classification only, stopping *before* the bleed rule.

    Reaches into the module's own helpers deliberately: the whole point of the
    pair of bleed tests is to show what the classifier does on its own and then
    what the rule does to it, and going through `decompose` can only show the
    second.
    """
    magnitude, freqs = drum_elements._stft_magnitude(audio, ANALYSIS_SAMPLE_RATE)
    envelopes = drum_elements._band_envelopes(magnitude, freqs)
    fluxes = {
        name: drum_elements._spectral_flux(envelope) for name, envelope in envelopes.items()
    }
    active, _dormant = drum_elements._active_bands(envelopes, fluxes)
    candidates = [
        drum_elements._Candidate(int(frame), band)
        for band in active
        for frame in drum_elements._pick_peaks(fluxes[band], ANALYSIS_SAMPLE_RATE)
    ]
    candidates.sort(key=lambda item: (item.frame, item.band))
    drum_elements._measure(candidates, magnitude, freqs, envelopes, ANALYSIS_SAMPLE_RATE)
    for candidate in candidates:
        candidate.scores = drum_elements._class_scores(candidate)
        candidate.drum, candidate.confidence = drum_elements._decide(candidate.scores)
    return candidates, active


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
    """The anchor is `beat_times[0]` **after** the phase snap, not before it.

    `grid_anchor_seconds` reports the anchor the steps were actually computed
    from, so it has to be the snapped one — reporting the caller's input beside
    steps derived from something else would make the record inconsistent with
    itself. The snap is bounded by half a step (0.0625 s here) and on this
    fixture moves 0.0092 s — under `WINDOW_HALF_SECONDS`, which is the real
    uncertainty on a hit time, because a frame centred before an onset still
    sees it through a 46 ms window and fluxes accordingly. See `_snap_anchor`.
    """
    result = _run(drum_pattern_120bpm)
    assert result.status == "ok"
    assert result.steps_per_cycle == DRUM_PATTERN_STEPS_PER_CYCLE
    assert result.cycle_seconds == pytest.approx(DRUM_PATTERN_CYCLE_SECONDS)
    assert result.grid_anchor_source == "beats"
    assert result.grid_anchor_seconds is not None
    assert result.grid_anchor_seconds == pytest.approx(
        DRUM_PATTERN_ANCHOR_SECONDS, abs=WINDOW_HALF_SECONDS
    )
    # And structurally bounded: a snap can never renumber a step.
    shift_steps = (
        abs(result.grid_anchor_seconds - DRUM_PATTERN_ANCHOR_SECONDS)
        / DRUM_PATTERN_STEP_SECONDS
    )
    assert shift_steps < 0.5
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
    assert result.grid_anchor_source in GRID_ANCHOR_SOURCES | SUPPLIED_ANCHOR_SOURCE
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


# ---------------------------------------------------------------------------
# Kick bleed: the same drum found twice
# ---------------------------------------------------------------------------
#
# These two tests pull in opposite directions on purpose and neither means
# anything alone. The first says a kick's own transient must not become a
# second hit; the second says a real hat sounding with a kick must still be two
# hits, which is the module's founding claim. A rule that satisfies one by
# giving up the other has not fixed anything.


def test_the_bright_kick_fixture_really_does_reproduce_the_failure() -> None:
    """Without the rule, a bare bright kick reports a hat on every kick.

    The premise of the pair below. If this ever stops failing, the fixture has
    drifted away from the material it was built to imitate and the two tests
    after it are testing nothing. Measured band shares are pinned here too, so
    a drift shows up as a number rather than as a silent pass.
    """
    candidates, active = _classified_candidates(_bright_kick_pattern())
    # Every band fires, which is the precondition the plain `_kick` cannot meet.
    assert set(active) == set(DETECTION_BANDS)

    air = [item for item in candidates if item.band == "air"]
    assert len(air) == len(DRUM_PATTERN_KICK_ONLY_STEPS) * DRUM_PATTERN_CYCLES == 16
    assert {item.drum for item in air} == {"hat"}, "the fixture no longer reproduces the bug"

    measured = air[0]
    assert measured.ratios["kick"] == pytest.approx(0.811, abs=5e-3)
    assert measured.ratios["noise"] == pytest.approx(0.147, abs=5e-3)
    assert measured.ratios["air"] == pytest.approx(0.036, abs=5e-3)
    assert drum_elements._air_over_noise(measured) == pytest.approx(0.195, abs=5e-3)
    # A closed hat's decay, which is exactly why the hat rule is fooled.
    assert measured.decay_ratio is not None
    assert measured.decay_ratio == pytest.approx(6.46, rel=0.02)


def test_a_bright_kick_alone_is_sixteen_kicks_and_nothing_else() -> None:
    """16 kicks, no hats. One drum struck once is one hit, however many bands hear it."""
    result = _run(_bright_kick_pattern())
    grouped = _by_class(result)
    assert len(grouped["kick"]) == 16
    assert grouped["hat"] == []
    assert grouped["snare"] == []
    assert grouped["unclassified"] == []
    assert {pattern.drum for pattern in result.patterns} == {"kick"}
    assert any("found a second time" in caveat for caveat in result.caveats)


def test_a_hat_over_a_bright_kick_is_still_two_hits() -> None:
    """The other direction, on the same kick that provokes the rule.

    `test_all_thirty_two_hats_survive_coincidence` already proves coincidence
    survives over the *plain* kick, where the kick contributes nothing to the
    bright bands and the question is easy. This asks it where it is hard: the
    kick has a beater click loud enough to fire the air detector on its own, and
    a genuine hat still has to come out as its own hit.
    """
    result = _run(_bright_kick_pattern(hat_peak=COINCIDENT_HAT_PEAK))
    grouped = _by_class(result)
    assert len(grouped["kick"]) == 16
    assert len(grouped["hat"]) == 16
    assert {hit.step for hit in grouped["hat"]} == set(DRUM_PATTERN_KICK_ONLY_STEPS)
    # Same instant, two hits — which is the whole design.
    for hat in grouped["hat"]:
        assert any(
            other.drum == "kick"
            and abs(other.time_seconds - hat.time_seconds)
            < THRESHOLDS["min_hit_separation_seconds"]
            for other in result.hits
        )


def test_a_hat_quieter_than_the_kicks_own_click_is_swallowed() -> None:
    """The documented cost of the rule, asserted rather than left to be discovered.

    `KICK_BLEED_AIR_OVER_NOISE` asks whether the bright half of a hit is
    weighted to the air side. A hat quiet enough that the kick's 1-6 kHz beater
    click still outweighs it fails that question and is suppressed with the
    bleed. Measured here: a hat at 0.4 against a kick at 0.5 survives
    (`air / (air + noise)` 0.523) and one at 0.2 does not (0.313).

    This is the same *shape* of cost as `_resolve_coincidences` rule 2's tom
    swallowed by a hat, and it is stated here so that a future change which
    makes it worse fails a test instead of quietly losing hats.
    """
    result = _run(_bright_kick_pattern(hat_peak=QUIET_COINCIDENT_HAT_PEAK))
    grouped = _by_class(result)
    assert len(grouped["kick"]) == 16
    assert grouped["hat"] == []


@pytest.mark.parametrize("fixture_name", sorted(_EXPECTED))
def test_the_bleed_rule_never_fires_on_the_synthetic_fixtures(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Zero suppressions on all four, so the v5 rule changed nothing there.

    Notably including `drum_pattern_120bpm`, where the kick band *does* fire on
    the snare's 200 Hz shell tone: co-detection alone would delete the hats on
    steps 4 and 12, and `KICK_BLEED_DOMINANCE` is the clause that stops it.
    """
    candidates, _active = _classified_candidates(request.getfixturevalue(fixture_name))
    before = [item.drum for item in candidates]
    assert drum_elements._suppress_kick_bleed(candidates, ANALYSIS_SAMPLE_RATE) == 0
    assert [item.drum for item in candidates] == before


def test_the_dominance_clause_is_what_spares_a_hat_over_a_snare(
    drum_pattern_120bpm: np.ndarray,
) -> None:
    """Name the mechanism, not just the outcome.

    The hats on steps 4 and 12 sit on a snare, are coincident with a kick-band
    detection, and measure `air / (air + noise)` 0.0526 — under the bleed
    threshold. The only thing keeping them is that their windows are 1.2% kick,
    fifty times under `KICK_BLEED_DOMINANCE`.
    """
    candidates, _active = _classified_candidates(drum_pattern_120bpm)
    separation = drum_elements._frames(
        THRESHOLDS["min_hit_separation_seconds"], ANALYSIS_SAMPLE_RATE
    )
    kick_frames = [item.frame for item in candidates if item.band == BLEED_SOURCE_BAND]
    over_snare = [
        item
        for item in candidates
        if item.band == "air"
        and item.drum == "hat"
        and any(abs(item.frame - frame) < separation for frame in kick_frames)
        and (item.ratios["body"] or 0.0) > 0.5
    ]
    assert len(over_snare) == 8
    for item in over_snare:
        assert drum_elements._air_over_noise(item) < KICK_BLEED_AIR_OVER_NOISE
        assert item.ratios["kick"] is not None
        assert item.ratios["kick"] < KICK_BLEED_DOMINANCE / 10.0


# ---------------------------------------------------------------------------
# Real material: the Madonna drums fixture
# ---------------------------------------------------------------------------
#
# Four per-frame band-energy arrays, no audio — see
# `tests/fixtures/real/PROVENANCE.md`. Everything that classifies a hit or
# fits a grid is a function of those four arrays, which is what makes this
# possible at all; `_decompose_bands` is the seam and its docstring says why.
#
# Ground truth, measured by the orchestrator independently of this module:
# 132.000 BPM exactly, 147 bars, kick on steps 0/4/8/12, a working downbeat at
# 1.6283 s (one of four equivalent bar phases on four-on-the-floor material).

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "real"
MADONNA_ENVELOPES = FIXTURE_DIR / "madonna__drums_band_envelopes.npz"

#: The verified tempo, as a beat period. Sequenced, not played.
MADONNA_BPM = 132.0
MADONNA_BEAT_PERIOD_SECONDS = 60.0 / MADONNA_BPM

#: A verified downbeat. Four-fold ambiguous by nature, and this is the phase
#: the orchestrator's ground truth quotes, so kick steps are asserted against
#: it and nothing else is.
MADONNA_DOWNBEAT_SECONDS = 1.6283

#: What v4 shipped, frozen at `calibration/v4/.../analysis/drums.json`: 1872
#: hits, 1240 of them hats, and `status: "no_grid"`. Quoted so the v5 numbers
#: below are a measured delta rather than a fresh assertion.
MADONNA_V4_HITS = 1872
MADONNA_V4_HATS = 1240
MADONNA_V4_BPM = 132.040405
MADONNA_V4_BEAT_TIME = 0.348299

#: The mix stem's tempo, which is what `strudel_hints.json` printed. Wrong by
#: 0.145 BPM, which is 3 sixteenth-steps of drift over this track.
MADONNA_MIX_BPM = 131.854843


@pytest.fixture(scope="module")
def madonna_envelopes() -> dict[str, np.ndarray]:
    """The committed drums-stem band envelopes, keyed as `DETECTION_BANDS` is."""
    with np.load(MADONNA_ENVELOPES, allow_pickle=True) as data:
        return {name: data[f"band_{name}"].astype(np.float64) for name in DETECTION_BANDS}


def _madonna(
    envelopes: dict[str, np.ndarray],
    *,
    beat_period_seconds: float | None = MADONNA_BEAT_PERIOD_SECONDS,
    downbeat_seconds: float | None = MADONNA_DOWNBEAT_SECONDS,
    bpm: float | None = None,
    beat_times: Sequence[float] = (),
) -> DrumDecomposition:
    return drum_elements._decompose_bands(
        envelopes,
        ANALYSIS_SAMPLE_RATE,
        bpm=bpm,
        beat_times=beat_times,
        beats_per_cycle=DRUM_PATTERN_BEATS_PER_CYCLE,
        beat_period_seconds=beat_period_seconds,
        downbeat_seconds=downbeat_seconds,
    )


def test_madonna_resolves_a_grid_at_the_corrected_period(
    madonna_envelopes: dict[str, np.ndarray],
) -> None:
    """`no_grid` in v4 at 132.040 BPM; a textbook grid at 132.000.

    Finding F1, closed. The allowance did not move: the same hits that scored
    0.2875 steps against 0.18 in v4 score 0.033 here, a factor of five inside
    it. What changed is the period and the anchor phase.
    """
    result = _madonna(madonna_envelopes)
    assert result.status == "ok"
    assert result.steps_per_cycle == 16
    assert result.cycle_seconds == pytest.approx(
        DRUM_PATTERN_BEATS_PER_CYCLE * MADONNA_BEAT_PERIOD_SECONDS
    )
    assert result.grid_anchor_source == "supplied"
    assert result.quantisation_error_steps is not None
    assert result.quantisation_error_steps == pytest.approx(0.0332, abs=5e-3)
    assert result.quantisation_error_steps < THRESHOLDS["max_quantisation_error_steps"]


def test_madonna_puts_the_kick_on_the_four_on_the_floor_steps(
    madonna_envelopes: dict[str, np.ndarray],
) -> None:
    """Kick on 0/4/8/12 above 0.9 occupancy, and off-grid leakage under 0.05.

    The grid `drum_elements` declared did not exist, asserted. `step_occupancy`
    is what carries the finding: the four steps read 0.906-0.953 and the twelve
    stray readings are all at or under 0.031, so the profile is a backbone with
    noise around it rather than a list of steps that happened to be touched.
    """
    kick = next(
        pattern for pattern in _madonna(madonna_envelopes).patterns if pattern.drum == "kick"
    )
    occupancy = dict(zip(kick.steps, kick.step_occupancy, strict=True))
    backbone = {step: value for step, value in occupancy.items() if value > 0.5}
    assert sorted(backbone) == [0, 4, 8, 12]
    assert min(backbone.values()) > 0.9
    assert max(value for step, value in occupancy.items() if step not in backbone) < 0.05


def test_madonna_hat_count_drops_by_the_duplicate_kick_detections(
    madonna_envelopes: dict[str, np.ndarray],
) -> None:
    """1240 hats in v4, 784 here, and the 456 that went were kicks found twice.

    The orchestrator's independent estimate was "roughly 736, which is eighth
    notes across the bars that are playing". 784 is 6.5% above that and is
    reported rather than tuned towards: the remainder is 16th-note decoration
    on steps 5, 11, 13 and 15, which the estimate did not count.

    What is *not* approximate is where the surviving hats sit: the offbeat
    eighths 2/6/10/14 carry 0.73-0.89 occupancy and the kick's own steps carry
    0.11 or less. W4C's bass notes land on the same four steps at the same
    grid, from a completely separate measurement.
    """
    result = _madonna(madonna_envelopes)
    grouped = _by_class(result)
    assert len(grouped["kick"]) == 487  # unchanged from v4: the rule never touches a kick
    assert len(grouped["hat"]) == 784
    assert len(grouped["hat"]) < MADONNA_V4_HATS
    assert len(result.hits) == 1422 < MADONNA_V4_HITS

    hat = next(pattern for pattern in result.patterns if pattern.drum == "hat")
    occupancy = dict(zip(hat.steps, hat.step_occupancy, strict=True))
    assert min(occupancy[step] for step in (2, 6, 10, 14)) > 0.7
    assert max(occupancy[step] for step in (0, 4, 8, 12)) < 0.15


def test_madonna_reports_the_suppression_rather_than_doing_it_quietly(
    madonna_envelopes: dict[str, np.ndarray],
) -> None:
    """Removing a quarter of a source's hits is a thing the reader is told."""
    caveats = _madonna(madonna_envelopes).caveats
    assert any("found a second time" in caveat for caveat in caveats)
    assert any("moved" in caveat and "phase" in caveat for caveat in caveats)


def test_madonna_never_emits_a_class_outside_the_schema(
    madonna_envelopes: dict[str, np.ndarray],
) -> None:
    """Including `clap`, which is deliberately absent — see `calibration/v5-progress.md`.

    Finding F2 proposed a clap class on steps 4 and 12. It did not survive
    verification: those hits carry 84% of their energy below 150 Hz, which is a
    kick, and this module now removes them rather than renaming them. The
    assertion is here so a future reading of Part 1 cannot quietly reintroduce
    it.
    """
    result = _madonna(madonna_envelopes)
    assert {hit.drum for hit in result.hits} <= DRUM_CLASSES
    assert "clap" not in DRUM_CLASSES
    assert "clap" not in {hit.drum for hit in result.hits}


def test_madonna_at_the_v4_period_reports_drift_rather_than_a_flat_refusal(
    madonna_envelopes: dict[str, np.ndarray],
) -> None:
    """The v4 failure, re-run, and now diagnosed.

    132.040 BPM is the drums stem's own estimate and is 0.040 BPM out. v4 said
    "no cycle grid" and stopped. The halves fit their own phases to 0.09 and
    0.08 steps and disagree by 0.37, which is a period error accumulating and
    not loose playing — so the caveat says so and names an implied period.
    """
    result = _madonna(
        madonna_envelopes,
        beat_period_seconds=None,
        downbeat_seconds=None,
        bpm=MADONNA_V4_BPM,
        beat_times=(MADONNA_V4_BEAT_TIME,),
    )
    assert result.status == "no_grid"
    assert result.steps_per_cycle is None
    assert result.quantisation_error_steps is not None
    assert result.quantisation_error_steps > THRESHOLDS["max_quantisation_error_steps"]
    drift = [caveat for caveat in result.caveats if "drifting" in caveat]
    assert len(drift) == 1
    assert "period error" in drift[0]
    assert "approximate" in drift[0]


def test_madonna_at_the_mix_tempo_reports_no_fit_rather_than_drift(
    madonna_envelopes: dict[str, np.ndarray],
) -> None:
    """131.855 BPM is 3 steps of drift over the track — past rescuing, and said so.

    The distinction task 5 of W4B asks for, in both directions: at 0.040 BPM
    out the halves still fit and the answer is "drifting"; at 0.145 BPM out
    they do not and the answer is "these hits do not fit any grid". Reporting
    the same sentence for both, as v4 did, loses the only actionable half.
    """
    result = _madonna(
        madonna_envelopes,
        beat_period_seconds=None,
        downbeat_seconds=None,
        bpm=MADONNA_MIX_BPM,
        beat_times=(MADONNA_V4_BEAT_TIME,),
    )
    assert result.status == "no_grid"
    assert not any("drifting" in caveat for caveat in result.caveats)
    assert any("do not fit any grid" in caveat for caveat in result.caveats)


def test_madonna_hits_do_not_depend_on_the_grid_at_all(
    madonna_envelopes: dict[str, np.ndarray],
) -> None:
    """Detection and classification are grid-free on real material too.

    The same claim `test_hits_are_unchanged_when_only_the_grid_input_changes`
    makes on 8.5 s of synthesis, made on 267 s of a record: a wrong tempo must
    not change which hits exist, only where they are said to sit.
    """
    right = _madonna(madonna_envelopes)
    wrong = _madonna(
        madonna_envelopes,
        beat_period_seconds=None,
        downbeat_seconds=None,
        bpm=MADONNA_MIX_BPM,
        beat_times=(MADONNA_V4_BEAT_TIME,),
    )
    assert [(hit.time_seconds, hit.drum) for hit in right.hits] == [
        (hit.time_seconds, hit.drum) for hit in wrong.hits
    ]


# ---------------------------------------------------------------------------
# The supplied period and downbeat
# ---------------------------------------------------------------------------


def test_a_supplied_period_beats_the_bpm_label(drum_pattern_120bpm: np.ndarray) -> None:
    """`beat_period_seconds` wins outright when both are given.

    Not a preference — a correctness requirement. `bpm` is a backend label
    accurate to about +/- 0.2 BPM and `beat_period_seconds` is a measurement;
    F1 is what happens when a grid is built from the first. Here the label is
    deliberately absurd and the grid must ignore it completely.
    """
    supplied = decompose(
        drum_pattern_120bpm,
        ANALYSIS_SAMPLE_RATE,
        bpm=97.3,
        beat_times=BEAT_TIMES,
        beats_per_cycle=DRUM_PATTERN_BEATS_PER_CYCLE,
        beat_period_seconds=60.0 / DRUM_PATTERN_BPM,
    )
    assert supplied.status == "ok"
    assert supplied.cycle_seconds == pytest.approx(DRUM_PATTERN_CYCLE_SECONDS)
    assert supplied.model_dump() == _run(drum_pattern_120bpm).model_dump()


def test_a_supplied_downbeat_beats_the_beat_times(drum_pattern_120bpm: np.ndarray) -> None:
    """`downbeat_seconds` wins, and the record says which source was used.

    `beat_times[0]` is a *beat*, not a downbeat, and on the Madonna fixture the
    two differ by three beats — enough to rotate every reported step by 12.
    Which one produced the anchor therefore has to be visible in the output.
    """
    shifted = decompose(
        drum_pattern_120bpm,
        ANALYSIS_SAMPLE_RATE,
        bpm=DRUM_PATTERN_BPM,
        beat_times=BEAT_TIMES,
        beats_per_cycle=DRUM_PATTERN_BEATS_PER_CYCLE,
        downbeat_seconds=DRUM_PATTERN_ANCHOR_SECONDS + 2 * DRUM_PATTERN_STEP_SECONDS,
    )
    assert shifted.grid_anchor_source == "supplied"
    assert shifted.status == "ok"
    # Two steps later an anchor, two steps earlier every step number.
    plain = {hit.step for hit in _run(drum_pattern_120bpm).hits}
    assert {hit.step for hit in shifted.hits} == {(step - 2) % 16 for step in plain}


@pytest.mark.parametrize("offset_steps", [-0.4, -0.2, 0.2, 0.4])
def test_the_anchor_snap_corrects_phase_without_renumbering_a_step(
    drum_pattern_120bpm: np.ndarray, offset_steps: float
) -> None:
    """A downbeat up to 0.4 steps out yields identical step numbers and a good fit.

    This is the property that makes accepting a supplied downbeat safe. The
    snap is the circular mean of the hits' fractional step positions, so it is
    bounded by half a step and cannot move a hit across a step boundary; what
    it *can* do is turn a 0.267-step phase error — the one the verified Madonna
    downbeat carries — from a rejected grid into a 0.033-step fit.
    """
    baseline = _run(drum_pattern_120bpm)
    nudged = decompose(
        drum_pattern_120bpm,
        ANALYSIS_SAMPLE_RATE,
        bpm=DRUM_PATTERN_BPM,
        beat_times=BEAT_TIMES,
        beats_per_cycle=DRUM_PATTERN_BEATS_PER_CYCLE,
        downbeat_seconds=DRUM_PATTERN_ANCHOR_SECONDS + offset_steps * DRUM_PATTERN_STEP_SECONDS,
    )
    assert nudged.status == "ok"
    assert [hit.step for hit in nudged.hits] == [hit.step for hit in baseline.hits]
    assert nudged.quantisation_error_steps == pytest.approx(
        baseline.quantisation_error_steps, abs=1e-9
    )
    assert nudged.grid_anchor_seconds == pytest.approx(baseline.grid_anchor_seconds, abs=1e-9)


def test_a_small_snap_is_applied_silently_and_a_large_one_is_reported(
    drum_pattern_120bpm: np.ndarray,
) -> None:
    """`ANCHOR_SNAP_CAVEAT_STEPS` is a tenth of a step — one STFT hop at 132 BPM.

    Below that there is nothing to tell a reader, because the move is smaller
    than the time resolution of the hits it was computed from.
    """
    big = decompose(
        drum_pattern_120bpm,
        ANALYSIS_SAMPLE_RATE,
        bpm=DRUM_PATTERN_BPM,
        beat_times=BEAT_TIMES,
        beats_per_cycle=DRUM_PATTERN_BEATS_PER_CYCLE,
        downbeat_seconds=DRUM_PATTERN_ANCHOR_SECONDS + 0.4 * DRUM_PATTERN_STEP_SECONDS,
    )
    assert any("moved" in caveat and "phase" in caveat for caveat in big.caveats)
    # The fixture's own anchor needs a 0.07-step move, which is under the floor.
    assert not any("moved" in caveat for caveat in _run(drum_pattern_120bpm).caveats)


def test_the_fold_refuses_material_with_hits_but_no_pulse() -> None:
    """200 uniformly random kicks over 60 s is not a grid, and the fold says which.

    The gate the per-hit quantisation error cannot be trusted to provide on its
    own, and the reason `_on_grid_share` exists. Measured: this material scores
    0.051 against a 0.50 floor and 0.25 chance, while every fixture with a real
    pattern scores 0.909 or better.
    """
    rng = np.random.default_rng(5)
    times = np.sort(rng.uniform(0.3, 59.0, 200))
    audio = _hit_train([(float(time_s), _kick()) for time_s in times], 60.0)
    result = decompose(
        audio,
        ANALYSIS_SAMPLE_RATE,
        bpm=DRUM_PATTERN_BPM,
        beat_times=[],
        beats_per_cycle=DRUM_PATTERN_BEATS_PER_CYCLE,
    )
    assert result.status == "no_grid"
    assert result.steps_per_cycle is None
    assert result.hits, "the hits themselves are still reported"
    assert any("do not fit any grid at this period" in caveat for caveat in result.caveats)


# ---------------------------------------------------------------------------
# Occupancy semantics
# ---------------------------------------------------------------------------


def test_step_occupancy_counts_the_cycles_a_class_was_playing_in() -> None:
    """A hat that plays for two cycles of four reads 1.0, not 0.5.

    The denominator is the choice that gives the field meaning. Counting cycles
    of the *file* would charge every element for the arrangement, and on a real
    track that is the difference between a kick reading 0.79 and reading 0.95
    on the same four-on-the-floor part.
    """
    hat = _hat_closed()
    placements = [(time_s, _kick()) for time_s in _step_times(DRUM_PATTERN_KICK_ONLY_STEPS)]
    placements += [
        (time_s, hat)
        for time_s in _step_times(DRUM_PATTERN_HAT_STEPS, cycles=2)
    ]
    placements.sort(key=lambda item: item[0])
    audio = _hit_train(placements, PATTERN_FIXTURE_DURATION_SECONDS)

    patterns = {pattern.drum: pattern for pattern in _run(audio).patterns}
    assert patterns["kick"].step_occupancy == pytest.approx([1.0] * 4)
    assert patterns["hat"].steps == list(DRUM_PATTERN_HAT_STEPS)
    assert patterns["hat"].step_occupancy == pytest.approx([1.0] * len(DRUM_PATTERN_HAT_STEPS))
    # It really did only play for half the source.
    assert patterns["hat"].hit_count == len(DRUM_PATTERN_HAT_STEPS) * 2


# ---------------------------------------------------------------------------
# Grid primitives
# ---------------------------------------------------------------------------


def test_fold_abstains_below_two_cycles() -> None:
    """A median across one cycle is that cycle, so the fold says nothing instead."""
    flux = np.ones(200, dtype=np.float64)
    one_cycle = 200 * STFT_HOP_LENGTH / ANALYSIS_SAMPLE_RATE
    assert drum_elements._fold(flux, ANALYSIS_SAMPLE_RATE, one_cycle, 0.0, 16) is None
    folded = drum_elements._fold(flux, ANALYSIS_SAMPLE_RATE, one_cycle / MIN_FOLD_CYCLES, 0.0, 16)
    assert folded is not None
    assert folded.shape == (MIN_FOLD_CYCLES, 16)


def test_fold_assigns_a_frame_to_its_nearest_slot_not_the_one_below() -> None:
    """`floor` reads the whole profile one slot early; that is how the Madonna
    kick first appeared on steps 3/7/11/15 instead of 0/4/8/12."""
    cycle = 1.0
    slot = cycle / 16
    flux = np.zeros(400, dtype=np.float64)
    # One frame a hair *before* a step boundary, which floor would misfile.
    frame = int(round((4 * slot - 0.2 * slot) * ANALYSIS_SAMPLE_RATE / STFT_HOP_LENGTH))
    flux[frame] = 5.0
    folded = drum_elements._fold(flux, ANALYSIS_SAMPLE_RATE, cycle, 0.0, 16)
    assert folded is not None
    assert int(np.argmax(folded[0])) == 4


def test_profile_contrast_is_a_sparsity_measure_and_is_documented_as_one() -> None:
    """Flat scores 0, a single spike scores nearly 1 — and random hits score high.

    Pinned because the number looks like a quality score and is not one. The
    module uses it to pick between 16 and 12 steps, never to decide whether a
    grid exists; `_on_grid_share` does that.
    """
    flat = np.ones(16, dtype=np.float64)
    assert drum_elements._profile_contrast(flat) == pytest.approx(0.0)

    spike = np.zeros(16, dtype=np.float64)
    spike[0] = 1.0
    assert drum_elements._profile_contrast(spike) == pytest.approx(15 / 16)

    assert drum_elements._profile_contrast(np.zeros(16, dtype=np.float64)) == 0.0


def test_on_grid_share_is_chance_on_a_flat_profile_and_one_on_a_locked_one() -> None:
    """The periodicity test, at both ends and on nothing."""
    flat = np.ones(16 * GRID_OVERSAMPLE, dtype=np.float64)
    assert drum_elements._on_grid_share(flat, GRID_OVERSAMPLE) == pytest.approx(
        1.0 / GRID_OVERSAMPLE
    )

    locked = np.zeros(16 * GRID_OVERSAMPLE, dtype=np.float64)
    locked[::GRID_OVERSAMPLE] = 1.0
    assert drum_elements._on_grid_share(locked, GRID_OVERSAMPLE) == pytest.approx(1.0)

    assert drum_elements._on_grid_share(np.zeros(64, dtype=np.float64), GRID_OVERSAMPLE) is None


def test_snap_anchor_is_bounded_by_half_a_step_whatever_the_hits_do() -> None:
    """The structural guarantee: a snap can never renumber a step.

    Tested on hits placed at every fractional offset, including the pathological
    half-step case where the circular mean is genuinely undefined.
    """
    cycle, steps_per_cycle = 2.0, 16
    step = cycle / steps_per_cycle
    for offset in np.linspace(-0.9, 0.9, 19):
        times = np.arange(16, dtype=np.float64) * step + offset * step
        _anchor, shift = drum_elements._snap_anchor(times, cycle, 0.0, steps_per_cycle)
        assert -0.5 <= shift <= 0.5

    scattered = np.array([0.0, 0.5 * step, step, 1.5 * step], dtype=np.float64)
    _anchor, shift = drum_elements._snap_anchor(scattered, cycle, 0.0, steps_per_cycle)
    assert -0.5 <= shift <= 0.5


def test_snap_anchor_does_not_rescue_a_wrong_period(drum_pattern_120bpm: np.ndarray) -> None:
    """0.2464 steps of error before the snap, 0.2473 after — very slightly worse.

    The guard against reading the snap as a way of forcing a fit. A wrong period
    spreads its residuals rather than offsetting them, so there is no phase for
    a circular mean to find.
    """
    times = np.asarray(
        [hit.time_seconds for hit in _run(drum_pattern_120bpm).hits], dtype=np.float64
    )
    cycle = DRUM_PATTERN_BEATS_PER_CYCLE * 60.0 / 97.3
    before = drum_elements._mean_error(times, cycle, float(times[0]), 16)
    anchor, _shift = drum_elements._snap_anchor(times, cycle, float(times[0]), 16)
    after = drum_elements._mean_error(times, cycle, anchor, 16)
    assert before == pytest.approx(0.2464, abs=5e-3)
    assert after == pytest.approx(0.2473, abs=5e-3)
    assert after > THRESHOLDS["max_quantisation_error_steps"]


def test_cycle_seconds_prefers_the_measurement_and_rejects_nonsense() -> None:
    assert drum_elements._cycle_seconds(120.0, None, 4) == pytest.approx(2.0)
    assert drum_elements._cycle_seconds(97.3, 0.5, 4) == pytest.approx(2.0)
    assert drum_elements._cycle_seconds(None, None, 4) is None
    assert drum_elements._cycle_seconds(0.0, None, 4) is None
    assert drum_elements._cycle_seconds(float("nan"), None, 4) is None
    assert drum_elements._cycle_seconds(120.0, None, 0) is None


# ---------------------------------------------------------------------------
# The v5 thresholds, and the interface `tempo.py` depends on
# ---------------------------------------------------------------------------


def test_the_bleed_threshold_is_not_the_hat_threshold() -> None:
    """Two thresholds on one descriptor, and collapsing them breaks a fixture.

    `hat_air_over_noise` runs on every hit and is held down at 0.015 by a closed
    hat sounding with a snare, measured at 0.0552. `kick_bleed_air_over_noise`
    only runs where a kick has already been found and can afford to ask a much
    harder question. Raising the first to the second's value would delete the 16
    hats on steps 4 and 12 of `drum_pattern_120bpm`.
    """
    assert KICK_BLEED_AIR_OVER_NOISE > THRESHOLDS["hat_air_over_noise"] * 30
    assert KICK_BLEED_DOMINANCE == THRESHOLDS["kick_low_ratio"]
    assert BLEED_SOURCE_BAND not in BLEED_TARGET_BANDS
    assert set(BLEED_TARGET_BANDS) < set(DETECTION_BANDS)
    # The bands the kick bleeds into are above it, never below.
    order = list(DETECTION_BANDS)
    assert all(order.index(band) > order.index(BLEED_SOURCE_BAND) for band in BLEED_TARGET_BANDS)


def test_the_grid_gates_sit_where_the_measurements_say_they_do() -> None:
    """The two thresholds whose values are the whole of the v5 grid fix."""
    # Twice chance, and chance is what a flat profile scores.
    assert GRID_ON_GRID_SHARE_MIN == pytest.approx(2.0 / GRID_OVERSAMPLE)
    # The allowance did NOT move. `KICKOFF-v2.md` calls this out by name.
    assert THRESHOLDS["max_quantisation_error_steps"] == 0.18


def test_every_new_threshold_is_in_the_thresholds_dict() -> None:
    """The module's own convention: a constant with no documented entry is a future bug."""
    exported = {
        "kick_bleed_dominance": KICK_BLEED_DOMINANCE,
        "kick_bleed_air_over_noise": KICK_BLEED_AIR_OVER_NOISE,
        "grid_on_grid_share_min": GRID_ON_GRID_SHARE_MIN,
        "anchor_snap_caveat_steps": drum_elements.ANCHOR_SNAP_CAVEAT_STEPS,
    }
    for key, value in exported.items():
        assert THRESHOLDS[key] == value


def test_the_helpers_tempo_py_imports_keep_their_names_and_shapes() -> None:
    """`tempo.py` imports these four rather than reimplementing them.

    That is the right call — it is what guarantees frame *k* means the same
    instant in both modules, and it is the convention `tools/make-fixtures/`
    already follows — but it makes them a shared interface. A rename or a
    signature change here breaks W4A silently, so it fails here loudly instead.

    Checked by origin and shape rather than by object identity:
    `test_runs_with_neither_librosa_nor_essentia_importable` reloads this
    module, which replaces its function objects while `tempo` still holds the
    originals. An `is` comparison would then fail for a reason that has nothing
    to do with the interface.
    """
    import inspect

    from audio_pipeline import tempo

    for name in ("_stft_magnitude", "_band_envelope", "_spectral_flux"):
        borrowed = getattr(tempo, name)
        assert borrowed.__module__ == drum_elements.__name__, name
        assert borrowed.__qualname__ == name
        assert inspect.signature(borrowed) == inspect.signature(
            getattr(drum_elements, name)
        ), name

    assert tempo.STFT_HOP_LENGTH == STFT_HOP_LENGTH
    assert list(inspect.signature(drum_elements._stft_magnitude).parameters) == [
        "audio",
        "sample_rate",
    ]
    assert list(inspect.signature(drum_elements._band_envelope).parameters) == [
        "magnitude",
        "freqs",
        "low_hz",
        "high_hz",
        "include_high",
    ]
    assert list(inspect.signature(drum_elements._spectral_flux).parameters) == ["envelope"]


def test_a_supplied_anchor_needs_a_vocabulary_entry_w6_has_not_added_yet(
    drum_pattern_120bpm: np.ndarray,
) -> None:
    """Documents the one schema change this package needs, as a failing-when-fixed test.

    `decompose` emits `grid_anchor_source="supplied"` when the caller hands in a
    downbeat. `GRID_ANCHOR_SOURCES` does not contain it, because `schemas.py` is
    frozen and only W6 may touch it. Pydantic does not validate the field, so
    nothing breaks today — but nothing would notice either, which is what this
    test is for.
    """
    result = decompose(
        drum_pattern_120bpm,
        ANALYSIS_SAMPLE_RATE,
        bpm=DRUM_PATTERN_BPM,
        beat_times=BEAT_TIMES,
        beats_per_cycle=DRUM_PATTERN_BEATS_PER_CYCLE,
        downbeat_seconds=DRUM_PATTERN_ANCHOR_SECONDS,
    )
    assert result.grid_anchor_source == "supplied"
    assert SUPPLIED_ANCHOR_SOURCE - GRID_ANCHOR_SOURCES == {"supplied"}, (
        "W6 has added `supplied` to GRID_ANCHOR_SOURCES — drop SUPPLIED_ANCHOR_SOURCE "
        "and assert against the frozenset directly"
    )


# ---------------------------------------------------------------------------
# The limit: a kick this module cannot find, and says so
# ---------------------------------------------------------------------------


def _reverberant_kit(decay_seconds: float = 0.5, drive: float = 6.0) -> np.ndarray:
    """`drum_pattern_120bpm`'s kit through a long room and a compressor.

    Reproduces the signature of "When the Levee Breaks", where this module
    reports no kick at all: measured kick-band sparsity 0.697 against that
    stem's 0.654 and a `FLUX_SPARSITY_MIN` of 0.72, with `q90 / peak` 0.089
    against 0.114. The two ingredients are both necessary — the room stops the
    band falling back between hits and the compressor lifts what is left of the
    gaps — and neither alone gets under the gate.
    """
    dry = _drum_pattern(
        kick_steps=(0, 8), snare_steps=(4, 12), hat_steps=DRUM_PATTERN_HAT_STEPS
    ).astype(np.float64)
    length = int(2.0 * ANALYSIS_SAMPLE_RATE)
    impulse = np.exp(-np.arange(length) / ANALYSIS_SAMPLE_RATE / decay_seconds)
    impulse = impulse * np.random.default_rng(31).standard_normal(length)
    impulse[0] += 6.0
    wet = np.convolve(dry, impulse)[: dry.size]
    return _normalised(np.tanh(wet / np.abs(wet).max() * drive), KICK_PEAK)


def test_a_reverberant_compressed_kit_is_refused_rather_than_invented() -> None:
    """No hits, and a caveat that names the loud band it could not read.

    The honest limit, pinned. Flux peak-picking needs a gap between one rise and
    the next; a stairwell and a compressor remove it. Opening the band anyway —
    measured on the real stem — yields 702 "kicks" that fold onto 13 of 16 steps
    at a quantisation error of 0.247, which is what uniformly random hits score
    by construction. Reporting nothing is the correct answer and this test
    exists so that a future sensitivity increase has to break it deliberately.
    """
    audio = _reverberant_kit()
    magnitude, freqs = drum_elements._stft_magnitude(audio, ANALYSIS_SAMPLE_RATE)
    envelopes = drum_elements._band_envelopes(magnitude, freqs)
    flux = drum_elements._spectral_flux(envelopes["kick"])

    # Loud, and full of transients — it fails neither of the cheap tests.
    assert float(np.sum(envelopes["kick"])) / sum(
        float(np.sum(value)) for value in envelopes.values()
    ) > THRESHOLDS["band_activity_floor"] * 100
    assert float(flux.max()) > THRESHOLDS["flux_peak_floor"]
    # It fails on density, and the reason is reported as its own value.
    assert drum_elements._is_percussive(flux) == "not_sparse"

    result = _run(audio)
    assert result.hits == []
    assert result.status == "too_few_hits"
    caveat = next(c for c in result.caveats if "never separate" in c)
    assert "kick" in caveat
    assert "missing from the counts" in caveat


def test_the_three_reasons_a_band_is_skipped_are_reported_separately(
    drum_pattern_kick_only: np.ndarray,
) -> None:
    """"Holds nothing" and "holds more than I can read" are different findings.

    v5 first shipped one sentence for both. On the Levee stem the unsearched
    kick band carries **37% of the source's energy**, and a reader told it held
    "nothing, or nothing transient" would go looking for a silent band and find
    the loudest one there is.
    """
    assert set(drum_elements._DORMANT_CAVEATS) == {"empty", "no_transient", "not_sparse"}

    # `drum_pattern_kick_only`: the bright bands are float residue under a sine.
    quiet = _run(drum_pattern_kick_only)
    assert any("residue rather than content" in caveat for caveat in quiet.caveats)
    assert not any("never separate" in caveat for caveat in quiet.caveats)

    # One caveat per distinct reason, never one per band, and it names its bands.
    pair = drum_elements._dormant_caveats({"noise": "empty", "air": "empty"})
    assert len(pair) == 1
    assert pair[0].endswith("noise, air")
    both = drum_elements._dormant_caveats({"kick": "not_sparse", "air": "empty"})
    assert len(both) == 2
