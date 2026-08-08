"""Unit tests for the heuristic labelling layer.

No audio, no fixtures, no backends: `SourceAnalysis` objects are hand-built with
synthetic descriptor values, which is the whole point of keeping `heuristics.py`
pure. Descriptor values are chosen to land on exact ramp fractions (usually 0.5)
so a threshold change breaks the test loudly rather than drifting quietly.

Threshold convention under test: **thresholds are inclusive and score 0.0**. A
descriptor exactly on its threshold fires its label with confidence 0.0; a
descriptor short of the threshold fires nothing.
"""

from __future__ import annotations

import math

import pytest

from audio_pipeline.heuristics import (
    THRESHOLDS,
    _dedupe,
    _ramp,
    _window,
    apply,
    chroma_entropy,
    label_bass,
    label_drums,
    label_generic,
    label_other,
    label_vocals,
)
from audio_pipeline.schemas import (
    BandEnergyRatios,
    DynamicsFeatures,
    HeuristicLabel,
    RhythmFeatures,
    SourceAnalysis,
    SpectralFeatures,
    TonalFeatures,
)


def chroma_with_entropy(target: float, bias: str = "nearest") -> list[float]:
    """A 12-bin chroma vector whose normalised entropy is `target`.

    Lets the measured tables below record the one number the labels actually
    read — chroma entropy — instead of 12 raw bins per signal per backend.
    Mass `q` sits on one pitch class and the rest spreads evenly over the other
    eleven; entropy falls monotonically as `q` rises, so bisection inverts it.

    `bias` controls which side of the target to land on when the exact value is
    not representable, which matters only for boundary tests: "at_least"
    guarantees entropy >= target, "at_most" guarantees entropy <= target, and
    the default "nearest" splits the final bracket. Without it, an
    exactly-on-the-threshold test is a coin flip on the last bit.
    `test_chroma_with_entropy_round_trips` pins the inversion.
    """
    if target >= 1.0:
        return [1.0] * 12
    if target <= 0.0:
        return [1.0] + [0.0] * 11

    def vector_for(q: float) -> list[float]:
        return [q] + [(1.0 - q) / 11.0] * 11

    # Entropy decreases as q rises, so `low` always sits above the target and
    # `high` always at or below it.
    low, high = 1.0 / 12.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if mid <= low or mid >= high:
            break  # bracket has collapsed to adjacent floats
        entropy = chroma_entropy(vector_for(mid))
        assert entropy is not None
        if entropy > target:
            low = mid
        else:
            high = mid
    if bias == "at_least":
        return vector_for(low)
    if bias == "at_most":
        return vector_for(high)
    return vector_for((low + high) / 2.0)


def make_analysis(
    source: str = "mix",
    *,
    onset_density: float | None = None,
    transient_sharpness: float | None = None,
    key_confidence: float | None = None,
    tonal_stability: float | None = None,
    chroma_entropy_value: float | None = None,
    chroma_bias: str = "nearest",
    hpcp_mean: list[float] | None = None,
    centroid_mean: float | None = None,
    centroid_energy_hz: float | None = None,
    centroid_std: float | None = None,
    band_low: float | None = None,
    band_high: float | None = None,
    crest_factor: float | None = None,
    rms_mean: float | None = None,
    loudness_lufs: float | None = None,
) -> SourceAnalysis:
    """Build a `SourceAnalysis` with only the descriptors a test cares about.

    Everything left unset stays `None` (or, for `hpcp_mean`, empty), which is
    exactly the shape a backend produces when a feature is unavailable. Pass
    `chroma_entropy_value` to have the 12-bin chroma synthesised to match.
    """
    if hpcp_mean is None:
        hpcp_mean = (
            []
            if chroma_entropy_value is None
            else chroma_with_entropy(chroma_entropy_value, chroma_bias)
        )
    return SourceAnalysis(
        source=source,
        audio_path=f"output/demo/stems/{source}.wav",
        duration_seconds=30.0,
        sample_rate=44100,
        backend="fake",
        rhythm=RhythmFeatures(
            onset_density=onset_density,
            transient_sharpness=transient_sharpness,
        ),
        tonal=TonalFeatures(
            key_confidence=key_confidence,
            tonal_stability=tonal_stability,
            hpcp_mean=hpcp_mean,
        ),
        spectral=SpectralFeatures(
            centroid_mean=centroid_mean,
            centroid_energy_hz=centroid_energy_hz,
            centroid_std=centroid_std,
            band_energy_ratios=BandEnergyRatios(low=band_low, high=band_high),
        ),
        dynamics=DynamicsFeatures(
            crest_factor=crest_factor, rms_mean=rms_mean, loudness_lufs=loudness_lufs
        ),
    )


def names(labels: list[HeuristicLabel]) -> set[str]:
    return {label.label for label in labels}


def by_name(labels: list[HeuristicLabel], name: str) -> HeuristicLabel:
    matches = [label for label in labels if label.label == name]
    assert matches, f"expected label {name!r} in {sorted(names(labels))}"
    assert len(matches) == 1, f"label {name!r} appeared {len(matches)} times"
    return matches[0]


# ---------------------------------------------------------------------------
# Cross-agent contract
# ---------------------------------------------------------------------------


def test_thresholds_keys_shared_with_strudel_hints_survive() -> None:
    """W1E imports these two by name. Values may be retuned; keys may not move."""
    assert "busy_drums_onsets_per_sec" in THRESHOLDS
    assert "sparse_onsets_per_sec" in THRESHOLDS
    assert THRESHOLDS["sparse_onsets_per_sec"] < THRESHOLDS["busy_drums_onsets_per_sec"]


def test_thresholds_are_all_floats() -> None:
    assert all(isinstance(value, float) for value in THRESHOLDS.values())


# (threshold key, saturation key, "up" if higher is more confident else "down").
# `_ramp` infers direction from the pair, so moving a threshold past its
# saturation silently inverts the label rather than failing. This table pins the
# intent so that mistake fails loudly instead.
RAMP_DIRECTIONS: list[tuple[str, str, str]] = [
    ("silence_floor_lufs", "silence_floor_lufs_saturation", "down"),
    ("silence_floor_rms", "silence_floor_rms_saturation", "down"),
    ("sparse_onsets_per_sec", "sparse_onsets_saturation", "down"),
    ("moderate_onsets_per_sec", "percussive_onsets_saturation", "up"),
    ("moderate_onsets_per_sec", "sustained_onsets_saturation", "down"),
    ("busy_drums_onsets_per_sec", "busy_drums_onsets_saturation", "up"),
    ("percussive_crest_factor", "percussive_crest_saturation", "up"),
    ("sustained_crest_factor", "sustained_crest_saturation", "down"),
    ("transient_bright_centroid_hz", "transient_bright_centroid_saturation", "up"),
    ("dark_centroid_hz", "dark_centroid_saturation", "down"),
    ("noisy_centroid_hz", "noisy_centroid_saturation", "up"),
    (
        "tonally_stable_max_chroma_entropy",
        "tonally_stable_chroma_entropy_saturation",
        "down",
    ),
    ("noisy_min_chroma_entropy", "noisy_chroma_entropy_saturation", "up"),
    ("kick_heavy_low_ratio", "kick_heavy_low_saturation", "up"),
    ("bright_hats_high_ratio", "bright_hats_high_saturation", "up"),
    ("bright_plucks_high_ratio", "bright_plucks_high_saturation", "up"),
    ("processed_vocal_high_ratio", "processed_vocal_high_saturation", "up"),
    ("sharp_transient_ratio", "sharp_transient_saturation", "up"),
    ("vocal_presence_rms", "vocal_presence_rms_saturation", "up"),
    ("processed_centroid_std_hz", "processed_centroid_std_saturation", "up"),
]


@pytest.mark.parametrize(("threshold_key", "saturation_key", "direction"), RAMP_DIRECTIONS)
def test_every_threshold_pair_ramps_in_its_intended_direction(
    threshold_key: str, saturation_key: str, direction: str
) -> None:
    threshold = THRESHOLDS[threshold_key]
    saturation = THRESHOLDS[saturation_key]
    assert threshold != saturation, f"{threshold_key} has no gradient"
    if direction == "up":
        assert saturation > threshold, f"{threshold_key} inverted: would fire on low values"
    else:
        assert saturation < threshold, f"{threshold_key} inverted: would fire on high values"


def test_ramp_direction_table_covers_every_saturation_threshold() -> None:
    """A new saturation constant must declare its direction."""
    declared = {saturation_key for _, saturation_key, _ in RAMP_DIRECTIONS}
    actual = {key for key in THRESHOLDS if key.endswith("_saturation")}
    assert actual == declared


def test_window_thresholds_do_not_cross() -> None:
    """Window margins must stay inside half the window, or the two edge ramps
    conflict and the label becomes unreachable in the middle."""
    for lower_key, upper_key, margin_key in (
        ("vocal_centroid_min_hz", "vocal_centroid_max_hz", "vocal_centroid_margin_hz"),
        ("sparse_onsets_per_sec", "busy_drums_onsets_per_sec", "vocal_onset_margin_per_sec"),
    ):
        lower, upper, margin = (THRESHOLDS[k] for k in (lower_key, upper_key, margin_key))
        assert lower < upper, f"{lower_key} must sit below {upper_key}"
        assert 0.0 < margin <= (upper - lower) / 2, f"{margin_key} too wide for its window"


# ---------------------------------------------------------------------------
# Chroma entropy
# ---------------------------------------------------------------------------


def test_chroma_entropy_is_zero_for_a_single_pitch_class() -> None:
    assert chroma_entropy([1.0] + [0.0] * 11) == 0.0


def test_chroma_entropy_is_one_for_a_perfectly_flat_chroma() -> None:
    assert chroma_entropy([1.0] * 12) == 1.0
    # Scale must not matter — only the shape of the distribution. Exact equality
    # is not guaranteed once the bins are not 1.0: the division rounds.
    assert chroma_entropy([0.004] * 12) == pytest.approx(1.0)


def test_chroma_entropy_rises_as_energy_spreads() -> None:
    one = chroma_entropy([1.0] + [0.0] * 11)
    three = chroma_entropy([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    six = chroma_entropy([1.0] * 6 + [0.0] * 6)
    twelve = chroma_entropy([1.0] * 12)
    assert one is not None and three is not None and six is not None and twelve is not None
    assert one < three < six < twelve


def test_chroma_entropy_of_n_equal_bins_matches_log_n_over_log_12() -> None:
    """The closed form the normalisation is derived from."""
    for count in (1, 2, 3, 4, 6, 12):
        vector = [1.0] * count + [0.0] * (12 - count)
        assert chroma_entropy(vector) == pytest.approx(math.log(count) / math.log(12))


def test_chroma_entropy_ignores_bin_order() -> None:
    peaked = [0.9, 0.1] + [0.0] * 10
    rotated = [0.0] * 5 + [0.9, 0.1] + [0.0] * 5
    assert chroma_entropy(peaked) == pytest.approx(chroma_entropy(rotated))


@pytest.mark.parametrize(
    "vector",
    [
        pytest.param([], id="empty"),
        pytest.param([1.0] * 11, id="eleven-bins"),
        pytest.param([1.0] * 13, id="thirteen-bins"),
        pytest.param([0.0] * 12, id="sums-to-zero"),
        pytest.param([float("nan")] + [1.0] * 11, id="nan"),
        pytest.param([float("inf")] + [1.0] * 11, id="inf"),
    ],
)
def test_chroma_entropy_returns_none_for_unusable_input(vector: list[float]) -> None:
    assert chroma_entropy(vector) is None


def test_chroma_entropy_stays_within_bounds_on_extreme_input() -> None:
    for vector in ([1e-300] * 12, [1e300] + [1e-300] * 11, [1.0] + [1e-12] * 11):
        value = chroma_entropy(vector)
        assert value is not None
        assert 0.0 <= value <= 1.0


def test_chroma_with_entropy_round_trips() -> None:
    """The test helper that inverts entropy must be faithful, or every measured
    row below would be testing the wrong number."""
    for target in (0.0, 0.0868, 0.35, 0.5286, 0.75, 0.9054, 0.9954, 1.0):
        assert chroma_entropy(chroma_with_entropy(target)) == pytest.approx(target, abs=1e-6)


def test_labels_do_not_fire_when_chroma_is_missing() -> None:
    """An empty `hpcp_mean` must degrade to no label, not to entropy 0.0 —
    which would otherwise read as 'maximally tonal'."""
    labels = apply(make_analysis("mix", centroid_energy_hz=9000.0))
    assert "tonally stable" not in names(labels)
    assert "noisy" not in names(labels)


# ---------------------------------------------------------------------------
# The confidence ramp
# ---------------------------------------------------------------------------


def test_ramp_below_threshold_does_not_fire() -> None:
    assert _ramp(3.9, 4.0, 8.0) is None


def test_ramp_exactly_at_threshold_fires_at_zero_confidence() -> None:
    """The documented convention: inclusive threshold, zero confidence."""
    assert _ramp(4.0, 4.0, 8.0) == 0.0


def test_ramp_is_graded_between_threshold_and_saturation() -> None:
    assert _ramp(5.0, 4.0, 8.0) == pytest.approx(0.25)
    assert _ramp(6.0, 4.0, 8.0) == pytest.approx(0.5)
    assert _ramp(7.0, 4.0, 8.0) == pytest.approx(0.75)


def test_ramp_saturates_at_one_and_never_exceeds_it() -> None:
    assert _ramp(8.0, 4.0, 8.0) == 1.0
    assert _ramp(1_000_000.0, 4.0, 8.0) == 1.0


def test_ramp_handles_the_descending_direction() -> None:
    """`saturation < threshold` means lower values are more confident."""
    assert _ramp(4.1, 4.0, 2.0) is None
    assert _ramp(4.0, 4.0, 2.0) == 0.0
    assert _ramp(3.0, 4.0, 2.0) == pytest.approx(0.5)
    assert _ramp(2.0, 4.0, 2.0) == 1.0
    assert _ramp(-50.0, 4.0, 2.0) == 1.0


def test_ramp_passes_none_through() -> None:
    assert _ramp(None, 4.0, 8.0) is None


def test_ramp_rejects_a_zero_width_gradient() -> None:
    with pytest.raises(ValueError, match="gradient"):
        _ramp(4.0, 4.0, 4.0)


def test_window_is_inclusive_at_both_edges_and_confident_inside() -> None:
    assert _window(299.0, 300.0, 3000.0, 400.0) is None
    assert _window(300.0, 300.0, 3000.0, 400.0) == 0.0
    assert _window(3000.0, 300.0, 3000.0, 400.0) == 0.0
    assert _window(3001.0, 300.0, 3000.0, 400.0) is None
    assert _window(500.0, 300.0, 3000.0, 400.0) == pytest.approx(0.5)
    assert _window(1500.0, 300.0, 3000.0, 400.0) == 1.0
    assert _window(None, 300.0, 3000.0, 400.0) is None


# ---------------------------------------------------------------------------
# Generic labels
# ---------------------------------------------------------------------------


def test_percussive_fires_with_graded_confidence_and_evidence() -> None:
    labels = label_generic(make_analysis(crest_factor=14.0, onset_density=4.0))
    percussive = by_name(labels, "percussive")
    assert percussive.confidence == pytest.approx(0.5)
    assert percussive.evidence == {
        "crest_factor": 14.0,
        "onset_density": 4.0,
        "min_crest_factor": THRESHOLDS["percussive_crest_factor"],
        "min_onset_density": THRESHOLDS["moderate_onsets_per_sec"],
    }


def test_percussive_boundary_either_side_and_exactly_on_the_line() -> None:
    crest = THRESHOLDS["percussive_crest_factor"]
    onsets = THRESHOLDS["moderate_onsets_per_sec"]

    on_line = label_generic(make_analysis(crest_factor=crest, onset_density=onsets))
    assert by_name(on_line, "percussive").confidence == 0.0

    just_under_crest = label_generic(make_analysis(crest_factor=crest - 0.01, onset_density=onsets))
    assert "percussive" not in names(just_under_crest)

    just_under_onsets = label_generic(
        make_analysis(crest_factor=crest, onset_density=onsets - 0.01)
    )
    assert "percussive" not in names(just_under_onsets)

    well_over = label_generic(make_analysis(crest_factor=40.0, onset_density=20.0))
    assert by_name(well_over, "percussive").confidence == 1.0


def test_percussive_needs_both_inputs() -> None:
    assert "percussive" not in names(label_generic(make_analysis(crest_factor=14.0)))
    assert "percussive" not in names(label_generic(make_analysis(onset_density=4.0)))


def test_sustained_fires_on_low_crest_and_low_onset_density() -> None:
    labels = label_generic(make_analysis(crest_factor=2.8, onset_density=1.1))
    sustained = by_name(labels, "sustained")
    assert sustained.confidence == pytest.approx(0.5)
    assert sustained.evidence["crest_factor"] == 2.8
    assert sustained.evidence["max_crest_factor"] == THRESHOLDS["sustained_crest_factor"]
    assert "percussive" not in names(labels)


def test_sustained_boundary_either_side_and_exactly_on_the_line() -> None:
    crest = THRESHOLDS["sustained_crest_factor"]
    onsets = THRESHOLDS["moderate_onsets_per_sec"]

    on_line = label_generic(make_analysis(crest_factor=crest, onset_density=onsets))
    assert by_name(on_line, "sustained").confidence == 0.0

    just_over = label_generic(make_analysis(crest_factor=crest + 0.01, onset_density=onsets))
    assert "sustained" not in names(just_over)

    just_over_onsets = label_generic(make_analysis(crest_factor=crest, onset_density=onsets + 0.01))
    assert "sustained" not in names(just_over_onsets)


def test_noisy_fires_on_bright_and_flat_chroma() -> None:
    labels = label_generic(make_analysis(centroid_energy_hz=5250.0, chroma_entropy_value=0.965))
    noisy = by_name(labels, "noisy")
    assert noisy.confidence == pytest.approx(0.5, abs=1e-3)
    assert noisy.evidence["centroid_energy_hz"] == 5250.0
    assert noisy.evidence["chroma_entropy"] == pytest.approx(0.965, abs=1e-6)
    assert noisy.evidence["min_centroid_hz"] == THRESHOLDS["noisy_centroid_hz"]
    assert noisy.evidence["min_chroma_entropy"] == THRESHOLDS["noisy_min_chroma_entropy"]


def test_noisy_reports_retired_descriptors_as_evidence_without_gating_on_them() -> None:
    labels = label_generic(
        make_analysis(
            centroid_energy_hz=5250.0,
            chroma_entropy_value=0.965,
            # The values essentia really measures for white noise, either of
            # which would once have suppressed this label.
            key_confidence=0.6953,
            tonal_stability=0.2417,
        )
    )
    noisy = by_name(labels, "noisy")
    assert noisy.evidence["key_confidence"] == 0.6953
    assert noisy.evidence["tonal_stability"] == 0.2417
    assert noisy.confidence == pytest.approx(0.5, abs=1e-3)


def test_noisy_boundary_either_side_and_exactly_on_the_line() -> None:
    centroid = THRESHOLDS["noisy_centroid_hz"]
    entropy = THRESHOLDS["noisy_min_chroma_entropy"]

    on_line = label_generic(
        make_analysis(
            centroid_energy_hz=centroid, chroma_entropy_value=entropy, chroma_bias="at_least"
        )
    )
    assert by_name(on_line, "noisy").confidence == pytest.approx(0.0, abs=1e-6)

    dark = label_generic(
        make_analysis(centroid_energy_hz=centroid - 1.0, chroma_entropy_value=entropy)
    )
    assert "noisy" not in names(dark)

    peaky = label_generic(
        make_analysis(centroid_energy_hz=centroid, chroma_entropy_value=entropy - 0.01)
    )
    assert "noisy" not in names(peaky)


def test_tonally_stable_fires_on_a_peaked_chroma() -> None:
    labels = label_generic(make_analysis(chroma_entropy_value=0.40))
    stable = by_name(labels, "tonally stable")
    assert stable.confidence == pytest.approx(0.5, abs=1e-3)
    assert stable.evidence["chroma_entropy"] == pytest.approx(0.40, abs=1e-6)
    assert (
        stable.evidence["max_chroma_entropy"] == THRESHOLDS["tonally_stable_max_chroma_entropy"]
    )


def test_tonally_stable_reports_retired_descriptors_without_gating_on_them() -> None:
    labels = label_generic(
        make_analysis(chroma_entropy_value=0.40, key_confidence=0.0221, tonal_stability=0.0)
    )
    stable = by_name(labels, "tonally stable")
    assert stable.evidence["key_confidence"] == 0.0221
    assert stable.evidence["tonal_stability"] == 0.0
    assert stable.confidence == pytest.approx(0.5, abs=1e-3)


def test_tonally_stable_boundary_either_side_and_exactly_on_the_line() -> None:
    ceiling = THRESHOLDS["tonally_stable_max_chroma_entropy"]

    on_line = label_generic(
        make_analysis(chroma_entropy_value=ceiling, chroma_bias="at_most")
    )
    assert by_name(on_line, "tonally stable").confidence == pytest.approx(0.0, abs=1e-6)

    flatter = label_generic(make_analysis(chroma_entropy_value=ceiling + 0.01))
    assert "tonally stable" not in names(flatter)


def test_a_dead_band_separates_tonally_stable_from_noisy() -> None:
    """Deliberate gap: mid-entropy material gets neither label rather than both."""
    ceiling = THRESHOLDS["tonally_stable_max_chroma_entropy"]
    floor = THRESHOLDS["noisy_min_chroma_entropy"]
    assert ceiling < floor, "the two labels must not overlap"
    midpoint = (ceiling + floor) / 2.0
    labels = label_generic(make_analysis(chroma_entropy_value=midpoint, centroid_energy_hz=9000.0))
    assert "tonally stable" not in names(labels)
    assert "noisy" not in names(labels)


def test_noisy_still_needs_a_bright_centroid() -> None:
    """Entropy alone would call a quiet atonal pad noisy."""
    labels = label_generic(make_analysis(chroma_entropy_value=0.999, centroid_energy_hz=400.0))
    assert "noisy" not in names(labels)


# ---------------------------------------------------------------------------
# Drums
# ---------------------------------------------------------------------------


def test_busy_drums_fires_and_is_graded() -> None:
    labels = label_drums(make_analysis("drums", onset_density=8.5))
    busy = by_name(labels, "busy drums")
    assert busy.confidence == pytest.approx(0.5)
    assert busy.evidence == {
        "onset_density": 8.5,
        "min_onset_density": THRESHOLDS["busy_drums_onsets_per_sec"],
    }
    assert "sparse percussion" not in names(labels)


def test_busy_drums_boundary_either_side_and_exactly_on_the_line() -> None:
    threshold = THRESHOLDS["busy_drums_onsets_per_sec"]
    on_line = label_drums(make_analysis("drums", onset_density=threshold))
    assert by_name(on_line, "busy drums").confidence == 0.0
    under = label_drums(make_analysis("drums", onset_density=threshold - 0.01))
    assert "busy drums" not in names(under)


def test_sparse_percussion_fires_and_is_graded() -> None:
    labels = label_drums(make_analysis("drums", onset_density=0.5))
    sparse = by_name(labels, "sparse percussion")
    assert sparse.confidence == pytest.approx(0.5)
    assert sparse.evidence["max_onset_density"] == THRESHOLDS["sparse_onsets_per_sec"]
    assert "busy drums" not in names(labels)


def test_sparse_percussion_boundary_either_side_and_exactly_on_the_line() -> None:
    threshold = THRESHOLDS["sparse_onsets_per_sec"]
    on_line = label_drums(make_analysis("drums", onset_density=threshold))
    assert by_name(on_line, "sparse percussion").confidence == 0.0
    over = label_drums(make_analysis("drums", onset_density=threshold + 0.01))
    assert "sparse percussion" not in names(over)


def test_kick_heavy_fires_from_the_low_band_with_auditable_evidence() -> None:
    labels = label_drums(make_analysis("drums", band_low=0.675))
    kick = by_name(labels, "kick-heavy")
    assert kick.confidence == pytest.approx(0.5)
    # The whole point of evidence: a reader can see why this fired.
    assert kick.evidence == {
        "band_energy_low": 0.675,
        "min_band_energy_low": THRESHOLDS["kick_heavy_low_ratio"],
    }


def test_kick_heavy_boundary_either_side_and_exactly_on_the_line() -> None:
    threshold = THRESHOLDS["kick_heavy_low_ratio"]
    on_line = label_drums(make_analysis("drums", band_low=threshold))
    assert by_name(on_line, "kick-heavy").confidence == 0.0
    under = label_drums(make_analysis("drums", band_low=threshold - 0.001))
    assert "kick-heavy" not in names(under)


def test_bright_hats_needs_both_high_band_and_centroid() -> None:
    labels = label_drums(make_analysis("drums", band_high=0.235, centroid_mean=6500.0))
    hats = by_name(labels, "bright hats")
    assert hats.confidence == pytest.approx(0.5)
    assert hats.evidence["band_energy_high"] == 0.235
    assert hats.evidence["centroid_mean"] == 6500.0

    dull = label_drums(make_analysis("drums", band_high=0.235, centroid_mean=500.0))
    assert "bright hats" not in names(dull)

    thin = label_drums(make_analysis("drums", band_high=0.01, centroid_mean=6500.0))
    assert "bright hats" not in names(thin)


def test_bright_hats_boundary_exactly_on_both_lines() -> None:
    labels = label_drums(
        make_analysis(
            "drums",
            band_high=THRESHOLDS["bright_hats_high_ratio"],
            centroid_mean=THRESHOLDS["transient_bright_centroid_hz"],
        )
    )
    assert by_name(labels, "bright hats").confidence == 0.0


def test_band_ratios_all_none_fires_no_band_labels() -> None:
    """Partial None: centroid present, every band ratio missing."""
    labels = label_drums(make_analysis("drums", centroid_mean=6500.0, onset_density=8.5))
    assert "kick-heavy" not in names(labels)
    assert "bright hats" not in names(labels)
    assert "busy drums" in names(labels)


# ---------------------------------------------------------------------------
# Bass
# ---------------------------------------------------------------------------


def test_sparse_bass_fires_and_is_graded() -> None:
    labels = label_bass(make_analysis("bass", onset_density=0.5))
    assert by_name(labels, "sparse bass").confidence == pytest.approx(0.5)


def test_sustained_sub_needs_dark_centroid_and_low_onset_density() -> None:
    labels = label_bass(make_analysis("bass", centroid_energy_hz=155.0, onset_density=1.125))
    sub = by_name(labels, "sustained sub")
    assert sub.confidence == pytest.approx(0.5)
    assert sub.evidence["centroid_energy_hz"] == 155.0
    assert sub.evidence["max_centroid_hz"] == THRESHOLDS["dark_centroid_hz"]

    bright = label_bass(make_analysis("bass", centroid_energy_hz=900.0, onset_density=1.125))
    assert "sustained sub" not in names(bright)

    busy = label_bass(make_analysis("bass", centroid_energy_hz=155.0, onset_density=6.0))
    assert "sustained sub" not in names(busy)


def test_sustained_sub_boundary_exactly_on_both_lines() -> None:
    labels = label_bass(
        make_analysis(
            "bass",
            centroid_energy_hz=THRESHOLDS["dark_centroid_hz"],
            onset_density=THRESHOLDS["moderate_onsets_per_sec"],
        )
    )
    assert by_name(labels, "sustained sub").confidence == 0.0


def test_plucked_bass_fires_from_transient_sharpness() -> None:
    labels = label_bass(make_analysis("bass", transient_sharpness=5.5))
    plucked = by_name(labels, "plucked bass")
    assert plucked.confidence == pytest.approx(0.5)
    assert plucked.evidence == {
        "transient_sharpness": 5.5,
        "min_transient_sharpness": THRESHOLDS["sharp_transient_ratio"],
    }


def test_plucked_bass_boundary_either_side_and_exactly_on_the_line() -> None:
    threshold = THRESHOLDS["sharp_transient_ratio"]
    on_line = label_bass(make_analysis("bass", transient_sharpness=threshold))
    assert by_name(on_line, "plucked bass").confidence == 0.0
    under = label_bass(make_analysis("bass", transient_sharpness=threshold - 0.01))
    assert "plucked bass" not in names(under)


# ---------------------------------------------------------------------------
# Vocals
# ---------------------------------------------------------------------------


def test_speech_vocal_dominant_needs_energy_mid_centroid_and_moderate_onsets() -> None:
    labels = label_vocals(
        make_analysis("vocals", rms_mean=0.055, centroid_energy_hz=1500.0, onset_density=3.0)
    )
    dominant = by_name(labels, "speech/vocal dominant")
    # RMS is the weakest ingredient at 0.5; the two windows are both saturated.
    assert dominant.confidence == pytest.approx(0.5)
    assert dominant.evidence["rms_mean"] == 0.055
    assert dominant.evidence["centroid_energy_hz"] == 1500.0
    assert dominant.evidence["onset_density"] == 3.0


def test_speech_vocal_dominant_rejected_outside_each_condition() -> None:
    too_quiet = label_vocals(
        make_analysis("vocals", rms_mean=0.001, centroid_energy_hz=1500.0, onset_density=3.0)
    )
    assert "speech/vocal dominant" not in names(too_quiet)

    too_bright = label_vocals(
        make_analysis("vocals", rms_mean=0.055, centroid_energy_hz=7000.0, onset_density=3.0)
    )
    assert "speech/vocal dominant" not in names(too_bright)

    too_dark = label_vocals(
        make_analysis("vocals", rms_mean=0.055, centroid_energy_hz=100.0, onset_density=3.0)
    )
    assert "speech/vocal dominant" not in names(too_dark)

    too_busy = label_vocals(
        make_analysis("vocals", rms_mean=0.055, centroid_energy_hz=1500.0, onset_density=9.0)
    )
    assert "speech/vocal dominant" not in names(too_busy)


def test_speech_vocal_dominant_boundary_exactly_on_every_line() -> None:
    labels = label_vocals(
        make_analysis(
            "vocals",
            rms_mean=THRESHOLDS["vocal_presence_rms"],
            centroid_energy_hz=THRESHOLDS["vocal_centroid_min_hz"],
            onset_density=THRESHOLDS["sparse_onsets_per_sec"],
        )
    )
    assert by_name(labels, "speech/vocal dominant").confidence == 0.0


def test_sparse_vocal_fires_and_is_graded() -> None:
    labels = label_vocals(make_analysis("vocals", onset_density=0.5))
    assert by_name(labels, "sparse vocal").confidence == pytest.approx(0.5)


def test_processed_wide_vocal_needs_air_and_a_moving_centroid() -> None:
    labels = label_vocals(make_analysis("vocals", band_high=0.12, centroid_std=1400.0))
    processed = by_name(labels, "processed/wide vocal")
    assert processed.confidence == pytest.approx(0.5)
    assert processed.evidence["band_energy_high"] == 0.12
    assert processed.evidence["centroid_std"] == 1400.0

    steady = label_vocals(make_analysis("vocals", band_high=0.12, centroid_std=100.0))
    assert "processed/wide vocal" not in names(steady)

    dull = label_vocals(make_analysis("vocals", band_high=0.001, centroid_std=1400.0))
    assert "processed/wide vocal" not in names(dull)


def test_processed_wide_vocal_boundary_exactly_on_both_lines() -> None:
    labels = label_vocals(
        make_analysis(
            "vocals",
            band_high=THRESHOLDS["processed_vocal_high_ratio"],
            centroid_std=THRESHOLDS["processed_centroid_std_hz"],
        )
    )
    assert by_name(labels, "processed/wide vocal").confidence == 0.0


# ---------------------------------------------------------------------------
# Other
# ---------------------------------------------------------------------------


def test_sustained_pad_like_texture_fires_from_crest_and_onsets() -> None:
    labels = label_other(make_analysis("other", crest_factor=2.8, onset_density=1.1))
    pad = by_name(labels, "sustained pad-like texture")
    assert pad.confidence == pytest.approx(0.5)
    assert pad.evidence["crest_factor"] == 2.8


def test_sustained_pad_like_texture_boundary_exactly_on_both_lines() -> None:
    labels = label_other(
        make_analysis(
            "other",
            crest_factor=THRESHOLDS["sustained_crest_factor"],
            onset_density=THRESHOLDS["moderate_onsets_per_sec"],
        )
    )
    assert by_name(labels, "sustained pad-like texture").confidence == 0.0


def test_other_noisy_matches_the_generic_definition() -> None:
    analysis = make_analysis("other", centroid_energy_hz=5250.0, chroma_entropy_value=0.965)
    assert by_name(label_other(analysis), "noisy").confidence == pytest.approx(0.5, abs=1e-3)


def test_bright_plucks_needs_air_brightness_and_sharp_transients() -> None:
    labels = label_other(
        make_analysis("other", band_high=0.575, centroid_mean=3600.0, transient_sharpness=5.5)
    )
    plucks = by_name(labels, "bright plucks")
    assert plucks.confidence == pytest.approx(0.5)
    assert plucks.evidence["band_energy_high"] == 0.575
    assert plucks.evidence["centroid_mean"] == 3600.0
    assert plucks.evidence["transient_sharpness"] == 5.5

    soft = label_other(
        make_analysis("other", band_high=0.575, centroid_mean=3600.0, transient_sharpness=1.0)
    )
    assert "bright plucks" not in names(soft)

    dark = label_other(
        make_analysis("other", band_high=0.575, centroid_mean=500.0, transient_sharpness=5.5)
    )
    assert "bright plucks" not in names(dark)

    dull = label_other(
        make_analysis("other", band_high=0.01, centroid_mean=3600.0, transient_sharpness=5.5)
    )
    assert "bright plucks" not in names(dull)


def test_bright_plucks_boundary_exactly_on_every_line() -> None:
    labels = label_other(
        make_analysis(
            "other",
            band_high=THRESHOLDS["bright_plucks_high_ratio"],
            centroid_mean=THRESHOLDS["transient_bright_centroid_hz"],
            transient_sharpness=THRESHOLDS["sharp_transient_ratio"],
        )
    )
    assert by_name(labels, "bright plucks").confidence == 0.0


# ---------------------------------------------------------------------------
# apply(): dispatch, dedupe, ordering, degradation
# ---------------------------------------------------------------------------


def test_apply_on_all_none_analysis_produces_no_labels_and_does_not_raise() -> None:
    for source in ("mix", "drums", "bass", "vocals", "other", "wildly-unexpected"):
        assert apply(make_analysis(source)) == []


def test_apply_on_unknown_source_falls_back_to_generic_labels_only() -> None:
    analysis = make_analysis(
        "strings",
        crest_factor=14.0,
        onset_density=8.5,
        band_low=0.9,
        transient_sharpness=5.5,
    )
    labels = apply(analysis)
    assert names(labels) == {"percussive"}
    # Nothing source-specific leaked in despite the descriptors being present.
    assert "busy drums" not in names(labels)
    assert "kick-heavy" not in names(labels)


def test_apply_on_mix_uses_generic_labels_only() -> None:
    analysis = make_analysis("mix", onset_density=8.5, band_low=0.9)
    assert names(apply(analysis)) == set()


def test_apply_combines_generic_and_source_labels() -> None:
    analysis = make_analysis(
        "drums",
        crest_factor=14.0,
        onset_density=8.5,
        band_low=0.675,
        centroid_mean=6500.0,
        band_high=0.235,
    )
    assert names(apply(analysis)) == {
        "percussive",
        "busy drums",
        "kick-heavy",
        "bright hats",
    }


def test_apply_dedupes_labels_reachable_by_two_routes() -> None:
    """`noisy` fires generically and from `label_other`; only one may survive."""
    analysis = make_analysis("other", centroid_energy_hz=5250.0, chroma_entropy_value=0.965)
    labels = apply(analysis)
    assert [label.label for label in labels].count("noisy") == 1


def test_dedupe_keeps_the_higher_confidence_reading() -> None:
    weak = HeuristicLabel(label="noisy", confidence=0.2, evidence={"centroid_mean": 3000.0})
    strong = HeuristicLabel(label="noisy", confidence=0.9, evidence={"centroid_mean": 9000.0})
    assert _dedupe([weak, strong]) == [strong]
    assert _dedupe([strong, weak]) == [strong]


def test_apply_sorts_by_confidence_descending() -> None:
    analysis = make_analysis(
        "drums",
        crest_factor=40.0,  # percussive saturates at 1.0
        onset_density=8.5,  # busy drums at 0.5
        band_low=0.5875,  # kick-heavy at 0.25
    )
    labels = apply(analysis)
    assert [label.label for label in labels] == ["percussive", "busy drums", "kick-heavy"]
    confidences = [label.confidence for label in labels]
    assert confidences == sorted(confidences, reverse=True)
    assert confidences == pytest.approx([1.0, 0.5, 0.25])


def test_apply_breaks_confidence_ties_alphabetically_for_stable_output() -> None:
    analysis = make_analysis(
        "bass",
        onset_density=THRESHOLDS["sparse_onsets_per_sec"],
        transient_sharpness=THRESHOLDS["sharp_transient_ratio"],
    )
    labels = apply(analysis)
    assert [label.label for label in labels] == ["plucked bass", "sparse bass"]
    assert all(label.confidence == 0.0 for label in labels)


def test_apply_with_partial_none_fires_only_the_labels_whose_inputs_exist() -> None:
    """Centroid present, band ratios and everything else missing."""
    labels = apply(make_analysis("drums", centroid_mean=6500.0))
    assert labels == []

    labels = apply(make_analysis("drums", centroid_mean=6500.0, onset_density=8.5))
    assert names(labels) == {"busy drums"}


def test_every_confidence_stays_within_the_schema_bounds() -> None:
    extremes = (-1e6, 0.0, 0.5, 1.0, 2.0, 12.0, 1e6)
    for source in ("mix", "drums", "bass", "vocals", "other", "unknown"):
        for value in extremes:
            analysis = make_analysis(
                source,
                onset_density=value,
                transient_sharpness=value,
                key_confidence=min(1.0, max(0.0, value)),
                tonal_stability=min(1.0, max(0.0, value)),
                centroid_mean=value,
                centroid_energy_hz=value,
                centroid_std=value,
                band_low=min(1.0, max(0.0, value)),
                band_high=min(1.0, max(0.0, value)),
                crest_factor=value,
                rms_mean=value,
            )
            for label in apply(analysis):
                assert 0.0 <= label.confidence <= 1.0


def test_confidence_is_graded_not_binary_across_a_label() -> None:
    """Just over the line scores near 0.0; far past it scores 1.0."""
    threshold = THRESHOLDS["busy_drums_onsets_per_sec"]
    saturation = THRESHOLDS["busy_drums_onsets_saturation"]

    just_over = by_name(apply(make_analysis("drums", onset_density=threshold + 0.05)), "busy drums")
    assert just_over.confidence < 0.05

    midway = by_name(
        apply(make_analysis("drums", onset_density=(threshold + saturation) / 2)), "busy drums"
    )
    assert midway.confidence == pytest.approx(0.5)

    far_past = by_name(apply(make_analysis("drums", onset_density=saturation * 3)), "busy drums")
    assert far_past.confidence == 1.0

    assert just_over.confidence < midway.confidence < far_past.confidence


# ---------------------------------------------------------------------------
# Reachability against real backend output, on BOTH backends
# ---------------------------------------------------------------------------
# Every threshold test above constructs descriptor values that satisfy the
# thresholds by assumption, so all of them pass even when a threshold sits
# somewhere a real backend can never reach. Several labels shipped broken
# exactly that way, and the last round was worse: thresholds calibrated on
# librosa left `noisy` unreachable and `tonally stable` universal on essentia,
# which is the DEFAULT backend. A librosa-only table cannot see that.
#
# So both backends get a table, and the reachability check runs against both.
# Values are transcribed from real `LibrosaBackend` / `EssentiaBackend` runs
# rather than imported, so this layer stays pure, fast, and runnable with
# neither library installed. `chroma_entropy` is the measured entropy of that
# run's `hpcp_mean`; `make_analysis` rebuilds a 12-bin vector matching it.
#
# Source: 8 s synthetic signals at 44.1 kHz, 44.1 kHz mono, both backends.
#
# `centroid_energy_hz` was added to every row in W8B, when `_noisy`,
# `sustained sub` and the `speech/vocal dominant` window migrated onto it. It is
# a LATER MEASUREMENT than the rest of each row and its provenance is stated
# plainly, because the original run's signal generators were never committed and
# could only be reconstructed:
#
#   * Reconstructed at 8 s / 44.1 kHz mono, then validated against each row's
#     committed `centroid_mean` and band ratios before its energy centroid was
#     recorded. `sine_a440` (441.0 / 439.9 against 441.04 / 439.93),
#     `a_minor_triad` (274.7 / 274.2 against 274.69 / 274.21), `white_noise`
#     (11036.7 / 9939.7 against 11034.78 / 9936.58) and `smooth_sub_60hz`
#     (61.4 / 60.0 against 61.28 / 60.04) reproduce to the last digit that
#     matters. `voiced_phrases` reproduces on the band that defines it
#     (low 0.6446 against 0.6422) from a 165 Hz stack of ten 1/n harmonics
#     gated into 0.45 s phrases.
#   * The click, dense and pluck rows reproduce their SPECTRA (band_high 0.657,
#     0.657, 0.998 against 0.700, 0.604, 0.977) but not their hit density. That
#     does not matter for this column and is the whole reason the column exists:
#     an energy centroid is invariant to how sparse the hits are. Measured on
#     one pluck train at 2, 4 and 8 hits/sec, `centroid_mean` reads 1698 / 3397
#     / 6820 Hz while `centroid_energy_hz` reads 9543 Hz at every rate.
#   * `processed_vocal` is deliberately left UNMEASURED. Its band split
#     (low 0.24, high 0.0415) could not be reproduced closely enough to attach a
#     spectrum-derived number to it, and nothing that row gates reads the field
#     — `processed/wide vocal` is built from `band_high` and `centroid_std`. A
#     `None` here is the honest shape: a descriptor that was not measured.
#
# The two backends agree on this descriptor to 0.05 Hz on every row above
# (W4D measured a worst case of 0.240 Hz on real stems), so both tables carry
# the same number rather than two transcriptions of one measurement.
MEASURED_LIBROSA: dict[str, dict[str, float | None]] = {
    "sine_a440": {
        "key_confidence": 0.1714, "tonal_stability": 0.99991, "chroma_entropy_value": 0.0868,
        "centroid_mean": 441.04,
        "centroid_energy_hz": 439.99, "centroid_std": 14.72, "crest_factor": 1.4142,
        "onset_density": 0.0, "transient_sharpness": None, "band_low": 0.0, "band_high": 0.0,
        "rms_mean": 0.3531,
    },
    "white_noise": {
        "key_confidence": 0.0221, "tonal_stability": 0.99448, "chroma_entropy_value": 0.9999,
        "centroid_mean": 11034.78,
        "centroid_energy_hz": 10029.3, "centroid_std": 138.19, "crest_factor": 4.7289,
        "onset_density": 8.125, "transient_sharpness": 1.3834, "band_low": 0.012,
        "band_high": 0.7013, "rms_mean": 0.1901,
    },
    "a_minor_triad": {
        "key_confidence": 0.2035, "tonal_stability": 0.99995, "chroma_entropy_value": 0.5145,
        "centroid_mean": 274.69,
        "centroid_energy_hz": 270.39, "centroid_std": 36.91, "crest_factor": 2.4481,
        "onset_density": 19.125, "transient_sharpness": 3.9485, "band_low": 0.3668,
        "band_high": 0.0, "rms_mean": 0.3669,
    },
    "click_120bpm": {
        "key_confidence": 0.0257, "tonal_stability": 0.99937, "chroma_entropy_value": 0.9947,
        "centroid_mean": 2098.81,
        "centroid_energy_hz": 9201.36, "centroid_std": 4296.46, "crest_factor": 19.0442,
        "onset_density": 2.0, "transient_sharpness": 100.0, "band_low": 0.0256,
        "band_high": 0.7001, "rms_mean": 0.017,
    },
    "dense_16ths": {
        "key_confidence": 0.0612, "tonal_stability": 0.99881, "chroma_entropy_value": 0.9955,
        "centroid_mean": 5472.81,
        "centroid_energy_hz": 9201.28, "centroid_std": 5216.42, "crest_factor": 19.9148,
        "onset_density": 7.875, "transient_sharpness": 100.0, "band_low": 0.0605,
        "band_high": 0.6041, "rms_mean": 0.0308,
    },
    "smooth_sub_60hz": {
        "key_confidence": 0.0709, "tonal_stability": 1.0, "chroma_entropy_value": 0.353,
        "centroid_mean": 61.28,
        "centroid_energy_hz": 60.0, "centroid_std": 3.19, "crest_factor": 1.5492,
        "onset_density": 0.0, "transient_sharpness": None, "band_low": 1.0, "band_high": 0.0,
        "rms_mean": 0.3707,
    },
    "bright_pluck_train": {
        "key_confidence": 0.2784, "tonal_stability": 0.99952, "chroma_entropy_value": 0.9983,
        "centroid_mean": 1301.85,
        "centroid_energy_hz": 10371.07, "centroid_std": 2838.91, "crest_factor": 21.3126,
        "onset_density": 2.875, "transient_sharpness": 100.0, "band_low": 0.0003,
        "band_high": 0.9765, "rms_mean": 0.0166,
    },
    "voiced_phrases": {
        "key_confidence": 0.0296, "tonal_stability": 0.99668, "chroma_entropy_value": 0.8591,
        "centroid_mean": 389.96,
        "centroid_energy_hz": 312.06, "centroid_std": 352.84, "crest_factor": 2.9841,
        "onset_density": 3.375, "transient_sharpness": 97.954, "band_low": 0.6422,
        "band_high": 0.0, "rms_mean": 0.0958,
    },
    "processed_vocal": {
        "key_confidence": 0.0219, "tonal_stability": 0.99587, "chroma_entropy_value": 0.9054,
        "centroid_mean": 4442.1, "centroid_std": 3812.96, "crest_factor": 4.7451,
        "onset_density": 2.5, "transient_sharpness": 68.7411, "band_low": 0.24,
        "band_high": 0.0415, "rms_mean": 0.1356,
    },
    # An absent stem: what Demucs returns when the source is not in the mix.
    # Every descriptor here is separation residue. Without the silence gate this
    # row alone fires noisy, busy drums and bright hats on pure numerical noise.
    "separation_residue": {
        "key_confidence": 0.1748, "tonal_stability": 0.99632, "chroma_entropy_value": 0.9761,
        "centroid_mean": 10861.67,
        "centroid_energy_hz": 9988.2, "centroid_std": 138.63, "crest_factor": 4.5354,
        "onset_density": 7.375, "transient_sharpness": 1.3934, "band_low": 0.0202,
        "band_high": 0.6323, "rms_mean": 0.000941, "loudness_lufs": -54.67,
    },
}

# Same signals, same run, essentia. Note how far the descriptors diverge:
# onset rate is far more conservative (white noise 0.125/s against librosa's
# 8.125/s), centroids sit ~10% lower, and chroma entropy is compressed into a
# much narrower band.
MEASURED_ESSENTIA: dict[str, dict[str, float | None]] = {
    "sine_a440": {
        "key_confidence": 0.688, "tonal_stability": 0.78205, "chroma_entropy_value": 0.5286,
        "centroid_mean": 439.93,
        "centroid_energy_hz": 439.99, "centroid_std": 1.14, "crest_factor": 1.4142,
        "onset_density": 0.25, "transient_sharpness": 2.6008, "band_low": 0.0, "band_high": 0.0,
        "rms_mean": 0.3529,
    },
    "white_noise": {
        "key_confidence": 0.6953, "tonal_stability": 0.24165, "chroma_entropy_value": 0.9954,
        "centroid_mean": 9936.58,
        "centroid_energy_hz": 10029.3, "centroid_std": 99.3, "crest_factor": 4.7289,
        "onset_density": 0.125, "transient_sharpness": 1.0452, "band_low": 0.012,
        "band_high": 0.7013, "rms_mean": 0.1899,
    },
    "a_minor_triad": {
        "key_confidence": 0.7663, "tonal_stability": 0.80031, "chroma_entropy_value": 0.75,
        "centroid_mean": 274.21,
        "centroid_energy_hz": 270.39, "centroid_std": 3.12, "crest_factor": 2.4481,
        "onset_density": 0.375, "transient_sharpness": 3.154, "band_low": 0.3669,
        "band_high": 0.0, "rms_mean": 0.3667,
    },
    "click_120bpm": {
        "key_confidence": 0.8098, "tonal_stability": 0.51637, "chroma_entropy_value": 0.7783,
        "centroid_mean": 1897.75,
        "centroid_energy_hz": 9201.36, "centroid_std": 3890.65, "crest_factor": 19.0442,
        "onset_density": 2.0, "transient_sharpness": 100.0, "band_low": 0.0256,
        "band_high": 0.7001, "rms_mean": 0.017,
    },
    "dense_16ths": {
        "key_confidence": 0.7927, "tonal_stability": 0.35185, "chroma_entropy_value": 0.7862,
        "centroid_mean": 4853.36,
        "centroid_energy_hz": 9201.28, "centroid_std": 4641.73, "crest_factor": 19.9148,
        "onset_density": 7.875, "transient_sharpness": 100.0, "band_low": 0.0605,
        "band_high": 0.6041, "rms_mean": 0.0307,
    },
    "smooth_sub_60hz": {
        "key_confidence": 0.688, "tonal_stability": 0.61243, "chroma_entropy_value": 0.7725,
        "centroid_mean": 60.04,
        "centroid_energy_hz": 60.0, "centroid_std": 2.4, "crest_factor": 1.5492,
        "onset_density": 0.875, "transient_sharpness": 1.0921, "band_low": 1.0,
        "band_high": 0.0, "rms_mean": 0.3705,
    },
    "bright_pluck_train": {
        "key_confidence": 0.766, "tonal_stability": 0.30535, "chroma_entropy_value": 0.7268,
        "centroid_mean": 1169.48,
        "centroid_energy_hz": 10371.07, "centroid_std": 2552.18, "crest_factor": 21.3126,
        "onset_density": 2.875, "transient_sharpness": 100.0, "band_low": 0.0003,
        "band_high": 0.9765, "rms_mean": 0.0166,
    },
    "voiced_phrases": {
        "key_confidence": 0.5699, "tonal_stability": 0.90966, "chroma_entropy_value": 0.5439,
        "centroid_mean": 273.54,
        "centroid_energy_hz": 312.06, "centroid_std": 235.72, "crest_factor": 2.9841,
        "onset_density": 2.5, "transient_sharpness": 7.26, "band_low": 0.6422,
        "band_high": 0.0, "rms_mean": 0.0957,
    },
    "processed_vocal": {
        "key_confidence": 0.4445, "tonal_stability": 0.82353, "chroma_entropy_value": 0.7947,
        "centroid_mean": 2065.99, "centroid_std": 1866.5, "crest_factor": 4.7451,
        "onset_density": 2.5, "transient_sharpness": 7.878, "band_low": 0.24,
        "band_high": 0.0415, "rms_mean": 0.1354,
    },
    # Same residue signal as the librosa table. The two backends agree on LUFS
    # to 0.05 dB, which is why the gate reads LUFS and not a backend-specific
    # descriptor — note how far apart their tonal readings of it are.
    "separation_residue": {
        "key_confidence": 0.688, "tonal_stability": 0.57664, "chroma_entropy_value": 0.8558,
        "centroid_mean": 9471.58,
        "centroid_energy_hz": 9988.2, "centroid_std": 121.79, "crest_factor": 4.5354,
        "onset_density": 0.125, "transient_sharpness": 0.9822, "band_low": 0.0202,
        "band_high": 0.6323, "rms_mean": 0.000941, "loudness_lufs": -54.62,
    },
}

MEASURED: dict[str, dict[str, dict[str, float | None]]] = {
    "librosa": MEASURED_LIBROSA,
    "essentia": MEASURED_ESSENTIA,
}

SOURCES = ("mix", "drums", "bass", "vocals", "other")

# Every label this module can emit. Each must be reachable on BOTH backends.
EXPECTED_LABELS = frozenset(
    {
        "silent/absent stem",
        "tonally stable",
        "noisy",
        "sustained",
        "percussive",
        "busy drums",
        "sparse percussion",
        "kick-heavy",
        "bright hats",
        "sparse bass",
        "sustained sub",
        "plucked bass",
        "speech/vocal dominant",
        "sparse vocal",
        "processed/wide vocal",
        "sustained pad-like texture",
        "bright plucks",
    }
)


def measured(backend: str, signal: str, source: str) -> SourceAnalysis:
    return make_analysis(source, **MEASURED[backend][signal])  # type: ignore[arg-type]


def labels_reachable_on(backend: str) -> dict[str, set[tuple[str, str]]]:
    """Map each label to every (signal, source) that produces it on `backend`."""
    found: dict[str, set[tuple[str, str]]] = {}
    for signal in MEASURED[backend]:
        for source in SOURCES:
            for label in names(apply(measured(backend, signal, source))):
                found.setdefault(label, set()).add((signal, source))
    return found


@pytest.mark.parametrize("backend", sorted(MEASURED))
@pytest.mark.parametrize("label", sorted(EXPECTED_LABELS))
def test_every_label_is_reachable_on_every_backend(label: str, backend: str) -> None:
    """The durable guard. A threshold tuned on one backend that leaves a label
    unreachable on the other fails here, which is the defect class that has now
    bitten this module three times."""
    reachable = labels_reachable_on(backend)
    assert label in reachable, (
        f"{label!r} is unreachable on {backend} from any measured signal or source"
    )


@pytest.mark.parametrize("backend", sorted(MEASURED))
def test_reachability_covers_exactly_the_labels_the_module_emits(backend: str) -> None:
    """Guards the guard, per backend: a new label must be shown reachable on
    both backends, not merely added."""
    emitted = set(labels_reachable_on(backend))
    assert emitted == set(EXPECTED_LABELS), (
        f"on {backend}: unexpected {sorted(emitted - EXPECTED_LABELS)}, "
        f"missing {sorted(EXPECTED_LABELS - emitted)}"
    )


@pytest.mark.parametrize("backend", sorted(MEASURED))
def test_maximally_tonal_measured_input_is_labelled_tonally_stable(backend: str) -> None:
    """A sustained pure tone is the most tonal signal there is. If the real
    backend's tonal descriptor for it cannot clear the threshold, the label is
    dead on arrival — which is how this shipped twice."""
    labels = apply(measured(backend, "sine_a440", "mix"))
    assert "tonally stable" in names(labels)
    assert by_name(labels, "tonally stable").confidence > 0.0
    assert "noisy" not in names(labels)


@pytest.mark.parametrize("backend", sorted(MEASURED))
def test_maximally_noisy_measured_input_is_labelled_noisy(backend: str) -> None:
    """White noise is the most atonal, broadband signal there is."""
    labels = apply(measured(backend, "white_noise", "mix"))
    assert "noisy" in names(labels)
    assert by_name(labels, "noisy").confidence > 0.5
    assert "tonally stable" not in names(labels)


@pytest.mark.parametrize("backend", sorted(MEASURED))
def test_tonal_and_noise_sit_either_side_of_the_dead_band(backend: str) -> None:
    """The measured separation both thresholds are derived from. Tonal material
    must clear the ceiling and broadband noise must clear the floor, on both
    backends — that simultaneous constraint is what ruled out `key_confidence`
    and `tonal_stability` and left chroma entropy."""
    ceiling = THRESHOLDS["tonally_stable_max_chroma_entropy"]
    floor = THRESHOLDS["noisy_min_chroma_entropy"]
    for signal in ("sine_a440", "a_minor_triad", "smooth_sub_60hz"):
        entropy = MEASURED[backend][signal]["chroma_entropy_value"]
        assert entropy is not None and entropy <= ceiling, f"{backend}/{signal}"
    for signal in ("white_noise",):
        entropy = MEASURED[backend][signal]["chroma_entropy_value"]
        assert entropy is not None and entropy >= floor, f"{backend}/{signal}"


def test_retired_descriptors_would_not_have_worked_on_both_backends() -> None:
    """Records why `key_confidence` and `tonal_stability` cannot gate anything:
    on essentia, white noise is not separable from tonal material by key
    confidence, and on librosa it is not separable by tonal stability. Any
    single threshold on either descriptor fails one backend."""
    essentia_kc = {
        name: MEASURED_ESSENTIA[name]["key_confidence"]
        for name in ("sine_a440", "white_noise", "a_minor_triad")
    }
    # Noise lands *between* the sine and the triad — no threshold separates it.
    assert essentia_kc["sine_a440"] is not None
    assert essentia_kc["white_noise"] is not None
    assert essentia_kc["a_minor_triad"] is not None
    assert essentia_kc["sine_a440"] < essentia_kc["white_noise"] < essentia_kc["a_minor_triad"]

    # On librosa the whole tonal_stability range is under a percent wide.
    librosa_stability = [
        value
        for value in (row["tonal_stability"] for row in MEASURED_LIBROSA.values())
        if value is not None
    ]
    assert max(librosa_stability) - min(librosa_stability) < 0.01


def test_no_threshold_reads_key_confidence_or_tonal_stability() -> None:
    """Both are reported as evidence everywhere and gate nothing anywhere."""
    banned = ("key_confidence", "tonal_stability", "stability")
    offenders = [key for key in THRESHOLDS if any(word in key for word in banned)]
    assert not offenders


@pytest.mark.parametrize("descriptor", ["key_confidence", "tonal_stability"])
def test_retired_descriptors_never_change_a_label(descriptor: str) -> None:
    """The behavioural version: sweeping either descriptor across its whole
    range must not alter any label or confidence."""
    row = dict(MEASURED_LIBROSA["white_noise"])
    baselines: list[list[tuple[str, float]]] = []
    for value in (0.0, 0.02, 0.25, 0.5, 0.7, 0.99, 1.0):
        row[descriptor] = value
        labels = apply(make_analysis("other", **row))  # type: ignore[arg-type]
        baselines.append([(item.label, item.confidence) for item in labels])
    assert all(entry == baselines[0] for entry in baselines)
    assert baselines[0], "expected at least one label, otherwise this proves nothing"


# ---------------------------------------------------------------------------
# The near-silence gate
# ---------------------------------------------------------------------------
# Measured on a real 12 s instrumental mix (kick, snare, hats, bass, pad, and
# deliberately no vocals). Demucs correctly returned an empty vocals stem; the
# labeller then described the residue as tonally stable percussive material
# with a key of C minor and 53 onsets. No synthetic-descriptor test could catch
# that, because each descriptor was individually plausible.
EMPTY_VOCALS_STEM: dict[str, float | None] = {
    "loudness_lufs": -68.23,
    "rms_mean": 0.0013,
    "crest_factor": 10.115,
    "onset_density": 4.4167,
    "chroma_entropy_value": 0.6238,
    "key_confidence": 0.6880,
    "tonal_stability": 0.8066,
}

# The same run's real sources, none of which may be gated.
REAL_SOURCE_LUFS = {
    "mix": -17.38,
    "drums": -24.58,
    "bass": -18.93,
    "other": -27.11,
}


def test_empty_stem_produces_only_the_silence_label() -> None:
    """The regression. Before the gate this returned ['tonally stable',
    'percussive'] — a confident description of inaudible separation residue."""
    labels = apply(make_analysis("vocals", **EMPTY_VOCALS_STEM))  # type: ignore[arg-type]
    assert [label.label for label in labels] == ["silent/absent stem"]
    assert "tonally stable" not in names(labels)
    assert "percussive" not in names(labels)


def test_empty_stem_label_carries_its_loudness_as_evidence() -> None:
    labels = apply(make_analysis("vocals", **EMPTY_VOCALS_STEM))  # type: ignore[arg-type]
    silence = by_name(labels, "silent/absent stem")
    assert silence.evidence["loudness_lufs"] == -68.23
    assert silence.evidence["max_loudness_lufs"] == THRESHOLDS["silence_floor_lufs"]
    assert silence.confidence > 0.5


@pytest.mark.parametrize("source", sorted(REAL_SOURCE_LUFS))
def test_real_sources_from_the_same_run_are_not_gated(source: str) -> None:
    """The other side of the regression: the gate must not silence real stems.
    The quietest was -27.11 LUFS, over 20 dB above the floor."""
    labels = apply(
        make_analysis(
            source,
            loudness_lufs=REAL_SOURCE_LUFS[source],
            rms_mean=0.0216,
            crest_factor=10.67,
            onset_density=4.0,
            chroma_entropy_value=0.5,
        )
    )
    assert "silent/absent stem" not in names(labels)
    assert labels, "a real source should still be described"


def test_the_gate_applies_uniformly_to_every_source_including_mix() -> None:
    for source in ("mix", "drums", "bass", "vocals", "other", "unknown-source"):
        labels = apply(make_analysis(source, **EMPTY_VOCALS_STEM))  # type: ignore[arg-type]
        assert [label.label for label in labels] == ["silent/absent stem"], source


def test_silence_floor_boundary_either_side_and_exactly_on_the_line() -> None:
    floor = THRESHOLDS["silence_floor_lufs"]

    on_line = apply(make_analysis("vocals", loudness_lufs=floor, crest_factor=10.0))
    assert by_name(on_line, "silent/absent stem").confidence == 0.0

    just_above = apply(
        make_analysis("vocals", loudness_lufs=floor + 0.01, crest_factor=10.0, onset_density=4.0)
    )
    assert "silent/absent stem" not in names(just_above)


def test_silence_confidence_rises_the_quieter_the_stem_gets() -> None:
    def confidence_at(lufs: float) -> float:
        labels = apply(make_analysis("vocals", loudness_lufs=lufs))
        return by_name(labels, "silent/absent stem").confidence

    assert confidence_at(-50.0) == 0.0
    assert confidence_at(-55.0) == pytest.approx(0.25)
    assert confidence_at(-60.0) == pytest.approx(0.5)
    assert confidence_at(-68.23) == pytest.approx(0.9115, abs=1e-4)
    # Essentia clamps at -70.0 for digital silence.
    assert confidence_at(-70.0) == 1.0
    assert confidence_at(-120.0) == 1.0


def test_lufs_is_authoritative_and_is_not_second_guessed_by_rms() -> None:
    """An audible LUFS reading must win even when rms_mean looks tiny — the
    fallback exists to cover missing LUFS, not to override a present one."""
    labels = apply(
        make_analysis("bass", loudness_lufs=-18.93, rms_mean=0.0001, chroma_entropy_value=0.5)
    )
    assert "silent/absent stem" not in names(labels)


def test_rms_fallback_catches_digital_silence_when_lufs_is_none() -> None:
    """The case that makes the fallback necessary rather than optional: on
    digital silence pyloudnorm returns non-finite and W1A maps it to None, so
    on librosa the most pathological input has no LUFS at all. Both backends
    report rms_mean 0.0 for it."""
    labels = apply(make_analysis("vocals", loudness_lufs=None, rms_mean=0.0))
    assert [label.label for label in labels] == ["silent/absent stem"]
    silence = labels[0]
    assert silence.confidence == 1.0
    assert silence.evidence["rms_mean"] == 0.0
    assert silence.evidence["max_rms_mean"] == THRESHOLDS["silence_floor_rms"]
    assert "loudness_lufs" not in silence.evidence


def test_rms_fallback_boundary_either_side_and_exactly_on_the_line() -> None:
    floor = THRESHOLDS["silence_floor_rms"]

    on_line = apply(make_analysis("vocals", rms_mean=floor))
    assert by_name(on_line, "silent/absent stem").confidence == 0.0

    just_above = apply(
        make_analysis("vocals", rms_mean=floor + 0.0001, crest_factor=10.0, onset_density=4.0)
    )
    assert "silent/absent stem" not in names(just_above)


def test_rms_fallback_does_not_gate_a_quiet_but_real_stem() -> None:
    """The quietest real stem measured was rms 0.0216, ~7x above the floor."""
    labels = apply(
        make_analysis("drums", rms_mean=0.0216, crest_factor=10.67, onset_density=4.0)
    )
    assert "silent/absent stem" not in names(labels)


def test_a_source_with_neither_loudness_nor_rms_is_not_gated() -> None:
    """No level information is not evidence of silence — it degrades to the
    ordinary all-None behaviour of producing nothing, not to a false claim."""
    assert apply(make_analysis("vocals")) == []
    labels = apply(make_analysis("drums", onset_density=8.5))
    assert names(labels) == {"busy drums"}


@pytest.mark.parametrize("backend", sorted(MEASURED))
def test_measured_separation_residue_is_gated_on_both_backends(backend: str) -> None:
    """Without the gate this row fires noisy, busy drums and bright hats on
    librosa — all computed on inaudible residue."""
    for source in SOURCES:
        labels = apply(measured(backend, "separation_residue", source))
        assert [label.label for label in labels] == ["silent/absent stem"], f"{backend}/{source}"


def test_labels_are_serialisable_with_their_evidence() -> None:
    analysis = make_analysis("drums", band_low=0.675, onset_density=8.5)
    payload = [label.model_dump() for label in apply(analysis)]
    kick = next(entry for entry in payload if entry["label"] == "kick-heavy")
    assert kick["evidence"]["band_energy_low"] == 0.675
    assert kick["evidence"]["min_band_energy_low"] == THRESHOLDS["kick_heavy_low_ratio"]


# ---------------------------------------------------------------------------
# W8B: the centroid migration, pinned against its eight-track corpus
# ---------------------------------------------------------------------------
# Transcribed from `calibration/v5/*/analysis/*.json` (essentia, the default
# backend). Real music rather than synthesis, and it is the evidence behind
# `noisy_centroid_hz`, `dark_centroid_hz` and `vocal_centroid_min_hz` now
# reading `centroid_energy_hz`. No audio and no stems are needed: these are the
# committed numbers the pipeline itself wrote.
#
# The point of the table is the SPREAD between the two columns. They are not two
# scales of one quantity — on a stem that is 89% kick they differ by 10x and
# both are correct.
CORPUS_SPECTRAL: dict[str, dict[str, float]] = {
    # stem                     centroid_mean  centroid_energy_hz  band_low
    "madonna/drums": {"mean": 4389.98, "energy": 411.79, "low": 0.890089},
    "badu/drums": {"mean": 2722.42, "energy": 187.19, "low": 0.958900},
    "chameleon/drums": {"mean": 2754.24, "energy": 501.86, "low": 0.779000},
    "roni/drums": {"mean": 3737.53, "energy": 1275.72, "low": 0.616000},
    "showers/drums": {"mean": 7791.96, "energy": 7335.63, "low": 0.001000},
    "eno/bass": {"mean": 473.75, "energy": 68.71, "low": 1.000000},
    "roni/bass": {"mean": 744.86, "energy": 78.99, "low": 0.989000},
    "levee/bass": {"mean": 334.32, "energy": 120.44, "low": 0.978000},
    "madonna/bass": {"mean": 1004.75, "energy": 138.81, "low": 0.917000},
    "badu/bass": {"mean": 371.75, "energy": 159.93, "low": 0.924000},
    "chameleon/bass": {"mean": 415.86, "energy": 161.42, "low": 0.864000},
    "badu/vocals": {"mean": 1867.61, "energy": 846.39, "low": 0.029000},
    "madonna/vocals": {"mean": 2712.51, "energy": 762.68, "low": 0.332000},
    "levee/vocals": {"mean": 3252.68, "energy": 1280.53, "low": 0.017000},
    "roni/vocals": {"mean": 3351.14, "energy": 1417.52, "low": 0.143000},
}

CORPUS_AUDIBLE_BASS = ("eno/bass", "roni/bass", "levee/bass", "madonna/bass", "badu/bass",
                       "chameleon/bass")
CORPUS_REAL_VOCALS = ("badu/vocals", "madonna/vocals", "levee/vocals", "roni/vocals")


def test_no_corpus_bass_stem_could_ever_reach_the_dark_centroid_on_the_frame_mean() -> None:
    """Why `sustained sub` migrated. Against `centroid_mean` the label had never
    fired on any real material: the quietest-registered bass stem in the corpus
    still reads 334 Hz, over the 250 Hz low-band edge the threshold is grounded
    in. Against the corrected descriptor every one of them clears it."""
    ceiling = THRESHOLDS["dark_centroid_hz"]
    for stem in CORPUS_AUDIBLE_BASS:
        row = CORPUS_SPECTRAL[stem]
        assert row["mean"] > ceiling, f"{stem} would have fired on the frame mean"
        assert row["energy"] < ceiling, f"{stem} does not read as low-register energy"


def test_a_real_sustained_sub_is_labelled_and_a_busy_bass_is_not() -> None:
    """Eno's bass: 68.7 Hz energy centroid, 1.000 low-band ratio, 0.37
    onsets/sec. The one thing in the corpus that is a sustained sub. Madonna's
    bass sits in the same register and plays 3.26 onsets/sec, so the
    onset clause — not the centroid — is what separates them."""
    eno = label_bass(
        make_analysis(
            "bass", centroid_energy_hz=CORPUS_SPECTRAL["eno/bass"]["energy"], onset_density=0.37
        )
    )
    assert "sustained sub" in names(eno)
    assert by_name(eno, "sustained sub").confidence > 0.9

    madonna = label_bass(
        make_analysis(
            "bass",
            centroid_energy_hz=CORPUS_SPECTRAL["madonna/bass"]["energy"],
            onset_density=3.26,
        )
    )
    assert "sustained sub" not in names(madonna)


@pytest.mark.parametrize("stem", ["madonna/drums", "badu/drums", "chameleon/drums", "roni/drums"])
def test_a_kick_dominated_drum_stem_is_not_called_noisy(stem: str) -> None:
    """The false positive the `noisy` migration removed. Each of these stems has
    most of its energy under 250 Hz and a frame-mean centroid over the 2500 Hz
    threshold, because the frames between kicks hold tails and air. A near-flat
    chroma is supplied so only the centroid clause can decide."""
    row = CORPUS_SPECTRAL[stem]
    assert row["low"] > 0.6, "this test only means something on a low-heavy stem"
    assert row["mean"] > THRESHOLDS["noisy_centroid_hz"], "would not have fired before either"
    labels = label_generic(
        make_analysis("drums", centroid_energy_hz=row["energy"], chroma_entropy_value=0.98)
    )
    assert "noisy" not in names(labels)


def test_the_one_genuinely_broadband_drum_stem_is_still_called_noisy() -> None:
    """The other side: showers-of-gold's drums stem is cymbals and nothing else
    — 0.001 of its energy below 250 Hz — and must keep the label."""
    row = CORPUS_SPECTRAL["showers/drums"]
    labels = label_generic(
        make_analysis("drums", centroid_energy_hz=row["energy"], chroma_entropy_value=0.95)
    )
    assert "noisy" in names(labels)


@pytest.mark.parametrize("stem", CORPUS_REAL_VOCALS)
def test_every_real_vocal_stem_lands_inside_the_vocal_centroid_window(stem: str) -> None:
    """Why the vocal window migrated. Two of these four sat outside even the
    widened 3000 Hz ceiling on the frame mean and got no label at all; on the
    corrected descriptor all four land inside the 500-2500 Hz range the
    threshold comment described in the first place."""
    row = CORPUS_SPECTRAL[stem]
    assert row["energy"] >= THRESHOLDS["vocal_centroid_min_hz"]
    assert row["energy"] <= THRESHOLDS["vocal_centroid_max_hz"]
    assert 500.0 <= row["energy"] <= 2500.0


def test_two_real_vocal_stems_were_excluded_by_the_frame_mean_ceiling() -> None:
    """Pins the defect rather than only the fix: Levee Breaks and Roni Size read
    3253 and 3351 Hz on `centroid_mean`, over the ceiling, and 1281 and 1418 Hz
    on the corrected descriptor."""
    ceiling = THRESHOLDS["vocal_centroid_max_hz"]
    for stem in ("levee/vocals", "roni/vocals"):
        assert CORPUS_SPECTRAL[stem]["mean"] > ceiling
        assert CORPUS_SPECTRAL[stem]["energy"] < ceiling


def test_the_two_centroids_are_not_a_rescaling_of_each_other() -> None:
    """W4D's warning, made a failing test if anyone ever 'converts' between
    them. The ratio between the two columns runs from 1.06 to 14.5 across the
    corpus — there is no factor, and a threshold moved by one would be wrong
    everywhere else."""
    ratios = [row["mean"] / row["energy"] for row in CORPUS_SPECTRAL.values()]
    assert min(ratios) < 1.1
    assert max(ratios) > 10.0
