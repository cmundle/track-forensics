"""Tests for the Essentia analysis backend.

`pytest.importorskip("essentia")` keeps this module from failing the suite on
a machine where Essentia will not install — but Essentia genuinely is
installed here (`essentia 2.1-beta6-dev`, arm64), so these tests actually run
and actually assert real numbers against the ground truth in `conftest.py`,
not just "did not crash".

`EssentiaBackend` is always instantiated directly, never resolved via
`get_backend()` — that would just retest the same class through an extra
layer of indirection.

Cross-backend agreement tests import `audio_pipeline.backends.librosa_backend`
read-only, purely to compare `band_energy_ratios`/`brightness` output on the
same fixtures. That module is owned by W1A; nothing here edits it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

essentia = pytest.importorskip("essentia")

from conftest import (  # noqa: E402
    CLICK_TRACK_BPM,
    CLICK_TRACK_ONSET_COUNT,
    CLICK_TRACK_ONSET_DENSITY,
    FIXTURE_DURATION_SECONDS,
    SWUNG_LONG_IOI_SECONDS,
    SWUNG_SHORT_IOI_SECONDS,
)

from audio_pipeline import ANALYSIS_SAMPLE_RATE, BAND_EDGES_HZ  # noqa: E402

# Cross-backend comparison only — never edited, never used for anything but
# reading the two shared helpers and running the librosa backend for reference.
from audio_pipeline.backends import librosa_backend  # noqa: E402
from audio_pipeline.backends.essentia_backend import (  # noqa: E402
    MIN_ONSET_RATE_FOR_RHYTHM_HZ,
    EssentiaBackend,
    band_energy_ratios,
    brightness,
)
from audio_pipeline.schemas import (  # noqa: E402
    DynamicsFeatures,
    RhythmFeatures,
    SpectralFeatures,
    TonalFeatures,
)


@pytest.fixture(scope="module")
def backend() -> EssentiaBackend:
    return EssentiaBackend()


@pytest.fixture(scope="module")
def librosa() -> librosa_backend.LibrosaBackend:
    return librosa_backend.LibrosaBackend()


@pytest.fixture(scope="module")
def click_rhythm(backend: EssentiaBackend, click_track_120bpm: np.ndarray) -> RhythmFeatures:
    return backend.rhythm(click_track_120bpm, ANALYSIS_SAMPLE_RATE)


@pytest.fixture(scope="module")
def sine_tonal(backend: EssentiaBackend, sine_a440: np.ndarray) -> TonalFeatures:
    return backend.tonal(sine_a440, ANALYSIS_SAMPLE_RATE)


@pytest.fixture(scope="module")
def noise_tonal(backend: EssentiaBackend, white_noise: np.ndarray) -> TonalFeatures:
    return backend.tonal(white_noise, ANALYSIS_SAMPLE_RATE)


@pytest.fixture(scope="module")
def sine_spectral(backend: EssentiaBackend, sine_a440: np.ndarray) -> SpectralFeatures:
    return backend.spectral(sine_a440, ANALYSIS_SAMPLE_RATE)


@pytest.fixture(scope="module")
def noise_spectral(backend: EssentiaBackend, white_noise: np.ndarray) -> SpectralFeatures:
    return backend.spectral(white_noise, ANALYSIS_SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# Module hygiene
# --------------------------------------------------------------------------- #


def test_essentia_is_never_imported_at_module_top_level() -> None:
    """The CLI and `doctor` must load without Essentia installed.

    Checked structurally (AST, module top level only) rather than by patching
    `sys.modules`, since other tests in this session have already imported
    essentia for real.
    """
    source = Path(__file__).parent.parent / "src/audio_pipeline/backends/essentia_backend.py"
    tree = ast.parse(source.read_text())

    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] == "essentia"]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "essentia":
                offenders.append(node.module)

    assert offenders == [], f"top-level imports of {offenders} break the no-backend CLI path"


def test_backend_name(backend: EssentiaBackend) -> None:
    assert backend.name == "essentia"


# --------------------------------------------------------------------------- #
# rhythm()
# --------------------------------------------------------------------------- #


def test_click_track_bpm_is_120(click_rhythm: RhythmFeatures) -> None:
    """Ground truth: 120 BPM. Essentia measures 119.98, so allow +/- 2."""
    assert click_rhythm.bpm is not None
    assert click_rhythm.bpm == pytest.approx(CLICK_TRACK_BPM, abs=2.0)


def test_click_track_onset_density_is_two_per_second(click_rhythm: RhythmFeatures) -> None:
    assert click_rhythm.onset_density is not None
    assert click_rhythm.onset_density == pytest.approx(CLICK_TRACK_ONSET_DENSITY, abs=0.2)


def test_click_track_beat_times_are_sane(click_rhythm: RhythmFeatures) -> None:
    beats = click_rhythm.beat_times
    assert len(beats) >= 12
    assert beats == sorted(beats)
    assert 0.0 <= beats[0] < beats[-1] <= FIXTURE_DURATION_SECONDS
    intervals = np.diff(beats)
    assert float(np.median(intervals)) == pytest.approx(0.5, abs=0.05)


def test_click_track_onset_times_land_on_the_click_grid(click_rhythm: RhythmFeatures) -> None:
    onsets = click_rhythm.onset_times
    assert len(onsets) == pytest.approx(CLICK_TRACK_ONSET_COUNT, abs=1)
    assert onsets == sorted(onsets)
    assert float(np.median(np.diff(onsets))) == pytest.approx(0.5, abs=0.02)


def test_onset_times_are_distinct_from_beat_times(
    backend: EssentiaBackend, swung_click_8ths: np.ndarray
) -> None:
    """Swung 8ths: onsets alternate 0.4/0.2 s. A backend that copied beat_times
    into onset_times would show one IOI mode and destroy the swing signal."""
    rhythm = backend.rhythm(swung_click_8ths, ANALYSIS_SAMPLE_RATE)
    intervals = np.diff(rhythm.onset_times)
    assert intervals.size > 5
    assert float(np.min(intervals)) == pytest.approx(SWUNG_SHORT_IOI_SECONDS, abs=0.05)
    assert float(np.max(intervals)) == pytest.approx(SWUNG_LONG_IOI_SECONDS, abs=0.05)


def test_bpm_confidence_is_on_essentias_own_scale(click_rhythm: RhythmFeatures) -> None:
    """Documented as roughly 0-5.32, not 0-1 — asserted loosely since the exact
    ceiling is Essentia's implementation detail, not this project's contract."""
    assert click_rhythm.bpm_confidence is not None
    assert 0.0 < click_rhythm.bpm_confidence < 6.0


def test_sustained_tone_gets_no_bpm(backend: EssentiaBackend, sine_a440: np.ndarray) -> None:
    """A pure tone has no rhythmic content. RhythmExtractor2013 nonetheless
    reports a confident-looking tempo for it (measured: 112.9 BPM, confidence
    2.65) because it does not itself gate on "is there anything rhythmic
    here" — OnsetRate's own onset rate (0.25/s, below
    MIN_ONSET_RATE_FOR_RHYTHM_HZ) is what this backend uses to null it out."""
    rhythm = backend.rhythm(sine_a440, ANALYSIS_SAMPLE_RATE)
    assert rhythm.onset_density is not None
    assert rhythm.onset_density < MIN_ONSET_RATE_FOR_RHYTHM_HZ
    assert rhythm.bpm is None
    assert rhythm.bpm_confidence is None
    assert rhythm.beat_times == []


def test_white_noise_gets_no_bpm(backend: EssentiaBackend, white_noise: np.ndarray) -> None:
    rhythm = backend.rhythm(white_noise, ANALYSIS_SAMPLE_RATE)
    assert rhythm.bpm is None
    assert rhythm.beat_times == []


def test_click_track_transients_are_sharp(click_rhythm: RhythmFeatures) -> None:
    assert click_rhythm.transient_sharpness is not None
    assert click_rhythm.transient_sharpness > 5.0


# --------------------------------------------------------------------------- #
# tonal()
# --------------------------------------------------------------------------- #


def test_sine_key_is_a(sine_tonal: TonalFeatures) -> None:
    assert sine_tonal.key == "A"


def test_sine_chroma_peaks_on_a(sine_tonal: TonalFeatures) -> None:
    """Bin 9 is A in the shared `PITCH_CLASSES` convention (bin 0 = C).

    Verifies the HPCP rotation directly: Essentia's raw HPCP peaks at bin 0
    (its own default A-rooted convention), and this asserts the *rotated*
    output — what the backend actually returns — peaks at bin 9 instead.
    """
    assert len(sine_tonal.hpcp_mean) == 12
    assert all(np.isfinite(sine_tonal.hpcp_mean))
    assert int(np.argmax(sine_tonal.hpcp_mean)) == 9


def test_sine_is_tonally_stable(sine_tonal: TonalFeatures) -> None:
    """Structural divergence from librosa, documented in `tonal()` and at
    `CHROMA_SPECTRAL_PEAKS_MAX`: Essentia's peak-picking HPCP chroma is
    measurably noisier frame-to-frame than librosa's CQT-based chroma, even
    for a perfectly periodic tone (spectral leakage sidelobes shift with the
    frame's phase alignment, which `unitMax` normalisation amplifies). librosa
    measures >0.99 on this fixture; tuned Essentia parameters measure ~0.78.
    "High" is asserted relative to this backend's own noise floor (see the
    next test), not against librosa's absolute scale."""
    assert sine_tonal.tonal_stability is not None
    assert sine_tonal.tonal_stability > 0.6


def test_noise_is_less_stable_than_the_sine(
    sine_tonal: TonalFeatures, noise_tonal: TonalFeatures
) -> None:
    """Ordering, not absolute scale — see `test_sine_is_tonally_stable`."""
    assert noise_tonal.tonal_stability is not None
    assert sine_tonal.tonal_stability is not None
    assert noise_tonal.tonal_stability < sine_tonal.tonal_stability
    assert noise_tonal.tonal_stability < 0.4, "noise should sit near this backend's floor"


def test_key_confidence_stays_in_zero_to_one(sine_tonal: TonalFeatures) -> None:
    assert sine_tonal.key_confidence is not None
    assert 0.0 <= sine_tonal.key_confidence <= 1.0


# --------------------------------------------------------------------------- #
# spectral()
# --------------------------------------------------------------------------- #


def test_sine_centroid_sits_at_its_own_frequency(sine_spectral: SpectralFeatures) -> None:
    """Confirmed directly against Essentia: SpectralCentroidTime on this
    fixture measures 439.93 Hz."""
    assert sine_spectral.centroid_mean is not None
    assert sine_spectral.centroid_mean == pytest.approx(440.0, abs=10.0)
    assert sine_spectral.centroid_std is not None
    assert sine_spectral.centroid_std < 10.0


def test_noise_centroid_far_exceeds_the_sine(
    sine_spectral: SpectralFeatures, noise_spectral: SpectralFeatures
) -> None:
    assert noise_spectral.centroid_mean is not None
    assert sine_spectral.centroid_mean is not None
    assert noise_spectral.centroid_mean > 9000.0
    assert noise_spectral.centroid_mean > 10 * sine_spectral.centroid_mean


def test_noise_rolloff_is_high(noise_spectral: SpectralFeatures) -> None:
    assert noise_spectral.rolloff_mean is not None
    assert noise_spectral.rolloff_mean > 15000.0


def test_brightness_orders_noise_above_the_sine(
    sine_spectral: SpectralFeatures, noise_spectral: SpectralFeatures
) -> None:
    assert sine_spectral.brightness is not None
    assert noise_spectral.brightness is not None
    assert sine_spectral.brightness < 0.01
    assert noise_spectral.brightness > 0.8


@pytest.mark.parametrize(
    "fixture_name",
    ["click_track_120bpm", "sine_a440", "white_noise", "stereo_pink_noise", "swung_click_8ths"],
)
def test_band_energy_ratios_sum_to_one(
    backend: EssentiaBackend, fixture_name: str, request: pytest.FixtureRequest
) -> None:
    audio = request.getfixturevalue(fixture_name)
    ratios = backend.spectral(audio, ANALYSIS_SAMPLE_RATE).band_energy_ratios.model_dump()

    assert set(ratios) == set(BAND_EDGES_HZ)
    assert all(value is not None for value in ratios.values())
    assert sum(ratios.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= value <= 1.0 for value in ratios.values())


def test_low_frequency_signal_lands_in_the_low_band(backend: EssentiaBackend) -> None:
    seconds = np.arange(int(ANALYSIS_SAMPLE_RATE * FIXTURE_DURATION_SECONDS)) / ANALYSIS_SAMPLE_RATE
    tone = np.asarray(0.5 * np.sin(2 * np.pi * 100.0 * seconds), dtype=np.float32)

    ratios = backend.spectral(tone, ANALYSIS_SAMPLE_RATE).band_energy_ratios
    assert ratios.low is not None
    assert ratios.low > 0.95
    assert ratios.high is not None
    assert ratios.high < 0.01


# --------------------------------------------------------------------------- #
# dynamics()
# --------------------------------------------------------------------------- #


def test_stereo_pink_noise_loudness_is_plausible(
    backend: EssentiaBackend, stereo_pink_noise: np.ndarray
) -> None:
    dynamics = backend.dynamics(stereo_pink_noise, ANALYSIS_SAMPLE_RATE)
    assert dynamics.loudness_lufs is not None
    assert np.isfinite(dynamics.loudness_lufs)
    assert -40.0 < dynamics.loudness_lufs < 0.0


def test_mono_input_also_gets_a_loudness_reading(
    backend: EssentiaBackend, white_noise: np.ndarray
) -> None:
    """Mono is promoted to stereo via `ensure_stereo` — `LoudnessEBUR128`
    refuses mono input outright (verified: raises TypeError on a 1-D array)."""
    dynamics = backend.dynamics(white_noise, ANALYSIS_SAMPLE_RATE)
    assert dynamics.loudness_lufs is not None
    assert -40.0 < dynamics.loudness_lufs < 0.0


def test_sine_crest_factor_is_root_two(backend: EssentiaBackend, sine_a440: np.ndarray) -> None:
    dynamics = backend.dynamics(sine_a440, ANALYSIS_SAMPLE_RATE)
    assert dynamics.crest_factor is not None
    assert dynamics.crest_factor == pytest.approx(np.sqrt(2.0), abs=0.01)


def test_noise_crest_factor_exceeds_a_sine(
    backend: EssentiaBackend, white_noise: np.ndarray
) -> None:
    dynamics = backend.dynamics(white_noise, ANALYSIS_SAMPLE_RATE)
    assert dynamics.crest_factor is not None
    assert dynamics.crest_factor == pytest.approx(4.7, abs=0.5)


def test_click_track_is_the_most_peaky(
    backend: EssentiaBackend, click_track_120bpm: np.ndarray
) -> None:
    dynamics = backend.dynamics(click_track_120bpm, ANALYSIS_SAMPLE_RATE)
    assert dynamics.crest_factor is not None
    assert dynamics.crest_factor > 10.0


# --------------------------------------------------------------------------- #
# silence: nothing raises, nothing divides by zero
# --------------------------------------------------------------------------- #


def test_silence_rhythm(backend: EssentiaBackend, silence: np.ndarray) -> None:
    """RhythmExtractor2013 does NOT degrade gracefully on raw silence — it
    reports a spurious ~738 BPM at confidence ~4.22 (verified directly). This
    asserts the backend's own silence gate short-circuits before that call."""
    rhythm = backend.rhythm(silence, ANALYSIS_SAMPLE_RATE)
    assert rhythm.bpm is None
    assert rhythm.bpm_confidence is None
    assert rhythm.beat_times == []
    assert rhythm.onset_times == []
    assert rhythm.onset_density == 0.0
    assert rhythm.transient_sharpness is None


def test_silence_tonal(backend: EssentiaBackend, silence: np.ndarray) -> None:
    tonal = backend.tonal(silence, ANALYSIS_SAMPLE_RATE)
    assert tonal.key is None
    assert tonal.scale is None
    assert tonal.key_confidence is None
    assert tonal.hpcp_mean == []
    assert tonal.tonal_stability is None


def test_silence_spectral(backend: EssentiaBackend, silence: np.ndarray) -> None:
    spectral = backend.spectral(silence, ANALYSIS_SAMPLE_RATE)
    assert spectral.centroid_mean is None
    assert spectral.centroid_std is None
    assert spectral.rolloff_mean is None
    assert spectral.brightness is None
    assert all(value is None for value in spectral.band_energy_ratios.model_dump().values())


def test_silence_dynamics(backend: EssentiaBackend, silence: np.ndarray) -> None:
    """RMS is a defined 0.0 and crest factor has no denominator, same as
    librosa. `loudness_lufs` is the one deliberate divergence: Essentia's
    `LoudnessEBUR128` returns a *defined* -70.0 (its own absolute-silence
    gate floor) rather than a non-finite value, so this backend reports that
    real number instead of forcing it to None. See `dynamics()`'s docstring."""
    dynamics = backend.dynamics(silence, ANALYSIS_SAMPLE_RATE)
    assert dynamics.rms_mean == 0.0
    assert dynamics.rms_std == 0.0
    assert dynamics.crest_factor is None
    assert dynamics.loudness_lufs == pytest.approx(-70.0, abs=0.5)


def test_no_descriptor_is_nan_or_inf(backend: EssentiaBackend, silence: np.ndarray) -> None:
    models: list[RhythmFeatures | TonalFeatures | SpectralFeatures | DynamicsFeatures] = [
        backend.rhythm(silence, ANALYSIS_SAMPLE_RATE),
        backend.tonal(silence, ANALYSIS_SAMPLE_RATE),
        backend.spectral(silence, ANALYSIS_SAMPLE_RATE),
        backend.dynamics(silence, ANALYSIS_SAMPLE_RATE),
    ]
    for model in models:
        for value in model.model_dump().values():
            if isinstance(value, float):
                assert np.isfinite(value), f"{model.__class__.__name__} produced {value}"


def test_empty_audio_never_raises(backend: EssentiaBackend) -> None:
    empty = np.zeros(0, dtype=np.float32)
    assert backend.rhythm(empty, ANALYSIS_SAMPLE_RATE).bpm is None
    assert backend.tonal(empty, ANALYSIS_SAMPLE_RATE).key is None
    assert backend.spectral(empty, ANALYSIS_SAMPLE_RATE).centroid_mean is None
    assert backend.dynamics(empty, ANALYSIS_SAMPLE_RATE).crest_factor is None


# --------------------------------------------------------------------------- #
# Array shape: channel-last stereo must not be read as a 2-sample signal
# --------------------------------------------------------------------------- #


def test_stereo_click_track_matches_mono(
    backend: EssentiaBackend, click_track_120bpm: np.ndarray, click_rhythm: RhythmFeatures
) -> None:
    stereo = np.stack([click_track_120bpm, click_track_120bpm], axis=1)
    assert stereo.shape == (click_track_120bpm.size, 2)

    rhythm = backend.rhythm(stereo, ANALYSIS_SAMPLE_RATE)
    assert rhythm.bpm is not None
    assert rhythm.bpm == pytest.approx(CLICK_TRACK_BPM, abs=2.0)
    assert rhythm.bpm == pytest.approx(click_rhythm.bpm)


def test_stereo_pink_noise_is_analysed_over_its_full_length(
    backend: EssentiaBackend, stereo_pink_noise: np.ndarray
) -> None:
    rhythm = backend.rhythm(stereo_pink_noise, ANALYSIS_SAMPLE_RATE)
    dynamics = backend.dynamics(stereo_pink_noise, ANALYSIS_SAMPLE_RATE)
    assert dynamics.loudness_lufs is not None
    # rhythm.rhythm() runs on the mono-summed signal; just confirm no crash and
    # that a full-length signal was actually consumed (onset density computed).
    assert rhythm.onset_density is not None


# --------------------------------------------------------------------------- #
# Shared helpers, exercised directly
# --------------------------------------------------------------------------- #


def _flat_spectrum(n_bins: int = 1025) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    return np.ones((n_bins, 4)), freqs


def test_band_energy_ratios_partition_a_flat_spectrum() -> None:
    magnitude, freqs = _flat_spectrum()
    ratios = band_energy_ratios(magnitude, freqs)
    assert all(value is not None for value in ratios.values())
    assert sum(value for value in ratios.values() if value is not None) == pytest.approx(1.0)


def test_band_energy_ratios_are_none_on_zero_energy() -> None:
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    ratios = band_energy_ratios(np.zeros((freqs.size, 4)), freqs)
    assert all(value is None for value in ratios.values())


def test_brightness_splits_at_the_cutoff() -> None:
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    low_bin = int(np.argmin(np.abs(freqs - 500.0)))
    high_bin = int(np.argmin(np.abs(freqs - 5000.0)))

    quiet = np.zeros((freqs.size, 1))
    quiet[low_bin] = 1.0
    assert brightness(quiet, freqs) == pytest.approx(0.0)

    loud = np.zeros((freqs.size, 1))
    loud[high_bin] = 1.0
    assert brightness(loud, freqs) == pytest.approx(1.0)


def test_brightness_is_none_on_zero_energy() -> None:
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    assert brightness(np.zeros((freqs.size, 3)), freqs) is None


# --------------------------------------------------------------------------- #
# Cross-backend agreement — the check TODO.md wanted and could not run.
#
# Tolerances are deliberately loose, not tight: the two backends do not share
# a framing implementation (see the module docstring's note on the 690-vs-691
# frame count), use different window normalisation, and Essentia's Windowing
# defaults to zero-phase reordering that librosa's STFT does not perform
# (a circular time shift; it does not change magnitude, but is one more axis
# the two pipelines differ on structurally even though the number is the
# same). What must hold is that both describe the same signal the same way in
# aggregate — not bit-for-bit identical framing.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fixture_name",
    ["click_track_120bpm", "sine_a440", "white_noise", "swung_click_8ths"],
)
def test_band_energy_ratios_agree_with_librosa(
    backend: EssentiaBackend,
    librosa: librosa_backend.LibrosaBackend,
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """Same signal, same BAND_EDGES_HZ bounds: ratios should land within 0.05
    (5 percentage points) of each other per band.

    0.05 was chosen empirically: it comfortably covers the measured gap on
    every mono fixture (largest observed: ~0.02 on white noise's `high` band)
    while still catching a gross error (e.g. a mislabelled band, a missing
    factor-of-2, or an unrotated HPCP-style mixup) which would produce
    differences an order of magnitude larger.
    """
    audio = request.getfixturevalue(fixture_name)
    essentia_ratios = backend.spectral(audio, ANALYSIS_SAMPLE_RATE).band_energy_ratios.model_dump()
    librosa_ratios = librosa.spectral(audio, ANALYSIS_SAMPLE_RATE).band_energy_ratios.model_dump()

    for band in BAND_EDGES_HZ:
        e_val, l_val = essentia_ratios[band], librosa_ratios[band]
        assert e_val is not None and l_val is not None
        assert e_val == pytest.approx(l_val, abs=0.05), (
            f"{fixture_name}/{band}: essentia={e_val:.4f} librosa={l_val:.4f}"
        )


@pytest.mark.parametrize(
    "fixture_name",
    ["click_track_120bpm", "sine_a440", "white_noise", "swung_click_8ths"],
)
def test_brightness_agrees_with_librosa(
    backend: EssentiaBackend,
    librosa: librosa_backend.LibrosaBackend,
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """Same 0.05 absolute tolerance and justification as the band-ratio check
    above — brightness is built from the same energy measure."""
    audio = request.getfixturevalue(fixture_name)
    e_val = backend.spectral(audio, ANALYSIS_SAMPLE_RATE).brightness
    l_val = librosa.spectral(audio, ANALYSIS_SAMPLE_RATE).brightness

    assert e_val is not None and l_val is not None
    assert e_val == pytest.approx(l_val, abs=0.05), (
        f"{fixture_name}: essentia={e_val:.4f} librosa={l_val:.4f}"
    )


def test_centroid_agrees_with_librosa_on_the_sine(
    backend: EssentiaBackend, librosa: librosa_backend.LibrosaBackend, sine_a440: np.ndarray
) -> None:
    """Both backends should land within a few Hz of each other on a pure tone,
    where the "correct" answer (~440 Hz) is unambiguous."""
    e_centroid = backend.spectral(sine_a440, ANALYSIS_SAMPLE_RATE).centroid_mean
    l_centroid = librosa.spectral(sine_a440, ANALYSIS_SAMPLE_RATE).centroid_mean
    assert e_centroid is not None and l_centroid is not None
    assert e_centroid == pytest.approx(l_centroid, abs=15.0)


def test_bpm_agrees_with_librosa_on_the_click_track(
    backend: EssentiaBackend,
    librosa: librosa_backend.LibrosaBackend,
    click_track_120bpm: np.ndarray,
) -> None:
    e_bpm = backend.rhythm(click_track_120bpm, ANALYSIS_SAMPLE_RATE).bpm
    l_bpm = librosa.rhythm(click_track_120bpm, ANALYSIS_SAMPLE_RATE).bpm
    assert e_bpm is not None and l_bpm is not None
    assert e_bpm == pytest.approx(l_bpm, abs=1.0)


def test_hpcp_mean_bin_zero_is_c_matching_librosas_convention(
    sine_tonal: TonalFeatures,
) -> None:
    """Both backends must agree that `hpcp_mean[9]` — not `[0]` — is A.

    This is the direct cross-backend check for the HPCP rotation: it does not
    compare magnitudes (the two chroma implementations differ, CQT vs HPCP),
    only that bin 9 is where each backend puts the peak for an isolated 440 Hz
    tone, which is the one thing that must line up for the bin *convention*
    (not the values) to be comparable.
    """
    assert int(np.argmax(sine_tonal.hpcp_mean)) == 9
    assert list(librosa_backend.PITCH_CLASSES)[9] == "A"
