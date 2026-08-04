"""Tests for the librosa analysis backend.

Every assertion is against the ground truth documented in `conftest.py`, not
against "it did not crash". Where a number is librosa's own measured output
rather than a mathematical fact (the 120.19 BPM the click train reports, say),
the tolerance says so.

`LibrosaBackend` is always instantiated directly. `get_backend()` is
deliberately never called: Essentia installs fine on this machine and would be
preferred, so routing through the resolver would silently test the other
backend.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    CLICK_TRACK_BPM,
    CLICK_TRACK_ONSET_COUNT,
    CLICK_TRACK_ONSET_DENSITY,
    FIXTURE_DURATION_SECONDS,
    SWUNG_LONG_IOI_SECONDS,
    SWUNG_SHORT_IOI_SECONDS,
)

from audio_pipeline import ANALYSIS_SAMPLE_RATE, BAND_EDGES_HZ
from audio_pipeline.backends.librosa_backend import (
    BRIGHTNESS_CUTOFF_HZ,
    MAX_TRANSIENT_SHARPNESS,
    LibrosaBackend,
    band_energy_ratios,
    brightness,
    transient_sharpness,
)
from audio_pipeline.schemas import (
    DynamicsFeatures,
    RhythmFeatures,
    SpectralFeatures,
    TonalFeatures,
)


@pytest.fixture(scope="module")
def backend() -> LibrosaBackend:
    """The backend under test, constructed directly rather than resolved."""
    return LibrosaBackend()


# Analysis results are cached per module: chroma_cqt over 8 s at 44.1 kHz is
# the expensive call in this suite and several tests want the same result.


@pytest.fixture(scope="module")
def click_rhythm(backend: LibrosaBackend, click_track_120bpm: np.ndarray) -> RhythmFeatures:
    return backend.rhythm(click_track_120bpm, ANALYSIS_SAMPLE_RATE)


@pytest.fixture(scope="module")
def sine_tonal(backend: LibrosaBackend, sine_a440: np.ndarray) -> TonalFeatures:
    return backend.tonal(sine_a440, ANALYSIS_SAMPLE_RATE)


@pytest.fixture(scope="module")
def noise_tonal(backend: LibrosaBackend, white_noise: np.ndarray) -> TonalFeatures:
    return backend.tonal(white_noise, ANALYSIS_SAMPLE_RATE)


@pytest.fixture(scope="module")
def sine_spectral(backend: LibrosaBackend, sine_a440: np.ndarray) -> SpectralFeatures:
    return backend.spectral(sine_a440, ANALYSIS_SAMPLE_RATE)


@pytest.fixture(scope="module")
def noise_spectral(backend: LibrosaBackend, white_noise: np.ndarray) -> SpectralFeatures:
    return backend.spectral(white_noise, ANALYSIS_SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# Module hygiene
# --------------------------------------------------------------------------- #


def test_librosa_is_never_imported_at_module_top_level() -> None:
    """The CLI and `doctor` must load without librosa or pyloudnorm installed.

    Checked structurally rather than by patching `sys.modules`, since other
    tests in this session have already imported librosa for real.
    """
    source = Path(__file__).parent.parent / "src/audio_pipeline/backends/librosa_backend.py"
    tree = ast.parse(source.read_text())

    forbidden = {"librosa", "pyloudnorm"}
    offenders: list[str] = []
    for node in tree.body:  # module top level only, not function bodies
        if isinstance(node, ast.Import):
            offenders += [
                alias.name for alias in node.names if alias.name.split(".")[0] in forbidden
            ]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in forbidden:
                offenders.append(node.module)

    assert offenders == [], f"top-level imports of {offenders} break the no-backend CLI path"


def test_backend_name(backend: LibrosaBackend) -> None:
    assert backend.name == "librosa"


# --------------------------------------------------------------------------- #
# rhythm()
# --------------------------------------------------------------------------- #


def test_click_track_bpm_is_120(click_rhythm: RhythmFeatures) -> None:
    """Ground truth: 120 BPM. librosa measures 120.19, so allow +/- 2."""
    assert click_rhythm.bpm is not None
    assert click_rhythm.bpm == pytest.approx(CLICK_TRACK_BPM, abs=2.0)


def test_click_track_onset_density_is_two_per_second(click_rhythm: RhythmFeatures) -> None:
    """Ground truth: 16 clicks over 8 s = 2.0 onsets/sec."""
    assert click_rhythm.onset_density is not None
    assert click_rhythm.onset_density == pytest.approx(CLICK_TRACK_ONSET_DENSITY, abs=0.2)


def test_click_track_beat_times_are_sane(click_rhythm: RhythmFeatures) -> None:
    """Beats are in seconds, ascending, and inside the 8 s fixture."""
    beats = click_rhythm.beat_times
    assert len(beats) >= 12, "a 120 BPM 8 s track has ~16 beats"
    assert beats == sorted(beats)
    assert 0.0 <= beats[0] < beats[-1] <= FIXTURE_DURATION_SECONDS
    # Beats land on the 0.5 s click grid, not on some frame-index scale.
    intervals = np.diff(beats)
    assert float(np.median(intervals)) == pytest.approx(0.5, abs=0.05)


def test_click_track_transients_are_sharp(click_rhythm: RhythmFeatures) -> None:
    """Isolated bursts over a silent floor saturate the sharpness ceiling."""
    assert click_rhythm.transient_sharpness == pytest.approx(MAX_TRANSIENT_SHARPNESS)


def test_noise_transients_are_dull(backend: LibrosaBackend, white_noise: np.ndarray) -> None:
    """Noise has peaks but no real attacks: sharpness sits near 1.0."""
    sharpness = backend.rhythm(white_noise, ANALYSIS_SAMPLE_RATE).transient_sharpness
    assert sharpness is not None
    assert 1.0 <= sharpness < 3.0


def test_sustained_tone_reports_no_onsets(backend: LibrosaBackend, sine_a440: np.ndarray) -> None:
    """Ground truth for the sine: onset density ~0, no transients.

    Guards the `ONSET_ENVELOPE_FLOOR` gate. librosa normalises the onset
    envelope before peak-picking, so without it a steady sine's numerical floor
    is rescaled into ~18 phantom onsets/sec and a nonsense 140 BPM.
    """
    rhythm = backend.rhythm(sine_a440, ANALYSIS_SAMPLE_RATE)
    assert rhythm.onset_density == 0.0
    assert rhythm.bpm is None
    assert rhythm.beat_times == []
    assert rhythm.transient_sharpness is None


def test_click_track_onset_times_land_on_the_click_grid(click_rhythm: RhythmFeatures) -> None:
    """Ground truth: 16 clicks, first at t=0.25 s, spaced 0.5 s apart.

    `onset_times` is observed rather than inferred, so unlike `beat_times` it
    must land on the actual hits — this is what W1E's subdivision detection
    reads.
    """
    onsets = click_rhythm.onset_times
    assert len(onsets) == pytest.approx(CLICK_TRACK_ONSET_COUNT, abs=1)
    assert onsets == sorted(onsets)
    assert onsets[0] == pytest.approx(0.25, abs=0.05)
    assert float(np.median(np.diff(onsets))) == pytest.approx(0.5, abs=0.02)


def test_onset_times_are_distinct_from_beat_times(
    backend: LibrosaBackend, swung_click_8ths: np.ndarray
) -> None:
    """Swung 8ths: onsets alternate 0.4/0.2 s while beats stay evenly spaced.

    A backend that populated `onset_times` from the beat grid would show a
    single IOI mode here and destroy the swing signal downstream.
    """
    rhythm = backend.rhythm(swung_click_8ths, ANALYSIS_SAMPLE_RATE)
    intervals = np.diff(rhythm.onset_times)
    assert intervals.size > 10
    assert float(np.min(intervals)) == pytest.approx(SWUNG_SHORT_IOI_SECONDS, abs=0.05)
    assert float(np.max(intervals)) == pytest.approx(SWUNG_LONG_IOI_SECONDS, abs=0.05)


def test_bpm_confidence_is_unavailable(click_rhythm: RhythmFeatures) -> None:
    """librosa reports no tempo confidence; the field stays None by design."""
    assert click_rhythm.bpm_confidence is None


# --------------------------------------------------------------------------- #
# tonal()
# --------------------------------------------------------------------------- #


def test_sine_key_is_a(sine_tonal: TonalFeatures) -> None:
    """Ground truth: a 440 Hz sine is A. Mode is not meaningful for one pitch."""
    assert sine_tonal.key == "A"
    assert sine_tonal.scale in {"major", "minor"}


def test_sine_chroma_peaks_on_a(sine_tonal: TonalFeatures) -> None:
    """librosa chroma bin 9 is A. Twelve bins, all finite."""
    assert len(sine_tonal.hpcp_mean) == 12
    assert all(np.isfinite(sine_tonal.hpcp_mean))
    assert int(np.argmax(sine_tonal.hpcp_mean)) == 9


def test_sine_is_tonally_stable(sine_tonal: TonalFeatures) -> None:
    """One unchanging pitch: chroma is identical frame to frame."""
    assert sine_tonal.tonal_stability is not None
    assert sine_tonal.tonal_stability > 0.99


def test_noise_is_less_stable_and_less_confident_than_the_sine(
    sine_tonal: TonalFeatures, noise_tonal: TonalFeatures
) -> None:
    """White noise has no tonal centre; the sine has nothing but.

    Both are ordering assertions rather than absolute thresholds. Chroma is
    per-frame normalised, so a *flat* chroma is also a stable one and noise
    still scores high on frame-to-frame cosine similarity — the ordering holds,
    the separation is narrow, and key confidence is the far better signal.
    """
    assert noise_tonal.tonal_stability is not None
    assert sine_tonal.tonal_stability is not None
    assert noise_tonal.tonal_stability < sine_tonal.tonal_stability

    assert noise_tonal.key_confidence is not None
    assert sine_tonal.key_confidence is not None
    assert noise_tonal.key_confidence < sine_tonal.key_confidence


def test_key_confidence_stays_in_zero_to_one(
    sine_tonal: TonalFeatures, noise_tonal: TonalFeatures
) -> None:
    """The heuristics threshold on this, so the range is part of the contract."""
    for tonal in (sine_tonal, noise_tonal):
        assert tonal.key_confidence is not None
        assert 0.0 <= tonal.key_confidence <= 1.0


def test_a_minor_triad_beats_noise_on_key_confidence(
    backend: LibrosaBackend, white_noise: np.ndarray
) -> None:
    """Real tonal material must clear atonal material by a wide margin."""
    seconds = np.arange(int(ANALYSIS_SAMPLE_RATE * FIXTURE_DURATION_SECONDS)) / ANALYSIS_SAMPLE_RATE
    triad = sum(0.25 * np.sin(2 * np.pi * f * seconds) for f in (220.0, 261.63, 329.63))
    tonal = backend.tonal(np.asarray(triad, dtype=np.float32), ANALYSIS_SAMPLE_RATE)

    assert tonal.key == "A"
    assert tonal.scale == "minor"
    assert tonal.key_confidence is not None
    noise_confidence = backend.tonal(white_noise, ANALYSIS_SAMPLE_RATE).key_confidence
    assert noise_confidence is not None
    assert tonal.key_confidence > 3 * noise_confidence


# --------------------------------------------------------------------------- #
# spectral()
# --------------------------------------------------------------------------- #


def test_sine_centroid_sits_at_its_own_frequency(sine_spectral: SpectralFeatures) -> None:
    """Ground truth: ~441 Hz with near-zero variance across frames."""
    assert sine_spectral.centroid_mean is not None
    assert sine_spectral.centroid_mean == pytest.approx(440.0, abs=25.0)
    assert sine_spectral.centroid_std is not None
    assert sine_spectral.centroid_std < 50.0


def test_noise_centroid_far_exceeds_the_sine(
    sine_spectral: SpectralFeatures, noise_spectral: SpectralFeatures
) -> None:
    """Ground truth: white noise centroid ~11 kHz, i.e. ~sr/4.

    This is also the canary for a 22.05 kHz downsample: at half rate the
    centroid would land near 5.5 kHz and fail the lower bound.
    """
    assert noise_spectral.centroid_mean is not None
    assert sine_spectral.centroid_mean is not None
    assert noise_spectral.centroid_mean > 9000.0
    assert noise_spectral.centroid_mean > 10 * sine_spectral.centroid_mean


def test_noise_rolloff_is_high(noise_spectral: SpectralFeatures) -> None:
    """Energy is flat to Nyquist, so the 85% rolloff sits far up the spectrum."""
    assert noise_spectral.rolloff_mean is not None
    assert noise_spectral.rolloff_mean > 15000.0


def test_brightness_orders_noise_above_the_sine(
    sine_spectral: SpectralFeatures, noise_spectral: SpectralFeatures
) -> None:
    """A 440 Hz sine has essentially no energy above the 1.5 kHz cutoff."""
    assert sine_spectral.brightness is not None
    assert noise_spectral.brightness is not None
    assert sine_spectral.brightness < 0.01
    assert noise_spectral.brightness > 0.8


@pytest.mark.parametrize(
    "fixture_name",
    ["click_track_120bpm", "sine_a440", "white_noise", "stereo_pink_noise", "swung_click_8ths"],
)
def test_band_energy_ratios_sum_to_one(
    backend: LibrosaBackend, fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Present ratios sum to 1.0 on every non-silent fixture, mono or stereo."""
    audio = request.getfixturevalue(fixture_name)
    ratios = backend.spectral(audio, ANALYSIS_SAMPLE_RATE).band_energy_ratios.model_dump()

    assert set(ratios) == set(BAND_EDGES_HZ)
    assert all(value is not None for value in ratios.values())
    assert sum(ratios.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= value <= 1.0 for value in ratios.values())


def test_low_frequency_signal_lands_in_the_low_band(backend: LibrosaBackend) -> None:
    """A 100 Hz sine is inside the 20-250 Hz `low` band and nowhere else."""
    seconds = np.arange(int(ANALYSIS_SAMPLE_RATE * FIXTURE_DURATION_SECONDS)) / ANALYSIS_SAMPLE_RATE
    tone = np.asarray(0.5 * np.sin(2 * np.pi * 100.0 * seconds), dtype=np.float32)

    ratios = backend.spectral(tone, ANALYSIS_SAMPLE_RATE).band_energy_ratios
    assert ratios.low is not None
    assert ratios.low > 0.95
    assert ratios.high is not None
    assert ratios.high < 0.01


def test_white_noise_spreads_across_every_band(noise_spectral: SpectralFeatures) -> None:
    """Ground truth: band energy roughly tracks each band's width in Hz.

    `high` is 14 kHz wide against `low`'s 230, so a flat spectrum is expected to
    be top-heavy — the point of the assertion is that no band is empty.
    """
    ratios = noise_spectral.band_energy_ratios
    values = [ratios.low, ratios.low_mid, ratios.high_mid, ratios.high]
    assert all(value is not None and value > 0.005 for value in values)
    assert ratios.high is not None
    assert ratios.high > ratios.low_mid  # type: ignore[operator]


# --------------------------------------------------------------------------- #
# dynamics()
# --------------------------------------------------------------------------- #


def test_stereo_pink_noise_loudness_is_plausible(
    backend: LibrosaBackend, stereo_pink_noise: np.ndarray
) -> None:
    """Ground truth: a finite negative LUFS, well clear of the -70 gate.

    Also the stereo path: pyloudnorm is handed channel-last `(n, 2)` audio.
    """
    assert stereo_pink_noise.shape == (int(ANALYSIS_SAMPLE_RATE * FIXTURE_DURATION_SECONDS), 2)

    dynamics = backend.dynamics(stereo_pink_noise, ANALYSIS_SAMPLE_RATE)
    assert dynamics.loudness_lufs is not None
    assert np.isfinite(dynamics.loudness_lufs)
    assert -40.0 < dynamics.loudness_lufs < 0.0


def test_mono_input_also_gets_a_loudness_reading(
    backend: LibrosaBackend, white_noise: np.ndarray
) -> None:
    """Mono is duplicated to stereo rather than skipped."""
    dynamics = backend.dynamics(white_noise, ANALYSIS_SAMPLE_RATE)
    assert dynamics.loudness_lufs is not None
    assert -40.0 < dynamics.loudness_lufs < 0.0


def test_sine_crest_factor_is_root_two(backend: LibrosaBackend, sine_a440: np.ndarray) -> None:
    """Ground truth: peak/RMS of a sine is sqrt(2) ~ 1.414."""
    dynamics = backend.dynamics(sine_a440, ANALYSIS_SAMPLE_RATE)
    assert dynamics.crest_factor is not None
    assert dynamics.crest_factor == pytest.approx(np.sqrt(2.0), abs=0.01)
    assert dynamics.rms_mean is not None
    assert dynamics.rms_mean == pytest.approx(0.5 / np.sqrt(2.0), abs=0.01)


def test_noise_crest_factor_exceeds_a_sine(
    backend: LibrosaBackend, white_noise: np.ndarray
) -> None:
    """Ground truth: gaussian peak/RMS over 8 s is ~4.7."""
    dynamics = backend.dynamics(white_noise, ANALYSIS_SAMPLE_RATE)
    assert dynamics.crest_factor is not None
    assert dynamics.crest_factor == pytest.approx(4.7, abs=0.5)


def test_click_track_is_the_most_peaky(
    backend: LibrosaBackend, click_track_120bpm: np.ndarray
) -> None:
    """Sparse loud bursts over silence: a very high crest factor."""
    dynamics = backend.dynamics(click_track_120bpm, ANALYSIS_SAMPLE_RATE)
    assert dynamics.crest_factor is not None
    assert dynamics.crest_factor > 10.0


# --------------------------------------------------------------------------- #
# silence: nothing raises, nothing divides by zero
# --------------------------------------------------------------------------- #


def test_silence_rhythm(backend: LibrosaBackend, silence: np.ndarray) -> None:
    rhythm = backend.rhythm(silence, ANALYSIS_SAMPLE_RATE)
    assert rhythm.bpm is None
    assert rhythm.bpm_confidence is None
    assert rhythm.beat_times == []
    assert rhythm.onset_times == []
    assert rhythm.onset_density == 0.0  # zero onsets over a known duration is a fact
    assert rhythm.transient_sharpness is None


def test_silence_tonal(backend: LibrosaBackend, silence: np.ndarray) -> None:
    tonal = backend.tonal(silence, ANALYSIS_SAMPLE_RATE)
    assert tonal.key is None
    assert tonal.scale is None
    assert tonal.key_confidence is None
    assert tonal.hpcp_mean == []
    assert tonal.tonal_stability is None


def test_silence_spectral(backend: LibrosaBackend, silence: np.ndarray) -> None:
    spectral = backend.spectral(silence, ANALYSIS_SAMPLE_RATE)
    assert spectral.centroid_mean is None
    assert spectral.centroid_std is None
    assert spectral.rolloff_mean is None
    assert spectral.brightness is None
    assert all(value is None for value in spectral.band_energy_ratios.model_dump().values())


def test_silence_dynamics(backend: LibrosaBackend, silence: np.ndarray) -> None:
    """RMS is a defined 0.0; crest factor and LUFS have no denominator."""
    dynamics = backend.dynamics(silence, ANALYSIS_SAMPLE_RATE)
    assert dynamics.rms_mean == 0.0
    assert dynamics.rms_std == 0.0
    assert dynamics.crest_factor is None
    assert dynamics.loudness_lufs is None


def test_no_descriptor_is_nan_or_inf(backend: LibrosaBackend, silence: np.ndarray) -> None:
    """Nothing may return NaN or inf — neither survives JSON serialisation."""
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


def test_empty_audio_never_raises(backend: LibrosaBackend) -> None:
    """A zero-length array is degenerate input, not an error."""
    empty = np.zeros(0, dtype=np.float32)
    assert backend.rhythm(empty, ANALYSIS_SAMPLE_RATE).bpm is None
    assert backend.tonal(empty, ANALYSIS_SAMPLE_RATE).key is None
    assert backend.spectral(empty, ANALYSIS_SAMPLE_RATE).centroid_mean is None
    assert backend.dynamics(empty, ANALYSIS_SAMPLE_RATE).crest_factor is None


# --------------------------------------------------------------------------- #
# Array shape: channel-last stereo must not be read as a 2-sample signal
# --------------------------------------------------------------------------- #


def test_stereo_click_track_matches_mono(
    backend: LibrosaBackend, click_track_120bpm: np.ndarray, click_rhythm: RhythmFeatures
) -> None:
    """`audio_io` gives `(n_samples, 2)`; librosa's own layout is the transpose.

    Reading the fixture channel-first would make it a 2-sample signal and the
    whole analysis would collapse. Duplicating the mono click train into two
    channels must reproduce the mono result exactly.
    """
    stereo = np.stack([click_track_120bpm, click_track_120bpm], axis=1)
    assert stereo.shape == (click_track_120bpm.size, 2)

    rhythm = backend.rhythm(stereo, ANALYSIS_SAMPLE_RATE)
    assert rhythm.bpm is not None
    assert rhythm.bpm == pytest.approx(CLICK_TRACK_BPM, abs=2.0)
    assert rhythm.onset_density == pytest.approx(CLICK_TRACK_ONSET_DENSITY, abs=0.2)
    assert rhythm.beat_times[-1] > 1.0, "beat times collapsed to a near-zero span"
    assert rhythm.bpm == pytest.approx(click_rhythm.bpm)


def test_stereo_spectral_matches_mono(
    backend: LibrosaBackend, white_noise: np.ndarray, noise_spectral: SpectralFeatures
) -> None:
    """The same signal in two channels describes the same spectrum."""
    stereo = np.stack([white_noise, white_noise], axis=1)
    spectral = backend.spectral(stereo, ANALYSIS_SAMPLE_RATE)
    assert spectral.centroid_mean is not None
    assert noise_spectral.centroid_mean is not None
    assert spectral.centroid_mean == pytest.approx(noise_spectral.centroid_mean, rel=1e-6)


def test_stereo_pink_noise_is_analysed_over_its_full_length(
    backend: LibrosaBackend, stereo_pink_noise: np.ndarray
) -> None:
    """Beat times must span the 8 s fixture, not two samples' worth of it."""
    rhythm = backend.rhythm(stereo_pink_noise, ANALYSIS_SAMPLE_RATE)
    assert rhythm.beat_times
    assert rhythm.beat_times[-1] > FIXTURE_DURATION_SECONDS / 2


# --------------------------------------------------------------------------- #
# Shared helpers, exercised directly (W1B mirrors these)
# --------------------------------------------------------------------------- #


def _flat_spectrum(n_bins: int = 1025) -> tuple[np.ndarray, np.ndarray]:
    """A unit-magnitude spectrum on the 2048-point FFT grid at 44.1 kHz."""
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    return np.ones((n_bins, 4)), freqs


def test_band_energy_ratios_partition_a_flat_spectrum() -> None:
    """Bins go to exactly one band, and the ratios sum to 1.0."""
    magnitude, freqs = _flat_spectrum()
    ratios = band_energy_ratios(magnitude, freqs)
    assert all(value is not None for value in ratios.values())
    assert sum(value for value in ratios.values() if value is not None) == pytest.approx(1.0)


def test_band_energy_ratios_place_a_single_bin() -> None:
    """One populated bin puts all the energy in that bin's band."""
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    magnitude = np.zeros((freqs.size, 1))
    magnitude[int(np.argmin(np.abs(freqs - 100.0)))] = 1.0

    ratios = band_energy_ratios(magnitude, freqs)
    assert ratios["low"] == pytest.approx(1.0)
    assert ratios["low_mid"] == pytest.approx(0.0)


def test_band_energy_ratios_are_none_on_zero_energy() -> None:
    """Silence has no ratios — not zeros, and never a division by zero."""
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    ratios = band_energy_ratios(np.zeros((freqs.size, 4)), freqs)
    assert all(value is None for value in ratios.values())


def test_band_energy_ratios_reject_a_mismatched_frequency_axis() -> None:
    """A frequency axis that does not match the spectrum is undefined, not wrong."""
    ratios = band_energy_ratios(np.ones((10, 2)), np.linspace(0.0, 20000.0, 11))
    assert all(value is None for value in ratios.values())


def test_band_energy_ratios_accept_a_single_frame() -> None:
    """A 1-D spectrum is treated as one frame rather than rejected."""
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    ratios = band_energy_ratios(np.ones(freqs.size), freqs)
    assert sum(value for value in ratios.values() if value is not None) == pytest.approx(1.0)


def test_band_energy_ratios_exclude_out_of_range_bins() -> None:
    """Energy below 20 Hz or above 20 kHz leaves the ratios untouched.

    Both are excluded from numerator and denominator, so adding DC energy to a
    spectrum must not change how the in-band energy is divided up.
    """
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    magnitude = np.zeros((freqs.size, 1))
    magnitude[int(np.argmin(np.abs(freqs - 1000.0)))] = 1.0
    before = band_energy_ratios(magnitude, freqs)

    magnitude[0] = 50.0  # DC, below the 20 Hz floor
    magnitude[-1] = 50.0  # 22.05 kHz, above the 20 kHz ceiling
    after = band_energy_ratios(magnitude, freqs)

    assert after == before
    assert after["low_mid"] == pytest.approx(1.0)


def test_brightness_splits_at_the_cutoff() -> None:
    """Energy placed either side of 1.5 kHz lands where it should."""
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    low_bin = int(np.argmin(np.abs(freqs - 500.0)))
    high_bin = int(np.argmin(np.abs(freqs - 5000.0)))

    quiet = np.zeros((freqs.size, 1))
    quiet[low_bin] = 1.0
    assert brightness(quiet, freqs) == pytest.approx(0.0)

    loud = np.zeros((freqs.size, 1))
    loud[high_bin] = 1.0
    assert brightness(loud, freqs) == pytest.approx(1.0)

    # Equal power either side: magnitudes are squared, so equal magnitudes here.
    both = np.zeros((freqs.size, 1))
    both[low_bin] = 1.0
    both[high_bin] = 1.0
    assert brightness(both, freqs) == pytest.approx(0.5)


def test_brightness_is_none_on_zero_energy() -> None:
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    assert brightness(np.zeros((freqs.size, 3)), freqs) is None


def test_brightness_agrees_with_the_band_ratios() -> None:
    """Both share a denominator, so brightness must equal the bands above it.

    With the cutoff at 1.5 kHz, energy placed only in `high_mid` and `high`
    means brightness is exactly those two ratios added together.
    """
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    magnitude = np.zeros((freqs.size, 1))
    for hz, weight in ((100.0, 3.0), (3000.0, 2.0), (10000.0, 1.0)):
        magnitude[int(np.argmin(np.abs(freqs - hz)))] = weight

    ratios = band_energy_ratios(magnitude, freqs)
    expected = (ratios["high_mid"] or 0.0) + (ratios["high"] or 0.0)
    assert brightness(magnitude, freqs) == pytest.approx(expected)


def test_brightness_cutoff_is_1500_hz() -> None:
    assert BRIGHTNESS_CUTOFF_HZ == 1500.0


def test_transient_sharpness_saturates_on_a_silent_floor() -> None:
    """A lone spike in a zero envelope has an infinite ratio: report the ceiling."""
    envelope = np.zeros(100)
    envelope[50] = 5.0
    assert transient_sharpness(envelope, np.array([50]), 10) == MAX_TRANSIENT_SHARPNESS


def test_transient_sharpness_is_one_for_a_flat_envelope() -> None:
    """No transients means peak equals its own local median."""
    envelope = np.ones(100)
    assert transient_sharpness(envelope, np.array([50]), 10) == pytest.approx(1.0)


def test_transient_sharpness_measures_the_ratio() -> None:
    """A peak of 4.0 over a floor of 1.0 is a sharpness of 4.0."""
    envelope = np.ones(101)
    envelope[50] = 4.0
    assert transient_sharpness(envelope, np.array([50]), 10) == pytest.approx(4.0)


def test_transient_sharpness_averages_over_peaks() -> None:
    envelope = np.ones(201)
    envelope[50] = 2.0
    envelope[150] = 4.0
    assert transient_sharpness(envelope, np.array([50, 150]), 10) == pytest.approx(3.0)


def test_transient_sharpness_is_none_without_peaks() -> None:
    assert transient_sharpness(np.ones(100), np.array([], dtype=int), 10) is None
    assert transient_sharpness(np.zeros(0), np.array([0]), 10) is None
    assert transient_sharpness(np.ones(100), np.array([50]), 0) is None


def test_transient_sharpness_ignores_out_of_range_peaks() -> None:
    envelope = np.ones(10)
    assert transient_sharpness(envelope, np.array([99]), 3) is None
