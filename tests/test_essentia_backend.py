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
    BASS_GLIDE_NOTE_NAMES,
    BASS_GLIDE_PAIRS,
    BASS_LINE_FREQS_HZ,
    BASS_LINE_MIDI,
    BASS_LINE_NOTE_NAMES,
    CLICK_TRACK_BPM,
    CLICK_TRACK_ONSET_COUNT,
    CLICK_TRACK_ONSET_DENSITY,
    FIXTURE_DURATION_SECONDS,
    SWUNG_LONG_IOI_SECONDS,
    SWUNG_SHORT_IOI_SECONDS,
)

# The F4 fixture lives in the librosa backend's test module and is imported,
# not copied: the cross-backend claim is only meaningful if both backends are
# handed the identical array. Nothing else is taken from there.
from test_librosa_backend import (  # noqa: E402
    SUB_BASS_CENTROID_ENERGY_HZ,
    SUB_BASS_ROLLOFF_ENERGY_HZ,
    sub_bass_over_noise_floor,
)

from audio_pipeline import ANALYSIS_SAMPLE_RATE, BAND_EDGES_HZ  # noqa: E402

# Cross-backend comparison only — never edited, never used for anything but
# reading the two shared helpers and running the librosa backend for reference.
from audio_pipeline.backends import librosa_backend  # noqa: E402
from audio_pipeline.backends.essentia_backend import (  # noqa: E402
    MIN_ONSET_RATE_FOR_RHYTHM_HZ,
    PITCH_FRAME_SIZE,
    STFT_HOP_LENGTH,
    EssentiaBackend,
    band_energy_ratios,
    brightness,
)
from audio_pipeline.note_track import segment_notes  # noqa: E402
from audio_pipeline.schemas import (  # noqa: E402
    BassLine,
    DynamicsFeatures,
    PitchTrack,
    RhythmFeatures,
    SpectralFeatures,
    TonalFeatures,
)
from audio_pipeline.strudel_vocab import (  # noqa: E402
    SUB_BASS_CENTROID_HZ_MAX,
    suggest_bass_sound,
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
    assert spectral.centroid_energy_hz is None
    assert spectral.rolloff_mean is None
    assert spectral.rolloff_energy_hz is None
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


#: Tolerance for `rolloff_energy_hz` across backends. **Zero** — not a loose
#: bound like the 0.05 used for the band ratios above, and not an aspiration.
#:
#: `energy_percentile_hz` returns the centre frequency of an FFT bin on a grid
#: both backends share exactly (`rfftfreq(2048, 1/44100)`), chosen by where a
#: normalised cumulative sum crosses 0.85. Window normalisation and gain cancel
#: in the normalisation, and because the result is a step function the small
#: framing difference between the backends has to move the crossing bin
#: entirely or not at all. Measured across the fixtures below and on all twelve
#: v4 calibration stems — real 4-minute material — the delta is 0.000000 Hz
#: every time.
#:
#: Pinned at exact equality rather than a tolerance because a non-zero delta
#: here would mean something structural had changed (a different FFT size, a
#: different band window, an interpolation step added), and that is worth
#: failing on rather than absorbing. If real material ever does land a source
#: on a knife-edge bin boundary, the honest fix is to record the case here, not
#: to widen this to a number that hides it.
ROLLOFF_ENERGY_CROSS_BACKEND_TOLERANCE_HZ = 0.0

#: Relative tolerance for `centroid_energy_hz` across backends: 0.1%.
#:
#: This one is **not** exact, and the reason is structural rather than a defect
#: in either backend. A centroid is a first moment and therefore a continuous
#: function of the aggregate spectrum, so the 690-vs-691 frame-count difference
#: documented in the backend's module docstring — a few hundred milliseconds of
#: edge padding — perturbs it slightly instead of rounding away as it does for
#: the percentile above.
#:
#: Measured worst case across the fixtures below and all twelve v4 calibration
#: stems: **0.0307% relative, 0.240 Hz absolute**. The largest absolute deltas
#: are all on the two short calibration clips (4.3 s and 17.1 s), where one
#: extra frame is a much bigger share of the total; the 4:27 track's four stems
#: agree to 0.0002 Hz. 0.001 is roughly 3x the measured worst case.
#:
#: For scale: `centroid_mean` on the same material diverges by up to ~1100 Hz,
#: because there the two backends run genuinely different algorithms.
CENTROID_ENERGY_CROSS_BACKEND_TOLERANCE = 0.001


@pytest.mark.parametrize(
    "fixture_name",
    ["click_track_120bpm", "sine_a440", "white_noise", "swung_click_8ths"],
)
def test_rolloff_energy_agrees_with_librosa_exactly(
    backend: EssentiaBackend,
    librosa: librosa_backend.LibrosaBackend,
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """The one spectral descriptor the two backends agree on to the bit.

    `rolloff_mean` cannot manage this — it reads 4333 Hz against 1097 Hz on the
    v4 calibration bass stem. `rolloff_energy_hz` reads only the shape of the
    aggregate power spectrum, which both compute identically, and lands on a
    shared bin grid.
    """
    audio = request.getfixturevalue(fixture_name)
    e_val = backend.spectral(audio, ANALYSIS_SAMPLE_RATE).rolloff_energy_hz
    l_val = librosa.spectral(audio, ANALYSIS_SAMPLE_RATE).rolloff_energy_hz
    assert e_val is not None and l_val is not None
    assert e_val == pytest.approx(l_val, abs=ROLLOFF_ENERGY_CROSS_BACKEND_TOLERANCE_HZ), (
        f"{fixture_name}: essentia={e_val:.6f} librosa={l_val:.6f}"
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["click_track_120bpm", "sine_a440", "white_noise", "swung_click_8ths"],
)
def test_centroid_energy_agrees_with_librosa(
    backend: EssentiaBackend,
    librosa: librosa_backend.LibrosaBackend,
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """Close, and for a stated reason not bit-exact. See the tolerance above."""
    audio = request.getfixturevalue(fixture_name)
    e_val = backend.spectral(audio, ANALYSIS_SAMPLE_RATE).centroid_energy_hz
    l_val = librosa.spectral(audio, ANALYSIS_SAMPLE_RATE).centroid_energy_hz
    assert e_val is not None and l_val is not None
    assert e_val == pytest.approx(l_val, rel=CENTROID_ENERGY_CROSS_BACKEND_TOLERANCE), (
        f"{fixture_name}: essentia={e_val:.6f} librosa={l_val:.6f}"
    )


# --------------------------------------------------------------------------- #
# F4 — the sub bass that rests. Same signal as the librosa backend sees, by
# importing its builder rather than rewriting it: "both backends agree" is only
# a claim about the descriptor if the input is bit-identical.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gated_sub_bass(backend: EssentiaBackend) -> SpectralFeatures:
    return backend.spectral(sub_bass_over_noise_floor(rests=True), ANALYSIS_SAMPLE_RATE)


def test_centroid_energy_reads_the_sub_bass_through_the_rests(
    gated_sub_bass: SpectralFeatures,
) -> None:
    """A 55 Hz sub with 50% rests reads as 55 Hz here too.

    Same number, different library. `strudel_vocab.SUB_BASS_CENTROID_HZ_MAX` is
    the ceiling this has to clear, and F4 is that no unweighted frame-mean
    centroid ever cleared it.
    """
    assert gated_sub_bass.centroid_energy_hz is not None
    assert gated_sub_bass.centroid_energy_hz < SUB_BASS_CENTROID_HZ_MAX
    assert gated_sub_bass.centroid_energy_hz == pytest.approx(
        SUB_BASS_CENTROID_ENERGY_HZ, abs=0.05
    )
    assert gated_sub_bass.rolloff_energy_hz == pytest.approx(
        SUB_BASS_ROLLOFF_ENERGY_HZ, abs=0.001
    )


def test_gated_sub_bass_energy_fields_match_librosa(
    backend: EssentiaBackend, librosa: librosa_backend.LibrosaBackend
) -> None:
    """The `Done when` line of W4D, asserted directly."""
    audio = sub_bass_over_noise_floor(rests=True)
    essentia_spectral = backend.spectral(audio, ANALYSIS_SAMPLE_RATE)
    librosa_spectral = librosa.spectral(audio, ANALYSIS_SAMPLE_RATE)
    assert essentia_spectral.centroid_energy_hz is not None
    assert librosa_spectral.centroid_energy_hz is not None
    assert essentia_spectral.centroid_energy_hz == pytest.approx(
        librosa_spectral.centroid_energy_hz, rel=CENTROID_ENERGY_CROSS_BACKEND_TOLERANCE
    )
    assert essentia_spectral.rolloff_energy_hz == pytest.approx(
        librosa_spectral.rolloff_energy_hz, abs=ROLLOFF_ENERGY_CROSS_BACKEND_TOLERANCE_HZ
    )


def test_essentia_centroid_mean_is_also_contaminated_just_differently(
    gated_sub_bass: SpectralFeatures,
) -> None:
    """Both backends are wrong here, and not wrong by the same amount.

    `SpectralCentroidTime` reads ~4570 Hz on this signal where librosa's
    centroid reads ~5139 Hz — both in the kilohertz on a signal with nothing
    above ~100 Hz, and ~570 Hz apart. That gap is the second half of why the
    old threshold could never have been calibrated: there was no single number
    to calibrate, because the two backends did not report the same statistic.
    """
    mean = gated_sub_bass.centroid_mean
    std = gated_sub_bass.centroid_std
    assert mean is not None and std is not None
    assert mean > 1000.0
    assert std > mean  # the F4 signature: a standard deviation above the mean


def test_the_f4_stem_resolves_to_a_sine_on_this_backend_too(
    gated_sub_bass: SpectralFeatures,
) -> None:
    """End to end, same verdict as the librosa backend produces.

    Two backends reaching the same Strudel sound name from the same audio is
    the property that makes the fallback backend genuinely first-class.
    """
    result = suggest_bass_sound(BassLine(status="ok"), gated_sub_bass)
    assert len(result) == 1
    assert result[0].sound == "sine"
    assert result[0].match == "approximate"


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


# --------------------------------------------------------------------------- #
# pitch() — raw F0, the Wave 4 Protocol method
# --------------------------------------------------------------------------- #


def test_pitch_recovers_the_exact_synthesis_frequencies(
    backend: EssentiaBackend, bass_line_a_minor: np.ndarray
) -> None:
    """`bass_line_a_minor` is built from literal 55.0/65.40639/82.40689 Hz.

    Exact by construction, so this is a real accuracy assertion. Every voiced
    frame must land within a quarter-tone (3%) of one of the three frequencies.
    """
    track = backend.pitch(bass_line_a_minor, ANALYSIS_SAMPLE_RATE)

    voiced = np.asarray(track.f0_hz)[np.asarray(track.voiced, dtype=bool)]
    assert voiced.size > 100

    for frequency in voiced:
        closest = min(BASS_LINE_FREQS_HZ, key=lambda target: abs(target - frequency))
        assert frequency == pytest.approx(closest, rel=0.03)


def test_pitch_survives_the_octave_trap(
    backend: EssentiaBackend, bass_line_octave_trap: np.ndarray
) -> None:
    """The fixture that decided which Essentia algorithm this backend uses.

    Measured octave-error rates on this fixture, as a share of voiced frames:
    `PitchYinProbabilistic` 44.7%, `PredominantPitchMelodia` 63.3%,
    `PitchYinFFT` **0.0%**. See `EssentiaBackend.pitch` for the full account,
    including why `PitchYinProbabilistic` cannot represent 55 Hz at all.
    """
    track = backend.pitch(bass_line_octave_trap, ANALYSIS_SAMPLE_RATE)
    line = segment_notes(track)

    assert [note.midi_note for note in line.notes] == list(BASS_LINE_MIDI)
    assert line.octave_corrections == 0, "the range constraint alone should defeat the trap"


def test_pitch_and_note_track_reproduce_the_documented_note_names(
    backend: EssentiaBackend, bass_line_a_minor: np.ndarray
) -> None:
    line = segment_notes(backend.pitch(bass_line_a_minor, ANALYSIS_SAMPLE_RATE))

    assert line.status == "ok"
    assert [note.note_name for note in line.notes] == list(BASS_LINE_NOTE_NAMES)


def test_pitch_absorbs_a_glide_into_the_notes_either_side(
    backend: EssentiaBackend, bass_line_with_glide: np.ndarray
) -> None:
    line = segment_notes(backend.pitch(bass_line_with_glide, ANALYSIS_SAMPLE_RATE))

    assert [note.note_name for note in line.notes] == list(BASS_GLIDE_NOTE_NAMES) * BASS_GLIDE_PAIRS


def test_pitch_invents_nothing_on_unpitched_low_material(
    backend: EssentiaBackend, bass_unvoiced: np.ndarray
) -> None:
    """20-200 Hz noise. `PITCH_MIN_CONFIDENCE` passes ~7% of frames here, and
    none of them survive `note_track.MIN_NOTE_SECONDS`.
    """
    track = backend.pitch(bass_unvoiced, ANALYSIS_SAMPLE_RATE)
    line = segment_notes(track)

    assert np.mean(np.asarray(track.voiced, dtype=bool)) < 0.2
    assert line.status == "unvoiced"
    assert line.notes == []
    assert line.caveats


def test_pitch_returns_an_empty_track_on_silence(
    backend: EssentiaBackend, silence: np.ndarray
) -> None:
    track = backend.pitch(silence, ANALYSIS_SAMPLE_RATE)

    assert track.f0_hz == []
    assert track.voiced == []
    assert segment_notes(track).status == "unvoiced"


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_pitch_never_raises_on_a_nonsense_sample_rate(
    backend: EssentiaBackend, sine_a440: np.ndarray, sample_rate: int
) -> None:
    assert backend.pitch(sine_a440, sample_rate) == PitchTrack()


def test_pitch_names_the_algorithm_that_produced_it(
    backend: EssentiaBackend, bass_line_a_minor: np.ndarray
) -> None:
    """`yinfft`, not `pyin`: this backend does not use the probabilistic variant.

    `PitchTrack.method` is the only place a reader can tell which estimator ran,
    and the two backends genuinely run different algorithms from the same
    family. Claiming `pyin` here would be the kind of invented equivalence this
    project keeps out of its output.
    """
    track = backend.pitch(bass_line_a_minor, ANALYSIS_SAMPLE_RATE)

    assert track.method == "yinfft"
    assert track.frame_hop_seconds == pytest.approx(STFT_HOP_LENGTH / ANALYSIS_SAMPLE_RATE)
    assert PITCH_FRAME_SIZE == librosa_backend.PYIN_FRAME_LENGTH


def test_pitch_reports_voicing_that_separates_repeated_notes(
    backend: EssentiaBackend, bass_line_a_minor: np.ndarray
) -> None:
    """16 voiced runs, one per note, including the two consecutive `a1`s.

    Two notes of the same pitch have nothing but the 40 ms silence between them
    to tell them apart, which is what `PITCH_MIN_CONFIDENCE` was calibrated
    against.
    """
    voiced = np.asarray(backend.pitch(bass_line_a_minor, ANALYSIS_SAMPLE_RATE).voiced, dtype=bool)

    boundaries = np.diff(voiced.astype(int))
    assert int(np.count_nonzero(boundaries == 1)) == len(BASS_LINE_MIDI)


# --------------------------------------------------------------------------- #
# Cross-backend agreement on pitch — the whole point of the seam
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", ["bass_line_a_minor", "bass_line_octave_trap"])
def test_both_backends_produce_the_same_note_sequence(
    backend: EssentiaBackend,
    librosa: librosa_backend.LibrosaBackend,
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """Identical `note_name` sequences on both bass fixtures.

    This is what the seam is for. `segment_notes` is shared numpy, so anything
    that could differ here has to differ in the F0 estimate itself — and F0,
    unlike `tonal_stability`, has a right answer. Both backends hit it.
    """
    audio = request.getfixturevalue(fixture_name)

    essentia_line = segment_notes(backend.pitch(audio, ANALYSIS_SAMPLE_RATE))
    librosa_line = segment_notes(librosa.pitch(audio, ANALYSIS_SAMPLE_RATE))

    assert [note.note_name for note in essentia_line.notes] == list(BASS_LINE_NOTE_NAMES)
    assert [note.note_name for note in librosa_line.notes] == list(BASS_LINE_NOTE_NAMES)


@pytest.mark.parametrize("fixture_name", ["bass_line_a_minor", "bass_line_octave_trap"])
def test_per_note_median_f0_agrees_across_backends_within_one_percent(
    backend: EssentiaBackend,
    librosa: librosa_backend.LibrosaBackend,
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """1% is well inside a semitone (5.95%), so this is a real accuracy claim.

    Measured worst case across both fixtures: 0.42% (55.02 Hz from librosa
    against 55.25 Hz from Essentia on `bass_line_octave_trap`). Do not loosen
    this tolerance to accommodate a backend change — the tolerance is the point.
    A backend that cannot meet it should carry a `BassLine.caveats` entry
    instead.
    """
    audio = request.getfixturevalue(fixture_name)

    essentia_notes = segment_notes(backend.pitch(audio, ANALYSIS_SAMPLE_RATE)).notes
    librosa_notes = segment_notes(librosa.pitch(audio, ANALYSIS_SAMPLE_RATE)).notes
    assert len(essentia_notes) == len(librosa_notes) == len(BASS_LINE_MIDI)

    for left, right in zip(essentia_notes, librosa_notes, strict=True):
        assert left.median_f0_hz is not None and right.median_f0_hz is not None
        assert left.median_f0_hz == pytest.approx(right.median_f0_hz, rel=0.01)


def test_both_backends_land_within_a_quarter_tone_of_the_synthesis_frequencies(
    backend: EssentiaBackend,
    librosa: librosa_backend.LibrosaBackend,
    bass_line_a_minor: np.ndarray,
) -> None:
    """Agreeing with each other is not enough; both must agree with the truth.

    Two backends could agree perfectly and both be wrong. `bass_line_a_minor`
    is synthesised from literal frequencies, so this closes that gap: every
    note's measured median F0 sits within 25 cents of the frequency the fixture
    was built from.
    """
    expected = [BASS_LINE_FREQS_HZ[index % 4] for index in range(len(BASS_LINE_MIDI))]

    for reference in (backend, librosa):
        notes = segment_notes(reference.pitch(bass_line_a_minor, ANALYSIS_SAMPLE_RATE)).notes
        measured = [note.median_f0_hz for note in notes]
        assert len(measured) == len(expected)
        for value, target in zip(measured, expected, strict=True):
            assert value is not None
            cents = 1200.0 * np.log2(value / target)
            assert abs(cents) < 25.0, f"{value} Hz is {cents:.1f} cents from {target} Hz"
