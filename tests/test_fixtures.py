"""Guards on the synthetic fixtures' ground truth.

Backend tests assert real numbers against these signals, so the signals
themselves need checking: shape and dtype unconditionally, and the librosa
measurements that the docstrings quote whenever librosa is installed. If a
fixture drifts, this file fails before the backend tests start lying.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    CLICK_TRACK_BPM,
    CLICK_TRACK_ONSET_COUNT,
    CLICK_TRACK_ONSET_DENSITY,
    FIXTURE_DURATION_SECONDS,
    SINE_FREQUENCY_HZ,
    SWUNG_CLICK_BPM,
    SWUNG_LONG_IOI_SECONDS,
    SWUNG_SHORT_IOI_SECONDS,
)

from audio_pipeline import ANALYSIS_SAMPLE_RATE

EXPECTED_SAMPLES = int(ANALYSIS_SAMPLE_RATE * FIXTURE_DURATION_SECONDS)


@pytest.mark.parametrize(
    "fixture_name",
    ["click_track_120bpm", "swung_click_8ths", "sine_a440", "white_noise", "silence"],
)
def test_mono_fixtures_are_float32_8s_at_full_rate(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    audio = request.getfixturevalue(fixture_name)
    assert audio.dtype == np.float32
    assert audio.shape == (EXPECTED_SAMPLES,)
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 1.0


def test_stereo_pink_noise_is_channel_last_and_decorrelated(
    stereo_pink_noise: np.ndarray,
) -> None:
    assert stereo_pink_noise.dtype == np.float32
    assert stereo_pink_noise.shape == (EXPECTED_SAMPLES, 2)
    left, right = stereo_pink_noise[:, 0], stereo_pink_noise[:, 1]
    assert not np.array_equal(left, right)
    assert abs(float(np.corrcoef(left, right)[0, 1])) < 0.1


def test_noise_fixtures_are_reproducible(
    white_noise: np.ndarray, stereo_pink_noise: np.ndarray
) -> None:
    """Fixed seeds, so a backend test asserting a number stays true tomorrow."""
    assert float(white_noise[:5].sum()) == pytest.approx(0.0386628, abs=1e-6)
    assert float(np.std(stereo_pink_noise)) == pytest.approx(0.0612736, abs=1e-6)


def test_white_noise_crest_factor(white_noise: np.ndarray) -> None:
    rms = float(np.sqrt(np.mean(white_noise.astype(np.float64) ** 2)))
    assert float(np.max(np.abs(white_noise))) / rms == pytest.approx(4.73, abs=0.05)


def test_silence_is_actually_silent(silence: np.ndarray) -> None:
    assert not np.any(silence)
    assert float(np.sqrt(np.mean(silence**2))) == 0.0


def test_sine_is_a440(sine_a440: np.ndarray) -> None:
    spectrum = np.abs(np.fft.rfft(sine_a440.astype(np.float64)))
    frequencies = np.fft.rfftfreq(sine_a440.size, 1.0 / ANALYSIS_SAMPLE_RATE)
    assert float(frequencies[int(np.argmax(spectrum))]) == pytest.approx(
        SINE_FREQUENCY_HZ, abs=1.0
    )
    # Crest factor of a sine is sqrt(2).
    rms = float(np.sqrt(np.mean(sine_a440.astype(np.float64) ** 2)))
    assert float(np.max(np.abs(sine_a440))) / rms == pytest.approx(np.sqrt(2), rel=0.01)


def test_click_fixtures_are_transient_not_continuous(
    click_track_120bpm: np.ndarray, sine_a440: np.ndarray
) -> None:
    """Clicks are short shaped bursts, so most of the file is near-silent."""
    quiet_fraction = float(np.mean(np.abs(click_track_120bpm) < 1e-4))
    assert quiet_fraction > 0.9

    click_crest = float(np.max(np.abs(click_track_120bpm))) / float(
        np.sqrt(np.mean(click_track_120bpm.astype(np.float64) ** 2))
    )
    sine_crest = float(np.max(np.abs(sine_a440))) / float(
        np.sqrt(np.mean(sine_a440.astype(np.float64) ** 2))
    )
    assert click_crest > sine_crest * 3


def test_swung_iois_are_two_to_one(swung_click_8ths: np.ndarray) -> None:
    """Read the hit positions straight off the waveform, no detector involved."""
    loud = np.flatnonzero(np.abs(swung_click_8ths) > 0.2)
    gaps = np.flatnonzero(np.diff(loud) > ANALYSIS_SAMPLE_RATE * 0.05)
    starts = np.concatenate([loud[:1], loud[gaps + 1]]) / ANALYSIS_SAMPLE_RATE
    iois = np.diff(starts)

    long_iois = iois[iois > 0.3]
    short_iois = iois[iois <= 0.3]
    assert long_iois.mean() == pytest.approx(SWUNG_LONG_IOI_SECONDS, abs=0.01)
    assert short_iois.mean() == pytest.approx(SWUNG_SHORT_IOI_SECONDS, abs=0.01)
    assert long_iois.mean() / short_iois.mean() == pytest.approx(2.0, rel=0.05)


def test_click_track_iois_are_uniform(click_track_120bpm: np.ndarray) -> None:
    loud = np.flatnonzero(np.abs(click_track_120bpm) > 0.2)
    gaps = np.flatnonzero(np.diff(loud) > ANALYSIS_SAMPLE_RATE * 0.05)
    starts = np.concatenate([loud[:1], loud[gaps + 1]]) / ANALYSIS_SAMPLE_RATE

    assert starts.size == CLICK_TRACK_ONSET_COUNT
    assert np.allclose(np.diff(starts), 0.5, atol=0.005)


# --- librosa-measured ground truth quoted in the fixture docstrings ----------


def test_click_track_measures_120_bpm_and_2_onsets_per_second(
    click_track_120bpm: np.ndarray,
) -> None:
    librosa = pytest.importorskip("librosa")

    tempo, _ = librosa.beat.beat_track(y=click_track_120bpm, sr=ANALYSIS_SAMPLE_RATE)
    onsets = librosa.onset.onset_detect(
        y=click_track_120bpm, sr=ANALYSIS_SAMPLE_RATE, units="time"
    )

    assert float(np.atleast_1d(tempo)[0]) == pytest.approx(CLICK_TRACK_BPM, abs=2.0)
    assert len(onsets) == CLICK_TRACK_ONSET_COUNT
    assert len(onsets) / FIXTURE_DURATION_SECONDS == pytest.approx(CLICK_TRACK_ONSET_DENSITY)


def test_swung_click_measures_100_bpm(swung_click_8ths: np.ndarray) -> None:
    librosa = pytest.importorskip("librosa")

    tempo, _ = librosa.beat.beat_track(y=swung_click_8ths, sr=ANALYSIS_SAMPLE_RATE)
    assert float(np.atleast_1d(tempo)[0]) == pytest.approx(SWUNG_CLICK_BPM, abs=2.0)


def test_noise_is_far_brighter_than_the_sine(
    white_noise: np.ndarray, sine_a440: np.ndarray, silence: np.ndarray
) -> None:
    librosa = pytest.importorskip("librosa")

    def centroid(audio: np.ndarray) -> float:
        return float(
            librosa.feature.spectral_centroid(y=audio, sr=ANALYSIS_SAMPLE_RATE).mean()
        )

    assert centroid(sine_a440) == pytest.approx(SINE_FREQUENCY_HZ, abs=20.0)
    assert centroid(white_noise) > 8000.0
    assert centroid(white_noise) > centroid(sine_a440) * 10
    # The edge case: silence must not produce NaN.
    assert np.isfinite(centroid(silence))


def test_pink_noise_sits_between_sine_and_white_noise(
    stereo_pink_noise: np.ndarray, white_noise: np.ndarray
) -> None:
    librosa = pytest.importorskip("librosa")

    pink_mono = stereo_pink_noise.mean(axis=1)
    pink_centroid = float(
        librosa.feature.spectral_centroid(y=pink_mono, sr=ANALYSIS_SAMPLE_RATE).mean()
    )
    white_centroid = float(
        librosa.feature.spectral_centroid(y=white_noise, sr=ANALYSIS_SAMPLE_RATE).mean()
    )
    assert 200.0 < pink_centroid < white_centroid
