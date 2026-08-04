"""Synthetic audio fixtures with known ground truth.

No test in this repo may need a real audio file. Every signal here is generated
from numpy with a fixed seed, so backend tests can assert real numbers rather
than "it did not crash".

Everything is `np.float32` at `ANALYSIS_SAMPLE_RATE` (44.1 kHz) and
`FIXTURE_DURATION_SECONDS` (8 s) long, mono `(n_samples,)` unless the fixture
name says stereo.

The ground truth in each docstring was measured against librosa 0.11 at 44.1
kHz, not assumed — in particular the click bursts are short shaped noise
transients (not single samples), and the trains start at
`CLICK_TRAIN_OFFSET_SECONDS` because an onset landing at t=0 has no preceding
frame for the detector to see a rise against, and gets dropped.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_pipeline import ANALYSIS_SAMPLE_RATE

FIXTURE_DURATION_SECONDS = 8.0

#: Lead-in before the first click. Without it the detector misses onset one.
CLICK_TRAIN_OFFSET_SECONDS = 0.25

#: Length of one click burst in samples (~6 ms at 44.1 kHz).
CLICK_LENGTH_SAMPLES = 264

# --- Ground truth, exported so tests can assert against named values ---------
CLICK_TRACK_BPM = 120.0
CLICK_TRACK_ONSET_COUNT = 16
CLICK_TRACK_ONSET_DENSITY = 2.0  # 16 onsets / 8 s
SWUNG_CLICK_BPM = 100.0
SWUNG_LONG_IOI_SECONDS = 0.4  # long-short 2:1 swing at 100 BPM
SWUNG_SHORT_IOI_SECONDS = 0.2
SINE_FREQUENCY_HZ = 440.0  # A4 -> key A


def _n_samples(duration: float = FIXTURE_DURATION_SECONDS) -> int:
    return int(round(ANALYSIS_SAMPLE_RATE * duration))


def _click(seed: int = 7) -> np.ndarray:
    """One click: an exponentially damped noise burst with a 1.8 kHz tone.

    Broadband enough for onset detection to fire on every hit, short enough
    (~6 ms) that 0.2 s inter-onset intervals stay clearly separated.
    """
    rng = np.random.default_rng(seed)
    n = CLICK_LENGTH_SAMPLES
    envelope = np.exp(-np.linspace(0.0, 9.0, n))
    tone = np.sin(2 * np.pi * 1800.0 * np.arange(n) / ANALYSIS_SAMPLE_RATE)
    body = rng.standard_normal(n) * 0.6 + tone * 0.4
    return (envelope * body).astype(np.float32)


def _click_train(onset_times: np.ndarray, duration: float = FIXTURE_DURATION_SECONDS) -> np.ndarray:
    """Place a click burst at each time in `onset_times`."""
    out = np.zeros(_n_samples(duration), dtype=np.float32)
    click = _click()
    for time_s in onset_times:
        start = int(round(float(time_s) * ANALYSIS_SAMPLE_RATE))
        end = start + click.size
        if end <= out.size:
            out[start:end] += click
    return np.clip(out, -1.0, 1.0).astype(np.float32)


@pytest.fixture(scope="session")
def sample_rate() -> int:
    """The one sample rate this project uses: 44100 Hz."""
    return ANALYSIS_SAMPLE_RATE


@pytest.fixture(scope="session")
def click_track_120bpm() -> np.ndarray:
    """8 s click train at 120 BPM, mono 44.1 kHz.

    Ground truth:
      * BPM 120 (librosa beat_track measures 120.19; assert within +/- 2)
      * 16 clicks, evenly spaced 0.5 s apart, first at t=0.25 s
      * onset density 2.0 onsets/sec
      * inter-onset intervals uniformly 0.5 s -> straight, not swung
      * percussive: high crest factor, broadband transients
    """
    return _click_train(CLICK_TRAIN_OFFSET_SECONDS + np.arange(16) * 0.5)


@pytest.fixture(scope="session")
def swung_click_8ths() -> np.ndarray:
    """8 s click train, swung 8ths at 100 BPM, mono 44.1 kHz.

    Ground truth:
      * BPM 100 (librosa measures 99.38)
      * inter-onset intervals alternate 0.4 s / 0.2 s — a 2:1 long-short ratio,
        i.e. triplet swing, against a 0.6 s beat
      * 26 clicks, first at t=0.25 s
      * a subdivision detector must call this swung, not straight 16ths
    """
    times: list[float] = []
    time_s = CLICK_TRAIN_OFFSET_SECONDS
    index = 0
    while time_s < FIXTURE_DURATION_SECONDS - 0.1:
        times.append(time_s)
        time_s += SWUNG_LONG_IOI_SECONDS if index % 2 == 0 else SWUNG_SHORT_IOI_SECONDS
        index += 1
    return _click_train(np.asarray(times))


@pytest.fixture(scope="session")
def sine_a440() -> np.ndarray:
    """8 s pure 440 Hz sine at 0.5 amplitude, mono 44.1 kHz.

    Ground truth:
      * key A (A4 = 440 Hz), major or minor is not meaningful for one pitch
      * spectral centroid ~441 Hz with near-zero variance across frames
      * tonal stability high — chroma is identical frame to frame
      * onset density ~0: one attack at most, no transients
      * crest factor sqrt(2) ~ 1.414 for a sine
    """
    t = np.arange(_n_samples(), dtype=np.float64) / ANALYSIS_SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * SINE_FREQUENCY_HZ * t)).astype(np.float32)


@pytest.fixture(scope="session")
def white_noise() -> np.ndarray:
    """8 s gaussian white noise, peak-normalised to 0.9, mono 44.1 kHz, seed 0.

    Peak-normalised rather than scaled by a constant so it stays inside full
    scale — gaussian tails reach ~4.7 sigma over 8 s and would otherwise clip.

    Ground truth:
      * spectral centroid ~11 kHz — roughly sr/4, and far above the sine's
      * spectral rolloff high (energy spread flat to Nyquist)
      * tonal stability low, key confidence low: chroma is flat and unstable
      * band energy roughly proportional to each band's width in Hz
      * crest factor ~4.7 (gaussian peak/RMS over 8 s), well above a sine's 1.41
    """
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(_n_samples())
    return (noise / np.max(np.abs(noise)) * 0.9).astype(np.float32)


@pytest.fixture(scope="session")
def stereo_pink_noise() -> np.ndarray:
    """8 s pink noise, stereo `(n_samples, 2)`, 44.1 kHz, seed 1.

    Two independent channels, so this exercises the stereo load path that
    integrated loudness needs (pyloudnorm, and Essentia's LoudnessEBUR128,
    which refuses mono input).

    Ground truth:
      * shape (352800, 2), dtype float32
      * 1/f power spectrum: centroid well below white noise's ~11 kHz
      * integrated loudness around -20 LUFS at this scaling — finite and well
        clear of -70 LUFS (the gate that means "silence")
    """
    rng = np.random.default_rng(1)
    n = _n_samples()
    channels = []
    for _ in range(2):
        spectrum = np.fft.rfft(rng.standard_normal(n))
        frequencies = np.fft.rfftfreq(n, d=1.0 / ANALYSIS_SAMPLE_RATE)
        scale = np.ones_like(frequencies)
        scale[1:] = 1.0 / np.sqrt(frequencies[1:])
        shaped = np.fft.irfft(spectrum * scale, n=n)
        channels.append(shaped / np.max(np.abs(shaped)) * 0.3)
    return np.stack(channels, axis=1).astype(np.float32)


@pytest.fixture(scope="session")
def silence() -> np.ndarray:
    """8 s of digital silence, mono 44.1 kHz. The divide-by-zero edge case.

    Ground truth: every descriptor should come back `None` (and be listed in
    `unavailable_features`) or a defined zero. Nothing may raise, and nothing
    may return NaN or inf — RMS is 0, so crest factor and any level-normalised
    ratio have no denominator.
    """
    return np.zeros(_n_samples(), dtype=np.float32)


@pytest.fixture
def wav_file(tmp_path: Path) -> Callable[..., Path]:
    """Factory writing an array to a temp wav. `wav_file(audio, rate, name)`.

    For `audio_io` tests only — analysis tests should take the arrays directly.
    """

    def _write(
        audio: np.ndarray,
        rate: int = ANALYSIS_SAMPLE_RATE,
        name: str = "fixture.wav",
        subtype: str = "FLOAT",
    ) -> Path:
        path = tmp_path / name
        sf.write(path, audio, rate, subtype=subtype)
        return path

    return _write
