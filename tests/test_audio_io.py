"""Tests for decoding, channel handling, and the never-downsample rule."""

from __future__ import annotations

import builtins
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from audio_pipeline import ANALYSIS_SAMPLE_RATE
from audio_pipeline.audio_io import (
    AudioDecodeError,
    FFmpegNotFoundError,
    ensure_stereo,
    load_audio,
    to_mono,
)


def test_load_mono_returns_1d_float32(
    wav_file: Callable[..., Path], sine_a440: np.ndarray, sample_rate: int
) -> None:
    path = wav_file(sine_a440, sample_rate)
    audio, rate = load_audio(path)

    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert rate == ANALYSIS_SAMPLE_RATE
    assert audio.shape == sine_a440.shape
    assert np.allclose(audio, sine_a440, atol=1e-6)


def test_load_stereo_is_channel_last(
    wav_file: Callable[..., Path], stereo_pink_noise: np.ndarray, sample_rate: int
) -> None:
    """Shape convention: `(n_samples, n_channels)`, never channel-first."""
    path = wav_file(stereo_pink_noise, sample_rate)
    audio, rate = load_audio(path, mono=False)

    assert audio.ndim == 2
    assert audio.shape == (stereo_pink_noise.shape[0], 2)
    assert audio.dtype == np.float32
    assert rate == ANALYSIS_SAMPLE_RATE


def test_mono_file_loaded_as_stereo_keeps_one_channel(
    wav_file: Callable[..., Path], sine_a440: np.ndarray, sample_rate: int
) -> None:
    """A mono source stays `(n, 1)`; use `ensure_stereo` to widen it."""
    path = wav_file(sine_a440, sample_rate)
    audio, _ = load_audio(path, mono=False)

    assert audio.shape == (sine_a440.size, 1)
    assert ensure_stereo(audio).shape == (sine_a440.size, 2)


def test_stereo_source_averaged_when_mono_requested(
    wav_file: Callable[..., Path], stereo_pink_noise: np.ndarray, sample_rate: int
) -> None:
    path = wav_file(stereo_pink_noise, sample_rate)
    audio, _ = load_audio(path, mono=True)

    assert audio.ndim == 1
    assert np.allclose(audio, stereo_pink_noise.mean(axis=1), atol=1e-6)


def test_22050_input_is_upsampled_not_silently_accepted(
    wav_file: Callable[..., Path], sample_rate: int
) -> None:
    """The load-bearing accuracy rule: a half-rate file must not pass through.

    Accepting 22.05 kHz would cap the spectrum at 11 kHz and quietly corrupt
    every centroid, rolloff, and high-band ratio downstream.
    """
    half_rate = ANALYSIS_SAMPLE_RATE // 2
    t = np.arange(half_rate * 2, dtype=np.float64) / half_rate
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    path = wav_file(tone, half_rate, name="half_rate.wav")

    audio, rate = load_audio(path)

    assert rate == ANALYSIS_SAMPLE_RATE
    assert audio.size == pytest.approx(tone.size * 2, rel=0.01)
    # Duration is preserved: 2 s in, 2 s out.
    assert audio.size / rate == pytest.approx(2.0, abs=0.01)
    # And it is still a 440 Hz tone, not garbage.
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64)))
    peak_hz = float(np.fft.rfftfreq(audio.size, 1.0 / rate)[int(np.argmax(spectrum))])
    assert peak_hz == pytest.approx(440.0, abs=2.0)


def test_upsampling_falls_back_to_interpolation_without_scipy(
    wav_file: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """scipy ships with the librosa extra, so a core-only install must still work.

    It warns rather than downsampling or silently accepting the low rate.
    """
    real_import = builtins.__import__

    def no_scipy(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("scipy"):
            raise ImportError("scipy is not installed")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_scipy)

    half_rate = ANALYSIS_SAMPLE_RATE // 2
    t = np.arange(half_rate, dtype=np.float64) / half_rate
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    path = wav_file(tone, half_rate, name="half_rate_no_scipy.wav")

    with pytest.warns(RuntimeWarning, match="scipy"):
        audio, rate = load_audio(path)

    assert rate == ANALYSIS_SAMPLE_RATE
    assert audio.dtype == np.float32
    assert audio.size == pytest.approx(tone.size * 2, rel=0.01)


def test_higher_rate_input_is_never_downsampled(wav_file: Callable[..., Path]) -> None:
    """48 kHz stays 48 kHz. The rate travels with the samples."""
    high_rate = 48000
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(high_rate) / high_rate)).astype(np.float32)
    path = wav_file(tone, high_rate, name="high_rate.wav")

    audio, rate = load_audio(path)

    assert rate == high_rate
    assert audio.size == tone.size


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "nope.wav")


def test_undecodable_file_raises_audio_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "not_audio.wav"
    path.write_bytes(b"this is not a wav file at all")

    with pytest.raises(AudioDecodeError):
        load_audio(path)


def test_ffmpeg_missing_raises_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When soundfile rejects a file and FFmpeg is absent, say so plainly."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    path = tmp_path / "mystery.m4a"
    path.write_bytes(b"\x00\x01\x02\x03")

    with pytest.raises(FFmpegNotFoundError) as excinfo:
        load_audio(path)

    assert "ffmpeg" in str(excinfo.value).lower()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not on PATH")
def test_ffmpeg_fallback_decodes_m4a(
    tmp_path: Path, wav_file: Callable[..., Path], sine_a440: np.ndarray, sample_rate: int
) -> None:
    """soundfile cannot read m4a; the FFmpeg path must, at the native rate."""
    source = wav_file(sine_a440[: sample_rate // 2], sample_rate, name="source.wav")
    target = tmp_path / "encoded.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(source), str(target)],
        check=True,
    )

    audio, rate = load_audio(target)

    assert rate == ANALYSIS_SAMPLE_RATE
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.size > 0


def test_to_mono_and_ensure_stereo_roundtrip(stereo_pink_noise: np.ndarray) -> None:
    mono = to_mono(stereo_pink_noise)
    assert mono.ndim == 1
    assert mono.dtype == np.float32

    widened = ensure_stereo(mono)
    assert widened.shape == (mono.size, 2)
    assert np.array_equal(widened[:, 0], widened[:, 1])

    assert ensure_stereo(stereo_pink_noise).shape == stereo_pink_noise.shape


def test_silence_loads_without_error(
    wav_file: Callable[..., Path], silence: np.ndarray, sample_rate: int
) -> None:
    path = wav_file(silence, sample_rate)
    audio, rate = load_audio(path)

    assert rate == ANALYSIS_SAMPLE_RATE
    assert audio.size == silence.size
    assert not np.any(audio)
