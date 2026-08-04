"""Decoding and rate handling. The single place audio enters the pipeline.

Array shape conventions — these hold everywhere downstream:

* ``mono=True``  → 1-D ``(n_samples,)``, channels averaged.
* ``mono=False`` → 2-D ``(n_samples, n_channels)``, soundfile's native layout,
  channel-last. A single-channel file gives ``(n_samples, 1)``; use
  :func:`ensure_stereo` when a measurement genuinely needs two channels.

Dtype is always ``np.float32``.

Sample rate — the accuracy-over-speed rule, in code:

* Files at or above :data:`ANALYSIS_SAMPLE_RATE` are returned untouched, at
  their own rate. **Nothing here ever downsamples.** Downsampling to 22.05 kHz
  would cap the spectrum at 11 kHz and quietly wreck every centroid, rolloff,
  and high-band ratio in the project. A 48 kHz file is therefore analysed at
  48 kHz, and the rate travels with the samples in the returned tuple.
* Files below it are resampled **up**, so a 22.05 kHz source never silently
  passes through as if it were full rate.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

from . import ANALYSIS_SAMPLE_RATE

__all__ = [
    "AudioArray",
    "AudioDecodeError",
    "FFmpegNotFoundError",
    "ensure_stereo",
    "load_audio",
    "to_mono",
]

#: Every array crossing this module boundary: float32, shape as documented above.
AudioArray = npt.NDArray[np.float32]

logger = logging.getLogger(__name__)

#: Seconds of wall clock before a stuck FFmpeg decode is abandoned.
_FFMPEG_TIMEOUT_SECONDS = 300


class AudioDecodeError(RuntimeError):
    """Raised when a file cannot be decoded by soundfile or FFmpeg."""


class FFmpegNotFoundError(AudioDecodeError):
    """Raised when FFmpeg is needed for decoding but is not on PATH."""


def to_mono(audio: AudioArray) -> AudioArray:
    """Collapse `(n_samples, n_channels)` to `(n_samples,)` by averaging.

    A 1-D array is returned unchanged.
    """
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return np.asarray(audio.mean(axis=1), dtype=np.float32)


def ensure_stereo(audio: AudioArray) -> AudioArray:
    """Return `(n_samples, 2)`, duplicating a single channel if needed.

    For measurements that structurally require two channels — Essentia's
    `LoudnessEBUR128` being the case that forced this helper into existence.
    More than two channels are truncated to the first two.
    """
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1).astype(np.float32, copy=False)
    return audio[:, :2].astype(np.float32, copy=False)


def load_audio(
    path: Path,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
    mono: bool = True,
) -> tuple[AudioArray, int]:
    """Decode an audio file to float32 samples at `sample_rate` or higher.

    Tries `soundfile` first and shells out to FFmpeg for anything it rejects
    (m4a, and some mp3 builds). The returned rate is the rate of the returned
    samples, which is `sample_rate` unless the source was already higher.

    Args:
        path: File to decode.
        sample_rate: Floor for the output rate. Sources below it are resampled
            up; sources above it are left alone. Defaults to
            `ANALYSIS_SAMPLE_RATE`.
        mono: `True` for `(n_samples,)`, `False` for `(n_samples, n_channels)`.

    Returns:
        `(samples, actual_sample_rate)`.

    Raises:
        FileNotFoundError: if `path` does not exist.
        FFmpegNotFoundError: if FFmpeg is needed for this file but absent.
        AudioDecodeError: if neither decoder can read the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, native_rate = _decode(path)

    if mono:
        audio = to_mono(audio)

    audio, actual_rate = _resample_up(audio, native_rate, sample_rate, source=path)
    return np.ascontiguousarray(audio, dtype=np.float32), actual_rate


def _decode(path: Path) -> tuple[AudioArray, int]:
    """Decode to `(n_samples, n_channels)` float32 at the file's native rate."""
    try:
        audio, native_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises LibsndfileError/RuntimeError
        logger.debug("soundfile could not read %s (%s); falling back to FFmpeg", path, exc)
        audio, native_rate = _decode_via_ffmpeg(path, soundfile_error=exc)
    return np.asarray(audio, dtype=np.float32), int(native_rate)


def _decode_via_ffmpeg(path: Path, soundfile_error: Exception) -> tuple[AudioArray, int]:
    """Transcode to float32 WAV in a temp dir, then read that with soundfile.

    Going via a WAV file rather than a raw pipe keeps the source's own sample
    rate and channel count: FFmpeg copies both into the WAV header, so nothing
    has to be probed or guessed up front, and no accidental resampling occurs.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FFmpegNotFoundError(
            f"soundfile could not decode {path.name} ({soundfile_error}) and FFmpeg is not "
            "on PATH. Install it with: brew install ffmpeg"
        )

    with tempfile.TemporaryDirectory(prefix="track-forensics-decode-") as tmp:
        wav_path = Path(tmp) / "decoded.wav"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "a:0",
            "-c:a",
            "pcm_f32le",
            "-f",
            "wav",
            str(wav_path),
        ]
        try:
            # Fixed argv, no shell: the only interpolated value is a path.
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioDecodeError(f"FFmpeg timed out decoding {path}") from exc

        if completed.returncode != 0 or not wav_path.is_file():
            raise AudioDecodeError(
                f"FFmpeg failed to decode {path} (exit {completed.returncode}): "
                f"{completed.stderr.strip() or 'no stderr'}"
            )

        try:
            audio, native_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        except Exception as exc:
            raise AudioDecodeError(f"Could not read FFmpeg output for {path}: {exc}") from exc

    return np.asarray(audio, dtype=np.float32), int(native_rate)


def _resample_up(
    audio: AudioArray,
    native_rate: int,
    target_rate: int,
    source: Path | None = None,
) -> tuple[AudioArray, int]:
    """Raise `audio` to `target_rate` if it is below it. Never lowers it."""
    if native_rate == target_rate or audio.size == 0:
        return audio, native_rate

    if native_rate > target_rate:
        logger.info(
            "%s is %d Hz, above the %d Hz analysis floor; keeping the native rate "
            "(this pipeline never downsamples)",
            source if source is not None else "input",
            native_rate,
            target_rate,
        )
        return audio, native_rate

    logger.info(
        "%s is %d Hz, below the %d Hz analysis floor; resampling up",
        source if source is not None else "input",
        native_rate,
        target_rate,
    )
    return _resample(audio, native_rate, target_rate), target_rate


def _resample(audio: AudioArray, native_rate: int, target_rate: int) -> AudioArray:
    """Rate-convert along axis 0.

    Prefers `scipy.signal.resample_poly` (polyphase FIR, properly band-limited).
    scipy ships with the `librosa` extra rather than the core install, so there
    is a linear-interpolation fallback for a core-only environment — audibly
    poorer above ~10 kHz, hence the warning.
    """
    try:
        from scipy.signal import resample_poly
    except ImportError:
        warnings.warn(
            "scipy is not installed; falling back to linear-interpolation resampling, "
            'which is imprecise in the high band. Install it with: pip install -e ".[librosa]"',
            RuntimeWarning,
            stacklevel=3,
        )
        return _resample_linear(audio, native_rate, target_rate)

    gcd = np.gcd(native_rate, target_rate)
    resampled = resample_poly(audio, target_rate // gcd, native_rate // gcd, axis=0)
    return np.asarray(resampled, dtype=np.float32)


def _resample_linear(audio: AudioArray, native_rate: int, target_rate: int) -> AudioArray:
    """Last-resort rate conversion with no third-party dependency."""
    n_out = int(round(audio.shape[0] * target_rate / native_rate))
    source_index = np.arange(audio.shape[0], dtype=np.float64)
    target_index = np.arange(n_out, dtype=np.float64) * native_rate / target_rate

    if audio.ndim == 1:
        return np.interp(target_index, source_index, audio).astype(np.float32)

    channels = [
        np.interp(target_index, source_index, audio[:, ch]) for ch in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)
