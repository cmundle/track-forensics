"""Decoding and rate handling. The single place audio enters the pipeline.

Array shape conventions — these hold everywhere downstream:

* ``mono=True``  → 1-D ``(n_samples,)``, channels averaged.
* ``mono=False`` → 2-D ``(n_samples, n_channels)``, soundfile's native layout,
  channel-last. A single-channel file gives ``(n_samples, 1)``; use
  :func:`ensure_stereo` when a measurement genuinely needs two channels.

Dtype is always ``np.float32``.

Sample rate — the accuracy-over-speed rule, in code:

**Everything comes back at exactly :data:`ANALYSIS_SAMPLE_RATE` (44,100 Hz).**
Up from lower rates, down from higher ones. 44,100 is a fixed target, never a
variable, and never negotiable downward — see :func:`_to_analysis_rate` for why
44.1 kHz specifically, and why 22.05 kHz would be a catastrophe rather than an
optimisation.

Locking the rate keeps a 48 kHz mix comparable with its own 44.1 kHz Demucs
stems, and keeps every `SourceAnalysis.sample_rate` in agreement with
`TrackSummary.analysis_sample_rate`.
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
    """Decode an audio file to float32 samples at exactly `sample_rate`.

    Tries `soundfile` first and shells out to FFmpeg for anything it rejects
    (m4a, and some mp3 builds). The returned rate always equals `sample_rate`:
    lower sources are resampled up, higher sources down. See
    :func:`_to_analysis_rate` before you touch that behaviour.

    Args:
        path: File to decode.
        sample_rate: Exact output rate. Defaults to `ANALYSIS_SAMPLE_RATE`
            (44,100 Hz), which is the only value the pipeline uses; the
            parameter exists for tests, not for tuning runs.
        mono: `True` for `(n_samples,)`, `False` for `(n_samples, n_channels)`.

    Returns:
        `(samples, sample_rate)`. The second element is returned rather than
        assumed so callers stay honest, but it is always `sample_rate`.

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

    audio, actual_rate = _to_analysis_rate(audio, native_rate, sample_rate, source=path)
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


def _to_analysis_rate(
    audio: AudioArray,
    native_rate: int,
    target_rate: int,
    source: Path | None = None,
) -> tuple[AudioArray, int]:
    """Convert `audio` to exactly `target_rate`, up or down as needed.

    READ THIS BEFORE CHANGING THE TARGET RATE.

    `target_rate` is always `ANALYSIS_SAMPLE_RATE` = 44,100 Hz. It is a fixed
    target, not a variable, and lowering it is not a valid optimisation:

    * 44.1 kHz puts Nyquist at 22.05 kHz, above the top of human hearing and
      above the 20 kHz ceiling of `BAND_EDGES_HZ`. Nothing this tool measures
      lives higher, so converting 48 kHz down to 44.1 kHz discards only the
      22-24 kHz band — inaudible, unmeasured, and free.
    * Dropping to **22.05 kHz would put Nyquist at 11 kHz**, cutting straight
      through hats and cymbals. Every spectral centroid, rolloff, brightness
      figure and `high` band ratio in the project would be silently wrong, and
      the output would still look plausible. That is the disaster this rule
      exists to prevent, and it is why the fix is never "just resample lower
      for speed".

    Converting in both directions is what keeps a 48 kHz input mix comparable
    with its own Demucs stems (which come back at 44.1 kHz), and keeps every
    `SourceAnalysis.sample_rate` consistent with
    `TrackSummary.analysis_sample_rate`.
    """
    if native_rate == target_rate or audio.size == 0:
        return audio, target_rate

    logger.info(
        "%s is %d Hz; converting to the %d Hz analysis rate (%s)",
        source if source is not None else "input",
        native_rate,
        target_rate,
        "up" if native_rate < target_rate else "down",
    )
    return _resample(audio, native_rate, target_rate), target_rate


def _resample(audio: AudioArray, native_rate: int, target_rate: int) -> AudioArray:
    """Rate-convert along axis 0 with `scipy.signal.resample_poly`.

    Polyphase FIR, properly band-limited in both directions: it anti-aliases on
    the way down from 48 kHz and suppresses imaging on the way up from 22.05
    kHz. scipy is a **base dependency**, so this path is the only one any
    supported install takes.

    The linear-interpolation fallback below it is a genuine last resort for a
    broken environment, not a supported configuration — it images badly above
    ~10 kHz, precisely the band this tool exists to measure. It should be
    unreachable; if the warning ever fires, the install is broken.
    """
    try:
        from scipy.signal import resample_poly
    except ImportError:
        warnings.warn(
            "scipy is missing, so resampling fell back to linear interpolation, which is "
            "imprecise above ~10 kHz and will skew spectral descriptors. scipy is a base "
            'dependency of this package: reinstall with pip install -e ".[dev,librosa]"',
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
