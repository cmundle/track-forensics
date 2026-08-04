"""Essentia analysis backend.

The preferred backend when it installs. Its macOS wheel is fragile, so nothing
outside this module may assume Essentia exists.

Implementation notes:

- `essentia.standard` is imported **inside** method bodies only, never at this
  module's top level, so the CLI and `doctor` load fine when Essentia is
  absent.
- Audio arrives from `audio_io` at exactly `sample_rate` (44.1 kHz or higher,
  in practice always exactly `ANALYSIS_SAMPLE_RATE`). Nothing here resamples.
  `RhythmExtractor2013` and `OnsetRate` are hard-wired to 44100 Hz internally
  (neither exposes a `sampleRate` parameter at all), so `rhythm()` refuses to
  run them against anything else rather than silently returning wrong numbers.
  Every other algorithm used here *does* take `sampleRate`, and it is always
  passed the real value explicitly.
- Band-energy ratios and brightness reimplement `librosa_backend`'s contract
  verbatim (see the module docstring there): same `BAND_EDGES_HZ`, same
  linear-magnitude/power/zero-energy-gate rules. This is a deliberate
  duplication rather than an import, so this backend has no load-time
  dependency on `librosa_backend` — only the cross-backend test imports that
  module, and only to compare numbers.
- `LoudnessEBUR128` requires stereo `(n_samples, 2)` input; mono is promoted
  via `audio_io.ensure_stereo`.
- Anything that cannot be computed is `None` on the model, never an exception.

Framing convention, and its documented divergence from librosa:

  Both backends target the same nominal STFT grid — `STFT_N_FFT` (2048),
  `STFT_HOP_LENGTH` (512), Hann window, centred at 44.1 kHz — because
  `band_energy_ratios`/`brightness` are pinned to that resolution. librosa's
  `stft(..., center=True)` reflect-pads by `n_fft // 2` so frame *k* is
  centred on sample `k * hop`. Essentia's `FrameGenerator(..., startFromZero=
  False)` centres the same way but pads/counts frames slightly differently at
  the signal's edges. Measured on an 8 s 44.1 kHz signal: librosa produces 690
  frames, Essentia produces 691 — one extra trailing frame. This is a few
  hundred milliseconds of edge padding out of 8 seconds and has no measurable
  effect on the *energy-summed-over-all-frames* descriptors this module
  reports (band ratios, brightness, rolloff mean, centroid mean/std); it is
  called out here rather than asserted away, per the brief.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .. import BAND_EDGES_HZ
from ..audio_io import ensure_stereo, to_mono
from ..schemas import (
    BandEnergyRatios,
    DynamicsFeatures,
    RhythmFeatures,
    SpectralFeatures,
    TonalFeatures,
)
from . import BackendName

__all__ = [
    "BRIGHTNESS_CUTOFF_HZ",
    "CHROMA_FRAME_SIZE",
    "CHROMA_HOP_SIZE",
    "CHROMA_SPECTRAL_PEAKS_MAX",
    "MAX_TRANSIENT_SHARPNESS",
    "MIN_ONSET_RATE_FOR_RHYTHM_HZ",
    "PITCH_CLASSES",
    "ROLLOFF_PERCENT",
    "STFT_HOP_LENGTH",
    "STFT_N_FFT",
    "EssentiaBackend",
    "band_energy_ratios",
    "brightness",
]

#: Matches `librosa_backend.STFT_N_FFT`. ~21.5 Hz/bin at 44.1 kHz.
STFT_N_FFT = 2048

#: Matches `librosa_backend.STFT_HOP_LENGTH`. ~11.6 ms at 44.1 kHz; the frame
#: rate for every framewise descriptor in this backend (chroma, centroid,
#: rolloff, RMS, the onset-detection envelope).
STFT_HOP_LENGTH = 512

#: Fraction of spectral energy below the reported rolloff frequency. Matches
#: `librosa_backend.ROLLOFF_PERCENT`.
ROLLOFF_PERCENT = 0.85

#: Brightness split point. Matches `librosa_backend.BRIGHTNESS_CUTOFF_HZ`.
BRIGHTNESS_CUTOFF_HZ = 1500.0

#: Lowest/highest frequency any band-relative descriptor considers, derived
#: from `BAND_EDGES_HZ` so the two cannot drift apart. Matches
#: `librosa_backend.BAND_FLOOR_HZ` / `BAND_CEILING_HZ`.
BAND_FLOOR_HZ = min(low for low, _ in BAND_EDGES_HZ.values())
BAND_CEILING_HZ = max(high for _, high in BAND_EDGES_HZ.values())

#: Saturation point for the transient-sharpness ratio. Matches
#: `librosa_backend.MAX_TRANSIENT_SHARPNESS`.
MAX_TRANSIENT_SHARPNESS = 100.0

#: Half-width, in seconds, of the local-median window transient sharpness
#: measures a peak against. Matches `librosa_backend.TRANSIENT_WINDOW_SECONDS`.
TRANSIENT_WINDOW_SECONDS = 0.5

#: Window, in onset-envelope frames either side of an `OnsetRate` onset time,
#: searched for the envelope's true local maximum before measuring sharpness.
#: `OnsetRate` runs its own internal onset-detection grid, which is not
#: identical to the HFC envelope computed here on the shared `STFT_HOP_LENGTH`
#: grid, so its reported onset time can land 1-2 frames off the envelope's
#: actual peak. Measured on the 120 BPM click fixture: the raw onset-time
#: frame index undershoots the true HFC peak by 1-2 frames roughly a third of
#: the time. A window of 2 frames (~23 ms) absorbs that without risking
#: latching onto a neighbouring, unrelated hit at this fixture's 0.5 s
#: spacing.
_PEAK_SEARCH_WINDOW_FRAMES = 2

#: `OnsetRate`'s onsets/sec below which `RhythmExtractor2013`'s BPM/beat grid
#: is discarded as unreliable rather than reported. `RhythmExtractor2013`
#: does not itself gate on "is there anything rhythmic here": measured
#: directly on the project fixtures, it reports a fully-formed, confident-
#: looking tempo for material with no real transients at all -
#: bpm=112.9 confidence=2.65 on the 440 Hz sine (2 spurious onsets, rate
#: 0.25/s), bpm=90.7 confidence=2.40 on white noise (0 onsets, rate 0.0/s),
#: and on pure digital silence bpm=738.3 confidence=4.22 with 98 fabricated
#: ticks - a *higher* confidence than most real material. `OnsetRate`'s own
#: rate, by contrast, correctly reads near-zero for exactly this material
#: (0.25, 0.0, 0.0 onsets/sec respectively) versus 2.0/s on the 120 BPM click
#: fixture and ~3.25/s on the swung-8ths fixture. 0.5 sits comfortably above
#: every non-rhythmic case measured and comfortably below both real fixtures,
#: so it is used as a trust gate: below it, bpm/beat_times/bpm_confidence are
#: nulled, but onset_density and onset_times (which come from `OnsetRate`
#: directly, not from `RhythmExtractor2013`) are kept — they are correct at
#: any rate, including exactly 0.
MIN_ONSET_RATE_FOR_RHYTHM_HZ = 0.5

#: Peak amplitude at or below which a signal is treated as digital silence.
#: `RhythmExtractor2013` in particular does not degrade gracefully on true
#: silence (see `MIN_ONSET_RATE_FOR_RHYTHM_HZ`), so silence is gated out
#: before any Essentia algorithm sees it, not just after.
_SILENCE_THRESHOLD = 1e-12

#: Minimum signal length for `LoudnessEBUR128`'s integrated-loudness figure to
#: be meaningful - matches the meter's own 400 ms gating block, and mirrors
#: `librosa_backend`'s pyloudnorm guard for parity. Essentia's implementation
#: does not itself raise below this length (see the dynamics() docstring for
#: what it returns instead), so this is a deliberate choice for cross-backend
#: consistency rather than a crash-avoidance measure.
_MIN_LOUDNESS_DURATION_SECONDS = 0.4

#: Frame size/hop used only for chroma (HPCP), distinct from the
#: `STFT_N_FFT`/`STFT_HOP_LENGTH` grid the shared spectral contract pins for
#: band ratios/brightness/rolloff/centroid. HPCP has no such pinned contract,
#: and the 2048/512 grid measurably misbehaves for it: see
#: `CHROMA_SPECTRAL_PEAKS_MAX` for the full story and the numbers that led
#: here.
CHROMA_FRAME_SIZE = 4096
CHROMA_HOP_SIZE = 2048

#: `SpectralPeaks(orderBy="magnitude")` is capped to its strongest 5 peaks
#: before HPCP folds them into chroma bins. Empirically necessary: a 440 Hz
#: sine does not land on a bin centre of the 2048-point FFT grid, and because
#: `STFT_HOP_LENGTH` (512 samples) is not an integer number of 440 Hz cycles,
#: the spectral-leakage sidelobe pattern rotates slightly frame to frame.
#: `HPCP`'s default `normalized="unitMax"` amplifies that jitter in the minor
#: bins, and with Essentia's default `SpectralPeaks(maxPeaks=100)` this made
#: the *perfectly periodic* sine measure LESS tonally stable (0.79) than
#: *white noise* (0.83) - backwards, and unusable for the `tonally stable`
#: heuristic. Capping to the 5 strongest peaks, and moving to the larger,
#: coarser `CHROMA_FRAME_SIZE`/`CHROMA_HOP_SIZE` grid above (174 frames over
#: 8 s instead of 691, so leakage jitter is both smaller per-frame and
#: averaged over fewer, longer frames), fixes the ordering with a wide
#: margin: measured sine 0.78, white noise 0.24. This is still well short of
#: the near-1.0 a smooth CQT-based chroma (`librosa_backend`) gets on the
#: same signal - documented as a structural divergence in `tonal()`, not
#: papered over.
CHROMA_SPECTRAL_PEAKS_MAX = 5

#: Chroma bin order: bin 0 is C, bin 9 is A. Must match
#: `librosa_backend.PITCH_CLASSES` exactly - see `_rotate_hpcp_to_c_root` for
#: why this needs a rotation on Essentia's raw HPCP output.
PITCH_CLASSES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

#: Essentia's `HPCP` defaults to `referenceFrequency=440` (concert A), which
#: places pitch class A - not C - at bin 0. Verified directly: an isolated
#: 440 Hz sine's spectral peak fed through `SpectralPeaks` -> `HPCP` produces
#: `argmax(hpcp) == 0`. Rotating the raw 12-bin output right by this many
#: places puts A at index 9 and C at index 0, matching
#: `librosa_backend.PITCH_CLASSES`. See `_rotate_hpcp_to_c_root`.
_HPCP_ROTATE_TO_C = 9


# --------------------------------------------------------------------------- #
# Shared spectral helpers - reimplemented, not imported, from
# `librosa_backend`. This is the canonical contract; both backends must stay
# numerically comparable. See that module's docstrings for the full spec this
# mirrors; kept brief here to avoid duplicating hundreds of comment lines.
# --------------------------------------------------------------------------- #


def band_energy_ratios(
    magnitude: npt.NDArray[np.floating[Any]],
    freqs: npt.NDArray[np.floating[Any]],
) -> dict[str, float | None]:
    """Share of in-band spectral energy per `BAND_EDGES_HZ` band.

    Mirrors `librosa_backend.band_energy_ratios` exactly: `magnitude` is a
    linear-magnitude spectrum (`abs(stft)`-equivalent, not power, not dB),
    shape `(n_bins, n_frames)` (a 1-D `(n_bins,)` spectrum is one frame).
    Energy is `magnitude ** 2` summed over all frames (no per-frame
    normalisation). A bin belongs to exactly one band by its centre
    frequency, half-open `[low, high)`, topmost band closed at 20000 Hz.
    Denominator is total energy in `[BAND_FLOOR_HZ, BAND_CEILING_HZ]`. Every
    band is `None` together on a zero or non-finite in-band total, or on
    empty/mismatched input - never 0.0, never a division by zero.
    """
    undefined: dict[str, float | None] = dict.fromkeys(BAND_EDGES_HZ, None)

    spectrum = np.asarray(magnitude, dtype=np.float64)
    frequencies = np.asarray(freqs, dtype=np.float64)
    if spectrum.size == 0 or frequencies.size == 0:
        return undefined
    if spectrum.ndim == 1:
        spectrum = spectrum[:, np.newaxis]
    if spectrum.ndim != 2 or spectrum.shape[0] != frequencies.shape[0]:
        return undefined

    power = np.square(spectrum).sum(axis=1)
    if not bool(np.all(np.isfinite(power))):
        return undefined

    per_band: dict[str, float] = {}
    for name, (low, high) in BAND_EDGES_HZ.items():
        if high >= BAND_CEILING_HZ:
            in_band = (frequencies >= low) & (frequencies <= high)
        else:
            in_band = (frequencies >= low) & (frequencies < high)
        per_band[name] = float(power[in_band].sum())

    total = float(sum(per_band.values()))
    if not np.isfinite(total) or total <= 0.0:
        return undefined
    return {name: value / total for name, value in per_band.items()}


def brightness(
    magnitude: npt.NDArray[np.floating[Any]],
    freqs: npt.NDArray[np.floating[Any]],
    cutoff_hz: float = BRIGHTNESS_CUTOFF_HZ,
) -> float | None:
    """Fraction of in-band spectral energy above `cutoff_hz`.

    Mirrors `librosa_backend.brightness` exactly: same energy measure and
    same `[BAND_FLOOR_HZ, BAND_CEILING_HZ]` denominator as
    `band_energy_ratios`, so the two agree on what "total energy" means.
    `None` on a zero or non-finite in-band total, or on empty/mismatched
    input.
    """
    spectrum = np.asarray(magnitude, dtype=np.float64)
    frequencies = np.asarray(freqs, dtype=np.float64)
    if spectrum.size == 0 or frequencies.size == 0:
        return None
    if spectrum.ndim == 1:
        spectrum = spectrum[:, np.newaxis]
    if spectrum.ndim != 2 or spectrum.shape[0] != frequencies.shape[0]:
        return None

    power = np.square(spectrum).sum(axis=1)
    if not bool(np.all(np.isfinite(power))):
        return None

    in_range = (frequencies >= BAND_FLOOR_HZ) & (frequencies <= BAND_CEILING_HZ)
    total = float(power[in_range].sum())
    if not np.isfinite(total) or total <= 0.0:
        return None

    above = float(power[in_range & (frequencies >= cutoff_hz)].sum())
    return above / total


# --------------------------------------------------------------------------- #
# Small shared utilities
# --------------------------------------------------------------------------- #


def _is_silent(audio: npt.NDArray[np.floating[Any]]) -> bool:
    """Whether `audio` is empty or digital silence (no sample above 1e-12)."""
    if audio.size == 0:
        return True
    peak = float(np.max(np.abs(audio)))
    return not np.isfinite(peak) or peak <= _SILENCE_THRESHOLD


def _finite_or_none(value: float) -> float | None:
    """`value` as a float, or `None` when it is NaN or infinite."""
    return float(value) if np.isfinite(value) else None


def _rotate_hpcp_to_c_root(hpcp: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rotate Essentia's A-rooted HPCP bins to `PITCH_CLASSES`' C-rooted order.

    Essentia bin ``i`` (default ``referenceFrequency=440``) holds pitch class
    ``(9 + i) mod 12`` in `PITCH_CLASSES` terms (bin 0 = A = index 9). The
    output's index ``j`` should hold that same pitch class, i.e. the bin
    where ``(9 + i) mod 12 == j`` -> ``i = (j - 9) mod 12``, which is exactly
    ``np.roll(hpcp, 9)``. Verified directly (see module docstring / test
    suite): feeding an isolated 440 Hz sine through `SpectralPeaks` -> `HPCP`
    -> this rotation puts the peak at index 9 (A), and `argmax` of the
    unrotated array is 0.
    """
    return np.roll(hpcp, _HPCP_ROTATE_TO_C)


def _tonal_stability(chroma: npt.NDArray[np.float64]) -> float | None:
    """1 - mean frame-to-frame cosine distance of the chroma sequence.

    Matches `librosa_backend`'s definition exactly (reimplemented, not
    imported, for the same reason as the spectral helpers above). Chroma bin
    order is irrelevant here - cosine similarity is invariant under any fixed
    permutation applied identically to both vectors - so this runs on
    Essentia's raw, unrotated HPCP frames rather than the C-rooted output.

    Returns `None` when fewer than two consecutive non-empty frames exist.
    """
    frames = np.asarray(chroma, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] < 2:
        return None

    norms = np.linalg.norm(frames, axis=0)
    left, right = frames[:, :-1], frames[:, 1:]
    left_norm, right_norm = norms[:-1], norms[1:]
    usable = (left_norm > 0.0) & (right_norm > 0.0)
    if not bool(np.any(usable)):
        return None

    similarity = (left * right).sum(axis=0)[usable] / (left_norm[usable] * right_norm[usable])
    distance = 1.0 - similarity
    stability = 1.0 - float(np.mean(distance))
    if not np.isfinite(stability):
        return None
    return float(np.clip(stability, 0.0, 1.0))


class EssentiaBackend:
    """Feature extraction backed by Essentia's standard algorithm suite."""

    name: BackendName = "essentia"

    def rhythm(self, audio: npt.NDArray[np.float32], sample_rate: int) -> RhythmFeatures:
        """BPM, beat times, onset times/density, transient sharpness.

        `RhythmExtractor2013(method="multifeature")` supplies `bpm`,
        `beat_times` (its `ticks` output) and `bpm_confidence` (its raw
        `confidence` output). `OnsetRate` supplies `onset_times` and
        `onset_density` independently - these are Essentia's *observed* hit
        positions, distinct from the *inferred*, evenly-spaced beat grid, and
        exactly what `onset_times` is specified to hold.

        `bpm_confidence` is Essentia's own scale, **not 0-1**: per the
        `RhythmExtractor2013` docs it is unbounded, and measured across this
        project's fixtures it ranges roughly 2.4-4.33 for real and spurious
        material alike (silence hit 4.22). It is passed through unnormalised;
        rescaling it into 0-1 would invent a mapping this backend has no
        basis for, and would make it look comparable to
        `librosa_backend`'s `bpm_confidence`, which is always `None` (librosa
        has no confidence figure at all). Downstream thresholds must treat
        this as backend-specific.

        Both `RhythmExtractor2013` and `OnsetRate` are hard-wired to 44100 Hz
        internally (neither accepts a `sampleRate` parameter), so if
        `sample_rate` is anything else this method returns
        `RhythmFeatures()` untouched rather than feed them wrong-rate audio.
        `audio_io.load_audio` never actually produces anything but 44100 Hz,
        so this branch should be unreachable in practice.

        On digital silence, returns `onset_density=0.0` immediately without
        calling either algorithm: `RhythmExtractor2013` does not degrade
        gracefully on silence (see `MIN_ONSET_RATE_FOR_RHYTHM_HZ` for the
        measured, spurious 738 BPM / confidence 4.22 it reports there), so
        silence is filtered out before either algorithm sees it. See
        `MIN_ONSET_RATE_FOR_RHYTHM_HZ` for how quiet-but-not-silent material
        (a sustained tone, noise) is handled instead.
        """
        mono = to_mono(np.asarray(audio, dtype=np.float32))
        duration = mono.size / float(sample_rate) if sample_rate > 0 else 0.0
        if duration <= 0.0:
            return RhythmFeatures()
        if _is_silent(mono):
            return RhythmFeatures(onset_density=0.0)
        if sample_rate != 44100:
            return RhythmFeatures()

        import essentia.standard as es

        onset_times: list[float] = []
        onset_density: float | None = None
        try:
            onsets, rate = es.OnsetRate()(mono)
            onset_times = [float(t) for t in np.asarray(onsets, dtype=np.float64)]
            onset_density = float(rate)
        except Exception:
            return RhythmFeatures()

        bpm: float | None = None
        bpm_confidence: float | None = None
        beat_times: list[float] = []
        if onset_density is not None and onset_density >= MIN_ONSET_RATE_FOR_RHYTHM_HZ:
            try:
                extractor = es.RhythmExtractor2013(method="multifeature")
                bpm_raw, ticks, confidence, _estimates, _intervals = extractor(mono)
                bpm = _finite_or_none(float(bpm_raw))
                bpm_confidence = _finite_or_none(float(confidence))
                beat_times = [float(t) for t in np.asarray(ticks, dtype=np.float64)]
            except Exception:
                bpm, bpm_confidence, beat_times = None, None, []

        sharpness = self._transient_sharpness(mono, sample_rate, onset_times)

        return RhythmFeatures(
            bpm=bpm,
            bpm_confidence=bpm_confidence,
            beat_times=beat_times,
            onset_times=onset_times,
            onset_density=onset_density,
            transient_sharpness=sharpness,
        )

    def _transient_sharpness(
        self,
        mono: npt.NDArray[np.float32],
        sample_rate: int,
        onset_times: list[float],
    ) -> float | None:
        """Mean ratio of each onset's local HFC-envelope peak to its median.

        Same specification as `librosa_backend.transient_sharpness` (peak
        value over the local median in a `TRANSIENT_WINDOW_SECONDS` window,
        clamped to `MAX_TRANSIENT_SHARPNESS`, `MAX_TRANSIENT_SHARPNESS` again
        when the local median is zero), reimplemented against Essentia's own
        onset-strength function rather than librosa's, because there is no
        equivalent of librosa's `onset_strength` in Essentia's standard mode:
        the nearest analogue is `OnsetDetection`, which yields one detection-
        function value per frame and must be run framewise by hand.

        The envelope used is `OnsetDetection(method="hfc")` over the same
        `STFT_N_FFT`/`STFT_HOP_LENGTH` frame grid as `spectral()`. `OnsetRate`
        runs its own internal onset-detection grid to produce `onset_times`,
        which does not exactly line up with this envelope's frame grid or
        peaks (see `_PEAK_SEARCH_WINDOW_FRAMES`), so each onset time is
        treated as an approximate location and the true local maximum is
        found within a small search window around it before measuring
        sharpness - rather than trusting the envelope value at the exact
        mapped frame index, which measurably undershoots the real peak.
        """
        if not onset_times:
            return None

        import essentia.standard as es

        try:
            windowing = es.Windowing(type="hann")
            spectrum_algo = es.Spectrum()
            hfc = es.OnsetDetection(method="hfc", sampleRate=sample_rate)
            envelope: list[float] = []
            for frame in es.FrameGenerator(
                mono, frameSize=STFT_N_FFT, hopSize=STFT_HOP_LENGTH, startFromZero=False
            ):
                spec = spectrum_algo(windowing(frame))
                phase = np.zeros(len(spec), dtype=np.float32)
                envelope.append(float(hfc(spec, phase)))
        except Exception:
            return None

        env = np.asarray(envelope, dtype=np.float64)
        if env.size == 0:
            return None

        window_frames = max(
            1, int(round(TRANSIENT_WINDOW_SECONDS * sample_rate / STFT_HOP_LENGTH))
        )

        ratios: list[float] = []
        for time_s in onset_times:
            centre = int(round(time_s * sample_rate / STFT_HOP_LENGTH))
            search_low = max(0, centre - _PEAK_SEARCH_WINDOW_FRAMES)
            search_high = min(env.size, centre + _PEAK_SEARCH_WINDOW_FRAMES + 1)
            if search_low >= search_high:
                continue
            local_window = env[search_low:search_high]
            peak_offset = int(np.argmax(local_window))
            peak_index = search_low + peak_offset
            value = float(env[peak_index])
            if not np.isfinite(value) or value <= 0.0:
                continue

            low = max(0, peak_index - window_frames)
            high = min(env.size, peak_index + window_frames + 1)
            local_median = float(np.median(env[low:high]))
            if not np.isfinite(local_median) or local_median <= 0.0:
                ratios.append(MAX_TRANSIENT_SHARPNESS)
            else:
                ratios.append(min(value / local_median, MAX_TRANSIENT_SHARPNESS))

        if not ratios:
            return None
        return float(np.mean(ratios))

    def tonal(self, audio: npt.NDArray[np.float32], sample_rate: int) -> TonalFeatures:
        """Key, scale, key confidence, 12-bin HPCP mean, tonal stability.

        `KeyExtractor` supplies `key`/`scale`/`key_confidence` (its `strength`
        output, already roughly 0-1 by construction). `HPCP`, fed by
        `SpectralPeaks`, is computed framewise over `CHROMA_FRAME_SIZE`/
        `CHROMA_HOP_SIZE` - deliberately *not* the `STFT_N_FFT`/
        `STFT_HOP_LENGTH` grid `spectral()` uses; see `CHROMA_SPECTRAL_PEAKS_MAX`
        for why that grid measurably breaks chroma stability. Frames are
        mean-pooled for `hpcp_mean` and rotated to C-rooted order (see
        `_rotate_hpcp_to_c_root`) to match `librosa_backend.PITCH_CLASSES`.
        `tonal_stability` is 1 minus the mean frame-to-frame cosine distance
        of those same chroma frames (see `_tonal_stability`), matching
        `librosa_backend`'s definition of the statistic, though not its
        absolute scale - see the module docstring's framing note and
        `CHROMA_SPECTRAL_PEAKS_MAX` for the measured gap.

        All `None` on silence. `KeyExtractor` does not itself gate on
        silence - it returns `("A", "minor", 0.0)` for an all-zero signal
        rather than raising - so silence is filtered out explicitly instead
        of trusting that zero-strength result.

        Structural divergence from `librosa_backend`, worth flagging loudly
        for whoever tunes heuristic thresholds: on white noise, this
        backend's `key_confidence` (0.695) sits almost as high as it does on
        the 440 Hz sine (0.688) - `KeyExtractor`'s default `bgate` profile
        does not clearly separate tonal from atonal material the way
        `librosa_backend`'s Krumhansl-Schmuckler margin does (0.171 sine vs
        0.022 noise, a >7x gap). The same is true of `tonal_stability`:
        0.782 (sine) vs 0.242 (noise) here, versus ~1.0 vs ~0.994 in
        `librosa_backend` (its CQT chroma stays near-flat, and a flat vector
        is trivially "stable" under cosine similarity - see that module's
        own comment on the same effect). Both backends correctly *order*
        sine above noise on both fields; neither field is comparable in
        absolute scale across backends, and a threshold tuned on one
        backend's output will not transfer to the other's.
        """
        mono = to_mono(np.asarray(audio, dtype=np.float32))
        if sample_rate <= 0 or _is_silent(mono):
            return TonalFeatures()

        import essentia.standard as es

        key: str | None = None
        scale: str | None = None
        key_confidence: float | None = None
        try:
            key_raw, scale_raw, strength = es.KeyExtractor(sampleRate=sample_rate)(mono)
            key = str(key_raw) or None
            scale = str(scale_raw) or None
            key_confidence = _finite_or_none(float(strength))
        except Exception:
            key, scale, key_confidence = None, None, None

        hpcp_mean: list[float] = []
        tonal_stability: float | None = None
        try:
            windowing = es.Windowing(type="hann")
            spectrum_algo = es.Spectrum()
            peaks_algo = es.SpectralPeaks(
                sampleRate=sample_rate,
                maxPeaks=CHROMA_SPECTRAL_PEAKS_MAX,
                orderBy="magnitude",
            )
            hpcp_algo = es.HPCP(sampleRate=sample_rate)

            frames: list[npt.NDArray[np.float64]] = []
            for frame in es.FrameGenerator(
                mono, frameSize=CHROMA_FRAME_SIZE, hopSize=CHROMA_HOP_SIZE, startFromZero=False
            ):
                spec = spectrum_algo(windowing(frame))
                freqs, mags = peaks_algo(spec)
                hpcp_vector = hpcp_algo(freqs, mags)
                frames.append(np.asarray(hpcp_vector, dtype=np.float64))

            if frames:
                chroma = np.stack(frames, axis=1)  # (12, n_frames), Essentia's A-rooted order
                chroma_mean = chroma.mean(axis=1)
                if bool(np.all(np.isfinite(chroma_mean))):
                    hpcp_mean = [float(v) for v in _rotate_hpcp_to_c_root(chroma_mean)]
                tonal_stability = _tonal_stability(chroma)
        except Exception:
            hpcp_mean, tonal_stability = [], None

        return TonalFeatures(
            key=key,
            scale=scale,
            key_confidence=key_confidence,
            hpcp_mean=hpcp_mean,
            tonal_stability=tonal_stability,
        )

    def spectral(self, audio: npt.NDArray[np.float32], sample_rate: int) -> SpectralFeatures:
        """Centroid, rolloff, brightness, band-energy ratios over BAND_EDGES_HZ.

        Centroid comes from `SpectralCentroidTime`, which - unlike everything
        else in this module - operates on raw time-domain frames rather than
        a magnitude spectrum (it applies its own first-difference filter
        internally); passing it a windowed frame would double-filter it, so
        it deliberately runs on the unwindowed frames. It is run once per
        frame on the shared `STFT_N_FFT`/`STFT_HOP_LENGTH` grid for
        `centroid_mean`/`centroid_std`, and separately verified against a
        whole-signal call to confirm the two agree (they do, to ~0.001 Hz on
        the sine fixture).

        `RollOff` and the band-energy helpers all consume one shared,
        Hann-windowed magnitude spectrogram (`Windowing` -> `Spectrum` per
        frame), so rolloff, brightness and the band ratios cannot disagree
        about what spectrum they are describing. `band_energy_ratios` and
        `brightness` are this module's own reimplementation of
        `librosa_backend`'s contract (see the module docstring).

        Measured cross-backend agreement (see `tests/test_essentia_backend.py`):
        `band_energy_ratios` and `brightness` agree with `librosa_backend` to
        within 0.0001 on every fixture - both are scale-invariant ratios over
        the same energy definition, so window-normalisation and the 690-vs-
        691-frame count wash out completely. `rolloff_mean` agrees closely
        (within ~5% on sparse/noisy material, under 1% on the sine and white
        noise). `centroid_mean` is the one real structural divergence:
        `SpectralCentroidTime` is a *different algorithm* from librosa's
        magnitude-weighted spectral centroid (a time-domain first-difference
        filter, not a frequency-domain weighted mean - see its own docstring),
        not just a different implementation of the same formula. Measured
        deltas: ~1 Hz on the sine (both agree almost exactly on an
        unambiguous single frequency), but 176 Hz (~18%) on the click train,
        283 Hz (~18%) on the swung click, and 1098 Hz (~10%) on white noise.
        Both orderings and both magnitudes are musically sane; they are just
        not the same statistic on transient-heavy or broadband material, and
        should not be compared across backends past a coarse sanity check.
        """
        mono = to_mono(np.asarray(audio, dtype=np.float32))
        if sample_rate <= 0 or _is_silent(mono):
            return SpectralFeatures()

        import essentia.standard as es

        try:
            windowing = es.Windowing(type="hann")
            spectrum_algo = es.Spectrum()
            centroid_algo = es.SpectralCentroidTime(sampleRate=sample_rate)
            rolloff_algo = es.RollOff(sampleRate=sample_rate, cutoff=ROLLOFF_PERCENT)

            magnitudes: list[npt.NDArray[np.float64]] = []
            centroids: list[float] = []
            rolloffs: list[float] = []
            for frame in es.FrameGenerator(
                mono, frameSize=STFT_N_FFT, hopSize=STFT_HOP_LENGTH, startFromZero=False
            ):
                spec = spectrum_algo(windowing(frame))
                magnitudes.append(np.asarray(spec, dtype=np.float64))
                centroids.append(float(centroid_algo(frame)))
                rolloffs.append(float(rolloff_algo(spec)))
        except Exception:
            return SpectralFeatures()

        if not magnitudes:
            return SpectralFeatures()

        magnitude = np.stack(magnitudes, axis=1)  # (n_bins, n_frames)
        freqs = np.fft.rfftfreq(STFT_N_FFT, d=1.0 / sample_rate)

        centroid_arr = np.asarray(centroids, dtype=np.float64)
        centroid_mean = _finite_or_none(float(np.mean(centroid_arr))) if centroid_arr.size else None
        centroid_std = _finite_or_none(float(np.std(centroid_arr))) if centroid_arr.size else None

        rolloff_arr = np.asarray(rolloffs, dtype=np.float64)
        rolloff_mean = _finite_or_none(float(np.mean(rolloff_arr))) if rolloff_arr.size else None

        ratios = band_energy_ratios(magnitude, freqs)
        return SpectralFeatures(
            centroid_mean=centroid_mean,
            centroid_std=centroid_std,
            rolloff_mean=rolloff_mean,
            brightness=brightness(magnitude, freqs),
            band_energy_ratios=BandEnergyRatios(
                low=ratios["low"],
                low_mid=ratios["low_mid"],
                high_mid=ratios["high_mid"],
                high=ratios["high"],
            ),
        )

    def dynamics(self, audio: npt.NDArray[np.float32], sample_rate: int) -> DynamicsFeatures:
        """Integrated LUFS, RMS mean/std, crest factor.

        `LoudnessEBUR128` **requires stereo** `(n_samples, 2)` input - it
        raises `TypeError` on a 1-D or `(n, 1)` array (verified directly). A
        mono `audio` is promoted via `audio_io.ensure_stereo` before the call;
        `RMS` and crest factor run on the mono-summed signal instead, since
        those are meant to describe the source's own level, not a stereo
        artefact of the promotion.

        RMS mean/std are framewise (`STFT_N_FFT`/`STFT_HOP_LENGTH` grid).
        Crest factor is peak amplitude over the **global** RMS of the whole
        mono signal (not the frame-mean RMS), matching
        `librosa_backend`'s definition - this is what gives the expected
        sqrt(2) for a sine.

        Structural divergence from `librosa_backend`, worth flagging: on pure
        digital silence, `LoudnessEBUR128` returns a **defined** `-70.0`
        LUFS (its own absolute-silence gate floor), not `-inf` or `NaN`.
        `librosa_backend` + pyloudnorm instead produces a non-finite result
        on silence, which that backend maps to `None`. Both are "correct" by
        their own algorithm's contract; this backend reports the real
        Essentia value (`-70.0`) rather than overriding it to `None`, since
        it is a legitimate, finite, documented output - but a caller
        comparing `loudness_lufs` across backends on a silent source will see
        `-70.0` from Essentia and `None` from librosa for the same input.
        """
        raw = np.asarray(audio, dtype=np.float32)
        mono = to_mono(raw)
        if mono.size == 0:
            return DynamicsFeatures()

        import essentia.standard as es

        rms_mean: float | None = None
        rms_std: float | None = None
        try:
            rms_algo = es.RMS()
            values = [
                float(rms_algo(frame))
                for frame in es.FrameGenerator(
                    mono, frameSize=STFT_N_FFT, hopSize=STFT_HOP_LENGTH, startFromZero=False
                )
            ]
            if values:
                arr = np.asarray(values, dtype=np.float64)
                rms_mean = _finite_or_none(float(np.mean(arr)))
                rms_std = _finite_or_none(float(np.std(arr)))
        except Exception:
            rms_mean, rms_std = None, None

        peak = float(np.max(np.abs(mono)))
        global_rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        crest: float | None = None
        if np.isfinite(peak) and np.isfinite(global_rms) and global_rms > 0.0:
            crest = _finite_or_none(peak / global_rms)

        return DynamicsFeatures(
            loudness_lufs=self._integrated_loudness(raw, sample_rate),
            rms_mean=rms_mean,
            rms_std=rms_std,
            crest_factor=crest,
        )

    def _integrated_loudness(
        self, audio: npt.NDArray[np.float32], sample_rate: int
    ) -> float | None:
        """ITU-R BS.1770-4 integrated loudness in LUFS via `LoudnessEBUR128`.

        Promotes to stereo with `audio_io.ensure_stereo` first -
        `LoudnessEBUR128` accepts exactly `(n_samples, 2)` and nothing else,
        the same channel-last layout `audio_io` produces natively (no
        transpose needed). Returns `None` below `_MIN_LOUDNESS_DURATION_SECONDS`
        (mirroring `librosa_backend`'s guard for parity, even though Essentia
        itself does not raise there - see that constant's docstring) or when
        the result is non-finite.
        """
        stereo = ensure_stereo(np.asarray(audio, dtype=np.float32))
        min_samples = int(round(_MIN_LOUDNESS_DURATION_SECONDS * sample_rate))
        if sample_rate <= 0 or stereo.shape[0] < min_samples:
            return None

        import essentia.standard as es

        try:
            _momentary, _short_term, integrated, _range = es.LoudnessEBUR128(
                sampleRate=sample_rate
            )(stereo)
            value = float(integrated)
        except Exception:
            return None
        return _finite_or_none(value)


if TYPE_CHECKING:  # pragma: no cover - static structural check only
    from . import AnalysisBackend

    _protocol_check: AnalysisBackend = EssentiaBackend()
