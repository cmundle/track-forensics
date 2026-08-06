"""librosa analysis backend.

The fallback backend, and in practice the one that actually runs on macOS when
Essentia refuses to build. Treat it as first-class.

Conventions this module holds to:

- `librosa` and `pyloudnorm` are imported **inside** function bodies only, so
  importing this module — and therefore the CLI, and `doctor` — never requires
  them.
- Audio arrives from `audio_io` at exactly `ANALYSIS_SAMPLE_RATE` (44,100 Hz).
  Nothing here resamples, and every librosa call taking an `sr` argument is
  passed the real rate explicitly: librosa's own default of 22,050 Hz would
  halve the spectrum and silently corrupt every centroid, rolloff and high-band
  ratio in the project.
- Stereo audio is channel-last `(n_samples, n_channels)` — soundfile's layout,
  which is *not* librosa's `(n_channels, n_samples)`. Everything is routed
  through `audio_io.to_mono()` / `ensure_stereo()` rather than handed to librosa
  two-dimensional, so the two conventions never meet.
- The energy-weighted descriptors live in module-level helpers
  (:func:`band_energy_ratios`, :func:`brightness`,
  :func:`energy_weighted_centroid_hz`, :func:`energy_percentile_hz`) taking a
  magnitude spectrum and its frequency axis, with no librosa dependency. All
  four share one energy measure — power summed over every frame, inside
  `[20, 20000]` Hz — so they cannot disagree about what they are describing,
  and none of them can be swayed by a silent frame. The Essentia backend
  mirrors them exactly; their docstrings are that mirror's specification.
- Anything that cannot be computed is `None` on the model, never an exception.
  The caller collects those names into `SourceAnalysis.unavailable_features`.
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
    PitchTrack,
    RhythmFeatures,
    SpectralFeatures,
    TonalFeatures,
)
from . import BASS_F0_MAX_HZ, BASS_F0_MIN_HZ, BackendName

__all__ = [
    "BRIGHTNESS_CUTOFF_HZ",
    "KS_MAJOR_PROFILE",
    "KS_MINOR_PROFILE",
    "MAX_TRANSIENT_SHARPNESS",
    "PITCH_CLASSES",
    "PYIN_FRAME_LENGTH",
    "PYIN_MIN_VOICED_PROBABILITY",
    "ROLLOFF_ENERGY_FRACTION",
    "ROLLOFF_PERCENT",
    "STFT_HOP_LENGTH",
    "STFT_N_FFT",
    "LibrosaBackend",
    "band_energy_ratios",
    "brightness",
    "energy_percentile_hz",
    "energy_weighted_centroid_hz",
    "transient_sharpness",
]

#: STFT window length in samples. 2048 at 44.1 kHz is ~46 ms and ~21.5 Hz per
#: bin — fine enough to resolve the 250 Hz band edge, coarse enough to stay
#: cheap over a full track.
STFT_N_FFT = 2048

#: STFT hop in samples: ~11.6 ms at 44.1 kHz. Also the frame rate for the onset
#: envelope, beat tracking and framewise RMS, so every framewise descriptor in
#: this backend shares one time grid.
STFT_HOP_LENGTH = 512

#: Fraction of spectral energy below the reported rolloff frequency.
ROLLOFF_PERCENT = 0.85

#: Brightness split point. Above it lives most of what makes a source read as
#: "bright": hats, cymbals, consonants, pluck attacks.
BRIGHTNESS_CUTOFF_HZ = 1500.0

#: Cumulative-energy fraction behind `SpectralFeatures.rolloff_energy_hz`.
#: Identical to `ROLLOFF_PERCENT` and deliberately not a separate number — it
#: is named separately only so :func:`energy_percentile_hz`'s default states
#: what the field it feeds is defined as. A rolloff *is* a percentile; a
#: centroid is not, which is why `centroid_energy_hz` does not come from this
#: function. See :func:`energy_weighted_centroid_hz`.
ROLLOFF_ENERGY_FRACTION = ROLLOFF_PERCENT

#: Saturation point for :func:`transient_sharpness`. A hit with a digitally
#: silent floor around it has an infinite peak-to-median ratio; reporting the
#: ceiling keeps the descriptor finite and JSON-serialisable.
MAX_TRANSIENT_SHARPNESS = 100.0

#: Half-width of the local-median window used by :func:`transient_sharpness`,
#: in seconds. Roughly one beat either side at moderate tempi.
TRANSIENT_WINDOW_SECONDS = 0.5

#: Minimum peak onset-strength for a signal to be considered to have onsets at
#: all. `librosa.onset.onset_detect` normalises the envelope to max 1.0 before
#: peak-picking, so a signal with *no* transients has its numerical floor
#: rescaled into a full-height envelope and reports a stream of phantom onsets:
#: an 8 s 440 Hz sine comes back at 18.6 onsets/sec without this gate.
#: onset_strength is a dB-domain difference taken against the signal's own
#: loudest bin (`ref=np.max`), so its scale is level-invariant and one fixed
#: threshold works across material. Measured envelope maxima at 44.1 kHz:
#: pure sine 0.06, three-harmonic pad 0.20, pink noise 3.44, white noise 3.12,
#: click train 64.73. 0.5 sits above every flat case and ~6x below real
#: material.
ONSET_ENVELOPE_FLOOR = 0.5

#: Window length for `librosa.pyin`, in samples. **A deliberate, documented
#: departure from the `STFT_N_FFT` (2048) grid every other framewise descriptor
#: in this backend shares** — the one place in the project that breaks it.
#:
#: The reason is numerical, not stylistic. pYIN estimates a period by
#: autocorrelation, so it needs roughly two periods of the lowest frequency it
#: is asked to find inside a single window. `BASS_F0_MIN_HZ` is 30 Hz, and
#: 2 / 30 Hz = 66.7 ms; 2048 samples at 44.1 kHz is only 46.4 ms. A 2048-sample
#: window therefore cannot resolve a low B at all, and YIN's failure mode is not
#: an error or a `None` — it silently locks onto the first period it *can* fit,
#: which is the octave above. 4096 samples is 92.9 ms, comfortably past the
#: 66.7 ms floor, and the next power of two.
#:
#: The hop stays at `STFT_HOP_LENGTH` (512), so the F0 track lands on the same
#: 11.6 ms time grid as the onset envelope and framewise RMS even though the
#: windows are twice as long. `essentia_backend.CHROMA_FRAME_SIZE` is the
#: precedent for this kind of pinned-grid exception: state the number, state
#: why, do not quietly diverge.
PYIN_FRAME_LENGTH = 4096

#: Minimum `voiced_probability` for `pitch()` to call a frame voiced, on top of
#: pYIN's own `voiced_flag`.
#:
#: The extra gate is not belt-and-braces; without it the note segmenter cannot
#: see note boundaries. Measured on `bass_line_a_minor` (16 notes, each 460 ms
#: sounding with a 40 ms silent gap): pYIN's `voiced_flag` alone yields only
#: **11** voiced runs, because its HMM smooths straight through the shorter gaps
#: and welds each pair of consecutive `a1` notes into one — and two notes of the
#: same pitch have nothing *but* the gap to separate them. Adding
#: `voiced_probability >= 0.2` yields exactly **16** runs of 35-39 frames.
#:
#: Measured range of usable thresholds on that fixture: 0.05 through 0.4 all
#: give 16 clean runs; 0.5 collapses four of them to 2 frames. 0.2 is the middle
#: of the working range, and it is also where this backend's voiced runs (35-39
#: frames) line up most closely with the Essentia backend's (34-38), which is
#: what keeps the two note sequences identical. On `bass_unvoiced` it passes
#: 6% of frames, none of which survive `note_track.MIN_NOTE_SECONDS`.
PYIN_MIN_VOICED_PROBABILITY = 0.2

#: Chroma bin order used by librosa: bin 0 is C, bin 9 is A.
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

#: Krumhansl-Kessler probe-tone key profiles (Krumhansl & Kessler 1982, as
#: tabulated in Krumhansl, *Cognitive Foundations of Musical Pitch*, 1990).
#: Index 0 is the tonic, so rotating by `k` semitones puts the tonic on pitch
#: class `k`. librosa ships no key detector, so these drive the template
#: correlation in :func:`_estimate_key`.
KS_MAJOR_PROFILE: tuple[float, ...] = (
    6.35,
    2.23,
    3.48,
    2.33,
    4.38,
    4.09,
    2.52,
    5.19,
    2.39,
    3.66,
    2.29,
    2.88,
)
KS_MINOR_PROFILE: tuple[float, ...] = (
    6.33,
    2.68,
    3.52,
    5.38,
    2.60,
    3.53,
    2.54,
    4.75,
    3.98,
    2.69,
    3.34,
    3.17,
)

#: Lowest and highest frequency any band-relative descriptor considers. Derived
#: from `BAND_EDGES_HZ` rather than hard-coded so the two cannot drift apart.
BAND_FLOOR_HZ = min(low for low, _ in BAND_EDGES_HZ.values())
BAND_CEILING_HZ = max(high for _, high in BAND_EDGES_HZ.values())

#: Peak amplitude at or below which a signal is treated as digital silence.
_SILENCE_THRESHOLD = 1e-12


# --------------------------------------------------------------------------- #
# Shared spectral helpers. The Essentia backend mirrors these exactly, so the
# docstrings below are a specification, not a description.
# --------------------------------------------------------------------------- #


def band_energy_ratios(
    magnitude: npt.NDArray[np.floating[Any]],
    freqs: npt.NDArray[np.floating[Any]],
) -> dict[str, float | None]:
    """Share of in-band spectral energy falling in each `BAND_EDGES_HZ` band.

    The canonical definition for the whole project: both analysis backends must
    produce comparable numbers, so the Essentia backend reimplements exactly
    this contract.

    Args:
        magnitude: **Linear magnitude** spectrum — that is, ``abs(stft)``, not
            power and not dB. Shape ``(n_bins, n_frames)``; a 1-D
            ``(n_bins,)`` single-frame spectrum is accepted and treated as one
            frame. Assumed to come from a one-sided STFT with
            ``n_fft = STFT_N_FFT`` (2048) and ``hop_length = STFT_HOP_LENGTH``
            (512) at 44,100 Hz, Hann window, centred — so ``n_bins`` is 1025.
            Those parameters fix the frequency resolution (~21.5 Hz per bin) and
            therefore how bins land near a band edge; a backend using a
            different transform size will differ slightly at the edges, which is
            why they are pinned here rather than left to the caller.
        freqs: Centre frequency in Hz of each row of `magnitude`, shape
            ``(n_bins,)``, ascending. Equivalent to
            ``librosa.fft_frequencies(sr=44100, n_fft=2048)``.

    Returns:
        One entry per `BAND_EDGES_HZ` key (``low``, ``low_mid``, ``high_mid``,
        ``high``). Values are floats summing to 1.0, or **all four are `None`**
        when the ratios are undefined.

    Definition, precisely:

    * Energy of a bin is ``magnitude ** 2`` (power), summed over **all frames**
      of the spectrogram. There is no per-frame normalisation and no averaging
      of per-frame ratios, so loud frames weigh more than quiet ones — the
      result describes the whole signal rather than its average frame.
    * A bin is assigned to exactly one band by its **centre frequency**, using
      half-open intervals ``[low, high)``. The topmost band is closed at the
      top, ``[6000, 20000]``, so a bin landing exactly on 20 kHz is counted
      rather than dropped. Bins are never split across a boundary and no
      interpolation is done: with ~21.5 Hz bins the resulting edge error is well
      under a percent of any band's width.
    * The denominator is the total energy of the bins inside the union of the
      bands, i.e. ``[BAND_FLOOR_HZ, BAND_CEILING_HZ]`` = ``[20, 20000]`` Hz. DC,
      sub-20 Hz rumble and everything above 20 kHz are excluded from numerator
      *and* denominator, so the four values sum to exactly 1.0 (to floating-point
      tolerance) rather than to "1.0 minus whatever was out of range".
    * Zero-energy case: if the in-band total is zero (digital silence, or energy
      only outside 20 Hz-20 kHz) or any value is non-finite, every band is
      `None` — never 0.0, and never a division by zero. Likewise for an empty
      spectrum, or a `freqs` axis whose length does not match `magnitude`.
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

    Companion to :func:`band_energy_ratios`, deliberately built on the same
    energy measure so brightness is directly comparable with the band ratios: it
    is the share of the same denominator lying above the cutoff. The Essentia
    backend mirrors this contract exactly.

    Args:
        magnitude: **Linear magnitude** spectrum (``abs(stft)``, not power, not
            dB), shape ``(n_bins, n_frames)``; a 1-D ``(n_bins,)`` spectrum is
            treated as a single frame. Same assumed STFT as
            :func:`band_energy_ratios`: ``n_fft`` 2048, ``hop_length`` 512, Hann
            window, centred, at 44,100 Hz.
        freqs: Centre frequency in Hz of each row, shape ``(n_bins,)``,
            ascending.
        cutoff_hz: Split point, default `BRIGHTNESS_CUTOFF_HZ` (1500.0).

    Returns:
        A float in ``[0.0, 1.0]``, or `None` when undefined.

    Definition, precisely:

    * Energy of a bin is ``magnitude ** 2`` summed over all frames, exactly as
      in :func:`band_energy_ratios`.
    * Numerator: bins whose centre frequency is ``>= cutoff_hz`` **and** inside
      ``[BAND_FLOOR_HZ, BAND_CEILING_HZ]`` = ``[20, 20000]`` Hz.
    * Denominator: all bins inside ``[20, 20000]`` Hz — the same window the band
      ratios use, so the two cannot disagree about what "total energy" means.
      DC and sub-20 Hz bins are excluded, as is everything above 20 kHz.
    * Bins are assigned by centre frequency, with no interpolation across the
      cutoff.
    * Zero-energy case: if the in-band total is zero (digital silence) or any
      value is non-finite, the result is `None`, never 0.0. Likewise for an
      empty spectrum or a mismatched `freqs`.
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


def _aggregate_band_power(
    magnitude: npt.NDArray[np.floating[Any]],
    freqs: npt.NDArray[np.floating[Any]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """`(power, freqs)` for the in-band bins of one whole-signal spectrum.

    The shared front half of :func:`energy_weighted_centroid_hz` and
    :func:`energy_percentile_hz`, and the reason neither can disagree with
    :func:`band_energy_ratios` or :func:`brightness` about what it is measuring:
    energy is ``magnitude ** 2`` summed over **all frames**, restricted to
    ``[BAND_FLOOR_HZ, BAND_CEILING_HZ]`` = ``[20, 20000]`` Hz.

    Returns `None` on empty, mismatched, non-finite or zero-energy input, which
    is what makes every caller's zero-energy case a `None` rather than a 0.0 or
    a division by zero.
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
    band_power = power[in_range]
    band_freqs = frequencies[in_range]
    if band_freqs.size == 0:
        return None

    total = float(band_power.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return band_power, band_freqs


def energy_weighted_centroid_hz(
    magnitude: npt.NDArray[np.floating[Any]],
    freqs: npt.NDArray[np.floating[Any]],
) -> float | None:
    """Energy-weighted spectral centroid: ``sum(f * P) / sum(P)``, in Hz.

    `SpectralFeatures.centroid_energy_hz`. Fourth member of the same family as
    :func:`band_energy_ratios`, :func:`brightness` and
    :func:`energy_percentile_hz`, built on the identical energy measure so all
    four describe the same thing; the Essentia backend mirrors this contract
    exactly, and the docstring here is that mirror's specification.

    **A first moment, not a percentile — the distinction is load-bearing.** An
    energy *median* was implemented first and rejected on measurement: it
    returns the centre frequency of one FFT bin, so a 55 Hz sine, a 55 Hz square
    and a 55 Hz sawtooth all read **64.6 Hz**, the same number. The only
    consumer of this descriptor, `strudel_vocab.suggest_bass_sound`, exists to
    choose between exactly those waveforms, so a median carries no information
    for the one decision it feeds. The centroid separates them: 54.9, 159.2 and
    215.8 Hz respectively, measured on band-limited additive synthesis gated to
    50% rests over a -82 dBFS floor. Same collapse on real material — the median
    reads 107.7 Hz for both the v4 calibration bass stem and its drums stem,
    where the centroid reads 139.7 and 412.2.

    **Why it is not `centroid_mean`.** `centroid_mean` is an unweighted mean
    over *per-frame* centroids, so a frame carrying no signal counts exactly as
    much as a frame carrying the hook. A stem's noise floor is broadband, which
    puts a resting frame's own centroid up in the kilohertz, and the mean
    follows it. Measured on a 55 Hz sine, continuous versus gated to 50% rests
    over that same -82 dBFS floor and nothing else changed: `centroid_mean`
    84.2 Hz -> **5141.3 Hz, +6007%**; this function 54.9 Hz -> 54.9 Hz,
    **+0.00%**. Silence carries no energy, so it gets no weight. The v4
    calibration bass stem is the real-material case: `centroid_mean` 1010.7 Hz
    with `centroid_std` 1573.3 Hz on a sub-bass with nothing above 4 kHz,
    against 139.7 Hz here.

    Args:
        magnitude: **Linear magnitude** spectrum (``abs(stft)``, not power, not
            dB), shape ``(n_bins, n_frames)``; a 1-D ``(n_bins,)`` spectrum is
            treated as a single frame. Same assumed STFT as
            :func:`band_energy_ratios`: ``n_fft`` 2048, ``hop_length`` 512, Hann
            window, centred, at 44,100 Hz.
        freqs: Centre frequency in Hz of each row, shape ``(n_bins,)``,
            ascending.

    Returns:
        A frequency in Hz, or `None` when undefined.

    Definition, precisely:

    * Energy of a bin is ``magnitude ** 2`` summed over **all frames** — one
      aggregate spectrum for the whole signal, no per-frame normalisation and no
      averaging of per-frame centroids. That single choice is the whole fix.
    * Only bins inside ``[BAND_FLOOR_HZ, BAND_CEILING_HZ]`` = ``[20, 20000]`` Hz
      count, in numerator and denominator both — the same window
      :func:`brightness` and :func:`band_energy_ratios` use.
    * Bins are weighted by their centre frequency with no interpolation, and the
      result is continuous in the input rather than quantised to a bin: a 55 Hz
      tone reads 54.9 Hz, not the 64.6 Hz a bin-quantised percentile gives it.
    * **Scale invariance.** Numerator and denominator scale together, so window
      normalisation and overall gain cancel exactly. This is why the two
      backends agree on it far more closely than they do on `centroid_mean`,
      where they run genuinely different algorithms.
    * **It is a mean, so it has a mean's weakness**: one distant, quiet
      component pulls it. A source with 99% of its energy at 50 Hz and 1% at
      10 kHz reads ~150 Hz, not ~50. That is arithmetically correct and can
      still mislead, which is one reason `brightness` is read alongside it in
      `strudel_vocab` rather than the centroid deciding alone — and why
      `centroid_mean` is kept in the schema rather than deleted.
    * Zero-energy case: if the in-band total is zero (digital silence) or any
      value is non-finite, the result is `None`, never 0.0. Likewise for an
      empty spectrum or a mismatched `freqs`.
    """
    aggregated = _aggregate_band_power(magnitude, freqs)
    if aggregated is None:
        return None
    band_power, band_freqs = aggregated
    return float((band_freqs * band_power).sum() / band_power.sum())


def energy_percentile_hz(
    magnitude: npt.NDArray[np.floating[Any]],
    freqs: npt.NDArray[np.floating[Any]],
    fraction: float = ROLLOFF_ENERGY_FRACTION,
) -> float | None:
    """Frequency below which `fraction` of the in-band spectral energy sits.

    At the default `fraction` (`ROLLOFF_ENERGY_FRACTION`, 0.85) this is
    `SpectralFeatures.rolloff_energy_hz`. Unlike a centroid, a rolloff genuinely
    *is* defined as a percentile, so this is the right shape for it — see
    :func:`energy_weighted_centroid_hz` for why the same function must not be
    used at 0.5 to produce a centroid.

    Same energy measure and same ``[20, 20000]`` Hz window as
    :func:`band_energy_ratios`, :func:`brightness` and
    :func:`energy_weighted_centroid_hz`: power summed over **all frames**, so a
    silent frame contributes nothing. `rolloff_mean`, the per-frame unweighted
    mean this supersedes, is contaminated exactly as `centroid_mean` is —
    measured on a 55 Hz sine gated to 50% rests over a -82 dBFS floor, 64.8 Hz
    continuous against **8693.9 Hz gated, +13319%**, against 64.6 Hz here either
    way. On the v4 calibration bass stem: `rolloff_mean` 4333 Hz (librosa) and
    1097 Hz (essentia), against 215.3 Hz here on both.

    Args:
        magnitude: **Linear magnitude** spectrum, shape ``(n_bins, n_frames)``;
            a 1-D ``(n_bins,)`` spectrum is treated as a single frame. Same
            assumed STFT as :func:`band_energy_ratios`.
        freqs: Centre frequency in Hz of each row, ascending.
        fraction: Cumulative-energy fraction in ``(0, 1]``.

    Returns:
        A frequency in Hz, or `None` when undefined.

    Definition, precisely:

    * The result is the centre frequency of the **first bin at which the
      cumulative in-band energy reaches `fraction`**. No interpolation across
      the crossing bin: the CDF of a tonal signal is a step, not a ramp, so
      interpolating assumes a distribution the signal does not have. Measured on
      pure tones, bin-exact overshoots (55 Hz reads 64.6) and linear
      interpolation undershoots by a similar margin (44.8), so interpolation
      buys no accuracy and costs exact cross-backend agreement.
    * Resolution is therefore one FFT bin, ~21.5 Hz. Acceptable for a rolloff,
      which is a coarse "where does the top of this source sit" figure; it was
      **not** acceptable for a centroid, which is the other reason
      `centroid_energy_hz` does not come from this function.
    * **Scale invariance**, and a step-function result on a shared bin grid,
      together mean the two backends agree on this to the bit: 0.000000 Hz on
      every fixture and on all twelve v4 calibration stems.
    * **Known instability, not a bug.** A percentile is ill-conditioned on a
      strongly bimodal spectrum: a source with half its energy at 55 Hz and half
      at 8 kHz can be tipped either way at `fraction=0.5` by a rounding
      difference. At 0.85 this is far less exposed, but it is why the centroid
      is a first moment and this is reserved for the rolloff.
    * Zero-energy case: `None`, never 0.0. Likewise for an empty spectrum, a
      mismatched `freqs`, or a `fraction` outside ``(0, 1]``.
    """
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        return None
    aggregated = _aggregate_band_power(magnitude, freqs)
    if aggregated is None:
        return None
    band_power, band_freqs = aggregated

    cumulative = np.cumsum(band_power) / band_power.sum()
    index = min(int(np.searchsorted(cumulative, fraction, side="left")), band_freqs.size - 1)
    return float(band_freqs[index])


def transient_sharpness(
    onset_envelope: npt.NDArray[np.floating[Any]],
    peak_frames: npt.NDArray[np.integer[Any]],
    window_frames: int,
    ceiling: float = MAX_TRANSIENT_SHARPNESS,
) -> float | None:
    """Mean ratio of each onset-strength peak to its local median.

    A made-up descriptor, so this definition *is* the specification — nothing
    downstream can reinterpret the number without it.

    For every detected onset peak `p`:

    1. Take the onset-envelope value at `p` (the numerator).
    2. Take the **median** of the envelope over the symmetric window
       ``[p - window_frames, p + window_frames]``, clipped to the array bounds
       (the denominator). Median rather than mean, so the window's own peak and
       any neighbouring hits inside it barely move the floor.
    3. The peak's sharpness is ``value / median``, clamped to `ceiling`. When the
       local median is zero — an isolated hit surrounded by digital silence,
       which is an infinitely sharp transient by this measure — the peak scores
       exactly `ceiling`. Peaks with a non-positive envelope value are skipped.

    The result is the arithmetic mean over all contributing peaks.

    Interpretation: ~1.0 means the envelope is flat around its own "peaks", i.e.
    no real transients (a pad, a sustained tone, noise); large values mean
    isolated, sharply attacked hits. The practical range is ``[1, ceiling]``,
    saturating at `ceiling` (100.0) rather than running off to infinity.

    Args:
        onset_envelope: 1-D onset strength over time, one value per frame.
        peak_frames: Frame indices of detected onsets. Out-of-range indices are
            ignored.
        window_frames: Half-width of the local-median window, in frames; must be
            >= 1.
        ceiling: Saturation value, default `MAX_TRANSIENT_SHARPNESS`.

    Returns:
        Mean sharpness, or `None` when there are no usable peaks, the envelope
        is empty or not 1-D, or `window_frames` < 1.
    """
    envelope = np.asarray(onset_envelope, dtype=np.float64)
    peaks = np.asarray(peak_frames, dtype=int).ravel()
    if envelope.ndim != 1 or envelope.size == 0 or peaks.size == 0 or window_frames < 1:
        return None

    ratios: list[float] = []
    for peak in peaks:
        index = int(peak)
        if not 0 <= index < envelope.size:
            continue
        value = float(envelope[index])
        if not np.isfinite(value) or value <= 0.0:
            continue
        low = max(0, index - window_frames)
        high = min(envelope.size, index + window_frames + 1)
        local_median = float(np.median(envelope[low:high]))
        if not np.isfinite(local_median) or local_median <= 0.0:
            ratios.append(ceiling)
        else:
            ratios.append(min(value / local_median, ceiling))

    if not ratios:
        return None
    return float(np.mean(ratios))


# --------------------------------------------------------------------------- #
# Tonal helpers
# --------------------------------------------------------------------------- #


def _estimate_key(
    chroma_mean: npt.NDArray[np.floating[Any]],
) -> tuple[str | None, str | None, float | None]:
    """Krumhansl-Schmuckler key estimate from a 12-bin mean chroma.

    Correlates `chroma_mean` (Pearson) against `KS_MAJOR_PROFILE` and
    `KS_MINOR_PROFILE` at all 12 rotations — 24 candidates — and takes the
    argmax as key and scale.

    `key_confidence` is the winning margin, normalised into 0-1 as::

        confidence = clip(r_best, 0, 1) * (r_best - r_rival) / (r_best - r_worst)

    where `r_rival` is the best correlation among candidates on a **different
    tonic** from the winner.

    The second factor is that margin as a fraction of the full spread across the
    24 candidates, so it does not depend on the overall scale of the
    correlations. The first factor is goodness of fit: an atonal or near-flat
    chroma correlates with nothing, so `r_best` sits near zero and the
    confidence collapses even when one candidate happens to win by a wide
    relative margin. Both factors are needed — either alone is easy to fool. A
    drums stem scores ~0.03 on the margin term alone, the same as a C major
    triad; only the `r_best` factor separates them.

    **Why the rival is the best *other tonic* rather than simply the
    second-best candidate.** A key's major and minor profiles share a tonic
    weight (6.35 vs 6.33), so for material centred on one pitch the runner-up is
    almost always the parallel key — same tonic, other mode — and the raw
    best-to-second margin collapses to nothing. Measured on the project
    fixtures, a pure 440 Hz sine (the most unambiguously tonal signal there is)
    scored 0.00007 that way, *below* white noise at 0.022, which would have made
    the descriptor worse than useless for the heuristics that threshold on it.
    Measuring against the best rival tonic scores the same sine at 0.171 and
    leaves every other fixture unchanged, because for real polyphonic material
    the runner-up already sits on a different tonic. The number therefore means
    "confidence in the tonal centre", which is what `StrudelHints.tonal_centre`
    actually consumes; mode confidence is deliberately not folded in.

    Observed range on the fixtures: white noise 0.022, click train 0.029, pink
    noise 0.033, C major triad 0.075, sustained sine 0.171, A minor triad 0.204.
    Treat anything below ~0.05 as "no usable key"; W3C calibrates on real
    material.

    Returns:
        `(key, scale, confidence)`. All `None` when the chroma is not 12 bins,
        has zero variance, or produces non-finite correlations.
    """
    chroma = np.asarray(chroma_mean, dtype=np.float64).ravel()
    if chroma.size != 12 or not bool(np.all(np.isfinite(chroma))):
        return None, None, None
    if float(np.std(chroma)) <= 0.0:
        # A perfectly flat chroma correlates with nothing: Pearson is undefined.
        return None, None, None

    candidates: list[tuple[float, int, str]] = []
    for scale, profile in (("major", KS_MAJOR_PROFILE), ("minor", KS_MINOR_PROFILE)):
        template = np.asarray(profile, dtype=np.float64)
        for tonic in range(12):
            rotated = np.roll(template, tonic)
            correlation = float(np.corrcoef(chroma, rotated)[0, 1])
            if not np.isfinite(correlation):
                return None, None, None
            candidates.append((correlation, tonic, scale))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_correlation, best_tonic, scale = candidates[0]
    key = PITCH_CLASSES[best_tonic]
    worst_correlation = candidates[-1][0]

    rivals = [value for value, tonic, _ in candidates if tonic != best_tonic]
    if not rivals:
        return key, scale, None
    rival_correlation = max(rivals)

    spread = best_correlation - worst_correlation
    if spread <= 0.0:
        return key, scale, None
    margin = (best_correlation - rival_correlation) / spread
    confidence = float(np.clip(best_correlation, 0.0, 1.0) * np.clip(margin, 0.0, 1.0))
    return key, scale, confidence


def _tonal_stability(chroma: npt.NDArray[np.floating[Any]]) -> float | None:
    """1 - mean frame-to-frame cosine distance of the chroma sequence.

    Chroma is non-negative, so the cosine similarity of consecutive frames lies
    in ``[0, 1]`` and so does the result: 1.0 means the harmonic content never
    changes, low values mean it churns frame to frame (noise, dense
    modulation). Frames with zero energy have no direction, so each pair
    involving one is skipped.

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


def _integrated_loudness(audio: npt.NDArray[np.float32], sample_rate: int) -> float | None:
    """ITU-R BS.1770-4 integrated loudness in LUFS, via pyloudnorm.

    Takes the raw (possibly stereo) array and routes it through
    `audio_io.ensure_stereo`, so the input is always channel-last ``(n, 2)`` —
    pyloudnorm's expected layout, and the one Essentia's `LoudnessEBUR128`
    requires. A mono source is duplicated into both channels, which reads about
    3 LU louder than metering the single channel would; that is deliberate, so
    the two backends agree, and it matches how mono material is actually
    auditioned.

    Returns `None` when pyloudnorm is missing, when the signal is shorter than
    the meter's 400 ms block (pyloudnorm raises there), or when the result is
    not finite — digital silence gates to -inf, which is not JSON-serialisable.
    """
    try:
        import pyloudnorm
    except ImportError:  # pragma: no cover - pyloudnorm ships with the extra
        return None

    stereo = ensure_stereo(np.asarray(audio, dtype=np.float32))
    if sample_rate <= 0 or stereo.shape[0] < int(round(0.4 * sample_rate)):
        return None

    try:
        meter = pyloudnorm.Meter(sample_rate)
        value = float(meter.integrated_loudness(stereo.astype(np.float64)))
    except Exception:
        # pyloudnorm raises ValueError on too-short input and can propagate
        # scipy filter errors; neither is worth losing the other descriptors.
        return None
    return _finite_or_none(value)


class LibrosaBackend:
    """Feature extraction backed by librosa (+ pyloudnorm for LUFS)."""

    name: BackendName = "librosa"

    def rhythm(self, audio: npt.NDArray[np.float32], sample_rate: int) -> RhythmFeatures:
        """BPM, beat times, onset times, onset density, transient sharpness.

        `beat_times` is the inferred, evenly spaced pulse from
        `librosa.beat.beat_track`; `onset_times` is where hits were actually
        observed by `librosa.onset.onset_detect`. They are different things —
        swing lives in the second one — and both come off the same onset
        envelope on the `STFT_HOP_LENGTH` grid.

        `bpm_confidence` is always `None`: librosa's `beat_track` reports no
        confidence figure, and inventing one from beat-interval regularity would
        measure something different from Essentia's `RhythmExtractor2013`
        confidence, quietly breaking cross-backend comparison.

        On digital silence, and on material whose onset envelope never clears
        `ONSET_ENVELOPE_FLOOR` (a sustained tone or pad, which has no transients
        to track a tempo against), `onset_density` is `0.0` — zero onsets over a
        known duration is a fact, not a missing value — while BPM, beat times
        and transient sharpness are `None`. Reporting a tempo for a signal with
        no detectable onsets would be a confident guess, and a wrong grid is
        worse downstream than no grid.
        """
        import librosa

        mono = to_mono(np.asarray(audio, dtype=np.float32))
        duration = mono.size / float(sample_rate) if sample_rate > 0 else 0.0
        if duration <= 0.0:
            return RhythmFeatures()
        if _is_silent(mono):
            return RhythmFeatures(onset_density=0.0)

        try:
            onset_envelope = np.asarray(
                librosa.onset.onset_strength(y=mono, sr=sample_rate, hop_length=STFT_HOP_LENGTH),
                dtype=np.float64,
            )
        except Exception:
            return RhythmFeatures()

        if onset_envelope.size == 0 or float(np.max(onset_envelope)) < ONSET_ENVELOPE_FLOOR:
            return RhythmFeatures(onset_density=0.0)

        bpm: float | None = None
        beat_times: list[float] = []
        try:
            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_envelope,
                sr=sample_rate,
                hop_length=STFT_HOP_LENGTH,
                units="frames",
            )
            # librosa 0.11 returns tempo as a 1-element array, older ones a float.
            tempo_value = float(np.atleast_1d(np.asarray(tempo, dtype=np.float64)).ravel()[0])
            bpm = tempo_value if np.isfinite(tempo_value) and tempo_value > 0.0 else None
            beat_times = [
                float(time)
                for time in librosa.frames_to_time(
                    np.asarray(beat_frames), sr=sample_rate, hop_length=STFT_HOP_LENGTH
                )
            ]
        except Exception:
            bpm, beat_times = None, []

        onset_times: list[float] = []
        onset_density: float | None = None
        sharpness: float | None = None
        try:
            onset_frames = np.asarray(
                librosa.onset.onset_detect(
                    onset_envelope=onset_envelope,
                    sr=sample_rate,
                    hop_length=STFT_HOP_LENGTH,
                    units="frames",
                ),
                dtype=int,
            ).ravel()
            onset_times = [
                float(time)
                for time in librosa.frames_to_time(
                    onset_frames, sr=sample_rate, hop_length=STFT_HOP_LENGTH
                )
            ]
            onset_density = float(onset_frames.size) / duration
            window_frames = max(
                1, int(round(TRANSIENT_WINDOW_SECONDS * sample_rate / STFT_HOP_LENGTH))
            )
            sharpness = transient_sharpness(onset_envelope, onset_frames, window_frames)
        except Exception:
            onset_times, onset_density, sharpness = [], None, None

        return RhythmFeatures(
            bpm=bpm,
            bpm_confidence=None,
            beat_times=beat_times,
            onset_times=onset_times,
            onset_density=onset_density,
            transient_sharpness=sharpness,
        )

    def tonal(self, audio: npt.NDArray[np.float32], sample_rate: int) -> TonalFeatures:
        """Key, scale, key confidence, 12-bin chroma mean, tonal stability.

        Chroma comes from `librosa.feature.chroma_cqt` at the full 44.1 kHz
        rate. Key and confidence come from Krumhansl-Schmuckler template
        correlation (see :func:`_estimate_key`), because librosa has no key
        detector. Everything is `None` on silence.

        `tuning=0.0` pins the chroma to concert pitch instead of letting librosa
        estimate a per-call tuning offset. Two reasons: the estimate is made per
        source, so mix, bass and vocals could each land on a different reference
        and stop being comparable; and on unpitched material (a drums stem) the
        estimator has no peaks to work from, warns, and returns 0.0 anyway. The
        cost is that a track recorded meaningfully off A440 will smear across
        adjacent chroma bins.
        """
        import librosa

        mono = to_mono(np.asarray(audio, dtype=np.float32))
        if _is_silent(mono) or sample_rate <= 0:
            return TonalFeatures()

        try:
            chroma = np.asarray(
                librosa.feature.chroma_cqt(
                    y=mono, sr=sample_rate, hop_length=STFT_HOP_LENGTH, tuning=0.0
                ),
                dtype=np.float64,
            )
        except Exception:
            return TonalFeatures()

        if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.size == 0:
            return TonalFeatures()

        chroma_mean = chroma.mean(axis=1)
        if not bool(np.all(np.isfinite(chroma_mean))):
            return TonalFeatures()

        key, scale, confidence = _estimate_key(chroma_mean)
        return TonalFeatures(
            key=key,
            scale=scale,
            key_confidence=confidence,
            hpcp_mean=[float(value) for value in chroma_mean],
            tonal_stability=_tonal_stability(chroma),
        )

    def spectral(self, audio: npt.NDArray[np.float32], sample_rate: int) -> SpectralFeatures:
        """Centroid, rolloff, brightness, band-energy ratios over BAND_EDGES_HZ.

        One magnitude STFT (`STFT_N_FFT` / `STFT_HOP_LENGTH`) feeds all of them:
        librosa computes the per-frame centroid and rolloff from it, and the
        shared helpers :func:`brightness`, :func:`band_energy_ratios`,
        :func:`energy_weighted_centroid_hz` and :func:`energy_percentile_hz`
        take the same spectrum, so no two descriptors here can disagree about
        what they are describing.

        **Two families of descriptor live in this one method, and they do not
        mean the same thing.** `centroid_mean`, `centroid_std` and
        `rolloff_mean` are unweighted means over *per-frame* values: every
        frame votes once, whether or not it carries any signal. `brightness`,
        `band_energy_ratios`, `centroid_energy_hz` and `rolloff_energy_hz` are
        computed from the energy of the whole spectrogram summed over all
        frames: a frame votes in proportion to the energy it carries, so
        silence does not vote.

        Measured on a 55 Hz sine gated to 50% rests over a -82 dBFS noise floor,
        against the same signal ungated — the only change is the rests:

        ============================  ============  ============  ==========
        descriptor                    continuous    50% rests     change
        ============================  ============  ============  ==========
        `centroid_mean`               84.2 Hz       5141.3 Hz     **+6007%**
        `rolloff_mean`                64.8 Hz       8693.9 Hz     **+13319%**
        `centroid_energy_hz`          54.9 Hz       54.9 Hz       +0.00%
        `rolloff_energy_hz`           64.6 Hz       64.6 Hz       +0.00%
        `brightness`                  ~0            ~0            —
        ============================  ============  ============  ==========

        And on a signal with real content in both registers (200 Hz + 4 kHz
        tones, same gating): `centroid_mean` +345%, `rolloff_mean` +169%, the
        two energy-weighted fields **+0.00%**, `brightness` **+0.00%** (0.13793
        either way, unchanged to five decimal places).

        So `brightness` needs no corrected companion — it was already summed
        over energy, and it is measurably immune. `centroid_mean` has
        `centroid_energy_hz` beside it and `rolloff_mean` has
        `rolloff_energy_hz`.

        None of the three per-frame fields is being fixed in place: v4 outputs
        must stay numerically comparable, so `centroid_mean` keeps its exact
        previous value and behaviour, including librosa's habit of reporting
        0 Hz for an energy-free frame (which pulls the mean *down* on sparse
        material such as a click train, the opposite contamination).
        """
        import librosa

        mono = to_mono(np.asarray(audio, dtype=np.float32))
        if _is_silent(mono) or sample_rate <= 0:
            return SpectralFeatures()

        try:
            magnitude = np.abs(librosa.stft(mono, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH))
            freqs = np.asarray(
                librosa.fft_frequencies(sr=sample_rate, n_fft=STFT_N_FFT), dtype=np.float64
            )
        except Exception:
            return SpectralFeatures()

        centroid_mean: float | None = None
        centroid_std: float | None = None
        try:
            centroid = np.asarray(
                librosa.feature.spectral_centroid(
                    S=magnitude, sr=sample_rate, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH
                ),
                dtype=np.float64,
            )
            if centroid.size:
                centroid_mean = _finite_or_none(float(np.mean(centroid)))
                centroid_std = _finite_or_none(float(np.std(centroid)))
        except Exception:
            centroid_mean, centroid_std = None, None

        rolloff_mean: float | None = None
        try:
            rolloff = np.asarray(
                librosa.feature.spectral_rolloff(
                    S=magnitude,
                    sr=sample_rate,
                    n_fft=STFT_N_FFT,
                    hop_length=STFT_HOP_LENGTH,
                    roll_percent=ROLLOFF_PERCENT,
                ),
                dtype=np.float64,
            )
            if rolloff.size:
                rolloff_mean = _finite_or_none(float(np.mean(rolloff)))
        except Exception:
            rolloff_mean = None

        ratios = band_energy_ratios(magnitude, freqs)
        return SpectralFeatures(
            centroid_mean=centroid_mean,
            centroid_std=centroid_std,
            centroid_energy_hz=energy_weighted_centroid_hz(magnitude, freqs),
            rolloff_mean=rolloff_mean,
            rolloff_energy_hz=energy_percentile_hz(magnitude, freqs),
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

        Accepts stereo `(n_samples, n_channels)` input; integrated loudness is
        only meaningful on the channel layout the file actually has, so the raw
        array goes to :func:`_integrated_loudness` while RMS and crest factor are
        measured on the mono sum.

        RMS mean/std are over framewise RMS (`STFT_N_FFT` frame,
        `STFT_HOP_LENGTH` hop). Crest factor is peak amplitude over the
        **global** RMS of the whole mono signal rather than over the frame mean:
        that is the textbook definition, and it gives the expected sqrt(2) for a
        sine. It is `None` on silence rather than a division by zero, while RMS
        itself stays a defined 0.0.
        """
        import librosa

        raw = np.asarray(audio, dtype=np.float32)
        mono = to_mono(raw)
        if mono.size == 0:
            return DynamicsFeatures()

        rms_mean: float | None = None
        rms_std: float | None = None
        try:
            rms = np.asarray(
                librosa.feature.rms(y=mono, frame_length=STFT_N_FFT, hop_length=STFT_HOP_LENGTH),
                dtype=np.float64,
            )
            if rms.size:
                rms_mean = _finite_or_none(float(np.mean(rms)))
                rms_std = _finite_or_none(float(np.std(rms)))
        except Exception:
            rms_mean, rms_std = None, None

        peak = float(np.max(np.abs(mono)))
        global_rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        crest: float | None = None
        if np.isfinite(peak) and np.isfinite(global_rms) and global_rms > 0.0:
            crest = _finite_or_none(peak / global_rms)

        return DynamicsFeatures(
            loudness_lufs=_integrated_loudness(raw, sample_rate),
            rms_mean=rms_mean,
            rms_std=rms_std,
            crest_factor=crest,
        )

    def pitch(self, audio: npt.NDArray[np.float32], sample_rate: int) -> PitchTrack:
        """Framewise F0 over `BASS_F0_MIN_HZ`-`BASS_F0_MAX_HZ`, via `librosa.pyin`.

        Raw physics only: one F0 estimate and one voicing decision per frame.
        Notes are `note_track.segment_notes()`'s job, shared between backends,
        so nothing musical is decided here.

        pYIN (Mauch & Dixon 2014) is YIN with a probabilistic threshold and an
        HMM over the candidates, which is exactly the machinery that would be
        foolish to reimplement — hence a Protocol method rather than a numpy
        module. `frame_length` is `PYIN_FRAME_LENGTH` (4096, not the project's
        2048; see that constant), `hop_length` is `STFT_HOP_LENGTH` so the track
        lands on the shared time grid, and `fill_na=0.0` writes 0.0 rather than
        NaN into unvoiced frames because `PitchTrack.f0_hz` is a JSON-shaped
        `list[float]` even though it never reaches JSON.

        A frame counts as voiced only when pYIN's own `voiced_flag` is set, the
        probability clears `PYIN_MIN_VOICED_PROBABILITY`, and the F0 lies inside
        the search range. That last check is nominally redundant — `pyin` is
        given the same `fmin`/`fmax` — and is kept because `note_track` treats
        `voiced` as authoritative and must never be handed a 0.0.

        Measured on the project's bass fixtures at 44.1 kHz: **100% of voiced
        frames land on the correct semitone on both `bass_line_a_minor` and
        `bass_line_octave_trap`, with a 0.0% octave-error rate** — the range
        constraint alone defeats the trap, since the buried fundamental's
        octave-up reading is still inside the window but pYIN's period search
        prefers the true period. Cost: ~2.1-3.5 s for 8.5 s of audio, i.e.
        roughly a third of real time and by far the most expensive thing this
        backend does. Digital silence, a non-positive sample rate and any
        internal failure all return an empty `PitchTrack()`.
        """
        import librosa

        mono = to_mono(np.asarray(audio, dtype=np.float32))
        if sample_rate <= 0 or _is_silent(mono):
            return PitchTrack()

        try:
            f0, voiced_flag, voiced_probability = librosa.pyin(
                mono,
                fmin=BASS_F0_MIN_HZ,
                fmax=BASS_F0_MAX_HZ,
                sr=sample_rate,
                frame_length=PYIN_FRAME_LENGTH,
                hop_length=STFT_HOP_LENGTH,
                fill_na=0.0,
            )
        except Exception:
            return PitchTrack()

        frequencies = np.nan_to_num(np.asarray(f0, dtype=np.float64), nan=0.0)
        flags = np.asarray(voiced_flag, dtype=bool)
        probabilities = np.nan_to_num(np.asarray(voiced_probability, dtype=np.float64), nan=0.0)
        if frequencies.ndim != 1 or flags.shape != frequencies.shape:
            return PitchTrack()

        voiced = (
            flags
            & (probabilities >= PYIN_MIN_VOICED_PROBABILITY)
            & (frequencies >= BASS_F0_MIN_HZ)
            & (frequencies <= BASS_F0_MAX_HZ)
        )

        return PitchTrack(
            f0_hz=[float(value) for value in frequencies],
            voiced=[bool(value) for value in voiced],
            voiced_probability=[float(value) for value in probabilities],
            frame_hop_seconds=STFT_HOP_LENGTH / float(sample_rate),
            method="pyin",
        )


if TYPE_CHECKING:  # pragma: no cover - static structural check only
    from . import AnalysisBackend

    _protocol_check: AnalysisBackend = LibrosaBackend()
