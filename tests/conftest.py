"""Synthetic audio fixtures with known ground truth.

No test in this repo may need a real audio file. Every signal here is generated
from numpy with a fixed seed, so backend tests can assert real numbers rather
than "it did not crash".

Everything is `np.float32` at `ANALYSIS_SAMPLE_RATE` (44.1 kHz), mono
`(n_samples,)` unless the fixture name says stereo, and
`FIXTURE_DURATION_SECONDS` (8 s) long — except the Wave 4 drum and bass pattern
fixtures, which are `PATTERN_FIXTURE_DURATION_SECONDS` (8.5 s) so the last
event of their fourth cycle keeps its tail. See that constant for why.

The ground truth in each docstring was measured, not assumed. For the Wave 0
signals that means librosa 0.11 at 44.1 kHz — in particular the click bursts
are short shaped noise transients (not single samples), and the trains start at
`CLICK_TRAIN_OFFSET_SECONDS` because an onset landing at t=0 has no preceding
frame for the detector to see a rise against, and gets dropped. For the Wave 4
drum one-shots it means library-free numpy band shares and flatness, measured
with the conventions `band_energy_ratios` fixes; `tests/test_fixtures.py`
recomputes every number quoted below and fails if a fixture drifts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_pipeline import ANALYSIS_SAMPLE_RATE

FIXTURE_DURATION_SECONDS = 8.0

#: Buffer length for the Wave 4 pattern fixtures (drums and bass).
#:
#: Their musical content is still 8 s — four two-second cycles starting at
#: `DRUM_PATTERN_ANCHOR_SECONDS` — but the *last* event of the fourth cycle
#: begins at exactly 8.0 s, and `_hit_train` drops any hit whose tail would run
#: past the end of the buffer. At 8.0 s that silently costs the fixture its
#: final open hat (0.4 s tail) and its final bass note (0.46 s), i.e. the
#: exported hit counts would be lies. 8.5 s leaves room for the longest tail
#: (open hat, ending 8.4 s) with nothing truncated and nothing dropped.
PATTERN_FIXTURE_DURATION_SECONDS = 8.5

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

# --- Wave 4: drum pattern ground truth ---------------------------------------
DRUM_PATTERN_BPM = 120.0
DRUM_PATTERN_BEATS_PER_CYCLE = 4
DRUM_PATTERN_STEPS_PER_CYCLE = 16
DRUM_PATTERN_ANCHOR_SECONDS = 0.25
DRUM_PATTERN_CYCLES = 4
#: 4 beats at 120 BPM.
DRUM_PATTERN_CYCLE_SECONDS = 2.0
#: One 16th at 120 BPM.
DRUM_PATTERN_STEP_SECONDS = 0.125

DRUM_PATTERN_KICK_STEPS = (0, 8)  # 8 hits total
DRUM_PATTERN_SNARE_STEPS = (4, 12)  # 8 hits total
DRUM_PATTERN_HAT_STEPS = (0, 2, 4, 6, 8, 10, 12, 14)  # 32 hits total

#: `drum_pattern_kick_only`: nothing but kicks, on every beat.
DRUM_PATTERN_KICK_ONLY_STEPS = (0, 4, 8, 12)  # 16 hits total

#: `drum_pattern_ambiguous`: kicks plus broadband `_click()` bursts that belong
#: to no drum class.
DRUM_PATTERN_AMBIGUOUS_KICK_STEPS = (0, 8)  # 8 hits total
DRUM_PATTERN_AMBIGUOUS_CLICK_STEPS = (2, 6, 10, 14)  # 16 hits total

#: Peak amplitude of each one-shot. Chosen so the loudest coincidence in
#: `drum_pattern_120bpm` (kick + hat on steps 0 and 8) stays inside full scale:
#: `_hit_train` sums hits and then clips, and a clipped sum would invalidate
#: every band-share number measured on the isolated one-shots.
KICK_PEAK = 0.8
SNARE_PEAK = 0.5
HAT_PEAK = 0.18

# --- Wave 4: bass line ground truth ------------------------------------------
BASS_LINE_BPM = 120.0
BASS_LINE_MIDI = (33, 33, 36, 40) * 4
BASS_LINE_NOTE_NAMES = ("a1", "a1", "c2", "e2") * 4
#: Exact synthesis inputs, so the fixture cannot disagree with its own
#: constants: A1 = 55 Hz by definition of A4 = 440 Hz, the other two are
#: 440 * 2 ** ((midi - 69) / 12).
BASS_LINE_FREQS_HZ = (55.0, 55.0, 65.40639, 82.40689)
BASS_LINE_NOTE_SECONDS = 0.5
BASS_LINE_GAP_SECONDS = 0.04
BASS_LINE_ANCHOR_SECONDS = 0.25
#: One note per beat, so notes land on every fourth 16th step.
BASS_LINE_STEPS = (0, 4, 8, 12)
BASS_LINE_PEAK = 0.7

#: `bass_line_octave_trap` amplitudes: fundamental buried under the 2nd
#: harmonic. Harmonics 3-5 keep their 1/n weighting.
BASS_OCTAVE_TRAP_AMPLITUDES = (0.15, 1.0, 1 / 3, 1 / 4, 1 / 5)

#: `bass_line_with_glide`: A1 held, linear glide, E2 held, repeated.
BASS_GLIDE_MIDI = (33, 40)
BASS_GLIDE_NOTE_NAMES = ("a1", "e2")
BASS_GLIDE_HOLD_SECONDS = 0.4
BASS_GLIDE_SECONDS = 0.1
BASS_GLIDE_GAP_SECONDS = 0.06
BASS_GLIDE_PAIRS = 8

#: `bass_unvoiced`: band-limited noise sitting squarely in the bass register.
BASS_UNVOICED_BAND_HZ = (20.0, 200.0)


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


def _hit_train(
    hits: Iterable[tuple[float, np.ndarray]],
    duration: float = FIXTURE_DURATION_SECONDS,
) -> np.ndarray:
    """Stamp `(time_seconds, one_shot)` pairs into one buffer.

    The single placer for every event-based fixture in this file, so every one
    of them shares the same two properties:

    * **Additive.** Coincident hits sum, which is exactly what a real drum
      pattern does and what the Wave 4 drum fixtures are built on — kick and
      hat land on the same instant at steps 0 and 8. (Peak levels are chosen in
      `KICK_PEAK` and friends so those sums stay inside full scale; the final
      clip is a backstop, not part of the design.)
    * **A hit whose tail would run past the end of the buffer is dropped, not
      truncated.** Silent truncation would leave a hit the exported counts claim
      is a whole one; dropping makes the count wrong loudly instead. See
      `PATTERN_FIXTURE_DURATION_SECONDS` for the length that keeps every Wave 4
      hit intact.
    """
    out = np.zeros(_n_samples(duration), dtype=np.float32)
    for time_s, hit in hits:
        start = int(round(float(time_s) * ANALYSIS_SAMPLE_RATE))
        end = start + hit.size
        if end <= out.size:
            out[start:end] += hit
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _click_train(onset_times: np.ndarray, duration: float = FIXTURE_DURATION_SECONDS) -> np.ndarray:
    """Place a click burst at each time in `onset_times`."""
    click = _click()
    return _hit_train(((float(time_s), click) for time_s in onset_times), duration)


# --- Wave 4 one-shots --------------------------------------------------------
#
# Ground truth in each docstring below was measured, not derived: band shares
# over the drum detection bands (kick 20-150, body 150-500, noise 1k-6k, air
# 6k-16k Hz), assigned by bin centre frequency over half-open intervals and
# summed as `magnitude ** 2` — the conventions `band_energy_ratios` fixes for
# the whole project. Flatness is the geometric/arithmetic mean ratio of the same
# power spectrum over 20 Hz-16 kHz, floored 120 dB below the peak bin so it is
# scale-invariant and cannot be decided by the nulls of a band-limited signal.
# `tests/test_fixtures.py` implements both and pins these numbers.


def _decay(n: int, tau_seconds: float) -> np.ndarray:
    """Exponential decay envelope, 1.0 at sample 0."""
    return np.exp(-(np.arange(n, dtype=np.float64) / ANALYSIS_SAMPLE_RATE) / tau_seconds)


def _band_limited_noise(
    rng: np.random.Generator, n: int, low_hz: float, high_hz: float
) -> np.ndarray:
    """Gaussian noise with everything outside `[low_hz, high_hz]` zeroed."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    frequencies = np.fft.rfftfreq(n, 1.0 / ANALYSIS_SAMPLE_RATE)
    spectrum[(frequencies < low_hz) | (frequencies > high_hz)] = 0.0
    shaped = np.fft.irfft(spectrum, n=n)
    peak = float(np.max(np.abs(shaped)))
    return shaped / peak if peak > 0.0 else shaped


def _normalised(signal: np.ndarray, peak: float) -> np.ndarray:
    """Scale to an exact peak amplitude and return float32."""
    largest = float(np.max(np.abs(signal)))
    scaled = signal / largest * peak if largest > 0.0 else signal
    return scaled.astype(np.float32)


def _kick(seed: int = 11) -> np.ndarray:
    """One kick: 60 Hz sine, 90 ms decay, with a 3 ms 150 -> 60 Hz pitch sweep.

    300 ms long, peak `KICK_PEAK`. `seed` is accepted for signature symmetry
    with the other one-shots and is unused — the kick has no stochastic
    component, which is the point: it is the tonal end of the drum spectrum.

    Measured ground truth:
      * band shares kick 0.9941, body 0.0058, noise 0.0000, air 0.0000 — so
        well past the >0.90 below 150 Hz that classification needs
      * flatness 0.000001, i.e. as tonal as a signal gets (< 0.15 comfortably)
      * decay to -20 dB in 37 ms
    """
    n = int(round(0.3 * ANALYSIS_SAMPLE_RATE))
    times = np.arange(n, dtype=np.float64) / ANALYSIS_SAMPLE_RATE
    sweep_seconds = 0.003
    frequency = np.where(
        times < sweep_seconds,
        150.0 + (60.0 - 150.0) * (times / sweep_seconds),
        60.0,
    )
    # Integrate instantaneous frequency so the sweep stays phase-continuous.
    phase = 2 * np.pi * np.cumsum(frequency) / ANALYSIS_SAMPLE_RATE
    return _normalised(_decay(n, 0.09) * np.sin(phase), KICK_PEAK)


def _snare(seed: int = 12) -> np.ndarray:
    """One snare: 200 Hz body (60 ms decay) at 0.4 plus 1-6 kHz noise at 0.6.

    200 ms long, peak `SNARE_PEAK`. The noise decays over 120 ms, so the tail is
    all rattle and no shell tone — the shape that separates a snare from a tom.

    Measured ground truth — **two numbers here contradict the obvious
    expectation, and a classifier tuned on theory rather than on these will
    misread this fixture**:
      * band shares kick 0.0118, body 0.5935, noise 0.3937, air 0.0010
      * kick share 0.0118, well under 0.15, as expected
      * noise share is **0.3937, not the > 0.45 a 0.4-body/0.6-noise mix
        suggests**. Amplitude is not energy: the body is a single spectral line
        at 200 Hz while the noise is spread over 5 kHz, and the body also has a
        60 ms decay against the noise's 120 ms. What actually identifies this
        one-shot is `body_ratio` — see `_click`, whose noise share is 0.5641,
        *higher* than the snare's. A snare score keyed on `noise_ratio` alone
        would call every broadband click a snare.
      * flatness 0.0015 — 1500x the kick's, but NOWHERE NEAR the > 0.6 a
        textbook reading suggests. Band-limited noise has near-zero bins outside
        its passband, and a geometric mean is decided by its smallest values.
        Flatness discriminates these one-shots by *orders of magnitude*, not on
        an absolute 0-1 scale: kick 1e-6, snare 1.5e-3, hats 3-4e-2, click 0.52.
      * decay to -20 dB in 141 ms
    """
    n = int(round(0.2 * ANALYSIS_SAMPLE_RATE))
    times = np.arange(n, dtype=np.float64) / ANALYSIS_SAMPLE_RATE
    rng = np.random.default_rng(seed)
    body = np.sin(2 * np.pi * 200.0 * times) * _decay(n, 0.06)
    noise = _band_limited_noise(rng, n, 1000.0, 6000.0) * _decay(n, 0.12)
    return _normalised(0.4 * body + 0.6 * noise, SNARE_PEAK)


def _hat_closed(seed: int = 13) -> np.ndarray:
    """One closed hat: noise above 6 kHz with a 15 ms decay. 50 ms, peak `HAT_PEAK`.

    Measured ground truth:
      * band shares kick 0.0000, body 0.0000, noise 0.0011, air 0.9989 — the
        > 0.80 air share hat detection needs, with room to spare
      * flatness 0.0414
      * decay to -20 dB in 34 ms, against the open hat's 277 ms: an 8x gap, and
        the whole basis of the `hh` versus `oh` suggestion
    """
    rng = np.random.default_rng(seed)
    n = int(round(0.05 * ANALYSIS_SAMPLE_RATE))
    return _normalised(_band_limited_noise(rng, n, 6000.0, 20000.0) * _decay(n, 0.015), HAT_PEAK)


def _hat_open(seed: int = 14) -> np.ndarray:
    """One open hat: the closed hat's spectrum with a 150 ms decay.

    400 ms long, peak `HAT_PEAK`. Deliberately identical in spectrum to
    `_hat_closed` so decay length is the *only* thing telling them apart.

    Measured ground truth:
      * band shares kick 0.0000, body 0.0000, noise 0.0003, air 0.9997
      * flatness 0.0277
      * decay to -20 dB in 277 ms
    """
    rng = np.random.default_rng(seed)
    n = int(round(0.4 * ANALYSIS_SAMPLE_RATE))
    return _normalised(_band_limited_noise(rng, n, 6000.0, 20000.0) * _decay(n, 0.15), HAT_PEAK)


def _step_times(
    steps: Sequence[int],
    cycles: int = DRUM_PATTERN_CYCLES,
    anchor: float = DRUM_PATTERN_ANCHOR_SECONDS,
) -> list[float]:
    """Absolute times of `steps` repeated over `cycles` cycles of the grid."""
    return [
        anchor + cycle * DRUM_PATTERN_CYCLE_SECONDS + step * DRUM_PATTERN_STEP_SECONDS
        for cycle in range(cycles)
        for step in steps
    ]


def _drum_pattern(
    kick_steps: Sequence[int] = (),
    snare_steps: Sequence[int] = (),
    hat_steps: Sequence[int] = (),
    click_steps: Sequence[int] = (),
    hat: np.ndarray | None = None,
) -> np.ndarray:
    """Build one drum-pattern buffer from per-class step sets."""
    closed_hat = _hat_closed() if hat is None else hat
    placements: list[tuple[float, np.ndarray]] = []
    for steps, one_shot in (
        (kick_steps, _kick()),
        (snare_steps, _snare()),
        (hat_steps, closed_hat),
        (click_steps, _click()),
    ):
        placements.extend((time_s, one_shot) for time_s in _step_times(steps))
    placements.sort(key=lambda item: item[0])
    return _hit_train(placements, PATTERN_FIXTURE_DURATION_SECONDS)


# --- Wave 4 bass ------------------------------------------------------------


def _bass_note(
    frequency_hz: float,
    seconds: float,
    amplitudes: Sequence[float] = (1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 5),
    peak: float = BASS_LINE_PEAK,
) -> np.ndarray:
    """One plucked-ish bass note: fundamental plus harmonics 2-5 at 1/n.

    5 ms linear attack and 20 ms linear release, so there is a transient to
    detect and no click at the release.
    """
    n = int(round(seconds * ANALYSIS_SAMPLE_RATE))
    times = np.arange(n, dtype=np.float64) / ANALYSIS_SAMPLE_RATE
    wave = np.zeros(n, dtype=np.float64)
    for harmonic, amplitude in enumerate(amplitudes, start=1):
        wave += amplitude * np.sin(2 * np.pi * harmonic * frequency_hz * times)
    return _normalised(wave * _attack_release(n), peak)


def _attack_release(n: int, attack: float = 0.005, release: float = 0.02) -> np.ndarray:
    """Linear 5 ms attack / 20 ms release envelope, flat in between."""
    envelope = np.ones(n, dtype=np.float64)
    attack_samples = min(n, int(round(attack * ANALYSIS_SAMPLE_RATE)))
    release_samples = min(n - attack_samples, int(round(release * ANALYSIS_SAMPLE_RATE)))
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, endpoint=False)
    if release_samples > 0:
        envelope[n - release_samples :] = np.linspace(1.0, 0.0, release_samples)
    return envelope


def _glide_pair(
    amplitudes: Sequence[float] = (1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 5),
    peak: float = BASS_LINE_PEAK,
) -> np.ndarray:
    """A1 held, a linear frequency glide, then E2 held — as one continuous signal.

    Built by integrating instantaneous frequency, so phase is continuous across
    the glide and there is no discontinuity for an onset detector to mistake for
    a third note.
    """
    hold = int(round(BASS_GLIDE_HOLD_SECONDS * ANALYSIS_SAMPLE_RATE))
    glide = int(round(BASS_GLIDE_SECONDS * ANALYSIS_SAMPLE_RATE))
    start_hz, end_hz = BASS_LINE_FREQS_HZ[0], BASS_LINE_FREQS_HZ[3]
    frequency = np.concatenate(
        [
            np.full(hold, start_hz),
            np.linspace(start_hz, end_hz, glide, endpoint=False),
            np.full(hold, end_hz),
        ]
    )
    phase = 2 * np.pi * np.cumsum(frequency) / ANALYSIS_SAMPLE_RATE
    wave = np.zeros(phase.size, dtype=np.float64)
    for harmonic, amplitude in enumerate(amplitudes, start=1):
        wave += amplitude * np.sin(harmonic * phase)
    return _normalised(wave * _attack_release(phase.size), peak)


def _bass_line(amplitudes: Sequence[float]) -> np.ndarray:
    """Place `BASS_LINE_FREQS_HZ` x 4 on the beat, one note per beat."""
    sounding = BASS_LINE_NOTE_SECONDS - BASS_LINE_GAP_SECONDS
    placements = [
        (
            BASS_LINE_ANCHOR_SECONDS + index * BASS_LINE_NOTE_SECONDS,
            _bass_note(BASS_LINE_FREQS_HZ[index % len(BASS_LINE_FREQS_HZ)], sounding, amplitudes),
        )
        for index in range(len(BASS_LINE_MIDI))
    ]
    return _hit_train(placements, PATTERN_FIXTURE_DURATION_SECONDS)


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


# --- Wave 4 composite fixtures ----------------------------------------------


@pytest.fixture(scope="session")
def drum_pattern_120bpm() -> np.ndarray:
    """8.5 s kick/snare/closed-hat pattern, 120 BPM, 16 steps per cycle.

    Four two-second cycles anchored at `DRUM_PATTERN_ANCHOR_SECONDS` (0.25 s —
    the same lead-in the click trains use, because a detector cannot see an
    onset at t=0). One step is 125 ms.

    Ground truth:
      * kick on steps `DRUM_PATTERN_KICK_STEPS` (0, 8)  -> 8 hits
      * snare on `DRUM_PATTERN_SNARE_STEPS` (4, 12)     -> 8 hits
      * hat on `DRUM_PATTERN_HAT_STEPS` (every other)   -> 32 hits
      * 48 hits total, but only **32 distinct instants**: the kick and snare
        steps are a subset of the hat steps, so 16 of the 48 hits share an
        instant with another hit

    **This is the load-bearing fixture of Wave 4, and the coincidences are the
    point.** Kick and hat sound together on steps 0 and 8; snare and hat on 4
    and 12. Any design that classifies a single global onset list assigns one
    class per instant and therefore finds at most 8 of the 32 hats here — it
    deletes the hat on every downbeat, which is wrong in a way that is obvious
    by ear. Per-band detection finds two hits at those instants because two
    detectors fired. Do not "simplify" this fixture by moving the hats off the
    kick and snare: that removes the only property it exists to test.
    """
    return _drum_pattern(
        kick_steps=DRUM_PATTERN_KICK_STEPS,
        snare_steps=DRUM_PATTERN_SNARE_STEPS,
        hat_steps=DRUM_PATTERN_HAT_STEPS,
    )


@pytest.fixture(scope="session")
def drum_pattern_kick_only() -> np.ndarray:
    """8.5 s of kicks on every beat and nothing else. 120 BPM, 4 cycles.

    Ground truth:
      * kick on steps `DRUM_PATTERN_KICK_ONLY_STEPS` (0, 4, 8, 12) -> 16 hits
      * zero snares, zero hats, zero unclassified — nothing else is present

    This pins the failure mode that matters most in practice: a classifier that
    hallucinates classes which are not there. Silence in a class is a real
    answer, and reporting "no hats" is far more useful than reporting a hat
    pattern invented from a kick's harmonics.
    """
    return _drum_pattern(kick_steps=DRUM_PATTERN_KICK_ONLY_STEPS)


@pytest.fixture(scope="session")
def drum_pattern_open_hats() -> np.ndarray:
    """`drum_pattern_120bpm` with open hats: identical steps, 10x the decay.

    Ground truth:
      * exactly the step sets of `drum_pattern_120bpm` (kick 0/8, snare 4/12,
        hat every other step), same 48 hits at 32 distinct instants
      * hats decay to -20 dB in 277 ms against the closed hat's 34 ms, and
        their spectrum is identical — so decay length is the only available
        discriminator, which is the `oh` versus `hh` suggestion in miniature
    """
    return _drum_pattern(
        kick_steps=DRUM_PATTERN_KICK_STEPS,
        snare_steps=DRUM_PATTERN_SNARE_STEPS,
        hat_steps=DRUM_PATTERN_HAT_STEPS,
        hat=_hat_open(),
    )


@pytest.fixture(scope="session")
def drum_pattern_ambiguous() -> np.ndarray:
    """Kicks plus broadband clicks that belong to no drum class. 8.5 s, 120 BPM.

    Ground truth:
      * kick on steps `DRUM_PATTERN_AMBIGUOUS_KICK_STEPS` (0, 8) -> 8 hits
      * the existing `_click()` burst on `DRUM_PATTERN_AMBIGUOUS_CLICK_STEPS`
        (2, 6, 10, 14) -> 16 hits. It is ~6 ms of shaped broadband noise: no low
        end, no tail, no shell tone, so it is not a kick, a snare or a hat.
      * those 16 clicks must land in `unclassified` **with their timing
        preserved**. Dropping them loses two thirds of the hits; forcing them
        into a class invents one.
      * measured click shares: kick 0.0000, body 0.0422, noise 0.5641, air
        0.3937; flatness 0.5201; decay to -20 dB in 2 ms. Note it is *noisier
        than the snare* (0.5641 against 0.3937) and has *more air than the
        snare* (0.3937 against 0.0010) while having a third of the closed hat's
        air share. It sits between the classes on every single band, which is
        exactly why the honest answer is `unclassified`.
    """
    return _drum_pattern(
        kick_steps=DRUM_PATTERN_AMBIGUOUS_KICK_STEPS,
        click_steps=DRUM_PATTERN_AMBIGUOUS_CLICK_STEPS,
    )


@pytest.fixture(scope="session")
def bass_line_a_minor() -> np.ndarray:
    """8.5 s bass line, (A1, A1, C2, E2) x 4, one note per beat at 120 BPM.

    Each note sounds for 460 ms with a 40 ms silent tail, synthesised as
    fundamental plus harmonics 2-5 at 1/n amplitude, 5 ms attack / 20 ms
    release, first note at `BASS_LINE_ANCHOR_SECONDS`.

    Ground truth is exact **by construction** — the signal is built from the
    literal frequencies in `BASS_LINE_FREQS_HZ`, so the fixture cannot disagree
    with its own constants:
      * 16 notes, MIDI `BASS_LINE_MIDI` = 33, 33, 36, 40 repeated
      * names `BASS_LINE_NOTE_NAMES` = a1, a1, c2, e2 repeated
      * fundamentals 55.0, 55.0, 65.40639, 82.40689 Hz
      * notes start on 16th steps `BASS_LINE_STEPS` (0, 4, 8, 12) of each cycle
      * measured: the first note's spectrum peaks at 55.0 Hz, and the strongest
        bin of each note is its own fundamental (harmonics are 1/n below it)
    """
    return _bass_line((1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 5))


@pytest.fixture(scope="session")
def bass_line_octave_trap() -> np.ndarray:
    """`bass_line_a_minor` with the fundamental buried under its 2nd harmonic.

    Amplitudes are `BASS_OCTAVE_TRAP_AMPLITUDES`: fundamental 0.15, 2nd
    harmonic 1.0, harmonics 3-5 unchanged at 1/n. This is what a bass sounds
    like after a high-pass filter, a small speaker, or an over-eager separation
    model — common enough that it is the single most likely way a pitch tracker
    embarrasses itself.

    Ground truth:
      * **the correct answer is still MIDI 33 / 33 / 36 / 40 (a1, a1, c2, e2)**
      * a tracker reporting 45 / 45 / 48 / 52 has made an octave error, not a
        different-but-defensible reading
      * measured: the loudest bin of the first note is 110 Hz, twice the
        fundamental — this fixture exists to fail a naive peak-picking tracker
    """
    return _bass_line(BASS_OCTAVE_TRAP_AMPLITUDES)


@pytest.fixture(scope="session")
def bass_line_with_glide() -> np.ndarray:
    """8.5 s of A1 held 0.4 s, a 0.1 s linear glide, E2 held 0.4 s, repeated.

    `BASS_GLIDE_PAIRS` (8) pairs with a 60 ms silence between them, first at
    `BASS_LINE_ANCHOR_SECONDS`. Each pair is one phase-continuous signal, so the
    glide is a real frequency ramp rather than a crossfade.

    Ground truth:
      * exactly **2 notes per pair** — 16 notes total, MIDI 33 and 40 alternating
      * the glide frames must be absorbed into the notes either side, not
        emitted as the string of chromatic junk a naive frame-to-note segmenter
        produces (a 0.1 s ramp over 9 semitones crosses 8 semitone boundaries)
      * the held sections are 400 ms, comfortably above any minimum note length;
        no individual glide semitone lasts more than ~11 ms
    """
    pair_seconds = 2 * BASS_GLIDE_HOLD_SECONDS + BASS_GLIDE_SECONDS
    placements = [
        (
            BASS_LINE_ANCHOR_SECONDS + index * (pair_seconds + BASS_GLIDE_GAP_SECONDS),
            _glide_pair(),
        )
        for index in range(BASS_GLIDE_PAIRS)
    ]
    return _hit_train(placements, PATTERN_FIXTURE_DURATION_SECONDS)


@pytest.fixture(scope="session")
def bass_unvoiced() -> np.ndarray:
    """8.5 s of band-limited noise, 20-200 Hz, peak 0.5, seed 21.

    Loud, low, and completely unpitched: the shape of a rumble, a kick bleeding
    into a bass stem, or a Demucs residue.

    Ground truth:
      * **no notes.** Nothing invented, nothing crashed. A pitch tracker will
        return f0 estimates for some frames regardless — there is always a
        strongest partial — so the voicing gate, not the tracker, is what has to
        hold here.
      * measured band shares: kick 0.7216, body 0.2784, noise 1e-16, air 1e-16
        — the 20-200 Hz passband straddles the 150 Hz kick/body boundary, and
        there is nothing whatsoever above 200 Hz
    """
    rng = np.random.default_rng(21)
    n = _n_samples(PATTERN_FIXTURE_DURATION_SECONDS)
    low_hz, high_hz = BASS_UNVOICED_BAND_HZ
    return _normalised(_band_limited_noise(rng, n, low_hz, high_hz), 0.5)


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
