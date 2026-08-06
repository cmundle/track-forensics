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
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from conftest import (
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

from audio_pipeline import ANALYSIS_SAMPLE_RATE, BAND_EDGES_HZ
from audio_pipeline.backends import BASS_F0_MIN_HZ
from audio_pipeline.backends.librosa_backend import (
    BRIGHTNESS_CUTOFF_HZ,
    MAX_TRANSIENT_SHARPNESS,
    PYIN_FRAME_LENGTH,
    STFT_HOP_LENGTH,
    STFT_N_FFT,
    LibrosaBackend,
    band_energy_ratios,
    brightness,
    energy_percentile_hz,
    energy_weighted_centroid_hz,
    transient_sharpness,
)
from audio_pipeline.note_track import segment_notes
from audio_pipeline.schemas import (
    BassLine,
    DynamicsFeatures,
    PitchTrack,
    RhythmFeatures,
    SpectralFeatures,
    TonalFeatures,
)
from audio_pipeline.strudel_vocab import SUB_BASS_CENTROID_HZ_MAX, suggest_bass_sound


@pytest.fixture(scope="module")
def backend() -> LibrosaBackend:
    """The backend under test, constructed directly rather than resolved."""
    return LibrosaBackend()


# --------------------------------------------------------------------------- #
# The F4 signal: a sub bass that rests, over a real stem's noise floor.
#
# `conftest.py` is owned by another package and has no fixture that rests over
# a noise floor, which is the exact condition finding F4 turns on — so this one
# is built here. `tests/test_essentia_backend.py` imports this builder rather
# than reimplementing it, because "both backends agree" is only a claim about
# the descriptor if both are fed a bit-identical signal.
# --------------------------------------------------------------------------- #

#: A1, the lowest note on a 4-string bass and squarely in sub territory.
SUB_BASS_FREQUENCY_HZ = 55.0

#: Peak of the tone while it is sounding. Nothing depends on the exact value —
#: every descriptor under test here is scale-invariant — but it is well clear
#: of the floor below.
SUB_BASS_PEAK = 0.7

#: On/off block length. 8 s of fixture divides into 16 whole blocks, 8 sounding
#: and 8 resting, so the silent fraction is exactly 0.5.
SUB_BASS_BLOCK_SECONDS = 0.5

#: RMS of the broadband floor a Demucs stem actually rests at: 7.7e-05, about
#: -82 dBFS. Measured on the v4 calibration bass stem, and it is the whole
#: point of the fixture — **digital silence would not reproduce F4**. librosa
#: reports a centroid of 0 Hz for a frame with no energy at all, which drags
#: `centroid_mean` *down*; it is a floor with a flat spectrum that puts a
#: resting frame's centroid up in the kilohertz and drags the mean up.
STEM_NOISE_FLOOR_RMS = 7.7e-5

#: Fixed so the numbers pinned below are reproducible.
SUB_BASS_SEED = 3


def sub_bass_over_noise_floor(*, rests: bool) -> np.ndarray:
    """A 55 Hz sine over a -82 dBFS floor, with or without 50% rests.

    The pair is the measurement: identical apart from the gate, so any
    descriptor that moves between them is being moved by silence and nothing
    else.
    """
    n_samples = int(round(ANALYSIS_SAMPLE_RATE * FIXTURE_DURATION_SECONDS))
    times = np.arange(n_samples) / ANALYSIS_SAMPLE_RATE
    tone = SUB_BASS_PEAK * np.sin(2 * np.pi * SUB_BASS_FREQUENCY_HZ * times)

    if rests:
        block = int(round(SUB_BASS_BLOCK_SECONDS * ANALYSIS_SAMPLE_RATE))
        gate = np.zeros(n_samples)
        for start in range(0, n_samples, 2 * block):
            gate[start : start + block] = 1.0
        assert gate.mean() == pytest.approx(0.5)  # the fixture's own ground truth
        tone = tone * gate

    floor = np.random.default_rng(SUB_BASS_SEED).standard_normal(n_samples)
    return (tone + floor * STEM_NOISE_FLOOR_RMS).astype(np.float32)


#: Measured `centroid_energy_hz` of both signals above: **54.91 Hz** against a
#: synthesis frequency of 55.0, on both backends. A first moment is continuous
#: in the input, so it lands on the tone rather than on a bin centre — the
#: 0.09 Hz shortfall is the -82 dBFS floor's own broadband energy pulling very
#: slightly, and nothing else.
#:
#: For contrast, and this is the reason the descriptor is a centroid and not a
#: percentile: `rolloff_energy_hz` on the same signal reads 64.5996 Hz, FFT bin
#: 3 of the 2048-point grid (3 * 44100 / 2048). A bin-quantised statistic reads
#: the same 64.6 Hz for a 55 Hz sine, a 55 Hz square and a 55 Hz sawtooth
#: alike, which is exactly the discrimination `suggest_bass_sound` needs.
SUB_BASS_CENTROID_ENERGY_HZ = 54.91
SUB_BASS_ROLLOFF_ENERGY_HZ = 64.599609375


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
# spectral(): silence contamination, and which descriptors survive it (F4)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gated_sub_bass_spectral(backend: LibrosaBackend) -> SpectralFeatures:
    return backend.spectral(sub_bass_over_noise_floor(rests=True), ANALYSIS_SAMPLE_RATE)


@pytest.fixture(scope="module")
def continuous_sub_bass_spectral(backend: LibrosaBackend) -> SpectralFeatures:
    return backend.spectral(sub_bass_over_noise_floor(rests=False), ANALYSIS_SAMPLE_RATE)


def test_centroid_energy_reads_the_sub_bass_through_the_rests(
    gated_sub_bass_spectral: SpectralFeatures,
) -> None:
    """The assertion finding F4 turns on: a 55 Hz sub reads as 55 Hz.

    `SUB_BASS_CENTROID_HZ_MAX` is the ceiling this has to clear, and the whole
    of F4 is that no unweighted frame-mean centroid ever clears it. The tighter
    assertion is that the number is *right*, not merely small: a descriptor
    that reported 100 Hz for everything in the bass register would also pass a
    threshold test and would be useless to the verdict it feeds.
    """
    assert gated_sub_bass_spectral.centroid_energy_hz is not None
    assert gated_sub_bass_spectral.centroid_energy_hz < SUB_BASS_CENTROID_HZ_MAX
    assert gated_sub_bass_spectral.centroid_energy_hz == pytest.approx(
        SUB_BASS_CENTROID_ENERGY_HZ, abs=0.05
    )


def test_rolloff_energy_survives_the_rests_too(
    gated_sub_bass_spectral: SpectralFeatures,
) -> None:
    """`rolloff_energy_hz`, the corrected companion to `rolloff_mean`.

    Bin-quantised, so 64.6 Hz rather than 54.9 — correct for a rolloff, and
    the reason the same helper is not used for the centroid.
    """
    assert gated_sub_bass_spectral.rolloff_energy_hz is not None
    assert gated_sub_bass_spectral.rolloff_energy_hz == pytest.approx(
        SUB_BASS_ROLLOFF_ENERGY_HZ, abs=0.001
    )
    assert gated_sub_bass_spectral.rolloff_mean is not None
    assert gated_sub_bass_spectral.rolloff_mean > 100 * gated_sub_bass_spectral.rolloff_energy_hz


def test_centroid_mean_is_the_broken_descriptor_and_still_is(
    gated_sub_bass_spectral: SpectralFeatures,
) -> None:
    """Pins the bug in place, deliberately.

    `centroid_mean` is kept unchanged so v4 outputs stay comparable (settled
    decision 1), so this asserts that it is still wrong: it reads in the
    kilohertz on a signal with no energy above ~100 Hz, and — the signature F4
    names — its standard deviation is larger than its mean. If this test ever
    starts failing, someone has changed `centroid_mean` in place and the v4
    calibration baseline no longer means anything.
    """
    mean = gated_sub_bass_spectral.centroid_mean
    std = gated_sub_bass_spectral.centroid_std
    assert mean is not None and std is not None
    assert mean > 1000.0
    assert std > mean


def test_only_the_per_frame_descriptors_move_when_rests_are_added(
    gated_sub_bass_spectral: SpectralFeatures,
    continuous_sub_bass_spectral: SpectralFeatures,
) -> None:
    """The measurement behind the whole package.

    Two signals identical but for a 50% gate. Energy-summed descriptors
    (`centroid_energy_hz`, `rolloff_energy_hz`, `brightness`, the band ratios)
    barely notice; unweighted per-frame means (`centroid_mean`, `rolloff_mean`)
    move by three or four orders of magnitude.

    `centroid_energy_hz` is not bit-identical between the two — 54.912 gated
    against 55.014 continuous, **0.185%**. That is not the silence leaking in:
    measured with the noise floor removed entirely, the same two signals differ
    by the same 0.185%, so it is the gate's own 16 hard edges redistributing a
    little energy, and it would shrink to nothing under a fade. The tolerance
    below is set at 0.5%, about 2.7x the measured figure. Compare the numbers
    it is standing next to: `centroid_mean` moves by 6007% across the same
    pair.
    """
    gated, continuous = gated_sub_bass_spectral, continuous_sub_bass_spectral

    assert gated.centroid_energy_hz is not None
    assert continuous.centroid_energy_hz is not None
    assert gated.centroid_energy_hz == pytest.approx(continuous.centroid_energy_hz, rel=0.005)
    assert gated.rolloff_energy_hz == continuous.rolloff_energy_hz
    assert gated.band_energy_ratios.low is not None
    assert continuous.band_energy_ratios.low is not None
    assert gated.band_energy_ratios.low == pytest.approx(
        continuous.band_energy_ratios.low, abs=1e-4
    )

    assert gated.centroid_mean is not None and continuous.centroid_mean is not None
    assert gated.centroid_mean > 20 * continuous.centroid_mean
    assert gated.rolloff_mean is not None and continuous.rolloff_mean is not None
    assert gated.rolloff_mean > 20 * continuous.rolloff_mean


def test_brightness_needs_no_corrected_variant(backend: LibrosaBackend) -> None:
    """Measured, not assumed — W4D task 2, and the answer is "no correction".

    `brightness` is already summed over the energy of every frame, so it should
    be immune to the same gating. On a signal with real content in both
    registers (200 Hz and 4 kHz), inserting 50% rests over the stem noise floor
    moves it by under 0.1% — measured 0.13793 either way. `centroid_mean` moves
    by +345% and `rolloff_mean` by +169% on the same pair.

    The sub-bass fixture cannot show this: its brightness is ~1e-9 either way,
    and a ratio of two near-zeros is not a measurement.
    """
    n_samples = int(round(ANALYSIS_SAMPLE_RATE * FIXTURE_DURATION_SECONDS))
    times = np.arange(n_samples) / ANALYSIS_SAMPLE_RATE
    two_register = 0.5 * np.sin(2 * np.pi * 200.0 * times) + 0.2 * np.sin(
        2 * np.pi * 4000.0 * times
    )
    floor = np.random.default_rng(SUB_BASS_SEED).standard_normal(n_samples) * STEM_NOISE_FLOOR_RMS

    block = int(round(SUB_BASS_BLOCK_SECONDS * ANALYSIS_SAMPLE_RATE))
    gate = np.zeros(n_samples)
    for start in range(0, n_samples, 2 * block):
        gate[start : start + block] = 1.0

    continuous = backend.spectral((two_register + floor).astype(np.float32), ANALYSIS_SAMPLE_RATE)
    gated = backend.spectral((two_register * gate + floor).astype(np.float32), ANALYSIS_SAMPLE_RATE)

    assert continuous.brightness is not None and gated.brightness is not None
    assert continuous.brightness == pytest.approx(0.13793, abs=0.001)
    assert gated.brightness == pytest.approx(continuous.brightness, rel=0.001)

    # The control: the per-frame descriptors on the very same pair of signals.
    assert continuous.centroid_mean is not None and gated.centroid_mean is not None
    assert gated.centroid_mean > 3 * continuous.centroid_mean
    assert continuous.rolloff_mean is not None and gated.rolloff_mean is not None
    assert gated.rolloff_mean > 2 * continuous.rolloff_mean


def test_the_f4_stem_now_resolves_to_a_sine(
    gated_sub_bass_spectral: SpectralFeatures,
) -> None:
    """End to end: audio in, Strudel sound name out. This is the F4 regression.

    Reaching across into `strudel_vocab` from a backend test is deliberate —
    F4 was invisible precisely because each half looked fine on its own. The
    descriptor was plausible, the threshold was correct, and only the join
    between them was broken.

    On `match`: the plan's W4D task 3 asks for `"exact"` here. This module's
    own settled semantics reserve `"exact"` for "Strudel ships a sound under
    this name for this thing" (a kick really is `bd`), and a bass verdict is a
    spectral reading mapped onto the nearest of four waveforms, which is
    `"approximate"`. The v5 fix makes the branch reachable; it does not make
    the inference exact. Flagged to the orchestrator rather than settled here.
    """
    result = suggest_bass_sound(BassLine(status="ok"), gated_sub_bass_spectral)
    assert len(result) == 1
    assert result[0].sound == "sine"
    assert result[0].match == "approximate"
    assert result[0].evidence["centroid_energy_hz"] < SUB_BASS_CENTROID_HZ_MAX


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
    assert spectral.centroid_energy_hz is None
    assert spectral.rolloff_mean is None
    assert spectral.rolloff_energy_hz is None
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


def _spectrum_of(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(magnitude, freqs)` on the project's pinned STFT grid.

    The shared helpers take a spectrum rather than audio, so tests that want to
    exercise them on a real signal need this one step. Same parameters
    `LibrosaBackend.spectral` uses, so a helper called through here sees exactly
    what it sees in production.
    """
    import librosa

    magnitude = np.abs(
        librosa.stft(
            np.asarray(audio, dtype=np.float32),
            n_fft=STFT_N_FFT,
            hop_length=STFT_HOP_LENGTH,
        )
    )
    freqs = np.asarray(
        librosa.fft_frequencies(sr=ANALYSIS_SAMPLE_RATE, n_fft=STFT_N_FFT), dtype=np.float64
    )
    return magnitude, freqs


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


def test_energy_percentile_finds_the_bin_holding_the_energy() -> None:
    """All the energy in one bin: every percentile is that bin's frequency."""
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    magnitude = np.zeros((freqs.size, 1))
    target = int(np.argmin(np.abs(freqs - 5000.0)))
    magnitude[target] = 1.0

    for fraction in (0.01, 0.5, 0.85, 1.0):
        assert energy_percentile_hz(magnitude, freqs, fraction) == pytest.approx(freqs[target])
    # And the centroid agrees, because there is only one place to be.
    assert energy_weighted_centroid_hz(magnitude, freqs) == pytest.approx(freqs[target])


def test_energy_percentile_is_ill_conditioned_on_a_bimodal_spectrum() -> None:
    """Half the energy at 100 Hz, half at 10 kHz: at 0.5 the answer is a coin toss.

    The result is the first bin at which cumulative energy *reaches* the
    fraction, which for an exact even split is the lower tone — and a hair more
    energy up top moves it by 9.9 kHz. Asserted rather than hidden, because it
    is the property that disqualified a percentile from being the centroid: a
    descriptor that can swing three orders of magnitude on a rounding
    difference cannot be thresholded. A first moment reads ~5 kHz for both of
    these and moves smoothly.
    """
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    low_bin = int(np.argmin(np.abs(freqs - 100.0)))
    high_bin = int(np.argmin(np.abs(freqs - 10000.0)))

    magnitude = np.zeros((freqs.size, 1))
    magnitude[low_bin] = 1.0
    magnitude[high_bin] = 1.0
    assert energy_percentile_hz(magnitude, freqs, 0.5) == pytest.approx(freqs[low_bin])
    steady = energy_weighted_centroid_hz(magnitude, freqs)

    magnitude[high_bin] = 1.001
    assert energy_percentile_hz(magnitude, freqs, 0.5) == pytest.approx(freqs[high_bin])
    assert energy_weighted_centroid_hz(magnitude, freqs) == pytest.approx(steady, rel=0.002)


def _bass_waveform(f0: float, harmonic_amplitude: Callable[[int], float]) -> np.ndarray:
    """Band-limited additive synthesis, gated to 50% rests over the stem floor.

    Harmonics are summed to Nyquist and no further, so nothing aliases and the
    measured centroid is a property of the waveform rather than of the
    synthesis. Same gate and same floor as :func:`sub_bass_over_noise_floor`.
    """
    n_samples = int(round(ANALYSIS_SAMPLE_RATE * FIXTURE_DURATION_SECONDS))
    times = np.arange(n_samples) / ANALYSIS_SAMPLE_RATE
    out = np.zeros(n_samples)
    harmonic = 1
    while f0 * harmonic < ANALYSIS_SAMPLE_RATE / 2:
        amplitude = harmonic_amplitude(harmonic)
        if amplitude:
            out += amplitude * np.sin(2 * np.pi * f0 * harmonic * times)
        harmonic += 1
    out = 0.7 * out / np.max(np.abs(out))

    block = int(round(SUB_BASS_BLOCK_SECONDS * ANALYSIS_SAMPLE_RATE))
    gate = np.zeros(n_samples)
    for start in range(0, n_samples, 2 * block):
        gate[start : start + block] = 1.0
    floor = np.random.default_rng(SUB_BASS_SEED).standard_normal(n_samples)
    return (out * gate + floor * STEM_NOISE_FLOOR_RMS).astype(np.float32)


#: Harmonic weights for the four Strudel waveforms `suggest_bass_sound` picks
#: between. Textbook series: triangle is odd harmonics at 1/n^2 with alternating
#: sign, square odd at 1/n, sawtooth all at 1/n.
BASS_WAVEFORMS: dict[str, Callable[[int], float]] = {
    "sine": lambda n: 1.0 if n == 1 else 0.0,
    "triangle": lambda n: (1.0 / n**2) * (1 if n % 4 == 1 else -1) if n % 2 else 0.0,
    "square": lambda n: (1.0 / n) if n % 2 else 0.0,
    "sawtooth": lambda n: 1.0 / n,
}


def test_the_centroid_tells_bass_waveforms_apart_and_a_percentile_does_not() -> None:
    """The measurement that made `centroid_energy_hz` a first moment.

    At one fundamental, the energy-weighted centroid orders the four waveforms
    by harmonic content — measured at 55 Hz: sine 54.9, triangle 57.0, square
    159.2, sawtooth 215.8. The energy **median**, which is what this descriptor
    was first implemented as, reads **64.6 Hz for all four**: more than half of
    every one of these waveforms' energy is in the fundamental, so the CDF
    crosses 0.5 in the same bin regardless of what sits above it. Same collapse
    at 35 Hz (43.1 for all four) and at 110 Hz (107.7 for three of four).

    `suggest_bass_sound` exists to choose between exactly these waveforms, so a
    descriptor that returns one number for all of them carries no information
    for the only decision it feeds. That is why the median was rejected and
    this is a centroid.
    """
    centroids = {
        name: energy_weighted_centroid_hz(*_spectrum_of(_bass_waveform(55.0, weights)))
        for name, weights in BASS_WAVEFORMS.items()
    }
    assert all(value is not None for value in centroids.values())
    assert centroids["sine"] == pytest.approx(54.9, abs=1.0)
    assert centroids["triangle"] == pytest.approx(57.0, abs=1.5)
    assert centroids["square"] == pytest.approx(159.2, abs=3.0)
    assert centroids["sawtooth"] == pytest.approx(215.8, abs=4.0)

    ordered = [centroids["sine"], centroids["triangle"], centroids["square"]]
    assert ordered == sorted(ordered)  # type: ignore[type-var]
    assert centroids["sawtooth"] > centroids["square"]  # type: ignore[operator]

    # The median, on the very same signals, cannot separate any of them.
    medians = {
        name: energy_percentile_hz(*_spectrum_of(_bass_waveform(55.0, weights)), 0.5)
        for name, weights in BASS_WAVEFORMS.items()
    }
    assert len(set(medians.values())) == 1, medians


def test_the_centroid_cannot_separate_waveforms_across_the_whole_register() -> None:
    """The honest limit of an absolute-Hz threshold, pinned so it stays visible.

    A square an octave down and a sine an octave up read the same: measured,
    square at 35 Hz is 108.3 and sine at 110 Hz is 109.8, so the two classes
    overlap and no single centroid ceiling separates them across 35-110 Hz.

    This is why `strudel_vocab.SUB_BASS_BRIGHTNESS_MAX` — pitch-independent by
    construction — carries the sine-versus-sawtooth discrimination, and
    `SUB_BASS_CENTROID_HZ_MAX` only answers "is this in the sub register".
    Both constants document the same sweep.
    """
    low_square = energy_weighted_centroid_hz(
        *_spectrum_of(_bass_waveform(35.0, BASS_WAVEFORMS["square"]))
    )
    high_sine = energy_weighted_centroid_hz(
        *_spectrum_of(_bass_waveform(110.0, BASS_WAVEFORMS["sine"]))
    )
    assert low_square is not None and high_sine is not None
    assert low_square == pytest.approx(108.3, abs=3.0)
    assert high_sine == pytest.approx(109.8, abs=2.0)
    assert low_square < high_sine  # the overlap, stated as an assertion


def test_energy_percentile_is_scale_invariant() -> None:
    """Gain cancels: it reads the spectrum's shape and nothing else.

    This is why the two backends agree exactly on it despite normalising their
    windows differently.
    """
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    rng = np.random.default_rng(19)
    magnitude = rng.random((freqs.size, 5))
    quiet = energy_percentile_hz(magnitude * 1e-6, freqs)
    loud = energy_percentile_hz(magnitude * 1e6, freqs)
    assert quiet is not None
    assert quiet == loud


def test_energy_percentile_excludes_out_of_range_bins() -> None:
    """DC, sub-20 Hz and above-20 kHz bins count for neither part of the ratio.

    Same window as `band_energy_ratios` and `brightness`, so all three agree on
    what "total energy" means.
    """
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    magnitude = np.zeros((freqs.size, 1))
    magnitude[int(np.argmin(np.abs(freqs - 1000.0)))] = 1.0
    before = energy_percentile_hz(magnitude, freqs)

    magnitude[0] = 50.0  # DC
    magnitude[freqs > 20000.0] = 50.0
    assert energy_percentile_hz(magnitude, freqs) == before


def test_energy_percentile_matches_the_band_ratios_it_shares_a_denominator_with() -> None:
    """A cross-check against an independently computed number.

    Put 60% of the energy below 250 Hz and 40% above 2 kHz. `band_energy_ratios`
    must report `low` = 0.6, and the median must therefore land on the low tone
    — the two are reading the same aggregate spectrum.
    """
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    low_bin = int(np.argmin(np.abs(freqs - 100.0)))
    high_bin = int(np.argmin(np.abs(freqs - 4000.0)))
    magnitude = np.zeros((freqs.size, 1))
    magnitude[low_bin] = np.sqrt(0.6)  # squared to power inside the helpers
    magnitude[high_bin] = np.sqrt(0.4)

    assert band_energy_ratios(magnitude, freqs)["low"] == pytest.approx(0.6)
    assert energy_percentile_hz(magnitude, freqs, 0.5) == pytest.approx(freqs[low_bin])
    assert energy_percentile_hz(magnitude, freqs, 0.85) == pytest.approx(freqs[high_bin])


def test_energy_percentile_is_none_when_undefined() -> None:
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    assert energy_percentile_hz(np.zeros((freqs.size, 3)), freqs) is None
    assert energy_percentile_hz(np.zeros((0, 0)), freqs) is None
    assert energy_percentile_hz(np.ones((10, 2)), np.linspace(0.0, 20000.0, 11)) is None
    assert energy_percentile_hz(np.ones((freqs.size, 1)), freqs, 0.0) is None
    assert energy_percentile_hz(np.ones((freqs.size, 1)), freqs, 1.5) is None
    assert energy_percentile_hz(np.ones((freqs.size, 1)), freqs, float("nan")) is None


def test_energy_percentile_accepts_a_single_frame() -> None:
    """1-D input is one frame, same as the other two shared helpers."""
    freqs = np.fft.rfftfreq(2048, d=1.0 / ANALYSIS_SAMPLE_RATE)
    magnitude = np.zeros(freqs.size)
    magnitude[int(np.argmin(np.abs(freqs - 800.0)))] = 1.0
    assert energy_percentile_hz(magnitude, freqs) == pytest.approx(800.0, abs=15.0)


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


# --------------------------------------------------------------------------- #
# pitch() — raw F0, the Wave 4 Protocol method
# --------------------------------------------------------------------------- #


def test_pitch_recovers_the_exact_synthesis_frequencies(
    backend: LibrosaBackend, bass_line_a_minor: np.ndarray
) -> None:
    """`bass_line_a_minor` is built from literal 55.0/65.40639/82.40689 Hz.

    Ground truth here is exact by construction rather than measured, so this is
    a real accuracy assertion and not a regression pin. Every voiced frame must
    land within a quarter-tone (3%) of one of the three fixture frequencies.
    """
    track = backend.pitch(bass_line_a_minor, ANALYSIS_SAMPLE_RATE)

    voiced = np.asarray(track.f0_hz)[np.asarray(track.voiced, dtype=bool)]
    assert voiced.size > 100

    for frequency in voiced:
        closest = min(BASS_LINE_FREQS_HZ, key=lambda target: abs(target - frequency))
        assert frequency == pytest.approx(closest, rel=0.03)


def test_pitch_survives_the_octave_trap(
    backend: LibrosaBackend, bass_line_octave_trap: np.ndarray
) -> None:
    """Fundamental at 0.15 under a 2nd harmonic at 1.0 — still MIDI 33/36/40.

    A tracker reporting 45/48/52 has made an octave error, not a
    different-but-defensible reading. Measured octave-error rate for
    `librosa.pyin` on this fixture: **0.0% of voiced frames**.
    """
    track = backend.pitch(bass_line_octave_trap, ANALYSIS_SAMPLE_RATE)
    line = segment_notes(track)

    assert [note.midi_note for note in line.notes] == list(BASS_LINE_MIDI)
    assert line.octave_corrections == 0, "the range constraint alone should defeat the trap"


def test_pitch_and_note_track_reproduce_the_documented_note_names(
    backend: LibrosaBackend, bass_line_a_minor: np.ndarray
) -> None:
    track = backend.pitch(bass_line_a_minor, ANALYSIS_SAMPLE_RATE)
    line = segment_notes(track)

    assert line.status == "ok"
    assert [note.note_name for note in line.notes] == list(BASS_LINE_NOTE_NAMES)


def test_pitch_absorbs_a_glide_into_the_notes_either_side(
    backend: LibrosaBackend, bass_line_with_glide: np.ndarray
) -> None:
    """Exactly two notes per pair — the ramp is not eight chromatic notes."""
    line = segment_notes(backend.pitch(bass_line_with_glide, ANALYSIS_SAMPLE_RATE))

    assert [note.note_name for note in line.notes] == list(BASS_GLIDE_NOTE_NAMES) * BASS_GLIDE_PAIRS


def test_pitch_invents_nothing_on_unpitched_low_material(
    backend: LibrosaBackend, bass_unvoiced: np.ndarray
) -> None:
    """20-200 Hz noise: loud, low, and completely unpitched."""
    line = segment_notes(backend.pitch(bass_unvoiced, ANALYSIS_SAMPLE_RATE))

    assert line.status == "unvoiced"
    assert line.notes == []
    assert line.caveats


def test_pitch_returns_an_empty_track_on_silence(
    backend: LibrosaBackend, silence: np.ndarray
) -> None:
    track = backend.pitch(silence, ANALYSIS_SAMPLE_RATE)

    assert track.f0_hz == []
    assert track.voiced == []
    assert track.frame_hop_seconds is None
    assert segment_notes(track).status == "unvoiced"


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_pitch_never_raises_on_a_nonsense_sample_rate(
    backend: LibrosaBackend, sine_a440: np.ndarray, sample_rate: int
) -> None:
    assert backend.pitch(sine_a440, sample_rate) == PitchTrack()


def test_pitch_uses_a_4096_window_on_the_512_hop_grid(
    backend: LibrosaBackend, bass_line_a_minor: np.ndarray
) -> None:
    """The documented departure from the pinned 2048 grid.

    2 / `BASS_F0_MIN_HZ` = 66.7 ms exceeds 2048 samples (46.4 ms), so a 2048
    window cannot resolve a low B and would silently report the octave above.
    The *hop* stays 512 so the F0 track shares the project's time grid — if
    either half of that drifts, the note timings stop lining up with the drum
    grid they are quantised against.
    """
    assert PYIN_FRAME_LENGTH == 4096
    assert 2.0 / BASS_F0_MIN_HZ > STFT_N_FFT / ANALYSIS_SAMPLE_RATE
    assert 2.0 / BASS_F0_MIN_HZ < PYIN_FRAME_LENGTH / ANALYSIS_SAMPLE_RATE

    track = backend.pitch(bass_line_a_minor, ANALYSIS_SAMPLE_RATE)
    assert track.frame_hop_seconds == pytest.approx(STFT_HOP_LENGTH / ANALYSIS_SAMPLE_RATE)
    assert track.method == "pyin"


def test_pitch_reports_voicing_that_separates_repeated_notes(
    backend: LibrosaBackend, bass_line_a_minor: np.ndarray
) -> None:
    """16 voiced runs, one per note — the gate `PYIN_MIN_VOICED_PROBABILITY` buys.

    pYIN's own `voiced_flag` alone gives 11 runs on this fixture: its HMM smooths
    through the 40 ms gaps and welds each pair of consecutive `a1` notes into
    one. Two notes of the same pitch have nothing but that gap to separate them,
    so the extra probability gate is what makes 16 notes reachable at all.
    """
    voiced = np.asarray(backend.pitch(bass_line_a_minor, ANALYSIS_SAMPLE_RATE).voiced, dtype=bool)

    boundaries = np.diff(voiced.astype(int))
    assert int(np.count_nonzero(boundaries == 1)) == len(BASS_LINE_MIDI)
