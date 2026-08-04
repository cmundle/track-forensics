"""Guards on the synthetic fixtures' ground truth.

Backend tests assert real numbers against these signals, so the signals
themselves need checking: shape and dtype unconditionally, and the librosa
measurements that the docstrings quote whenever librosa is installed. If a
fixture drifts, this file fails before the backend tests start lying.

The Wave 4 section at the bottom is **library-free by design**: drum
classification and note segmentation are shared numpy that must run with
neither backend installed, so their ground truth is measured here with
`np.fft.rfft` alone. Every number those tests assert was measured from the
fixture, not derived from theory — where the two disagree the docstrings say so
explicitly, because three later work packages are calibrated against them.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    BASS_GLIDE_GAP_SECONDS,
    BASS_GLIDE_HOLD_SECONDS,
    BASS_GLIDE_MIDI,
    BASS_GLIDE_PAIRS,
    BASS_GLIDE_SECONDS,
    BASS_LINE_ANCHOR_SECONDS,
    BASS_LINE_FREQS_HZ,
    BASS_LINE_GAP_SECONDS,
    BASS_LINE_MIDI,
    BASS_LINE_NOTE_NAMES,
    BASS_LINE_NOTE_SECONDS,
    CLICK_TRACK_BPM,
    CLICK_TRACK_ONSET_COUNT,
    CLICK_TRACK_ONSET_DENSITY,
    DRUM_PATTERN_AMBIGUOUS_CLICK_STEPS,
    DRUM_PATTERN_AMBIGUOUS_KICK_STEPS,
    DRUM_PATTERN_ANCHOR_SECONDS,
    DRUM_PATTERN_CYCLE_SECONDS,
    DRUM_PATTERN_CYCLES,
    DRUM_PATTERN_HAT_STEPS,
    DRUM_PATTERN_KICK_ONLY_STEPS,
    DRUM_PATTERN_KICK_STEPS,
    DRUM_PATTERN_SNARE_STEPS,
    DRUM_PATTERN_STEP_SECONDS,
    DRUM_PATTERN_STEPS_PER_CYCLE,
    FIXTURE_DURATION_SECONDS,
    PATTERN_FIXTURE_DURATION_SECONDS,
    SINE_FREQUENCY_HZ,
    SWUNG_CLICK_BPM,
    SWUNG_LONG_IOI_SECONDS,
    SWUNG_SHORT_IOI_SECONDS,
    _click,
    _hat_closed,
    _hat_open,
    _kick,
    _snare,
)

from audio_pipeline import ANALYSIS_SAMPLE_RATE

EXPECTED_SAMPLES = int(ANALYSIS_SAMPLE_RATE * FIXTURE_DURATION_SECONDS)
EXPECTED_PATTERN_SAMPLES = int(ANALYSIS_SAMPLE_RATE * PATTERN_FIXTURE_DURATION_SECONDS)

#: Drum detection bands, in Hz. Deliberately not `BAND_EDGES_HZ`: a snare's body
#: sits inside that scheme's `low` band, so it cannot separate kick from snare.
#: `drum_elements.py` owns the definition these numbers mirror; they are
#: restated here because this file must stay library-free and must not depend on
#: the module it is guarding the fixtures *for*.
_DETECTION_BANDS: dict[str, tuple[float, float]] = {
    "kick": (20.0, 150.0),
    "body": (150.0, 500.0),
    "noise": (1000.0, 6000.0),
    "air": (6000.0, 16000.0),
}
_FLATNESS_BAND_HZ = (20.0, 16000.0)
#: Floor for the flatness geometric mean, relative to the peak bin. Without one,
#: the deep nulls of a band-limited signal decide the answer on their own.
_FLATNESS_FLOOR = 1e-12


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


# --- Wave 4 fixtures: library-free ground truth ------------------------------
#
# Nothing below imports librosa or essentia. These fixtures are the ground truth
# for drum classification and pitch tracking, and both of those are meant to be
# checkable with neither backend installed — so are their guards.


def _band_shares(audio: np.ndarray) -> dict[str, float]:
    """Share of in-band energy per `_DETECTION_BANDS` band.

    Mirrors `band_energy_ratios`' conventions exactly: bins assigned by centre
    frequency, half-open intervals `[low, high)` with the top band closed, energy
    as `magnitude ** 2`, denominator the union of the bands.
    """
    power = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
    frequencies = np.fft.rfftfreq(audio.size, 1.0 / ANALYSIS_SAMPLE_RATE)
    top = max(high for _, high in _DETECTION_BANDS.values())
    per_band: dict[str, float] = {}
    for name, (low, high) in _DETECTION_BANDS.items():
        if high >= top:
            in_band = (frequencies >= low) & (frequencies <= high)
        else:
            in_band = (frequencies >= low) & (frequencies < high)
        per_band[name] = float(power[in_band].sum())
    total = sum(per_band.values())
    assert total > 0.0
    return {name: value / total for name, value in per_band.items()}


def _flatness(audio: np.ndarray) -> float:
    """Geometric/arithmetic mean of the in-band power spectrum, floored.

    Scale-invariant: the floor sits `_FLATNESS_FLOOR` (120 dB) below the peak
    bin rather than at an absolute level, so a quiet hat and a loud kick are
    measured on the same scale.
    """
    power = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
    frequencies = np.fft.rfftfreq(audio.size, 1.0 / ANALYSIS_SAMPLE_RATE)
    low, high = _FLATNESS_BAND_HZ
    band = power[(frequencies >= low) & (frequencies <= high)]
    band = np.maximum(band, band.max() * _FLATNESS_FLOOR)
    return float(np.exp(np.mean(np.log(band))) / np.mean(band))


def _decay_to_minus_20db_ms(audio: np.ndarray) -> float:
    """Milliseconds from the envelope peak to 10% of it, on a 1 ms RMS envelope."""
    window = int(round(0.001 * ANALYSIS_SAMPLE_RATE))
    usable = (audio.size // window) * window
    envelope = np.sqrt(
        (audio[:usable].astype(np.float64) ** 2).reshape(-1, window).mean(axis=1)
    )
    after_peak = envelope[int(np.argmax(envelope)) :]
    below = np.flatnonzero(after_peak < envelope.max() * 0.1)
    return float(below[0]) if below.size else float(after_peak.size)


def _step_time(cycle: int, step: int) -> float:
    return (
        DRUM_PATTERN_ANCHOR_SECONDS
        + cycle * DRUM_PATTERN_CYCLE_SECONDS
        + step * DRUM_PATTERN_STEP_SECONDS
    )


def _amplitude_jump(audio: np.ndarray, time_s: float, window_s: float = 0.012) -> float:
    """Peak amplitude just after `time_s` minus peak amplitude just before it.

    A hit starting at `time_s` produces a large positive jump; anywhere else the
    signal is decaying, so the jump is zero or negative. Deliberately not an
    onset detector — a threshold-and-gap reading of the waveform merges a hat
    into the kick tail it lands on, and a short-window flux reading oscillates
    on the kick's own 60 Hz period. Measured separation on every pattern fixture
    is at least 0.138 for a hit against at most 0.000 for an empty step.
    """
    window = int(round(window_s * ANALYSIS_SAMPLE_RATE))
    index = int(round(time_s * ANALYSIS_SAMPLE_RATE))
    before = float(np.max(np.abs(audio[max(0, index - window) : index]))) if index > 0 else 0.0
    after = float(np.max(np.abs(audio[index : index + window])))
    return after - before


def _hit_steps(audio: np.ndarray) -> set[int]:
    """Grid steps at which the buffer actually starts a hit, folded onto a cycle."""
    found: set[int] = set()
    for cycle in range(DRUM_PATTERN_CYCLES):
        for step in range(DRUM_PATTERN_STEPS_PER_CYCLE):
            if _amplitude_jump(audio, _step_time(cycle, step)) > 0.05:
                found.add(step)
    return found


def _hit_count(audio: np.ndarray) -> int:
    """Total grid positions across all cycles that start a hit."""
    return sum(
        1
        for cycle in range(DRUM_PATTERN_CYCLES)
        for step in range(DRUM_PATTERN_STEPS_PER_CYCLE)
        if _amplitude_jump(audio, _step_time(cycle, step)) > 0.05
    )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "drum_pattern_120bpm",
        "drum_pattern_kick_only",
        "drum_pattern_open_hats",
        "drum_pattern_ambiguous",
        "bass_line_a_minor",
        "bass_line_octave_trap",
        "bass_line_with_glide",
        "bass_unvoiced",
    ],
)
def test_pattern_fixtures_are_float32_at_full_rate(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Same guard as the Wave 0 fixtures, at the longer pattern buffer length."""
    audio = request.getfixturevalue(fixture_name)
    assert audio.dtype == np.float32
    assert audio.shape == (EXPECTED_PATTERN_SAMPLES,)
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 1.0


@pytest.mark.parametrize(
    "fixture_name",
    ["drum_pattern_120bpm", "drum_pattern_open_hats", "drum_pattern_ambiguous"],
)
def test_pattern_fixtures_do_not_clip(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Coincident hits sum, and a clipped sum would invalidate every band share.

    `_hit_train` clips as a backstop; this pins that the backstop never fires,
    so the one-shot measurements below still describe what is in the buffer.
    """
    audio = request.getfixturevalue(fixture_name)
    assert np.max(np.abs(audio)) < 0.99


def test_kick_one_shot_is_low_and_tonal() -> None:
    shares = _band_shares(_kick())
    assert shares["kick"] == pytest.approx(0.9941, abs=0.005)
    assert shares["kick"] > 0.90
    assert shares["noise"] < 0.01
    assert shares["air"] < 0.01
    assert _flatness(_kick()) < 0.15
    assert _decay_to_minus_20db_ms(_kick()) == pytest.approx(37.0, abs=3.0)


def test_snare_one_shot_is_body_plus_noise() -> None:
    """Note the measured noise share is 0.394, NOT the > 0.45 theory suggests.

    See `_snare`'s docstring: amplitude is not energy. `body_ratio` is what
    actually identifies a snare here, which matters because `_click`'s noise
    share (0.564) is higher than the snare's.
    """
    shares = _band_shares(_snare())
    assert shares["kick"] == pytest.approx(0.0118, abs=0.005)
    assert shares["kick"] < 0.15
    assert shares["body"] == pytest.approx(0.5935, abs=0.02)
    assert shares["noise"] == pytest.approx(0.3937, abs=0.02)
    assert shares["body"] + shares["noise"] > 0.95
    assert _decay_to_minus_20db_ms(_snare()) == pytest.approx(141.0, abs=10.0)


@pytest.mark.parametrize("one_shot", [_hat_closed, _hat_open])
def test_hat_one_shots_are_almost_entirely_air(one_shot: object) -> None:
    shares = _band_shares(one_shot())  # type: ignore[operator]
    assert shares["air"] > 0.80
    assert shares["air"] > 0.99
    assert shares["kick"] < 0.001


def test_hats_differ_only_in_decay() -> None:
    """The `hh` versus `oh` discrimination, in its purest form."""
    closed_shares = _band_shares(_hat_closed())
    open_shares = _band_shares(_hat_open())
    assert closed_shares["air"] == pytest.approx(open_shares["air"], abs=0.005)

    closed_decay = _decay_to_minus_20db_ms(_hat_closed())
    open_decay = _decay_to_minus_20db_ms(_hat_open())
    assert closed_decay == pytest.approx(34.0, abs=4.0)
    assert open_decay == pytest.approx(277.0, abs=15.0)
    assert open_decay > closed_decay * 5


def test_flatness_orders_the_one_shots_tonal_to_noisy() -> None:
    """Flatness separates these by orders of magnitude, not on an absolute scale.

    Measured: kick 1e-6, snare 1.5e-3, hat_open 2.8e-2, hat_closed 4.1e-2,
    click 5.2e-1. A classifier wanting "flat means noisy" must compare these
    against each other, not against a fixed 0.6.
    """
    values = {
        "kick": _flatness(_kick()),
        "snare": _flatness(_snare()),
        "hat_open": _flatness(_hat_open()),
        "hat_closed": _flatness(_hat_closed()),
        "click": _flatness(_click()),
    }
    assert list(values) == sorted(values, key=lambda name: values[name])
    assert values["kick"] < 1e-4
    assert values["snare"] == pytest.approx(0.0015, abs=0.0008)
    assert values["click"] > 0.4


def test_click_sits_between_every_drum_class() -> None:
    """Why `drum_pattern_ambiguous` has an honest answer of `unclassified`."""
    click = _band_shares(_click())
    snare = _band_shares(_snare())
    hat = _band_shares(_hat_closed())

    assert click["kick"] < 0.001  # not a kick: no low end at all
    assert click["noise"] > snare["noise"]  # noisier than the snare...
    assert click["body"] < 0.1  # ...but with no shell tone
    assert click["air"] < hat["air"] / 2  # and nowhere near a hat's air
    assert _decay_to_minus_20db_ms(_click()) < 5.0  # and no tail whatsoever


def test_drum_pattern_120bpm_has_the_documented_steps(
    drum_pattern_120bpm: np.ndarray,
) -> None:
    expected = set(DRUM_PATTERN_KICK_STEPS) | set(DRUM_PATTERN_SNARE_STEPS) | set(
        DRUM_PATTERN_HAT_STEPS
    )
    assert _hit_steps(drum_pattern_120bpm) == expected
    assert _hit_count(drum_pattern_120bpm) == len(expected) * DRUM_PATTERN_CYCLES == 32


def test_drum_pattern_120bpm_really_has_coincident_hits() -> None:
    """The property the whole per-band design exists for. Guard it explicitly.

    48 hits at 32 instants: kick and hat coincide on steps 0 and 8, snare and
    hat on 4 and 12. If someone "simplifies" the fixture by moving the hats off
    the kick and snare, this fails rather than quietly making a broken design
    look correct.
    """
    kick = set(DRUM_PATTERN_KICK_STEPS)
    snare = set(DRUM_PATTERN_SNARE_STEPS)
    hat = set(DRUM_PATTERN_HAT_STEPS)
    assert kick <= hat
    assert snare <= hat
    assert not kick & snare

    total_hits = (len(kick) + len(snare) + len(hat)) * DRUM_PATTERN_CYCLES
    distinct = len(kick | snare | hat) * DRUM_PATTERN_CYCLES
    assert total_hits == 48
    assert distinct == 32


def test_drum_pattern_kick_only_contains_nothing_else(
    drum_pattern_kick_only: np.ndarray,
) -> None:
    assert _hit_steps(drum_pattern_kick_only) == set(DRUM_PATTERN_KICK_ONLY_STEPS)
    assert _hit_count(drum_pattern_kick_only) == 16

    shares = _band_shares(drum_pattern_kick_only)
    assert shares["kick"] > 0.99
    # There is genuinely no hat and no snare here to find.
    assert shares["air"] < 1e-4
    assert shares["noise"] < 1e-4


def test_drum_pattern_open_hats_matches_the_closed_hat_layout(
    drum_pattern_open_hats: np.ndarray, drum_pattern_120bpm: np.ndarray
) -> None:
    assert _hit_steps(drum_pattern_open_hats) == _hit_steps(drum_pattern_120bpm)
    assert _hit_count(drum_pattern_open_hats) == 32
    # The long tails put far more air in the buffer for the same step layout.
    assert _band_shares(drum_pattern_open_hats)["air"] > _band_shares(drum_pattern_120bpm)["air"]


def test_drum_pattern_ambiguous_keeps_the_clicks_off_the_kicks(
    drum_pattern_ambiguous: np.ndarray,
) -> None:
    kick_steps = set(DRUM_PATTERN_AMBIGUOUS_KICK_STEPS)
    click_steps = set(DRUM_PATTERN_AMBIGUOUS_CLICK_STEPS)
    assert not kick_steps & click_steps  # nothing here is a sum of two classes

    assert _hit_steps(drum_pattern_ambiguous) == kick_steps | click_steps
    assert _hit_count(drum_pattern_ambiguous) == 24
    assert len(click_steps) * DRUM_PATTERN_CYCLES == 16


def _note_window(audio: np.ndarray, index: int) -> np.ndarray:
    start = BASS_LINE_ANCHOR_SECONDS + index * BASS_LINE_NOTE_SECONDS
    first = int(round(start * ANALYSIS_SAMPLE_RATE))
    length = int(round((BASS_LINE_NOTE_SECONDS - BASS_LINE_GAP_SECONDS) * ANALYSIS_SAMPLE_RATE))
    return audio[first : first + length].astype(np.float64)


def _peak_bin_hz(segment: np.ndarray) -> tuple[float, float]:
    """Frequency of the loudest bin, and the bin width, both in Hz."""
    spectrum = np.abs(np.fft.rfft(segment))
    frequencies = np.fft.rfftfreq(segment.size, 1.0 / ANALYSIS_SAMPLE_RATE)
    return float(frequencies[int(np.argmax(spectrum))]), float(frequencies[1])


def test_bass_line_first_note_really_is_55_hz(bass_line_a_minor: np.ndarray) -> None:
    peak_hz, bin_hz = _peak_bin_hz(_note_window(bass_line_a_minor, 0))
    assert peak_hz == pytest.approx(BASS_LINE_FREQS_HZ[0], abs=bin_hz)
    assert BASS_LINE_FREQS_HZ[0] == 55.0  # A1, by definition of A4 = 440 Hz


def test_every_bass_note_peaks_at_its_own_fundamental(bass_line_a_minor: np.ndarray) -> None:
    """Ground truth exact by construction: the constants *are* the synthesis inputs."""
    assert len(BASS_LINE_MIDI) == len(BASS_LINE_NOTE_NAMES) == 16
    for index in range(len(BASS_LINE_MIDI)):
        expected = BASS_LINE_FREQS_HZ[index % len(BASS_LINE_FREQS_HZ)]
        peak_hz, bin_hz = _peak_bin_hz(_note_window(bass_line_a_minor, index))
        assert peak_hz == pytest.approx(expected, abs=bin_hz), f"note {index}"


def test_bass_line_midi_and_frequencies_agree() -> None:
    """The two exported representations must describe the same four pitches."""
    for midi, frequency in zip(BASS_LINE_MIDI[:4], BASS_LINE_FREQS_HZ, strict=True):
        assert 440.0 * 2 ** ((midi - 69) / 12) == pytest.approx(frequency, abs=1e-4)
    assert BASS_LINE_MIDI[:4] == (33, 33, 36, 40)
    assert BASS_LINE_NOTE_NAMES[:4] == ("a1", "a1", "c2", "e2")


def test_bass_notes_are_separated_by_real_silence(bass_line_a_minor: np.ndarray) -> None:
    """The 40 ms tail is digital silence, so voicing has an unambiguous gap."""
    for index in range(len(BASS_LINE_MIDI) - 1):
        start = BASS_LINE_ANCHOR_SECONDS + index * BASS_LINE_NOTE_SECONDS
        gap_start = start + BASS_LINE_NOTE_SECONDS - BASS_LINE_GAP_SECONDS
        first = int(round(gap_start * ANALYSIS_SAMPLE_RATE))
        last = int(round((gap_start + BASS_LINE_GAP_SECONDS) * ANALYSIS_SAMPLE_RATE))
        assert not np.any(bass_line_a_minor[first:last])


def test_octave_trap_really_traps(bass_line_octave_trap: np.ndarray) -> None:
    """The loudest partial is the 2nd harmonic, but the answer is still MIDI 33.

    A tracker that peak-picks reports 45/45/48/52 here. That is the error this
    fixture exists to catch, so pin that the trap is actually set.
    """
    for index in range(4):
        expected = BASS_LINE_FREQS_HZ[index]
        peak_hz, bin_hz = _peak_bin_hz(_note_window(bass_line_octave_trap, index))
        assert peak_hz == pytest.approx(2 * expected, abs=bin_hz), f"note {index}"
    # ...and the correct answer is unchanged.
    assert BASS_LINE_MIDI[:4] == (33, 33, 36, 40)


def test_octave_trap_still_contains_its_fundamental(
    bass_line_octave_trap: np.ndarray,
) -> None:
    """Buried at 0.15, not removed — an octave guard has something to find."""
    segment = _note_window(bass_line_octave_trap, 0)
    spectrum = np.abs(np.fft.rfft(segment))
    frequencies = np.fft.rfftfreq(segment.size, 1.0 / ANALYSIS_SAMPLE_RATE)
    fundamental = spectrum[int(np.argmin(np.abs(frequencies - BASS_LINE_FREQS_HZ[0])))]
    assert 0.1 < float(fundamental / spectrum.max()) < 0.25


def test_glide_fixture_holds_two_pitches_per_pair(bass_line_with_glide: np.ndarray) -> None:
    """Each pair is two held notes with a short ramp between, not a chromatic run."""
    pair_seconds = 2 * BASS_GLIDE_HOLD_SECONDS + BASS_GLIDE_SECONDS
    period = pair_seconds + BASS_GLIDE_GAP_SECONDS
    for pair in range(BASS_GLIDE_PAIRS):
        start = BASS_LINE_ANCHOR_SECONDS + pair * period
        for offset, expected in (
            (0.05, BASS_LINE_FREQS_HZ[0]),  # inside the A1 hold
            (0.55, BASS_LINE_FREQS_HZ[3]),  # inside the E2 hold
        ):
            first = int(round((start + offset) * ANALYSIS_SAMPLE_RATE))
            last = first + int(round(0.3 * ANALYSIS_SAMPLE_RATE))
            peak_hz, bin_hz = _peak_bin_hz(bass_line_with_glide[first:last].astype(np.float64))
            assert peak_hz == pytest.approx(expected, abs=bin_hz), f"pair {pair}"


def test_glide_is_short_relative_to_the_notes_it_joins() -> None:
    """Why the merge rule can absorb it: 100 ms of ramp against 400 ms of note."""
    assert BASS_GLIDE_SECONDS * 4 == BASS_GLIDE_HOLD_SECONDS
    semitones = BASS_GLIDE_MIDI[1] - BASS_GLIDE_MIDI[0]
    # Each intermediate semitone lasts ~14 ms, far below any sane minimum note.
    assert BASS_GLIDE_SECONDS / semitones < 0.02


def test_bass_unvoiced_is_loud_low_and_pitchless(bass_unvoiced: np.ndarray) -> None:
    shares = _band_shares(bass_unvoiced)
    assert shares["kick"] == pytest.approx(0.7216, abs=0.02)
    assert shares["body"] == pytest.approx(0.2784, abs=0.02)
    assert shares["noise"] < 1e-6
    assert shares["air"] < 1e-6
    assert float(np.max(np.abs(bass_unvoiced))) == pytest.approx(0.5, abs=1e-6)

    # No harmonic series: the loudest bin of one half does not predict the other.
    half = bass_unvoiced.size // 2
    first_peak, _ = _peak_bin_hz(bass_unvoiced[:half].astype(np.float64))
    second_peak, _ = _peak_bin_hz(bass_unvoiced[half:].astype(np.float64))
    assert first_peak != second_peak


def test_wave4_fixtures_are_reproducible(
    drum_pattern_120bpm: np.ndarray, bass_unvoiced: np.ndarray
) -> None:
    """Fixed seeds, so a Wave 4 feature test asserting a number stays true tomorrow."""
    assert float(np.abs(_snare()).sum()) == pytest.approx(490.28137, abs=1e-3)
    assert float(np.abs(_hat_closed()).sum()) == pytest.approx(35.62051, abs=1e-3)
    assert float(drum_pattern_120bpm.sum()) == pytest.approx(419.21631, abs=1e-3)
    assert float(bass_unvoiced[: ANALYSIS_SAMPLE_RATE].sum()) == pytest.approx(
        -23.87624, abs=1e-3
    )
