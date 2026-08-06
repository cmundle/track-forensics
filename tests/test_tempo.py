"""Tests for `tempo.py` — BPM refinement to 0.01, and an honest downbeat.

Two kinds of material here, and the split is deliberate.

**Synthetic click trains**, built from numpy at `ANALYSIS_SAMPLE_RATE`, carry a
tempo that is exactly known, so they pin absolute accuracy: 120, 132 and 128.5
BPM must come back inside 0.01 BPM. They also let a ritardando be dialled in to
a stated percentage, which no real recording does.

**The committed Madonna flux fixture** carries what synthetic material cannot:
147 bars of real arrangement with sections where elements drop out, a real
mastering chain, and a tempo that is 132.000 by construction because the record
was sequenced. No audio is committed — `tests/fixtures/real/PROVENANCE.md`
records what the arrays are and why they are an irreversible reduction.

The bug this module exists to fix is worth restating, because two of the tests
below are its regression cover. The pipeline reported 131.855 BPM for a
132.000 track. Four hundredths of a BPM accumulate 82 ms over 147 bars, which
is 0.72 of a sixteenth step, and a textbook four-on-the-floor grid was rejected
as `no_grid`. Nothing here may be relaxed to make a fit pass; the tolerance is
the requirement.
"""

from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from audio_pipeline import ANALYSIS_SAMPLE_RATE, tempo
from audio_pipeline.tempo import (
    BEAT_MULTIPLES,
    BPM_TOLERANCE,
    DOWNBEAT_TIE_FRACTION,
    HOP_SECONDS,
    MIN_AUTOCORRELATION_R,
    MULTIPLE_AGREEMENT_BPM,
    STABILITY_HIGH_BPM,
    THRESHOLDS,
    DownbeatFit,
    TempoFit,
    find_downbeat,
    find_downbeat_from_envelopes,
    refine_bpm,
    refine_bpm_from_envelope,
    stability_from_envelope,
    within_bpm_tolerance,
)

#: The accuracy this module exists to deliver, in BPM. Not a tolerance chosen
#: for convenience: at 132 BPM over a four-minute track, 0.01 BPM is 21 ms of
#: accumulated drift, a fifth of a sixteenth step, which a grid absorbs. The
#: 0.04 BPM error calibration found was 0.72 of a step, which it does not.
BPM_TOLERANCE_BPM = 0.01

#: The true tempo of the fixture track, exactly. Sequenced, not played.
MADONNA_BPM = 132.0

#: `V2-PLAN.md`'s finding F1 re-folded the drums stem at a downbeat of 0.228 s.
#: This module lands on the STFT frame containing it; one frame is 11.6 ms.
MADONNA_DOWNBEAT_SECONDS = 0.228

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "real"
DRUM_ENVELOPES = FIXTURE_DIR / "madonna__drums_band_envelopes.npz"


# --------------------------------------------------------------------------- #
# Synthetic material
# --------------------------------------------------------------------------- #


def _kick(duration_seconds: float = 0.18) -> npt.NDArray[np.float64]:
    """A 52 Hz thump with a 45 ms decay — energy squarely inside 20-110 Hz."""
    length = int(duration_seconds * ANALYSIS_SAMPLE_RATE)
    time = np.arange(length) / ANALYSIS_SAMPLE_RATE
    return np.sin(2 * np.pi * 52.0 * time) * np.exp(-time / 0.045)


def _bright(duration_seconds: float, decay_seconds: float, seed: int) -> npt.NDArray[np.float64]:
    """Noise burst differenced once, so its energy sits in the 6-16 kHz band.

    A first difference is a crude +6 dB/octave tilt. It is enough to put the
    burst where `BRIGHT_BAND_HZ` reads, which is all the bar-phase objective
    needs, and it keeps the fixture generator to two lines of numpy.
    """
    length = int(duration_seconds * ANALYSIS_SAMPLE_RATE)
    time = np.arange(length) / ANALYSIS_SAMPLE_RATE
    burst = np.random.default_rng(seed).normal(0.0, 1.0, length) * np.exp(-time / decay_seconds)
    return np.diff(np.concatenate(([0.0], burst)))


def _render(
    events: list[tuple[float, npt.NDArray[np.float64], float]], seconds: float
) -> npt.NDArray[np.float64]:
    """Sum `(start_seconds, waveform, gain)` events into a silent buffer."""
    out = np.zeros(int(seconds * ANALYSIS_SAMPLE_RATE), dtype=np.float64)
    for start_seconds, waveform, gain in events:
        start = int(round(start_seconds * ANALYSIS_SAMPLE_RATE))
        if start < 0 or start >= out.size:
            continue
        span = min(waveform.size, out.size - start)
        out[start : start + span] += waveform[:span] * gain
    return out


def click_train(
    bpm: float, seconds: float, *, ritardando: float = 0.0
) -> npt.NDArray[np.float64]:
    """Kicks on every beat at exactly `bpm`.

    `ritardando` is the **total** fractional slowdown across the source: 0.01
    means the beat period is 1% longer at the end than at the start, growing
    linearly in between. That is a tempo curve no autocorrelation can call a
    single number, which is the point of the stability test.
    """
    kick = _kick()
    events: list[tuple[float, npt.NDArray[np.float64], float]] = []
    beat = 60.0 / bpm
    position = 0.0
    while position < seconds - 0.25:
        events.append((position, kick, 1.0))
        position += beat * (1.0 + ritardando * position / seconds)
    return _render(events, seconds)


def four_on_the_floor(
    bpm: float, bars: int, *, downbeat_seconds: float = 0.25, downbeat_gain: float = 1.0
) -> npt.NDArray[np.float64]:
    """Kick on every beat, clap on 2 and 4, hat on every offbeat eighth.

    The pattern that makes bar phase ambiguous, and the reason `find_downbeat`
    could not be written as the plan specified it. With `downbeat_gain` at 1.0
    beats 1 and 3 are indistinguishable by any bar-level measurement. Raising
    it accents beat one, which is the conventional asymmetry the objective can
    actually resolve.
    """
    beat = 60.0 / bpm
    kick, clap, hat = _kick(), _bright(0.12, 0.020, seed=11), _bright(0.04, 0.006, seed=12)
    events: list[tuple[float, npt.NDArray[np.float64], float]] = []
    for bar in range(bars):
        for index in range(4):
            at = downbeat_seconds + (4 * bar + index) * beat
            events.append((at, kick, downbeat_gain if index == 0 else 1.0))
            events.append((at + 0.5 * beat, hat, 0.5))
            if index in (1, 3):
                events.append((at, clap, 0.8))
    return _render(events, downbeat_seconds + bars * 4 * beat + 1.0)


def white_noise(seconds: float, seed: int = 0) -> npt.NDArray[np.float64]:
    """Stationary noise: energy everywhere, periodicity nowhere."""
    return np.random.default_rng(seed).normal(
        0.0, 0.2, int(seconds * ANALYSIS_SAMPLE_RATE)
    )


@pytest.fixture(scope="module")
def madonna() -> dict[str, npt.NDArray[np.float64]]:
    """The committed drums band envelopes, or a skip if they are absent."""
    if not DRUM_ENVELOPES.exists():
        pytest.skip(f"real-material fixture missing: {DRUM_ENVELOPES}")
    with np.load(DRUM_ENVELOPES) as data:
        return {key: np.asarray(data[key]) for key in ("band_tempo", "band_air", "hop_seconds")}


# --------------------------------------------------------------------------- #
# Accuracy: the whole point of the module
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("true_bpm", [120.0, 132.0, 128.5])
def test_click_train_recovers_bpm_to_a_hundredth(true_bpm: float) -> None:
    """A known tempo must come back inside 0.01 BPM from raw audio.

    The coarse estimate is deliberately handed in 0.15 BPM low, which is a
    little worse than the +/-0.2 BPM a real beat tracker delivers, so the test
    exercises refinement rather than passing through a value that was already
    right.

    128.5 is in the set on purpose: a tempo whose beat period is not a round
    number of STFT frames is where a peak interpolator's bias shows up.
    """
    samples = click_train(true_bpm, 70.0)

    fit = refine_bpm(samples, ANALYSIS_SAMPLE_RATE, true_bpm - 0.15)

    assert fit.status == "refined", fit.caveats
    assert fit.bpm is not None
    assert abs(fit.bpm - true_bpm) <= BPM_TOLERANCE_BPM, (
        f"{fit.bpm} is {abs(fit.bpm - true_bpm):.4f} BPM from {true_bpm}"
    )
    assert fit.period_seconds is not None
    assert fit.period_seconds == pytest.approx(60.0 / fit.bpm)


def test_madonna_fixture_recovers_132_bpm(madonna: dict[str, npt.NDArray[np.float64]]) -> None:
    """The regression for finding F1, on the record that exposed it.

    The pipeline reported 131.855 for this track and `drum_elements` built its
    cycle from 132.040. Either error rejects the grid. 132.000 +/- 0.01 is the
    contract.
    """
    fit = refine_bpm_from_envelope(
        madonna["band_tempo"], float(madonna["hop_seconds"]), 131.854843
    )

    assert fit.status == "refined", fit.caveats
    assert fit.bpm is not None
    assert abs(fit.bpm - MADONNA_BPM) <= BPM_TOLERANCE_BPM, (
        f"{fit.bpm} is {abs(fit.bpm - MADONNA_BPM):.4f} BPM from {MADONNA_BPM}"
    )
    assert fit.confidence_label == "high"
    assert fit.coarse_bpm == 131.854843, "the coarse estimate must survive for comparison"


def test_madonna_multiples_agree_and_are_both_accepted(
    madonna: dict[str, npt.NDArray[np.float64]],
) -> None:
    """Two independent readings of the same tempo, and they must corroborate.

    This is the cross-check that turns a single autocorrelation peak into
    evidence. Both multiples measured this fixture at 131.998 and 132.001.
    """
    fit = refine_bpm_from_envelope(
        madonna["band_tempo"], float(madonna["hop_seconds"]), 131.854843
    )

    assert [multiple.beats for multiple in fit.multiples] == list(BEAT_MULTIPLES)
    assert all(multiple.accepted for multiple in fit.multiples), fit.multiples
    values = [multiple.bpm for multiple in fit.multiples]
    assert all(value is not None for value in values)
    spread = max(v for v in values if v is not None) - min(v for v in values if v is not None)
    assert spread <= MULTIPLE_AGREEMENT_BPM
    assert all(multiple.r > 0.5 for multiple in fit.multiples), "real material correlates strongly"


def test_madonna_fixture_reports_high_stability(
    madonna: dict[str, npt.NDArray[np.float64]],
) -> None:
    """This record is machine-timed, and the halves must say so.

    Measured: 131.9986 and 132.0017, a difference of 0.0031 BPM. A tool that
    called that `medium` would be useless for telling a sequenced track from a
    played one.
    """
    fit = refine_bpm_from_envelope(
        madonna["band_tempo"], float(madonna["hop_seconds"]), 131.854843
    )

    assert fit.stability.label == "high"
    assert fit.stability.delta_bpm is not None
    assert fit.stability.delta_bpm < STABILITY_HIGH_BPM
    for half in (fit.stability.first_half_bpm, fit.stability.second_half_bpm):
        assert half is not None
        assert abs(half - MADONNA_BPM) <= 0.05


def test_samples_and_envelope_entry_points_agree() -> None:
    """`refine_bpm` must be `refine_bpm_from_envelope` plus one STFT, nothing more.

    The split exists so the committed fixtures are usable at all. If the two
    paths could diverge, a fixture-based assertion would stop being evidence
    about what the pipeline does.
    """
    samples = click_train(132.0, 45.0)
    magnitude, freqs = tempo._stft_magnitude(samples, ANALYSIS_SAMPLE_RATE)
    envelope = tempo._band_envelope(magnitude, freqs, *tempo.TEMPO_BAND_HZ)

    from_samples = refine_bpm(samples, ANALYSIS_SAMPLE_RATE, 131.85)
    from_envelope = refine_bpm_from_envelope(envelope, HOP_SECONDS, 131.85)

    assert from_samples.bpm == from_envelope.bpm
    assert from_samples.status == from_envelope.status


# --------------------------------------------------------------------------- #
# Refusing to answer
# --------------------------------------------------------------------------- #


def test_pure_noise_returns_the_coarse_estimate_with_low_confidence() -> None:
    """Noise has no tempo, and inventing one is the failure mode that matters.

    A confident wrong BPM propagates into a grid, a bar count and an
    arrangement. Measured autocorrelation r on white noise is 0.016-0.038
    across seeds, an order of magnitude under `MIN_AUTOCORRELATION_R`, and the
    two multiples land 1.5-3.6 BPM apart. Either guard alone catches it.
    """
    fit = refine_bpm(white_noise(60.0), ANALYSIS_SAMPLE_RATE, 120.0)

    assert fit.status == "coarse"
    assert fit.bpm == 120.0, "the coarse estimate is returned unchanged, not adjusted"
    assert fit.confidence_label == "low"
    assert fit.confidence < MIN_AUTOCORRELATION_R
    assert fit.caveats, "refusing to refine must say why"


def test_a_sustained_tone_has_no_tempo_to_find() -> None:
    """A 55 Hz sine fills the tempo band completely and fluxes almost nowhere.

    The band-activity case: plenty of energy, no transients. Measured r 0.0003.
    """
    time = np.arange(int(40.0 * ANALYSIS_SAMPLE_RATE)) / ANALYSIS_SAMPLE_RATE
    fit = refine_bpm(np.sin(2 * np.pi * 55.0 * time), ANALYSIS_SAMPLE_RATE, 120.0)

    assert fit.status == "coarse"
    assert fit.bpm == 120.0
    assert fit.confidence_label == "low"


def test_no_coarse_estimate_means_nothing_to_refine() -> None:
    """This module locates a peak *near* a starting point; it cannot invent one.

    Saying so beats reporting a plausible number from an unconstrained search.
    """
    fit = refine_bpm(click_train(132.0, 40.0), ANALYSIS_SAMPLE_RATE, None)

    assert fit.status == "unavailable"
    assert fit.bpm is None
    assert fit.period_seconds is None
    assert fit.caveats


@pytest.mark.parametrize("coarse", [0.0, -132.0, float("nan"), float("inf")])
def test_a_nonsense_coarse_estimate_is_unavailable_not_a_crash(coarse: float) -> None:
    """Zero, negative, NaN and infinity all arrive from somewhere eventually."""
    fit = refine_bpm(click_train(132.0, 20.0), ANALYSIS_SAMPLE_RATE, coarse)

    assert fit.status == "unavailable"
    assert fit.bpm is None


def test_a_source_too_short_for_a_multiple_says_so() -> None:
    """32 beats at 132 BPM is 14.5 s, and the lag must fit in half the record.

    A 10 s source cannot measure either multiple. It must report that rather
    than correlating a pattern against one repetition of itself and calling the
    result confident.
    """
    fit = refine_bpm(click_train(132.0, 10.0), ANALYSIS_SAMPLE_RATE, 131.9)

    assert fit.status == "coarse"
    assert {multiple.reason for multiple in fit.multiples} == {"too_short"}


def test_nan_and_infinity_in_the_envelope_never_raise() -> None:
    """A poisoned envelope must degrade, not take an analysis down with it."""
    envelope = np.full(4096, 1.0)
    envelope[100] = np.nan
    envelope[200] = np.inf
    envelope[300] = -np.inf

    fit = refine_bpm_from_envelope(envelope, HOP_SECONDS, 132.0)

    assert isinstance(fit, TempoFit)
    assert fit.status in {"refined", "coarse"}


@pytest.mark.parametrize("envelope", [np.zeros(0), np.zeros(3), np.zeros(4096)])
def test_degenerate_envelopes_never_raise(envelope: npt.NDArray[np.float64]) -> None:
    """Empty, too short, and all-silent. None of them is an exception."""
    fit = refine_bpm_from_envelope(envelope, HOP_SECONDS, 132.0)

    assert isinstance(fit, TempoFit)
    assert fit.bpm == 132.0, "with nothing to measure, the coarse value stands"


def test_a_non_analysis_sample_rate_is_flagged() -> None:
    """Every threshold here was measured at 44.1 kHz and the output must admit it.

    The project never downsamples, so this should not happen — which is exactly
    why it must be visible when it does.
    """
    fit = refine_bpm(click_train(132.0, 40.0), 48000, 131.9)

    assert any("48000" in caveat for caveat in fit.caveats), fit.caveats


# --------------------------------------------------------------------------- #
# The octave and multiple guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("wrong", [66.0, 264.0, 88.0, 176.0])
def test_the_guard_rejects_half_time_double_time_and_triplet_relatives(wrong: float) -> None:
    """132 against its own metric relatives, every one of which must fail.

    Half time (66), double time (264), and the two-thirds and four-thirds
    relatives (88, 176) are the candidates an unguarded peak search picks up,
    because each of them is a genuine autocorrelation peak of the same signal.
    They sit 33% to 100% away from the coarse estimate; the guard allows 3%.
    """
    assert not within_bpm_tolerance(wrong, 132.0)


@pytest.mark.parametrize("close", [132.0, 131.855, 132.2, 135.9, 128.1])
def test_the_guard_accepts_a_backend_grade_error(close: float) -> None:
    """The guard must not fire on the errors it exists to tolerate.

    `RhythmExtractor2013` is quoted at +/-0.2 BPM; the two extremes here are
    +/-3%, which is fifteen times that. A guard that rejected a real backend
    reading would refuse to refine anything.
    """
    assert within_bpm_tolerance(close, 132.0)


@pytest.mark.parametrize("bad", [(float("nan"), 132.0), (132.0, 0.0), (-1.0, 132.0)])
def test_the_guard_rejects_values_that_are_not_tempi(bad: tuple[float, float]) -> None:
    """NaN, a zero reference and a negative candidate are all `False`, not errors."""
    assert not within_bpm_tolerance(*bad)


def test_refinement_never_changes_octave() -> None:
    """Handed 66 for a 132 BPM track, this returns 66 — and that is correct.

    16 beats at 66 BPM is a real autocorrelation peak of a 132 BPM signal,
    because it is 32 beats at 132. The guard is what keeps the module inside
    the octave its caller asked about. Correcting an octave error in the coarse
    estimate is a different job and deliberately not this one; a caller that
    silently jumped octave would make every downstream bar count wrong by a
    factor of two with no record of it having happened.
    """
    samples = click_train(132.0, 70.0)

    halved = refine_bpm(samples, ANALYSIS_SAMPLE_RATE, 66.0)

    assert halved.bpm is not None
    assert abs(halved.bpm - 66.0) <= 0.05, "must stay near the octave it was asked about"
    for multiple in halved.multiples:
        if multiple.bpm is not None and multiple.accepted:
            assert within_bpm_tolerance(multiple.bpm, 66.0)


def test_every_accepted_multiple_satisfies_the_guard(
    madonna: dict[str, npt.NDArray[np.float64]],
) -> None:
    """The guard is applied per reading, not only to the combined answer."""
    fit = refine_bpm_from_envelope(
        madonna["band_tempo"], float(madonna["hop_seconds"]), 131.854843
    )

    for multiple in fit.multiples:
        if multiple.accepted:
            assert multiple.bpm is not None
            assert within_bpm_tolerance(multiple.bpm, 131.854843)


def test_no_accepted_reading_can_violate_the_guard_at_any_coarse_estimate() -> None:
    """A property sweep, because the guard's real strength is structural.

    Worth being precise about how the guard works, since it is easy to
    mis-test. The search window is *contained within* the tolerance, so nothing
    outside +/-3% is ever reachable in the first place; the explicit
    `within_bpm_tolerance` check afterwards catches only the case where
    interpolation pushes the peak past the edge. The guard is therefore not a
    filter that fires often — it is a contract, and this asserts the contract
    holds across the whole range of coarse estimates a caller might hand in.

    Note what is *not* claimed. Handed a coarse estimate of 100 for a 132 BPM
    track, this module returns 100.57 and is right to: 21 beats at 132 BPM is a
    genuine autocorrelation peak, and read as 16 beats it implies 100.57, which
    is a legitimate reading of the question that was asked. Refinement answers
    "where exactly is the peak near here", not "what is the tempo".
    """
    samples = click_train(132.0, 70.0)
    magnitude, freqs = tempo._stft_magnitude(samples, ANALYSIS_SAMPLE_RATE)
    flux = tempo._spectral_flux(tempo._band_envelope(magnitude, freqs, *tempo.TEMPO_BAND_HZ))
    correlation = tempo._autocorrelation(flux)

    for coarse in np.linspace(100.0, 180.0, 41):
        for beats in BEAT_MULTIPLES:
            fit = tempo._fit_multiple(correlation, HOP_SECONDS, float(coarse), beats)
            if fit.accepted:
                assert fit.bpm is not None
                assert within_bpm_tolerance(fit.bpm, float(coarse)), (
                    f"N={beats} accepted {fit.bpm} against a coarse {coarse}"
                )
            else:
                assert fit.reason is not None, "a rejection must carry its reason"


def test_the_search_window_cannot_reach_a_neighbouring_beat_multiple() -> None:
    """Regression for the bug that cost this package an afternoon.

    The plan specified a +/-3% search window. At 32 beats that is +/-0.96 of a
    beat, so the window reaches the 31- and 33-beat peaks sitting one beat
    either side. Measured with that window, a 132 BPM click train returned
    **128.00** — the 33-beat peak read as if it were 32 beats, an error of four
    whole BPM that looked like a confident answer.

    The window is now sized in beats, so it cannot span a neighbouring
    multiple no matter what the tempo is.
    """
    for beats in BEAT_MULTIPLES:
        half_width = min(BPM_TOLERANCE, THRESHOLDS["peak_window_beats"] / beats)
        assert half_width * beats < 0.5, (
            f"window at N={beats} spans {half_width * beats:.3f} beats and can "
            "reach the adjacent multiple"
        )

    fit = refine_bpm(click_train(132.0, 70.0), ANALYSIS_SAMPLE_RATE, 131.85)

    assert fit.bpm is not None
    assert abs(fit.bpm - 132.0) <= BPM_TOLERANCE_BPM
    assert abs(fit.bpm - 128.0) > 1.0, "the 33-beat peak must not be readable as 32"


def test_a_multiple_tolerates_a_coarse_error_of_point_four_five_over_n(
    madonna: dict[str, npt.NDArray[np.float64]],
) -> None:
    """The measured rule that decides `BEAT_MULTIPLES`, swept on real material.

    Each multiple's search window is `PEAK_WINDOW_BEATS / N` wide, so each one
    is exact until the coarse estimate drifts far enough that the true peak
    falls outside its own window — at which point it reads a neighbouring
    beat-multiple and is wrong by several BPM. N=32 survives a 1% error and
    fails at 2%; N=64 survives 0.7% and fails at 1%.

    This is the trade `BEAT_MULTIPLES` settles, and it is worth pinning because
    a later reader will be tempted to add 64 for the extra resolution.
    """
    hop = float(madonna["hop_seconds"])
    flux = tempo._spectral_flux(np.asarray(madonna["band_tempo"], dtype=np.float64))
    correlation = tempo._autocorrelation(flux)

    for beats in (16, 32, 64):
        inside = 132.0 * (1.0 - 0.9 * THRESHOLDS["peak_window_beats"] / beats)
        outside = 132.0 * (1.0 - 1.5 * THRESHOLDS["peak_window_beats"] / beats)

        held = tempo._fit_multiple(correlation, hop, inside, beats)
        assert held.bpm is not None
        assert abs(held.bpm - MADONNA_BPM) <= BPM_TOLERANCE_BPM, (
            f"N={beats} should still be exact {0.9 * 0.45 / beats:.2%} from truth"
        )

        lost = tempo._fit_multiple(correlation, hop, outside, beats)
        assert lost.bpm is None or abs(lost.bpm - MADONNA_BPM) > 0.5, (
            f"N={beats} should have lost the peak {1.5 * 0.45 / beats:.2%} from truth"
        )


def test_the_multiples_stop_at_thirty_two_beats(
    madonna: dict[str, npt.NDArray[np.float64]],
) -> None:
    """Adding 64 beats would cost tolerance and buy no accuracy. Measured.

    **This contradicts `V2-PLAN.md` and the dispatch note**, which both record
    N=64 returning 129.97 as evidence that long lags are intrinsically
    unreliable. They are not. That reading is what a coarse estimate 1% off
    does to a window 0.70% wide; with the window sized in beats and an accurate
    coarse estimate, N=64 returns 132.0005 on this fixture — exact.

    The real reason to stop at 32 is the trade in the test above. Including 64
    would pull the module's tolerance for a bad coarse estimate down to 0.70%,
    against the 3% its own guard advertises, in exchange for an answer no
    closer to the truth than N=32's and a measurably weaker correlation.
    """
    assert BEAT_MULTIPLES == (16, 32)

    hop = float(madonna["hop_seconds"])
    flux = tempo._spectral_flux(np.asarray(madonna["band_tempo"], dtype=np.float64))
    correlation = tempo._autocorrelation(flux)

    readings = {
        beats: tempo._fit_multiple(correlation, hop, 131.854843, beats) for beats in (32, 64)
    }

    assert readings[64].bpm is not None
    assert abs(readings[64].bpm - MADONNA_BPM) <= BPM_TOLERANCE_BPM, (
        "N=64 is accurate here; the plan's claim that it returns a wrong peak "
        "was an artefact of the +/-3% search window, not of the lag length"
    )
    assert readings[64].r < readings[32].r, (
        "the honest cost of a longer lag is fewer repetitions to average"
    )


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #


def test_a_ritardando_reports_low_stability() -> None:
    """A track that slows by 1% does not have one tempo, and must not claim one.

    The halves are fitted independently and land 131.673 against 131.019, a
    difference of 0.654 BPM — thirteen times `STABILITY_HIGH_BPM`.

    Worth noting what the whole-source fit does here, because it is the reason
    stability is a separate signal rather than a derived one. Drift smears the
    autocorrelation peak, so the full fit still succeeds but its r collapses
    from the 0.74 real material gives to 0.28. A caller reading only `bpm`
    would get a plausible number that is true at no point in the track; the
    stability label is what says so.
    """
    samples = click_train(132.0, 120.0, ritardando=0.01)

    fit = refine_bpm(samples, ANALYSIS_SAMPLE_RATE, 132.0)

    assert fit.stability.label == "low", fit.stability
    assert fit.stability.delta_bpm is not None
    assert fit.stability.delta_bpm > 0.1
    assert fit.stability.first_half_bpm is not None
    assert fit.stability.second_half_bpm is not None
    assert fit.stability.first_half_bpm > fit.stability.second_half_bpm, (
        "a ritardando means the second half is slower, and the sign must survive"
    )


def test_a_steady_click_train_reports_high_stability() -> None:
    """The control for the test above: same generator, no drift."""
    fit = refine_bpm(click_train(132.0, 120.0), ANALYSIS_SAMPLE_RATE, 132.0)

    assert fit.stability.label == "high"
    assert fit.stability.delta_bpm is not None
    assert fit.stability.delta_bpm <= STABILITY_HIGH_BPM


def test_stability_from_envelope_takes_a_period_not_a_bpm(
    madonna: dict[str, npt.NDArray[np.float64]],
) -> None:
    """The public stability entry point is period-shaped, matching the plan."""
    result = stability_from_envelope(
        madonna["band_tempo"], float(madonna["hop_seconds"]), 60.0 / MADONNA_BPM
    )

    assert result.label == "high"
    assert result.delta_bpm is not None
    assert result.delta_bpm < STABILITY_HIGH_BPM


def test_stability_is_unknown_when_a_half_cannot_be_fitted() -> None:
    """No fit is not the same fact as no drift, and must not be reported as it."""
    result = stability_from_envelope(np.zeros(8192), HOP_SECONDS, 60.0 / 132.0)

    assert result.label == "unknown"
    assert result.delta_bpm is None


@pytest.mark.parametrize("period", [None, 0.0, -1.0, float("nan")])
def test_stability_without_a_usable_period_is_unknown(period: float | None) -> None:
    assert stability_from_envelope(np.zeros(4096), HOP_SECONDS, period).label == "unknown"


# --------------------------------------------------------------------------- #
# Downbeat: beat phase is easy, bar phase is not
# --------------------------------------------------------------------------- #


def test_beat_phase_is_found_within_one_frame() -> None:
    """The easy half, and it must be solid: which offset inside a beat is the beat.

    Resolution is one STFT frame and no better, deliberately — measured against
    known onset times the folded peak lands between 1.2 frames early and 0.4
    late depending on where the transient sits inside a frame, so interpolating
    would be fitting the analysis window rather than the music. One frame is
    11.6 ms, a tenth of a sixteenth step at 132 BPM.
    """
    for start in (0.100, 0.228, 0.350):
        samples = click_train(132.0, 40.0)
        padded = np.concatenate(
            (np.zeros(int(start * ANALYSIS_SAMPLE_RATE)), samples)
        )

        fit = find_downbeat(padded, ANALYSIS_SAMPLE_RATE, 60.0 / 132.0)

        assert fit.beat_offset_seconds is not None
        beat = 60.0 / 132.0
        error = min(
            (fit.beat_offset_seconds - start) % beat, (start - fit.beat_offset_seconds) % beat
        )
        assert error <= 2 * HOP_SECONDS, f"start {start}: off by {error * 1000:.1f} ms"
        assert fit.beat_confidence > 0.8, "percussive material must phase-lock strongly"


def test_four_on_the_floor_bar_phase_is_reported_as_degenerate() -> None:
    """The amendment this package was dispatched with, as an executable claim.

    The plan's objective — maximise folded energy on steps 0/4/8/12 — scores
    identically at four offsets when the kick plays every beat. Even the
    corrected objective, which reads the backbeat out of the bright band, can
    only separate beats {1, 3} from {2, 4}: nothing in a bar-level fold
    distinguishes beat 1 from beat 3, and folding at two and four bars shows no
    asymmetry either.

    So the requirement is not that the answer is right. It is that the answer
    says it might not be: a low phase confidence, and the surviving alternative
    named rather than discarded.
    """
    samples = four_on_the_floor(132.0, bars=24)

    fit = find_downbeat(samples, ANALYSIS_SAMPLE_RATE, 60.0 / 132.0)

    assert fit.status == "ambiguous"
    assert fit.phase_confidence < 1.0
    assert fit.confidence_label == "low"
    assert fit.unresolved_offsets, "an unresolved candidate must be named, not dropped"
    assert any("degenerate" in caveat for caveat in fit.caveats), fit.caveats

    half_bar = 2 * 60.0 / 132.0
    assert fit.offset_seconds is not None
    gaps = [abs(abs(other - fit.offset_seconds) - half_bar) for other in fit.unresolved_offsets]
    assert min(gaps) < 0.05, (
        f"the survivor should be half a bar away; offsets {fit.unresolved_offsets}"
    )


def test_the_backbeat_still_rules_out_beats_two_and_four() -> None:
    """Degenerate is not the same as uninformative: half the candidates do fall.

    The bright band separates the backbeat from the downbeat by a factor of
    eleven on real material. The objective must use that, or the ambiguity
    would be four-fold rather than two-fold and the downbeat would be a coin
    toss among all four beats.
    """
    samples = four_on_the_floor(132.0, bars=24)

    fit = find_downbeat(samples, ANALYSIS_SAMPLE_RATE, 60.0 / 132.0)

    scores = np.asarray(fit.candidate_scores)
    assert scores.size == 4
    assert scores[0] > 0 and scores[2] > 0, "beats 1 and 3 hold the low-heavy slots"
    assert scores[1] < 0 and scores[3] < 0, "beats 2 and 4 carry the backbeat"
    assert len(fit.unresolved_offsets) == 1, "only one candidate should survive, not three"


def test_an_accented_downbeat_resolves_the_bar_phase() -> None:
    """Give the objective a real asymmetry and it must take it, confidently.

    A louder kick on beat one is the conventional accent, and it is the case
    where a confident answer is warranted. If this failed while the degenerate
    test passed, the module would be refusing to answer rather than measuring.
    """
    samples = four_on_the_floor(132.0, bars=24, downbeat_seconds=0.25, downbeat_gain=2.5)

    fit = find_downbeat(samples, ANALYSIS_SAMPLE_RATE, 60.0 / 132.0)

    assert fit.bar_phase == 0, fit.candidate_scores
    assert fit.offset_seconds is not None
    assert abs(fit.offset_seconds - 0.25) < 3 * HOP_SECONDS
    assert fit.phase_confidence > DOWNBEAT_TIE_FRACTION
    assert fit.resolved_by == "spectral"


def test_madonna_downbeat_lands_on_the_beat_and_admits_the_bar_ambiguity(
    madonna: dict[str, npt.NDArray[np.float64]],
) -> None:
    """Real material, and both halves of the answer at once.

    The beat phase is pinned hard: 0.2322 s, the STFT frame containing the
    0.228 s that finding F1 re-folded this stem at, with a fold contrast of
    0.92. The bar phase is not: the winning margin is 0.013 against a spread of
    0.454, i.e. 3%, so beat 3 cannot be ruled out and is reported.

    The tie is broken by convention rather than measurement — the stem's first
    significant onset is on this beat, and music generally starts on a
    downbeat — and `resolved_by` records that it happened.
    """
    hop = float(madonna["hop_seconds"])

    fit = find_downbeat_from_envelopes(
        madonna["band_tempo"], madonna["band_air"], hop, 60.0 / MADONNA_BPM
    )

    assert fit.beat_offset_seconds is not None
    assert abs(fit.beat_offset_seconds - MADONNA_DOWNBEAT_SECONDS) <= HOP_SECONDS
    assert fit.beat_confidence > 0.85

    assert fit.status == "ambiguous"
    assert fit.phase_confidence < 0.5
    assert fit.resolved_by == "onset"
    assert fit.offset_seconds is not None
    assert abs(fit.offset_seconds - MADONNA_DOWNBEAT_SECONDS) <= HOP_SECONDS
    assert len(fit.unresolved_offsets) == 1
    assert any("convention, not measurement" in caveat for caveat in fit.caveats), fit.caveats


def test_the_four_candidate_offsets_are_one_beat_apart(
    madonna: dict[str, npt.NDArray[np.float64]],
) -> None:
    """Every beat of the bar is a candidate, and all four stay in the output.

    A caller that disagrees with the chosen phase — or that has an
    arrangement-level reason to prefer another — needs the alternatives rather
    than a single number it has to re-derive.
    """
    hop = float(madonna["hop_seconds"])

    fit = find_downbeat_from_envelopes(
        madonna["band_tempo"], madonna["band_air"], hop, 60.0 / MADONNA_BPM
    )

    assert len(fit.candidate_offsets) == 4
    gaps = np.diff(fit.candidate_offsets)
    assert np.allclose(gaps, 60.0 / MADONNA_BPM)
    assert fit.offset_seconds in fit.candidate_offsets


def test_a_source_with_no_bright_content_cannot_read_bar_phase() -> None:
    """A kick-only stem has a beat and no backbeat, and must say which it has.

    `beat_confidence` stays high because the beat grid is perfectly readable;
    `phase_confidence` is zero because nothing in the source distinguishes one
    beat of the bar from another. Collapsing those two into one number would
    hide the useful half.

    This is the regression for a bug found writing these tests. Measured, the
    bright band of a kick-only train holds 1.4e-07 of the low band's energy —
    pure float residue — and the objective happily read a shape out of it,
    picked a clear winner and reported a wrong downbeat at 0.5 confidence.
    `DOWNBEAT_BRIGHT_ACTIVITY_FLOOR` is what stops it.
    """
    fit = find_downbeat(click_train(132.0, 40.0), ANALYSIS_SAMPLE_RATE, 60.0 / 132.0)

    assert fit.beat_confidence > 0.8
    assert fit.phase_confidence == 0.0
    assert fit.status == "ambiguous"
    assert len(fit.unresolved_offsets) == 3, "all three other beats remain live"


@pytest.mark.parametrize("period", [None, 0.0, -1.0, float("nan")])
def test_downbeat_without_a_usable_period_is_unavailable(period: float | None) -> None:
    """The downbeat search is a phase problem, and phase needs a period."""
    fit = find_downbeat(click_train(132.0, 20.0), ANALYSIS_SAMPLE_RATE, period)

    assert fit.status == "unavailable"
    assert fit.offset_seconds is None
    assert fit.caveats


@pytest.mark.parametrize(
    "samples",
    [np.zeros(0), np.zeros(1024), np.zeros(ANALYSIS_SAMPLE_RATE * 5)],
)
def test_downbeat_on_degenerate_audio_never_raises(
    samples: npt.NDArray[np.float64],
) -> None:
    """Empty, too short, and digitally silent. All statuses, no exceptions."""
    fit = find_downbeat(samples, ANALYSIS_SAMPLE_RATE, 60.0 / 132.0)

    assert isinstance(fit, DownbeatFit)
    assert fit.status in {"ok", "ambiguous", "unavailable"}


def test_downbeat_survives_a_poisoned_envelope() -> None:
    """NaN and infinity in an envelope must not reach the output as a number."""
    low = np.abs(np.random.default_rng(4).normal(0.0, 1.0, 8192))
    low[10] = np.nan
    low[20] = np.inf

    fit = find_downbeat_from_envelopes(low, low, HOP_SECONDS, 60.0 / 132.0)

    assert isinstance(fit, DownbeatFit)
    if fit.offset_seconds is not None:
        assert math.isfinite(fit.offset_seconds)


def test_a_single_beat_bar_has_no_phase_to_resolve() -> None:
    """`beats_per_bar=1` is degenerate by definition, and must not divide by zero."""
    fit = find_downbeat(
        click_train(132.0, 30.0), ANALYSIS_SAMPLE_RATE, 60.0 / 132.0, beats_per_bar=1
    )

    assert fit.status == "ok"
    assert fit.bar_phase == 0
    assert fit.unresolved_offsets == ()


# --------------------------------------------------------------------------- #
# House rules
# --------------------------------------------------------------------------- #


def test_every_threshold_has_a_module_level_alias() -> None:
    """Mirror test, same convention as `drum_elements` and `heuristics`.

    A threshold that lives only in the dict is one nobody imports and nobody
    tests; a constant that has drifted from its documented dict entry is worse.
    Both are caught here.
    """
    for key, value in THRESHOLDS.items():
        alias = key.upper()
        assert hasattr(tempo, alias), f"THRESHOLDS['{key}'] has no {alias} alias"
        assert getattr(tempo, alias) == pytest.approx(value), (
            f"{alias} has drifted from THRESHOLDS['{key}']"
        )


def test_every_threshold_carries_its_provenance() -> None:
    """Ground rule 10: an undocumented constant is a future bug with no paper trail.

    Three of the four bugs in Part 1 were thresholds that were plausible in the
    abstract and wrong against real material, so each entry must be tagged
    `[grounded]` or `[guess]` where it is defined.

    Scanned as the file reads. One comment block may cover a **family** of
    keys sharing a first token — `confidence_high_r` with `confidence_medium_r`,
    `stability_high_bpm` with `stability_medium_bpm` — which is the same
    convention `drum_elements.THRESHOLDS` uses for its `*_saturation` partners
    and reads correctly, since a pair like that is one decision with two
    boundaries.

    Requiring the shared prefix is what keeps this test honest. Without it a
    new key dropped in underneath a documented one inherits its tag and the
    check passes vacuously; `tests/` verified that hole exists before the
    prefix rule closed it.
    """
    source = (Path(__file__).parent.parent / "src/audio_pipeline/tempo.py").read_text()
    body = source.split("THRESHOLDS: Final[dict[str, float]] = {", 1)[1].split("\n}\n", 1)[0]

    documented: set[str] = set()
    tagged = False
    previous_key: str | None = None
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            if previous_key is not None:  # a fresh comment block, so a fresh claim
                tagged = False
                previous_key = None
            tagged = tagged or "[grounded" in line or "[guess" in line
        elif line.startswith('"'):
            key = line.split('"')[1]
            family = previous_key is None or key.split("_")[0] == previous_key.split("_")[0]
            if tagged and family:
                documented.add(key)
            else:
                tagged = False
            previous_key = key

    assert set(THRESHOLDS) - documented == set(), (
        f"untagged thresholds: {sorted(set(THRESHOLDS) - documented)}"
    )


def test_tempo_never_imports_an_analysis_library_at_any_level() -> None:
    """Not at module top level, and not inside a function body either.

    Same rule as `drum_elements` and `note_track`, for the same reason: the
    tempo a track reports must not depend on which analysis wheel happened to
    install. This module is arithmetic numpy can do exactly.
    """
    source = Path(__file__).parent.parent / "src/audio_pipeline/tempo.py"
    tree = ast.parse(source.read_text())

    forbidden = {"librosa", "essentia"}
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [
                alias.name for alias in node.names if alias.name.split(".")[0] in forbidden
            ]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in forbidden:
                offenders.append(node.module)

    assert offenders == [], f"tempo must be numpy-only; found {offenders}"


def test_tempo_runs_with_neither_backend_importable() -> None:
    """The AST check proves nothing is written; this proves nothing is reached.

    Run out of process because this session has already imported both libraries
    for real.
    """
    program = """
import sys

class _Blocked:
    def find_module(self, name, path=None):
        if name.split(".")[0] in {"librosa", "essentia"}:
            raise ImportError(f"{name} is blocked for this test")
        return None

sys.meta_path.insert(0, _Blocked())

import numpy as np
from audio_pipeline.tempo import refine_bpm_from_envelope

rng = np.random.default_rng(0)
envelope = np.zeros(8192)
envelope[::40] = 1.0
fit = refine_bpm_from_envelope(envelope, 512 / 44100, 129.2)
print(fit.status)
print(sorted(m for m in sys.modules if m.split(".")[0] in {"essentia", "librosa"}))
"""
    src = Path(__file__).resolve().parent.parent / "src"
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    lines = result.stdout.strip().splitlines()
    assert lines[0] in {"refined", "coarse"}, result.stdout
    assert lines[1] == "[]", result.stdout


def test_the_result_types_are_plain_dataclasses_not_models() -> None:
    """W6 promotes these into `schemas.py`; until then pydantic stays out.

    Keeping the boundary explicit is what lets this module be developed and
    tested in a wave where `schemas.py` is frozen.
    """
    import dataclasses

    assert dataclasses.is_dataclass(TempoFit)
    assert dataclasses.is_dataclass(DownbeatFit)
    fit = refine_bpm_from_envelope(np.zeros(4096), HOP_SECONDS, 132.0)
    assert not hasattr(fit, "model_dump"), "TempoFit must not be a pydantic model yet"
