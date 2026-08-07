"""Refine a coarse BPM estimate into a grid-grade one, and find the downbeat.

Pure numpy, no scipy, no backend. Same architectural shape as
`drum_elements.py` and `note_track.py`: **no librosa, no essentia**, not at
module level and not inside a function, so the tempo a track reports does not
depend on which analysis wheel happened to install.

Why this module exists
----------------------

A backend's `rhythm.bpm` is accurate to roughly +/- 0.2 BPM. That is fine for a
label and useless for a grid. Calibration measured the cost exactly: the
pipeline reported **131.855 BPM** for a track that is **132.000** exactly, and
`drum_elements` built its cycle from the drums stem's own 132.040. Four
hundredths of a BPM accumulate 82 ms over 147 bars, which is **0.72 of a
sixteenth step**, and a textbook four-on-the-floor pattern was rejected as
`no_grid` with a quantisation error of 0.287 steps against a 0.18 allowance.

So the accuracy target here is **0.01 BPM**, and it is a requirement rather
than an aspiration: at 132 BPM over four minutes, 0.01 BPM is 21 ms, a fifth of
a sixteenth step, which a grid can absorb. 0.04 cannot.

`rhythm.bpm` is not replaced. It keeps its meaning and this lands beside it —
the project's additive-only rule, so the frozen v4 calibration outputs stay
numerically comparable.


How the refinement works
------------------------

1. **One band, 20-110 Hz.** `TEMPO_BAND_HZ` is narrower than `drum_elements`'
   20-150 Hz `kick` band on purpose: 110 Hz is A2, so the bound keeps a bass
   fundamental out of the envelope while the kick fundamental (40-60 Hz) sits
   fully inside it. Tempo is carried by the kick; a bass note landing off the
   beat is contamination.
2. **Flux, not energy.** `drum_elements._spectral_flux` — the half-wave
   rectified first difference of `log1p(energy)`. A tempo fit wants the
   instants at which energy *rose*, not how loud the passage is.
3. **Autocorrelate, and look near a multiple of the coarse period.** The peak
   at `N` beats is `N` times better resolved than the peak at one beat, because
   the same absolute lag error is divided by `N`. `BEAT_MULTIPLES` is `(16, 32)`
   and stops there — see below.
4. **Interpolate the peak by its centre of mass, not by a parabola.** Measured
   below; this is the single change that buys the last decimal place.
5. **Cross-check the two multiples.** They are independent estimates of the same
   quantity. If they disagree by more than `MULTIPLE_AGREEMENT_BPM` the fit is
   refused and the coarse value is returned, rather than one of them being
   picked on a coin toss.


Why the peak is interpolated by its centroid
--------------------------------------------

The plan for this module said "parabolic-interpolate the peak". Measured
against real and synthetic material, a three-point parabola is the accuracy
bottleneck, so this uses the peak lobe's **centre of mass** instead. The reason
is structural rather than incidental: the autocorrelation peak of a percussive
flux envelope is a symmetric lobe that is *not* a parabola, so a parabola
fitted through three samples of it is biased by wherever those samples happen
to fall relative to the apex — a bias that changes sign with tempo and does not
shrink with track length. A centroid over a symmetric lobe is unbiased at any
sampling phase.

Error against known truth, in BPM, on 90 s synthetic click trains and on the
committed Madonna flux fixture, mean of the two multiples:

========= ========== ========== ==========
true BPM  parabola   5-pt LSQ   centroid
========= ========== ========== ==========
100.0     -0.0063    -0.0070    -0.0001
120.0     +0.0050    +0.0051    -0.0004
128.5     +0.0014    +0.0181    -0.0001
132.0     +0.0041    +0.0143    -0.0004
145.0     +0.0112    +0.0028    -0.0002
174.0     +0.0230    +0.0343    -0.0005
madonna   +0.0015    +0.0083    -0.0000
========= ========== ========== ==========

The parabola misses the 0.01 BPM target outright above 140 BPM, which is
exactly where the corpus' drum-and-bass row lives. The centroid is inside
0.001 BPM everywhere measured.


Why the multiples stop at 32 beats
-----------------------------------

Longer lags look like free precision. What they actually buy is a tighter
requirement on the coarse estimate, and that is the trade `BEAT_MULTIPLES`
settles.

The search window below is `PEAK_WINDOW_BEATS / N` wide, because it has to be
narrower than the beat spacing to keep the neighbouring beat-multiples out. So
**a multiple tolerates a coarse error of 0.45/N and no more**: 2.8% at N=16,
1.4% at N=32, 0.70% at N=64. Measured on the Madonna fixture by sweeping the
coarse estimate away from the truth, every multiple is exact until its own
window loses the peak and then reads a neighbouring one:

========== ========== ========== ==========
coarse err N=16       N=32       N=64
========== ========== ========== ==========
-0.11%     131.9985   132.0014   132.0005
-0.70%     131.9985   132.0014   132.0005
-1.00%     131.9985   132.0014   **129.97**
-2.00%     131.9985   **128.00** 129.97
========== ========== ========== ==========

(The plan and the dispatch note both record N=64 returning 129.97 as evidence
that long lags are intrinsically unreliable. They are not — that reading is
what a coarse estimate 1% off does to a window 0.70% wide. Once the window is
sized in beats, N=64 is exact on this fixture. The contradiction is recorded
here rather than quietly fixed.)

So 64 is excluded for what it costs rather than for being wrong: it would drag
the module's tolerance for a bad coarse estimate down to 0.70%, against the 3%
its own guard advertises, while buying nothing — its answer, 132.0005, is no
better than N=32's 132.0014, and its correlation is measurably weaker (r 0.615
against 0.735), because a 4:27 track holds only nine 64-beat spans to average.
`BEAT_MULTIPLES` is therefore `(16, 32)` and is not a tuning knob.


The search window is sized in beats, not in percent
----------------------------------------------------

The plan specified a +/- 3% guard on the implied BPM. That is right as a
*guard* and wrong as a *search window*, and the difference cost a whole
afternoon: 3% of a 32-beat lag is 0.96 beats, so a +/-3% window around the
32-beat peak reaches almost exactly the 31- and 33-beat peaks either side of
it. Measured, a 132 BPM click train with a coarse estimate of 131.85 returned
**128.00** — the 33-beat peak, read as if it were 32 beats.

So the window half-width is `min(BPM_TOLERANCE, PEAK_WINDOW_BEATS / N)`, which
is +/-1.4% at N=32 and cannot reach a neighbouring multiple. `BPM_TOLERANCE`
still applies afterwards as an explicit guard on the interpolated result, which
is what rejects half-time and double-time candidates.

**Refinement never changes octave, by design.** Handed a coarse estimate of 66
for a 132 BPM track it returns 66, not 132: 16 beats at 66 BPM is a genuine
autocorrelation peak (it is 32 beats at 132), and the +/-3% guard is what stops
the module wandering off the octave its caller asked about. Fixing an octave
error in the coarse estimate is a different job and is not this one.


Why the octave is reported and not corrected
---------------------------------------------

That last paragraph was challenged by corpus material and it survived, so the
evidence is recorded here rather than in a commit message.

Essentia reports **84.92 BPM** for Roni Size' "Brown Paper Bag", exactly half
the true ~170, at the highest `bpm_confidence` of any track in the corpus.
`refine_bpm` faithfully refines the wrong octave to 85.04. The obvious fix is
to score the x0.5, x1 and x2 candidates by correlation and take the winner.
**Measured, that does not work**, and it fails in the direction that breaks
working tracks.

Two independent statistics were tried. Both separate the two tracks the fix was
designed against, and both invert on the controls.

*Correlation ratio at fixed N* — r at the doubled candidate over r at the base,
both at N=16, which is the only comparison where the two octaves are distinct
lags (r at x2/N=16 and r at x1/N=32 are the **same lag** and the same number):

============================ ====== =========
case                         r*2/r  required
============================ ====== =========
roni/drums                    1.21  double
badu/bass                     0.03  stay
badu/drums                    1.00  stay
madonna drums                 1.12  stay
synthetic click 132           1.23  stay
synthetic 4-on-the-floor 132  1.24  stay
badu/other                    1.81  stay
eno/other                     1.76  stay
============================ ====== =========

There is no threshold. Every value that must stay put brackets the one value
that must move. The cause is that autocorrelation decays with lag, so the
doubled candidate — a shorter lag — is favoured on nearly all material
regardless of what the beat is doing, and that bias is the same size as the
harmonic structure the rule would need to read.

*Half-beat fold occupancy* — whether the midpoint between two assumed beats
carries as much onset energy as the beats themselves, which autocorrelation
cannot see at all. It inverts outright: roni/drums, which must double, measures
**0.267**, while madonna drums, which must not, measures **0.760**. The
statistic turns out to measure whether the material subdivides, not whether the
estimate is halved — and a drum-and-bass break is syncopated where a house
track puts a hat on every offbeat.

So this module **reports the octave and refuses to pick one**. `TempoFit`
carries an `octave_candidates` list with each candidate's own correlation, and
`octave_status` says whether a neighbouring octave survives at all. `bpm` never
moves. The information a caller needs to arbitrate is genuinely here — on the
badu bass stem the doubled candidate scores 0.018 and is ruled out outright,
which is exactly the move that would have broken that track — but the
arbitration itself needs evidence this module does not have.

**What does resolve it** is grid quality, measured on the drums stem:
`drum_elements.decompose` fits Didn't Cha Know at 135.264 with a quantisation
error of 0.112 and `status="ok"`, and at 90.176 with 0.242 and `no_grid`. That
is a decisive separation and it belongs in W6, which holds both the stems and
the grid fitter. This module supplies the candidates; it does not choose.


Finding the downbeat is two problems, and the second one is hard
----------------------------------------------------------------

**Beat phase** — where the beat grid sits inside one beat period — is easy and
solid: fold the low-band flux at the beat period and take the offset with the
most energy. Measured contrast on real and synthetic material is 0.92-0.96 on a
scale where noise sits near 0.

**Bar phase** — which of the `beats_per_bar` beats is beat one — is not.
The obvious objective, "maximise folded energy on steps 0/4/8/12", is
**four-fold degenerate on four-on-the-floor material**, because the kick plays
every beat and therefore scores identically at four different offsets. It would
return a confidently wrong bar phase and silently rotate every downstream step
number by four.

What is actually asymmetric, measured on the Madonna drums stem folded at 132
BPM (median across 147 bars, normalised to sum 1 across the four beats):

======== ======== ======== ======== ========
band     beat 1   beat 2   beat 3   beat 4
======== ======== ======== ======== ========
20-110   0.2624   0.2370   0.2696   0.2310
6-16k    0.0415   0.4603   0.0353   0.4629
======== ======== ======== ======== ========

The bright band separates beats 2 and 4 from beats 1 and 3 by a factor of
eleven. That is the backbeat, and `_spectral_balance` reads it: a beat is a
downbeat candidate when it holds a large share of the low-band energy and a
small share of the bright-band energy.

It still leaves a **two-fold** ambiguity, and that ambiguity is real rather
than a defect of the objective — folding the same stem at two bars and at four
bars shows no asymmetry beyond one bar at all, so beat 1 and beat 3 are not
distinguishable from this stem by any amount of cleverness. The measured margin
between them is 0.013 against a 0.454 spread, i.e. 3%.

So when the top two candidates fall within `DOWNBEAT_TIE_FRACTION` of each
other the tie is broken by a **convention, not a measurement**: music generally
starts on a downbeat, so the candidate nearest the source's first significant
onset wins. On the Madonna stem that first onset is at 0.2322 s and it picks
the right one. `resolved_by` records that this happened, `unresolved_offsets`
lists what was not ruled out, and `confidence` stays low. A downbeat with an
honest "ambiguous to within half a bar" attached is far more useful than a
confident wrong one.

**Known failure mode, stated plainly.** A crash cymbal on beat one with nothing
else asymmetric inverts the objective — the downbeat becomes the *brightest*
beat, not the dimmest — and the fit will prefer the wrong phase. When the
first-onset convention disagrees with a clearly separated spectral winner the
spectral winner is kept, but the confidence is capped at
`DOWNBEAT_CONFLICT_CEILING` and the conflict is recorded in `caveats`. Nothing
here is ever silently confident about bar phase.


Nothing in this module raises
------------------------------

Every entry point returns a dataclass whose `status` and `caveats` say what
happened. A tempo fit is not worth losing an analysis over. The dataclasses are
plain (W6 promotes them into `schemas.py`); this module deliberately does not
import pydantic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from . import ANALYSIS_SAMPLE_RATE
from .drum_elements import (
    STFT_HOP_LENGTH,
    _band_envelope,
    _spectral_flux,
    _stft_magnitude,
)

__all__ = [
    "BEAT_MULTIPLES",
    "BPM_TOLERANCE",
    "BRIGHT_BAND_HZ",
    "DOWNBEAT_BRIGHT_ACTIVITY_FLOOR",
    "DOWNBEAT_CONFLICT_CEILING",
    "DOWNBEAT_ONSET_FRACTION",
    "DOWNBEAT_ONSET_SEARCH_BARS",
    "DOWNBEAT_SLOT_WINDOW_SECONDS",
    "DOWNBEAT_TIE_FRACTION",
    "HOP_SECONDS",
    "MIN_AUTOCORRELATION_R",
    "MULTIPLE_AGREEMENT_BPM",
    "OCTAVE_RATIOS",
    "PEAK_CENTROID_HALF_FRAMES",
    "PEAK_WINDOW_BEATS",
    "STABILITY_HIGH_BPM",
    "STABILITY_MEDIUM_BPM",
    "TEMPO_BAND_HZ",
    "THRESHOLDS",
    "DownbeatFit",
    "MultipleFit",
    "OctaveCandidate",
    "TempoFit",
    "TempoStability",
    "find_downbeat",
    "find_downbeat_from_envelopes",
    "refine_bpm",
    "refine_bpm_from_envelope",
    "stability",
    "stability_from_envelope",
    "within_bpm_tolerance",
]


# ---------------------------------------------------------------------------
# The pinned analysis grid
# ---------------------------------------------------------------------------

#: The band tempo is refined from, in Hz. **Deliberately not a
#: `DETECTION_BANDS` band** — it overlaps `kick` (20-150 Hz) and would break
#: that dict's disjointness, so it is named separately. 110 Hz is A2: the bound
#: keeps a bass fundamental out of the envelope the fit reads while the kick
#: fundamental at 40-60 Hz stays fully inside it. `tools/make-fixtures/` holds
#: an identical copy, because the committed `band_tempo` fixture is this band
#: and the two must not drift.
TEMPO_BAND_HZ: Final[tuple[float, float]] = (20.0, 110.0)

#: The band the downbeat's bar phase is read from, in Hz. Identical to
#: `drum_elements.DETECTION_BANDS["air"]`, so the fixture's `band_air` is
#: exactly this measurement. Hats, claps and cymbals live here; kicks do not,
#: which is the asymmetry the bar-phase objective needs.
BRIGHT_BAND_HZ: Final[tuple[float, float]] = (6000.0, 16000.0)

#: Seconds between STFT frames, pinned by `drum_elements.STFT_HOP_LENGTH` at
#: `ANALYSIS_SAMPLE_RATE`: 512 / 44100 = 11.60998 ms. Every committed fixture
#: stores this same number as `hop_seconds`, so a test never has to assume it.
HOP_SECONDS: Final[float] = STFT_HOP_LENGTH / ANALYSIS_SAMPLE_RATE

#: Beat multiples the autocorrelation peak is looked for at. Two, so they can
#: cross-check each other. **Not a tuning knob** — see the module docstring for
#: the measurement showing N=64 returns a wrong peak on real material.
BEAT_MULTIPLES: Final[tuple[int, ...]] = (16, 32)

#: Octaves of the coarse estimate that are measured and reported alongside it.
#: Half and double are the errors beat trackers actually make — Essentia halved
#: a drum-and-bass track at its highest confidence in the corpus. The x1 entry
#: is included so a reader can compare like with like without recomputing it.
#:
#: **These are reported, never applied.** The module docstring carries the
#: measurement showing that correlation cannot pick between them.
OCTAVE_RATIOS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Same candour convention as `heuristics.THRESHOLDS` and
# `drum_elements.THRESHOLDS`: `[grounded]` means the number follows from
# something physical, arithmetic or *measured*; `[guess]` means it is a
# plausible starting point that the calibration corpus should revisit.
#
# Every number here was measured against the committed Madonna fixture, against
# synthetic click trains at 100-174 BPM, against white noise, and against a
# click train with a deliberate ritardando. Where only one real track backs a
# number that is said so, because one house record is not a corpus.

THRESHOLDS: Final[dict[str, float]] = {
    # -- Peak search -------------------------------------------------------
    # [grounded, measured] Half-width of the autocorrelation search window,
    # **in beats**, before the BPM tolerance caps it. The neighbouring
    # beat-multiple peaks sit exactly one beat either side of the target, so
    # anything below 0.5 cannot reach them; 0.45 leaves a 10% margin against a
    # peak sitting right on the boundary. Measured failure at the plain +/-3%
    # window the plan specified: at N=32 that window is +/-0.96 beats and a 132
    # BPM click train returned 128.00, reading the 33-beat peak as 32 beats.
    #
    # The cost is a bound on how wrong the coarse estimate may be: the true lag
    # must land inside the window, which needs the coarse BPM within
    # 0.45/N = 1.4% of the truth at N=32. `RhythmExtractor2013` is accurate to
    # ~0.15% at 132 BPM, so there is a factor of nine of headroom.
    "peak_window_beats": 0.45,
    # [grounded, measured] Half-width, in STFT frames, of the window the peak's
    # centre of mass is taken over. Measured error, mean of both multiples,
    # across six synthetic tempi and the Madonna fixture:
    #
    #   half=2  up to 0.0039 BPM      half=6   up to 0.0039 BPM
    #   half=3  up to 0.0003 BPM      half=8   up to 0.0141 BPM
    #   half=4  up to 0.0006 BPM      half=12  up to 0.0266 BPM
    #
    # Both tails have a cause. Below 3 the window no longer contains the flux
    # lobe, which is 1-2 frames wide for a percussive onset and wider for a
    # soft one. Above 4 it starts collecting the autocorrelation's own decay
    # with lag, which tilts the centroid earlier — that is the real-material
    # tail, and it is why Madonna degrades faster than the click trains do.
    # 3 frames is 34.8 ms either side, comfortably clear of the eighth-note
    # sub-peak, which sits 19.6 frames away at 132 BPM.
    "peak_centroid_half_frames": 3.0,
    # -- Guards ------------------------------------------------------------
    # [grounded] How far the refined BPM may sit from the coarse estimate,
    # as a fraction. This is the guard against octave and multiple errors, and
    # 3% is wide enough to absorb any error a working beat tracker makes
    # (`RhythmExtractor2013` is quoted at +/-0.2 BPM, 0.15% at 132) while being
    # nowhere near the 50% or 100% an octave error needs. A candidate outside
    # it is rejected with a reason rather than silently clamped.
    "bpm_tolerance": 0.03,
    # [grounded, measured] How far the two beat multiples may disagree before
    # the whole fit is refused and the coarse value returned. Measured
    # disagreement: Madonna 0.0107 BPM, synthetic click trains 0.0002-0.0284,
    # white noise 1.5-3.6. The plan specified 0.05 and real material agrees
    # with it — Madonna uses a fifth of the allowance and noise overshoots it
    # by a factor of thirty.
    "multiple_agreement_bpm": 0.05,
    # [grounded, measured] Autocorrelation r below which a peak is not a beat.
    # Measured r at the winning lag: white noise 0.016-0.038 across four seeds,
    # a pure 55 Hz sine 0.0003, a click train with a 1% ritardando 0.118, real
    # and synthetic percussive material 0.65-0.91. 0.15 is four times the
    # loudest noise reading and four times below the weakest real one. The
    # ritardando case sitting below the floor is correct, not collateral: a
    # track that drifts 1% does not have one tempo to report.
    "min_autocorrelation_r": 0.15,
    # -- Confidence and stability labels -----------------------------------
    # [grounded, measured] r at or above which the fit is called `high`
    # confidence, and above which `medium`. Placed against the same measured
    # spread: everything genuinely periodic came in at 0.65 or better, so 0.50
    # has a clear margin below it, and 0.25 sits between the ritardando's 0.118
    # and that floor.
    "confidence_high_r": 0.50,
    "confidence_medium_r": 0.25,
    # [grounded, measured] Difference between the two halves' independently
    # fitted BPM, at or below which tempo is called `high` stability, and
    # `medium`. Measured: the Madonna halves fit 131.9986 and 132.0017, which
    # differ by **0.0030 BPM** — that track is machine-timed and must read as
    # high. A synthetic click train with a 1% ritardando splits 131.673 against
    # 131.019, a difference of 0.654, and must read as low. 0.05 is seventeen
    # times the machine-timed reading and reuses `multiple_agreement_bpm`'s
    # number for the same reason it was chosen there. 0.50 is the low boundary,
    # comfortably under the 0.654 a 1% drift produces.
    #
    # [guess] on the 0.50 in particular: no real material with a genuinely
    # floating tempo has been measured yet. Corpus row 3 (live band) exists to
    # settle it.
    "stability_high_bpm": 0.05,
    "stability_medium_bpm": 0.50,
    # -- Downbeat ----------------------------------------------------------
    # [grounded, measured] How close, as a fraction of the full score spread
    # across the candidates, the runner-up may be before the bar phase counts
    # as unresolved. Measured on the Madonna stem: the winning margin over the
    # runner-up is 0.013 against a spread of 0.454, i.e. **0.029** — genuinely
    # degenerate, and it must be caught. The separated pair (beats 2 and 4) sit
    # a full 1.0 away. 0.25 is an order of magnitude above the degenerate case
    # and a factor of four below the separated one.
    #
    # [guess] beyond that: one house record. The gap either side is large, but
    # nothing has yet measured a track whose bar phase is *weakly* separated,
    # which is where this number will actually be tested.
    "downbeat_tie_fraction": 0.25,
    # [grounded] Length of the window, in seconds, over which a beat slot's
    # band energy is taken as the peak value. 60 ms is a little over five STFT
    # frames: long enough to contain a hit whose onset landed anywhere in the
    # frame the beat rounds to, short enough not to reach the following
    # sixteenth, which is 114 ms away at 132 BPM.
    "downbeat_slot_window_seconds": 0.060,
    # [grounded, measured] Share of the low band's total energy that the bright
    # band must hold before the bar-phase objective is allowed to read it. A
    # numerical-noise gate, not a musical one, and deliberately the same value
    # and the same reasoning as
    # `drum_elements.THRESHOLDS["band_activity_floor"]`.
    #
    # Measured bright-over-low energy ratio: a kick-only click train **1.4e-07**
    # (float residue under a 52 Hz sine — no bright content whatsoever), the
    # Madonna drums stem 2.0e-02, synthetic four-on-the-floor with hats and
    # claps 1.5e-01 to 3.4e-01, white noise 9.4e+01. Five orders of magnitude of
    # headroom below the floor and a factor of twenty above it.
    #
    # Without this, a kick-only stem was measured resolving a bar phase out of
    # nothing: the objective read float residue, found what looked like a clear
    # winner, and reported a wrong downbeat at 0.5 confidence. A source with no
    # backbeat has no bar phase to find and must say so.
    "downbeat_bright_activity_floor": 1e-3,
    # [grounded, measured] Fraction of the flux envelope's own peak that counts
    # as the source's first significant onset. Measured on the Madonna drums
    # stem, the first frame clearing 0.2, 0.3 and 0.5 of the peak is the same
    # frame 20 in all three cases, so the exact value is not delicate. 0.2 is
    # the loosest of the three that still ignores a noise floor.
    "downbeat_onset_fraction": 0.20,
    # [grounded] How many bars from the start the first significant onset must
    # fall within for the "music starts on a downbeat" convention to be applied
    # at all. Past four bars the source plainly did not start with the music
    # and the convention has nothing to say.
    "downbeat_onset_search_bars": 4.0,
    # [guess] Ceiling placed on the bar-phase confidence when a clearly
    # separated spectral winner disagrees with the first-onset convention. Half
    # is deliberately unhelpful: it says the two available signals point
    # different ways and the caller should not build anything load-bearing on
    # the answer. See the module docstring's crash-on-beat-one failure mode.
    "downbeat_conflict_ceiling": 0.50,
}

PEAK_WINDOW_BEATS: Final[float] = THRESHOLDS["peak_window_beats"]
PEAK_CENTROID_HALF_FRAMES: Final[int] = int(THRESHOLDS["peak_centroid_half_frames"])
BPM_TOLERANCE: Final[float] = THRESHOLDS["bpm_tolerance"]
MULTIPLE_AGREEMENT_BPM: Final[float] = THRESHOLDS["multiple_agreement_bpm"]
MIN_AUTOCORRELATION_R: Final[float] = THRESHOLDS["min_autocorrelation_r"]
CONFIDENCE_HIGH_R: Final[float] = THRESHOLDS["confidence_high_r"]
CONFIDENCE_MEDIUM_R: Final[float] = THRESHOLDS["confidence_medium_r"]
STABILITY_HIGH_BPM: Final[float] = THRESHOLDS["stability_high_bpm"]
STABILITY_MEDIUM_BPM: Final[float] = THRESHOLDS["stability_medium_bpm"]
DOWNBEAT_TIE_FRACTION: Final[float] = THRESHOLDS["downbeat_tie_fraction"]
DOWNBEAT_SLOT_WINDOW_SECONDS: Final[float] = THRESHOLDS["downbeat_slot_window_seconds"]
DOWNBEAT_BRIGHT_ACTIVITY_FLOOR: Final[float] = THRESHOLDS["downbeat_bright_activity_floor"]
DOWNBEAT_ONSET_FRACTION: Final[float] = THRESHOLDS["downbeat_onset_fraction"]
DOWNBEAT_ONSET_SEARCH_BARS: Final[float] = THRESHOLDS["downbeat_onset_search_bars"]
DOWNBEAT_CONFLICT_CEILING: Final[float] = THRESHOLDS["downbeat_conflict_ceiling"]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
# Plain dataclasses on purpose. W6 promotes these into `schemas.py` as pydantic
# models; until then this module stays importable without pydantic and testable
# without a schema version bump.

Confidence = Literal["high", "medium", "low"]
StabilityLabel = Literal["high", "medium", "low", "unknown"]
TempoStatus = Literal["refined", "coarse", "unavailable"]
DownbeatStatus = Literal["ok", "ambiguous", "unavailable"]
ResolvedBy = Literal["spectral", "onset", "none"]


@dataclass(frozen=True)
class MultipleFit:
    """One beat multiple's independent reading of the tempo.

    Kept in the result rather than collapsed, because "the two multiples
    agreed" and "one of them was rejected" are different facts and the second
    one is what a reader needs when a fit looks wrong.
    """

    #: How many beats the autocorrelation lag spans. One of `BEAT_MULTIPLES`.
    beats: int
    #: BPM implied by the interpolated peak, or `None` when nothing usable was
    #: found at this multiple.
    bpm: float | None
    #: The interpolated lag in STFT frames, for auditing against `hop_seconds`.
    lag_frames: float | None
    #: Autocorrelation r at the integer peak, before interpolation. 0.0 when no
    #: peak was searched at all.
    r: float
    #: Whether this reading was allowed to contribute to the final BPM.
    accepted: bool
    #: Why not, when `accepted` is False. One of `too_short`,
    #: `below_correlation_floor`, `outside_bpm_tolerance`.
    reason: str | None = None


@dataclass(frozen=True)
class OctaveCandidate:
    """One octave of the coarse estimate, and how well the source supports it.

    Evidence, not a decision. See the module docstring for why this module
    reports octaves rather than choosing between them.
    """

    #: Multiplier against the coarse estimate. One of `OCTAVE_RATIOS`.
    ratio: float
    #: BPM this octave implies — the refined value when its own peak search
    #: succeeded, otherwise `coarse_bpm * ratio`.
    bpm: float
    #: Autocorrelation r at this octave, measured at `BEAT_MULTIPLES[0]`.
    #: Fixed N is the only comparison in which the octaves are distinct lags:
    #: r at x2/N=16 and r at x1/N=32 are the same lag and the same number.
    r: float
    #: `live` if r clears `MIN_AUTOCORRELATION_R`, `ruled_out` if it does not,
    #: `unmeasurable` if the lag did not fit in half the source.
    status: Literal["live", "ruled_out", "unmeasurable"]


@dataclass(frozen=True)
class TempoStability:
    """Whether the tempo is the same at the end of the source as at the start.

    Both halves are fitted **independently**, with the same machinery and the
    same guards as the whole-source fit. A track whose tempo genuinely moves
    should be detectable rather than silently averaged into a number that is
    true nowhere in it.
    """

    first_half_bpm: float | None
    second_half_bpm: float | None
    #: Absolute difference in BPM, or `None` when either half could not be fit.
    delta_bpm: float | None
    #: `high` under `STABILITY_HIGH_BPM`, `medium` under `STABILITY_MEDIUM_BPM`,
    #: `low` above it, `unknown` when a half could not be fitted at all.
    label: StabilityLabel


@dataclass(frozen=True)
class TempoFit:
    """A refined tempo, or an honest statement that refinement was refused."""

    #: The refined BPM when `status` is `refined`, the caller's coarse estimate
    #: when it is `coarse`, and `None` when it is `unavailable`.
    bpm: float | None
    #: `60 / bpm`. The number every downstream grid actually wants.
    period_seconds: float | None
    #: What was handed in, kept so a reader can see how far the refinement
    #: moved and in which direction.
    coarse_bpm: float | None
    #: Autocorrelation r, the weakest across the accepted multiples. 0.0 when
    #: nothing was accepted.
    confidence: float
    confidence_label: Confidence
    status: TempoStatus
    #: Per-multiple evidence, including rejected readings and their reasons.
    multiples: tuple[MultipleFit, ...]
    stability: TempoStability
    #: The x0.5, x1 and x2 readings of the coarse estimate, in `OCTAVE_RATIOS`
    #: order. **`bpm` is never taken from here** — this module reports the
    #: octave and does not correct it. See the module docstring.
    octave_candidates: tuple[OctaveCandidate, ...] = ()
    #: `single` when no neighbouring octave survives, so the reported octave is
    #: the only one the source supports; `ambiguous` when one does and a caller
    #: with more evidence should arbitrate; `unavailable` when nothing was
    #: measured.
    octave_status: Literal["single", "ambiguous", "unavailable"] = "unavailable"
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class DownbeatFit:
    """Where bar one starts, and how much that should be trusted.

    Two independent quantities, and conflating them hides the interesting one.
    `beat_confidence` says how well the beat grid's phase is pinned, which on
    percussive material is close to certain. `phase_confidence` says whether
    the right beat of the bar was identified, which on four-on-the-floor
    material frequently cannot be — see the module docstring.
    """

    #: Seconds from the start of the source to the first downbeat, or `None`.
    #: Always the earliest such position, so it lies in `[0, bar_seconds)`.
    offset_seconds: float | None
    #: `min(beat_confidence, phase_confidence)` — a downbeat is only as good as
    #: the weaker of the two stages that produced it.
    confidence: float
    confidence_label: Confidence
    #: Beat-grid phase alone, before a bar phase was chosen. Useful on its own:
    #: it is correct even when the bar phase is not.
    beat_offset_seconds: float | None
    #: Contrast of the folded flux at the winning beat phase, `1 - mean/peak`.
    beat_confidence: float
    #: How cleanly the winning bar phase beat the runner-up, `1.0` for a clear
    #: separation and near `0.0` for a degenerate objective.
    phase_confidence: float
    #: Which beat of the bar won, `0` for the first.
    bar_phase: int | None
    #: Every candidate downbeat position considered, in beat order.
    candidate_offsets: tuple[float, ...]
    #: The spectral-balance score of each candidate, same order.
    candidate_scores: tuple[float, ...]
    #: Candidates that could not be ruled out. Empty on a clean resolution.
    unresolved_offsets: tuple[float, ...]
    #: `spectral` when the objective separated the candidates on its own,
    #: `onset` when the "music starts on a downbeat" convention broke a tie,
    #: `none` when neither was available.
    resolved_by: ResolvedBy
    status: DownbeatStatus
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Envelopes:
    """The band envelopes an entry point derived from raw samples."""

    low: npt.NDArray[np.float64]
    bright: npt.NDArray[np.float64]
    hop_seconds: float
    caveats: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _clean(values: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """Finite float64 view of an envelope. A NaN or infinity is not energy."""
    array = np.asarray(values, dtype=np.float64).ravel()
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _autocorrelation(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalised autocorrelation of a flux envelope, `r[0] == 1`.

    Mean-subtracted, so a constant offset in the envelope does not manufacture
    correlation, and computed through the FFT with zero padding to at least
    twice the length so no lag wraps around onto itself.

    The **biased** estimator (divided by `r[0]`, not by the overlap `n - lag`)
    is deliberate. Unbiasing scales long lags up by `n / (n - lag)`, which
    amplifies exactly the noisiest part of the curve; the price is a gentle
    downward tilt with lag, which is why `PEAK_CENTROID_HALF_FRAMES` is small
    enough not to collect it.

    Returns all zeros when the input is constant or empty — a signal with no
    variance has no periodicity, and saying so beats dividing by zero.
    """
    if values.size < 2:
        return np.zeros(values.size, dtype=np.float64)
    centred = values - values.mean()
    size = 1 << int(math.ceil(math.log2(max(4, 2 * values.size))))
    spectrum = np.fft.rfft(centred, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[: values.size]
    zero_lag = float(correlation[0])
    if not math.isfinite(zero_lag) or zero_lag <= 0.0:
        return np.zeros(values.size, dtype=np.float64)
    return np.asarray(correlation / zero_lag, dtype=np.float64)


def _centroid_lag(
    correlation: npt.NDArray[np.float64], index: int, half: int = PEAK_CENTROID_HALF_FRAMES
) -> float:
    """Sub-frame peak position, as the centre of mass of the peak lobe.

    The window's own minimum is subtracted first so the centroid measures the
    lobe rather than the baseline it sits on. An asymmetric window would
    reintroduce exactly the sampling-phase bias this replaces, so a peak too
    close to either end of the array falls back to the integer index rather
    than being interpolated badly.

    See the module docstring for the measurement that chose this over a
    parabola.
    """
    if index - half < 0 or index + half >= correlation.size:
        return float(index)
    window = correlation[index - half : index + half + 1]
    weights = window - window.min()
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        return float(index)
    positions = np.arange(index - half, index + half + 1, dtype=np.float64)
    return float((positions * weights).sum() / total)


def within_bpm_tolerance(candidate_bpm: float, coarse_bpm: float) -> bool:
    """Is `candidate_bpm` within `BPM_TOLERANCE` of `coarse_bpm`?

    The guard that rejects octave and multiple errors, exposed because it is
    the one piece of this module worth asserting on directly: half-time and
    double-time candidates are 50% and 100% away from the coarse estimate and
    must never survive it.
    """
    if not (math.isfinite(candidate_bpm) and math.isfinite(coarse_bpm)):
        return False
    if coarse_bpm <= 0.0 or candidate_bpm <= 0.0:
        return False
    return abs(candidate_bpm - coarse_bpm) / coarse_bpm <= BPM_TOLERANCE


def _fit_multiple(
    correlation: npt.NDArray[np.float64],
    hop_seconds: float,
    coarse_bpm: float,
    beats: int,
) -> MultipleFit:
    """Locate the autocorrelation peak `beats` beats away from zero lag.

    The window is `min(BPM_TOLERANCE, PEAK_WINDOW_BEATS / beats)` wide either
    side of the lag the coarse estimate predicts — sized in beats so a
    neighbouring beat-multiple can never win it, capped by the BPM tolerance so
    the window is never wider than the guard that follows it.

    The lag is also required to be at most half the record, so the correlation
    at that lag averages at least two repetitions. Below that the peak is one
    observation of the pattern against itself and reads as confident noise.
    """
    coarse_lag = 60.0 / coarse_bpm / hop_seconds
    target = beats * coarse_lag
    relative = min(BPM_TOLERANCE, PEAK_WINDOW_BEATS / beats)
    low = max(1, int(math.floor(target * (1.0 - relative))))
    high = int(math.ceil(target * (1.0 + relative)))
    ceiling = correlation.size // 2 - PEAK_CENTROID_HALF_FRAMES - 1
    if high > ceiling or high <= low:
        return MultipleFit(
            beats=beats, bpm=None, lag_frames=None, r=0.0, accepted=False, reason="too_short"
        )

    index = low + int(np.argmax(correlation[low : high + 1]))
    peak_r = float(correlation[index])
    if not math.isfinite(peak_r) or peak_r < MIN_AUTOCORRELATION_R:
        return MultipleFit(
            beats=beats,
            bpm=None,
            lag_frames=float(index),
            r=max(0.0, peak_r) if math.isfinite(peak_r) else 0.0,
            accepted=False,
            reason="below_correlation_floor",
        )

    lag = _centroid_lag(correlation, index)
    if lag <= 0.0:
        return MultipleFit(
            beats=beats, bpm=None, lag_frames=None, r=peak_r, accepted=False, reason="too_short"
        )
    bpm = 60.0 * beats / (lag * hop_seconds)
    if not within_bpm_tolerance(bpm, coarse_bpm):
        return MultipleFit(
            beats=beats,
            bpm=bpm,
            lag_frames=lag,
            r=peak_r,
            accepted=False,
            reason="outside_bpm_tolerance",
        )
    return MultipleFit(beats=beats, bpm=bpm, lag_frames=lag, r=peak_r, accepted=True)


def _octave_candidates(
    correlation: npt.NDArray[np.float64], hop_seconds: float, coarse_bpm: float
) -> tuple[tuple[OctaveCandidate, ...], Literal["single", "ambiguous", "unavailable"]]:
    """Score each octave of the coarse estimate. Reports; never decides.

    Reuses `_fit_multiple` unchanged, so an octave candidate is measured by
    exactly the machinery that measures the reported tempo — including the
    beat-sized window and the centroid interpolator.
    """
    candidates: list[OctaveCandidate] = []
    for ratio in OCTAVE_RATIOS:
        fit = _fit_multiple(correlation, hop_seconds, coarse_bpm * ratio, BEAT_MULTIPLES[0])
        if fit.reason == "too_short":
            status: Literal["live", "ruled_out", "unmeasurable"] = "unmeasurable"
        elif fit.r >= MIN_AUTOCORRELATION_R:
            status = "live"
        else:
            status = "ruled_out"
        candidates.append(
            OctaveCandidate(
                ratio=ratio,
                bpm=fit.bpm if fit.bpm is not None else coarse_bpm * ratio,
                r=fit.r,
                status=status,
            )
        )

    base = next((item for item in candidates if item.ratio == 1.0), None)
    if base is None or base.status == "unmeasurable":
        return tuple(candidates), "unavailable"
    neighbours_live = any(
        item.ratio != 1.0 and item.status == "live" for item in candidates
    )
    return tuple(candidates), "ambiguous" if neighbours_live else "single"


def _confidence_label(r: float) -> Confidence:
    """Bucket an autocorrelation r. Thresholds documented in `THRESHOLDS`."""
    if r >= CONFIDENCE_HIGH_R:
        return "high"
    if r >= CONFIDENCE_MEDIUM_R:
        return "medium"
    return "low"


def _combine(fits: list[MultipleFit]) -> tuple[float | None, float, list[str]]:
    """Reduce the per-multiple readings to one BPM, or refuse to.

    Returns `(bpm, r, caveats)` with `bpm` `None` when the readings cannot be
    trusted. Averaging the accepted multiples rather than picking the strongest
    is deliberate: they are independent estimates whose residual interpolation
    errors have no reason to share a sign, and on the Madonna fixture the mean
    lands 0.0015 BPM from truth where the individual readings sit 0.0068 and
    0.0039 away.
    """
    accepted = [fit for fit in fits if fit.accepted and fit.bpm is not None]
    caveats: list[str] = []
    if not accepted:
        reasons = sorted({fit.reason for fit in fits if fit.reason})
        caveats.append(
            "no beat multiple produced a usable autocorrelation peak"
            + (f" ({', '.join(reasons)})" if reasons else "")
        )
        return None, 0.0, caveats

    values = [float(fit.bpm) for fit in accepted if fit.bpm is not None]
    weakest_r = min(fit.r for fit in accepted)
    if len(accepted) < len(BEAT_MULTIPLES):
        missing = sorted(
            f"{fit.beats} beats ({fit.reason})" for fit in fits if not fit.accepted and fit.reason
        )
        caveats.append(
            "only one beat multiple was usable, so the two estimates could not "
            f"cross-check each other: {', '.join(missing)}"
        )
        # One reading cannot be corroborated, so it never reports `high`.
        weakest_r = min(weakest_r, CONFIDENCE_HIGH_R - 1e-9)
        return values[0], weakest_r, caveats

    spread = max(values) - min(values)
    if spread > MULTIPLE_AGREEMENT_BPM:
        caveats.append(
            f"beat multiples disagreed by {spread:.4f} BPM, over the "
            f"{MULTIPLE_AGREEMENT_BPM} allowance, so the coarse estimate was kept"
        )
        return None, weakest_r, caveats
    return float(np.mean(values)), weakest_r, caveats


# ---------------------------------------------------------------------------
# Tempo refinement
# ---------------------------------------------------------------------------


def refine_bpm_from_envelope(
    envelope: npt.NDArray[np.floating[Any]],
    hop_seconds: float,
    coarse_bpm: float | None,
) -> TempoFit:
    """Refine `coarse_bpm` from a pre-computed 20-110 Hz **energy** envelope.

    The entry point the committed fixtures use, and the reason this module is
    testable against real material without shipping audio: a
    `tests/fixtures/real/*__drums_band_envelopes.npz` `band_tempo` array plus
    its `hop_seconds` feeds straight in.

    Args:
        envelope: Per-frame band **energy**, not flux. `_spectral_flux` is
            applied here, so callers hand over the same array the fixtures
            store and the project's own flux definition is applied once.
        hop_seconds: Seconds between frames. `HOP_SECONDS` for anything on the
            pipeline's own grid.
        coarse_bpm: The backend's estimate. `None` or a non-positive value
            means there is nothing to refine towards, and the result is
            `status="unavailable"` — this module locates a peak *near* a
            starting point and cannot invent one.

    Returns:
        A `TempoFit`. Never raises.
    """
    caveats: list[str] = []
    flux = _spectral_flux(_clean(envelope))
    if coarse_bpm is None or not math.isfinite(coarse_bpm) or coarse_bpm <= 0.0:
        return _unavailable("no coarse BPM to refine from", coarse_bpm)
    if not math.isfinite(hop_seconds) or hop_seconds <= 0.0:
        return _unavailable("frame hop is not a positive number of seconds", coarse_bpm)
    if flux.size < 4:
        return _unavailable("envelope too short to autocorrelate", coarse_bpm)

    correlation = _autocorrelation(flux)
    fits = [
        _fit_multiple(correlation, hop_seconds, coarse_bpm, beats) for beats in BEAT_MULTIPLES
    ]
    refined, weakest_r, combine_caveats = _combine(fits)
    caveats.extend(combine_caveats)

    octaves, octave_status = _octave_candidates(correlation, hop_seconds, coarse_bpm)
    if octave_status == "ambiguous":
        alternatives = ", ".join(
            f"{item.bpm:.2f} BPM (r={item.r:.2f})"
            for item in octaves
            if item.ratio != 1.0 and item.status == "live"
        )
        caveats.append(
            "the octave is not settled by correlation alone: the reported tempo "
            f"was not shifted, but {alternatives} also fits this source. Arbitrate "
            "on grid quality, not on r — see the module docstring"
        )

    fit_stability = _stability_from_flux(flux, hop_seconds, coarse_bpm)

    if refined is None:
        return TempoFit(
            bpm=coarse_bpm,
            period_seconds=60.0 / coarse_bpm,
            coarse_bpm=coarse_bpm,
            confidence=weakest_r,
            confidence_label="low",
            status="coarse",
            multiples=tuple(fits),
            stability=fit_stability,
            octave_candidates=octaves,
            octave_status=octave_status,
            caveats=tuple(caveats),
        )
    return TempoFit(
        bpm=refined,
        period_seconds=60.0 / refined,
        coarse_bpm=coarse_bpm,
        confidence=weakest_r,
        confidence_label=_confidence_label(weakest_r),
        status="refined",
        multiples=tuple(fits),
        stability=fit_stability,
        octave_candidates=octaves,
        octave_status=octave_status,
        caveats=tuple(caveats),
    )


def _unavailable(reason: str, coarse_bpm: float | None) -> TempoFit:
    """A `TempoFit` for the cases where there is nothing to fit."""
    usable = coarse_bpm is not None and math.isfinite(coarse_bpm) and coarse_bpm > 0.0
    return TempoFit(
        bpm=coarse_bpm if usable else None,
        period_seconds=60.0 / coarse_bpm if usable and coarse_bpm else None,
        coarse_bpm=coarse_bpm if usable else None,
        confidence=0.0,
        confidence_label="low",
        status="coarse" if usable else "unavailable",
        multiples=(),
        stability=TempoStability(None, None, None, "unknown"),
        caveats=(reason,),
    )


def refine_bpm(
    samples: npt.NDArray[np.floating[Any]],
    sample_rate: int,
    coarse_bpm: float | None,
) -> TempoFit:
    """Refine `coarse_bpm` from raw audio.

    Computes the `TEMPO_BAND_HZ` energy envelope on the project's pinned STFT
    grid and delegates to `refine_bpm_from_envelope`, which is where all the
    reasoning lives.

    Args:
        samples: Mono or multi-channel float audio. Multi-channel is averaged
            to mono; nothing here is stereo-aware.
        sample_rate: Must be `ANALYSIS_SAMPLE_RATE`. Any other rate still
            works and adds a caveat saying every threshold was calibrated at
            44.1 kHz.
        coarse_bpm: The backend's estimate.

    Returns:
        A `TempoFit`. Never raises.
    """
    try:
        envelopes = _envelopes_from_samples(samples, sample_rate)
        fit = refine_bpm_from_envelope(envelopes.low, envelopes.hop_seconds, coarse_bpm)
        if envelopes.caveats:
            return TempoFit(
                bpm=fit.bpm,
                period_seconds=fit.period_seconds,
                coarse_bpm=fit.coarse_bpm,
                confidence=fit.confidence,
                confidence_label=fit.confidence_label,
                status=fit.status,
                multiples=fit.multiples,
                stability=fit.stability,
                octave_candidates=fit.octave_candidates,
                octave_status=fit.octave_status,
                caveats=(*envelopes.caveats, *fit.caveats),
            )
        return fit
    except Exception as error:  # noqa: BLE001 - deliberate: never break an analysis
        return _unavailable(f"tempo refinement failed with {type(error).__name__}", coarse_bpm)


def _envelopes_from_samples(
    samples: npt.NDArray[np.floating[Any]], sample_rate: int
) -> _Envelopes:
    """One STFT, both bands this module reads."""
    caveats: list[str] = []
    if sample_rate != ANALYSIS_SAMPLE_RATE:
        caveats.append(
            f"analysed at {sample_rate} Hz; every threshold here was calibrated "
            f"at {ANALYSIS_SAMPLE_RATE} Hz"
        )
    magnitude, freqs = _stft_magnitude(samples, sample_rate)
    low = _band_envelope(magnitude, freqs, *TEMPO_BAND_HZ)
    bright = _band_envelope(magnitude, freqs, *BRIGHT_BAND_HZ, include_high=True)
    return _Envelopes(
        low=low, bright=bright, hop_seconds=STFT_HOP_LENGTH / sample_rate, caveats=tuple(caveats)
    )


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def _stability_from_flux(
    flux: npt.NDArray[np.float64], hop_seconds: float, coarse_bpm: float
) -> TempoStability:
    """Fit both halves independently and report how far apart they landed.

    Only the multiples **both** halves could measure are compared, so a half
    that fell back to 16 beats is never held against the other half's 32-beat
    reading. Each half is half as long, so at 132 BPM a 32-beat lag needs a
    58 s source before it can be measured in halves at all.
    """
    midpoint = flux.size // 2
    if midpoint < 4:
        return TempoStability(None, None, None, "unknown")

    halves = []
    for segment in (flux[:midpoint], flux[midpoint:]):
        correlation = _autocorrelation(segment)
        halves.append(
            {
                fit.beats: fit
                for fit in (
                    _fit_multiple(correlation, hop_seconds, coarse_bpm, beats)
                    for beats in BEAT_MULTIPLES
                )
            }
        )

    shared = [
        beats
        for beats in BEAT_MULTIPLES
        if all(half[beats].accepted and half[beats].bpm is not None for half in halves)
    ]
    if not shared:
        return TempoStability(None, None, None, "unknown")

    first, second = (
        float(np.mean([half[beats].bpm for beats in shared])) for half in halves  # type: ignore[arg-type]
    )
    delta = abs(first - second)
    if delta <= STABILITY_HIGH_BPM:
        label: StabilityLabel = "high"
    elif delta <= STABILITY_MEDIUM_BPM:
        label = "medium"
    else:
        label = "low"
    return TempoStability(first, second, delta, label)


def stability_from_envelope(
    envelope: npt.NDArray[np.floating[Any]],
    hop_seconds: float,
    period_seconds: float | None,
) -> TempoStability:
    """Tempo stability from a pre-computed 20-110 Hz energy envelope.

    Exposed separately from `refine_bpm_from_envelope` — which already computes
    it — for callers holding a period rather than a BPM, and so the halves can
    be asserted on directly.
    """
    if period_seconds is None or not math.isfinite(period_seconds) or period_seconds <= 0.0:
        return TempoStability(None, None, None, "unknown")
    if not math.isfinite(hop_seconds) or hop_seconds <= 0.0:
        return TempoStability(None, None, None, "unknown")
    flux = _spectral_flux(_clean(envelope))
    return _stability_from_flux(flux, hop_seconds, 60.0 / period_seconds)


def stability(
    samples: npt.NDArray[np.floating[Any]],
    sample_rate: int,
    period_seconds: float | None,
) -> TempoStability:
    """Tempo stability from raw audio, around a known beat period."""
    try:
        envelopes = _envelopes_from_samples(samples, sample_rate)
        return stability_from_envelope(envelopes.low, envelopes.hop_seconds, period_seconds)
    except Exception:  # noqa: BLE001 - deliberate: never break an analysis
        return TempoStability(None, None, None, "unknown")


# ---------------------------------------------------------------------------
# Downbeat
# ---------------------------------------------------------------------------


def _beat_phase(
    flux: npt.NDArray[np.float64], hop_seconds: float, period_seconds: float
) -> tuple[int, float]:
    """Which frame offset inside one beat period carries the most flux.

    Returns `(frame_offset, contrast)` where contrast is `1 - mean/peak` over
    the candidate offsets: 1.0 for a grid that is entirely concentrated on one
    offset, 0.0 for a signal with no beat at all. Measured 0.92 on the Madonna
    drums stem and 0.95-0.96 on synthetic percussion.

    Resolution is one STFT frame and no better, and deliberately so. Measured
    against click trains with known onset times, the folded peak lands between
    1.15 frames early and 0.4 frames late depending on where the transient sits
    inside a frame, so a sub-frame interpolation here would be fitting the STFT
    window's own shape rather than the music. One frame is 11.6 ms, which is
    0.10 of a sixteenth step at 132 BPM — well inside the 0.18-step
    quantisation allowance the grid is judged against.
    """
    beat_frames = period_seconds / hop_seconds
    offsets = max(1, int(round(beat_frames)))
    if flux.size <= offsets:
        return 0, 0.0
    scores = np.zeros(offsets, dtype=np.float64)
    for offset in range(offsets):
        count = int((flux.size - 1 - offset) / beat_frames) + 1
        if count < 1:
            continue
        indices = np.rint(offset + np.arange(count) * beat_frames).astype(np.intp)
        indices = indices[indices < flux.size]
        if indices.size:
            scores[offset] = float(flux[indices].mean())
    peak = float(scores.max())
    if peak <= 0.0:
        return 0, 0.0
    return int(np.argmax(scores)), float(max(0.0, 1.0 - scores.mean() / peak))


def _slot_energy(
    envelope: npt.NDArray[np.float64],
    hop_seconds: float,
    offset_seconds: float,
    period_seconds: float,
    slots: int,
) -> npt.NDArray[np.float64]:
    """Median-across-bars peak energy at each beat slot of the bar.

    The envelope-fold method from finding F8, applied at beat rather than
    sixteenth resolution: fold, then take the **median** across bars. That is
    what makes this survive an arrangement — a sixteen-bar breakdown with the
    kick out drops out of the median instead of dragging the profile with it,
    and it needs no threshold to do it.
    """
    window = max(1, int(round(DOWNBEAT_SLOT_WINDOW_SECONDS / hop_seconds)))
    duration = envelope.size * hop_seconds
    beats = int((duration - offset_seconds) / period_seconds)
    buckets: list[list[float]] = [[] for _ in range(slots)]
    for beat in range(max(0, beats)):
        start = int(round((offset_seconds + beat * period_seconds) / hop_seconds))
        if start < 0 or start + window > envelope.size:
            continue
        buckets[beat % slots].append(float(envelope[start : start + window].max()))
    return np.array(
        [float(np.median(values)) if values else 0.0 for values in buckets], dtype=np.float64
    )


def _bright_band_is_active(
    low: npt.NDArray[np.float64], bright: npt.NDArray[np.float64]
) -> bool:
    """Does the bright band hold enough energy for its shape to mean anything?

    The gate that stops a kick-only source resolving a bar phase out of float
    residue. Measured ratios and the reasoning behind the floor are on
    `DOWNBEAT_BRIGHT_ACTIVITY_FLOOR` in `THRESHOLDS`.
    """
    low_total = float(low.sum())
    if low_total <= 0.0:
        return False
    return float(bright.sum()) / low_total >= DOWNBEAT_BRIGHT_ACTIVITY_FLOOR


def _spectral_balance(
    low: npt.NDArray[np.float64], bright: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | None:
    """Per-slot "low-heavy and not bright" score, or `None` when unreadable.

    Each band's slot profile is normalised to sum to 1 first, so the score is a
    comparison of *distributions across the bar* and not of band levels — which
    matters, because the two bands differ by orders of magnitude in absolute
    energy and a raw difference would just report the low band.
    """
    low_total = float(low.sum())
    bright_total = float(bright.sum())
    if low_total <= 0.0 or bright_total <= 0.0:
        return None
    return np.asarray(low / low_total - bright / bright_total, dtype=np.float64)


def _first_onset_seconds(flux: npt.NDArray[np.float64], hop_seconds: float) -> float | None:
    """When the source first does something, by its own flux scale."""
    peak = float(flux.max()) if flux.size else 0.0
    if peak <= 0.0:
        return None
    above = np.flatnonzero(flux >= DOWNBEAT_ONSET_FRACTION * peak)
    if above.size == 0:
        return None
    return float(above[0]) * hop_seconds


def find_downbeat_from_envelopes(
    low_envelope: npt.NDArray[np.floating[Any]],
    bright_envelope: npt.NDArray[np.floating[Any]],
    hop_seconds: float,
    period_seconds: float | None,
    *,
    beats_per_bar: int = 4,
) -> DownbeatFit:
    """Locate the first downbeat from pre-computed band **energy** envelopes.

    The fixture-facing entry point, matching `refine_bpm_from_envelope`.

    Args:
        low_envelope: `TEMPO_BAND_HZ` per-frame energy — the fixtures'
            `band_tempo`. Carries the beat.
        bright_envelope: `BRIGHT_BAND_HZ` per-frame energy — the fixtures'
            `band_air`. Carries the backbeat, and is the only asymmetric
            feature available at bar level on four-on-the-floor material.
        hop_seconds: Seconds between frames.
        period_seconds: The **beat** period, not the bar. `TempoFit.period_seconds`.
        beats_per_bar: 4 for 4/4.

    Returns:
        A `DownbeatFit`. Never raises. Read `phase_confidence` before trusting
        `offset_seconds` to a finer resolution than one bar.
    """
    low = _clean(low_envelope)
    bright = _clean(bright_envelope)
    if period_seconds is None or not math.isfinite(period_seconds) or period_seconds <= 0.0:
        return _downbeat_unavailable("no beat period to search within")
    if not math.isfinite(hop_seconds) or hop_seconds <= 0.0:
        return _downbeat_unavailable("frame hop is not a positive number of seconds")
    if beats_per_bar < 1:
        return _downbeat_unavailable("beats_per_bar must be at least 1")
    flux = _spectral_flux(low)
    if flux.size < 4 or float(flux.max()) <= 0.0:
        return _downbeat_unavailable("low band holds no transients to phase-lock to")

    frame, beat_confidence = _beat_phase(flux, hop_seconds, period_seconds)
    beat_offset = frame * hop_seconds
    candidates = tuple(beat_offset + slot * period_seconds for slot in range(beats_per_bar))

    caveats: list[str] = []
    scores = (
        _spectral_balance(
            _slot_energy(low, hop_seconds, beat_offset, period_seconds, beats_per_bar),
            _slot_energy(bright, hop_seconds, beat_offset, period_seconds, beats_per_bar),
        )
        if _bright_band_is_active(low, bright)
        else None
    )
    onset = _first_onset_seconds(flux, hop_seconds)
    onset_phase = _onset_phase(onset, candidates, period_seconds, beats_per_bar)

    if beats_per_bar == 1:
        return DownbeatFit(
            offset_seconds=candidates[0],
            confidence=beat_confidence,
            confidence_label=_confidence_label(beat_confidence),
            beat_offset_seconds=beat_offset,
            beat_confidence=beat_confidence,
            phase_confidence=1.0,
            bar_phase=0,
            candidate_offsets=candidates,
            candidate_scores=(0.0,),
            unresolved_offsets=(),
            resolved_by="spectral",
            status="ok",
            caveats=tuple(caveats),
        )

    if scores is None:
        return _downbeat_from_onset_only(
            beat_offset, beat_confidence, candidates, onset_phase, beats_per_bar
        )

    order = np.argsort(scores)[::-1]
    spread = float(scores.max() - scores.min())
    if spread <= 0.0:
        return _downbeat_from_onset_only(
            beat_offset, beat_confidence, candidates, onset_phase, beats_per_bar
        )

    best = int(order[0])
    margin = float(scores[best] - scores[int(order[1])]) / spread
    phase_confidence = float(min(1.0, margin / DOWNBEAT_TIE_FRACTION))
    def _gap(slot: int) -> float:
        """How far behind the winner a candidate scored, as a share of the spread."""
        return float(scores[best] - scores[slot]) / spread

    tied = tuple(
        int(slot) for slot in order if slot != best and _gap(int(slot)) <= DOWNBEAT_TIE_FRACTION
    )
    resolved_by: ResolvedBy = "spectral"
    unresolved = [candidates[slot] for slot in tied]

    if tied:
        caveats.append(
            f"bar phase is degenerate: {len(tied)} other beat(s) scored within "
            f"{DOWNBEAT_TIE_FRACTION:.0%} of the winner"
        )
        if onset_phase is not None and onset_phase in tied:
            best = onset_phase
            resolved_by = "onset"
            caveats.append(
                "tie broken by convention, not measurement: the source's first "
                "significant onset lands on this beat, and music generally "
                "starts on a downbeat"
            )
            unresolved = [
                candidates[slot] for slot in (int(order[0]), *tied) if slot != best
            ]
    elif onset_phase is not None and onset_phase != best:
        caveats.append(
            "the spectral objective and the source's first onset disagree about "
            "bar phase; confidence capped and the alternative listed"
        )
        phase_confidence = min(phase_confidence, DOWNBEAT_CONFLICT_CEILING)
        unresolved = [candidates[onset_phase]]

    confidence = min(beat_confidence, phase_confidence)
    return DownbeatFit(
        offset_seconds=candidates[best],
        confidence=confidence,
        confidence_label=_confidence_label(confidence),
        beat_offset_seconds=beat_offset,
        beat_confidence=beat_confidence,
        phase_confidence=phase_confidence,
        bar_phase=best,
        candidate_offsets=candidates,
        candidate_scores=tuple(float(value) for value in scores),
        unresolved_offsets=tuple(unresolved),
        resolved_by=resolved_by,
        status="ok" if phase_confidence >= 1.0 else "ambiguous",
        caveats=tuple(caveats),
    )


def _onset_phase(
    onset_seconds: float | None,
    candidates: tuple[float, ...],
    period_seconds: float,
    beats_per_bar: int,
) -> int | None:
    """Which candidate the source's first significant onset sits on.

    `None` when there is no onset, or when it arrives more than
    `DOWNBEAT_ONSET_SEARCH_BARS` bars in — past that the source plainly did not
    start with the music and the convention has nothing to say.
    """
    if onset_seconds is None:
        return None
    bar = period_seconds * beats_per_bar
    if onset_seconds > DOWNBEAT_ONSET_SEARCH_BARS * bar:
        return None
    distances = [
        min((onset_seconds - candidate) % bar, (candidate - onset_seconds) % bar)
        for candidate in candidates
    ]
    return int(np.argmin(distances))


def _downbeat_from_onset_only(
    beat_offset: float,
    beat_confidence: float,
    candidates: tuple[float, ...],
    onset_phase: int | None,
    beats_per_bar: int,
) -> DownbeatFit:
    """Bar phase when the spectral objective had nothing to read.

    Happens on a source with no bright-band content at all — an isolated bass
    stem, a sub-only kick track. The beat phase is still solid; only the bar
    phase is guesswork, and it says so.
    """
    slot = onset_phase if onset_phase is not None else 0
    return DownbeatFit(
        offset_seconds=candidates[slot],
        confidence=0.0,
        confidence_label="low",
        beat_offset_seconds=beat_offset,
        beat_confidence=beat_confidence,
        phase_confidence=0.0,
        bar_phase=slot,
        candidate_offsets=candidates,
        candidate_scores=tuple(0.0 for _ in candidates),
        unresolved_offsets=tuple(
            candidate for index, candidate in enumerate(candidates) if index != slot
        ),
        resolved_by="onset" if onset_phase is not None else "none",
        status="ambiguous",
        caveats=(
            "no bright-band content to read bar phase from; every beat of the "
            f"bar remains a candidate ({beats_per_bar} in total)",
        ),
    )


def _downbeat_unavailable(reason: str) -> DownbeatFit:
    """A `DownbeatFit` for the cases where there is nothing to fit."""
    return DownbeatFit(
        offset_seconds=None,
        confidence=0.0,
        confidence_label="low",
        beat_offset_seconds=None,
        beat_confidence=0.0,
        phase_confidence=0.0,
        bar_phase=None,
        candidate_offsets=(),
        candidate_scores=(),
        unresolved_offsets=(),
        resolved_by="none",
        status="unavailable",
        caveats=(reason,),
    )


def find_downbeat(
    samples: npt.NDArray[np.floating[Any]],
    sample_rate: int,
    period_seconds: float | None,
    *,
    beats_per_bar: int = 4,
) -> DownbeatFit:
    """Locate the first downbeat in raw audio, given a known beat period.

    Computes both band envelopes on the pinned STFT grid and delegates to
    `find_downbeat_from_envelopes`, which is where all the reasoning lives.

    Returns:
        A `DownbeatFit`. Never raises.
    """
    try:
        envelopes = _envelopes_from_samples(samples, sample_rate)
        fit = find_downbeat_from_envelopes(
            envelopes.low,
            envelopes.bright,
            envelopes.hop_seconds,
            period_seconds,
            beats_per_bar=beats_per_bar,
        )
        if not envelopes.caveats:
            return fit
        return DownbeatFit(
            offset_seconds=fit.offset_seconds,
            confidence=fit.confidence,
            confidence_label=fit.confidence_label,
            beat_offset_seconds=fit.beat_offset_seconds,
            beat_confidence=fit.beat_confidence,
            phase_confidence=fit.phase_confidence,
            bar_phase=fit.bar_phase,
            candidate_offsets=fit.candidate_offsets,
            candidate_scores=fit.candidate_scores,
            unresolved_offsets=fit.unresolved_offsets,
            resolved_by=fit.resolved_by,
            status=fit.status,
            caveats=(*envelopes.caveats, *fit.caveats),
        )
    except Exception as error:  # noqa: BLE001 - deliberate: never break an analysis
        return _downbeat_unavailable(f"downbeat search failed with {type(error).__name__}")
