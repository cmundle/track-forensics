"""Decompose a drum source into kick, snare and hat hits, in pure numpy.

**No librosa. No essentia. Not at module level, not inside a function, not
lazily.** That is the architectural point of this module, not an incidental
property: `analyze.py` resolves one of two backends whose onset detection
diverges hard (measured on white noise: essentia 0.125 onsets/s against
librosa's 8.125), so anything built on a backend's onset list would report a
different drum pattern depending on which wheel happened to install. Band
energy, flux, decay and flatness are exact arithmetic that numpy can do
directly, so this module owns them and every backend yields identical drum
output. `tests/test_drum_elements.py` pins that with both libraries blocked out
of `sys.modules`.

The rule this follows is the project's architecture rule: put it behind the
backend Protocol only when the computation is an algorithm this project should
not own; put it in a shared numpy module when it is a measurement numpy can
make exactly.


Why per-band detection
----------------------

`RhythmFeatures.onset_times` is a **single global list of instants**. Kick and
hat routinely land together, so classifying that list assigns one class per
instant and systematically deletes the hat on every downbeat. In
`tests/conftest.py::drum_pattern_120bpm` that is 16 of 48 hits — the fixture has
48 hits at only 32 distinct instants, and a design working from one global list
can find at most 32 of them.

So this module computes a **separate spectral flux per detection band and
peak-picks each band independently**. A coincident kick and hat are two hits
because two detectors found them. `rhythm.onset_times` keeps its own meaning
(density, subdivision feel) and is never read here.


The detection bands, and why they are not `BAND_EDGES_HZ`
---------------------------------------------------------

`BAND_EDGES_HZ` splits at 250 / 2000 / 6000 Hz, which puts a kick's fundamental
and a snare's 200 Hz shell tone in the *same* `low` band. That scheme cannot
separate kick from snare no matter how it is thresholded. `DETECTION_BANDS`
splits where the drums do:

===== ============ =========================================================
band  Hz           what lives there
===== ============ =========================================================
kick  20 - 150     kick fundamental and its pitch sweep
body  150 - 500    snare shell tone, tom body, the part of a snare that is
                   not rattle
noise 1k - 6k      snare rattle, clap, broadband transient content
air   6k - 16k     hats, cymbals, the top of a bright transient
===== ============ =========================================================

The gaps (500 Hz - 1 kHz, above 16 kHz) are deliberate: nothing that separates
these three classes lives there, and excluding them keeps each ratio a share of
material that is actually diagnostic.

`_band_energy()` mirrors `backends.librosa_backend.band_energy_ratios()`
exactly — bins assigned by centre frequency, half-open `[low, high)` with the
topmost bound closed, energy as `magnitude ** 2` summed over frames — so the
two cannot drift apart. `tests/test_drum_elements.py` pins that numerically by
handing this module's helper that function's own bounds. That test imports
`band_energy_ratios`; **this module must not**, and does not.


Measurement conventions (WP-CAL: these are the numbers to recalibrate against)
-----------------------------------------------------------------------------

Everything is measured on one magnitude STFT: `STFT_N_FFT` 2048,
`STFT_HOP_LENGTH` 512, periodic Hann, centred by zero-padding `n_fft // 2` on
each side, at 44,100 Hz. Power is `magnitude ** 2`. A band envelope is that
power summed over the band's bins, per frame.

*Detection.* Per band, flux is the half-wave-rectified first difference of
`log1p(envelope)`. The log domain is for level invariance, the same reasoning
behind `librosa_backend.ONSET_ENVELOPE_FLOOR`. A frame is a peak when it
exceeds a rolling median over +/- `FLUX_MEDIAN_HALF_SECONDS` (mirroring
`TRANSIENT_WINDOW_SECONDS`), is the largest flux within +/-
`MIN_HIT_SEPARATION_SECONDS`, and clears that rolling median by
`PEAK_DELTA_FRACTION` of the band's own `PEAK_REFERENCE_QUANTILE` of candidate
flux values.

*Per-hit features.* The window runs from the hit's frame to
`FEATURE_WINDOW_SECONDS` later, truncated at the next hit **in the same band**.
Band ratios are that window's band energies normalised to sum to 1.
`decay_ratio` is the RMS of the *detecting band's* envelope over the first
`ATTACK_SECONDS` divided by the RMS over the rest of the window, clamped at
`MAX_DECAY_RATIO` exactly as `MAX_TRANSIENT_SHARPNESS` clamps. `flatness` is
the geometric over arithmetic mean of the window's summed power spectrum across
`FLATNESS_RANGE_HZ`, floored `FLATNESS_FLOOR` below the peak bin so it is
scale-invariant.

**Measured within exactly those conventions**, on the one-shots of
`tests/conftest.py`, each alone in an 8.5 s silent buffer with its first sample
at 0.25 s. The placement is part of the convention, not incidental: a one-shot
starting at t=0 has no preceding frame to have risen from and produces zero
flux. Every row is the reading from that one-shot's own dominant band, and
`tests/test_drum_elements.py` pins all of them.

========= ====== ======= ======= ======= ======= ======= ======== =========
one-shot  band   kick    body    noise   air     decay   air/     flatness
                                                 _ratio  (air+
                                                         noise)
========= ====== ======= ======= ======= ======= ======= ======== =========
kick      kick   0.9928  0.0072  0.0000  0.0000    2.01  0.0000   9.0e-07
snare     body   0.0118  0.5939  0.3931  0.0012    2.59  0.0030   2.5e-03
hat       air    0.0000  0.0000  0.0012  0.9988   10.36  0.9988   4.9e-02
open hat  air    0.0000  0.0000  0.0009  0.9991    1.47  0.9991   4.1e-02
click     noise  0.0152  0.0508  0.5468  0.3872  100.00  0.4146   5.2e-01
========= ====== ======= ======= ======= ======= ======= ======== =========

Four of those numbers contradict the obvious reading, and each one drives a
design decision:

* **The click is noisier than the snare** (0.5468 against 0.3931) and has 300x
  its air (0.3872 against 0.0012). A snare score keyed on `noise_ratio` calls
  every broadband click a snare, which is exactly what
  `drum_pattern_ambiguous` exists to catch. `body_ratio` is the discriminator:
  0.5939 against 0.0508, a 12x separation.
* **The isolated snare's air share, 0.0012, is enough to fire the air
  detector**, and its air-band decay of 1.94 is indistinguishable from an open
  hat's 1.47. A snare-only stem reported a hat on every snare until
  `hat_air_over_noise` existed. None of the four drum fixtures can show this:
  they either put a hat on every snare or contain no snares at all.
* **The kick's `decay_ratio` is 2.01, not some large number.** It is a 300 ms
  one-shot with a 90 ms decay constant, so over a 250 ms window it is mostly
  tail. `decay_ratio` measures how front-loaded a hit is, not how long it
  rings, and only the *air* band's reading discriminates anything here.
* **Flatness spans six decades across five one-shots**, 9.0e-07 to 0.52, and it
  is violently window-sensitive: WP0 measured this same open hat at 0.0277 over
  its own length, an earlier probe at 6.0e-05 over a 93 ms window, and this
  convention gives 0.041. Any absolute threshold on it is meaningless. It is
  reported for audit and deliberately **not** used to classify.


How a hit is classified
-----------------------

Three scores, each built from `heuristics.ramp` so a hit sitting exactly on a
threshold scores 0.0 and climbs to 1.0 — the project's graded-confidence
convention, one implementation of it. Each score is the product of a **shape
term** measured from the hit and a **detector affinity**: how likely a hit found
by *that band's* detector is to belong to that class. Kick and snare read one
descriptor each (`kick_ratio` and `body_ratio`); the hat takes the weaker of two
(`decay_ratio` and `air / (air + noise)`), combined as a minimum in the same
weakest-link way `heuristics._all` combines conditions.

Before any of that, a band is only searched at all when it holds real energy
(`BAND_ACTIVITY_FLOOR`) **and** its flux looks like hits rather than a level or
a rumble (`_is_percussive`). Both gates exist because an *adaptive* threshold is
by construction happy to threshold noise: without the first, a kick-only stem
grows a full hat pattern out of float residue; without the second, a 20-200 Hz
rumble reports 128 kicks.

The affinity is load-bearing, not decoration. At an instant where a kick and a
hat sound together, every instant-level measurement is shared between the two
detectors — the window band ratios are the kick's, because the kick carries
roughly 100x the hat's energy. The only thing that differs is which band found
the hit. Dropping the affinity term collapses both detections onto one class
and deletes the hat, which is the failure this module exists to avoid.

Decision is argmax, but only when the winner clears `DECISION_FLOOR` **and**
beats the runner-up by `DECISION_MARGIN`. Otherwise the hit is `unclassified`
with its honest winner confidence and its timing intact — the same
best-minus-best-rival logic `librosa_backend._estimate_key` uses, for the same
reason: an argmax without a margin is a coin toss dressed as a decision.

Two post-passes then run, both consequences of detecting per band:

* Two hits of the same class within `MIN_HIT_SEPARATION_SECONDS` are **one**
  hit (the more confident one). A snare fires the body *and* noise detectors;
  that is one snare, not two.
* An `unclassified` hit within `MIN_HIT_SEPARATION_SECONDS` of a **classified**
  hit is dropped as leakage. A kick puts a little energy in the body band, and
  the body detector duly fires; that bump is the kick, already reported. The
  cost is honest and documented: a tom struck exactly with a hat is swallowed
  by the hat.


What this cannot do
-------------------

Three classes cannot describe a real kit. Toms, rides, crashes, claps and
shakers all land in `unclassified`, and on percussive material that bucket will
be large and will *look* like failure while being correct. Every threshold here
was calibrated on synthetic one-shots; real stems carry bleed and pre-echo, and
a real kick has beater click at 1-4 kHz that pushes `noise_ratio` up. WP-CAL
recalibrates against real material.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from . import ANALYSIS_SAMPLE_RATE
from .heuristics import ramp
from .schemas import DrumDecomposition, DrumHit, DrumPattern

__all__ = [
    "ATTACK_SECONDS",
    "BAND_ACTIVITY_FLOOR",
    "DECISION_FLOOR",
    "DECISION_MARGIN",
    "DETECTION_BANDS",
    "DETECTOR_CLASS_AFFINITY",
    "FEATURE_WINDOW_SECONDS",
    "FLATNESS_FLOOR",
    "FLATNESS_RANGE_HZ",
    "FLUX_MEDIAN_HALF_SECONDS",
    "FLUX_NEAR_ZERO_FRACTION",
    "FLUX_PEAK_FLOOR",
    "FLUX_SPARSITY_MIN",
    "GRID_IMPROVEMENT_FRACTION",
    "GRID_STEP_CANDIDATES",
    "MAX_DECAY_RATIO",
    "MAX_QUANTISATION_ERROR_STEPS",
    "MIN_HIT_SEPARATION_SECONDS",
    "PEAK_DELTA_FRACTION",
    "PEAK_REFERENCE_QUANTILE",
    "STFT_HOP_LENGTH",
    "STFT_N_FFT",
    "THRESHOLDS",
    "decompose",
]


# ---------------------------------------------------------------------------
# The pinned analysis grid
# ---------------------------------------------------------------------------

#: STFT window length in samples. Matches `librosa_backend.STFT_N_FFT` and
#: `essentia_backend.STFT_N_FFT`: ~46 ms and ~21.5 Hz per bin at 44.1 kHz.
STFT_N_FFT: Final[int] = 2048

#: STFT hop in samples: ~11.6 ms at 44.1 kHz, and therefore the time resolution
#: of every hit reported here. Matches both backends.
STFT_HOP_LENGTH: Final[int] = 512

#: Drum detection bands in Hz. **Deliberately not `BAND_EDGES_HZ`** — see the
#: module docstring for why that scheme cannot separate kick from snare. The
#: dict order is the reporting order of the ratio fields on `DrumHit`.
DETECTION_BANDS: Final[dict[str, tuple[float, float]]] = {
    "kick": (20.0, 150.0),
    "body": (150.0, 500.0),
    "noise": (1000.0, 6000.0),
    "air": (6000.0, 16000.0),
}

#: Top of the union of the detection bands. Bins landing exactly on it are
#: counted rather than dropped, mirroring `band_energy_ratios`' treatment of its
#: own ceiling.
_BAND_CEILING_HZ: Final[float] = max(high for _, high in DETECTION_BANDS.values())

#: Range over which spectral flatness is measured, and the floor applied to the
#: power spectrum first, relative to its own peak bin (120 dB down). Without the
#: floor the deep nulls of a band-limited signal decide a geometric mean on
#: their own. Matches the convention `tests/test_fixtures.py` uses on the
#: one-shots, so the two sets of numbers are comparable.
FLATNESS_RANGE_HZ: Final[tuple[float, float]] = (20.0, 16000.0)
FLATNESS_FLOOR: Final[float] = 1e-12


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Same candour convention as `heuristics.THRESHOLDS`: `[grounded]` means the
# number follows from something physical, arithmetic or *measured on the
# fixtures*; `[guess]` means it is a plausible starting point for WP-CAL.
#
# This dict is deliberately separate from `heuristics.THRESHOLDS`. That one is
# mirrored by `tests/test_heuristics.py::RAMP_DIRECTIONS`, which requires a row
# per `*_saturation` key; drum thresholds live here with their own mirror test
# in `tests/test_drum_elements.py`, so neither file has to know about the other.

THRESHOLDS: Final[dict[str, float]] = {
    # -- Detection ---------------------------------------------------------
    # [grounded] Mirrors `librosa_backend.TRANSIENT_WINDOW_SECONDS`: roughly one
    # beat either side at moderate tempi, which is long enough that a rolling
    # median tracks the passage rather than the hit.
    "flux_median_half_seconds": 0.5,
    # [grounded] 30 ms is about 2.5 STFT hops, so it is the shortest separation
    # this grid can actually resolve. Applied *within* a band only: two bands
    # firing at the same instant are two hits, which is the whole design.
    "min_hit_separation_seconds": 0.030,
    # [grounded, measured] A peak must clear its rolling median by this fraction
    # of the band's reference flux. Measured working range on all four drum
    # fixtures simultaneously is 0.30-0.50; below 0.30 the body detector starts
    # firing on the 300 ms truncation click at the end of `conftest._kick`
    # (measured flux 1.73 against a real kick's 4.14-7.36), and above 0.50 the
    # kick detector starts missing kicks in `drum_pattern_kick_only`. 0.40 is
    # the middle of that range.
    "peak_delta_fraction": 0.40,
    # [grounded, measured] Which quantile of a band's candidate flux values is
    # the band's reference. The maximum was tried first and works only in a
    # single-point window (0.25 +/- nothing): in `drum_pattern_open_hats` the
    # first hat rises from silence and fluxes 4.685 while every later hat lands
    # on the previous hat's tail and fluxes 1.35-1.82, so the max is a 3.5x
    # outlier and a fraction of it silently drops two thirds of the hats. The
    # 90th percentile is not.
    "peak_reference_quantile": 0.90,
    # [guess] A band holding less than this share of the source's in-band energy
    # is not searched at all. This is a numerical-noise gate, not a musical one:
    # in `drum_pattern_kick_only` the air band holds 3e-07 of the energy (pure
    # float residue under a 60 Hz sine) and its flux, adaptively thresholded
    # against its own noise floor, otherwise produces 32 phantom hats. Measured
    # air shares on material that does have hats: 4.5e-03 to 5.1e-02. There is a
    # factor of 4.5 of headroom above and 3000 below, so the exact value is not
    # delicate — but it is the only thing standing between a hatless stem and an
    # invented hat pattern, and WP-CAL should check it on a real kick-only stem.
    "band_activity_floor": 1e-3,
    # [grounded, measured] A band whose loudest flux is below this has no
    # transients in it at all, only a level. In `log1p` units a flux of 1.0 is
    # the band's energy rising by a factor of e - 1 ~ 1.7 in one hop, so this
    # asks for very little. It exists for the case `ONSET_ENVELOPE_FLOOR` exists
    # for in the librosa backend: a sustained tone has a flat envelope, its flux
    # is float residue, and a detector thresholding adaptively against residue
    # reports a stream of phantom hits. Measured band peak flux: an 8 s 440 Hz
    # sine 1e-10, band-limited noise 1.8-2.4, every drum fixture 4.69-9.56.
    "flux_peak_floor": 1.0,
    # [grounded, measured] Share of a band's frames whose flux is at or below
    # `flux_near_zero_fraction` of that band's peak flux, before the band is
    # searched at all. This is a **sparsity** test and it is what separates a
    # drum pattern from a rumble: a stationary process fluctuates constantly, so
    # roughly half its frames carry appreciable flux, while a percussive band is
    # quiet between hits. Measured: `bass_unvoiced` 0.520-0.520 and white noise
    # 0.520-0.600, against 0.847-0.950 across all four drum fixtures. 0.72 is
    # the midpoint of that gap.
    #
    # The cost is real and worth stating: a band this dense *is* gated off, so a
    # 32nd-note hat roll with no gaps would be missed. Estimated crossover is
    # around 16 hits per second at 44.1 kHz, which is a 32nd-note roll at 120
    # BPM; 16ths measure about 0.81 and still pass.
    "flux_sparsity_min": 0.72,
    "flux_near_zero_fraction": 0.02,
    # -- Per-hit features --------------------------------------------------
    # [guess] How long after a hit its features are measured, truncated at the
    # next hit in the same band. Long enough to see an open hat's tail (150 ms
    # decay constant), short enough that at 16th notes and 120 BPM the
    # truncation, not this number, sets the window.
    "feature_window_seconds": 0.25,
    # [grounded] The attack half of `decay_ratio`, in seconds. 35 ms is three
    # STFT hops, the shortest attack this grid can separate from its own tail.
    "attack_seconds": 0.035,
    # [grounded] Mirrors `librosa_backend.MAX_TRANSIENT_SHARPNESS`. A hit with a
    # digitally silent tail has an infinite ratio; reporting the ceiling keeps
    # the descriptor finite and JSON-serialisable.
    "max_decay_ratio": 100.0,
    # -- Classification ----------------------------------------------------
    # [grounded, measured] Share of the hit window's band energy below 150 Hz
    # for a kick. Measured: real kicks 0.979-0.992 in every fixture; the
    # loudest non-kick reading in the kick band is 0.42, from a `_click` landing
    # inside a kick's decay tail in `drum_pattern_ambiguous`. 0.60 sits in the
    # middle of that gap in log terms and gives every real kick full confidence.
    "kick_low_ratio": 0.60,
    "kick_low_ratio_saturation": 0.90,
    # [grounded, measured] Share of the hit window's band energy in 150-500 Hz
    # for a snare. **This, not `noise_ratio`, is the snare discriminator.**
    # Measured: snares 0.530-0.600; every other detected hit in every fixture is
    # at or below 0.051 (kick leakage 0.0078, click 0.031-0.051, hat leakage
    # 0.0007). A 10x gap on both sides of 0.25.
    "snare_body_ratio": 0.25,
    "snare_body_ratio_saturation": 0.55,
    # [grounded, measured] `decay_ratio` in the *detecting* band at or below
    # which a hit has a hat's tail rather than a click's. Measured in the air
    # band across all four drum fixtures: closed hats 5.40-12.49, open hats
    # 1.46-1.72, `_click` 20.55-100.0. 16.0 is the geometric midpoint of 12.49
    # and 20.55. Lower is more confident, so this pair ramps downwards — see
    # `heuristics._ramp`'s footgun note and move the two together.
    #
    # The closed hat's 2.3x internal spread is **sub-frame phase**, not noise:
    # `_hat_closed` has a 15 ms decay constant against an 11.6 ms hop, so it is
    # essentially gone within two frames and the split between attack and tail
    # depends on where inside a frame the hit landed. Hats 0.25 s apart advance
    # 21.53 frames each time, and the measured ratio drifts smoothly with that
    # 0.53-frame residue. Widening `attack_seconds` does not help — measured at
    # 4 frames the closed hats spread 8.55-24.56 and start overlapping the
    # clicks; at 2 frames they spread 2.37-6.51 against clicks from 3.16, which
    # overlaps outright. Three frames is the measured optimum.
    "hat_decay_ratio": 16.0,
    "hat_decay_ratio_saturation": 3.0,
    # [grounded, measured] The hat's second condition: `air / (air + noise)` of
    # the hit window, i.e. how much of the bright half of the spectrum is
    # genuinely *air* rather than the top of a 1-6 kHz rattle.
    #
    # This exists because of a hallucination none of the four drum fixtures can
    # show, since all four either have hats on every snare or have no snares at
    # all: **an isolated `_snare` fires the air detector**. Its air share is
    # 0.0012, which clears the band-activity floor, and its air-band decay is
    # 1.94, which reads as a perfectly good open hat. Measured, a snare-only
    # stem reported a hat on every snare until this term existed.
    #
    # Measured `air / (air + noise)`: isolated `_snare` 0.0030; the tightest
    # real hat in any fixture (a closed hat sounding *with* a snare) 0.0552;
    # open hat with a snare 0.2846; hat alone 0.9988; hat over a kick 1.0. The
    # threshold sits at 0.015, a factor of 5 above the snare and 3.7 below the
    # tightest hat.
    #
    # It is the thinnest margin in this module and the one WP-CAL should attack
    # first: a real snare is far brighter than this synthetic one, and a genuinely
    # crisp snare with no hat over it may well cross 0.015.
    "hat_air_over_noise": 0.015,
    "hat_air_over_noise_saturation": 0.060,
    # [guess] The winning score must reach this, and beat the runner-up by the
    # margin, or the hit is `unclassified`. Inherited from the plan; measured
    # winners on the fixtures are 0.38-1.00 for classified hits and exactly 0.0
    # for every `_click`, so nothing currently sits near either number and
    # neither is yet tested by real material.
    "decision_floor": 0.15,
    "decision_margin": 0.10,
    # -- Grid --------------------------------------------------------------
    # [grounded, measured] Mean absolute distance from a hit to its nearest
    # step, in steps, above which no grid is reported at all.
    #
    # **0.25 would be the wrong number**, even though it looks like the obvious
    # one: hits falling uniformly at random score exactly 0.25 by construction,
    # so a limit of 0.25 accepts any fit at all that is better than chance.
    # Measured, `drum_pattern_120bpm` fitted to a deliberately wrong 97.3 BPM
    # scores 0.2464 and would have been reported as a grid. The fixtures fitted
    # to their own tempo score 0.03-0.09, so 0.18 leaves them a factor of two of
    # headroom while requiring a fit meaningfully better than random.
    "max_quantisation_error_steps": 0.18,
    # [grounded] How much better a later `GRID_STEP_CANDIDATES` entry must fit
    # before it displaces an earlier one, as a fraction of the incumbent's
    # error. Not decoration: a quarter-note pattern lands exactly on both a 16-
    # and a 12-step grid, so the choice would otherwise be decided by float
    # noise — `drum_pattern_kick_only` measured 12 steps per cycle and reported
    # its kicks on steps 0/3/6/9 instead of 0/4/8/12. 10% asks a triplet grid to
    # actually fit better before this module claims a triplet feel.
    "grid_improvement_fraction": 0.10,
    # [guess] Share of hits that may be `unclassified` before that fact is
    # raised as a caveat rather than left to the reader to notice.
    "unclassified_caveat_fraction": 0.25,
}

FLUX_MEDIAN_HALF_SECONDS: Final[float] = THRESHOLDS["flux_median_half_seconds"]
MIN_HIT_SEPARATION_SECONDS: Final[float] = THRESHOLDS["min_hit_separation_seconds"]
PEAK_DELTA_FRACTION: Final[float] = THRESHOLDS["peak_delta_fraction"]
PEAK_REFERENCE_QUANTILE: Final[float] = THRESHOLDS["peak_reference_quantile"]
BAND_ACTIVITY_FLOOR: Final[float] = THRESHOLDS["band_activity_floor"]
FLUX_PEAK_FLOOR: Final[float] = THRESHOLDS["flux_peak_floor"]
FLUX_SPARSITY_MIN: Final[float] = THRESHOLDS["flux_sparsity_min"]
FLUX_NEAR_ZERO_FRACTION: Final[float] = THRESHOLDS["flux_near_zero_fraction"]
FEATURE_WINDOW_SECONDS: Final[float] = THRESHOLDS["feature_window_seconds"]
ATTACK_SECONDS: Final[float] = THRESHOLDS["attack_seconds"]
MAX_DECAY_RATIO: Final[float] = THRESHOLDS["max_decay_ratio"]
DECISION_FLOOR: Final[float] = THRESHOLDS["decision_floor"]
DECISION_MARGIN: Final[float] = THRESHOLDS["decision_margin"]
MAX_QUANTISATION_ERROR_STEPS: Final[float] = THRESHOLDS["max_quantisation_error_steps"]
GRID_IMPROVEMENT_FRACTION: Final[float] = THRESHOLDS["grid_improvement_fraction"]

#: How likely a hit found by each band's detector is to belong to each class,
#: as a multiplier on that class's shape score.
#:
#: Read a row as "what can make this band's detector fire". The kick band fires
#: on kicks and, weakly, on a snare's shell tone; the noise band fires on a
#: snare's rattle and on a hat's lower reaches; nothing in the air band is ever
#: a kick.
#:
#: A zero is a hard veto and is used only where the physics allows it: a hit
#: found by the air detector cannot be a kick, because a kick has no measurable
#: energy above 6 kHz.
#:
#: These are `[guess]` values, but the *ordering* is grounded: it follows the
#: measured band shares of the `tests/conftest.py` one-shots quoted in the
#: module docstring. WP-CAL should retune the magnitudes, not the zeroes.
DETECTOR_CLASS_AFFINITY: Final[dict[str, dict[str, float]]] = {
    "kick": {"kick": 1.0, "snare": 0.4, "hat": 0.0},
    "body": {"kick": 0.4, "snare": 1.0, "hat": 0.0},
    "noise": {"kick": 0.0, "snare": 1.0, "hat": 0.5},
    "air": {"kick": 0.0, "snare": 0.3, "hat": 1.0},
}

#: Steps per cycle to try, in preference order on a tie. 16 is straight 16ths;
#: 12 is a triplet or shuffle feel. Nothing else is offered: a grid this module
#: cannot justify is worse than no grid, which is the same bias `strudel_hints`
#: takes.
GRID_STEP_CANDIDATES: Final[tuple[int, ...]] = (16, 12)

#: The classes `_class_scores` scores, in reporting order. `unclassified` is not
#: here because nothing scores *for* it — it is what a failed decision leaves.
_SCORED_CLASSES: Final[tuple[str, ...]] = ("kick", "snare", "hat")


# ---------------------------------------------------------------------------
# Spectral primitives
# ---------------------------------------------------------------------------


def _stft_magnitude(
    audio: npt.NDArray[np.floating[Any]], sample_rate: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Magnitude STFT on the pinned grid, plus its bin centre frequencies.

    Periodic Hann of `STFT_N_FFT`, hop `STFT_HOP_LENGTH`, centred by padding
    `STFT_N_FFT // 2` zeros on each side so frame `k` is centred on sample
    `k * STFT_HOP_LENGTH`. Zero padding rather than reflection: reflection
    invents a mirror-image transient at the start of the file, which a flux
    detector then reports as a hit.

    Returns:
        `(magnitude, freqs)` with magnitude shaped `(n_bins, n_frames)` to match
        what `band_energy_ratios` expects, and `freqs` shaped `(n_bins,)`.
        Both are empty when the input is too short to make a single frame.
    """
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=tuple(range(1, mono.ndim)))
    # A NaN or an infinity is not audio. Zeroing them keeps the FFT finite and
    # warning-free; propagating them would poison every band of every frame the
    # window touches and turn one bad sample into a silent whole-source failure.
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    freqs = np.fft.rfftfreq(STFT_N_FFT, 1.0 / float(sample_rate))
    if mono.size == 0:
        return np.zeros((freqs.size, 0), dtype=np.float64), freqs

    pad = STFT_N_FFT // 2
    padded = np.pad(mono, (pad, pad))
    n_frames = 1 + (padded.size - STFT_N_FFT) // STFT_HOP_LENGTH
    if n_frames < 1:
        return np.zeros((freqs.size, 0), dtype=np.float64), freqs

    starts = STFT_HOP_LENGTH * np.arange(n_frames)
    indices = starts[:, np.newaxis] + np.arange(STFT_N_FFT)[np.newaxis, :]
    # Periodic Hann, matching what an STFT library uses (np.hanning is
    # symmetric, which is a window for filter design, not for analysis).
    window = np.hanning(STFT_N_FFT + 1)[:-1]
    frames = padded[indices] * window
    return np.abs(np.fft.rfft(frames, axis=1)).T, freqs


def _band_envelope(
    magnitude: npt.NDArray[np.floating[Any]],
    freqs: npt.NDArray[np.floating[Any]],
    low_hz: float,
    high_hz: float,
    *,
    include_high: bool = False,
) -> npt.NDArray[np.float64]:
    """Per-frame `magnitude ** 2` summed over the bins of one band.

    The single place this module decides which bin belongs to which band, and it
    decides it exactly as `backends.librosa_backend.band_energy_ratios()` does:
    a bin belongs to the band containing its **centre frequency**, intervals are
    half-open `[low, high)`, and energy is `magnitude ** 2`. No bin is split and
    nothing is interpolated across an edge.

    Args:
        magnitude: Linear magnitude spectrum `(n_bins, n_frames)`; a 1-D
            `(n_bins,)` spectrum is treated as a single frame.
        freqs: Bin centre frequencies in Hz, `(n_bins,)`, ascending.
        low_hz: Inclusive lower bound.
        high_hz: Exclusive upper bound, unless `include_high`.
        include_high: Close the interval at the top, `[low, high]`. Used for the
            topmost band of a scheme so a bin landing exactly on the ceiling is
            counted rather than dropped.

    Returns:
        `(n_frames,)` of float64, empty when the inputs do not line up.
    """
    spectrum = np.asarray(magnitude, dtype=np.float64)
    frequencies = np.asarray(freqs, dtype=np.float64)
    if spectrum.ndim == 1:
        spectrum = spectrum[:, np.newaxis]
    if spectrum.ndim != 2 or spectrum.shape[0] != frequencies.shape[0]:
        return np.zeros(0, dtype=np.float64)
    if include_high:
        in_band = (frequencies >= low_hz) & (frequencies <= high_hz)
    else:
        in_band = (frequencies >= low_hz) & (frequencies < high_hz)
    summed: npt.NDArray[np.float64] = np.square(spectrum[in_band]).sum(axis=0)
    return summed


def _band_energy(
    magnitude: npt.NDArray[np.floating[Any]],
    freqs: npt.NDArray[np.floating[Any]],
    low_hz: float,
    high_hz: float,
    *,
    include_high: bool = False,
) -> float:
    """Total `magnitude ** 2` in one band, summed over every frame.

    `_band_envelope` before the sum over time. Handed `BAND_EDGES_HZ`' bounds
    this reproduces `band_energy_ratios()`' numerator exactly, which
    `tests/test_drum_elements.py` pins numerically.
    """
    return float(_band_envelope(magnitude, freqs, low_hz, high_hz, include_high=include_high).sum())


def _band_envelopes(
    magnitude: npt.NDArray[np.floating[Any]], freqs: npt.NDArray[np.floating[Any]]
) -> dict[str, npt.NDArray[np.float64]]:
    """One envelope per `DETECTION_BANDS` band, keyed by band name."""
    return {
        name: _band_envelope(
            magnitude, freqs, low, high, include_high=high >= _BAND_CEILING_HZ
        )
        for name, (low, high) in DETECTION_BANDS.items()
    }


def _spectral_flux(envelope: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Half-wave-rectified first difference of `log1p(envelope)`.

    The log domain makes the flux depend on the *ratio* by which a band's energy
    rose rather than the absolute amount, so one threshold works across a quiet
    passage and a loud one. Same reasoning as
    `librosa_backend.ONSET_ENVELOPE_FLOOR`.

    Frame 0 is defined as 0.0: there is no preceding frame to have risen from.
    That is why every fixture in `tests/conftest.py` starts at 0.25 s rather
    than at t=0.
    """
    if envelope.size == 0:
        return np.zeros(0, dtype=np.float64)
    rise = np.diff(np.log1p(np.maximum(envelope, 0.0)))
    return np.concatenate(([0.0], np.maximum(rise, 0.0)))


def _rolling_median(values: npt.NDArray[np.float64], half_width: int) -> npt.NDArray[np.float64]:
    """Median over `+/- half_width` samples, shrinking at both ends.

    Shrinking rather than edge-padded: padding with the edge value biases the
    first half-second of the file towards whatever happened to be in frame 0,
    and every fixture here puts its first hit inside that region.
    """
    if values.size == 0:
        return values
    half = max(0, int(half_width))
    if half == 0:
        return values.copy()
    padded = np.full(values.size + 2 * half, np.nan, dtype=np.float64)
    padded[half : half + values.size] = values
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    median: npt.NDArray[np.float64] = np.nanmedian(windows, axis=-1).astype(np.float64)
    return median


def _frames(seconds: float, sample_rate: int, *, minimum: int = 1) -> int:
    """Convert seconds to whole STFT frames, never below `minimum`."""
    return max(minimum, int(round(seconds * sample_rate / STFT_HOP_LENGTH)))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _pick_peaks(flux: npt.NDArray[np.float64], sample_rate: int) -> npt.NDArray[np.intp]:
    """Peak-pick one band's flux. Returns frame indices, ascending.

    Three conditions, in order:

    1. above a rolling median over `+/- FLUX_MEDIAN_HALF_SECONDS`;
    2. the largest flux within `+/- MIN_HIT_SEPARATION_SECONDS`, with ties
       resolved by keeping the earlier frame, then greedily by amplitude, so the
       result does not depend on iteration order;
    3. above that rolling median by `PEAK_DELTA_FRACTION` of the band's
       reference flux, which is the `PEAK_REFERENCE_QUANTILE` quantile of the
       candidates surviving (1) and (2).

    Condition 3 is computed *after* 1 and 2 on purpose: the reference has to be
    a quantile of plausible hits, not of every frame, most of which are silence.
    """
    n = flux.size
    if n < 3:
        return np.zeros(0, dtype=np.intp)

    median_half = _frames(FLUX_MEDIAN_HALF_SECONDS, sample_rate)
    separation = _frames(MIN_HIT_SEPARATION_SECONDS, sample_rate)
    median = _rolling_median(flux, median_half)

    width = 2 * separation + 1
    padded = np.full(n + 2 * separation, -np.inf, dtype=np.float64)
    padded[separation : separation + n] = flux
    local_max = np.lib.stride_tricks.sliding_window_view(padded, width).max(axis=-1)

    candidates: list[int] = []
    for index in range(1, n - 1):
        if flux[index] <= median[index] or flux[index] < local_max[index]:
            continue
        if candidates and index - candidates[-1] < separation:
            if flux[index] > flux[candidates[-1]]:
                candidates[-1] = index
            continue
        candidates.append(index)
    if not candidates:
        return np.zeros(0, dtype=np.intp)

    reference = float(np.quantile(flux[candidates], PEAK_REFERENCE_QUANTILE))
    if not math.isfinite(reference) or reference <= 0.0:
        return np.zeros(0, dtype=np.intp)
    delta = PEAK_DELTA_FRACTION * reference
    kept = [index for index in candidates if flux[index] > median[index] + delta]
    return np.asarray(kept, dtype=np.intp)


def _is_percussive(flux: npt.NDArray[np.float64]) -> bool:
    """Whether a band's flux looks like hits rather than a level or a rumble.

    Two tests, each aimed at a different way a peak picker invents a pattern out
    of nothing, and both needed because neither catches the other's case:

    * **A transient at all.** The band's loudest flux must reach
      `FLUX_PEAK_FLOOR`. A sustained tone has a flat envelope, so its flux is
      float residue and the adaptive threshold — being adaptive — happily
      thresholds residue. Measured: an 8 s 440 Hz sine peaks at 1e-10, every
      drum fixture at 4.69 and above. Same failure `ONSET_ENVELOPE_FLOOR`
      exists for in the librosa backend.
    * **Sparsity.** At least `FLUX_SPARSITY_MIN` of frames must sit at or below
      `FLUX_NEAR_ZERO_FRACTION` of the band's peak flux. A stationary process
      fluctuates on roughly half its frames; a percussive band is quiet between
      hits. Measured: band-limited noise 0.520, white noise 0.520-0.600, drum
      fixtures 0.847-0.950. Without this, `bass_unvoiced` — a 20-200 Hz rumble
      — reports 128 hits, 119 of them kicks.
    """
    if flux.size == 0:
        return False
    peak = float(flux.max())
    if not math.isfinite(peak) or peak < FLUX_PEAK_FLOOR:
        return False
    near_zero = float(np.mean(flux <= FLUX_NEAR_ZERO_FRACTION * peak))
    return near_zero >= FLUX_SPARSITY_MIN


def _active_bands(
    envelopes: dict[str, npt.NDArray[np.float64]], fluxes: dict[str, npt.NDArray[np.float64]]
) -> tuple[list[str], list[str]]:
    """Split the detection bands into those worth searching and those that are not.

    A band is searched when it holds at least `BAND_ACTIVITY_FLOOR` of the
    source's in-band energy **and** its flux passes `_is_percussive`. The first
    test rejects numerical residue (a band that is not really there); the second
    rejects a band that is there but holds no hits.

    Returns:
        `(active, dormant)`, both in `DETECTION_BANDS` order.
    """
    totals = {name: float(envelope.sum()) for name, envelope in envelopes.items()}
    grand = sum(totals.values())
    if not math.isfinite(grand) or grand <= 0.0:
        return [], list(DETECTION_BANDS)
    active = [
        name
        for name in DETECTION_BANDS
        if totals[name] / grand >= BAND_ACTIVITY_FLOOR and _is_percussive(fluxes[name])
    ]
    dormant = [name for name in DETECTION_BANDS if name not in active]
    return active, dormant


# ---------------------------------------------------------------------------
# Per-hit features
# ---------------------------------------------------------------------------


class _Candidate:
    """One detected hit, before classification. Deliberately not a pydantic model.

    `DrumHit` is the output contract; this is working state, and keeping them
    separate means the intermediate fields (which band found it, which frame)
    cannot leak into JSON.
    """

    __slots__ = (
        "band",
        "confidence",
        "decay_ratio",
        "drum",
        "flatness",
        "frame",
        "ratios",
        "scores",
    )

    def __init__(self, frame: int, band: str) -> None:
        self.frame = frame
        self.band = band
        self.ratios: dict[str, float | None] = dict.fromkeys(DETECTION_BANDS, None)
        self.decay_ratio: float | None = None
        self.flatness: float | None = None
        self.scores: dict[str, float] = dict.fromkeys(_SCORED_CLASSES, 0.0)
        self.drum: str = "unclassified"
        self.confidence: float = 0.0


def _window_end(candidate: _Candidate, candidates: list[_Candidate], limit: int, span: int) -> int:
    """Last frame (exclusive) of a candidate's feature window.

    `FEATURE_WINDOW_SECONDS` after the hit, truncated at the next hit **in the
    same band** so a measurement never runs into the next hit of the same
    instrument. Truncating on the next hit in *any* band would make a hat's
    window depend on where the kicks are.
    """
    end = min(candidate.frame + span, limit)
    for other in candidates:
        if other.frame > candidate.frame and other.band == candidate.band:
            end = min(end, other.frame)
            break
    return max(candidate.frame + 1, end)


def _measure(
    candidates: list[_Candidate],
    magnitude: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    envelopes: dict[str, npt.NDArray[np.float64]],
    sample_rate: int,
) -> None:
    """Fill in every candidate's ratios, decay and flatness, in place."""
    n_frames = magnitude.shape[1]
    span = _frames(FEATURE_WINDOW_SECONDS, sample_rate)
    attack = _frames(ATTACK_SECONDS, sample_rate)
    low_hz, high_hz = FLATNESS_RANGE_HZ
    in_range = (freqs >= low_hz) & (freqs <= high_hz)

    for candidate in candidates:
        start = candidate.frame
        end = _window_end(candidate, candidates, n_frames, span)

        per_band = {name: float(envelopes[name][start:end].sum()) for name in DETECTION_BANDS}
        total = sum(per_band.values())
        if math.isfinite(total) and total > 0.0:
            candidate.ratios = {name: value / total for name, value in per_band.items()}

        candidate.decay_ratio = _decay_ratio(envelopes[candidate.band], start, end, attack)
        candidate.flatness = _flatness(magnitude[:, start:end], in_range)


def _decay_ratio(
    envelope: npt.NDArray[np.float64], start: int, end: int, attack_frames: int
) -> float | None:
    """RMS of the attack over RMS of the tail, in the detecting band. Higher is shorter.

    Measured on the detecting band's envelope rather than the full spectrum,
    which is what makes it survive coincidence: at an instant where a snare and
    a hat sound together, the air band's decay is the hat's, because the snare
    puts 0.1% of its energy up there.

    Clamped at `MAX_DECAY_RATIO`, mirroring `MAX_TRANSIENT_SHARPNESS`: a hit
    with a digitally silent tail otherwise divides by zero.

    Returns:
        `None` when the window has no tail to measure or no attack energy at
        all, so the hat score simply does not fire rather than reading a
        fabricated number.
    """
    attack_end = min(start + attack_frames, end)
    if attack_end <= start or end <= attack_end:
        return None
    head = float(np.sqrt(np.mean(envelope[start:attack_end])))
    tail = float(np.sqrt(np.mean(envelope[attack_end:end])))
    if not math.isfinite(head) or head <= 0.0:
        return None
    if not math.isfinite(tail) or tail <= 0.0:
        return MAX_DECAY_RATIO
    return min(head / tail, MAX_DECAY_RATIO)


def _flatness(
    window: npt.NDArray[np.float64], in_range: npt.NDArray[np.bool_]
) -> float | None:
    """Geometric over arithmetic mean of the window's power spectrum. Low is tonal.

    Reported for audit only — see the module docstring for why it does not
    classify. Floored `FLATNESS_FLOOR` below its own peak bin, so the answer is
    scale-invariant and is not decided by the nulls of a band-limited signal.
    """
    if window.size == 0:
        return None
    power = np.square(window).sum(axis=1)[in_range]
    if power.size == 0:
        return None
    peak = float(power.max())
    if not math.isfinite(peak) or peak <= 0.0:
        return None
    floored = np.maximum(power, peak * FLATNESS_FLOOR)
    value = float(np.exp(np.mean(np.log(floored))) / np.mean(floored))
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _score(value: float | None, threshold: float, saturation: float) -> float:
    """`heuristics.ramp`, with "did not fire" folded to 0.0.

    The scores here are compared against each other and against
    `DECISION_FLOOR`, so they need to be numbers rather than
    number-or-None. 0.0 is the honest value: `ramp` returns `None` precisely
    when the descriptor has not reached the threshold, which is a score of
    nothing.
    """
    graded = ramp(value, threshold, saturation)
    return 0.0 if graded is None else float(graded)


def _class_scores(candidate: _Candidate) -> dict[str, float]:
    """Score a candidate for kick, snare and hat.

    Each score is `shape * affinity`, where the shape term is a `ramp` over one
    measured descriptor and the affinity is `DETECTOR_CLASS_AFFINITY` for the
    band that found the hit.

    The three shape terms, and why each is the one that survived measurement:

    * **kick: `kick_ratio`.** A kick is the loudest thing at its own instant by
      a factor of ~100, so its share of the window survives coincidence
      untouched. Measured 0.979-0.992 against 0.42 for anything else.
    * **snare: `body_ratio`.** Not `noise_ratio` — a plain broadband click
      measures *noisier than a snare* (0.5468 against 0.3653), so a noise-keyed
      rule calls every click a snare. `body_ratio` separates them 12:1. A
      secondary `noise_ratio` term was measured and dropped: it adds no
      separation any fixture can demonstrate, and adding an untested condition
      to a rule that already works is how a classifier stops firing on real
      material.
    * **hat: `decay_ratio` in the detecting band and `air / (air + noise)`,
      whichever is weaker.** The air band's raw *share* is useless — at a
      kick-and-hat instant the window is 99% kick — but the air band's *decay*
      belongs to the hat alone, because neither kick nor snare has measurable
      energy up there. Measured: closed hats 5.40-12.49, open hats 1.46-1.72,
      `_click` 20.55 and up. `decay_ratio` is also the field `strudel_vocab`
      reads to choose `hh` over `oh`, so it is load-bearing twice over. The
      second term rejects the top shoulder of a snare's rattle, which decays
      exactly like an open hat; see its threshold note.

    The two hat terms combine as a minimum rather than a product, matching
    `heuristics._all`: a label is never more confident than its weakest
    ingredient, and a barely-met condition must not be masked by a
    comfortably-met one.

    Spectral flatness is measured on every hit and used by none of them; see the
    module docstring.
    """
    affinity = DETECTOR_CLASS_AFFINITY.get(candidate.band, {})
    air = candidate.ratios["air"]
    noise = candidate.ratios["noise"]
    bright = air + noise if air is not None and noise is not None else None
    air_over_noise = None if not bright or air is None else air / bright
    shapes = {
        "kick": _score(
            candidate.ratios["kick"],
            THRESHOLDS["kick_low_ratio"],
            THRESHOLDS["kick_low_ratio_saturation"],
        ),
        "snare": _score(
            candidate.ratios["body"],
            THRESHOLDS["snare_body_ratio"],
            THRESHOLDS["snare_body_ratio_saturation"],
        ),
        "hat": min(
            _score(
                candidate.decay_ratio,
                THRESHOLDS["hat_decay_ratio"],
                THRESHOLDS["hat_decay_ratio_saturation"],
            ),
            _score(
                air_over_noise,
                THRESHOLDS["hat_air_over_noise"],
                THRESHOLDS["hat_air_over_noise_saturation"],
            ),
        ),
    }
    return {name: shapes[name] * affinity.get(name, 0.0) for name in _SCORED_CLASSES}


def _decide(scores: dict[str, float]) -> tuple[str, float]:
    """argmax with a floor and a margin. Returns `(class, confidence)`.

    Ranked by score, then alphabetically so an exact tie is resolved the same
    way on every run. The winner is only accepted when it reaches
    `DECISION_FLOOR` *and* beats the runner-up by `DECISION_MARGIN`; otherwise
    the class is `unclassified` and the confidence is the winner's score
    anyway. Reporting the honest near-miss is the point: a hit that scored 0.14
    for snare and a hit that scored 0.00 for everything are both unclassified
    and are not the same thing.
    """
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    winner, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best >= DECISION_FLOOR and best - runner_up >= DECISION_MARGIN:
        return winner, best
    return "unclassified", best


def _resolve_coincidences(
    candidates: list[_Candidate], separation_seconds: float, sample_rate: int
) -> list[_Candidate]:
    """Collapse the two artefacts of detecting the same instant in several bands.

    1. **Same class, same instant, one hit.** A snare fires the body detector
       *and* the noise detector; that is one snare. The more confident copy
       wins.
    2. **An `unclassified` hit beside a classified one is leakage.** A kick
       leaks into the body band and the body detector fires; that bump is
       already reported as a kick. Keeping it would put a phantom unclassified
       hit under every kick in the track.

    Both use `MIN_HIT_SEPARATION_SECONDS` — the same resolution the per-band
    peak picker uses, so "the same instant" means one thing in this module.

    The documented cost of rule 2: a tom or clap struck at exactly the same
    instant as a hat is swallowed by the hat rather than reported as
    unclassified. Struck anywhere else, it survives.
    """
    if not candidates:
        return []
    separation = _frames(separation_seconds, sample_rate)

    kept: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.frame, item.band)):
        duplicate = next(
            (
                other
                for other in kept
                if other.drum == candidate.drum
                and abs(other.frame - candidate.frame) < separation
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
        elif candidate.confidence > duplicate.confidence:
            kept[kept.index(duplicate)] = candidate

    classified_frames = [item.frame for item in kept if item.drum != "unclassified"]
    return [
        item
        for item in kept
        if item.drum != "unclassified"
        or all(abs(item.frame - frame) >= separation for frame in classified_frames)
    ]


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


class _Grid:
    """A cycle grid fitted to the hits, or the reason there isn't one."""

    __slots__ = (
        "anchor_seconds",
        "anchor_source",
        "cycle_seconds",
        "error_steps",
        "steps",
        "steps_per_cycle",
    )

    def __init__(
        self,
        steps_per_cycle: int | None = None,
        cycle_seconds: float | None = None,
        anchor_seconds: float | None = None,
        anchor_source: str | None = None,
        error_steps: float | None = None,
        steps: list[int] | None = None,
    ) -> None:
        self.steps_per_cycle = steps_per_cycle
        self.cycle_seconds = cycle_seconds
        self.anchor_seconds = anchor_seconds
        self.anchor_source = anchor_source
        self.error_steps = error_steps
        self.steps = steps


def _fit_grid(
    times: list[float],
    bpm: float | None,
    beat_times: Sequence[float],
    beats_per_cycle: int,
) -> _Grid:
    """Fit a cycle grid to the hit times, or return one with `steps_per_cycle=None`.

    The anchor is `beat_times[0]` when there is one, else the first hit, and
    which of those it was is recorded — a grid anchored on a tempo estimate and
    one anchored on whatever happened to be loudest first deserve different
    amounts of trust.

    `GRID_STEP_CANDIDATES` are tried in order and the one that fits best wins,
    where "best" is the mean absolute distance from a hit to its nearest step
    **in seconds**, not in steps.

    That distinction is load-bearing and easy to get wrong. For a fixed amount
    of timing jitter, the error measured *in steps* is smaller on a coarser
    grid purely because its steps are longer — a 12-step cycle scores 25% lower
    than a 16-step one on identical hits, so comparing step errors always
    prefers 12 regardless of the material. Seconds are neutral. On a genuine
    tie (a quarter-note pattern lands exactly on both grids) the earlier
    candidate keeps its place unless the challenger improves by
    `GRID_IMPROVEMENT_FRACTION`.

    The winner's error is then converted back to steps for reporting and
    compared against `MAX_QUANTISATION_ERROR_STEPS`; over that, no grid is
    reported at all. The bias throughout this project is that a wrong grid is
    worse than none, and a mean error of a quarter of a step means the
    assignment carries no information.

    No BPM means no cycle length, so no grid at all.
    """
    if bpm is None or not math.isfinite(bpm) or bpm <= 0.0 or not times:
        return _Grid()
    if beats_per_cycle <= 0:
        return _Grid()

    cycle_seconds = beats_per_cycle * 60.0 / float(bpm)
    if not math.isfinite(cycle_seconds) or cycle_seconds <= 0.0:
        return _Grid()

    anchor_source = "beats" if len(beat_times) else "first_hit"
    anchor = float(beat_times[0]) if len(beat_times) else float(times[0])
    if not math.isfinite(anchor):
        return _Grid()

    positions = (np.asarray(times, dtype=np.float64) - anchor) / cycle_seconds
    best: tuple[float, int, npt.NDArray[np.float64]] | None = None
    for steps_per_cycle in GRID_STEP_CANDIDATES:
        scaled = positions * steps_per_cycle
        error_steps = float(np.mean(np.abs(scaled - np.round(scaled))))
        error_seconds = error_steps * cycle_seconds / steps_per_cycle
        if best is None or error_seconds < best[0] * (1.0 - GRID_IMPROVEMENT_FRACTION):
            best = (error_seconds, steps_per_cycle, scaled)
    assert best is not None  # noqa: S101 - GRID_STEP_CANDIDATES is never empty
    _, steps_per_cycle, scaled = best
    error = float(np.mean(np.abs(scaled - np.round(scaled))))
    if error > MAX_QUANTISATION_ERROR_STEPS:
        return _Grid(
            cycle_seconds=cycle_seconds,
            anchor_seconds=anchor,
            anchor_source=anchor_source,
            error_steps=error,
        )
    absolute = np.round(scaled).astype(np.int64)
    return _Grid(
        steps_per_cycle=steps_per_cycle,
        cycle_seconds=cycle_seconds,
        anchor_seconds=anchor,
        anchor_source=anchor_source,
        error_steps=error,
        steps=[int(value) for value in absolute],
    )


def _patterns(
    hits: list[DrumHit], absolute_steps: list[int] | None, steps_per_cycle: int | None
) -> list[DrumPattern]:
    """Fold the hits onto one cycle, one `DrumPattern` per class present.

    `step_occupancy` divides by the number of cycles the hits actually span, not
    by the number of cycles that fit in the file. A four-bar loop with two bars
    of silence after it should read as a full pattern, not a half-occupied one.
    """
    if steps_per_cycle is None or absolute_steps is None:
        return [
            DrumPattern(drum=drum, hit_count=sum(1 for hit in hits if hit.drum == drum))
            for drum in _pattern_order(hits)
        ]

    cycles = [step // steps_per_cycle for step in absolute_steps]
    span = max(1, max(cycles) - min(cycles) + 1) if cycles else 1

    patterns: list[DrumPattern] = []
    for drum in _pattern_order(hits):
        occupied: dict[int, set[int]] = {}
        for hit, absolute, cycle in zip(hits, absolute_steps, cycles, strict=True):
            if hit.drum != drum:
                continue
            occupied.setdefault(absolute % steps_per_cycle, set()).add(cycle)
        steps = sorted(occupied)
        patterns.append(
            DrumPattern(
                drum=drum,
                steps=steps,
                step_occupancy=[min(1.0, len(occupied[step]) / span) for step in steps],
                hit_count=sum(1 for hit in hits if hit.drum == drum),
            )
        )
    return patterns


def _pattern_order(hits: list[DrumHit]) -> list[str]:
    """Classes actually present, in a stable reporting order."""
    present = {hit.drum for hit in hits}
    return [drum for drum in (*_SCORED_CLASSES, "unclassified") if drum in present]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def decompose(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    *,
    bpm: float | None,
    beat_times: Sequence[float],
    beats_per_cycle: int = 4,
) -> DrumDecomposition:
    """Decompose a drum source into kick, snare and hat hits on a cycle grid.

    The one supported entry point. Pure numpy throughout — see the module
    docstring for why that is the architecture rather than an implementation
    detail.

    Args:
        audio: Mono or multi-channel float array. Multi-channel input is
            averaged to mono; nothing here is stereo-aware.
        sample_rate: Must be the project's `ANALYSIS_SAMPLE_RATE`. Any other
            rate still works, but every threshold here was calibrated at 44.1
            kHz and a caveat says so.
        bpm: Tempo estimate, or `None`. `None` means no cycle length and
            therefore `status="no_grid"` — hits are still reported in full.
        beat_times: The rhythm block's beat positions. Only `beat_times[0]` is
            read, as the grid anchor. **`onset_times` is deliberately not a
            parameter**: this module finds its own onsets per band, so its
            output is identical whichever backend produced the rhythm block.
        beats_per_cycle: Beats in one Strudel cycle. 4 is one bar of 4/4.

    Returns:
        A `DrumDecomposition`. Never raises: any internal failure is caught and
        returned as `status="failed"` with the exception's class name in
        `caveats`, because a drum classifier is not worth losing the rest of an
        analysis over.

    Status:
        `ok` grid and hits; `no_grid` hits but no usable grid; `too_few_hits`
        nothing detected; `failed` an exception was swallowed.
    """
    try:
        return _decompose(audio, sample_rate, bpm, beat_times, beats_per_cycle)
    except Exception as error:  # noqa: BLE001 - deliberate: never break an analysis
        return DrumDecomposition(
            status="failed",
            caveats=[f"drum decomposition failed with {type(error).__name__}"],
        )


def _decompose(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    bpm: float | None,
    beat_times: Sequence[float],
    beats_per_cycle: int,
) -> DrumDecomposition:
    """The body of `decompose`, without the never-raise wrapper."""
    caveats: list[str] = []
    if sample_rate != ANALYSIS_SAMPLE_RATE:
        caveats.append(
            f"analysed at {sample_rate} Hz; every threshold here was calibrated "
            f"at {ANALYSIS_SAMPLE_RATE} Hz"
        )

    magnitude, freqs = _stft_magnitude(audio, sample_rate)
    if magnitude.shape[1] < 3:
        return DrumDecomposition(
            status="too_few_hits", caveats=[*caveats, "source too short to analyse"]
        )

    envelopes = _band_envelopes(magnitude, freqs)
    fluxes = {name: _spectral_flux(envelope) for name, envelope in envelopes.items()}
    active, dormant = _active_bands(envelopes, fluxes)
    if dormant:
        caveats.append(
            "no hits looked for in these bands — they hold nothing, or nothing "
            f"transient: {', '.join(dormant)}"
        )

    candidates: list[_Candidate] = []
    for band in active:
        for frame in _pick_peaks(fluxes[band], sample_rate):
            candidates.append(_Candidate(int(frame), band))
    candidates.sort(key=lambda item: (item.frame, item.band))
    if not candidates:
        return DrumDecomposition(status="too_few_hits", caveats=[*caveats, "no drum hits detected"])

    _measure(candidates, magnitude, freqs, envelopes, sample_rate)
    for candidate in candidates:
        candidate.scores = _class_scores(candidate)
        candidate.drum, candidate.confidence = _decide(candidate.scores)
    candidates = _resolve_coincidences(candidates, MIN_HIT_SEPARATION_SECONDS, sample_rate)
    candidates.sort(key=lambda item: (item.frame, item.drum))

    times = [candidate.frame * STFT_HOP_LENGTH / sample_rate for candidate in candidates]
    grid = _fit_grid(times, bpm, beat_times, beats_per_cycle)

    hits = [
        DrumHit(
            time_seconds=time_seconds,
            drum=candidate.drum,
            confidence=candidate.confidence,
            step=(
                grid.steps[index] % grid.steps_per_cycle
                if grid.steps is not None and grid.steps_per_cycle is not None
                else None
            ),
            kick_ratio=candidate.ratios["kick"],
            body_ratio=candidate.ratios["body"],
            noise_ratio=candidate.ratios["noise"],
            air_ratio=candidate.ratios["air"],
            decay_ratio=candidate.decay_ratio,
            flatness=candidate.flatness,
        )
        for index, (candidate, time_seconds) in enumerate(zip(candidates, times, strict=True))
    ]

    unclassified = sum(1 for hit in hits if hit.drum == "unclassified")
    if unclassified:
        caveats.append(
            f"{unclassified} of {len(hits)} hits are unclassified. Three classes "
            "cannot describe a full kit: toms, rides, crashes, claps and shakers "
            "all land here, and on percussive material that is correct rather "
            "than a failure."
        )
    if unclassified > THRESHOLDS["unclassified_caveat_fraction"] * len(hits):
        caveats.append(
            "most hits are unclassified, so read the kick/snare/hat pattern as a "
            "partial transcription of this source rather than a complete one"
        )
    if grid.steps_per_cycle is None:
        caveats.append(
            "no cycle grid: no usable tempo estimate"
            if bpm is None
            else (
                "no cycle grid: the best of "
                f"{'/'.join(str(value) for value in GRID_STEP_CANDIDATES)} steps per cycle "
                f"still misplaced hits by {grid.error_steps:.2f} steps on average, over the "
                f"{MAX_QUANTISATION_ERROR_STEPS:g} allowed"
            )
        )

    return DrumDecomposition(
        status="ok" if grid.steps_per_cycle is not None else "no_grid",
        steps_per_cycle=grid.steps_per_cycle,
        cycle_seconds=grid.cycle_seconds,
        grid_anchor_seconds=grid.anchor_seconds,
        grid_anchor_source=grid.anchor_source,
        quantisation_error_steps=grid.error_steps,
        patterns=_patterns(hits, grid.steps, grid.steps_per_cycle),
        hits=hits,
        unclassified_count=unclassified,
        caveats=caveats,
    )
