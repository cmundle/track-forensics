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


Kick bleed: when the second detection is the same drum again (WP-CAL v5)
------------------------------------------------------------------------

Those two post-passes handle leakage that stays *unclassified*. They do nothing
about leakage that gets classified **confidently and wrongly**, and on real
material that is the larger failure. Measured on the committed Madonna drums
envelopes, folded onto the verified 132.000 BPM grid: of 1240 hits reported as
`hat`, **504 sat on steps 0/4/8/12 — the kick's own steps — carrying a median
`kick_ratio` of 0.81**. They are the kick's broadband transient, found a second
time by the noise and air detectors and then scored as a hat, because the hat
rule normalises the kick away: `air / (air + noise)` is a *share of the bright
half* and says nothing about how much bright content there is.

The obvious fixes all fail, and each failure is worth recording because each
looks right on paper:

* **`kick_ratio` alone cannot do it.** A genuine hat sounding *with* a kick
  measures `kick_ratio` 0.99 in `drum_pattern_120bpm`, higher than the bleed's
  0.81, because the kick carries ~100x the hat's energy either way.
* **Absolute air-band level cannot do it.** The bleed's air-band peak on
  Madonna is 4.5x the median air-band peak of the hits that fire the air
  detector alone. The kick deposits *more* 6-16 kHz energy than this record's
  hats do.
* **Raising `hat_air_over_noise` cannot do it.** The bleed measures 0.21 and a
  closed hat sounding with a snare measures 0.0552, so any threshold that
  catches the first deletes the second — and the second is
  `test_all_thirty_two_hats_survive_coincidence`, the module's load-bearing
  assertion.

What does work is the same descriptor made **conditional on a kick being
present**. A kick's transient is broadband but falls off steeply with
frequency: its beater click lives at 1-6 kHz and its 6-16 kHz content is a
shoulder of that click. A hat is the other way round. So at an instant where
the kick band has *also* fired, and the window is kick-dominated, a hit found
by the noise or air detector belongs to the kick unless the bright half of its
spectrum is genuinely air-weighted. Measured `air / (air + noise)` at exactly
those instants:

===================================== ==================================
material                              `air / (air + noise)`
===================================== ==================================
Madonna, the 878 hits inside this     0.201 median, 0.295 at the 95th
rule's exact scope                    percentile, 0.375 at the 99th
`drum_pattern_120bpm`, hat over kick  0.9894
`drum_pattern_open_hats`, over kick   0.9981
===================================== ==================================

`KICK_BLEED_AIR_OVER_NOISE` sits at 0.50, which is not a midpoint reached for
convenience: it is the point where the two halves of the bright spectrum hold
equal energy, so the question it asks — *is this hit weighted to the air side
or the noise side?* — has a physical answer rather than a tuned one. It happens
to leave a factor of 1.6 of headroom below the real material and 2.0 above the
synthetic hat.

The dominance clause is not decoration either: in `drum_pattern_120bpm` the
kick band fires on the **snare's** shell tone, so co-detection alone would
catch every hat sounding with a snare. Those windows measure `kick_ratio`
0.0116, far under `KICK_BLEED_DOMINANCE`, and the rule never looks at them.

A suppressed hit is demoted to `unclassified` rather than deleted, so it is the
existing leakage post-pass that removes it, and a bleed detection beside a kick
that was *not* itself classified survives as an honest unclassified hit instead
of vanishing. Measured effect: Madonna 1240 hats -> 784 and 1872 hits -> 1422,
with the kick count unchanged at 487, and **zero** hits changed on any of the
four synthetic drum fixtures.

The cost, stated rather than discovered later: where a kick's beater click
carries more 1-6 kHz energy than a coincident hat carries at 6-16 kHz, the hat
goes with the bleed. On a synthetic kick built to the Madonna kick's own band
shares that boundary sits at a hat peaking at 0.4 against a kick at 0.5 — a
hat much quieter than the kick's own click is swallowed by it, the same shape
of cost as `_resolve_coincidences`' tom swallowed by a hat, and
`test_a_hat_quieter_than_the_kicks_own_click_is_swallowed` pins it.


The grid comes from an envelope fold, not from the hit list (WP-CAL v5)
-----------------------------------------------------------------------

Folding the *band flux envelope* into a (cycle x step) matrix and taking the
median across cycles is far more robust than folding peak-picked onsets: it
needs no detection threshold and it degrades gracefully across sections where
an element drops out. That is finding F8 of `V2-PLAN.md` and it is what this
module now uses to choose the subdivision and to decide whether the material is
periodic at all.

Three things had to be measured before that could replace the old rule, and two
of them contradict the plan:

1. **The fold cannot judge a grid from the shape of the profile.** The obvious
   statistic, `1 - mean/max` of the normalised median profile, measures
   *sparsity*, not periodicity: 40 uniformly random kicks in 8.5 s score 0.755
   and a genuine 8th-note pattern with 5% jitter scores 0.532. It is still the
   right statistic for **choosing** `steps_per_cycle`, because a pattern that
   does not divide into 12 smears across the 12-step fold, and it is used for
   exactly that and nothing else.
2. **What does judge a grid is where the folded energy sits *within* a step.**
   `_on_grid_share` folds at `GRID_OVERSAMPLE` sub-slots per step and reports
   the share of the median profile that lands on the sub-slots that are step
   centres. Chance is `1 / GRID_OVERSAMPLE` = 0.25. Measured: Madonna at the
   corrected period 1.000, `drum_pattern_120bpm` 0.962, `drum_pattern_kick_only`
   0.909, `drum_pattern_ambiguous` 0.991; against Madonna at the v4 period
   0.052, Madonna at 97.3 BPM 0.303, `drum_pattern_120bpm` at 97.3 BPM 0.369,
   200 random kicks over 60 s 0.051. `GRID_ON_GRID_SHARE_MIN` is 0.50 — twice
   chance, with the nearest accepted reading at 0.909 and the nearest rejected
   one at 0.369.
3. **The fold still cannot replace the per-hit quantisation check on short
   material.** `drum_pattern_120bpm` fitted to a deliberately wrong 97.3 BPM
   scores a *higher* profile contrast (0.773) than it does at its own tempo
   (0.801 is barely above it), because three cycles of a sparse pattern is not
   enough for a median to mean anything. `MAX_QUANTISATION_ERROR_STEPS` still
   has to hold, and it is unchanged at 0.18. So the plan's "stop using per-hit
   picking to decide whether a grid exists" is only half right, and this module
   requires **both** gates and says which one failed.


The anchor is phase-snapped, and that is not optional
-----------------------------------------------------

A supplied downbeat fixes which step is step 0. It does **not** fix the
sub-step phase, and the difference is the whole grid. The downbeat verified for
the Madonna fixture, 1.6283 s, puts the kick on steps 0/4/8/12 correctly and
still sits 0.267 steps — 30 ms — away from where the hits actually are. Fitted
raw it scores a mean quantisation error of 0.2672 and the grid is rejected;
snapped it scores **0.0332** and the grid is textbook. The same material
anchored on `beat_times[0]` = 0.348299 s needs no snap at all (0.0334), so this
is a property of the supplied offset, not of the record.

`_snap_anchor` therefore moves the anchor by the circular mean of the hits'
fractional step positions, which is bounded by half a step by construction and
so **cannot renumber the steps** — the four-fold downbeat ambiguity documented
in `calibration/v5-progress.md` is the caller's to resolve and this module does
not touch it. The snap is measured and reported in `caveats` whenever it moves
the anchor by more than `ANCHOR_SNAP_CAVEAT_STEPS`.

It also cannot rescue a wrong period: on `drum_pattern_120bpm` fitted to 97.3
BPM the snap moves the error from 0.2464 to 0.2473, because the residuals of a
wrong period are spread rather than offset.


Two ways to fail a grid, and they are not the same failure
-----------------------------------------------------------

`BLOCK_STATUSES` has one value for both, so the distinction lives in `caveats`:

* **The hits do not fit any grid.** Each half of the source, snapped to its own
  phase, still misses the allowance.
* **The hits fit a grid that is drifting.** Each half fits comfortably on its
  own but the two halves disagree about phase, which is what a period error
  looks like once it has accumulated. Measured on Madonna at the v4 period of
  132.040 BPM: whole-source error 0.1841, halves 0.0926 and 0.0841, phase
  difference +0.366 steps over 131.8 s. The implied period is reported with it,
  and it is explicitly approximate — assignments have already begun to wrap by
  the time the whole-source fit fails, so it recovers 132.08 rather than
  132.000. It is a pointer at the real cause, not a tempo estimate; `tempo.py`
  is where a tempo estimate comes from.


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
    "ANCHOR_SNAP_CAVEAT_STEPS",
    "ATTACK_SECONDS",
    "BAND_ACTIVITY_FLOOR",
    "BLEED_SOURCE_BAND",
    "BLEED_TARGET_BANDS",
    "DECISION_FLOOR",
    "DECISION_MARGIN",
    "DETECTION_BANDS",
    "DETECTOR_CLASS_AFFINITY",
    "FEATURE_WINDOW_SECONDS",
    "FLATNESS_FLOOR",
    "FLATNESS_RANGE_HZ",
    "FLUX_MEDIAN_HALF_SECONDS",
    "FLUX_PEAK_FLOOR",
    "FLUX_NEAR_ZERO_FRACTION",
    "FLUX_SPARSITY_MIN",
    "GRID_IMPROVEMENT_FRACTION",
    "GRID_ON_GRID_SHARE_MIN",
    "GRID_OVERSAMPLE",
    "GRID_STEP_CANDIDATES",
    "KICK_BLEED_AIR_OVER_NOISE",
    "KICK_BLEED_DOMINANCE",
    "MAX_DECAY_RATIO",
    "MAX_QUANTISATION_ERROR_STEPS",
    "MIN_HIT_SEPARATION_SECONDS",
    "MIN_FOLD_CYCLES",
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
    # the midpoint of that gap. Corpus material sits with the fixtures: Madonna
    # 0.794-0.806, Erykah Badu 0.791-0.885, Roni Size 0.730-0.799.
    #
    # The cost is real and worth stating: a band this dense *is* gated off, so a
    # 32nd-note hat roll with no gaps would be missed. Estimated crossover is
    # around 16 hits per second at 44.1 kHz, which is a 32nd-note roll at 120
    # BPM; 16ths measure about 0.81 and still pass.
    #
    # **A reverberant room is the other way to be dense, and it is the reason
    # this module reports no kick at all on "When the Levee Breaks".** That
    # stem's kick band scores 0.654 — under the gate, and *between* white noise
    # at 0.538 and every other drum stem at 0.79 and up. See
    # `_is_percussive` for what was tried instead and why the gate stayed.
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
    # -- Kick bleed --------------------------------------------------------
    # [grounded, measured] Share of a hit window's band energy below 150 Hz
    # above which the window is "kick-dominated" and the bleed rule is allowed
    # to look at it at all. Deliberately the same number as `kick_low_ratio`,
    # and for the same reason: it is the bar a window has to clear before this
    # module is willing to say the kick is what is in it.
    #
    # This clause is what keeps the rule off the snare. In
    # `drum_pattern_120bpm` the kick band fires on the snare's 200 Hz shell
    # tone, so a hat sounding with a snare *is* coincident with a kick-band
    # detection — but its window measures `kick_ratio` 0.0116, fifty times
    # under this floor, so the rule never reaches it. Measured kick-dominated
    # windows: real kicks 0.979-0.992, the Madonna bleed 0.72-0.84, a genuine
    # hat over a synthetic kick 0.99.
    "kick_bleed_dominance": 0.60,
    # [grounded, measured] `air / (air + noise)` at or above which a hit found
    # by the noise or air detector, at a kick-dominated instant where the kick
    # band also fired, is kept as a hit of its own rather than suppressed as
    # the kick's own transient.
    #
    # **0.50 is physical, not a tuned midpoint**: it is the point at which the
    # two halves of the bright spectrum hold equal energy. A kick's bright
    # content is its 1-6 kHz beater click with a 6-16 kHz shoulder, so it is
    # noise-weighted by construction; a hat is air-weighted. Measured over the
    # 878 Madonna hits that fall inside this rule's exact scope: 0.201 median,
    # 0.295 at the 95th percentile, 0.375 at the 99th, 0.574 at the maximum —
    # against 0.9894 for a closed hat over a synthetic kick and 0.9981 for an
    # open one. A factor of 1.7 of headroom over the 95th percentile below and
    # 2.0 below the synthetic hat above. 875 of the 878 are suppressed; the
    # three that survive are the only ones on the record with an air-weighted
    # bright half, and letting them through is the honest outcome.
    #
    # This is a *different threshold from* `hat_air_over_noise` (0.015) and
    # must stay that way: that one runs on every hit and is held down by a hat
    # sounding with a snare measuring 0.0552, while this one only ever runs
    # where a kick has already been found. Raising the shared threshold to
    # catch this would delete the snare-coincident hats, which is
    # `test_all_thirty_two_hats_survive_coincidence`.
    "kick_bleed_air_over_noise": 0.50,
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
    # [grounded, measured] Share of the oversampled median fold profile that
    # must land on the sub-slots that are step centres before the material is
    # accepted as periodic at this cycle length. This is the envelope-fold test
    # of F8 and it replaces nothing: it runs *beside*
    # `max_quantisation_error_steps`, because neither catches the other's case.
    #
    # Chance is `1 / GRID_OVERSAMPLE` = 0.25, so 0.50 asks for twice chance.
    # Measured accepted: Madonna at the corrected period 1.000,
    # `drum_pattern_ambiguous` 0.991, `drum_pattern_120bpm` 0.962,
    # `drum_pattern_kick_only` 0.909. Measured rejected: 200 uniformly random
    # kicks over 60 s 0.051, Madonna at the v4 period 0.052, Madonna at 97.3
    # BPM 0.303, `drum_pattern_120bpm` at 97.3 BPM 0.369. The nearest accepted
    # reading is 2.5x this and the nearest rejected one is 0.74x it.
    "grid_on_grid_share_min": 0.50,
    # [grounded] Anchor snap, in steps, above which the move is reported as a
    # caveat rather than applied silently. A tenth of a step is 11 ms at 132
    # BPM, one STFT hop, so anything smaller is below the time resolution of
    # every hit in this module and there is nothing to tell the reader.
    "anchor_snap_caveat_steps": 0.10,
    # [guess] Share of hits that may be `unclassified` before that fact is
    # raised as a caveat rather than left to the reader to notice.
    "unclassified_caveat_fraction": 0.25,
    # [grounded, measured — six-track corpus] Share of the kick band's own
    # candidates that must survive classification as kicks before the kick
    # pattern is reported without comment. Below it, `_kick_survival_caveat`
    # says so.
    #
    # This covers the case `_dormant_caveats` cannot: that helper reports a band
    # that was never searched, and a reader chasing a missing kick on Levee
    # Breaks is told the band was closed. Roni Size is the opposite failure and
    # was silent — the band is active, the grid fits at 0.098 quantisation
    # error, `status` is `"ok"`, and the kick class holds 73 of 2636 hits with
    # nothing in the output admitting it. A drum & bass grid with no kick in it,
    # presented as clean.
    #
    # Measured across all six tracks in `calibration/v5/`, candidates detected
    # in the kick band against those classified as kick:
    #
    #     madonna    491 -> 473   0.963
    #     badu       279 -> 270   0.968
    #     chameleon  827 -> 563   0.681
    #     roni       293 ->  57   0.195
    #     levee, eno   kick band dormant — `_dormant_caveats` owns these
    #
    # 0.5 is the majority bound and it is a statable fact, not a midpoint: below
    # it the kick band found more things that were not kicks than things that
    # were. It sits in a 3.5x gap with nothing in it — 2.6x above Roni, 1.36x
    # below the nearest track that is not being complained about.
    #
    # Chameleon at 0.681 stays quiet deliberately. It already returns `no_grid`
    # and says why, 695 kicks over 947 s is a plausible funk rate, and moving
    # the bound up to catch it would put the threshold 1.4x under Madonna and
    # Badu with no gap left to absorb a seventh track.
    "kick_survival_caveat_fraction": 0.5,
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
KICK_BLEED_DOMINANCE: Final[float] = THRESHOLDS["kick_bleed_dominance"]
KICK_BLEED_AIR_OVER_NOISE: Final[float] = THRESHOLDS["kick_bleed_air_over_noise"]
GRID_ON_GRID_SHARE_MIN: Final[float] = THRESHOLDS["grid_on_grid_share_min"]
ANCHOR_SNAP_CAVEAT_STEPS: Final[float] = THRESHOLDS["anchor_snap_caveat_steps"]

#: The band whose transient bleeds upward, and the bands it bleeds into. A
#: kick is the only drum in a kit loud enough and broad enough to fire a
#: detector two bands above its own, which is why this is a named pair rather
#: than a loop over every band below every other one — every other such pair
#: was measured and none of them fires. `body` is absent from the targets
#: because a body-band detection at a kick instant is already classified as the
#: kick and collapsed by `_resolve_coincidences`.
BLEED_SOURCE_BAND: Final[str] = "kick"
BLEED_TARGET_BANDS: Final[tuple[str, ...]] = ("noise", "air")

#: Sub-slots per step in the oversampled fold. 4 puts chance at 0.25 for
#: `_on_grid_share` and keeps a sub-slot at 28 ms at 132 BPM — still two to
#: three STFT hops wide, so a slot is never narrower than the measurement grid
#: that fills it.
GRID_OVERSAMPLE: Final[int] = 4

#: Cycles a source must span before a fold means anything. Two is the minimum
#: at which a median across cycles is not simply the one cycle there was; below
#: it the fold abstains and says so rather than returning a verdict from a
#: sample of one.
MIN_FOLD_CYCLES: Final[int] = 2

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


def _is_percussive(flux: npt.NDArray[np.float64]) -> str | None:
    """Why a band's flux is not worth searching, or `None` if it is.

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

    A compressed, reverberant source is dense in the same way a roll is, and
    this is where the module gives up honestly (WP-CAL v5, corpus). On "When
    the Levee Breaks" the kick band carries 37% of the stem's energy and peaks
    at 6.48 flux, so it fails neither for want of energy nor for want of
    transients — it scores 0.654 sparsity because the stairwell tail never lets
    the band fall back to its floor, and the whole band is switched off. The
    track reports no kick at all.

    **The obvious replacement was measured and rejected**, and the measurement
    is worth keeping because it looks like a clean win right up to the point
    where you check the output. `q90 / peak` separates the two populations far
    better than frame-counting sparsity does — percussive 0.000-0.205 across
    every fixture and every corpus track, stationary 0.369-0.459 — where
    sparsity puts this stem *between* white noise and every other drum stem, so
    no threshold on it can ever admit Bonham without also admitting noise.
    Swapping the test in changes no band on any track that currently works and
    turns this one on.

    It was still wrong, because what it admits cannot be resolved into hits.
    With the band open the picker returns 702 kicks over 430 s, and folded at
    the best tempo and phase available they occupy **13 of 16 steps** at
    0.18-0.53 occupancy with a quantisation error of 0.247 — 0.25 is what
    uniformly random hits score by construction. The between-hit floor in that
    band is a quarter of its peak; on Madonna it is a fiftieth. A longer
    differencing interval does help the *envelope*: at 4 frames instead of 1 the
    folded kick profile at 71 BPM sharpens from 0.560 to 0.645 contrast and
    shows a real pattern on steps 1/8/11. It does not help the *picker*, because
    the problem is not that the rise is missed, it is that there is no gap to
    separate one rise from the next.

    So: the fold can see this kick and per-hit detection cannot, and reporting
    702 unplaceable hits is worse than reporting none. The gate stays, and the
    caveat now says which of the two tests refused the band so a reader is not
    sent looking for silence.

    Returns:
        `None` when the band is worth searching, else which test refused it —
        `"no_transient"` or `"not_sparse"`. The two are different findings and
        `_decompose` reports them as different caveats: a band that holds no
        transient holds nothing, while a band that holds transients it cannot
        separate holds something this module cannot read. Saying "or nothing
        transient" for both, as v5 first shipped, sends a reader looking for
        silence in a band carrying 37% of the source's energy.
    """
    if flux.size == 0:
        return "no_transient"
    peak = float(flux.max())
    if not math.isfinite(peak) or peak < FLUX_PEAK_FLOOR:
        return "no_transient"
    near_zero = float(np.mean(flux <= FLUX_NEAR_ZERO_FRACTION * peak))
    return None if near_zero >= FLUX_SPARSITY_MIN else "not_sparse"


def _active_bands(
    envelopes: dict[str, npt.NDArray[np.float64]], fluxes: dict[str, npt.NDArray[np.float64]]
) -> tuple[list[str], dict[str, str]]:
    """Split the detection bands into those worth searching and those that are not.

    A band is searched when it holds at least `BAND_ACTIVITY_FLOOR` of the
    source's in-band energy **and** its flux passes `_is_percussive`. The first
    test rejects numerical residue (a band that is not really there); the second
    rejects a band that is there but holds no hits it can separate.

    Returns:
        `(active, dormant)`, where `dormant` maps each rejected band to *why* —
        `"empty"`, `"no_transient"` or `"not_sparse"`. Three different facts
        about a source, and a reader chasing a missing kick needs to know which
        one they are looking at.
    """
    totals = {name: float(envelope.sum()) for name, envelope in envelopes.items()}
    grand = sum(totals.values())
    if not math.isfinite(grand) or grand <= 0.0:
        return [], dict.fromkeys(DETECTION_BANDS, "empty")

    active: list[str] = []
    dormant: dict[str, str] = {}
    for name in DETECTION_BANDS:
        if totals[name] / grand < BAND_ACTIVITY_FLOOR:
            dormant[name] = "empty"
            continue
        refusal = _is_percussive(fluxes[name])
        if refusal is None:
            active.append(name)
        else:
            dormant[name] = refusal
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
    magnitude: npt.NDArray[np.float64] | None,
    freqs: npt.NDArray[np.float64] | None,
    envelopes: dict[str, npt.NDArray[np.float64]],
    sample_rate: int,
) -> None:
    """Fill in every candidate's ratios, decay and flatness, in place.

    `magnitude` and `freqs` are optional and only `flatness` needs them, which
    is why they are: everything that *classifies* a hit is computed from the
    band envelopes alone. That is what lets `tests/fixtures/real/` ship four
    envelope arrays per frame instead of audio and still exercise this module
    on real material — ground rule 9 of `KICKOFF-v2.md`. Without a spectrum,
    `flatness` is `None`, which is what it already means everywhere else:
    not measurable here.
    """
    n_frames = next(iter(envelopes.values())).size if envelopes else 0
    span = _frames(FEATURE_WINDOW_SECONDS, sample_rate)
    attack = _frames(ATTACK_SECONDS, sample_rate)
    low_hz, high_hz = FLATNESS_RANGE_HZ
    in_range = None if freqs is None else (freqs >= low_hz) & (freqs <= high_hz)

    for candidate in candidates:
        start = candidate.frame
        end = _window_end(candidate, candidates, n_frames, span)

        per_band = {name: float(envelopes[name][start:end].sum()) for name in DETECTION_BANDS}
        total = sum(per_band.values())
        if math.isfinite(total) and total > 0.0:
            candidate.ratios = {name: value / total for name, value in per_band.items()}

        candidate.decay_ratio = _decay_ratio(envelopes[candidate.band], start, end, attack)
        candidate.flatness = (
            None
            if magnitude is None or in_range is None
            else _flatness(magnitude[:, start:end], in_range)
        )


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


def _air_over_noise(candidate: _Candidate) -> float | None:
    """`air / (air + noise)` of a candidate's window, or `None` if neither fired.

    How much of the bright half of this hit's spectrum is genuinely *air*
    rather than the top of a 1-6 kHz rattle or a beater click. Read twice: by
    `_class_scores` against `hat_air_over_noise`, and by `_suppress_kick_bleed`
    against `KICK_BLEED_AIR_OVER_NOISE`. Those two thresholds sit a factor of
    33 apart on purpose — see their notes.
    """
    air = candidate.ratios["air"]
    noise = candidate.ratios["noise"]
    if air is None or noise is None:
        return None
    bright = air + noise
    return None if bright <= 0.0 else air / bright


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
    air_over_noise = _air_over_noise(candidate)
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


def _suppress_kick_bleed(
    candidates: list[_Candidate], sample_rate: int
) -> int:
    """Demote upper-band detections that are the kick's own transient. In place.

    The failure this exists for is documented at length in the module
    docstring: on real material the noise and air detectors fire on a kick's
    broadband transient, and the hat rule — which normalises the kick away by
    construction — then scores those detections a confident 1.0. Measured on
    the Madonna drums fixture, 504 of 1240 reported hats were the kick found
    twice.

    Three conditions, all required, and every one of them is load-bearing:

    1. the hit was found by a `BLEED_TARGET_BANDS` detector;
    2. a `BLEED_SOURCE_BAND` detection lies within
       `MIN_HIT_SEPARATION_SECONDS` — the same resolution "the same instant"
       means everywhere else in this module — **and** the window is
       kick-dominated past `KICK_BLEED_DOMINANCE`. Co-detection alone is not
       enough: the kick band fires on a synthetic snare's shell tone, so
       without the dominance clause this would delete every hat sounding with
       a snare;
    3. `air / (air + noise)` is below `KICK_BLEED_AIR_OVER_NOISE`, i.e. the
       bright half of the hit is weighted to the beater-click side rather than
       the air side. This is the only one of the three that separates the
       bleed from a genuine coincident hat, and it separates them by a factor
       of five.

    A suppressed candidate becomes `unclassified` with its winning score
    intact rather than being deleted here. Deletion is then
    `_resolve_coincidences`' rule 2, which drops it *because* a classified
    kick is beside it — so a bleed detection next to a kick that somehow did
    not classify survives as an honest unclassified hit rather than
    disappearing on the strength of a rule about kicks.

    Returns:
        How many candidates were demoted. Reported as a caveat, because
        silently removing a third of a source's hits is exactly the kind of
        thing a reader should be told about.
    """
    separation = _frames(MIN_HIT_SEPARATION_SECONDS, sample_rate)
    source_frames = np.asarray(
        [item.frame for item in candidates if item.band == BLEED_SOURCE_BAND], dtype=np.int64
    )
    if source_frames.size == 0:
        return 0

    suppressed = 0
    for candidate in candidates:
        if candidate.band not in BLEED_TARGET_BANDS or candidate.drum == "unclassified":
            continue
        kick_ratio = candidate.ratios[BLEED_SOURCE_BAND]
        if kick_ratio is None or kick_ratio < KICK_BLEED_DOMINANCE:
            continue
        if not bool(np.any(np.abs(source_frames - candidate.frame) < separation)):
            continue
        air_over_noise = _air_over_noise(candidate)
        if air_over_noise is not None and air_over_noise >= KICK_BLEED_AIR_OVER_NOISE:
            continue
        candidate.drum = "unclassified"
        suppressed += 1
    return suppressed


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
    """A cycle grid fitted to the hits, or the reason there isn't one.

    Working state, like `_Candidate`: it carries the diagnostic quantities that
    decided the grid (`contrast`, `on_grid_share`, `anchor_shift_steps`, the
    drift readings) so `_decompose` can turn them into caveats, and none of
    them leak into the schema.
    """

    __slots__ = (
        "anchor_seconds",
        "anchor_shift_steps",
        "anchor_source",
        "contrast",
        "cycle_seconds",
        "drift_steps",
        "error_steps",
        "failure",
        "half_error_steps",
        "implied_cycle_seconds",
        "on_grid_share",
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
        *,
        anchor_shift_steps: float | None = None,
        contrast: float | None = None,
        on_grid_share: float | None = None,
        half_error_steps: tuple[float, float] | None = None,
        drift_steps: float | None = None,
        implied_cycle_seconds: float | None = None,
        failure: str | None = None,
    ) -> None:
        self.steps_per_cycle = steps_per_cycle
        self.cycle_seconds = cycle_seconds
        self.anchor_seconds = anchor_seconds
        self.anchor_source = anchor_source
        self.error_steps = error_steps
        self.steps = steps
        self.anchor_shift_steps = anchor_shift_steps
        self.contrast = contrast
        self.on_grid_share = on_grid_share
        self.half_error_steps = half_error_steps
        self.drift_steps = drift_steps
        self.implied_cycle_seconds = implied_cycle_seconds
        self.failure = failure


def _fold(
    values: npt.NDArray[np.float64],
    sample_rate: int,
    cycle_seconds: float,
    anchor_seconds: float,
    slots_per_cycle: int,
) -> npt.NDArray[np.float64] | None:
    """Fold one band's flux into a `(cycle, slot)` matrix. The F8 primitive.

    Every frame is assigned to its **nearest** slot and a slot takes the
    largest flux assigned to it, so a hit that straddles two frames counts once
    at its peak rather than being split between neighbours. Frames before the
    anchor are dropped; there is no cycle -1 to put them in.

    Nearest rather than floor is not cosmetic: with `floor`, an event sitting a
    hair before a slot boundary lands in the previous slot, and the whole
    profile reads one slot early. That is how the Madonna kick first appeared
    on steps 3/7/11/15.

    Returns:
        `(n_cycles, slots_per_cycle)`, or `None` when the source does not span
        `MIN_FOLD_CYCLES` whole cycles — below that a median across cycles is
        a median of one thing, and abstaining is the honest answer.
    """
    if values.size == 0 or slots_per_cycle <= 0:
        return None
    if not math.isfinite(cycle_seconds) or cycle_seconds <= 0.0:
        return None
    slot_seconds = cycle_seconds / slots_per_cycle
    times = np.arange(values.size, dtype=np.float64) * STFT_HOP_LENGTH / sample_rate
    slots = np.rint((times - anchor_seconds) / slot_seconds).astype(np.int64)
    keep = slots >= 0
    if not bool(keep.any()):
        return None
    slots, kept = slots[keep], values[keep]
    total = int(slots.max()) + 1
    folded = np.zeros(total, dtype=np.float64)
    np.maximum.at(folded, slots, kept)
    cycles = total // slots_per_cycle
    if cycles < MIN_FOLD_CYCLES:
        return None
    return folded[: cycles * slots_per_cycle].reshape(cycles, slots_per_cycle)


def _fold_profile(
    fluxes: dict[str, npt.NDArray[np.float64]],
    bands: Sequence[str],
    sample_rate: int,
    cycle_seconds: float,
    anchor_seconds: float,
    slots_per_cycle: int,
) -> dict[str, npt.NDArray[np.float64]]:
    """Median-across-cycles profile per band. The median is the point of F8.

    A mean would let one crash in one bar decide a step; a median asks "does
    this step fire in *most* cycles", which is what a pattern is, and it is
    what makes the fold survive a breakdown where an element drops out for
    sixteen bars.
    """
    profiles: dict[str, npt.NDArray[np.float64]] = {}
    for band in bands:
        matrix = _fold(fluxes[band], sample_rate, cycle_seconds, anchor_seconds, slots_per_cycle)
        if matrix is None:
            continue
        profile = np.median(matrix, axis=0).astype(np.float64)
        if float(profile.max()) > 0.0:
            profiles[band] = profile
    return profiles


def _profile_contrast(profile: npt.NDArray[np.float64]) -> float:
    """`1 - mean/peak` of a normalised profile. The subdivision chooser.

    Matches `tempo.DownbeatFit.beat_confidence`, which measures the same thing
    for the same reason, so the two modules describe a fold the same way.

    **This does not measure periodicity** and must not be used as if it did:
    40 uniformly random kicks in 8.5 s score 0.755 on it, because a sparse
    random set leaves most steps empty in most cycles and emptiness reads as
    contrast. What it does measure well is whether the material divides into
    `slots_per_cycle` at all — a pattern that does not smears across the wrong
    fold and flattens it. `_on_grid_share` is the periodicity test.
    """
    peak = float(profile.max())
    if not math.isfinite(peak) or peak <= 0.0:
        return 0.0
    return float(1.0 - (profile / peak).mean())


def _on_grid_share(profile: npt.NDArray[np.float64], oversample: int) -> float | None:
    """Share of an oversampled profile that lands on the step centres.

    The periodicity test. `profile` is folded at `oversample` sub-slots per
    step, so the sub-slots at indices `0, oversample, 2 * oversample, ...` are
    the step centres and everything between them is off-grid. A pattern locked
    to the grid puts nearly all of its folded energy on the centres; a wrong
    period puts each cycle's events at a different in-cycle position, the
    median flattens, and the share falls to chance, `1 / oversample`.

    Measured, against `GRID_ON_GRID_SHARE_MIN` = 0.50 and chance = 0.25:
    Madonna at the corrected period 1.000, `drum_pattern_ambiguous` 0.991,
    `drum_pattern_120bpm` 0.962, `drum_pattern_kick_only` 0.909; Madonna at
    the v4 period 0.052, Madonna at 97.3 BPM 0.303, `drum_pattern_120bpm` at
    97.3 BPM 0.369, 200 uniformly random kicks over 60 s 0.051.

    Returns:
        `None` when the profile is empty — every step silent in more than half
        the cycles — because that is no evidence either way rather than
        evidence against.
    """
    total = float(profile.sum())
    if not math.isfinite(total) or total <= 0.0:
        return None
    return float(profile[::oversample].sum() / total)


def _fractional_steps(
    times: npt.NDArray[np.float64],
    cycle_seconds: float,
    anchor_seconds: float,
    steps_per_cycle: int,
) -> npt.NDArray[np.float64]:
    """Signed distance from each hit to its nearest step, in steps, in [-0.5, 0.5]."""
    scaled = (times - anchor_seconds) / cycle_seconds * steps_per_cycle
    residual: npt.NDArray[np.float64] = (scaled - np.round(scaled)).astype(np.float64)
    return residual


def _snap_anchor(
    times: npt.NDArray[np.float64],
    cycle_seconds: float,
    anchor_seconds: float,
    steps_per_cycle: int,
) -> tuple[float, float]:
    """Move the anchor onto the hits' own phase. Returns `(anchor, shift_steps)`.

    The shift is the **circular** mean of the fractional step positions, taken
    as an angle so that hits sitting either side of a step boundary average to
    the boundary rather than to the middle of the cycle. It is bounded by half
    a step by construction, so it cannot renumber a single step: which step is
    step 0 stays exactly as the caller supplied it, and the four-fold downbeat
    ambiguity that `calibration/v5-progress.md` documents is not this module's
    to resolve.

    Why this is necessary rather than tidy: the downbeat verified for the
    Madonna fixture, 1.6283 s, is 0.267 steps — 30 ms — from where the hits
    are. Fitted raw it scores 0.2672 steps of mean error and the grid is
    rejected; snapped it scores 0.0332. A supplied downbeat pins *which* step
    is step 0; it does not pin the sub-step phase, and the grid needs both.

    Why it is not a way of forcing a fit: on `drum_pattern_120bpm` fitted to a
    deliberately wrong 97.3 BPM the snap moves the error from 0.2464 to 0.2473
    — it makes it very slightly worse. A wrong period spreads its residuals
    instead of offsetting them, so there is no phase for the snap to find.
    """
    if times.size == 0:
        return anchor_seconds, 0.0
    fractional = _fractional_steps(times, cycle_seconds, anchor_seconds, steps_per_cycle)
    angles = 2.0 * np.pi * fractional
    mean_angle = float(
        np.arctan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    )
    shift = mean_angle / (2.0 * np.pi)
    if not math.isfinite(shift):
        return anchor_seconds, 0.0
    return anchor_seconds + shift * cycle_seconds / steps_per_cycle, shift


def _mean_error(
    times: npt.NDArray[np.float64],
    cycle_seconds: float,
    anchor_seconds: float,
    steps_per_cycle: int,
) -> float:
    """Mean absolute distance from a hit to its nearest step, in steps."""
    if times.size == 0:
        return 0.0
    return float(
        np.mean(np.abs(_fractional_steps(times, cycle_seconds, anchor_seconds, steps_per_cycle)))
    )


def _drift_reading(
    times: npt.NDArray[np.float64],
    cycle_seconds: float,
    anchor_seconds: float,
    steps_per_cycle: int,
) -> tuple[tuple[float, float], float, float | None] | None:
    """Fit each half of the source on its own phase. The drift-versus-no-fit test.

    A period error and a loose performance both blow the whole-source
    quantisation allowance, and they are not the same finding: the first is the
    caller's tempo being slightly wrong and is fixable, the second is the
    material. Splitting at the midpoint and snapping each half separately
    separates them, because a period error is a *phase ramp* — each half fits
    its own phase comfortably and the two phases disagree.

    Measured on Madonna at the v4 period of 132.040 BPM: whole-source 0.1841,
    halves 0.0926 and 0.0841, phase difference +0.366 steps over 131.8 s. At
    the truly wrong 97.3 BPM on `drum_pattern_120bpm` both halves score 0.243
    and 0.247, so nothing is rescued and the answer is honestly "no grid".

    Returns:
        `((first_error, second_error), drift_steps, implied_cycle_seconds)`, or
        `None` when either half holds too few hits to fit. The implied cycle is
        **approximate and says so**: by the time a whole-source fit fails, some
        hits have already wrapped to the wrong step, which biases the ramp. On
        the Madonna case it recovers 132.08 BPM against a true 132.000. It
        points at the cause; `tempo.py` measures the tempo.
    """
    if times.size < 8:
        return None
    midpoint = 0.5 * (float(times.min()) + float(times.max()))
    halves = (times[times <= midpoint], times[times > midpoint])
    if any(half.size < 4 for half in halves):
        return None

    anchors: list[float] = []
    errors: list[float] = []
    centres: list[float] = []
    for half in halves:
        snapped, _shift = _snap_anchor(half, cycle_seconds, anchor_seconds, steps_per_cycle)
        anchors.append(snapped)
        errors.append(_mean_error(half, cycle_seconds, snapped, steps_per_cycle))
        centres.append(float(half.mean()))

    step_seconds = cycle_seconds / steps_per_cycle
    drift = (anchors[1] - anchors[0]) / step_seconds
    # Both anchors are phases, so their difference is only meaningful modulo a
    # whole step; take the representative in (-0.5, 0.5].
    drift = float((drift + 0.5) % 1.0 - 0.5)
    elapsed = centres[1] - centres[0]
    implied: float | None = None
    if elapsed > 0.0:
        rate = drift * step_seconds / elapsed
        if abs(rate) < 0.5:
            implied = cycle_seconds * (1.0 - rate)
    return (errors[0], errors[1]), drift, implied


def _cycle_seconds(
    bpm: float | None, beat_period_seconds: float | None, beats_per_cycle: int
) -> float | None:
    """One cycle in seconds, preferring a supplied beat period over a BPM label.

    `beat_period_seconds` is `tempo.TempoFit.period_seconds` — a refined
    measurement. `bpm` is whichever backend produced the rhythm block, accurate
    to roughly +/- 0.2 BPM, which finding F1 showed is four decimal places too
    coarse to extend a grid across a four-minute track. When both are present
    the measurement wins and the label is ignored.
    """
    if beats_per_cycle <= 0:
        return None
    period = beat_period_seconds
    if period is None and bpm is not None and math.isfinite(bpm) and bpm > 0.0:
        period = 60.0 / float(bpm)
    if period is None or not math.isfinite(period) or period <= 0.0:
        return None
    cycle = beats_per_cycle * period
    return cycle if math.isfinite(cycle) and cycle > 0.0 else None


def _fit_grid(
    times: list[float],
    fluxes: dict[str, npt.NDArray[np.float64]],
    active: Sequence[str],
    sample_rate: int,
    *,
    bpm: float | None,
    beat_period_seconds: float | None,
    beat_times: Sequence[float],
    downbeat_seconds: float | None,
    beats_per_cycle: int,
) -> _Grid:
    """Fit a cycle grid, or return one with `steps_per_cycle=None` and a reason.

    Four stages, in this order, because each depends on the one before it:

    1. **Cycle length.** A supplied `beat_period_seconds` if there is one, else
       `bpm`. No length, no grid.
    2. **Anchor.** `downbeat_seconds` if supplied, else `beat_times[0]`, else
       the first hit — recorded either way, because a grid anchored on a
       measured downbeat, one anchored on a tempo estimate and one anchored on
       whatever happened to be loudest first deserve different amounts of
       trust. Then phase-snapped onto the hits, which cannot renumber a step
       and which the Madonna fixture does not fit without.
    3. **Subdivision.** `GRID_STEP_CANDIDATES` in order, scored by the best
       band's `_profile_contrast` on the median fold. 16 is the incumbent and
       12 must beat it by `GRID_IMPROVEMENT_FRACTION` to displace it, because a
       quarter-note pattern lands exactly on both grids and the choice would
       otherwise be decided by float noise — `drum_pattern_kick_only` reported
       its kicks on steps 0/3/6/9 until this rule existed.
    4. **Two gates, both required, and the failure names which one gave way.**
       `_on_grid_share` against `GRID_ON_GRID_SHARE_MIN` asks whether the
       *envelope* is periodic at this cycle length; the mean per-hit error
       against `MAX_QUANTISATION_ERROR_STEPS` asks whether the *hits* land on
       it. Neither subsumes the other: the fold is blind on material spanning
       three cycles (`drum_pattern_120bpm` scores a higher contrast at a wrong
       97.3 BPM than at its own tempo), and the hit error is marginal on
       material where the fold is decisive (Madonna at the v4 period scores
       0.1841 against a 0.18 allowance, and 0.052 against a 0.50 fold floor).

    The allowance is **not** relaxed anywhere in here. Finding F1 was a tempo
    error wearing a threshold's clothes: at the corrected period the same hits
    score 0.0332.
    """
    cycle_seconds = _cycle_seconds(bpm, beat_period_seconds, beats_per_cycle)
    if cycle_seconds is None or not times:
        return _Grid(failure="no_cycle_length")

    if downbeat_seconds is not None and math.isfinite(downbeat_seconds):
        anchor, anchor_source = float(downbeat_seconds), "supplied"
    elif len(beat_times):
        anchor, anchor_source = float(beat_times[0]), "beats"
    else:
        anchor, anchor_source = float(times[0]), "first_hit"
    if not math.isfinite(anchor):
        return _Grid(failure="no_cycle_length")

    hit_times = np.asarray(times, dtype=np.float64)
    best: tuple[float, int] | None = None
    for steps_per_cycle in GRID_STEP_CANDIDATES:
        snapped, _shift = _snap_anchor(hit_times, cycle_seconds, anchor, steps_per_cycle)
        profiles = _fold_profile(
            fluxes, active, sample_rate, cycle_seconds, snapped, steps_per_cycle
        )
        contrast = max((_profile_contrast(p) for p in profiles.values()), default=0.0)
        if best is None or contrast > best[0] * (1.0 + GRID_IMPROVEMENT_FRACTION):
            best = (contrast, steps_per_cycle)
    assert best is not None  # noqa: S101 - GRID_STEP_CANDIDATES is never empty
    contrast, steps_per_cycle = best

    anchor, shift = _snap_anchor(hit_times, cycle_seconds, anchor, steps_per_cycle)
    error = _mean_error(hit_times, cycle_seconds, anchor, steps_per_cycle)
    oversampled = _fold_profile(
        fluxes,
        active,
        sample_rate,
        cycle_seconds,
        anchor,
        steps_per_cycle * GRID_OVERSAMPLE,
    )
    shares = [
        share
        for share in (_on_grid_share(p, GRID_OVERSAMPLE) for p in oversampled.values())
        if share is not None
    ]
    on_grid = max(shares) if shares else None

    partial = _Grid(
        cycle_seconds=cycle_seconds,
        anchor_seconds=anchor,
        anchor_source=anchor_source,
        error_steps=error,
        anchor_shift_steps=shift,
        contrast=contrast,
        on_grid_share=on_grid,
    )
    if on_grid is not None and on_grid < GRID_ON_GRID_SHARE_MIN:
        partial.failure = "not_periodic"
        return partial
    if error > MAX_QUANTISATION_ERROR_STEPS:
        reading = _drift_reading(hit_times, cycle_seconds, anchor, steps_per_cycle)
        partial.failure = "no_fit"
        if reading is not None:
            partial.half_error_steps, partial.drift_steps, partial.implied_cycle_seconds = reading
            if max(reading[0]) <= MAX_QUANTISATION_ERROR_STEPS:
                partial.failure = "drifting"
        return partial

    scaled = (hit_times - anchor) / cycle_seconds * steps_per_cycle
    partial.steps_per_cycle = steps_per_cycle
    partial.steps = [int(value) for value in np.round(scaled).astype(np.int64)]
    return partial


def _patterns(
    hits: list[DrumHit], absolute_steps: list[int] | None, steps_per_cycle: int | None
) -> list[DrumPattern]:
    """Fold the hits onto one cycle, one `DrumPattern` per class present.

    `step_occupancy` is the share of the cycles **this class was playing in**
    that hold a hit of it on this step, and the choice of denominator is the
    whole value of the field. Three candidates, and only one of them says
    anything:

    * *cycles in the file* punishes every element for the arrangement. On the
      Madonna track the kick sits out a sixteen-bar breakdown, so a four-on-the
      floor kick would read 0.79-0.83 and look intermittent.
    * *cycles the hits span* is the same number for anything that plays at the
      start and at the end, which on a full track is everything.
    * *cycles this class plays in* answers the question a reader actually has:
      when this drum is playing, does it hit this step every time? Measured on
      Madonna the kick reads 0.906/0.953/0.929/0.945 on steps 0/4/8/12 — a
      backbone — and its off-grid steps read 0.008-0.031, which is the same
      pattern the envelope fold shows independently.

    On a fixture where a class plays throughout, all three agree, which is why
    the synthetic expectations are unchanged: a repeating four-cycle pattern
    still reads 1.0 and one ghost kick in one cycle of four still reads 0.25.

    `step_occupancy` was empty in every v4 output because v4 never resolved a
    grid, not because it was never computed — there is no schema change here
    and nothing to migrate.
    """
    if steps_per_cycle is None or absolute_steps is None:
        return [
            DrumPattern(drum=drum, hit_count=sum(1 for hit in hits if hit.drum == drum))
            for drum in _pattern_order(hits)
        ]

    cycles = [step // steps_per_cycle for step in absolute_steps]

    patterns: list[DrumPattern] = []
    for drum in _pattern_order(hits):
        occupied: dict[int, set[int]] = {}
        playing: set[int] = set()
        for hit, absolute, cycle in zip(hits, absolute_steps, cycles, strict=True):
            if hit.drum != drum:
                continue
            occupied.setdefault(absolute % steps_per_cycle, set()).add(cycle)
            playing.add(cycle)
        steps = sorted(occupied)
        denominator = max(1, len(playing))
        patterns.append(
            DrumPattern(
                drum=drum,
                steps=steps,
                step_occupancy=[min(1.0, len(occupied[step]) / denominator) for step in steps],
                hit_count=sum(1 for hit in hits if hit.drum == drum),
            )
        )
    return patterns


def _pattern_order(hits: list[DrumHit]) -> list[str]:
    """Classes actually present, in a stable reporting order."""
    present = {hit.drum for hit in hits}
    return [drum for drum in (*_SCORED_CLASSES, "unclassified") if drum in present]


#: Why a band was not searched, keyed by the reason `_active_bands` gives.
#:
#: v5 first shipped one sentence for all three — "they hold nothing, or nothing
#: transient" — and the corpus showed why that is not good enough. On "When the
#: Levee Breaks" the kick band holds **37% of the stem's energy** and peaks at
#: 6.48 flux, and the tool reported no kick at all; a reader told the band held
#: "nothing, or nothing transient" would go looking for a silent band and find
#: the loudest one in the source. The third message is the one that finding
#: bought, and it names what a reader can actually do about it.
_DORMANT_CAVEATS: Final[dict[str, str]] = {
    "empty": (
        "no hits looked for in these bands — they hold under "
        f"{BAND_ACTIVITY_FLOOR:g} of this source's energy, which is residue "
        "rather than content: {bands}"
    ),
    "no_transient": (
        "no hits looked for in these bands — they hold energy but no transient, "
        "so there is a level or a tone in them and nothing that starts: {bands}"
    ),
    "not_sparse": (
        "no hits looked for in these bands — they are full of transients that "
        "never separate. The energy does not fall back between hits, which is a "
        "compressed or reverberant source (or a roll), and this module cannot "
        "tell one hit from the next in it. **Any drum living in these bands is "
        "missing from the counts below, however loud it is.** An envelope fold "
        "can still see the pattern where per-hit detection cannot: {bands}"
    ),
}


def _dormant_caveats(dormant: dict[str, str]) -> list[str]:
    """One caveat per distinct reason, naming the bands it applies to."""
    return [
        _DORMANT_CAVEATS[reason].format(
            bands=", ".join(name for name in DETECTION_BANDS if dormant.get(name) == reason)
        )
        for reason in _DORMANT_CAVEATS
        if reason in dormant.values()
    ]


def _kick_survival_caveat(candidates: list[_Candidate], active: Sequence[str]) -> str | None:
    """Report a kick band that was searched and produced almost no kicks.

    The sibling of `_dormant_caveats`, which covers a band that was never
    searched at all. This is the case that was silent: the band is active, hits
    come out of it, and classification then throws nearly all of them away — so
    the kick pattern is a transcription of whatever survived rather than of the
    source, and nothing in the output said so. See
    `kick_survival_caveat_fraction` in `THRESHOLDS` for the corpus measurement
    and why the bound sits where it does.

    Only the kick is checked. It is the one detection band whose name maps 1:1
    to a class -- a snare is found in `body` and a hat in `noise`/`air`, and
    neither mapping is clean enough to count survivors against. It is also the
    class whose absence most changes what a reader builds from the grid.

    Returns `None` when the band is dormant (`_dormant_caveats` has already
    said so, and 0 of 0 is not a failure rate) or when enough survived.
    """
    if "kick" not in active:
        return None
    detected = [candidate for candidate in candidates if candidate.band == "kick"]
    if not detected:
        return None
    survived = sum(1 for candidate in detected if candidate.drum == "kick")
    share = survived / len(detected)
    if share >= THRESHOLDS["kick_survival_caveat_fraction"]:
        return None
    return (
        f"the kick band was searched and found {len(detected)} hits, but only "
        f"{survived} of them ({share:.0%}) classified as kicks. Read the kick "
        "pattern as what survived classification, not as what this source "
        "plays -- on dense material the feature window catches the whole "
        "pattern, so no window is kick-dominated and genuine kicks are lost "
        "here rather than never detected"
    )


#: Why a grid was not reported, keyed by `_Grid.failure`. `BLOCK_STATUSES` is
#: frozen and has one `no_grid` for all of these, so the distinction lives here
#: — and it is a real distinction: "your tempo is 0.04 BPM out" and "this
#: material has no pulse" are different findings with different fixes, and v4
#: printed the same sentence for both.
_GRID_FAILURE_CAVEATS: Final[dict[str, str]] = {
    "no_cycle_length": (
        "no cycle grid: no usable tempo estimate, so there is no cycle length to "
        "fold onto"
    ),
    "not_periodic": (
        # `on_grid` is printed to three places, not two: it is compared against
        # `floor` in the same sentence, and a value just under the floor rounds
        # to the floor at two places. On Herbie Hancock's "Chameleon" the v5
        # calibration run produced "puts only 0.50 of the profile on the grid,
        # against 0.5 required", which reads as a contradiction. The measurement
        # was right and the format string was hiding the margin. (W8B; the
        # `{chance:.2f}` term is left at two places — it is a reference value,
        # not one of the two numbers being compared.)
        "no cycle grid: folding the band flux onto this cycle length puts only "
        "{on_grid:.3f} of the profile on the grid, against {floor:g} required and "
        "{chance:.2f} expected by chance. The hits do not fit any grid at this "
        "period — this is not a near miss."
    ),
    "no_fit": (
        "no cycle grid: hits sit {error:.2f} steps from their nearest step on "
        "average, over the {allowance:g} allowed, and each half of the source "
        "fails on its own phase too. The hits do not fit any grid here; the "
        "allowance is not the problem."
    ),
    "drifting": (
        "no cycle grid, but the hits DO fit a grid that is drifting: each half of "
        "the source fits its own phase to {first:.2f} and {second:.2f} steps, "
        "while the two halves disagree by {drift:+.2f} steps. That is a period "
        "error accumulating, not loose playing. Implied cycle {implied:.6f} s "
        "({implied_bpm:.3f} BPM at {beats} beats per cycle) — approximate, "
        "because some hits have already wrapped to the wrong step by the time a "
        "whole-source fit fails. Re-fit with a measured period from tempo.py."
    ),
}


def _grid_caveats(grid: _Grid, beats_per_cycle: int) -> list[str]:
    """Turn a grid's diagnostics into plain English. Nothing else reads `_Grid`."""
    caveats: list[str] = []
    if (
        grid.anchor_shift_steps is not None
        and abs(grid.anchor_shift_steps) >= ANCHOR_SNAP_CAVEAT_STEPS
    ):
        caveats.append(
            f"the grid anchor was moved {grid.anchor_shift_steps:+.2f} steps onto the "
            "hits' own phase. A supplied downbeat fixes which step is step 0, not "
            "the sub-step phase, and this move cannot change any step number."
        )
    if grid.on_grid_share is None and grid.cycle_seconds is not None:
        caveats.append(
            "the envelope fold could not be read — the source spans fewer than "
            f"{MIN_FOLD_CYCLES} whole cycles, or no step fires in more than half of "
            "them — so the grid rests on the per-hit fit alone"
        )
    if grid.failure is None:
        return caveats

    template = _GRID_FAILURE_CAVEATS.get(grid.failure)
    if template is None:  # pragma: no cover - every failure has a template
        return [*caveats, "no cycle grid"]
    implied = grid.implied_cycle_seconds
    caveats.append(
        template.format(
            on_grid=grid.on_grid_share if grid.on_grid_share is not None else float("nan"),
            floor=GRID_ON_GRID_SHARE_MIN,
            chance=1.0 / GRID_OVERSAMPLE,
            error=grid.error_steps if grid.error_steps is not None else float("nan"),
            allowance=MAX_QUANTISATION_ERROR_STEPS,
            first=grid.half_error_steps[0] if grid.half_error_steps else float("nan"),
            second=grid.half_error_steps[1] if grid.half_error_steps else float("nan"),
            drift=grid.drift_steps if grid.drift_steps is not None else float("nan"),
            implied=implied if implied else float("nan"),
            implied_bpm=(beats_per_cycle * 60.0 / implied) if implied else float("nan"),
            beats=beats_per_cycle,
        )
    )
    return caveats


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
    beat_period_seconds: float | None = None,
    downbeat_seconds: float | None = None,
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
        bpm: Tempo estimate, or `None`. Ignored when `beat_period_seconds` is
            supplied. Neither means no cycle length and therefore
            `status="no_grid"` — hits are still reported in full.
        beat_times: The rhythm block's beat positions. Only `beat_times[0]` is
            read, as the grid anchor, and only when `downbeat_seconds` is not
            supplied. **`onset_times` is deliberately not a parameter**: this
            module finds its own onsets per band, so its output is identical
            whichever backend produced the rhythm block.
        beats_per_cycle: Beats in one Strudel cycle. 4 is one bar of 4/4.
        beat_period_seconds: Seconds per beat from a real measurement —
            `tempo.TempoFit.period_seconds`. Takes precedence over `bpm`,
            because F1 showed a backend's BPM label is accurate to about
            +/- 0.2 BPM and a grid across a four-minute track needs four
            decimal places. `None` falls back to `bpm`, so this module stays
            testable on its own.
        downbeat_seconds: Where bar one starts —
            `tempo.DownbeatFit.offset_seconds`. Taken as given: **which** step
            is step 0 is the caller's call, and on four-on-the-floor material
            it is four-fold ambiguous, so re-deriving it here would silently
            rotate the caller's answer. Its sub-step *phase* is refined, which
            cannot renumber anything — see `_snap_anchor`. `None` falls back to
            `beat_times[0]`, then to the first hit.

    Returns:
        A `DrumDecomposition`. Never raises: any internal failure is caught and
        returned as `status="failed"` with the exception's class name in
        `caveats`, because a drum classifier is not worth losing the rest of an
        analysis over.

    Status:
        `ok` grid and hits; `no_grid` hits but no usable grid; `too_few_hits`
        nothing detected; `failed` an exception was swallowed. `no_grid` covers
        three different failures and `caveats` says which one — see
        `_GRID_FAILURE_CAVEATS`.
    """
    try:
        return _decompose(
            audio,
            sample_rate,
            bpm,
            beat_times,
            beats_per_cycle,
            beat_period_seconds,
            downbeat_seconds,
        )
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
    beat_period_seconds: float | None = None,
    downbeat_seconds: float | None = None,
) -> DrumDecomposition:
    """The body of `decompose`, without the never-raise wrapper.

    Does exactly two things the envelope path cannot: the STFT, and the
    too-short check that needs to know how many frames came out of it.
    Everything after that is `_decompose_bands`.
    """
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

    return _decompose_bands(
        _band_envelopes(magnitude, freqs),
        sample_rate,
        bpm=bpm,
        beat_times=beat_times,
        beats_per_cycle=beats_per_cycle,
        beat_period_seconds=beat_period_seconds,
        downbeat_seconds=downbeat_seconds,
        magnitude=magnitude,
        freqs=freqs,
        caveats=caveats,
    )


def _decompose_bands(
    envelopes: dict[str, npt.NDArray[np.float64]],
    sample_rate: int,
    *,
    bpm: float | None,
    beat_times: Sequence[float],
    beats_per_cycle: int,
    beat_period_seconds: float | None = None,
    downbeat_seconds: float | None = None,
    magnitude: npt.NDArray[np.float64] | None = None,
    freqs: npt.NDArray[np.float64] | None = None,
    caveats: list[str] | None = None,
) -> DrumDecomposition:
    """Everything after the STFT: detect, classify, suppress bleed, fit a grid.

    Split out from `_decompose` because **every measurement that decides a hit's
    class or the grid is a function of the four band envelopes alone** — only
    `flatness`, which is reported and never read, needs the spectrum. That makes
    the four committed arrays in `tests/fixtures/real/` a complete input to this
    module, so real-material regressions can be asserted without shipping audio
    (ground rule 9 of `KICKOFF-v2.md`). It is a seam with a purpose, not a
    convenience: the alternative is a real-material test that cannot exist.
    """
    caveats = [] if caveats is None else caveats
    fluxes = {name: _spectral_flux(envelope) for name, envelope in envelopes.items()}
    active, dormant = _active_bands(envelopes, fluxes)
    caveats.extend(_dormant_caveats(dormant))

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
    bleed = _suppress_kick_bleed(candidates, sample_rate)
    if bleed:
        caveats.append(
            f"{bleed} detections in the {'/'.join(BLEED_TARGET_BANDS)} bands were the "
            "kick's own transient found a second time, not hits of their own — "
            f"their windows are over {KICK_BLEED_DOMINANCE:g} kick energy and their "
            f"air/(air+noise) is under {KICK_BLEED_AIR_OVER_NOISE:g}, which is a "
            "beater click rather than a hat"
        )
    candidates = _resolve_coincidences(candidates, MIN_HIT_SEPARATION_SECONDS, sample_rate)
    candidates.sort(key=lambda item: (item.frame, item.drum))

    times = [candidate.frame * STFT_HOP_LENGTH / sample_rate for candidate in candidates]
    grid = _fit_grid(
        times,
        fluxes,
        active,
        sample_rate,
        bpm=bpm,
        beat_period_seconds=beat_period_seconds,
        beat_times=beat_times,
        downbeat_seconds=downbeat_seconds,
        beats_per_cycle=beats_per_cycle,
    )

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
    kick_survival = _kick_survival_caveat(candidates, active)
    if kick_survival is not None:
        caveats.append(kick_survival)
    caveats.extend(_grid_caveats(grid, beats_per_cycle))

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
