"""Recover a track's arrangement: which stems are playing, bar by bar.

Pure numpy, no scipy, no backend. Same architectural shape as
`drum_elements.py`, `note_track.py` and `tempo.py`: **no librosa, no essentia**,
not at module level and not inside a function.

Why this module exists
----------------------

Calibration finding F7 calls this "the single highest-value addition", and the
reason is that everything else the pipeline produces describes *a loop*. One
BPM, one key, one drum grid, one bass line. A four-minute record is not a loop —
it is an intro, a build, a breakdown, a drop and an outro, and the only thing
separating those is which stems are playing.

That fact is already sitting in data the pipeline throws away. Per-bar RMS per
stem, thresholded, gives the structure directly. On the calibration track it
recovers, with no tuning beyond one threshold:

======== ======= ============= ==========================================
bars     label   length        measured presence
======== ======= ============= ==========================================
0-16     intro   17 bars       drums + other + kick; no bass, no vocals
19-24    groove  6 bars        drums + bass + kick
27-75    full    49 bars       all five
76-90    breakdn 15 bars       vocals + other only; **kick and bass out**
91-99    drop    9 bars        all five, straight out of the breakdown
144-146  silence 3 bars        nothing at all
======== ======= ============= ==========================================

F7 predicted a 16-bar intro on kick and pad, an 8-bar kick-and-bass section,
48 bars of full band, a 16-bar breakdown with kick and bass out, a drop, and
total silence near the end. Every one of those is above, every one with the
right content, and each is within a bar or two of the length F7 states. This is
the one Part-1 finding that reproduced essentially without correction.

**The differences are reported, not tuned away.** F7 says the breakdown is
16 bars at 75-90; measured here it is **15 bars at 76-90**, because bar 75 is a
transitional bar whose kick-band level sits at 17.4 against a threshold of 16.9
— 3% over the line. Raising `PRESENCE_FRACTION` from 0.15 to 0.18 drops bar 75
out and reproduces F7's 75-90 exactly. That was not done: nothing recommends
0.18 over 0.15 except that it matches a number arrived at by hand, and this
cycle has spent an afternoon on what happens when a threshold is moved to make
one track agree.

The breakdown row above is also **two adjacent `Section` objects**, 76-88 and
89-90, both labelled `breakdown`. The drums stem comes back for the last two
bars of it while the kick stays out, and that is a genuine change in who is
playing — a fill leading back into the drop. Merging it away would need a
smoothing rule whose only justification is that the answer would look tidier.
The kick and bass are absent across all 15 bars either way, which is the fact
F7 recorded and the fact the test asserts.

**Verse and chorus are deliberately not attempted.** Telling a verse from a
chorus needs repetition analysis — the same material returning — which is a
different measurement from "who is playing", and is out of scope. The labels
here are `intro`, `breakdown`, `drop`, `outro`, `full`, `groove` and `silence`,
every one of them derived from a presence pattern and every one of them carrying
that pattern in `label_reason` so a reader can disagree with the label without
losing the measurement. Same auditability rule as `heuristics.py`.


The period and the downbeat are arguments, not derivations
-----------------------------------------------------------

`tempo.py` owns both, and W6 wires them. Nothing here re-derives a grid: handed
a wrong period this module will fold onto a wrong grid and say so only through
whatever the caller already knows about the fit's confidence. That is the
correct division — a module that quietly disagreed with `tempo.py` about the bar
length would be the F1 bug again, from the other end.

When no period is supplied at all the answer is `status="no_grid"` and no
sections. That is the intended outcome on ambient material: `refine_bpm` refuses
a tempo for the corpus' Brian Eno row, so arrangement refuses too, rather than
inventing a bar length in order to have something to say.


"The drums stem is loud" and "the kick is playing" are different facts
----------------------------------------------------------------------

Hence `kick` is a sixth track alongside the four stems, taken from the drums
stem's own 20-150 Hz band rather than from its broadband RMS. It earns its place
on the calibration track's breakdown: across bars 76-90 the drums stem's RMS
clears its own presence threshold in **3 of those 15 bars** — percussion and a
reverb tail are still there — while the kick band clears its threshold in
**0 of 15**. A structure built on drums-stem RMS alone reports a breakdown with
holes punched in it; the kick band says plainly that the four-on-the-floor
stopped and did not restart until bar 91.

The kick track is not gated independently against the other stems — its units
are STFT band energy and theirs are sample RMS, which are not on a common scale.
It inherits the drums stem's absence gate instead, which is exactly right: the
kick band is computed *from* the drums stem, so a drums stem that is not in the
record has no kick in it either.


The threshold, and the failure it is guarding against
------------------------------------------------------

Presence is two tests, and the second one is the one that matters.

**Per-bar, relative to the stem itself.** A bar counts as active when the stem's
RMS in it clears `PRESENCE_FRACTION` of that stem's own 90th-percentile bar. A
relative test is unavoidable — stems arrive at wildly different levels, and an
absolute dBFS floor would be a mastering-level threshold wearing a musical
costume. The 90th percentile rather than the peak because one crash cymbal
should not set the bar for a whole stem.

**Per stem, relative to the record.** A relative test alone is *scale-free*,
which is the trap: run it on a stem that contains nothing but separation bleed
and it will faithfully report that the bleed is loud in some bars and quiet in
others, and hand back a drum arrangement for a record with no drums. The
calibration corpus contains exactly that case on purpose. So a stem is first
compared against the loudest stem in the same record, and one below
`STEM_ACTIVITY_FLOOR` is marked `status="absent"` and reported as playing in no
bar at all.

Measured across all five corpus tracks, as each stem's 90th-percentile bar over
the loudest stem's:

=========== ======== ======== ======== ========
track       drums    bass     vocals   other
=========== ======== ======== ======== ========
madonna     0.9106   1.0000   0.7806   0.6172
badu        1.0000   0.4706   0.5071   0.3771
roni        0.5142   1.0000   0.4118   0.2696
levee       0.6765   0.7526   0.7846   1.0000
**eno**     0.00138  0.2994   0.00033  1.0000
=========== ======== ======== ======== ========

Eighteen stems are genuinely in their record and the quietest of them reads
0.2696. Two are separation residue — the Eno drums stem peaks at -55 dBFS and
its vocals stem at -71 dBFS on a track with neither — and the louder of those
reads 0.00138. The gap is a factor of **195**, and it is a gap in the right
place: the four stems that are quiet *because the mix is quiet* sit with the
present ones, because the comparison is within a record.

`STEM_ACTIVITY_FLOOR` is placed at the geometric midpoint of that gap.


What this produces on the corpus, including where it is thin
--------------------------------------------------------------

============ ======== =========== ========================================
track        bars     sections    outcome
============ ======== =========== ========================================
madonna      147      16          F7's structure, at close to F7's lengths
badu         137      6           intro, groove, full, groove, full, outro
roni         214      44          faithful, and not useful — see below
levee        257      21          solo-drum intro isolated as bars 0-3
eno          --       0           refused: no period, `status="no_grid"`
============ ======== =========== ========================================

Three of those need saying plainly.

**Levee's tempo genuinely drifts.** `refine_bpm` returns `status="coarse"` with
confidence 0.000 on it, so the ~143.5 BPM this folds at is a coarse estimate and
the bar boundaries slide against the music across seven minutes. The *sections*
survive that — an entry or an exit is a step of two or three orders of magnitude
in a stem's level, and a bar boundary landing half a bar out moves the reported
start bar, not the existence of the section. Nothing here may be read as placing
a section to the bar on that track, and `_assemble` adds a caveat saying so
whenever the caller reports a weak grid. What it does recover cleanly is the
solo-drum opening: bars 0-3 are drums and kick alone, and the bass and pad enter
together at bar 4 with the bass stepping from 0.00005 to 0.0698, a factor of
1400. (That is a 6.7 s intro at 143.5 BPM, shorter than the record's reputation.
Either the corpus file is an edit or the coarse BPM is doubled; the *boundary*
is not in doubt, and this module does not get a vote on the tempo.)

**Roni is faithful and it is not useful, and the distinction matters.** 44
sections on a 214-bar track looks like a threshold firing on noise. It is not.
That track's bassline runs a four-bar cycle — two bars at 0.05, two bars at
0.00005, a factor of a thousand, over and over from bar 11 to bar 35 — so
"which stems are playing in this bar" genuinely alternates every two bars, and
44 sections is the correct answer to the question this module asks. It is the
question that is wrong for the material: at 170 BPM a bar is 1.4 s, and a
riff with rests in it is not an arrangement. **This is a real limitation of
presence-based segmentation on dense, fast, sparse-bass material and it is not
fixable by moving a threshold** — raising `MIN_SECTION_BARS` to 4 does take
Roni from 44 sections to 14, and it also takes the hip-hop row from 6 sections
to 2, deleting a real 2-bar intro and a real 3-bar outro. Repetition analysis
would answer it, and that is the same tool verse/chorus needs and is equally
out of scope here.

**Eno produces nothing, and that is the result.** With no period there is no
fold. Forced onto a nominal 120 BPM grid as an experiment, the absence gate does
its job — drums and vocals are correctly marked `absent` rather than reported as
an arrangement, and no `breakdown` or `drop` label can fire once there is no
kick in the record — and what remains is bass and pad drifting in and out over
521 bars. The section count on that forced grid rises monotonically with
`PRESENCE_FRACTION`, 37 at 0.03 to 116 at 0.50, which is the signature of a
threshold sliding up and down a continuous swell rather than of a structure
being found. The honest answer on ambient material is that this module has
nothing to say, and the way it says so is by requiring a grid it will not
invent.


Nothing in this module raises
------------------------------

Every entry point returns a dataclass whose `status` and `caveats` say what
happened. The dataclasses are plain — W6 promotes them into `schemas.py` — so
this module stays importable without pydantic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from . import ANALYSIS_SAMPLE_RATE, STEM_NAMES
from .drum_elements import DETECTION_BANDS, STFT_HOP_LENGTH, _band_envelope, _stft_magnitude

__all__ = [
    "BEATS_PER_BAR",
    "HOP_SECONDS",
    "KICK_BAND_HZ",
    "KICK_TRACK",
    "MIN_FINAL_BAR_FRACTION",
    "MIN_SECTION_BARS",
    "PRESENCE_FRACTION",
    "PRESENCE_REFERENCE_PERCENTILE",
    "STEM_ACTIVITY_FLOOR",
    "THRESHOLDS",
    "TRACK_NAMES",
    "Arrangement",
    "BarEnergy",
    "Presence",
    "Section",
    "TrackPresence",
    "arrangement",
    "arrangement_from_frames",
    "frame_rms",
    "label_sections",
    "per_bar_energy",
    "per_bar_energy_from_frames",
    "presence",
    "segment",
]


# ---------------------------------------------------------------------------
# The pinned analysis grid
# ---------------------------------------------------------------------------

#: Seconds between frames, pinned by `drum_elements.STFT_HOP_LENGTH` at
#: `ANALYSIS_SAMPLE_RATE`: 512 / 44100 = 11.60998 ms. Every committed
#: `*__stem_frame_rms.npz` fixture stores this same number as `hop_seconds`, so
#: a test never has to assume it. The per-stem RMS blocks are non-overlapping
#: and this long, so frame `k` here is samples `[k*512, (k+1)*512)`.
HOP_SECONDS: Final[float] = STFT_HOP_LENGTH / ANALYSIS_SAMPLE_RATE

#: The band the kick track is read from, in Hz. **Identical to
#: `drum_elements.DETECTION_BANDS["kick"]`** — not a new band and not a tuning
#: choice, so "the kick is playing" means the same thing here as it does in the
#: drum decomposition. `tools/make-fixtures/` writes the committed
#: `kick_band_energy` array from this same definition.
KICK_BAND_HZ: Final[tuple[float, float]] = DETECTION_BANDS["kick"]

#: Name of the kick track in every result. Deliberately not a stem name: it is
#: derived from the drums stem, not separated alongside it.
KICK_TRACK: Final[str] = "kick"

#: Every track a result may carry, in a fixed order so two runs are comparable.
TRACK_NAMES: Final[tuple[str, ...]] = (*STEM_NAMES, KICK_TRACK)

#: Beats per bar assumed when a caller does not say. Four, because nothing
#: upstream measures a time signature — `tempo.find_downbeat` takes the same
#: assumption as a parameter and this mirrors it. A caller that knows better
#: passes `beats_per_bar`.
BEATS_PER_BAR: Final[int] = 4


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Same candour convention as `heuristics.THRESHOLDS`, `drum_elements.THRESHOLDS`
# and `tempo.THRESHOLDS`: `[grounded]` means the number follows from something
# physical, arithmetic or *measured*; `[guess]` means it is a plausible starting
# point that the calibration corpus should revisit.
#
# Every number here was measured against all five committed
# `tests/fixtures/real/*__stem_frame_rms.npz` fixtures — house, hip-hop, drum
# and bass, live band and ambient — not against one record. That is deliberate:
# a presence threshold is exactly the shape of threshold this project spent an
# afternoon discovering had been calibrated against a single house track.

THRESHOLDS: Final[dict[str, float]] = {
    # -- Presence ----------------------------------------------------------
    # [grounded, measured] Fraction of a stem's own reference bar that a bar
    # must clear to count as active. The plan proposed 0.15 from one track.
    # Swept 0.03-0.50 across all five fixtures, scored on the longest run of
    # bars with kick and bass both absent on the calibration track — the
    # breakdown F7 measured by hand:
    #
    #   0.03   bars 78-84, 7 bars     -- broken
    #   0.05   bars 76-89, 14 bars
    #   0.08   bars 76-90, 15 bars
    #   0.10   bars 76-90, 15 bars
    #   0.15   bars 76-90, 15 bars    <- here
    #   0.18   bars 75-90, 16 bars    -- F7's exact answer
    #   0.25   bars 75-90, 16 bars
    #   0.50   bars 75-90, 16 bars
    #
    # It is a **plateau, not a peak**, and that is the useful thing to know
    # about it: everything from 0.05 to 0.50 recovers the breakdown, a
    # sixteen-fold range. The reason is that the thresholded quantity is
    # bimodal by three orders of magnitude — a stem that is not playing is
    # *silent*, not merely quiet. Within the breakdown the bass reads 0.0001
    # against its own 0.158 reference, so no threshold in this range can
    # disagree about it.
    #
    # Kept at the plan's 0.15. Note 0.18 would reproduce F7's hand-measured
    # 75-90 exactly, by dropping a single transitional bar whose kick sits 3%
    # over the line. That is not a reason to move it — see the module
    # docstring. The one edge that is real is the bottom: below 0.05 reverb
    # tails and bleed start reading as presence and the breakdown fragments.
    "presence_fraction": 0.15,
    # [grounded] Percentile of a stem's own per-bar distribution the fraction
    # above is taken against. The peak would let one crash cymbal set the
    # reference for a whole stem; the median would be dragged down by a stem
    # that rests for half the record, which is normal for vocals (the
    # calibration track's vocals stem is silent in 39% of its bars). The 90th
    # percentile is "a loud bar for this stem" without being "the loudest bar
    # in the record", which is the quantity actually wanted.
    "presence_reference_percentile": 90.0,
    # [grounded, measured] How loud a stem's reference bar must be relative to
    # the loudest stem's, before the stem is believed to be in the record at
    # all. **This is the gate that stops the module manufacturing an
    # arrangement out of separation bleed**, and it is the one threshold here
    # with a real corpus behind it. Measured, reference bar over the loudest
    # stem's reference bar, twenty stems across five tracks:
    #
    #   genuinely present (18 stems)   0.2696 .. 1.0000
    #   separation residue  (2 stems)  0.00033 and 0.00138
    #
    # The two residue readings are the ambient track's drums and vocals stems,
    # on a record with no drums and no voice: they peak at -55 and -71 dBFS.
    # The gap between the quietest real stem and the loudest fake one is a
    # factor of **195**, and 0.02 is its geometric midpoint — 13.5x below the
    # lowest present reading, 14.5x above the highest absent one.
    #
    # Note it is a *within-record* comparison, which is what makes it work: a
    # quiet mix moves every stem together and the ratio does not move. An
    # absolute dBFS floor would have been a mastering-level threshold wearing
    # a musical costume.
    #
    # [guess] on the absent side, honestly: n=2, from one ambient track. The
    # margin is enormous and the failure direction is safe (a genuinely very
    # quiet stem is reported as absent, which loses a section boundary, rather
    # than bleed being reported as an arrangement, which invents one). But two
    # readings are not a distribution.
    "stem_activity_floor": 0.02,
    # -- Segmentation ------------------------------------------------------
    # [grounded, measured] Shortest section kept. Anything shorter is merged
    # into a neighbour. Two bars is the plan's number, and the corpus both
    # confirms it and rules out the obvious next value. A one-bar change of
    # who is playing is a fill or a dropped beat, not a section: on the
    # calibration track the raw presence map holds 40 runs, **22 of them a
    # single bar**, and merging those leaves 16 sections whose lengths are
    # 17, 6, 49, 15, 9 and so on -- the arrangement.
    #
    # Section count against this threshold, all five fixtures:
    #
    #        madonna  badu  roni  levee  eno(forced)
    #   1        40      9    89     29     108
    #   2        16      6    44     21      72     <- here
    #   3        11      3    18     18      53
    #   4        11      2    14     12      33
    #   8         6      2     9      8      14
    #
    # Raising it to 4 is tempting because it takes the drum-and-bass row from
    # 44 sections to 14. **It also takes the hip-hop row from 6 to 2**, where
    # what is lost is a real 2-bar pad intro and a real 3-bar outro, and what
    # is left calls the first 14 bars "intro" and the remaining 123 "full".
    # Deleting true sections on a conventional arrangement to tidy up an
    # unconventional one is the wrong trade, so this stays at 2 and Roni's 44
    # sections are reported as what they are.
    "min_section_bars": 2.0,
    # [grounded, measured] Fraction of a bar's worth of audio that must be
    # present for a trailing partial bar to be counted as a bar. Half, which
    # makes the bar count independent of the downbeat within the range a
    # downbeat measurement actually lives in — it is a measured quantity with
    # its own error, and the bar count is a property of the record.
    #
    # The calibration track is 267.5 s at 132.000 BPM: 147.12 bars counted
    # from zero and 146.99 from its measured 0.2322 s downbeat. Measured, the
    # count is **147 for every offset from 0 to 0.9 s** and 146 beyond that,
    # so it agrees with the verified 147 across the whole first half-bar. A
    # plain floor would have reported 147 and 146 for those two offsets, and
    # a plain ceiling 148 and 147.
    "min_final_bar_fraction": 0.5,
}

PRESENCE_FRACTION: Final[float] = THRESHOLDS["presence_fraction"]
PRESENCE_REFERENCE_PERCENTILE: Final[float] = THRESHOLDS["presence_reference_percentile"]
STEM_ACTIVITY_FLOOR: Final[float] = THRESHOLDS["stem_activity_floor"]
MIN_SECTION_BARS: Final[int] = int(THRESHOLDS["min_section_bars"])
MIN_FINAL_BAR_FRACTION: Final[float] = THRESHOLDS["min_final_bar_fraction"]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
# Plain dataclasses on purpose, exactly as in `tempo.py`. W6 promotes these into
# `schemas.py` as pydantic models; until then this module stays importable
# without pydantic and testable without a schema version bump.

EnergyStatus = Literal["ok", "no_grid", "too_short"]
PresenceStatus = Literal["ok", "unavailable"]
TrackStatus = Literal["measured", "absent"]
ArrangementStatus = Literal["ok", "no_grid", "too_short", "unavailable"]
SectionLabel = Literal["intro", "outro", "breakdown", "drop", "full", "groove", "silence"]


@dataclass(frozen=True)
class BarEnergy:
    """One level per bar per track — the whole measurement this module makes.

    `levels` maps a name in `TRACK_NAMES` to `bar_count` values. Every value is
    an amplitude, not a power: stem tracks are the RMS of the bar's samples, and
    the kick track is the square root of its mean band energy, so the two scale
    the same way with level and a single fractional threshold means the same
    thing on both.
    """

    levels: dict[str, tuple[float, ...]]
    bar_count: int
    #: Seconds per bar, `beat_period_seconds * beats_per_bar`.
    bar_seconds: float
    #: Where bar zero starts, in seconds from the start of the source.
    downbeat_seconds: float
    beats_per_bar: int
    status: EnergyStatus
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrackPresence:
    """Whether one track is playing, bar by bar, and what decided that.

    Both the threshold and the reference it came from are kept, because
    "this stem was quiet in bar 40" and "this stem is quiet everywhere" are
    different facts and only the second one is a property of the record.
    """

    name: str
    present: tuple[bool, ...]
    #: The absolute level a bar had to clear. `reference * PRESENCE_FRACTION`.
    threshold: float
    #: This track's own `PRESENCE_REFERENCE_PERCENTILE` bar.
    reference: float
    #: `reference` over the loudest stem's `reference`. The quantity
    #: `STEM_ACTIVITY_FLOOR` is applied to. `None` for the kick track, which
    #: inherits the drums stem's decision rather than being gated on its own —
    #: band energy and sample RMS are not on a common scale.
    relative_level: float | None
    #: `absent` means the track was gated out of the record entirely and
    #: `present` is all-False regardless of what its own distribution said.
    status: TrackStatus
    #: Number of bars the track is active in. Redundant, and worth carrying:
    #: it is the first thing anyone checks when a section looks wrong.
    active_bars: int


@dataclass(frozen=True)
class Presence:
    """Binary per-bar presence for every track."""

    tracks: tuple[TrackPresence, ...]
    bar_count: int
    #: Names gated out by `STEM_ACTIVITY_FLOOR`, in `TRACK_NAMES` order.
    absent_tracks: tuple[str, ...]
    status: PresenceStatus
    caveats: tuple[str, ...] = ()

    def by_name(self) -> dict[str, TrackPresence]:
        """The tracks keyed by name, for callers that want a lookup."""
        return {track.name: track for track in self.tracks}


@dataclass(frozen=True)
class Section:
    """A run of bars over which the same set of tracks is playing.

    `label` is derived and `label_reason` is the evidence, so a reader who
    disagrees with the name still has the measurement. Same rule as
    `heuristics.py`: a label that cannot be audited is decoration.
    """

    start_bar: int
    length_bars: int
    start_seconds: float
    #: Tracks playing throughout, in `TRACK_NAMES` order.
    active: tuple[str, ...]
    label: SectionLabel | None = None
    #: Human-readable statement of the presence pattern and the position that
    #: produced `label`.
    label_reason: str | None = None


@dataclass(frozen=True)
class Arrangement:
    """The sections of a track, or an honest statement that there are none."""

    sections: tuple[Section, ...]
    bar_count: int
    bar_seconds: float | None
    downbeat_seconds: float | None
    #: Tracks that are not in this record at all — see `STEM_ACTIVITY_FLOOR`.
    absent_tracks: tuple[str, ...] = ()
    status: ArrangementStatus = "unavailable"
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Frames:
    """Per-frame levels an entry point derived from raw samples."""

    levels: dict[str, npt.NDArray[np.float64]]
    hop_seconds: float
    caveats: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _clean(values: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """Finite float64 view of a level array. A NaN or infinity is not energy.

    Deliberately the same helper, with the same reasoning, as `tempo._clean`.
    Not imported from there: `tempo.py` owns three names shared with
    `drum_elements` already and a fourth cross-module coupling to save four
    lines is not worth the merge-order risk.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def frame_rms(samples: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """RMS per non-overlapping `STFT_HOP_LENGTH` block of mono samples.

    The definition the committed fixtures were written with — see
    `tools/make-fixtures/make_real_fixtures.py`, which uses the identical
    reshape — so a fixture array and a live one are the same measurement on the
    same grid. Any trailing partial block is dropped rather than being averaged
    over a shorter window, which would give it a level it did not have.
    """
    mono = _clean(samples)
    count = mono.size // STFT_HOP_LENGTH
    if count < 1:
        return np.zeros(0, dtype=np.float64)
    blocks = mono[: count * STFT_HOP_LENGTH].reshape(count, STFT_HOP_LENGTH)
    return np.asarray(np.sqrt(np.square(blocks).mean(axis=1)), dtype=np.float64)


def _bar_count(frames: int, hop_seconds: float, bar_seconds: float, offset: float) -> int:
    """How many bars fit after `offset`, counting a long-enough final partial.

    See `MIN_FINAL_BAR_FRACTION` for why the partial bar is conditional: it is
    what makes the bar count independent of where the downbeat landed inside the
    first bar, which it must be, because the downbeat is a measurement with its
    own error and the bar count is a property of the record.
    """
    span = frames * hop_seconds - offset
    if span <= 0.0:
        return 0
    whole = int(math.floor(span / bar_seconds))
    remainder = span - whole * bar_seconds
    return whole + (1 if remainder >= MIN_FINAL_BAR_FRACTION * bar_seconds else 0)


def _fold(
    values: npt.NDArray[np.float64],
    hop_seconds: float,
    bar_seconds: float,
    offset: float,
    bar_count: int,
    *,
    already_power: bool,
) -> tuple[float, ...]:
    """Reduce a per-frame array to one amplitude per bar.

    RMS across the bar, not the mean of the frames' RMS values. The blocks are
    equal length, so the root of the mean of the squares *is* the RMS of the
    bar's samples, while a mean of RMS values is a different and smaller
    quantity that under-reads any bar containing both a hit and a gap.

    `already_power` covers the kick track, whose fixture array is band energy
    (`magnitude ** 2` summed over bins) rather than an amplitude. Rooting the
    mean of that puts it on the same amplitude-like scale as the stems, so
    `PRESENCE_FRACTION` means the same thing on both.
    """
    power = values if already_power else np.square(values)
    out: list[float] = []
    for index in range(bar_count):
        low = int(round((offset + index * bar_seconds) / hop_seconds))
        high = int(round((offset + (index + 1) * bar_seconds) / hop_seconds))
        low = max(0, low)
        high = min(values.size, high)
        out.append(float(np.sqrt(power[low:high].mean())) if high > low else 0.0)
    return tuple(out)


# ---------------------------------------------------------------------------
# Task 1 — per-bar energy
# ---------------------------------------------------------------------------


def per_bar_energy_from_frames(
    levels: Mapping[str, npt.NDArray[np.floating[Any]]],
    hop_seconds: float,
    beat_period_seconds: float | None,
    downbeat_seconds: float | None = 0.0,
    *,
    kick_band_energy: npt.NDArray[np.floating[Any]] | None = None,
    beats_per_bar: int = BEATS_PER_BAR,
) -> BarEnergy:
    """Fold pre-computed per-frame levels onto a bar grid.

    The entry point the committed fixtures use, and the reason this module is
    testable against five real tracks without shipping audio: a
    `tests/fixtures/real/<track>__stem_frame_rms.npz` feeds straight in, its
    `rms_*` arrays as `levels` and its `kick_band_energy` as the keyword.

    Args:
        levels: Stem name to per-frame **RMS**. Keys outside `STEM_NAMES` are
            ignored; missing stems are simply not in the result, so a source
            with no vocals stem is not a failure.
        hop_seconds: Seconds per frame. `HOP_SECONDS` for anything on the
            pipeline's own grid, and stored in every fixture.
        beat_period_seconds: From `tempo.TempoFit.period_seconds`. `None` or
            non-positive gives `status="no_grid"` and no bars — this module
            does not invent a bar length.
        downbeat_seconds: From `tempo.DownbeatFit.offset_seconds`. `None` is
            treated as 0.0 with a caveat, because folding from the start of the
            file is a real answer whose step numbering is rotated, not a
            failure.
        kick_band_energy: The drums stem's `KICK_BAND_HZ` **energy** per frame.
            Optional; without it the result simply has no kick track.
        beats_per_bar: See `BEATS_PER_BAR`.

    Returns:
        A `BarEnergy`. Never raises.
    """
    caveats: list[str] = []
    if beat_period_seconds is None or not math.isfinite(beat_period_seconds):
        return _no_grid("no beat period was supplied, so there is no bar to fold onto")
    if beat_period_seconds <= 0.0:
        return _no_grid(f"beat period {beat_period_seconds} is not a positive number of seconds")
    if not math.isfinite(hop_seconds) or hop_seconds <= 0.0:
        return _no_grid("frame hop is not a positive number of seconds")
    if beats_per_bar < 1:
        return _no_grid(f"beats_per_bar {beats_per_bar} is not a positive whole number")

    offset = downbeat_seconds if downbeat_seconds is not None else 0.0
    if downbeat_seconds is None:
        caveats.append(
            "no downbeat was supplied, so bars are counted from the start of the "
            "source and every section boundary may be rotated within a bar"
        )
    if not math.isfinite(offset) or offset < 0.0:
        caveats.append(f"downbeat {downbeat_seconds} is not usable; counted from zero instead")
        offset = 0.0

    arrays: dict[str, tuple[npt.NDArray[np.float64], bool]] = {}
    for name in STEM_NAMES:
        if name in levels:
            arrays[name] = (_clean(levels[name]), False)
    if kick_band_energy is not None:
        arrays[KICK_TRACK] = (_clean(kick_band_energy), True)
    if not arrays:
        return _no_grid("no stem levels were supplied")

    bar_seconds = beat_period_seconds * beats_per_bar
    frames = min(array.size for array, _ in arrays.values())
    bar_count = _bar_count(frames, hop_seconds, bar_seconds, offset)
    if bar_count < 1:
        return BarEnergy(
            levels={},
            bar_count=0,
            bar_seconds=bar_seconds,
            downbeat_seconds=offset,
            beats_per_bar=beats_per_bar,
            status="too_short",
            caveats=(
                *caveats,
                f"the source holds less than one {bar_seconds:.3f} s bar after the downbeat",
            ),
        )

    folded = {
        name: _fold(
            array, hop_seconds, bar_seconds, offset, bar_count, already_power=already_power
        )
        for name, (array, already_power) in arrays.items()
    }
    ordered = {name: folded[name] for name in TRACK_NAMES if name in folded}
    missing = [name for name in STEM_NAMES if name not in ordered]
    if missing:
        caveats.append(f"no levels supplied for {', '.join(missing)}; not represented in sections")
    return BarEnergy(
        levels=ordered,
        bar_count=bar_count,
        bar_seconds=bar_seconds,
        downbeat_seconds=offset,
        beats_per_bar=beats_per_bar,
        status="ok",
        caveats=tuple(caveats),
    )


def _no_grid(reason: str) -> BarEnergy:
    """A `BarEnergy` for the cases where there is no grid to fold onto."""
    return BarEnergy(
        levels={},
        bar_count=0,
        bar_seconds=0.0,
        downbeat_seconds=0.0,
        beats_per_bar=BEATS_PER_BAR,
        status="no_grid",
        caveats=(reason,),
    )


def per_bar_energy(
    stems: Mapping[str, npt.NDArray[np.floating[Any]]],
    sample_rate: int,
    beat_period_seconds: float | None,
    downbeat_seconds: float | None = 0.0,
    *,
    beats_per_bar: int = BEATS_PER_BAR,
) -> BarEnergy:
    """Per-bar level per stem, plus a kick track, from raw stem audio.

    Computes each stem's block RMS and — when a drums stem is present — the
    drums stem's `KICK_BAND_HZ` energy envelope on the project's pinned STFT
    grid, then delegates to `per_bar_energy_from_frames`, where all the
    reasoning lives.

    Args:
        stems: Stem name to mono or multi-channel float audio. Multi-channel is
            averaged to mono; nothing here is stereo-aware. Names outside
            `STEM_NAMES` are ignored.
        sample_rate: Must be `ANALYSIS_SAMPLE_RATE`. Any other rate still works
            and adds a caveat saying every threshold was calibrated at 44.1 kHz.
        beat_period_seconds: See `per_bar_energy_from_frames`.
        downbeat_seconds: See `per_bar_energy_from_frames`.
        beats_per_bar: See `BEATS_PER_BAR`.

    Returns:
        A `BarEnergy`. Never raises.
    """
    try:
        frames = _frames_from_stems(stems, sample_rate)
        result = per_bar_energy_from_frames(
            frames.levels,
            frames.hop_seconds,
            beat_period_seconds,
            downbeat_seconds,
            kick_band_energy=frames.levels.get(KICK_TRACK),
            beats_per_bar=beats_per_bar,
        )
        if not frames.caveats:
            return result
        return BarEnergy(
            levels=result.levels,
            bar_count=result.bar_count,
            bar_seconds=result.bar_seconds,
            downbeat_seconds=result.downbeat_seconds,
            beats_per_bar=result.beats_per_bar,
            status=result.status,
            caveats=(*frames.caveats, *result.caveats),
        )
    except Exception as error:  # noqa: BLE001 - deliberate: never break an analysis
        return _no_grid(f"per-bar energy failed with {type(error).__name__}")


def _to_mono(samples: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """Channel average. `audio_io.to_mono` without the import — this module
    deliberately depends on nothing that touches the filesystem."""
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim <= 1:
        return np.nan_to_num(array.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    axis = 0 if array.shape[0] < array.shape[-1] else -1
    averaged = array.mean(axis=axis).ravel()
    return np.asarray(np.nan_to_num(averaged, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)


def _frames_from_stems(
    stems: Mapping[str, npt.NDArray[np.floating[Any]]], sample_rate: int
) -> _Frames:
    """Block RMS per stem, plus the drums stem's kick-band energy."""
    caveats: list[str] = []
    if sample_rate != ANALYSIS_SAMPLE_RATE:
        caveats.append(
            f"analysed at {sample_rate} Hz; every threshold here was calibrated "
            f"at {ANALYSIS_SAMPLE_RATE} Hz"
        )
    levels: dict[str, npt.NDArray[np.float64]] = {}
    for name in STEM_NAMES:
        if name in stems:
            levels[name] = frame_rms(_to_mono(stems[name]))
    if "drums" in stems:
        magnitude, freqs = _stft_magnitude(_to_mono(stems["drums"]), sample_rate)
        levels[KICK_TRACK] = _band_envelope(magnitude, freqs, *KICK_BAND_HZ)
    return _Frames(levels=levels, hop_seconds=STFT_HOP_LENGTH / sample_rate, caveats=tuple(caveats))


# ---------------------------------------------------------------------------
# Task 2 — presence
# ---------------------------------------------------------------------------


def presence(energy: BarEnergy) -> Presence:
    """Threshold each track against a percentile of its own distribution.

    Two tests, and the module docstring argues at length for why the second one
    is the load-bearing half. Briefly: the per-bar test is relative to the
    track, so it is scale-free and would happily report an arrangement made of
    separation bleed; the per-track test is relative to the loudest stem in the
    same record, and that is what says "this stem is not in this record".

    The kick track is not gated on its own — it is band energy against the
    stems' sample RMS, which are not on a common scale — so it inherits the
    drums stem's verdict. A drums stem that is not in the record has no kick.

    Returns:
        A `Presence`. Never raises.
    """
    if energy.status != "ok" or energy.bar_count < 1 or not energy.levels:
        return Presence(
            tracks=(),
            bar_count=0,
            absent_tracks=(),
            status="unavailable",
            caveats=(f"per-bar energy is {energy.status}, so there is nothing to threshold",),
        )

    references = {
        name: float(
            np.percentile(np.asarray(values, dtype=np.float64), PRESENCE_REFERENCE_PERCENTILE)
        )
        for name, values in energy.levels.items()
    }
    stem_references = [references[name] for name in STEM_NAMES if name in references]
    loudest = max(stem_references) if stem_references else 0.0

    caveats: list[str] = []
    absent: list[str] = []
    tracks: list[TrackPresence] = []
    drums_absent = False
    for name in TRACK_NAMES:
        if name not in energy.levels:
            continue
        values = np.asarray(energy.levels[name], dtype=np.float64)
        reference = references[name]
        if name == KICK_TRACK:
            relative: float | None = None
            gated = drums_absent
        else:
            relative = reference / loudest if loudest > 0.0 else 0.0
            gated = relative < STEM_ACTIVITY_FLOOR
        if name == "drums":
            drums_absent = gated
        threshold = reference * PRESENCE_FRACTION
        flags = (
            np.zeros(values.size, dtype=bool)
            if gated or threshold <= 0.0
            else values >= threshold
        )
        if gated:
            absent.append(name)
        tracks.append(
            TrackPresence(
                name=name,
                present=tuple(bool(flag) for flag in flags),
                threshold=threshold,
                reference=reference,
                relative_level=relative,
                status="absent" if gated else "measured",
                active_bars=int(flags.sum()),
            )
        )

    if absent:
        detail = ", ".join(
            f"{track.name} ({track.relative_level:.5f} of the loudest stem)"
            if track.relative_level is not None
            else f"{track.name} (inherited from the drums stem)"
            for track in tracks
            if track.status == "absent"
        )
        caveats.append(
            f"treated as not present in this record at all: {detail}. Below "
            f"{STEM_ACTIVITY_FLOOR} of the loudest stem a separated stem is bleed, "
            "and thresholding bleed against itself invents an arrangement"
        )
    return Presence(
        tracks=tuple(tracks),
        bar_count=energy.bar_count,
        absent_tracks=tuple(absent),
        status="ok",
        caveats=tuple(caveats),
    )


# ---------------------------------------------------------------------------
# Task 3 — segmentation
# ---------------------------------------------------------------------------


def segment(
    presence_result: Presence,
    bar_seconds: float = 0.0,
    downbeat_seconds: float = 0.0,
    *,
    min_section_bars: int = MIN_SECTION_BARS,
) -> tuple[Section, ...]:
    """Collapse runs of identical presence patterns into sections.

    Two passes. The first is exact — consecutive bars with the same active set
    are one section, no tolerance and no smoothing, because the presence map is
    already binary and a soft boundary here would only hide the threshold.

    The second merges anything shorter than `min_section_bars` into a
    neighbour, shortest first so that a merge cannot strand an even shorter
    section behind it. The neighbour chosen is the one whose active set differs
    in the fewest tracks — a one-bar bass rest belongs to the section that has
    bass, not to whichever side happens to be longer — with ties broken by
    length and then by taking the earlier neighbour, so the result does not
    depend on iteration order.

    `bar_seconds` and `downbeat_seconds` only fill in `Section.start_seconds`;
    the segmentation itself is in bars and does not read them.
    """
    if presence_result.status != "ok" or presence_result.bar_count < 1:
        return ()

    patterns = _patterns(presence_result)
    runs: list[list[Any]] = []
    for index, pattern in enumerate(patterns):
        if runs and runs[-1][2] == pattern:
            runs[-1][1] += 1
        else:
            runs.append([index, 1, pattern])

    minimum = max(1, min_section_bars)
    while len(runs) > 1:
        shortest = min(range(len(runs)), key=lambda i: (runs[i][1], i))
        if runs[shortest][1] >= minimum:
            break
        target = _merge_target(runs, shortest)
        runs[target][1] += runs[shortest][1]
        runs.pop(shortest)
        runs = _coalesce(runs)

    return tuple(
        Section(
            start_bar=start,
            length_bars=length,
            start_seconds=downbeat_seconds + start * bar_seconds,
            active=pattern,
        )
        for start, length, pattern in runs
    )


def _coalesce(runs: Sequence[list[Any]]) -> list[list[Any]]:
    """Re-join adjacent runs holding the same pattern, and renumber the starts.

    Load-bearing rather than tidying. A merge replaces a short run's pattern
    with its neighbour's, which routinely leaves that neighbour abutting a
    third run carrying the *same* pattern — a one-bar fill between two halves
    of one section is exactly that shape. Without this step the calibration
    track reports its 48-bar full-band stretch as five consecutive sections
    with identical active sets, which is not wrong so much as useless.
    """
    joined: list[list[Any]] = []
    for run in runs:
        if joined and joined[-1][2] == run[2]:
            joined[-1][1] += run[1]
        else:
            joined.append([run[0], run[1], run[2]])
    start = 0
    for run in joined:
        run[0] = start
        start += run[1]
    return joined


def _patterns(presence_result: Presence) -> list[tuple[str, ...]]:
    """The active track set for each bar, in `TRACK_NAMES` order."""
    measured = [track for track in presence_result.tracks if track.status == "measured"]
    return [
        tuple(
            track.name
            for track in measured
            if index < len(track.present) and track.present[index]
        )
        for index in range(presence_result.bar_count)
    ]


def _merge_target(runs: Sequence[list[Any]], index: int) -> int:
    """Which neighbour a too-short run is folded into.

    Nearest in *content* first — the fewest tracks differing — because a
    one-bar gap in one stem is a fill inside a section, not a boundary between
    two. Length and then position break the ties, so the answer does not depend
    on which end the sweep started from.
    """
    candidates = [i for i in (index - 1, index + 1) if 0 <= i < len(runs)]
    own = set(runs[index][2])
    return min(
        candidates,
        key=lambda i: (len(own.symmetric_difference(set(runs[i][2]))), -runs[i][1], i),
    )


# ---------------------------------------------------------------------------
# Task 4 — labels
# ---------------------------------------------------------------------------


def label_sections(sections: Sequence[Section]) -> tuple[Section, ...]:
    """Attach a derived name to each section, with the evidence that produced it.

    **Held loosely, and the naming says so.** These are the six names F7 asked
    for plus `silence`, and each is a statement about who is playing and where
    in the record, nothing more:

    ``silence``
        No track active. Distinct from `outro` because "the record has stopped"
        and "the record is thinning out" are different, and F7 specifically
        names two bars of total silence on the calibration track.
    ``intro``
        The first section, when it is thinner than the full band.
    ``outro``
        Any section after the *last* full-band one. Deliberately asymmetric
        with `intro`: a record starts once and ends by degrees, and on the
        corpus the ending is routinely two or three sections of successive
        thinning before the silence. Measured on the calibration track, this
        is what stops the pad-only bars at 258 s being called a breakdown.
    ``breakdown``
        Neither kick nor bass, away from the ends — **and only on a record
        that has a kick at all**. Without that guard the label is vacuous:
        on the corpus' ambient row, where the drums stem is separation
        residue and correctly gated out, every section trivially has no kick
        and the whole track would come back as alternating breakdowns and
        drops. That is precisely the "manufacture structure out of nothing"
        failure the ambient fixture exists to catch, and it lives in the
        labels, not in the threshold.
    ``drop``
        Full band, immediately after a breakdown. Position, not loudness — a
        drop is defined here by what preceded it.
    ``full``
        Every track that is in the record is playing at once.
    ``groove``
        Everything else. The honest default, and on a dense record most of
        the track will be this.

    "In the record" means the union of every section's active set, not the
    five names this module can produce: a stem gated out by
    `STEM_ACTIVITY_FLOOR` must not make every section look thin.

    Verse and chorus are not attempted — see the module docstring.
    """
    if not sections:
        return ()

    in_record = {name for section in sections for name in section.active}
    full_indices = [i for i, s in enumerate(sections) if len(s.active) == len(in_record)]
    # With no full-band section anywhere, only the closing section can be an
    # outro — otherwise the rule would label the entire track one.
    last_full = max(full_indices) if full_indices else len(sections) - 2

    labelled: list[Section] = []
    previous_label: SectionLabel | None = None
    for index, section in enumerate(sections):
        label, reason = _label_one(
            active=set(section.active),
            active_names=section.active,
            in_record=in_record,
            first=index == 0,
            after_last_full=index > last_full,
            previous_label=previous_label,
        )
        labelled.append(
            Section(
                start_bar=section.start_bar,
                length_bars=section.length_bars,
                start_seconds=section.start_seconds,
                active=section.active,
                label=label,
                label_reason=reason,
            )
        )
        previous_label = label
    return tuple(labelled)


def _label_one(
    *,
    active: set[str],
    active_names: tuple[str, ...],
    in_record: set[str],
    first: bool,
    after_last_full: bool,
    previous_label: SectionLabel | None,
) -> tuple[SectionLabel, str]:
    """One section's label and the sentence justifying it.

    Order matters and is deliberate: silence beats everything, the two
    positional labels beat the content ones because "this is how the record
    starts" is the more useful statement about an opening section, and
    `breakdown` beats `groove` because a missing kick *and* a missing bass is
    the strongest content signal available here.
    """
    playing = ", ".join(active_names) if active_names else "nothing"
    band = ", ".join(sorted(in_record))
    if not active:
        return "silence", "no track is playing"
    thin = len(active) < len(in_record)
    if first and thin:
        return "intro", f"first section, thinner than the full band ({band}): {playing}"
    if after_last_full and thin:
        return "outro", f"after the last full-band section ({band}): {playing}"
    if (
        not first
        and not after_last_full
        and KICK_TRACK in in_record
        and KICK_TRACK not in active
        and "bass" not in active
    ):
        return "breakdown", f"neither kick nor bass, away from the ends: {playing}"
    if not thin:
        if previous_label == "breakdown":
            return "drop", f"the whole band ({band}) is back, straight out of a breakdown"
        return "full", f"every track that is in this record is playing: {playing}"
    return "groove", f"playing: {playing}"


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------


def arrangement_from_frames(
    levels: Mapping[str, npt.NDArray[np.floating[Any]]],
    hop_seconds: float,
    beat_period_seconds: float | None,
    downbeat_seconds: float | None = 0.0,
    *,
    kick_band_energy: npt.NDArray[np.floating[Any]] | None = None,
    beats_per_bar: int = BEATS_PER_BAR,
    grid_confidence: str | None = None,
) -> Arrangement:
    """All four steps, from pre-computed per-frame levels.

    Exists so a caller does not have to re-derive the order the four public
    functions run in. The steps stay separately callable — each is separately
    assertable, and W6 may well want the intermediate `Presence`.

    Args:
        grid_confidence: The confidence label the caller's tempo fit reported,
            if it has one. Purely for the caveat: `low`, `coarse` or
            `unavailable` adds a plain statement that section boundaries on
            this track are approximate. The corpus' live-band row genuinely
            drifts and its sections must not read as bar-exact.

    Returns:
        An `Arrangement`. Never raises.
    """
    try:
        energy = per_bar_energy_from_frames(
            levels,
            hop_seconds,
            beat_period_seconds,
            downbeat_seconds,
            kick_band_energy=kick_band_energy,
            beats_per_bar=beats_per_bar,
        )
        return _assemble(energy, grid_confidence)
    except Exception as error:  # noqa: BLE001 - deliberate: never break an analysis
        return Arrangement(
            sections=(),
            bar_count=0,
            bar_seconds=None,
            downbeat_seconds=None,
            status="unavailable",
            caveats=(f"arrangement failed with {type(error).__name__}",),
        )


def arrangement(
    stems: Mapping[str, npt.NDArray[np.floating[Any]]],
    sample_rate: int,
    beat_period_seconds: float | None,
    downbeat_seconds: float | None = 0.0,
    *,
    beats_per_bar: int = BEATS_PER_BAR,
    grid_confidence: str | None = None,
) -> Arrangement:
    """All four steps, from raw stem audio. See `arrangement_from_frames`."""
    try:
        energy = per_bar_energy(
            stems, sample_rate, beat_period_seconds, downbeat_seconds, beats_per_bar=beats_per_bar
        )
        return _assemble(energy, grid_confidence)
    except Exception as error:  # noqa: BLE001 - deliberate: never break an analysis
        return Arrangement(
            sections=(),
            bar_count=0,
            bar_seconds=None,
            downbeat_seconds=None,
            status="unavailable",
            caveats=(f"arrangement failed with {type(error).__name__}",),
        )


#: Grid confidence labels that make section boundaries approximate. `coarse` is
#: `TempoFit.status` when refinement was refused and `low` is its
#: `confidence_label`; both mean the bar length is an estimate rather than a
#: measurement, and on a seven-minute track an estimate slides.
_WEAK_GRID: Final[frozenset[str]] = frozenset({"low", "coarse", "unavailable", "unknown"})


def _assemble(energy: BarEnergy, grid_confidence: str | None) -> Arrangement:
    """Presence, segmentation and labels over an already-folded `BarEnergy`."""
    caveats = list(energy.caveats)
    if grid_confidence is not None and grid_confidence.lower() in _WEAK_GRID:
        caveats.append(
            f"the supplied bar grid is {grid_confidence}, so section boundaries are "
            "approximate: an entry or an exit is a step of orders of magnitude in a "
            "stem's level and survives a drifting grid, but the bar it is reported at "
            "does not"
        )
    if energy.status != "ok":
        return Arrangement(
            sections=(),
            bar_count=energy.bar_count,
            bar_seconds=energy.bar_seconds or None,
            downbeat_seconds=energy.downbeat_seconds,
            status=energy.status,
            caveats=tuple(caveats),
        )

    found = presence(energy)
    caveats.extend(found.caveats)
    sections = label_sections(
        segment(found, energy.bar_seconds, energy.downbeat_seconds)
    )
    return Arrangement(
        sections=sections,
        bar_count=energy.bar_count,
        bar_seconds=energy.bar_seconds,
        downbeat_seconds=energy.downbeat_seconds,
        absent_tracks=found.absent_tracks,
        status="ok",
        caveats=tuple(caveats),
    )
