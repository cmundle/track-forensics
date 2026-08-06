"""Map measured drum and bass evidence onto Strudel's default sound vocabulary.

Pure functions only: no audio, no I/O, no backend imports, no network at
runtime. Same contract as `heuristics.py`. Separate module from
`strudel_hints.py` (a different concern — that module condenses a whole
`TrackSummary`, this one maps individual measurements to sound names) and kept
file-disjoint from it so the two can be built in parallel.

`strudel_hints.build()` is the only intended caller: analysis modules emit
measurements, this module and that one turn them into hints. Nothing here
reads a `TrackSummary` or writes JSON.

---------------------------------------------------------------------------
Provenance — READ THIS BEFORE TRUSTING A "no default match" VERDICT
---------------------------------------------------------------------------
Strudel is actively developed and its sound list changes. The tables below
were transcribed by fetching the live docs, not written from memory. There is
no offline way to detect that they have gone stale — the date and URLs below
are the entire mitigation. A "match=none" verdict from this module is only as
current as `STRUDEL_DOCS_READ`; if that date is old, re-fetch the pages below
and diff the tables before trusting a "none" as still true. Every
`StrudelHints` this pipeline writes carries this date in
`strudel_vocabulary_read` for the same reason.

STRUDEL_DOCS_READ = "2026-08-04"

URLs actually fetched this session:

  - https://strudel.cc/learn/samples/   (drum/percussion sample names, .bank())
  - https://strudel.cc/learn/synths/    (waveforms, noise, FM, additive, wavetables, ZZFX)
  - https://strudel.cc/learn/notes/     (note-name syntax, MIDI convention)

What the docs said, and how it compares to the brief's "confirmed as of this
session" starting list:

  Drum/percussion samples (`learn/samples/`). The starting list named
  `bd, sd, hh, oh, cp`. The docs table names those five plus **more that the
  starting list did not mention**: `rim` (rimshot), `cr` (crash), `rd` (ride),
  `ht`/`mt`/`lt` (high/mid/low tom), `sh` (shaker), `cb` (cowbell), `tb`
  (tambourine), and generic `perc` (percussion), plus `misc` and `fx` grab-bag
  categories. `.bank()` is confirmed exactly as described: it prepends the
  drum-machine name to the sample name with `_`, so `s("bd sd").bank(
  "RolandTR808")` is shorthand for `s("RolandTR808_bd RolandTR808_sd")`, and
  the bank argument itself can be patterned (e.g.
  `.bank("<RolandTR808 RolandTR909>")`). The docs use `RolandTR808` and
  `RolandTR909` as examples, not as an exhaustive list of banks — this module
  does not attempt to enumerate banks, per the "no `.bank()` inference" rule
  below.

  Synth waveforms (`learn/synths/`). `sine`, `sawtooth`, `square`, `triangle`
  all confirmed, and `triangle` is confirmed as the default when `note` is set
  without a `sound`, exactly as the starting list said. Noise types `white`,
  `pink`, `brown`, `crackle` all confirmed, described in the docs as "written
  from hard to soft". FM parameters `fm`, `fmh`, `fmattack`/`fmatt`,
  `fmdecay`/`fmdec`, `fmsustain`/`fmsus`, `fmenv`/`fme` all confirmed (the
  short aliases are additional detail the starting list did not spell out).
  Additive `partials` and `phases` both confirmed. AKWF wavetables via a `wt_`
  prefix confirmed. ZZFX confirmed, but the starting list undersold it: ZZFX
  waveforms are their own `z_`-prefixed set (`z_sine`, `z_sawtooth`,
  `z_square`, `z_tan`, `z_noise`), not the plain waveform names, with their own
  parameter set (`zrand`, `curve`, `slide`, `deltaSlide`, `zmod`, `zcrush`,
  `zdelay`, `pitchJump`, `lfo`, `tremolo`). This module never emits a `z_`
  name — nothing in the pipeline's evidence points specifically at a chiptune
  register, so suggesting one would be exactly the kind of invented claim this
  module exists to avoid. Also found and not previously listed: `vib`/
  `vibrato`/`v` and `vibmod`/`vmod` for vibrato depth in semitones; not used
  here either, for the same reason.

  Notes and MIDI (`learn/notes/`). The docs state note names are "the note
  letter, followed by the octave number", flats as `b` and sharps as `#`, and
  that the numeric form uses "MIDI numbers, where adjacent whole numbers are
  one semitone apart" — but the page does not contain a bare "C4 = 60"
  sentence. It does give a worked equivalence: `note("a3 c#4 e4 a4")` is
  documented as identical to `note("57 61 64 69")`. Since a3 to c4 is 3
  semitones up (a-a#-b-c), that pins c4 = 57 + 3 = **60**, matching the
  project's own convention already encoded in `schemas.BassNote.midi_note`
  ("60 is c4") and the `69 + 12*log2(f0/440)` formula in `note_track.py`
  (A4 = 69). Confirmed from the docs, not assumed.
---------------------------------------------------------------------------

Mapping rules, from the plan and pinned here:

  - kick        -> `bd`                              match="exact"
  - snare       -> `sd`, `cp` noted as an alternative when there is no
                    tonal shell (low body-band energy)   match="exact"
  - hat         -> `hh` (closed) or `oh` (open), decided purely by decay
                    length — a real, cheap discrimination the pipeline
                    measures directly                     match="exact"
  - unclassified -> `match="none"`; toms/ride/crash/percussion named as
                    what it might be, Strudel's percussion set offered so
                    the user can pick by ear
  - sub bass    -> `sine`                             match="approximate"
  - harmonically rich bass -> `sawtooth`               match="approximate"
  - neither     -> `match="none"`, full waveform palette offered

The two bass verdicts each read three descriptors, and one of those three
changed in schema v5: they threshold `SpectralFeatures.centroid_energy_hz`,
never `centroid_mean`. `centroid_mean` is an unweighted mean over per-frame
centroids and a stem's silent frames drag it into the kilohertz, which is what
made the sub-bass branch unreachable in v4 (finding F4 in `V2-PLAN.md`). Both
values are reported in `evidence` so the contamination stays visible rather
than merely fixed.

Fixing the descriptor made the thresholds measurable for the first time, so
two of them moved: see the calibration block above `SUB_BASS_LOW_RATIO_MIN`
for the waveform sweep behind it, including the finding that **no absolute
centroid threshold separates a square from a sine across the bass register**
and that `brightness` has to carry that discrimination.

A note on `match="approximate"` for both bass verdicts, since it looks like an
undersell for a source that really is a sine: `"exact"` in this module means
"Strudel ships a sound under this name for this thing" — it is the verdict a
kick gets, because `bd` *is* the kick sample. A bass verdict is a reading of
spectral shape mapped onto the nearest of four waveforms, and the pipeline has
no way to know whether the source was a sine, a heavily filtered saw, or a
sampled 808. `"approximate"` is that gap, stated. The v5 fix makes the branch
reachable; it does not make the inference exact, and the two should not be
confused.

Two things this module deliberately refuses to claim, matching the plan:

  1. **`square` vs `sawtooth`.** Telling them apart needs an odd/even
     harmonic ratio this pipeline does not measure. A "harmonically rich"
     verdict always maps to `sawtooth` (never to `square`), always at
     `match="approximate"` rather than `"exact"`, and always carries
     `square` in `alternatives` so the honest uncertainty is visible rather
     than silently resolved one way.
  2. **`.bank()` drum-machine identification.** Inferring "TR-909" from a
     spectrum is exactly the kind of invented claim that made three rounds
     of heuristic labels unusable in Wave 1. `.bank()` is surfaced only as
     the static `DRUM_BANK_HINT` string below — no `bank` field on any
     model, no attempt to guess which machine.

`StrudelSoundSuggestion.match="none"` paired with `sound=None` is the
machine-readable "source it elsewhere" flag; the two always travel together
(`tests/test_strudel_vocab.py` pins the iff). `evidence` carries the raw
measurements that drove each verdict, floats only, mirroring
`HeuristicLabel.evidence`, so a reader can audit a suggestion without
re-running the analysis. `reason` is always populated prose.

On `alternatives`: WP0 added this field speculatively. It earns its place
here — it is where the honest "I don't know, but here is what Strudel offers"
lives for the `unclassified` drum bucket, the "neither" bass bucket, the
"no decay data" hat fallback, the snare's `cp` note, and the sawtooth-vs-square
disclosure. It stays empty only when the suggestion is genuinely unambiguous
(kick, and a hat or snare with nothing further to disclose).
"""

from __future__ import annotations

from statistics import fmean, median
from typing import Final

from .schemas import BassLine, DrumDecomposition, DrumHit, SpectralFeatures, StrudelSoundSuggestion

# ---------------------------------------------------------------------------
# Transcribed vocabulary tables. See the module docstring for provenance.
# ---------------------------------------------------------------------------

STRUDEL_DOCS_READ: Final[str] = "2026-08-04"

STRUDEL_DOCS_URLS: Final[tuple[str, ...]] = (
    "https://strudel.cc/learn/samples/",
    "https://strudel.cc/learn/synths/",
    "https://strudel.cc/learn/notes/",
)

#: Default drum/percussion sample abbreviations, from `learn/samples/`.
#: `misc` and `fx` are grab-bag categories in the docs, not specific sounds,
#: and are deliberately excluded — this module only ever names a sound it
#: could plausibly suggest or list as an alternative.
STRUDEL_DRUM_SAMPLES: Final[frozenset[str]] = frozenset(
    {
        "bd",  # bass/kick drum
        "sd",  # snare drum
        "rim",  # rimshot
        "cp",  # clap
        "hh",  # closed hi-hat
        "oh",  # open hi-hat
        "cr",  # crash
        "rd",  # ride
        "ht",  # high tom
        "mt",  # mid tom
        "lt",  # low tom
        "sh",  # shaker
        "cb",  # cowbell
        "tb",  # tambourine
        "perc",  # generic percussion
    }
)

#: Basic synth waveforms, from `learn/synths/`. `triangle` is the default
#: when `note` is set without a `sound`.
STRUDEL_WAVEFORMS: Final[frozenset[str]] = frozenset({"sine", "sawtooth", "square", "triangle"})

#: Noise flavours, "hard to soft" per the docs.
STRUDEL_NOISE_TYPES: Final[frozenset[str]] = frozenset({"white", "pink", "brown", "crackle"})

#: ZZFX has its own waveform names, distinct from the basic set above. Not
#: emitted by this module (see the module docstring) but transcribed for
#: completeness and for the meta-test that checks every emitted name against
#: a real table.
STRUDEL_ZZFX_WAVEFORMS: Final[frozenset[str]] = frozenset(
    {"z_sine", "z_sawtooth", "z_square", "z_tan", "z_noise"}
)

#: Every sound name this module might emit in `sound` or `alternatives`.
#: Deliberately a subset of the full transcribed vocabulary above — noise and
#: ZZFX names are real Strudel sounds but this module never has evidence that
#: points at them, so they are excluded here rather than left available for a
#: future accidental use.
ALL_STRUDEL_SOUND_NAMES: Final[frozenset[str]] = STRUDEL_DRUM_SAMPLES | STRUDEL_WAVEFORMS

#: What a "neither sub nor harmonically rich" bass verdict offers instead of a
#: single guess.
BASS_WAVEFORM_PALETTE: Final[tuple[str, ...]] = ("sine", "sawtooth", "square", "triangle")

#: What an `unclassified` drum hit might be, offered so the user can pick by
#: ear. Excludes `bd`/`sd`/`hh`/`oh`/`cp`/`rim` — those are exactly the sounds
#: the classifier already had a chance to name and did not.
UNCLASSIFIED_PERCUSSION_ALTERNATIVES: Final[tuple[str, ...]] = (
    "ht",
    "mt",
    "lt",
    "cr",
    "rd",
    "perc",
    "sh",
    "cb",
    "tb",
)

#: Static hint about `.bank()`, never used to infer a specific machine. Not
#: injected into every suggestion's `reason` (that would repeat on every hit
#: class); exported for `strudel_hints.build()` or the CLI to surface once,
#: e.g. in `StrudelHints.notes`.
DRUM_BANK_HINT: Final[str] = (
    "Strudel drum samples can be swapped to a specific drum machine with "
    '.bank(), e.g. s("bd sd").bank("RolandTR808") — this pipeline does not '
    "guess which machine the source used, only that the sample names above "
    "are available across many banks."
)

#: Closed vocabulary for `StrudelSoundSuggestion.role` as emitted by this
#: module. Not declared on the schema itself (`role` is a plain `str` there,
#: same reasoning as `DRUM_CLASSES`/`SOUND_MATCH_TERMS`), so it is pinned
#: here and enforced by `tests/test_strudel_vocab.py`.
ROLES: Final[frozenset[str]] = frozenset({"kick", "snare", "hat", "unclassified", "bass"})

# ---------------------------------------------------------------------------
# Thresholds. All `[guess]` — there is no calibration fixture for this WP
# (WP-A and WP-B, which produce the `DrumHit`/`BassLine` measurements this
# module reads, have not landed yet), so these are plausible starting points
# only, in the same spirit as `heuristics.THRESHOLDS`. WP-CAL should retune
# against real material; do not treat these as settled.
# ---------------------------------------------------------------------------

#: [guess] Mean `body_ratio` (150-500 Hz share) below this reads as "no
#: tonal shell", the condition under which `cp` is offered as a possibly
#: closer alternative to `sd`.
SNARE_SHELL_BODY_RATIO_MIN: Final[float] = 0.15

#: [guess] Median `decay_ratio` (attack RMS / tail RMS) across a source's hat
#: hits at or above this reads as closed (`hh`); below it reads as open
#: (`oh`). The fixture ground truth in `tests/conftest.py` gives closed and
#: open hats an 8x gap in time-to-decay-20dB (34 ms vs 277 ms), so the two
#: populations should sit well apart on any reasonable decay-ratio measure —
#: the exact number here is not load-bearing to get right on the first try,
#: but WP-CAL should confirm it against `drum_elements.py`'s actual output.
HAT_CLOSED_DECAY_RATIO_MIN: Final[float] = 4.0

# ---------------------------------------------------------------------------
# The sub-bass verdict, and the calibration behind it
# ---------------------------------------------------------------------------
#
# All three clauses must hold: `low` band-energy ratio at or above
# `SUB_BASS_LOW_RATIO_MIN`, `brightness` at or below
# `SUB_BASS_BRIGHTNESS_MAX`, and **`centroid_energy_hz`** at or below
# `SUB_BASS_CENTROID_HZ_MAX`. Two of the three were recalibrated in schema v5
# against a descriptor that, for the first time, means what they assumed.
#
# **The calibration sweep.** Band-limited additive synthesis (harmonics summed
# to Nyquist, so nothing aliases), fundamentals across the bass register at
# 35.0, 41.2, 49.0, 55.0, 65.4, 82.4, 98.0 and 110.0 Hz, each gated to 50%
# rests over a -82 dBFS noise floor and measured through the same
# 2048/512 Hann STFT the backends use. Measured `centroid_energy_hz`:
#
#     waveform    centroid Hz      brightness       low ratio
#     sine         35.3 - 109.8   0.0000 - 0.0001  0.999 - 1.000
#     triangle     36.6 - 113.9   0.0000 - 0.0001  0.985 - 1.000
#     square      108.3 - 288.4   0.0088 - 0.0266  0.813 - 0.944
#     sawtooth    147.4 - 385.6   0.0131 - 0.0415  0.764 - 0.915
#
# Plus a one-pole-filtered sawtooth, which is what a real synth sub usually
# is, at cutoffs from 4x to 32x the fundamental: centroid 46.3 - 246.6,
# brightness 0.0000 - 0.0157. And the one real sub-bass stem available, the
# v4 calibration bass: centroid **139.7 Hz**, brightness 0.0021, low 0.916.
#
# **Finding: no absolute centroid threshold separates these classes across the
# register.** Square starts at 108.3 Hz and sine reaches 109.8 Hz — they
# overlap, because a centroid in Hz is pitch-dependent and a square an octave
# down looks exactly like a sine an octave up. The overlap is not an artefact
# of the sweep's edges; it is intrinsic to comparing an absolute frequency
# against material whose fundamental varies by 3:1. Reported to the
# orchestrator as a real result rather than papered over with a boundary
# fitted to whichever stem was in hand.
#
# **What carries the discrimination instead: `brightness`.** It is
# pitch-independent by construction (a fixed 1500 Hz split), and on the same
# sweep it separates {sine, triangle, filtered saw, the real sub-bass stem}
# from {square, sawtooth} across the whole register with a wide margin —
# 0.0001 or less against 0.0088 or more, roughly 90x. So the two clauses have
# distinct jobs, and both are needed:
#
#   * `brightness` answers "does this have harmonic content" — the sine-vs-saw
#     question the verdict is actually about.
#   * `centroid_energy_hz` answers "is the energy in the sub register at all",
#     which brightness cannot see: a bass filtered to 16x its fundamental has
#     brightness ~0.005 whether the fundamental is 35 Hz or 110 Hz.
#
# **What this cannot do, stated plainly.** A heavily filtered sawtooth is
# indistinguishable from a sine on every descriptor here (cutoff 4x f0:
# brightness 0.0001, centroid 46-145 Hz), because at that point the harmonics
# really are gone and it *is* a sub. That is the correct answer, not a failure,
# and it is a large part of why both bass verdicts are `match="approximate"`
# rather than `"exact"`.

#: [guess] Below this share of energy under 250 Hz, a source is not a sub
#: whatever else it looks like. Unchanged in v5, and it is the clause that
#: keeps the `ancient-heavy-tech-donjon` bass (low ratio 0.628) out. Note from
#: the sweep above that it does **not** separate square or sawtooth from sine
#: (0.764 - 0.944 against 0.985 - 1.000, all above 0.75); tightening it to
#: ~0.96 would separate them synthetically but would reject the one real
#: sub-bass stem, at 0.916, because separated stems carry bleed that synthesis
#: does not. Left alone rather than fitted to n=1.
SUB_BASS_LOW_RATIO_MIN: Final[float] = 0.75

#: [measured, v5] **Lowered 10x, from 0.05.** This is the clause that actually
#: decides sine versus sawtooth; at 0.05 it decided nothing, admitting every
#: square and every sawtooth in the register (max 0.0415).
#:
#: From the sweep above: sine and triangle reach 0.0001, the real sub-bass stem
#: sits at 0.0021 (Demucs bleed puts it 20x above synthesis), the lowest
#: square is 0.0088 and the lowest sawtooth 0.0131. 0.005 is roughly the
#: geometric middle of the 0.0021-0.0088 gap: 2.4x above the one real sub-bass
#: reading, 1.8x below the brightest thing that must be rejected.
#:
#: Both margins are under 3x, which is thin, and the accept side rests on a
#: single real stem. If a real sub-bass with more bleed than that stem starts
#: reading `none`, this number is the first thing to check — but it should be
#: rechecked against more real material, not widened back toward a value the
#: sweep shows is inert.
SUB_BASS_BRIGHTNESS_MAX: Final[float] = 0.005

#: [measured, v5] **Raised from 120.0**, and the reason is not that a stem
#: failed to fit — it is that the descriptor underneath changed twice.
#:
#: The v4 history is finding **F4** in `V2-PLAN.md`. This clause read
#: `SpectralFeatures.centroid_mean`, an unweighted mean over per-frame
#: centroids. A resting frame in a real stem sits on a broadband noise floor,
#: so its own centroid lands in the kilohertz and the mean follows it: the v4
#: calibration bass stem — 0.916 low-band ratio, 0.0021 brightness, energy
#: 2e-06 above 6 kHz, audibly a pure sine sub — measured `centroid_mean`
#: 1010.7 Hz with `centroid_std` **1573.3 Hz, larger than the mean**, and
#: returned `match="none"`. No unweighted frame-mean centroid over real
#: material falls below 120 Hz, so the branch had most likely never fired on
#: any input in the tool's history. 120 Hz was never calibrated, because
#: nothing it was measured against was measurable.
#:
#: Against `centroid_energy_hz` the sweep above shows 120 Hz is **too
#: permissive, not too strict** — it admits a square down to about 41 Hz
#: (125.6 Hz) — while simultaneously rejecting the real sub-bass stem at
#: 139.7 Hz and a sine at the top of the register. It is wrong in both
#: directions at once, which is what an uncalibrated number looks like.
#:
#: 150.0 is set from the measurement: it clears the whole sine and triangle
#: range (max 113.9 Hz) with headroom for the bleed real separation adds, and
#: it clears the real sub-bass stem at 139.7 Hz. It sits just above the lowest
#: synthetic sawtooth (147.4 Hz at a 35 Hz fundamental), so it does **not**
#: reject that on its own — `SUB_BASS_BRIGHTNESS_MAX` does, and that division
#: of labour is the point. Read this clause as "is the energy in the sub
#: register", not as the harmonic-content test.
#:
#: **Pitch-dependence is a genuine weakness of any absolute-Hz threshold
#: here**, and it is worth stating even though the number is kept. The real
#: sub-bass stem plays both a1 (55 Hz) and a2 (110 Hz), so its 139.7 Hz is
#: pushed up by its own note range as much as by its waveform; the same patch
#: playing an octave lower would read near 70 Hz, and an octave higher would
#: fail. A ratio against a measured fundamental would be the right descriptor
#: and the pipeline does not produce one at this point in the analysis.
#:
#: Do not widen this to make a stubborn stem fit. Widening was the mistake F4
#: exists to prevent, and the two v5 changes here were made only because the
#: descriptor was fixed first and then re-measured against known signals.
SUB_BASS_CENTROID_HZ_MAX: Final[float] = 150.0

#: [guess] Harmonically-rich verdict requires ALL three, in the opposite
#: direction: energy spread beyond the fundamental, some brightness, and a
#: centroid pulled up above the fundamental by the harmonics. Also now read
#: against `centroid_energy_hz` rather than `centroid_mean`.
#:
#: **These numbers have not been recalibrated and W8B must.** They were chosen
#: against a contaminated descriptor, and only one stem in the calibration set
#: reaches this branch (`showers-of-gold` bass: centroid 2676.5 Hz, brightness
#: 0.482, low 0.223 — it passes all three comfortably either way). There is
#: therefore no evidence here for moving them and none for keeping them.
#:
#: One thing the sub-bass sweep above does show: `HARMONIC_BASS_LOW_RATIO_MAX`
#: at 0.55 means a genuinely harmonic bass low in the register cannot reach
#: this branch at all — a synthetic 55 Hz sawtooth has a low ratio of 0.868,
#: because at that fundamental even the harmonics are mostly under 250 Hz. It
#: falls through to `match="none"` with the full palette, which is honest but
#: is not the `sawtooth` a listener would pick. Flagged for W8B rather than
#: changed here on synthetic evidence alone.
HARMONIC_BASS_LOW_RATIO_MAX: Final[float] = 0.55
HARMONIC_BASS_BRIGHTNESS_MIN: Final[float] = 0.12
HARMONIC_BASS_CENTROID_HZ_MIN: Final[float] = 180.0


# --- small helpers ----------------------------------------------------------


def _mean(values: list[float | None]) -> float | None:
    """Mean of the non-`None` values, or `None` if there are none."""
    finite = [v for v in values if v is not None]
    if not finite:
        return None
    return fmean(finite)


def _evidence(pairs: list[tuple[str, float | None]]) -> dict[str, float]:
    """Build an evidence dict from `(name, value)` pairs, dropping `None`s.

    `StrudelSoundSuggestion.evidence` is `dict[str, float]` — no `None`s
    allowed — so a missing measurement is simply absent from the dict rather
    than crashing or being coerced to a fake `0.0`.
    """
    return {name: value for name, value in pairs if value is not None}


# --- drum mapping -------------------------------------------------------


def _suggest_kick(hits: list[DrumHit]) -> StrudelSoundSuggestion:
    kick_ratio = _mean([h.kick_ratio for h in hits])
    confidence = _mean([h.confidence for h in hits])
    evidence = _evidence(
        [
            ("hit_count", float(len(hits))),
            ("mean_kick_ratio", kick_ratio),
            ("mean_confidence", confidence),
        ]
    )
    return StrudelSoundSuggestion(
        role="kick",
        match="exact",
        sound="bd",
        reason=(
            f"{len(hits)} hit(s) classified as kick; 'bd' is Strudel's default "
            "kick/bass-drum sample."
        ),
        alternatives=[],
        evidence=evidence,
    )


def _suggest_snare(hits: list[DrumHit]) -> StrudelSoundSuggestion:
    body_ratio = _mean([h.body_ratio for h in hits])
    confidence = _mean([h.confidence for h in hits])
    evidence = _evidence(
        [
            ("hit_count", float(len(hits))),
            ("mean_body_ratio", body_ratio),
            ("mean_confidence", confidence),
        ]
    )

    alternatives: list[str] = []
    shell_note = ""
    if body_ratio is not None and body_ratio < SNARE_SHELL_BODY_RATIO_MIN:
        alternatives.append("cp")
        shell_note = (
            f" Body-band energy is low (mean body ratio {body_ratio:.2f}), so "
            "there is little tonal shell resonance here — 'cp' (clap) may read "
            "closer by ear than 'sd'."
        )

    return StrudelSoundSuggestion(
        role="snare",
        match="exact",
        sound="sd",
        reason=(
            f"{len(hits)} hit(s) classified as snare; 'sd' is Strudel's default "
            "snare sample." + shell_note
        ),
        alternatives=alternatives,
        evidence=evidence,
    )


def _suggest_hat(hits: list[DrumHit]) -> StrudelSoundSuggestion:
    decay_ratios = [h.decay_ratio for h in hits if h.decay_ratio is not None]
    confidence = _mean([h.confidence for h in hits])
    base_evidence = [
        ("hit_count", float(len(hits))),
        ("mean_confidence", confidence),
    ]

    if not decay_ratios:
        return StrudelSoundSuggestion(
            role="hat",
            match="none",
            sound=None,
            reason=(
                f"{len(hits)} hit(s) classified as hat, but none carry a decay "
                "measurement, so open versus closed cannot be told apart."
            ),
            alternatives=["hh", "oh"],
            evidence=_evidence(base_evidence),
        )

    decay_median = median(decay_ratios)
    evidence = _evidence([*base_evidence, ("median_decay_ratio", decay_median)])

    if decay_median >= HAT_CLOSED_DECAY_RATIO_MIN:
        sound, label = "hh", "closed"
    else:
        sound, label = "oh", "open"

    return StrudelSoundSuggestion(
        role="hat",
        match="exact",
        sound=sound,
        reason=(
            f"{len(hits)} hit(s) classified as hat; median decay ratio "
            f"{decay_median:.2f} reads as {label}, so '{sound}' is the closer "
            "default sample. Open versus closed is decided by decay length "
            "alone, which this pipeline measures directly."
        ),
        alternatives=[],
        evidence=evidence,
    )


def _suggest_unclassified(hits: list[DrumHit], unclassified_count: int) -> StrudelSoundSuggestion:
    # `hits` may be empty even when hits happened: `track_summary.json`
    # strips `DrumDecomposition.hits` down to `total_hit_count`, and this
    # function is written to degrade gracefully if it is ever handed that
    # stripped shape rather than the full `analysis/drums.json` one.
    count = len(hits) if hits else unclassified_count
    confidence = _mean([h.confidence for h in hits]) if hits else None
    evidence = _evidence(
        [
            ("hit_count", float(count)),
            ("mean_confidence", confidence),
        ]
    )

    return StrudelSoundSuggestion(
        role="unclassified",
        match="none",
        sound=None,
        reason=(
            f"{count} hit(s) did not clear the kick/snare/hat decision margin. "
            "This pipeline recognises only three drum classes; toms, rides, "
            "crashes, claps and shakers all land here on percussive material, "
            "correctly, not as a failure. Source by ear from Strudel's "
            "percussion set."
        ),
        alternatives=list(UNCLASSIFIED_PERCUSSION_ALTERNATIVES),
        evidence=evidence,
    )


def suggest_drum_sounds(decomposition: DrumDecomposition) -> list[StrudelSoundSuggestion]:
    """Map a `DrumDecomposition` onto Strudel sound suggestions.

    One suggestion per class actually present — `kick`, `snare`, `hat`, and
    `unclassified` — never a fixed four; a track with no snare hits gets no
    snare suggestion. Works from `decomposition.hits`, so call this on the
    full decomposition from `analysis/drums.json`, not the stripped shape in
    `track_summary.json` (`decomposition.hits` there would already be gone —
    see `_suggest_unclassified` for the one case handled anyway via the
    surviving count).

    Never crashes: an all-default `DrumDecomposition()` (status
    "not_attempted", no hits) returns `[]`. Any hit whose `drum` field is
    outside `DRUM_CLASSES` is silently ignored rather than raising — the
    schema does not validate that field's contents, and a classifier bug
    should not take down sound-suggestion generation too.
    """
    kicks: list[DrumHit] = []
    snares: list[DrumHit] = []
    hats: list[DrumHit] = []
    unclassified: list[DrumHit] = []
    for hit in decomposition.hits:
        if hit.drum == "kick":
            kicks.append(hit)
        elif hit.drum == "snare":
            snares.append(hit)
        elif hit.drum == "hat":
            hats.append(hit)
        elif hit.drum == "unclassified":
            unclassified.append(hit)
        # else: unrecognised class string — ignored, not raised.

    suggestions: list[StrudelSoundSuggestion] = []
    if kicks:
        suggestions.append(_suggest_kick(kicks))
    if snares:
        suggestions.append(_suggest_snare(snares))
    if hats:
        suggestions.append(_suggest_hat(hats))
    if unclassified or decomposition.unclassified_count > 0:
        suggestions.append(_suggest_unclassified(unclassified, decomposition.unclassified_count))
    return suggestions


# --- bass mapping ------------------------------------------------------


def _sub_bass_match(
    low_ratio: float | None, brightness: float | None, centroid_energy: float | None
) -> bool:
    """All three clauses, or no verdict. `centroid_energy_hz`, never `centroid_mean`."""
    if low_ratio is None or brightness is None or centroid_energy is None:
        return False
    return (
        low_ratio >= SUB_BASS_LOW_RATIO_MIN
        and brightness <= SUB_BASS_BRIGHTNESS_MAX
        and centroid_energy <= SUB_BASS_CENTROID_HZ_MAX
    )


def _harmonic_bass_match(
    low_ratio: float | None, brightness: float | None, centroid_energy: float | None
) -> bool:
    """All three clauses, or no verdict. `centroid_energy_hz`, never `centroid_mean`."""
    if low_ratio is None or brightness is None or centroid_energy is None:
        return False
    return (
        low_ratio <= HARMONIC_BASS_LOW_RATIO_MAX
        and brightness >= HARMONIC_BASS_BRIGHTNESS_MIN
        and centroid_energy >= HARMONIC_BASS_CENTROID_HZ_MIN
    )


def suggest_bass_sound(
    bass_line: BassLine, spectral: SpectralFeatures
) -> list[StrudelSoundSuggestion]:
    """Map a bass stem's spectral shape onto a Strudel synth waveform.

    Always returns exactly one suggestion (role `"bass"`), never `[]` — unlike
    drums, "was there a bass stem at all" is the caller's question to answer
    before calling this, not this function's.

    Deliberately reads `spectral`, not `bass_line.notes`: sub-vs-harmonic is a
    question about the stem's overall spectral shape, which is measured
    whether or not note segmentation succeeded. So this still produces a
    (caveated) verdict when `bass_line.status` is e.g. `"unvoiced"` or
    `"too_few_hits"` — the caller should pass the bass stem's own
    `SpectralFeatures` regardless of pitch-tracking outcome. `bass_line.status`
    itself is not a float and cannot go in `evidence`; when it is not `"ok"`
    or `"not_attempted"` it is folded into `reason` as a caveat instead.

    Reads `spectral.centroid_energy_hz`, **not** `spectral.centroid_mean` —
    see `SUB_BASS_CENTROID_HZ_MAX` for finding F4 and why. `centroid_mean` is
    still carried in `evidence` alongside it, so a reader can see both the
    corrected number the verdict was made on and the contaminated one a v4
    output would have shown; it is never thresholded on. An analysis produced
    before schema v5 has no `centroid_energy_hz` at all, in which case both
    branches decline and the verdict is an honest `match="none"` with a caveat
    naming the missing field, rather than a guess made from the field known to
    be wrong.

    Never crashes: a `SpectralFeatures()` with every field `None` produces a
    `match="none"` suggestion with empty `evidence` rather than raising.
    """
    low_ratio = spectral.band_energy_ratios.low
    brightness = spectral.brightness
    centroid = spectral.centroid_energy_hz

    evidence = _evidence(
        [
            ("low_band_ratio", low_ratio),
            ("brightness", brightness),
            ("centroid_energy_hz", centroid),
            ("centroid_mean_hz", spectral.centroid_mean),
        ]
    )

    status_note = ""
    if centroid is None and spectral.centroid_mean is not None:
        status_note += (
            " (no centroid_energy_hz on this analysis — it arrived in schema "
            "v5, and centroid_mean is contaminated by silent frames, so it "
            "is reported as evidence but deliberately not used to decide)"
        )
    if bass_line.status not in ("ok", "not_attempted"):
        status_note += (
            f" (bass line status is '{bass_line.status}'; this verdict is based "
            "on the stem's overall spectral shape only, independent of note "
            "segmentation)"
        )

    if _sub_bass_match(low_ratio, brightness, centroid):
        return [
            StrudelSoundSuggestion(
                role="bass",
                match="approximate",
                sound="sine",
                reason=(
                    "Energy is concentrated at the fundamental (low-band ratio "
                    f"{low_ratio:.2f}, brightness {brightness:.2f}, median "
                    f"centroid {centroid:.0f} Hz) with little harmonic content: "
                    "reads as sub bass. 'sine' is the closest default waveform."
                    + status_note
                ),
                alternatives=[],
                evidence=evidence,
            )
        ]

    if _harmonic_bass_match(low_ratio, brightness, centroid):
        return [
            StrudelSoundSuggestion(
                role="bass",
                match="approximate",
                sound="sawtooth",
                reason=(
                    "Energy reaches beyond the fundamental (low-band ratio "
                    f"{low_ratio:.2f}, brightness {brightness:.2f}, median "
                    f"centroid {centroid:.0f} Hz): reads as a harmonically rich "
                    "bass. "
                    "'sawtooth' is the closest default waveform. Whether the "
                    "source is odd-harmonic (closer to 'square') or full-"
                    "harmonic cannot be told apart without an odd/even "
                    "harmonic ratio, which this pipeline does not measure."
                    + status_note
                ),
                alternatives=["square"],
                evidence=evidence,
            )
        ]

    return [
        StrudelSoundSuggestion(
            role="bass",
            match="none",
            sound=None,
            reason=(
                "Spectral shape does not clearly read as sub bass or as "
                "harmonically rich; picking a synth waveform from this "
                "evidence alone would be a guess, not a measurement."
                + status_note
            ),
            alternatives=list(BASS_WAVEFORM_PALETTE),
            evidence=evidence,
        )
    ]


def suggest_sounds(
    drums: DrumDecomposition, bass: BassLine, bass_spectral: SpectralFeatures
) -> list[StrudelSoundSuggestion]:
    """Convenience combinator: `suggest_drum_sounds` + `suggest_bass_sound`.

    Exists only so a caller with all three inputs in hand does not have to
    remember to call and concatenate both; identical to doing so by hand.
    """
    return [*suggest_drum_sounds(drums), *suggest_bass_sound(bass, bass_spectral)]
