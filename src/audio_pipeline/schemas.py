"""Pydantic models defining every JSON artefact this tool writes.

Single source of truth for output shape. Bump `SCHEMA_VERSION` in `__init__.py`
whenever a field changes meaning.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import ANALYSIS_SAMPLE_RATE, SCHEMA_VERSION


class RhythmFeatures(BaseModel):
    """Tempo and event-timing descriptors."""

    bpm: float | None = None
    bpm_confidence: float | None = None
    beat_times: list[float] = Field(
        default_factory=list,
        description=(
            "The pulse you would tap along to, in seconds. Inferred and evenly "
            "spaced by construction, so it says nothing about what was actually "
            "played between beats. Not the same as onset_times."
        ),
    )
    onset_times: list[float] = Field(
        default_factory=list,
        description=(
            "When notes and hits actually start, in seconds. Observed, not "
            "inferred, and unevenly spaced: this is where swing lives, so it is "
            "what subdivision-feel detection reads. Not the same as beat_times."
        ),
    )
    onset_density: float | None = Field(default=None, description="Onsets per second")
    transient_sharpness: float | None = None


class TonalFeatures(BaseModel):
    """Key, scale, and chroma descriptors."""

    key: str | None = None
    scale: str | None = None
    key_confidence: float | None = None
    hpcp_mean: list[float] = Field(default_factory=list, description="12-bin chroma/HPCP means")
    tonal_stability: float | None = None


class BandEnergyRatios(BaseModel):
    """Share of total spectral energy per band. Present fields sum to ~1.0.

    Band edges are fixed in `BAND_EDGES_HZ` and shared by both backends so the
    heuristic thresholds tuned on one backend remain meaningful on the other.
    """

    low: float | None = Field(default=None, description="20-250 Hz")
    low_mid: float | None = Field(default=None, description="250-2000 Hz")
    high_mid: float | None = Field(default=None, description="2000-6000 Hz")
    high: float | None = Field(default=None, description="6000-20000 Hz")


class SpectralFeatures(BaseModel):
    """Timbre-related descriptors."""

    centroid_mean: float | None = Field(
        default=None,
        description=(
            "Unweighted mean of the per-frame spectral centroid. **Contaminated "
            "by silent frames** — a frame at the noise floor has a flat spectrum "
            "and therefore a centroid up in the kilohertz, so a stem that is half "
            "silence reads far brighter than it sounds. Measured on the v4 "
            "calibration bass stem: mean 1010.7 Hz with std 1573.3 Hz, on a "
            "sub-bass with no energy at all above 4 kHz. A standard deviation "
            "larger than the mean is the signature. Prefer `centroid_energy_hz`; "
            "this field is kept so v4 outputs stay comparable."
        ),
    )
    centroid_std: float | None = None
    centroid_energy_hz: float | None = Field(
        default=None,
        description=(
            "Energy-weighted spectral centroid: `sum(f * power) / sum(power)` over "
            "every bin of every frame, in Hz. A first moment, not a median — the "
            "distinction is load-bearing. Silent frames carry almost no energy and "
            "so almost no weight, which is what makes this immune to the "
            "contamination `centroid_mean` suffers. Both backends compute it from "
            "the shared magnitude spectrogram and must agree exactly.\n\n"
            "Measured against known signals, all gated 50% over a −82 dBFS floor: "
            "a 55 Hz sine reads 55.0, a 55 Hz square 86.6, a 55 Hz sawtooth 109.0. "
            "An energy *median* reads 64.6 for all three and cannot tell them "
            "apart, which is why this is a centroid. This is the field thresholds "
            "should read; `centroid_mean` is retained only for continuity with v4."
        ),
    )
    rolloff_mean: float | None = Field(
        default=None,
        description=(
            "Unweighted mean of the per-frame 85% rolloff. **Contaminated by "
            "silent frames** for the same reason as `centroid_mean`. Measured on "
            "the v4 calibration bass stem: 4333 Hz (librosa) and 1097 Hz "
            "(essentia), on a stem with 2e-06 of its energy above 6 kHz, against "
            "an energy-weighted 215 Hz. Prefer `rolloff_energy_hz`."
        ),
    )
    rolloff_energy_hz: float | None = Field(
        default=None,
        description=(
            "Energy-weighted 85% rolloff: the frequency below which 85% of the "
            "source's total spectral energy sits, aggregated over all frames so "
            "silent frames carry no weight. Both backends identical. Unlike a "
            "centroid this one *is* a percentile, because a rolloff is defined as "
            "one. Resolution is a single FFT bin, ~21.5 Hz."
        ),
    )
    brightness: float | None = None
    band_energy_ratios: BandEnergyRatios = Field(default_factory=BandEnergyRatios)


class DynamicsFeatures(BaseModel):
    """Level and dynamic-range descriptors."""

    loudness_lufs: float | None = None
    rms_mean: float | None = None
    rms_std: float | None = None
    crest_factor: float | None = None


class HeuristicLabel(BaseModel):
    """A human-readable production clue plus the evidence behind it."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema v4: drum decomposition and bass notes
# ---------------------------------------------------------------------------
# Closed vocabularies, as module-level frozensets rather than `Literal`
# annotations. Two reasons, both load-bearing:
#
# 1. `tests/test_analyze.py::_fill_value` synthesises a value per field type to
#    keep the fake backends honest as the schema grows, and it raises on a
#    `Literal`. A plain `str` plus a frozenset keeps that machinery working.
# 2. A validator that rejects an unknown string turns a classifier bug into a
#    crash mid-analysis. These sets are the contract; the tests enforce it.
#
# `strudel_hints.DENSITY_TERMS` and `SUBDIVISION_TERMS` already work this way.

#: The three classes Wave 4 claims to recognise, plus the honest fourth.
#: `unclassified` is a real answer, not a failure: toms, rides, crashes, claps
#: and shakers all land there, and on percussive material that bucket is large.
DRUM_CLASSES: frozenset[str] = frozenset({"kick", "snare", "hat", "unclassified"})

#: Status of a `DrumDecomposition` or `BassLine` block. Carried explicitly
#: rather than signalled by an empty list, because "not attempted" (the source
#: is not drums), "attempted and found nothing" and "attempted and failed" are
#: three different facts and only the first is uninteresting.
#:
#:   not_attempted  this source is not eligible (only `drums` gets a
#:                  decomposition, only `bass` gets a note track)
#:   ok             ran and produced a result
#:   no_grid        hits found, but no usable cycle grid to fold them onto
#:                  (drums only) — hits are still reported
#:   too_few_hits   ran, but there was not enough material to describe
#:   unvoiced       ran, and the source has no pitch to track (bass only)
#:   silent         the source is below `SILENCE_RMS_FLOOR`, so nothing was
#:                  attempted on it at all (schema v5) — see that constant
#:   failed         raised, and the exception was swallowed to keep the
#:                  analysis going
BLOCK_STATUSES: frozenset[str] = frozenset(
    {"not_attempted", "ok", "no_grid", "too_few_hits", "unvoiced", "silent", "failed"}
)

#: RMS below which a source is treated as separation residue rather than as a
#: stem, in linear amplitude. Everything derived from it (`drum_decomposition`,
#: `bass_line`, the `strudel_hints` tonal-centre fallback) is skipped and
#: reported as `status="silent"` instead of being computed from a noise floor.
#:
#: **[grounded, measured]** across 35 stems of the seven tracks under
#: `calibration/`, which is the whole corpus plus the two example tracks:
#:
#:   six residue stems, -65.9 to -70.0 LUFS   rms 8e-06 .. 1.38e-04
#:   quietest genuinely-present stem          rms 3.82e-03 (-37.4 LUFS)
#:
#: A 27.7x gap with nothing in it. 1e-03 is -60 dBFS, which sits 7.2x above the
#: loudest residue stem and 3.8x below the quietest real one, and is inaudible
#: in any mix — a physical bound, not the midpoint of the gap.
#:
#: The failure this exists to stop is measured, not theoretical: a bass stem at
#: -70 LUFS produced a two-note bass line and an `e4` median, and that "E minor"
#: then beat the mix's own F major into `strudel_hints.json` because the mix's
#: key confidence was 0.445. An `arrangement.STEM_ACTIVITY_FLOOR` of 0.02 already
#: catches the same stems, but it is a *within-record ratio* answering a
#: different question and it is not consulted before pitch-tracking a stem.
SILENCE_RMS_FLOOR: float = 1e-3

#: What the drum grid was anchored on: a supplied downbeat, `rhythm.beat_times[0]`,
#: or failing both the first detected hit. Recorded either way, because a grid
#: anchored on a measured downbeat, one anchored on a tempo estimate and one
#: anchored on whatever happened to be loudest first deserve different amounts
#: of trust.
#:
#: `supplied` is what `analyze.py` produces in v5 and the only one that should
#: appear on a full pipeline run: `beat_times[0]` is *not* a downbeat (on the
#: calibration track it lands the kick on steps 3/7/11/15), so the grid is
#: anchored on `DownbeatFit.offset_seconds` instead. The other two survive for
#: `drum_elements.decompose` called on its own.
#:
#: Spelt `beats` rather than `beat_times` deliberately — a *value* that reads
#: like a stripped *field name* makes `track_summary.json` ambiguous to grep and
#: breaks the guard that no event list survives into it.
GRID_ANCHOR_SOURCES: frozenset[str] = frozenset({"supplied", "beats", "first_hit"})

#: How well a Strudel sound matches what was measured. `none` paired with
#: `sound=None` is the machine-readable "source this sample elsewhere" flag.
SOUND_MATCH_TERMS: frozenset[str] = frozenset({"exact", "approximate", "none"})


class DrumHit(BaseModel):
    """One detected drum hit, with the band measurements that classified it.

    Every ratio is a share of the hit's own in-band energy over the drum
    detection bands, which are deliberately **not** `BAND_EDGES_HZ` — a snare's
    body sits inside that scheme's `low` band, so it cannot separate kick from
    snare. `drum_elements.py` owns the exact bounds.

    A hit whose winning class did not clear the decision margin is reported as
    `unclassified` with its honest winner confidence and its timing intact.
    Dropping it would silently delete part of the pattern.
    """

    time_seconds: float
    drum: str = Field(description=f"One of: {', '.join(sorted(DRUM_CLASSES))}")
    confidence: float = Field(description="Winning class score, 0.0-1.0")
    step: int | None = Field(default=None, description="Grid step, or None with no grid")
    kick_ratio: float | None = None
    body_ratio: float | None = None
    noise_ratio: float | None = None
    air_ratio: float | None = None
    decay_ratio: float | None = Field(
        default=None, description="Attack RMS over tail RMS; higher means shorter"
    )
    flatness: float | None = Field(
        default=None, description="Spectral flatness: low is tonal, high is noisy"
    )


class DrumPattern(BaseModel):
    """One drum class folded onto a single cycle: the bar you would type out.

    Small on purpose — at most `steps_per_cycle` integers per class — and
    deliberately *not* stripped from `track_summary.json`, unlike the hit list.
    This compact one-bar view is the whole point of the feature.
    """

    drum: str = Field(description=f"One of: {', '.join(sorted(DRUM_CLASSES))}")
    steps: list[int] = Field(
        default_factory=list, description="Occupied steps within one cycle, ascending"
    )
    step_occupancy: list[float] = Field(
        default_factory=list,
        description=(
            "Parallel to `steps`: the share of cycles in which that step was "
            "actually hit. An occasional ghost kick reads 0.25, not full "
            "membership, so a reader can tell the backbone from the decoration."
        ),
    )
    hit_count: int = Field(
        default=0, description="Hits of THIS class across the whole source, not per cycle"
    )


class DrumDecomposition(BaseModel):
    """Which drum plays where, per band rather than per onset.

    Built from per-band spectral flux with each band peak-picked independently,
    so a coincident kick and hat are two hits because two detectors found them.
    Classifying the single global `rhythm.onset_times` list instead would assign
    one class per instant and delete the hat on every downbeat.

    `rhythm.onset_times` keeps its own meaning (density, subdivision feel) and
    is untouched by any of this.
    """

    status: str = Field(
        default="not_attempted", description=f"One of: {', '.join(sorted(BLOCK_STATUSES))}"
    )
    steps_per_cycle: int | None = None
    cycle_seconds: float | None = None
    grid_anchor_seconds: float | None = None
    grid_anchor_source: str | None = Field(
        default=None, description=f"One of: {', '.join(sorted(GRID_ANCHOR_SOURCES))}"
    )
    quantisation_error_steps: float | None = Field(
        default=None, description="Mean absolute distance from a hit to its step, in steps"
    )
    patterns: list[DrumPattern] = Field(default_factory=list)
    hits: list[DrumHit] = Field(
        default_factory=list,
        description=(
            "Every hit, in time order. Replaced by `total_hit_count` in "
            "`track_summary.json`; complete in `analysis/<source>.json`."
        ),
    )
    unclassified_count: int = 0
    # `caveats`, not `notes`: in a `SourceAnalysis`, `notes` already means bass
    # notes one block down. Two fields named `notes` meaning different things in
    # the same file is how a reader — or a strip table — gets it wrong.
    caveats: list[str] = Field(
        default_factory=list, description="Plain-English caveats about this decomposition"
    )


class BassNote(BaseModel):
    """One note of the bass line: what to type, and how sure the tracker was."""

    start_seconds: float
    duration_seconds: float
    midi_note: int = Field(description="MIDI number; 60 is c4, matching Strudel")
    note_name: str = Field(description="Strudel spelling, e.g. 'a1'")
    median_f0_hz: float | None = None
    cents_offset: float | None = Field(
        default=None, description="Median distance from equal temperament, in cents"
    )
    confidence: float | None = Field(default=None, description="Mean voicing confidence, 0.0-1.0")
    step: int | None = Field(default=None, description="Drum-grid step, or None with no grid")


class BassLine(BaseModel):
    """The bass stem as a note sequence, plus what could go wrong with it."""

    status: str = Field(
        default="not_attempted", description=f"One of: {', '.join(sorted(BLOCK_STATUSES))}"
    )
    notes: list[BassNote] = Field(
        default_factory=list,
        description=(
            "Every note, in time order. Replaced by `note_count` in "
            "`track_summary.json`; complete in `analysis/<source>.json`."
        ),
    )
    median_midi_note: int | None = None
    median_cents_offset: float | None = Field(
        default=None,
        description=(
            "Median cents offset across the line. A consistent non-zero value is "
            "a 432 Hz master or a pitched remix, not a line of wrong notes — and "
            "it cannot catch an octave error, which is 0 cents."
        ),
    )
    voiced_fraction: float | None = Field(
        default=None, description="Share of frames the tracker called voiced, 0.0-1.0"
    )
    octave_corrections: int = Field(
        default=0, description="Frames the octave guard moved; a high rate is a caveat"
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Plain-English warnings: octave risk, low voicing, bleed",
    )


class PitchTrack(BaseModel):
    """Raw per-frame F0 from a backend. **Transport only — never written to JSON.**

    A five-minute track at hop 512 is ~25,800 F0 floats, which belongs in no
    output file. `note_track.segment_notes()` turns this into a `BassLine`, and
    only the `BassLine` is written. No field of `SourceAnalysis` may reference
    this model; `tests/test_schemas_summary.py` walks the model graph and fails
    if one ever does.

    Floats here are deliberately **unbounded** even where the range is 0.0-1.0.
    This is a backend return type, and the fake-backend filler in
    `tests/test_analyze.py` synthesises values that can exceed 1.0; a
    `Field(le=1.0)` would turn that into a validation error far from its cause.
    """

    f0_hz: list[float] = Field(
        default_factory=list, description="Per-frame fundamental estimate; 0.0 where unvoiced"
    )
    voiced: list[bool] = Field(default_factory=list, description="Per-frame voicing decision")
    voiced_probability: list[float] = Field(
        default_factory=list, description="Per-frame voicing confidence, nominally 0.0-1.0"
    )
    frame_hop_seconds: float | None = Field(
        default=None, description="Seconds between consecutive frames"
    )
    method: str | None = Field(default=None, description="Algorithm used, e.g. 'pyin'")


class StrudelSoundSuggestion(BaseModel):
    """A Strudel sound to reach for, or an explicit admission there isn't one.

    `match="none"` with `sound=None` is the machine-readable "source it
    elsewhere" flag, and the two always travel together. Inventing a plausible
    sound name is worse than saying nothing: it reads as a measurement.
    """

    role: str = Field(description="What this suggestion is for, e.g. kick, snare, hat, bass")
    match: str = Field(description=f"One of: {', '.join(sorted(SOUND_MATCH_TERMS))}")
    sound: str | None = Field(default=None, description="Strudel sound name, or None")
    reason: str = ""
    alternatives: list[str] = Field(
        default_factory=list,
        description="What Strudel does offer for this role when nothing matched",
    )
    evidence: dict[str, float] = Field(
        default_factory=dict, description="Descriptor values behind the suggestion"
    )


class DrumGridHint(BaseModel):
    """One cycle of drums, condensed for someone typing a pattern by hand."""

    status: str = Field(
        default="not_attempted", description=f"One of: {', '.join(sorted(BLOCK_STATUSES))}"
    )
    steps_per_cycle: int | None = None
    kick_steps: list[int] = Field(default_factory=list)
    snare_steps: list[int] = Field(default_factory=list)
    hat_steps: list[int] = Field(default_factory=list)
    unclassified_count: int = Field(
        default=0,
        description=(
            "Hits that are none of the three. Large on percussive material and "
            "correct there — three classes cannot describe a real kit."
        ),
    )
    caveats: list[str] = Field(default_factory=list)


class BassLineHint(BaseModel):
    """The bass line as a short list of note names you can type straight in."""

    status: str = Field(
        default="not_attempted", description=f"One of: {', '.join(sorted(BLOCK_STATUSES))}"
    )
    note_sequence: list[str] = Field(
        default_factory=list, description="Strudel note names in time order, capped"
    )
    truncated_from: int | None = Field(
        default=None,
        description=(
            "Real note count when `note_sequence` was capped, else None. This "
            "file is documented as hand-readable, so the full line lives in "
            "`analysis/bass.json`."
        ),
    )
    median_midi_note: int | None = None
    steps: list[int] = Field(
        default_factory=list,
        description=(
            "Grid steps the bass lands on within one cycle, ascending and "
            "de-duplicated. Empty when there is no grid. **Read this with "
            "`step_share`, not on its own** — a busy line touches all sixteen "
            "steps at some point, and on three of the five corpus tracks this "
            "list is exactly `[0..15]`, which says nothing by itself."
        ),
    )
    step_share: list[float] = Field(
        default_factory=list,
        description=(
            "Parallel to `steps`: the share of all notes starting on that step. "
            "The same shape as `DrumPattern.step_occupancy`, and here for the "
            "same reason — it is what separates the backbone from the "
            "decoration. On the calibration track steps 2, 6, 10 and 14 read "
            "0.145 each against 0.07-0.10 elsewhere: the offbeat eighths, the "
            "same four steps the hats land on, found by an independent code "
            "path. Deliberately a measured share rather than a thresholded "
            "list — no threshold here has more than one record behind it."
        ),
    )
    caveats: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Schema v5: one tempo and one structure, at track level
# ---------------------------------------------------------------------------
# These mirror the frozen dataclasses `tempo.py` and `arrangement.py` return,
# field for field and name for name, so `analyze.py` converts with a plain
# `dataclasses.asdict()` into `model_validate()` and there is no hand-written
# mapping to drift. `tests/test_schemas_summary.py` pins that correspondence.
#
# They are *not* defined in those modules because both are pure-numpy and
# deliberately importable without pydantic, and `tempo.py` already imports from
# `drum_elements.py`, which imports from here — defining them there would close
# an import cycle.
#
# The closed vocabularies below are frozensets rather than `Literal`
# annotations, for the two reasons `DRUM_CLASSES` gives above. The source
# modules do use `Literal`; the values are identical and a test asserts it.

#: Confidence bands shared by `TempoFit` and `DownbeatFit`.
CONFIDENCE_LABELS: frozenset[str] = frozenset({"high", "medium", "low"})

#: `TempoStability.label`. `unknown` means a half could not be fitted at all,
#: which is a different fact from "the tempo moved".
STABILITY_LABELS: frozenset[str] = CONFIDENCE_LABELS | {"unknown"}

#: `TempoFit.status`. `coarse` means refinement was refused and `bpm` is the
#: backend's own estimate passed straight through — the single most important
#: distinction in the block, and the one v4's bare `bpm` could not make.
TEMPO_STATUSES: frozenset[str] = frozenset({"refined", "coarse", "unavailable"})

#: `OctaveCandidate.status`. `ruled_out` is a positive finding, not a gap.
OCTAVE_CANDIDATE_STATUSES: frozenset[str] = frozenset({"live", "ruled_out", "unmeasurable"})

#: `TempoFit.octave_status`.
OCTAVE_STATUSES: frozenset[str] = frozenset({"single", "ambiguous", "unavailable"})

#: `DownbeatFit.status`.
DOWNBEAT_STATUSES: frozenset[str] = frozenset({"ok", "ambiguous", "unavailable"})

#: `DownbeatFit.resolved_by`.
DOWNBEAT_RESOLVED_BY: frozenset[str] = frozenset({"spectral", "onset", "none"})

#: `Arrangement.status`.
ARRANGEMENT_STATUSES: frozenset[str] = frozenset({"ok", "no_grid", "too_short", "unavailable"})

#: `Section.label`. `silence` is a real section: the calibration track ends with
#: three bars of nothing and calling that an `outro` would be wrong.
SECTION_LABELS: frozenset[str] = frozenset(
    {"intro", "outro", "breakdown", "drop", "full", "groove", "silence"}
)

#: Octaves of the coarse tempo estimate that are measured and reported.
#: **Must equal `tempo.OCTAVE_RATIOS`**; a test asserts it rather than trusting
#: the copy, because this module may not import that one (see above).
OCTAVE_RATIOS: tuple[float, ...] = (0.5, 1.0, 2.0)


class MultipleFit(BaseModel):
    """One beat multiple's independent reading of the tempo.

    Kept rather than collapsed, because "the two multiples agreed" and "one of
    them was rejected" are different facts and the second is what a reader
    needs when a fit looks wrong.
    """

    beats: int = 0
    bpm: float | None = None
    lag_frames: float | None = Field(
        default=None, description="Interpolated peak lag in STFT frames, for auditing"
    )
    r: float = Field(default=0.0, description="Autocorrelation at the integer peak")
    accepted: bool = False
    reason: str | None = Field(
        default=None,
        description="Why not, when not accepted: too_short | below_correlation_floor | "
        "outside_bpm_tolerance",
    )


class OctaveCandidate(BaseModel):
    """One octave of the coarse estimate, and how well the source supports it.

    Evidence produced by `tempo.py`, which deliberately reports octaves rather
    than choosing between them: autocorrelation decays with lag, so the doubled
    candidate scores higher on nearly all material and `r` cannot arbitrate.
    Choosing is `TempoFit.octave_arbitration`'s job, on grid quality.
    """

    ratio: float = Field(
        description=f"Multiplier against the coarse estimate; one of {OCTAVE_RATIOS}"
    )
    bpm: float = 0.0
    r: float = Field(
        default=0.0, description="Autocorrelation at this octave, at the first beat multiple"
    )
    status: str = Field(
        default="unmeasurable",
        description=f"One of: {', '.join(sorted(OCTAVE_CANDIDATE_STATUSES))}",
    )


class OctaveGridFit(BaseModel):
    """How well one live octave candidate's grid actually fits the drums.

    New in v5 and not mirrored from any dataclass: this is the arbitration
    `tempo.py` proved it could not do from correlation alone, decided in
    `analyze.py` by fitting `drum_elements.decompose` at each live candidate.

    **Both error columns are carried because neither is scale-free.**
    `quantisation_error_steps` shrinks when the tempo halves, because the step
    it is measured in doubles; `quantisation_error_seconds` shrinks when the
    tempo doubles, for the mirror-image reason. Measured on the corpus: at half
    tempo the swung hip-hop track scores 0.0689 steps against its true tempo's
    0.1115 and would win on that statistic alone. A candidate is therefore only
    allowed to displace the incumbent when it wins on **both** — that is, when
    it wins despite the bias running against it in either direction. It is a
    comparison, not a threshold, so there is nothing here to tune.
    """

    ratio: float
    bpm: float
    grid_status: str = Field(
        default="not_attempted",
        description=f"`DrumDecomposition.status` at this octave; one of: "
        f"{', '.join(sorted(BLOCK_STATUSES))}",
    )
    quantisation_error_steps: float | None = None
    quantisation_error_seconds: float | None = Field(
        default=None, description="The same error in seconds: error_steps * cycle / steps_per_cycle"
    )
    chosen: bool = False


class TempoStability(BaseModel):
    """Whether the tempo is the same at the end of the source as at the start.

    Both halves are fitted independently with the same guards as the whole, so
    a track whose tempo genuinely moves is detectable rather than silently
    averaged into a number that is true nowhere in it.
    """

    first_half_bpm: float | None = None
    second_half_bpm: float | None = None
    delta_bpm: float | None = None
    label: str = Field(
        default="unknown", description=f"One of: {', '.join(sorted(STABILITY_LABELS))}"
    )


class TempoFit(BaseModel):
    """The track's one refined tempo, or an honest refusal to refine it.

    **Additive, never a replacement.** Every source's `rhythm.bpm` keeps its
    backend estimate untouched; this carries the refined value. The two differ
    by enough to matter: on the calibration track the backend reported 131.855
    on the mix and 132.040 on the drums, and a 0.040 BPM error accumulates 82 ms
    over 147 bars — three quarters of a sixteenth step, which is what made the
    tool declare that a textbook four-on-the-floor grid did not exist.

    `status="coarse"` means refinement was **refused** and `bpm` is just the
    backend's estimate. Read `status` and `confidence_label` before `bpm`: on
    the live-band corpus row the backend reports `bpm_confidence` 2.12 while
    this block reports r = 0.108 and confidence 0.000, and the two disagree
    completely.
    """

    bpm: float | None = Field(
        default=None,
        description="Refined when status is `refined`, the coarse estimate when `coarse`",
    )
    period_seconds: float | None = Field(
        default=None, description="60 / bpm — the number every downstream grid wants"
    )
    coarse_bpm: float | None = Field(
        default=None, description="What was handed in, so the size of the correction is visible"
    )
    confidence: float = Field(
        default=0.0, description="Autocorrelation r, weakest across the accepted multiples"
    )
    confidence_label: str = Field(
        default="low", description=f"One of: {', '.join(sorted(CONFIDENCE_LABELS))}"
    )
    status: str = Field(
        default="unavailable", description=f"One of: {', '.join(sorted(TEMPO_STATUSES))}"
    )
    multiples: list[MultipleFit] = Field(default_factory=list)
    stability: TempoStability = Field(default_factory=TempoStability)
    octave_candidates: list[OctaveCandidate] = Field(
        default_factory=list, description="The x0.5, x1 and x2 readings, in `OCTAVE_RATIOS` order"
    )
    octave_status: str = Field(
        default="unavailable", description=f"One of: {', '.join(sorted(OCTAVE_STATUSES))}"
    )
    #: v5 only, and the reason `bpm` may differ from what `tempo.py` returned.
    octave_arbitration: list[OctaveGridFit] = Field(
        default_factory=list,
        description=(
            "Grid quality at each live octave candidate. Empty when there was "
            "nothing to arbitrate. When an entry has `chosen=True` and a `ratio` "
            "other than 1.0, `bpm` here is that octave's — never silently: "
            "`caveats` says so in words as well."
        ),
    )
    caveats: list[str] = Field(default_factory=list)


class DownbeatFit(BaseModel):
    """Where bar one starts, and how much that should be trusted.

    Two independent quantities, and conflating them hides the interesting one.
    `beat_confidence` says how well the beat grid's phase is pinned, which on
    percussive material is near certain. `phase_confidence` says whether the
    right *beat of the bar* was identified, which on four-on-the-floor material
    frequently cannot be — a kick on every beat scores identically at every
    offset, and a wrong pick rotates every step number downstream while the
    output still looks plausible.
    """

    offset_seconds: float | None = Field(
        default=None, description="First downbeat, in seconds. Always the earliest such position"
    )
    confidence: float = Field(
        default=0.0, description="min(beat_confidence, phase_confidence) — the weaker stage"
    )
    confidence_label: str = Field(
        default="low", description=f"One of: {', '.join(sorted(CONFIDENCE_LABELS))}"
    )
    beat_offset_seconds: float | None = Field(
        default=None, description="Beat-grid phase alone, correct even when the bar phase is not"
    )
    beat_confidence: float = 0.0
    phase_confidence: float = Field(
        default=0.0, description="How cleanly the winning bar phase beat the runner-up"
    )
    bar_phase: int | None = Field(default=None, description="Which beat of the bar won; 0 is first")
    candidate_offsets: list[float] = Field(default_factory=list)
    candidate_scores: list[float] = Field(default_factory=list)
    unresolved_offsets: list[float] = Field(
        default_factory=list, description="Candidates that could not be ruled out"
    )
    resolved_by: str = Field(
        default="none", description=f"One of: {', '.join(sorted(DOWNBEAT_RESOLVED_BY))}"
    )
    status: str = Field(
        default="unavailable", description=f"One of: {', '.join(sorted(DOWNBEAT_STATUSES))}"
    )
    caveats: list[str] = Field(default_factory=list)


class Section(BaseModel):
    """A run of bars over which the same set of tracks is playing.

    `label` is derived and `label_reason` is the evidence, so a reader who
    disagrees with the name still has the measurement. Same rule as every label
    in `heuristics.py`: one that cannot be audited is decoration.
    """

    start_bar: int = 0
    length_bars: int = 0
    start_seconds: float = 0.0
    active: list[str] = Field(
        default_factory=list, description="Tracks playing throughout, including the kick"
    )
    label: str | None = Field(
        default=None, description=f"One of: {', '.join(sorted(SECTION_LABELS))}"
    )
    label_reason: str | None = None


class Arrangement(BaseModel):
    """The track's structure, or an honest statement that there is none.

    `sections` is deliberately **not** in `_SUMMARY_LIST_FIELDS`. It is one
    entry per section rather than per event — sixteen on the calibration track
    — and it is the entire point of the feature, exactly the reasoning that
    keeps `DrumPattern.patterns` out of the strip table too.

    Like `DrumDecomposition`, this carries an explicit `status` and is not
    routed through `analyze._collect_unavailable`: an empty `sections` list on
    an ambient record is a *correct* answer, not a missing feature.
    """

    sections: list[Section] = Field(default_factory=list)
    bar_count: int = 0
    bar_seconds: float | None = None
    downbeat_seconds: float | None = None
    absent_tracks: list[str] = Field(
        default_factory=list,
        description=(
            "Tracks not in this record at all, gated out by a within-record "
            "level ratio rather than by their own distribution — the test that "
            "stops a stem of separation bleed reporting an arrangement."
        ),
    )
    status: str = Field(
        default="unavailable", description=f"One of: {', '.join(sorted(ARRANGEMENT_STATUSES))}"
    )
    caveats: list[str] = Field(default_factory=list)


class ArrangementHint(BaseModel):
    """The track's structure as a handful of one-line strings.

    Deliberately prose rather than the full `Arrangement`: this file is
    documented as small and hand-readable, and `track_summary.json` already
    carries every section with the evidence behind its label. `"bar 76 x15
    breakdown"` is what you need while typing a pattern; `label_reason` is what
    you need while arguing with the tool, and it lives in the other file.
    """

    status: str = Field(
        default="unavailable", description=f"One of: {', '.join(sorted(ARRANGEMENT_STATUSES))}"
    )
    bar_count: int = 0
    bar_seconds: float | None = None
    sections: list[str] = Field(
        default_factory=list, description="e.g. 'bar 76 x15 breakdown: drums, vocals, other'"
    )
    truncated_from: int | None = Field(
        default=None, description="Real section count when `sections` was capped, else None"
    )
    absent_tracks: list[str] = Field(
        default_factory=list, description="Tracks not in this record at all — do not write parts"
    )
    caveats: list[str] = Field(default_factory=list)


class SourceAnalysis(BaseModel):
    """Full analysis of one source: the mix or a single stem."""

    schema_version: int = SCHEMA_VERSION
    source: str = Field(description="One of: mix, drums, bass, vocals, other")
    audio_path: str
    duration_seconds: float
    sample_rate: int
    backend: str = Field(description="Analysis backend used, e.g. essentia or librosa")
    rhythm: RhythmFeatures = Field(default_factory=RhythmFeatures)
    tonal: TonalFeatures = Field(default_factory=TonalFeatures)
    spectral: SpectralFeatures = Field(default_factory=SpectralFeatures)
    dynamics: DynamicsFeatures = Field(default_factory=DynamicsFeatures)
    # Placed here, between `dynamics` and `labels`, on purpose:
    # `write_source_analysis` serialises with `sort_keys=False`, so declaration
    # order is JSON key order, and the file reads measurements first, then the
    # things derived from them.
    drum_decomposition: DrumDecomposition = Field(default_factory=DrumDecomposition)
    bass_line: BassLine = Field(default_factory=BassLine)
    labels: list[HeuristicLabel] = Field(default_factory=list)
    unavailable_features: list[str] = Field(default_factory=list)


#: Per-source event lists dropped from `track_summary.json`, mapped to the count
#: field that takes their place. Keyed by the `SourceAnalysis` block the list
#: lives on, then by field name. They stay complete in
#: `analysis/<source>.json`, so nothing is lost — the summary just stops being
#: thousands of floats of duplicated data across five sources.
#:
#: This one table drives both directions: `TrackSummary.summary_payload()`
#: strips, `rehydrate_stripped_lists()` puts back. They cannot drift apart
#: because neither of them names a field itself.
#:
#: **`DrumDecomposition.patterns` is deliberately absent.** It is cycle-folded
#: and tiny (at most `steps_per_cycle` ints per class), and that compact one-bar
#: pattern is the entire point of the feature — it belongs in the file you read
#: by hand.
#:
#: `hits` becomes `total_hit_count` rather than `hit_count` because
#: `DrumPattern` already carries a per-class `hit_count`, and `patterns` is not
#: stripped, so both would appear in the same block of the same file meaning
#: different things.
_SUMMARY_LIST_FIELDS: dict[str, dict[str, str]] = {
    "rhythm": {"beat_times": "beat_count", "onset_times": "onset_count"},
    "drum_decomposition": {"hits": "total_hit_count"},
    "bass_line": {"notes": "note_count"},
}


def rehydrate_stripped_lists(target: SourceAnalysis, full: SourceAnalysis) -> None:
    """Copy every list `summary_payload()` strips from `full` onto `target`.

    `track_summary.json` carries counts where the event lists were, so a
    `TrackSummary` loaded back off disk cannot answer questions that need the
    times themselves. This restores them from a source's own
    `analysis/<source>.json`, in place.

    Driven by `_SUMMARY_LIST_FIELDS`, the same table the stripping reads, so
    adding a list to one side automatically adds it to the other. Blocks or
    fields a model does not have are skipped rather than raising, so a summary
    written by an older schema version still loads.
    """
    for block_name, list_fields in _SUMMARY_LIST_FIELDS.items():
        target_block = getattr(target, block_name, None)
        full_block = getattr(full, block_name, None)
        if target_block is None or full_block is None:
            continue
        for field_name in list_fields:
            if hasattr(target_block, field_name) and hasattr(full_block, field_name):
                setattr(target_block, field_name, getattr(full_block, field_name))


class TrackSummary(BaseModel):
    """Combined view across the mix and all stems.

    Beat and onset times live in the per-source `analysis/*.json` files only.
    They are omitted from the written summary — a six-minute track produces
    roughly 720 beat floats per source, and a busy drum stem several times that
    in onsets, which is pure duplication and makes the one file you actually
    read by hand unreadable. Use `summary_payload()` to serialise.

    **Schema v5 put `tempo`, `downbeat` and `arrangement` here rather than on
    `SourceAnalysis`.** There is one tempo and one structure per record. v4 gave
    every source its own, and the five disagreed — 131.855 / 132.040 / 131.815 /
    130.359 / 131.992 on the calibration track — with different modules silently
    consuming different ones. Whichever a downstream module happened to read, it
    built a grid that drifted. These three fields are what "resolve it once and
    share it" means in the output shape.

    A `harmony` block belongs in the same position, between `arrangement` and
    `sources`, whenever the deferred harmony package lands. Reserving the slot
    now is why adding it later is a field addition rather than a second
    structural change.
    """

    schema_version: int = SCHEMA_VERSION
    track_name: str
    input_path: str
    duration_seconds: float
    backend: str
    separation_model: str | None = None
    separation_device: str | None = None
    analysis_sample_rate: int = ANALYSIS_SAMPLE_RATE
    tempo: TempoFit = Field(default_factory=TempoFit)
    downbeat: DownbeatFit = Field(default_factory=DownbeatFit)
    arrangement: Arrangement = Field(default_factory=Arrangement)
    # (harmony goes here — see the class docstring)
    sources: dict[str, SourceAnalysis] = Field(default_factory=dict)

    def summary_payload(self) -> dict[str, object]:
        """Dict for writing `track_summary.json`, with event lists stripped.

        Every list named in `_SUMMARY_LIST_FIELDS` — beat and onset times, drum
        hits, bass notes — is replaced by its length under the paired count
        name, so nothing is silently lost and the full lists stay in each
        source's own analysis file. `DrumDecomposition.patterns` is not one of
        them: the folded one-bar pattern is small and is what you came for.

        Each count takes the slot its list occupied, so key order across the rest
        of the payload is unchanged and run-to-run diffs stay readable.
        """
        payload = self.model_dump(mode="json")
        for source in payload.get("sources", {}).values():
            if not isinstance(source, dict):
                continue
            for block_name, list_fields in _SUMMARY_LIST_FIELDS.items():
                block = source.get(block_name)
                if not isinstance(block, dict):
                    continue
                source[block_name] = {
                    list_fields.get(key, key): (len(value) if key in list_fields else value)
                    for key, value in block.items()
                }
        return payload


class StrudelHints(BaseModel):
    """Compact, hand-readable starting point for rebuilding the track in Strudel."""

    schema_version: int = SCHEMA_VERSION
    track_name: str
    bpm: float | None = None
    #: The two fields v4 was missing, and the reason it could print a confident
    #: `bpm: 143.25` for a track whose tempo the tool had explicitly declined to
    #: measure. `tempo_status="coarse"` means `bpm` is the backend's raw estimate
    #: and no grid was ever fitted to it; `tempo_confidence="low"` means the
    #: refinement found nothing periodic. On the ambient corpus row both fire, on
    #: a record with no pulse at all. Read them before `bpm`.
    tempo_status: str | None = Field(
        default=None, description=f"One of: {', '.join(sorted(TEMPO_STATUSES))}"
    )
    tempo_confidence: str | None = Field(
        default=None, description=f"One of: {', '.join(sorted(CONFIDENCE_LABELS))}"
    )
    suggested_cycle_seconds: float | None = None
    subdivision_feel: str | None = Field(
        default=None, description="e.g. 'straight 16ths', 'swung 8ths'"
    )
    drum_density: str | None = Field(default=None, description="sparse | moderate | busy")
    bass_activity: str | None = None
    tonal_centre: str | None = None
    drum_grid: DrumGridHint = Field(default_factory=DrumGridHint)
    bass_line: BassLineHint = Field(default_factory=BassLineHint)
    arrangement: ArrangementHint = Field(default_factory=ArrangementHint)
    sound_suggestions: list[StrudelSoundSuggestion] = Field(default_factory=list)
    strudel_vocabulary_read: str | None = Field(
        default=None,
        description=(
            "ISO date the Strudel sound vocabulary was transcribed from the live "
            "docs. Strudel is actively developed and nothing offline can detect "
            "the tables going stale, so every hints file carries the date it was "
            "built against and identifies itself as old."
        ),
    )
    notes: list[str] = Field(default_factory=list)
