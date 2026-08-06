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
            "larger than the mean is the signature. Prefer `centroid_median`; "
            "this field is kept so v4 outputs stay comparable."
        ),
    )
    centroid_std: float | None = None
    centroid_median: float | None = Field(
        default=None,
        description=(
            "Energy-weighted centroid: the frequency below which half the "
            "source's total spectral energy sits. Silent frames carry no weight, "
            "so this reads what the stem sounds like rather than what its noise "
            "floor looks like. Both backends use an identical definition. This is "
            "the field thresholds should read; `centroid_mean` is retained only "
            "for continuity with v4."
        ),
    )
    rolloff_mean: float | None = None
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
#:   failed         raised, and the exception was swallowed to keep the
#:                  analysis going
BLOCK_STATUSES: frozenset[str] = frozenset(
    {"not_attempted", "ok", "no_grid", "too_few_hits", "unvoiced", "failed"}
)

#: What the drum grid was anchored on: `rhythm.beat_times[0]`, or failing that
#: the first detected hit. Recorded either way, because a grid anchored on a
#: tempo estimate and one anchored on whatever happened to be loudest first
#: deserve different amounts of trust.
#:
#: Spelt `beats` rather than `beat_times` deliberately — a *value* that reads
#: like a stripped *field name* makes `track_summary.json` ambiguous to grep and
#: breaks the guard that no event list survives into it.
GRID_ANCHOR_SOURCES: frozenset[str] = frozenset({"beats", "first_hit"})

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
    """

    schema_version: int = SCHEMA_VERSION
    track_name: str
    input_path: str
    duration_seconds: float
    backend: str
    separation_model: str | None = None
    separation_device: str | None = None
    analysis_sample_rate: int = ANALYSIS_SAMPLE_RATE
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
    suggested_cycle_seconds: float | None = None
    subdivision_feel: str | None = Field(
        default=None, description="e.g. 'straight 16ths', 'swung 8ths'"
    )
    drum_density: str | None = Field(default=None, description="sparse | moderate | busy")
    bass_activity: str | None = None
    tonal_centre: str | None = None
    drum_grid: DrumGridHint = Field(default_factory=DrumGridHint)
    bass_line: BassLineHint = Field(default_factory=BassLineHint)
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
