"""Tests for `TrackSummary.summary_payload()` and `rehydrate_stripped_lists()`.

Beat and onset times, drum hits and bass notes stay complete in
`analysis/<source>.json`; the summary carries only counts, so the one file you
read by hand is not thousands of duplicated floats. Onset lists were the
largest of these — a busy drum stem at 6 onsets/sec over five minutes is ~1,800
floats on its own — until per-band drum detection started emitting a hit per
band per instant.

`DrumDecomposition.patterns` is deliberately *not* stripped, and there are
tests here for that: it is cycle-folded, tiny, and the reason the feature
exists.
"""

from __future__ import annotations

import json
import typing

from pydantic import BaseModel

from audio_pipeline import SCHEMA_VERSION, SOURCE_NAMES
from audio_pipeline.schemas import (
    _SUMMARY_LIST_FIELDS,
    BLOCK_STATUSES,
    DRUM_CLASSES,
    GRID_ANCHOR_SOURCES,
    SOUND_MATCH_TERMS,
    Arrangement,
    BassLine,
    BassLineHint,
    BassNote,
    DrumDecomposition,
    DrumGridHint,
    DrumHit,
    DrumPattern,
    PitchTrack,
    RhythmFeatures,
    Section,
    SourceAnalysis,
    StrudelHints,
    StrudelSoundSuggestion,
    TrackSummary,
    rehydrate_stripped_lists,
)

STRIPPED_LIST_FIELDS = ("beat_times", "onset_times")
COUNT_FIELDS = ("beat_count", "onset_count")


def _drums(hits: int = 3) -> DrumDecomposition:
    return DrumDecomposition(
        status="ok",
        steps_per_cycle=16,
        cycle_seconds=2.0,
        grid_anchor_seconds=0.25,
        grid_anchor_source="beats",
        quantisation_error_steps=0.02,
        patterns=[
            DrumPattern(drum="kick", steps=[0, 8], step_occupancy=[1.0, 1.0], hit_count=8),
            DrumPattern(drum="snare", steps=[4, 12], step_occupancy=[1.0, 0.75], hit_count=7),
        ],
        hits=[
            DrumHit(time_seconds=0.25 * index, drum="kick", confidence=0.8, step=index)
            for index in range(hits)
        ],
        unclassified_count=1,
    )


def _bass(notes: int = 2) -> BassLine:
    return BassLine(
        status="ok",
        notes=[
            BassNote(
                start_seconds=0.5 * index,
                duration_seconds=0.46,
                midi_note=33,
                note_name="a1",
                median_f0_hz=55.0,
            )
            for index in range(notes)
        ],
        median_midi_note=33,
        median_cents_offset=1.5,
        voiced_fraction=0.8,
    )


def _source(
    name: str,
    beat_times: list[float],
    onset_times: list[float] | None = None,
) -> SourceAnalysis:
    return SourceAnalysis(
        source=name,
        audio_path=f"output/demo/stems/{name}.wav",
        duration_seconds=12.0,
        sample_rate=44100,
        backend="librosa",
        rhythm=RhythmFeatures(
            bpm=120.0,
            bpm_confidence=0.9,
            beat_times=beat_times,
            onset_times=[t * 0.25 for t in beat_times] if onset_times is None else onset_times,
            onset_density=2.0,
            transient_sharpness=3.5,
        ),
        drum_decomposition=_drums(),
        bass_line=_bass(),
    )


def _summary() -> TrackSummary:
    return TrackSummary(
        track_name="demo",
        input_path="examples/demo.wav",
        duration_seconds=12.0,
        backend="librosa",
        separation_model="htdemucs_ft",
        separation_device="mps",
        sources={
            name: _source(name, [i * 0.5 for i in range(1 + index * 3)])
            for index, name in enumerate(SOURCE_NAMES)
        },
    )


def _walk_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        keys: list[str] = []
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(child))
        return keys
    if isinstance(value, list):
        return [key for item in value for key in _walk_keys(item)]
    return []


def test_payload_contains_no_event_lists_anywhere() -> None:
    keys = _walk_keys(_summary().summary_payload())
    for field in STRIPPED_LIST_FIELDS:
        assert field not in keys


def test_counts_match_the_original_list_lengths() -> None:
    summary = _summary()
    payload = summary.summary_payload()

    sources = payload["sources"]
    assert isinstance(sources, dict)
    for name, source in summary.sources.items():
        rhythm = sources[name]["rhythm"]
        assert rhythm["beat_count"] == len(source.rhythm.beat_times)
        assert rhythm["onset_count"] == len(source.rhythm.onset_times)

    # Non-trivial: at least one source really did have events to count.
    for count_field in COUNT_FIELDS:
        assert any(sources[name]["rhythm"][count_field] > 0 for name in summary.sources)


def test_onsets_without_beats_and_beats_without_onsets() -> None:
    """The two lists are independent — a stem can have either, both, or neither.

    A sustained pad has a pulse the mix implies but no detectable attacks; an
    unquantised percussion loop can have plenty of onsets and no stable beat.
    Each count must reflect its own list, not the other one.
    """
    summary = TrackSummary(
        track_name="demo",
        input_path="examples/demo.wav",
        duration_seconds=12.0,
        backend="librosa",
        sources={
            "bass": _source("bass", beat_times=[0.0, 0.5, 1.0], onset_times=[]),
            "drums": _source("drums", beat_times=[], onset_times=[0.1, 0.2, 0.35, 0.5]),
            "other": _source("other", beat_times=[], onset_times=[]),
        },
    )

    sources = summary.summary_payload()["sources"]
    assert isinstance(sources, dict)

    assert sources["bass"]["rhythm"] == {
        "bpm": 120.0,
        "bpm_confidence": 0.9,
        "beat_count": 3,
        "onset_count": 0,
        "onset_density": 2.0,
        "transient_sharpness": 3.5,
    }
    assert sources["drums"]["rhythm"]["beat_count"] == 0
    assert sources["drums"]["rhythm"]["onset_count"] == 4
    assert sources["other"]["rhythm"]["beat_count"] == 0
    assert sources["other"]["rhythm"]["onset_count"] == 0


def test_everything_else_survives_intact() -> None:
    summary = _summary()
    payload = summary.summary_payload()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["track_name"] == "demo"
    assert payload["separation_model"] == "htdemucs_ft"
    assert payload["separation_device"] == "mps"
    assert payload["analysis_sample_rate"] == 44100

    sources = payload["sources"]
    assert isinstance(sources, dict)
    assert set(sources) == set(SOURCE_NAMES)
    rhythm = sources["mix"]["rhythm"]
    assert rhythm["bpm"] == 120.0
    assert rhythm["bpm_confidence"] == 0.9
    assert rhythm["onset_density"] == 2.0
    assert rhythm["transient_sharpness"] == 3.5
    assert sources["mix"]["spectral"]["band_energy_ratios"] == {
        "low": None,
        "low_mid": None,
        "high_mid": None,
        "high": None,
    }


def test_each_count_keeps_the_slot_its_list_had() -> None:
    """Stable key order keeps run-to-run diffs of the summary readable."""
    payload = _summary().summary_payload()
    sources = payload["sources"]
    assert isinstance(sources, dict)
    assert list(sources["mix"]["rhythm"]) == [
        "bpm",
        "bpm_confidence",
        "beat_count",
        "onset_count",
        "onset_density",
        "transient_sharpness",
    ]


def test_source_analysis_still_carries_the_full_lists() -> None:
    """Stripping is a property of the summary only — nothing is actually lost."""
    source = _summary().sources["vocals"]
    dumped = source.model_dump(mode="json")

    assert dumped["rhythm"]["beat_times"] == source.rhythm.beat_times
    assert dumped["rhythm"]["onset_times"] == source.rhythm.onset_times
    assert dumped["rhythm"]["onset_times"]


def test_payload_is_json_serialisable() -> None:
    """It is written straight to disk, so it must not contain numpy or models."""
    text = json.dumps(_summary().summary_payload(), indent=2, sort_keys=False)
    for field in STRIPPED_LIST_FIELDS:
        assert field not in text
    assert json.loads(text)["track_name"] == "demo"


def test_schema_version_is_current() -> None:
    """drum_decomposition and bass_line were added at schema version 4."""
    assert SCHEMA_VERSION >= 4
    assert _summary().summary_payload()["schema_version"] == SCHEMA_VERSION


def test_empty_summary_payload_does_not_raise() -> None:
    summary = TrackSummary(
        track_name="demo",
        input_path="examples/demo.wav",
        duration_seconds=0.0,
        backend="librosa",
    )
    assert summary.summary_payload()["sources"] == {}


# --- schema v4: drum and bass blocks -----------------------------------------


def test_summary_list_fields_all_name_real_fields() -> None:
    """Meta-test: every key in the table must name a real field on a real model.

    The table drives stripping *and* rehydration, so a typo in it would silently
    stop doing both rather than failing.
    """
    for block_name, list_fields in _SUMMARY_LIST_FIELDS.items():
        assert block_name in SourceAnalysis.model_fields, block_name
        block_model = SourceAnalysis.model_fields[block_name].annotation
        assert isinstance(block_model, type) and issubclass(block_model, BaseModel)
        for field_name, count_name in list_fields.items():
            assert field_name in block_model.model_fields, f"{block_name}.{field_name}"
            assert typing.get_origin(block_model.model_fields[field_name].annotation) is list
            # The count name must not collide with a real field, or the strip
            # would overwrite one value with another.
            assert count_name not in block_model.model_fields, count_name


def test_drum_hits_and_bass_notes_are_stripped_to_counts() -> None:
    payload = _summary().summary_payload()
    sources = payload["sources"]
    assert isinstance(sources, dict)

    drums = sources["drums"]["drum_decomposition"]
    assert "hits" not in drums
    assert drums["total_hit_count"] == 3

    bass = sources["bass"]["bass_line"]
    assert "notes" not in bass
    assert bass["note_count"] == 2

    assert "hits" not in _walk_keys(payload)
    assert "notes" not in _walk_keys(payload)


def test_folded_patterns_survive_the_summary() -> None:
    """The one-bar pattern is the point of the feature — it must not be stripped."""
    sources = _summary().summary_payload()["sources"]
    assert isinstance(sources, dict)
    patterns = sources["drums"]["drum_decomposition"]["patterns"]

    assert [entry["drum"] for entry in patterns] == ["kick", "snare"]
    assert patterns[0]["steps"] == [0, 8]
    assert patterns[0]["step_occupancy"] == [1.0, 1.0]
    # Per-class count, distinct from the whole-source `total_hit_count` above.
    assert patterns[1]["hit_count"] == 7


def test_the_two_hit_counts_do_not_collide() -> None:
    """`DrumPattern.hit_count` is per class; the stripped one is per source.

    They live in the same block of the same file, so they are named differently
    on purpose. This pins that neither name leaks into the other's model.
    """
    assert "hit_count" in DrumPattern.model_fields
    assert "hit_count" not in DrumDecomposition.model_fields
    assert "total_hit_count" not in DrumPattern.model_fields
    assert _SUMMARY_LIST_FIELDS["drum_decomposition"]["hits"] == "total_hit_count"


def test_counts_keep_the_slots_their_lists_had_in_every_block() -> None:
    sources = _summary().summary_payload()["sources"]
    assert isinstance(sources, dict)

    assert list(sources["mix"]["drum_decomposition"]) == [
        "status",
        "steps_per_cycle",
        "cycle_seconds",
        "grid_anchor_seconds",
        "grid_anchor_source",
        "quantisation_error_steps",
        "patterns",
        "total_hit_count",
        "unclassified_count",
        "caveats",
    ]
    assert list(sources["mix"]["bass_line"]) == [
        "status",
        "note_count",
        "median_midi_note",
        "median_cents_offset",
        "voiced_fraction",
        "octave_corrections",
        "caveats",
    ]


def test_new_blocks_sit_between_dynamics_and_labels() -> None:
    """JSON key order is declaration order — measurements, then derived."""
    order = list(SourceAnalysis.model_fields)
    assert order[order.index("dynamics") + 1] == "drum_decomposition"
    assert order[order.index("drum_decomposition") + 1] == "bass_line"
    assert order[order.index("bass_line") + 1] == "labels"


def test_default_blocks_are_present_and_not_attempted() -> None:
    """Every source carries both blocks, so their absence is never ambiguous."""
    analysis = SourceAnalysis(
        source="vocals",
        audio_path="output/demo/stems/vocals.wav",
        duration_seconds=1.0,
        sample_rate=44100,
        backend="librosa",
    )
    assert analysis.drum_decomposition.status == "not_attempted"
    assert analysis.bass_line.status == "not_attempted"
    assert analysis.drum_decomposition.hits == []
    assert analysis.bass_line.notes == []


def _referenced_models(model: type[BaseModel], seen: set[type[BaseModel]]) -> set[type[BaseModel]]:
    """Every pydantic model reachable from `model`'s fields, recursively."""
    if model in seen:
        return seen
    seen.add(model)
    for info in model.model_fields.values():
        annotations = [info.annotation, *typing.get_args(info.annotation)]
        for annotation in annotations:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                _referenced_models(annotation, seen)
    return seen


def test_pitch_track_is_never_reachable_from_source_analysis() -> None:
    """It is transport only: ~25,800 F0 floats for a five-minute track.

    `note_track.segment_notes()` turns a `PitchTrack` into a `BassLine`, and
    only the `BassLine` is written. If a field ever references it, every
    `analysis/bass.json` silently grows by megabytes.
    """
    reachable = _referenced_models(SourceAnalysis, set())
    assert PitchTrack not in reachable
    assert BassLine in reachable  # the walk really does find nested models
    assert DrumPattern in reachable  # ...including through a list


def test_pitch_track_floats_are_unbounded() -> None:
    """A backend return type: the fake-backend filler can hand it 1.06.

    `tests/test_analyze.py::_PRIMITIVE_FILLERS[float]` returns
    `0.1 + (hash % 97) / 100`, which exceeds 1.0 for plenty of field names. A
    `Field(le=1.0)` on a nominally 0-1 field would turn that into a validation
    error a long way from its cause.
    """
    track = PitchTrack(
        f0_hz=[55.0, 0.0],
        voiced=[True, False],
        voiced_probability=[1.06, -0.02],
        frame_hop_seconds=512 / 44100,
        method="pyin",
    )
    assert track.voiced_probability == [1.06, -0.02]


def test_rehydrate_restores_every_stripped_list() -> None:
    """The strip and the rehydrate read the same table, so they cannot drift."""
    full = _source("drums", [0.0, 0.5, 1.0])
    reloaded = TrackSummary.model_validate(
        json.loads(json.dumps(_summary().summary_payload()))
    ).sources["drums"]

    # Loaded from the summary, the lists are gone and the counts are not fields.
    assert reloaded.rhythm.beat_times == []
    assert reloaded.drum_decomposition.hits == []
    assert reloaded.bass_line.notes == []

    rehydrate_stripped_lists(reloaded, full)

    assert reloaded.rhythm.beat_times == full.rhythm.beat_times
    assert reloaded.rhythm.onset_times == full.rhythm.onset_times
    assert reloaded.drum_decomposition.hits == full.drum_decomposition.hits
    assert reloaded.bass_line.notes == full.bass_line.notes


def test_rehydrate_touches_nothing_else() -> None:
    target = _source("drums", [])
    target.drum_decomposition.status = "no_grid"
    target.bass_line.median_midi_note = 99

    rehydrate_stripped_lists(target, _source("drums", [0.0, 0.5]))

    assert target.drum_decomposition.status == "no_grid"
    assert target.bass_line.median_midi_note == 99
    assert target.rhythm.beat_times == [0.0, 0.5]


def test_rehydrate_from_an_empty_source_is_a_no_op_not_a_crash() -> None:
    """A missing `analysis/<source>.json` must not take the hints path down."""
    target = _source("drums", [1.0, 2.0])
    empty = SourceAnalysis(
        source="drums",
        audio_path="missing.wav",
        duration_seconds=0.0,
        sample_rate=44100,
        backend="librosa",
    )
    rehydrate_stripped_lists(target, empty)
    assert target.rhythm.beat_times == []
    assert target.drum_decomposition.hits == []


# --- schema v4: the closed vocabularies --------------------------------------
#
# These frozensets are cross-agent contracts, and they exist instead of `Literal`
# annotations because `tests/test_analyze.py::_fill_value` raises on a `Literal`.
# Retune what produces them; never rename or remove a member without saying so.


def test_vocabularies_hold_exactly_the_documented_values() -> None:
    assert DRUM_CLASSES == {"kick", "snare", "hat", "unclassified"}
    assert BLOCK_STATUSES == {
        "not_attempted",
        "ok",
        "no_grid",
        "too_few_hits",
        "unvoiced",
        "silent",
        "failed",
    }
    assert GRID_ANCHOR_SOURCES == {"supplied", "beats", "first_hit"}
    assert SOUND_MATCH_TERMS == {"exact", "approximate", "none"}


# --- schema v5: the promoted dataclasses -------------------------------------
#
# `analyze.py` converts `tempo.py`'s and `arrangement.py`'s frozen dataclasses
# into the models above with a plain `dataclasses.asdict()` fed to
# `model_validate()`. That works only while the field names match exactly, and
# it fails silently the day they do not: pydantic fills the missing field from
# its default and drops the unknown one, so a renamed field becomes a plausible
# zero rather than an error. These tests are what make that conversion safe to
# write as one line.


def _dataclass_field_names(dataclass_type: object) -> set[str]:
    import dataclasses

    return {field.name for field in dataclasses.fields(dataclass_type)}  # type: ignore[arg-type]


def test_promoted_models_mirror_their_dataclasses_field_for_field() -> None:
    from audio_pipeline import arrangement, schemas, tempo

    pairs = [
        (schemas.MultipleFit, tempo.MultipleFit),
        (schemas.OctaveCandidate, tempo.OctaveCandidate),
        (schemas.TempoStability, tempo.TempoStability),
        (schemas.DownbeatFit, tempo.DownbeatFit),
        (schemas.Section, arrangement.Section),
        (schemas.Arrangement, arrangement.Arrangement),
    ]
    for model, dataclass_type in pairs:
        assert set(model.model_fields) == _dataclass_field_names(dataclass_type), model.__name__


def test_tempo_fit_adds_only_the_arbitration_w6_decides() -> None:
    """`TempoFit` is the one model with a field its dataclass does not have.

    `tempo.py` measures the octave candidates and deliberately refuses to
    choose between them — correlation cannot, and it proved so. `analyze.py`
    arbitrates on grid quality and records the evidence here.
    """
    from audio_pipeline import schemas, tempo

    extra = set(schemas.TempoFit.model_fields) - _dataclass_field_names(tempo.TempoFit)
    assert extra == {"octave_arbitration"}
    assert _dataclass_field_names(tempo.TempoFit) <= set(schemas.TempoFit.model_fields)


def test_v5_vocabularies_match_the_literals_they_were_copied_from() -> None:
    from audio_pipeline import arrangement, schemas, tempo

    assert schemas.CONFIDENCE_LABELS == set(typing.get_args(tempo.Confidence))
    assert schemas.STABILITY_LABELS == set(typing.get_args(tempo.StabilityLabel))
    assert schemas.TEMPO_STATUSES == set(typing.get_args(tempo.TempoStatus))
    assert schemas.DOWNBEAT_STATUSES == set(typing.get_args(tempo.DownbeatStatus))
    assert schemas.DOWNBEAT_RESOLVED_BY == set(typing.get_args(tempo.ResolvedBy))
    assert schemas.ARRANGEMENT_STATUSES == set(typing.get_args(arrangement.ArrangementStatus))
    assert schemas.SECTION_LABELS == set(typing.get_args(arrangement.SectionLabel))
    assert schemas.OCTAVE_RATIOS == tempo.OCTAVE_RATIOS


def test_track_level_blocks_are_not_per_source() -> None:
    """One tempo and one structure per record — v4's five disagreed.

    A field appearing on both would let the two drift apart again, which is
    the whole failure v5 exists to close.
    """
    for name in ("tempo", "downbeat", "arrangement"):
        assert name in TrackSummary.model_fields
        assert name not in SourceAnalysis.model_fields


def test_arrangement_sections_survive_into_the_summary() -> None:
    """`sections` is not in the strip table, for `DrumPattern.patterns`' reason.

    It is one entry per section, not per event, and it is the point of the
    feature. `_SUMMARY_LIST_FIELDS` is keyed by `SourceAnalysis` block anyway,
    so nothing track-level can be stripped by accident — this pins that.
    """
    summary = TrackSummary(
        track_name="t",
        input_path="t.wav",
        duration_seconds=1.0,
        backend="fake",
        arrangement=Arrangement(
            sections=[Section(start_bar=0, length_bars=17, active=["drums", "kick"])],
            bar_count=17,
            status="ok",
        ),
    )
    payload = summary.summary_payload()
    assert payload["arrangement"]["sections"][0]["length_bars"] == 17  # type: ignore[index,call-overload]
    assert set(_SUMMARY_LIST_FIELDS) <= set(SourceAnalysis.model_fields)


def test_unclassified_is_a_real_class_not_a_sentinel() -> None:
    """It carries timing and an honest confidence, like any other hit."""
    hit = DrumHit(time_seconds=1.5, drum="unclassified", confidence=0.11, step=6)
    assert hit.drum in DRUM_CLASSES
    assert hit.time_seconds == 1.5
    assert hit.step == 6


def test_no_new_field_uses_a_literal_annotation() -> None:
    """`_fill_value` in test_analyze.py raises on `Literal`; keep it plain `str`.

    Walks every model reachable from `SourceAnalysis` and from `StrudelHints`,
    so this holds for models added later too.
    """
    models = _referenced_models(SourceAnalysis, set()) | _referenced_models(StrudelHints, set())
    for model in models:
        for name, info in model.model_fields.items():
            annotations = [info.annotation, *typing.get_args(info.annotation)]
            for annotation in annotations:
                assert typing.get_origin(annotation) is not typing.Literal, (
                    f"{model.__name__}.{name}"
                )


def test_status_defaults_are_members_of_the_vocabulary() -> None:
    for model in (DrumDecomposition, BassLine, DrumGridHint, BassLineHint):
        assert model().status in BLOCK_STATUSES, model.__name__


def test_no_measurement_float_is_bounded_to_one() -> None:
    """`_PRIMITIVE_FILLERS[float]` can return 1.06, so bounds break fake backends.

    `HeuristicLabel.confidence` is the one exception: it is only ever built by
    `heuristics._emit`, never machine-filled.
    """
    models = _referenced_models(SourceAnalysis, set()) | _referenced_models(StrudelHints, set())
    for model in models:
        if model.__name__ == "HeuristicLabel":
            continue
        for name, info in model.model_fields.items():
            for meta in info.metadata:
                assert not hasattr(meta, "le"), f"{model.__name__}.{name}"
                assert not hasattr(meta, "ge"), f"{model.__name__}.{name}"


def test_no_sound_means_no_match_and_vice_versa() -> None:
    """`match='none'` with `sound=None` is the "source it elsewhere" flag.

    The iff is the machine-readable part: a consumer must be able to test one
    field and know the other. Pinned here so WP-C's mapping cannot half-fill it.
    """
    unavailable = StrudelSoundSuggestion(
        role="bass",
        match="none",
        reason="square vs sawtooth needs an odd/even harmonic ratio nothing measures",
        alternatives=["sawtooth", "square", "triangle"],
    )
    assert unavailable.sound is None
    assert unavailable.match in SOUND_MATCH_TERMS

    matched = StrudelSoundSuggestion(role="kick", match="exact", sound="bd")
    assert (matched.match == "none") == (matched.sound is None)
    assert (unavailable.match == "none") == (unavailable.sound is None)


def test_strudel_hints_carries_the_new_blocks_and_the_vocabulary_date() -> None:
    hints = StrudelHints(track_name="demo")
    assert hints.drum_grid.status == "not_attempted"
    assert hints.bass_line.status == "not_attempted"
    assert hints.sound_suggestions == []
    # Absent until WP-C transcribes the tables, but always a key in the file:
    # a hints file with no date is one built before anyone recorded it.
    assert hints.strudel_vocabulary_read is None
    assert "strudel_vocabulary_read" in hints.model_dump(mode="json")


def test_strudel_hints_key_order_keeps_notes_last() -> None:
    """`notes` is the prose trailer; the structured blocks read before it."""
    order = list(StrudelHints.model_fields)
    assert order[-1] == "notes"
    assert order.index("drum_grid") < order.index("bass_line") < order.index("sound_suggestions")
    assert order.index("tonal_centre") < order.index("drum_grid")


def test_bass_line_hint_truncation_is_recorded() -> None:
    """A capped sequence must say what it was capped from, or it reads as short."""
    hint = BassLineHint(
        status="ok",
        note_sequence=["a1"] * 32,
        truncated_from=91,
        median_midi_note=33,
    )
    assert len(hint.note_sequence) == 32
    assert hint.truncated_from == 91
    assert BassLineHint().truncated_from is None
