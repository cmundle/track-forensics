"""Tests for `TrackSummary.summary_payload()`.

Beat and onset times stay complete in `analysis/<source>.json`; the summary
carries only counts, so the one file you read by hand is not thousands of
duplicated floats. Onset lists are the larger of the two — a busy drum stem at
6 onsets/sec over five minutes is ~1,800 floats on its own.
"""

from __future__ import annotations

import json

from audio_pipeline import SCHEMA_VERSION, SOURCE_NAMES
from audio_pipeline.schemas import RhythmFeatures, SourceAnalysis, TrackSummary

STRIPPED_LIST_FIELDS = ("beat_times", "onset_times")
COUNT_FIELDS = ("beat_count", "onset_count")


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
    """onset_times was added at schema version 3."""
    assert SCHEMA_VERSION >= 3
    assert _summary().summary_payload()["schema_version"] == SCHEMA_VERSION


def test_empty_summary_payload_does_not_raise() -> None:
    summary = TrackSummary(
        track_name="demo",
        input_path="examples/demo.wav",
        duration_seconds=0.0,
        backend="librosa",
    )
    assert summary.summary_payload()["sources"] == {}
