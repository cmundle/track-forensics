"""Tests for `TrackSummary.summary_payload()`.

Beat times stay complete in `analysis/<source>.json`; the summary carries only a
count, so the one file you read by hand is not 3,600 duplicated floats.
"""

from __future__ import annotations

import json

from audio_pipeline import SCHEMA_VERSION, SOURCE_NAMES
from audio_pipeline.schemas import RhythmFeatures, SourceAnalysis, TrackSummary


def _source(name: str, beat_times: list[float]) -> SourceAnalysis:
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


def test_payload_contains_no_beat_times_anywhere() -> None:
    payload = _summary().summary_payload()
    assert "beat_times" not in _walk_keys(payload)


def test_beat_count_matches_the_original_list_length() -> None:
    summary = _summary()
    payload = summary.summary_payload()

    sources = payload["sources"]
    assert isinstance(sources, dict)
    for name, source in summary.sources.items():
        assert sources[name]["rhythm"]["beat_count"] == len(source.rhythm.beat_times)

    # Non-trivial: at least one source really did have beats to count.
    assert any(sources[name]["rhythm"]["beat_count"] > 0 for name in summary.sources)


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


def test_beat_count_keeps_the_slot_beat_times_had() -> None:
    """Stable key order keeps run-to-run diffs of the summary readable."""
    payload = _summary().summary_payload()
    sources = payload["sources"]
    assert isinstance(sources, dict)
    assert list(sources["mix"]["rhythm"]) == [
        "bpm",
        "bpm_confidence",
        "beat_count",
        "onset_density",
        "transient_sharpness",
    ]


def test_payload_is_json_serialisable() -> None:
    """It is written straight to disk, so it must not contain numpy or models."""
    text = json.dumps(_summary().summary_payload(), indent=2, sort_keys=False)
    assert "beat_times" not in text
    assert json.loads(text)["track_name"] == "demo"


def test_empty_summary_payload_does_not_raise() -> None:
    summary = TrackSummary(
        track_name="demo",
        input_path="examples/demo.wav",
        duration_seconds=0.0,
        backend="librosa",
    )
    assert summary.summary_payload()["sources"] == {}
