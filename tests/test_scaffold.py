"""Scaffold-level tests. Replace with real coverage as modules get implemented.

Tests must never require a real audio file — generate synthetic signals instead.
"""

from __future__ import annotations

import numpy as np

from audio_pipeline import SOURCE_NAMES, STEM_NAMES
from audio_pipeline.schemas import SourceAnalysis, StrudelHints, TrackSummary


def test_source_names() -> None:
    assert SOURCE_NAMES == ("mix", *STEM_NAMES)
    assert STEM_NAMES == ("drums", "bass", "vocals", "other")


def test_source_analysis_defaults_are_serialisable() -> None:
    analysis = SourceAnalysis(
        source="drums",
        audio_path="output/demo/stems/drums.wav",
        duration_seconds=12.0,
        sample_rate=44100,
        backend="librosa",
    )
    payload = analysis.model_dump()
    assert payload["schema_version"] == 1
    assert payload["rhythm"]["bpm"] is None
    assert payload["unavailable_features"] == []


def test_track_summary_and_hints_roundtrip() -> None:
    summary = TrackSummary(
        track_name="demo",
        input_path="examples/demo.wav",
        duration_seconds=12.0,
        backend="librosa",
    )
    assert TrackSummary.model_validate_json(summary.model_dump_json()) == summary

    hints = StrudelHints(track_name="demo")
    assert StrudelHints.model_validate_json(hints.model_dump_json()) == hints


def test_synthetic_signal_fixture_shape() -> None:
    """Sanity check the pattern real tests should use for audio input."""
    sample_rate = 44100
    t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    assert tone.shape == (sample_rate,)
    assert tone.dtype == np.float32
