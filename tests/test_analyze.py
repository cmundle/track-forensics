"""Tests for analysis orchestration: `analyze_source`, `analyze_track`, and the
JSON-writing helpers.

Everything here uses a hand-written fake backend implementing the
`AnalysisBackend` Protocol with canned values -- no real audio decoding of
interest, no real essentia, no real librosa (`tests/conftest.py`'s synthetic
fixtures plus its `wav_file` factory supply just enough real audio to exercise
`audio_io.load_audio`). This is what makes the orchestration layer testable in
isolation from the backends, which are owned by other in-flight agents.

Several tests deliberately do *not* monkeypatch `heuristics.apply` -- at the
time this file was written, `heuristics.py` (owned by W1D) still raises
`NotImplementedError` unconditionally. Exercising that real stub and asserting
the analysis survives it is the point: this package must not depend on W1D
being finished.

Full-coverage fixtures (`_full_rhythm`, `_full_tonal`, etc.) are built through
`_fill_model`, not by listing every field as a keyword by hand. The first
version of this file *did* list fields by hand, and it broke in five places
the moment `onset_times` was added to `RhythmFeatures` -- a schema change in a
file this module does not own, landed by a concurrent agent, that these fakes
had no way to see coming. `_fill_model` walks `model_cls.model_fields` and
synthesizes a concrete value for anything not given explicitly, so a *new*
optional field silently gets a real value instead of being left `None`/empty
and silently reported as unavailable. A genuinely novel field *type* (not
covered by `_PRIMITIVE_FILLERS`) still fails, but once, at `_fill_value`, with
a message saying exactly what to add -- not as a confusing
`unavailable_features` mismatch in five unrelated tests.
"""

from __future__ import annotations

import json
import logging
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest
from pydantic import BaseModel

from audio_pipeline import analyze
from audio_pipeline.backends import AnalysisBackend
from audio_pipeline.schemas import (
    BandEnergyRatios,
    DynamicsFeatures,
    HeuristicLabel,
    PitchTrack,
    RhythmFeatures,
    SourceAnalysis,
    SpectralFeatures,
    TonalFeatures,
    TrackSummary,
)

_M = TypeVar("_M", bound=BaseModel)

#: One synthesized value per primitive field type, keyed by the field name so
#: different fields of the same type still get distinguishable values. Add an
#: entry here (not to every fixture) when a genuinely new field *type* shows
#: up in a feature model -- see the module docstring.
_PRIMITIVE_FILLERS: dict[type, Callable[[str], object]] = {
    float: lambda name: round(0.1 + (abs(hash(name)) % 97) / 100, 6),
    int: lambda name: 1 + (abs(hash(name)) % 5),
    str: lambda name: f"{name}-value",
    bool: lambda name: True,
}


def _unwrap_optional(annotation: object) -> object:
    """Strip a `X | None` union down to `X`; leaves other annotations alone."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _fill_value(field_name: str, annotation: object) -> object:
    """Synthesize a concrete, non-`None`/non-empty value for one field's type."""
    annotation = _unwrap_optional(annotation)
    origin = typing.get_origin(annotation)

    if origin is list:
        (item_type,) = typing.get_args(annotation) or (float,)
        return [_fill_value(field_name, item_type) for _ in range(3)]

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _fill_model(annotation)

    if isinstance(annotation, type):
        filler = _PRIMITIVE_FILLERS.get(annotation)
        if filler is not None:
            return filler(field_name)

    raise TypeError(
        f"_fill_value has no filler for {field_name!r}: {annotation!r}. Add one to "
        "_PRIMITIVE_FILLERS (or a new branch here) -- that is the one place a new "
        "schema field type should need a test update, instead of every hand-built "
        "fake in this file."
    )


def _fill_model(model_cls: type[_M], **overrides: object) -> _M:
    """Build `model_cls` with every field populated by construction.

    Fields named in `overrides` get exactly that value (used for the fields a
    given test actually asserts on); every other field -- including any added
    to the model after this helper was written -- gets a synthesized,
    type-appropriate, non-`None`/non-empty value from `_fill_value`. See the
    module docstring for why this replaced explicit per-field fixtures.
    """
    values: dict[str, object] = dict(overrides)
    for name, info in model_cls.model_fields.items():
        if name in values:
            continue
        values[name] = _fill_value(name, info.annotation)
    return model_cls(**values)


def _full_rhythm(**overrides: object) -> RhythmFeatures:
    return _fill_model(
        RhythmFeatures,
        bpm=120.123456789,
        bpm_confidence=0.9,
        beat_times=[0.25, 0.75, 1.25],
        onset_density=2.0,
        transient_sharpness=3.5,
        **overrides,
    )


def _full_tonal(**overrides: object) -> TonalFeatures:
    return _fill_model(
        TonalFeatures,
        key="A",
        scale="major",
        key_confidence=0.8,
        hpcp_mean=[0.1] * 12,
        tonal_stability=0.9,
        **overrides,
    )


def _full_spectral(**overrides: object) -> SpectralFeatures:
    return _fill_model(
        SpectralFeatures,
        centroid_mean=2000.123456789,
        centroid_std=50.0,
        rolloff_mean=5000.0,
        brightness=0.3,
        band_energy_ratios=BandEnergyRatios(low=0.25, low_mid=0.35, high_mid=0.25, high=0.15),
        **overrides,
    )


def _full_dynamics(**overrides: object) -> DynamicsFeatures:
    return _fill_model(
        DynamicsFeatures,
        loudness_lufs=-14.0,
        rms_mean=0.1,
        rms_std=0.02,
        crest_factor=3.0,
        **overrides,
    )


@dataclass
class FakeBackend:
    """Canned-value stand-in for `AnalysisBackend`. See module docstring."""

    name: str = "fake"
    rhythm_result: RhythmFeatures | Exception = field(default_factory=RhythmFeatures)
    tonal_result: TonalFeatures | Exception = field(default_factory=TonalFeatures)
    spectral_result: SpectralFeatures | Exception = field(default_factory=SpectralFeatures)
    dynamics_result: DynamicsFeatures | Exception = field(default_factory=DynamicsFeatures)
    # `pitch()` joined the Protocol in Wave 4. Unlike the other four it is not
    # called for every source -- only the bass stem gets a note track -- so its
    # default is an empty `PitchTrack`, which is exactly what a backend returns
    # for silence and what `note_track.segment_notes()` reports as `unvoiced`.
    pitch_result: PitchTrack | Exception = field(default_factory=PitchTrack)

    def rhythm(self, audio: np.ndarray, sample_rate: int) -> RhythmFeatures:
        if isinstance(self.rhythm_result, Exception):
            raise self.rhythm_result
        return self.rhythm_result

    def tonal(self, audio: np.ndarray, sample_rate: int) -> TonalFeatures:
        if isinstance(self.tonal_result, Exception):
            raise self.tonal_result
        return self.tonal_result

    def spectral(self, audio: np.ndarray, sample_rate: int) -> SpectralFeatures:
        if isinstance(self.spectral_result, Exception):
            raise self.spectral_result
        return self.spectral_result

    def dynamics(self, audio: np.ndarray, sample_rate: int) -> DynamicsFeatures:
        if isinstance(self.dynamics_result, Exception):
            raise self.dynamics_result
        return self.dynamics_result

    def pitch(self, audio: np.ndarray, sample_rate: int) -> PitchTrack:
        if isinstance(self.pitch_result, Exception):
            raise self.pitch_result
        return self.pitch_result


def _full_backend() -> FakeBackend:
    return FakeBackend(
        rhythm_result=_full_rhythm(),
        tonal_result=_full_tonal(),
        spectral_result=_full_spectral(),
        dynamics_result=_full_dynamics(),
    )


@pytest.fixture
def click_wav(
    click_track_120bpm: np.ndarray, wav_file: Callable[..., Path]
) -> Path:
    """A short real wav file so `load_audio` has something to decode."""
    return wav_file(click_track_120bpm, name="mix.wav")


def test_fake_backend_satisfies_the_protocol() -> None:
    assert isinstance(_full_backend(), AnalysisBackend)


# --- analyze_source -----------------------------------------------------


def test_full_backend_produces_complete_analysis(click_wav: Path) -> None:
    analysis = analyze.analyze_source(click_wav, "mix", _full_backend())

    assert analysis.source == "mix"
    assert analysis.backend == "fake"
    assert analysis.sample_rate == 44100
    assert analysis.duration_seconds == pytest.approx(8.0, abs=0.01)
    assert analysis.rhythm.bpm == pytest.approx(120.123456789)
    assert analysis.tonal.key == "A"
    assert analysis.spectral.band_energy_ratios.low == 0.25
    assert analysis.dynamics.loudness_lufs == -14.0
    assert analysis.unavailable_features == []


def test_none_descriptors_populate_unavailable_features(click_wav: Path) -> None:
    backend = FakeBackend(
        rhythm_result=RhythmFeatures(bpm=120.0),  # everything else left None/empty
        tonal_result=_full_tonal(),
        spectral_result=_full_spectral(),
        dynamics_result=_full_dynamics(),
    )
    analysis = analyze.analyze_source(click_wav, "mix", backend)

    assert "rhythm.bpm_confidence" in analysis.unavailable_features
    assert "rhythm.beat_times" in analysis.unavailable_features
    assert "rhythm.onset_density" in analysis.unavailable_features
    assert "rhythm.transient_sharpness" in analysis.unavailable_features
    # bpm was supplied, so it must not be reported as unavailable.
    assert "rhythm.bpm" not in analysis.unavailable_features
    # Other categories were fully populated.
    assert not any(f.startswith("tonal.") for f in analysis.unavailable_features)


def test_nested_band_energy_ratios_none_are_reported_with_dotted_path(
    click_wav: Path,
) -> None:
    backend = FakeBackend(
        rhythm_result=_full_rhythm(),
        tonal_result=_full_tonal(),
        spectral_result=SpectralFeatures(centroid_mean=1000.0),  # ratios left default (all None)
        dynamics_result=_full_dynamics(),
    )
    analysis = analyze.analyze_source(click_wav, "mix", backend)

    for band in ("low", "low_mid", "high_mid", "high"):
        assert f"spectral.band_energy_ratios.{band}" in analysis.unavailable_features


def test_one_method_raising_still_yields_the_other_three(
    click_wav: Path, caplog: pytest.LogCaptureFixture
) -> None:
    backend = FakeBackend(
        rhythm_result=_full_rhythm(),
        tonal_result=RuntimeError("tonal blew up"),
        spectral_result=_full_spectral(),
        dynamics_result=_full_dynamics(),
    )
    with caplog.at_level(logging.WARNING, logger="audio_pipeline.analyze"):
        analysis = analyze.analyze_source(click_wav, "mix", backend)

    # rhythm, spectral, dynamics came through untouched.
    assert analysis.rhythm.bpm == pytest.approx(120.123456789)
    assert analysis.spectral.centroid_mean == pytest.approx(2000.123456789)
    assert analysis.dynamics.loudness_lufs == -14.0

    # tonal degraded to all-None and every one of its fields is recorded.
    assert analysis.tonal == TonalFeatures()
    assert "tonal.key" in analysis.unavailable_features
    assert "tonal.scale" in analysis.unavailable_features
    assert "tonal.key_confidence" in analysis.unavailable_features
    assert "tonal.hpcp_mean" in analysis.unavailable_features
    assert "tonal.tonal_stability" in analysis.unavailable_features

    assert any("tonal" in record.message for record in caplog.records)
    assert not any(f.startswith("rhythm.") for f in analysis.unavailable_features)


def test_labelling_failure_does_not_lose_the_analysis(
    click_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising `heuristics.apply` (e.g. W1D's stub mid-implementation, or any
    future bug in it) must degrade to an empty label list rather than losing
    the rest of the analysis. Monkeypatched rather than relying on the real
    `heuristics.apply` being unfinished, so this test holds regardless of
    W1D's progress.
    """
    monkeypatch.setattr(
        analyze.heuristics_module,
        "apply",
        lambda analysis: (_ for _ in ()).throw(NotImplementedError("still a stub")),
    )

    analysis = analyze.analyze_source(click_wav, "mix", _full_backend())

    assert analysis.labels == []
    # The rest of the analysis is untouched by the labelling failure.
    assert analysis.rhythm.bpm == pytest.approx(120.123456789)
    assert analysis.unavailable_features == []


def test_labels_are_attached_when_heuristics_succeeds(
    click_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned = [HeuristicLabel(label="percussive", confidence=0.8, evidence={"crest": 12.0})]
    monkeypatch.setattr(analyze.heuristics_module, "apply", lambda analysis: canned)

    analysis = analyze.analyze_source(click_wav, "mix", _full_backend())

    assert analysis.labels == canned


def test_analyze_source_defaults_to_get_backend(
    click_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`backend=None` should resolve via `get_backend()`, not crash."""
    monkeypatch.setattr(analyze, "get_backend", lambda: _full_backend())
    analysis = analyze.analyze_source(click_wav, "mix")
    assert analysis.backend == "fake"


# --- analyze_track --------------------------------------------------------


def test_analyze_track_skips_missing_stems_with_a_warning(
    click_wav: Path,
    click_track_120bpm: np.ndarray,
    wav_file: Callable[..., Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    drums_path = wav_file(click_track_120bpm, name="drums.wav")
    bass_path = wav_file(click_track_120bpm, name="bass.wav")
    missing_vocals = drums_path.parent / "vocals.wav"  # never written
    stems = {
        "drums": drums_path,
        "bass": bass_path,
        "vocals": missing_vocals,
        # "other" entirely absent from the dict
    }

    with caplog.at_level(logging.WARNING, logger="audio_pipeline.analyze"):
        results = analyze.analyze_track(click_wav, stems, _full_backend())

    assert set(results) == {"mix", "drums", "bass"}
    assert "vocals" not in results
    assert "other" not in results
    warnings = [r.message for r in caplog.records if "Skipping stem" in r.message]
    assert any("vocals" in message for message in warnings)
    assert any("other" in message for message in warnings)


def test_analyze_track_analyzes_mix_plus_all_present_stems(
    click_wav: Path,
    click_track_120bpm: np.ndarray,
    wav_file: Callable[..., Path],
) -> None:
    stems = {
        name: wav_file(click_track_120bpm, name=f"{name}.wav")
        for name in ("drums", "bass", "vocals", "other")
    }
    results = analyze.analyze_track(click_wav, stems, _full_backend())

    assert set(results) == {"mix", "drums", "bass", "vocals", "other"}
    for source_name, analysis in results.items():
        assert analysis.source == source_name
        assert analysis.unavailable_features == []


def test_one_source_raising_does_not_prevent_the_others_from_being_analyzed(
    click_wav: Path,
    click_track_120bpm: np.ndarray,
    wav_file: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stems = {
        name: wav_file(click_track_120bpm, name=f"{name}.wav")
        for name in ("drums", "bass", "vocals", "other")
    }

    real_analyze_source = analyze.analyze_source

    def flaky(
        path: Path, source: str, backend: AnalysisBackend | None = None
    ) -> SourceAnalysis:
        if source == "bass":
            raise RuntimeError("simulated corrupt stem")
        return real_analyze_source(path, source, backend)

    monkeypatch.setattr(analyze, "analyze_source", flaky)

    with caplog.at_level(logging.ERROR, logger="audio_pipeline.analyze"):
        results = analyze.analyze_track(click_wav, stems, _full_backend())

    # All five sources are present -- the failing one as a placeholder.
    assert set(results) == {"mix", "drums", "bass", "vocals", "other"}

    bass = results["bass"]
    assert bass.unavailable_features  # everything is unavailable
    assert any("simulated corrupt stem" in f for f in bass.unavailable_features)
    assert bass.labels == []

    # The others analyzed normally and were not affected.
    assert results["mix"].unavailable_features == []
    assert results["drums"].unavailable_features == []
    assert results["vocals"].unavailable_features == []
    assert results["other"].unavailable_features == []

    assert any("bass" in record.message for record in caplog.records)


def test_analyze_track_resolution_failure_is_not_swallowed(
    click_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backend resolution failure is an environment error, distinct from a
    per-source failure, and must propagate rather than being caught here.
    """
    from audio_pipeline.backends import BackendUnavailableError

    def boom() -> AnalysisBackend:
        raise BackendUnavailableError("nothing installed")

    monkeypatch.setattr(analyze, "get_backend", boom)

    with pytest.raises(BackendUnavailableError):
        analyze.analyze_track(click_wav, {})


# --- JSON writing ----------------------------------------------------------


def _summary_with_precise_floats(track_name: str = "demo") -> TrackSummary:
    mix = SourceAnalysis(
        source="mix",
        audio_path="output/demo/demo.wav",
        duration_seconds=12.3456789,
        sample_rate=44100,
        backend="fake",
        rhythm=RhythmFeatures(
            bpm=120.0000004,
            bpm_confidence=0.9,
            beat_times=[0.1111111, 0.2222222],
            onset_times=[0.05555555, 0.16666666, 0.27777777],
            onset_density=2.0,
            transient_sharpness=3.5,
        ),
        spectral=SpectralFeatures(
            centroid_mean=1234.56789012,
            band_energy_ratios=BandEnergyRatios(low=0.123456789),
        ),
        labels=[HeuristicLabel(label="percussive", confidence=0.87654321, evidence={"x": 1.0})],
    )
    return TrackSummary(
        track_name=track_name,
        input_path="examples/demo.wav",
        duration_seconds=12.3456789,
        backend="fake",
        separation_model="htdemucs_ft",
        separation_device="cpu",
        sources={"mix": mix},
    )


def test_write_source_analysis_rounds_floats_and_round_trips(tmp_path: Path) -> None:
    summary = _summary_with_precise_floats()
    mix = summary.sources["mix"]

    path = analyze.write_source_analysis(mix, tmp_path, summary.track_name)

    assert path == tmp_path / "demo" / "analysis" / "mix.json"
    raw = json.loads(path.read_text())
    assert raw["rhythm"]["bpm"] == round(120.0000004, 6)
    assert raw["rhythm"]["beat_times"] == [round(0.1111111, 6), round(0.2222222, 6)]
    assert raw["rhythm"]["onset_times"] == [
        round(0.05555555, 6),
        round(0.16666666, 6),
        round(0.27777777, 6),
    ]
    assert raw["spectral"]["centroid_mean"] == round(1234.56789012, 6)
    assert raw["spectral"]["band_energy_ratios"]["low"] == round(0.123456789, 6)
    assert raw["labels"][0]["confidence"] == round(0.87654321, 6)

    # Round-trips back through the model.
    round_tripped = SourceAnalysis.model_validate_json(path.read_text())
    assert round_tripped.source == "mix"
    assert round_tripped.rhythm.beat_times == [round(0.1111111, 6), round(0.2222222, 6)]
    assert round_tripped.rhythm.onset_times == [
        round(0.05555555, 6),
        round(0.16666666, 6),
        round(0.27777777, 6),
    ]


def test_write_source_analysis_key_order_is_stable_across_runs(tmp_path: Path) -> None:
    summary = _summary_with_precise_floats()
    mix = summary.sources["mix"]

    path_a = analyze.write_source_analysis(mix, tmp_path / "run_a", summary.track_name)
    path_b = analyze.write_source_analysis(mix, tmp_path / "run_b", summary.track_name)

    assert path_a.read_text() == path_b.read_text()


def test_write_track_summary_has_no_beat_or_onset_times_and_matches_counts(
    tmp_path: Path,
) -> None:
    """`onset_times` lists run several times longer than `beat_times` on a busy
    drum stem, so stripping them from the summary (in favour of `onset_count`)
    matters more than the beat-time stripping this test used to check alone.
    """
    summary = _summary_with_precise_floats()
    path = analyze.write_track_summary(summary, tmp_path)

    assert path == tmp_path / "demo" / "track_summary.json"
    text = path.read_text()
    assert "beat_times" not in text
    assert "onset_times" not in text

    raw = json.loads(text)
    rhythm = raw["sources"]["mix"]["rhythm"]
    mix_rhythm = summary.sources["mix"].rhythm
    assert rhythm["beat_count"] == len(mix_rhythm.beat_times)
    assert rhythm["onset_count"] == len(mix_rhythm.onset_times)


def test_write_analysis_outputs_writes_every_source_plus_summary(tmp_path: Path) -> None:
    summary = _summary_with_precise_floats()
    written = analyze.write_analysis_outputs(summary, tmp_path)

    assert set(written) == {"mix", "track_summary"}
    assert written["mix"].is_file()
    assert written["track_summary"].is_file()
    assert written["mix"] == tmp_path / "demo" / "analysis" / "mix.json"
    assert written["track_summary"] == tmp_path / "demo" / "track_summary.json"


def test_round_floats_recurses_through_nested_structures() -> None:
    value = {
        "a": 1.123456789,
        "b": [1.987654321, {"c": 2.5555555}],
        "d": "not a float",
        "e": None,
        "f": True,
    }
    result = analyze.round_floats(value)
    assert result == {
        "a": round(1.123456789, 6),
        "b": [round(1.987654321, 6), {"c": round(2.5555555, 6)}],
        "d": "not a float",
        "e": None,
        "f": True,
    }


def test_analysis_output_dir_and_paths_are_rooted_under_output_root_and_track_name(
    tmp_path: Path,
) -> None:
    assert analyze.analysis_output_dir(tmp_path, "demo") == tmp_path / "demo" / "analysis"
    assert (
        analyze.source_analysis_path(tmp_path, "demo", "drums")
        == tmp_path / "demo" / "analysis" / "drums.json"
    )
    assert analyze.track_summary_path(tmp_path, "demo") == tmp_path / "demo" / "track_summary.json"


# --- Re-exports (CLI depends on these staying importable from `analyze`) ---


def test_backend_api_is_still_reexported_from_analyze() -> None:
    from audio_pipeline import backends

    assert analyze.get_backend is backends.get_backend
    assert analyze.available_backends is backends.available_backends
    assert analyze.AnalysisBackend is backends.AnalysisBackend
    assert analyze.BackendUnavailableError is backends.BackendUnavailableError
