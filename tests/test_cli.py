"""Tests for the Typer CLI: wiring, exit codes, and message content.

Every pipeline call (`separate.separate`, `analyze.get_backend`/`analyze_track`/
`write_analysis_outputs`, `strudel_hints.build`) is monkeypatched. Nothing here
may trigger a real Demucs run or a real backend analysis -- see the module
docstring in `cli.py` and the ground rules in `CLAUDE.md`.

Assertions target exit codes and message content (what the user would see),
not implementation details of how a command gets there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from audio_pipeline import SCHEMA_VERSION
from audio_pipeline import cli as cli_module
from audio_pipeline.backends import BackendUnavailableError
from audio_pipeline.schemas import RhythmFeatures, SourceAnalysis, StrudelHints, TrackSummary
from audio_pipeline.separate import DEFAULT_MODEL, FAST_MODEL, SeparationResult

runner = CliRunner()


# --- shared fixtures / fakes -------------------------------------------------


class _FakeBackend:
    """Minimal stand-in for a resolved `AnalysisBackend` -- name only, no I/O."""

    def __init__(self, name: str = "librosa") -> None:
        self.name = name


def _make_source_analysis(source: str = "mix", **rhythm_kwargs: Any) -> SourceAnalysis:
    return SourceAnalysis(
        source=source,
        audio_path=f"output/demo/stems/{source}.wav" if source != "mix" else "input.wav",
        duration_seconds=8.0,
        sample_rate=44100,
        backend="librosa",
        rhythm=RhythmFeatures(**rhythm_kwargs),
    )


def _make_written_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "mix": tmp_path / "analysis" / "mix.json",
        "track_summary": tmp_path / "track_summary.json",
    }


def _make_separation_result(**overrides: Any) -> SeparationResult:
    defaults: dict[str, Any] = {
        "stems": {
            "drums": Path("output/demo/stems/drums.wav"),
            "bass": Path("output/demo/stems/bass.wav"),
            "vocals": Path("output/demo/stems/vocals.wav"),
            "other": Path("output/demo/stems/other.wav"),
        },
        "model": DEFAULT_MODEL,
        "device": "cpu",
        "skipped": False,
    }
    defaults.update(overrides)
    return SeparationResult(**defaults)


@pytest.fixture
def input_wav(tmp_path: Path) -> Path:
    path = tmp_path / "demo.wav"
    path.write_bytes(b"not real audio, never decoded in these tests")
    return path


# --- input validation, shared across commands --------------------------------


def test_bad_suffix_exits_1_and_names_accepted_suffixes(tmp_path: Path) -> None:
    bad = tmp_path / "demo.txt"
    bad.write_text("nope")
    result = runner.invoke(cli_module.app, ["analyze", str(bad)])
    assert result.exit_code == 1
    assert ".txt" in result.output
    for suffix in (".wav", ".mp3", ".aiff", ".aif", ".m4a"):
        assert suffix in result.output


def test_missing_file_exits_1(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.wav"
    result = runner.invoke(cli_module.app, ["analyze", str(missing)])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_uppercase_suffix_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "demo.WAV"
    path.write_bytes(b"x")
    monkeypatch.setattr(
        cli_module.analyze_module, "get_backend", lambda *_a, **_k: _FakeBackend()
    )
    monkeypatch.setattr(
        cli_module.analyze_module,
        "analyze_track",
        lambda *_a, **_k: {"mix": _make_source_analysis()},
    )
    monkeypatch.setattr(
        cli_module.analyze_module,
        "write_analysis_outputs",
        lambda *_a, **_k: {"mix": tmp_path / "mix.json"},
    )
    result = runner.invoke(
        cli_module.app, ["analyze", str(path), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 0


# --- separate ------------------------------------------------------------


def test_separate_happy_path(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_separate(input_path: Path, output_root: Path, **kwargs: Any) -> SeparationResult:
        captured.update(kwargs)
        return _make_separation_result()

    monkeypatch.setattr(cli_module.separate_module, "separate", fake_separate)
    result = runner.invoke(
        cli_module.app, ["separate", str(input_wav), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 0
    assert "drums" in result.output
    assert captured["model"] == DEFAULT_MODEL


def test_separate_reports_model_and_device_before_run(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_separate(input_path: Path, output_root: Path, **kwargs: Any) -> SeparationResult:
        calls.append("separate")
        return _make_separation_result()

    monkeypatch.setattr(cli_module.separate_module, "separate", fake_separate)
    monkeypatch.setattr(cli_module.separate_module, "pick_device", lambda: "cpu")
    result = runner.invoke(
        cli_module.app, ["separate", str(input_wav), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 0
    # The model/device line must appear before "separate" was ever called --
    # i.e. before any output that could only follow a completed run.
    model_line_index = result.output.find(f"model={DEFAULT_MODEL!r}")
    stems_line_index = result.output.find("drums:")
    assert model_line_index != -1
    assert stems_line_index != -1
    assert model_line_index < stems_line_index
    assert "cpu" in result.output


def test_separate_bad_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "demo.ogg"
    bad.write_text("nope")
    result = runner.invoke(cli_module.app, ["separate", str(bad)])
    assert result.exit_code == 1
    assert ".ogg" in result.output


def test_separate_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(cli_module.app, ["separate", str(tmp_path / "missing.wav")])
    assert result.exit_code == 1


def test_separate_import_error_is_environment_error(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_separate(*_a: Any, **_k: Any) -> SeparationResult:
        raise ImportError("no module named 'demucs'")

    monkeypatch.setattr(cli_module.separate_module, "separate", fake_separate)
    result = runner.invoke(
        cli_module.app, ["separate", str(input_wav), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 2


# --- --fast / --model resolution, shared behaviour across separate/all ------


def test_fast_resolves_to_fast_model(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_separate(input_path: Path, output_root: Path, **kwargs: Any) -> SeparationResult:
        captured.update(kwargs)
        return _make_separation_result(model=kwargs["model"])

    monkeypatch.setattr(cli_module.separate_module, "separate", fake_separate)
    result = runner.invoke(
        cli_module.app, ["separate", str(input_wav), "--fast", "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 0
    assert captured["model"] == FAST_MODEL


def test_model_flag_overrides_default(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_separate(input_path: Path, output_root: Path, **kwargs: Any) -> SeparationResult:
        captured.update(kwargs)
        return _make_separation_result(model=kwargs["model"])

    monkeypatch.setattr(cli_module.separate_module, "separate", fake_separate)
    result = runner.invoke(
        cli_module.app,
        ["separate", str(input_wav), "--model", "htdemucs_6s", "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0
    assert captured["model"] == "htdemucs_6s"


def test_neither_flag_resolves_via_default_model(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_separate(input_path: Path, output_root: Path, **kwargs: Any) -> SeparationResult:
        captured.update(kwargs)
        return _make_separation_result(model=kwargs["model"])

    monkeypatch.setattr(cli_module.separate_module, "separate", fake_separate)
    result = runner.invoke(
        cli_module.app, ["separate", str(input_wav), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 0
    assert captured["model"] == DEFAULT_MODEL


def test_env_var_is_honoured(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRACK_FORENSICS_MODEL", "htdemucs_6s")
    captured: dict[str, Any] = {}

    def fake_separate(input_path: Path, output_root: Path, **kwargs: Any) -> SeparationResult:
        captured.update(kwargs)
        return _make_separation_result(model=kwargs["model"])

    monkeypatch.setattr(cli_module.separate_module, "separate", fake_separate)
    result = runner.invoke(
        cli_module.app, ["separate", str(input_wav), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 0
    assert captured["model"] == "htdemucs_6s"


def test_model_and_fast_both_given_model_wins(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_separate(input_path: Path, output_root: Path, **kwargs: Any) -> SeparationResult:
        captured.update(kwargs)
        return _make_separation_result(model=kwargs["model"])

    monkeypatch.setattr(cli_module.separate_module, "separate", fake_separate)
    result = runner.invoke(
        cli_module.app,
        [
            "separate",
            str(input_wav),
            "--model",
            "htdemucs_6s",
            "--fast",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0
    assert captured["model"] == "htdemucs_6s"
    # The conflict is surfaced, not silently resolved.
    assert "--fast" in result.output


# --- analyze -----------------------------------------------------------------


def test_analyze_happy_path(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_module.analyze_module, "get_backend", lambda *_a, **_k: _FakeBackend()
    )
    monkeypatch.setattr(
        cli_module.analyze_module,
        "analyze_track",
        lambda *_a, **_k: {"mix": _make_source_analysis()},
    )
    written = _make_written_paths(tmp_path)
    monkeypatch.setattr(
        cli_module.analyze_module, "write_analysis_outputs", lambda *_a, **_k: written
    )

    result = runner.invoke(
        cli_module.app, ["analyze", str(input_wav), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 0
    assert "librosa" in result.output
    for path in written.values():
        assert str(path) in result.output


def test_analyze_backend_unavailable_exits_2(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_unavailable(*_a: Any, **_k: Any) -> Any:
        raise BackendUnavailableError("no backend installed: neither essentia nor librosa")

    monkeypatch.setattr(cli_module.analyze_module, "get_backend", raise_unavailable)
    result = runner.invoke(
        cli_module.app, ["analyze", str(input_wav), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 2
    assert "backend" in result.output.lower()


def test_analyze_unknown_backend_name_exits_1(input_wav: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        cli_module.app,
        [
            "analyze",
            str(input_wav),
            "--backend",
            "not-a-real-backend",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "essentia" in result.output
    assert "librosa" in result.output


# --- export-strudel-hints -----------------------------------------------------


def test_export_hints_missing_summary_names_command_to_run_first(
    input_wav: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        cli_module.app, ["export-strudel-hints", str(input_wav), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 1
    assert "analyze" in result.output or "all" in result.output


def _write_track_summary(
    output_root: Path, track_name: str, *, schema_version: int = SCHEMA_VERSION
) -> Path:
    summary = TrackSummary(
        track_name=track_name,
        input_path="demo.wav",
        duration_seconds=8.0,
        backend="librosa",
        sources={"mix": _make_source_analysis("mix")},
    )
    payload = summary.summary_payload()
    payload["schema_version"] = schema_version
    path = output_root / track_name / "track_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_export_hints_happy_path(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "out"
    track_name = cli_module.separate_module.slugify(input_wav.stem)
    _write_track_summary(output_root, track_name)

    hints = StrudelHints(track_name=track_name, bpm=120.0, notes=["a note"])
    monkeypatch.setattr(cli_module.strudel_hints_module, "build", lambda _summary: hints)

    result = runner.invoke(
        cli_module.app, ["export-strudel-hints", str(input_wav), "--output", str(output_root)]
    )
    assert result.exit_code == 0
    hints_path = output_root / track_name / "strudel_hints.json"
    assert hints_path.is_file()
    assert json.loads(hints_path.read_text())["bpm"] == 120.0
    assert "a note" in result.output


def test_export_hints_schema_version_mismatch_warns_but_proceeds(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "out"
    track_name = cli_module.separate_module.slugify(input_wav.stem)
    _write_track_summary(output_root, track_name, schema_version=SCHEMA_VERSION - 1)

    monkeypatch.setattr(
        cli_module.strudel_hints_module,
        "build",
        lambda _summary: StrudelHints(track_name=track_name),
    )
    result = runner.invoke(
        cli_module.app, ["export-strudel-hints", str(input_wav), "--output", str(output_root)]
    )
    assert result.exit_code == 0
    assert "schema_version" in result.output


def test_export_hints_rehydrates_onset_times_from_analysis_files(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`summary_payload()` strips onset_times/beat_times to counts; this command
    must read them back from the per-source analysis files so
    `infer_subdivision_feel` (which reads onset_times) is not silently
    weakened when run standalone. See `cli._load_track_summary_with_timing`.
    """
    output_root = tmp_path / "out"
    track_name = cli_module.separate_module.slugify(input_wav.stem)
    _write_track_summary(output_root, track_name)

    drums_analysis = _make_source_analysis(
        "drums", onset_times=[0.1, 0.2, 0.3], beat_times=[0.0, 0.5, 1.0]
    )
    drums_path = output_root / track_name / "analysis" / "drums.json"
    drums_path.parent.mkdir(parents=True, exist_ok=True)
    drums_path.write_text(drums_analysis.model_dump_json())

    # track_summary.json only listed "mix", not "drums" -- add drums to the
    # in-memory summary by patching build() to inspect what it is handed and
    # verifying the mix source itself was rehydrated (it has no onset_times in
    # this test, but the mechanism under test is generic: whatever is present
    # in analysis/<source>.json is copied onto the reloaded summary).
    captured: dict[str, Any] = {}

    def fake_build(summary: TrackSummary) -> StrudelHints:
        captured["summary"] = summary
        return StrudelHints(track_name=track_name)

    # Make the summary also carry a "drums" source so rehydration has
    # something to act on.
    summary_path = output_root / track_name / "track_summary.json"
    payload = json.loads(summary_path.read_text())
    mix_payload = payload["sources"]["mix"]
    drums_payload = json.loads(drums_analysis.model_dump_json())
    drums_payload["rhythm"] = {
        "bpm": None,
        "bpm_confidence": None,
        "beat_count": len(drums_analysis.rhythm.beat_times),
        "onset_count": len(drums_analysis.rhythm.onset_times),
        "onset_density": None,
        "transient_sharpness": None,
    }
    payload["sources"]["drums"] = drums_payload
    payload["sources"]["mix"] = mix_payload
    summary_path.write_text(json.dumps(payload))

    monkeypatch.setattr(cli_module.strudel_hints_module, "build", fake_build)
    result = runner.invoke(
        cli_module.app, ["export-strudel-hints", str(input_wav), "--output", str(output_root)]
    )
    assert result.exit_code == 0
    rehydrated_drums = captured["summary"].sources["drums"]
    assert rehydrated_drums.rhythm.onset_times == [0.1, 0.2, 0.3]
    assert rehydrated_drums.rhythm.beat_times == [0.0, 0.5, 1.0]


# --- all -----------------------------------------------------------------


def test_all_happy_path(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_module.analyze_module, "get_backend", lambda *_a, **_k: _FakeBackend()
    )
    monkeypatch.setattr(
        cli_module.separate_module, "separate", lambda *_a, **_k: _make_separation_result()
    )
    monkeypatch.setattr(
        cli_module.analyze_module,
        "analyze_track",
        lambda *_a, **_k: {"mix": _make_source_analysis()},
    )
    written = _make_written_paths(tmp_path)
    monkeypatch.setattr(
        cli_module.analyze_module, "write_analysis_outputs", lambda *_a, **_k: written
    )
    monkeypatch.setattr(
        cli_module.strudel_hints_module,
        "build",
        lambda _summary: StrudelHints(track_name="demo", notes=["heads up"]),
    )

    result = runner.invoke(
        cli_module.app, ["all", str(input_wav), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 0
    assert "[1/3]" in result.output
    assert "[2/3]" in result.output
    assert "[3/3]" in result.output
    assert "heads up" in result.output
    track_name = cli_module.separate_module.slugify(input_wav.stem)
    strudel_path = tmp_path / "out" / track_name / "strudel_hints.json"
    assert strudel_path.is_file()


def test_all_backend_unavailable_exits_2_before_separating(
    input_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_unavailable(*_a: Any, **_k: Any) -> Any:
        raise BackendUnavailableError("no backend installed")

    separate_calls: list[Any] = []

    def fake_separate(*_a: Any, **_k: Any) -> SeparationResult:
        separate_calls.append(1)
        return _make_separation_result()

    monkeypatch.setattr(cli_module.analyze_module, "get_backend", raise_unavailable)
    monkeypatch.setattr(cli_module.separate_module, "separate", fake_separate)

    result = runner.invoke(
        cli_module.app, ["all", str(input_wav), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 2
    assert not separate_calls


def test_all_bad_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "demo.ogg"
    bad.write_text("nope")
    result = runner.invoke(cli_module.app, ["all", str(bad)])
    assert result.exit_code == 1


# --- doctor --------------------------------------------------------------


def test_doctor_exits_0_and_names_backend_and_device() -> None:
    result = runner.invoke(cli_module.app, ["doctor"])
    assert result.exit_code == 0
    assert "Resolved Demucs device:" in result.output
    assert "Active backend:" in result.output or "none available" in result.output


def test_doctor_survives_a_fully_broken_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """`doctor` is the command that runs when everything else is broken, so it
    must not raise no matter which dependency is missing. Everything here is
    actually installed on the dev machine, so the absence is simulated.
    """
    # Simulate torch being uninstalled: assigning None makes `import torch`
    # raise ImportError, exactly like a genuinely missing package.
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    monkeypatch.setattr(
        cli_module.separate_module,
        "pick_device",
        lambda: (_ for _ in ()).throw(RuntimeError("torch is gone")),
    )
    monkeypatch.setattr(
        cli_module.analyze_module,
        "available_backends",
        lambda: (_ for _ in ()).throw(RuntimeError("backend discovery is gone")),
    )
    monkeypatch.setattr(
        cli_module.analyze_module,
        "get_backend",
        lambda *_a, **_k: (_ for _ in ()).throw(BackendUnavailableError("nothing installed")),
    )

    result = runner.invoke(cli_module.app, ["doctor"])

    assert result.exit_code == 0
    assert "torch: NOT installed" in result.output
    assert "Resolved Demucs device:" in result.output
    assert "none available" in result.output
