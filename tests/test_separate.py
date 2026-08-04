"""Tests for `audio_pipeline.separate`.

No real audio file and no real Demucs model load in the default run: the
Separator/save_audio calls are monkeypatched everywhere except the single
`@pytest.mark.slow` test, which is additionally gated behind an env var so
it can never trigger an accidental multi-hundred-MB weights download.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from audio_pipeline import ANALYSIS_SAMPLE_RATE, STEM_NAMES
from audio_pipeline.separate import (
    DEFAULT_MODEL,
    MODEL_ENV_VAR,
    SeparationResult,
    default_model,
    pick_device,
    separate,
    slugify,
    stem_paths,
)

# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("My Track Name", "my-track-name"),
        ("  leading and trailing spaces  ", "leading-and-trailing-spaces"),
        ("weird___punctuation!!!", "weird-punctuation"),
        ("Already-Slugged-123", "already-slugged-123"),
        ("multiple   internal    spaces", "multiple-internal-spaces"),
        ("--leading-and-trailing-hyphens--", "leading-and-trailing-hyphens"),
        ("Track (Remix) [2024]", "track-remix-2024"),
        ("café résumé naïve", "caf-r-sum-na-ve"),
        ("123", "123"),
    ],
)
def test_slugify_expected(raw: str, expected: str) -> None:
    assert slugify(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "!!!", "---", "🎧🎵", "___"])
def test_slugify_falls_back_when_nothing_usable_remains(raw: str) -> None:
    slug = slugify(raw)
    assert slug != ""
    assert slug == slugify(raw)  # deterministic


def test_slugify_is_deterministic_and_idempotent() -> None:
    raw = "Some Track — Final MIX (v2).wav"
    slug = slugify(raw)
    assert slug == slugify(raw)
    assert slugify(slug) == slug


# ---------------------------------------------------------------------------
# stem_paths
# ---------------------------------------------------------------------------


def test_stem_paths_shape(tmp_path: Path) -> None:
    paths = stem_paths(tmp_path, "my-track")
    assert set(paths.keys()) == set(STEM_NAMES)
    for stem, path in paths.items():
        assert path == tmp_path / "my-track" / "stems" / f"{stem}.wav"


def test_stem_paths_does_no_io(tmp_path: Path) -> None:
    paths = stem_paths(tmp_path, "no-io-track")
    for path in paths.values():
        assert not path.exists()
    assert not (tmp_path / "no-io-track").exists()


# ---------------------------------------------------------------------------
# pick_device
# ---------------------------------------------------------------------------


def test_pick_device_mps_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device() == "mps"


def test_pick_device_mps_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device() == "cpu"


def test_pick_device_mps_attribute_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a torch build with no `mps` backend at all.

    A plain `delattr(torch.backends, "mps")` does not work here: torch's own
    `torch.backends.__getattr__` lazily re-imports and re-attaches known
    submodules (mps included) on access, since the real `mps` submodule is
    still sitting in `sys.modules`. Swapping the whole `backends` namespace
    for a bare object with no such fallback is what actually reproduces
    "the attribute does not exist".
    """
    import types

    import torch

    monkeypatch.setattr(torch, "backends", types.SimpleNamespace())
    assert pick_device() == "cpu"


def test_pick_device_mps_is_available_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    def _boom() -> bool:
        raise RuntimeError("backend probe failure")

    monkeypatch.setattr(torch.backends.mps, "is_available", _boom)
    assert pick_device() == "cpu"


def test_pick_device_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    assert pick_device() == "cpu"


def test_pick_device_never_returns_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device() != "cuda"


# ---------------------------------------------------------------------------
# default_model
# ---------------------------------------------------------------------------


def test_default_model_without_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    assert default_model() == DEFAULT_MODEL


def test_default_model_with_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "htdemucs")
    assert default_model() == "htdemucs"


def test_default_model_with_empty_env_var_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "")
    assert default_model() == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# separate(): idempotent skip path
# ---------------------------------------------------------------------------


class _ExplodingSeparator:
    """Stands in for demucs.api.Separator; constructing it is a test failure."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Separator must not be constructed on the idempotent-skip path")


def test_separate_skips_when_all_stems_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("demucs.api.Separator", _ExplodingSeparator)

    input_path = tmp_path / "input" / "Some Track.wav"
    input_path.parent.mkdir(parents=True)
    input_path.touch()

    output_root = tmp_path / "output"
    track_name = slugify(input_path.stem)
    expected_paths = stem_paths(output_root, track_name)
    for path in expected_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    result = separate(input_path, output_root, model="htdemucs", device="cpu", force=False)

    assert isinstance(result, SeparationResult)
    assert result.skipped is True
    assert result.stems == expected_paths
    assert result.model == "htdemucs"
    assert result.device == "cpu"


def test_separate_force_does_not_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force=True must attempt a real run even if stems exist."""
    calls: list[str] = []

    class _FakeSeparator:
        samplerate = ANALYSIS_SAMPLE_RATE

        def __init__(self, model: str, device: str) -> None:
            calls.append(device)

        def separate_audio_file(self, file: str) -> tuple[object, dict[str, object]]:
            return None, {stem: object() for stem in STEM_NAMES}

    written: list[Path] = []

    def _fake_save_audio(wav: object, path: Path, samplerate: int) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()
        written.append(Path(path))

    monkeypatch.setattr("demucs.api.Separator", _FakeSeparator)
    monkeypatch.setattr("demucs.api.save_audio", _fake_save_audio)

    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_root = tmp_path / "output"
    track_name = slugify(input_path.stem)
    expected_paths = stem_paths(output_root, track_name)
    for path in expected_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    result = separate(input_path, output_root, model="htdemucs", device="cpu", force=True)

    assert result.skipped is False
    assert calls == ["cpu"]
    assert set(written) == set(expected_paths.values())


# ---------------------------------------------------------------------------
# separate(): MPS -> CPU retry path
# ---------------------------------------------------------------------------


def test_separate_retries_on_cpu_after_mps_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class _FlakyMpsSeparator:
        samplerate = ANALYSIS_SAMPLE_RATE

        def __init__(self, model: str, device: str) -> None:
            calls.append(device)

        def separate_audio_file(self, file: str) -> tuple[object, dict[str, object]]:
            if calls[-1] == "mps":
                raise RuntimeError("simulated MPS op gap")
            return None, {stem: object() for stem in STEM_NAMES}

    def _fake_save_audio(wav: object, path: Path, samplerate: int) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    monkeypatch.setattr("demucs.api.Separator", _FlakyMpsSeparator)
    monkeypatch.setattr("demucs.api.save_audio", _fake_save_audio)

    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_root = tmp_path / "output"

    result = separate(input_path, output_root, model="htdemucs", device="mps", force=False)

    assert calls == ["mps", "cpu"]
    assert result.skipped is False
    assert result.device == "cpu"
    assert result.model == "htdemucs"
    for path in result.stems.values():
        assert path.is_file()


def test_separate_reraises_when_cpu_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _AlwaysFailsSeparator:
        samplerate = ANALYSIS_SAMPLE_RATE

        def __init__(self, model: str, device: str) -> None:
            pass

        def separate_audio_file(self, file: str) -> tuple[object, dict[str, object]]:
            raise RuntimeError("boom")

    monkeypatch.setattr("demucs.api.Separator", _AlwaysFailsSeparator)

    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_root = tmp_path / "output"

    with pytest.raises(RuntimeError, match="boom"):
        separate(input_path, output_root, model="htdemucs", device="cpu", force=False)


def test_separate_sets_mps_fallback_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    class _FakeSeparator:
        samplerate = ANALYSIS_SAMPLE_RATE

        def __init__(self, model: str, device: str) -> None:
            pass

        def separate_audio_file(self, file: str) -> tuple[object, dict[str, object]]:
            return None, {stem: object() for stem in STEM_NAMES}

    def _fake_save_audio(wav: object, path: Path, samplerate: int) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    monkeypatch.setattr("demucs.api.Separator", _FakeSeparator)
    monkeypatch.setattr("demucs.api.save_audio", _fake_save_audio)

    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_root = tmp_path / "output"

    separate(input_path, output_root, model="htdemucs", device="cpu", force=False)

    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_separate_rejects_wrong_samplerate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _WrongRateSeparator:
        samplerate = 22050

        def __init__(self, model: str, device: str) -> None:
            pass

        def separate_audio_file(self, file: str) -> tuple[object, dict[str, object]]:
            return None, {stem: object() for stem in STEM_NAMES}

    monkeypatch.setattr("demucs.api.Separator", _WrongRateSeparator)

    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_root = tmp_path / "output"

    with pytest.raises(RuntimeError, match="44100"):
        separate(input_path, output_root, model="htdemucs", device="cpu", force=False)


# ---------------------------------------------------------------------------
# separate(): model/device resolution and reporting
# ---------------------------------------------------------------------------


def test_separate_reports_resolved_model_when_none_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "htdemucs")

    class _FakeSeparator:
        samplerate = ANALYSIS_SAMPLE_RATE

        def __init__(self, model: str, device: str) -> None:
            assert model == "htdemucs"

        def separate_audio_file(self, file: str) -> tuple[object, dict[str, object]]:
            return None, {stem: object() for stem in STEM_NAMES}

    def _fake_save_audio(wav: object, path: Path, samplerate: int) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    monkeypatch.setattr("demucs.api.Separator", _FakeSeparator)
    monkeypatch.setattr("demucs.api.save_audio", _fake_save_audio)

    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_root = tmp_path / "output"

    result = separate(input_path, output_root, model=None, device="cpu", force=False)

    assert result.model == "htdemucs"


# ---------------------------------------------------------------------------
# Real Demucs run: slow, opt-in only, never part of the default suite.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_separation_end_to_end(tmp_path: Path) -> None:
    """Exercises the real Demucs model. Downloads ~300MB of weights on first
    use and takes minutes. Gated behind an explicit env var on top of the
    `slow` marker so this can never run by accident -- e.g. before the
    `-m "not slow"` deselect default is wired into pyproject.toml, or if
    someone runs `pytest -m slow` directly.
    """
    if not os.environ.get("TRACK_FORENSICS_RUN_REAL_SEPARATION"):
        pytest.skip("set TRACK_FORENSICS_RUN_REAL_SEPARATION=1 to run a real Demucs pass")

    import numpy as np
    import soundfile as sf

    input_path = tmp_path / "clip.wav"
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(ANALYSIS_SAMPLE_RATE * duration_s), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype("float32")
    sf.write(input_path, tone, ANALYSIS_SAMPLE_RATE)

    output_root = tmp_path / "output"
    result = separate(input_path, output_root, model="htdemucs", force=False)

    assert not result.skipped
    for path in result.stems.values():
        assert path.is_file()

    # Re-running should now be a no-op.
    result_again = separate(input_path, output_root, model="htdemucs", force=False)
    assert result_again.skipped is True
