"""Tests for backend discovery and resolution.

These cover the seam only. The backends themselves are owned by other agents,
so nothing here asserts anything about their numbers or their implementation
state — only that resolution never crashes the CLI, that failure messages tell
the user what to install, and that neither analysis library gets imported
eagerly.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from audio_pipeline import analyze
from audio_pipeline.backends import (
    BACKEND_PREFERENCE,
    BASS_F0_MAX_HZ,
    BASS_F0_MIN_HZ,
    AnalysisBackend,
    BackendUnavailableError,
    available_backends,
    get_backend,
)
from audio_pipeline.backends.essentia_backend import EssentiaBackend
from audio_pipeline.backends.librosa_backend import LibrosaBackend
from audio_pipeline.schemas import RhythmFeatures


def test_preference_puts_essentia_first() -> None:
    assert BACKEND_PREFERENCE == ("essentia", "librosa")


def test_available_backends_never_raises() -> None:
    """`doctor` calls this on machines where neither library installed."""
    found = available_backends()
    assert set(found) <= set(BACKEND_PREFERENCE)
    assert found == [name for name in BACKEND_PREFERENCE if name in found]


@pytest.mark.parametrize(
    ("backend_class", "expected_name"),
    [(EssentiaBackend, "essentia"), (LibrosaBackend, "librosa")],
)
def test_backend_classes_satisfy_the_protocol(backend_class: type, expected_name: str) -> None:
    """Structure only — deliberately says nothing about implementation state.

    This asserted `NotImplementedError` while both backends were stubs, which
    made it fail the moment W1A implemented `LibrosaBackend`. Whether a method
    is filled in yet is that agent's business; what this file guards is that the
    seam holds: the class is constructible, names itself correctly, and exposes
    every Protocol method.
    """
    backend = backend_class()

    assert isinstance(backend, AnalysisBackend)
    assert backend.name == expected_name
    for method in ("rhythm", "tonal", "spectral", "dynamics", "pitch"):
        assert callable(getattr(backend, method))


def test_importing_the_package_does_not_pull_in_the_analysis_libraries() -> None:
    """The CLI must load with neither library installed, so nothing may import
    them eagerly. Run in a subprocess so an earlier test's import cannot mask it.
    """
    program = (
        "import sys; import audio_pipeline.cli, audio_pipeline.analyze, "
        "audio_pipeline.audio_io, audio_pipeline.backends; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] "
        "in {'essentia', 'librosa'}))"
    )
    src = Path(__file__).resolve().parent.parent / "src"
    env = {**os.environ, "PYTHONPATH": str(src)}
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert result.stdout.strip() == "[]", result.stdout


def test_get_backend_raises_actionable_error_when_nothing_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("audio_pipeline.backends._is_importable", lambda _name: False)

    assert available_backends() == []
    with pytest.raises(BackendUnavailableError) as excinfo:
        get_backend()

    message = str(excinfo.value)
    assert "essentia" in message
    assert "librosa" in message
    assert "pip install" in message


def test_get_backend_does_not_silently_swap_a_requested_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for essentia and getting librosa would make runs incomparable."""
    monkeypatch.setattr(
        "audio_pipeline.backends._is_importable", lambda name: name == "librosa"
    )

    assert available_backends() == ["librosa"]
    assert get_backend().name == "librosa"
    with pytest.raises(BackendUnavailableError):
        get_backend("essentia")


def test_get_backend_prefers_essentia_when_both_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("audio_pipeline.backends._is_importable", lambda _name: True)
    assert get_backend().name == "essentia"


def test_analyze_reexports_backend_api() -> None:
    """Kept so callers predating the backends package split still work."""
    assert analyze.get_backend is get_backend
    assert analyze.available_backends is available_backends
    assert analyze.AnalysisBackend is AnalysisBackend


def test_a_hand_written_fake_backend_satisfies_the_protocol() -> None:
    """The pattern W1F's orchestration tests use: canned values, no audio."""

    class FakeBackend:
        name = "librosa"

        def rhythm(self, audio: np.ndarray, sample_rate: int) -> RhythmFeatures:
            return RhythmFeatures(bpm=120.0)

        def tonal(self, audio: np.ndarray, sample_rate: int) -> object:
            raise NotImplementedError

        def spectral(self, audio: np.ndarray, sample_rate: int) -> object:
            raise NotImplementedError

        def dynamics(self, audio: np.ndarray, sample_rate: int) -> object:
            raise NotImplementedError

        def pitch(self, audio: np.ndarray, sample_rate: int) -> object:
            raise NotImplementedError

    assert isinstance(FakeBackend(), AnalysisBackend)


def test_the_pitch_range_is_a_module_constant_not_a_parameter() -> None:
    """Both backends must analyse the same register, so the range cannot be
    per-call. Narrowing it per backend would reintroduce exactly the divergence
    the shared `note_track` seam exists to remove.
    """
    assert (BASS_F0_MIN_HZ, BASS_F0_MAX_HZ) == (30.0, 400.0)

    signature = inspect.signature(AnalysisBackend.pitch)
    assert list(signature.parameters) == ["self", "audio", "sample_rate"]


def test_both_backends_read_the_pitch_range_from_the_one_constant() -> None:
    """Imported, not re-declared: mirrored numbers are how two backends drift."""
    from audio_pipeline.backends import essentia_backend, librosa_backend

    for module in (essentia_backend, librosa_backend):
        assert module.BASS_F0_MIN_HZ is BASS_F0_MIN_HZ
        assert module.BASS_F0_MAX_HZ is BASS_F0_MAX_HZ


def test_the_protocol_documents_why_drums_are_not_a_protocol_method() -> None:
    """The pitch/drums asymmetry is deliberate and easy to "fix" by mistake.

    Pitch is behind the Protocol because YIN is an algorithm this project should
    not own; drum decomposition is not, because it is arithmetic numpy does
    exactly. If that reasoning ever leaves the docstring, the next reader has
    every reason to make the two symmetrical.
    """
    documentation = AnalysisBackend.__doc__ or ""
    assert "drum" in documentation.lower()
    assert "note_track" in documentation
