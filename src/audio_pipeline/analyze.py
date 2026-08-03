"""Backend-agnostic feature extraction over the mix and each stem.

Essentia is preferred; librosa is the fallback when Essentia will not install.
Neither is imported at module top level — resolve the backend lazily so the CLI
always loads, and `doctor` can report what is actually available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .schemas import (
    DynamicsFeatures,
    RhythmFeatures,
    SourceAnalysis,
    SpectralFeatures,
    TonalFeatures,
)

BackendName = Literal["essentia", "librosa"]


class AnalysisBackend(Protocol):
    """Interface every analysis backend must satisfy.

    A backend returns `None` for any descriptor it cannot compute; the caller
    records those in `SourceAnalysis.unavailable_features` rather than crashing.
    """

    name: BackendName

    def rhythm(self, audio: np.ndarray, sample_rate: int) -> RhythmFeatures: ...
    def tonal(self, audio: np.ndarray, sample_rate: int) -> TonalFeatures: ...
    def spectral(self, audio: np.ndarray, sample_rate: int) -> SpectralFeatures: ...
    def dynamics(self, audio: np.ndarray, sample_rate: int) -> DynamicsFeatures: ...


def available_backends() -> list[BackendName]:
    """Which analysis backends are importable right now. Used by `doctor`."""
    raise NotImplementedError


def get_backend(preferred: BackendName | None = None) -> AnalysisBackend:
    """Resolve a backend, defaulting to Essentia and falling back to librosa."""
    raise NotImplementedError


def load_audio(path: Path, sample_rate: int = 44100, mono: bool = True) -> tuple[np.ndarray, int]:
    """Decode any FFmpeg-readable file to a float32 numpy array."""
    raise NotImplementedError


def analyze_source(
    path: Path,
    source: str,
    backend: AnalysisBackend | None = None,
) -> SourceAnalysis:
    """Analyze one audio file and attach heuristic labels."""
    raise NotImplementedError


def analyze_track(
    input_path: Path,
    stems: dict[str, Path],
    backend: AnalysisBackend | None = None,
) -> dict[str, SourceAnalysis]:
    """Analyze the mix plus every stem. Keys: mix, drums, bass, vocals, other."""
    raise NotImplementedError
