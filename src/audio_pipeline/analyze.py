"""Analysis orchestration: resolve a backend, loop over sources, write JSON.

Feature extraction itself lives in `backends/`; decoding lives in `audio_io`.
This module only sequences them and assembles `SourceAnalysis` models.

`AnalysisBackend`, `BackendName`, `available_backends`, and `get_backend` are
re-exported here for callers that predate the `backends` package split.
"""

from __future__ import annotations

from pathlib import Path

from .backends import (
    AnalysisBackend,
    BackendName,
    BackendUnavailableError,
    available_backends,
    get_backend,
)
from .schemas import SourceAnalysis

__all__ = [
    "AnalysisBackend",
    "BackendName",
    "BackendUnavailableError",
    "analyze_source",
    "analyze_track",
    "available_backends",
    "get_backend",
]


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
