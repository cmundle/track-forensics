"""Turn raw descriptors into human-readable production clues.

Pure functions only: no audio, no I/O, no backend imports. This is the layer
that is cheap to unit test and easy to tune, so keep it that way.
"""

from __future__ import annotations

from .schemas import HeuristicLabel, SourceAnalysis

# Tune these against real material; they are placeholders.
THRESHOLDS: dict[str, float] = {
    "busy_drums_onsets_per_sec": 4.0,
    "sparse_onsets_per_sec": 1.0,
    "bright_centroid_hz": 4000.0,
    "dark_centroid_hz": 800.0,
    "sustained_crest_factor": 4.0,
    "percussive_crest_factor": 10.0,
    "tonally_stable_confidence": 0.7,
}


def label_drums(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """e.g. 'busy drums', 'kick-heavy', 'bright hats', 'sparse percussion'."""
    raise NotImplementedError


def label_bass(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """e.g. 'sparse bass', 'sustained sub', 'plucked/transient bass'."""
    raise NotImplementedError


def label_vocals(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """e.g. 'speech/vocal dominant', 'sparse vocal', 'processed/wide vocal'."""
    raise NotImplementedError


def label_other(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """e.g. 'sustained pad-like texture', 'noisy', 'bright plucks'."""
    raise NotImplementedError


def label_generic(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """Source-independent labels: percussive, sustained, noisy, tonally stable."""
    raise NotImplementedError


def apply(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """Dispatch to the right labeller(s) for `analysis.source` and merge results."""
    raise NotImplementedError
