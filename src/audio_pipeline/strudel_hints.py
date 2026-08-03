"""Condense a full analysis into a compact, hand-readable hints file.

v1 emits descriptive hints only. It does NOT generate Strudel code — the point
is to give you something to read while writing patterns yourself.
"""

from __future__ import annotations

from .schemas import StrudelHints, TrackSummary


def infer_subdivision_feel(summary: TrackSummary) -> str | None:
    """Guess the rhythmic grid, e.g. 'straight 16ths' or 'swung 8ths'.

    Derive from inter-onset interval distribution on the drums stem relative to
    the beat grid. Return None when confidence is low — a wrong guess is worse
    than no guess.
    """
    raise NotImplementedError


def classify_density(onset_density: float | None) -> str | None:
    """Map onsets/sec to 'sparse' | 'moderate' | 'busy'."""
    raise NotImplementedError


def suggest_cycle_seconds(bpm: float | None, beats_per_cycle: int = 4) -> float | None:
    """Strudel cycle length in seconds for the detected tempo."""
    raise NotImplementedError


def build(summary: TrackSummary) -> StrudelHints:
    """Assemble the hints object from a completed track summary."""
    raise NotImplementedError
