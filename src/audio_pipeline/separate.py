"""Demucs stem separation with Apple-silicon (MPS) preference and CPU fallback.

Writes stems to `output/<track-name>/stems/{drums,bass,vocals,other}.wav`.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL = "htdemucs"


def pick_device() -> str:
    """Return the best available torch device: 'mps' on Apple silicon, else 'cpu'.

    Must not raise if torch is missing an MPS build — fall back silently to 'cpu'.
    """
    raise NotImplementedError


def separate(
    input_path: Path,
    output_root: Path,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Run Demucs and return a mapping of stem name -> written wav path.

    If all stems already exist and `force` is False, skip the run and return the
    existing paths. If the chosen device fails mid-run, retry once on CPU.
    """
    raise NotImplementedError


def stem_paths(output_root: Path, track_name: str) -> dict[str, Path]:
    """Expected stem paths for a track, whether or not they exist yet."""
    raise NotImplementedError
