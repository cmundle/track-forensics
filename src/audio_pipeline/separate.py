"""Demucs stem separation with Apple-silicon (MPS) preference and CPU fallback.

Writes stems to `output/<track-name>/stems/{drums,bass,vocals,other}.wav`.
"""

from __future__ import annotations

from pathlib import Path

# Quality-first default. `htdemucs_ft` is the fine-tuned model: noticeably
# cleaner stems, roughly 4x slower than plain `htdemucs`. Override per-run with
# `--model`, or change the standing default with the TRACK_FORENSICS_MODEL
# environment variable.
DEFAULT_MODEL = "htdemucs_ft"
FAST_MODEL = "htdemucs"
MODEL_ENV_VAR = "TRACK_FORENSICS_MODEL"


def default_model() -> str:
    """Standing model default: TRACK_FORENSICS_MODEL if set, else DEFAULT_MODEL."""
    raise NotImplementedError


def pick_device() -> str:
    """Return the best available torch device: 'mps' on Apple silicon, else 'cpu'.

    Must not raise if torch is missing an MPS build — fall back silently to 'cpu'.
    """
    raise NotImplementedError


def separate(
    input_path: Path,
    output_root: Path,
    model: str | None = None,
    device: str | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Run Demucs and return a mapping of stem name -> written wav path.

    `model=None` resolves via `default_model()`. If all stems already exist and
    `force` is False, skip the run and return the existing paths. If the chosen
    device fails mid-run, retry once on CPU.

    Stems are written at 44.1 kHz. Never downsample — analysis accuracy on hats
    and cymbals depends on the full band.
    """
    raise NotImplementedError


def stem_paths(output_root: Path, track_name: str) -> dict[str, Path]:
    """Expected stem paths for a track, whether or not they exist yet."""
    raise NotImplementedError
