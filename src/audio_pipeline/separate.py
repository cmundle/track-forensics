"""Demucs stem separation with Apple-silicon (MPS) preference and CPU fallback.

Writes stems to `output/<track-name>/stems/{drums,bass,vocals,other}.wav`.

`torch` and `demucs` are heavyweight, sometimes-broken dependencies (MPS
support in particular varies by torch build). Neither is imported at module
top level: every function that needs them imports lazily, inside its own
body, so `track-forensics doctor` and the rest of the CLI keep working even
on a machine where torch/demucs are missing or broken.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import ANALYSIS_SAMPLE_RATE, STEM_NAMES

if TYPE_CHECKING:
    # Only for static typing -- stripped at runtime by `from __future__ import
    # annotations`, so this does not violate the "no torch at module top
    # level" rule. Every real (runtime) torch/demucs import stays inside a
    # function body below.
    import torch

logger = logging.getLogger(__name__)

# Quality-first default. `htdemucs_ft` is the fine-tuned model: noticeably
# cleaner stems, roughly 4x slower than plain `htdemucs`. Override per-run with
# `--model`, or change the standing default with the TRACK_FORENSICS_MODEL
# environment variable.
DEFAULT_MODEL = "htdemucs_ft"
FAST_MODEL = "htdemucs"
MODEL_ENV_VAR = "TRACK_FORENSICS_MODEL"

_SLUG_DISALLOWED_RE = re.compile(r"[^a-z0-9]+")
_SLUG_FALLBACK = "track"


@dataclass(frozen=True)
class SeparationResult:
    """Outcome of a `separate()` call.

    `separate()`'s stub signature declared a plain `dict[str, Path]` return
    type. That is not enough information for `TrackSummary.separation_model`
    / `separation_device` to be accurate, since the model/device actually
    used can differ from what was requested (env override, CPU fallback
    after an MPS failure, or a skipped run reusing whatever produced the
    existing stems). Returning a small frozen dataclass instead of a bare
    dict, or stashing the info in a module-level "last run" global, were the
    two options on the table; the dataclass was chosen because it keeps the
    result self-contained and race-safe if `separate()` is ever called
    concurrently for different tracks. This is a deliberate deviation from
    the stub's declared return type -- flagged for W2 (CLI) and W1F, who
    both consume this return value.

    Attributes:
        stems: stem name -> written (or pre-existing) wav path.
        model: the Demucs model that actually produced the stems.
        device: the torch device that actually produced the stems ("mps" or
            "cpu"). Reflects the post-fallback device on a run that retried.
        skipped: True if existing stems were reused without loading a model.
    """

    stems: dict[str, Path]
    model: str
    device: str
    skipped: bool


def default_model() -> str:
    """Standing model default: TRACK_FORENSICS_MODEL if set, else DEFAULT_MODEL."""
    return os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL


def pick_device() -> str:
    """Return the best available torch device: 'mps' on Apple silicon, else 'cpu'.

    Never raises: missing torch, a torch build without an `mps` backend
    attribute, or an `is_available()` call that itself errors all degrade to
    'cpu'. Never returns 'cuda' -- this project targets Apple silicon only.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    try:
        mps = torch.backends.mps
    except AttributeError:
        return "cpu"

    try:
        if mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - any backend probe failure means "not usable"
        return "cpu"

    return "cpu"


def slugify(name: str) -> str:
    """Lowercase, filesystem/URL-safe slug: non-alphanumerics -> hyphens.

    Deterministic and simple by design: lowercase, collapse any run of
    non-alphanumeric characters (including unicode letters, which are not
    matched by [a-z0-9]) into a single hyphen, strip leading/trailing
    hyphens, and fall back to a fixed non-empty string if nothing usable
    remains (e.g. an input name made entirely of punctuation or emoji).
    """
    slug = _SLUG_DISALLOWED_RE.sub("-", name.strip().lower()).strip("-")
    return slug or _SLUG_FALLBACK


def stem_paths(output_root: Path, track_name: str) -> dict[str, Path]:
    """Expected stem paths for a track, whether or not they exist yet.

    Pure path construction, no I/O. `track_name` is expected to already be
    slugified by the caller (see `separate()`).
    """
    stems_dir = output_root / track_name / "stems"
    return {stem: stems_dir / f"{stem}.wav" for stem in STEM_NAMES}


def _stems_exist(paths: dict[str, Path]) -> bool:
    return all(path.is_file() for path in paths.values())


def _write_stem_wavs(
    separated: dict[str, torch.Tensor], paths: dict[str, Path], samplerate: int
) -> None:
    """Write each separated stem tensor to its target wav path.

    Uses `demucs.api.save_audio`, which handles clip-safe rescaling, rather
    than a raw torchaudio call. Imports demucs lazily, matching the
    module-level lazy-import policy.
    """
    from demucs.api import save_audio  # type: ignore[attr-defined]

    for stem, path in paths.items():
        tensor = separated[stem]
        path.parent.mkdir(parents=True, exist_ok=True)
        # demucs.api returns torch tensors shaped (channels, samples).
        save_audio(tensor, path, samplerate)


def _run_separator(
    input_path: Path, model: str, device: str
) -> tuple[dict[str, torch.Tensor], int]:
    """Load the Demucs model on `device` and separate `input_path`.

    Returns the stem-name -> tensor mapping from `demucs.api.Separator`
    plus the model's native sample rate. Raises whatever the underlying
    Separator/torch call raises; the retry policy lives in `separate()`,
    not here.
    """
    from demucs.api import Separator

    separator = Separator(model=model, device=device)
    _origin, separated = separator.separate_audio_file(input_path)
    return dict(separated), int(separator.samplerate)


def separate(
    input_path: Path,
    output_root: Path,
    model: str | None = None,
    device: str | None = None,
    force: bool = False,
) -> SeparationResult:
    """Run Demucs and return the written (or reused) stem paths plus provenance.

    `model=None` resolves via `default_model()`. `device=None` resolves via
    `pick_device()`. If all four stems already exist under
    `output_root/<slug>/stems/` and `force` is False, the run is skipped
    entirely -- no model is loaded -- and the existing paths are returned
    with `skipped=True`. In the skipped case there is no way to know what
    model/device actually produced those files, so the *requested*
    model/device (after resolving `None`) is reported instead; this is
    documented on `SeparationResult`.

    If separation on the chosen device raises, and the device is 'mps', the
    run is retried exactly once on 'cpu' with a clear warning logged.
    `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in-process before the first
    attempt, so partial MPS gaps in individual ops can fall back to CPU
    within a single run rather than aborting outright.

    Stems are always written at `ANALYSIS_SAMPLE_RATE` (44.1 kHz) -- never
    downsampled -- since analysis accuracy on hats and cymbals depends on
    the full band.
    """
    resolved_model = model or default_model()
    resolved_device = device or pick_device()

    track_name = slugify(input_path.stem)
    paths = stem_paths(output_root, track_name)

    if not force and _stems_exist(paths):
        logger.info(
            "Skipping separation for %r: all stems already exist at %s "
            "(pass force=True to re-run)",
            track_name,
            paths[STEM_NAMES[0]].parent,
        )
        return SeparationResult(
            stems=paths,
            model=resolved_model,
            device=resolved_device,
            skipped=True,
        )

    # Known real failure mode: some ops used by Demucs have gaps in MPS
    # support depending on the torch build. Enabling the fallback lets torch
    # silently route unsupported ops to CPU instead of raising, and it must
    # be set before the model loads.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    run_device = resolved_device
    try:
        separated, samplerate = _run_separator(input_path, resolved_model, run_device)
    except Exception as exc:  # noqa: BLE001 - any device-side failure triggers fallback
        if run_device != "cpu":
            logger.warning(
                "Demucs separation failed on device %r (%s: %s); retrying once on CPU.",
                run_device,
                type(exc).__name__,
                exc,
            )
            run_device = "cpu"
            separated, samplerate = _run_separator(input_path, resolved_model, run_device)
        else:
            raise

    if samplerate != ANALYSIS_SAMPLE_RATE:
        # Every htdemucs-family model is trained and run at 44.1 kHz, so this
        # would indicate an unexpected model choice rather than something we
        # should silently paper over by resampling (this project never
        # resamples down, and resampling up here would be lying about
        # provenance). Fail loudly instead.
        raise RuntimeError(
            f"Demucs model {resolved_model!r} produced audio at {samplerate} Hz, "
            f"expected {ANALYSIS_SAMPLE_RATE} Hz. Refusing to write stems at the "
            "wrong sample rate."
        )

    _write_stem_wavs(separated, paths, samplerate)

    return SeparationResult(
        stems=paths,
        model=resolved_model,
        device=run_device,
        skipped=False,
    )
