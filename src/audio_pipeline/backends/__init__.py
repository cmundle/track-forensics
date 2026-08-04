"""Analysis backend interface and lazy resolution.

Essentia is preferred; librosa is the fallback for the (common) case where
Essentia will not install on macOS. Neither library is imported at module top
level: resolution happens inside the function bodies below so that the CLI
always loads and `doctor` can report what is actually installed.

This package is the *only* place in `audio_pipeline` allowed to import
`essentia` or `librosa`.
"""

from __future__ import annotations

import importlib
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..schemas import (
    DynamicsFeatures,
    PitchTrack,
    RhythmFeatures,
    SpectralFeatures,
    TonalFeatures,
)

__all__ = [
    "BACKEND_PREFERENCE",
    "BASS_F0_MAX_HZ",
    "BASS_F0_MIN_HZ",
    "AnalysisBackend",
    "BackendName",
    "BackendUnavailableError",
    "available_backends",
    "get_backend",
]

BackendName = Literal["essentia", "librosa"]

#: Resolution order when no backend is requested explicitly. Essentia first —
#: it has purpose-built key/beat extractors — with librosa as the fallback.
BACKEND_PREFERENCE: tuple[BackendName, ...] = ("essentia", "librosa")

#: Third-party module each backend needs. Probed with a real import rather than
#: `importlib.util.find_spec` because a broken Essentia wheel installs fine and
#: only fails when its shared libraries are loaded.
_REQUIRED_MODULE: dict[BackendName, str] = {
    "essentia": "essentia.standard",
    "librosa": "librosa",
}

_INSTALL_HINT = (
    'Install one: pip install -e ".[librosa]" (recommended on macOS) '
    'or pip install -e ".[essentia]".'
)

#: Frequency range every backend's `pitch()` searches, in Hz. **A module
#: constant, deliberately not a parameter of `pitch()`**, so both backends
#: analyse exactly the same register and their F0 tracks stay comparable — a
#: caller that could narrow the range per backend would reintroduce the very
#: divergence this seam exists to remove.
#:
#: The bounds are a bass guard as much as a search window. 30 Hz sits just below
#: B0 (30.87 Hz), the lowest note on a 5-string bass, and 400 Hz is roughly g4 —
#: above anything a bass line plays and below most of what bleeds into a bass
#: stem. **Constraining the range is itself the single most effective octave
#: guard available**: an estimator that cannot represent 110 Hz's neighbours
#: above 400 Hz cannot report them, and one that cannot look below 30 Hz cannot
#: halve a 55 Hz fundamental into subsonic nonsense. `note_track`'s octave snap
#: cleans up what is left inside the window.
BASS_F0_MIN_HZ = 30.0
BASS_F0_MAX_HZ = 400.0


class BackendUnavailableError(RuntimeError):
    """Raised when no analysis backend can be imported."""


@runtime_checkable
class AnalysisBackend(Protocol):
    """Interface every analysis backend must satisfy.

    A backend returns `None` for any descriptor it cannot compute; the caller
    records those in `SourceAnalysis.unavailable_features` rather than crashing.

    All five methods take mono audio as a 1-D float32 array at the file's own
    sample rate (44.1 kHz or higher — see `audio_io.load_audio`). `dynamics()`
    additionally accepts stereo audio shaped `(n_samples, n_channels)`, which
    integrated-loudness measurement needs.

    **Why pitch is a Protocol method and drum decomposition is not.** The two
    Wave 4 features look symmetrical and are deliberately not implemented
    symmetrically. The rule is:

        Put it behind this Protocol only when the computation is an algorithm
        this project should not own. Put it in a shared numpy module when it is
        a measurement numpy can make exactly.

    Drum decomposition is band energy, decay ratios and spectral flatness —
    exact arithmetic with one right answer, which `band_energy_ratios` already
    proves (both backends agree on it to 0.0001 precisely because it is
    library-free). So it lives in `drum_elements.py`, is computed once, and is
    identical whichever backend ran. Adding it here would buy nothing and create
    two implementations that could drift.

    F0 estimation is the opposite: YIN and pYIN are real algorithms with
    published papers, and reimplementing one would be strictly worse than
    depending on `librosa.pyin` or Essentia's YIN family. So `pitch()` is here —
    but it returns a **raw framewise `PitchTrack`, not notes**. Turning F0 into
    notes is musical interpretation, so it lives in `note_track.segment_notes()`
    and is shared: given identical F0, both backends produce byte-identical
    `BassLine`s, and the only thing that can differ across backends is the F0
    estimate itself, which has a right answer and is tested against a fixture
    synthesised at exactly 55.000 Hz.

    Do not "fix" the asymmetry in either direction.
    """

    name: BackendName

    def rhythm(self, audio: npt.NDArray[np.float32], sample_rate: int) -> RhythmFeatures: ...
    def tonal(self, audio: npt.NDArray[np.float32], sample_rate: int) -> TonalFeatures: ...
    def spectral(self, audio: npt.NDArray[np.float32], sample_rate: int) -> SpectralFeatures: ...
    def dynamics(self, audio: npt.NDArray[np.float32], sample_rate: int) -> DynamicsFeatures: ...
    def pitch(self, audio: npt.NDArray[np.float32], sample_rate: int) -> PitchTrack: ...


def _is_importable(backend: BackendName) -> bool:
    """Whether `backend`'s third-party dependency imports cleanly right now.

    Catches `Exception` rather than `ImportError` alone: a half-installed
    Essentia typically raises `OSError` from the dynamic loader.
    """
    try:
        importlib.import_module(_REQUIRED_MODULE[backend])
    except Exception:
        return False
    return True


def _instantiate(backend: BackendName) -> AnalysisBackend:
    """Construct a backend. Imports the backend module lazily."""
    if backend == "essentia":
        from .essentia_backend import EssentiaBackend

        return EssentiaBackend()
    from .librosa_backend import LibrosaBackend

    return LibrosaBackend()


def available_backends() -> list[BackendName]:
    """Which analysis backends are importable right now, in preference order.

    Used by `doctor`. Never raises — an uninstallable backend is simply absent.
    """
    return [name for name in BACKEND_PREFERENCE if _is_importable(name)]


def get_backend(preferred: BackendName | None = None) -> AnalysisBackend:
    """Resolve a backend, defaulting to Essentia and falling back to librosa.

    Args:
        preferred: Force a specific backend. When given and that backend is not
            importable, this raises rather than silently using the other one —
            a run that quietly swapped backends would produce numbers that are
            not comparable with the previous run.

    Raises:
        BackendUnavailableError: when nothing usable is installed.
    """
    if preferred is not None:
        if not _is_importable(preferred):
            raise BackendUnavailableError(
                f"Analysis backend {preferred!r} was requested but "
                f"{_REQUIRED_MODULE[preferred]!r} could not be imported. {_INSTALL_HINT}"
            )
        return _instantiate(preferred)

    for name in BACKEND_PREFERENCE:
        if _is_importable(name):
            return _instantiate(name)

    raise BackendUnavailableError(
        "No analysis backend is installed: neither essentia nor librosa could be "
        f"imported. {_INSTALL_HINT}"
    )
