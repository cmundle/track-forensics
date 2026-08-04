"""librosa analysis backend.

The fallback backend, and in practice the one that actually runs on macOS when
Essentia refuses to build. Treat it as first-class.

Implementation notes for whoever fills these in:

- Import `librosa` (and `pyloudnorm`) **inside** the method bodies or at this
  module's top level only — never from any module outside `backends/`.
- Audio arrives at its native rate, which is 44.1 kHz or higher. Never pass
  `sr=None`/omit `sr=` to a librosa call that defaults to 22050, and never
  resample.
- Band-energy ratios must use `BAND_EDGES_HZ` verbatim. Write that computation
  as a documented module-level helper, not inline in `spectral()`: the Essentia
  backend mirrors it and the two must agree.
- Anything you cannot compute is `None` on the model, not an exception.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..schemas import (
    DynamicsFeatures,
    RhythmFeatures,
    SpectralFeatures,
    TonalFeatures,
)
from . import BackendName


class LibrosaBackend:
    """Feature extraction backed by librosa (+ pyloudnorm for LUFS)."""

    name: BackendName = "librosa"

    def rhythm(self, audio: npt.NDArray[np.float32], sample_rate: int) -> RhythmFeatures:
        """BPM, beat times, onset density, transient sharpness."""
        raise NotImplementedError

    def tonal(self, audio: npt.NDArray[np.float32], sample_rate: int) -> TonalFeatures:
        """Key, scale, key confidence, 12-bin chroma mean, tonal stability."""
        raise NotImplementedError

    def spectral(self, audio: npt.NDArray[np.float32], sample_rate: int) -> SpectralFeatures:
        """Centroid, rolloff, brightness, band-energy ratios over BAND_EDGES_HZ."""
        raise NotImplementedError

    def dynamics(self, audio: npt.NDArray[np.float32], sample_rate: int) -> DynamicsFeatures:
        """Integrated LUFS, RMS mean/std, crest factor.

        Accepts stereo `(n_samples, n_channels)` input; integrated loudness is
        only meaningful on the channel layout the file actually has.
        """
        raise NotImplementedError


if TYPE_CHECKING:  # pragma: no cover - static structural check only
    from . import AnalysisBackend

    _protocol_check: AnalysisBackend = LibrosaBackend()
