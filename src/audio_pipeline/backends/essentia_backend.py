"""Essentia analysis backend.

The preferred backend when it installs. Its macOS wheel is fragile, so nothing
outside this module may assume Essentia exists.

Implementation notes for whoever fills these in:

- Import `essentia.standard` **inside** the method bodies or at this module's
  top level only — never from any module outside `backends/`.
- Essentia's algorithms want `np.float32`; audio arrives as float32 already, at
  its native rate (44.1 kHz or higher). Do not resample.
- Band-energy ratios must use `BAND_EDGES_HZ` verbatim and mirror the helper in
  `librosa_backend.py` exactly, so the two backends stay comparable. Where an
  algorithm forces a structural difference, say why in a comment.
- `LoudnessEBUR128` requires stereo input; document that at the call site.
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


class EssentiaBackend:
    """Feature extraction backed by Essentia's standard algorithm suite."""

    name: BackendName = "essentia"

    def rhythm(self, audio: npt.NDArray[np.float32], sample_rate: int) -> RhythmFeatures:
        """BPM, beat times, onset density, transient sharpness."""
        raise NotImplementedError

    def tonal(self, audio: npt.NDArray[np.float32], sample_rate: int) -> TonalFeatures:
        """Key, scale, key confidence, 12-bin HPCP mean, tonal stability."""
        raise NotImplementedError

    def spectral(self, audio: npt.NDArray[np.float32], sample_rate: int) -> SpectralFeatures:
        """Centroid, rolloff, brightness, band-energy ratios over BAND_EDGES_HZ."""
        raise NotImplementedError

    def dynamics(self, audio: npt.NDArray[np.float32], sample_rate: int) -> DynamicsFeatures:
        """Integrated LUFS, RMS mean/std, crest factor.

        Accepts stereo `(n_samples, n_channels)` input, which `LoudnessEBUR128`
        requires.
        """
        raise NotImplementedError


if TYPE_CHECKING:  # pragma: no cover - static structural check only
    from . import AnalysisBackend

    _protocol_check: AnalysisBackend = EssentiaBackend()
