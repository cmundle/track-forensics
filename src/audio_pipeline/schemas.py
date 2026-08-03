"""Pydantic models defining every JSON artefact this tool writes.

Single source of truth for output shape. Bump `SCHEMA_VERSION` in `__init__.py`
whenever a field changes meaning.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import SCHEMA_VERSION


class RhythmFeatures(BaseModel):
    """Tempo and event-timing descriptors."""

    bpm: float | None = None
    bpm_confidence: float | None = None
    beat_times: list[float] = Field(default_factory=list, description="Beat positions in seconds")
    onset_density: float | None = Field(default=None, description="Onsets per second")
    transient_sharpness: float | None = None


class TonalFeatures(BaseModel):
    """Key, scale, and chroma descriptors."""

    key: str | None = None
    scale: str | None = None
    key_confidence: float | None = None
    hpcp_mean: list[float] = Field(default_factory=list, description="12-bin chroma/HPCP means")
    tonal_stability: float | None = None


class SpectralFeatures(BaseModel):
    """Timbre-related descriptors."""

    centroid_mean: float | None = None
    centroid_std: float | None = None
    rolloff_mean: float | None = None
    brightness: float | None = None


class DynamicsFeatures(BaseModel):
    """Level and dynamic-range descriptors."""

    loudness_lufs: float | None = None
    rms_mean: float | None = None
    rms_std: float | None = None
    crest_factor: float | None = None


class HeuristicLabel(BaseModel):
    """A human-readable production clue plus the evidence behind it."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, float] = Field(default_factory=dict)


class SourceAnalysis(BaseModel):
    """Full analysis of one source: the mix or a single stem."""

    schema_version: int = SCHEMA_VERSION
    source: str = Field(description="One of: mix, drums, bass, vocals, other")
    audio_path: str
    duration_seconds: float
    sample_rate: int
    backend: str = Field(description="Analysis backend used, e.g. essentia or librosa")
    rhythm: RhythmFeatures = Field(default_factory=RhythmFeatures)
    tonal: TonalFeatures = Field(default_factory=TonalFeatures)
    spectral: SpectralFeatures = Field(default_factory=SpectralFeatures)
    dynamics: DynamicsFeatures = Field(default_factory=DynamicsFeatures)
    labels: list[HeuristicLabel] = Field(default_factory=list)
    unavailable_features: list[str] = Field(default_factory=list)


class TrackSummary(BaseModel):
    """Combined view across the mix and all stems."""

    schema_version: int = SCHEMA_VERSION
    track_name: str
    input_path: str
    duration_seconds: float
    backend: str
    separation_model: str | None = None
    separation_device: str | None = None
    sources: dict[str, SourceAnalysis] = Field(default_factory=dict)


class StrudelHints(BaseModel):
    """Compact, hand-readable starting point for rebuilding the track in Strudel."""

    schema_version: int = SCHEMA_VERSION
    track_name: str
    bpm: float | None = None
    suggested_cycle_seconds: float | None = None
    subdivision_feel: str | None = Field(
        default=None, description="e.g. 'straight 16ths', 'swung 8ths'"
    )
    drum_density: str | None = Field(default=None, description="sparse | moderate | busy")
    bass_activity: str | None = None
    tonal_centre: str | None = None
    notes: list[str] = Field(default_factory=list)
