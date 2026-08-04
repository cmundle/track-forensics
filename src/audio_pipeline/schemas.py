"""Pydantic models defining every JSON artefact this tool writes.

Single source of truth for output shape. Bump `SCHEMA_VERSION` in `__init__.py`
whenever a field changes meaning.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import ANALYSIS_SAMPLE_RATE, SCHEMA_VERSION


class RhythmFeatures(BaseModel):
    """Tempo and event-timing descriptors."""

    bpm: float | None = None
    bpm_confidence: float | None = None
    beat_times: list[float] = Field(
        default_factory=list,
        description=(
            "The pulse you would tap along to, in seconds. Inferred and evenly "
            "spaced by construction, so it says nothing about what was actually "
            "played between beats. Not the same as onset_times."
        ),
    )
    onset_times: list[float] = Field(
        default_factory=list,
        description=(
            "When notes and hits actually start, in seconds. Observed, not "
            "inferred, and unevenly spaced: this is where swing lives, so it is "
            "what subdivision-feel detection reads. Not the same as beat_times."
        ),
    )
    onset_density: float | None = Field(default=None, description="Onsets per second")
    transient_sharpness: float | None = None


class TonalFeatures(BaseModel):
    """Key, scale, and chroma descriptors."""

    key: str | None = None
    scale: str | None = None
    key_confidence: float | None = None
    hpcp_mean: list[float] = Field(default_factory=list, description="12-bin chroma/HPCP means")
    tonal_stability: float | None = None


class BandEnergyRatios(BaseModel):
    """Share of total spectral energy per band. Present fields sum to ~1.0.

    Band edges are fixed in `BAND_EDGES_HZ` and shared by both backends so the
    heuristic thresholds tuned on one backend remain meaningful on the other.
    """

    low: float | None = Field(default=None, description="20-250 Hz")
    low_mid: float | None = Field(default=None, description="250-2000 Hz")
    high_mid: float | None = Field(default=None, description="2000-6000 Hz")
    high: float | None = Field(default=None, description="6000-20000 Hz")


class SpectralFeatures(BaseModel):
    """Timbre-related descriptors."""

    centroid_mean: float | None = None
    centroid_std: float | None = None
    rolloff_mean: float | None = None
    brightness: float | None = None
    band_energy_ratios: BandEnergyRatios = Field(default_factory=BandEnergyRatios)


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


#: Per-source event lists dropped from `track_summary.json`, mapped to the count
#: field that takes their place. They stay complete in `analysis/<source>.json`,
#: so nothing is lost — the summary just stops being thousands of floats of
#: duplicated data across five sources.
_SUMMARY_LIST_FIELDS: dict[str, str] = {
    "beat_times": "beat_count",
    "onset_times": "onset_count",
}


class TrackSummary(BaseModel):
    """Combined view across the mix and all stems.

    Beat and onset times live in the per-source `analysis/*.json` files only.
    They are omitted from the written summary — a six-minute track produces
    roughly 720 beat floats per source, and a busy drum stem several times that
    in onsets, which is pure duplication and makes the one file you actually
    read by hand unreadable. Use `summary_payload()` to serialise.
    """

    schema_version: int = SCHEMA_VERSION
    track_name: str
    input_path: str
    duration_seconds: float
    backend: str
    separation_model: str | None = None
    separation_device: str | None = None
    analysis_sample_rate: int = ANALYSIS_SAMPLE_RATE
    sources: dict[str, SourceAnalysis] = Field(default_factory=dict)

    def summary_payload(self) -> dict[str, object]:
        """Dict for writing `track_summary.json`, with event lists stripped.

        Every list named in `_SUMMARY_LIST_FIELDS` — `beat_times`, `onset_times`
        — is replaced by its length under the paired count name, so nothing is
        silently lost and the full lists stay in each source's own analysis file.

        Each count takes the slot its list occupied, so key order across the rest
        of the payload is unchanged and run-to-run diffs stay readable.
        """
        payload = self.model_dump(mode="json")
        for source in payload.get("sources", {}).values():
            rhythm = source.get("rhythm")
            if not isinstance(rhythm, dict):
                continue
            source["rhythm"] = {
                _SUMMARY_LIST_FIELDS.get(key, key): (
                    len(value) if key in _SUMMARY_LIST_FIELDS else value
                )
                for key, value in rhythm.items()
            }
        return payload


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
