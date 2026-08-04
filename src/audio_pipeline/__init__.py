"""track-forensics: local-only stem separation and audio analysis pipeline."""

__version__ = "0.1.0"

# 3: RhythmFeatures gained onset_times — the observed hit positions subdivision
#    detection needs, distinct from the inferred, evenly spaced beat_times.
# 2: SpectralFeatures gained band_energy_ratios over BAND_EDGES_HZ.
SCHEMA_VERSION = 3

STEM_NAMES: tuple[str, ...] = ("drums", "bass", "vocals", "other")
SOURCE_NAMES: tuple[str, ...] = ("mix", *STEM_NAMES)
SUPPORTED_INPUT_SUFFIXES: frozenset[str] = frozenset({".wav", ".mp3", ".aiff", ".aif", ".m4a"})

# Analysis runs at full rate. Accuracy on hats and cymbals beats speed here, so
# nothing in the pipeline may downsample before feature extraction.
ANALYSIS_SAMPLE_RATE = 44100

# Shared band edges in Hz. Both analysis backends must use these exact bounds so
# their band-energy ratios stay comparable and heuristic thresholds transfer.
BAND_EDGES_HZ: dict[str, tuple[float, float]] = {
    "low": (20.0, 250.0),
    "low_mid": (250.0, 2000.0),
    "high_mid": (2000.0, 6000.0),
    "high": (6000.0, 20000.0),
}
