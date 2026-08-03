"""track-forensics: local-only stem separation and audio analysis pipeline."""

__version__ = "0.1.0"

SCHEMA_VERSION = 1

STEM_NAMES: tuple[str, ...] = ("drums", "bass", "vocals", "other")
SOURCE_NAMES: tuple[str, ...] = ("mix", *STEM_NAMES)
SUPPORTED_INPUT_SUFFIXES: frozenset[str] = frozenset({".wav", ".mp3", ".aiff", ".aif", ".m4a"})
