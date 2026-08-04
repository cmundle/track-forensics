# track-forensics

Local-only, offline audio forensics CLI for macOS Apple silicon. Takes one audio file, separates it into stems with Demucs, analyzes the mix and each stem, writes structured JSON, and exports compact hints you can use to rebuild the track by hand in [Strudel](https://strudel.cc).

No web app, no GUI, no cloud. v1 is deterministic analysis output only — Strudel pattern *generation* is deliberately out of scope.

## Pipeline

```
input track -> Demucs stems -> per-stem analysis -> JSON summaries -> Strudel hints
```

## Install

Requires Python 3.11 (Demucs and Essentia are both happiest there).

```bash
# System deps
brew install ffmpeg
brew install python@3.11

# Project
cd track-forensics
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Analyzer backend — try Essentia first
pip install -e ".[essentia]"

# If Essentia fails to build/install on your machine, use the fallback
pip install -e ".[librosa]"
```

The analyzer is modular: it picks Essentia when importable and falls back to librosa otherwise. Check which backend is live with:

```bash
track-forensics doctor
```

### Apple silicon notes

- Demucs runs on `mps` when available, with automatic CPU fallback.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if you hit an unimplemented MPS op.
- Essentia has no universal Apple-silicon wheel for every Python version. If `pip install essentia` fails, don't fight it — use the librosa backend.

## Usage

```bash
track-forensics all input.wav              # default: separate + analyze + hints
track-forensics all input.wav --fast       # quicker separation, rougher stems
track-forensics separate input.wav
track-forensics analyze input.wav
track-forensics export-strudel-hints input.wav
track-forensics doctor                     # report backend/device availability
```

Accepted input: `.wav`, `.mp3`, `.aiff`, `.m4a` (anything FFmpeg can decode).

### Separation model

Defaults to `htdemucs_ft` — the fine-tuned Demucs model. Cleaner stems, roughly four times slower than plain `htdemucs`. Expect minutes, not seconds, on a full track.

```bash
track-forensics all input.wav --model htdemucs   # or just --fast
export TRACK_FORENSICS_MODEL=htdemucs            # standing preference
```

### Sample rate

The whole pipeline runs at 44.1 kHz and never downsamples. Analysis accuracy on hi-hats and cymbals is the point of this tool, and a 22.05 kHz shortcut would cap the usable spectrum at 11 kHz.

## Output layout

```
output/<track-name>/
  stems/
    drums.wav
    bass.wav
    vocals.wav
    other.wav
  analysis/
    mix.json
    drums.json
    bass.json
    vocals.json
    other.json
  track_summary.json
  strudel_hints.json
```

## What gets analyzed

Per source (mix + each stem):

| Group | Descriptors |
|---|---|
| Rhythm | BPM estimate, beat positions, onset density, transient sharpness |
| Tonal | key, scale, key confidence, chroma/HPCP summary, tonal stability |
| Spectral | spectral centroid, spectral rolloff, brightness |
| Dynamics | loudness (integrated LUFS or equivalent), RMS, crest factor |

Heuristics layer turns those into human-readable production clues: `kick-heavy`, `bright hats`, `sustained pad-like texture`, `speech/vocal dominant`, `sparse bass`, `busy drums`, `percussive`, `noisy`.

## Strudel hints

`strudel_hints.json` is intentionally small and hand-readable: drum density, likely subdivision feel, bass activity, tonal centre, and suggested cycle length. It's a starting point for writing patterns yourself, not generated code.

## Development

```bash
pytest
ruff check .
mypy src
```

## Status

Scaffold only. Every module is a typed stub with a docstring describing its contract. See `CLAUDE.md` for the implementation brief.
