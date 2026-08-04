# track-forensics

Local-only, offline audio forensics CLI for macOS Apple silicon. Takes one audio file, separates it into stems with Demucs, analyzes the mix and each stem, writes structured JSON, and exports compact hints you can use to rebuild the track by hand in [Strudel](https://strudel.cc).

No web app, no GUI, no cloud. v1 is deterministic analysis output only — Strudel pattern *generation* is deliberately out of scope.

## Pipeline

```
input track -> Demucs stems -> per-stem analysis -> JSON summaries -> Strudel hints
```

## Install

Requires **Python 3.11 exactly** (`>=3.11,<3.12`) — Demucs and Essentia are both happiest there, and it is very likely *not* the `python3` your shell already resolves to. On a stock macOS/Homebrew setup `python3` is commonly 3.9.x; you need an explicit 3.11 interpreter (`brew install python@3.11`, `pyenv`, `uv python install 3.11`, etc.) and must point the venv at it directly, or the very first command below fails with a confusing "no matching distribution" error before you get anywhere near audio code.

```bash
# System deps
brew install ffmpeg          # decoding for non-wav input; must end up on PATH
brew install python@3.11     # or pyenv/uv — any Python 3.11.x interpreter works

# Project
cd track-forensics
python3.11 -m venv .venv     # NOT `python3 -m venv` — that's almost never 3.11
source .venv/bin/activate
pip install -e ".[dev]"

# Analyzer backend — try Essentia first
pip install -e ".[essentia]"

# If Essentia fails to build/install on your machine, use the fallback instead
pip install -e ".[librosa]"
```

The analyzer is modular: it picks Essentia when importable and falls back to librosa otherwise. Check which backend is live with:

```bash
track-forensics doctor
```

Note that `scipy` is a base dependency, not part of either extra — `audio_io` needs its polyphase resampler regardless of which analyzer backend you install. `pyloudnorm` (the LUFS meter librosa lacks) lives in the `librosa` extra.

### Apple silicon notes

- Demucs runs on `mps` when available, with automatic CPU fallback.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if you hit an unimplemented MPS op.
- **Essentia currently installs cleanly and fast on Apple silicon.** As of this writing there is a published arm64 wheel (`essentia-2.1b6.dev1389-cp311-cp311-macosx_15_0_arm64.whl`) that installs in well under a minute on Python 3.11, and every algorithm this project uses works with it — `track-forensics doctor` resolves to `essentia` by default on a machine like this. That said, the wheel is tied to a specific Python version and macOS release and is not guaranteed to exist for yours. If `pip install -e ".[essentia]"` fails to find a matching wheel or won't build, don't fight it — install `.[librosa]` instead. The librosa backend is treated as first-class, not a stopgap; it substitutes for essentially every feature Essentia provides.

### Verifying the install

`doctor` is the first thing to run after installing — it never raises, so it's safe to run on a half-broken environment to see what's missing. Genuine output from a fresh venv built with the steps above, on Apple silicon with the Essentia extra installed:

```
$ track-forensics doctor
Python: 3.11.15 (/Users/you/track-forensics/.venv/bin/python3.11)
FFmpeg: found (/opt/homebrew/bin/ffmpeg)

-- Demucs / separation --
torch: 2.13.0
  MPS available: True
Resolved Demucs device: 'mps'
Resolved Demucs model: 'htdemucs_ft' (from default)

-- Analysis backend --
Installed backends: essentia
Active backend: 'essentia'
```

If `.[librosa]` is installed instead (or alongside), `Installed backends` lists both and `Active backend` still prefers `essentia` when it's importable.

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

### First run downloads model weights

The first time you run `separate` or `all`, Demucs pulls the model weights from the HuggingFace Hub: roughly 80 MB for `htdemucs`, roughly 320 MB for the `htdemucs_ft` default. You'll see a line like:

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

That warning is harmless — no account or token is required, it's just Hub telemetry noise — and it is the only network activity this tool ever performs. Weights are cached under `~/.cache/huggingface/hub/` afterward, so every subsequent run, and every other guarantee in this README, is genuinely offline.

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

### Real sample output

From `track-forensics all synth_mix_100bpm.wav --fast` against a 12-second synthetic mix (kick, snare, hats on 8ths, a moving bass line, a sustained A-minor pad, and deliberately no vocals). `strudel_hints.json` in full:

```json
{
  "schema_version": 3,
  "track_name": "synth-mix-100bpm",
  "bpm": 100.03359985351562,
  "suggested_cycle_seconds": 2.399194,
  "subdivision_feel": "straight 8ths",
  "drum_density": "moderate",
  "bass_activity": "moderate",
  "tonal_centre": "A minor",
  "notes": []
}
```

An excerpt of `track_summary.json` for the `drums` source shows the shape every source follows — rhythm/tonal/spectral/dynamics blocks plus auditable labels:

```json
"drums": {
  "schema_version": 3,
  "source": "drums",
  "rhythm": {
    "bpm": 100.0336,
    "bpm_confidence": 3.976937,
    "beat_count": 19,
    "onset_count": 40,
    "onset_density": 3.333333,
    "transient_sharpness": 100.0
  },
  "spectral": {
    "centroid_mean": 3544.332726,
    "band_energy_ratios": { "low": 0.9034, "low_mid": 0.007091, "high_mid": 0.016505, "high": 0.073004 }
  },
  "labels": [
    { "label": "kick-heavy", "confidence": 1.0,
      "evidence": { "band_energy_low": 0.9034, "min_band_energy_low": 0.5 } },
    { "label": "percussive", "confidence": 0.222814,
      "evidence": { "crest_factor": 10.673765, "onset_density": 3.333333, "min_crest_factor": 8.0, "min_onset_density": 2.0 } }
  ],
  "unavailable_features": []
}
```

Worth knowing before you look at real output from an instrumental track: this test mix had no vocals, so Demucs still produces a `vocals` stem (separation always writes all four), but it's silence plus separation artifacts. Its analysis comes back at `loudness_lufs: -68.23` and picks up a single `"silent/absent stem"` label at high confidence — that's the intended behaviour, not a bug, and is exactly what you should expect to see on any track that genuinely has no vocal content.

## What gets analyzed

Per source (mix + each stem):

| Group | Descriptors |
|---|---|
| Rhythm | BPM estimate, beat positions, onset density, transient sharpness |
| Tonal | key, scale, key confidence, chroma/HPCP summary, tonal stability |
| Spectral | spectral centroid, spectral rolloff, brightness, band energy ratios (low / low-mid / high-mid / high) |
| Dynamics | loudness (integrated LUFS), RMS, crest factor |

Beat times live only in the per-source `analysis/*.json` files; `track_summary.json` carries a `beat_count` instead so the one file you're likely to read by hand doesn't balloon into thousands of floats. Any descriptor a backend can't compute comes back `null` and its name is recorded in that source's `unavailable_features` list rather than crashing the run.

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

Implemented and working end to end: `track-forensics all input.wav` runs separation, per-source analysis, heuristic labelling, and Strudel-hint export, and produces the full output tree described above. Schema is at `schema_version` 3. `pytest`, `ruff check .`, and `mypy src` are all clean. See `CLAUDE.md` for the implementation brief and module contracts if you're extending it.
