# track-forensics — implementation brief

Read this before writing code. The repo is currently a scaffold: structure, config, and typed stubs. Your job is to fill the stubs in.

## Goal

A local-only macOS Apple-silicon Python CLI that takes an audio file, separates stems with Demucs, analyzes each stem, and writes JSON for `mix`, `drums`, `bass`, `vocals`, and `other` — plus compact Strudel-friendly hints for rebuilding the track by hand.

Must run fully offline. No network calls at runtime, no cloud services, no web app, no GUI.

## Constraints

- Python 3.11, type hints everywhere, pydantic v2 models for all JSON output.
- Demucs for separation, prefer `mps` device on Apple silicon with CPU fallback. Default model is `htdemucs_ft` (quality over speed), overridable via `--model`, `--fast`, or `TRACK_FORENSICS_MODEL`.
- **44.1 kHz everywhere. Never downsample.** Accuracy on hats and cymbals is the whole point; use `ANALYSIS_SAMPLE_RATE`.
- Essentia is the preferred analysis backend, **but** its macOS install is fragile. Keep the analyzer modular behind a backend interface so librosa can substitute for most features. Never hard-import Essentia at module top level in a way that breaks the CLI.
- FFmpeg handles decoding of non-wav input.
- pytest for tests. Tests must not require a real audio file to run — generate synthetic signals with numpy.

## Modules and contracts

| File | Responsibility |
|---|---|
| `cli.py` | Typer app. Commands: `separate`, `analyze`, `all`, `export-strudel-hints`, `doctor`. `all` is the default path for `track-forensics all input.wav`. |
| `separate.py` | Runs Demucs, writes stems to `output/<track-name>/stems/`. Device selection + fallback lives here. Skips work if stems already exist unless `--force`. |
| `analyze.py` | Backend-agnostic feature extraction over the mix and each stem. Writes `analysis/<source>.json`. Emits `AnalysisResult` models. |
| `heuristics.py` | Pure functions: raw descriptors in, human-readable production clues out. No I/O, no audio. Easiest thing to unit test — do it properly. |
| `strudel_hints.py` | Condenses the full analysis into a small `strudel_hints.json`. |
| `schemas.py` | All pydantic models. Single source of truth for the JSON shape. Version the schema with a `schema_version` field. |

## Features to extract (per source)

- Rhythm: BPM estimate, beat positions (seconds), onset density (onsets/sec), transient sharpness
- Tonal: key, scale, key confidence, chroma/HPCP 12-bin summary, tonal stability
- Spectral: spectral centroid (mean/std), spectral rolloff, brightness, band energy ratios over `BAND_EDGES_HZ`
- Dynamics: integrated loudness, RMS, crest factor

Beat times are written to `analysis/*.json` only; `track_summary.json` carries a `beat_count` instead. See `TrackSummary.summary_payload()`.

If a feature is unavailable on the active backend, emit `null` and record it in a `unavailable_features` list rather than crashing.

## Heuristic labels

Derived, not measured. Examples: `kick-heavy`, `bright hats`, `sustained pad-like texture`, `speech/vocal dominant`, `sparse bass`, `busy drums`, `percussive`, `noisy`, `tonally stable`. Each label should carry the descriptor values that triggered it so the output is auditable.

## Strudel hints shape

Small and hand-readable. Drum density, likely subdivision feel (e.g. straight 16ths vs swung 8ths), bass activity, tonal centre, suggested cycle length. **Do not generate Strudel code in v1.**

## Non-goals for v1

- Generating Strudel patterns or any code output
- Web UI, GUI, notebooks
- Any cloud/API service
- Batch processing of directories (single file in, single output tree out)

## Definition of done

- `track-forensics all examples/<some file>.wav` produces the full output tree
- `track-forensics doctor` reports Python version, Demucs device, and active analysis backend
- `pytest` green, `ruff check .` clean, `mypy src` clean
- README install steps actually work from a fresh venv
