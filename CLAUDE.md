# track-forensics — implementation brief

Read this before writing code. The pipeline is implemented and working end to end (`track-forensics all input.wav` produces the full output tree); this document is the standing contract for anyone extending it, not a stub-filling checklist. `schema_version` is currently 3. See `README.md` for install steps, verified `doctor` output, and real sample output.

## Goal

A local-only macOS Apple-silicon Python CLI that takes an audio file, separates stems with Demucs, analyzes each stem, and writes JSON for `mix`, `drums`, `bass`, `vocals`, and `other` — plus compact Strudel-friendly hints for rebuilding the track by hand.

Must run fully offline. No network calls at runtime, no cloud services, no web app, no GUI.

## Constraints

- Python 3.11, type hints everywhere, pydantic v2 models for all JSON output.
- Demucs for separation, prefer `mps` device on Apple silicon with CPU fallback. Default model is `htdemucs_ft` (quality over speed), overridable via `--model`, `--fast`, or `TRACK_FORENSICS_MODEL`.
- **44.1 kHz everywhere. Never downsample.** Accuracy on hats and cymbals is the whole point; use `ANALYSIS_SAMPLE_RATE`.
- Essentia is the preferred analysis backend. On the current arm64/Python 3.11 combination it installs in seconds from a published wheel (`essentia-2.1b6.dev1389-cp311-cp311-macosx_15_0_arm64.whl`) and every algorithm this project needs works, so `get_backend()` resolves to it by default — it is **not** the fragile install this project originally assumed. That wheel is still tied to a specific Python version and macOS release and will not exist on every machine, so the librosa backend stays genuine, first-class fallback, not dead weight: keep the analyzer modular behind the `AnalysisBackend` Protocol in `backends/__init__.py`, and never hard-import `essentia` or `librosa` at module top level in a way that breaks the CLI when one or both are absent.
- FFmpeg handles decoding of non-wav input.
- pytest for tests. Tests must not require a real audio file to run — generate synthetic signals with numpy.

## Modules and contracts

| File | Responsibility |
|---|---|
| `cli.py` | Typer app. Commands: `separate`, `analyze`, `all`, `export-strudel-hints`, `doctor`. `all` is the default path for `track-forensics all input.wav`. |
| `separate.py` | Runs Demucs, writes stems to `output/<track-name>/stems/`. Device selection + fallback lives here. Skips work if stems already exist unless `--force`. |
| `audio_io.py` | The one place that decodes and (up-only) resamples audio. Loads via `soundfile`, shells out to FFmpeg for anything `soundfile` rejects. Returns `(np.float32 array, sample_rate)`, mono or stereo. |
| `backends/` | `__init__.py` holds the `AnalysisBackend` Protocol, `available_backends()`, and `get_backend()` (lazy, try/except-guarded imports, prefers Essentia). `essentia_backend.py` and `librosa_backend.py` are the two concrete implementations — same Protocol, comparable output, deliberately no schema drift between them. |
| `analyze.py` | Orchestration only, not extraction: resolves a backend, loops mix + each present stem, calls the backend's four feature methods, runs `heuristics.apply()`, and writes `analysis/<source>.json` plus `track_summary.json`. |
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
