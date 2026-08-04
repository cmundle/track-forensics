# track-forensics — remaining work

Everything left to reach v1. Written to be split across parallel Claude Code agents.

Current state: scaffold committed. `schemas.py` is fully implemented at schema version 2. Every other module is a typed stub raising `NotImplementedError`. `pytest` passes the scaffold tests.

All four open decisions are settled — see [Settled decisions](#settled-decisions) at the bottom. They are already reflected in the packages below and in `schemas.py`, so no agent needs to revisit them.

---

## How to run this with agents

Work is grouped into **waves**. Everything inside a wave can run in parallel because no two packages own the same file. Do not start a wave until the previous one is merged and green.

```
Wave 0  (1 agent, sequential)   W0 — restructure + fixtures
Wave 1  (6 agents, parallel)    W1A W1B W1C W1D W1E W1F
Wave 2  (1 agent, sequential)   W2 — CLI + orchestration
Wave 3  (3 agents, parallel)    W3A W3B W3C
```

Each package below lists **Owns** (files that agent may edit), **Blocked by**, **Tasks**, and **Done when**. An agent must not edit files outside its Owns list — if it thinks it needs to, it should stop and report instead.

### Ground rules for every agent

- Python 3.11. Full type hints. `from __future__ import annotations` at the top of every module.
- No network calls at runtime. No cloud SDKs. Ever.
- **Everything runs at 44.1 kHz.** Do not resample or downsample anywhere in the pipeline — not for speed, not for a library's convenience. Use `ANALYSIS_SAMPLE_RATE` from `audio_pipeline/__init__.py`. This project trades speed for accuracy on hats and cymbals; a 22.05 kHz shortcut silently caps the spectrum at 11 kHz and quietly corrupts every centroid, rolloff, and high-band ratio.
- Band-energy ratios use the shared `BAND_EDGES_HZ` bounds. Do not invent per-backend bands.
- Never hard-import `essentia` or `librosa` at module top level outside `backends/`. The CLI must load even when neither is installed.
- Any descriptor that cannot be computed returns `None` and appends its name to `SourceAnalysis.unavailable_features`. Never crash on a missing feature.
- Tests must not require a real audio file. Use the synthetic fixtures from W0.
- Every package finishes with `pytest`, `ruff check .`, and `mypy src` clean.
- Do not edit `schemas.py` without flagging it — it is the contract everything else is built against. If a field genuinely needs to change, report it rather than changing it unilaterally.

---

## Wave 0 — restructure and fixtures

### W0 · Backend package split + test fixtures

**Owns:** `src/audio_pipeline/analyze.py`, new `src/audio_pipeline/backends/`, new `src/audio_pipeline/audio_io.py`, new `tests/conftest.py`, `pyproject.toml`
**Blocked by:** nothing

The current `analyze.py` holds the backend Protocol, the loader, and the orchestration in one file. Three agents would collide there. Split it first.

Target layout:

```
src/audio_pipeline/
  audio_io.py            # decode/resample, one place only
  analyze.py             # orchestration only: resolve backend, loop sources, write JSON
  backends/
    __init__.py          # AnalysisBackend Protocol, available_backends(), get_backend()
    essentia_backend.py  # stub for W1B
    librosa_backend.py   # stub for W1A
```

**Tasks**

1. Move `AnalysisBackend` Protocol, `available_backends()`, `get_backend()` into `backends/__init__.py`. Backend resolution imports lazily inside the function body, in a try/except ImportError, preferring Essentia.
2. Move `load_audio()` into `audio_io.py`. Decode via `soundfile`, shelling out to FFmpeg for anything soundfile rejects (m4a, some mp3). Return `(np.float32 array, sample_rate)`. Support `mono=True|False` — LUFS needs stereo, most other features want mono. Decode at native rate and assert 44.1 kHz; resample **up** if a source file is lower, never down. Add a test that a 22.05 kHz input is upsampled rather than silently accepted.
3. Leave `essentia_backend.py` and `librosa_backend.py` as class stubs implementing the Protocol, all methods raising `NotImplementedError`.
4. `analyze.py` keeps only `analyze_source()` and `analyze_track()`, re-exporting `get_backend`/`available_backends` for compatibility.
5. Add `tests/conftest.py` with synthetic fixtures that have **known ground truth**, so backend tests can assert real numbers rather than "did not crash":
   - `click_track_120bpm` — impulse train, 8 s, 44.1 kHz → BPM 120, onset density 2.0/s
   - `sine_a440` — 8 s pure tone → key A, low spectral centroid variance, high tonal stability
   - `white_noise` — 8 s → high centroid, high rolloff, low tonal stability
   - `swung_click_8ths` — 2:1 long-short IOI pattern at 100 BPM → used by W1D swing detection
   - `stereo_pink_noise` — for LUFS, which needs two channels
   - `silence` — edge case; every feature should degrade to `None`, nothing should divide by zero
6. Add `pyloudnorm` to the `librosa` extra in `pyproject.toml` (librosa has no LUFS meter).
7. Implement `TrackSummary.summary_payload()` in `schemas.py` — this is the one sanctioned edit to that file, and it belongs to W0 alone. Strip `rhythm.beat_times` from every source, replace it with a `beat_count` int, leave everything else intact. Unit-test that the payload contains no `beat_times` key and that `beat_count` matches the original list length.

**Done when:** the four W1 backend/analysis agents can each open exactly one file and not see each other's work. `pytest` still green.

---

## Wave 1 — parallel implementation

### W1A · librosa backend

**Owns:** `src/audio_pipeline/backends/librosa_backend.py`, `tests/test_librosa_backend.py`
**Blocked by:** W0

This is the backend that will actually run on your machine if Essentia won't install, so treat it as first-class, not a stopgap.

**Tasks**

1. `rhythm()` — `librosa.beat.beat_track` for BPM and beat frames; `librosa.onset.onset_detect` for onset times. Onset density = onsets / duration. Transient sharpness = mean ratio of onset-strength peak to its local median (implement as a small helper, document the definition).
2. `tonal()` — `librosa.feature.chroma_cqt` for the 12-bin HPCP mean. **librosa has no key detector**: implement Krumhansl-Schmuckler template correlation against major and minor profiles, take the argmax as key/scale and the correlation margin between best and second-best as `key_confidence`. Tonal stability = 1 − mean frame-to-frame cosine distance of chroma.
3. `spectral()` — `librosa.feature.spectral_centroid`, `spectral_rolloff` (0.85). Brightness = fraction of spectral energy above 1.5 kHz. Populate `band_energy_ratios` from a magnitude STFT summed over the `BAND_EDGES_HZ` bounds, normalised to sum to 1.0. **W1B copies your formula, so write it as a documented module-level helper**, not inline in the method.
4. `dynamics()` — `librosa.feature.rms` for mean/std; `pyloudnorm.Meter` for integrated LUFS (needs the stereo load path); crest factor = peak amplitude / RMS.
5. Set `name = "librosa"`. Populate `unavailable_features` for anything you skip.
6. Tests assert against W0 ground truth: BPM within ±2 of 120, detected key == A for the sine, noise centroid > sine centroid, silence returns `None` without raising.

**Done when:** every Protocol method returns a populated model for all six fixtures.

---

### W1B · Essentia backend

**Owns:** `src/audio_pipeline/backends/essentia_backend.py`, `tests/test_essentia_backend.py`
**Blocked by:** W0

**Tasks**

1. `rhythm()` — `RhythmExtractor2013(method="multifeature")` for BPM, beat times, and confidence. `OnsetRate` for onset density.
2. `tonal()` — `KeyExtractor` for key/scale/strength. `HPCP` (via `SpectralPeaks`) for the 12-bin chroma summary.
3. `spectral()` — `SpectralCentroidTime`, `RollOff`. Brightness and `band_energy_ratios` from the same definitions W1A uses over the shared `BAND_EDGES_HZ` bounds — **the two backends must agree**, so read the helper in `librosa_backend.py` and mirror it exactly, even though you may not edit that file.
4. `dynamics()` — `LoudnessEBUR128` for integrated LUFS (stereo input required; document that clearly). `RMS` framewise for mean/std. Crest factor as peak/RMS.
5. Set `name = "essentia"`. Populate `unavailable_features` for anything unavailable.
6. Mark the whole test module `pytest.importorskip("essentia")` so the suite stays green on machines where Essentia won't install.

**Done when:** on a machine with Essentia, outputs are within a documented tolerance of W1A for the same fixtures. Where they diverge structurally, record why in a comment.

---

### W1C · Demucs separation

**Owns:** `src/audio_pipeline/separate.py`, `tests/test_separate.py`
**Blocked by:** nothing (can start in Wave 0 alongside W0)

**Tasks**

1. `pick_device()` — return `"mps"` when `torch.backends.mps.is_available()`, else `"cpu"`. Must not raise if torch lacks MPS support. Never touch CUDA.
2. `separate()` — use `demucs.api.Separator(model=..., device=...)`. Write four stems as 44.1 kHz wav to `output/<track-name>/stems/`. Track name = input filename stem, slugified.
2b. Model resolution: `model=None` falls through to `default_model()`, which reads `TRACK_FORENSICS_MODEL` from the environment and otherwise returns `DEFAULT_MODEL` (`htdemucs_ft`). Expose `FAST_MODEL` (`htdemucs`) as the documented escape hatch when a run is taking too long. `separate()` must report which model it actually used so `TrackSummary.separation_model` is accurate — never assume the default was the one that ran.
3. Idempotency: if all four stems exist and `force` is False, log a skip and return the existing paths without loading the model.
4. Fallback: wrap the run so any MPS-side failure retries once on CPU with a clear warning. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in-process before the model loads.
5. Return `dict[str, Path]` keyed by `STEM_NAMES`, plus expose the model name and device actually used so `TrackSummary` can record them.
6. `stem_paths()` — pure path construction, no I/O. Test this one properly.
7. Tests: `stem_paths()` and slugification are unit-tested. Device selection is tested with a monkeypatched `torch.backends.mps`. `default_model()` is tested with and without the env var set. The actual Demucs run is marked `@pytest.mark.slow` and skipped by default — register the marker in `pyproject.toml` via W0's config, or note it for W3B.

**Done when:** a real 30 s file separates end-to-end on your Mac and re-running is a no-op.

---

### W1D · Heuristics

**Owns:** `src/audio_pipeline/heuristics.py`, `tests/test_heuristics.py`
**Blocked by:** nothing — pure functions over `schemas.py` models, needs no audio and no backend

The highest-value package to get right, and the cheapest to test. Every label carries the descriptor values that triggered it, so output stays auditable.

**Tasks**

1. `label_generic()` — `percussive` (high crest factor, high onset density), `sustained` (low crest, low onset density), `noisy` (high centroid + low tonal stability), `tonally stable` (high key confidence + high stability).
2. `label_drums()` — `busy drums` / `sparse percussion` from onset density; `kick-heavy` from `spectral.band_energy_ratios.low`; `bright hats` from `band_energy_ratios.high` plus centroid on the drums stem.
3. `label_bass()` — `sparse bass`, `sustained sub` (low centroid, low onset density), `plucked bass` (sharp transients).
4. `label_vocals()` — `speech/vocal dominant` (energy present, mid-band centroid, moderate onset density), `sparse vocal`, `processed/wide vocal`.
5. `label_other()` — `sustained pad-like texture`, `noisy`, `bright plucks`.
6. `apply()` — dispatch on `analysis.source`, always include generic labels, dedupe, sort by confidence descending.
7. Confidence should be a graded function of how far a descriptor sits past its threshold, not a hard 0/1. A value just over the line should not report 1.0.
8. Tune `THRESHOLDS` — the current values are placeholders. Document the reasoning for each in a comment.
9. Tests: build `SourceAnalysis` objects by hand with synthetic descriptor values and assert the expected labels appear. Include boundary cases either side of each threshold, and an all-`None` analysis that must produce zero labels rather than crashing.

**Note:** `SpectralFeatures.band_energy_ratios` already exists in the schema (added at version 2) with bounds fixed in `BAND_EDGES_HZ`. Use it — no schema change needed. Handle the case where a backend leaves all four ratios `None`: those labels simply do not fire.

---

### W1E · Strudel hints

**Owns:** `src/audio_pipeline/strudel_hints.py`, `tests/test_strudel_hints.py`
**Blocked by:** nothing — operates on `TrackSummary`, needs no audio

**Tasks**

1. `suggest_cycle_seconds()` — `beats_per_cycle * 60 / bpm`, `None` when BPM is unknown.
2. `classify_density()` — onsets/sec → `sparse` | `moderate` | `busy`. Share thresholds with `heuristics.py` by importing them; do not redefine constants.
3. `infer_subdivision_feel()` — histogram the inter-onset intervals of the drums stem against the beat grid. Near-uniform short IOIs → `straight 16ths`; bimodal long-short around a 2:1 ratio → `swung 8ths`. **Return `None` when confidence is low** — a wrong grid is worse than no grid, since it sends you down the wrong path in Strudel.
4. `build()` — assemble `StrudelHints` from the summary. `tonal_centre` from the mix's key/scale where confidence clears a threshold, else from the bass stem. `notes` carries short prose caveats, e.g. "key confidence low, verify by ear".
5. Tests: hand-built `TrackSummary` objects. Assert 120 BPM → 2.0 s cycle. Assert a low-confidence, ambiguous IOI distribution returns `None` rather than guessing.

**Explicitly out of scope for v1: do not generate Strudel code.** Descriptive hints only.

---

### W1F · Analysis orchestration

**Owns:** `src/audio_pipeline/analyze.py`, `tests/test_analyze.py`
**Blocked by:** W0

**Tasks**

1. `analyze_source()` — load audio via `audio_io`, call all four backend methods, assemble `SourceAnalysis`, call `heuristics.apply()` for labels, collect `unavailable_features`.
2. `analyze_track()` — analyze mix plus every stem present. Missing stems are skipped with a warning, not an error, so `analyze` works on a bare mix before separation has run.
3. Write `analysis/<source>.json` with the full `SourceAnalysis`, beat times included. Write `track_summary.json` via `TrackSummary.summary_payload()`, which strips beat times and leaves a `beat_count` — the full lists stay in the per-source files, so nothing is lost and the one file you read by hand stays readable. Pretty-printed, stable key order, 6-decimal float rounding so diffs between runs stay readable.
4. Wrap per-source failures: one stem blowing up must not lose the other four. Record the failure in that source's `unavailable_features`.
5. Tests use a fake backend implementing the Protocol with canned values — no real audio, no real Essentia, no real librosa. This is what makes the orchestration testable in isolation.

---

## Wave 2 — integration

### W2 · CLI and end-to-end wiring

**Owns:** `src/audio_pipeline/cli.py`, `tests/test_cli.py`
**Blocked by:** all of Wave 1

**Tasks**

1. Implement `separate`, `analyze`, `export-strudel-hints`, `all`, `doctor`.
2. `all` = separate → analyze → hints, with a progress line per phase and a summary of what was written.
3. `doctor` reports: Python version, whether FFmpeg is on PATH, torch version, MPS availability, resolved Demucs device, the resolved Demucs model and whether it came from the env var, installed backends, and the active backend. This is the first thing to run when something breaks, so make the output genuinely diagnostic.
4. Input validation: reject suffixes outside `SUPPORTED_INPUT_SUFFIXES` with a clear message listing what is accepted.
4b. Expose `--model` on `separate` and `all`, defaulting to `None` so `default_model()` resolves it. Add a `--fast` flag as sugar for `--model htdemucs`. Since the default is now the 4x-slower `htdemucs_ft`, print the model and device before the run starts and warn that a long wait is expected — a silent four-minute pause reads as a hang.
5. Exit codes: 0 success, 1 user error (bad input, missing file), 2 environment error (no backend, no FFmpeg).
6. `export-strudel-hints` reads an existing `track_summary.json` and errors helpfully if it is absent, naming the command to run first.
7. Tests use Typer's `CliRunner` with mocked pipeline functions. Assert exit codes and message content, not implementation details.

**Done when:** `track-forensics all examples/<file>.wav` produces the full output tree described in the README.

---

## Wave 3 — polish

### W3A · README verification

**Owns:** `README.md`, `CLAUDE.md`
**Blocked by:** W2

Run the documented install from a genuinely fresh venv on Apple silicon and fix whatever is wrong. Record the actual Essentia outcome — whether it installed, and if not, the exact error — so the fallback guidance is grounded in what really happened rather than what was assumed. Add a real sample output tree and an example `strudel_hints.json`. Update the Status section away from "scaffold only".

### W3B · Tooling and CI

**Owns:** `pyproject.toml`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`
**Blocked by:** W2

Register the `slow` pytest marker and default to deselecting it. Add a GitHub Actions workflow running ruff, mypy, and the fast test suite on macOS — Essentia-skipped, Demucs-skipped, so CI stays quick. Optional pre-commit hooks for ruff and mypy. Add coverage reporting if it is cheap.

### W3C · Real-material calibration

**Owns:** `src/audio_pipeline/heuristics.py` thresholds, `examples/`
**Blocked by:** W2

Run the pipeline over several tracks you already know well and check the labels against your own ears. The Wave 1 thresholds are guesses; this is where they become useful. Record a short calibration note in `examples/` covering which labels fired correctly, which misfired, and what changed. This package is judgement work — it needs your ears, so treat it as collaborative rather than fully autonomous.

---

## Settled decisions

Closed. Recorded here so nobody relitigates them mid-wave.

1. **`band_energy_ratios` — added.** `SpectralFeatures` now carries a `BandEnergyRatios` model (low / low_mid / high_mid / high), with bounds fixed in `BAND_EDGES_HZ` so both backends stay comparable. `SCHEMA_VERSION` bumped to 2. Already committed; W1D can use it directly.
2. **Beat times — per-source files only.** They stay complete in `analysis/*.json` and are stripped from `track_summary.json`, which keeps a `beat_count` instead. Nothing is lost; the summary just stops being 3,600 floats of duplicated data across five sources. Implemented via `TrackSummary.summary_payload()`, which W0 fills in.
3. **Demucs model — quality by default.** `DEFAULT_MODEL` is `htdemucs_ft`, the fine-tuned model: better separation, roughly 4x slower. Override per-run with `--model`, reach for `--fast` when a run is dragging, or set a standing preference with `TRACK_FORENSICS_MODEL`. Because the slow path is now the default, the CLI must announce the model and expected wait up front.
4. **Analysis sample rate — 44.1 kHz throughout, no exceptions.** Accuracy on hats and cymbals is the point of this tool. Downsampling to 22.05 kHz would cap the spectrum at 11 kHz and quietly wreck every high-band ratio and centroid. `audio_io` upsamples anything lower and never downsamples. This is in the ground rules because it is the easiest constraint for an agent to break by accident while reaching for speed.

## Risks

- **Essentia install on macOS is the single most likely thing to fail.** W1A exists so the project still works when it does. Do not let Essentia block the critical path.
- **MPS gaps in Demucs.** Some ops fall back to CPU or fail outright depending on torch version. W1C's retry-on-CPU path is load-bearing, not defensive decoration.
- **Backend divergence.** Essentia and librosa will not produce identical numbers for "the same" descriptor. Thresholds tuned on one backend may misfire on the other. W3C should calibrate on whichever backend you actually run, and the other should be treated as approximate until separately checked.
- **Separation is now slow by default.** `htdemucs_ft` on a full track is minutes, not seconds. Any agent running the pipeline end-to-end should use a short clip, and the `slow` pytest marker exists so CI never touches it.
