# track-forensics — implementation brief

Read this before writing code. The pipeline is implemented and working end to end (`track-forensics all input.wav` produces the full output tree); this document is the standing contract for anyone extending it, not a stub-filling checklist. `schema_version` is currently 5. See `README.md` for install steps, verified `doctor` output, and real sample output.

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
| `backends/` | `__init__.py` holds the `AnalysisBackend` Protocol, `available_backends()`, and `get_backend()` (lazy, try/except-guarded imports, prefers Essentia). `essentia_backend.py` and `librosa_backend.py` are the two concrete implementations — same Protocol, comparable output, deliberately no schema drift between them. The Protocol has five methods: `rhythm`, `tonal`, `spectral`, `dynamics`, and `pitch` (raw F0 track, added in schema v4 for bass note extraction). |
| `analyze.py` | Orchestration only, not extraction: resolves a backend, resolves **one track-level tempo and downbeat** via `tempo.py`, loops mix + each present stem, calls the backend's feature methods, runs `heuristics.apply()`, calls `drum_elements.decompose()` for the drums source and `backend.pitch()` + `note_track.segment_notes()` for the bass source (a policy here, not in the Protocol, because pitch tracking is costly on the librosa backend), builds the track's `Arrangement` via `arrangement.py`, and writes `analysis/<source>.json` plus `track_summary.json`. It also **arbitrates the tempo octave** by fitting a drum grid at each live candidate — see `TempoFit.octave_arbitration`. |
| `tempo.py` | Pure numpy, no backend dependency, new in v5. Refines the backend's coarse BPM against the low-band flux envelope's autocorrelation (`refine_bpm`), finds the downbeat (`find_downbeat`, returning a `DownbeatFit` with `beat_offset_seconds` and `phase_confidence` kept apart), reports tempo stability across halves, and surfaces x0.5/x1/x2 `octave_candidates` as **evidence rather than a correction** — it never moves `bpm` itself. Imports `_stft_magnitude`, `_band_envelope` and `_spectral_flux` from `drum_elements.py`; those three names are a pinned shared surface. |
| `arrangement.py` | Pure numpy, no backend dependency, new in v5. Per-bar presence for each stem plus the kick band, folded into labelled sections (`intro`, `full`, `breakdown`, `drop`, `outro`, `silence`, ...). Presence needs **two** tests, not one: each stem against a percentile of its own distribution, *and* each stem against the loudest stem in the same record. The first alone is scale-free and will faithfully report an arrangement for a stem holding only separation bleed. |
| `heuristics.py` | Pure functions: raw descriptors in, human-readable production clues out. No I/O, no audio. Easiest thing to unit test — do it properly. |
| `drum_elements.py` | Pure numpy, no backend dependency: classifies onsets in the drums stem into kick/snare/hat via per-band spectral flux (peak-picked per band, so coincident kick+hat hits are both recovered), with an honest `unclassified` bucket and a cycle-folded grid pattern per class. |
| `note_track.py` | Pure numpy, no backend dependency: turns a backend's raw F0 track into a `BassLine` note sequence (MIDI note, note name, timing) via median filtering, an exact-octave snap, and note segmentation — shared so both backends yield byte-identical notes from identical F0. |
| `strudel_vocab.py` | Pure functions over models: maps drum/bass measurements onto Strudel's default sound vocabulary (`match="exact" \| "approximate" \| "none"`), transcribed from the live Strudel docs with a pinned read date, not from memory. |
| `strudel_hints.py` | Condenses the full analysis into a small `strudel_hints.json`, including drum grid, bass line, and sound suggestions built via `strudel_vocab.py`. |
| `schemas.py` | All pydantic models. Single source of truth for the JSON shape. Version the schema with a `schema_version` field. |

## Features to extract (per source)

- Rhythm: BPM estimate, beat positions (seconds), onset density (onsets/sec), transient sharpness
- Tonal: key, scale, key confidence, chroma/HPCP 12-bin summary, tonal stability
- Spectral: spectral centroid (`centroid_mean`/`centroid_std` **and** `centroid_energy_hz`), spectral rolloff (`rolloff_mean` **and** `rolloff_energy_hz`), brightness, band energy ratios over `BAND_EDGES_HZ`
- Dynamics: integrated loudness, RMS, crest factor

Beat times are written to `analysis/*.json` only; `track_summary.json` carries a `beat_count` instead. See `TrackSummary.summary_payload()`.

If a feature is unavailable on the active backend, emit `null` and record it in a `unavailable_features` list rather than crashing.

Schema v4 added two more `SourceAnalysis` fields, populated only for specific sources rather than all five:

- `drum_decomposition` (drums source only): per-hit kick/snare/hat/unclassified classification plus a cycle-folded grid pattern, from `drum_elements.py`.
- `bass_line` (bass source only): a MIDI note sequence with timing, from `backend.pitch()` + `note_track.py`.

Both carry an explicit `status` (`not_attempted | ok | no_grid | too_few_hits | unvoiced | silent | failed`) rather than signalling absence through an empty list — see `analyze._collect_unavailable()`, which does not route these through the empty-list-means-unavailable check that the four Protocol features use, because an empty list is a legitimate result here (e.g. no snares found), not a missing feature.

## Schema v5

Three additions and one policy. `calibration/v5-vs-v4.md` documents the measured delta; `calibration/v5-progress.md` is the running record of the cycle and **overrides `V2-PLAN.md` wherever they disagree.**

**Track-level tempo, resolved once.** `TrackSummary` gained `tempo: TempoFit` and `downbeat: DownbeatFit`, and they are the numbers every grid in the output is built on. The tempo is now **refined** by `tempo.py` from the audio, not taken from the backend, and it is resolved **once per track** rather than per source — v4 had every source reporting its own estimate (mix 131.855, drums 132.040, bass 131.815) with different modules silently consuming different ones. A 0.040 BPM error accumulates 82 ms over 147 bars, which is what made v4 declare that a textbook four-on-the-floor grid did not exist. Read `TempoFit.status` and `confidence_label` before `bpm`: `coarse` means refinement was **refused** and `bpm` is just the backend's estimate, which on a genuinely drifting record is the correct answer.

**Arrangement.** `TrackSummary.arrangement: Arrangement` — per-bar presence folded into labelled sections, from `arrangement.py`.

**Silence gating.** `SILENCE_RMS_FLOOR = 1e-3` (−60 dBFS), measured across all 35 stems under `calibration/`: six residue stems at 8e-06–1.38e-04 against a quietest-real of 3.82e-03, a 27.7× gap with nothing in it. A source below it gets `status="silent"` on its per-source blocks, is excluded from the `tonal_centre` fallback and from density reporting in `strudel_hints`, and appears in `unavailable_features`. This is not theoretical: a −70 LUFS bass stem's "E minor" used to beat the mix's own F major into `strudel_hints.json`, and an ambient record's empty vocals stem produced a full analysis with a negative-confidence tempo.

### Additive only, and what it costs

**A corrected descriptor lands beside the one it supersedes, never in place of it.** `centroid_energy_hz` joined `centroid_mean`; `rolloff_energy_hz` joined `rolloff_mean`; `TempoFit.bpm` joined every source's `rhythm.bpm`. That rule is what keeps `calibration/v4/` usable as a baseline, and it is not negotiable inside a cycle.

The cost is real and worth stating plainly: **`centroid_mean`, `rolloff_mean` and `rhythm.bpm` survive as fields that look authoritative and are not.** Nothing in the type system distinguishes them from the corrected ones. Docstrings in `schemas.py` are the only guard, and docstrings are easy to ignore — `heuristics.py` had four thresholds tuned against a `centroid_mean` contaminated by silent frames, and one of them (`dark_centroid_hz`) made its label unreachable on every real bass stem ever measured. Revisit at v6 and consider removing them once nothing reads them.

**They are not two scales of one quantity.** The Madonna drums stem reads `centroid_mean` 4390 Hz and `centroid_energy_hz` 412 Hz and *both are correct*, because 89% of that stem's energy is kick. A threshold cannot be migrated by rescaling; it has to be re-derived from measurement. See the `THRESHOLDS` centroid block in `heuristics.py`, which records one migration that was made, one that was measured and refused, and one that had nowhere to go.

## Heuristic labels

Derived, not measured. Examples: `kick-heavy`, `bright hats`, `sustained pad-like texture`, `speech/vocal dominant`, `sparse bass`, `busy drums`, `percussive`, `noisy`, `tonally stable`. Each label should carry the descriptor values that triggered it so the output is auditable.

## Strudel hints shape

Small and hand-readable. Drum density, likely subdivision feel (e.g. straight 16ths vs swung 8ths), bass activity, tonal centre, suggested cycle length. Schema v4 added a cycle-folded `drum_grid`, a `bass_line` note sequence (capped at 32 entries with `truncated_from`), a `sound_suggestions` list mapping measurements onto Strudel's default sound vocabulary (`match="exact" | "approximate" | "none"`, built by `strudel_vocab.py`), and `strudel_vocabulary_read` recording the date that vocabulary was transcribed from the live docs. Schema v5 added `tempo_status` and `tempo_confidence` beside `bpm` (v4 printed a bare `bpm: 143.25` for a track whose refinement had explicitly declined), an `arrangement` block capped at 24 one-line sections, and `bass_line.step_share`. Still hand-readable by design — this is measurements and sound names, not Strudel syntax. **Do not generate Strudel code.**

This file is where most readers meet the analysis, so it owes them one coherent account of anything two modules disagree about. `_octave_note()` is the worked example: `tempo.py` reports that it did not shift the tempo and `analyze.py` reports that it corrected the octave ×2, both true of their own layer and both landing in the same `TempoFit.caveats` list. `track_summary.json` keeps both for audit; the hints file emits one note, built from the structured `octave_arbitration` fields rather than by matching on either module's prose.

## Non-goals

Unchanged from v1 except where noted.

- **Generating Strudel patterns or any code output.** Codegen (`V2-PLAN.md` W7) stays deferred and stays in that document's appendix.
- **Harmony / chord extraction.** W5B, likewise deferred and likewise in the appendix.
- Web UI, GUI, notebooks
- Any cloud/API service
- Batch processing of directories (single file in, single output tree out)
- **Terminal visualisation.** W8A specced a `track-forensics show` command and it was **cut** from the v4→v5 cycle as unnecessary: `strudel_hints.json` is already hand-readable, which was the need the command existed to meet. There is no `viz.py`. If it is ever revived, terminal output only — the GUI/HTML/notebook non-goals above are not weakened by it.

## Definition of done

- `track-forensics all examples/<some file>.wav` produces the full output tree
- `track-forensics doctor` reports Python version, Demucs device, and active analysis backend
- `pytest` green, `ruff check .` clean, `mypy src` clean
- README install steps actually work from a fresh venv

## Every threshold is documented where it is defined

With its reasoning, its provenance, and the measurement behind it. Three of the four bugs found calibrating v1 were thresholds that were plausible in the abstract and wrong against real material, and an undocumented constant is a future bug with no paper trail. A `[guess]` marker is not a defect — it is the honest state of a number nothing has yet tested, and it tells the next reader what to go and measure.

**A measured refusal is a result.** Four packages in the v4→v5 cycle built the thing they were asked for, measured that it did not work, and reverted it: no octave corrector exists on the autocorrelation statistic; a kick-detection fix produced 702 unplaceable hits; centroid alone cannot separate bass waveforms across a 3:1 register; 44 correct-but-useless arrangement sections on drum & bass were reported rather than tuned away. Do not widen a threshold to make a branch fire. If a branch has never fired, find out whether its *input* is broken before touching its bounds — that was F4, and it was the descriptor.

## Calibration corpus

Six tracks under `calibration/v5/`, JSON only (`.gitignore` excludes the stems; no source material is ever committed). One row per failure mode: house at a fixed tempo, swung hip-hop, live band with a floating tempo, ambient with no pulse, drum & bass at 170, plus a funk record. Every row is a full-length record; two short clips were carried through the v4 → v5 cycle and dropped afterwards, since Demucs put nearly all of both into `other` and they were too short for tempo drift to show. **The ambient row is the important one and the easiest to skip** — it exists so that "the tool correctly refused" is a tested outcome. A tool only ever tested on material it handles well will quietly learn to always answer.
