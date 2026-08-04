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
  "schema_version": 4,
  "track_name": "synth-mix-100bpm",
  "bpm": 100.0336,
  "suggested_cycle_seconds": 2.399194,
  "subdivision_feel": "straight 8ths",
  "drum_density": "moderate",
  "bass_activity": "moderate",
  "tonal_centre": "A minor",
  "drum_grid": {
    "status": "ok",
    "steps_per_cycle": 16,
    "kick_steps": [4, 12],
    "snare_steps": [],
    "hat_steps": [0, 8],
    "unclassified_count": 20,
    "caveats": [
      "20 of 39 hits are unclassified. Three classes cannot describe a full kit: toms, rides, crashes, claps and shakers all land here, and on percussive material that is correct rather than a failure.",
      "most hits are unclassified, so read the kick/snare/hat pattern as a partial transcription of this source rather than a complete one"
    ]
  },
  "bass_line": {
    "status": "ok",
    "note_sequence": ["a2", "f2", "g2", "a2"],
    "truncated_from": null,
    "median_midi_note": 45,
    "caveats": []
  },
  "sound_suggestions": [
    {
      "role": "kick", "match": "exact", "sound": "bd",
      "reason": "9 hit(s) classified as kick; 'bd' is Strudel's default kick/bass-drum sample.",
      "alternatives": [],
      "evidence": { "hit_count": 9.0, "mean_kick_ratio": 0.996146, "mean_confidence": 1.0 }
    },
    {
      "role": "hat", "match": "exact", "sound": "oh",
      "reason": "10 hit(s) classified as hat; median decay ratio 3.99 reads as open, so 'oh' is the closer default sample. Open versus closed is decided by decay length alone, which this pipeline measures directly.",
      "alternatives": [],
      "evidence": { "hit_count": 10.0, "mean_confidence": 0.929763, "median_decay_ratio": 3.991593 }
    },
    {
      "role": "unclassified", "match": "none", "sound": null,
      "reason": "20 hit(s) did not clear the kick/snare/hat decision margin. This pipeline recognises only three drum classes; toms, rides, crashes, claps and shakers all land here on percussive material, correctly, not as a failure. Source by ear from Strudel's percussion set.",
      "alternatives": ["ht", "mt", "lt", "cr", "rd", "perc", "sh", "cb", "tb"],
      "evidence": { "hit_count": 20.0, "mean_confidence": 0.001084 }
    },
    {
      "role": "bass", "match": "approximate", "sound": "sine",
      "reason": "Energy is concentrated at the fundamental (low-band ratio 1.00, brightness 0.00, centroid 107 Hz) with little harmonic content: reads as sub bass. 'sine' is the closest default waveform.",
      "alternatives": [],
      "evidence": { "low_band_ratio": 0.998611, "brightness": 0.000002, "centroid_mean_hz": 106.730525 }
    }
  ],
  "strudel_vocabulary_read": "2026-08-04",
  "notes": []
}
```

An excerpt of `track_summary.json` for the `drums` source shows the shape every source follows — rhythm/tonal/spectral/dynamics blocks plus auditable labels:

```json
"drums": {
  "schema_version": 4,
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

`track_summary.json` strips `drum_decomposition.hits` down to a `total_hit_count`; the full per-hit list, and the cycle-folded pattern it's built from, live only in `analysis/drums.json`:

```json
"drum_decomposition": {
  "status": "ok",
  "steps_per_cycle": 16,
  "patterns": [
    { "drum": "kick", "steps": [4, 12], "step_occupancy": [0.833333, 0.666667], "hit_count": 9 },
    { "drum": "hat", "steps": [0, 8], "step_occupancy": [0.833333, 0.833333], "hit_count": 10 },
    { "drum": "unclassified", "steps": [2, 6, 10, 14], "step_occupancy": [0.833333, 0.833333, 0.833333, 0.833333], "hit_count": 20 }
  ],
  "hits": [
    { "time_seconds": 0.290249, "drum": "unclassified", "confidence": 0.0, "step": 14,
      "kick_ratio": 0.026058, "body_ratio": 0.005693, "noise_ratio": 0.013401, "air_ratio": 0.954848,
      "decay_ratio": 24.57339, "flatness": 0.086486 }
  ],
  "unclassified_count": 20
}
```

Note the real result: on this run the classifier found kick and hat cleanly but reported **zero snares** — see "Status" below on why that's expected right now, not a bug.

`analysis/bass.json` carries the full note sequence behind `bass_line` in `strudel_hints.json` (which caps at 32 entries), plus per-note diagnostics:

```json
"bass_line": {
  "status": "ok",
  "notes": [
    { "start_seconds": 0.01161, "duration_seconds": 4.783311, "midi_note": 45, "note_name": "a2",
      "median_f0_hz": 111.040894, "cents_offset": 16.305071, "confidence": 0.937927, "step": 12 },
    { "start_seconds": 4.794921, "duration_seconds": 2.380045, "midi_note": 41, "note_name": "f2",
      "median_f0_hz": 88.951698, "cents_offset": 32.308638, "confidence": 0.906406, "step": 12 }
  ],
  "median_midi_note": 45,
  "median_cents_offset": 21.740231,
  "voiced_fraction": 0.994203,
  "octave_corrections": 0,
  "caveats": []
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

### Drum element decomposition (drums stem only)

The `drums` source additionally gets a `drum_decomposition` block: every onset in that stem classified as `kick`, `snare`, `hat`, or `unclassified`, with per-hit timing, a winning-class confidence, and the evidence (band-energy ratios, decay ratio, spectral flatness) behind the call. Classification runs on independent per-band spectral flux — one detector per frequency band, peak-picked separately — specifically so that a kick and a hat landing on the same instant are recovered as two hits instead of one detector's onset swallowing the other's. A hit that doesn't clear the decision margin lands in `unclassified` with its timing preserved rather than being dropped; three drum classes cannot describe a full kit, so toms, rides, crashes, claps and shakers all end up there, correctly, not as a failure.

Hits are also cycle-folded onto a 16- or 12-step grid (whichever quantises better) into a `DrumPattern` per class — step positions plus `step_occupancy`, so an occasional ghost kick reads as partial membership rather than full. If no reliable grid can be fit, `status` reports `no_grid` and hits are still returned ungridded; a wrong grid is treated as worse than none. This block is pure numpy with no analysis-backend dependency, so it's identical whichever of essentia/librosa is active. The full per-hit list lives in `analysis/drums.json`; `track_summary.json` strips it down to a `total_hit_count` to stay small, keeping the cycle-folded pattern (the compact, useful part) intact.

### Bass note extraction (bass stem only)

The `bass` source additionally gets a `bass_line` block: a note sequence (MIDI number, Strudel-style note name like `a2`, start time, duration, and per-note confidence) extracted from a raw pitch (F0) track. Pitch tracking is a fifth method on the analysis backend Protocol — `pitch()`, implemented via `librosa.pyin` or Essentia's `PitchYinFFT` — restricted to the bass register (30-400 Hz) as an octave guard. Turning that F0 track into discrete notes (median filtering, an exact-octave snap against a running median, segmenting on note changes, merging short glide/vibrato segments) is shared, backend-independent numpy, so both backends produce byte-identical note sequences from identical F0.

Note extraction is the one place cost is worth calling out: on librosa, `pitch()` runs at roughly 0.12x real time (a genuinely material cost on a long track); on Essentia it's negligible. For that reason it only ever runs against the bass stem, never all five sources. `bass_line.median_cents_offset` flags a track tuned away from A440 (e.g. a 432 Hz master) as a consistent offset rather than a wrong note; `octave_corrections` counts how often the snap step intervened. A stem with no usable pitch (noise, silence) comes back `status="unvoiced"` rather than inventing notes.

## Strudel hints

`strudel_hints.json` is intentionally small and hand-readable: drum density, likely subdivision feel, bass activity, tonal centre, suggested cycle length, a cycle-folded `drum_grid`, a capped `bass_line` note sequence, and `sound_suggestions` mapping what was measured onto Strudel's default sound vocabulary. It's a starting point for writing patterns yourself, not generated code.

### The Strudel-mapping philosophy

Every entry in `sound_suggestions` carries a `match`:

- **`match="exact"`** — a real Strudel default sample or waveform for what was measured. A hit classified as `kick` maps to `bd`. A hat's decay length picks `hh` (short) or `oh` (long) directly, because that's what the open/closed distinction physically is.
- **`match="approximate"`** — the closest default Strudel offers, not a perfect model. Sub bass with energy concentrated at the fundamental maps to the `sine` waveform because that's the nearest built-in shape, not because the source was literally a sine oscillator.
- **`match="none"`, `sound=null`** — the tool has no basis for a claim and says so, listing `alternatives` you can choose from by ear instead of guessing on your behalf. This is deliberate rather than a gap to be filled later: distinguishing `square` from `sawtooth` bass would need a harmonic-ratio measurement this pipeline doesn't make, and inferring a specific drum machine via `.bank()` from a spectrum would be an invented claim — a `sh`/`cb`/`ht`/`mt`/`lt`/`cr`/`rd`/`perc` unclassified drum hit gets exactly this treatment.

Refusing to guess is the selling point here, not a limitation: every suggestion in the file is either grounded in a real measurement or is honestly flagged as unfounded, so nothing downstream has to guess which is which.

The vocabulary tables themselves are transcribed from the live Strudel docs, not written from memory, and every output file records the date they were read in `strudel_vocabulary_read` (currently `2026-08-04`). Strudel is actively developed and there is no offline way to detect that the tables have drifted — treat that date as a "recheck against the live docs" reminder, especially if it's more than a few months old.

## Development

```bash
pytest
ruff check .
mypy src
```

## Status

Implemented and working end to end: `track-forensics all input.wav` runs separation, per-source analysis, heuristic labelling, drum decomposition, bass note extraction, Strudel sound mapping, and hints export, and produces the full output tree described above. Schema is at `schema_version` 4. `pytest`, `ruff check .`, and `mypy src` are all clean.

Drum decomposition and bass note extraction are new in this schema version and honest about where they stand: numerically verified against synthetic fixtures (including a fixture specifically constructed so a naive single-onset-list classifier cannot pass it) and structurally verified end to end against real Demucs separation. They are **not yet calibrated against a wide range of real tracks** — that's an explicit, ongoing, separate activity, done by ear against known material. On the one real mix verified so far, the drum classifier recovered kick and hat correctly but reported zero snares, and grid positions didn't perfectly match known ground truth. That's the expected state pre-calibration, not a bug, and it's why the drum classifier is deliberately conservative about what it claims (three classes only, an honest `unclassified` bucket, no invented drum-machine identity) rather than confidently wrong.

See `CLAUDE.md` for the implementation brief and module contracts if you're extending it.
