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

### Schema v5 migration

**v5 adds fields and removes none, so a v4 reader keeps working unchanged.**
Every field a v4 consumer expects is still present, still in the same place, and
still means the same thing. Nothing was renamed and nothing was replaced in
place — a corrected descriptor lands *beside* the one it supersedes, and the
superseded field's description says what is wrong with it and names the
replacement.

What is new:

| where | field | why |
|---|---|---|
| `track_summary.json` | `tempo` | One refined tempo for the record. `rhythm.bpm` on every source is untouched and still carries that source's own backend estimate. |
| `track_summary.json` | `downbeat` | Where bar one starts, with the beat-phase and bar-phase confidences kept apart. |
| `track_summary.json` | `arrangement` | Sections, each with the presence pattern that labelled it. |
| `strudel_hints.json` | `tempo_status`, `tempo_confidence` | So a printed BPM can be told apart from a measured one. |
| `strudel_hints.json` | `arrangement` | One line per section. |
| `strudel_hints.json` | `bass_line.steps`, `bass_line.step_share` | Which grid steps the bass lands on, and what share of its notes each holds. A measured share rather than a thresholded list. |
| `analysis/*.json` | `spectral.centroid_energy_hz`, `spectral.rolloff_energy_hz` | Energy-weighted, and immune to the silent-frame contamination `centroid_mean`/`rolloff_mean` suffer. Both old fields are still there and still populated. |
| `analysis/*.json` | — | Otherwise unchanged in shape. |

The three new blocks are at **track level**, not per source. In v4 each of the
five sources refined its own tempo and they disagreed; whichever a downstream
module happened to read, it built a grid that drifted apart from the audio. A
`harmony` block is reserved in the same position.

Two blocks may now report `status: "silent"` — a source below the documented
RMS floor is separation residue, and everything derived from it is skipped
rather than computed from a noise floor.

**`calibration/v4/` is a frozen reference and is never regenerated in place.**
Every claim in the v4→v5 findings is measured against those exact files, so
overwriting them destroys the only baseline that can show whether the fixes
worked. v5 runs write to `calibration/v5/`.

### Real sample output

From `track-forensics all Madonna_-_I_Feel_So_Free_Peggy_Gou_Energy_Mix_Official.wav` — the 4:27
house record this tool was calibrated against, and the source of most of the v4 → v5 findings. The
full files are committed under `calibration/v5/`; this is `strudel_hints.json`, abridged only where
a list repeats.

```jsonc
{
  "schema_version": 5,
  "track_name": "madonna-i-feel-so-free-peggy-gou-energy-mix-official",
  "bpm": 131.99969088787668,
  "tempo_status": "refined",
  "tempo_confidence": "high",
  "suggested_cycle_seconds": 1.818186,
  "subdivision_feel": null,
  "drum_density": "busy",
  "bass_activity": "moderate",
  "tonal_centre": "A minor",
  "drum_grid": {
    "status": "ok",
    "steps_per_cycle": 16,
    "kick_steps": [0, 1, 4, 6, 7, 8, 11, 12, 15],
    "snare_steps": [0, 1, 2, /* ... */ 14, 15],
    "hat_steps":   [0, 1, 2, /* ... */ 14, 15],
    "unclassified_count": 65,
    "caveats": [
      "877 detections in the noise/air bands were the kick's own transient found a second time, not hits of their own — their windows are over 0.6 kick energy and their air/(air+noise) is under 0.5, which is a beater click rather than a hat",
      "65 of 1418 hits are unclassified. Three classes cannot describe a full kit: toms, rides, crashes, claps and shakers all land here, and on percussive material that is correct rather than a failure."
    ]
  },
  "bass_line": {
    "status": "ok",
    "note_sequence": ["a1", "a2", "a1", "a2", "a1", "a2", "a1", "a2", "g2", /* ... 32 of */],
    "truncated_from": 709,
    "median_midi_note": 41,
    "steps":      [0,      1,      2,      4,      5,      6,      8,      9,      10,     12,     13,     14,     15],
    "step_share": [0.0959, 0.0028, 0.1453, 0.0959, 0.0028, 0.1453, 0.0790, 0.0042, 0.1453, 0.0564, 0.0085, 0.1453, 0.0733],
    "caveats": []
  },
  "arrangement": {
    "status": "ok",
    "bar_count": 147,
    "bar_seconds": 1.8181860759345343,
    "sections": [
      "bar 0 x17 intro: drums, other, kick",
      "bar 17 x2 breakdown: drums",
      "bar 19 x6 groove: drums, bass, kick",
      "bar 25 x2 groove: drums, bass, other, kick",
      "bar 27 x48 full: drums, bass, vocals, other, kick",
      "bar 75 x14 breakdown: vocals, other",
      "bar 89 x2 breakdown: drums, vocals, other",
      "bar 91 x9 drop: drums, bass, vocals, other, kick",
      "bar 100 x2 breakdown: drums",
      "bar 102 x10 groove: drums, bass, other, kick",
      "bar 112 x6 groove: drums, bass, vocals, kick",
      "bar 118 x4 full: drums, bass, vocals, other, kick",
      "bar 122 x14 groove: drums, bass, other, kick",
      "bar 136 x6 full: drums, bass, vocals, other, kick",
      "bar 142 x2 outro: other",
      "bar 144 x3 silence: nothing playing"
    ],
    "truncated_from": null,
    "absent_tracks": [],
    "caveats": []
  },
  "sound_suggestions": [
    {
      "role": "kick", "match": "exact", "sound": "bd",
      "reason": "483 hit(s) classified as kick; 'bd' is Strudel's default kick/bass-drum sample.",
      "alternatives": [],
      "evidence": { "hit_count": 483.0, "mean_kick_ratio": 0.842356, "mean_confidence": 0.763809 }
    },
    {
      "role": "snare", "match": "exact", "sound": "sd",
      "reason": "87 hit(s) classified as snare; 'sd' is Strudel's default snare sample.",
      "alternatives": [],
      "evidence": { "hit_count": 87.0, "mean_body_ratio": 0.673968, "mean_confidence": 0.777226 }
    },
    {
      "role": "hat", "match": "exact", "sound": "oh",
      "reason": "783 hit(s) classified as hat; median decay ratio 2.20 reads as open, so 'oh' is the closer default sample. Open versus closed is decided by decay length alone, which this pipeline measures directly.",
      "alternatives": [],
      "evidence": { "hit_count": 783.0, "mean_confidence": 0.952919, "median_decay_ratio": 2.195861 }
    },
    {
      "role": "unclassified", "match": "none", "sound": null,
      "reason": "65 hit(s) did not clear the kick/snare/hat decision margin. This pipeline recognises only three drum classes; toms, rides, crashes, claps and shakers all land here on percussive material, correctly, not as a failure. Source by ear from Strudel's percussion set.",
      "alternatives": ["ht", "mt", "lt", "cr", "rd", "perc", "sh", "cb", "tb"],
      "evidence": { "hit_count": 65.0, "mean_confidence": 0.099924 }
    },
    {
      "role": "bass", "match": "approximate", "sound": "sine",
      "reason": "Energy is concentrated at the fundamental (low-band ratio 0.92, brightness 0.00, median centroid 139 Hz) with little harmonic content: reads as sub bass. 'sine' is the closest default waveform.",
      "alternatives": [],
      "evidence": { "low_band_ratio": 0.917462, "brightness": 0.002052, "centroid_energy_hz": 138.759392, "centroid_mean_hz": 1004.751763 }
    }
  ],
  "strudel_vocabulary_read": "2026-08-04",
  "notes": [
    "tempo octave: correlation could not settle it, so the drum grid was fitted at each candidate and the backend's own octave won (other live candidates: 66.00, 263.99 BPM). bpm above is the right one",
    "drum inter-onset intervals fit neither an even nor a swung grid, subdivision not inferred — verify by ear"
  ]
}
```

Four things in there are worth pointing at, because each is a specific thing the tool was getting
wrong a version ago.

**`bpm` is 131.99969, not 131.855.** v4 printed the mix stem's backend estimate. The record is
132.000. That 0.145 BPM difference — and the 0.040 BPM difference against the *drums* stem's own
estimate, which is what the grid was actually built from — accumulated 82 ms of drift over 147 bars
and made the tool report `no_grid` for a textbook four-on-the-floor pattern. `rhythm.bpm` on every
source still carries its own backend estimate, untouched.

**`subdivision_feel` is `null`, and the note says so.** The tool will not name a grid it cannot
read. Across eight corpus tracks it has named one (`straight 8ths`, on Erykah Badu). That is the
intended hit rate: a wrong grid sends you down the wrong path for a bar of patterns before you
notice.

**`step_share`, not a list of "the steps the bass plays".** 0.1453 on steps 2, 6, 10 and 14 against
0.003–0.096 elsewhere — offbeat 16ths, and the same four steps the hats land on, from a completely
independent code path. It is a measured share rather than a thresholded list because the threshold
that returns exactly those four steps would have exactly one record behind it.

**The arrangement is the structure, not a guess at it.** 17-bar intro, a 48-bar full-band section
from bar 27, a 14-bar breakdown at 75 with kick and bass absent in every bar, a drop at 91, and
three bars of silence at the end. Each section carries the presence pattern that labelled it in
`track_summary.json`.

An excerpt of `analysis/drums.json` shows the shape every source follows — rhythm/tonal/spectral/
dynamics blocks plus auditable labels:

```json
"source": "drums",
"rhythm": {
  "bpm": 132.015411,
  "bpm_confidence": 2.82432,
  "onset_density": 5.315915,
  "transient_sharpness": 3.120116
},
"spectral": {
  "centroid_mean": 4389.982379,
  "centroid_energy_hz": 411.793353,
  "rolloff_mean": 6285.672077,
  "rolloff_energy_hz": 172.265625,
  "brightness": 0.062798,
  "band_energy_ratios": { "low": 0.890089, "low_mid": 0.056397, "high_mid": 0.039051, "high": 0.014463 }
},
"dynamics": { "loudness_lufs": -15.669995, "rms_mean": 0.067463, "crest_factor": 7.608849 },
"labels": [
  { "label": "kick-heavy", "confidence": 1.0,
    "evidence": { "band_energy_low": 0.890089, "min_band_energy_low": 0.5 } },
  { "label": "busy drums", "confidence": 0.045131,
    "evidence": { "onset_density": 5.315915, "min_onset_density": 5.0 } }
],
"unavailable_features": []
```

`centroid_mean` 4390 Hz and `centroid_energy_hz` 412 Hz on the same stem, and **both are correct**:
89% of its energy is the kick, so the energy-weighted centroid sits down there, while the mean of
per-frame centroids is dragged up by every frame between kicks that holds a hat tail. They are
different statistics, not two scales of one, which is why `centroid_energy_hz` was added beside
`centroid_mean` rather than replacing it — and why the heuristic thresholds that read a centroid had
to be re-derived one at a time rather than rescaled.

`track_summary.json` strips `drum_decomposition.hits` down to a `total_hit_count`; the full per-hit
list, and the cycle-folded pattern it is built from, live only in `analysis/drums.json`:

```json
"drum_decomposition": {
  "status": "ok",
  "steps_per_cycle": 16,
  "cycle_seconds": 1.818186,
  "grid_anchor_source": "supplied",
  "quantisation_error_steps": 0.033797,
  "patterns": [
    { "drum": "kick",  "steps": [0, 1, 4, 6, 7, 8, 11, 12, 15],
      "step_occupancy": [0.96, 0.01, 0.94, 0.01, 0.01, 0.96, 0.02, 0.90, 0.03], "hit_count": 483 }
  ],
  "hits": [
    { "time_seconds": 0.232200, "drum": "kick", "confidence": 0.94, "step": 0,
      "kick_ratio": 0.87, "body_ratio": 0.07, "noise_ratio": 0.04, "air_ratio": 0.02,
      "decay_ratio": 1.9, "flatness": 0.004 }
  ],
  "unclassified_count": 65
}
```

`step_occupancy` is the useful column: 0.96 / 0.94 / 0.96 / 0.90 on steps 0, 4, 8 and 12, and ≤ 0.03
on every other step the kick ever touched. That is what a four-on-the-floor kick looks like when the
grid is right, and it is what v4 could not report at all.

`analysis/bass.json` carries the full note sequence behind `bass_line` in `strudel_hints.json`
(which caps at 32 entries), plus per-note diagnostics:

```json
"bass_line": {
  "status": "ok",
  "notes": [
    { "start_seconds": 0.243810, "duration_seconds": 0.208798, "midi_note": 33, "note_name": "a1",
      "median_f0_hz": 55.31, "cents_offset": 9.7, "confidence": 0.89, "step": 2 }
  ],
  "median_midi_note": 41,
  "voiced_fraction": 0.4533,
  "octave_corrections": 43,
  "caveats": []
}
```

On a source with no bass in it at all, that block reads `"status": "silent"` with a caveat rather
than a note list — a pitch tracker pointed at a noise floor returns confident nonsense, and v4
returned it. On one of the short test clips a −70 LUFS bass stem reported two notes and a key of E
minor, and that key then beat the mix's own reading into `strudel_hints.json`. See
`calibration/v5-vs-v4.md`.

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

Implemented and working end to end: `track-forensics all input.wav` runs separation, per-source analysis, heuristic labelling, tempo and downbeat resolution, drum decomposition, bass note extraction, arrangement extraction, Strudel sound mapping, and hints export, and produces the full output tree described above. Schema is at `schema_version` 5 — see the migration note above; v4 readers keep working. `pytest`, `ruff check .`, and `mypy src` are all clean.

Drum decomposition and bass note extraction are numerically verified against synthetic fixtures (including one specifically constructed so a naive single-onset-list classifier cannot pass it), against committed real-material fixtures in `tests/fixtures/real/`, and end to end against real Demucs separation. As of schema v5 they are also calibrated against an **eight-track corpus** — house, swung hip-hop, funk, live band, ambient, drum & bass, and two short clips — with the outputs committed under `calibration/v5/` and the v4 → v5 delta written up in `calibration/v5-vs-v4.md`.

What the corpus found is worth knowing before you trust any single number:

- **The classifier fails on drums it was not designed for, and now says so.** Bonham's kick on "When the Levee Breaks" yields **zero** kick hits in seven minutes — not because the band is quiet (it holds 37% of the stem's energy) but because it is compressed and reverberant enough that the energy never falls back between hits, so per-hit peak-picking cannot separate them. The caveat says exactly that, and says that any drum living in those bands is missing from the counts however loud it is. A global sensitivity increase would have manufactured kicks on the three tracks that work; it was built, measured, and reverted.
- **Roni Size's "Brown Paper Bag" is reported at 170 BPM, and the backend reads 84.92 at its highest confidence anywhere in the corpus.** The octave is arbitrated by fitting a drum grid at each candidate, not by correlation — correlation was measured and cannot do it.
- **Brian Eno's "1/1" returns no grid, no arrangement and no sections**, and that is a pass. A tool only ever tested on material it handles well learns to always answer.
- Some thresholds are still marked `[guess]` in the source, with the measurement they need named. That is deliberate: an undocumented constant is a future bug with no paper trail.

The drum classifier stays deliberately conservative about what it claims — three classes only, an honest `unclassified` bucket, no invented drum-machine identity — rather than confidently wrong.

See `CLAUDE.md` for the implementation brief and module contracts if you're extending it.
