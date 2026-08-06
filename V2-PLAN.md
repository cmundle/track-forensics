# track-forensics v2 — findings and work breakdown

Written after calibrating schema v4 against three real tracks. v1 is done and works; this document
is what the first real-material calibration exposed, and the plan to act on it.

Same conventions as `TODO.md`: work is grouped into **waves**, everything inside a wave is
file-disjoint and can run in parallel, waves run strictly in sequence. Each package lists
**Owns**, **Blocked by**, **Tasks**, and **Done when**.

Target schema version: **5**.

## Decisions settled before dispatch

Closed. Recorded here so nobody relitigates them mid-wave.

1. **New descriptors are added, never substituted.** `centroid_mean` keeps its current definition and
   a correctly-weighted `centroid_median` is added alongside it (W4D). Same rule for tempo: the
   backend's `rhythm.bpm` stays as-is and the refined value lands in a new `TempoFit` block (W4A).
   v4 and v5 outputs stay numerically comparable, and thresholds migrate one at a time instead of all
   at once. The cost is a known-misleading field left in the schema — mitigate it with an explicit
   docstring on `centroid_mean` saying what it is contaminated by and pointing at the replacement.
2. **Scope for this cycle is Waves 4, 5A, 6 and 8.** Harmony (W5B) and code generation (W7) are
   deferred. Neither is load-bearing for anything else, both are designed against a single track, and
   both are better decided after re-calibration proves the fixes generalise. Their specs are kept
   intact in [Appendix: deferred packages](#appendix--deferred-packages).
3. **v4 outputs are frozen as a reference, then everything is re-run.** Move the current outputs to
   `calibration/v4/` and commit them before any code changes land; v5 runs write to
   `calibration/v5/`, so W8B has a real before-and-after diff. This means a full Demucs re-run per
   track at the end of the cycle — budget for it.
4. **The calibration corpus is five tracks, not two.** See W8B.
5. **The clap is a fourth class, not a merged "backbeat" class.** Merging snare and clap would lose
   exactly the distinction that made F2 visible, and the two map to different Strudel sounds
   (`sd` vs `cp`), which is the point.
6. **Strudel expressions are verified by a dev-time harness**, not by trusting the docs or memory.
   `tools/strudel-verify/` evaluates each emitted expression against the real library and asserts
   the events land where the analysis says. Runtime stays offline; the harness needs the network once
   at `npm install`. See the appendix.

---

## Part 1 — What calibration found

Three tracks were run end to end: two short clips (`ancient-heavy-tech-donjon` 4.3 s,
`showers-of-gold` 17.1 s) and one full track (`madonna-i-feel-so-free-peggy-gou-energy-mix-official`,
267.5 s, 4:27). The full track is the one that matters — it is well separated, has real levels on
all four stems, and is a genre the tool should handle cleanly. Everything below is measured, and
every finding has a reproducible number attached.

### F1 · Tempo error destroys the drum grid (critical)

The pipeline reported **131.854843 BPM**. The true tempo is **132.000 BPM exactly**:
autocorrelation of the 20–110 Hz flux gives a lag of 0.454545 s at r = 0.97, identical at 16-beat
and 32-beat multiples, and stable to four decimal places across both halves of the track.

That 0.145 BPM error accumulates 0.38 s of drift over 588 beats, about **3 sixteenth-steps** by the
end. `drum_elements.decompose()` anchors once at 0.348 s and extends a fixed cycle across 147 bars,
so hits at the end of the track land several steps away from where the grid says they should. The
reported quantisation error was 0.287 steps against a 0.18 allowance, and the grid was rejected as
`no_grid`.

Re-fold the same stem at exactly 132 BPM with the downbeat at 0.228 s and the pattern is
unambiguous:

```
step     0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
kick  0.93 0.01 0.00 0.01 0.99 0.01 0.00 0.00 0.96 0.01 0.00 0.01 1.00 0.01 0.00 0.00
clap  0.81 0.03 0.04 0.03 1.00 0.03 0.05 0.04 0.82 0.04 0.04 0.03 1.00 0.03 0.05 0.02
air   0.27 0.15 0.30 0.07 1.00 0.08 0.28 0.10 0.24 0.19 0.29 0.07 1.00 0.08 0.31 0.07
```

Four-on-the-floor with 1% off-grid leakage. This is a textbook grid that the tool declared did not
exist. **Any track over about two minutes is exposed to this**; the error scales with duration, so
short calibration clips will never reveal it.

Two contributing causes, both fixable:

1. The frame-level BPM estimate is not refined. Essentia's `RhythmExtractor2013` is accurate to
   roughly ±0.2 BPM, which is fine for a label and useless for a grid.
2. A single global anchor plus a fixed cycle length cannot absorb any residual error.

### F2 · The three-class drum model mislabels an entire genre

87 hits classified as snare. The arrangement implies roughly 294 (two per bar over 147 bars). The
missing backbeat is visible in the profile above: steps 4 and 12 peak in both the 1.2–5 kHz band and
the 6–14 kHz band, with no low-mid body. That is a **clap**, and claps do not clear a `body_ratio`
test, so they land in the hat bucket — which is why `hat` reported 1240 hits, roughly double the
8th-note count the track actually plays.

House, techno, disco, and most of pop uses a clap or rimshot on the backbeat rather than a snare.
The current kick/snare/hat trio systematically misreads all of it. The existing `unclassified`
bucket is honest but does not help here, because the claps are not landing there — they are being
confidently mislabelled as hats.

### F3 · Bass note onsets carry a constant 32 ms lag

Quantising all 709 detected bass notes to 16ths at the corrected tempo gives a mean error of
**0.276 steps**. Removing a single constant offset — the circular mean of the fractional step
position, 0.282 steps = **32 ms** — drops it to **0.137 steps**, comfortably under the 0.18
allowance.

This is not musical looseness. It is segmentation latency in `note_track.py`: the median filter plus
the note-segmentation logic needs several frames of stable F0 before it declares a note started, so
every onset is reported late by roughly the same amount. It is systematic, it is measurable, and it
is currently being charged against the music.

### F4 · The sub-bass branch in `strudel_vocab.py` is unreachable

The Madonna bass stem measures `low_band_ratio` 0.916, `brightness` 0.0021, and nothing at all above
4 kHz (`high` ratio 2e-06). That is a pure sine sub, and `suggest_bass_sound()` returned
`match: "none"`.

It failed on one clause: `SUB_BASS_CENTROID_HZ_MAX = 120.0` against a measured `centroid_mean` of
1010.7 Hz. But `centroid_std` is **1573 Hz, larger than the mean** — the signature of an unweighted
mean taken over frames that include silence, where a flat noise floor puts the per-frame centroid up
in the kilohertz. 55% of this stem is unvoiced.

An unweighted mean centroid over a real stem will essentially never fall below 120 Hz, so this
branch has probably never fired on any input. The fix is not to loosen the threshold; it is to
compute an energy-weighted (or median) centroid so the descriptor means what the threshold assumes
it means.

### F5 · A silent stem still reports `status: "ok"`

On both short clips, Demucs put the entire track into `other` and left `bass` and `vocals` at the
noise floor (−70 LUFS, RMS 7.7e-05, about −82 dBFS). `bass_line` still ran, still returned
`status: "ok"`, and still emitted notes — pitch-tracked from noise. The `e4` median it reported is
not a bass at all, and the caveat it generated ("check the octave by ear") sends the reader off to
verify something that does not exist.

Related: on the same track, `tonal_centre` in `strudel_hints.json` was taken from that silent bass
stem ("E minor"), overriding the mix's own F major reading. Falling back to a silent stem is worse
than emitting `null`.

### F6 · The `voiced_fraction` caveat misfires at normal values

"Only 45% of frames were voiced, so this line is built from less than half the stem" reads as a
defect. A bassline with rests is unvoiced roughly half the time by construction. The caveat earns
its keep at 9%; at 45% it is noise, and noise in the caveat list trains the reader to skip caveats.

### F7 · Three things worth extracting that the tool does not attempt

All three were derived from data the pipeline already produces or from the stems it already writes,
and all three turned out to be the difference between "a loop" and "the track":

- **Arrangement.** Per-bar RMS per stem, thresholded, gives the song structure directly: 16-bar
  intro on kick and pad, an 8-bar kick-and-bass section, 48 bars of full band, a 16-bar breakdown
  with kick and bass entirely out, a drop, two bars of total silence, main groove, outro. 146 bars,
  matching the measured duration to within 1.8 s. This is the single highest-value addition.
- **Harmony.** Per-bar chroma on the `other` stem matched against triad templates gives long static
  A minor stretches with a recurring `Em F G Am` turnaround at phrase ends, and a different colour
  (`Am C Em`) in the breakdown. The existing `tonal` block gives one key for the whole track; this
  gives the actual progression.
- **Bass placement.** The 709 notes cluster hard on steps 2, 6, 10, 14 — offbeat 8ths — in 103 of
  roughly 110 playing bars. `bass_line` records pitch and time but never reports where the notes sit
  relative to the bar, which is the thing you need to type it in.

### F8 · Envelope folding beats onset picking

Worth recording as a method, because it is what made F1 and F2 visible. Peak-picking onsets and then
folding them onto a grid is fragile: threshold choice dominates the result, and I got anywhere from
487 to 2061 "kicks" out of the same stem depending on parameters. Folding the **band flux envelope**
itself into a (bar × step) matrix and taking the median across bars is far more robust — it needs no
threshold, it degrades gracefully across sections where an element drops out, and it produced the
clean profile above on the first attempt. `drum_elements.py` should use it for grid inference even
if it keeps per-hit picking for hit counts.

---

## Part 2 — Ground rules for v2

Everything in `TODO.md` still applies (44.1 kHz throughout, no network, lazy backend imports,
`None` plus `unavailable_features` rather than crashing, full type hints, `from __future__ import
annotations`). Four additions:

1. **`schemas.py` is frozen except for W6.** Same rule as v1. Any agent that thinks the schema needs
   to change reports it rather than editing.
2. **No new runtime dependencies.** numpy and scipy are already base dependencies and cover
   everything here. `rich` arrives transitively via Typer and may be used for terminal output, but
   must not be imported at module top level in a way that breaks the CLI if it is absent.
3. **Tests still must not require a real audio file — but real-material regressions must be
   captured.** Resolve this by committing *derived intermediate data*, not audio: a decimated flux
   envelope, a per-bar RMS array, the note list from `bass.json`. These are small, they are not
   copyrightable audio, and they let a test assert "this real track yields 132.000 BPM" without
   shipping the track. Put them in `tests/fixtures/real/` with a short provenance note. The existing
   `.gitignore` already allows synthetic `tests/fixtures/*.wav`; that exception stands and no
   source material is ever committed.
4. **Add, do not substitute.** Settled decision 1. A new descriptor that corrects an old one lands
   beside it, with the old one's docstring updated to say what is wrong with it. This is what keeps
   the v4 calibration outputs usable as a baseline.

---

## Wave 4 — fix what is broken

Four packages, fully parallel. None touches `schemas.py`, `analyze.py`, or `cli.py`.

### W4A · Tempo refinement and downbeat anchoring

**Owns:** new `src/audio_pipeline/tempo.py`, `tests/test_tempo.py`
**Blocked by:** nothing

Pure numpy + scipy, no backend dependency — same shape as `drum_elements.py` and `note_track.py`.
This is the highest-priority package in the wave; F2 and every grid-dependent feature downstream is
gated on it.

**Tasks**

1. `refine_bpm(samples, sample_rate, coarse_bpm) -> TempoFit` — take the backend's estimate as a
   starting point, compute the low-band (20–110 Hz) flux envelope, autocorrelate, and locate the
   peak near `N * coarse_period` for N in (16, 32). Parabolic-interpolate the peak for sub-bin
   resolution. Return refined period, BPM, and the autocorrelation r at that lag as a confidence.
   Guard against the octave and multiple-peak errors that bit the first implementation: reject any
   candidate whose implied BPM is not within ±3% of the coarse estimate, and if the N=16 and N=32
   estimates disagree by more than 0.05 BPM, return the coarse value with low confidence rather than
   picking one.
2. `find_downbeat(samples, sample_rate, period) -> float` — search offsets across one beat period,
   maximising folded energy on steps 0/4/8/12. Return seconds.
3. `stability(samples, sample_rate, period)` — fit the first and second half independently and
   report the difference. A track that genuinely changes tempo should be detectable, not silently
   averaged.
4. Model the result as a plain dataclass here; W6 promotes it into `schemas.py`.
5. Tests: synthetic click trains at 120, 132, and 128.5 BPM must recover to within 0.01 BPM. A click
   train with a deliberate 1% ritardando must report low stability. A pure-noise input must return
   the coarse estimate with low confidence rather than a confident wrong answer. Add the committed
   Madonna flux fixture and assert **132.000 ± 0.01**.

**Done when:** the Madonna fixture returns 132.000 and the ±3% guard demonstrably rejects the
half-time and double-time candidates.

### W4B · Clap class and envelope-fold grid inference

**Owns:** `src/audio_pipeline/drum_elements.py`, `tests/test_drum_elements.py`
**Blocked by:** nothing (but see note on W4A)

**Tasks**

1. Add a **clap/rimshot** class. Distinguishing feature versus hat is not brightness — both are
   bright — it is the presence of 1.2–5 kHz body with a short, dense, multi-transient attack, and
   crucially its *placement* correlating with steps 4 and 12. Start with the band contrast
   (1.2–5 kHz energy relative to 6–14 kHz) and a decay measure; document the thresholds and where
   they came from.
2. Replace grid inference with the envelope-fold method from F8: build a (bar × step) matrix from
   the band flux, take the median across bars, and read the pattern off the profile. Keep per-hit
   peak-picking for `hit_count`, but stop using it to decide whether a grid exists.
3. Accept an externally supplied period and downbeat offset (from W4A) rather than deriving the
   cycle from BPM internally. Keep the current internal derivation as a fallback when none is
   supplied, so this module stays independently testable.
4. Report per-step **occupancy fraction** (in what proportion of playing bars does this class hit
   this step) alongside the binary step list. `0.93 0.99 0.96 1.00` is much more useful than
   `[0,4,8,12]`, and it is what tells you an element is intermittent rather than absent.
5. Revisit the `no_grid` rejection. With a corrected period the 0.18 allowance is probably right,
   but the failure mode should distinguish "hits do not fit any grid" from "hits fit a grid that is
   drifting", and report which.

**Done when:** the committed Madonna drums fixture yields kick on steps 0/4/8/12 with occupancy
above 0.9, a clap class on steps 4 and 12, and hats on even steps. Synthetic fixtures still pass.

**Note:** W4B can be written and tested against a supplied period without W4A existing. Wire the two
together in W6.

### W4C · Onset lag correction in note segmentation

**Owns:** `src/audio_pipeline/note_track.py`, `tests/test_note_track.py`
**Blocked by:** nothing

**Tasks**

1. Find and remove the systematic lag documented in F3. Do not apply a blanket 32 ms constant —
   derive it. The lag is a function of the median filter width and the minimum-frames-to-confirm
   threshold, both of which are known at call time, so compute the expected latency and subtract it.
2. Verify empirically: after correction, the circular mean fractional step position over a
   quantised note set should be near zero, not 0.28.
3. Add `voiced_fraction` gating. Below a documented floor (start at 0.15 and calibrate), return
   `status: "unvoiced"` and no notes rather than `ok` with noise-derived pitches. This is half of F5.
4. Fix the F6 caveat: only emit the low-voiced-fraction warning below roughly 0.3, and reword it so
   it describes coverage rather than implying failure. A bass that rests is not a broken measurement.
5. Add `step` population. The `NoteEvent.step` field already exists in the schema and is `null`
   everywhere. Given a period and downbeat offset, fill it. This directly addresses F7's third point.

**Done when:** the committed Madonna note fixture quantises to a mean error under 0.15 steps with no
constant offset removed, and steps 2/6/10/14 hold 103 notes each.

### W4D · Spectral descriptor correctness

**Owns:** `src/audio_pipeline/backends/librosa_backend.py`,
`src/audio_pipeline/backends/essentia_backend.py`, `src/audio_pipeline/strudel_vocab.py`,
`tests/test_librosa_backend.py`, `tests/test_essentia_backend.py`, `tests/test_strudel_vocab.py`
**Blocked by:** nothing

One agent owns both backends because the two must stay numerically comparable — that constraint is
already in `CLAUDE.md` and splitting this package would break it.

**Tasks**

1. Add `centroid_median` as a **new** field beside `centroid_mean` (settled decision 1 — do not
   replace). Compute it energy-weighted, or as a median across frames weighted by frame energy, so
   silent frames stop dominating. The current unweighted frame mean is the bug (F4: mean 1010.7 Hz,
   std 1573 Hz on a stem that is 55% silence and has no content above 4 kHz). Both backends must use
   an identical definition. Update `centroid_mean`'s docstring to state that it is contaminated by
   silent frames and point at the replacement. W6 adds the schema field; report the exact shape you
   need rather than editing `schemas.py`.
2. Apply the same reasoning to `rolloff_mean` and `brightness` — check whether they suffer the same
   silence contamination. Do not assume; measure on a fixture with known silence, and only add a
   corrected variant where the measurement shows one is warranted.
3. Point `SUB_BASS_CENTROID_HZ_MAX = 120.0` in `strudel_vocab.py` at the new descriptor. The
   threshold itself is probably correct; it was the input that was broken. Add a test asserting a
   synthetic 55 Hz sine padded with 50% silence returns `match: "exact"`, `sound: "sine"` — that test
   would have caught F4.
4. Add a `sound` mapping for the new clap class from W4B, defaulting to `cp`.

**Done when:** a synthetic 55 Hz sine with 50% silence reports a centroid under 120 Hz and resolves
to `sine`, and both backends agree within the documented tolerance.

---

## Wave 5 — new extraction

One package this cycle. W5B (harmony) is deferred — see the appendix.

### W5A · Arrangement extraction

**Owns:** new `src/audio_pipeline/arrangement.py`, `tests/test_arrangement.py`
**Blocked by:** W4A (needs a reliable bar grid)

**Tasks**

1. `per_bar_energy(stems, period, offset)` — RMS per bar per stem, plus kick-band RMS from the drums
   stem specifically, since "the drums stem is loud" and "the kick is playing" are different facts.
2. `presence(energy)` — threshold each stem against a percentile of its own distribution (15% of the
   90th percentile worked well; calibrate it). Binary per-bar presence.
3. `segment(presence)` — collapse runs of identical presence patterns into sections, merging
   sections shorter than a minimum length (2 bars) into their neighbours. Return start bar, length,
   and the active stem set.
4. `label_sections(sections)` — heuristic names, held loosely and clearly marked as derived:
   `intro` (early, no bass), `breakdown` (no kick and no bass, mid-track), `drop` (full band
   immediately after a breakdown), `outro` (late, thinning), `full`, `groove`. Every label carries
   the presence pattern that produced it, same auditability rule as `heuristics.py`.
5. Do not attempt verse/chorus. That needs repetition analysis and is out of scope here.

**Done when:** the committed Madonna per-bar RMS fixture yields 146 bars and recovers the 16-bar
breakdown at bars 75–90 with kick and bass both absent.

## Wave 6 — schema v5 and wiring

One package, sequential. This is the integration point and it owns the contract, so nothing runs
alongside it.

### W6 · Schema v5, orchestration, hints

**Owns:** `src/audio_pipeline/schemas.py`, `src/audio_pipeline/analyze.py`,
`src/audio_pipeline/strudel_hints.py`, `tests/test_analyze.py`, `tests/test_strudel_hints.py`
**Blocked by:** all of Wave 4 and Wave 5

**Tasks**

1. Bump `SCHEMA_VERSION` to 5. New models: `TempoFit` (refined BPM, downbeat offset, confidence,
   stability) and `Arrangement` (list of sections). Add `centroid_median` to `SpectralFeatures` per
   W4D's reported shape. Extend `DrumPattern` with per-step occupancy and the clap class. Extend
   `NoteEvent.step` usage.
2. **Additive only** (settled decision 1). `rhythm.bpm` keeps the backend estimate untouched;
   `TempoFit.bpm` carries the refined one. `centroid_mean` stays; `centroid_median` joins it. Update
   the docstrings of both superseded fields to state what is wrong with them and name the
   replacement. A v4 reader must still find every field it expects.
3. Add `TempoFit` and `Arrangement` at track level, not per source — there is one tempo and one
   structure. This is a structural change to `TrackSummary`; think it through before writing it.
   Leave room for a `Harmony` block in the same position (deferred W5B) so adding it later is not a
   second structural change.
4. `analyze.py`: resolve tempo once via `tempo.py`, pass the period and downbeat into
   `drum_elements`, `note_track`, and `arrangement`. Today each derives its own grid; after this they
   share one.
5. **Silence gating** (the rest of F5). Before analyzing any stem, check `dynamics.rms_mean` against
   a documented floor. A stem below it gets `status: "silent"` on its derived features, is excluded
   from `tonal_centre` fallback, and is noted in `unavailable_features`. Nothing downstream should
   ever pitch-track a noise floor again.
6. `strudel_hints.py`: fix the `tonal_centre` fallback chain so a silent stem can never win (F5,
   second half). Surface arrangement and bass step placement in the hints. Keep the file small and
   hand-readable — that constraint has held up well, do not lose it.
7. Migration note in the README: v5 adds fields and removes none, so v4 readers keep working. State
   that `calibration/v4/` is a frozen reference and is never regenerated in place; v5 runs write to
   `calibration/v5/`.

**Done when:** `track-forensics all` on the Madonna wav produces a v5 tree with a populated grid and
arrangement, every v4 field is still present, and `pytest`, `ruff check .`, `mypy src` are clean.

---

## Wave 8 — terminal visualization and recalibration

Two packages, parallel.

### W8A · Terminal visualization

**Owns:** new `src/audio_pipeline/viz.py`, `src/audio_pipeline/cli.py`, `tests/test_viz.py`
**Blocked by:** W6

Terminal only. No HTML, no images, no notebooks, no server — the v1 non-goals stand. numpy is
already a dependency and Unicode block characters are enough. `rich` is available transitively via
Typer and may be used, but must be guarded so the CLI still loads without it.

**Tasks**

1. New CLI command `track-forensics show <output-dir> [--section drums|bass|arrangement|spectral|all]`.
   Reads the v5 JSON tree, writes nothing. Leave the `--section` enum open so a `harmony` view can be
   added later without a CLI change.
2. **Drum grid.** The step profile as a labelled character grid, one row per class, 16 columns,
   density mapped to block characters. This is the single most useful view and should be the
   default:
   ```
        1 e & a 2 e & a 3 e & a 4 e & a
   bd   █ · · · █ · · · █ · · · █ · · ·
   cp   · · · · █ · · · · · · · █ · · ·
   oh   ▄ · ▄ · █ · ▄ · ▄ · ▄ · █ · ▄ ·
   ```
3. **Arrangement map.** One character per bar, one row per stem, wrapped at 64 bars with bar-number
   gutters and section labels underneath. This is the view that makes a 4-minute track legible on
   one screen.
4. **Bass.** Step histogram in the same 16-column frame as the drum grid so they align visually, plus
   a note-name sequence for the first few bars.
5. **Spectral.** Four-band energy ratios as horizontal bars per stem, with LUFS and RMS beside them.
   Small, but it is the fastest way to spot a silent or misassigned stem — F5 would have been obvious
   in one glance.
6. Every view prints the measurement that backs it and any caveat attached. The point of this command
   is to make the JSON legible, not to replace reading it.
7. Degrade cleanly: a v4 tree, a missing section, or a `status: "silent"` feature prints a clear note
   rather than an empty frame or a traceback. Since v5 is additive, a v4 tree should render
   everything except the new sections rather than refusing outright.
8. Tests: build model objects by hand, assert on rendered string content. No audio, no terminal
   capture.

**Done when:** `track-forensics show` on the Madonna output renders grid, arrangement, bass, and
spectral views, and the drum grid visually matches the profile in F1.

### W8B · Recalibration and regression fixtures

**Owns:** `src/audio_pipeline/heuristics.py` thresholds, `tests/fixtures/real/`, `calibration/`,
`README.md`
**Blocked by:** W6

**Tasks**

0. **Before any Wave 4 code lands**, commit the current v4 `calibration/` JSON as the frozen
   reference (settled decision 3). This step is a prerequisite for the whole cycle, not part of
   Wave 8 — pull it forward and do it first. Directory is currently untracked; `.gitignore` already
   excludes the stems, so only JSON is committed.
1. Build the committed derived-data fixtures described in ground rule 3 — flux envelope, per-bar RMS,
   note list — from the Madonna track. Small arrays, `.npz` or JSON, with a provenance note recording
   what they came from and that no audio is committed.
2. Re-run all existing calibration tracks under v5 into `calibration/v5/`, leaving `calibration/v4/`
   untouched. Record what changed, especially whether the two short clips still fail in the same way
   and whether the failures are now reported honestly.
3. Re-tune `heuristics.py` thresholds against `centroid_median` from W4D. Every threshold that reads
   `centroid_mean` is suspect: it was tuned against a value contaminated by silent frames. Migrate
   them one at a time and record what each change did to the labels.
4. **Build the calibration corpus** (settled decision 4). Five full-length tracks, one per row. Each
   needs a short note recording what you expect by ear before the tool runs, so the comparison is
   honest rather than post-hoc:

   | # | material | what it tests |
   |---|---|---|
   | 1 | house, fixed tempo (Madonna, have it) | the baseline everything was fixed against |
   | 2 | swung hip-hop or broken beat | `infer_subdivision_feel`, which has returned `null` on everything so far, and the swung-versus-straight branch in W4B |
   | 3 | live band or acoustic | W4A's tempo stability check against a genuinely floating tempo — the likeliest source of a confidently wrong refined BPM |
   | 4 | ambient or rubato | that the tool correctly refuses. Expected output is `no_grid` and `status: "unvoiced"`, and that is a pass, not a failure |
   | 5 | drum and bass or breakbeat, 170+ | the half-time and double-time guards in W4A, which nothing has exercised |

   Row 4 is the important one and the easiest to skip. A tool that only ever gets tested on material
   it handles well will quietly learn to always answer.
5. Update `CLAUDE.md`: schema v5, the new modules (`tempo.py`, `arrangement.py`, `viz.py`), the
   additive-only descriptor policy, and the fact that the tempo estimate is now refined rather than
   taken from the backend. Note that codegen remains a non-goal for this cycle.

**Done when:** the fixture suite catches F1, F3, and F4 as regressions; all five corpus tracks have
run; and `calibration/v5/` documents the delta against `calibration/v4/`, including at least one
track where the correct answer was "no grid".

**Time cost, stated plainly:** five full-length tracks through `htdemucs_ft` is a few minutes each,
plus analysis. Budget an afternoon for step 4 and do the separation runs before you need the results.

---

## Suggested order if running solo

The waves exist for parallel agents. Working alone, the dependency-honest order is:

0. **W8B step 0** — commit the v4 baseline. Everything else is measured against it.
1. **W4A** — everything downstream is gated on a correct tempo. Nothing else is worth doing first.
2. **W4C** and **W4D** — small, self-contained, immediately verifiable.
3. **W4B** — the largest of the fixes.
4. **W8B step 1**, pulled forward, so the rest has regression cover.
5. **W5A** — arrangement, the highest-value new feature.
6. **W6** — integration.
7. **W8A**, then the rest of **W8B**.

---

## Risks

- **W6 is a big structural change.** Moving tempo and arrangement to track level is right, but it
  touches the one file everything else is built against. Budget for it going slower than it looks.
- **The clap class may not generalise.** It is being designed against one track in one genre. Band
  contrast plus placement is a reasonable first cut, but expect it to need revisiting once the corpus
  includes material that is not house.
- **Additive-only leaves traps in the schema.** `centroid_mean` and `rhythm.bpm` survive as fields
  that look authoritative and are not. Docstrings are the only guard, and docstrings are easy to
  ignore. Revisit at v6 and consider removing them once nothing reads them.
- **The corpus is the long pole.** Four new full-length tracks through `htdemucs_ft` plus honest
  by-ear notes is the most time-consuming item in the plan and the easiest to cut. Cutting it means
  shipping fixes calibrated against exactly one record.
- **Only one track has been properly calibrated.** Three of the eight findings above rest on a single
  4:27 house record. F1, F3, and F4 are mechanical and will hold anywhere. F2 and the arrangement
  thresholds are genre-shaped and should be treated as provisional until W8B step 4 lands.

---

## Appendix — deferred packages

Specced, not scheduled. Both are deferred by settled decision 2: neither is load-bearing, both were
designed against a single track, and both are better decided once the corpus exists. Kept verbatim so
picking them up later costs nothing.

**The Strudel API prerequisite is now resolved.** It was listed here as a blocker for W7; it is done.
`tools/strudel-verify/` is a dev-time Node harness that installs `@strudel/core` and `@strudel/mini`
at a pinned version, builds each expression the pipeline emits, queries one cycle, and asserts the
event onsets land on the 16th-steps the analysis claimed. It needs the network once at
`npm install`, on a developer's machine — the runtime offline constraint is untouched, and no Python
module imports it.

Two results from the first run, both relevant to W7 whenever it is scheduled:

- All eight placement claims in the hand-written patch verified. `note("[~ a1]*4")` really does
  place notes on steps 2, 6, 10, 14, and the gain pattern really does accent the offbeat. This is the
  check that matters most: a wrong function name fails loudly, but a right function name with wrong
  placement produces a patch that runs, sounds plausible, and does not match the record.
- **`setcpm` is a REPL global, not a library export.** It works pasted into strudel.cc and fails when
  the same text is evaluated as a library. `cpm` is a real core export. `npm run api` classifies
  every name the project emits on either side of that line, and exits non-zero on a name it cannot
  account for. W7 must consult it before emitting anything.

Also worth recording: Strudel's source has moved to Codeberg (`codeberg.org/uzu/strudel`), not
GitHub, and its docs are generated from JSDoc into a `doc.json` via `npm run jsdoc-json`. That file
is the machine-readable API reference if a fuller surface check is ever wanted.

When W7 is scheduled, add a case to `verify.mjs` for every construct codegen learns to emit, and
treat a red harness as a failed package. That closes the loop the tier system opens: the tiers say
how confident a line is, and the harness says whether the line does what the tier claims.




### W5B · Harmony extraction  *(deferred)*

**Owns:** new `src/audio_pipeline/harmony.py`, `tests/test_harmony.py`
**Blocked by:** nothing

**Tasks**

1. `bar_chroma(samples, sample_rate, period, offset)` — per-bar 12-bin chroma. Restrict to roughly
   65–2100 Hz before folding; above that, harmonics smear the estimate.
2. `match_triads(chroma)` — correlate against the 24 major and minor triad templates, return best
   match plus the margin over second-best as a confidence. Emit `None` below a confidence floor
   rather than guessing; an ambiguous bar is a real thing.
3. `find_progression(bars)` — locate the most common repeating 4-bar and 8-bar chord loops, and
   report them with the bar ranges where they occur. Static-harmony tracks should report exactly
   that rather than a spurious progression.
4. Run on the `other` stem by default, with the mix as fallback when `other` is absent or silent.
   Document why: `other` carries the harmonic content once drums, bass, and vocals are removed.
5. Tests: synthesise triads directly (three sine stacks) and assert the correct chord comes back.
   Assert a single sustained note returns `None` rather than inventing a triad. Assert the committed
   Madonna chroma fixture recovers the `Em F G Am` turnaround.

**Done when:** synthetic triads round-trip correctly and the Madonna fixture reports A minor as the
dominant chord with the turnaround present.

---

### Wave 7 · Strudel code generation  *(deferred)*

One package, sequential. This reverses an explicit v1 non-goal, deliberately.

#### W7 · `export-strudel-patch`

**Owns:** new `src/audio_pipeline/strudel_codegen.py`, `tests/test_strudel_codegen.py`
**Blocked by:** W6

The v1 rule was "do not generate Strudel code, a wrong pattern is worse than no pattern." That rule
was right when the grid was unreliable. With Wave 4 landed it no longer is, and a hand-written patch
built from these measurements did reproduce the track. So: generate code, and carry the original
concern forward as a constraint on *how*.

**The honesty rule.** Every emitted line falls into exactly one of three tiers, and the tier is
visible in the output:

- `measured` — traces directly to a descriptor that cleared its threshold. Emitted plain, with a
  trailing comment citing the value: `s("bd*4")  // steps 0,4,8,12 occupancy 0.93-1.00`
- `inferred` — a defensible reading of a measurement that could reasonably go another way. Emitted
  plain, comment states the alternative: `s("oh*8")  // 8ths measured; on/offbeat not determinable`
- `guess` — convention or genre prior, not measured. Emitted **commented out** with the reason.

`--strict` omits the `guess` tier entirely. The patch header carries a manifest: how many lines in
each tier, and which features were unavailable. A reader must be able to tell at a glance how much
of the file is evidence and how much is convention.

**Tasks**

1. Emit `setcpm(bpm/4)` from `TempoFit`, with the measured confidence in a comment.
2. Emit one named pattern per drum class from the occupancy profile. Steps above the occupancy
   threshold become hits; borderline steps become a commented alternative line.
3. Emit the bass from `bass_line` steps and note names. Where the same step alternates octaves
   across bars, emit the octave-doubled form rather than picking one.
4. Emit chords from `harmony` as explicit note stacks, not chord-name helpers — the helper API
   surface is larger than what has been verified against the live docs, and explicit stacks cannot
   be misinterpreted.
5. Emit `arrange(...)` from `Arrangement`, one entry per section, with bar ranges and timestamps in
   comments so the reader can scrub the original and check.
6. Sound selection comes from `strudel_vocab.py`. Never invent a sound name. A `match: "none"`
   result emits a commented-out line listing the alternatives, not a silent default.
7. Record `strudel_vocabulary_read` in the header, same as `strudel_hints.json` already does.
8. Tests: golden-file comparison against a committed expected patch built from the Madonna v5 JSON.
   Assert that `--strict` output contains no `guess` lines. Assert that a v5 tree with
   `status: "no_grid"` emits no drum pattern at all rather than a fabricated one.

**Done when:** the generated patch for the Madonna track is playable in Strudel without edits, and
every line in it is traceable to a number in the JSON.

**Honest expectation to hold onto:** "playable without edits" is the goal, and it is achievable for
four-on-the-floor material with a clean grid. It will not generalise to swung, rubato, or
polyrhythmic material, and the tool should say so rather than degrade quietly. The tier manifest is
what makes that visible.

---
