# calibration

Real-material output, kept per schema version. This directory is a **reference**, not a workspace.

## `v4/` is frozen

`calibration/v4/` is the output of `track-forensics all` at `schema_version: 4`, commit `752ab76`,
captured before any v2 work landed. Every finding in Part 1 of `V2-PLAN.md` is measured against
these exact files, and W8B's job is diffing v5 against them.

**Never regenerate `v4/` in place.** If a number in it looks wrong, that is the point — the wrong
numbers are the baseline. v5 runs write to `calibration/v5/`.

## What is committed

JSON only: `analysis/<source>.json`, `track_summary.json`, `strudel_hints.json`. The `stems/`
directories exist on disk and are excluded by `.gitignore` (`*.wav`) — no source material is ever
committed, here or in `tests/fixtures/real/`.

`madonna-.../strudel_patch.js` is the hand-written Strudel patch referenced in the `V2-PLAN.md`
appendix, verified against the real library by `tools/strudel-verify/`. It is committed because it
is the evidence that these measurements are enough to rebuild the track; it is not tool output.

## Tracks

| track | duration | why it is here |
|---|---|---|
| `madonna-i-feel-so-free-peggy-gou-energy-mix-official` | 267.5 s | the calibration track. Well separated, real levels on all four stems, fixed tempo. F1–F7 all trace to it. |
| `showers-of-gold` | 17.1 s | short clip. Demucs put nearly everything in `other`; `bass` and `vocals` sit at the noise floor. F5. |
| `ancient-heavy-tech-donjon` | 4.3 s | short clip, same failure mode as above, plus the silent-stem `tonal_centre` override. F5. |

The two short clips are too short to expose F1 — tempo drift scales with duration. That is itself a
finding: a calibration corpus of clips would have shipped v1's tempo bug indefinitely.
