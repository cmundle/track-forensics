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

## The two write-ups

| file | what |
|---|---|
| `v5-vs-v4.md` | the measured v4 → v5 delta, including what got *worse* and why that is the right answer. |
| `v5-progress.md` | the running record of the whole v4 → v5 cycle. **Overrides `V2-PLAN.md` wherever they disagree** — Part 1 of that plan was written from one track and four of its eight findings did not survive verification. |

## Tracks

Eight, and each one exists to break something different. Four of the five plan rows were added
during the cycle rather than at the end of it, which is why two real bugs and three previously
guessed thresholds were caught before anything was built on top of them.

| track | duration | why it is here |
|---|---|---|
| `madonna-i-feel-so-free-peggy-gou-energy-mix-official` | 267.5 s | **row 1, house at a fixed tempo.** The calibration track: well separated, real levels on all four stems. F1–F7 all trace to it. |
| `erykah-badu-didnt-cha-know` | 243.2 s | **row 2, swung hip-hop.** Chosen to exercise `infer_subdivision_feel`'s swung branch. It returned `straight 8ths` — Dilla's loose micro-timing on a straight hat pattern, which is the correct answer and leaves the swung branch still unexercised by real material. |
| `when-the-levee-breaks-remaster` | 430.3 s | **row 3, live band.** Bonham drifts, and `refine_bpm` declines rather than reporting a confidently wrong tempo (r = 0.108). Also the track that proved kick detection fails on a compressed, reverberant source: **zero** kick hits in seven minutes. |
| `brian-eno-1-1-remastered-2004` | 1041.5 s | **row 4, ambient. The important one.** Expected output is a refusal, and a refusal is a pass. Returns `no_grid`, no arrangement, no sections, and two stems gated as silent. Without this row nothing checks that the tool can say no. |
| `roni-size-reprazent-brown-paper-bag` | 302.7 s | **row 5, drum & bass at 170.** The backend reads 84.92 — exactly half — at the highest confidence anywhere in the corpus. The octave is arbitrated on grid quality, not correlation. |
| `herbie-hancock-chameleon-official-audio` | 947.3 s | funk. Added alongside row 2 as a second candidate for the swung branch; also returned nothing, and its grid is a genuine near miss (0.50 of the profile on grid against 0.5 required). |
| `showers-of-gold` | 17.1 s | short clip. Demucs put nearly everything in `other`; `bass` and `vocals` sit at the noise floor. F5. |
| `ancient-heavy-tech-donjon` | 4.3 s | short clip, same failure mode as above, plus the silent-stem `tonal_centre` override. F5. |

The two short clips are too short to expose F1 — tempo drift scales with duration. That is itself a
finding: a calibration corpus of clips would have shipped v1's tempo bug indefinitely.

**Demucs is not bit-reproducible on this machine**, so every stem descriptor moves a little between
runs and the residue stems move by up to 2×. The `mix` source is the input file rather than a
separator output and is identical across runs, which makes it the control: where `mix` differs
between two versions the code changed, where only stems differ the separator did.
