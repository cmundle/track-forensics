# calibration

Real-material output, kept per schema version. This directory is a **reference**, not a workspace.

## `v4/` is frozen

`calibration/v4/` is the output of `track-forensics all` at `schema_version: 4`, commit `752ab76`,
captured before any v2 work landed. Every finding in Part 1 of `V2-PLAN.md` is measured against
these exact files, and W8B's job is diffing v5 against them. Only the calibration track remains
here — the two short clips it also covered were dropped from the corpus (see **Tracks** below).

**Never regenerate `v4/` in place.** If a number in it looks wrong, that is the point — the wrong
numbers are the baseline. v5 runs write to `calibration/v5/`.

## What is committed

JSON only: `analysis/<source>.json`, `track_summary.json`, `strudel_hints.json`. The `stems/`
directories exist on disk and are excluded by `.gitignore` (`*.wav`) — no source material is ever
committed, here or in `tests/fixtures/real/`.

## The two write-ups

| file | what |
|---|---|
| `v5-vs-v4.md` | the measured v4 → v5 delta, including what got *worse* and why that is the right answer. |
| `v5-progress.md` | the running record of the whole v4 → v5 cycle. **Overrides `V2-PLAN.md` wherever they disagree** — Part 1 of that plan was written from one track and four of its eight findings did not survive verification. |

## Tracks

Ten, and each one exists to break something different. Four of the five plan rows were added
during the cycle rather than at the end of it, which is why two real bugs and three previously
guessed thresholds were caught before anything was built on top of them.

The **first six** are the v4 → v5 cycle's corpus. The **last four** were added afterwards, each to
take a known bug from one failing example to two, because a threshold tuned against a single track
with Madonna as its only guard is how four packages last cycle got built, measured and reverted.

| track | duration | why it is here |
|---|---|---|
| `madonna-i-feel-so-free-peggy-gou-energy-mix-official` | 267.5 s | **row 1, house at a fixed tempo.** The calibration track: well separated, real levels on all four stems. F1–F7 all trace to it. |
| `erykah-badu-didnt-cha-know` | 243.2 s | **row 2, swung hip-hop.** Chosen to exercise `infer_subdivision_feel`'s swung branch. It returned `straight 8ths` — Dilla's loose micro-timing on a straight hat pattern, which is the correct answer and leaves the swung branch still unexercised by real material. |
| `when-the-levee-breaks-remaster` | 430.3 s | **row 3, live band.** Bonham drifts, and `refine_bpm` declines rather than reporting a confidently wrong tempo (r = 0.108). Also the track that proved kick detection fails on a compressed, reverberant source: **zero** kick hits in seven minutes. |
| `brian-eno-1-1-remastered-2004` | 1041.5 s | **row 4, ambient. The important one.** Expected output is a refusal, and a refusal is a pass. Returns `no_grid`, no arrangement, no sections, and two stems gated as silent. Without this row nothing checks that the tool can say no. |
| `roni-size-reprazent-brown-paper-bag` | 302.7 s | **row 5, drum & bass at 170.** The backend reads 84.92 — exactly half — at the highest confidence anywhere in the corpus. The octave is arbitrated on grid quality, not correlation. |
| `herbie-hancock-chameleon-official-audio` | 947.3 s | funk. Added alongside row 2 as a second candidate for the swung branch; also returned nothing, and its grid is a genuine near miss (0.50 of the profile on grid against 0.5 required). |
| `stevie-ray-vaughan-double-trouble-pride-and-joy-official-audio` | 223.5 s | **shuffle.** Returns `subdivision_feel: "swung 8ths"` — the **first time that branch has fired on real material** in the project's history. Badu and Chameleon were both tried and both came back straight. Tempo refinement declines (`coarse`, `no_grid`), which is the right answer on a live blues trio and does not stop the swing being read. |
| `amy-winehouse-back-to-black` | 240.0 s | **second reverberant kit**, and it reproduces Levee's failure with a different drummer and room: `kick:not_sparse`, the band closed, **zero kick-band candidates**. Milder than Levee, which closes `body` as well — 223 kicks are still recovered from other bands here, so a fix to the sparsity gate can be checked for what it recovers *and* what it breaks. |
| `pendulum-slam-hd` | 259.0 s | **the drum & bass that works.** Roni Size is the only d&b in the corpus and it fails, so nothing guarded a fix to `feature_window_seconds`. Refined 174.00 at high confidence, grid `ok`, kick survival 0.757. Any change that improves Roni must leave this alone. |
| `daft-punk-get-lucky-official-audio-ft-pharrell-williams-nile-rodgers` | 248.7 s | **second fixed-tempo guard**, added because Madonna was the only one. Real drums, bass, vocal and guitar, so all four stems carry genuine content. Refined 116.04 at high confidence, kick survival **1.000** — the cleanest in the corpus. Also found a new bug on arrival: **7 snares**, see below. |

### What the four new rows found immediately

`daft-punk-get-lucky` reports **7 snares against 730 hats** over four minutes of a record with a
backbeat on every bar. The body band produced only 53 candidates (Madonna: 88 over a comparable
length) at a median `body_ratio` of **0.016** against Madonna's 0.445 — a 28x difference. That
snare is bright and thin, it puts almost nothing in 150–500 Hz, and it is being detected in `air`
and classified as a hat. Nothing in the output says so: this is the snare-shaped version of the
kick failure `_kick_survival_caveat` was written for, and that caveat is deliberately kick-only
because `body` does not map 1:1 to `snare` the way `kick` does. Open.

Every row is a full-length record. Two short clips (4.3 s and 17.1 s) were carried through the
v4 → v5 cycle and have since been dropped: they were too short to expose F1 — tempo drift scales
with duration — and Demucs put nearly all of both into `other`, so what they mostly exercised was
separation residue rather than anything worth extracting. The measurements they contributed are
still recorded in `v5-progress.md` and `v5-vs-v4.md`, which are dated write-ups and are not
rewritten after the fact.

**Demucs is not bit-reproducible on this machine**, so every stem descriptor moves a little between
runs and the residue stems move by up to 2×. The `mix` source is the input file rather than a
separator output and is identical across runs, which makes it the control: where `mix` differs
between two versions the code changed, where only stems differ the separator did.
