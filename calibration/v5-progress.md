# v5 progress

Running note, per `KICKOFF-v2.md`. Part 1 of `V2-PLAN.md` was written from one track; where real
material disagrees with it, the document is wrong and this file records the correction.

## Wave 0 — complete

- `calibration/v4/` frozen and committed (`2bf2fad`). JSON only; stems stay local and gitignored.
- Real-material fixtures built and committed (`1d13bd0`), pulled forward from W8B step 1 because
  three Wave 4 packages list them in their own "Done when" criteria and the plan schedules them four
  waves later. Generator at `tools/make-fixtures/`, provenance at `tests/fixtures/real/PROVENANCE.md`.
- `SpectralFeatures.centroid_median` added additively so W4D has somewhere to put its output.

### F1 — reproduced exactly

Tempo 132.00 ± 0.01 BPM (N=16 → 132.0068, N=32 → 131.9961; halves agree to 0.003). 147 bars. Kick on
steps 0/4/8/12 at 0.99–1.00 occupancy with off-grid leakage ≤ 0.07. The grid `drum_elements`
declared did not exist is exactly as textbook as F1 says.

One correction to the arithmetic: F1 attributes the drift to the 131.855 BPM figure, which is the
*mix* stem's estimate and what `strudel_hints.json` prints. The grid was actually built from the
*drums* stem's own 132.040 (`cycle_seconds` 1.817625). So the drift that killed it came from a
**0.040 BPM error, not 0.145** — 82 ms over 147 bars, 0.72 sixteenth-steps. The conclusion is
unchanged and the requirement is tighter than stated: four-hundredths of a BPM destroys a
four-minute grid.

Also unstated in Part 1: every source reports its own tempo (mix 131.855, drums 132.040, bass
131.815, other 130.359, vocals 131.992) and different modules silently consume different ones. W6
task 4 fixes this, but it deserves to be a finding in its own right.

### Two cautions found while verifying

1. **The downbeat is four-fold ambiguous on this material.** W4A task 2 says to find it by
   maximising folded energy on steps 0/4/8/12. The kick plays every beat, so that objective scores
   identically at four offsets. A wrong pick rotates every downstream step number by 4 and the output
   still looks plausible. W4A was dispatched with an amended task and a required phase confidence.
2. **A raw band fold cannot see past the kick.** Its broadband transient dominates every detection
   band at steps 0/4/8/12, so folding `band_noise` or `band_air` profiles the kick, not whatever else
   shares those steps. This is what broke F2 below.

## F2 does not survive verification

**Claim.** 87 hits classified as snare against roughly 294 implied by the arrangement; the missing
backbeat is a clap; claps fail the `body_ratio` test so they land in the hat bucket, which is why hat
reported 1240 hits. Therefore add a clap class.

**Measured against the 1872 committed hits in `calibration/v4/.../analysis/drums.json`**, folded onto
the verified 132.000 BPM grid. Four independent checks, all pointing the same way:

| check | result |
|---|---|
| Band ratios of the 246 "hat" hits on steps 4 and 12 | `kick_ratio` **0.84**, `noise_ratio` 0.038, `air_ratio` 0.052 |
| Band ratios of the 436 "hat" hits on steps 2/6/10/14 | `kick_ratio` 0.04, `noise_ratio` 0.302, `air_ratio` 0.527 |
| Where the 87 snare-classified hits sit | steps 10, 11, 13 (15, 11, 15 hits); steps 4 and 12 hold 1 and 3 |
| 1–6 kHz flux at steps 4/12 vs 0/8 (raw fold, no classification) | 0.90 / 0.71 vs 1.00 / 0.96 — **lower**, not higher |

The hits at steps 4 and 12 are not claps. A clap has no low end; these have 84% of their energy in
20–150 Hz. They are the **kick being detected a second time** by the noise- and air-band detectors,
which is a direct consequence of the per-band independent peak-picking that `DrumDecomposition`'s
docstring describes as a feature — coincident kick and hat are two hits by design, and nothing
downstream notices when the "hat" is the kick's own transient.

The snare hits cannot be a backbeat either. Only rotations that are multiples of 4 keep the kick on
0/4/8/12, and none of them map step 10 or 13 onto 4 or 12.

**What is actually wrong.** Of the 1240 hats, 504 sit on steps 0/4/8/12 carrying `kick_ratio`
0.72–0.84. Strip them and the count is ~736, which is eighth notes across the bars that are playing.
F2 was right that 1240 is wrong. It was wrong about why: the inflation is duplicate kick detection,
not misfiled claps.

**The honest limit of this result.** House does normally put a clap on 2 and 4, coincident with the
kick, and a clap buried under a kick would have its features measured over a window the kick
dominates. So this does not prove no clap is present in the audio. But if one were present *and
detectable*, the 1–6 kHz energy at steps 4/12 would exceed that at steps 0/8, and it is lower on both
the classified hits and the raw band fold. Either it is absent or it is invisible to the method W4B
was told to use. Either way W4B's "Done when — a clap class on steps 4 and 12" is not reachable.

**Consequence.** W4B is held, not dispatched. `"clap"` was deliberately *not* added to
`DRUM_CLASSES`. W4D was dispatched with its clap-sound task (`cp`) struck.

## Wave 4 — partial

| package | state |
|---|---|
| W4A tempo | **complete and independently verified.** Committed. |
| W4B drums | **complete and independently verified.** Committed. |
| W4C note track | **complete and independently verified.** Committed. |
| W4D spectral | **complete and independently verified**, after one revision. Committed. |

### W4A — verified

`refine_bpm_from_envelope` on the committed fixture returns **131.99996 BPM**, 0.00004 from the
verified 132.000, r=0.735, confidence `high`. Stability halves 131.9986 / 132.0017, delta 0.0031 →
`high`. 67 tests, ruff and mypy clean on both its files.

It corrected the orchestrator on two measurements and was right both times:

1. **Parabolic interpolation cannot reach 0.01 BPM.** The autocorrelation peak is a symmetric lobe
   but not a parabola, so a 3-point fit is biased by sampling phase and the bias does not shrink with
   track length: +0.0050 BPM at 120, +0.0112 at 145, **+0.0230 at 174** — failing outright exactly
   where corpus row 5 lives. The peak lobe's centroid is unbiased (≤0.0005 across the same range).
   `V2-PLAN.md` W4A task 1 specifies the parabola; the plan is wrong.
2. **The orchestrator's "N=64 is a wrong peak, do not extend" was a search-window artifact.**
   Reproduced exactly: ±3% *of the lag* is ±1.9 beats at N=64, wide enough to reach the neighbouring
   multiples, giving 129.97. With a beat-sized window the same lag gives **131.9973**. Verified
   independently. The plan's ±3% is correct as a *guard* on the result and wrong as a *search window*;
   the module now uses `min(3%, 0.45/N)` beats. Corrected in `tests/fixtures/real/PROVENANCE.md`.

It also found a real bug while writing tests: a kick-only source resolved a bar phase out of float
residue in the bright band (1.4e-07 of the low band's energy) and returned a wrong downbeat at 0.5
confidence. Fixed with an activity floor matching `drum_elements`' existing `band_activity_floor`.

**On the downbeat:** the bar phase is **two-fold** degenerate here, not four-fold as the orchestrator
reported. The 6–16 kHz band separates beats {1,3} from {2,4} by 11×, halving the ambiguity; folding
at 2 and 4 bars shows no asymmetry beyond one bar at all. The survivor is broken by convention (first
significant onset) at **0.2322 s**, agreeing with F1's 0.228 to within one STFT frame, with phase
confidence 0.115 and the alternative named in `unresolved_offsets`. The orchestrator's 1.6283 s came
from a deliberately degenerate objective and is 32 ms late; prefer 0.2322.

`find_downbeat` therefore returns a `DownbeatFit`, not a `float` as the plan specifies, and keeps
`beat_offset_seconds` separate from `offset_seconds` — on this track the first is trustworthy and the
second is a coin toss, and one number cannot carry both facts.

**Merge-order coupling:** `tempo.py` imports `_stft_magnitude`, `_band_envelope`, `_spectral_flux`
and `_clean` from `drum_elements.py`, which W4B owns. W4B has been told to keep all four stable.

### W4C — verified

Re-ran `segment_notes` on `tests/fixtures/real/madonna__bass_f0.npz` at the verified 132.000 BPM,
independently of the agent's own numbers:

| metric | v4 | v5 | target |
|---|---|---|---|
| mean absolute quantisation error, no offset removed | 0.2814 | **0.1090** | < 0.15 |
| circular mean offset | 0.2916 steps (33 ms) | **0.0685** (7.8 ms) | near zero |
| concentration `R` | 0.548 | **0.724** | — |
| notes on steps 2 / 6 / 10 / 14 | n/a (`step` was null) | **103 / 103 / 103 / 103** | F7's prediction |

`R` rising is the load-bearing part: subtracting a constant leaves `R` unchanged, so the rise proves
per-note variance was removed rather than a mean shifted. Residual 7.8 ms is below the 11.6 ms frame
hop — that is the frame grid, not an error.

### F3's mechanism is wrong (third Part-1 finding to fail verification)

F3 attributes the 32 ms lag to "the median filter plus the note-segmentation logic" in
`note_track.py`. It is not there. W4C measured the lag at 0.2925 steps in the **raw voicing-run
onsets, before this module sees them**, and sweeping the median filter width 1→11 moves it by
0.0006 steps. A centred odd-width median has zero group delay at a step edge, so the "obvious"
derivation `(W-1)/2 * hop` would have been a 23 ms over-correction dressed as physics.

The real mechanism is the tracker's voicing-confidence ramp: a YIN-family estimator needs several
periods of the new fundamental before periodicity clears its gate, so the lag is **pitch-dependent**
(23.6 ms at 110 Hz, 31.6 ms at 65 Hz), not constant. It is recoverable because backends write an F0
for every frame including the ones their own gate rejects, and that F0 is already correct while
confidence is still climbing.

The finding — a real, roughly-constant 33 ms lag that reproduces exactly — stands. Its stated cause
does not.

### Also from W4C, for later waves

- ~225 of the 709 notes are octave-flip points, not onsets: the tracker flips between the 55 Hz
  fundamental and its 110 Hz second harmonic mid-note. The existing octave guard cannot catch it —
  the runs are longer than `OCTAVE_MEDIAN_FRAMES` (21) and voicing dropouts split them. Deliberately
  left alone; a v6 item.
- The F6 caveat window (0.15–0.30 voiced fraction) is reasoned, not measured — nothing in the corpus
  lands in it. W8B's ambient/rubato track should decide it.

### W4B — verified, and it corrected the orchestrator's reasoning

Run end to end on the real drums stem, wired to W4A's live output (131.99996 BPM, downbeat 0.2322 s):

| | v4 | v5 |
|---|---|---|
| grid status | `no_grid` | **`ok`**, 16 steps |
| quantisation error | 0.2875 | **0.0332** (allowance untouched at 0.18) |
| kick occupancy on 0/4/8/12 | not reported | **0.96 / 0.94 / 0.95 / 0.91** |
| hat count | 1240 | **784** |
| hat occupancy on 2/6/10/14 | not reported | 0.76 / 0.84 / 0.92 / 0.94 |
| hat occupancy on the kick's own steps | — | ≤ 0.11 |
| total hits | 1872 | 1422 |
| kick count | 487 | **487, unchanged** |

The offbeat eighths the hats land on are the same four steps W4C's bass lands on, from a completely
independent code path.

**The orchestrator's central argument for the F2 disconfirmation was void, and W4B said so.** The
claim was: the 246 backbeat "hats" carry `kick_ratio` 0.84, a clap has no low end, therefore they are
duplicate kick detections. But a *genuine* hat sitting on top of a kick measures **0.99** on the same
statistic, because the analysis window contains the kick either way. `kick_ratio` cannot separate
bleed from a real coincident hit and should not be cited again.

The conclusion survives on evidence the orchestrator did not have. The discriminator that works is
`air/(air+noise)` **conditional on a kick being present**: a kick's bright content is a 1–6 kHz beater
click with a 6–16 kHz shoulder, a hat is the reverse. Measured in scope: n=878, median 0.201, p95
0.295, max 0.574, against 0.9894 for a closed hat over a kick and 0.9981 for an open hat. The 0.50
boundary is where the two halves of the bright spectrum hold equal energy — physical, not a tuned
midpoint. 875 of 878 suppressed.

Two further discriminators were measured and **rejected**: absolute air level (this kick deposits
4.5× the air energy of the record's actual hats) and raising `hat_air_over_noise` (bleed 0.20 against
a closed hat over a snare at 0.0552 — any threshold catching the first deletes real hats).

**Stated cost, pinned by a test rather than left to be discovered:** where the kick's beater click
outweighs the hat, the hat is suppressed too.

### Three things W4B found that W6 must act on

1. **A supplied downbeat needs its sub-step phase refined or the grid is rejected.** The
   orchestrator's 1.6283 s is 0.267 steps (30 ms) from where the hits actually are: fitted raw it
   scores 0.2672 and fails; snapped to the hits' own phase it scores 0.0332. The snap is a circular
   mean bounded by half a step, so it **cannot renumber a step** — the two-fold bar ambiguity stays
   the caller's problem. Nothing anticipated this and it would have looked exactly like F1 again.
2. **`beat_times[0]` is not a downbeat.** At 0.348299 s the kick lands on steps 3/7/11/15. W6 must
   pass `downbeat_seconds` from `tempo.find_downbeat`, never fall back to the beat list.
3. **Steps 0/8 and 4/12 are not the same sound.** Air-band envelope peaks measure 1909 / 208 / 1904 /
   145 while the kick band is equal across all four (137k / 155k / 138k / 159k). Two alternating kick
   layers, or a bright element on beats 1 and 3 only.

Point 3 independently corroborates W4A, which found the 6–16 kHz band separates beats {1,3} from
{2,4} by 11× (this measures 9.2× and 13×), from a different module on a different code path. It also
makes the clap story *less* supported than the orchestrator's disconfirmation did: the bright energy
sits on steps 0 and 8, not on 4 and 12, which is the opposite of where a backbeat clap would put it.

### F1, now diagnosed rather than merely failing

At the v4 tempo the module no longer reports a bare `no_grid`. It reports: *"the hits DO fit a grid
that is drifting: each half fits its own phase to 0.09 and 0.08 steps, while the two halves disagree
by +0.37 steps. That is a period error accumulating, not loose playing. Implied cycle 1.817053 s
(132.082 BPM) — approximate, because some hits have already wrapped. Re-fit with a measured period."*
At the mix's 131.855 BPM it correctly says the hits fit no grid at all.

### Also from W4B

- The plan's "stop using per-hit picking to decide whether a grid exists" is **half right**. A fold
  contrast can score *higher* at a wrong tempo than at the right one (0.773 at a wrong 97.3 BPM
  against 0.801 at its own), and three cycles is not enough material for a median. Both gates are
  required; the module reports which failed.
- `1 − mean/peak` on a fold is a **sparsity** measure, not a periodicity one — 40 uniformly random
  kicks score 0.755. Used only to choose the subdivision.
- Corrected hat count is **784, not the ~736 the orchestrator estimated** — reported, not tuned
  toward. The extra ~48 are 16th-note decoration the "eighths across playing bars" estimate misses.
- Correction to the orchestrator's shared-helper list: `_clean` is defined **in `tempo.py`**, not
  imported from `drum_elements`. The shared surface is three names, not four, and a new test pins
  their origin, qualname and signature.

## W4D's `centroid_median` definition is wrong and must be revised

W4D implemented `centroid_median` as the **global energy median**: the frequency below which half
the source's total spectral energy sits. That follows the docstring the orchestrator committed, which
was itself muddled — it said "energy-weighted centroid" and then described a median. A centroid is a
first moment, not a median. The naming error is the orchestrator's; the consequence is measurable.

Measured on synthetic signals with known answers, all gated 50% over a −82 dBFS floor:

| case | frame-mean (v4) | energy median | **energy-weighted centroid** |
|---|---|---|---|
| pure sine 55 Hz | 4987.8 | 64.6 | **55.0** |
| sine sub 41 Hz (+2% h2) | — | 43.1 | **41.0** |
| saw bass 55 Hz | — | 64.6 | **109.0** |
| square bass 55 Hz | — | 64.6 | **86.6** |

**The median returns 64.6 Hz for a sine, a saw and a square alike.** `suggest_bass_sound()` exists to
choose between `sine`, `saw`, `square` and `triangle`. A descriptor that returns one number for all
three carries no information for the only decision it feeds. The branch now fires, but its `sine`
verdict is not evidence-backed — the same value comes out for a saw.

The same collapse shows on real stems: energy median reads **107.7 Hz for both the bass stem and the
drums stem**; the energy-weighted centroid reads 139.7 and 412.2.

The energy-weighted mean of per-frame centroids and the aggregate-spectrum centroid are the **same
statistic** (both are `Σ f·P / Σ P` over all bins and frames) — verified identical to 0.1 Hz on four
real stems. So there are only two candidates, not three, and the centroid wins on every test.

It is also exact where the median is not: 55.0 Hz against a truth of 55, versus the median's 64.6
(bin-quantised, ±21.5 Hz) or 44.8 (interpolated). W4D documented that cost honestly as "read it as a
register, not a pitch" — but a register is not enough to pick a waveform.

**Consequence for the threshold.** `SUB_BASS_CENTROID_HZ_MAX = 120.0` is marked `[guess]` and was
never calibrated. Against the centroid, a synthetic saw reads 109 and a square 86.6, so 120 admits
both — it is too permissive, not too strict. The Madonna bass reads 139.7 and would not clear it.
Recalibrating it against a correct descriptor is the work F4 actually implies, and it is *not* the
"widen the threshold to hide a broken descriptor" trap: the descriptor is being fixed first, and the
threshold has never been calibrated against any correct input. That distinction must be held.

### W4D revision — verified, and the calibration answer was "no"

`centroid_energy_hz` is now `Σ f·P / Σ P` over the aggregate spectrum. A 55 Hz tone reads **54.91 Hz**
against the median's 64.6. `rolloff_energy_hz` is wired from the same aggregate power. Cross-backend:
`rolloff_energy_hz` exact in all 16 cases; `centroid_energy_hz` worst delta 0.240 Hz (0.031%), and the
reason is structural rather than a defect — a centroid is continuous in the input, so the known
690-vs-691 frame difference perturbs it where a percentile absorbs it on the shared bin grid. The two
worst cases are both short clips; the 4:27 track's four stems agree to 0.0002 Hz.

**The threshold could not be calibrated, and that is the result.** Band-limited additive synthesis
across f0 ∈ 35–110 Hz, re-measured independently by the orchestrator through the real backend:

| | max sine/triangle | min square/saw |
|---|---|---|
| `centroid_energy_hz` | **114.0** | **108.8** |
| `brightness` | 0.00006 | 0.00890 |

**The classes overlap on centroid** — a square an octave down looks exactly like a sine an octave up —
so no absolute-Hz ceiling separates them across a 3:1 register. This is intrinsic. `brightness`,
which is pitch-independent by construction, separates the same signals by **149×**.

So the sub-bass branch was restructured rather than retuned: `SUB_BASS_CENTROID_HZ_MAX` 120 → **150**,
demoted to a "is the energy in the sub register" sanity check, and `SUB_BASS_BRIGHTNESS_MAX`
0.05 → **0.005**, now carrying the actual discrimination.

The orchestrator's prediction that 120 was "too permissive" was half right and the full picture is
worse: **120 was wrong in both directions at once**, admitting a square down to ~41 Hz while
rejecting the real sub-bass stem at 139.7. And at 0.05 the brightness clause was entirely inert — the
brightest sawtooth in the register reads 0.0418, so every square and every sawtooth passed it.

`SUB_BASS_LOW_RATIO_MIN` was deliberately **left at 0.75**. Tightening to ~0.96 would separate the
synthetic classes perfectly and reject the one real sub-bass stem at 0.916, because real stems carry
separation bleed that synthesis does not. Correctly refused.

**Caveat to carry into W8B:** the accept side of `SUB_BASS_BRIGHTNESS_MAX` rests on **n=1** — the
single Madonna bass stem at 0.0021. Synthetic sines read 0.00006, thirty-five times lower, so real
bleed dominates the margin and one record is setting it. Both margins are under 3×. The failure
direction is safe (a real sub-bass reported as `none` rather than a saw reported as `sine`), which
matches this module's standing rule that inventing a plausible sound name is worse than saying
nothing — but it needs the corpus.

Also for W8B: `HARMONIC_BASS_LOW_RATIO_MAX = 0.55` makes that branch unreachable for a low-register
harmonic bass — a synthetic 55 Hz sawtooth has a low ratio of 0.868, because at that fundamental even
its harmonics sit mostly under 250 Hz. Documented, not changed. And a heavily filtered sawtooth is
indistinguishable from a sine on every descriptor here, which is the correct answer (the harmonics
really are gone) and part of why both bass verdicts stay `match="approximate"`.

Verdicts across all three v4 tracks, identical on both backends: Madonna `approximate`/`sine`
(**F4 fixed**), the other two unchanged.

### Accepted from W4D without change

- **`brightness` needs no correction** — measured immune (0.13793 either way, unchanged to five
  decimals), because it was already energy-summed over all frames. Nothing added. Correct call.
- **`rolloff_mean` is badly contaminated** — 4333 Hz (librosa) / 1097 Hz (essentia) on the Madonna
  bass against an energy-weighted 85% rolloff of 215 Hz, on a stem with 2e-06 of its energy above
  6 kHz. A corrected `rolloff_energy_hz` is warranted.
- **`centroid_median` is not a scale replacement for `centroid_mean`,** and W8B step 3 must not treat
  it as one. The Madonna drums stem reads `centroid_mean` 4373 Hz and a corrected centroid 412 Hz —
  both correct, because 89% of that stem's energy is kick. Every centroid threshold in
  `heuristics.py` must be re-derived from measurement, never rescaled.
- **`match="approximate"` rather than the plan's `"exact"`** for the bass waveform verdict. W4D's
  reasoning — the pipeline cannot know whether the source was a sine, a filtered saw or a sampled
  808 — is strengthened, not weakened, by the descriptor finding above. Keep `approximate`.
  Contradicts W4D task 3 of the plan; the plan is wrong.

---

## Wave 4 closed

`pytest` **873 passed**, `ruff check .` clean repo-wide, `mypy src` clean. Verified by the
orchestrator, not taken from any agent's report.

### Accumulated for W6

| from | what |
|---|---|
| W4A | promote `TempoFit`, `MultipleFit`, `TempoStability`, `DownbeatFit` into `schemas.py`, **track-level not per-source** |
| W4B | add `"supplied"` to `GRID_ANCHOR_SOURCES`; a tripwire test fails once it lands, telling W6 to delete its shim |
| W4B | pass `downbeat_seconds` into `decompose`, never `beat_times[0]` |
| W4C | call `segment_notes(track, beat_period_seconds=…, downbeat_seconds=…)` directly; it no longer needs to route through `DrumDecomposition` |
| W4D | `centroid_energy_hz` and `rolloff_energy_hz` are already in `schemas.py` and populated |

Note `DrumPattern.step_occupancy` already existed in v4 and is now populated, so W6 task 1's "extend
`DrumPattern` with per-step occupancy" is already done and its clap half is dead.

### Standing caveats for W8B

- `SUB_BASS_BRIGHTNESS_MAX` accept side rests on **n=1**.
- `stability_medium_bpm` (0.50), `downbeat_tie_fraction` (0.25) and `downbeat_conflict_ceiling` (0.50)
  are `[guess]` — corpus rows 3 and 4 settle them.
- The F6 voiced-fraction caveat window (0.15–0.30) is reasoned, not measured.
- `HARMONIC_BASS_LOW_RATIO_MAX` (0.55) makes that branch unreachable for a low-register harmonic bass.
- The kick-bleed rule swallows a hat quieter than the kick's own beater click.

### Scoreboard on Part 1

Four of the eight findings needed correction. Every underlying *problem* was real; the *explanations*
were written from one track without re-derivation and did not survive.

| finding | verdict |
|---|---|
| F1 tempo destroys the grid | **holds.** Arithmetic wrong (0.040 BPM, not 0.145) and the requirement is tighter than stated. Fixed. |
| F2 clap class | **fails.** No clap. The real bug was duplicate cross-band detection. |
| F3 32 ms bass lag | **holds**, cause misattributed. Not the median filter; the tracker's confidence ramp, and pitch-dependent. Fixed. |
| F4 sub-bass branch unreachable | **holds.** The threshold was wrong in both directions at once, and centroid alone cannot carry the branch. Fixed. |
| F5 silent stem reports ok | **holds.** Half fixed in W4C; the `tonal_centre` half is W6. |
| F6 voiced-fraction caveat | **holds.** Fixed. |
| F7 arrangement / harmony / bass placement | bass placement **confirmed** (103/103/103/103). Arrangement is W5A. |
| F8 envelope folding beats onset picking | **half.** Both gates are needed; a fold can score higher at a wrong tempo. |

---

## Corpus (W8B step 4), pulled forward before Wave 5

Run ahead of schedule, on the orchestrator's recommendation, because every threshold Wave 4 set was
calibrated against one house record and W5A was about to add more of the same kind. Four tracks, run
by the user via the CLI. Outputs in `calibration/v5/`.

| row | track | verdict |
|---|---|---|
| 4 · ambient | Brian Eno — "1/1" | **pass** — refused the drum grid |
| 5 · drum & bass | Roni Size — "Brown Paper Bag" | **fail** — half-time at the highest confidence in the corpus |
| 2 · swung | Erykah Badu — "Didn't Cha Know" | **pass** — first non-null `subdivision_feel` in the project's history |
| 3 · live band | Led Zeppelin — "When the Levee Breaks" | **pass** on tempo, **fail** on kick detection |

It found two real bugs and validated three previously-guessed thresholds, in an afternoon, before
anything was built on top of them. Running it last as the plan schedules would have meant
recalibrating W5A and W6 afterwards.

### Bug 1 — nothing detects an octave error in the coarse tempo

Brown Paper Bag is ~170. Essentia reports **84.92**, exactly half, at `bpm_confidence` **1.788** — the
highest confidence anywhere in the corpus, and wrong. `refine_bpm(84.918991)` returns **85.041**:
it faithfully refines the wrong octave. Handed ~170 it returns **170.075** correctly, so the
machinery is right and only the input is wrong.

The evidence is already in the array `tempo.py` computes. On that stem's 20–110 Hz flux
autocorrelation, N=16: **0.380 at 170 BPM against 0.106 at 85**. The true tempo correlates 3.6×
better and nothing looks.

W4A declined this deliberately and said so. That was defensible on one house track and is not
defensible now. Reopened.

**The negative case matters more than the positive one.** Didn't Cha Know reads 135.27 on the mix and
90.08 on the bass — a clean 3:2 that looks exactly like a shuffle misread. `refine_bpm` returns
**high** confidence for *both*. The arbitration is `decompose`: at 135.264 the grid resolves,
`status="ok"`, error **0.112**, hats on all eight even steps at 0.95–0.97. At 90.176 it returns
**`no_grid`** at 0.242. **135 is correct.** A naive "prefer the stronger or faster candidate" rule
fixes Roni Size and breaks Badu. Both tracks were handed to W4A as its test set, with "correlation
alone cannot separate these" named as an acceptable answer.

### Bug 2 — kick detection fails on anything that is not a modern sharp transient

| track | kick hits | |
|---|---|---|
| Madonna | 487 | house, sharp transient |
| Roni Size | **73** | in 303 s of drum and bass — far too few |
| Levee Breaks | **0** | in 430 s |

**Not a band-activity problem.** On the Levee drums stem the `kick` band holds **0.3748 of total
energy, the largest share of any band**, and all four bands clear `BAND_ACTIVITY_FLOOR`. W4B's own
caveat names the real cause correctly — *"they hold nothing, or nothing transient"*. Bonham's kick is
slow-attack, compressed, and recorded in a reverberant stairwell, so the energy rise is gradual and
flux peak-picking has no sharp rise to pick. Reopened, with an explicit instruction that a documented
limitation is preferable to a global sensitivity increase that manufactures kicks on the three tracks
that work.

### Validated: `MIN_AUTOCORRELATION_R = 0.15`

Previously a single-track guess. Across the corpus it separates machine-timed from human cleanly:

| track | autocorrelation r | outcome |
|---|---|---|
| Badu | 0.804 | refined |
| Madonna | 0.735 | refined |
| Roni Size | 0.476 | refined |
| **Levee Breaks** | **0.108** | **`status="coarse"`, confidence 0.000, `low`** |

Row 3's purpose was to catch a confidently wrong refined BPM on a floating tempo. It did the opposite
of failing: `refine_bpm` declined, with the caveat *"no beat multiple produced a usable
autocorrelation peak"*. Bonham drifts and the tool says so.

### Confirmed for W6

- **`bpm_confidence ≤ 0` occurs on every track**, on quiet stems: −0.187 (Eno vocals), −0.030 and
  −0.043 (Badu other/vocals), −0.004 (Levee vocals). It is not a meaningful quantity and nothing
  rejects it. Treat ≤ 0 as unusable.
- **Tempo confidence never reaches the hints file.** On Levee Breaks essentia reports
  `bpm_confidence` 2.12 and `tempo.py` reports r = 0.108 unusable; `strudel_hints.json` prints a bare
  `bpm: 143.25`. The two measurements disagree completely and the hints file trusts the wrong one.
- **Silence gating is confirmed necessary, not theoretical.** The Eno vocals stem at −69.4 LUFS got a
  full analysis and produced a tempo with negative confidence.

### What the corpus did not test

Row 2 was chosen to exercise the swung branch of `infer_subdivision_feel` and did not. "Didn't Cha
Know" has straight 8th hats (all eight even steps at 0.96) with loose micro-timing — quantisation
error 0.112 against Madonna's 0.033, which is the tool measuring human feel rather than failing. That
is a different thing from a shuffle, where onsets sit on triplet subdivisions. **The swung branch
remains unexercised.** A genuine shuffle is still wanted.

The orchestrator suspected a 3:2 tempo error here and was wrong; the grid test above rejected it.

---

## Corpus follow-ups — both closed as measured refusals

Both reopened packages came back with negative results, and both built the thing first and measured
it before refusing. That is the standard this cycle has converged on.

### W4A — no octave corrector exists on this statistic

`r(×2)/r(×1)` at 16 beats, verified independently by the orchestrator:

| case | ratio | required |
|---|---|---|
| badu/bass | 0.03 | stay |
| badu/drums | 1.00 | stay |
| madonna/drums | 1.12 | stay |
| **roni/drums** | **1.21** | **double** |
| eno/other | 1.76 | stay |

The one must-move value sits **inside** the must-stay range, bracketed by Madonna and by the ambient
track whose whole job is to be refused. Structural, not bad luck: autocorrelation decays with lag, so
the doubled candidate is always a shorter lag and is favoured on nearly all material — by about the
magnitude of the harmonic structure such a rule would need to read. A second, independent statistic
(half-beat fold occupancy) inverts outright: 0.267 on the track that must double against 0.760 on the
one that must not, because it measures whether material *subdivides*, not whether the estimate is
halved.

The orchestrator's instruction that a 132 track handed a coarse 66 should now correct **does not
survive**: that case measures 0.96, which says do not double.

Shipped instead: `octave_candidates` as evidence, `bpm` never moves. Roni's 170.069 is surfaced
`live` at the highest r; Badu's 180 is `ruled_out` at r=0.018 — the exact move a naive
prefer-the-faster rule makes. `ruled_out` is a fingerprint rather than a threshold: 8 beats at 90 is
not a whole number of beats at 135, so a 3:2 misread leaves nothing at the doubled lag. Choosing
between *live* candidates needs grid quality, not correlation, and belongs in W6.

`roni__tempo_band.npz` and `badu__tempo_band.npz` are committed so the table is re-derived every run
rather than transcribed from gitignored stems.

### W4B — the fold can see Bonham's kick; per-hit detection cannot

**The orchestrator's diagnosis was wrong, and so was the caveat it quoted.** Levee's kick band is not
slow-attack-with-no-transient: it carries 37% of the stem's energy and peaks at 6.48 flux. It is
gated off by the *sparsity* clause — 0.654 against a 0.72 floor — because the energy never falls back
*between* hits. The band is full of transients that never separate.

**Roni Size is a different bug entirely.** Its kick band is active (sparsity 0.730) and yields 1121
candidates; they die in *classification*, `kick_ratio` median 0.248 against a 0.60 threshold, because
the 250 ms feature window in dense drum & bass catches the whole break so no window is kick-dominated.
That is `feature_window_seconds`, with real Madonna regression risk. **Untouched — open follow-up.**

W4B found a statistic that does separate where sparsity cannot — `q90/peak`, percussive 0.000–0.205
including Levee at 0.057–0.114, stationary 0.369–0.459, a 1.8× gap, and swapping it in changes no
band on any track that currently works. **It built that, measured the result, and reverted it:** with
the band open the picker returns 702 kicks occupying 13 of 16 steps at 0.18–0.53 occupancy and a
quantisation error of **0.247**, where 0.25 is what uniformly random hits score by construction. 702
unplaceable hits is worse than none. The longer-differencing suggestion helps the envelope (fold
contrast 0.560 → 0.645 at n=4, showing a real pattern) and not the picker.

Shipped: a three-way split of why a band was skipped, and a caveat that says the bands are full of
transients that never separate, that this is a compressed or reverberant source, that **any drum
living there is missing from the counts however loud it is**, and that an envelope fold can still see
what per-hit detection cannot. The whole rejected investigation with its numbers is in the docstring
so nobody re-runs it.

All four corpus tracks and Madonna are byte-identical before and after. 881 passed, ruff and mypy
clean.

### Open follow-ups

1. `feature_window_seconds` (250 ms) on dense material — the Roni Size kick classification failure.
2. Envelope-fold kick recovery for compressed/reverberant sources, where the fold sees a pattern the
   picker cannot place.
3. The swung branch of `infer_subdivision_feel` is still unexercised; a genuine shuffle is wanted.
