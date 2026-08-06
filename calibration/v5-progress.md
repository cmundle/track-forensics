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
| W4A tempo | `tempo.py` written (1315 lines, imports clean, mypy clean) but **no tests and 2 ruff errors** — agent killed by a session limit before finishing. Resume, do not restart. |
| W4B drums | **not started** — killed by the same session limit at the first tool call. |
| W4C note track | **complete and independently verified.** Committed. |
| W4D spectral | complete, but its central design choice is wrong. **Send back.** |

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
