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

## Wave 4 — dispatched

W4A (tempo), W4C (note lag), W4D (spectral) are running. W4B is held pending a decision on the
recharter above.
