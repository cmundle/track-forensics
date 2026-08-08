# Real-material fixtures

Derived arrays from the calibration corpus, committed so tests can assert against real material
without shipping audio. Ground rule 9 of `KICKOFF-v2.md`.

Regenerate or extend to a new track with:

```bash
python tools/make-fixtures/make_real_fixtures.py <stems-dir> --track <slug>
```

## Why this is not source material

Every array here is an irreversible reduction. The band envelopes are four energy numbers per
11.6 ms frame — no phase, no per-sample detail, a ~128:1 reduction before compression. The F0 track
is one frequency estimate per frame with all timbre discarded. Nothing here can be turned back into
audible audio. No `.wav` is committed, here or anywhere else in the repo.

## Source

All stems come from `calibration/v5/<track>/stems/`, separated with `htdemucs_ft`. The audio and
the stems are local only and are excluded by `.gitignore`.

| slug | track | duration |
|---|---|---|
| `madonna` | `madonna-i-feel-so-free-peggy-gou-energy-mix-official` | 267.5 s |
| `badu` | `erykah-badu-didnt-cha-know` | 243.2 s |
| `levee` | `when-the-levee-breaks-remaster` | 430.3 s |
| `roni` | `roni-size-reprazent-brown-paper-bag` | 302.7 s |
| `eno` | `brian-eno-1-1-remastered-2004` | 1041.5 s |

`madonna`'s v4 analysis of the same stems is frozen at `calibration/v4/`.

Regenerating a fixture from unchanged stems is byte-identical — verified when `levee` and `roni`
band envelopes were added and the existing `__stem_frame_rms.npz` files came back unchanged. Demucs
itself is *not* reproducible, so that guarantee holds only while the stems on disk stay put.

## The grid every fixture sits on

STFT: `n_fft` 2048, hop 512, periodic Hann, centred by zero-padding `n_fft // 2` each side. This is
`drum_elements._stft_magnitude` — the fixtures call it rather than reimplementing it, so frame `k`
here and frame `k` in the pipeline are the same instant. Hop is 512 / 44100 = **11.60998 ms**, stored
in every file as `hop_seconds`.

## Files

### `madonna__drums_band_envelopes.npz` — 23041 frames

Per-frame energy (`magnitude ** 2` summed over in-band bins) from the drums stem.

| key | band | for |
|---|---|---|
| `band_kick` | 20–150 Hz | `DETECTION_BANDS` |
| `band_body` | 150–500 Hz | `DETECTION_BANDS` |
| `band_noise` | 1000–6000 Hz | `DETECTION_BANDS` |
| `band_air` | 6000–16000 Hz | `DETECTION_BANDS` |
| `band_tempo` | 20–110 Hz | W4A tempo refinement |

Energy, not flux — flux is a pure function of this (`drum_elements._spectral_flux`), so each package
applies the project's own definition rather than inheriting one caller's choice.

`band_tempo` is narrower than `band_kick` on purpose: 110 Hz is A2, so the bound keeps a bass
fundamental out of the envelope the tempo fit reads, while the kick fundamental (40–60 Hz) is fully
inside it.

### `madonna__stem_frame_rms.npz` — 23040 frames

`rms_drums`, `rms_bass`, `rms_vocals`, `rms_other`: RMS per non-overlapping 512-sample block.
`kick_band_energy`: the drums stem's 20–150 Hz envelope, because "the drums stem is loud" and "the
kick is playing" are different facts (W5A task 1).

Deliberately **not** folded into bars. Folding needs a period and a downbeat; baking the answer W4A
computes into the fixture W5A asserts against would make the two agree by construction.

### `madonna__bass_f0.npz` — 23042 frames

`f0_hz`, `voiced`, `voiced_probability` from `essentia_backend.pitch()` (`PitchYinFFT`), plus
`frame_hop_seconds`, `method`, `backend`.

This is the *input* to note segmentation, not the v4 note list. W4C's job is to re-run segmentation
and show the 32 ms onset lag is gone; a committed note list would already have the lag in it.

### `levee__drums_band_envelopes.npz` · `roni__drums_band_envelopes.npz`

Same five arrays and the same grid as `madonna__drums_band_envelopes.npz` above: 37065 frames for
`levee`, 26072 for `roni`.

These exist because kick detection fails differently on each, and neither failure could be measured
deterministically before they were committed — the only way to re-check a fix was to regenerate
stems through Demucs, which is not reproducible, so a real change and separator noise looked alike.

| track | the failure it pins |
|---|---|
| `levee` | the kick band is never searched: flux sparsity 0.654 against a 0.72 floor, because a compressed, reverberant room never lets the band fall back between hits. Zero kicks in 430 s. |
| `roni` | the kick band *is* searched and 236 of its 293 hits die in classification, because the 250 ms feature window catches a whole break and no window is kick-dominated. This is what `_kick_survival_caveat` reports. |

`chameleon` is deliberately absent: its bug is over-detection and a near-miss grid, nobody is
working it, and at 947 s it would be the largest file here.

### `<track>__stem_frame_rms.npz` — five tracks

`rms_drums`, `rms_bass`, `rms_vocals`, `rms_other`, plus `kick_band_energy`, for **all five corpus
tracks**: `madonna`, `roni`, `badu`, `levee`, `eno`. Written by `--rms-only`.

Arrangement extraction thresholds a stem's energy against a percentile of its own distribution to
decide "is this playing in this bar". That is exactly the shape of threshold this cycle spent an
afternoon discovering had been calibrated against one house record. These exist so W5A cannot repeat
it — the four non-Madonna tracks have no committed ground-truth section labels, but they constrain
the threshold's *behaviour*:

| track | what it should constrain |
|---|---|
| `eno` | 1042 s of ambient with no drums at all — presence detection must not manufacture sections |
| `levee` | a famously long solo-drum intro before anything else enters |
| `roni` | dense drum & bass where almost everything is playing almost all the time |
| `badu` | mid-tempo hip-hop with a conventional arrangement |
| `madonna` | the baseline: 147 bars, 16-bar breakdown with kick and bass out |

### `roni__tempo_band.npz` · `badu__tempo_band.npz`

One 20–110 Hz envelope per named stem, same grid and same `hop_seconds`. Written by
`--tempo-band-only`, which skips the full fixture set: for an octave-ambiguity regression a single
band envelope per stem is the whole evidence.

| key | from | what it proves |
|---|---|---|
| `roni__tempo_band.npz` → `tempo_drums` | Roni Size — "Brown Paper Bag" | the coarse estimate is an octave low. r at ×1 is 0.476, at ×2 **0.575** — the true 170 BPM is the stronger peak and nothing acts on it |
| `badu__tempo_band.npz` → `tempo_bass` | Erykah Badu — "Didn't Cha Know" | **the case that catches a naive corrector.** Coarse 90.08, and r at ×2 collapses to **0.018** — 8 beats at 90 is not a whole number of beats at 135, so a 3:2 misread leaves nothing at the doubled lag |
| `badu__tempo_band.npz` → `tempo_drums` | same track | coarse 135.27, r at ×1 0.856 and at ×2 0.857 — indistinguishable, which is why correlation alone cannot decide |

These three exist because the octave measurement in `test_tempo.py` was otherwise a transcribed
table: numbers copied from a run against gitignored stems. A transcribed table can only fail when
someone edits the table. With these committed, the measurement itself is re-derived on every run.

The Badu bass row is the valuable one. Every "prefer the faster candidate" rule ever written passes
Roni and fails here.

## Verified ground truth

Measured by the orchestrator from these fixtures before dispatch, independently of the numbers in
`V2-PLAN.md`, because Wave 4 asserts against them.

| quantity | value | how |
|---|---|---|
| tempo | **132.00 ± 0.01 BPM** | autocorrelation of `band_tempo` flux, parabolic-interpolated peak: N=16 beats → 132.0068, N=32 → 131.9961. `tempo.refine_bpm_from_envelope` later reached **131.99996** on the same array with a better peak estimator — see the correction below. |
| tempo stability | first half 131.9952, second half 131.9982 | agree to 0.003 BPM; this track is machine-timed |
| bars | **147** | at 132 BPM over 267.5 s |
| kick placement | steps **0, 4, 8, 12**, occupancy 0.99–1.00 | envelope-fold of `band_kick`, median across 147 bars; off-grid leakage ≤ 0.07 |

Two cautions attached to that table, both found while measuring it:

1. **The downbeat is four-fold ambiguous on this material.** The kick plays every beat, so
   maximising folded kick energy on steps 0/4/8/12 — W4A task 2 as written — scores identically at
   four different offsets. Any objective for `find_downbeat` needs an asymmetric feature.
2. **A raw band fold cannot see the backbeat here.** The kick's broadband transient dominates every
   band at steps 0/4/8/12, so `band_noise` and `band_air` peak on the kick, not on whatever else
   shares those steps. See `calibration/v5-progress.md` for what that did to F2.

### Correction to this table (W4A)

An earlier revision of this file carried a third caution: *"N=64 returns 129.97 at r=0.63 — a wrong
peak. Do not extend to longer lags."* **That was wrong, and the cause was the measuring script, not
the data.**

The 129.97 came from searching a window of ±3% *of the lag*. At N=64 that window is ±1.9 beats wide,
so it reaches the 63- and 65-beat peaks either side and picks a neighbour. Re-measured with a
beat-sized window (±0.45 beats) the same lag returns **131.9973**. Both numbers reproduce exactly.

The rule that actually holds is that each multiple tolerates a coarse-estimate error of about
`0.45/N` beats — 2.8% at N=16, 1.4% at N=32, 0.70% at N=64. `tempo.py` stops at N=32 anyway, but for
the honest reason: going to 64 would drag the module's usable coarse tolerance down to 0.70% against
the 3% its own guard advertises, while buying no accuracy (132.0005 against 132.0014) at a weaker
correlation (r 0.615 against 0.735).

Two further corrections from the same package, both recorded so nobody re-derives them:

- **Parabolic interpolation of the autocorrelation peak cannot reach 0.01 BPM.** The peak is a
  symmetric lobe but not a parabola, so a 3-point fit carries a sampling-phase bias that does not
  shrink with track length: +0.0050 BPM at 120, +0.0112 at 145, **+0.0230 at 174**. The centroid of
  the peak lobe is unbiased (≤0.0005 across the same range). The numbers in the table above were
  produced with the parabola and are correspondingly ~0.005 out.
- **The bar phase here is two-fold degenerate, not four-fold.** The first caution above understated
  it in one direction and overstated it in another: the 6–16 kHz band does separate beats {1,3} from
  {2,4} by a factor of 11, which halves the ambiguity, but folding at 2 and 4 bars shows no
  asymmetry beyond one bar at all. The surviving two-way choice is broken by convention, not by
  measurement. The verified downbeat is **0.2322 s** (agreeing with F1's 0.228 to within one STFT
  frame); the 1.6283 s offset quoted above came from a deliberately degenerate objective and is
  32 ms late — prefer 0.2322.
