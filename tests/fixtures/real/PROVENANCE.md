# Real-material fixtures

Derived arrays from one real track, committed so tests can assert against real material without
shipping audio. Ground rule 9 of `KICKOFF-v2.md`.

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

`madonna-i-feel-so-free-peggy-gou-energy-mix-official`, 267.5 s, separated with `htdemucs_ft`.
The v4 analysis of the same stems is frozen at `calibration/v4/`. The audio and the stems are local
only and are excluded by `.gitignore`.

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

## Verified ground truth

Measured by the orchestrator from these fixtures before dispatch, independently of the numbers in
`V2-PLAN.md`, because Wave 4 asserts against them.

| quantity | value | how |
|---|---|---|
| tempo | **132.00 ± 0.01 BPM** | autocorrelation of `band_tempo` flux, parabolic-interpolated peak: N=16 beats → 132.0068, N=32 → 131.9961 |
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
