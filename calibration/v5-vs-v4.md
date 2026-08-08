# v4 → v5, measured

W8B, task 1. Three tracks exist in both `calibration/v4/` and `calibration/v5/` and this is the
diff between them. `v4/` is frozen and was never regenerated; every number below is read out of the
committed JSON.

The question this document exists to answer is not "did the output get better". It is **whether the
two short clips still fail in the same way, and whether the failures are now reported honestly.**
v1's failures were silent. The thesis of this cycle is that an honest refusal beats a confident
guess, and a delta that only listed the new capabilities would be marking its own homework.

| | v4 | v5 |
|---|---|---|
| `schema_version` | 4 | 5 |
| tracks | 3 | 8 |
| tracks with a drum grid | 0 | 3 |
| tracks with an arrangement | 0 | 8 (one of which correctly refuses) |
| tracks whose tempo is refined rather than the backend's | 0 | 3 |

---

## Read this before comparing any single number

**The stems were re-separated.** `calibration/v5/` is a fresh `track-forensics all` run, so Demucs
ran again, and it is not bit-reproducible on this machine. Every stem descriptor moved a little:
Madonna's drums `centroid_mean` 4373.2 → 4390.0, its bass `crest_factor` 4.902 → 5.193, its vocals
`loudness_lufs` −15.21 → −15.45. None of that is code.

The `mix` source is identical in all three tracks, to the last decimal place of every field, because
the mix is the input file rather than a Demucs output. That is the control: **where `mix` differs
between v4 and v5, the code changed; where only stems differ, the separator did.**

The residue stems moved enormously, and that is the most useful single observation in this document:

| showers-of-gold | v4 | v5 |
|---|---|---|
| `vocals.centroid_mean` | 2852.6 | **5511.2** |
| `vocals.brightness` | 0.347 | **0.690** |
| `vocals.crest_factor` | 38.7 | **65.1** |
| `vocals.onset_density` | 3.98 | **6.03** |

Nothing about that stem changed except which run of a stochastic separator produced it. Its
descriptors are not measurements of anything, they are measurements of the residue, and they swing
by 2× run to run. Both v4 and v5 gate it — but v4 gated it only in `heuristics.apply()`, and
everything else in the pipeline went on reading it. v5's `SILENCE_RMS_FLOOR` gate is what stops the
rest of the pipeline doing so, and this table is why it had to exist.

---

## The two short clips: same failures, now stated

Both clips fail for exactly the reasons they failed in v4. Nothing was fixed for them and nothing
should have been — they are 4.3 s and 17.1 s of audio, too short for any of this to work. What
changed is that in v4 they failed quietly and in v5 they say what happened.

### `ancient-heavy-tech-donjon` (4.3 s)

| | v4 | v5 |
|---|---|---|
| drum grid | `too_few_hits`, 0 hits | `too_few_hits`, 0 hits — **unchanged** |
| why no hits | *"they hold nothing, or nothing transient: kick, body, noise, air"* | **split into two distinct reasons** (below) |
| bass line | `ok`, 2 notes, median MIDI 64 | **`silent`**, 0 notes |
| `tonal_centre` | **`E minor`** | **`None`** |
| tempo | bare `bpm: 90.39` | `90.39` + `status: coarse` + `confidence: low` |
| arrangement | — | `ok`, 2 bars, `absent_tracks: [bass, vocals]` |
| notes in `strudel_hints.json` | 3 | 6 |

Three things here are the whole point of the exercise.

**`tonal_centre` was `E minor` and it came from a −70 LUFS stem.** That is F5. The bass stem's
`rms_mean` is 8.2e-05 — digital-silence-adjacent separation residue — and essentia's key detector
read "E minor" off it at 0.688 confidence, which beat the mix's own reading and won the field. v5
prints `None` and says why: *"bass stem is below the silence floor, so it was not used as a
fallback tonal centre — a key detector reads separation residue as confidently as it reads a
bassline."* **Confirmed fixed.**

**The bass line was `ok` with two notes.** Two notes over 4.3 seconds of nothing, presented with the
same status word a 709-note Madonna bass line gets. v5 says `silent`.

**The "no hits" caveat was one sentence covering two different failures.** v4 said the bands hold
"nothing, or nothing transient" — a disjunction, because the module could not tell which. v5
separates them, and on this clip both apply to different bands: the noise and air bands hold
**under 0.001 of the source's energy** (residue), while the kick and body bands are **full of
transients that never separate** (compressed or reverberant, and any drum living there is missing
from the counts however loud it is). That is W4B's three-way split, and it turns "we found nothing"
into two actionable diagnoses.

**What did not change, and should not have:** the grid is still `too_few_hits` and there are still
zero hits. 4.3 seconds is not enough material. The tool now says so instead of leaving the reader to
infer it from an empty list.

### `showers-of-gold` (17.1 s)

| | v4 | v5 |
|---|---|---|
| drum grid | `no_grid`, error 0.238 | `no_grid`, error 0.188 — **still no grid** |
| grid anchor | `beats` (i.e. `beat_times[0]`) | `supplied` (a real downbeat) |
| hits | 72 (65 hat, 7 unclassified) | 45 (**0 hat, 45 unclassified**) |
| drum sound suggestion | `hat` → `oh`, `match="exact"` | **gone** |
| bass line | `ok`, 1 note | **`silent`** |
| `tonal_centre` | `F major` | `F major` (from the mix — unchanged and correct) |
| grid caveats | 3 | 6 |
| tempo | bare `bpm: 150.90` | `150.90` + `coarse` + `low` |

**This clip lost a capability, and the loss is honest.** v4 confidently named 65 hats and suggested
Strudel's `oh`. v5 finds 45 hits, classifies none of them, and explains: the kick band holds under
0.001 of the source's energy, the noise and air bands are full of transients that never separate,
and *"most hits are unclassified, so read the kick/snare/hat pattern as a partial transcription of
this source rather than a complete one."* This is a 17-second clip of a cymbal-dominated loop, and
"I found 45 events and cannot tell you what they are" is a more accurate report than "65 open hats".

It is worth being blunt that this is a real regression in output richness, not only a framing
change. If the v4 answer was right, v5 now withholds it. The evidence says it was not right: the
stem's `band_energy_ratios.low` is 0.001, so there is no kick to anchor anything, and the same
kick-bleed suppression that stripped 457 phantom hats off Madonna's drums is what is running here.

**Still fails the grid, and the reason is now numeric.** v4: *"the best of 16/12 steps per cycle
still misplaced hits by 0.24 steps on average, over the 0.18 allowed."* v5 adds the periodicity
test that preceded it: *"folding the band flux onto this cycle length puts only 0.44 of the profile
on the grid, against 0.5 required and 0.25 expected by chance."* Two independent gates, both
reported, and 0.44 against a 0.25 chance floor says there is *some* structure and not enough.

**One residual leak, in a file W8B does not own.** The bass sound suggestion still reads
`match="approximate", sound="sawtooth"` off this clip's −70 LUFS bass stem, with the evidence
`low_band_ratio 0.356, brightness 0.366, centroid_energy_hz 1899`. Those are numbers from the
residue table above — the ones that moved by 2× between runs. `strudel_vocab.suggest_bass_sound()`
is not silence-gated the way `_tonal_centre` and `_bass_line_hint` now are; it appends a
parenthetical noting the `silent` status and then gives the verdict anyway. It is also the **only**
place in the whole corpus that reaches the harmonic-bass branch at all (see the caveat notes below).
Reported, not fixed: that module belongs to W4D.

---

## Madonna: the track everything was calibrated against

| | v4 | v5 |
|---|---|---|
| reported `bpm` | 131.855 (the mix's backend estimate) | **131.99969** (refined, `high`) |
| `mix.rhythm.bpm` | 131.855 | **131.855 — untouched** |
| drum grid | **`no_grid`**, error 0.288 | **`ok`, 16 steps**, error 0.0338 |
| grid anchor | `beats` | `supplied` |
| hits | 1872 | 1418 |
| kick / snare / hat | 487 / 87 / **1240** | 483 / 87 / **783** |
| kick occupancy on steps 0/4/8/12 | not reported | **0.96 / 0.94 / 0.96 / 0.90** |
| `unavailable_features` on drums | `["drum_decomposition (status=no_grid)"]` | `[]` |
| bass line | `ok`, 709 notes, no `step` | `ok`, 709 notes, **placed on steps 2/6/10/14** |
| bass sound | `match="none"` | **`match="approximate"`, `sine`** |
| arrangement | — | **`ok`, 147 bars, 16 sections** |
| `tonal_centre` | `A minor` | `A minor` (unchanged) |

**F1 is closed.** The tool declared that a textbook four-on-the-floor grid did not exist; it now
reports the grid, and the kick sits on 0, 4, 8 and 12 at 0.90–0.96 occupancy with everything else in
that class under 0.03. The correction was 0.145 BPM on the printed number and 0.040 BPM on the one
`drum_elements` was actually using, and `mix.rhythm.bpm` still reads 131.855 — additive-only held,
so this table is possible at all.

**F2's real bug is visible in the hat count.** 1240 → 783. The 457 that vanished are the kick's own
transient found a second time by the noise and air detectors, and v5 says so in a caveat naming 877
suppressed detections. There is still no clap class and there should not be one; see
`v5-progress.md`.

**F4 is closed.** The bass sound suggestion went from `none` to `approximate`/`sine`, because
`centroid_energy_hz` replaced a descriptor that returned 107.7 Hz for the bass stem and the drums
stem alike.

**The one number that moved for the wrong reason:** `unclassified` went 58 → 65. Seven more hits
landed in the honest bucket. Given the separator re-ran, this is inside the noise.

---

## Was anything lost?

Three things, and all three are recorded here rather than in a footnote.

1. **showers-of-gold's hat transcription**, above. 65 named hats became 45 unnamed hits.
2. **Nothing else.** Every v4 field is present on every source of all three tracks. Spot-checked
   field by field: `rhythm`, `tonal`, `spectral`, `dynamics` and `labels` are populated in v5
   wherever they were populated in v4, and `centroid_mean` / `rolloff_mean` / `rhythm.bpm` all
   survive beside their corrected replacements exactly as ground rule 11 requires.
3. **Heuristic labels barely moved between v4 and v5**, and the small changes that did happen came
   from the re-separation rather than from code — Madonna's vocals `speech/vocal dominant` 0.46 →
   0.44, donjon's drums `percussive` 0.46 → 0.31. The *deliberate* label changes are W8B's centroid
   migration, which landed after these files were written; they are recorded in `v5-progress.md` and
   are not visible in `calibration/v5/`.

## "The correct answer was no grid"

Required by W8B's "Done when", and the corpus supplies four cases rather than one:

| track | verdict | how it says so |
|---|---|---|
| Brian Eno — "1/1" | **no grid, no arrangement, no sections** | `arrangement.status="no_grid"`, drums stem gated `silent` at −67 LUFS, tempo `coarse`/`low` with *"no beat multiple produced a usable autocorrelation peak"*. 17 minutes of ambient and the tool declines all of it. |
| Herbie Hancock — "Chameleon" | no grid | *"folding the band flux onto this cycle length puts only 0.50 of the profile on the grid, against 0.5 required"* — a genuine near miss, and the reason `_GRID_FAILURE_CAVEATS` now prints that figure to three places instead of two. The measurement was right; the format string was hiding the margin. |
| Led Zeppelin — "When the Levee Breaks" | no grid, arrangement kept | tempo refinement refused at r = 0.108; the kick band is *"full of transients that never separate"*, so its 0 kick hits are explained rather than implied |
| showers-of-gold | no grid | above |

Eno is the one that matters. Wiring the refined period straight into `arrangement` initially gave it
52 sections labelled `full` and `groove` across a record with no pulse — inverting the single corpus
outcome that proves the tool can say no. The discriminator that fixed it is not tempo precision but
whether the record has percussion at all.
