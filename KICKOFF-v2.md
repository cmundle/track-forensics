# Kickoff prompt — v2

Paste the block below into Claude Code from the repo root, or just say:
*"Read KICKOFF-v2.md and start Wave 4."*

---

You are the orchestrator for taking `track-forensics` from schema v4 to v5. Do not write
implementation code yourself — dispatch subagents, enforce the plan, arbitrate when they disagree.

**Read first, in this order:** `CLAUDE.md` (project constraints), `V2-PLAN.md` (findings and the full
work breakdown), `src/audio_pipeline/schemas.py` (the contract everything is built against). Then
`git log` and the existing calibration outputs to see what the tool currently produces.

`TODO.md` is the v1 build plan and is complete. It is useful as a model for how packages are
specified, not as a list of outstanding work.

## Why this exists

v1 shipped and works. Calibrating it against a real 4:27 house track found four mechanical bugs and
three missing capabilities, all documented with reproducible numbers in Part 1 of `V2-PLAN.md`. The
headline: the tool reported 131.855 BPM for a track that is exactly 132.000, and that 0.145 BPM error
accumulated enough drift over 588 beats to make the tool declare that a textbook four-on-the-floor
grid did not exist.

Read Part 1 properly before dispatching anything. Every package traces back to a numbered finding
there, and an agent that has not read the evidence will re-litigate decisions that are already
settled.

## How the work is structured

`V2-PLAN.md` defines 8 packages across four waves for this cycle. Packages inside a wave are
file-disjoint and run in parallel; waves run strictly in sequence.

```
Wave 0  do this first    W8B step 0    commit the v4 baseline
Wave 4  4 agents         W4A W4B W4C W4D    fix what is broken
Wave 5  1 agent          W5A                arrangement extraction
Wave 6  1 agent          W6                 schema v5 and wiring
Wave 8  2 agents         W8A W8B            terminal visualization, recalibration
```

Harmony (W5B) and code generation (W7) are **deferred** and live in the appendix. Do not dispatch
them. If an agent proposes pulling one forward, the answer is no — settled decision 2.

**Before dispatching anything**, commit the current untracked `calibration/` JSON as the frozen v4
baseline. Every claim in Part 1 is measured against those files, and W8B's whole job is diffing
against them. `.gitignore` already excludes the stems, so this commits JSON only and no audio.

Spawn one subagent per package. Give each agent: its package ID, its **Owns** file list, its
**Tasks**, its **Done when**, the finding(s) in Part 1 it is fixing, and the ground rules below. Tell
it explicitly that it may not edit files outside its Owns list — if it believes it needs to, it must
stop and report to you rather than reaching across the boundary.

Do not open Wave N+1 until every package in Wave N is merged and `pytest`, `ruff check .`, and
`mypy src` are clean on the combined result. Run that check yourself; do not take an agent's word
for it.

## Non-negotiable ground rules

Pass these verbatim to every subagent. The first seven are carried over from v1 and still bind.

1. Python 3.11, full type hints, `from __future__ import annotations` at the top of every module.
2. **44.1 kHz everywhere. Never downsample.** Use `ANALYSIS_SAMPLE_RATE`. Watch for library calls
   that default to `sr=22050`.
3. No network calls at runtime. No cloud SDKs.
4. Never import `essentia` or `librosa` at module top level outside `backends/`.
5. Unavailable descriptors return `None` and append their name to `unavailable_features`. Never crash
   on a missing feature.
6. Band energy uses the shared `BAND_EDGES_HZ` bounds. No per-backend bands.
7. `schemas.py` is frozen. **Only W6 may touch it.** Any other agent that thinks the schema needs to
   change must report the field shape it needs to you, and you hand it to W6.
8. **No new runtime dependencies.** numpy and scipy are already base dependencies and cover
   everything in this plan. `rich` arrives transitively via Typer and may be used for terminal
   output, but must be guarded so the CLI still loads without it.
9. **Tests must not require a real audio file, but real-material regressions must be captured.**
   Commit *derived intermediate data* — flux envelopes, per-bar RMS arrays, note lists — not audio.
   `tests/fixtures/real/` with a provenance note. `.gitignore` already permits synthetic
   `tests/fixtures/*.wav`; no source material is ever committed.
10. **Every threshold is documented where it is defined**, with the reasoning and where the number
    came from. Three of the four bugs in Part 1 are thresholds that were plausible in the abstract
    and wrong against real material. An undocumented constant is a future bug with no paper trail.
11. **Additive only.** A corrected descriptor lands *beside* the one it supersedes, never in place of
    it — `centroid_median` joins `centroid_mean`, `TempoFit.bpm` joins `rhythm.bpm`. Update the old
    field's docstring to say what is wrong with it. This is what keeps the frozen v4 calibration
    outputs usable as a baseline, and it is settled decision 1. An agent that "cleans up" by replacing
    a field has broken the cycle's only means of measuring whether the fixes worked.

## Wave 4 is load-bearing

W4A (tempo refinement) gates almost everything downstream. `drum_elements`, `note_track`,
`arrangement`, and `harmony` all need a correct period and downbeat, and today each derives its own
grid from an estimate that is good enough for a label and not good enough for a grid.

The other three Wave 4 packages are written and tested against a *supplied* period so they do not
block on W4A. W6 wires them to the real one. If W4A slips, the wave still completes.

Do not let an agent "fix" a grid problem by loosening the quantisation allowance. The allowance is
not the bug.

## What to watch for

- **Agents optimising away the 44.1 kHz constraint.** It has happened before. It is in the ground
  rules because it is the easiest thing to break by accident while reaching for speed.
- **Agents widening a threshold instead of fixing the descriptor.** F4 in `V2-PLAN.md` is exactly
  this trap: the sub-bass centroid threshold looks wrong and is actually correct — the descriptor
  feeding it is what is broken. Loosening the threshold would have hidden the real bug.
- **Agents replacing a field instead of adding one.** Ground rule 11. It will look like good
  housekeeping and it destroys the baseline.
- **Scope creep in W8A.** Terminal output only. If an agent proposes HTML, images, a notebook, or a
  server, the answer is no — that is a standing non-goal in `CLAUDE.md` and it has not changed.
- **Agents adding a Node dependency to the Python package.** `tools/strudel-verify/` is dev-time
  only: a developer runs `npm install` there once. Nothing in `src/audio_pipeline/` may import it,
  shell out to it, or require Node to be present. The runtime offline constraint is unchanged.
- **W8B step 4 getting quietly dropped.** Building a five-track corpus is the slowest item in the
  plan and the one with the least visible output. It is also the only thing standing between you and
  a tool tuned to exactly one record. In particular, the ambient/rubato track exists so that "the
  tool correctly refused" is a tested outcome; without it, nothing checks that the tool can say no.

## Reporting

After each wave, write a short note to `calibration/v5-progress.md`: which packages landed, what
changed in the outputs, and anything a package discovered that contradicts `V2-PLAN.md`. Part 1 was
written from one track; if real material disagrees with it, the document is wrong, not the material.
