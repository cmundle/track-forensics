# Kickoff prompt

Paste the block below into Claude Code from the repo root, or just say: *"Read KICKOFF.md and start Wave 0."*

---

You are the orchestrator for building `track-forensics` to v1. Do not write implementation code yourself — your job is to dispatch subagents, enforce the plan, and arbitrate when they disagree.

**Read first, in this order:** `CLAUDE.md` (project constraints), `TODO.md` (the full work breakdown), `src/audio_pipeline/schemas.py` (the data contract everything is built against). Then `git log` to see what already exists.

## How the work is structured

`TODO.md` defines 11 work packages across four waves. Packages inside a wave are file-disjoint and run in parallel; waves run strictly in sequence.

```
Wave 0  1 agent    W0
Wave 1  6 agents   W1A W1B W1C W1D W1E W1F
Wave 2  1 agent    W2
Wave 3  3 agents   W3A W3B W3C
```

Spawn one subagent per package using the Task tool. Give each agent: its package ID, its **Owns** file list, its **Tasks**, its **Done when**, and the ground rules below. Tell it explicitly that it may not edit files outside its Owns list — if it believes it needs to, it must stop and report to you rather than reaching across the boundary.

Do not open Wave N+1 until every package in Wave N is merged and `pytest`, `ruff check .`, and `mypy src` are all clean on the combined result. Run that check yourself; do not take an agent's word for it.

## Non-negotiable ground rules

Pass these verbatim to every subagent:

1. Python 3.11, full type hints, `from __future__ import annotations` at the top of every module.
2. **44.1 kHz everywhere. Never downsample.** Use `ANALYSIS_SAMPLE_RATE`. Watch for library calls that default to `sr=22050` — librosa's loader does, and it will silently cap the spectrum at 11 kHz and corrupt every centroid, rolloff, and high-band ratio in the project.
3. No network calls at runtime. No cloud SDKs.
4. Never import `essentia` or `librosa` at module top level outside `backends/`. The CLI must load and `doctor` must run when neither is installed.
5. Unavailable descriptors return `None` and append their name to `unavailable_features`. Never crash on a missing feature.
6. Band energy uses the shared `BAND_EDGES_HZ` bounds. No per-backend bands.
7. Tests never require a real audio file — use the synthetic fixtures W0 builds.
8. `schemas.py` is frozen. Only W0 may touch it, and only to implement `summary_payload()`. Any other agent that thinks the schema needs to change must report to you instead of editing.

## Wave 0 is load-bearing

`analyze.py` currently holds the backend Protocol, the audio loader, and the orchestration in one file. Three Wave 1 agents would collide there immediately. W0 splits it into `backends/` plus `audio_io.py` and builds the ground-truth fixtures. Nothing else starts until W0 lands.

W1C (Demucs separation) is the one exception — it touches no shared file and can run alongside W0.

## Sequencing detail worth enforcing

W1A and W1B must agree on the brightness and band-energy formulas. W1A writes them as documented module-level helpers; W1B mirrors them exactly. Start W1A slightly ahead of W1B so there is something to mirror, and check the two implementations agree before closing the wave.

## Reporting protocol

Each subagent finishes with: files changed, tests added, what it verified, anything it could not do, and any assumption it made that it is not confident about. You compile these into a short wave summary for Christopher before opening the next wave.

Stop and ask Christopher rather than guessing when: a schema change looks necessary, a package's acceptance criteria turn out to be wrong or unachievable, or two agents produce genuinely incompatible approaches.

## Model assignment

Use per-agent model overrides:

| Package | Model | Why |
|---|---|---|
| Orchestrator (you) | Opus | Holds cross-package invariants, arbitrates conflicts |
| W0 | Opus | Restructure everything else depends on; a bad split costs the whole wave |
| W1A librosa backend | Opus | Krumhansl-Schmuckler key detection is real DSP, not boilerplate |
| W1B Essentia backend | Sonnet | Mostly wiring documented Essentia algorithms |
| W1C Demucs | Sonnet | Well-trodden API, clear acceptance criteria |
| W1D heuristics | Opus | Threshold and confidence design is judgement work |
| W1E strudel hints | Opus | Swing detection from IOI distributions is the subtlest algorithm here |
| W1F orchestration | Sonnet | Plumbing against a fixed contract |
| W2 CLI | Sonnet | Well-specified surface, integration-heavy but not novel |
| W3A README | Sonnet | Run the install, fix what breaks |
| W3B CI | Haiku | Config files |
| W3C calibration | Opus, with Christopher | Needs ears and judgement, not autonomy |

Begin with Wave 0. Report back before starting Wave 1.
