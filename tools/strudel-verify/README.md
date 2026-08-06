# strudel-verify

Dev-time only. Not part of the Python package, never imported by it, never run during analysis.
The offline-at-runtime constraint in `CLAUDE.md` is untouched: this directory needs the network
once, at `npm install`, on a developer's machine.

## Why this exists

`strudel_vocab.py` pins a date recording when Strudel's **sound names** were transcribed from the
live docs. That was the right call and it covers `bd`, `sd`, `oh`, `cp` and the rest.

It does not cover two other things the project depends on:

1. **Function names.** `setcpm`, `stack`, `arrange`, gain-pattern syntax. These were written from
   memory, never checked.
2. **Placement semantics.** The analysis measures a bass on 16th-steps 2, 6, 10, 14 and emits
   `note("[~ a1]*4")`. Whether that expression actually places notes on those steps was an
   assumption, not a verified fact.

The second is the dangerous one. A wrong function name fails loudly. A right function name with
wrong placement produces a patch that runs, sounds plausible, and does not match the record.

## What it does

`npm run verify` builds each expression the pipeline emits, queries one cycle, and asserts the
event onsets land on the 16th-steps the analysis claimed. Exit code 1 on any mismatch.

`npm run api` lists every Strudel name the project emits and classifies it as a library export or a
REPL-only global.

## Findings from the first run

- All eight placement claims verified against `@strudel/core` 1.1.0. In particular
  `note("[~ a1]*4")` does put notes on steps 2, 6, 10, 14, and `s("oh*8").gain("[0.35 0.5]*4")`
  does put the quieter value on the downbeat and the louder one on the offbeat.
- **`setcpm` is not a library export.** It is a REPL global, defined in strudel.cc's eval scope.
  Patches pasted into the browser REPL work; the same text evaluated as a library does not. `cpm`
  *is* a core export. Any future codegen must know which side of that line each name sits on, which
  is what `api-surface.mjs` is for.
- The Strudel source has moved to Codeberg (`codeberg.org/uzu/strudel`), not GitHub. Docs are
  generated from JSDoc into a `doc.json` via `npm run jsdoc-json` in that repo. That file is the
  machine-readable API reference if a fuller surface check is ever wanted.

## Versions

Pinned to `@strudel/core` and `@strudel/mini` **1.1.0**. Latest at time of writing is 1.2.6; 1.2.x
pulls a browser-only dependency (`@kabelsalat/web`) that does not resolve under Node, so the pin is
deliberate and load-bearing, not laziness. Revisit when 1.2.x is headless-clean.

Verified: 2026-08-06.

## Adding a case

Add to `CASES` in `verify.mjs` whenever the pipeline learns to emit a new construct. The claim
string should name the measurement, not the syntax — the point is to check that measurement and
expression agree.
