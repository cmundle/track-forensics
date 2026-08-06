// Dev-time verification that the Strudel expressions this project emits do what
// the analysis claims they do.
//
// This is the part `strudel_vocab.py`'s pinned read date cannot cover. That date
// records when the *sound names* were transcribed from the docs. It says nothing
// about whether `note("[~ a1]*4")` actually places notes on 16th-steps 2, 6, 10
// and 14 — which is a claim the analysis makes and which, until now, nothing
// checked.
//
// Run:  npm install && npm run verify
//
// Exit code 0 = every expression parsed and landed where expected.
// Exit code 1 = at least one mismatch. The diff is printed.

import { stack, arrange, s, note, sound, silence, setStringParser } from '@strudel/core';
import { mini } from '@strudel/mini';
import { readFileSync } from 'node:fs';

setStringParser(mini);

const STEPS_PER_CYCLE = 16; // one cycle = one bar of 4/4, so a step is a 16th

/** Query one cycle and return the 16th-step index of every event onset. */
function steps(pattern, cycles = 1) {
  return pattern
    .queryArc(0, cycles)
    .filter((h) => h.whole !== undefined)
    .map((h) => +(h.whole.begin.valueOf() * STEPS_PER_CYCLE).toFixed(4))
    .sort((a, b) => a - b);
}

/** Query one cycle and return the event values, in onset order. */
function values(pattern, cycles = 1) {
  return pattern
    .queryArc(0, cycles)
    .filter((h) => h.whole !== undefined)
    .sort((a, b) => a.whole.begin.valueOf() - b.whole.begin.valueOf())
    .map((h) => h.value);
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// ---------------------------------------------------------------------------
// The claims. Each entry is an assertion the analysis pipeline makes when it
// maps a measurement onto a Strudel expression. If a case here fails, the
// pipeline is emitting a pattern that does not match what it measured.
//
// `expect` is the list of 16th-steps the events should land on. Add a case here
// whenever codegen learns a new construct.
// ---------------------------------------------------------------------------
const CASES = [
  {
    claim: 'four-on-the-floor kick, measured on steps 0/4/8/12',
    build: () => s('bd*4'),
    expect: [0, 4, 8, 12],
  },
  {
    claim: 'backbeat clap on beats 2 and 4, measured on steps 4/12',
    build: () => s('~ cp ~ cp'),
    expect: [4, 12],
  },
  {
    claim: 'straight 8th hats, measured on even steps',
    build: () => s('oh*8'),
    expect: [0, 2, 4, 6, 8, 10, 12, 14],
  },
  {
    claim: 'offbeat 8th bass, measured on steps 2/6/10/14',
    build: () => note('[~ a1]*4'),
    expect: [2, 6, 10, 14],
  },
  {
    claim: 'octave-doubled offbeat bass keeps the same placement',
    build: () => note('[~ [a1,a2]]*4'),
    expect: [2, 2, 6, 6, 10, 10, 14, 14],
  },
  {
    claim: 'stacking kick and clap preserves both placements',
    build: () => stack(s('bd*4'), s('~ cp ~ cp')),
    expect: [0, 4, 4, 8, 12, 12],
  },
];

// Value-level checks, where placement is not the point.
const VALUE_CASES = [
  {
    claim: 'gain pattern accents the offbeat, not the downbeat',
    build: () => s('oh*8').gain('[0.35 0.5]*4'),
    check: (vals) => {
      const g = vals.map((v) => v.gain);
      const onbeat = [g[0], g[2], g[4], g[6]];
      const offbeat = [g[1], g[3], g[5], g[7]];
      return (
        onbeat.every((v) => v === 0.35) &&
        offbeat.every((v) => v === 0.5) &&
        'onbeat 0.35 / offbeat 0.5'
      );
    },
  },
  {
    claim: 'arrange gives each section the stated number of cycles',
    build: () => arrange([2, s('bd*4')], [2, s('~ cp ~ cp')]),
    cycles: 4,
    check: (vals) => {
      const names = vals.map((v) => v.s);
      return (
        names.slice(0, 8).every((n) => n === 'bd') &&
        names.slice(8).every((n) => n === 'cp') &&
        `2 cycles bd then 2 cycles cp (${names.length} events)`
      );
    },
  },
];

// ---------------------------------------------------------------------------

let failures = 0;
console.log('Strudel expression verification\n');

for (const c of CASES) {
  let got;
  try {
    got = steps(c.build(), c.cycles ?? 1);
  } catch (err) {
    console.log(`  FAIL  ${c.claim}\n        threw: ${err.message}`);
    failures++;
    continue;
  }
  if (same(got, c.expect)) {
    console.log(`  ok    ${c.claim}\n        steps ${got.join(', ')}`);
  } else {
    console.log(
      `  FAIL  ${c.claim}\n        expected ${c.expect.join(', ')}\n        got      ${got.join(', ')}`,
    );
    failures++;
  }
}

for (const c of VALUE_CASES) {
  try {
    const result = c.check(values(c.build(), c.cycles ?? 1));
    if (result) {
      console.log(`  ok    ${c.claim}\n        ${result}`);
    } else {
      console.log(`  FAIL  ${c.claim}`);
      failures++;
    }
  } catch (err) {
    console.log(`  FAIL  ${c.claim}\n        threw: ${err.message}`);
    failures++;
  }
}

console.log(
  `\n${CASES.length + VALUE_CASES.length - failures}/${CASES.length + VALUE_CASES.length} passed`,
);
process.exit(failures === 0 ? 0 : 1);
