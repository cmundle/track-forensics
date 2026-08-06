// Dumps the Strudel API surface this project is allowed to emit, and flags any
// name the project uses that the library does not actually export.
//
// The distinction that matters: some names work in the strudel.cc REPL but are
// NOT library exports. `setcpm` and `setcps` are the important examples — they
// are REPL globals. A patch pasted into strudel.cc runs fine; the same patch
// evaluated as a library does not. Codegen needs to know which side of that line
// every name it emits sits on.
//
// Run:  npm install && npm run api

import * as core from '@strudel/core';
import * as miniPkg from '@strudel/mini';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const version = require('@strudel/core/package.json').version;

const exported = new Set([...Object.keys(core), ...Object.keys(miniPkg)]);

// Every name this project currently emits or plans to emit.
const USED = [
  'stack',
  'arrange',
  's',
  'sound',
  'note',
  'n',
  'gain',
  'cutoff',
  'silence',
  'setcpm',
  'cpm',
];

// Names known to exist only in the REPL's eval scope, not as library exports.
// Keep this list short and evidenced; anything here must be verified by hand
// in the REPL and dated.
const REPL_ONLY = {
  setcpm: 'verified working in the strudel.cc REPL, not a library export',
  setcps: 'verified working in the strudel.cc REPL, not a library export',
};

console.log(`@strudel/core ${version} — ${exported.size} exported names\n`);

let unknown = 0;
for (const name of USED) {
  if (exported.has(name)) {
    console.log(`  library   ${name}`);
  } else if (name in REPL_ONLY) {
    console.log(`  REPL only ${name}   (${REPL_ONLY[name]})`);
  } else {
    console.log(`  UNKNOWN   ${name}   <-- not exported and not a known REPL global`);
    unknown++;
  }
}

console.log(
  `\n${unknown === 0 ? 'All names accounted for.' : `${unknown} name(s) unaccounted for.`}`,
);
console.log(
  'Emitting an UNKNOWN name produces a patch that fails silently and looks like a Strudel bug.',
);
process.exit(unknown === 0 ? 0 : 1);
