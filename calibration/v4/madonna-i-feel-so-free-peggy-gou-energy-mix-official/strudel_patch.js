// madonna-i-feel-so-free-peggy-gou-energy-mix-official
//
// Derived from the schema v4 analysis plus follow-up measurement of the stems.
// Sound names come from strudel_vocab.py (docs read 2026-08-04). Function names
// (arrange, stack, gain patterns) are from memory and are the part most worth
// checking against the live docs before trusting.
//
// MEASURED, not guessed:
//   tempo        132.000 BPM exactly (autocorrelation of the kick band, r=0.97,
//                stable across both halves of the track)
//   first beat   0.228 s
//   length       146 bars
//   key          A minor, confidence 0.957, agreed by 4 stems independently
//   kick         16th-steps 0, 4, 8, 12. Nothing else. Off-step energy is 1% of peak.
//   backbeat     steps 4 and 12, broadband. Reads as a clap, not a snare.
//   hats         even 16th-steps = straight 8ths, decay ratio 2.14 = open
//   bass         steps 2, 6, 10, 14 = offbeat 8ths, in 103 of ~110 playing bars

setcpm(132 / 4); // one cycle = one bar of 4/4 = 1.81818 s

// ---------------------------------------------------------------- drums
const kick = s("bd*4");
const clap = s("~ cp ~ cp").gain(0.8);
const hats = s("oh*8").gain("[0.35 0.5]*4"); // offbeat slightly louder, as measured

// ---------------------------------------------------------------- bass
// Offbeat 8ths. "[~ x]*4" puts the note on steps 2, 6, 10, 14.
// brightness 0.0021 and low-band ratio 0.916 say pure sub, so sine.
const bassPedal = note("[~ a1]*4").sound("sine").gain(0.9);
const bassOct = note("[~ [a1,a2]]*4").sound("sine").gain(0.9); // octave-doubled variant
const bassMove = note("[~ <d2 e2 f2 d2>]*4").sound("sine").gain(0.9);

// ---------------------------------------------------------------- chords
// Bar-level chord estimate off the 'other' stem. Long stretches of static Am
// with an Em - F - G - Am turnaround at phrase ends.
const padHold = note("[a2,c3,e3]").sound("sawtooth").gain(0.35).cutoff(900);
const padTurn = note("<[e2,g2,b2] [f2,a2,c3] [g2,b2,d3] [a2,c3,e3]>")
  .sound("sawtooth")
  .gain(0.35)
  .cutoff(900);
// breakdown leans Am - C - Em instead
const padBreak = note("<[a2,c3,e3] [c3,e3,g3] [e2,g2,b2]>")
  .sound("sawtooth")
  .gain(0.3)
  .cutoff(1400);

// ---------------------------------------------------------------- sections
// Bar boundaries read off per-bar stem RMS. Cycles == bars, so these are literal.
const intro = stack(kick, hats, padHold); //  0- 15
const stab = stack(padHold); // 16- 18
const groove = stack(kick, clap, hats, bassPedal); // 19- 26
const full = stack(kick, clap, hats, bassOct, padTurn); // 27- 74
const breakdown = stack(hats.gain(0.2), padBreak); // 75- 90  kick and bass both out
const drop = stack(kick, clap, hats, bassOct, padHold); // 91- 99
const stop = silence; // 100-101
const main = stack(kick, clap, hats, bassMove, padHold); // 102-140
const outro = stack(padHold); // 141-145

arrange(
  [16, intro],
  [3, stab],
  [8, groove],
  [48, full],
  [16, breakdown],
  [9, drop],
  [2, stop],
  [39, main],
  [5, outro],
); // = 146 bars
