"""Turn raw descriptors into human-readable production clues.

Pure functions only: no audio, no I/O, no backend imports. This is the layer
that is cheap to unit test and easy to tune, so keep it that way.

Three rules hold everywhere in this module:

1. **Any descriptor may be `None`.** A label whose inputs are missing simply
   does not fire. An all-`None` `SourceAnalysis` yields zero labels, never an
   exception.
2. **Confidence is graded, not binary.** Every label runs its inputs through
   `_ramp()`, so a value sitting exactly on its threshold scores 0.0 and
   confidence climbs from there, saturating at 1.0. See `_ramp` for the
   inclusive/exclusive convention.
3. **Every label carries its evidence.** The descriptor values that triggered
   it *and* the thresholds they were compared against land in
   `HeuristicLabel.evidence`, so a reader of the JSON can audit any label
   without re-running the analysis.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from .schemas import HeuristicLabel, SourceAnalysis

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Each label is defined by a *threshold* (where it starts firing, at confidence
# 0.0) and a *saturation* point (where confidence reaches 1.0). Saturation may
# sit below its threshold; that simply means "lower is more confident".
#
# The confidence notes below are candid. `[grounded]` means the number follows
# from something physical or arithmetic; `[guess]` means it is a plausible
# starting point that W3C must calibrate against real material.
#
# W1E (`strudel_hints.py`) imports this dict. `busy_drums_onsets_per_sec` and
# `sparse_onsets_per_sec` are a cross-agent contract: retune the values, never
# rename or remove the keys.
THRESHOLDS: dict[str, float] = {
    # -- Near-silence gate -------------------------------------------------
    # Demucs returns an essentially empty stem whenever the source is not
    # present — every instrumental track yields an empty `vocals`, and plenty
    # yield a near-empty `other`. Every descriptor is then computed on
    # separation residue, and the labeller confidently describes numerical
    # noise. Measured on a real 12 s instrumental mix, the empty vocals stem
    # came back "tonally stable, percussive" with a key of C minor and 53
    # onsets. That is worse than silence, because it does not look wrong.
    #
    # Observed LUFS on that run:
    #     mix     -17.38     drums   -24.58     bass    -18.93
    #     other   -27.11     vocals  -68.23  <- empty stem
    # and on synthetic separation residue, -54.67 (librosa) / -54.62
    # (essentia) — the two backends agree on LUFS to within 0.05 dB, which is
    # why this gate reads LUFS rather than anything backend-specific.
    #
    # [grounded] -50 LUFS sits in the very wide gap between the quietest real
    # source (-27.11) and the residue (-54.6 and below). For scale, mastered
    # music runs -14 to -9 LUFS, so this is ~36 dB below a finished master:
    # inaudible under any playback condition. There is over 20 dB of headroom
    # on both sides, so the exact value is not delicate.
    "silence_floor_lufs": -50.0,
    # [grounded] Essentia's LoudnessEBUR128 clamps at -70.0 for digital
    # silence, so that is the floor of the scale and full confidence.
    "silence_floor_lufs_saturation": -70.0,
    # [grounded] Fallback for when LUFS is unavailable. This is NOT a second
    # calibrated scale — it is a backstop for one specific measured case: on
    # digital silence pyloudnorm returns non-finite and W1A maps that to None,
    # so on librosa the *most* pathological input produces no LUFS at all.
    # Essentia returns -70.0 for the same input. Both report rms_mean 0.0.
    # 0.003 linear RMS is about -50 dBFS, mirroring the LUFS floor. RMS-dBFS
    # and LUFS differ by a few dB on broadband material (the residue above is
    # -60.5 dBFS against -54.6 LUFS), so treat this as coarse. It only decides
    # cases LUFS cannot, and it sits ~7x below the quietest real stem measured
    # (rms 0.0216), so it will not fire on quiet-but-present material.
    "silence_floor_rms": 0.003,
    # [grounded] Digital silence.
    "silence_floor_rms_saturation": 0.0,
    # -- Onset density, onsets/sec -----------------------------------------
    # Reference points are note values at a typical 120 BPM: quarter notes =
    # 2.0/s, 8ths = 4.0/s, 16ths = 8.0/s.
    #
    # [grounded] Below one onset per second is slower than half notes at 120
    # BPM — sparse by any reading, and slow enough that a thin result is the
    # part being empty rather than the onset detector under-counting.
    "sparse_onsets_per_sec": 1.0,
    # [grounded] Total silence is maximal sparseness.
    "sparse_onsets_saturation": 0.0,
    # [grounded] Quarter notes at 120 BPM. The dividing line between "holds a
    # note" and "plays a rhythm", used in both directions (above = active,
    # below = sustained).
    "moderate_onsets_per_sec": 2.0,
    # [guess] Between 8ths (4.0/s) and 16ths (8.0/s) at 120 BPM. Deliberately
    # under straight 16ths because onset detectors merge closely spaced hits and
    # systematically under-report on dense material — a literal 8.0 would mean
    # "busy drums" almost never fires on a real break.
    "busy_drums_onsets_per_sec": 5.0,
    # [guess] 16ths at 120 BPM plus ghost notes: as busy as the label can say.
    "busy_drums_onsets_saturation": 12.0,
    # [guess] Full confidence in "percussive" arrives well before full
    # confidence in "busy" — a part can be unambiguously percussive at 8ths.
    "percussive_onsets_saturation": 6.0,
    # [guess] Roughly one event every four seconds: a pad or drone, not a part
    # that merely happens to be playing slowly.
    "sustained_onsets_saturation": 0.25,
    # -- Crest factor, linear peak/RMS -------------------------------------
    # Reference points: a pure sine is 1.41 (3 dB), pink noise about 4 (12 dB),
    # a heavily limited master 3-5, isolated drum hits with space between them
    # 10-20+.
    #
    # [grounded] 8.0 is 18 dB peak-to-RMS. Sustained material essentially cannot
    # reach this; transient material routinely does.
    "percussive_crest_factor": 8.0,
    # [guess] 26 dB peak-to-RMS: dry one-shots with silence between them.
    "percussive_crest_saturation": 20.0,
    # [grounded] 12 dB peak-to-RMS, the pink-noise figure. At or below it the
    # waveform is dense and continuous rather than event-driven.
    "sustained_crest_factor": 4.0,
    # [grounded] 4 dB, just above a pure sine — nothing real gets flatter.
    "sustained_crest_saturation": 1.6,
    # -- Spectral centroid, Hz ---------------------------------------------
    # [grounded, measured] The "bright" line for TRANSIENT material, which is
    # much lower than intuition suggests. `centroid_mean` averages over frames,
    # and sparse hits leave most frames holding a decaying tail or near-silence,
    # which drags the mean down hard. Measured on the librosa backend: a pluck
    # train with 98% of its energy above 6 kHz still averages only 1302 Hz, and
    # a broadband click train with 70% above 6 kHz averages 2099 Hz. An earlier
    # 4000 Hz gate here made `bright hats` and `bright plucks` unreachable on
    # exactly the material they describe. Band-energy share is the robust
    # brightness cue for percussive sources; this centroid term is only a
    # sanity check alongside it.
    #
    # Lowered from 1200 once essentia could be measured: its centroids run
    # consistently ~10% below librosa's on identical input (white noise 9937 vs
    # 11035, pluck train 1169 vs 1302, click train 1898 vs 2099), and 1200 left
    # `bright plucks` unreachable on essentia by 30 Hz.
    "transient_bright_centroid_hz": 1000.0,
    # [guess] Sustained cymbal territory, where the tails keep the mean up.
    "transient_bright_centroid_saturation": 6000.0,
    # [grounded] The upper bound of the `low` band in `BAND_EDGES_HZ`. A
    # centroid below it means the energy really is sub/low-bass rather than a
    # bass with a bright harmonic stack — exactly the distinction "sustained
    # sub" is drawing.
    "dark_centroid_hz": 250.0,
    # [grounded] Near the bottom of the low band: a pure sub sine.
    "dark_centroid_saturation": 60.0,
    # [grounded] The `low_mid`/`high_mid` boundary region of `BAND_EDGES_HZ`.
    # Above it, most of the spectral mass sits past the musical midrange, which
    # is where broadband noise lives.
    "noisy_centroid_hz": 2500.0,
    # [guess] Roughly where band-limited white noise lands.
    "noisy_centroid_saturation": 8000.0,
    # -- Tonal -------------------------------------------------------------
    # Both labels here read CHROMA ENTROPY, computed by `chroma_entropy()` from
    # the 12-bin `hpcp_mean` that both backends populate. Neither reads
    # `key_confidence` or `tonal_stability`, and there is deliberately no
    # threshold on either descriptor anywhere in this dict.
    #
    # Why: the two backends have opposite working descriptors, so no single
    # threshold on either one can serve both. Measured on the same signals:
    #
    #                    key_confidence      tonal_stability     chroma_entropy
    #                  librosa  essentia   librosa  essentia   librosa  essentia
    #     sine A440     0.1714    0.6880    0.9999    0.7821    0.0868    0.5286
    #     A min triad   0.2035    0.7663    0.9999    0.8003    0.5145    0.7500
    #     white noise   0.0221    0.6953    0.9945    0.2417    0.9999    0.9954
    #
    # `key_confidence` separates cleanly on librosa but on essentia noise
    # (0.695) lands between the sine (0.688) and the triad (0.766) — no signal.
    # `tonal_stability` is the mirror image: excellent on essentia, half a
    # percent of range on librosa. Chroma entropy is the only column that
    # orders correctly on both, and it needs no schema or backend change.
    #
    # [grounded, both backends] Ceiling for "has a tonal centre". Must accept
    # the most tonal material each backend produces — librosa's sine at 0.0868
    # and triad at 0.5145, essentia's sine at 0.5286 and triad at 0.7500 — and
    # reject broadband noise at 0.9939+. 0.80 clears essentia's triad by 0.05,
    # which is the binding constraint; a lower ceiling would make the label
    # unreachable for genuinely tonal material on essentia.
    "tonally_stable_max_chroma_entropy": 0.80,
    # [grounded] Zero entropy is a single pure pitch class.
    "tonally_stable_chroma_entropy_saturation": 0.0,
    # [grounded, both backends] Floor for "no tonal centre at all". Must accept
    # librosa's white noise at 0.9999 and essentia's at 0.9954 and its pink
    # noise at 0.9939, while rejecting librosa's swept vocal at 0.9054. 0.93
    # sits in that gap.
    "noisy_min_chroma_entropy": 0.93,
    # [grounded] A perfectly flat chroma is maximal atonality.
    "noisy_chroma_entropy_saturation": 1.0,
    #
    # The 0.80-0.93 gap between them is a deliberate DEAD BAND: mid-entropy
    # material gets neither label rather than both. Measured examples that fall
    # in it and are correctly left unlabelled: librosa voiced phrases (0.8591)
    # and librosa swept/processed vocal (0.9054).
    #
    # TWO HONEST CAVEATS, both for W3C to settle against real material.
    # 1. This is calibrated on a handful of synthetic signals, not music.
    # 2. Chroma entropy orders tonal-versus-atonal correctly on both backends
    #    but must NOT be used to rank degrees of tonality finely. On essentia
    #    the sine (0.5286) scores flatter than the triad (0.7500), which is
    #    backwards, and essentia's percussive material interleaves with its
    #    tonal material completely (clicks 0.56-0.79 against tonal 0.53-0.77).
    #    The practical consequence: on essentia, `tonally stable` also fires on
    #    click trains and other percussive sources. It is a precision loss, not
    #    a reachability failure, and it does not affect the tonal-versus-noise
    #    split these two labels actually need. On librosa the separation is
    #    clean (tonal 0.09-0.51, percussive and noise 0.99+).
    # -- Band energy ratios, fraction of total spectral energy -------------
    # [guess] On an isolated drum stem the kick is the only sustained source of
    # 20-250 Hz energy, so half the stem's total energy landing there is a
    # strong kick signature. Meaningless on the mix, which is why "kick-heavy"
    # only fires on `drums`.
    "kick_heavy_low_ratio": 0.5,
    # [guess] A stem that is essentially kick and little else.
    "kick_heavy_low_saturation": 0.85,
    # [guess] The 6-20 kHz band is wide but natural spectral tilt keeps very
    # little energy up there. 12% of total energy above 6 kHz on a drum stem is
    # a lot, and points at hats and cymbals rather than kick and snare.
    "bright_hats_high_ratio": 0.12,
    # [guess] An open hat or ride carrying the pattern.
    "bright_hats_high_saturation": 0.35,
    # [guess] `bright plucks` asks for more high-band share than `bright hats`
    # because it is competing against pads and sustained texture on the `other`
    # stem rather than against kick and snare. Measured: a bright pluck train
    # scores 0.977.
    "bright_plucks_high_ratio": 0.25,
    # [guess]
    "bright_plucks_high_saturation": 0.9,
    # [guess] The human voice puts very little energy above 6 kHz unaided, so
    # any meaningful share up there suggests an exciter, aggressive de-essing,
    # or air EQ. Lowered from an initial 0.06 after a synthetic air-boosted
    # vocal measured only 0.041 — 6% was on the wrong side of reachable. Still
    # the shakiest number in this block; W3C should check it against a real
    # vocal stem with and without air EQ.
    "processed_vocal_high_ratio": 0.04,
    # [guess]
    "processed_vocal_high_saturation": 0.2,
    # -- Transients --------------------------------------------------------
    # W1A defines transient sharpness as the mean ratio of an onset-strength
    # peak to its local median, so 1.0 means "no peak at all".
    #
    # [guess] Fully dependent on W1A's helper. 3x the local median is a clear
    # attack, but the scale is unverified — a prime candidate for W3C.
    "sharp_transient_ratio": 3.0,
    # [guess]
    "sharp_transient_saturation": 8.0,
    # -- Vocals ------------------------------------------------------------
    # [grounded] 0.01 linear RMS is -40 dBFS. Below that a separated stem is
    # bleed and separation artefacts, not a performance.
    "vocal_presence_rms": 0.01,
    # [grounded] 0.1 linear RMS is -20 dBFS: a mixed-in lead vocal level.
    "vocal_presence_rms_saturation": 0.1,
    # [guess] Sung and spoken voice centroids typically sit between about 500 Hz
    # and 2.5 kHz. The window is opened a little wider at both ends so a bright
    # or dark vocal is not excluded outright.
    "vocal_centroid_min_hz": 300.0,
    # [guess]
    "vocal_centroid_max_hz": 3000.0,
    # [guess] How far inside each edge of the vocal centroid window confidence
    # takes to reach 1.0.
    "vocal_centroid_margin_hz": 400.0,
    # [guess] How far inside each edge of the "moderate" onset-density window
    # confidence takes to reach 1.0.
    "vocal_onset_margin_per_sec": 1.0,
    # [guess] A centroid standard deviation this large means the timbre is
    # sweeping: a filter, a wide modulated effect, or heavy automation. This is
    # a *proxy* — stereo width is measured nowhere in the pipeline, so
    # "processed/wide vocal" infers width it cannot see. Read it as
    # "processed", and expect W3C to re-scope the label or drop the width claim.
    "processed_centroid_std_hz": 800.0,
    # [guess]
    "processed_centroid_std_saturation": 2000.0,
}


# ---------------------------------------------------------------------------
# Derived descriptors
# ---------------------------------------------------------------------------


def chroma_entropy(hpcp_mean: Sequence[float]) -> float | None:
    """Normalised Shannon entropy of a 12-bin chroma vector, in [0.0, 1.0].

    A third core descriptor, derived here rather than in a backend because it
    needs nothing but `TonalFeatures.hpcp_mean`, which both backends already
    populate on the same 12-bin layout (bin 0 = C, verified in both).

    The 12 values are normalised to sum to 1 and read as a probability
    distribution over pitch classes, then Shannon entropy is divided by
    `log(12)` — the entropy of a perfectly flat chroma — to land on a fixed
    scale:

        0.0  all energy in a single pitch class (one pure tone)
        1.0  energy spread evenly over all twelve (broadband noise)

    This is the only tonal descriptor that orders tonal-versus-atonal material
    correctly on *both* backends; see the `THRESHOLDS` tonal block for the
    measurements and for why `key_confidence` and `tonal_stability` cannot.

    Returns:
        `None` when the vector is empty, is not 12 bins, sums to zero, or holds
        a non-finite value — so a label reading it simply does not fire rather
        than propagating a bad number.
    """
    values = list(hpcp_mean)
    if len(values) != 12:
        return None
    if any(not math.isfinite(value) for value in values):
        return None
    # Chroma is non-negative by definition; clamp rather than trust the backend.
    weights = [max(0.0, float(value)) for value in values]
    total = sum(weights)
    if total <= 0.0:
        return None
    entropy = 0.0
    for weight in weights:
        probability = weight / total
        # Test the probability, not the weight: a tiny-but-positive weight
        # against a huge total underflows to exactly 0.0 here, and log(0.0)
        # raises. A zero bin contributes nothing anyway, since p*log(p) -> 0
        # as p -> 0.
        if probability <= 0.0:
            continue
        entropy -= probability * math.log(probability)
    # Guard the [0, 1] bound against floating-point drift at the extremes.
    return min(1.0, max(0.0, entropy / math.log(12)))


# ---------------------------------------------------------------------------
# Graded-confidence machinery
# ---------------------------------------------------------------------------


def _ramp(value: float | None, threshold: float, saturation: float) -> float | None:
    """Grade how far `value` sits past `threshold`, on a 0.0-1.0 scale.

    The single calibration primitive in this module. Every label's confidence is
    built from one or more of these, so all labels are graded the same way.

    Returns:
        `None` when `value` is `None` or has not reached `threshold` — the
        caller reads that as "this label does not fire". Otherwise a float in
        [0.0, 1.0]: exactly 0.0 at `threshold`, rising linearly, clamped to 1.0
        at `saturation` and beyond.

    Convention:
        **The threshold is inclusive and scores 0.0.** A descriptor sitting
        exactly on the line produces a label with zero confidence rather than no
        label — the label is then honest about being a coin-flip instead of
        silently vanishing. Anything short of the threshold produces no label.

    Direction is inferred from the arguments: `saturation > threshold` means
    higher values are more confident (a bright centroid), `saturation <
    threshold` means lower values are (a dark one).

    **Footgun.** Because direction is inferred rather than declared, retuning a
    threshold past its saturation point silently inverts the label instead of
    failing: it will start firing on exactly the material it was meant to
    exclude. Always move a threshold and its saturation together.
    `test_every_threshold_pair_ramps_in_its_intended_direction` pins the
    intended direction of every pair in `THRESHOLDS` so an inversion fails loudly.
    """
    if value is None:
        return None
    span = saturation - threshold
    if span == 0.0:
        raise ValueError("_ramp needs threshold != saturation to have a gradient")
    progress = (value - threshold) / span
    if progress < 0.0:
        return None
    return min(1.0, progress)


def _all(*confidences: float | None) -> float | None:
    """Combine ramps that must *all* hold, weakest-link style.

    `None` if any input is `None` — a missing descriptor or an unmet condition
    kills the whole label. Otherwise the minimum. Taking the minimum rather than
    a mean stops a barely-met condition being masked by a comfortably-met one,
    so a multi-condition label is never more confident than its weakest
    ingredient.
    """
    resolved: list[float] = []
    for confidence in confidences:
        if confidence is None:
            return None
        resolved.append(confidence)
    return min(resolved) if resolved else None


def _window(value: float | None, lower: float, upper: float, margin: float) -> float | None:
    """Ramp for "value sits inside [lower, upper]", most confident in the middle.

    Both edges are inclusive and score 0.0; confidence reaches 1.0 once the
    value is `margin` inside either edge.
    """
    return _all(_ramp(value, lower, lower + margin), _ramp(value, upper, upper - margin))


def _evidence(**values: float | None) -> dict[str, float]:
    """Build an evidence dict, dropping anything that was not measured.

    Callers pass the descriptor values *and* the thresholds those values were
    judged against, so the JSON shows both what fired and what it had to clear.
    """
    return {name: float(value) for name, value in values.items() if value is not None}


def _emit(
    label: str, confidence: float | None, evidence: dict[str, float]
) -> HeuristicLabel | None:
    """Make a `HeuristicLabel`, or `None` when the label did not fire."""
    if confidence is None:
        return None
    # `_ramp` already clamps, but the schema constrains confidence to [0.0, 1.0]
    # and a validation error would be a poor way to discover a future bug here.
    return HeuristicLabel(label=label, confidence=min(1.0, max(0.0, confidence)), evidence=evidence)


def _collect(*labels: HeuristicLabel | None) -> list[HeuristicLabel]:
    """Drop the labels that did not fire."""
    return [label for label in labels if label is not None]


def _dedupe(labels: Iterable[HeuristicLabel]) -> list[HeuristicLabel]:
    """Keep one entry per label name — the most confident one.

    Some labels are reachable by more than one route: `noisy` fires both
    generically and from `label_other`. The stronger reading wins.
    """
    best: dict[str, HeuristicLabel] = {}
    for label in labels:
        incumbent = best.get(label.label)
        if incumbent is None or label.confidence > incumbent.confidence:
            best[label.label] = label
    return list(best.values())


# ---------------------------------------------------------------------------
# Shared label builders
# ---------------------------------------------------------------------------


def _silent_stem(analysis: SourceAnalysis) -> HeuristicLabel | None:
    """Detect a source too quiet to carry any meaning: an absent stem.

    This is a *gate*, not an ordinary label. When it fires, `apply()` returns it
    alone and suppresses everything else, because every other descriptor on a
    near-empty stem is computed on separation residue and is meaningless. See
    the `silence_floor_lufs` comment for the measurements.

    Emitting a label rather than an empty list is deliberate: "this stem is
    empty" is useful to someone rebuilding a track, whereas no labels at all
    reads as the analysis having failed.

    `loudness_lufs` is the authority whenever it is present — it is perceptual
    and the two backends agree on it closely. It is only absent in one measured
    case, and it is the worst one: on digital silence pyloudnorm returns
    non-finite and W1A maps that to `None`. `rms_mean` covers exactly that gap.
    Note the fallback is only consulted when LUFS is missing, never to
    second-guess a LUFS reading that says the stem is audible.
    """
    loudness = analysis.dynamics.loudness_lufs
    if loudness is not None:
        return _emit(
            "silent/absent stem",
            _ramp(
                loudness,
                THRESHOLDS["silence_floor_lufs"],
                THRESHOLDS["silence_floor_lufs_saturation"],
            ),
            _evidence(
                loudness_lufs=loudness,
                max_loudness_lufs=THRESHOLDS["silence_floor_lufs"],
            ),
        )

    rms = analysis.dynamics.rms_mean
    return _emit(
        "silent/absent stem",
        _ramp(rms, THRESHOLDS["silence_floor_rms"], THRESHOLDS["silence_floor_rms_saturation"]),
        _evidence(
            rms_mean=rms,
            max_rms_mean=THRESHOLDS["silence_floor_rms"],
        ),
    )


def _noisy(analysis: SourceAnalysis) -> HeuristicLabel | None:
    """High spectral centroid plus a near-flat chroma: broadband noise.

    **Why neither `tonal_stability` nor `key_confidence` gates this.** Both were
    tried and both made the label unreachable on one backend or the other.

    `tonal_stability` is 1 - the mean frame-to-frame cosine distance of the
    chroma sequence, and chroma is per-frame normalised — so a *flat* chroma is
    also a perfectly *self-similar* one. On librosa, white noise scores 0.99448
    against a sine's 0.99991: correct ordering, but the whole dynamic range is
    0.0055. On essentia it discriminates well but in the opposite direction from
    librosa's useful range.

    `key_confidence` is the mirror image: it separates cleanly on librosa
    (0.0221 noise against 0.1714 sine) but not at all on essentia, where noise
    at 0.6953 lands *between* the sine at 0.6880 and the triad at 0.7663.

    Chroma entropy is the one descriptor that orders correctly on both, so it is
    what this label reads. The centroid term stays because entropy alone would
    call a quiet atonal pad noisy. Both retired descriptors are still reported
    as evidence — informative for a human reading the JSON, just not
    load-bearing.
    """
    centroid = analysis.spectral.centroid_mean
    entropy = chroma_entropy(analysis.tonal.hpcp_mean)
    confidence = _all(
        _ramp(centroid, THRESHOLDS["noisy_centroid_hz"], THRESHOLDS["noisy_centroid_saturation"]),
        _ramp(
            entropy,
            THRESHOLDS["noisy_min_chroma_entropy"],
            THRESHOLDS["noisy_chroma_entropy_saturation"],
        ),
    )
    return _emit(
        "noisy",
        confidence,
        _evidence(
            centroid_mean=centroid,
            chroma_entropy=entropy,
            min_centroid_hz=THRESHOLDS["noisy_centroid_hz"],
            min_chroma_entropy=THRESHOLDS["noisy_min_chroma_entropy"],
            # Reported for audit only — deliberately not part of the gate.
            key_confidence=analysis.tonal.key_confidence,
            tonal_stability=analysis.tonal.tonal_stability,
        ),
    )


def _sustained(analysis: SourceAnalysis, label: str) -> HeuristicLabel | None:
    """Low crest factor plus few onsets: continuous rather than event-driven."""
    crest = analysis.dynamics.crest_factor
    onsets = analysis.rhythm.onset_density
    confidence = _all(
        _ramp(
            crest,
            THRESHOLDS["sustained_crest_factor"],
            THRESHOLDS["sustained_crest_saturation"],
        ),
        _ramp(
            onsets,
            THRESHOLDS["moderate_onsets_per_sec"],
            THRESHOLDS["sustained_onsets_saturation"],
        ),
    )
    return _emit(
        label,
        confidence,
        _evidence(
            crest_factor=crest,
            onset_density=onsets,
            max_crest_factor=THRESHOLDS["sustained_crest_factor"],
            max_onset_density=THRESHOLDS["moderate_onsets_per_sec"],
        ),
    )


def _sparse(analysis: SourceAnalysis, label: str) -> HeuristicLabel | None:
    """Fewer onsets per second than the sparse threshold allows."""
    onsets = analysis.rhythm.onset_density
    confidence = _ramp(
        onsets,
        THRESHOLDS["sparse_onsets_per_sec"],
        THRESHOLDS["sparse_onsets_saturation"],
    )
    return _emit(
        label,
        confidence,
        _evidence(
            onset_density=onsets,
            max_onset_density=THRESHOLDS["sparse_onsets_per_sec"],
        ),
    )


# ---------------------------------------------------------------------------
# Labellers
# ---------------------------------------------------------------------------


def label_generic(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """Source-independent labels: percussive, sustained, noisy, tonally stable."""
    crest = analysis.dynamics.crest_factor
    onsets = analysis.rhythm.onset_density
    key_confidence = analysis.tonal.key_confidence
    stability = analysis.tonal.tonal_stability

    percussive = _emit(
        "percussive",
        _all(
            _ramp(
                crest,
                THRESHOLDS["percussive_crest_factor"],
                THRESHOLDS["percussive_crest_saturation"],
            ),
            _ramp(
                onsets,
                THRESHOLDS["moderate_onsets_per_sec"],
                THRESHOLDS["percussive_onsets_saturation"],
            ),
        ),
        _evidence(
            crest_factor=crest,
            onset_density=onsets,
            min_crest_factor=THRESHOLDS["percussive_crest_factor"],
            min_onset_density=THRESHOLDS["moderate_onsets_per_sec"],
        ),
    )

    # Reads chroma entropy alone. `key_confidence` and `tonal_stability` each
    # gated this at some point and each made the label unreachable on one
    # backend; see `_noisy` for the measurements. Both are still reported.
    entropy = chroma_entropy(analysis.tonal.hpcp_mean)
    tonally_stable = _emit(
        "tonally stable",
        _ramp(
            entropy,
            THRESHOLDS["tonally_stable_max_chroma_entropy"],
            THRESHOLDS["tonally_stable_chroma_entropy_saturation"],
        ),
        _evidence(
            chroma_entropy=entropy,
            max_chroma_entropy=THRESHOLDS["tonally_stable_max_chroma_entropy"],
            # Reported for audit only — deliberately not part of the gate.
            key_confidence=key_confidence,
            tonal_stability=stability,
        ),
    )

    return _collect(percussive, _sustained(analysis, "sustained"), _noisy(analysis), tonally_stable)


def label_drums(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """e.g. 'busy drums', 'kick-heavy', 'bright hats', 'sparse percussion'."""
    onsets = analysis.rhythm.onset_density
    centroid = analysis.spectral.centroid_mean
    bands = analysis.spectral.band_energy_ratios

    busy = _emit(
        "busy drums",
        _ramp(
            onsets,
            THRESHOLDS["busy_drums_onsets_per_sec"],
            THRESHOLDS["busy_drums_onsets_saturation"],
        ),
        _evidence(
            onset_density=onsets,
            min_onset_density=THRESHOLDS["busy_drums_onsets_per_sec"],
        ),
    )

    kick_heavy = _emit(
        "kick-heavy",
        _ramp(
            bands.low,
            THRESHOLDS["kick_heavy_low_ratio"],
            THRESHOLDS["kick_heavy_low_saturation"],
        ),
        _evidence(
            band_energy_low=bands.low,
            min_band_energy_low=THRESHOLDS["kick_heavy_low_ratio"],
        ),
    )

    bright_hats = _emit(
        "bright hats",
        _all(
            _ramp(
                bands.high,
                THRESHOLDS["bright_hats_high_ratio"],
                THRESHOLDS["bright_hats_high_saturation"],
            ),
            _ramp(
                centroid,
                THRESHOLDS["transient_bright_centroid_hz"],
                THRESHOLDS["transient_bright_centroid_saturation"],
            ),
        ),
        _evidence(
            band_energy_high=bands.high,
            centroid_mean=centroid,
            min_band_energy_high=THRESHOLDS["bright_hats_high_ratio"],
            min_centroid_hz=THRESHOLDS["transient_bright_centroid_hz"],
        ),
    )

    return _collect(busy, _sparse(analysis, "sparse percussion"), kick_heavy, bright_hats)


def label_bass(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """e.g. 'sparse bass', 'sustained sub', 'plucked/transient bass'."""
    onsets = analysis.rhythm.onset_density
    centroid = analysis.spectral.centroid_mean
    sharpness = analysis.rhythm.transient_sharpness

    sustained_sub = _emit(
        "sustained sub",
        _all(
            _ramp(centroid, THRESHOLDS["dark_centroid_hz"], THRESHOLDS["dark_centroid_saturation"]),
            _ramp(
                onsets,
                THRESHOLDS["moderate_onsets_per_sec"],
                THRESHOLDS["sustained_onsets_saturation"],
            ),
        ),
        _evidence(
            centroid_mean=centroid,
            onset_density=onsets,
            max_centroid_hz=THRESHOLDS["dark_centroid_hz"],
            max_onset_density=THRESHOLDS["moderate_onsets_per_sec"],
        ),
    )

    plucked = _emit(
        "plucked bass",
        _ramp(
            sharpness,
            THRESHOLDS["sharp_transient_ratio"],
            THRESHOLDS["sharp_transient_saturation"],
        ),
        _evidence(
            transient_sharpness=sharpness,
            min_transient_sharpness=THRESHOLDS["sharp_transient_ratio"],
        ),
    )

    return _collect(_sparse(analysis, "sparse bass"), sustained_sub, plucked)


def label_vocals(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """e.g. 'speech/vocal dominant', 'sparse vocal', 'processed/wide vocal'."""
    rms = analysis.dynamics.rms_mean
    centroid = analysis.spectral.centroid_mean
    centroid_std = analysis.spectral.centroid_std
    onsets = analysis.rhythm.onset_density
    bands = analysis.spectral.band_energy_ratios

    dominant = _emit(
        "speech/vocal dominant",
        _all(
            _ramp(
                rms,
                THRESHOLDS["vocal_presence_rms"],
                THRESHOLDS["vocal_presence_rms_saturation"],
            ),
            _window(
                centroid,
                THRESHOLDS["vocal_centroid_min_hz"],
                THRESHOLDS["vocal_centroid_max_hz"],
                THRESHOLDS["vocal_centroid_margin_hz"],
            ),
            _window(
                onsets,
                THRESHOLDS["sparse_onsets_per_sec"],
                THRESHOLDS["busy_drums_onsets_per_sec"],
                THRESHOLDS["vocal_onset_margin_per_sec"],
            ),
        ),
        _evidence(
            rms_mean=rms,
            centroid_mean=centroid,
            onset_density=onsets,
            min_rms_mean=THRESHOLDS["vocal_presence_rms"],
            min_centroid_hz=THRESHOLDS["vocal_centroid_min_hz"],
            max_centroid_hz=THRESHOLDS["vocal_centroid_max_hz"],
            min_onset_density=THRESHOLDS["sparse_onsets_per_sec"],
            max_onset_density=THRESHOLDS["busy_drums_onsets_per_sec"],
        ),
    )

    processed = _emit(
        "processed/wide vocal",
        _all(
            _ramp(
                bands.high,
                THRESHOLDS["processed_vocal_high_ratio"],
                THRESHOLDS["processed_vocal_high_saturation"],
            ),
            _ramp(
                centroid_std,
                THRESHOLDS["processed_centroid_std_hz"],
                THRESHOLDS["processed_centroid_std_saturation"],
            ),
        ),
        _evidence(
            band_energy_high=bands.high,
            centroid_std=centroid_std,
            min_band_energy_high=THRESHOLDS["processed_vocal_high_ratio"],
            min_centroid_std=THRESHOLDS["processed_centroid_std_hz"],
        ),
    )

    return _collect(dominant, _sparse(analysis, "sparse vocal"), processed)


def label_other(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """e.g. 'sustained pad-like texture', 'noisy', 'bright plucks'."""
    centroid = analysis.spectral.centroid_mean
    sharpness = analysis.rhythm.transient_sharpness
    bands = analysis.spectral.band_energy_ratios

    # High-band share is the load-bearing brightness cue here; the centroid term
    # only guards against a dark source that happens to have some air. See
    # `transient_bright_centroid_hz` for why that gate sits so much lower than
    # the 4 kHz a "bright" label might suggest.
    bright_plucks = _emit(
        "bright plucks",
        _all(
            _ramp(
                bands.high,
                THRESHOLDS["bright_plucks_high_ratio"],
                THRESHOLDS["bright_plucks_high_saturation"],
            ),
            _ramp(
                centroid,
                THRESHOLDS["transient_bright_centroid_hz"],
                THRESHOLDS["transient_bright_centroid_saturation"],
            ),
            _ramp(
                sharpness,
                THRESHOLDS["sharp_transient_ratio"],
                THRESHOLDS["sharp_transient_saturation"],
            ),
        ),
        _evidence(
            band_energy_high=bands.high,
            centroid_mean=centroid,
            transient_sharpness=sharpness,
            min_band_energy_high=THRESHOLDS["bright_plucks_high_ratio"],
            min_centroid_hz=THRESHOLDS["transient_bright_centroid_hz"],
            min_transient_sharpness=THRESHOLDS["sharp_transient_ratio"],
        ),
    )

    return _collect(
        _sustained(analysis, "sustained pad-like texture"),
        _noisy(analysis),
        bright_plucks,
    )


# Sources without an entry here — including "mix", and anything unrecognised —
# get the generic labels only.
_SOURCE_LABELLERS: dict[str, Callable[[SourceAnalysis], list[HeuristicLabel]]] = {
    "drums": label_drums,
    "bass": label_bass,
    "vocals": label_vocals,
    "other": label_other,
}


def apply(analysis: SourceAnalysis) -> list[HeuristicLabel]:
    """Dispatch to the right labeller(s) for `analysis.source` and merge results.

    The single supported entry point to this module. The near-silence gate
    lives here rather than in the individual labellers, which stay pure "what do
    these descriptors say" functions — so call `apply()`, not `label_drums()`
    and friends, unless you specifically want the ungated reading.

    A source below the silence floor returns `silent/absent stem` alone. Every
    other descriptor on an empty stem is separation residue, so reporting any of
    them would be worse than reporting nothing. The rule is uniform across
    sources: `mix` is never empty in practice, but it is not special-cased.

    Otherwise generic labels always apply, plus the source-specific ones. An
    unknown source name is not an error — it falls back to the generic labels
    alone. Duplicates collapse to the most confident reading, and the result is
    sorted by confidence descending, ties broken alphabetically so output is
    stable run to run.
    """
    silent = _silent_stem(analysis)
    if silent is not None:
        return [silent]

    labels = label_generic(analysis)
    source_labeller = _SOURCE_LABELLERS.get(analysis.source)
    if source_labeller is not None:
        labels.extend(source_labeller(analysis))
    return sorted(_dedupe(labels), key=lambda item: (-item.confidence, item.label))
