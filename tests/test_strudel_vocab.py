"""Tests for `strudel_vocab.py`.

Every model here is hand-built — no audio, no fixtures, no backend. That is
the point of the module: it is pure functions over already-existing models,
buildable and fully testable before WP-A (`drum_elements.py`) or WP-B
(`note_track.py`, bass pitch) exist.
"""

from __future__ import annotations

from audio_pipeline.schemas import (
    SOUND_MATCH_TERMS,
    BandEnergyRatios,
    BassLine,
    DrumDecomposition,
    DrumHit,
    SpectralFeatures,
    StrudelSoundSuggestion,
)
from audio_pipeline.strudel_vocab import (
    ALL_STRUDEL_SOUND_NAMES,
    BASS_WAVEFORM_PALETTE,
    ROLES,
    STRUDEL_DOCS_READ,
    STRUDEL_DOCS_URLS,
    STRUDEL_DRUM_SAMPLES,
    STRUDEL_WAVEFORMS,
    SUB_BASS_BRIGHTNESS_MAX,
    SUB_BASS_CENTROID_HZ_MAX,
    suggest_bass_sound,
    suggest_drum_sounds,
    suggest_sounds,
)


def _hit(
    drum: str,
    *,
    time_seconds: float = 0.0,
    confidence: float = 0.8,
    kick_ratio: float | None = None,
    body_ratio: float | None = None,
    decay_ratio: float | None = None,
) -> DrumHit:
    return DrumHit(
        time_seconds=time_seconds,
        drum=drum,
        confidence=confidence,
        kick_ratio=kick_ratio,
        body_ratio=body_ratio,
        decay_ratio=decay_ratio,
    )


# --- provenance --------------------------------------------------------


def test_docs_read_date_and_urls_are_pinned() -> None:
    assert STRUDEL_DOCS_READ.count("-") == 2  # cheap ISO-date-shape check
    assert len(STRUDEL_DOCS_URLS) >= 3
    assert any("samples" in u for u in STRUDEL_DOCS_URLS)
    assert any("synths" in u for u in STRUDEL_DOCS_URLS)
    assert any("notes" in u for u in STRUDEL_DOCS_URLS)


# --- kick ----------------------------------------------------------------


def test_kick_hits_map_to_bd() -> None:
    decomp = DrumDecomposition(
        status="ok",
        hits=[_hit("kick", kick_ratio=0.7), _hit("kick", kick_ratio=0.9)],
    )
    suggestions = suggest_drum_sounds(decomp)
    kick = next(s for s in suggestions if s.role == "kick")
    assert kick.match == "exact"
    assert kick.sound == "bd"
    assert kick.evidence["hit_count"] == 2.0
    assert kick.evidence["mean_kick_ratio"] == 0.8


# --- snare -----------------------------------------------------------------


def test_snare_with_shell_tone_maps_to_sd_no_alternative() -> None:
    decomp = DrumDecomposition(status="ok", hits=[_hit("snare", body_ratio=0.6)])
    snare = next(s for s in suggest_drum_sounds(decomp) if s.role == "snare")
    assert snare.sound == "sd"
    assert snare.match == "exact"
    assert snare.alternatives == []


def test_snare_without_shell_tone_notes_clap_alternative() -> None:
    decomp = DrumDecomposition(status="ok", hits=[_hit("snare", body_ratio=0.02)])
    snare = next(s for s in suggest_drum_sounds(decomp) if s.role == "snare")
    assert snare.sound == "sd"  # still the primary suggestion
    assert "cp" in snare.alternatives
    assert "shell" in snare.reason.lower()


# --- hat ---------------------------------------------------------------


def test_closed_hat_decay_maps_to_hh() -> None:
    decomp = DrumDecomposition(status="ok", hits=[_hit("hat", decay_ratio=12.0)])
    hat = next(s for s in suggest_drum_sounds(decomp) if s.role == "hat")
    assert hat.sound == "hh"
    assert hat.match == "exact"


def test_open_hat_decay_maps_to_oh() -> None:
    decomp = DrumDecomposition(status="ok", hits=[_hit("hat", decay_ratio=0.9)])
    hat = next(s for s in suggest_drum_sounds(decomp) if s.role == "hat")
    assert hat.sound == "oh"
    assert hat.match == "exact"


def test_hat_without_decay_data_falls_through_to_none() -> None:
    decomp = DrumDecomposition(status="ok", hits=[_hit("hat", decay_ratio=None)])
    hat = next(s for s in suggest_drum_sounds(decomp) if s.role == "hat")
    assert hat.match == "none"
    assert hat.sound is None
    assert set(hat.alternatives) == {"hh", "oh"}


# --- unclassified ------------------------------------------------------


def test_unclassified_hits_present_yields_none_with_alternatives() -> None:
    decomp = DrumDecomposition(status="ok", hits=[_hit("unclassified"), _hit("unclassified")])
    entry = next(s for s in suggest_drum_sounds(decomp) if s.role == "unclassified")
    assert entry.match == "none"
    assert entry.sound is None
    assert entry.evidence["hit_count"] == 2.0
    assert len(entry.alternatives) > 0
    for name in entry.alternatives:
        assert name in ALL_STRUDEL_SOUND_NAMES


def test_unclassified_count_survives_stripped_hits_list() -> None:
    # Mirrors track_summary.json's stripped shape: hits gone, count remains.
    decomp = DrumDecomposition(status="ok", hits=[], unclassified_count=5)
    entry = next(s for s in suggest_drum_sounds(decomp) if s.role == "unclassified")
    assert entry.evidence["hit_count"] == 5.0
    assert "mean_confidence" not in entry.evidence  # no hits to average


# --- coincident classes / robustness ------------------------------------


def test_all_empty_drum_decomposition_yields_no_suggestions() -> None:
    assert suggest_drum_sounds(DrumDecomposition()) == []


def test_mixed_pattern_yields_one_suggestion_per_present_class() -> None:
    decomp = DrumDecomposition(
        status="ok",
        hits=[
            _hit("kick", kick_ratio=0.8),
            _hit("hat", decay_ratio=8.0),
            _hit("hat", decay_ratio=9.0),
        ],
    )
    roles = {s.role for s in suggest_drum_sounds(decomp)}
    assert roles == {"kick", "hat"}


def test_unrecognised_drum_class_string_is_ignored_not_raised() -> None:
    decomp = DrumDecomposition(status="ok", hits=[_hit("tom-tom-not-a-real-class")])
    # Should not raise, and should not invent a role for the bogus class.
    suggestions = suggest_drum_sounds(decomp)
    assert all(s.role in ROLES for s in suggestions)


# --- bass ----------------------------------------------------------------


def _spectral(
    low: float | None,
    brightness: float | None,
    centroid: float | None,
    *,
    centroid_mean: float | None = None,
) -> SpectralFeatures:
    """A `SpectralFeatures` carrying only what `suggest_bass_sound` reads.

    `centroid` is the **energy-weighted** centroid — the field the bass
    verdict thresholds on since schema v5. `centroid_mean` defaults to `None`
    rather than mirroring `centroid`, so any test that accidentally starts
    depending on the contaminated field fails instead of quietly passing.
    """
    return SpectralFeatures(
        centroid_mean=centroid_mean,
        centroid_energy_hz=centroid,
        brightness=brightness,
        band_energy_ratios=BandEnergyRatios(low=low),
    )


# --- F4: the sub-bass branch reads centroid_energy_hz, not centroid_mean ----
#
# Measured on the v4 calibration bass stem
# (`calibration/v4/madonna-.../analysis/bass.json` plus a recomputation of the
# v5 fields from `stems/bass.wav`). A pure sine sub: 91.6% of its energy under
# 250 Hz, 2e-06 of it above 6 kHz. The two backends read the energy-weighted
# centroid as 139.7256 and 139.7254 Hz.
F4_BASS_LOW_RATIO = 0.916392
F4_BASS_BRIGHTNESS = 0.002093
F4_BASS_CENTROID_MEAN_HZ = 1010.696524  # contaminated: 55% of the stem is unvoiced
F4_BASS_CENTROID_ENERGY_HZ = 139.725  # energy-weighted, the number the branch needs


def test_f4_sub_bass_stem_now_resolves_to_sine() -> None:
    """The regression that finding F4 is about, at the numbers it was found at.

    Before v5 this stem returned `match="none"`: the sub-bass branch was
    thresholding `centroid_mean` (1010.7 Hz) against a 120 Hz ceiling, which no
    real stem's frame-mean centroid ever clears. Nothing about the threshold
    changed; the descriptor it reads did.
    """
    spectral = _spectral(
        low=F4_BASS_LOW_RATIO,
        brightness=F4_BASS_BRIGHTNESS,
        centroid=F4_BASS_CENTROID_ENERGY_HZ,
        centroid_mean=F4_BASS_CENTROID_MEAN_HZ,
    )
    result = suggest_bass_sound(BassLine(status="ok"), spectral)
    assert len(result) == 1
    assert result[0].sound == "sine"
    assert result[0].match == "approximate"
    # Both numbers travel in the evidence: the one the verdict was made on and
    # the one a v4 output would have shown, so the fix is auditable.
    assert result[0].evidence["centroid_energy_hz"] == F4_BASS_CENTROID_ENERGY_HZ
    assert result[0].evidence["centroid_mean_hz"] == F4_BASS_CENTROID_MEAN_HZ


def test_contaminated_centroid_mean_alone_cannot_produce_a_verdict() -> None:
    """A v4-shaped analysis has no median, and must not fall back to the mean.

    This is the guard on the fix rather than the fix itself: reading
    `centroid_mean` when `centroid_energy_hz` is absent would reintroduce exactly
    the descriptor F4 blames, and on this stem it would say "not a sub bass".
    Declining, with the reason saying why, is the honest answer.
    """
    spectral = SpectralFeatures(
        centroid_mean=F4_BASS_CENTROID_MEAN_HZ,
        brightness=F4_BASS_BRIGHTNESS,
        band_energy_ratios=BandEnergyRatios(low=F4_BASS_LOW_RATIO),
    )
    result = suggest_bass_sound(BassLine(status="ok"), spectral)
    assert result[0].match == "none"
    assert result[0].sound is None
    assert "centroid_energy_hz" in result[0].reason
    assert result[0].evidence["centroid_mean_hz"] == F4_BASS_CENTROID_MEAN_HZ
    assert "centroid_energy_hz" not in result[0].evidence


def test_sub_bass_threshold_is_not_quietly_loosened() -> None:
    """Pins which clause decides, not just the numbers.

    F4's trap was that the threshold looked wrong and was right, so widening
    one to make a stubborn stem fit is the mistake this file exists to catch.
    The centroid ceiling is now the top of the `low` band and is a sanity
    bound only — lowering it back toward the corpus (highest real sub bass
    161.4 Hz) would restore the two false negatives it used to produce.
    """
    assert SUB_BASS_CENTROID_HZ_MAX == 250.0
    assert SUB_BASS_BRIGHTNESS_MAX == 0.005

    # Brightness is the discriminator. A synthetic 35 Hz sawtooth reads
    # brightness 0.0131 with a centroid of 147.4 Hz: it clears both other
    # clauses and must be rejected here or not at all.
    low_saw = _spectral(low=0.915, brightness=0.0131, centroid=147.4)
    assert suggest_bass_sound(BassLine(status="ok"), low_saw)[0].match == "none"

    # The ceiling only catches what brightness is blind to: energy between
    # 250 Hz and the 1500 Hz brightness split.
    midrange = _spectral(low=0.75, brightness=0.0, centroid=287.0)
    assert suggest_bass_sound(BassLine(status="ok"), midrange)[0].match == "none"


#: Every audible bass stem in the eight-track v5 corpus, read from the
#: committed `calibration/v5/*/analysis/bass.json`. The stems themselves are
#: gitignored; these are the numbers, which is all the verdict reads.
#: All six are sub basses, spanning a 2.3:1 centroid range.
CORPUS_SUB_BASSES = {
    "eno": (68.662122, 0.000005, 0.999753),
    "roni": (79.013851, 0.000111, 0.988909),
    "levee": (120.358758, 0.000421, 0.977866),
    "madonna": (138.759392, 0.002052, 0.917462),
    "badu": (159.852626, 0.000007, 0.923566),
    "chameleon": (161.391024, 0.000539, 0.864159),
}


def test_every_audible_corpus_bass_resolves_to_sine() -> None:
    """The regression the corpus caught: 150 Hz rejected two of these six.

    Badu (brightness 0.000007) and Chameleon (0.000539) are purer sines than
    Madonna (0.002052) by the discriminator that actually separates the
    classes, and both were rejected for missing a ten-hertz margin on a ceiling
    derived from Madonna alone.
    """
    for name, (centroid, brightness, low) in CORPUS_SUB_BASSES.items():
        result = suggest_bass_sound(
            BassLine(status="ok"), _spectral(low=low, brightness=brightness, centroid=centroid)
        )
        assert result[0].sound == "sine", f"{name}: {result[0].reason}"
        assert result[0].match == "approximate", name


# --- F5: a silent stem gets no waveform ---------------------------------
#
# Both silent stems in the corpus, from the same committed JSON. `rms_mean`
# 8.2e-05 and 6.4e-05 against a `SILENCE_RMS_FLOOR` of 1e-3, both metered at
# -70.0 LUFS. `analyze.py` already sets `bass_line.status="silent"` for them.
SILENT_SHOWERS = (1899.043088, 0.365997, 0.355655)
SILENT_ANCIENT = (736.203341, 0.074505, 0.532178)


def test_silent_bass_stem_gets_no_sound_even_when_it_clears_every_clause() -> None:
    """`showers-of-gold`: empty stem, and it passed the harmonic branch.

    Low ratio 0.356 <= 0.55, brightness 0.366 >= 0.12, centroid 1899 >= 180 —
    all three, so the tool recommended a sawtooth bass for a stem containing
    nothing. That is the live half of F5, and the descriptors behind it are not
    even reproducible: residue stems move by up to 2x between separator runs.
    """
    centroid, brightness, low = SILENT_SHOWERS
    spectral = _spectral(low=low, brightness=brightness, centroid=centroid)

    # Ungated, this really does read as a harmonically rich bass.
    assert suggest_bass_sound(BassLine(status="ok"), spectral)[0].sound == "sawtooth"

    result = suggest_bass_sound(BassLine(status="silent"), spectral)
    assert len(result) == 1
    assert result[0].match == "none"
    assert result[0].sound is None
    assert "silence floor" in result[0].reason
    # The measurements still travel, so a reader can see what was discarded.
    assert result[0].evidence["brightness"] == brightness


def test_silence_gate_covers_the_sub_bass_branch_too() -> None:
    """Not only the branch that happened to leak.

    `ancient-heavy-tech-donjon` returns `none` on its own descriptors, so it is
    no evidence the gate works. A silent stem whose residue happened to look
    like a sub must be gated by status, not by luck.
    """
    centroid, brightness, low = SILENT_ANCIENT
    ancient = _spectral(low=low, brightness=brightness, centroid=centroid)
    assert suggest_bass_sound(BassLine(status="silent"), ancient)[0].sound is None
    looks_like_a_sub = _spectral(low=0.99, brightness=0.0, centroid=60.0)
    assert suggest_bass_sound(BassLine(status="ok"), looks_like_a_sub)[0].sound == "sine"
    assert suggest_bass_sound(BassLine(status="silent"), looks_like_a_sub)[0].sound is None


def test_unvoiced_is_a_caveat_and_silent_is_a_stop() -> None:
    """The distinction the gate turns on.

    `unvoiced` means there is a signal whose pitch could not be tracked, and
    its spectral shape is still a real measurement. `silent` means there is no
    signal. Only the second one blocks a verdict.
    """
    spectral = _spectral(low=0.9, brightness=0.002, centroid=50.0)
    assert suggest_bass_sound(BassLine(status="unvoiced"), spectral)[0].sound == "sine"
    assert suggest_bass_sound(BassLine(status="silent"), spectral)[0].sound is None


def test_sub_bass_spectral_shape_maps_to_sine() -> None:
    spectral = _spectral(low=0.9, brightness=0.002, centroid=50.0)
    result = suggest_bass_sound(BassLine(status="ok"), spectral)
    assert len(result) == 1
    assert result[0].role == "bass"
    assert result[0].sound == "sine"
    assert result[0].match == "approximate"


def test_harmonically_rich_bass_maps_to_sawtooth_with_square_alternative() -> None:
    spectral = _spectral(low=0.2, brightness=0.3, centroid=400.0)
    result = suggest_bass_sound(BassLine(status="ok"), spectral)
    assert result[0].sound == "sawtooth"
    assert result[0].match == "approximate"
    assert "square" in result[0].alternatives


def test_ambiguous_bass_spectral_shape_maps_to_none_with_full_palette() -> None:
    spectral = _spectral(low=0.5, brightness=0.08, centroid=150.0)
    result = suggest_bass_sound(BassLine(status="ok"), spectral)
    assert result[0].match == "none"
    assert result[0].sound is None
    assert set(result[0].alternatives) == set(BASS_WAVEFORM_PALETTE)


def test_bass_line_with_no_spectral_data_does_not_crash() -> None:
    result = suggest_bass_sound(BassLine(status="not_attempted"), SpectralFeatures())
    assert len(result) == 1
    assert result[0].match == "none"
    assert result[0].sound is None
    assert result[0].evidence == {}


def test_unvoiced_bass_line_still_classifies_from_spectral_shape() -> None:
    # status != "ok" must not crash and should still use the spectral
    # evidence, since sub-vs-harmonic is a spectral-shape question
    # independent of whether note segmentation succeeded.
    spectral = _spectral(low=0.9, brightness=0.002, centroid=50.0)
    result = suggest_bass_sound(BassLine(status="unvoiced"), spectral)
    assert result[0].sound == "sine"
    assert "unvoiced" in result[0].reason


# --- the sound-is-None-iff-match-is-none invariant --------------------


def _battery() -> list[StrudelSoundSuggestion]:
    """A wide sample of suggestions across every branch this module has."""
    drum_cases = [
        DrumDecomposition(),
        DrumDecomposition(status="ok", hits=[_hit("kick", kick_ratio=0.5)]),
        DrumDecomposition(status="ok", hits=[_hit("snare", body_ratio=0.6)]),
        DrumDecomposition(status="ok", hits=[_hit("snare", body_ratio=0.01)]),
        DrumDecomposition(status="ok", hits=[_hit("hat", decay_ratio=10.0)]),
        DrumDecomposition(status="ok", hits=[_hit("hat", decay_ratio=0.5)]),
        DrumDecomposition(status="ok", hits=[_hit("hat", decay_ratio=None)]),
        DrumDecomposition(status="ok", hits=[_hit("unclassified")]),
        DrumDecomposition(status="no_grid", hits=[], unclassified_count=3),
        DrumDecomposition(
            status="ok",
            hits=[
                _hit("kick", kick_ratio=0.8),
                _hit("snare", body_ratio=0.6),
                _hit("hat", decay_ratio=12.0),
                _hit("hat", decay_ratio=0.4),
                _hit("unclassified"),
            ],
        ),
    ]
    bass_spectral_cases = [
        (BassLine(), SpectralFeatures()),
        (BassLine(status="ok"), _spectral(low=0.9, brightness=0.002, centroid=50.0)),
        (BassLine(status="ok"), _spectral(low=0.2, brightness=0.3, centroid=400.0)),
        (BassLine(status="ok"), _spectral(low=0.5, brightness=0.08, centroid=150.0)),
        (BassLine(status="unvoiced"), _spectral(low=0.9, brightness=0.002, centroid=50.0)),
        (BassLine(status="too_few_hits"), SpectralFeatures()),
        (BassLine(status="silent"), _spectral(low=0.36, brightness=0.37, centroid=1899.0)),
        (BassLine(status="silent"), SpectralFeatures()),
        (BassLine(status="failed"), _spectral(low=None, brightness=None, centroid=None)),
        # v4-shaped: centroid_mean present, centroid_energy_hz absent.
        (
            BassLine(status="ok"),
            SpectralFeatures(
                centroid_mean=1010.7,
                brightness=0.002,
                band_energy_ratios=BandEnergyRatios(low=0.92),
            ),
        ),
    ]

    out: list[StrudelSoundSuggestion] = []
    for decomp in drum_cases:
        out.extend(suggest_drum_sounds(decomp))
    for bass_line, spectral in bass_spectral_cases:
        out.extend(suggest_bass_sound(bass_line, spectral))
    return out


def test_sound_is_none_iff_match_is_none() -> None:
    battery = _battery()
    assert len(battery) > 10  # sanity: the battery actually exercises branches
    for suggestion in battery:
        if suggestion.match == "none":
            assert suggestion.sound is None, suggestion
        else:
            assert suggestion.sound is not None, suggestion


def test_closed_vocabulary_of_role_and_match() -> None:
    for suggestion in _battery():
        assert suggestion.role in ROLES, suggestion.role
        assert suggestion.match in SOUND_MATCH_TERMS, suggestion.match


def test_every_emitted_sound_is_in_the_transcribed_vocabulary() -> None:
    """Meta-test: stops a typo'd sound name shipping.

    Every `sound` this module can emit, plus every string in `alternatives`,
    must be a real, transcribed Strudel sound name.
    """
    for suggestion in _battery():
        if suggestion.sound is not None:
            assert suggestion.sound in ALL_STRUDEL_SOUND_NAMES, suggestion.sound
        for alt in suggestion.alternatives:
            assert alt in ALL_STRUDEL_SOUND_NAMES, alt


def test_evidence_values_are_all_floats() -> None:
    for suggestion in _battery():
        for key, value in suggestion.evidence.items():
            assert type(value) is float, f"{key}={value!r} is {type(value)}"


def test_reason_is_always_populated() -> None:
    for suggestion in _battery():
        assert suggestion.reason
        assert isinstance(suggestion.reason, str)


# --- combinator ----------------------------------------------------------


def test_suggest_sounds_combines_drum_and_bass() -> None:
    decomp = DrumDecomposition(status="ok", hits=[_hit("kick", kick_ratio=0.7)])
    bass_line = BassLine(status="ok")
    spectral = _spectral(low=0.9, brightness=0.002, centroid=50.0)
    combined = suggest_sounds(decomp, bass_line, spectral)
    roles = {s.role for s in combined}
    assert roles == {"kick", "bass"}


# --- sanity on the transcribed tables themselves ------------------------


def test_transcribed_drum_samples_include_the_ones_this_module_uses() -> None:
    for name in ("bd", "sd", "cp", "hh", "oh"):
        assert name in STRUDEL_DRUM_SAMPLES


def test_transcribed_waveforms_include_the_ones_this_module_uses() -> None:
    for name in ("sine", "sawtooth", "square"):
        assert name in STRUDEL_WAVEFORMS
