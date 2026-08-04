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
    low: float | None, brightness: float | None, centroid: float | None
) -> SpectralFeatures:
    return SpectralFeatures(
        centroid_mean=centroid,
        brightness=brightness,
        band_energy_ratios=BandEnergyRatios(low=low),
    )


def test_sub_bass_spectral_shape_maps_to_sine() -> None:
    spectral = _spectral(low=0.9, brightness=0.01, centroid=50.0)
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
    spectral = _spectral(low=0.9, brightness=0.01, centroid=50.0)
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
        (BassLine(status="ok"), _spectral(low=0.9, brightness=0.01, centroid=50.0)),
        (BassLine(status="ok"), _spectral(low=0.2, brightness=0.3, centroid=400.0)),
        (BassLine(status="ok"), _spectral(low=0.5, brightness=0.08, centroid=150.0)),
        (BassLine(status="unvoiced"), _spectral(low=0.9, brightness=0.01, centroid=50.0)),
        (BassLine(status="too_few_hits"), SpectralFeatures()),
        (BassLine(status="failed"), _spectral(low=None, brightness=None, centroid=None)),
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
    spectral = _spectral(low=0.9, brightness=0.01, centroid=50.0)
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
