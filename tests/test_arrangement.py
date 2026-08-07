"""Tests for `arrangement.py` — which stems are playing, bar by bar.

Two kinds of material, and the split carries the argument.

**Synthetic stems**, built from numpy, pin the mechanics: a stem that is
switched off for known bars must read absent for exactly those bars, RMS must
be RMS and not a mean of RMS values, and every entry point must refuse rather
than raise when it is handed something impossible.

**All five committed per-bar RMS fixtures** pin the behaviour that actually
matters. This module's single load-bearing threshold decides "is this stem
playing", and that is exactly the shape of threshold this project spent an
afternoon discovering had been calibrated against one house record. So the
threshold is asserted against house, hip-hop, drum and bass, a live band and
seventeen minutes of ambient, and the ambient row is the important one: it
asserts what the module **does not** find.

The tests are deliberately asymmetric about what they pin. On the calibration
track there is a hand-measured ground truth (F7) and the assertions are
specific — 147 bars, a 15-bar breakdown at bar 76 with the kick out in every
bar of it. On the other four there is no committed section list, so nothing
here asserts what their sections *are*. What those four constrain is how the
threshold **behaves**: it must not manufacture drums for a record with none, it
must isolate a solo-drum intro, and it must not need moving from track to
track. A test that asserted Roni Size has 44 sections would fail the next time
anyone touched the segmenter for a good reason, and would be pinning a number
nobody has verified by ear.

No audio is committed — `tests/fixtures/real/PROVENANCE.md` records what the
arrays are and why they are an irreversible reduction of the stems.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from audio_pipeline import ANALYSIS_SAMPLE_RATE, STEM_NAMES
from audio_pipeline import arrangement as arr
from audio_pipeline.arrangement import (
    KICK_TRACK,
    MIN_SECTION_BARS,
    PRESENCE_FRACTION,
    STEM_ACTIVITY_FLOOR,
    THRESHOLDS,
    TRACK_NAMES,
    Arrangement,
    arrangement_from_frames,
    frame_rms,
    label_sections,
    per_bar_energy,
    per_bar_energy_from_frames,
    presence,
    segment,
)
from audio_pipeline.drum_elements import DETECTION_BANDS, STFT_HOP_LENGTH

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "real"

#: Verified tempo and downbeat per corpus track, from `PROVENANCE.md` and
#: `calibration/v5-progress.md`. Eno carries no BPM: `refine_bpm` declines on
#: it, which is the point of that row.
CORPUS: dict[str, tuple[float | None, float]] = {
    "madonna": (132.000, 0.2322),
    "badu": (135.264, 0.0),
    "roni": (170.07, 0.0),
    "levee": (143.5, 0.0),
    "eno": (None, 0.0),
}

#: A nominal grid used **only** to force the ambient track through a fold it
#: would never be given in production, so the absence gate and the label guards
#: can be asserted on it. Nothing about 120 BPM is claimed to be Eno's tempo.
ENO_FORCED_BPM = 120.0

# --- Ground truth for the calibration track, from F7 as corrected by measurement
MADONNA_BARS = 147
MADONNA_BREAKDOWN_START = 76
MADONNA_BREAKDOWN_BARS = 15


# --------------------------------------------------------------------------- #
# Fixture loading
# --------------------------------------------------------------------------- #


def _load(track: str) -> tuple[dict[str, npt.NDArray[np.float64]], float, npt.NDArray[np.float64]]:
    """`(stem levels, hop_seconds, kick band energy)` for one corpus track."""
    data = np.load(FIXTURE_DIR / f"{track}__stem_frame_rms.npz")
    levels = {key[4:]: data[key] for key in data.files if key.startswith("rms_")}
    return levels, float(data["hop_seconds"]), data["kick_band_energy"]


def _energy(track: str, *, bpm: float | None = None, offset: float | None = None):
    """Fold one corpus track onto its verified grid."""
    verified_bpm, verified_offset = CORPUS[track]
    use_bpm = bpm if bpm is not None else verified_bpm
    levels, hop, kick = _load(track)
    period = 60.0 / use_bpm if use_bpm else None
    return per_bar_energy_from_frames(
        levels,
        hop,
        period,
        verified_offset if offset is None else offset,
        kick_band_energy=kick,
    )


def _sections(track: str, **kwargs) -> Arrangement:
    verified_bpm, verified_offset = CORPUS[track]
    bpm = kwargs.pop("bpm", verified_bpm)
    levels, hop, kick = _load(track)
    return arrangement_from_frames(
        levels,
        hop,
        60.0 / bpm if bpm else None,
        verified_offset,
        kick_band_energy=kick,
        **kwargs,
    )


def _longest_absent_run(result, *names: str) -> tuple[int, int]:
    """`(start_bar, length)` of the longest run with every named track absent."""
    by_name = result.by_name()
    absent = np.ones(result.bar_count, dtype=bool)
    for name in names:
        absent &= ~np.asarray(by_name[name].present, dtype=bool)
    best_start, best = 0, 0
    run_start, run = 0, 0
    for index, flag in enumerate(absent):
        if flag:
            if run == 0:
                run_start = index
            run += 1
            if run > best:
                best, best_start = run, run_start
        else:
            run = 0
    return best_start, best


# --------------------------------------------------------------------------- #
# Synthetic material
# --------------------------------------------------------------------------- #


def _switching_stem(bar_seconds: float, playing: list[bool], level: float = 0.2):
    """Mono noise at `level` in the bars flagged True, digital silence elsewhere.

    Digital silence, not a low level, because that is what a separated stem
    actually holds when its instrument is not playing, and it is what makes the
    thresholded quantity bimodal rather than continuous.
    """
    rng = np.random.default_rng(7)
    per_bar = int(round(bar_seconds * ANALYSIS_SAMPLE_RATE))
    out = np.zeros(per_bar * len(playing), dtype=np.float64)
    for index, active in enumerate(playing):
        if active:
            out[index * per_bar : (index + 1) * per_bar] = rng.normal(0.0, level, per_bar)
    return out


def _synthetic_stems(bar_seconds: float):
    """Four stems with a known arrangement: 4 bars drums, 8 full, 4 no-drums."""
    return {
        "drums": _switching_stem(bar_seconds, [True] * 12 + [False] * 4),
        "bass": _switching_stem(bar_seconds, [False] * 4 + [True] * 12),
        "vocals": _switching_stem(bar_seconds, [False] * 4 + [True] * 8 + [False] * 4),
        "other": _switching_stem(bar_seconds, [True] * 16),
    }


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


class TestFrameRms:
    def test_matches_the_fixture_generator_definition(self) -> None:
        """Non-overlapping 512-sample blocks, the same reshape the fixtures use."""
        samples = np.arange(STFT_HOP_LENGTH * 3, dtype=np.float64)
        measured = frame_rms(samples)
        assert measured.size == 3
        for index in range(3):
            block = samples[index * STFT_HOP_LENGTH : (index + 1) * STFT_HOP_LENGTH]
            assert measured[index] == pytest.approx(math.sqrt(float((block**2).mean())))

    def test_constant_signal_reads_its_own_amplitude(self) -> None:
        assert frame_rms(np.full(STFT_HOP_LENGTH * 4, 0.5))[0] == pytest.approx(0.5)

    def test_trailing_partial_block_is_dropped_not_averaged(self) -> None:
        """A short final block would otherwise report a level it did not have."""
        assert frame_rms(np.ones(STFT_HOP_LENGTH + 5)).size == 1

    def test_shorter_than_one_block_is_empty_not_an_error(self) -> None:
        assert frame_rms(np.ones(10)).size == 0

    def test_non_finite_samples_are_not_energy(self) -> None:
        samples = np.zeros(STFT_HOP_LENGTH)
        samples[0] = np.nan
        samples[1] = np.inf
        assert frame_rms(samples)[0] == pytest.approx(0.0)


class TestBarFolding:
    def test_a_bar_is_rms_not_the_mean_of_frame_rms(self) -> None:
        """The distinction under-reads any bar holding both a hit and a gap.

        One bar, half of it at 1.0 and half digital silence. The RMS is
        `sqrt(0.5)` = 0.707; a mean of the frames' RMS values is 0.5. Getting
        this wrong makes every sparse stem look quieter than it is, which is
        exactly the population a presence threshold is deciding about.
        """
        frames = np.concatenate([np.ones(50), np.zeros(50)])
        result = per_bar_energy_from_frames({"drums": frames}, 0.01, 0.25)
        assert result.bar_count == 1
        assert result.levels["drums"][0] == pytest.approx(math.sqrt(0.5), abs=1e-6)

    def test_kick_band_energy_is_rooted_onto_the_stems_scale(self) -> None:
        """It arrives as power; a shared fractional threshold needs amplitude."""
        energy = per_bar_energy_from_frames(
            {"drums": np.ones(100)}, 0.01, 0.25, kick_band_energy=np.full(100, 9.0)
        )
        assert energy.levels[KICK_TRACK][0] == pytest.approx(3.0)

    @pytest.mark.parametrize("offset", [0.0, 0.2322, 0.5, 0.9])
    def test_bar_count_does_not_move_with_the_downbeat(self, offset: float) -> None:
        """The downbeat is a measurement with error; the bar count is a fact.

        Measured on the calibration track: 267.5 s at 132.000 BPM is 147.12
        bars from zero and 146.99 from the verified 0.2322 s downbeat, and
        `MIN_FINAL_BAR_FRACTION` makes both — and everything in between —
        report the verified 147.
        """
        assert _energy("madonna", offset=offset).bar_count == MADONNA_BARS

    def test_a_downbeat_past_half_a_bar_legitimately_loses_one(self) -> None:
        """Stated so the rule above is understood as bounded, not universal."""
        assert _energy("madonna", offset=1.5).bar_count == MADONNA_BARS - 1


# --------------------------------------------------------------------------- #
# Refusals — nothing in this module raises
# --------------------------------------------------------------------------- #


class TestRefusals:
    @pytest.mark.parametrize("period", [None, 0.0, -1.0, float("nan"), float("inf")])
    def test_an_unusable_period_is_no_grid(self, period: float | None) -> None:
        result = per_bar_energy_from_frames({"drums": np.ones(1000)}, 0.01, period)
        assert result.status == "no_grid"
        assert result.bar_count == 0
        assert result.caveats

    def test_no_stems_is_no_grid(self) -> None:
        assert per_bar_energy_from_frames({}, 0.01, 0.5).status == "no_grid"

    def test_a_source_shorter_than_a_bar_says_so(self) -> None:
        result = per_bar_energy_from_frames({"drums": np.ones(10)}, 0.01, 1.0)
        assert result.status == "too_short"
        assert "less than one" in result.caveats[-1]

    def test_a_missing_downbeat_is_a_caveat_not_a_failure(self) -> None:
        """Folding from the start of the file is a real answer, just rotated."""
        result = per_bar_energy_from_frames({"drums": np.ones(1000)}, 0.01, 0.25, None)
        assert result.status == "ok"
        assert any("no downbeat" in caveat for caveat in result.caveats)

    def test_presence_of_a_refused_fold_is_unavailable(self) -> None:
        found = presence(per_bar_energy_from_frames({}, 0.01, None))
        assert found.status == "unavailable"
        assert segment(found) == ()

    def test_label_sections_of_nothing_is_nothing(self) -> None:
        assert label_sections(()) == ()

    def test_non_finite_levels_do_not_crash_the_fold(self) -> None:
        frames = np.full(1000, np.nan)
        frames[::3] = np.inf
        result = per_bar_energy_from_frames({"drums": frames}, 0.01, 0.25)
        assert result.status == "ok"
        assert all(math.isfinite(value) for value in result.levels["drums"])

    def test_a_stem_that_is_pure_silence_is_reported_absent_not_divided_by(self) -> None:
        energy = per_bar_energy_from_frames(
            {"drums": np.ones(2000), "bass": np.zeros(2000)}, 0.01, 0.25
        )
        found = presence(energy).by_name()
        assert found["bass"].status == "absent"
        assert not any(found["bass"].present)

    def test_the_arrangement_wrapper_never_raises_on_a_missing_grid(self) -> None:
        result = arrangement_from_frames({"drums": np.ones(100)}, 0.01, None)
        assert result.status == "no_grid"
        assert result.sections == ()


# --------------------------------------------------------------------------- #
# Presence on synthetic material with a known arrangement
# --------------------------------------------------------------------------- #


class TestPresenceSynthetic:
    def test_a_known_arrangement_is_recovered_bar_for_bar(self) -> None:
        bar_seconds = 2.0
        energy = per_bar_energy(_synthetic_stems(bar_seconds), ANALYSIS_SAMPLE_RATE, 0.5, 0.0)
        assert energy.bar_count == 16
        found = presence(energy).by_name()
        assert list(found["drums"].present) == [True] * 12 + [False] * 4
        assert list(found["bass"].present) == [False] * 4 + [True] * 12
        assert list(found["vocals"].present) == [False] * 4 + [True] * 8 + [False] * 4
        assert all(found["other"].present)

    def test_every_stem_carries_the_threshold_that_decided_it(self) -> None:
        """Auditability: a label whose evidence is missing is decoration."""
        energy = per_bar_energy(_synthetic_stems(2.0), ANALYSIS_SAMPLE_RATE, 0.5, 0.0)
        for track in presence(energy).tracks:
            assert track.threshold == pytest.approx(track.reference * PRESENCE_FRACTION)
            assert track.active_bars == sum(track.present)

    def test_a_quiet_mix_is_not_an_absent_mix(self) -> None:
        """The gate is a within-record comparison, so overall level cancels."""
        loud = _synthetic_stems(2.0)
        quiet = {name: samples * 1e-4 for name, samples in loud.items()}
        loud_found = presence(per_bar_energy(loud, ANALYSIS_SAMPLE_RATE, 0.5, 0.0))
        quiet_found = presence(per_bar_energy(quiet, ANALYSIS_SAMPLE_RATE, 0.5, 0.0))
        assert quiet_found.absent_tracks == ()
        assert [t.present for t in quiet_found.tracks] == [t.present for t in loud_found.tracks]

    def test_a_stem_of_pure_bleed_is_gated_out_of_the_record(self) -> None:
        """The whole point of `STEM_ACTIVITY_FLOOR`, on a signal built to fail."""
        stems = _synthetic_stems(2.0)
        stems["vocals"] = _switching_stem(2.0, [True] * 16, level=1e-5)
        found = presence(per_bar_energy(stems, ANALYSIS_SAMPLE_RATE, 0.5, 0.0))
        vocals = found.by_name()["vocals"]
        assert vocals.status == "absent"
        assert vocals.active_bars == 0
        assert "vocals" in found.absent_tracks
        assert any("not present in this record" in caveat for caveat in found.caveats)

    def test_the_kick_inherits_the_drums_stems_verdict(self) -> None:
        """Band energy and sample RMS are not on a common scale, so it must."""
        stems = _synthetic_stems(2.0)
        stems["drums"] = _switching_stem(2.0, [True] * 16, level=1e-5)
        found = presence(per_bar_energy(stems, ANALYSIS_SAMPLE_RATE, 0.5, 0.0)).by_name()
        assert found["drums"].status == "absent"
        assert found[KICK_TRACK].status == "absent"
        assert found[KICK_TRACK].relative_level is None

    def test_the_gate_sits_between_the_two_measured_populations(self) -> None:
        """0.2696 is the quietest genuinely-present stem in the corpus and
        0.00138 the loudest piece of separation residue; the floor is the
        geometric midpoint of that 195x gap."""
        assert STEM_ACTIVITY_FLOOR < 0.2696
        assert STEM_ACTIVITY_FLOOR > 0.00138
        assert STEM_ACTIVITY_FLOOR == pytest.approx(math.sqrt(0.2696 * 0.00138), rel=0.15)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


class TestSegment:
    def _presence_of(self, patterns: list[list[bool]]):
        """Build a `Presence` directly from per-bar flags for two stems."""
        energy = per_bar_energy_from_frames(
            {
                "drums": _rms_from_flags(patterns[0]),
                "bass": _rms_from_flags(patterns[1]),
            },
            0.01,
            0.25,
        )
        return presence(energy)

    def test_runs_of_one_pattern_become_one_section(self) -> None:
        found = self._presence_of([[True] * 8 + [False] * 8, [True] * 16])
        sections = segment(found, bar_seconds=2.0)
        assert [(s.start_bar, s.length_bars) for s in sections] == [(0, 8), (8, 8)]
        assert sections[0].active == ("drums", "bass")
        assert sections[1].active == ("bass",)

    def test_start_seconds_follows_the_grid(self) -> None:
        found = self._presence_of([[True] * 8 + [False] * 8, [True] * 16])
        sections = segment(found, bar_seconds=2.0, downbeat_seconds=0.5)
        assert [s.start_seconds for s in sections] == [0.5, 16.5]

    def test_a_one_bar_flicker_is_merged_away(self) -> None:
        """A single bar of a stem dropping out is a fill, not a section."""
        flags = [True] * 16
        flags[7] = False
        found = self._presence_of([flags, [True] * 16])
        assert len(segment(found, bar_seconds=2.0)) == 1

    def test_the_two_halves_a_flicker_split_are_rejoined(self) -> None:
        """Without `_coalesce` this returns two adjacent identical sections.

        The regression this pins is real and was measured: the calibration
        track's 49-bar full-band stretch came back as five consecutive
        sections carrying the same active set.
        """
        flags = [True] * 16
        flags[7] = False
        found = self._presence_of([flags, [True] * 16])
        sections = segment(found, bar_seconds=2.0)
        assert len(sections) == 1
        assert sections[0].length_bars == 16

    def test_a_short_run_joins_the_neighbour_it_resembles(self) -> None:
        """Content before length: a one-bar bass rest belongs with the bass."""
        drums = [True] * 16
        bass = [False] * 4 + [True] * 12
        bass[10] = False
        found = self._presence_of([drums, bass])
        sections = segment(found, bar_seconds=2.0)
        assert [(s.start_bar, s.length_bars, s.active) for s in sections] == [
            (0, 4, ("drums",)),
            (4, 12, ("drums", "bass")),
        ]

    def test_nothing_shorter_than_the_minimum_survives(self) -> None:
        rng = np.random.default_rng(3)
        flags = [bool(value) for value in rng.integers(0, 2, 64)]
        found = self._presence_of([flags, [True] * 64])
        sections = segment(found, bar_seconds=1.0)
        assert all(s.length_bars >= MIN_SECTION_BARS for s in sections)

    def test_sections_tile_the_track_exactly(self) -> None:
        rng = np.random.default_rng(11)
        flags = [bool(value) for value in rng.integers(0, 2, 64)]
        found = self._presence_of([flags, [True] * 64])
        sections = segment(found, bar_seconds=1.0)
        assert sections[0].start_bar == 0
        assert sum(s.length_bars for s in sections) == found.bar_count
        for previous, following in zip(sections, sections[1:], strict=False):
            assert following.start_bar == previous.start_bar + previous.length_bars


#: Frames per bar in the synthetic segmentation fixtures: a 0.25 s beat and a
#: 0.01 s hop put 100 frames in a four-beat bar, so one flag is exactly one bar.
FRAMES_PER_BAR = 100


def _rms_from_flags(flags: list[bool], frames_per_bar: int = FRAMES_PER_BAR):
    """Per-frame RMS that is 0.2 in flagged bars and digital silence elsewhere."""
    return np.repeat(np.where(np.asarray(flags), 0.2, 0.0), frames_per_bar)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


class TestLabels:
    def _label(self, patterns: list[tuple[str, ...]]) -> list[str | None]:
        sections = tuple(
            arr.Section(start_bar=index * 4, length_bars=4, start_seconds=0.0, active=active)
            for index, active in enumerate(patterns)
        )
        return [section.label for section in label_sections(sections)]

    def test_the_canonical_shape(self) -> None:
        band = ("drums", "bass", "vocals", "other", "kick")
        labels = self._label(
            [
                ("drums", "other", "kick"),  # intro
                band,  # full
                ("vocals", "other"),  # breakdown
                band,  # drop
                ("drums", "other"),  # outro
                (),  # silence
            ]
        )
        assert labels == ["intro", "full", "breakdown", "drop", "outro", "silence"]

    def test_every_label_carries_the_pattern_that_produced_it(self) -> None:
        sections = label_sections(
            (
                arr.Section(0, 4, 0.0, ("drums", "kick")),
                arr.Section(4, 4, 0.0, ("drums", "bass", "kick")),
            )
        )
        for section in sections:
            assert section.label_reason
            for name in section.active:
                assert name in section.label_reason

    def test_silence_beats_outro(self) -> None:
        """"The record has stopped" and "the record is thinning" differ."""
        assert self._label([("drums", "bass"), ("drums",), ()]) == ["full", "outro", "silence"]

    def test_the_outro_is_a_trailing_run_not_only_the_last_section(self) -> None:
        """Measured need: the calibration track thins across three sections."""
        band = ("drums", "bass", "kick")
        labels = self._label([band, band, ("drums", "kick"), ("drums",), ()])
        assert labels == ["full", "full", "outro", "outro", "silence"]

    def test_a_drop_needs_a_breakdown_in_front_of_it(self) -> None:
        band = ("drums", "bass", "vocals", "kick")
        assert self._label([band, ("vocals",), band]) == ["full", "breakdown", "drop"]
        assert self._label([band, ("drums", "vocals", "kick"), band]) == [
            "full",
            "groove",
            "full",
        ]

    def test_breakdown_needs_the_record_to_have_a_kick(self) -> None:
        """Otherwise the label is vacuous and fires on everything.

        This is the guard that stops seventeen minutes of ambient coming back
        as alternating breakdowns and drops once the drums stem has been
        correctly gated out.
        """
        labels = self._label([("bass", "other"), ("other",), ("bass", "other"), ("other",)])
        assert "breakdown" not in labels
        assert "drop" not in labels

    def test_in_record_ignores_tracks_that_were_gated_out(self) -> None:
        """`full` means every track in *this* record, not all five names."""
        assert self._label([("bass",), ("bass", "other"), ("other",)]) == [
            "intro",
            "full",
            "outro",
        ]

    def test_verse_and_chorus_are_not_attempted(self) -> None:
        """Out of scope by decision, not by omission — it needs repetition."""
        produced = {
            section.label
            for section in label_sections(
                tuple(
                    arr.Section(i * 4, 4, 0.0, ("drums", "bass", "kick") if i % 2 else ("drums",))
                    for i in range(8)
                )
            )
        }
        assert not produced & {"verse", "chorus"}


# --------------------------------------------------------------------------- #
# The calibration track — F7's arrangement, as corrected by measurement
# --------------------------------------------------------------------------- #


class TestMadonna:
    def test_the_track_is_147_bars(self) -> None:
        """`V2-PLAN.md` says 146; 267.5 s at 132.000 BPM is 147.13. Report
        what is measured — the plan is wrong and `PROVENANCE.md` says 147."""
        assert _energy("madonna").bar_count == MADONNA_BARS

    def test_the_breakdown_has_the_kick_and_the_bass_out_together(self) -> None:
        """F7's headline: a 16-bar breakdown around bars 75-90.

        Measured, it is **15 bars from 76**. The difference is bar 75, a
        transitional bar whose kick sits 3% over the threshold; see
        `THRESHOLDS["presence_fraction"]` for the sweep and for why that was
        not tuned away.
        """
        start, length = _longest_absent_run(presence(_energy("madonna")), "kick", "bass")
        assert start == MADONNA_BREAKDOWN_START
        assert length == MADONNA_BREAKDOWN_BARS

    def test_inside_the_breakdown_the_drums_stem_lies_and_the_kick_does_not(self) -> None:
        """The measurement that earns the kick track its place in the result.

        The drums stem clears its own presence threshold in 3 of the 15
        breakdown bars — percussion and a reverb tail — while the kick band
        clears its threshold in none of them.
        """
        by_name = presence(_energy("madonna")).by_name()
        window = slice(MADONNA_BREAKDOWN_START, MADONNA_BREAKDOWN_START + MADONNA_BREAKDOWN_BARS)
        assert sum(by_name["drums"].present[window]) == 3
        assert sum(by_name[KICK_TRACK].present[window]) == 0

    def test_the_breakdown_bars_are_all_labelled_breakdown(self) -> None:
        result = _sections("madonna")
        for bar in range(
            MADONNA_BREAKDOWN_START, MADONNA_BREAKDOWN_START + MADONNA_BREAKDOWN_BARS
        ):
            section = next(
                s for s in result.sections if s.start_bar <= bar < s.start_bar + s.length_bars
            )
            assert section.label == "breakdown", f"bar {bar} is {section.label}"

    def test_the_drop_lands_on_the_bar_after_the_breakdown(self) -> None:
        result = _sections("madonna")
        drop = next(s for s in result.sections if s.label == "drop")
        assert drop.start_bar == MADONNA_BREAKDOWN_START + MADONNA_BREAKDOWN_BARS
        assert set(drop.active) == set(TRACK_NAMES)

    def test_the_intro_is_kick_and_pad_with_no_bass(self) -> None:
        """F7: "16-bar intro on kick and pad". Measured 17 bars, same content."""
        intro = _sections("madonna").sections[0]
        assert intro.label == "intro"
        assert intro.start_bar == 0
        assert set(intro.active) == {"drums", "other", KICK_TRACK}
        assert intro.length_bars == pytest.approx(16, abs=2)

    def test_the_track_ends_in_total_silence(self) -> None:
        """F7 names two bars of it; measured three, at the very end."""
        last = _sections("madonna").sections[-1]
        assert last.label == "silence"
        assert last.active == ()
        assert last.start_bar + last.length_bars == MADONNA_BARS

    def test_there_is_a_long_full_band_stretch(self) -> None:
        """F7: "48 bars of full band". Measured 49, in one section."""
        result = _sections("madonna")
        longest = max(result.sections, key=lambda s: s.length_bars)
        assert longest.label == "full"
        assert longest.length_bars == pytest.approx(48, abs=2)
        assert set(longest.active) == set(TRACK_NAMES)

    def test_no_two_adjacent_sections_share_an_active_set(self) -> None:
        """`_coalesce`'s job, asserted on real material rather than a toy."""
        sections = _sections("madonna").sections
        for previous, following in zip(sections, sections[1:], strict=False):
            assert previous.active != following.active


# --------------------------------------------------------------------------- #
# The corpus — what the threshold must do on four tracks it was not tuned on
# --------------------------------------------------------------------------- #


class TestCorpusBehaviour:
    @pytest.mark.parametrize("track", ["madonna", "badu", "roni", "levee"])
    def test_every_stem_of_every_percussive_track_is_in_its_record(self, track: str) -> None:
        """The gate must not fire on anything that is genuinely playing.

        Eighteen stems across four tracks, the quietest reading 0.2696 of its
        record's loudest stem against a floor of 0.02.
        """
        found = presence(_energy(track))
        assert found.absent_tracks == ()
        for name in STEM_NAMES:
            assert found.by_name()[name].relative_level > STEM_ACTIVITY_FLOOR

    @pytest.mark.parametrize("track", ["madonna", "badu", "roni", "levee"])
    def test_every_percussive_track_yields_sections_that_tile_it(self, track: str) -> None:
        result = _sections(track)
        assert result.status == "ok"
        assert result.sections
        assert sum(s.length_bars for s in result.sections) == result.bar_count
        assert all(s.label is not None and s.label_reason for s in result.sections)

    def test_levee_isolates_its_solo_drum_intro(self) -> None:
        """The bass and pad enter together, and the entry is not marginal:
        bar 3 reads 0.00005 on the bass stem and bar 4 reads 0.0698, a factor
        of 1400. A section boundary that survives any threshold in the sweep."""
        intro = _sections("levee").sections[0]
        assert intro.label == "intro"
        assert intro.start_bar == 0
        assert set(intro.active) == {"drums", KICK_TRACK}
        assert "bass" not in intro.active

    def test_levee_says_its_grid_is_approximate(self) -> None:
        """That track's tempo genuinely drifts — `refine_bpm` returns
        `status="coarse"` at confidence 0.000 — so nothing here may read as
        placing a section to the bar."""
        result = _sections("levee", grid_confidence="coarse")
        assert any("approximate" in caveat for caveat in result.caveats)
        clean = _sections("levee", grid_confidence="high")
        assert not any("approximate" in caveat for caveat in clean.caveats)

    def test_roni_finds_its_silent_tail(self) -> None:
        """The file runs 302.7 s and the music stops at 276.6 s."""
        last = _sections("roni").sections[-1]
        assert last.label == "silence"
        assert last.length_bars >= 8

    def test_badu_has_an_intro_and_an_outro_around_a_long_full_stretch(self) -> None:
        result = _sections("badu")
        labels = [s.label for s in result.sections]
        assert labels[0] == "intro"
        assert labels[-1] == "outro"
        assert max(s.length_bars for s in result.sections) > result.bar_count // 2


class TestAmbientRefusal:
    """The row whose job is to make the tool say no.

    1042 s of Brian Eno with no drums and no voice. `refine_bpm` declines a
    tempo on it, so in production this module is never handed a grid — but the
    interesting assertions are about what happens when it *is*, because that is
    where a scale-free threshold would invent a drum arrangement.
    """

    def test_with_no_period_there_is_no_arrangement(self) -> None:
        result = _sections("eno")
        assert result.status == "no_grid"
        assert result.sections == ()
        assert any("no beat period" in caveat for caveat in result.caveats)

    def test_a_forced_grid_gates_out_the_stems_that_are_not_in_the_record(self) -> None:
        """The drums stem peaks at -55 dBFS and the vocals stem at -71 dBFS on
        a record with neither. Both must read absent, not quiet."""
        found = presence(_energy("eno", bpm=ENO_FORCED_BPM))
        assert set(found.absent_tracks) == {"drums", "vocals", KICK_TRACK}
        for name in ("drums", "vocals", KICK_TRACK):
            assert found.by_name()[name].active_bars == 0

    def test_the_stems_that_are_in_the_record_are_kept(self) -> None:
        """The gate must not simply reject everything quiet."""
        found = presence(_energy("eno", bpm=ENO_FORCED_BPM)).by_name()
        assert found["bass"].status == "measured"
        assert found["other"].status == "measured"
        assert found["other"].active_bars > 0

    def test_no_drum_arrangement_is_manufactured(self) -> None:
        """Without the gate this track reports drums playing in most of its
        521 bars, which is the single failure this fixture exists to catch."""
        result = _sections("eno", bpm=ENO_FORCED_BPM)
        assert all("drums" not in s.active for s in result.sections)
        assert all(KICK_TRACK not in s.active for s in result.sections)

    def test_no_breakdown_or_drop_is_claimed(self) -> None:
        """Both labels are defined against a kick. On a record with none they
        would fire on every section, which is structure out of nothing."""
        labels = {s.label for s in _sections("eno", bpm=ENO_FORCED_BPM).sections}
        assert not labels & {"breakdown", "drop"}


# --------------------------------------------------------------------------- #
# The thresholds themselves
# --------------------------------------------------------------------------- #


class TestThresholdEvidence:
    """These pin the measurements the threshold comments cite.

    A threshold documented with a number nobody re-derives is a transcribed
    table: it can only fail when someone edits the table. These re-run the
    sweep on every test run.
    """

    @pytest.mark.parametrize("fraction", [0.05, 0.08, 0.10, 0.15, 0.25, 0.40, 0.50])
    def test_the_breakdown_survives_a_tenfold_range_of_the_threshold(
        self, fraction: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The plateau that makes 0.15 safe rather than lucky. The quantity is
        bimodal by three orders of magnitude — in the breakdown the bass reads
        0.0001 against its own 0.158 reference — so no threshold in this range
        can disagree about it."""
        monkeypatch.setattr(arr, "PRESENCE_FRACTION", fraction)
        start, length = _longest_absent_run(presence(_energy("madonna")), "kick", "bass")
        assert start in (75, 76)
        assert length >= 14

    def test_below_the_plateau_the_breakdown_fragments(self) -> None:
        """The one edge that is real: reverb tails start reading as presence."""
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(arr, "PRESENCE_FRACTION", 0.03)
            _, length = _longest_absent_run(presence(_energy("madonna")), "kick", "bass")
        assert length < 10

    def test_raising_the_minimum_section_to_four_bars_deletes_real_sections(self) -> None:
        """Why `MIN_SECTION_BARS` stays at 2 despite tidying up dense material.

        Four bars takes the drum-and-bass row from 44 sections to 14, which
        looks like an improvement, and takes the hip-hop row from 6 to 2 —
        losing a real 2-bar pad intro and a real 3-bar outro.
        """
        energy = _energy("badu")
        found = presence(energy)
        at_two = segment(found, energy.bar_seconds, min_section_bars=2)
        at_four = segment(found, energy.bar_seconds, min_section_bars=4)
        assert len(at_two) == 6
        assert len(at_four) == 2
        assert at_two[0].length_bars == 2
        assert at_two[-1].length_bars == 3
        assert label_sections(at_four)[0].length_bars > 4

    def test_the_ambient_section_count_only_tracks_the_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A structure would be stable under the threshold; a swell is not.

        The calibration track's breakdown does not move across the sweep
        above. The ambient track's section count rises monotonically with it,
        which is the signature of a line sliding up and down a continuous
        curve rather than of sections being found.
        """
        counts = []
        for fraction in (0.05, 0.15, 0.30, 0.50):
            monkeypatch.setattr(arr, "PRESENCE_FRACTION", fraction)
            counts.append(len(_sections("eno", bpm=ENO_FORCED_BPM).sections))
        assert counts == sorted(counts)
        assert counts[-1] > 2 * counts[0]

    def test_every_threshold_is_exported_and_documented(self) -> None:
        """Ground rule 10, asserted rather than trusted."""
        source = (Path(arr.__file__)).read_text(encoding="utf-8")
        for name in THRESHOLDS:
            assert f'"{name}"' in source
            assert "[grounded" in source or "[guess" in source
        assert set(THRESHOLDS) == {
            "presence_fraction",
            "presence_reference_percentile",
            "stem_activity_floor",
            "min_section_bars",
            "min_final_bar_fraction",
        }


# --------------------------------------------------------------------------- #
# Contracts this module must not break
# --------------------------------------------------------------------------- #


class TestContracts:
    def test_the_kick_band_is_the_drum_decompositions_own_band(self) -> None:
        """"The kick is playing" must mean the same thing in both modules."""
        assert arr.KICK_BAND_HZ == DETECTION_BANDS["kick"]

    def test_the_hop_is_the_fixtures_hop(self) -> None:
        for track in CORPUS:
            data = np.load(FIXTURE_DIR / f"{track}__stem_frame_rms.npz")
            assert float(data["hop_seconds"]) == pytest.approx(arr.HOP_SECONDS)

    def test_nothing_here_imports_a_backend(self) -> None:
        """Same rule as `tempo.py` and `drum_elements.py`: an arrangement must
        not depend on which analysis wheel happened to install."""
        source = Path(arr.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "essentia" not in stripped
                assert "librosa" not in stripped
                assert "scipy" not in stripped

    def test_a_foreign_sample_rate_is_flagged_not_silently_accepted(self) -> None:
        """Ground rule 2. Every threshold here was placed at 44.1 kHz."""
        result = per_bar_energy(_synthetic_stems(2.0), 22050, 0.5, 0.0)
        assert any("22050" in caveat for caveat in result.caveats)

    def test_missing_stems_are_a_caveat_not_a_crash(self) -> None:
        result = per_bar_energy_from_frames({"drums": np.ones(2000)}, 0.01, 0.25)
        assert result.status == "ok"
        assert any("bass" in caveat for caveat in result.caveats)

    def test_tracks_come_back_in_a_fixed_order(self) -> None:
        """So two runs are comparable and the kick is always last."""
        found = presence(_energy("madonna"))
        assert tuple(track.name for track in found.tracks) == TRACK_NAMES
