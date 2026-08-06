"""Build the committed real-material fixtures in `tests/fixtures/real/`.

Dev-time only. Nothing in `src/audio_pipeline/` imports this, and it is never
run by the test suite — it is the recorded provenance of files that *are*
committed, so a fixture can be regenerated or extended to a new corpus track
without anyone reverse-engineering how the first one was made.

**Why derived arrays and not audio.** Ground rule 9: tests must not require a
real audio file, but real-material regressions must be captured. Every array
written here is a heavy irreversible reduction of the stem it came from — a
4-band energy summary at 86 frames/sec, or an F0 contour. Phase is gone,
per-sample detail is gone, and the reduction is roughly 128:1 before
compression. None of it can be turned back into audible audio, so committing it
does not commit source material.

**Why it reuses `drum_elements`' private helpers.** The fixtures and the code
under test must sit on the same STFT grid, or an assertion about frame `k` in a
test means something different from frame `k` in the pipeline. Re-implementing
the transform here would create exactly the silent drift the fixtures exist to
catch. `_stft_magnitude` and `_band_envelope` are the definitions; this script
calls them rather than copying them.

Usage:
    python tools/make-fixtures/make_real_fixtures.py <stems-dir> --track <name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from audio_pipeline import ANALYSIS_SAMPLE_RATE, STEM_NAMES
from audio_pipeline.audio_io import load_audio, to_mono
from audio_pipeline.drum_elements import (
    DETECTION_BANDS,
    STFT_HOP_LENGTH,
    _band_envelope,
    _stft_magnitude,
)

#: The band W4A refines tempo from, in Hz. Narrower than `DETECTION_BANDS`'
#: `kick` band (20-150 Hz) on purpose: 110 Hz is A2, so this bound keeps a bass
#: guitar or synth fundamental out of the envelope the tempo fit reads. The kick
#: fundamental sits at 40-60 Hz with its click above, so nothing load-bearing is
#: lost. Named separately rather than added to `DETECTION_BANDS` because it
#: overlaps `kick` and would break that dict's disjointness.
TEMPO_BAND_HZ: tuple[float, float] = (20.0, 110.0)

#: Seconds between frames. Pinned by `STFT_HOP_LENGTH` at
#: `ANALYSIS_SAMPLE_RATE`: 512 / 44100 = 11.61 ms. Written into every fixture so
#: a test never has to recompute it and never has to assume 44.1 kHz.
HOP_SECONDS: float = STFT_HOP_LENGTH / ANALYSIS_SAMPLE_RATE


def band_envelopes(mono: npt.NDArray[np.float32], sample_rate: int) -> dict[str, Any]:
    """Per-frame energy in each detection band plus the tempo band.

    Energy, not flux. Flux is a pure function of this (`_spectral_flux`), so
    committing the envelope lets each package apply the project's own flux
    definition rather than freezing one caller's choice into the fixture.
    """
    magnitude, freqs = _stft_magnitude(mono, sample_rate)
    ceiling = max(high for _, high in DETECTION_BANDS.values())
    out: dict[str, Any] = {
        f"band_{name}": _band_envelope(
            magnitude, freqs, low, high, include_high=high >= ceiling
        ).astype(np.float32)
        for name, (low, high) in DETECTION_BANDS.items()
    }
    out["band_tempo"] = _band_envelope(magnitude, freqs, *TEMPO_BAND_HZ).astype(np.float32)
    return out


def frame_rms(mono: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """RMS per non-overlapping `STFT_HOP_LENGTH` block.

    Deliberately *not* folded into bars. Folding needs a period and a downbeat,
    and baking the answer W4A is meant to compute into the fixture W5A asserts
    against would make the pair of them agree by construction.
    """
    n_frames = mono.size // STFT_HOP_LENGTH
    blocks = mono[: n_frames * STFT_HOP_LENGTH].reshape(n_frames, STFT_HOP_LENGTH)
    return np.sqrt(np.square(blocks.astype(np.float64)).mean(axis=1)).astype(np.float32)


def bass_f0(mono: npt.NDArray[np.float32], sample_rate: int) -> dict[str, Any]:
    """Raw framewise F0 from the active backend, as `PitchTrack` arrays.

    The fixture W4C needs is the *input* to note segmentation, not the v4 note
    list: the whole point is to re-run segmentation and show the 32 ms onset lag
    is gone. A committed note list would already have the lag baked in.
    """
    from audio_pipeline.backends import get_backend

    backend = get_backend()
    track = backend.pitch(mono, sample_rate)
    return {
        "f0_hz": np.asarray(track.f0_hz, dtype=np.float32),
        "voiced": np.asarray(track.voiced, dtype=bool),
        "voiced_probability": np.asarray(track.voiced_probability, dtype=np.float32),
        "frame_hop_seconds": np.float64(track.frame_hop_seconds or float("nan")),
        "method": np.str_(track.method or "unknown"),
        "backend": np.str_(backend.name),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems_dir", type=Path, help="Directory holding <stem>.wav")
    parser.add_argument("--track", required=True, help="Track slug, used in filenames")
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/real"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"track": args.track, "hop_seconds": HOP_SECONDS, "files": {}}

    drums, rate = load_audio(args.stems_dir / "drums.wav", mono=True)
    assert rate == ANALYSIS_SAMPLE_RATE, f"expected {ANALYSIS_SAMPLE_RATE} Hz, got {rate}"
    mono_drums = to_mono(drums)
    envelopes = band_envelopes(mono_drums, rate)
    path = args.out / f"{args.track}__drums_band_envelopes.npz"
    np.savez_compressed(
        path,
        hop_seconds=np.float64(HOP_SECONDS),
        sample_rate=np.int32(rate),
        tempo_band_hz=np.asarray(TEMPO_BAND_HZ, dtype=np.float64),
        detection_bands=np.str_(json.dumps(DETECTION_BANDS)),
        **envelopes,
    )
    manifest["files"][path.name] = {
        "from": "drums stem",
        "frames": int(envelopes["band_tempo"].size),
        "contents": "per-frame energy in each DETECTION_BANDS band, plus band_tempo (20-110 Hz)",
    }

    rms = {}
    for stem in STEM_NAMES:
        stem_path = args.stems_dir / f"{stem}.wav"
        if not stem_path.exists():
            continue
        audio, stem_rate = load_audio(stem_path, mono=True)
        assert stem_rate == ANALYSIS_SAMPLE_RATE
        rms[f"rms_{stem}"] = frame_rms(to_mono(audio))
    kick_env = band_envelopes(mono_drums, rate)["band_kick"]
    path = args.out / f"{args.track}__stem_frame_rms.npz"
    np.savez_compressed(
        path,
        hop_seconds=np.float64(HOP_SECONDS),
        kick_band_energy=kick_env,
        **rms,
    )
    manifest["files"][path.name] = {
        "from": "all stems",
        "frames": int(next(iter(rms.values())).size),
        "contents": (
            "RMS per 512-sample block per stem, plus the drums stem's kick-band "
            "energy: 'the drums stem is loud' and 'the kick is playing' are "
            "different facts (W5A task 1)"
        ),
    }

    bass, bass_rate = load_audio(args.stems_dir / "bass.wav", mono=True)
    assert bass_rate == ANALYSIS_SAMPLE_RATE
    track_arrays = bass_f0(to_mono(bass), bass_rate)
    path = args.out / f"{args.track}__bass_f0.npz"
    np.savez_compressed(path, **track_arrays)
    manifest["files"][path.name] = {
        "from": "bass stem",
        "frames": int(track_arrays["f0_hz"].size),
        "contents": "raw framewise F0, voicing decision and voicing probability",
    }

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
