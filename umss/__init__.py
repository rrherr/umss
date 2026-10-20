"""Pitch-informed separation of a vocal ensemble with the pretrained UMSS models.

    model = umss.load("cpu")                      # 4 sources; or "cuda", "mps"
    voices = umss.separate(model, mono_16k, f0)   # float32 (voices, samples)

`f0` holds one column of pitches in Hz per voice (0 = silent), on the model's 16 ms frame grid:
`umss.n_frames(len(mono_16k))` rows, frame `i` at `i * HOP_LENGTH` samples.
"""

from __future__ import annotations

import json
from importlib import resources

import numpy as np
import torch
from numpy.typing import NDArray

from .ddsp import core, spectral_ops
from .model import SourceFilterMixtureAutoencoder2

__all__ = ["CHECKPOINTS", "HOP_LENGTH", "SAMPLE_RATE", "f0_frames", "load", "n_frames", "separate"]

SAMPLE_RATE = 16000
"""The models' sample rate (mono)."""
HOP_LENGTH = 256
"""Samples per f0 frame (16 ms)."""
CHUNK = 64000
"""The models' fixed input length (4 s); `separate` runs it on chunks hopped by half of it."""
CHECKPOINTS = {
    2: "unsupervised_2s_satb_bcbq_mf0_1",
    4: "unsupervised_4s_satb_bcbq_mf0_1_again",
}
"""The pretrained checkpoints shipped in the package, by number of sources."""

_N_FFT = 2048
"""The STFT the synthesised sources' masks are applied in (hop `HOP_LENGTH`)."""


def load(device: str = "cpu", n_sources: int = 4) -> SourceFilterMixtureAutoencoder2:
    """The pretrained model with `n_sources` sources (2 or 4), on `device`, in eval mode."""
    tag = CHECKPOINTS[n_sources]
    files = resources.files(__package__) / "checkpoints"
    config = json.loads((files / f"{tag}.json").read_text(encoding="utf-8"))
    with resources.as_file(files / f"{tag}.pth") as path:
        state = torch.load(path, map_location="cpu", weights_only=True)
    model = SourceFilterMixtureAutoencoder2.from_config(config)
    model.load_state_dict(state)
    model.return_sources = True
    moved: SourceFilterMixtureAutoencoder2 = model.to(device).eval()
    return moved


def n_frames(n_samples: int) -> int:
    """The number of f0 frames `separate` expects for `n_samples` of audio."""
    return n_samples // HOP_LENGTH + 1


def f0_frames(f0: NDArray[np.floating], n_samples: int) -> NDArray[np.float32]:
    """Pitch contours spanning `n_samples` of audio, on `separate`'s frames: (n_frames, voices).

    `f0` is (voices, frames) on any grid covering the same span. It is resampled linearly by
    frame count, zeros included, to `n_samples // HOP_LENGTH` frames, as the upstream
    preprocessing does, and a silent frame is added at the end.
    """
    x = torch.as_tensor(np.asarray(f0, dtype=np.float32))
    out = core.resample(x, n_samples // HOP_LENGTH).transpose(0, 1)
    out = torch.cat([out, torch.zeros(1, out.shape[1])])
    return np.ascontiguousarray(out.numpy(), dtype=np.float32)


def _masks(
    mix: torch.Tensor, sources: torch.Tensor, f0: torch.Tensor, n_voices: int
) -> torch.Tensor:
    """The voices from one chunk: the synthesised sources' magnitude ratios applied to `mix`.

    `mix` [1, samples], `sources` [1, n_sources, samples], `f0` [frames, n_sources]. In each STFT
    frame only the voices with a pitch share the mix (one such voice takes all of it); in a frame
    where none has one, the `n_voices` real voices share it. The other sources never get any.
    """
    _, n_sources, n_samples = sources.shape
    overlap = 1 - HOP_LENGTH / _N_FFT
    est = sources.reshape((n_sources, -1))
    mix_stft = spectral_ops.stft(mix, frame_size=_N_FFT, overlap=overlap, center=True, pad_end=False)
    mix_stft = torch.view_as_complex(mix_stft.repeat_interleave(n_sources, dim=0))
    mag = spectral_ops.compute_mag(est, _N_FFT, overlap, center=True, pad_end=False)
    mask = mag / (mag.sum(0, keepdim=True) + 1e-12)
    frames = mask.shape[2]
    rows = torch.clamp(torch.arange(frames, device=f0.device), max=f0.shape[0] - 1)
    voiced = (f0 > 0)[rows].clone()  # [frames, n_sources]
    real = torch.zeros(n_sources, dtype=torch.bool, device=f0.device)
    real[:n_voices] = True
    voiced[~voiced.any(1)] = real
    keep = voiced.T.float()[:, None, :]  # [n_sources, 1, frames]
    mask = mask * keep
    denom = mask.sum(0, keepdim=True)
    # bins where every allowed voice's mask is ~0 are split equally among them, so nothing is lost
    mask = torch.where(denom > 1e-6, mask / (denom + 1e-12), keep / keep.sum(0, keepdim=True))
    single = voiced.sum(1) == 1
    mask[:, :, single] = voiced[single].T.float()[:, None, :]
    src = torch.view_as_real(mask * mix_stft)
    out: torch.Tensor = spectral_ops.istft(src, _N_FFT, overlap, center=True, length=n_samples)
    return out


def separate(
    model: SourceFilterMixtureAutoencoder2,
    mono: NDArray[np.floating],
    f0: NDArray[np.floating],
    seed: int = 0,
) -> NDArray[np.float32]:
    """The voices of `mono` (16 kHz) guided by `f0` [n_frames(len(mono)), voices]: (voices, samples).

    Voices are at most the model's sources; missing sources get no pitch and no output. The input
    is cut into 4 s chunks hopped by 2 s (padded by 2 s at both ends, so every sample is in two
    chunks), each scaled to a peak of 1 for the model and back after, and overlap-added with
    periodic Hann windows. Silent chunks give silence. The noise part of the synthesis is
    random; `seed` fixes it on the CPU without touching the global random state (on a GPU the
    noise differs from run to run, by about as much as between two seeds). Within one STFT
    window (128 ms) of either end, the frames reach past the input, where no voice has a pitch,
    so those samples are shared as in a frame without pitch: give the input some context.
    """
    x_np = np.asarray(mono, dtype=np.float32)
    f0_np = np.asarray(f0, dtype=np.float32)
    n = len(x_np)
    n_voices = f0_np.shape[1]
    n_sources = model.encoder.n_sources
    if x_np.ndim != 1 or f0_np.ndim != 2 or f0_np.shape[0] != n_frames(n):
        raise ValueError(f"expected mono audio and f0 of shape ({n_frames(n)}, voices)")
    if not 1 <= n_voices <= n_sources:
        raise ValueError(f"the model separates at most {n_sources} voices, not {n_voices}")
    device = next(model.parameters()).device
    step = CHUNK // 2
    pad = step
    total = pad + n + pad + CHUNK
    x = torch.zeros(total)
    x[pad : pad + n] = torch.from_numpy(x_np)
    fp = torch.zeros(total // HOP_LENGTH + 1, n_sources)
    fp[pad // HOP_LENGTH : pad // HOP_LENGTH + len(f0_np), :n_voices] = torch.from_numpy(f0_np)
    out = torch.zeros(n_sources, total)
    window = torch.hann_window(CHUNK, periodic=True)
    per_chunk = CHUNK // HOP_LENGTH
    np_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(seed)
            np.random.seed(seed)
            for s in range(0, pad + n, step):
                chunk = x[s : s + CHUNK]
                peak = chunk.abs().max()
                if peak < 1e-5:
                    continue
                first = s // HOP_LENGTH
                mix = (chunk / peak)[None].to(device)
                _, sources = model(mix, fp[first : first + per_chunk][None].to(device))
                f0m = fp[first : first + per_chunk + 1].to(device)
                voices = _masks(mix, sources, f0m, n_voices).cpu()
                out[:, s : s + CHUNK] += voices * peak * window
    finally:
        np.random.set_state(np_state)
    return np.ascontiguousarray(out[:n_voices, pad : pad + n].numpy(), dtype=np.float32)
