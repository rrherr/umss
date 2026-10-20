"""How tests/data/reference.npz was made: the repository's own code at commit 05725fa (before it became
a package), run with 4 s chunks and the absent-voice mask override, seeded after loading, on the CPU.

    git worktree add /tmp/umss-05725fa 05725fa
    python tests/make_input.py            # writes input.npz
    REPO=/tmp/umss-05725fa python tests/make_reference.py input.npz out.npz

`voices` in reference.npz is `masked` from out.npz; `audio` and `f0` (first three columns) come from
input.npz. COMPAT=1 instead loads a shim restoring torch 1.6's FFT functions, for a checkout older
than 05725fa; it needs that shim on the path.
"""
import os, sys
import numpy as np, torch
REPO = os.environ["REPO"]
if os.environ.get("COMPAT"):
    import torch16_compat  # noqa
sys.path.insert(0, REPO); os.chdir(REPO)
import models  # noqa
import utils, ddsp.spectral_ops
NAMES = ["tenor", "lead", "baritone"]
SR, CHUNK, HOP_SAMPLES = 16000, 64000, 256
STEP = CHUNK // 2
def masking_with_override(true_mix, estimated_sources, f0, n_fft=2048, n_hop=256):
    _, n_sources, n_samples = estimated_sources.shape
    overlap = 1 - n_hop / n_fft
    est = estimated_sources.reshape((n_sources, -1))
    mix_stft = ddsp.spectral_ops.stft(true_mix, frame_size=n_fft, overlap=overlap, center=True, pad_end=False)
    mix_stft = torch.view_as_complex(mix_stft.repeat_interleave(n_sources, dim=0))
    mag = ddsp.spectral_ops.compute_mag(est, n_fft, overlap, center=True, pad_end=False)
    mask = mag / (mag.sum(0, keepdim=True) + 1e-12)
    n_frames = mask.shape[2]
    voiced = (f0 > 0)[torch.clamp(torch.arange(n_frames), max=f0.shape[0] - 1)].clone()
    real = torch.zeros(n_sources, dtype=torch.bool); real[:len(NAMES)] = True
    none = ~voiced.any(1); voiced[none] = real
    keep = voiced.T.float()[:, None, :]
    mask = mask * keep
    denom = mask.sum(0, keepdim=True)
    mask = torch.where(denom > 1e-6, mask / (denom + 1e-12), keep / keep.sum(0, keepdim=True))
    single = voiced.sum(1) == 1
    mask[:, :, single] = voiced[single].T.float()[:, None, :]
    src = torch.view_as_real(mask * mix_stft)
    return ddsp.spectral_ops.istft(src, n_fft, overlap, center=True, length=n_samples)
torch.set_grad_enabled(False)

model = utils.load_model("unsupervised_4s_satb_bcbq_mf0_1_again", "cpu"); model.return_sources = True
torch.manual_seed(0); np.random.seed(0)
window = torch.hann_window(CHUNK, periodic=True)
d = np.load(sys.argv[1]); audio = d["audio"]; f0 = torch.from_numpy(d["f0"]).float()
n = len(audio); pad = STEP; total = pad + n + pad + CHUNK
x = torch.zeros(total); x[pad:pad + n] = torch.from_numpy(audio)
fp = torch.zeros(total // HOP_SAMPLES + 1, f0.shape[1]); fp[pad // HOP_SAMPLES:pad // HOP_SAMPLES + len(f0)] = f0
masked = torch.zeros(f0.shape[1], total)
for s in range(0, pad + n, STEP):
    chunk = x[s:s + CHUNK]; peak = chunk.abs().max()
    if peak < 1e-5: continue
    f0c = fp[s // HOP_SAMPLES:s // HOP_SAMPLES + CHUNK // HOP_SAMPLES]
    mix = chunk / peak
    _, sources = model(mix[None], f0c[None])
    f0m = fp[s // HOP_SAMPLES:s // HOP_SAMPLES + CHUNK // HOP_SAMPLES + 1]
    m = masking_with_override(mix[None], sources, f0m, 2048, 256)
    masked[:, s:s + CHUNK] += m * peak * window
np.savez(sys.argv[2], masked=masked[:3, pad:pad + n].numpy())
print("done", masked.abs().max().item())
