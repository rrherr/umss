"""Synthetic three-voice input for the reference test: harmonic tones with vibrato, one voice pausing, one entering late."""
import numpy as np
SR, HOP, N = 16000, 256, 6 * 16000
rng = np.random.default_rng(0)
t_frames = np.arange(N // HOP + 1) * HOP / SR
def contour(base, on):
    f = base * 2 ** (0.3 / 12 * np.sin(2 * np.pi * 5.5 * t_frames))
    return np.where(on, f, 0.0)
on_all = (t_frames > 0.3) & (t_frames < 5.6)
f0 = np.zeros((len(t_frames), 4), np.float32)
f0[:, 0] = contour(392.0, on_all & ~((t_frames > 2.0) & (t_frames < 2.6)))   # tenor, a gap
f0[:, 1] = contour(329.6, on_all)                                             # lead
f0[:, 2] = contour(261.6, on_all & (t_frames > 1.0))                          # baritone enters late
t = np.arange(N) / SR
x = np.zeros(N)
for v in range(3):
    fs = np.interp(t, t_frames, f0[:, v])
    ph = 2 * np.pi * np.cumsum(fs) / SR
    amp = (fs > 0) * 0.15
    for h in range(1, 12):
        x += amp * np.sin(h * ph) / h ** 1.2 * (h * fs < SR / 2)
x += 0.002 * rng.standard_normal(N)
np.savez("input.npz", audio=x.astype(np.float32), f0=f0)
print(x.shape, f0.shape, np.abs(x).max())
