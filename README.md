# Unsupervised Music Source Separation Using Differentiable Parametric Source Models

This is the source code for the experiments related to the paper [Unsupervised Music Source Separation Using Differentiable Parametric Source Models](https://arxiv.org/abs/2201.09592).  

It contains a re-implementation of parts of the DDSP library in PyTorch. We added a differentiable all-pole filter which can be parameterized by line spectral frequencies or reflection coefficients. 

Please cite the paper, if you use parts of the code in your work.

## Links
[:loud_sound: Audio examples](https://schufo.github.io/umss/)

[:page_facing_up: Paper](https://arxiv.org/abs/2201.09592)

[:file_folder: CSD Database](https://zenodo.org/record/1286570#.Y0ZsbNJByUk)

[:microphone: Multiple-f0 estimation](https://github.com/helenacuesta/multif0-estimation-polyvocals)

## About this fork

This fork keeps only **inference** with the two pretrained models, as an installable package
(`umss`) whose only dependencies are PyTorch, NumPy and SciPy. Training, evaluation, the
datasets, the U-Net baselines, the Cuesta multiple-f0 preprocessing and the robustness tests
were removed; see the [upstream repository](https://github.com/schufo/umss) for them. The
weights of `unsupervised_2s_satb_bcbq_mf0_1` and `unsupervised_4s_satb_bcbq_mf0_1_again`
(21 MB each) ship inside the package with their training arguments; the optimizer checkpoints
and training histories do not.

Changes to the code:

- It runs on PyTorch 2.x: the removed `torch.rfft` / `torch.irfft` are replaced by `torch.fft`,
  keeping the old `irfft` output length of `2n - 1` in `frequency_impulse_response`, which the
  models were trained with, and `torch.stft` / `torch.istft` use complex tensors.
- It runs on Apple Silicon (MPS): the float64 LSF-to-filter conversion runs on the CPU.
- `separate` adds what the upstream evaluation did around the network for a whole recording of
  any length: 4 s chunks hopped by 2 s, each scaled to a peak of 1 and back, overlap-added with
  periodic Hann windows; and the masking of the mixture by the synthesised sources, with one
  change: in each STFT frame only the voices that have a pitch share the mixture (a lone voice
  takes all of it, and in a frame where none has a pitch, all voices share it), so a silent
  voice gets nothing.
- `models.py` keeps only `SourceFilterMixtureAutoencoder2` (now `umss/model.py`), the
  `ddsp` and `network_components` modules lose their plotting imports, and the code imports as
  a package.

The weights and the network's forward pass are otherwise unchanged. Before the cut, the port
to PyTorch 2.x was checked against the original environment (Python 3.8, PyTorch 1.6): every
metric of `eval.py` on both models matched to 4 decimals. A test checks that the package's
output is identical, on the CPU, to that of the code before the cut (`tests/make_reference.py`).

## Install

```bash
uv add git+https://github.com/rrherr/umss
```

## Use

```python
import umss

model = umss.load("cpu")                     # 4 sources (n_sources=2 for the 2-source model); or "cuda", "mps"
f0 = umss.f0_frames(contours, len(mono_16k)) # contours (voices, frames) in Hz, 0 = silent, over the same span
voices = umss.separate(model, mono_16k, f0)  # float32 array (voices, samples)
```

Audio is mono at 16 kHz (`umss.SAMPLE_RATE`); pitches are on 16 ms frames (`umss.HOP_LENGTH`
samples), voices ordered as the model expects (high to low for the SATB models), at most the
model's number of sources. The synthesis noise is random; `seed` (default 0) fixes it on the CPU.
The outputs are band-limited to 8 kHz. On MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1` before
importing torch.

## Tests

```bash
uv run --group dev pytest
```

## Acknowledgment

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068.

## Copyright

Copyright 2021 Kilian Schulze-Forster of Télécom Paris, Institut Polytechnique de Paris.
All rights reserved.
