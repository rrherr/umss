"""The package against the repository's original code (see make_reference.py)."""

from pathlib import Path

import numpy as np
import pytest

import umss

DATA = np.load(Path(__file__).parent / "data" / "reference.npz")


@pytest.fixture(scope="module")
def model():
    return umss.load("cpu")


def test_matches_the_original_code_exactly(model):
    voices = umss.separate(model, DATA["audio"], DATA["f0"])
    assert voices.dtype == np.float32
    np.testing.assert_array_equal(voices, DATA["voices"])


def test_leaves_the_global_random_state_alone(model):
    before = np.random.get_state()[1].copy()
    umss.separate(model, DATA["audio"][:16000], DATA["f0"][: umss.n_frames(16000)])
    assert np.array_equal(np.random.get_state()[1], before)


def test_a_lone_voiced_voice_takes_the_whole_input(model):
    n = 32000
    x = DATA["audio"][:n]
    f0 = np.zeros((umss.n_frames(n), 3), np.float32)
    f0[:, 1] = 220.0
    voices = umss.separate(model, x, f0)
    # within one STFT window of either end, frames past the input have no pitch and are shared
    inner = slice(2048, n - 2048)
    assert not voices[0][inner].any() and not voices[2][inner].any()
    assert np.abs(voices[1][inner] - x[inner]).max() < 1e-5


def test_silence_gives_silence(model):
    n = 16000
    voices = umss.separate(model, np.zeros(n, np.float32), np.zeros((umss.n_frames(n), 2), np.float32))
    assert voices.shape == (2, n) and not voices.any()


def test_rejects_a_misaligned_f0(model):
    with pytest.raises(ValueError):
        umss.separate(model, np.zeros(16000, np.float32), np.zeros((10, 2), np.float32))
    with pytest.raises(ValueError):
        umss.separate(model, np.zeros(16000, np.float32), np.zeros((umss.n_frames(16000), 5), np.float32))
