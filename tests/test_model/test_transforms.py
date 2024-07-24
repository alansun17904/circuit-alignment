from circuit_brain.model import transforms
from circuit_brain.utils import PCA

import torch
import pytest
import numpy as np


@pytest.fixture
def layer_repr():
    layers = []
    for l in range(13):
        layers.append(torch.randn(4, 35, 20))
    return layers


@pytest.fixture
def pca_layer_repr():
    layers = []
    for l in range(13):
        layers.append(torch.randn(120, 20))
    return layers


@pytest.fixture
def word2idx():
    return [
        [7] * 5 + [6] * 8 + [5] * 3 + [4] * 2 + [3] * 10 + [2] * 1 + [1] * 4 + [0] * 2,
        [5] * 25 + [4] * 2 + [3] * 1 + [2] * 1 + [1] * 4 + [0] * 2,
        [2] * 15 + [1] * 10 + [0] * 10,
        [0] * 35,
    ]


@pytest.fixture
def filter():
    return (
        torch.ones(
            35,
        )
        / 35
    )


@pytest.fixture
def pcas():
    return [PCA(5), PCA(10), PCA(15), PCA(17)] * 3 + [PCA(10)]


def test_avg(layer_repr):
    avg = transforms.Avg()
    rproc = avg(layer_repr)
    assert len(rproc) == 13
    for l in rproc:
        assert l.shape[0] == 4
        assert l.shape[1] == 20


def test_normalize(layer_repr):
    norm = transforms.Normalize()
    rproc = norm(layer_repr)
    assert len(rproc) == 13
    for l in rproc:
        assert l.shape[0] == 4
        assert l.shape[1] == 35
        assert l.shape[2] == 20
        # check that the mean is about 0
        m = torch.mean(l, dim=1)
        np.testing.assert_array_almost_equal(m, np.zeros(m.shape))


def test_wordavg(layer_repr, word2idx):
    assert len(word2idx) == 4
    for wi in word2idx:
        assert len(wi) == 35
    wordavg = transforms.WordAvg(word2idx)
    rproc = wordavg(layer_repr)
    # most words in a sentence
    maxed = max(max(w) for w in word2idx)
    assert len(rproc) == 13
    for idx, l in enumerate(rproc):
        assert l.shape[0] == 4
        assert l.shape[1] == maxed + 1
        assert l.shape[2] == 20


def test_concat(layer_repr):
    concat = transforms.Concat()
    rproc = concat(layer_repr)
    assert len(rproc) == 13
    for l in rproc:
        assert l.shape[0] == 4
        assert l.shape[1] == 35 * 20


def test_ccrop(layer_repr):
    ccrop = transforms.ContextCrop(20)
    rproc = ccrop(layer_repr)
    assert len(rproc) == 13
    for idx, l in enumerate(rproc):
        assert l.shape[0] == 4
        assert l.shape[1] == 20
        assert l.shape[2] == 20
        np.testing.assert_array_almost_equal(layer_repr[idx][:, -20:, :], l)


def test_convolve(layer_repr, filter):
    conv = transforms.ConvolveHRF(filter)
    rproc = conv(layer_repr)
    assert len(rproc) == 13
    for idx, l in enumerate(rproc):
        assert l.shape[0] == 4
        assert l.shape[1] == 1
        assert l.shape[2] == 20
        torch.testing.assert_close(l, layer_repr[idx].mean(axis=1).unsqueeze(1))


def test_compose(layer_repr): ...


def test_pcat(pcas, pca_layer_repr):
    pcat = transforms.PCAt(pcas, fit=True)

    assert len(pca_layer_repr) == 13
    pcaed = pcat(pca_layer_repr)
    assert len(pcaed) == 13
    for idx, l in enumerate(pcaed):
        assert l.shape[0] == 120
        assert l.shape[1] == pcas[idx].n_components
