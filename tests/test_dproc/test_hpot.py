import random

from circuit_brain.dproc import fMRIDataset

import pytest
from transformers import AutoTokenizer


@pytest.fixture
def gpt2_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def ds_no_pool(gpt2_tokenizer):
    return fMRIDataset.get_dataset(
        "hp",
        "tests/toy_data/HP_data",
        context_size=128,
        window_size=10,
        tokenizer=gpt2_tokenizer,
        remove_format_chars=True,
        remove_punc_spacing=True,
        pool_rois=False,
    )


@pytest.fixture
def ds_pool(gpt2_tokenizer):
    return fMRIDataset.get_dataset(
        "hp",
        "tests/toy_data/HP_data",
        context_size=128,
        window_size=10,
        tokenizer=gpt2_tokenizer,
        remove_format_chars=True,
        remove_punc_spacing=True,
        pool_rois=True,
    )


@pytest.fixture
def subj_idx():
    return ["F", "H", "I", "J", "K", "L", "M", "N"]


@pytest.fixture
def regions():
    return [
        "all",
        "PostTemp",
        "MFG",
        "IFGorb",
        "dmpfc",
        "pCingulate",
        "AntTemp",
        "AngularG",
        "IFG",
    ]


def test_ds_init(ds_no_pool, subj_idx):
    assert ds_no_pool.remove_format_chars
    assert ds_no_pool.remove_punc_spacing
    assert ds_no_pool.pool_rois == False
    assert len(ds_no_pool.idxmap) == len(ds_no_pool.contexts)
    assert len(ds_no_pool.toks) == len(ds_no_pool.contexts)
    assert subj_idx == ds_no_pool.subject_idxs
    assert ds_no_pool.dataset_id == "hp"
    # check that all of the token dimension are the same
    for i in range(len(ds_no_pool.toks)):
        assert len(ds_no_pool.toks[i]) == len(ds_no_pool.idxmap[i])


def test_len_getitem_no_pool(ds_no_pool):
    assert len(ds_no_pool) == 8
    for idx in range(len(ds_no_pool)):
        sfmri = ds_no_pool[idx]
        assert sfmri.shape[0] == 240
        assert sfmri.shape[1] == 80


def test_len_getitem_pool(ds_pool, regions):
    assert len(ds_pool) == 8
    assert ds_pool.pool_rois == True
    for idx in range(len(ds_pool)):
        sfmri, rois = ds_pool[idx]
        assert sfmri.shape[0] == 240
        assert sfmri.shape[1] == 80
        assert type(rois) == dict
        assert len(rois) == 9
        for r in regions:
            assert r in rois.keys()
        for k, v in rois.items():
            assert k in regions
            if k != "all":
                assert v.shape[0] == 80


def test_idx2samples_pool(ds_pool):
    # get 4 different fMRI measurements
    idxs = random.sample(range(0, 80), 15)
    for j in range(len(ds_pool)):
        measures, toks, idxmap = ds_pool.idx2samples(j, idxs)
        assert measures.shape[0] == 15
        assert measures.shape[1] == 8
        assert toks.shape[0] == idxmap.shape[0] == 15
        assert toks.shape[1] == idxmap.shape[1] == 128


def test_idx2samples_no_pool(ds_no_pool): ...


def test_kfold(ds_pool): ...
