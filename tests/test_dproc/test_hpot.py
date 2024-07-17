import os

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
        "tests/data/HP_data",
        context_size=128,
        window_size=10,
        tokenizer=gpt2_tokenizer,
        remove_format_chars=True,
        remove_punc_spacing=True,
        pool_rois=False
    )


@pytest.fixture
def ds_pool(gpt2_tokenizer):
    return fMRIDataset.get_dataset(
        "hp",
        "tests/data/HP_data",
        context_size=128,
        window_size=10,
        tokenizer=gpt2_tokenizer,
        remove_format_chars=True,
        remove_punc_spacing=True,
        pool_rois=False
    )

def test_testdir():
    print(os.listdir())
    assert False