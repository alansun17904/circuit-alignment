from circuit_brain.model import BrainAlignTransformer

import torch
import pytest
from pytest_lazyfixture import lazy_fixture
from transformers import AutoTokenizer


@pytest.fixture
def ba_gpt2():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok, BrainAlignTransformer.from_pretrained("gpt2-small")


@pytest.fixture
def ba_t5():
    return AutoTokenizer.from_pretrained(
        "google-t5/t5-small"
    ), BrainAlignTransformer.from_pretrained("t5-small")


@pytest.fixture
def ba_bert():  # at the moment BERT is not supported
    return AutoTokenizer.from_pretrained(
        "bert-base-cased"
    ), BrainAlignTransformer.from_pretrained("bert-base-cased")


@pytest.fixture
def sents():
    return [
        "Malfoy certainly did talk a lot about flying",
        "Neville had never been on a broomstick in his life, because his",
        "Hermione Granger was almost as nervous about flying as Neville was",
        "No sooner were they out of earshot than Malfoy burst into laughter.",
        "Blood was pounding on his ears. He mounted the broom and"
        "kicked hard against the ground and up, up he soard.",
        "his robes were whipped out behind him–and in a rush of fierce joy he realized he'd"
        "found something he could do without being tauhgt.",
    ]


def model_logit_reprs(ba_tmodel, sents, **kwargs):
    tok, model = ba_tmodel
    tok.padding_side = "left"
    tokenized_sents = tok.batch_encode_plus(
        sents, padding="max_length", max_length=128
    )["input_ids"]
    if "T5" in str(model.ht):
        return model.resid_post(
            torch.LongTensor(tokenized_sents),
            decoder_input=torch.LongTensor(tokenized_sents),
            **kwargs
        )
    return model.resid_post(torch.LongTensor(tokenized_sents), **kwargs)


def test_resid_post_identity_bert(ba_bert, sents):
    model_logit_reprs(ba_bert, sents)


def test_resid_post_identity_t5(ba_t5, sents):
    model_logit_reprs(ba_t5, sents)


def test_resid_post_identity_gpt(ba_gpt2, sents):
    model_logit_reprs(ba_gpt2, sents)


def test_generate_w_logits(ba_gpt2, sents):
    _, model = ba_gpt2
    for sent in sents:
        tokens, logits = model.generate(sent)
        assert logits.shape[0] == 1
        assert logits.shape[1] == len(tokens)
        assert logits.shape[2] == ba_gpt2.cfg.d_vocab
