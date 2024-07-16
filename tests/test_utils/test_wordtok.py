from circuit_brain.utils import word_token_corr

import pytest
import numpy as np
from transformers import AutoTokenizer


@pytest.fixture(scope="function")
def token_ans(request):
    name = request.param
    tokenizer = AutoTokenizer.from_pretrained(name)
    ans = None
    if name == "gpt2":  # BPE
        ans = [
            [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7],
            [0, 1, 2, 2, 2, 3, 4, 5, 5],
        ]
        tokenizer.pad_token = tokenizer.eos_token
    elif name == "bert-base-uncased":  # WordPiece
        ans = [
            [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7],
            [0, 0, 1, 2, 2, 2, 3, 4, 5, 5, 5],
        ]
    elif name == "roberta-base":  # BPE w/ special tokens
        ans = [
            [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7],
            [0, 0, 1, 2, 2, 2, 3, 4, 5, 5, 5],
        ]
    elif name == "t5-base":  # SentencePiece
        ans = [
            [0, 0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 7, 7],
            [0, 1, 2, 2, 2, 2, 2, 3, 4, 5, 5, 5],
        ]
    return tokenizer, ans


@pytest.fixture
def sents():
    return [
        "Malfoy certainly did talk a lot about flying.",
        "8 8 ksd! some more tricky_",
    ]


@pytest.mark.parametrize(
    "token_ans", ["gpt2", "bert-base-uncased", "roberta-base", "t5-base"], indirect=True
)
def test_word_token_corr_single(token_ans, sents):
    tokenizer, ans = token_ans
    for i, sent in enumerate(sents):
        _, words, idx = word_token_corr(tokenizer, sent)
        np.testing.assert_array_equal(ans[i], idx)
        assert words == sent.split(" ")


@pytest.mark.parametrize(
    "token_ans", ["gpt2", "bert-base-uncased", "roberta-base", "t5-base"], indirect=True
)
def test_word_token_corr_multi(token_ans, sents):
    tokenizer, ans = token_ans

    # extend the second sentence of the answer token to the same length
    pad = [ans[1][-1]] * (len(ans[0]) - len(ans[1]))
    ans[1].extend(pad)

    _, words, idx = word_token_corr(tokenizer, sents, padding=True)
    np.testing.assert_array_equal(ans, idx)
    assert words == [s.split(" ") for s in sents]