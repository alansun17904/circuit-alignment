from typing import Tuple, List, Union

import tqdm
import numpy as np
from transformers import PreTrainedTokenizer, BertTokenizer, BertTokenizerFast


def word_token_corr(
    tokenizer: PreTrainedTokenizer, text: Union[str, List[str]], **kwargs
) -> Tuple[
    Union[List[int], List[List[int]]], Union[List[str], List[List[str]]], np.array
]:
    """Returns the correspondance between tokens and their words. Note that we
    define a word as any sequence of characters that is surrounded by whitespace
    on either side.

    Args:
        tokenizer: A HuggingFace PreTrainedTokenizer
        text: A str that we wish to tokenize or a list of strings that we want to tokenize.
        kwargs: Keywords arguments that go into the tokenizer.

    Returns:
        A tuple where the first entry the tokenized text, then a list of words that
        we have extracted from the text. The third entry is an array that has the
        same length as the number of tokens in the provided text. Each value
        in the array correspond to the index of the word it represents.
    """
    not_list = False
    if not isinstance(text, list):
        text, not_list = [text], True

    toks = tokenizer.batch_encode_plus(text, return_offsets_mapping=True, **kwargs)
    word_split = []
    index_mapping = []
    for idx, t in tqdm.tqdm(
        enumerate(toks["input_ids"]), desc="Tokenizing", total=len(toks["input_ids"])
    ):
        word, indices = _per_example_token_mapping(
            text[idx], t, toks["offset_mapping"][idx]
        )
        word_split.append(word)
        index_mapping.append(indices)

    if not_list:
        return toks["input_ids"][0], word_split[0], index_mapping[0]

    return toks, word_split, index_mapping


def _per_example_token_mapping(
    text: str,
    toks: List[int],
    offset_mapping: List[Tuple[int, int]],
) -> Tuple[List[str], np.array]:
    """The same as the previous function, however, this function only
    operates on the level of a single string. For information on arguments
    and return types see above. We start at the last word and build the string
    from right to left since the left-side could be truncated.
    """
    lc_words = text.lower().split()
    indices = np.zeros(len(toks))
    word_counter, accum_word = 0, str()

    for i in range(len(toks) - 1, -1, -1):
        # ignore count if we see special tokens
        om = offset_mapping[i]
        indices[i] = word_counter if word_counter < len(lc_words) else word_counter - 1
        if om[0] == 0 and om[1] == 0:
            continue
        accum_word = text.lower()[om[0] : om[1]] + accum_word
        if lc_words[-word_counter - 1] in accum_word:
            word_counter += 1
            accum_word = str()
    return text.split(), indices
