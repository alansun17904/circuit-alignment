import re
from typing import Tuple, List, Union

import numpy as np
from transformers import PreTrainedTokenizer, BertTokenizer, BertTokenizerFast


def word_token_corr(
    tokenizer: PreTrainedTokenizer, text: Union[str, List[str]], **kwargs
) -> Tuple[Union[List[int], List[List[int]]], Union[List[str], List[List[str]]], np.array]:
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

    if not isinstance(text, list):
        toks = tokenizer.encode(text)
        return toks, *per_example_token_mapping(tokenizer, text, toks)
    toks = tokenizer.batch_encode_plus(text, **kwargs)
    word_mapping = []
    index_mapping = []
    for idx, t in enumerate(toks["input_ids"]):
        word, indices = per_example_token_mapping(tokenizer, text[idx], t)
        word_mapping.append(word)
        index_mapping.append(indices)
    return toks, word_mapping, np.array(index_mapping)


def per_example_token_mapping(
    tokenizer: PreTrainedTokenizer, text: str, toks: List[int]
) -> Tuple[List[str], np.array]:
    words = text.split()
    lc_words = text.lower().split()
    indices = np.zeros(len(toks))
    word_counter = 0
    decode_accum = ""

    for i in range(len(toks)):
        accum_text = " ".join(lc_words[: word_counter + 1])
        indices[i] = word_counter

        # check if we are working with WordPiece tokenizer
        if isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):
            # # WordPiece removes all white space in front of non-alphanumeric symbols
            symb = tokenizer.decode(toks[i])
            if re.search(r"##[a-z]+", symb):
                decode_accum += symb[2:]
            elif not any(c.isalnum() for c in symb):
                if words[min(word_counter + 1, len(words) - 1)] == symb:
                    decode_accum += f" {symb}"
                else:
                    decode_accum += symb
            else:
                decode_accum += f" {symb}"
        else:
            decode_accum = tokenizer.decode(toks[: i + 1]).lower()

        if accum_text in decode_accum and word_counter < len(lc_words) - 1:
            word_counter += 1
    return words, indices
