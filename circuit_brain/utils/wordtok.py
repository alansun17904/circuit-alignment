import re
from typing import Tuple, List

import numpy as np
from transformers import PreTrainedTokenizer, BertTokenizer, BertTokenizerFast


def word_token_corr(tokenizer: PreTrainedTokenizer, text: str) -> Tuple[List[str], np.array]:
    """Returns the correspondance between tokens and their words. Note that we
    define a word as any sequence of characters that is surrounded by whitespace
    on either side.

    Args:
        tokenizer: A HuggingFace PreTrainedTokenizer
        text: A str that we wish to tokenize. 
    
    Returns:
        A tuple where the first entry the list of words that we have extracted from
        the text. The second entry is an array that has the same length as the number
        of tokens in the provided text. Each value in the array correspond to the index
        of the word it represents.

    
    """
    words = text.split()
    lc_words = text.lower().split()
    toks = tokenizer.encode(text)
    indices = np.zeros(len(toks))
    word_counter = 0
    decode_accum = ""

    for i in range(len(toks)):
        accum_text = " ".join(lc_words[:word_counter+1])
        indices[i] = word_counter
        
        # check if we are working with WordPiece tokenizer
        if isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):
            # if re.search(r"##[a-z]+", tokenizer.decode(toks[i])):
            #     continue
            # # WordPiece removes all white space in front of non-alphanumeric symbols
            symb = tokenizer.decode(toks[i])
            if re.search(r"##[a-z]+", symb):
                decode_accum += symb[2:]
            elif not any(c.isalnum() for c in symb):
                if words[min(word_counter+1, len(words)-1)] == symb:
                    decode_accum += f" {symb}"
                else:
                    decode_accum += symb
            else:
                decode_accum += f" {symb}"  
        else:
            decode_accum = tokenizer.decode(toks[:i+1]).lower()

        if accum_text in decode_accum and word_counter < len(lc_words) - 1:
            word_counter += 1
    return words, indices
    
