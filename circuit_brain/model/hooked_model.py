from typing import Union, List, Optional

import torch
import numpy as np

import transformer_lens as tl
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
from rich.progress import track


torch.set_grad_enabled(False)


class BrainAlignedLMModel:
    """A wrapper for the `HookedTransformer` from `transformer_lens`. This
    implements many useful features that will be useful for computing brain-alignment
    on the fly.
    """

    def __init__(self, hf_model_id):
        """Instantiates a pre-trained `HookedTransformer` model.

        Args:
            hf_model_id: A hugging face repository identifier.
        """
        self.ht = HookedTransformer.from_pretrained(hf_model_id)

    def to_tokens(self, sentences: List[str]) -> List[List[int]]:
        """Converts a list of sentences or text into their model-specific tokens.

        Args:
            sentences: A list of strings that represent separate input prompts.

        Returns:
            A list of list of integers that correspond to the token index of
            each provided sentence.
        """
        return self.ht.to_tokens(sentences)

    def to_string(self, tokens: List[List[int]]) -> List[str]:
        """Converts the model-specific token representation of some text back
        into its textual form.

        Args:
            tokens: A list of list of integers that correspond to the token index of each
            sentence that we wish to convert back into text.

        Returns:
            A list of strings that correspond to the text of each list of input tokens.
        """
        return self.ht.to_string(tokens)

    def hidden_repr(
        self,
        tokens: List[List[int]],
        batch_size: int = 8,
        normalize: bool = True,
        avg: bool = True,
        chunk: bool = False,
        chunk_size: int = 4,
    ):
        """Gets the hidden representations of `tokens` in the model at every layer. For
        a transformer model, we define each layer as after the residuals have been added
        to the MLP outputs.

        Args:
            tokens: A list of list of model-specific tokens that correspond to some
                set of input prompts. We recommend generating this using
                `BrainAlignedLMModel.to_tokens`.
            batch_size: The number of examples per-batch during inference
            normalize: If this is true, then the hidden representations are normalized
                across the dimension of all input prompts.
            avg: If true, the embeddings are averaged across the token dimension for each
                input. It should be noted that both `avg` and `chunk` cannot be true at the
                same time since they are competing pooling strategies.
            chunk: If true, chunks the token embeddings into `chunk_size` first. That is,
                every `chunk_size` number of tokens are concatenated together then averaged
                across all chunks.
        Return:
            A torch tensor that has dimension `[layers, num_examples, d_model]` where
            `d_model` is the dimension of the embeddings. If `chunk=True`, then we
            should expect an output of dimension `[layers, num_examples, d_model * chunk_size]`.
        """
        tok_batch = tokens.chunk(len(tokens) // batch_size)
        reprs = []
        for toks in track(tok_batch, description="Infer w/ cache..."):
            l, c = self.ht.run_with_cache(toks)
            reprs.append(self.resid_post(c))
            del c
        reprs = torch.cat(reprs, dim=1)
        if normalize:
            bf = reprs.transpose(1, 0)
            bf = (bf - torch.mean(bf, axis=0)) / torch.std(bf, axis=0)
            return bf.transpose(1, 0)
        return reprs

    def run_with_cache(self, tokens, batch_size=8) -> ActivationCache:
        """A wrapper for `HookedTransformer.run_with_cache` that implements
        batching for larger models.
        """
        tok_batch = tokens.chunk(len(tokens) // batch_size)

        logits = []
        caches = []

        for toks in track(tok_batch, description="Infer w/ cache..."):
            l, c = self.ht.run_with_cache(toks)
            l, c = l.to("cpu"), c.to("cpu")  # free gpu memory
            logits.append(l)
            caches.append(c)

        agg_logits = torch.cat(logits, dim=0)
        return agg_logits, self.concat_activation_cache(caches)

    def resid_post(
        self,
        cache: ActivationCache,
        avg: bool = True,
        chunk: bool = False,
        chunk_size: int = 4,
        apply_ln: bool = True,
    ):
        """Calculates the model representations at every layer after the residual stream
        has been added."""
        if avg and chunk:
            raise ValueError(
                "Pooling schemes average and chunking cannot be used at the same time!"
            )
        accum_resid = cache.accumulated_resid(
            apply_ln=apply_ln
        )  # layer, batch, pos, d_model

        if avg:
            return torch.mean(accum_resid, axis=2)
        if chunk:
            pos_chunked = accum_resid.chunk(chunk_size, axis=2)
            return torch.cat([torch.mean(c, axis=2) for c in pos_chunked], dim=-1)
        return accum_resid

    @staticmethod
    def split_activation_cache(acache: ActivationCache, batch_idxs) -> ActivationCache:
        caches = []
        for batch in batch_idxs:
            caches.append(
                ActivationCache(
                    {k: acache[k][batch] for k in acache.keys()}, acache.model
                )
            )
        return caches

    @staticmethod
    def concat_activation_cache(acaches: List[ActivationCache]) -> ActivationCache:
        model = acaches[0].model
        agg_cache_dict = {}
        for k in acaches[0].keys():
            agg_cache_dict[k] = torch.cat([c[k] for c in acaches], dim=0)
        return ActivationCache(agg_cache_dict, model)
