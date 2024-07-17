from typing import Union, List, Optional, Callable

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
    HookedEncoder,
    HookedEncoderDecoder,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
from rich.progress import track


torch.set_grad_enabled(False)


class BrainAlignTransformer:
    """A wrapper for the `HookedTransformer`, `HookedEncoder`, nad `HookedEncoderDecoder`
    from the `TransformerLens <https://transformerlensorg.github.io/TransformerLens/>`_ package.
    This implements many features that will be useful for computing brain-alignment score on the fly.
    """

    def __init__(
        self,
        ht: Union[HookedTransformer, HookedEncoder, HookedEncoderDecoder],
        encoder_decoder=False,
    ):
        """Instantiates a pre-trained `HookedTransformer` model. This constructors should
        almost never be called directly. Instead, one should use the factory method
        `BrainAlignTransformer.from_pretrained`.

        Args:
            ht: A hooked model could be an encoder, decoder, or encoder-decoder architecture.
        """
        self.ht = ht
        self.encoder_decoder = encoder_decoder

    def __str__(self):
        return self.ht.__str__()

    def run_with_cache(
        self,
        tokens: torch.LongTensor,
        decoder_input: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        if self.encoder_decoder:
            return self.ht.run_with_cache(tokens, decoder_input, **kwargs)
        return self.ht.run_with_cache(tokens, **kwargs)

    def resid_post(
        self,
        tokens: torch.LongTensor,
        decoder_input: Optional[torch.LongTensor] = None,
        batch_size: Optional[int] = None,
        rpros: Optional[Callable] = None,
    ):
        """Gets the hidden representations of `tokens` in the model at every layer. For
        a transformer model, we define each layer as after the residuals have been added
        to the MLP outputs.

        Args:
            tokens: Indices of the input sequence tokens in the vocabulary. It is assumed
                that this will have dimension :math:`\text{batch}\times\text{num_tokens}`.
            decoder_inputs: Indices of the decoder sequence tokens. This is required if the model
                uses an encoder-decoder architecture (T5).
            batch_size: The number of examples per-batch during inference
            rpros: A function that takes in the hidden representations from all of the layers
                and performs post-processing.
        Return:
            A tuple where the first entry is the logits of the model and the second entry is
            a torch tensor that has dimension ``layers x [num_examples, d_model]`` where
            ``d_model`` is the dimension of the embeddings. If ``rpros`` is specified, then
            this is called on all of the hidden layers (by-batch) before being returned.
        """
        resid_name_filter = lambda name: name.endswith("hook_resid_post")
        if rpros is None:
            rpros = lambda x: x  # identity function
        if batch_size is None:
            logits, ac = self.run_with_cache(
                tokens, decoder_input, names_filter=resid_name_filter
            )
            return logits, rpros(ac.values())
        tok_batch = tokens.chunk(len(tokens) // batch_size)
        reprs, logits, layers = [], [], 0
        for toks in track(tok_batch):
            l, c = self.run_with_cache(
                toks, decoder_input, names_filter=resid_name_filter
            )
            layers = len(c)
            logits.append(l)
            reprs.append(rpros(c))
        return torch.cat(logits), [
            torch.cat([b[i] for b in reprs]) for i in range(layers)
        ]
        # if normalize:
        #     bf = reprs.transpose(1, 0)
        #     bf = (bf - torch.mean(bf, axis=0)) / torch.std(bf, axis=0)
        #     return bf.transpose(1, 0)

    # def run_with_cache(self, tokens, batch_size=8) -> ActivationCache:
    #     """A wrapper for `HookedTransformer.run_with_cache` that implements
    #     batching for larger models.
    #     """
    #     tok_batch = tokens.chunk(len(tokens) // batch_size)

    #     logits = []
    #     caches = []

    #     for toks in track(tok_batch, description="Infer w/ cache..."):
    #         l, c = self.ht.run_with_cache(toks)
    #         l, c = l.to("cpu"), c.to("cpu")  # free gpu memory
    #         logits.append(l)
    #         caches.append(c)

    #     agg_logits = torch.cat(logits, dim=0)
    #     return agg_logits, self.concat_activation_cache(caches)

    # def resid_post(
    #     self,
    #     cache: ActivationCache,
    #     avg: bool = True,
    #     chunk: bool = False,
    #     chunk_size: int = 4,
    #     apply_ln: bool = True,
    # ):
    #     """Calculates the model representations at every layer after the residual stream
    #     has been added."""
    #     if avg and chunk:
    #         raise ValueError(
    #             "Pooling schemes average and chunking cannot be used at the same time!"
    #         )
    #     accum_resid = cache.accumulated_resid(
    #         apply_ln=apply_ln
    #     )  # layer, batch, pos, d_model

    #     if avg:
    #         return torch.mean(accum_resid, axis=2)
    #     if chunk:
    #         pos_chunked = accum_resid.chunk(chunk_size, axis=2)
    #         return torch.cat([torch.mean(c, axis=2) for c in pos_chunked], dim=-1)
    #     return accum_resid

    # @staticmethod
    # def split_activation_cache(acache: ActivationCache, batch_idxs) -> ActivationCache:
    #     caches = []
    #     for batch in batch_idxs:
    #         caches.append(
    #             ActivationCache(
    #                 {k: acache[k][batch] for k in acache.keys()}, acache.model
    #             )
    #         )
    #     return caches

    # @staticmethod
    # def concat_activation_cache(acaches: List[ActivationCache]) -> ActivationCache:
    #     model = acaches[0].model
    #     agg_cache_dict = {}
    #     for k in acaches[0].keys():
    #         agg_cache_dict[k] = torch.cat([c[k] for c in acaches], dim=0)
    #     return ActivationCache(agg_cache_dict, model)
