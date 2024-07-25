from typing import Union, List, Optional, Callable, Literal, Tuple
from jaxtyping import Int, Float
import tqdm
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
from torch.utils.data import DataLoader, Dataset
from transformer_lens.utilities import devices
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache


torch.set_grad_enabled(False)


class Toks(Dataset):
    def __init__(self, toks):
        self.toks = toks
    
    def __len__(self):
        return len(self.toks)
    
    def __getitem__(self, idx):
        return self.toks[idx]



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

    @torch.inference_mode
    def resid_post(
        self,
        tokens: torch.LongTensor,
        decoder_input: Optional[torch.LongTensor] = None,
        batch_size: Optional[int] = None,
        rpre: Optional[Callable] = lambda x: x,
        rpost: Optional[Callable] = lambda x: x,
        verbose=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the hidden representations of `tokens` in the model at every layer. For
        a transformer model, we define each layer as after the residuals have been added
        to the MLP outputs.

        Args:
            tokens: Indices of the input sequence tokens in the vocabulary. It is assumed
                that this will have dimension :math:`\text{batch}\times\text{num_tokens}`.
            decoder_inputs: Indices of the decoder sequence tokens. This is required if the model
                uses an encoder-decoder architecture (T5).
            batch_size: The number of examples per-batch during inference
            rpre: A function that takes in the hidden representations from all of the layers
                and performs post-processing after each batch.
            rpost: A function that takes in the hidden representations from all of the layers
                and performs post-processing after all batches.
        Return:
            A tuple where the first entry is the logits of the model and the second entry is
            a torch tensor that has dimension ``layers x [num_examples, d_model]`` where
            ``d_model`` is the dimension of the embeddings. If ``rpre`` or ``rpost`` is specified, then
            this is called on all of the hidden layers (by-batch) before being returned.
        """
        resid_name_filter = lambda name: name.endswith("hook_resid_post")
        if batch_size is None:
            _, ac = self.run_with_cache(
                tokens, decoder_input, names_filter=resid_name_filter
            )
            return [], [v.to("cpu") for v in rpost(rpre(list(c.values())))]
        dl = DataLoader(Toks(tokens), batch_size=batch_size, shuffle=False, pin_memory=True)
        reprs, logits = [], []
        for toks in tqdm.tqdm(dl, disable=not verbose):
            _, c = self.run_with_cache(
                toks, decoder_input, names_filter=resid_name_filter
            )
            reprs.append([v.to("cpu") for v in rpre(list(c.values()))])
        return logits, rpost(
            [torch.cat([b[i] for b in reprs]) for i in range(self.ht.cfg.n_layers)]
        )

    @torch.inference_mode
    def generate(
        self,
        input: Union[str, Float[torch.Tensor, "batch pos"]] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[Literal["left", "right"]] = None,
        return_type: Optional[str] = "input",
    ) -> Union[Int[torch.Tensor, "batch pos_plus_new_tokens"], str]:
        """Sample Tokens from the Model.

        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.

        To avoid fiddling with ragged tensors, if we input a batch of text and some sequences finish
        (by producing an EOT token), we keep running the model on the entire batch, but throw away
        the output for a finished sequence and just keep adding EOTs to pad.

        This supports entering a single string, but not a list of strings - if the strings don't
        tokenize to exactly the same length, this gets messy. If that functionality is needed,
        convert them to a batch of tokens and input that instead.

        Args:
            input (Union[str, Int[torch.Tensor, "batch pos"])]): Either a batch of tokens ([batch,
                pos]) or a text string (this will be converted to a batch of tokens with batch size
                1).
            max_new_tokens (int): Maximum number of tokens to generate.
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token.
            eos_token_id (Optional[Union[int, Sequence]]): The token ID to use for end
                of sentence. If None, use the tokenizer's eos_token_id - required if using
                stop_at_eos. It's also possible to provide a list of token IDs (not just the
                eos_token_id), in which case the generation will stop when any of them are output
                (useful e.g. for stable_lm).
            do_sample (bool): If True, sample from the model's output distribution. Otherwise, use
                greedy search (take the max logit each time).
            top_k (int): Number of tokens to sample from. If None, sample from all tokens.
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens. If <1.0,
                we take the top tokens with cumulative probability >= top_p.
            temperature (float): Temperature for sampling. Higher values will make the model more
                random (limit of temp -> 0 is just taking the top token, limit of temp -> inf is
                sampling from a uniform distribution).
            freq_penalty (float): Frequency penalty for sampling - how much to penalise previous
                tokens. Higher values will make the model more random.
            use_past_kv_cache (bool): If True, create and use cache to speed up generation.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (applicable when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos (default is True unless specified
                otherwise). Pass True or False to override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.
            return_type (Optional[str]): The type of the output to return - either a string (str),
                a tensor of tokens (tensor) or whatever the format of the input was (input).
        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens
                (by default returns same type as input). Also, returns the logits corresponding to each
                step of generation.
        """

        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            if type(input) == str:
                # If text, convert to tokens (batch_size=1)
                assert (
                    self.ht.tokenizer is not None
                ), "Must provide a tokenizer if passing a string to the model"
                tokens = self.ht.to_tokens(
                    input, prepend_bos=prepend_bos, padding_side=padding_side
                )
            else:
                tokens = input

            if return_type == "input":
                if type(input) == str:
                    return_type = "str"
                else:
                    return_type = "tensor"

            assert isinstance(tokens, torch.Tensor)
            batch_size, ctx_length = tokens.shape
            device = devices.get_device_for_block_index(0, self.ht.cfg)
            tokens = tokens.to(device)
            if use_past_kv_cache:
                past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                    self.ht.cfg, self.ht.cfg.device, batch_size
                )
            else:
                past_kv_cache = None

            stop_tokens: List[int] = []
            eos_token_for_padding = 0
            assert self.ht.tokenizer is not None
            if stop_at_eos:
                tokenizer_has_eos_token = (
                    self.ht.tokenizer is not None
                    and self.ht.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert (
                        tokenizer_has_eos_token
                    ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

                    eos_token_id = self.ht.tokenizer.eos_token_id

                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    # eos_token_id is a Sequence (e.g. list or tuple)
                    stop_tokens = eos_token_id
                    eos_token_for_padding = (
                        self.ht.tokenizer.eos_token_id
                        if tokenizer_has_eos_token
                        else eos_token_id[0]
                    )

            # An array to track which sequences in the batch have finished.
            finished_sequences = torch.zeros(
                batch_size, dtype=torch.bool, device=self.ht.cfg.device
            )
            aggregated_logits = torch.zeros(
                (batch_size, ctx_length + max_new_tokens, self.ht.cfg.d_vocab),
                device=self.ht.cfg.device,
            )

            # Currently nothing in HookedTransformer changes with eval, but this is here in case
            # that changes in the future.
            self.ht.eval()
            for index in range(max_new_tokens):
                # While generating, we keep generating logits, throw away all but the final logits,
                # and then use those logits to sample from the distribution We keep adding the
                # sampled tokens to the end of tokens.
                if use_past_kv_cache:
                    # We just take the final tokens, as a [batch, 1] tensor
                    if index > 0:
                        logits = self.ht.forward(
                            tokens[:, -1:],
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                        )
                        aggregated_logits[:, ctx_length + index, :] = logits[:, -1, :]
                    else:
                        logits = self.ht.forward(
                            tokens,
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                        )
                        aggregated_logits[:, :ctx_length, :] = logits
                else:
                    # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                    # the cache.
                    logits = self.ht.forward(
                        tokens,
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                    )
                final_logits = logits[:, -1, :]

                if do_sample:
                    sampled_tokens = utils.sample_logits(
                        final_logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        tokens=tokens,
                    ).to(devices.get_device_for_block_index(0, self.ht.cfg))
                else:
                    sampled_tokens = final_logits.argmax(-1).to(
                        devices.get_device_for_block_index(0, self.ht.cfg)
                    )

                if stop_at_eos:
                    # For all unfinished sequences, add on the next token. If a sequence was
                    # finished, throw away the generated token and add eos_token_for_padding
                    # instead.
                    sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(
                            sampled_tokens.to(self.ht.cfg.device),
                            torch.tensor(stop_tokens).to(self.ht.cfg.device),
                        )
                    )

                tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

                if stop_at_eos and finished_sequences.all():
                    break

            if return_type == "str":
                if self.ht.cfg.default_prepend_bos:
                    # If we prepended a BOS token, remove it when returning output.
                    return self.ht.tokenizer.decode(tokens[0, 1:]), logits
                else:
                    return self.ht.tokenizer.decode(tokens[0]), logits

            else:
                return tokens, logits
