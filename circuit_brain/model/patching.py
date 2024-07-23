from __future__ import annotations
import itertools
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, overload

import einops
import pandas as pd
import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from typing_extensions import Literal

import transformer_lens.utils as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer


# %%
Logits = torch.Tensor
AxisNames = Literal["layer", "pos", "head_index", "head", "src_pos", "dest_pos"]


# %%
from typing import Sequence


def make_df_from_ranges(
    column_max_ranges: Sequence[int], column_names: Sequence[str]
) -> pd.DataFrame:
    """
    Takes in a list of column names and max ranges for each column, and returns a dataframe with the cartesian product of the range for each column (ie iterating through all combinations from zero to column_max_range - 1, in order, incrementing the final column first)
    """
    rows = list(
        itertools.product(
            *[range(axis_max_range) for axis_max_range in column_max_ranges]
        )
    )
    df = pd.DataFrame(rows, columns=column_names)
    return df


# %%
CorruptedActivation = torch.Tensor
PatchedActivation = torch.Tensor


@overload
def generic_activation_patch(
    model: HookedTransformer,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable[
        [Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]
    ],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: Literal[False] = False,
    batch_size: int = 64,
) -> torch.Tensor: ...


@overload
def generic_activation_patch(
    model: HookedTransformer,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable[
        [Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]
    ],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]],
    index_df: Optional[pd.DataFrame],
    return_index_df: Literal[True],
    batch_size: int = 64,
) -> Tuple[torch.Tensor, pd.DataFrame]: ...


def generic_activation_patch(
    wrapper: HookedTransformer,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable[
        [Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]
    ],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: bool = False,
    batch_size: int = 128,
) -> Union[torch.Tensor, Tuple[torch.Tensor, pd.DataFrame]]:
    """
    A generic function to do activation patching, will be specialised to specific use cases.

    Activation patching is about studying the counterfactual effect of a specific activation between a clean run and a corrupted run. The idea is have two inputs, clean and corrupted, which have two different outputs, and differ in some key detail. Eg "The Eiffel Tower is in" vs "The Colosseum is in". Then to take a cached set of activations from the "clean" run, and a set of corrupted.

    Internally, the key function comes from three things: A list of tuples of indices (eg (layer, position, head_index)), a index_to_act_name function which identifies the right activation for each index, a patch_setter function which takes the corrupted activation, the index and the clean cache, and a metric for how well the patched model has recovered.

    The indices can either be given explicitly as a pandas dataframe, or by listing the relevant axis names and having them inferred from the tokens and the model config. It is assumed that the first column is always layer.

    This function then iterates over every tuple of indices, does the relevant patch, and stores it

    Args:
        model: The relevant model
        corrupted_tokens: The input tokens for the corrupted run
        clean_cache: The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)
        patch_setter: A function which acts on (corrupted_activation, index, clean_cache) to edit the activation and patch in the relevant chunk of the clean activation
        activation_name: The name of the activation being patched
        index_axis_names: The names of the axes to (fully) iterate over, implicitly fills in index_df
        index_df: The dataframe of indices, columns are axis names and each row is a tuple of indices. Will be inferred from index_axis_names if not given. When this is input, the output will be a flattened tensor with an element per row of index_df
        return_index_df: A Boolean flag for whether to return the dataframe of indices too

    Returns:
        patched_output: The tensor of the patching metric for each patch. By default it has one dimension for each index dimension, via index_df set explicitly it is flattened with one element per row.
        index_df *optional*: The dataframe of indices
    """

    print("batch-size", batch_size)
    n_examples = corrupted_tokens.shape[0]
    batch_idxs = torch.arange(n_examples).chunk(n_examples // batch_size)

    if index_df is None:
        assert index_axis_names is not None

        # Get the max range for all possible axes
        max_axis_range = {
            "layer": wrapper.ht.cfg.n_layers,
            "pos": corrupted_tokens.shape[-1],
            "head_index": wrapper.ht.cfg.n_heads,
        }
        max_axis_range["src_pos"] = max_axis_range["pos"]
        max_axis_range["dest_pos"] = max_axis_range["pos"]
        max_axis_range["head"] = max_axis_range["head_index"]

        # Get the max range for each axis we iterate over
        index_axis_max_range = [
            max_axis_range[axis_name] for axis_name in index_axis_names
        ]

        # Get the dataframe where each row is a tuple of indices
        index_df = make_df_from_ranges(index_axis_max_range, index_axis_names)

        flattened_output = False
    else:
        # A dataframe of indices was provided. Verify that we did not *also* receive index_axis_names
        assert index_axis_names is None
        index_axis_max_range = index_df.max().to_list()

        flattened_output = True

    # Create an empty tensor to show the patched metric for each patch
    if flattened_output:
        patched_metric_output = torch.zeros(len(index_df), device=wrapper.ht.cfg.device)
    else:
        patched_metric_output = torch.zeros(
            index_axis_max_range, device=wrapper.ht.cfg.device
        )

    # A generic patching hook - for each index, it applies the patch_setter appropriately to patch the activation
    def patching_hook(corrupted_activation, hook, index, clean_activation):
        return patch_setter(corrupted_activation, index, clean_activation)

    # split the clean activation cache and the corrupted tokens in batches
    clean_caches = wrapper.split_activation_cache(clean_cache, batch_idxs)
    ct = [corrupted_tokens[v] for v in batch_idxs]

    # Iterate over every list of indices, and make the appropriate patch!
    for c, index_row in enumerate(tqdm((list(index_df.iterrows())))):
        caches = []
        for batch in tqdm(range(len(batch_idxs))):
            index = index_row[1].to_list()
            # The current activation name is just the activation name plus the layer
            # (assumed to be the first element of the input)
            current_activation_name = utils.get_act_name(
                activation_name, layer=index[0]
            )

            # The hook function cannot receive additional inputs, so we use partial to
            # include the specific index and the corresponding clean activation
            current_hook = partial(
                patching_hook,
                index=index,
                clean_activation=clean_caches[batch][current_activation_name],
            )

            # Run the model with the patching hook and get the logits!
            with wrapper.ht.hooks(fwd_hooks=[(current_activation_name, current_hook)]):
                patched_logits, patched_cache = wrapper.ht.run_with_cache(ct[batch])
                caches.append(patched_cache.to("cpu"))
                del patched_cache

            joined_patched_cache = wrapper.concat_activation_cache(caches)

        # Calculate the patching metric and store
        if flattened_output:
            patched_metric_output[c] = patching_metric(
                patched_logits, cache=joined_patched_cache
            )
        else:
            patched_metric_output[tuple(index)] = patching_metric(
                patched_logits, cache=joined_patched_cache
            )

    if return_index_df:
        return patched_metric_output, index_df
    else:
        return patched_metric_output


# %%
# Defining patch setters for various shapes of activations
def layer_pos_patch_setter(corrupted_activation, index, clean_activation):
    """
    Applies the activation patch where index = [layer, pos]

    Implicitly assumes that the activation axis order is [batch, pos, ...], which is true of everything that is not an attention pattern shaped tensor.
    """
    assert len(index) == 2
    layer, pos = index
    corrupted_activation[:, pos, ...] = clean_activation[:, pos, ...]
    return corrupted_activation


def layer_pos_head_vector_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer, pos, head_index]

    Implicitly assumes that the activation axis order is [batch, pos, head_index, ...], which is true of all attention head vector activations (q, k, v, z, result) but *not* of attention patterns.
    """
    assert len(index) == 3
    layer, pos, head_index = index
    corrupted_activation[:, pos, head_index] = clean_activation[:, pos, head_index]
    return corrupted_activation


def layer_head_vector_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index]

    Implicitly assumes that the activation axis order is [batch, pos, head_index, ...], which is true of all attention head vector activations (q, k, v, z, result) but *not* of attention patterns.
    """
    assert len(index) == 2
    layer, head_index = index
    corrupted_activation[:, :, head_index] = clean_activation[:, :, head_index]

    return corrupted_activation


def layer_head_pattern_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index]

    Implicitly assumes that the activation axis order is [batch, head_index, dest_pos, src_pos], which is true of attention scores and patterns.
    """
    assert len(index) == 2
    layer, head_index = index
    corrupted_activation[:, head_index, :, :] = clean_activation[:, head_index, :, :]

    return corrupted_activation


def layer_head_pos_pattern_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index, dest_pos]

    Implicitly assumes that the activation axis order is [batch, head_index, dest_pos, src_pos], which is true of attention scores and patterns.
    """
    assert len(index) == 3
    layer, head_index, dest_pos = index
    corrupted_activation[:, head_index, dest_pos, :] = clean_activation[
        :, head_index, dest_pos, :
    ]

    return corrupted_activation


def layer_head_dest_src_pos_pattern_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index, dest_pos, src_pos]

    Implicitly assumes that the activation axis order is [batch, head_index, dest_pos, src_pos], which is true of attention scores and patterns.
    """
    assert len(index) == 4
    layer, head_index, dest_pos, src_pos = index
    corrupted_activation[:, head_index, dest_pos, src_pos] = clean_activation[
        :, head_index, dest_pos, src_pos
    ]

    return corrupted_activation


# %%
# Defining activation patching functions for a range of common activation patches.
resid_pre = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="resid_pre",
    index_axis_names=("layer", "pos"),
)
resid_pre.__doc__ = """
    Function to get activation patching results for the residual stream (at the start of each block) (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each resid_pre patch. Has shape [n_layers, pos]
    """

resid_mid = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="resid_mid",
    index_axis_names=("layer", "pos"),
)
resid_mid.__doc__ = """
    Function to get activation patching results for the residual stream (between the attn and MLP layer of each block) (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos]
    """

attn_out = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="attn_out",
    index_axis_names=("layer", "pos"),
)
attn_out.__doc__ = """
    Function to get activation patching results for the output of each Attention layer (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos]
    """

mlp_out = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="mlp_out",
    index_axis_names=("layer", "pos"),
)
mlp_out.__doc__ = """
    Function to get activation patching results for the output of each MLP layer (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos]
    """
# %%
attn_head_out_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "pos", "head"),
)
attn_head_out_by_pos.__doc__ = """
    Function to get activation patching results for the output of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads]
    """

attn_head_q_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="q",
    index_axis_names=("layer", "pos", "head"),
)
attn_head_q_by_pos.__doc__ = """
    Function to get activation patching results for the queries of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads]
    """

attn_head_k_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="k",
    index_axis_names=("layer", "pos", "head"),
)
attn_head_k_by_pos.__doc__ = """
    Function to get activation patching results for the keys of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads]
    """

attn_head_v_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="v",
    index_axis_names=("layer", "pos", "head"),
)
attn_head_v_by_pos.__doc__ = """
    Function to get activation patching results for the values of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads]
    """
# %%
attn_head_pattern_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_pos_pattern_patch_setter,
    activation_name="pattern",
    index_axis_names=("layer", "head_index", "dest_pos"),
)
attn_head_pattern_by_pos.__doc__ = """
    Function to get activation patching results for the attention pattern of each Attention Head (by destination position). Returns a tensor of shape [n_layers, n_heads, dest_pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads, dest_pos]
    """

attn_head_pattern_dest_src_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_dest_src_pos_pattern_patch_setter,
    activation_name="pattern",
    index_axis_names=("layer", "head_index", "dest_pos", "src_pos"),
)
attn_head_pattern_dest_src_pos.__doc__ = """
    Function to get activation patching results for each destination, source entry of the attention pattern for each Attention Head. Returns a tensor of shape [n_layers, n_heads, dest_pos, src_pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads, dest_pos, src_pos]
    """

# %%
attn_head_out_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "head"),
)
attn_head_out_all_pos.__doc__ = """
    Function to get activation patching results for the outputs of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

attn_head_q_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="q",
    index_axis_names=("layer", "head"),
)
attn_head_q_all_pos.__doc__ = """
    Function to get activation patching results for the queries of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

attn_head_k_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="k",
    index_axis_names=("layer", "head"),
)
attn_head_k_all_pos.__doc__ = """
    Function to get activation patching results for the keys of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

attn_head_v_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="v",
    index_axis_names=("layer", "head"),
)
attn_head_v_all_pos.__doc__ = """
    Function to get activation patching results for the values of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

attn_head_pattern_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_pattern_patch_setter,
    activation_name="pattern",
    index_axis_names=("layer", "head_index"),
)
attn_head_pattern_all_pos.__doc__ = """
    Function to get activation patching results for the attention pattern of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

# %%


def attn_head_all_pos_every(
    model,
    corrupted_tokens,
    clean_cache,
    metric,
    batch_size=8,
) -> Float[torch.Tensor, "patch_type layer head"]:
    """Helper function to get activation patching results for every head (across all positions) for every act type (output, query, key, value, pattern). Wrapper around each's patching function, returns a stacked tensor of shape [5, n_layers, n_heads]

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [5, n_layers, n_heads]
    """
    act_patch_results: list[torch.Tensor] = []
    act_patch_results.append(
        attn_head_out_all_pos(
            model, corrupted_tokens, clean_cache, metric
        )
    )
    act_patch_results.append(
        attn_head_q_all_pos(
            model, corrupted_tokens, clean_cache, metric, batch_size=batch_size
        )
    )
    act_patch_results.append(
        attn_head_k_all_pos(
            model, corrupted_tokens, clean_cache, metric, batch_size=batch_size
        )
    )
    act_patch_results.append(
        attn_head_v_all_pos(
            model, corrupted_tokens, clean_cache, metric, batch_size=batch_size
        )
    )
    act_patch_results.append(
        attn_head_pattern_all_pos(
            model, corrupted_tokens, clean_cache, metric, batch_size=batch_size
        )
    )
    return torch.stack(act_patch_results, dim=0)


def attn_head_by_pos_every(
    model, corrupted_tokens, clean_cache, metric, batch_size
) -> Float[torch.Tensor, "patch_type layer pos head"]:
    """Helper function to get activation patching results for every head (by position) for every act type (output, query, key, value, pattern). Wrapper around each's patching function, returns a stacked tensor of shape [5, n_layers, pos, n_heads]

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [5, n_layers, pos, n_heads]
    """
    act_patch_results = []
    act_patch_results.append(
        attn_head_out_by_pos(
            model, corrupted_tokens, clean_cache, metric, batch_size=batch_size
        )
    )
    act_patch_results.append(
        attn_head_q_by_pos(
            model, corrupted_tokens, clean_cache, metric, batch_size=batch_size
        )
    )
    act_patch_results.append(
        attn_head_k_by_pos(
            model, corrupted_tokens, clean_cache, metric, batch_size=batch_size
        )
    )
    act_patch_results.append(
        attn_head_v_by_pos(
            model, corrupted_tokens, clean_cache, metric, batch_size=batch_size
        )
    )

    # Reshape pattern to be compatible with the rest of the results
    pattern_results = attn_head_pattern_by_pos(
        model, corrupted_tokens, clean_cache, metric, batch_size
    )
    act_patch_results.append(
        einops.rearrange(pattern_results, "batch head pos -> batch pos head")
    )
    return torch.stack(act_patch_results, dim=0)


def block_every(
    model, corrupted_tokens, clean_cache, metric, batch_size
) -> Float[torch.Tensor, "patch_type layer pos"]:
    """Helper function to get activation patching results for the residual stream (at the start of each block), output of each Attention layer and output of each MLP layer. Wrapper around each's patching function, returns a stacked tensor of shape [3, n_layers, pos]

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [3, n_layers, pos]
    """
    act_patch_results = []
    act_patch_results.append(
        resid_pre(
            model, corrupted_tokens, clean_cache, metric, batch_size=batch_size
        )
    )
    act_patch_results.append(
        attn_out(
            model, corrupted_tokens, clean_cache, metric, batch_size=batch_size
        )
    )
    act_patch_results.append(
        mlp_out(
            model, corrupted_tokens, clean_cache, metric, batch_size=batch_size
        )
    )
    return torch.stack(act_patch_results, dim=0)
