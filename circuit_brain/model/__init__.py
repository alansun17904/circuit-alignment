from . import patching
from .hooked_model import BrainAlignTransformer


from transformer_lens import HookedEncoder, HookedEncoderDecoder, HookedTransformer


def from_pretrained(hf_id):
    if "bert" in hf_id:
        ht = HookedEncoder.from_pretrained(hf_id)
    elif "t5" in hf_id:
        ht = HookedEncoderDecoder.from_pretrained(hf_id)
        return BrainAlignTransformer(ht, True)
    else:
        ht = HookedTransformer.from_pretrained(hf_id)
    return BrainAlignTransformer(ht)


BrainAlignTransformer.from_pretrained = from_pretrained
