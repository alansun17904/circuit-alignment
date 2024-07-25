import math
import pickle
import argparse
import itertools
import functools
from typing import List, Tuple, Callable, Literal, Optional

import circuit_brain.utils as utils
from circuit_brain.dproc import fMRIDataset
from circuit_brain.model import transforms
from circuit_brain.model import BrainAlignTransformer

import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformer_lens import ActivationCache
import transformer_lens.utils as tlutils


HeadIndex = Tuple[int, int]
Hook = Callable[ActivationCache, str]
HeadType = Literal["z", "q", "k", "v"]
PatchDir = Literal["noising", "denoising"]


torch.set_grad_enabled(False)

class Toks(Dataset):
    def __init__(self, src, des):
        self.src = src
        self.des = des
    
    def __len__(self):
        return self.src.shape[0]
    
    def __getitem__(self, idx):
        return self.src[idx], self.des[idx]


def make_attn_head_hooks(heads: List[HeadIndex], clean: ActivationCache, htype="z"):
    hooks = []
    for head in heads:
        act_name = tlutils.get_act_name(f"{htype}{head[0]}")
        hooks.append(
            (
                act_name,
                functools.partial(attn_head_setter, index=head, clean=clean[act_name]),
            )
        )
    return hooks


def attn_head_setter(corrupted, hook, index, clean):
    """
    Index needs to be in the form of (layer, head_index)
    """
    assert len(index) == 2
    _, head_index = index
    corrupted[:, :, head_index] = clean[:, :, head_index]
    return corrupted


def run_with_patch(
    m: BrainAlignTransformer,
    heads: List[HeadIndex],
    toks: Toks,
    batch_size: int,
    htype: HeadType = "z",
    logits_only:bool=True,
    clean_cache: Optional[ActivationCache] = None,
    rpre: Optional[Callable] = lambda x: x,
    rpost: Optional[Callable] = lambda x: x,
    cache: List=None
):
    # only cache the activations of the components that we are patching
    layers = m.ht.cfg.n_layers
    comp_names = {tlutils.get_act_name(f"{htype}{h[0]}") for h in heads}
    names_filter = lambda x: x in comp_names
    logits, reprs = [], []
    dl = DataLoader(toks, batch_size=batch_size)
    for i, (src, des) in enumerate(dl):
        if clean_cache is None:
            _, src_cache = m.ht.run_with_cache(src, names_filter=names_filter)
            if cache is not None:
                cache.append(src_cache.to("cpu"))
        else:
            src_cache = clean_cache[i]
        hooks = make_attn_head_hooks(heads, src_cache, htype=htype)
        with m.ht.hooks(fwd_hooks=hooks):
            #logits.append(m.ht(des)[:, -1, :].to("cpu"))
            if not logits_only:
                _, c = m.resid_post(des, rpre=rpre, verbose=False)
                reprs.append(c)
    #logits = torch.cat(logits)
    if logits_only:
        return logits
    return logits, rpost([torch.cat([b[i] for b in reprs]) for i in range(layers)])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("tok_name", type=str, help="id of tokenizer")
    parser.add_argument("ofname", type=str, help="name of output file")
    parser.add_argument("--batch_size", type=int, default=128)

    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    m = BrainAlignTransformer.from_pretrained(opts.model_name)
    atok = AutoTokenizer.from_pretrained(opts.tok_name)
    rpre = transforms.Compose(
        [
            transforms.ContextCrop(1),
            lambda x: [v.squeeze(1) for v in x],
        ]
    )
    hp = fMRIDataset.get_dataset(
        "hp",
        "data/HP_data",
        context_size=128,
        tokenizer=atok,
        remove_format_chars=True,
        remove_punc_spacing=True,
        pool_rois=True,
    )
    N_HEADS = m.ht.cfg.n_heads
    plato = torch.LongTensor(
        atok(" ".join(open("data/related_texts/the-republic.txt", "r").read().split()))[
            "input_ids"
        ]
    )
    plato = torch.stack(plato.chunk(math.ceil(len(plato) / 128))[:-1])
    ridge_cv = utils.RidgeCV(n_splits=5)
    heads = list(itertools.product(range(m.ht.cfg.n_layers), range(N_HEADS)))

    print(f"Patching {len(heads)} heads.")

    for sidx in range(len(hp)):
        head_scores = dict()
        for j, head in tqdm.tqdm(enumerate(heads), desc=f"Patching subj-{sidx+1}", total=len(heads)):
            plato_idxs = torch.randperm(len(plato))
            ds_generator = hp.kfold(sidx, 5, 10)
            fold_scores = torch.zeros(5, m.ht.cfg.n_layers)
            if j % N_HEADS == 0:
                train_caches, test_caches = [], []
            for k, test, train in ds_generator:
                tr_cache, tt_cache = [], []
                train_fmri, train_toks, _ = train
                test_fmri, test_toks, _ = test
                train_corr, test_corr = (
                    plato[plato_idxs[: len(train_fmri)]],
                    plato[plato_idxs[-len(test_fmri) :]],
                )
                train_dstok, test_dstok = Toks(train_toks, train_corr), Toks(test_toks, test_corr)
                _, train_reprs = run_with_patch(
                    m,
                    [head],
                    train_dstok,
                    opts.batch_size,
                    logits_only=False,
                    clean_cache=None if (j % N_HEADS == 0) else train_caches[k],
                    rpre=rpre,
                    cache=tr_cache,
                )
                _, test_reprs = run_with_patch(
                    m,
                    [head],
                    test_dstok,
                    opts.batch_size,
                    clean_cache=None if (j % N_HEADS == 0) else test_caches[k],
                    logits_only=False,
                    rpre=rpre,
                    cache=tt_cache
                )
                if j % N_HEADS == 0:
                    train_caches.append(tr_cache)
                    test_caches.append(tt_cache)
                l2 = torch.zeros(m.ht.cfg.n_layers)
                train_fmri = train_fmri.to("cuda")
                test_fmri = test_fmri.to("cuda")
                for l in range(head[0], m.ht.cfg.n_layers):
                    ridge_cv.fit(train_reprs[l].to("cuda"), train_fmri)
                    l2[l] = torch.mean(ridge_cv.score(test_reprs[l].to("cuda"), test_fmri)).item()
                fold_scores[k] = l2
            head_scores[f"{head[0]}.{head[1]}"] = torch.mean(fold_scores, dim=0)
        pickle.dump(head_scores, open(f"{opts.ofname}-subj{sidx}.pkl", "wb"))
