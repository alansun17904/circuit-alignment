import sys
import math
import pickle
import argparse

import circuit_brain.utils as utils
from circuit_brain.model import BrainAlignTransformer
from circuit_brain.dproc import fMRIDataset
from circuit_brain.model import transforms

import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str)
parser.add_argument("tok_name", type=str, help="id of tokenizer")
parser.add_argument("ofname", type=str, help="name of output file")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument(
    "--rfc", default=True, action="store_true", help="remove formatting characters"
)
parser.add_argument(
    "--rps", default=True, action="store_true", help="remove punctuation spacing"
)
parser.add_argument(
    "--text_type", choices=["norm", "plato", "randtok", "poa"], default="norm"
)
parser.add_argument(
    "--hf_model", default=False, action="store_true", help="huggingface gated model"
)
parser.add_argument(
    "--hf_model_id",
    type=str,
    help="huggingface id of model, only used if hf_model is true",
)
parser.add_argument(
    "--avg_subjs",
    default=False,
    action="store_true",
    help="average fmri recordings of all subjects",
)
opts = parser.parse_args()


if __name__ == "__main__":
    if not opts.hf_model:
        m = BrainAlignTransformer.from_pretrained(opts.model_name)
        atok = AutoTokenizer.from_pretrained(opts.tok_name)
    else:
        hf_m = AutoModelForCausalLM.from_pretrained(opts.hf_model_id)
        m = BrainAlignTransformer.from_pretrained(opts.model_name, hf_m=hf_m)
        atok = AutoTokenizer.from_pretrained(opts.hf_model_id)
    poa = np.load("data/HP_data/fMRI/prisoner-of-ask.npy")
    print("Initializing dataset...")
    hp = fMRIDataset.get_dataset(
        "hp",
        "data/HP_data",
        context_size=128,
        tokenizer=atok,
        remove_format_chars=opts.rfc,
        remove_punc_spacing=opts.rps,
        pool_rois=True,
        words=poa if opts.text_type == "poa" else None,
    )
    if opts.text_type == "plato":
        # get plato's republic instead and replace the contexts
        plato = torch.LongTensor(
            atok(
                " ".join(
                    open("data/related_texts/the-republic.txt", "r").read().split()
                )
            )["input_ids"]
        )
        plato = torch.stack(plato.chunk(math.ceil(len(plato) / 128))[:-1]).repeat(2, 1)
        hp.toks = plato
    if opts.text_type == "randtok":
        ran = torch.randint(0, m.ht.cfg.d_vocab, (len(hp.toks), 128))
        hp.toks = ran
    print("Dataset pre-processed!")
    ridge_cv = utils.RidgeCV(n_splits=5)
    gen_rpre = lambda w: transforms.Compose(
        [
            transforms.ContextCrop(1),
            lambda x: [v.squeeze(1) for v in x],
        ]
    )
    if opts.avg_subjs:
        gen_rpre = lambda w: transforms.Compose(
            [
                transforms.ContextCrop(45),
                transforms.Avg(),
            ]
        )
        print("Alignment by averaging all subjects together...")
        sscores, _, _ = utils.sub_align(
            m,
            hp,
            ridge_cv,
            None,
            opts.batch_size,
            1,
            10,
            gen_train_rpre=gen_rpre,
            gen_test_rpre=gen_rpre,
            train_reprs=list(),
            test_reprs=list(),
        )
        pickle.dump(
            sscores,
            open(f"{opts.ofname}", "wb+"),
        )
        sys.exit(0)
    subjs, train_reprs, test_reprs = [], [], []
    for sidx in tqdm.tqdm(range(len(hp))):
        sscores, train_reprs, test_reprs = utils.sub_align(
            m,
            hp,
            ridge_cv,
            sidx,
            opts.batch_size,
            5,
            5,
            gen_train_rpre=gen_rpre,
            gen_test_rpre=gen_rpre,
            train_reprs=train_reprs,
            test_reprs=test_reprs,
        )
        subjs.append(sscores)
    pickle.dump(
        subjs,
        open(f"{opts.ofname}", "wb+"),
    )
