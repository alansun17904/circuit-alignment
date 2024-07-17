import pickle
import argparse

import circuit_brain.utils as utils
from circuit_brain.model import BrainAlignedLMModel
from circuit_brain.dproc import fMRIDataset

import torch
from transformers import AutoTokenizer


torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str)
parser.add_argument("tok_name", type=str, help="id of tokenizer")
parser.add_argument("batch_size", type=int)
parser.add_argument("window_size", type=int)
parser.add_argument("ofname", type=str, help="name of output file")
parser.add_argument("--rfc", default=False, action="store_true", help="remove formatting characters")
parser.add_argument("--rps", default=False, action="store_true", help="remove punctuation spacing")
opts = parser.parse_args()


if __name__ == "__main__":
    m = BrainAlignedLMModel(opts.model_name)
    atok = AutoTokenizer.from_pretrained(opts.tok_name)
    hp = fMRIDataset.get_dataset(
        "hp",
        "data/HP_data",
        context_size=512,
        window_size=opts.window_size,
        tokenizer=atok,
        remove_format_chars=opts.rfc,
        remove_punc_spacing=opts.rps,
        pool_rois=True
    )

    model_repr = utils.per_subject_model_repr(
        hp.fmri_contexts, m, opts.batch_size
    )
    # compute brain-alignment scores
    ridge_cv = utils.RidgeCV(n_splits=5)
    pickle.dump(
        utils.across_subject_alignment(hp, model_repr, 5, 10, ridge_cv, pca=None),
        open(f"data/base_align_data/{opts.ofname}", "wb+"),
    )
