import pickle
import argparse

import circuit_brain.utils as utils
from circuit_brain.model import BrainAlignedLMModel
from circuit_brain.dproc import fMRIDataset
from circuit_brain.model import transforms

import torch
from transformers import AutoTokenizer


torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str)
parser.add_argument("tok_name", type=str, help="id of tokenizer")
parser.add_argument("batch_size", type=int)
parser.add_argument("window_size", type=int)
parser.add_argument("ofname", type=str, help="name of output file")
parser.add_argument(
    "--rfc", default=False, action="store_true", help="remove formatting characters"
)
parser.add_argument(
    "--rps", default=False, action="store_true", help="remove punctuation spacing"
)
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
        pool_rois=True,
    )

    rpros = transforms.Compose([
        transforms.ContextCrop(25),
        transforms.Normalize(),
        transforms.ContextCrop(1),
    ])

    ridge_cv = utils.RidgeCV(n_splits=5)

    subjs = []
    for sidx in range(len(hp)):
        ds_generator = hp.kfold(sidx, 5, 5)
        layers = []
        for _, train, test in ds_generator:
            train_fmri, train_toks, train_idxmap = train
            test_fmri, test_toks, test_idxmap = test
            train_model_repr = m.resid_post(train_toks, batch_size=opts.batch_size, rpros=rpros)
            test_model_repr = m.resid_post(test_toks, batch_size=opts.batch_size, rpros=rpros)
            lr2 = torch.zeros(len(train_model_repr))
            for layer in train_model_repr:
                ridge_cv.fit(train_model_repr[layer], train_fmri)
                lr2[layer] = torch.mean(ridge_cv.score(test_model_repr[layer], test_fmri)).item()
            layers.append(lr2)
        subjs.append(torch.mean(torch.Tensor(layers), dim=1))
    print(subjs)

    # compute brain-alignment scores
    
    pickle.dump(
        subjs,
        open(f"data/base_align_data/{opts.ofname}", "wb+"),
    )
