import pickle
import argparse

import circuit_brain.utils as utils
from circuit_brain.model import BrainAlignTransformer
from circuit_brain.dproc import fMRIDataset
from circuit_brain.model import transforms

import tqdm
import torch
from transformers import AutoTokenizer


torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str)
parser.add_argument("tok_name", type=str, help="id of tokenizer")
parser.add_argument("ofname", type=str, help="name of output file")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--window_size", type=int, default=20)
parser.add_argument(
    "--rfc", default=True, action="store_true", help="remove formatting characters"
)
parser.add_argument(
    "--rps", default=True, action="store_true", help="remove punctuation spacing"
)
opts = parser.parse_args()


if __name__ == "__main__":
    m = BrainAlignTransformer.from_pretrained(opts.model_name)
    atok = AutoTokenizer.from_pretrained(opts.tok_name)
    print("Initializing dataset...")
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

    print("Dataset pre-processed!")
    ridge_cv = utils.RidgeCV(n_splits=5)

    subjs = []
    train_reprs = []
    test_reprs = []
    for sidx in tqdm.tqdm(range(len(hp))):
        ds_generator = hp.kfold(sidx, 5, 5)
        folds = None
        for k, test, train in ds_generator:
            train_fmri, train_toks, train_idxmap = train
            test_fmri, test_toks, test_idxmap = test
            pcas = [utils.PCA(100)] * m.ht.cfg.n_layers
            train_rpros = transforms.Compose([
                transforms.WordAvg(train_idxmap), 
                transforms.Concat(),
                transforms.PCAt(pcas),
                transforms.Normalize(),
            ])
            test_rpros = transforms.Compose([
                transforms.WordAvg(test_idxmap),
                transforms.Concat(),
                transforms.PCAt(pcas, fit=False),
                transforms.Normalize(),
            ])

            if sidx == 0:
                _, train_model_repr = m.resid_post(train_toks, batch_size=opts.batch_size, rpros=train_rpros)
                _, test_model_repr = m.resid_post(test_toks, batch_size=opts.batch_size, rpros=test_rpros)
                train_reprs.append(train_model_repr)
                test_reprs.append(test_model_repr)
            else:
                train_model_repr = train_reprs[k]
                test_model_repr = test_reprs[k]
            lr2 = torch.zeros(len(train_model_repr), test_fmri.shape[1])
            if k == 0:
                folds = torch.zeros(5, len(train_model_repr), test_fmri.shape[1])
            for layer in range(len(train_model_repr)):
                ridge_cv.fit(train_model_repr[layer], train_fmri)
                lr2[layer] = ridge_cv.score(test_model_repr[layer], test_fmri)
            folds[k] = lr2
        subjs.append(torch.mean(folds, dim=0))
    # compute brain-alignment scores
    pickle.dump(
        subjs,
        open(f"{opts.ofname}", "wb+"),
    )
