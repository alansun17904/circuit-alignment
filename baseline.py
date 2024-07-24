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
        context_size=128,
        tokenizer=atok,
        remove_format_chars=opts.rfc,
        remove_punc_spacing=opts.rps,
        pool_rois=True,
    )
    print("Dataset pre-processed!")
    ridge_cv = utils.RidgeCV(n_splits=5)
    gen_rpre = lambda w: transforms.Compose(
        [
            # transforms.WordAvg(w, truncation=20),
            transforms.ContextCrop(1),
            lambda x: [v.squeeze(1) for v in x],
            # transforms.Normalize(),
        ]
    )
    # pcas = [utils.PCA(100)] * m.ht.cfg.n_layers
    # gen_train_rpost = lambda: transforms.PCAt(pcas)
    # gen_test_rpost = lambda: transforms.PCAt(pcas, fit=False)
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
            # gen_train_rpost,
            gen_test_rpre=gen_rpre,
            # gen_test_rpost,
            train_reprs=train_reprs,
            test_reprs=test_reprs,
        )
        subjs.append(sscores)

    pickle.dump(
        subjs,
        open(f"{opts.ofname}", "wb+"),
    )
