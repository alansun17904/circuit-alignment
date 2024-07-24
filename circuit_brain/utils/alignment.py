import pickle
import argparse
from typing import Optional, List, Callable

import circuit_brain.utils as utils
from circuit_brain.model import BrainAlignTransformer
from circuit_brain.dproc import fMRIDataset
from circuit_brain.model import transforms

import tqdm
import torch
from transformers import AutoTokenizer


GenTransform = Callable[..., transforms.Transform]
EncoderModel = utils.RidgeCV
ReprCache = List[torch.Tensor]
IDENTITY = lambda x: x


def sub_align(
    m: BrainAlignTransformer,
    ds: fMRIDataset,
    ridge: EncoderModel,
    subject_idx: int,
    batch_size: int = 64,
    folds: int = 5,
    trim: int = 10,
    gen_train_rpre: Optional[GenTransform] = lambda x: IDENTITY,
    gen_train_rpost: Optional[GenTransform] = lambda x: IDENTITY,
    gen_test_rpre: Optional[GenTransform] = lambda x: IDENTITY,
    gen_test_rpost: Optional[GenTransform] = lambda x: IDENTITY,
    train_reprs: Optional[ReprCache] = list(),
    test_reprs: Optional[ReprCache] = list(),
):
    ds_generator = ds.kfold(subject_idx, folds, trim)
    fold_scores = None
    for k, test, train in ds_generator:
        train_fmri, train_toks, train_idxmap = train
        test_fmri, test_toks, test_idxmap = test
        train_rpre = gen_train_rpre(train_idxmap)
        train_rpost = gen_train_rpost(train_idxmap)
        test_rpre = gen_test_rpre(test_idxmap)
        test_rpost = gen_test_rpost(test_idxmap)
        if (len(train_reprs) != folds) or (len(test_reprs) != folds):
            _, train_model_repr = m.resid_post(
                train_toks,
                batch_size=batch_size,
                rpre=train_rpre,
                rpost=train_rpost,
            )
            _, test_model_repr = m.resid_post(
                test_toks,
                batch_size=batch_size,
                rpre=test_rpre,
                rpost=test_rpost,
            )
            train_reprs.append(train_model_repr)
            test_reprs.append(test_model_repr)
        else:
            train_model_repr = train_reprs[k]
            test_model_repr = test_reprs[k]
        lr2 = torch.zeros(len(train_model_repr), test_fmri.shape[1])
        if k == 0:
            fold_scores = torch.zeros(5, len(train_model_repr), test_fmri.shape[1])
        for layer in range(len(train_model_repr)):
            ridge.fit(train_model_repr[layer], train_fmri)
            lr2[layer] = ridge.score(test_model_repr[layer], test_fmri)
        fold_scores[k] = lr2
    return torch.mean(fold_scores, dim=0), train_reprs, test_reprs
