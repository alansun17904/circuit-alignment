import torch
import numpy as np
from rich.progress import track


def per_subject_model_repr(sents, model, batch_size):
    sent_toks = model.to_tokens(sents)
    return model.hidden_repr(sent_toks, batch_size)


def per_subject_alignment(
    dataset, model_repr, subject_idx, folds, trim, ridge_cv, pca=None
):
    fold_r2 = []
    for k, test_idx, train_idx in track(
        dataset.kfold(folds, trim), description=f"S{subject_idx+1}", total=folds
    ):
        layer_r2 = []
        train_repr, test_repr = model_repr[:, train_idx, :], model_repr[:, test_idx, :]
        train_fmri, test_fmri = (
            dataset[subject_idx][train_idx, :],
            dataset[subject_idx][test_idx, :],
        )
        train_fmri = torch.from_numpy(train_fmri).float()
        test_fmri = torch.from_numpy(test_fmri).float()
        for l in range(len(train_repr)):
            train_repr_l, test_repr_l = train_repr[l], test_repr[l]
            if pca is not None:
                pvar = torch.var(train_repr_l)
                train_repr_l = pca.fit_transform(train_repr_l)
                test_repr_l = pca.transform(test_repr_l)
            ridge_cv.fit(train_repr_l, train_fmri)
            r2 = ridge_cv.score(test_repr_l, test_fmri)
            layer_r2.append(r2.cpu().item())
        fold_r2.append(layer_r2)
    print(f"s{subject_idx}-alignment: {np.mean(fold_r2, (0,1))}")
    return fold_r2


def across_subject_alignment(dataset, model_repr, folds, trim, ridge_cv, pca=None):
    return [
        per_subject_alignment(dataset, model_repr, sidx, folds, trim, ridge_cv, pca=pca)
        for sidx in range(len(dataset))
    ]
