import torch
import torch.linalg as tla

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV


def r2_score(pred, actual):
    ss_res = torch.mean((pred - actual) ** 2, dim=0)
    ss_tot = torch.var(actual, dim=0)
    return torch.nan_to_num(1 - ss_res / ss_tot)


def r2r_score(pred, actual):
    r2 = r2_score(pred, actual)
    return torch.where(r2 < 0, -1.0, 1.0) * torch.sqrt(torch.abs(r2))


def ridge(x, y, lam):
    nfeats = x.shape[1]
    return torch.matmul(
        tla.inv(torch.matmul(x.T, x) + lam * torch.eye(nfeats, device=x.device)),
        torch.matmul(x.T, y),
    )


def ridge_lam_per_target(x, y, x_val, y_val, lams=[1e-4, 1e-3]):
    error = torch.zeros(len(lams), y.shape[1], device=y.device)
    for i, lam in enumerate(lams):
        weights = ridge(x, y, lam)
        error[i] = 1 - r2_score(torch.matmul(x_val, weights), y_val)
    return error


def cross_val_ridge(x_train, y_train, n_splits=5, lams=[1e-3, 1e-4,1e-5]):
    r_cv = torch.zeros(len(lams), y_train.shape[1], device=y_train.device)
    kf = KFold(n_splits=n_splits)
    for f, (t_idx, v_idx) in enumerate(kf.split(y_train)):
        fx_train, fy_train = x_train[t_idx], y_train[t_idx]
        fx_val, fy_val = x_train[v_idx], y_train[v_idx]
        r_cv += ridge_lam_per_target(fx_train, fy_train, fx_val, fy_val, lams=lams)
    min_lams = torch.argmin(r_cv, axis=0)
    weights = torch.zeros((x_train.shape[1], y_train.shape[1]), device=x_train.device)
    for i in range(len(lams)):
        mask = min_lams == i
        if torch.any(mask):
            weights[:, mask] = ridge(x_train, y_train[:, mask], lams[i])
    return weights #, np.array([lams[i] for i in min_lams])
