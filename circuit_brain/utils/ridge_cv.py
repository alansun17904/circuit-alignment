import torch
import numpy as np

import circuit_brain.utils.ridge_torch_utils as rtu
import circuit_brain.utils.ridge_np_utils as rnu


class RidgeCV:
    def __init__(
        self,
        lam_per_target=True,
        n_splits=10,
        lams=[10**i for i in range(-6, 10)],
        device="cuda",
    ):
        assert device in ["cpu", "cuda"], "Device must be either 'cpu' or 'cuda'."
        self.n_splits = n_splits
        self.lams = lams
        if device == "cpu":
            self.r2_score = rnu.R2
            self.r2r_score = rnu.R2r
            self._cvr = rnu.cross_val_ridge
            self.mean = np.mean
            self.device = device
        else:
            self.r2_score = rtu.r2_score
            self.r2r_score = rtu.r2r_score
            self._cvr = rtu.cv_ridge_lam_per_target if lam_per_target else rtu.cv_ridge
            self.mean = torch.mean
            self.device = torch.device(device)
        self.W = None

    def fit(self, x, y):
        # compute the zscore of both x and y
        x = (x - x.mean(dim=0)) / x.std(dim=0)
        y = (y - y.mean(dim=0)) / y.std(dim=0)
        if self.device != "cpu":
            x, y = x.to(self.device), y.to(self.device)
        self.W = self._cvr(x, y, self.n_splits, self.lams)

    def predict(self, x):
        x = (x - x.mean(dim=0)) / x.std(dim=0)
        if self.W is None:
            raise ValueError("RidgeCV.fit needs to be run before calling inference.")
        if self.device == "cpu":
            return x.dot(self.W)
        x = x.to(self.W.device).float()
        return torch.matmul(x, self.W)

    def score(self, x, y):
        if self.device != "cpu":
            y = y.to(x.device)
        return self.r2_score(self.predict(x), y)
