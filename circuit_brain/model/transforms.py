import abc
from typing import List, Optional

from circuit_brain.utils import PCA

import torch


class Transform:
    @abc.abstractmethod
    def __call__(self, layer_repr: List[torch.Tensor]): ...


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, layer_repr):
        out = layer_repr
        for trans in self.transforms:
            out = trans(out)
        return out


class Avg(Transform):
    def __call__(self, layer_repr):
        out = []
        for l in layer_repr:
            out.append(torch.mean(l, dim=1))
        return out


class Normalize(Transform):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, layer_repr):
        out = []
        for l in layer_repr:
            lt = l.transpose(0, self.dim)
            normed = (lt - torch.mean(lt, dim=0)) / torch.std(lt, dim=0)
            out.append(normed.transpose(self.dim, 0))
        return out


class WordAvg(Transform):
    def __init__(self, word2idx):
        self.word2idx = torch.LongTensor(word2idx)
        # get the maximum number of words
        self.mwords = torch.max(word2idx).item() + 1
        self._wordavg = torch.vmap(self._avg_single)

    def __call__(self, layer_repr):
        out = []

        for l in layer_repr:
            for b in l:
                out.append(self._wordavg(l, self.word2idx))
        return out

    # def _avg_single(self, repr: torch.Tensor, w2i: torch.LongTensor):
    #     _, counts = torch.unique(w2i, return_counts=True, dim=0)
    #     idxs = torch.cumsum(counts)
    #     nt = torch.zeros(self.mwords, repr.shape[1])
    #     words = len(nt) - 1
    #     for j in range(len(idxs)-1,-1,-1):
    #         start, end = idxs[j-1] if j - 1 >= 0 else 0, idxs[j]
    #         nt[words] = torch.mean(repr[:,start:end,:], dim=1)
    #         words -= 1
    #     nt[:words] = torch.mean(nt[words:], dim=0)
    #     return nt


class Concat(Transform):
    def __init__(self, source_dim=1, target_dim=2):
        self.source_dim = source_dim
        self.target_dim = target_dim

    def __call__(self, layer_repr):
        out = []
        for l in layer_repr:
            sh = list(l.shape)
            sh[self.target_dim] = sh[self.source_dim] * sh[self.target_dim]
            del sh[self.source_dim]
            out.append(l.reshape(*sh))
        return out


class ContextCrop(Transform):
    def __init__(self, window_size: int):
        self.window_size = window_size

    def __call__(self, layer_repr):
        out = []
        for l in layer_repr:
            out.append(l[:, -self.window_size :, :])
        return out


class ConvolveHRF(Transform):
    def __init__(self):
        pass


class PCAt(Transform):
    def __init__(self, pcas: List[PCA], fit: bool = True):
        self.pcas = pcas
        self.fit = fit

    def __call__(self, layer_repr):
        out = []
        for idx, l in enumerate(layer_repr):
            if self.fit:
                out.append(self.pcas[idx].fit_transform(l))
            else:
                out.append(self.pcas[idx].transform(l))
        return out
