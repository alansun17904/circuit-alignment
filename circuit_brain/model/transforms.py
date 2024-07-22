import abc
from typing import List, Optional
from functools import partial

from circuit_brain.utils import PCA

import torch
import torch.nn.functional as F


class Transform:
    @abc.abstractmethod
    def __call__(self, layer_repr: List[torch.Tensor]): ...


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    @torch.no_grad()
    def __call__(self, layer_repr):
        out = layer_repr
        for trans in self.transforms:
            out = trans(out)
        return out


class Avg(Transform):
    @torch.no_grad()
    def __call__(self, layer_repr):
        out = []
        for l in layer_repr:
            out.append(torch.mean(l, dim=1))
        return out


class Normalize(Transform):
    def __init__(self, dim=1):
        self.dim = dim

    @torch.no_grad()
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
        self.mwords = torch.max(self.word2idx).item() + 1

    @torch.no_grad()
    def __call__(self, layer_repr):
        out = []
        for l in layer_repr:
            ls = torch.zeros(l.shape[0], self.mwords, l.shape[2])
            for i, b in enumerate(l):
                ls[i] = self._avg_single(b, self.word2idx[i])
            out.append(ls)
        return out

    def _avg_single(self, repr: torch.Tensor, w2i: torch.LongTensor):
        _, counts = torch.unique(w2i, return_counts=True, dim=0)  # get the counts of each word
        idxs = torch.cumsum(counts, dim=0)
        print(idxs)
        nt = torch.zeros(self.mwords, repr.shape[1])
        words = len(nt) - 1
        for j in range(len(idxs)-1,-1,-1):
            start, end = idxs[j-1] if j - 1 >= 0 else 0, idxs[j]
            nt[words] = torch.mean(repr[start:end,:], dim=0)
            words -= 1
        nt[:words+1] = torch.mean(nt[words:], dim=0)
        return nt


class Concat(Transform):
    def __init__(self, source_dim=1, target_dim=2):
        self.source_dim = source_dim
        self.target_dim = target_dim

    @torch.no_grad()
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

    @torch.no_grad()
    def __call__(self, layer_repr):
        out = []
        for l in layer_repr:
            out.append(l[:, -self.window_size :, :])
        return out


class ConvolveHRF(Transform):
    def __init__(self, filter: torch.Tensor):
        self.filter = filter

    @torch.no_grad()    
    def __call__(self, layer_repr):
        out = []
        for l in layer_repr:
            cin = l.shape[-1]
            weight = torch.zeros(cin, cin, len(self.filter))
            weight[range(cin),range(cin),:] = self.filter
            out.append(F.conv1d(l.transpose(1,2), weight).transpose(1,2))
        return out


class PCAt(Transform):
    def __init__(self, pcas: List[PCA], fit: bool = True):
        self.pcas = pcas
        self.fit = fit

    @torch.no_grad()
    def __call__(self, layer_repr):
        out = []
        for idx, l in enumerate(layer_repr):
            if self.fit:
                out.append(self.pcas[idx].fit_transform(l))
            else:
                out.append(self.pcas[idx].transform(l))
        return out
