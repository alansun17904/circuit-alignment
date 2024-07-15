"""Base class for all datasets.

This class outlines all of the properties and functionalities a generic fMRI
dataset w/ stimuli should have. This class is not meant to be instantiated, but
meant to be used as a factory class.

Typical usage example:

.. code::python
    
    dataset = fMRIDataset.get_dataset(dataset_id, ddir, **kwargs)

"""

from typing import List, Union
import numpy as np
from torch.utils.data import Dataset


class fMRIDataset(Dataset):
    """A typical fMRI dataset that includes fMRI recordings from several
    participants and the corresponding stimuli. Most importantly, the dataset
    assumes that the stimuli is the same across all subjects and presented to
    each participant at fixed time intervals.
    """

    dataset_id: str
    ddir: str
    subjects: List[np.array]
    subject_idxs: List[Union[int, str]]

    def __init__(self):
        raise NotImplemented()

    def __len__(self):
        """Gets the total number of subjects."""
        raise NotImplemented()

    def __getitem__(self, idx: int):
        """Given the index of the subject get their corresponding fMRI recording."""
        raise NotImplemented()

    def kfold(self, folds: int, trim: int):
        """A generator that yields `folds` number of training/test folds while
        trimming off `trim` number of samples at the ends of the test fold.
        """
        raise NotImplemented()
