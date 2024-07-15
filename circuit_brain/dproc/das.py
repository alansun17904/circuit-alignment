from typing import List

from .base import fMRIDataset

import pandas as pd
from pathlib import Path


class DAS(fMRIDataset):
    """Sentences from *Driving and suppressing the human language network
    using large language models* by Tuckute et al. (2024). The dataset contains
    1000 corpus-extracted sentences that formed the baseline. Then, an encoding model
    was created to predict the blood-oxygen-level-dependent (BOLD) response of each
    sentence. 1000 sentences were then generated, according to this encoding model,
    to either drive or suppress BOLD response in the language network (500 each).

    The data directory can be downloaded from
    `here <https://github.com/gretatuckute/drive_suppress_brains>`__, the instantiations
    of this class assumes this exact directory structure and file naming.
    """

    dataset_id = "das"
    base_sents: List[str]
    drive_sents: List[str]
    supp_sents: List[str]

    def __init__(self, ddir: str):
        """Initializes the dataset. This method should not be called directly. Instead,
        one should use the factory method in the `fMRIDataset` class.

        Args:
            ddir: Path to the downloaded data directory. 
        """
        self.ddir = Path(ddir)
        self.df = pd.read_csv(self.ddir / "brain-lang-data_participant_20230728.csv")

    @property
    def base_sents(self):
        """Gets all 1000 of the corpus-extracted baseline sentences"""
        return self.df[self.df["cond"] == "B"].sentence.unique()

    @property
    def drive_sents(self):
        """Gets all 500 of the sentences that drive BOLD responses in
        the language network."""
        return self.df[self.df["cond"] == "D"].sentence.unique()

    @property
    def supp_sents(self):
        """Gets all 500 of the sentences that suppress the BOLD responses
        in the language network."""
        return self.df[self.df["cond"] == "S"].sentence.unique()
