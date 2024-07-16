from .base import fMRIDataset

import h5py
import numpy as np
from pathlib import Path


class Narratives(fMRIDataset):
    subject_idxs = [f"{idx:03}" for idx in range(1, 346)]
    stories = [
        "pieman",
        "tunnel",
        "notthefallintact",
        "notthefalllongscram",
        "notthefallshortscram",
        "lucy",
        "prettymouth",
        "21styear",
        "merlin",
        "sherlock",
        "shapesphysical",
        "shapessocial",
        "bronx",
        "slumlordreach",
        "milkway",
        "schema",
        "forgot",
        "black",
        "piemanpni",
    ]
    exclude = {
        "pieman": ["001", "013", "014", "021", "022", "038", "056", "068", "069"],
        "tunnel": ["004", "013"],
        "lucy": ["053", "065"],
        "prettymouth": ["038", "205"],
        "milkyway": ["038", "105", "123"],
        "slumlordreach": ["139"],
        "notthefallintact": ["317", "335"],
        "notthefalllongscram": ["066", "335"],
        "notthefallshortscram": ["333"],
        "merlin": ["158"],
        "sherlock": ["139"],
    }

    def __init__(self, ddir):
        self.window_size = window_size
        self.ddir = Path(ddir)
        self.fmri_dir = self.ddir / "reponses"
        self.text_dir = self.ddir / "stimuli"
