import re
from typing import List

from .base import fMRIDataset

import numpy as np
from pathlib import Path


class HarryPotter(fMRIDataset):
    """Harry Potter dataset from *Simultaneously Uncovering the Patterns of
    Brain Regions Involved in Different Story
    Reading Subprocesses* by Wehbe et al. (2014). The dataset contains eight
    subjects and each subject is read Chapter 9 from *Harry
    Potter and the Sorcerer’s Stone*.

    The preprocessed data can be downloaded from
    [here](https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8).
    Every subject's fMRI recording is normalized across time for each voxel.
    Subjects are shown the stimuli word-by-word
    at intervals of 0.5 seconds. Text formatting such as italics or newline characters
    are displayed to the participants separately as `@` and `+`, respectively.
    """

    dataset_id = "hp"
    subject_idxs = ["F", "H", "I", "J", "K", "L", "M", "N"]

    def __init__(
        self,
        ddir: str,
        window_size: int,
        remove_format_chars: bool = False,
        remove_punc_spacing: bool = False,
    ):
        """Initializes the dataset. This method should not be called directly. Instead,
        one should use the factory method in the `fMRIDataset` class.

        Args:
            ddir: Path to the downloaded data directory. It is assumed that the
                subdirectory structure and file naming follows
                [here](https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8).
            window_size: For a given fMRI measurement, the number of previous words
                that are assumed to be a part of this fMRI's context.
            remove_format_chars: Whether or not to remove the special formatting
                characters that were displayed to participants such as `@` and `+`.
            remove_punc_spacing: Punctuation such as ellipses `...` or em-dashes
                `—` were displayed as `. . .` (period-by-period) and ` --- `,
                respectively, to participants. If this flag is true, punctuation is
                reformatted to what is conventional (i.e. `...` and `—` with no spaces
                around it).
        """
        self.window_size = window_size
        self.ddir = Path(ddir)
        self.fmri_dir = self.ddir / "fMRI"
        self.voxel_n = self.ddir / "voxel_neighborhoods"

        # load metadata
        self.words = np.load(self.fmri_dir / "words_fmri.npy")
        self.word_timing = np.load(self.fmri_dir / "time_words_fmri.npy")
        self.fmri_timing = np.load(self.fmri_dir / "time_fmri.npy")
        runs = np.load(self.fmri_dir / "runs_fmri.npy")

        # remove the edges of each run
        self.fmri_timing = np.concatenate(
            [self.fmri_timing[runs == i][20:-15] for i in range(1, 5)]
        )

        # load subject recordings
        self.subjects = [
            np.load(self.fmri_dir / f"data_subject_{i}.npy") for i in self.subject_idxs
        ]

        # normalize subject recordings
        for i in range(len(self.subjects)):
            self.subjects[i] = (
                self.subjects[i] - np.mean(self.subjects[i], axis=0)
            ) / np.std(self.subjects[i], axis=0)

        # for each fMRI measurement find its word context according to window size
        self.fmri_contexts = []
        for mri_time in self.fmri_timing:
            f = filter(lambda x: x[0] <= mri_time, zip(self.word_timing, self.words))
            m = map(lambda x: x[1], f)
            self.fmri_contexts.append(" ".join(list(m)[-self.window_size :]))

        if remove_format_chars:
            self.fmri_contexts = list(
                map(lambda x: re.sub(r"@|\+", "", x), self.fmri_contexts)
            )

        if remove_punc_spacing:
            # unify all em-dash
            uni_em_d = lambda x: re.sub(r"--", "—", x)
            self.fmri_contexts = map(uni_em_d, self.fmri_contexts)

            # remove the spacing around a em-dash
            remove_em_spacing = lambda x: re.sub(r"\s*—\s*", "—", x)
            self.fmri_contexts = list(map(remove_em_spacing, self.fmri_contexts))

            # remove the spacing around ellipses
            es_pat = r"\.\s+\."
            remove_ellipse_spacing = lambda x: re.sub(es_pat, "..", x)

            while any([re.search(es_pat, v) for v in self.fmri_contexts]):
                self.fmri_contexts = list(
                    map(remove_ellipse_spacing, self.fmri_contexts)
                )

    def __len__(self):
        """Gets the total number of subjects."""
        return len(self.subjects)

    def __getitem__(self, idx):
        """Given the index of the subject get their corresponding fMRI recording."""
        return self.subjects[idx]

    def kfold(self, folds: int, trim: int):
        """A generator that yields `folds` number of training/test folds while trimming
        off `trim` number of samples at the ends of the training folds.

        Note that since all subjects are using the same stimuli, the generator is
        subject-independent as all subjects share the same measurement indices.

        Args:
            folds: The number of folds.
            trim: The number of samples to remove from either end of the training and
                test folds.

        Yields:
            A tuple of the index of the current fold, the indices of the test examples,
            and the indices of the training examples.

        Raises:
            AssertionError: If the number of trimmed samples is greater than the total
                number of examples in the test fold.
        """
        print(
            f"There are {len(self.fmri_timing)} fMRI measurements available.",
            f"Splitting {folds} folds means roughly {len(self.fmri_timing) // folds}",
            f"measurements / fold, and {len(self.words) // folds} words / fold.",
        )
        fold_size = len(self.fmri_timing) // folds
        assert 2 * trim <= fold_size

        for f in range(folds):
            if f == 0:
                start = 0
            else:
                start = trim + fold_size * f
            if f == folds - 1:
                end = len(self.fmri_timing)
            else:
                end = fold_size * (f + 1) - trim

            train_st = max(start - trim, 0)
            train_ed = min(end + trim, len(self.fmri_timing))

            yield f, list(range(start, end)), list(range(0, train_st)) + list(
                range(train_ed, len(self.fmri_timing))
            )
