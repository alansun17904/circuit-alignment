import re
from typing import List, Union, Tuple, Generator

from .base import fMRIDataset
from circuit_brain.utils import word_token_corr

import torch
import numpy as np
from pathlib import Path
from transformers import PreTrainedTokenizer


class HarryPotter(fMRIDataset):
    """Harry Potter dataset from *Simultaneously Uncovering the Patterns of
    Brain Regions Involved in Different Story
    Reading Subprocesses* by Wehbe et al. (2014). The dataset contains eight
    subjects and each subject is read Chapter 9 from *Harry
    Potter and the Sorcerer's Stone*.

    The preprocessed data can be downloaded from
    `here <https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8>`_.
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
        context_size: int,
        tokenizer: PreTrainedTokenizer,
        remove_format_chars: bool = False,
        remove_punc_spacing: bool = False,
        pool_rois: bool = True,
    ):
        """Initializes the dataset. This method should not be called directly. Instead,
        one should use the factory method in the `fMRIDataset` class.

        Args:
            ddir: Path to the downloaded data directory. It is assumed that the
                subdirectory structure and file naming follows
                `here <https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8>`__.
            context_size: For a given fMRI measurement, the number of previous tokens that we use
                to compute a given window.
            tokenizer: a HuggingFace tokenizer
            remove_format_chars: Whether or not to remove the special formatting
                characters that were displayed to participants such as `@` and `+`.
            remove_punc_spacing: Punctuation such as ellipses `...` or em-dashes
                `—` were displayed as `. . .` (period-by-period) and ` --- `,
                respectively, to participants. If this flag is true, punctuation is
                reformatted to what is conventional (i.e. `...` and `—` with no spaces
                around it).
            pool_rois: If true, then the voxels are aggregated according to the brain regions
            that they belong to. In particular, we only focus on eight regions that are associated
            with language processing.
        """
        self.context_size = context_size
        self.ddir = Path(ddir)
        self.fmri_dir = self.ddir / "fMRI"
        self.voxel_n = self.ddir / "voxel_neighborhoods"
        self.rois = self.fmri_dir / "HP_subj_roi_inds.npy"
        self.pool_rois = pool_rois
        self.remove_format_chars = remove_format_chars
        self.remove_punc_spacing = remove_punc_spacing
        self.tokenizer = tokenizer

        # truncate and pad from the left side
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        self.subject_rois = [
            np.load(self.rois, allow_pickle=True).item()[i] for i in self.subject_idxs
        ]

        self.remove_format_chars = lambda x: re.sub(r"@|\+", "", x)
        self.unify_em_dash = lambda x: re.sub(r"--", "—", x)
        self.remove_em_spacing = lambda x: re.sub(r"\s*—\s*", "—", x)

        self.contexts = np.empty(len(self.fmri_timing), dtype=object)
        for i, mri_time in enumerate(self.fmri_timing):
            f = filter(lambda x: x[0] <= mri_time, zip(self.word_timing, self.words))
            w = map(lambda x: x[1], f)
            t_proc = " ".join(list(w))
            if self.remove_punc_spacing:
                t_proc = self.remove_ellipses_spacing(t_proc)
                t_proc = self.unify_em_dash(t_proc)
                t_proc = self.remove_em_spacing(t_proc)
            if self.remove_format_chars:
                t_proc = self.remove_format_chars(t_proc)
            self.contexts[i] = t_proc
        self.contexts = self.contexts.tolist()

        # tokenize the context and get the word correspondance
        self.toks, _, self.idxmap = word_token_corr(
            tokenizer,
            self.contexts,
            truncation=True,
            padding=True,
            max_length=self.context_size,
        )
        self.idxmap = np.array(self.idxmap)
        self.toks = self.toks["input_ids"]

    def remove_ellipses_spacing(self, text):
        """Given some text, transformers ellipses in the form of ". . ." to
        the grammatically correct format of "...". In the stimulus all punctuation
        is formatted with the former spacing.
        """
        es_pat = r"\.\s+\."
        self.res = lambda x: re.sub(es_pat, "..", x)
        while re.search(es_pat, text):
            text = self.res(text)
        return text

    def __len__(self):
        """Gets the total number of subjects."""
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Union[np.array, Tuple[dict, np.array]]:
        """Given the index of the subject get their corresponding fMRI recording.
        If :code:`self.pool_rois` is true, then the subject recordings along with a
        dictionary is returned where the keys are the names of the regions of interest
        and the values are a boolean mask (of dimension equal to the number of voxels)
        which denote whether each voxel is a part of that particular region of interest.

        Args:
            idx: A numerical index that correspond to the subject whose recordings we wish to
                retrieve. The total number of subjects can be found by running
                :code:`HarryPotter.__len__()`. Moreover, the correspondance between the indices
                and the actual participant codes is stored in `HarryPotter.subject_idxs`.

        Returns:
            If `self.pool_rois` is true, then a tuple is returned where the first entry
            is the fMRI measurements of the subject across all time intervals, the second
            entry is a dictionary that maps the name of the brain region to a boolean mask
            which corresponds voxels to particular brain regions.

            On the other hand, if `self.pool_rois` is true, only the aforementioned first entry
            is returned as an array.
        """
        if self.pool_rois:
            return self.subjects[idx], self.subject_rois[idx]
        return self.subjects[idx]

    def kfold(
        self, subject_idx: int, folds: int, trim: int
    ) -> Generator[Tuple[int, np.array, np.array], None, None]:
        """A generator that yields `folds` number of training/test folds while trimming
        off `trim` number of samples at the ends of the training folds.

        Note that since all subjects are using the same stimuli, the generator is
        subject-independent as all subjects share the same measurement indices.

        Args:
            folds: The number of folds.
            trim: The number of fMRI measurements to remove from either end of the training and
                test folds.

        Yields:
            A tuple of the index of the current fold, (for both training and testing folds)
            normalized fMRI measurements within fold for each of the points of interest in the
            fold as well as the tokens associated with each of these measurements. The last entry
            in each of these pairs is the mapping between words and token indices.

        Raises:
            AssertionError: If the number of trimmed samples is greater than the total
                number of examples in the test fold.
        """
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

            test_idxs = list(range(start, end))
            train_idxs = list(range(0, train_st)) + list(
                range(train_ed, len(self.fmri_timing))
            )

            yield f, self._idx2samples(subject_idx, test_idxs), self._idx2samples(
                subject_idx, train_idxs
            )

    def _idx2samples(self, subject_idx, idxs):
        measures = self.subjects[subject_idx][idxs]

        if self.pool_rois:
            rois = self.subject_rois[subject_idx]
            # do not get the "all" region
            roi_measures = np.zeros((len(idxs), 8))
            if "all" in rois:
                del rois["all"]
            for i, (label, mask) in enumerate(rois.items()):
                # average across all voxels that are in the same region
                roi_measures[:, i] = np.mean(measures[:, mask], axis=1)

            measures = roi_measures

        # normalize each voxel/roi across time
        measures = (measures - np.mean(measures, axis=0)) / np.std(measures, axis=0)
        return (
            torch.Tensor(measures),
            torch.LongTensor(self.toks)[idxs],
            torch.LongTensor(self.idxmap)[idxs],
        )
