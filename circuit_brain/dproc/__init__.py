from .base import fMRIDataset
from .hpot import HarryPotter
from .das import DAS


def get_dataset(dataset_id, ddir, **kwargs):
    """Factory method for creating datasets given the `dataset_id` as well as
    its data directory and other important keyword arguments, specified by each
    dataset.
    """
    if dataset_id == "hp":
        assert (
            kwargs.get("window_size", None) is not None
        ), "Must provide window size for Harry Potter dataset!"
        remove_format_chars = kwargs.get("remove_format_chars", False)
        remove_punc_spacing = kwargs.get("remove_punc_spacing", False)
        return HarryPotter(
            ddir, kwargs["window_size"], remove_format_chars, remove_punc_spacing
        )
    elif dataset_id == "das":
        return DAS(ddir)
    else:
        raise ValueError(f"Invalid dataset ID: {dataset_id}")


fMRIDataset.get_dataset = get_dataset
