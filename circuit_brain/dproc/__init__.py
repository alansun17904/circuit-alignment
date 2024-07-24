from .base import fMRIDataset
from .hpot import HarryPotter
from .das import DAS


def get_dataset(dataset_id, ddir, *args, **kwargs):
    """Factory method for creating datasets given the `dataset_id` as well as
    its data directory and other important keyword arguments, specified by each
    dataset.
    """
    if dataset_id == "hp":
        kwargs["remove_format_chars"] = kwargs.get("remove_format_chars", False)
        kwargs["remove_punc_spacing"] = kwargs.get("remove_punc_spacing", False)
        return HarryPotter(ddir, *args, **kwargs)
    elif dataset_id == "das":
        return DAS(ddir, *args, **kwargs)
    else:
        raise ValueError(f"Invalid dataset ID: {dataset_id}")


fMRIDataset.get_dataset = get_dataset
