import itertools
import pandas as pd
from typing import Sequence


def make_df_from_ranges(
    column_max_ranges: Sequence[int], column_names: Sequence[str]
) -> pd.DataFrame:
    """
    Takes in a list of column names and max ranges for each column, and returns a dataframe with the cartesian product of the range for each column (ie iterating through all combinations from zero to column_max_range - 1, in order, incrementing the final column first)
    """
    rows = list(
        itertools.product(
            *[range(axis_max_range) for axis_max_range in column_max_ranges]
        )
    )
    df = pd.DataFrame(rows, columns=column_names)
    return df
