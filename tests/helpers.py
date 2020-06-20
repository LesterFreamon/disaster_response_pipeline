import pandas as pd
from pandas.testing import assert_frame_equal


def sort_and_assert_frame_equal(df_1: pd.DataFrame, df_2: pd.DataFrame) -> None:
    """Sort the data frames by columns and index and then compare"""
    df_1 = df_1.reindex(sorted(df_1.columns), axis=1)
    df_1 = df_1.sort_values(df_1.columns.tolist())
    df_2 = df_2.reindex(sorted(df_1.columns), axis=1)
    df_2 = df_2.sort_values(df_1.columns.tolist())
    assert_frame_equal(df_1, df_2)
