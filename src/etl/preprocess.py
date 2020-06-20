from typing import List

import pandas as pd
from sqlalchemy.engine.base import Connection


def _get_categories_col_names(raw_categories_df: pd.DataFrame, cat_col: str) -> List[str]:
    """Extract the column names from raw form: 'related-1;request-0' -> ['related', 'request']"""
    first_index = raw_categories_df.index[0]
    category_colnames = [
        col_name_with_label[:-2]
        for col_name_with_label in raw_categories_df.loc[first_index, cat_col].split(';')
    ]
    return category_colnames


def parse_categories_df(
        raw_categories_df: pd.DataFrame,
        orig_cols: List[str],
        cat_col: str
) -> pd.DataFrame:
    """Parse a categories dataframe that initially containes all of the labels in one string"""
    split_cat_df = raw_categories_df[cat_col].str.split(';', expand=True)
    cat_col_names = _get_categories_col_names(raw_categories_df, cat_col)
    split_cat_df.columns = cat_col_names
    for col in cat_col_names:
        # set each value to be the last character of the string
        split_cat_df[col] = split_cat_df[col].apply(lambda x: x[-1])

        # convert column from string to numeric
        split_cat_df[col] = pd.to_numeric(split_cat_df[col])

    merged_cat_df = pd.concat([raw_categories_df[orig_cols], split_cat_df], axis=1)
    return merged_cat_df


def preprocess_disaster_input(
        raw_messages_df,
        raw_categories_df
) -> pd.DataFrame:
    """Preprocess, merge and deduplicate raw dataframes"""
    processed_cat_df = parse_categories_df(raw_categories_df, ['id'], 'categories')
    merged_df = raw_messages_df.merge(processed_cat_df, on='id')
    deduplicated_df = merged_df.drop_duplicates()
    return deduplicated_df


def save_data(engine: Connection, df: pd.DataFrame, table_name, is_index: bool = False) -> None:
    """Save a dataframe to remote db"""
    df.to_sql(table_name, engine, index=is_index)
