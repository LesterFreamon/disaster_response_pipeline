from typing import List
import sys

import pandas as pd
from sqlalchemy import create_engine


def _get_category_names(concat_categories_str: str, delimiter: str) -> List[str]:
    """Parse a raw target string in order to extract all of the categories"""
    split_categories = concat_categories_str.split(delimiter)
    parsed_categories = [
        category_with_binary[:-2]
        for category_with_binary in split_categories
    ]
    return parsed_categories


def _one_hot_encode_targets(
        message_categories_df: pd.DataFrame,
        raw_cat_col: str
) -> pd.DataFrame:
    """Parse the dataframe target to get a one hot encoding of the message categories"""
    cat_names = _get_category_names(message_categories_df[raw_cat_col].iloc[0], ';')
    cat_df = message_categories_df[raw_cat_col].str.split(';', expand=True)
    cat_df.columns = cat_names
    for cat_name in cat_names:
        cat_df[cat_name] = cat_df[cat_name].str.split('-').str[-1].astype(int)
    message_categories_df = message_categories_df.drop(raw_cat_col, axis=1)
    return message_categories_df.join(cat_df)


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """Load messages (features) and categories(targets) and return a merged dataframe"""
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    merged_df = messages_df.merge(categories_df, on='id')
    return merged_df


def clean_data(message_categories_df: pd.DataFrame) -> pd.DataFrame:
    """Parse the target column into a multilabel output and remove duplicate rows"""
    parsed_targets_df = _one_hot_encode_targets(message_categories_df, 'categories')
    clean_df = parsed_targets_df.drop_duplicates()
    return clean_df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """Save the data on a sqlite database"""
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_preprocess', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print(
            'Please provide the filepaths of the messages and categories '
            'datasets as the first and second argument respectively, as '
            'well as the filepath of the database to save the cleaned data '
            'to as the third argument. \n\nExample: python process_data.py '
            'disaster_messages.csv disaster_categories.csv DisasterResponse.db'
        )


if __name__ == '__main__':
    main()
