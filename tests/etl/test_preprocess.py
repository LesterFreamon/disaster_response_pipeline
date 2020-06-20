import unittest

import pandas as pd

from src.etl.preprocess import (
    _get_categories_col_names,
    parse_categories_df,
    preprocess_disaster_input
)
from tests.helpers import sort_and_assert_frame_equal


class TestProcess(unittest.TestCase):

    def test_get_categories_col_names(self):
        raw_categories_df = pd.DataFrame([
            [1, 'related-1;request-0'],
            [2, 'related-1;request-1']
        ],
            columns=['id', 'categories']
        )
        expected_output = ['related', 'request']
        outputs = _get_categories_col_names(raw_categories_df, 'categories')

        self.assertListEqual(
            expected_output,
            outputs,
            f'_get_categories_col_names Expected to get {expected_output}. Got {outputs} instead.'
        )

    def test_parse_categories_df(self):
        raw_categories_df = pd.DataFrame([
            [1, 'a', 'related-1;request-0'],
            [2, 'b', 'related-1;request-1']
        ],
            columns=['id', 'ABC', 'categories']
        )
        expected_output = pd.DataFrame([
            [1, 'a', 1, 0],
            [2, 'b', 1, 1]
        ], columns=['id', 'ABC', 'related', 'request'])
        outputs = parse_categories_df(raw_categories_df, ['id', 'ABC'], 'categories')
        sort_and_assert_frame_equal(expected_output, outputs)

    def test_preprocess_disaster_input(self):
        raw_messages_df = pd.DataFrame(
            [
                [1, 'Weather update - hot', 'Un front froid se retrouve sur Cuba', 'direct'],
                [7, 'Is the Hurricane over', 'Cyclone nan fini osinon li pa fini', 'direct']
            ],
            columns=['id', 'message', 'original', 'genre']
        )
        raw_categories_df = pd.DataFrame([
            [1, 'related-1;request-0'],
            [2, 'related-1;request-1']
        ],
            columns=['id', 'categories']
        )
        expected_output = pd.DataFrame(
            [
                [1, 'Weather update - hot', 'Un front froid se retrouve sur Cuba', 'direct', 1, 0]
            ],
            columns=['id', 'message', 'original', 'genre', 'related', 'request']
        )
        outputs = preprocess_disaster_input(raw_messages_df, raw_categories_df)
        sort_and_assert_frame_equal(expected_output, outputs)
