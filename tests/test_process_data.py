import unittest

import pandas as pd

from ..src.process_data import (
    _get_category_names,
    _one_hot_encode_targets,
    clean_data
)
from .helpers import sort_and_assert_frame_equal


class TestProcess(unittest.TestCase):

    def test_get_category_names(self):
        expected_output = ['related', 'request']
        outputs = _get_category_names('related-1;request-0', ';')

        self.assertEqual(
            expected_output,
            outputs,
            f'_get_category_names Expected to get {expected_output}. Got {outputs} instead.'
        )

    def test_one_hot_encode_targets(self):
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

        outputs = _one_hot_encode_targets(raw_categories_df, 'categories')
        sort_and_assert_frame_equal(expected_output, outputs)

    def test_clean_data(self):
        raw_categories_df = pd.DataFrame([
            [1, 'a', 'related-1;request-0'],
            [2, 'b', 'related-1;request-1'],
            [2, 'b', 'related-1;request-1']
        ],
            columns=['id', 'ABC', 'categories']
        )
        expected_output = pd.DataFrame([
            [1, 'a', 1, 0],
            [2, 'b', 1, 1]
        ], columns=['id', 'ABC', 'related', 'request'])

        outputs = clean_data(raw_categories_df)
        sort_and_assert_frame_equal(expected_output, outputs)
