import unittest

import pandas as pd
from sklearn.utils.estimator_checks import check_estimator

from src.train_classifier import (
    _split_to_feature_and_targets,
    evaluate_model,
    SpecialCharExtractor,
    tokenize
)
from tests.helpers import sort_and_assert_frame_equal


class TestTrainClassifier(unittest.TestCase):

    def test_split_to_feature_and_targets(self):
        df = pd.DataFrame(
            [
                [1, 'hello', 'greeting', 'something', 'nice'],
                [2, 'bye', 'greeting', 'something', 'not nice'],
                [3, 'you!', 'command', 'something', 'not nice']
            ]
            ,
            columns=['id', 'message', 'genre', 'who_knows', 'target']
        )
        output_X, output_Y = _split_to_feature_and_targets(df, ['message', 'genre', 'who_knows'])
        expected_X = pd.DataFrame([
            ['hello', 'greeting'],
            ['bye', 'greeting'],
            ['you!', 'command']

        ],
            columns=['message', 'genre'],
            index=pd.Index([1, 2, 3], name='id')
        )
        expected_Y = pd.DataFrame([
            ['nice'],
            ['not nice'],
            ['not nice']

        ],
            columns=['target'],
            index=pd.Index([1, 2, 3], name='id')
        )
        sort_and_assert_frame_equal(expected_X, output_X)
        sort_and_assert_frame_equal(expected_Y, output_Y)

    def test_evaluate_model(self):
        Y_test = pd.DataFrame(
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 0],
                [0, 1, 0]

            ],
            columns=['a', 'b', 'c']
        )
        Y_pred = pd.DataFrame(
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 0],
                [0, 0, 0]

            ],
            columns=['a', 'b', 'c']
        ).values

        output = evaluate_model(Y_test, Y_pred)
        expected = [
            'a: Accuracy: 1.000 Precision: 1.0 Recall: 1.000 F1_score: 1.000',
            'b: Accuracy: 0.750 Precision: 1.0 Recall: 0.750 F1_score: 0.857',
            'c: Accuracy: 1.000 Precision: 1.0 Recall: 1.000 F1_score: 1.000'
        ]
        self.assertListEqual(expected, output)


    def test_SpecialCharExtractor(self):
        special_char_extractor = SpecialCharExtractor()
        output = special_char_extractor.transform(['what are you filming this for?!'])
        expected = pd.DataFrame(  # question_mark, exclamation_mark, number_of_commas, text_len
            [
                [1, 1, 0, 31],

            ],
            columns=[0, 1, 2, 3]
        )
        sort_and_assert_frame_equal(expected, output)

    def test_tokenize(self):
        output = tokenize(
            'We need many more people here right now. Stop filming http bit.ly 7ENICX'
        )
        expected = ['need', 'many', 'people', 'right', 'stop', 'film', 'urlplaceholder']
        self.assertListEqual(expected, output)
