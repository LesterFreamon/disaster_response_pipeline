from collections import Counter
import pickle
import re
import sys
from typing import (
    List,
    Tuple
)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
URL_BIT_REGEX = 'http bit\.ly ......'


def _split_to_feature_and_targets(
        df: pd.DataFrame,
        non_target_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    X = df[['id'] + non_target_cols].set_index('id')[['message', 'genre']]
    Y = df.drop(non_target_cols, axis=1).set_index('id')
    return X, Y


def load_data(database_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Load data and split to features and targets"""
    df = pd.read_sql_table('disaster_preprocess', 'sqlite:///{}'.format(database_filepath))
    non_target_cols = ['message', 'original', 'genre']
    X, Y = _split_to_feature_and_targets(df, non_target_cols)
    return X, Y, Y.columns


def tokenize(text: str) -> List[str]:
    """Tokenize text"""
    detected_urls = re.findall(URL_REGEX, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        text = text.replace(URL_BIT_REGEX, "urlplaceholder")

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopwords_ = stopwords.words("english")
    tokens = [token for token in tokens if token not in stopwords_]
    clean_tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    return clean_tokens


class SpecialCharExtractor(BaseEstimator, TransformerMixin):

    def special_char_extractor(self, text: str) -> pd.Series:
        """Extract special characters and delimiters"""
        counter = Counter(text)
        question_mark = 0 if (counter['?'] == 0) else 1
        exclamation_mark = 0 if (counter['!'] == 0) else 1
        number_of_commas = counter[',']
        text_len = len(text)
        return pd.Series([question_mark, exclamation_mark, number_of_commas, text_len])

    def fit(self, x, y=None):
        """Here for scikit-learn compliance"""
        return self

    def transform(self, X: np.ndarray) -> pd.DataFrame:
        """Transform text into special characters bag of words"""
        X_tagged = pd.Series(X).apply(self.special_char_extractor)
        return pd.DataFrame(X_tagged)


def build_model():
    """Build a scikit learn pipeline"""
    tfidf_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
    ])
    pipeline = Pipeline(
        [
            (
                'features', ColumnTransformer(
                    transformers=[
                        ('tfidf_pipeline', tfidf_pipeline, 'message'),
                        ('genre_cat', OneHotEncoder(), ['genre']),
                        ('stuff', SpecialCharExtractor(), 'message')
                    ]
                )
            ),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    # specify parameters for grid search
    parameters = {
        'features__tfidf_pipeline__vect__ngram_range': [(1, 2)],
        'clf__estimator__criterion': ['gini'],
        'clf__estimator__min_samples_split': [4, 8]

    }
    # create grid search object
    cv = GridSearchCV(pipeline, parameters, verbose=3, cv=3)

    return cv


def evaluate_model(Y_test: pd.DataFrame, Y_pred: np.ndarray) -> None:
    """Evaluate model performance on test data"""
    category_names = Y_test.columns
    for index, name in enumerate(category_names):
        this_y_test = Y_test.values[:, index]
        this_y_pred = Y_pred[:, index]
        print(
            name + ": Accuracy: {:.3f} Precision: {:.3} Recall: {:.3f} F1_score: {:.3f}".format(
                accuracy_score(this_y_test, this_y_pred),
                precision_score(this_y_test, this_y_pred, average='weighted'),
                recall_score(this_y_test, this_y_pred, average='weighted'),
                f1_score(this_y_test, this_y_pred, average='weighted')
            ))


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """Save a model locally"""
    with open(model_filepath, 'wb') as file_object:
        pickle.dump(model, file_object)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)


        print('Evaluating model...')
        evaluate_model(Y_test, Y_pred)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print(
            'Please provide the filepath of the disaster messages database '
            'as the first argument and the filepath of the pickle file to '
            'save the model to as the second argument. \n\nExample: python '
            'train_classifier.py ../data/DisasterResponse.db classifier.pkl'
        )


if __name__ == '__main__':
    main()
