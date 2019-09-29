#!/usr/bin/python3
"""This modules is part of the desaster recovery pipeline.
It loads data from an SQLite database, uses it to train a classifere and saves
the classifier as a pickl file.
"""

# import libraries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import sqlalchemy
import sys
from tokenizer import tokenize

# download nltk packages
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """Load data from database.

    database_filepath -- file path of the database
    """

    # load data from database
    engine = sqlalchemy.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('cleandata', engine)
    # Ignoring 'original' text because translation is available in message.
    X = df[['message', 'genre']]
    # child_alone is always 0 in our data set. Thus remove the column.
    y = df.drop(columns=['id', 'message', 'original', 'genre', 'child_alone'])

    return X, y, y.columns


def build_model():
    """Build the model."""

    # Parameters obtained by prior grid search
    params = {
        'clf__estimator__C': 1000,
        'clf__estimator__gamma': 0.001,
        'clf__estimator__kernel': 'rbf'}

    #
    #   Inputs          message             genre
    #                   |                   |
    #                   CountVectorizer     OneHotEncoder
    #                   |                   |
    #                   TfidTransformer     |
    #                   |                   |
    #                   +-------------------
    #                   |
    #                   MultiOutputClassifier(SVC)
    #                   |
    #   Outputs         classes
    #
    ct = ColumnTransformer(transformers = [
            ('msg', Pipeline(steps = [
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
                ]), 'message'),
           ('gnr', OneHotEncoder(), ['genre'] )
        ])

    pipeline = Pipeline([
        ('ct', ct),
        ('clf', MultiOutputClassifier(SVC()))
    ])

    pipeline.set_params(**params)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model.

    Print a classification report for each category.

    model           -- trained model
    X_test          -- features: message, genre
    Y_test          -- true categories
    category_names  -- names of the categories
    """

    Y_pred = pd.DataFrame(model.predict(X_test))
    Y_pred.columns = category_names
    Y_test = pd.DataFrame(Y_test)
    Y_test.columns = category_names

    for column in category_names:
        print('** {} **'.format(column).upper())
        print(classification_report(Y_test[column], Y_pred[column]))


def save_model(model, model_filepath):
    """Save model.

    model           -- trained model
    model_filepath  -- file path for pickle file to create
    """

    outfile = open('model_filepath','wb')
    pickle.dump(model, outfile)
    outfile.close()


def main():
    """Main entry point."""

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python3 '\
              'train_classifier.py ../data/DisasterResponse.db '\
              '../models/classifier.pkl')


if __name__ == '__main__':
    main()
