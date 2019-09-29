#!/usr/bin/python3

import json
import plotly
import pandas as pd
import os

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine
from tokenizer import tokenize

import sys


app = Flask(__name__)

# get the path to the currently running script
path = os.path.dirname(os.path.realpath(__file__))

# load data
engine = create_engine('sqlite:///' + path + '/../data/DisasterResponse.db')
df = pd.read_sql_table('cleandata', engine)

# category columns (except for child_alone where we have no data)
category_columns = [c for c in df.columns if c not in
                    ['id', 'message', 'original', 'genre', 'child_alone']]

# load model
model = joblib.load(path + '/../models/classifier.pkl')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = list(df[category_columns].sum())
    category_names = [c.replace('_', ' ').capitalize() for c in category_columns]

    print(type(category_counts))
    print(type(category_names))

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=category_names,
                    values=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    genre = request.args.get('genre', '')
    df_query = pd.DataFrame({'message':[query], 'genre':[genre]})


    # use model to predict classification for query
    classification_labels = model.predict(df_query)[0]
    classification_results = dict(zip(category_columns, classification_labels))
    print(classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        genre=genre,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
