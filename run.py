# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse.db', engine)

# load model
model = joblib.load("classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # Visualization one data
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Visulization two data, for death category the genre share
    genre_share_by_death = df[df['death'] == 1].groupby(
        'genre').count()['message']

    # Visulization three data, for each cold category the genre bar
    genre_share_by_cold = df[df['cold'] == 1].groupby(
        'genre').count()['message']

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [{
        'data': [Bar(x=genre_names, y=genre_counts)],
        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        }
    },
              {
                  'data':
                  [Pie(labels=genre_names, values=genre_share_by_death)],
                  'layout': {
                      'title':
                      'Distribution of Message Genres in terms of death'
                  }
              },
              {
                  'data':
                  [Bar(x=genre_names, y=genre_share_by_cold)],
                  'layout': {
                      'title':
                      'Distribution of Message Genres in terms of Child Alone'
                  }
              }]

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

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html', query=query, classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
