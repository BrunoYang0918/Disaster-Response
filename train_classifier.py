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

# # Train Classifier

# ## Load Libraries

import re
import sys
import nltk
nltk.download(['stopwords', 'punkt', 'wordnet'])
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# ## Load Data

# + {"code_folding": []}
def load_data(database_filepath):
    """
    Input:
        1. database_filepath: the path of cleaned datasets
    Output:
        1. X: all messages
        2. y: category columns generated by cleaning process
        3. category_names: category columns' names
    Process:
        1. Read-in the datafrmae
        2. Select required datasets
        3. Generate category columns' names
    """
    
    # 1. Read-in dataframe
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, engine)
    
    # 2. Select required datasets
    X = df['message']
    y = df.iloc[:, 4:]
    
    # 3. Generate category columns' names
    category_names = y.columns
    return X, y, category_names
# -

# ## Tokenize

# + {"code_folding": [0]}
def tokenize(text):
    """
    Input:
        1. text: loaded-in messages
    Output:
        1. tokens: tokenized messages data
    Process:
        1. Define common parameters
        2. Normalize and remove punctuation
        3. Tokenize text
        4. Lemmatize and remove stop words
    """
    # Define common paras
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # Normalize and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', '', text.lower())
    
    # Tokenize text
    text = word_tokenize(text)
    
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    
    return tokens
# -

# ## Build Model

# + {"code_folding": [0]}
def build_model():
    """
    Input: None
    Output: cv model
    Process:
        1. Build pipeline for model
        2. Define grid search parameters
        3. Build GridSearchCV model
    """
    # Build Pipeline
    pipeline = Pipeline([
         ('vect', CountVectorizer(tokenizer=tokenize)),
         ('tfidf', TfidfTransformer()),
         ('mutclf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))])
    
    # Use grid search to find better parameters. 
    parameters = {
    'vect__max_features': (None, 5000, 10000),
    'tfidf__use_idf': (True, False)}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
# -

# ## Evaluate Model

# + {"code_folding": [0]}
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input:
        1. model:
        2. X_test:
        3. Y_test:
        4. category_name:
    Output: 
        1. Printed classification report
    Process:
        1. Predict using the trained model
        2. Use classification report to compare predicted and test data
    """
    
    # Predict use the trained model
    Y_pred = model.predict(X_test)
    
    # Report Model Effectiveness
    for i, col in enumerate(category_names):
        target_names = ['class 0', 'class 1', 'class 2']
        print(classification_report(Y_test[col].tolist(), list(Y_pred[:, i]), target_names=target_names))
# -

# ## Save Model

# + {"code_folding": []}
def save_model(model, model_filepath):
    """
    Input: 
        1. model: tranined model
        2. model_filepath: path of trained model file
    Output: None
    Process:
        1. save model file
    """
    # Save CV Model
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
# -

# ## Main

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
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

# ## Call Main

if __name__ == '__main__':
    main()
