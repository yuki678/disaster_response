import sys
import numpy as np
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
import time

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    """Load data from sqlite database
    Returns:
        X - Training data
        Y - True label
        Category Names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message'].values
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    """Convert the input text into tokens"""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """Returns a ML model"""
    parameters_short = {
        'clf__estimator__n_estimators': [10, 20, 50],
        'clf__estimator__max_depth': [2, 5],
        'clf__estimator__min_samples_leaf':[1, 5, 10],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'vect__ngram_range': [(1, 1), (1, 2)]
    }
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    cv = GridSearchCV(pipeline, param_grid=parameters_short, cv=None, n_jobs=-1, verbose=10)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Predict on X_test and print the result against true label of Y_test"""
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=Y_test.columns)

    # print(classification_report(Y_test, Y_pred_df, target_names=category_names))
    for column in category_names:
        print(column)
        print(classification_report(Y_test[column], Y_pred_df[column]))
    


def save_model(model, model_filepath):
    """Save the model to a pickle file"""
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        start = time.process_time()
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Elapsed time: ', time.process_time() - start)
        
        print('Building model...')
        model = build_model()
        print('Elapsed time: ', time.process_time() - start)

        print('Training model...')
        model.fit(X_train, Y_train)
        print('Elapsed time: ', time.process_time() - start)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print('Elapsed time: ', time.process_time() - start)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Elapsed time: ', time.process_time() - start)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
