# Disaster Response Pipeline

This project is part of the Udacity Nanodegree Data Science.

It demonstrates how to train a classifier for text messages and to provide an
online application to classify new messages.

The raw data comes in files data/messages.csv and data/categories.csv.
It contains real world messages collected in disaster scenarios and their
classification.

The project was conducted in two steps. First the data preparation and the
classification were conducted in Jupyter notebooks. Than the results were
converted into standalone Python programs.

# Jupyter notebooks

ETL Pipeline Preparation.ipynb

* reads the CSV files
* separates the categories in separate columns with values 0, 1
* removes dumplicats
* saves the data as an SQLite database (data/DisasterResponse.db).

ML Pipeline Preparation.ipynb

* reads the database
* trains different classification models
* uses grid search to optimize the model parameters
* provides an evalution report
* saves the classification model as pickle file (./models/classifier.pkl)

# Python code

The program src/process\_data.py

* reads the CSV files
* separates the categories in separate columns with values 0, 1
* removes dumplicats
* saves the data as an SQLite database (data/DisasterResponse.db).

The program src/train\_classifier.py

* reads the database
* trains a classification model
* provides an evalution report
* saves the classification model as pickle file (./models/classifier.pkl)

The program src/run.py

* reads the database and the classification model
* provides a web page at http://0.0.0.0:3001/

The web page

* displays statistics for the original data set
* displays the classification for user provided messages

# Software dependencies

For running the Python code on Debian Bullseye the following packages have to be
installed:

    apt-get install python3-flask python3-plotly python3-pandas python3-nltk \
    python3-sklearn python3-sqlalchemy

To run the Jupyter notebooks additionally install:

    apt-get install jupyter
