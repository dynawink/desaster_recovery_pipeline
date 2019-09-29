#!/usr/bin/python3
"""This modules is part of the desaster recovery pipeline.
It loads messages and their categories from CSV, cleans it, and saves it to an
SQLite data base.
"""

import sys
import pandas as pd
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    """Load data from CSV files.

    The two CSV files are loaded and merged on column 'id'.

    messages_filepath   -- file path for messages
    categories_filepath -- file path for categories
    return              -- data frame
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left', left_on='id', right_on='id')
    return df


def split_categories(df):
    """Convert categories column to multiple columns.

    The categories column contains multiple categories and their values. Replace
    it by mulitple columns containing the values only.

    df     -- data frame
    return -- data frame with splitted categories
    """

    # split categories column and create data frame
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True))
    # get first row
    row = categories.loc[0, :]
    # extract column names
    category_colnames = [r[:-2] for r in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.partition('-')[2]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # make sure we only have values 0 and 1
    categories[categories > 0] = 1

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original data frame with the new `categories` data frame
    df = df.join(categories, sort=False)

    return df


def remove_duplicates(df):
    """Drop duplicates from the data frame.

    df     -- data frame with duplicates
    return -- data frame without duplicates
    """
    # drop duplicates
    df = df.drop_duplicates()
    return df


def clean_data(df):
    """Clean data.

    df     -- data frame with unclean data
    return -- data frame with cleaned data
    """
    # split the cateogories column
    df = split_categories(df)
    # remove duplicates
    df = remove_duplicates(df)
    return df


def save_data(df, database_filename):
    """Save data to SQLite database

    df                -- data frame
    database_filename -- file path of the the data base
    """

    # create data base engine
    engine = sqlalchemy.create_engine('sqlite:///' + database_filename)
    # drop preexisting messages table
    conn = engine.connect()
    conn.execute('DROP TABLE IF EXISTS messages')
    # create messages table
    df.to_sql('messages', engine, index=False)
    conn.close()
    engine.dispose()


def main():
    """Entry point.

    The program takes 3 command line arguments:

    - the path of the CSV file with the messages
    - the path of the CSV file with the categories of the messages
    - the path of the SQLite database to create or update
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
