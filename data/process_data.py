"""
Preprocessing of Data
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)
Sample Script Syntax:
> python process_data.py <path to messages csv file> <path to categorie csv file> <path to sqllite  destination db>
Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db
Arguments Description:
    1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response_db.db)
"""

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load Messages Data with Categories Function
    
    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> Combined data containing messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    print(df.shape)

    return df
       
    
def clean_data(df):
    """
    Clean Categories Data Function
    
    Arguments:
        df -> Combined data containing messages and categories
    Outputs:
        df -> Combined data containing messages and categories with categories cleaned up
    """
    
    # Split the categories
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # Fix the categories columns name
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    print(df.shape)
    
    return df


def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        db_filename -> Path to SQLite destination database
    """
    
    engine = create_engine(f'sqlite:///{database_filename}')
    table_name = database_filename.replace('.db', '') + '_table'
    df.to_sql(table_name, con=engine, index=False, if_exists='replace')
      

def main():
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this
    function:
        1) Load Messages Data with Categories
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """
    
    # Print the system arguments
    # print(sys.argv)
    
    # Execute the ETL pipeline if the count of arguments is matching to 4
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print("Please provide the arguments correctly: \nSample Script Execution:\n\
> python process_data.py messages.csv categories.csv disaster_response.db \n\
Arguments Description: \n\
1) Path to the CSV file containing messages (e.g. messages.csv)\n\
2) Path to the CSV file containing categories (e.g. categories.csv)\n\
3) Path to SQLite destination database (e.g. disaster_response.db)")


if __name__ == '__main__':
    main()
