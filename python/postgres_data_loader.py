# postgres_data_loader.py
# Author: Quentin Willems
# Project: Smart EV Charge recommendations leveraging AI and Elia open data
# Purpose: Connect to PostgreSQL database and load the masterdata in a dataframe

import psycopg2
import pandas as pd
from psycopg2 import Error

###################################
### 0. Configuration Variables  ###
###################################

# Database connection parameters 
db_params = {
    'host': 'localhost',
    'port': '5432',
    'database': 'EV_AI_DB',
    'user': 'postgres',
    'password': 'postgrespassword'
}


#######################################################################
### 1. Connect to PostgreSQL DB and Load master data into DataFrame ###
#######################################################################


# SQL query to select all columns from master_data
query = "SELECT * FROM master_data;"

try:
    # Establish connection to PostgreSQL
    connection = psycopg2.connect(**db_params)
    print("Successfully connected to the database.")

    # Load master_data into a pandas DataFrame
    df_input = pd.read_sql_query(query, connection)

    # Set timestamp as the index for time series analysis
    df_input['timestamp'] = pd.to_datetime(df_input['timestamp'])
    df_input.set_index('timestamp', inplace=True)

    # Basic data exploration
    print("\nDataFrame Info:")
    print(df_input.info())
    print("\nFirst 5 rows:")
    print(df_input.head())
    print(f"\nDataFrame shape: {df_input.shape}")

    # Check for missing values
    print("\nMissing values per column:")
    print(df_input.isnull().sum())

except (Exception, Error) as error:
    print("Error while connecting to PostgreSQL or loading data:", error)

finally:
    # Close database connection
    if connection:
        connection.close()
        print("\nDatabase connection closed.")