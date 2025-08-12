# data_processing.py
# Author: Quentin Willems
# Project: Smart EV Charge recommendations leveraging AI and Elia open data
# Purpose: Centralized module for data loading, feature engineering and persistent file caching (for the web app).
# Dependencies:
#     - Available data in external PostgreSQL data base (data wrangling handeled in postgres_data_wrangler.sql)
#     - config.ini for configuration variables/paths

import psycopg2
import pandas as pd
import numpy as np
from psycopg2 import Error
import configparser
import os # Import to handle file paths

###################################
### 0. Configuration Variables ###
###################################

def load_config(config_file='config.ini'):
    """
    Loads configuration settings from a specified INI file.
    Assumes the config file is in the same directory as this script.
    
    Returns:
        configparser.ConfigParser: The loaded configuration object (with database parameters as one of its items)
    """
    # Get the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the config file
    config_path = os.path.join(script_dir, config_file)
    
    config = configparser.ConfigParser()
    config.read(config_path)
    if not config.sections():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found or is empty.")
    return config


#############################################################################
### 1. Connect to PostgreSQL Database and Load master data into DataFrame ###
#############################################################################

def load_data_from_postgres():
    """
    Connects to the PostgreSQL database using parameters from config.ini,
    loads the master data and performs initial data cleaning.

    Returns:
        pd.DataFrame: The loaded and pre-processed DataFrame (or None if an error occurs).
    """
    connection = None
    df_input = pd.DataFrame()
    query = "SELECT * FROM master_data;"

    try:
        # Load DB parameters from config.ini
        config = load_config()
        db_params = dict(config.items('Database'))
        db_params['port'] = int(db_params['port']) # Ensure port is an integer

        connection = psycopg2.connect(**db_params)
        print("Successfully connected to the database.")

        # Load master_data into a pandas DataFrame
        df_input = pd.read_sql_query(query, connection)

        # Set timestamp as the index for time series analysis
        df_input['timestamp'] = pd.to_datetime(df_input['timestamp'])
        df_input.set_index('timestamp', inplace=True)

        print("\nDataFrame Info:")
        print(df_input.info())
        print("\nFirst 5 rows:")
        print(df_input.head())
        print(f"\nDataFrame number of rows and columns: {df_input.shape}")

        # Check for missing values
        print("\nMissing values per column:")
        print(df_input.isnull().sum())
        
        return df_input

    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL or loading data:", error)
        return None

    finally:
        # Close database connection
        if connection:
            connection.close()
            print("\nDatabase connection closed.")

################################################################
### 2. Feature engineering and treatment of missing values   ###
################################################################

def engineer_features(df):
    """
    Performs all the feature engineering steps on the input DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
    
    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """
    if df is None or df.empty:
        print("Input DataFrame is empty or None. Skipping feature engineering.")
        return df

    # Work on a copy to avoid modifying the original DataFrame directly
    df_engineered = df.copy()
    
    # 2.1 Handle missing values
    def fill_missing_values_by_group(df_imputed, group_cols):
        columns_with_na = df_imputed.columns[df_imputed.isnull().any()].tolist()
        if not columns_with_na:
            print("No columns with missing values found. Imputation skipped.")
            return df_imputed
        print("Columns with missing values identified:", columns_with_na)
        for col in columns_with_na:
            print(f"  > Imputing missing values in '{col}'...")
            grouped_means = df_imputed.groupby(group_cols)[col].transform('mean')
            df_imputed[col] = df_imputed[col].fillna(grouped_means)
        return df_imputed

    df_engineered = fill_missing_values_by_group(df_engineered, ['month', 'weekday', 'hour'])
    final_na_count = df_engineered.isnull().sum().sum()
    print(f"\nTotal NA count in the final DataFrame after imputation: {final_na_count}")
    if final_na_count == 0:
        print("Successfully imputed all identified missing values.")
    else:
        print("Warning: Some missing values may still remain.")
    print(f"\nThe final DataFrame looks like this (with imputed values):\n{df_engineered.head()}")

    # 2.2 Electricity Market Price proxy
    df_engineered['MarketPriceProxy'] = np.where(
        df_engineered['avg_systemimbalance'] > 0,
        df_engineered['avg_positive_price'],
        df_engineered['avg_negative_price']
    )
    print("\nSuccessfully created 'MarketPriceProxy' column.")

    # 2.3 Photovoltaic (PV) production
    # Drop unnecessary aggregated views
    aggregated_regions = ['belgium', 'flanders', 'wallonia']
    cols_to_drop = [f'area_{region}_pv_production' for region in aggregated_regions] + [f'area_{region}_pv_capacity' for region in aggregated_regions]
    df_engineered.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Dropped aggregated PV columns: {cols_to_drop}")

    # Normalize PV production by calculating utilization rate (i.e ratio between observed production and measured capacity on average for a given hour)
    provinces = ['antwerp', 'eastflanders', 'flemishbrabant', 'hainaut', 'limburg', 'liège', 'luxembourg', 'namur', 'walloonbrabant', 'westflanders', 'brussels']
    cols_to_drop_pv = []
    for p in provinces:
        prod_col = f'area_{p}_pv_production'
        cap_col = f'area_{p}_pv_capacity'
        utilization_rate_col = prod_col.replace('_production', '_utilization_rate')
        df_engineered[utilization_rate_col] = np.where(df_engineered[cap_col] > 0, df_engineered[prod_col] / df_engineered[cap_col], 0)
        cols_to_drop_pv.extend([prod_col, cap_col])
    df_engineered.drop(columns=cols_to_drop_pv, inplace=True, errors='ignore')
    print("\nNormalized PV production features created as utilization rates.")
    
    # Create lagged features for PV
    new_pv_columns = [f'area_{p}_pv_utilization_rate' for p in provinces]
    lag_periods = [1, 6, 24]
    for col in new_pv_columns:
        for lag in lag_periods:
            df_engineered[f'{col}_lag_{lag}h'] = df_engineered[col].shift(periods=lag)
    print("Successfully created lagged features for PV utilization rates.")

    # 2.4 Wind production
    # Drop unnecessary aggregated views
    wind_federal = ['federal']
    agg_cols_to_drop_wind = [f'area_{region}_pv_production' for region in wind_federal] + [f'area_{region}_pv_capacity' for region in wind_federal]
    df_engineered.drop(columns=agg_cols_to_drop_wind, inplace=True, errors='ignore')
    print(f"Dropped aggregated PV columns: {agg_cols_to_drop_wind}")
    
    # Normalize wind production by calculating utilization rate (i.e ratio between observed production and measured capacity on average for a given hour)
    wind_regions = ['flanders', 'wallonia']
    cols_to_drop_wind = []
    for r in wind_regions:
        prod_col = f'area_{r}_wind_production'
        cap_col = f'area_{r}_wind_capacity'
        utilization_rate_col = prod_col.replace('_production', '_utilization_rate')
        df_engineered[utilization_rate_col] = np.where(df_engineered[cap_col] > 0, df_engineered[prod_col] / df_engineered[cap_col], 0)
        cols_to_drop_wind.extend([prod_col, cap_col])
    df_engineered.drop(columns=cols_to_drop_wind, inplace=True, errors='ignore')
    print("\nNormalized wind production features created as utilization rates.")
    
    # Create lagged features for wind
    new_wind_columns = [f'area_{r}_wind_utilization_rate' for r in wind_regions]
    for col in new_wind_columns:
        for lag in lag_periods:
            df_engineered[f'{col}_lag_{lag}h'] = df_engineered[col].shift(periods=lag)
    print("Successfully created lagged features for wind utilization rates.")

    # 2.5 Time Series features (cyclical and weekend features)
    df_engineered['hour'] = pd.to_numeric(df_engineered['hour'], errors='coerce').astype('Int64')
    df_engineered['weekday'] = pd.to_numeric(df_engineered['weekday'], errors='coerce').astype('Int64')
    df_engineered['calendar_week'] = pd.to_numeric(df_engineered['calendar_week'], errors='coerce').astype('Int64')
    df_engineered['hour_sin'] = np.sin(2 * np.pi * df_engineered['hour'] / 24)
    df_engineered['hour_cos'] = np.cos(2 * np.pi * df_engineered['hour'] / 24)
    df_engineered['weekday_sin'] = np.sin(2 * np.pi * df_engineered['weekday'] / 7)
    df_engineered['weekday_cos'] = np.cos(2 * np.pi * df_engineered['weekday'] / 7)
    df_engineered['is_weekend'] = ((df_engineered['weekday'] == 5) | (df_engineered['weekday'] == 6)).astype(int)
    print("Created cyclical and weekend features.")
    
     # 2.6 Create lagged features for MarketPriceProxy
    lag_periods = [1, 6, 24]
    for lag in lag_periods:
        df_engineered[f'MarketPriceProxy_lag_{lag}h'] = df_engineered['MarketPriceProxy'].shift(periods=lag)
    print("Successfully created lagged features for MarketPriceProxy.")
    # Lag 1h captures immediate short-term momentum or recent fluctuations.
    # Lag 6h reflects medium-term trends or intra-day cycles (e.g., morning vs. afternoon).
    # Lag 24h captures daily seasonality by referencing the same hour on the previous day.

    return df_engineered

if __name__ == '__main__':
    # This block runs only when data_processing.py is executed as a script
    df_raw = load_data_from_postgres()
    if df_raw is not None:
        df_processed = engineer_features(df_raw)
        if df_processed is not None:
            print("\nData processing complete. Final dataframe preview:")
            print(df_processed.head())

            # --- Save the processed DataFrame to a file for persistent caching ---
            try:
                # Load the path from the config file
                config = load_config()
                data_path = config['Paths']['data_path']
                cache_file = os.path.join(data_path, "processed_data.parquet")
            
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                df_processed.to_parquet(cache_file)
                print(f"Processed data saved to {cache_file} for deployment.")
            except (FileNotFoundError, KeyError) as e:
                print(f"Error loading path from config.ini: {e}. Cannot save data to a file.")
            except Exception as e:
                print(f"Error saving data to cache file: {e}")
