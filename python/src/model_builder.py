# model_builder.py
# Author: Quentin Willems
# Project: Smart EV Charge recommendations leveraging AI and Elia open data
# Purpose: Train different models (Random Forest, XGBoost, ARIMA and TCN) to predict 'MarketPriceProxy'. Select the best one then save it.
# Dependencies:
#     - data_processing.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os # Import the 'os' module to handle file paths
import configparser

# Import XGBoost
from xgboost import XGBRegressor

# Import Darts libraries for time series modeling
from darts import TimeSeries
from darts.models import AutoARIMA, TCNModel

# --- Import the data_processing module and its functions ---
from data_processing import load_data_from_postgres, engineer_features

def load_config(config_file='config.ini'):
    """
    Loads configuration settings from a specified INI file.
    Assumes the config file is in the same directory as the script.
    """
    # Get the absolute path to the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the config file
    config_path = os.path.join(script_dir, config_file)
    
    config = configparser.ConfigParser()
    config.read(config_path)
    if not config.sections():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found or is empty.")
    return config

def prepare_features_and_target(df):
    """
    Prepares features and target variable for modeling.
    Ensures 'timestamp' is a datetime index and converts relevant columns to numeric.
    
    Args:
        df (pd.DataFrame): The pre-processed DataFrame.

    Returns:
        tuple: X (features DataFrame), y (target Series), features_list (list of feature names)
    """
    if df is None:
        return None, None, None
    
    # Ensure 'timestamp' is datetime object and set as index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
        else:
            print("Error: 'timestamp' column not found and index is not DatetimeIndex. Cannot proceed.")
            return None, None, None

    # Set a consistent hourly frequency to ensure there are no missing time steps
    df = df.asfreq('H')

    # Convert relevant columns to numeric if they are objects
    for col in ['weekday', 'hour', 'calendar_week', 'month', 'year']:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    target = 'MarketPriceProxy'
    
    # --- UPDATED: Feature list to reflect the changes in data_processing.py ---
    features_list = [
        'avg_load', 'avg_systemimbalance', 'avg_netregulationvolume',
        'area_antwerp_pv_utilization_rate', 'area_eastflanders_pv_utilization_rate',
        'area_flemishbrabant_pv_utilization_rate', 'area_hainaut_pv_utilization_rate',
        'area_limburg_pv_utilization_rate', 'area_liège_pv_utilization_rate',
        'area_luxembourg_pv_utilization_rate', 'area_namur_pv_utilization_rate',
        'area_walloonbrabant_pv_utilization_rate', 'area_westflanders_pv_utilization_rate',
        'area_brussels_pv_utilization_rate',
        'area_flanders_wind_utilization_rate', 'area_wallonia_wind_utilization_rate',
        'area_antwerp_pv_utilization_rate_lag_1h', 'area_antwerp_pv_utilization_rate_lag_6h',
        'area_antwerp_pv_utilization_rate_lag_24h',
        'area_eastflanders_pv_utilization_rate_lag_1h', 'area_eastflanders_pv_utilization_rate_lag_6h',
        'area_eastflanders_pv_utilization_rate_lag_24h',
        'area_flemishbrabant_pv_utilization_rate_lag_1h', 'area_flemishbrabant_pv_utilization_rate_lag_6h',
        'area_flemishbrabant_pv_utilization_rate_lag_24h',
        'area_hainaut_pv_utilization_rate_lag_1h', 'area_hainaut_pv_utilization_rate_lag_6h',
        'area_hainaut_pv_utilization_rate_lag_24h',
        'area_limburg_pv_utilization_rate_lag_1h', 'area_limburg_pv_utilization_rate_lag_6h',
        'area_limburg_pv_utilization_rate_lag_24h',
        'area_liège_pv_utilization_rate_lag_1h', 'area_liège_pv_utilization_rate_lag_6h',
        'area_liège_pv_utilization_rate_lag_24h',
        'area_luxembourg_pv_utilization_rate_lag_1h', 'area_luxembourg_pv_utilization_rate_lag_6h',
        'area_luxembourg_pv_utilization_rate_lag_24h',
        'area_namur_pv_utilization_rate_lag_1h', 'area_namur_pv_utilization_rate_lag_6h',
        'area_namur_pv_utilization_rate_lag_24h',
        'area_walloonbrabant_pv_utilization_rate_lag_1h', 'area_walloonbrabant_pv_utilization_rate_lag_6h',
        'area_walloonbrabant_pv_utilization_rate_lag_24h',
        'area_westflanders_pv_utilization_rate_lag_1h', 'area_westflanders_pv_utilization_rate_lag_6h',
        'area_westflanders_pv_utilization_rate_lag_24h',
        'area_brussels_pv_utilization_rate_lag_1h', 'area_brussels_pv_utilization_rate_lag_6h',
        'area_brussels_pv_utilization_rate_lag_24h',
        'area_flanders_wind_utilization_rate_lag_1h', 'area_flanders_wind_utilization_rate_lag_6h',
        'area_flanders_wind_utilization_rate_lag_24h',
        'area_wallonia_wind_utilization_rate_lag_1h', 'area_wallonia_wind_utilization_rate_lag_6h',
        'area_wallonia_wind_utilization_rate_lag_24h',
        'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'is_weekend',
        'year', 'month', 'calendar_week',
        'MarketPriceProxy_lag_1h', 'MarketPriceProxy_lag_6h', 'MarketPriceProxy_lag_24h',
        'avg_load_lag_1h', 'avg_load_lag_24h', 'avg_systemimbalance_lag_1h', 'avg_systemimbalance_lag_24h',
    ]

    # --- Filter out features that might not exist in the new data_processing output ---
    features_list = [f for f in features_list if f in df.columns]

    X = df[features_list].copy()
    y = df[target].copy()

    # Handle NaNs from lagged features at the beginning of the series
    combined_df = pd.concat([X, y], axis=1)
    # The new `.asfreq('H')` can create NaNs, so we need to fill them.
    # We use ffill and bfill to carry forward/backward known values.
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
    combined_df.dropna(inplace=True)

    X = combined_df[features_list]
    y = combined_df[target]

    return X, y, features_list

def train_compare_and_save_best_model(df):
    """
    Prepares data, trains and compares multiple models,
    evaluates them, and saves the best-performing model.

    Args:
        df (pd.DataFrame): The pre-processed DataFrame from feature_engineering.py.

    Returns:
        tuple: (best trained model, list of features used for training)
    """
    print("\n--- Starting Model Training and Comparison ---")

    X, y, features_list = prepare_features_and_target(df.copy())
    
    if X is None or y is None:
        print("Data preparation failed. Cannot train models.")
        return None, None

    # Time-based Train/Test Split
    train_size = int(len(X) * 0.8)
    X_train_df, X_test_df = X.iloc[:train_size], X.iloc[train_size:]
    y_train_df, y_test_df = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"Train set size: {len(X_train_df)} samples")
    print(f"Test set size: {len(X_test_df)} samples")
    print(f"Number of features: {len(features_list)}")

    # Model Definition Dictionary
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1,
                                 early_stopping_rounds=10, eval_metric='rmse'),
        'DartsAutoARIMA': AutoARIMA(),
        'DartsTCN': TCNModel(
            input_chunk_length=24 * 7, # Use a week's worth of data as input
            output_chunk_length=24,   # Predict the next 24 hours
            n_epochs=1,               # Keep epochs low for a faster demonstration
            random_state=42,
            kernel_size=5,
            num_filters=4,
            dropout=0.1,
            optimizer_kwargs={"lr": 0.001},
        )
    }

    # Store performance metrics
    performance = {}
    best_model_name = ''
    best_score = float('inf')
    best_model = None

    # Train and evaluate models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        y_pred = None
        
        if name in ['RandomForest', 'XGBoost']:
            # Scikit-learn style models
            if name == 'XGBoost':
                model.fit(X_train_df, y_train_df,
                          eval_set=[(X_test_df, y_test_df)],
                          verbose=False)
            else: # RandomForest
                model.fit(X_train_df, y_train_df)
            y_pred = model.predict(X_test_df)
        
        elif name.startswith('Darts'):
            # Darts requires TimeSeries objects
            # Combine train and test data for easy creation of series
            full_y_series = TimeSeries.from_series(pd.concat([y_train_df, y_test_df]))
            full_X_series = TimeSeries.from_dataframe(pd.concat([X_train_df, X_test_df]))
            
            # Re-split the TimeSeries objects
            train_target_series = full_y_series[:train_size]
            test_target_series = full_y_series[train_size:]
            train_covariate_series = full_X_series[:train_size]
            # Darts models need the full covariate series to make predictions over the test set, as they need the past context.
            full_covariate_series = full_X_series
            
            if name == 'DartsAutoARIMA':
                # AutoARIMA does not use covariates directly
                model.fit(train_target_series)
                y_pred_series = model.predict(len(test_target_series))
            
            elif name == 'DartsTCN':
                # TCN uses both target and past covariates.
                model.fit(
                    series=train_target_series, 
                    past_covariates=train_covariate_series, 
                    verbose=False
                )
                y_pred_series = model.predict(
                    n=len(test_target_series),
                    series=train_target_series,
                    past_covariates=full_covariate_series
                )

            y_pred = y_pred_series.values().flatten()

        if y_pred is not None:
            mae = mean_absolute_error(y_test_df, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_df, y_pred))
            performance[name] = {'MAE': mae, 'RMSE': rmse}
            
            print(f"{name} Evaluation:")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")

            if rmse < best_score:
                best_score = rmse
                best_model_name = name
                best_model = model

    print("\n--- Final Model Comparison ---")
    for name, metrics in performance.items():
        print(f"{name}: RMSE = {metrics['RMSE']:.2f}, MAE = {metrics['MAE']:.2f}")
    
    print(f"\n🏆 The best performing model is: {best_model_name}")

    # --- Save the best model and features for prediction using config values ---
    try:
        config = load_config()
        model_path = config['Paths']['model_path']
        model_filename = config['Paths']['model_filename']
        
        # Create the directory if it doesn't exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print(f"Created directory: {model_path}")
        
        full_path = os.path.join(model_path, model_filename)
        
        joblib.dump({'model': best_model, 'features': features_list}, full_path)
        print(f"\nBest model and features saved to {full_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Cannot save the model.")
        print("Please ensure the 'config.ini' file is in the same folder as the running script and formatted correctly.")
    except KeyError as e:
        print(f"Error: Missing key in 'config.ini' - {e}.")
        print("Please ensure the 'config.ini' file has a '[Paths]' section with 'model_path' and 'model_filename' keys.")

    return best_model, features_list


if __name__ == "__main__":
    # The new execution block now loads and processes the data directly
    # First, load the raw data from PostgreSQL
    df_raw = load_data_from_postgres()

    if df_raw is not None:
        # Then, engineer the features
        df_processed = engineer_features(df_raw)
        trained_model, used_features = train_compare_and_save_best_model(df_processed.copy())
    else:
        print("Script cannot run without a valid DataFrame.")
    print("\n--- Model Training Script Finished ---")