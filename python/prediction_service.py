# prediction_script.py
# Author: Quentin Willems
# Purpose: Use a trained model to make 24-hour forecasts and recommend the best charging time.

import pandas as pd
import numpy as np
import joblib

# Set a pandas option to suppress the FutureWarning about downcasting
pd.set_option('future.no_silent_downcasting', True)


def load_model_and_features(model_path):
    """
    Loads the trained model and feature list from a joblib file.

    Args:
        model_path (str): The path to the saved joblib file.

    Returns:
        tuple: A tuple containing the loaded model and the list of features.
               Returns (None, None) if the file cannot be loaded.
    """
    try:
        data = joblib.load(model_path)
        print(f"Model and features successfully loaded from {model_path}.")
        return data['model'], data['features']
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please train a model first.")
        return None, None
    except KeyError:
        print(f"Error: The model file is corrupted. It should contain 'model' and 'features' keys.")
        return None, None

def create_future_features(start_date, features_list, last_known_data, historical_data):
    """
    Generates a DataFrame of future features for a 24-hour period.
    It populates lagged features using the last known data point and simulates daily patterns for others
    by taking the average of historical data for the given day of the year.

    Args:
        start_date (str): The date to start the prediction, in 'YYYY-MM-DD' format.
        features_list (list): The list of features required by the model.
        last_known_data (pd.DataFrame): The last DataFrame used for training to get initial lagged values.
        historical_data (pd.DataFrame): A DataFrame of historical data to get representative averages.

    Returns:
        pd.DataFrame: A DataFrame of features for the next 24 hours, ready for prediction.
    """
    # Create an empty DataFrame for the next 24 hours.
    prediction_range = pd.date_range(start=start_date, periods=24, freq='h')
    future_features = pd.DataFrame(index=prediction_range, columns=features_list)

    # Get the last known values from the training data for initialization
    last_known_values = last_known_data.iloc[-1]
    
    # Identify the day, month, and week of the prediction date to find historical averages
    prediction_day_of_week = prediction_range[0].weekday()
    prediction_month = prediction_range[0].month
    prediction_calendar_week = prediction_range[0].isocalendar().week

    # Filter historical data for the same day of the week, month, and calendar week
    historical_matches = historical_data[
        (historical_data.index.weekday == prediction_day_of_week) &
        (historical_data.index.month == prediction_month) &
        (historical_data.index.isocalendar().week.astype(int) == prediction_calendar_week)
    ]
    
    # If no matches are found, fall back to a simpler simulation
    if historical_matches.empty:
        print("Warning: No historical data found for the given day. Using a simpler simulation.")
        # Fallback simulation logic (similar to previous version)
        for feature in features_list:
            if 'avg_load' in feature:
                month_effect = np.sin(2 * np.pi * (future_features.index.month - 1) / 12) * 20
                daily_pattern = np.sin(2 * np.pi * future_features.index.hour / 24) * 50
                future_features[feature] = 100 + month_effect + daily_pattern
            elif 'avg_systemimbalance' in feature:
                month_effect = np.cos(2 * np.pi * (future_features.index.month - 1) / 12) * 5
                daily_pattern = np.cos(2 * np.pi * future_features.index.hour / 24) * 10
                future_features[feature] = month_effect + daily_pattern
            elif feature in last_known_values:
                future_features[feature] = last_known_values[feature]
            else:
                future_features[feature] = 0
    else:
        # Calculate the average values for each hour of the day
        daily_average = historical_matches.groupby(historical_matches.index.hour).mean()

        # Populate future features using these historical averages
        for feature in features_list:
            if feature in daily_average.columns:
                # Use the averaged data for the 24 hours
                future_features[feature] = daily_average[feature].values
            elif feature in last_known_values:
                # Use the last known data for initial lagged values
                future_features[feature] = last_known_values[feature]
            else:
                # Populate time-based features and others that don't have a direct average
                if feature == 'hour_sin':
                    future_features[feature] = np.sin(2 * np.pi * future_features.index.hour / 24)
                elif feature == 'hour_cos':
                    future_features[feature] = np.cos(2 * np.pi * future_features.index.hour / 24)
                elif feature == 'weekday_sin':
                    future_features[feature] = np.sin(2 * np.pi * future_features.index.weekday / 7)
                elif feature == 'weekday_cos':
                    future_features[feature] = np.cos(2 * np.pi * future_features.index.weekday / 7)
                elif feature == 'is_weekend':
                    future_features[feature] = future_features.index.weekday.isin([5, 6]).astype(int)
                elif feature == 'year':
                    future_features[feature] = future_features.index.year
                elif feature == 'month':
                    future_features[feature] = future_features.index.month
                elif feature == 'calendar_week':
                    future_features[feature] = future_features.index.isocalendar().week.astype(int)
                else:
                    future_features[feature] = 0

    return future_features.fillna(0) # Fill any remaining NaNs


def make_24_hour_prediction(model, features_df, features_list):
    """
    Makes an iterative 24-hour prediction using the trained model.

    Args:
        model: The trained model object.
        features_df (pd.DataFrame): A DataFrame of features for the 24-hour period,
                                    with initial lagged values populated.
        features_list (list): The list of features the model expects.

    Returns:
        list: A list of 24 predicted prices.
    """
    predictions = []
    # Create a working copy of the feature DataFrame
    predictions_features_df = features_df.copy()

    for i in range(24):
        # Prepare the features for the current hour, ensuring column order matches the model
        features_for_hour = predictions_features_df.iloc[[i]][features_list]
        
        # Make a prediction for the current hour
        prediction = model.predict(features_for_hour)
        predictions.append(prediction[0])
        
        # Update lagged features for the next hour with the new prediction
        if i < 23:
            # Update MarketPriceProxy lags
            predictions_features_df.loc[predictions_features_df.index[i+1], 'MarketPriceProxy_lag_1h'] = prediction[0]
            if i >= 5:
                predictions_features_df.loc[predictions_features_df.index[i+1], 'MarketPriceProxy_lag_6h'] = predictions[i-5]
            if i >= 23:
                predictions_features_df.loc[predictions_features_df.index[i+1], 'MarketPriceProxy_lag_24h'] = predictions[i-23]
            
            # Update other lagged features
            for lag_hours in [1, 6, 24]:
                lag_suffix = f'_lag_{lag_hours}h'
                for feature in features_list:
                    if feature.endswith(lag_suffix) and not feature.startswith('MarketPriceProxy'):
                        base_feature = feature[:-len(lag_suffix)]
                        if base_feature in predictions_features_df.columns:
                            # Use a value from the correct previous time step
                            if i - (lag_hours - 1) >= 0:
                                predictions_features_df.loc[predictions_features_df.index[i+1], feature] = predictions_features_df.loc[predictions_features_df.index[i-(lag_hours-1)], base_feature]
    
    return predictions

def find_best_charging_hours(predictions, num_hours=3):
    """
    Finds the best charging hours based on the lowest predicted prices.
    
    Args:
        predictions (list): A list of 24 predicted prices.
        num_hours (int): The number of hours to recommend.
        
    Returns:
        list: A list of tuples, each containing the hour (0-23) and its price, sorted by price.
    """
    # Combine predictions with their corresponding hours
    hourly_predictions = [(predictions[i], i) for i in range(24)]
    # Sort by price in ascending order
    hourly_predictions.sort(key=lambda x: x[0])
    # Return the top 'num_hours' recommendations
    return hourly_predictions[:num_hours]

if __name__ == '__main__':
    # --- Example Usage ---
    
    # Define paths and user input
    MODEL_PATH = 'ev_charge_model.joblib'
    USER_PREDICTION_DATE = '2025-08-28' # Example date
    
    # For this example, we create a dummy DataFrame to simulate the last known data
    # In a real app, this would be the last row of your full historical dataset.
    # Note: In a real scenario, you'd load a much larger historical dataset.
    dummy_historical_data = pd.DataFrame(
        data=np.random.rand(48, 75) * 100, # 48 hours of data
        index=pd.date_range('2025-08-27', periods=48, freq='h'),
        columns=['MarketPriceProxy', 'avg_load', 'avg_systemimbalance', 'avg_netregulationvolume', 'area_antwerp_pv_utilization_rate', 'area_eastflanders_pv_utilization_rate', 'area_flemishbrabant_pv_utilization_rate', 'area_hainaut_pv_utilization_rate', 'area_limburg_pv_utilization_rate', 'area_liège_pv_utilization_rate', 'area_luxembourg_pv_utilization_rate', 'area_namur_pv_utilization_rate', 'area_walloonbrabant_pv_utilization_rate', 'area_westflanders_pv_utilization_rate', 'area_brussels_pv_utilization_rate', 'area_federal_wind_utilization_rate', 'area_flanders_wind_utilization_rate', 'area_wallonia_wind_utilization_rate', 'area_antwerp_pv_utilization_rate_lag_1h', 'area_antwerp_pv_utilization_rate_lag_6h', 'area_antwerp_pv_utilization_rate_lag_24h', 'area_eastflanders_pv_utilization_rate_lag_1h', 'area_eastflanders_pv_utilization_rate_lag_6h', 'area_eastflanders_pv_utilization_rate_lag_24h', 'area_flemishbrabant_pv_utilization_rate_lag_1h', 'area_flemishbrabant_pv_utilization_rate_lag_6h', 'area_flemishbrabant_pv_utilization_rate_lag_24h', 'area_hainaut_pv_utilization_rate_lag_1h', 'area_hainaut_pv_utilization_rate_lag_6h', 'area_hainaut_pv_utilization_rate_lag_24h', 'area_limburg_pv_utilization_rate_lag_1h', 'area_limburg_pv_utilization_rate_lag_6h', 'area_limburg_pv_utilization_rate_lag_24h', 'area_liège_pv_utilization_rate_lag_1h', 'area_liège_pv_utilization_rate_lag_6h', 'area_liège_pv_utilization_rate_lag_24h', 'area_luxembourg_pv_utilization_rate_lag_1h', 'area_luxembourg_pv_utilization_rate_lag_6h', 'area_luxembourg_pv_utilization_rate_lag_24h', 'area_namur_pv_utilization_rate_lag_1h', 'area_namur_pv_utilization_rate_lag_6h', 'area_namur_pv_utilization_rate_lag_24h', 'area_walloonbrabant_pv_utilization_rate_lag_1h', 'area_walloonbrabant_pv_utilization_rate_lag_6h', 'area_walloonbrabant_pv_utilization_rate_lag_24h', 'area_westflanders_pv_utilization_rate_lag_1h', 'area_westflanders_pv_utilization_rate_lag_6h', 'area_westflanders_pv_utilization_rate_lag_24h', 'area_brussels_pv_utilization_rate_lag_1h', 'area_brussels_pv_utilization_rate_lag_6h', 'area_brussels_pv_utilization_rate_lag_24h', 'area_federal_wind_utilization_rate_lag_1h', 'area_federal_wind_utilization_rate_lag_6h', 'area_federal_wind_utilization_rate_lag_24h', 'area_flanders_wind_utilization_rate_lag_1h', 'area_flanders_wind_utilization_rate_lag_6h', 'area_flanders_wind_utilization_rate_lag_24h', 'area_wallonia_wind_utilization_rate_lag_1h', 'area_wallonia_wind_utilization_rate_lag_6h', 'area_wallonia_wind_utilization_rate_lag_24h', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'is_weekend', 'year', 'month', 'calendar_week', 'MarketPriceProxy_lag_1h', 'MarketPriceProxy_lag_6h', 'MarketPriceProxy_lag_24h', 'avg_load_lag_1h', 'avg_load_lag_24h', 'avg_systemimbalance_lag_1h', 'avg_systemimbalance_lag_24h', ]
    )
    dummy_last_known_data = dummy_historical_data.iloc[-1].to_frame().T

    # 1. Load the model and features
    model, features_list = load_model_and_features(MODEL_PATH)

    if model is not None:
        print(f"\nMaking prediction for {USER_PREDICTION_DATE}...")

        # 2. Prepare the input features
        future_features = create_future_features(
            start_date=USER_PREDICTION_DATE,
            features_list=features_list,
            last_known_data=dummy_last_known_data,
            historical_data=dummy_historical_data
        )

        # 3. Make the 24-hour forecast
        predicted_prices = make_24_hour_prediction(model, future_features, features_list)

        # 4. Find the best charging hour
        best_hours = find_best_charging_hours(predicted_prices)

        # 5. Display the results
        print("\n--- 24-Hour Price Forecast ---")
        for i, price in enumerate(predicted_prices):
            print(f"Hour {i:02d}: {price:.2f} €/MWh")
        
        print("\n--- Recommendation ---")
        print(f"The best hours to charge your EV on {USER_PREDICTION_DATE} are:")
        for price, hour in best_hours:
            print(f"  - Hour {hour:02d}: {price:.2f} €/MWh")