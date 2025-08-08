# prediction_service.py
# Author: Quentin Willems
# Project: Smart EV Charge recommendations leveraging AI and Elia open data
# Purpose: Predict a full day's market price, recommend best charging hours and provide contextual insights.
# Dependencies:
#     - data_processing.py (for initial data generation)

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
import os
import configparser

# Suppress joblib-related user warnings
warnings.filterwarnings("ignore", message="The feature names should match")

###################################
### 0. Configuration Variables ###
###################################

def load_config(config_file='config.ini'):
    """
    Loads configuration settings from a specified INI file.
    Assumes the config file is in the same directory as the script.
    
    Returns:
        configparser.ConfigParser: The loaded configuration object.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_file)
    
    config = configparser.ConfigParser()
    config.read(config_path)
    if not config.sections():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found or is empty.")
    return config

def load_data_from_file():
    """
    Loads the pre-computed data from the .parquet file for persistent caching (avoid connecting to DB again to align the app deployment strategy)
    
    Returns:
        pd.DataFrame: The loaded and processed DataFrame, or None if an error occurs.
    """
    try:
        config = load_config()
        data_path = config['Paths']['data_path']
        cache_file = os.path.join(data_path, "processed_data.parquet")

        if not os.path.exists(cache_file):
            print(
                "Error: Data file not found! Please run `data_processing.py` locally first "
                "to generate the `processed_data.parquet` file."
            )
            return None
        
        print("Loading pre-computed data from local file...")
        df_processed = pd.read_parquet(cache_file)
        return df_processed
    
    except KeyError:
        print("Error: Missing 'data_path' in the 'Paths' section of `config.ini`.")
        return None
    except Exception as e:
        print(f"Error loading the data file: {e}")
        return None

def generate_recommendation_insights(best_hour, month):
    """
    Generates a contextual text insight based on the predicted best charging hour and month.

    Args:
        best_hour (int): The best predicted hour for charging (0-23).
        month (int): The month of the prediction (1-12).

    Returns:
        str: A formatted string with a title and explanation.
    """
    # A. Summer Period (May - August)
    # Primary driver: High solar energy production in midday.
    if month in [5, 6, 7, 8]:
        if 10 <= best_hour <= 16:
            # Midday is the core opportunity due to high solar output.
            return f"**Solar-Powered Charge!**\n\n" \
                   f"In summer, the cheapest electricity is often in the middle of the day. This is when solar energy production is at its peak in Belgium, flooding the grid with green power and lowering prices. Take advantage of this surplus to charge at a lower cost!"
        else:
            # Generic advice for low-price moments outside the solar peak.
            return f"**Off-Peak Savings!**\n\n" \
                   f"Even outside of midday, other low-cost hours can appear in summer. Our prediction indicates this is the most economical window for today, driven by a combination of lower demand and favorable weather conditions."

    # B. Winter Period (November - February)
    # Primary driver: Low demand at night and potential for wind energy.
    elif month in [11, 12, 1, 2]:
        if 1 <= best_hour <= 9:
            # Late night/early morning is the historical sweet spot for low prices.
            return f"**Overnight Power-Up!**\n\n" \
                   f"In winter, the best time to charge is often during the night or early morning. This is when demand on the national grid is at its lowest, leading to a natural drop in prices. High wind power generation during these hours can further amplify the savings."
        else:
            # Advice for rare low-price moments during the day.
            return f"**Daytime Opportunity!**\n\n" \
                   f"While prices are typically highest in winter, our analysis shows a low-cost window during the day. This could be due to a sudden increase in wind energy or a temporary dip in industrial demand. Seize the opportunity!"

    # C. Transition Periods (March - April & September - October)
    # Primary driver: A mix of solar and wind, leading to variable price profiles.
    elif month in [3, 4, 9, 10]:
        if 10 <= best_hour <= 15:
            # Midday is best on sunny days due to solar.
            return f"**Midday Opportunity!**\n\n" \
                   f"In spring and autumn, the best times to charge can vary with the weather. A sunny day will favor low-cost charging in the afternoon thanks to solar power. Our prediction accounts for these factors!"
        elif 23 <= best_hour or best_hour <= 4:
            # Late night/early morning is best on windier days or when demand is low.
            return f"**Late Night Special!**\n\n" \
                   f"In spring and autumn, charging times can vary. A day with high winds may offer great opportunities late in the evening or during the night. Our prediction takes these factors into account!"
        else:
            # Fallback for any other hour in the transition period.
            return f"**Opportunistic Charging!**\n\n" \
                   f"During the transitional seasons, the best charging times can vary based on weather and demand. Our prediction helps you find the most economical window for today!"

    # D. Default Case (Error or unexpected month)
    else:
        return f"**Smart Charging Recommendation!**\n\n" \
               f"Our analysis indicates that this is the most economical time to charge your vehicle today."


def get_historical_averages(historical_df, date_to_predict, features_to_average):
    """
    Calculates the hourly average of specified features from historically similar days.
    A 'similar day' is defined as having the same month and weekday.

    Args:
        historical_df (pd.DataFrame): The pre-processed historical data.
        date_to_predict (datetime): The date for which to predict.
        features_to_average (list): A list of feature names to calculate averages for.

    Returns:
        pd.DataFrame: A DataFrame with the hourly average values for each feature.
    """
    # Filter historical data for the same month and weekday
    historical_df['weekday'] = historical_df.index.weekday
    historical_df['month'] = historical_df.index.month

    # Find the month and weekday of the target date
    target_month = date_to_predict.month
    target_weekday = date_to_predict.weekday()

    similar_days_df = historical_df[
        (historical_df['month'] == target_month) &
        (historical_df['weekday'] == target_weekday)
    ]

    if similar_days_df.empty:
        print(f"Warning: No historical data found for {date_to_predict.strftime('%B')} and {date_to_predict.strftime('%A')}. Returning zeros.")
        # If no similar days, return a DataFrame of zeros
        hourly_averages = pd.DataFrame(0.0, index=range(24), columns=features_to_average)
        hourly_averages.index.name = 'hour'
        return hourly_averages

    # Group by hour and calculate the mean for the specified features
    hourly_averages = similar_days_df.groupby(similar_days_df.index.hour)[features_to_average].mean()
    hourly_averages.index.name = 'hour'
    return hourly_averages


def create_time_features(dt):
    """
    Generates time-based features from a datetime object.

    Args:
        dt (datetime): The datetime object to process.

    Returns:
        dict: A dictionary of time-based features.
    """
    features = {
        'year': dt.year,
        'month': dt.month,
        'weekday': dt.weekday(),
        'hour': dt.hour,
        'calendar_week': dt.isocalendar()[1],
        'is_weekend': 1 if dt.weekday() >= 5 else 0,
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24.0),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24.0),
        'weekday_sin': np.sin(2 * np.pi * dt.weekday() / 7.0),
        'weekday_cos': np.cos(2 * np.pi * dt.weekday() / 7.0),
    }
    return features


def get_prediction_interval(predicted_price, rmse, confidence_level=0.95):
    """
    Calculates a prediction interval based on the model's RMSE.

    Args:
        predicted_price (float): The point prediction.
        rmse (float): The Root Mean Squared Error of the model.
        confidence_level (float): The desired confidence level for the interval (e.g., 0.95 for 95%).

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if np.isnan(predicted_price):
        return (np.nan, np.nan)
    
    # Z-score for the given confidence level. For 95%, it's 1.96.
    z_score = 1.96
    if confidence_level == 0.90:
        z_score = 1.645
    elif confidence_level == 0.99:
        z_score = 2.576
    
    margin = z_score * rmse
    lower_bound = predicted_price - margin
    upper_bound = predicted_price + margin
    return (lower_bound, upper_bound)


def predict_and_recommend_charging(date_str, region):
    """
    Loads the best trained model, predicts the market price for each hour of a given date,
    and recommends the best hours to charge based on the lowest predicted prices.
    Includes a sanity check against historical median prices.

    Args:
        date_str (str): The date to predict in 'YYYY-MM-DD' format.
        region (str): The region for which to provide a recommendation.
                      (e.g., 'antwerp', 'brussels', 'liège', etc.)

    Returns:
        tuple: (list of dicts with hourly predictions, dict with recommendations)
    """
    try:
        config = load_config()
        model_path = config['Paths']['model_path']
        model_filename = config['Paths']['model_filename']
        model_file = os.path.join(model_path, model_filename)

        # Load the trained model, feature list, and RMSE
        model_data = joblib.load(model_file)
        model = model_data['model']
        features_list = model_data['features']
        
        try:
            rmse = model_data['rmse']
            print(f"Model RMSE successfully loaded: {rmse:.2f}")
        except KeyError:
            rmse = 41.55  # Fallback to the hardcoded value if not found
            print(f"Warning: 'rmse' not found in joblib file. Using default value: {rmse:.2f}")
            
        print("Model and features successfully loaded.")
    except (FileNotFoundError, KeyError):
        print("Error: Model file or path not found. Please train a model first and check your config.ini.")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    # Load historical data from the pre-computed file
    print("Loading historical data from file...")
    df_input = load_data_from_file()
    if df_input is None:
        print("Failed to load historical data from file. Cannot generate predictions.")
        return None, None
    print("Historical data successfully loaded.")

    # Get the target date
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    print(f"\n--- Generating predictions for {target_date.strftime('%Y-%m-%d')} ---")
    
    # Identify features for which to calculate historical averages
    # These are all non-time and non-autoregressive features.
    autoregressive_lags = [f for f in features_list if 'MarketPriceProxy_lag' in f]
    time_features = list(create_time_features(target_date).keys())
    
    covariate_features_to_average = [
        f for f in features_list if f not in time_features and f not in autoregressive_lags
    ]

    # Calculate hourly averages for relevant features from similar past days
    historical_averages = get_historical_averages(df_input.copy(), target_date, covariate_features_to_average)

    # Initialize a list to store the predictions
    predictions = []
    
    # Get the last known values from the historical data for the initial autoregressive lags
    last_known_data = df_input.iloc[-1].to_dict()
    
    # Initialize autoregressive lag features for the first hour of the prediction
    current_autoregressive_lags = {
        'MarketPriceProxy_lag_1h': last_known_data.get('MarketPriceProxy'),
        'MarketPriceProxy_lag_6h': last_known_data.get('MarketPriceProxy_lag_6h'),
        'MarketPriceProxy_lag_24h': last_known_data.get('MarketPriceProxy_lag_24h'),
    }

    # Iterate through each hour of the day (0 to 23)
    for hour in range(24):
        current_datetime = target_date + timedelta(hours=hour)
        
        # Build the feature vector for the current hour
        feature_vector = {}
        
        # Add time-based features
        feature_vector.update(create_time_features(current_datetime))
        
        # Add historical average covariates (including their lags)
        if not historical_averages.empty:
            hourly_avg_row = historical_averages.loc[hour]
            feature_vector.update(hourly_avg_row.to_dict())
        else:
            # Fallback to zeros if no historical averages were found
            for feature in covariate_features_to_average:
                feature_vector[feature] = 0

        # Add the autoregressive lagged features, which are being updated iteratively
        feature_vector.update(current_autoregressive_lags)
        
        # Convert to a DataFrame row and reorder columns to match the features used in training
        feature_df = pd.DataFrame([feature_vector], columns=features_list)
        
        # Make a prediction
        try:
            predicted_price = model.predict(feature_df)[0]
        except Exception as e:
            print(f"Prediction failed for hour {hour}: {e}")
            predicted_price = np.nan
        
        # Calculate the prediction interval with the dynamically fetched RMSE
        lower, upper = get_prediction_interval(predicted_price, rmse=rmse)
        
        # Store the prediction and the interval
        predictions.append({
            'datetime': current_datetime,
            'predicted_price': predicted_price,
            'lower_bound': lower,
            'upper_bound': upper
        })

        # Update the autoregressive lagged features for the next hour's prediction
        # The 1-hour lag becomes the current prediction, while the others shift
        current_autoregressive_lags['MarketPriceProxy_lag_24h'] = current_autoregressive_lags['MarketPriceProxy_lag_6h']
        current_autoregressive_lags['MarketPriceProxy_lag_6h'] = current_autoregressive_lags['MarketPriceProxy_lag_1h']
        current_autoregressive_lags['MarketPriceProxy_lag_1h'] = predicted_price
    
    # Sort predictions by price to find the cheapest hours
    predictions.sort(key=lambda x: x['predicted_price'])
    
    # Get the top 3 predicted cheapest hours
    cheapest_predicted_hours = predictions[:3]
    predicted_hours_only = [p['datetime'].hour for p in cheapest_predicted_hours]

    # --- Sanity Check against Historical Data ---
    print("\n--- Performing Sanity Check against Historical Data ---")
    historical_df_for_check = df_input.copy()
    historical_df_for_check['weekday'] = historical_df_for_check.index.weekday
    historical_df_for_check['month'] = historical_df_for_check.index.month

    # Filter historical data for the same month and weekday
    target_month = target_date.month
    target_weekday = target_date.weekday()
    similar_days_df = historical_df_for_check[
        (historical_df_for_check['month'] == target_month) &
        (historical_df_for_check['weekday'] == target_weekday)
    ].copy()
    
    # Find the hour with the lowest median price from historical data
    if not similar_days_df.empty:
        hourly_medians = similar_days_df.groupby(similar_days_df.index.hour)['MarketPriceProxy'].median()
        best_historical_hour = hourly_medians.idxmin()
        print(f"Historically cheapest hour (based on median price) for similar days is: {best_historical_hour}h")
    else:
        best_historical_hour = None
        print("Warning: No historical data available to perform sanity check. Recommending based on prediction only.")

    final_recommendations = []
    
    if best_historical_hour is not None and best_historical_hour in predicted_hours_only:
        print("Sanity check passed: Historically cheapest hour is in the top 3 predictions.")
        cheapest_to_recommend = cheapest_predicted_hours
    elif best_historical_hour is not None:
        print("Sanity check failed: Historically cheapest hour is NOT in the top 3 predictions. Recommending only the best historical hour.")
        print("The initial top 3 predicted hours were:")
        for p in cheapest_predicted_hours:
            print(f" - {p['datetime'].strftime('%H:%M')} (Predicted price: {p['predicted_price']:.2f} €)")
        
        # Find the prediction for the best historical hour to get its full data
        best_historical_prediction = next((p for p in predictions if p['datetime'].hour == best_historical_hour), None)
        if best_historical_prediction:
            cheapest_to_recommend = [best_historical_prediction]
        else:
            cheapest_to_recommend = cheapest_predicted_hours # Fallback if historical hour prediction is missing
            print("Warning: Prediction for the best historical hour could not be found. Falling back to predicted recommendations.")
    else:
        print("Cannot perform sanity check. Recommending based on model prediction only.")
        cheapest_to_recommend = cheapest_predicted_hours

    # Now, format the final recommendations and add the insights
    cheapest_hours_formatted = []
    for p in cheapest_to_recommend:
        insights_text = generate_recommendation_insights(p['datetime'].hour, p['datetime'].month)
        cheapest_hours_formatted.append({
            'time': p['datetime'].strftime('%H:%M'),
            'predicted_price': f"{p['predicted_price']:.2f}",
            'price_interval': f"[{p['lower_bound']:.2f} - {p['upper_bound']:.2f}]",
            'insight': insights_text
        })
        
    most_expensive_hours = [
        (p['datetime'].strftime('%H:%M'), f"{p['predicted_price']:.2f}", f"[{p['lower_bound']:.2f} - {p['upper_bound']:.2f}]") for p in predictions[-3:]
    ]

    recommendations = {
        'date': date_str,
        'region': region,
        'cheapest_hours': cheapest_hours_formatted,
        'most_expensive_hours': most_expensive_hours
    }
    
    print("\n--- Final Charging Recommendations ---")
    if cheapest_to_recommend:
        print(f"For {region.capitalize()} on {date_str}, the best hours to charge are:")
        
        # Group recommendations by insight text to avoid duplication
        grouped_recs = {}
        for rec in cheapest_hours_formatted:
            insight = rec['insight']
            if insight not in grouped_recs:
                grouped_recs[insight] = []
            grouped_recs[insight].append(rec['time'])

        for insight, times in grouped_recs.items():
            hours_str = ", ".join(times)
            print(f"Hours: {hours_str}")
            print(f"Insight:\n{insight}")
            print("-" * 20)
    else:
        print("No charging recommendations could be made.")

    return predictions, recommendations


if __name__ == "__main__":
    # Example usage
    target_date_str = '2025-08-28'  
    target_region = 'namur'
    
    all_predictions, recs = predict_and_recommend_charging(target_date_str, target_region)
    
    if recs:
        print("\n--- Final Recommendation Summary ---")
        print(recs)
