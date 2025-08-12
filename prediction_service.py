# prediction_service.py
# Author: Quentin Willems
# Project: Smart EV Charge recommendations leveraging AI and Elia open data
# Purpose: Predict a full day's market price, recommend best charging hours and provide contextual insights (including a the recommendation plot used in the web app).
# Dependencies:
#     - data_processing.py (indirect - for initial data prep for model_builder.py)
#     - model_builder.py (to generate the stored model in ev_charge_model.joblib)
#     - config.ini for confirguration variables/paths

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
import configparser
import plotly.graph_objects as go

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
    Loads the pre-computed data from the .parquet file for persistent caching (avoid connecting to DB again to align with the app deployment strategy)
    
    Returns:
        pd.DataFrame: The loaded and processed DataFrame, or None if an error occurs.
    """
    try:
        config = load_config()
        data_path = config['Paths']['data_path']
        cache_file = os.path.join(data_path, "processed_data.parquet")

        if not os.path.exists(cache_file):
            print(
                "Error: Data file not found! Please run 'data_processing.py' locally first to generate the 'processed_data.parquet' file."
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
    if month in [5, 6, 7, 8]:
        if 10 <= best_hour <= 16:
            return f"**🔆 Midday Solar Charging Window (Summer)**\n\n" \
                   f"Solar energy production is at its peak, driving prices down."
        else:
            return f"**🌅 Off-Peak Charging Opportunity (Summer)**\n\n" \
                   f"Reduced grid demand and favorable weather conditions outside peak solar hours. It's a smart moment to charge efficiently."
        
    # B. Winter Period (November – February)
    elif month in [11, 12, 1, 2]:
        if 22 <= best_hour or best_hour <= 6:
            return f"**🌙 Low-Demand Charging Window (Winter)**\n\n" \
                   f"Grid demand is minimal and wind energy generation is strong, leading to lower prices."
        else:
            return f"**🌤️ Favorable Grid Conditions (Winter)**\n\n" \
                   f"Temporary surge in solar or wind production combined with favorable grid dynamics. It's a great opportunity to charge efficiently."

       
# D. Transition Periods (March – April & September – October)
    elif month in [3, 4, 9, 10]:
        if 10 <= best_hour <= 15:
            return f"**🌤️ Midday Charging Window**\n\n" \
                   f"Solar energy production is favorable during these hours, driving price down."
        else:
            return f"**🍃 Off-Peak Charging Opportunity**\n\n" \
                   f"Increased wind generation and lower grid demand during off-peak hours. It's a cost-effective moment to charge your EV."

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
    ### Store weekday and month in new columns in dataframe based on index value (timestamp index)
    historical_df['weekday'] = historical_df.index.weekday
    historical_df['month'] = historical_df.index.month

    ### Find the month and weekday of the target date
    target_month = date_to_predict.month
    target_weekday = date_to_predict.weekday()

    ### Filter historical data for the same month and weekday
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
    Loads the best trained model, predicts the market price for each hour of a given date
    and recommends the best hours to charge based on the lowest predicted prices.
    
    This version blends model predictions with historical price profiles to create
    a more varied and realistic hourly price curve for better recommendations. 
    In a future version, a more dynamic version for future value of covariates
    will include real-time connection to source data API.

    Args:
        date_str (str): The date to predict in 'YYYY-MM-DD' format.
        region (str): The region for which to provide a recommendation.
                      (e.g., 'antwerp', 'brussels', 'liège', etc.)

    Returns:
        tuple: (list of dicts with hourly predictions, dict with recommendations)
    """
    try:
        # Find the path where the model (.joblib) is stored using the config.ini file.
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
            rmse = 41.55  # Fallback to a hardcoded value if not found (i.e. averaged best rmse observed during manual testing)
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
    autoregressive_lags = [f for f in features_list if 'MarketPriceProxy_lag' in f] # Past values of the target variable used as predictors for the current value
    time_features = list(create_time_features(target_date).keys())
    covariate_features_to_average = [f for f in features_list if f not in time_features and f not in autoregressive_lags]
    
    # Include the target variable for historical averaging, which is crucial for the blending approach
    features_for_historical_averages = covariate_features_to_average + ['MarketPriceProxy']

    historical_averages = get_historical_averages(df_input.copy(), target_date, features_for_historical_averages)

    # Initialize a list to store the predictions
    predictions = []
    
    # Get the last known values from the historical data for the initial autoregressive lags
    last_known_data = df_input.iloc[-1].to_dict()
    
    # current_autoregressive_lags is a dictionary containing lagged values of the target variable
    # Initialize autoregressive lag features for the first hour of the prediction
    current_autoregressive_lags = {
        'MarketPriceProxy_lag_1h': last_known_data.get('MarketPriceProxy'),
        'MarketPriceProxy_lag_6h': last_known_data.get('MarketPriceProxy_lag_6h'),
        'MarketPriceProxy_lag_24h': last_known_data.get('MarketPriceProxy_lag_24h'),
    }
    ### More details on the logic: Use current lag to predict and then take the new prediction to update the lag dictionary.
    ###                            Predict the next target value using current lag features, then update the lag dictionary by shifting previous predictions and inserting the new one as the most recent value

    # Lag 1h captures immediate short-term momentum or recent fluctuations.
    # Lag 6h reflects medium-term trends or intra-day cycles (e.g., morning vs. afternoon).
    # Lag 24h captures daily seasonality by referencing the same hour on the previous day.


    # Iterate through each hour of the day (0 to 23) for the hourly prediction
    initial_predictions_list = []
    for hour in range(24):
        current_datetime = target_date + timedelta(hours=hour)
        
        # feature_vector is a dictionary  built to represent all the input features for the model at a specific hour
        # Build the feature vector for the current hour
        feature_vector = {}
        feature_vector.update(create_time_features(current_datetime))  # Update feature_vector for the given hour for the time features
        
        if not historical_averages.empty:
            hourly_avg_row = historical_averages.loc[hour]
            feature_vector.update(hourly_avg_row[covariate_features_to_average].to_dict())   # Update feature_vector for the given hour for the covariates to average features
        else:
            for feature in covariate_features_to_average:
                feature_vector[feature] = 0

        feature_vector.update(current_autoregressive_lags)  # Update feature_vector for the given hour for the autoregressive lags (i.e. target lag)
        
        feature_df = pd.DataFrame([feature_vector], columns=features_list) # Create a one-row DataFrame from a feature vector (assigning column names from features_list) to prepare it for model prediction
        
        try:
            predicted_price = model.predict(feature_df)[0]
        except Exception as e:
            print(f"Prediction failed for hour {hour}: {e}")
            predicted_price = np.nan
        
        # Store initial prediction and update autoregressive (using last prediction for the new lag_1h) lags for the next hour
        initial_predictions_list.append(predicted_price)
        current_autoregressive_lags['MarketPriceProxy_lag_24h'] = current_autoregressive_lags['MarketPriceProxy_lag_6h']
        current_autoregressive_lags['MarketPriceProxy_lag_6h'] = current_autoregressive_lags['MarketPriceProxy_lag_1h']
        current_autoregressive_lags['MarketPriceProxy_lag_1h'] = predicted_price  # Last prediction becomes the new lag_1h for the next loop iteration (i.e. prediction for the next hour)
    
    # --- Blending Logic ---
    print("\n--- Blending Model Prediction with Historical Price Profile for Variance ---")
    
    # Motivation for Blending:
    # This approach is a strategic compromise for the current status of the project. The initial predictions, based on historical
    # averages for future features, often lack the natural daily variance needed for meaningful recommendations (e.g., distinguishing between cheap and expensive hours).
    #
    # By blending the model's overall predicted price level with the typical hourly shape from historical data, we create a more realistic and actionable price curve. 
    # This provides a much better user experience while we await the implementation of a more advanced system that uses real-time API data for weather and Elia load forecasts.
    # Future versions will replace this blending method with a live-data-driven approach.
    
    # Convert initial predictions to a DataFrame
    df_predictions = pd.DataFrame({
        'datetime': [target_date + timedelta(hours=h) for h in range(24)],
        'predicted_price': initial_predictions_list
    })
    df_predictions['hour'] = df_predictions['datetime'].dt.hour
    
    if not historical_averages.empty and 'MarketPriceProxy' in historical_averages.columns:
        # Get the historical price profile from the averages
        historical_price_profile = historical_averages['MarketPriceProxy']
        
        # Calculate the daily average from both predictions and history
        predicted_daily_average = df_predictions['predicted_price'].mean()
        historical_daily_average = historical_price_profile.mean()
        
        # Calculate the difference of each historical hour from its daily average
        historical_diff = historical_price_profile - historical_daily_average
        
        # Create the blended price curve: predicted average + historical shape
        # This preserves the model's overall price level while adding the hourly variance of history
        df_predictions['blended_price'] = predicted_daily_average + historical_diff
        
        print("Blending successful. Recommendations will be based on the blended price curve.")
        final_price_column = 'blended_price'
    else:
        print("Historical data for 'MarketPriceProxy' not available for blending. Reverting to original predictions.")
        final_price_column = 'predicted_price'

    # Now, re-populate the final predictions list with the chosen prices and intervals
    predictions = []
    for index, row in df_predictions.iterrows():
        price = row[final_price_column]
        lower, upper = get_prediction_interval(price, rmse=rmse)
        predictions.append({
            'datetime': row['datetime'],
            'predicted_price': price,
            'lower_bound': lower,
            'upper_bound': upper
        })
    
    # Sort predictions by price to find the cheapest hours
    predictions.sort(key=lambda x: x['predicted_price'])
    
    # Get the top 4 cheapest hours, the middle 10, and the most expensive 10
    cheapest_to_recommend = predictions[:4]
    most_expensive_hours = predictions[-10:]
    middle_hours = predictions[4:-10]

    # --- Recommendations Formatting ---
    cheapest_hours_formatted = []
    for p in cheapest_to_recommend:
        insights_text = generate_recommendation_insights(p['datetime'].hour, p['datetime'].month)
        cheapest_hours_formatted.append({
            'time': p['datetime'].strftime('%H:%M'),
            'predicted_price': f"{p['predicted_price']:.2f}",
            'price_interval': f"[{p['lower_bound']:.2f} - {p['upper_bound']:.2f}]",
            'insight': insights_text
        })
        
    most_expensive_hours_formatted = [
        (p['datetime'].strftime('%H:%M'), f"{p['predicted_price']:.2f}", f"[{p['lower_bound']:.2f} - {p['upper_bound']:.2f}]") for p in most_expensive_hours
    ]
    
    middle_hours_formatted = [
        (p['datetime'].strftime('%H:%M'), f"{p['predicted_price']:.2f}", f"[{p['lower_bound']:.2f} - {p['upper_bound']:.2f}]") for p in middle_hours
    ]

    recommendations = {
        'date': date_str,
        'region': region,
        'cheapest_hours': cheapest_hours_formatted,
        'middle_hours': middle_hours_formatted,
        'most_expensive_hours': most_expensive_hours_formatted,
        'df_predictions': df_predictions, # Return the DataFrame for plotting
        'final_price_column': final_price_column # Return the column name for plotting
    }
    
    print("\n--- Final Charging Recommendations ---")
    if cheapest_to_recommend:
        # Group hours by their insight text to avoid duplication
        grouped_recs = {}
        for rec in cheapest_hours_formatted:
            insight = rec['insight']
            if insight not in grouped_recs:
                grouped_recs[insight] = []
            grouped_recs[insight].append(rec['time'])
        
        # Sort the cheapest hours chronologically before printing
        all_cheapest_times = sorted([rec['time'] for rec in cheapest_hours_formatted])
        print(f"Best Times to Charge: {', '.join(all_cheapest_times)}\n")
        
        # New, restructured printing loop
        for insight, times in grouped_recs.items():
            sorted_times = sorted(times) # Sort times chronologically for each insight
            print(f"{', '.join(sorted_times)}  :   {insight}")
            print("-" * 20)

    else:
        print("No charging recommendations could be made.")
    
    return predictions, recommendations


def generate_charging_recommendation_plot(df_predictions, price_column, date_string):
    """
    Generates a visual summary of recommended charging hours based on predicted prices,
    displaying hours as color-coded squares on a single line.

    Args:
        df_predictions (pd.DataFrame): The DataFrame with blended predicted prices.
        price_column (str): The name of the column containing the prices to use for categorization.
        date_string (str): The date for the plot title.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    try:
        print("Generating Charging Recommendation plot with Plotly...")
        
        # Sort hours by predicted price to find the cheapest/most expensive hours
        df_sorted = df_predictions.sort_values(by=price_column).reset_index(drop=True)
        
        # Categorize hours into three groups based on price
        cheapest_hours = df_sorted.iloc[:4]['hour'].tolist()
        middle_hours = df_sorted.iloc[4:14]['hour'].tolist()
        expensive_hours = df_sorted.iloc[14:]['hour'].tolist()

        # Define color schemes for fill and border
        color_schemes = {
            'optimal': {'fill': '#2ECC71', 'border': '#27AE60'},  # Green
            'moderate': {'fill': '#F39C12', 'border': '#D38A10'}, # Amber
            'avoid': {'fill': '#E74C3C', 'border': '#C0392B'}      # Red
        }

        # Create a color mapping for the hours
        fill_color_map = {}
        border_color_map = {}
        category_map = {}
        for h in cheapest_hours:
            fill_color_map[h] = color_schemes['optimal']['fill']
            border_color_map[h] = color_schemes['optimal']['border']
            category_map[h] = 'Optimal Charging Hours'
        for h in middle_hours:
            fill_color_map[h] = color_schemes['moderate']['fill']
            border_color_map[h] = color_schemes['moderate']['border']
            category_map[h] = 'Moderate Price Hours'
        for h in expensive_hours:
            fill_color_map[h] = color_schemes['avoid']['fill']
            border_color_map[h] = color_schemes['avoid']['border']
            category_map[h] = 'Avoid Charging Hours'

        # Create a DataFrame for the plot
        plot_df = pd.DataFrame({
            'hour': range(24),
            'y_value': [0] * 24,  # All points on the same line
            'marker_fill_color': [fill_color_map.get(h, '#808080') for h in range(24)],
            'marker_border_color': [border_color_map.get(h, '#666666') for h in range(24)],
            'category': [category_map.get(h, 'Uncategorized') for h in range(24)]
        })

        fig = go.Figure()

        # Add the scatter plot for the charging recommendations
        fig.add_trace(go.Scatter(
            x=plot_df['hour'],
            y=plot_df['y_value'],
            mode='markers',
            marker=dict(
                color=plot_df['marker_fill_color'],
                symbol='square',
                size=30,
                line=dict(color=plot_df['marker_border_color'], width=4) # Bold border lines with darker color
            ),
            hovertemplate='<b>Hour: %{x}h</b><br>Category: %{customdata[0]}<extra></extra>',
            customdata=plot_df[['category']],
            showlegend=False  # Remove this trace from the legend
        ))

        # Add invisible traces for the custom legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=14, color=color_schemes['optimal']['fill'], symbol='square', line=dict(width=2, color=color_schemes['optimal']['border'])), name='Optimal Charging Hours'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=14, color=color_schemes['moderate']['fill'], symbol='square', line=dict(width=2, color=color_schemes['moderate']['border'])), name='Moderate Price Hours'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=14, color=color_schemes['avoid']['fill'], symbol='square', line=dict(width=2, color=color_schemes['avoid']['border'])), name='Avoid Charging Hours'))

        # Update layout for a clean look
        fig.update_layout(
            title={
                'text': '<b>Recommended Charging Hours</b>',
                'font': dict(size=20, family='Arial', color='#333'),
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="", # X-axis title is set blank as title is self-sufficient for understanding
            xaxis_title_font=dict(size=18, family='Arial Black'),
            xaxis_tickfont=dict(size=18, family='Arial Black'),
            xaxis={'tickmode': 'linear', 'dtick': 1, 'range': [-0.5, 23.5], 'title_standoff': 10},
            yaxis={'showticklabels': False, 'showgrid': False, 'zeroline': False},
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5,
                font=dict(size=18)
            ),
            height=300
        )
        
        return fig

    except Exception as e:
        print(f"An error occurred while generating the charging recommendation plot: {e}")
        return None

def main():
    """
    Main function to run the prediction and generate the visualization.
    """
    # Example usage
    target_date_str = '2025-08-28'  
    target_region = 'namur'
    
    all_predictions, recs = predict_and_recommend_charging(target_date_str, target_region)

    if recs:
        print("\n--- Final Recommendation Summary ---")
        print(recs)
        
        # Call the plotting function with the returned data
        plot = generate_charging_recommendation_plot(recs['df_predictions'], recs['final_price_column'], target_date_str)
        
        if plot:
            print("\n--- Plotly Figure Data (Local Only) ---")
            plot.show()
        else:
            print("\n--- Plotly Figure Data ---")
            print("Failed to generate plot.")
    else:
        print("Failed to generate predictions and recommendations.")


if __name__ == "__main__":
    main()