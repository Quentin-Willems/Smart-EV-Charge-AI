# insights_visuals.py
# Author: Quentin Willems
# Project: Smart EV Charge recommendations leveraging AI and Elia open data
# Purpose: Create historical insight visuals for the "Did you know?" section of the web app.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# It is assumed that the `df_input` DataFrame has already been loaded and processed by the feature_engineering.py script.
try:
    from feature_engineering import df_input
    print("DataFrame `df_input` successfully loaded from feature_engineering.py for insights visuals.")
except ImportError:
    print("Warning: `df_input` not found. Creating a dummy DataFrame for demonstration.")

def _prepare_data_for_plots(df, date_string, user_province):
    """
    Helper function to prepare and filter data for plotting.
    """
    user_province = user_province.lower()
    
    # Ensure 'weekday', 'hour', 'calendar_week' are numeric for filtering and grouping
    df['weekday'] = pd.to_numeric(df['weekday'], errors='coerce').astype('Int64')
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce').astype('Int64')
    df['calendar_week'] = pd.to_numeric(df['calendar_week'], errors='coerce').astype('Int64')
    
    # Extract weekday and calendar week from the user-provided date
    date_obj = datetime.strptime(date_string, '%d-%m-%Y')
    target_weekday = date_obj.weekday()
    target_calendar_week = date_obj.isocalendar()[1]
    target_weekday_name = date_obj.strftime('%A')
    
    print(f"\nAnalyzing historical data for {target_weekday_name}s in calendar week {target_calendar_week}...")
    
    # Filter the DataFrame to get historical data for similar days
    df_filtered = df[(df['weekday'] == target_weekday) & (df['calendar_week'] == target_calendar_week)].copy()
    
    if df_filtered.empty:
        print("No historical data found for the specified weekday and calendar week.")
        return None, None, None, None, None, None

    # Define province-to-region mapping for wind production
    pv_flanders_provinces = ['antwerp', 'eastflanders', 'flemishbrabant', 'limburg', 'westflanders']
    pv_wallonia_provinces = ['hainaut', 'liège', 'luxembourg', 'namur', 'walloonbrabant']
    pv_brussels_provinces = ['brussels']
    
    # Find the user's region based on their province for wind data
    user_region_wind_col = ''
    if user_province in pv_flanders_provinces:
        user_region_wind_col = 'area_flanders_wind_utilization_rate'
    elif user_province in pv_wallonia_provinces:
        user_region_wind_col = 'area_wallonia_wind_utilization_rate'
    elif user_province in pv_brussels_provinces:
        # Brussels doesn't have a regional wind rate, so we'll use the federal one
        user_region_wind_col = 'area_federal_wind_utilization_rate'
    else:
        print(f"Warning: Province '{user_province}' not recognized. Defaulting to federal wind data.")
        user_region_wind_col = 'area_federal_wind_utilization_rate'

    # Get the PV column name for the user's province
    user_pv_col_name = f'area_{user_province}_pv_utilization_rate'
    
    if user_pv_col_name not in df_filtered.columns:
        print(f"PV data for province '{user_province}' not found in the DataFrame.")
        return None, None, None, None, None, None
    
    return df_filtered, target_calendar_week, target_weekday_name, user_province, user_pv_col_name, user_region_wind_col


def generate_renewable_energy_plot(df, date_string, user_province):
    """
    Generates the historical renewable energy utilization profile plot.
    
    Args:
        df (pd.DataFrame): The pre-processed DataFrame with historical data.
        date_string (str): The date to analyze in 'DD-MM-YYYY' format.
        user_province (str): The province selected by the user, e.g., 'antwerp'.
    """
    try:
        df_filtered, target_calendar_week, target_weekday_name, user_province, user_pv_col_name, user_region_wind_col = \
            _prepare_data_for_plots(df.copy(), date_string, user_province) # Use a copy to avoid modifying original df

        if df_filtered is None:
            return

        print("Generating Historical Renewable Energy Profile plot...")
        
        df_filtered['total_renewable_utilization'] = (
            df_filtered[user_pv_col_name] +
            df_filtered[user_region_wind_col]
        )
        
        df_total_renewable = df_filtered.groupby('hour').agg(
            mean_utilization=('total_renewable_utilization', 'mean'),
            std_utilization=('total_renewable_utilization', 'std')
        ).reset_index()
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 6))
        plt.plot(df_total_renewable['hour'], df_total_renewable['mean_utilization'], marker='o', color='green', label=f'Average Renewable Generation (Wind and Solar - {user_province.capitalize()})')
        plt.fill_between(df_total_renewable['hour'],
                         df_total_renewable['mean_utilization'] - df_total_renewable['std_utilization'],
                         df_total_renewable['mean_utilization'] + df_total_renewable['std_utilization'],
                         color='green', alpha=0.2, label='Standard Deviation')
        plt.title(f'Historical Renewable Energy Generation Rate (Wind and Solar - historical days similar to {date_string} in {user_province.capitalize()})', fontsize=16)
        plt.xlabel('Hour of the Day', fontsize=12)
        plt.ylabel('Average Total Generation Rate', fontsize=12)
        plt.xticks(df_total_renewable['hour'])
        plt.legend()
        plt.tight_layout()
        plt.figtext(0.1, 0.005,
            "Note: The Generation Rate is the ratio between the measured Wind & Solar energy generation and the respective estimated total capacity.", ha='left', fontsize=7, color='gray')
        plt.show()

    except Exception as e:
        print(f"An error occurred while generating the renewable energy plot: {e}")


def generate_electricity_price_plot(df, date_string, user_province):
    """
    Generates the historical electricity price distribution plot.
    
    Args:
        df (pd.DataFrame): The pre-processed DataFrame with historical data.
        date_string (str): The date to analyze in 'DD-MM-YYYY' format.
        user_province (str): The province selected by the user, e.g., 'antwerp'.
    """
    try:
        df_filtered, target_calendar_week, target_weekday_name, user_province, _, _ = \
            _prepare_data_for_plots(df.copy(), date_string, user_province) # Use a copy to avoid modifying original df

        if df_filtered is None:
            return

        print("Generating Historical Electricity Price Distribution plot...")

        # Find the hour with the lowest median price to highlight it
        hourly_medians = df_filtered.groupby('hour')['MarketPriceProxy'].median()
        best_hour = hourly_medians.idxmin()
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='hour', y='MarketPriceProxy', data=df_filtered, palette='viridis')
        
        # Highlight the best hour
        x_pos = list(hourly_medians.index).index(best_hour)
        plt.axvspan(x_pos - 0.5, x_pos + 0.5, color='green', alpha=0.3, label=f'Historically Cheapest Hour ({best_hour}h)')

        plt.title(f'Electricity Price (€/MWh) Distribution by Hour (historical days similar to {date_string} in {user_province.capitalize()})', fontsize=16)
        plt.xlabel('Hour of the Day', fontsize=12)
        plt.ylabel('Market Price (€/MWh)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred while generating the electricity price plot: {e}")
        
if __name__ == "__main__":
    # Example usage of the functions with a user-provided date and province
    # In a full pipeline, these would be called from main.py
    # For standalone testing, ensure df_input is available (e.g., by running feature_engineering.py first)
    # or by loading a pre-saved version of df_input.
    
    # Call the individual plot functions
    generate_renewable_energy_plot(df_input.copy(), '05-08-2025', 'Namur')
    generate_electricity_price_plot(df_input.copy(), '05-08-2025', 'Namur')
