# alternativeplot.py
# Author: Quentin Willems
# Project: Smart EV Charge recommendations leveraging AI and Elia open data
# Purpose: Create historical insight visuals for the "Did you know?" section of the web app.
# Dependencies:
#     - data_processing.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from data_processing import load_data_from_postgres, engineer_features

def _prepare_data_for_plots(df, date_string, user_province):
    """
    Helper function to prepare and filter data for plotting.

    Args:
        df (pd.DataFrame): The pre-processed DataFrame with historical data.
        date_string (str): The date to analyze in 'DD-MM-YYYY' format.
        user_province (str): The province selected by the user, e.g., 'antwerp'.

    Returns:
        tuple: Filtered DataFrame and metadata for plotting.
    """
    # Ensure province name is lowercase for consistency
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
    
    # --- UPDATED: Logic for determining the wind region based on the new data_processing.py output ---
    user_region_wind_col = ''
    if user_province in pv_flanders_provinces or user_province in pv_brussels_provinces:
        # Brussels is now handled by the Flanders wind data as federal data is not available
        user_region_wind_col = 'area_flanders_wind_utilization_rate'
    elif user_province in pv_wallonia_provinces:
        user_region_wind_col = 'area_wallonia_wind_utilization_rate'
    else:
        print(f"Warning: Province '{user_province}' not recognized. Skipping wind data.")
        user_region_wind_col = None # Set to None to handle this case later

    # Get the PV column name for the user's province
    user_pv_col_name = f'area_{user_province}_pv_utilization_rate'
    
    if user_pv_col_name not in df_filtered.columns:
        print(f"PV data for province '{user_province}' not found in the DataFrame.")
        return None, None, None, None, None, None
    
    return df_filtered, target_calendar_week, target_weekday_name, user_province, user_pv_col_name, user_region_wind_col


def generate_renewable_energy_plot(df, date_string, user_province):
    """
    Generates the historical renewable energy utilization profile plot using Plotly with enhanced visuals.
    
    Args:
        df (pd.DataFrame): The pre-processed DataFrame with historical data.
        date_string (str): The date to analyze in 'DD-MM-YYYY' format.
        user_province (str): The province selected by the user, e.g., 'antwerp'.
    
    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    try:
        df_filtered, target_calendar_week, target_weekday_name, user_province, user_pv_col_name, user_region_wind_col = \
            _prepare_data_for_plots(df.copy(), date_string, user_province)

        if df_filtered is None:
            return None

        print("Generating Historical Renewable Energy Profile plot with Plotly...")
        
        # Calculate the total renewable utilization
        # --- UPDATED: Handle cases where wind column is None ---
        pv_series = df_filtered.get(user_pv_col_name, 0)
        wind_series = df_filtered.get(user_region_wind_col, 0) if user_region_wind_col else 0
        df_filtered['total_renewable_utilization'] = pv_series + wind_series
        
        # Aggregate to get the mean and standard deviation for each hour
        df_total_renewable = df_filtered.groupby('hour').agg(
            mean_utilization=('total_renewable_utilization', 'mean'),
            std_utilization=('total_renewable_utilization', 'std')
        ).reset_index()

        # Create a Plotly Figure object
        fig = go.Figure()

        # Add the line plot for the mean utilization with a modern, clean color
        fig.add_trace(go.Scatter(
            x=df_total_renewable['hour'],
            y=df_total_renewable['mean_utilization'],
            mode='lines+markers',
            name=f'Average Renewable Generation (Wind and Solar - {user_province.capitalize()})',
            marker=dict(color='#2ECC71', size=8),  # A vibrant green
            line=dict(color='#27AE60', width=3),  # A slightly darker green for the line
            hovertemplate='Hour: %{x}<br>Avg. Rate: %{y:.2f}<extra></extra>' # Rounding hover text
        ))

        # Add the shaded area for the standard deviation with a subtle, modern fill
        fig.add_trace(go.Scatter(
            x=df_total_renewable['hour'],
            y=df_total_renewable['mean_utilization'] + df_total_renewable['std_utilization'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hovertemplate='<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=df_total_renewable['hour'],
            y=df_total_renewable['mean_utilization'] - df_total_renewable['std_utilization'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(46, 204, 113, 0.2)', # Semi-transparent vibrant green
            name='Standard Deviation',
            hovertemplate='<extra></extra>'
        ))
        
        # Update layout for title and axis labels, with increased font size
        fig.update_layout(
            title={
                'text': f'<b>Historical Renewable Energy Generation Rate (Wind and Solar - similar days in {user_province.capitalize()})</b>',
                'font': dict(size=18, family='Arial', color='#333'), # Increased font size
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title_font=dict(size=14), # Increased font size
            yaxis_title_font=dict(size=14), # Increased font size
            xaxis_tickfont=dict(size=12), # Increased font size
            yaxis_tickfont=dict(size=12), # Increased font size
            xaxis={'tickmode': 'linear', 'dtick': 1},
            template='plotly_white'
        )
        # Round the y-axis ticks to two decimal places
        fig.update_yaxes(tickformat=".2f")

        # Add the note at the bottom as an annotation, with increased font size
        fig.add_annotation(
            text="Note: The Generation Rate is the ratio between the measured Wind & Solar energy generation and the respective estimated total capacity.",
            xref="paper", yref="paper",
            x=0.05, y=-0.2,
            showarrow=False,
            font=dict(size=12, color='gray'), # Increased font size
            align="left"
        )
        
        return fig

    except Exception as e:
        print(f"An error occurred while generating the renewable energy plot: {e}")
        return None


def generate_electricity_price_plot(df, date_string, user_province):
    """
    Generates the historical electricity price distribution plot using Plotly with enhanced visuals and a categorical color scheme.
    
    Args:
        df (pd.DataFrame): The pre-processed DataFrame with historical data.
        date_string (str): The date to analyze in 'DD-MM-YYYY' format.
        user_province (str): The province selected by the user, e.g., 'antwerp'.
    
    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    try:
        df_filtered, target_calendar_week, target_weekday_name, user_province, _, _ = \
            _prepare_data_for_plots(df.copy(), date_string, user_province)

        if df_filtered is None:
            return None

        print("Generating Historical Electricity Price Distribution plot with Plotly...")

        # Calculate median price for each hour and sort to find the cheapest/most expensive hours
        hourly_medians = df_filtered.groupby('hour')['MarketPriceProxy'].median().reset_index()
        hourly_medians = hourly_medians.sort_values(by='MarketPriceProxy').reset_index(drop=True)

        # Categorize hours into three groups based on price
        cheapest_hours = hourly_medians.iloc[:4]['hour'].tolist()
        middle_hours = hourly_medians.iloc[4:14]['hour'].tolist()
        expensive_hours = hourly_medians.iloc[14:]['hour'].tolist()
        
        # Create a color mapping for the hours
        color_map = {}
        for h in cheapest_hours:
            color_map[h] = '#2ECC71'  # Green for cheap hours
        for h in middle_hours:
            color_map[h] = '#F39C12'  # Orange for middle-priced hours
        for h in expensive_hours:
            color_map[h] = '#E74C3C'  # Red for expensive hours
        
        fig = go.Figure()
        
        # Add invisible traces to create a custom legend with updated names
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#2ECC71'), name='Optimal Charging Hours'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#F39C12'), name='Moderate Price Hours'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#E74C3C'), name='Avoid Charging Hours'))


        # Add a box plot trace for each hour with the new color scheme
        for hour in range(24):
            hour_data = df_filtered[df_filtered['hour'] == hour]
            if not hour_data.empty:
                fig.add_trace(go.Box(
                    y=hour_data['MarketPriceProxy'],
                    name=f'{hour:02d}h',
                    marker_color=color_map.get(hour, '#bdc3c7'),
                    hovertemplate='Hour: %{x}<br>Price: %{y:.2f} €/MWh<extra></extra>',
                    showlegend=False,
                    boxpoints=False,
                    showwhiskers=False # This line removes the whiskers
                ))
        
        # Update layout for a cleaner look, with increased font size and centered legend
        fig.update_layout(
            title={
                'text': f'<b>Electricity Price (€/MWh) Distribution by Hour (similar days in {user_province.capitalize()})</b>',
                'font': dict(size=20, family='Arial', color='#333'),
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='<b>Hour of Day</b>',
            xaxis_title_font=dict(size=16),
            yaxis_title='<b>Electricity Price (€/MWh)</b>',
            yaxis_title_font=dict(size=16),
            xaxis_tickfont=dict(size=14),
            yaxis_tickfont=dict(size=14),
            xaxis={'tickmode': 'linear', 'dtick': 1},
            yaxis={'dtick': 50},
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.03,  # Adjusted legend position to be below the title
                xanchor="center",
                x=0.5,
                font=dict(size=18)
            ),
            height=600 # This line sets the height of the plot
        )
        # Round the y-axis ticks to two decimal places
        fig.update_yaxes(tickformat=".2f")
        
        # Add a single annotation for the footnote
        footnote_text = "Negative prices can occur when electricity supply (especially from renewables) exceeds demand (e.g.: power plants must pay to offload excess energy to avoid grid instability)."
        fig.add_annotation(
            text=footnote_text,
            xref="paper", yref="paper",
            x=0.01, y=-0.1,
            showarrow=False,
            font=dict(size=10, color='gray'),
            align="left"
        )
        
        return fig

    except Exception as e:
        print(f"An error occurred while generating the electricity price plot: {e}")
        return None

if __name__ == "__main__":
    # --- UPDATED: The new execution block now loads and processes the data directly ---
    # First, load the raw data from PostgreSQL
    df_raw = load_data_from_postgres()

    if df_raw is not None:
        # Then, engineer the features
        df_processed = engineer_features(df_raw)

        if df_processed is not None:
            # Use the processed DataFrame to generate sample plots to check code.
            date_to_plot = '15-09-2025'
            province_to_plot = 'Namur'

            renewable_fig = generate_renewable_energy_plot(df_processed.copy(), date_to_plot, province_to_plot)
            if renewable_fig:
                renewable_fig.show()

            price_fig = generate_electricity_price_plot(df_processed.copy(), date_to_plot, province_to_plot)
            if price_fig:
                price_fig.show()