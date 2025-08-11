# streamlit_app.py
# Author: Quentin Willems
# Project: Smart EV Charge recommendations leveraging AI and Elia open data
# Purpose: A Streamlit web application to visualize historical EV charge insights and provide charging recommendations.

import streamlit as st
import pandas as pd
from datetime import date
import os
import configparser

# Import the core logic from the existing scripts.
# The plotting function is now part of the prediction service,
# so we remove the import from insights_visuals and get it from here.
from prediction_service import predict_and_recommend_charging, generate_charging_recommendation_plot

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
        st.error(f"Configuration file '{config_path}' not found or is empty.")
        st.stop() # Stop the app if config is missing
    return config

# --- 1. Streamlit Page Configuration ---
st.set_page_config(
    page_title="Smart EV Charge Insights",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Electric Vehicle Charge Insights")
st.markdown("Explore AI augmented recommendations for the best times to charge your EV.")
st.markdown("The recommended charging hours are based on the predicted fluctuations in the day-ahead electricity spot market price.")
st.markdown("""---""")

# --- 2. Caching Data for Performance ---
# This function is now responsible for loading the pre-computed data file.
@st.cache_data
def get_processed_data():
    """
    Loads the pre-computed data from the .parquet file for persistent caching.
    This function will only run once per app run, then use the in-memory cache.
    
    Returns:
        pd.DataFrame: The loaded and processed DataFrame, or None if an error occurs.
    """
    try:
        # Load the data path from the config file
        config = load_config()
        data_path = config['Paths']['data_path']
        cache_file = os.path.join(data_path, "processed_data.parquet")

        if not os.path.exists(cache_file):
            st.error(
                "Data file not found! Please run `data_processing.py` locally first "
                "to generate the `processed_data.parquet` file and include it in your deployment."
            )
            st.stop() # Stop the app if the data file is missing
        
        df_processed = pd.read_parquet(cache_file)
        return df_processed
    
    except KeyError:
        st.error("Missing 'data_path' in the 'Paths' section of `config.ini`.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the data file: {e}")
        st.stop()
    
    return None

# --- 3. User Input Widgets in Sidebar ---
with st.sidebar:
    st.header("Settings")
    # Define the list of provinces for the dropdown
    provinces = [
        'antwerp', 'eastflanders', 'flemishbrabant', 'hainaut', 'limburg', 'liège', 
        'luxembourg', 'namur', 'walloonbrabant', 'westflanders', 'brussels'
    ]

    # Create a date picker.
    selected_date = st.date_input(
        "Select a date for analysis:",
        min_value=date.today(),
        value=date.today(),
        help="Select the current day or a future day."
    )

    # Create a dropdown list for provinces.
    selected_province = st.selectbox(
        "Select your province:",
        options=[p.capitalize() for p in provinces],
        help="Photovoltaic and wind production vary across provinces."
    )
    
    st.markdown("""---""")
    
    # This button is in the sidebar, but its action affects the main page
    generate_button = st.button("Generate Insights", type="primary")

# --- 4. Prediction, Plot Generation and Display (on the main page) ---
if generate_button:
    # Format date for the prediction service (YYYY-MM-DD)
    prediction_date_str = selected_date.strftime('%Y-%m-%d')
    # Format date for the visuals (DD-MM-YYYY)
    visuals_date_str = selected_date.strftime('%d-%m-%Y')
    
    province_lower = selected_province.lower()
    
    st.subheader(f"Insights and Recommendations for {selected_province} on {visuals_date_str}")
    
    with st.spinner("Generating recommendations and plots..."):
        # Get the recommendation from the prediction service
        # The function now returns a DataFrame and column name for plotting
        _, recommendations = predict_and_recommend_charging(prediction_date_str, province_lower)

        # Check if recommendations were successfully generated
        if recommendations and recommendations['cheapest_hours']:
            # Use an expander to make the recommendation a collapsible section
            with st.expander("⚡ **Best hours to charge**", expanded=True):
                # Group recommendations by their insight text to display them concisely
                grouped_recs = {}
                for rec in recommendations['cheapest_hours']:
                    insight = rec['insight']
                    if insight not in grouped_recs:
                        grouped_recs[insight] = []
                    grouped_recs[insight].append(rec['time'])
                
                # Sort the cheapest hours chronologically
                all_cheapest_times = sorted([rec['time'] for rec in recommendations['cheapest_hours']])
            
                # Display the grouped recommendations with their insights
                for insight, times in grouped_recs.items():
                    # Use bold for the times and the insight title
                    sorted_times = sorted(times)
                    hours_str = ", ".join(sorted_times)
                    st.markdown(f"**{hours_str}** : {insight}")
                    st.markdown("""---""") # Use a horizontal rule for separation

        else:
            # Display a warning if no recommendations could be generated
            st.warning(
                "Could not generate recommendations. This is likely due to a missing model or data file. "
                "Please ensure 'ev_charge_model.joblib' and 'processed_data.parquet' are in the "
                "correct directories as specified in your `config.ini` file."
            )
        

        if recommendations:
            # Generate and display the Charging Recommendation Plot.
            # We now use the dataframe and column name from the `recommendations` dictionary,
            # and the formatted date string.
            charging_rec_fig = generate_charging_recommendation_plot(
                recommendations['df_predictions'], 
                recommendations['final_price_column'], 
                visuals_date_str
            )

            if charging_rec_fig:
                # The plot is now displayed in the main content area, without columns
                st.plotly_chart(charging_rec_fig, use_container_width=True)
            else:
                st.warning("Could not generate the Charging Recommendation plot. Please check the data and scripts.")
        else:
            st.error("Failed to generate predictions and recommendations.")