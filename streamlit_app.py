# streamlit_app.py
# Author: Quentin Willems
# Project: Smart EV Charge recommendations leveraging AI and Elia open data
# Purpose: A Streamlit web application to visualize historical EV charge insights and provide charging recommendations.
# Dependencies:
#     - data_processing.py (for generation of processed_data.parquet)
#     - model_builder.py (for generation of stored model in ev_charge_model.joblib)
#     - prediction_service.py (for core logic of prediction and recommendations plot)
#     - config.ini for configuration variables/paths

import streamlit as st
import pandas as pd
from datetime import date
import os
import configparser

# Import the core logic from the existing scripts. The plotting function is now part of the prediction_service.py
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

st.title("⚡ Optimal Electric Vehicle Charging Times")
st.markdown("""
Select your day and province from the left dropdown menu, then click **Generate Insights** for AI-driven EV charging recommendations based on predicted day-ahead electricity prices.  
Data provided by ELIA under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0). For the best plot visibility, use a computer browser.
""")
st.markdown("---")

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

    # The mapping from internal code to display name (Dutch/French)
    province_map = {
        'antwerp': 'Antwerpen',
        'eastflanders': 'Oost-Vlaanderen',
        'flemishbrabant': 'Vlaams-Brabant',
        'limburg': 'Limburg',
        'westflanders': 'West-Vlaanderen',
        'hainaut': 'Hainaut',
        'liège': 'Liège',
        'luxembourg': 'Luxembourg',
        'namur': 'Namur',
        'walloonbrabant': 'Brabant Wallon',
        'brussels': 'Brussel / Bruxelles',
    }
    
    # Create the sorted list of display names for the selectbox
    display_provinces = list(province_map.values())
    display_provinces.sort()
    
    # Create a date picker.
    selected_date = st.date_input(
        "Select a date for analysis:",
        min_value=date.today(),
        value=date.today(),
        help="Select the current day or a future day."
    )

    # Create a dropdown list for provinces using the display names
    selected_display_province = st.selectbox(
        "Select your province:",
        options=display_provinces,
        help="Photovoltaic and wind production vary across provinces."
    )
    
    st.markdown("""---""")
    
    # This button is in the sidebar but its action affects the main page
    generate_button = st.button("Generate Insights", type="primary")

# --- 4. Prediction, Plot Generation and Display (on the main page) ---
if generate_button:
    # We need to map the selected display name back to the internal key for the prediction service.
    # This is done by creating a reverse lookup from the original map.
    reverse_province_map = {v: k for k, v in province_map.items()}
    selected_province_key = reverse_province_map.get(selected_display_province)

    # Format date for the prediction service (YYYY-MM-DD)
    prediction_date_str = selected_date.strftime('%Y-%m-%d')
    # Format date for the visuals (DD-MM-YYYY)
    visuals_date_str = selected_date.strftime('%d-%m-%Y')
    
    st.subheader(f"Insights and Recommendations for {selected_display_province} on {visuals_date_str}")
    
    with st.spinner("Generating recommendations and plots..."):
        # Get the recommendation from the prediction service
        _, recommendations = predict_and_recommend_charging(prediction_date_str, selected_province_key)

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
            charging_rec_fig = generate_charging_recommendation_plot(
                recommendations['df_predictions'], 
                recommendations['final_price_column'], 
                visuals_date_str
            )

            if charging_rec_fig:
                st.plotly_chart(charging_rec_fig, use_container_width=True)
            else:
                st.warning("Could not generate the Charging Recommendation plot. Please check the data and scripts.")
        else:
            st.error("Failed to generate predictions and recommendations.")