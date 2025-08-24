Smart EV Charge Insights

Summary
Smart EV Charge Insights is an AI-powered Streamlit web application that helps electric vehicle (EV) owners optimize charging times to minimize costs. It predicts day-ahead electricity spot market price fluctuations using historical data from ELIA (Belgium's electricity transmission system operator) and recommends the best hours to charge based on these forecasts. The app supports province-specific recommendations for all Belgian regions and visualizes results with interactive plots. You can check out a live demo of the app at https://smart-ev-charge-ai.streamlit.app/.

Key Features
AI-Driven Predictions: Trains and selects the best model from a set of machine learning algorithms (Random Forest, XGBoost, AutoARIMA, or TCN) to forecast hourly electricity prices. The best model is selected based on performance metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

Custom Recommendations: Users select a date and province to get tailored charging advice, including the top 4 cheapest hours and contextual insights (e.g., solar/wind influences).

Interactive Visualization: Displays a color-coded hourly recommendation plot (green for optimal, amber for moderate, red to avoid).

Data Processing Pipeline: Includes SQL-based data ingestion and transformation in PostgreSQL, feature engineering (e.g., lagged variables, cyclical time features), and model training.

Deployment: Live demo available at https://smart-ev-charge-ai.streamlit.app/.

Data Source: Utilizes four open datasets from ELIA Open Data Portal under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Motivations
This project aims to empower EV owners in Belgium to make data-informed decisions amid fluctuating energy prices and growing renewable integration. By leveraging open data and AI, it promotes cost savings, grid efficiency, and sustainable energy use. It also demonstrates a full-stack data pipeline from raw open data ingestion to a user-friendly web app, encouraging open-source contributions in energy analytics.

Project Architecture
The project is structured into modular components to handle the full data and application pipeline.

config.ini: Configuration for paths and database credentials.

config.ini.example: An example file showing the required configuration settings.

data_processing.py: Loads data from PostgreSQL, performs feature engineering, and caches processed data as a Parquet file.

model_builder.py: Trains multiple models, evaluates their performance, and saves the best one.

prediction_service.py: Generates predictions, recommendations, and Plotly visualizations.

streamlit_app.py: The Streamlit web app interface.

postgres_data_wrangler.sql: A SQL script for cleaning and joining raw ELIA datasets in PostgreSQL.

requirements.txt: Python dependencies.

Reproducibility and Setup
The raw ELIA data is not included in this repository due to its size and licensing. Users must download it from the ELIA Open Data Portal and load it into a local PostgreSQL database.

Step 1: Download ELIA Datasets
Go to the ELIA Open Data Portal.

Download the four historical datasets mentioned in the Summary (CSV format recommended).

Note: Comply with the CC BY 4.0 license by attributing ELIA in any reuse.

Step 2: Set Up PostgreSQL Database
Install PostgreSQL (e.g., via the official installer or Docker).

docker run -d -p 5432:5432 --name ev-db -e POSTGRES_PASSWORD=yourpassword postgres

Create a database: createdb EV_AI_DB (or use pgAdmin/psql).

Import the downloaded CSVs into tables using pgAdmin's import tool or the COPY command. Table names should match the script:

imbalance_prices_per_quarter_hour

measured_and_forecasted_total_load_on_the_belgian_grid

pv_production_belgian_grid

wind_power_production_estimation_and_forecast_on_belgian_grid

Create and update config.ini with your DB credentials.

[database]
host = localhost
port = 5432
database = EV_AI_DB
user = postgres
password = yourpassword

Run the SQL script: psql -d EV_AI_DB -f postgres_data_wrangler.sql. This cleans the data and creates the master_data table.

Step 3: Install Dependencies
git clone https://github.com/your-username/smart-ev-charge-insights.git
cd smart-ev-charge-insights
pip install -r requirements.txt

Step 4: Process Data and Train Model
python data_processing.py
python model_builder.py

Step 5: Run the App Locally
streamlit run streamlit_app.py

Access the app at http://localhost:8501. Select a date and province, then click "Generate Insights".

Step 6: Deploy (Optional)
Deploy to Streamlit Cloud: Push to GitHub, connect via Streamlit Sharing, and ensure processed_data.parquet and ev_charge_model.joblib are included in your repository (or regenerate them in the cloud setup).

Notes
Desktop Preferred: Plots are optimized for computer browsers; mobile may distort visuals.

Data Limitations: Predictions rely on historical patterns; real-time factors (e.g., weather events) may affect accuracy.

Contributing: Pull requests are welcome for improvements like additional models, real-time data integration, or enhanced visualizations. Open issues for bugs or features.

License
This project is licensed under the MIT License - see the LICENSE.md file for details. Data usage must attribute ELIA per the CC BY 4.0 license.

Contact
For questions, open a GitHub Issue or email willems.quentin@gmail.com.