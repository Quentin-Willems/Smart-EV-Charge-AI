# ⚡ Smart Eelectric Vehicle (EV) Charge Insights

## 🧾 Summary

**Smart EV Charge Insights** is an AI-powered Streamlit web application that helps EV owners optimize charging times to minimize costs. It predicts day-ahead electricity spot market price fluctuations using historical data from **ELIA** (Belgium's electricity transmission system operator) and recommends the best hours to charge based on these forecasts.

The app supports **province-specific recommendations** for all Belgian regions and visualizes results with interactive plots.

🔗 **Live Demo**: [smart-ev-charge-ai.streamlit.app](https://smart-ev-charge-ai.streamlit.app/)

---

## 🚀 Key Features

- **AI-Driven Predictions**  
  Trains and selects the best model from a set of machine learning algorithms (Random Forest, XGBoost, AutoARIMA, or TCN) to forecast hourly electricity prices. Model selection is based on performance metrics like MAE and RMSE.

- **Custom Recommendations**  
  Users select a date and province to get tailored charging advice, including the top 4 cheapest hours and contextual insights (e.g., solar/wind influences).

- **Interactive Visualization**  
  Displays a color-coded hourly recommendation plot:  
  🟩 Green = optimal hours  
  🟨 Amber = moderate  
  🟥 Red = avoid

- **Data Processing Pipeline**  
  Includes SQL-based ingestion and transformation in PostgreSQL, feature engineering (lagged variables, cyclical time features), and model training.

- **Deployment**  
  Live demo available via Streamlit Cloud.

- **Data Source**  
  Uses four open datasets from the [ELIA Open Data Portal](https://www.elia.be/en/open-data) under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

---

## 🎯 Motivations

This project empowers EV owners in Belgium to make **data-informed decisions** amid fluctuating energy prices and growing renewable integration. By leveraging open data and AI, it promotes:

- Cost savings  
- Grid efficiency  
- Sustainable energy use  

It also demonstrates a **full-stack data pipeline** from raw open data ingestion to a user-friendly web app, encouraging open-source contributions in energy analytics.

---

## 🏗️ Project Architecture

The project is modular and includes:

- `config.ini`: Configuration for paths and DB credentials   
- `postgres_data_wrangler.sql`: Cleans and joins raw ELIA datasets (create a master data for analytics)
- `data_processing.py`: Centralized module for data loading (from PostgreSQL master data), feature engineering and persistent file caching (for the web app).
- `model_builder.py`: Trains and selects the best model  
- `prediction_service.py`: Generates predictions and visualizations  
- `streamlit_app.py`: Streamlit web interface  
- `requirements.txt`: Python dependencies

---

## 🔁 Reproducibility & Setup

### Step 1: Download ELIA Datasets

- Visit the [ELIA Open Data Portal](https://www.elia.be/en/open-data)
- Download the following datasets (CSV format recommended):
  - `imbalance_prices_per_quarter_hour`
  - `measured_and_forecasted_total_load_on_the_belgian_grid`
  - `pv_production_belgian_grid`
  - `wind_power_production_estimation_and_forecast_on_belgian_grid`

> ⚠️ Comply with the CC BY 4.0 license by attributing ELIA in any reuse.

---

### Step 2: Set Up PostgreSQL

Install PostgreSQL locally or via Docker:

```bash
docker run -d -p 5432:5432 --name ev-db -e POSTGRES_PASSWORD=yourpassword postgres
```

Create the database:

```bash
createdb EV_AI_DB
```

Import CSVs into tables (via pgAdmin or `COPY` command). Table names must match the SQL script.

Update `config.ini`:

```ini
[database]
host = localhost
port = 5432
database = EV_AI_DB
user = postgres
password = yourpassword
```

Run the SQL script:

```bash
psql -d EV_AI_DB -f postgres_data_wrangler.sql
```

---

### Step 3: Install Dependencies

```bash
git clone https://github.com/your-username/smart-ev-charge-insights.git
cd smart-ev-charge-insights
pip install -r requirements.txt
```

---

### Step 4: Process Data & Train Model

```bash
python data_processing.py
python model_builder.py
```

---

### Step 5: Run the App Locally

```bash
streamlit run streamlit_app.py
```

Visit [http://localhost:8501](http://localhost:8501) and generate insights by selecting a date and province.

---

### Step 6: Deploy (Optional)

- Push to GitHub
- Connect via Streamlit Sharing
- Ensure `processed_data.parquet` and `ev_charge_model.joblib` are included (or regenerate them in cloud setup)

---

## 📌 Notes

- **Desktop Preferred**: Plots are optimized for desktop browsers  
- **Data Limitations**: Predictions rely on historical patterns; real-time events may affect accuracy  
- **Contributing**: Pull requests welcome! Add models, improve visuals, or integrate real-time data

---

## 📄 License

This project is licensed under the **MIT License**.  
Data usage must attribute ELIA per the **CC BY 4.0 License**.

---

## 📬 Contact

For questions, feel free to contact me via email: **willems.quentin@gmail.com**
