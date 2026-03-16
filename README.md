# 🚗 Vehicle Fleet ML System & Dashboard

An end-to-end Machine Learning project written in Python that generates synthetic vehicle telemetry data, trains multiple Random Forest predictive models to detect component failures, and visualizes the results on an interactive web dashboard.

## 🌟 Features

- **Synthetic Data Generation:** Simulates realistic fleet telemetry (Fuel level/pressure, Engine RPM/Vibration/Temp, Brake pressure/temp, etc.) for 5000+ records.
- **Machine Learning Diagnostics:** Trains three separate `RandomForestClassifier` models to predict:
  - Fuel System Issues
  - Brake Line Failures
  - Engine Maintenance Requirements
- **Interactive Streamlit Dashboard:**
  - **Secure Access:** Built-in dashboard authentication.
  - **Weekly Analysis:** Aggregated reporting of fleet health over a rolling 1-year window.
  - **Data Explorer:** View raw synthetic data with advanced attributes like Vehicle Names, License Plates, and nearest Petrol Stations.
  - **Report Generation:** Export filtered telemetry views to `.csv`.
  - **Predictive Tool:** Adjust real-time sensor sliders to instantly classify system health via the ML models.
  - **Correlation Heatmaps:** Interactive Plotly charts mapping variable relationships.

## 🛠️ Tech Stack
- **Data & Processing:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (`RandomForestClassifier`)
- **Web Dashboard:** `streamlit`
- **Visualization:** `plotly`, `matplotlib`, `seaborn`

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
python -m pip install -r requirements.txt
```

### 2. Run the Dashboard
To start the dashboard, run the Streamlit server from your terminal:
```bash
python -m streamlit run dashboard.py
```

### 3. Login
Navigate to the provided localhost URL and use the default mock credentials:
- **Username:** `admin` | **Password:** `password123`
- **Username:** `manager` | **Password:** `fleet2026`

## 📁 Project Structure
- `vehicle_prediction_ml.py`: Core utility script containing data generation, preprocessing, model training, and evaluation logic.
- `dashboard.py`: The frontend Streamlit application interface.
- `requirements.txt`: Python package dependencies.