import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from vehicle_prediction_ml import generate_data, preprocess_data, train_models, predict_vehicle_issues

st.set_page_config(page_title="Vehicle Fleet ML Dashboard", layout="wide", page_icon="🚗")

# --- Authentication Configuration ---
# Hardcoded for demonstration purposes
USER_CREDENTIALS = {
    "admin": "password123",
    "manager": "fleet2026"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; margin-top: 50px;'>🚗 Vehicle Fleet ML System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>🔒 Secure Dashboard Access</h4><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
        st.info("💡 Hint: Use admin / password123 to log in.")
    st.stop() # Prevents the rest of the script from executing until authenticated

# --- Data Loading & Model Training ---
# Cache data generation and model training so it doesn't run on every slider change!
@st.cache_data
def load_and_prep_data():
    df = generate_data(5000)
    features, splits = preprocess_data(df)
    return df, features, splits

@st.cache_resource
def load_models(_splits):
    return train_models(_splits)

try:
    with st.spinner("Loading Dashboard Data & Models..."):
        df, features, splits = load_and_prep_data()
        models = load_models(splits)
except Exception as e:
    st.error(f"Error initializing data or models: {e}")
    st.stop()

st.title("🚗 Vehicle Fleet ML Dashboard")

# Shows after successful login
st.markdown("Interactive dashboard with weekly analysis and machine learning predictions for vehicle maintenance.")

# --- Sidebar Navigation ---
st.sidebar.button("Logout", on_click=lambda: st.session_state.update(logged_in=False))
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Weekly Analysis", "Maintenance Data", "Predictive Diagnostics"])

# --- 1. Weekly Analysis Page ---
if page == "Weekly Analysis":
    st.header("📅 Weekly Issue Trends Analysis")
    
    # Aggregate data by week
    df['timestamp'] = pd.to_datetime(df['timestamp']) # Ensure it's datetime
    
    # Let users filter by vehicle if they want
    vehicles = ["All Vehicles"] + list(sorted(df['vehicle_id'].unique()))
    selected_vehicle = st.selectbox("Filter by Vehicle ID", vehicles)
    
    analysis_df = df.copy()
    if selected_vehicle != "All Vehicles":
        analysis_df = analysis_df[analysis_df['vehicle_id'] == selected_vehicle]
    
    # Group by week and sum the issues
    weekly_data = analysis_df.groupby(pd.Grouper(key='timestamp', freq='W'))[['fuel_issue', 'brake_line_failure', 'engine_maintenance_required']].sum().reset_index()
    
    # Interactive Plotly line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_data['timestamp'], y=weekly_data['fuel_issue'], mode='lines+markers', name='Fuel Issues'))
    fig.add_trace(go.Scatter(x=weekly_data['timestamp'], y=weekly_data['brake_line_failure'], mode='lines+markers', name='Brake Failures'))
    fig.add_trace(go.Scatter(x=weekly_data['timestamp'], y=weekly_data['engine_maintenance_required'], mode='lines+markers', name='Engine Maintenance'))
    
    fig.update_layout(
        title="Fleet Issues Reported Per Week (Past Year)",
        xaxis_title="Date (Weekly)",
        yaxis_title="Number of Issues",
        hovermode="x unified",
        template="plotly_white",
        legend_title="Issue Type",
        height=500
    )
    st.plotly_chart(fig, width="stretch")
    
    st.subheader("Weekly Summary Report")
    
    # Format dataframe for display
    display_weekly = weekly_data.rename(columns={
        'timestamp': 'Week Starting',
        'fuel_issue': 'Total Fuel Issues',
        'brake_line_failure': 'Total Brake Failures',
        'engine_maintenance_required': 'Total Engine Maint.'
    }).sort_values('Week Starting', ascending=False)
    
    # Don't show index
    st.dataframe(display_weekly, width="stretch", hide_index=True)

# --- 2. Maintenance Data Page ---
elif page == "Maintenance Data":
    st.header("🗄️ Raw Telemetry Data Explorer")
    
    st.dataframe(df, width="stretch", hide_index=True)
    
    # Report Generation (CSV Download)
    st.markdown("---")
    st.subheader("📥 Generate Telemetry Report")
    st.markdown("Download the current dataset view (including names, plates, and nearest petrol stations) as a CSV file.")
    
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df(df)
    
    st.download_button(
        label="Download Report as CSV",
        data=csv_data,
        file_name='vehicle_telemetry_report.csv',
        mime='text/csv',
    )
    
    st.markdown("---")
    st.subheader("Correlation Analysis")
    # Exclude non-numeric or irrelevant columns for correlation
    corr_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['timestamp', 'vehicle_id']]
    corr = df[corr_cols].corr()
    
    fig = px.imshow(
        corr, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale="RdBu_r",
        title="Feature and Target Correlation Heatmap"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")

# --- 3. Predictive Diagnostics Page ---
elif page == "Predictive Diagnostics":
    st.header("🔮 Interactive Predictive Diagnostics")
    st.markdown("Adjust the sensor readings below to see real-time ML predictions for potential vehicle issues. These predictions use the Random Forest models trained on the synthetic dataset.")
    
    # Create layout columns for sliders
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Fuel System Sensors")
        fuel_level = st.slider("Fuel Level (%)", 0.0, 100.0, 50.0)
        fuel_pressure = st.slider("Fuel Pressure (psi)", 0.0, 100.0, 30.0) # Default setup for issue 
        fuel_consumption = st.slider("Fuel Consumption Rate", 0.0, 50.0, 25.0) # Default setup for issue
        
    with col2:
        st.subheader("Brake System Sensors")
        brake_pressure = st.slider("Brake Pressure (psi)", 0.0, 2000.0, 1000.0)
        brake_fluid = st.slider("Brake Fluid Level (%)", 0.0, 100.0, 80.0)
        brake_temp = st.slider("Brake Temperature (°C)", 0.0, 800.0, 250.0)
        
    with col3:
        st.subheader("Engine Sensors")
        engine_temp = st.slider("Engine Temperature (°C)", 100.0, 350.0, 200.0)
        engine_vib = st.slider("Engine Vibration", 0.0, 15.0, 3.0)
        engine_rpm = st.slider("Engine RPM", 0.0, 8000.0, 2500.0)
        
    # Additional sensors
    col4, col5, col6 = st.columns(3)
    with col4:
        oil_level = st.slider("Oil Level (%)", 0.0, 100.0, 80.0)
    with col5:
        mileage = st.slider("Mileage", 0.0, 400000.0, 50000.0)
    with col6:
        maint_history = st.selectbox("Maintenance History (0=Good, 1=Issues in past)", [0, 1])

    new_reading = {
        'fuel_level': fuel_level,
        'fuel_pressure': fuel_pressure,
        'fuel_consumption_rate': fuel_consumption,
        'brake_pressure': brake_pressure,
        'brake_fluid_level': brake_fluid,
        'brake_temperature': brake_temp,
        'engine_temperature': engine_temp,
        'engine_vibration': engine_vib,
        'engine_rpm': engine_rpm,
        'oil_level': oil_level,
        'mileage': mileage,
        'maintenance_history': maint_history
    }
    
    st.markdown("---")
    st.subheader("Prediction Results")
    
    # Run prediction
    predictions = predict_vehicle_issues(models, features, new_reading)
    
    # Display results prominently with styling
    res_col1, res_col2, res_col3 = st.columns(3)
    
    def display_status(col, title, status):
        color = "#ff4b4b" if status == "Issue Detected" else "#00cc66" # Streamlit red/green
        icon = "⚠️" if status == "Issue Detected" else "✅"
        
        html_str = f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {color}; text-align: center;">
            <p style="margin:0; font-size: 1.2rem; color: #31333F;"><b>{title}</b></p>
            <h3 style="margin:10px 0 0 0; color: {color};">{icon} {status}</h3>
        </div>
        """
        col.markdown(html_str, unsafe_allow_html=True)

    display_status(res_col1, "Fuel System", predictions['fuel_issue'])
    display_status(res_col2, "Brake System", predictions['brake_line_failure'])
    display_status(res_col3, "Engine", predictions['engine_maintenance_required'])

