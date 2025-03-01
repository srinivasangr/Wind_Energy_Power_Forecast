import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os
import time
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor

# Streamlit interface
st.title('Wind Energy Power Prediction and Forecast')

# Display image (same as original)
st.image('Wind_turbine.JPG', use_container_width=True)

# 1. Power Prediction Section
st.header('Wind Power Prediction')
st.write('Enter the input features to predict wind power output.')

# Define input features with limits and sliders
features = {
    'temperature_2m': {'type': 'slider', 'min': -32, 'max': 102, 'default': 45, 'label': 'Temperature (2m) (°C)'},
    'relativehumidity_2m': {'type': 'slider', 'min': 0, 'max': 100, 'default': 70, 'label': 'Relative Humidity (2m) (%)'},
    'dewpoint_2m': {'type': 'slider', 'min': -36, 'max': 78, 'default': 15, 'label': 'Dew Point (2m) (°C)'},
    'windspeed_10m': {'type': 'number', 'min': 0, 'max': 20, 'default': 5, 'label': 'Wind Speed (10m) (m/s)'},
    'windspeed_100m': {'type': 'number', 'min': 0, 'max': 30, 'default': 10, 'label': 'Wind Speed (100m) (m/s)'},
    'winddirection_10m': {'type': 'number', 'min': 0, 'max': 360, 'default': 180, 'label': 'Wind Direction (10m) (degrees)'},
    'winddirection_100m': {'type': 'number', 'min': 0, 'max': 360, 'default': 180, 'label': 'Wind Direction (100m) (degrees)'},
    'windgusts_10m': {'type': 'number', 'min': 0, 'max': 30, 'default': 8, 'label': 'Wind Gusts (10m) (m/s)'}
}

user_inputs = {}
for feature, config in features.items():
    if config['type'] == 'slider':
        user_inputs[feature] = st.slider(config['label'], min_value=config['min'], max_value=config['max'], value=config['default'])
    else:
        user_inputs[feature] = st.number_input(config['label'], min_value=config['min'], max_value=config['max'], value=config['default'])

# Select prediction model (excluding Random Forest)
model_options = ["Linear Regression", "Lasso", "Ridge", "K-Neighbors Regressor", "Decision Tree", 
                 "XGBRegressor", "CatBoosting Regressor", "AdaBoost Regressor"]
selected_model = st.selectbox('Select Prediction Model', model_options)

# Load models
@st.cache_resource
def load_models():
    """Load prediction models from artifacts folder (faster with resource caching)."""
    models = {
        "Linear Regression": pickle.load(open('artifacts/Linear_Regression_model.pkl', 'rb')),
        "Lasso": pickle.load(open('artifacts/Lasso_model.pkl', 'rb')),
        "Ridge": pickle.load(open('artifacts/Ridge_model.pkl', 'rb')),
        "K-Neighbors Regressor": pickle.load(open('artifacts/K-Neighbors_Regressor_model.pkl', 'rb')),
        "Decision Tree": pickle.load(open('artifacts/Decision_Tree_model.pkl', 'rb')),
        "XGBRegressor": pickle.load(open('artifacts/XGBRegressor_model.pkl', 'rb')),
        "CatBoosting Regressor": pickle.load(open('artifacts/CatBoosting_Regressor_model.pkl', 'rb')),
        "AdaBoost Regressor": pickle.load(open('artifacts/AdaBoost_Regressor_model.pkl', 'rb'))
    }
    return models

models = load_models()

# Predict power
if st.button('Predict Power'):
    start_time = time.time()
    input_df = pd.DataFrame([user_inputs])  # Use raw inputs directly
    
    # Predict using the selected model
    prediction = models[selected_model].predict(input_df)[0]
    st.write(f'Predicted Wind Power: {prediction:.2f} units')
    end_time = time.time()
    st.write(f"Prediction time: {end_time - start_time:.2f} seconds")

# 2. Power Forecasting Section (2017-2025)
st.header('Wind Power Forecasting (2017-2025)')
st.write('Forecast wind power output using pre-forecasted input features from combined_forecast.csv.')

# Date range input for forecasting
start_date = st.date_input('Start Date', value=pd.to_datetime('2017-01-02 00:00:00'))
end_date = st.date_input('End Date', value=pd.to_datetime('2025-12-30 23:00:00'))

if start_date > end_date:
    st.error('Error: End date must fall after start date.')

# Load combined forecast CSV
@st.cache_resource
def load_combined_forecast():
    """Load combined forecast CSV with caching for performance and ensure full date range."""
    df = pd.read_csv('artifacts/combined_forecast.csv', parse_dates=['Date'], index_col='Date')
    st.write("Combined Forecast Index Sample:", df.index[:5])  # Debug: Show first 5 index values
    return df

combined_forecast_df = load_combined_forecast()

if st.button('Forecast Power'):
    if start_date <= end_date:
        start_time = time.time()
        
        # Filter the combined forecast for the selected date range
        forecast_dates = pd.date_range(start=start_date, end=end_date, freq='h')
        st.write("Forecast Dates Sample:", forecast_dates[:5])  # Debug: Show first 5 forecast dates
        
        forecast_input = combined_forecast_df.loc[forecast_dates, 
                                                ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 
                                                 'windspeed_10m', 'windspeed_100m', 'winddirection_10m', 
                                                 'winddirection_100m', 'windgusts_10m']]
        
        # Ensure no missing values (fill with forward fill or interpolate if needed)
        forecast_input = forecast_input.fillna(method='ffill').fillna(method='bfill')
        
        # Predict wind power using the selected model (raw data, no scaling)
        power_forecast = models[selected_model].predict(forecast_input)
        forecast_output = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted_Power': power_forecast
        })
        
        # Plot forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_output['Date'], y=forecast_output['Predicted_Power'], 
                                 mode='lines', name='Forecasted Power', line=dict(color='red')))
        fig.update_layout(title='Wind Power Forecast (2017-2025)',
                          xaxis_title='Date',
                          yaxis_title='Power',
                          xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
        end_time = time.time()
        st.write(f"Forecasting time: {end_time - start_time:.2f} seconds")
