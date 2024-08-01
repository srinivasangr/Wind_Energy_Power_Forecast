import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Function to load data
def load_data():
    df = pd.read_csv('Location1_final.csv', parse_dates=['ds'], index_col='ds')
    return df

# Function to preprocess data
def preprocess_data(df, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length, 1:]
            y = data[i+seq_length, 0]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    X, y = create_sequences(scaled_data, seq_length)
    return X, y, scaler

# Function to plot results and display accuracy metrics
def plot_and_evaluate(df, model, start_date, end_date):
    # Filter data based on user input
    df_lstm = df[['y', 'Wind_power_density', 'daily_ma_y', 'Windspeed_100m_lagged_1', 'Windspeed_100m_lagged_2', 
                  'daily_ma_wind', 'windspeed_100m', 'Wind_shear', 'Wind_cos_100m', 'windgusts_10m', 'temperature_sum']]
    
    SEQ_LENGTH = 24
    X, y, scaler = preprocess_data(df_lstm, SEQ_LENGTH)
    
    predictions = model.predict(X)
    predictions_reshaped = np.zeros((predictions.shape[0], len(df_lstm.columns)))
    predictions_reshaped[:, 0] = predictions.flatten()
    predictions_original_scale = scaler.inverse_transform(predictions_reshaped)
    predictions_power = predictions_original_scale[:, 0]
    
    # Create a DataFrame with predictions and datetime index
    predictions_df = pd.DataFrame(predictions_power, index=df.index[SEQ_LENGTH:], columns=['Predicted_Power'])
    
    # Filter the data for the plot
    actual_data = df.loc[start_date:end_date, 'y']
    predicted_data = predictions_df.loc[start_date:end_date, 'Predicted_Power']
    
    # Create interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data, mode='lines', name='Actual',line=dict(color='blue' )))
    fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data, mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted Power',
                      xaxis_title='Time',
                      yaxis_title='Power',
                      xaxis_rangeslider_visible=True)
    
    # Display plot in Streamlit
    st.plotly_chart(fig)
    
    # Accuracy metrics
    mse = mean_squared_error(actual_data, predicted_data)
    mae = mean_absolute_error(actual_data, predicted_data)
    r2 = r2_score(actual_data, predicted_data)
    
    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'Mean Absolute Error (MAE): {mae}')
    st.write(f'R-squared (R²): {r2}')

# Streamlit interface
st.title('Wind Energy Power (y) Forecast')

# Display image
st.image('Wind_turbine.jpg', use_column_width=True)

# Load the model
model = load_model('lstm_model.h5')

# Load and display the data
df = load_data()
#st.write('Data:', df)

# Input from the user
min_date = pd.to_datetime('2017-04-02')
max_date = pd.to_datetime('2021-12-31')

start_date = st.date_input('Select start date', value=pd.to_datetime('2021-01-01'), min_value=min_date, max_value=max_date)
end_date = st.date_input('Select end date', value=pd.to_datetime('2021-12-31'), min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.error('Error: End date must fall after start date.')

# Plot and evaluate
if st.button('Plot and Evaluate'):
    if start_date <= end_date:
        plot_and_evaluate(df, model, start_date, end_date)
    else:
        st.error('Error: End date must fall after start date.')
