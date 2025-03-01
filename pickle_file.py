import pickle
from prophet import Prophet
import pandas as pd

# Specify the target and model file
target = 'temperature_2m'  # Example target
model_path = f'artifacts/{target}_prophet_model.pkl'

# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define a specific date or date range for prediction (e.g., hourly for one week starting March 1, 2025)
start_date = pd.to_datetime('2025-03-01 00:00:00')
end_date = start_date + pd.Timedelta(hours=167)  # 7 days * 24 hours = 168 hours, so 167 steps forward

# Create a future DataFrame with the specific dates
future_dates = pd.date_range(start=start_date, end=end_date, freq='H')
future_df = pd.DataFrame({'ds': future_dates})

# Make predictions
forecast = model.predict(future_df)

# Display or save the predictions
print(f"\nPredictions for {target} from {start_date} to {end_date}:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# Optionally, save to CSV
forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': target}).to_csv(
    f'artifacts/{target}_custom_forecast.csv', index=False, date_format='%Y-%m-%d %H:%M:%S'
)
print(f"Saved custom forecast for {target} to artifacts/{target}_custom_forecast.csv")