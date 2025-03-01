from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

# Create artifacts directory (if not already present in Kaggle)
os.makedirs('artifacts', exist_ok=True)

def train_and_forecast_prophet(df, ds, target, periods=8760*4, freq='H'):
    """
    Train a Prophet model, forecast, and return the model, historical data, and forecast with datetime indices.
    """
    # Prepare data for Prophet
    prophet_df = df[[ds, target]].rename(columns={ds: 'ds', target: 'y'})
    
    # Train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_df)
    
    # Create future dataframe and predict
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    
    # Save the model as a pickle file
    model_filename = f'artifacts/{target}_prophet_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Saved {target} Prophet model to {model_filename}")
    
    return model, prophet_df['y'], forecast['yhat']

# Define target variables
target_variables = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 
                    'windspeed_10m', 'windspeed_100m', 'winddirection_10m', 
                    'winddirection_100m', 'windgusts_10m']

# Initialize a dictionary to store forecasts for all targets
combined_forecast = {}

# Train and forecast for each target variable
for target in target_variables:
    print(f"\nTraining Prophet model for {target}...")
    model, historical, forecast = train_and_forecast_prophet(df, 'Time', target)
    
    # Store forecast with datetime as index
    forecast_df = pd.DataFrame({
        'Date': forecast.index,  # Use the 'ds' column (datetime) from Prophet's forecast
        target: forecast.values
    })
    combined_forecast[target] = forecast_df[target]
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(historical.index, historical, label='Historical (Hourly)', color='blue', alpha=0.5)
    plt.plot(forecast.index, forecast, label='Forecast (4 Years Hourly)', color='orange')
    plt.title(f'Prophet Forecast for {target}')
    plt.xlabel('Date')
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'artifacts/{target}_forecast.png')
    plt.show()

# Combine all forecasts into a single DataFrame with datetime
# Use the 'ds' (datetime) from the last forecast (they're all aligned)
last_forecast_dates = pd.DataFrame({'Date': forecast.index})  # Use actual datetime from forecast
combined_df = pd.DataFrame({'Date': last_forecast_dates['Date']})
for target in target_variables:
    combined_df[target] = combined_forecast[target]

# Save combined forecast to CSV with proper datetime format
combined_csv_path = 'artifacts/combined_forecast.csv'
combined_df.to_csv(combined_csv_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
print(f"Saved combined forecast to {combined_csv_path}")

# Optional: Display the first few rows of the combined forecast
print("\nCombined Forecast Preview:")
print(combined_df.head())