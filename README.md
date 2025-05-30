# Wind Energy Power Prediction and Forecast

This project provides a web-based application for predicting and forecasting wind energy power output using multiple machine learning models. The application leverages Streamlit for an interactive interface, allowing users to input weather features to predict wind power and forecast power output over a specified date range (2017-2025) using pre-forecasted input feature data. The models include Linear Regression, Lasso, Ridge, K-Neighbors Regressor, Decision Tree, XGBRegressor, CatBoosting Regressor, and AdaBoost Regressor, with predictions based on raw, unscaled input features.

## Features

- **Wind Power Prediction**: Users can input weather features (`temperature_2m`, `relativehumidity_2m`, `dewpoint_2m`, `windspeed_10m`, `windspeed_100m`, `winddirection_10m`, `winddirection_100m`, `windgusts_10m`) via sliders and numeric inputs to predict wind power output instantly using one of eight machine learning models.
- **Model Selection**: Choose from eight prediction models (Linear Regression, Lasso, Ridge, K-Neighbors Regressor, Decision Tree, XGBRegressor, CatBoosting Regressor, AdaBoost Regressor) to predict and forecast wind power.
- **Wind Power Forecasting **: Forecast wind power output for a user-specified date range  using pre-forecasted input features from forecasted dataset, plotted interactively with Plotly.
- **Interactive Visualization**: Use Plotly to display interactive plots of predicted vs. forecasted wind power, with a range slider for easy navigation.
- **Performance Metrics**: Display prediction and forecasting times for performance monitoring.
- Streamlit link https://windenergypowerforecast-8hkutcfqkwzskfelragrjc.streamlit.app

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/wind-energy-power-forecast.git
   cd wind-energy-power-forecast

  ```bash
  git add README.md
  git commit -m "Update README.md to reflect current wind energy power prediction and forecast project"
  git push origin main

```
2.
   ![Wind Power Forecast Example](./Power_forecast.jpg)

