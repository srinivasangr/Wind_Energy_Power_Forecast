# Wind Energy Power Prediction and Forecast

This project provides a web-based application for predicting and forecasting wind energy power output using multiple machine learning models. The application leverages Streamlit for an interactive interface, allowing users to input weather features to predict wind power and forecast power output over a specified date range (2017-2025) using pre-forecasted input feature data. The models include Linear Regression, Lasso, Ridge, K-Neighbors Regressor, Decision Tree, XGBRegressor, CatBoosting Regressor, and AdaBoost Regressor, with predictions based on raw, unscaled input features.

## Features

- **Wind Power Prediction**: Users can input weather features (`temperature_2m`, `relativehumidity_2m`, `dewpoint_2m`, `windspeed_10m`, `windspeed_100m`, `winddirection_10m`, `winddirection_100m`, `windgusts_10m`) via sliders and numeric inputs to predict wind power output instantly using one of eight machine learning models.
- **Model Selection**: Choose from eight prediction models (Linear Regression, Lasso, Ridge, K-Neighbors Regressor, Decision Tree, XGBRegressor, CatBoosting Regressor, AdaBoost Regressor) to predict and forecast wind power.
- **Wind Power Forecasting (2017-2025)**: Forecast wind power output for a user-specified date range (e.g., 2017-01-02 to 2025-12-30) using pre-forecasted input features from `combined_forecast.csv`, plotted interactively with Plotly.
- **Interactive Visualization**: Use Plotly to display interactive plots of predicted vs. forecasted wind power, with a range slider for easy navigation.
- **Performance Metrics**: Display prediction and forecasting times for performance monitoring.

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/wind-energy-power-forecast.git
   cd wind-energy-power-forecast
Install Dependencies: Ensure you have Python 3.8+ installed. Then, install the required packages using pip:
bash
Wrap
Copy
pip install -r requirements.txt
Create a requirements.txt file in your repository with the following content if it doesn’t exist:
text
Wrap
Copy
streamlit
pandas
numpy
plotly
scikit-learn
xgboost
catboost
Prepare Data and Models:
Place your input data file Location1_final.csv in the project root directory.
Ensure the artifacts folder contains:
Prediction model .pkl files: Linear_Regression_model.pkl, Lasso_model.pkl, Ridge_model.pkl, K-Neighbors_Regressor_model.pkl, Decision_Tree_model.pkl, XGBRegressor_model.pkl, CatBoosting_Regressor_model.pkl, AdaBoost_Regressor_model.pkl.
Pre-forecasted input feature data combined_forecast.csv with columns for Date, temperature_2m, relativehumidity_2m, dewpoint_2m, windspeed_10m, windspeed_100m, winddirection_10m, winddirection_100m, and windgusts_10m.
The image file Wind_turbine.JPG for the app’s UI.
Run the App: Launch the Streamlit app locally:
bash
Wrap
Copy
streamlit run app.py
Usage
Power Prediction: On the app’s “Wind Power Prediction” section, input weather feature values using sliders and number inputs, select a prediction model, and click “Predict Power” to see the predicted wind power output and prediction time.
Power Forecasting: In the “Wind Power Forecasting (2017-2025)” section, specify a date range (e.g., 2017-01-02 to 2025-12-30), click “Forecast Power” to generate and visualize a forecast of wind power output using pre-forecasted input features, and view the forecasting time.
Dependencies
Python 3.8+
Streamlit
Pandas
NumPy
Plotly
Scikit-learn
XGBoost
CatBoost
Notes
The app assumes all prediction models were trained on raw, unscaled input features matching the columns in combined_forecast.csv.
Ensure combined_forecast.csv covers the full date range (2017-2025) with hourly data and proper datetime formatting (e.g., YYYY-MM-DD HH:MM:SS).
The app is optimized for performance using Streamlit caching and efficient data handling, but forecasting large date ranges may take a few minutes depending on the data size and model complexity.
Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request with a description of your changes.

License
[MIT License] or specify your preferred license here.

Contact
For questions or feedback, contact [yourusername@example.com] or open an issue on this repository.

text
Wrap
Copy

### Key Changes and Explanations
1. **Project Title and Description**:
   - Updated the title and description to reflect the current focus on wind energy power prediction and forecasting using multiple machine learning models (not LSTM, XGBoost, Prophet, or SARIMA, as those were replaced or omitted in your latest app).

2. **Features**:
   - Rewrote the features section to describe the current app’s functionality: user input for power prediction, model selection, forecasting using `combined_forecast.csv`, and interactive Plotly visualizations.
   - Removed references to LSTM, Prophet, and SARIMA, as they’re no longer used.

3. **Installation**:
   - Updated the installation instructions to match your current app’s dependencies and file structure (e.g., `Location1_final.csv`, `artifacts` folder, `.pkl` files, and `combined_forecast.csv`).
   - Specified the need for `catboost` in `requirements.txt`.

4. **Usage**:
   - Detailed how users interact with the app for both prediction and forecasting, including input features, model selection, and date range inputs.

5. **Dependencies**:
   - Listed only the libraries used in your current code (Streamlit, Pandas, NumPy, Plotly, Scikit-learn, XGBoost, CatBoost).

6. **Notes**:
   - Added notes about raw, unscaled input features, the expected format of `combined_forecast.csv`, and performance optimization.

7. **Contributing and Contact**:
   - Added placeholder sections for contributing guidelines and contact information. Replace `[yourusername@example.com]` with your actual contact details.

### How to Use This `README.md`
- Save this content as `README.md` in your GitHub repository (e.g., `wind-energy-power-forecast`).
- Commit and push the changes to your repository:
  ```bash
  git add README.md
  git commit -m "Update README.md to reflect current wind energy power prediction and forecast project"
  git push origin main
