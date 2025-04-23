import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from itertools import product
import warnings

# Ignore warnings from statsmodels
warnings.filterwarnings("ignore")


# Load time series data
def load_data():
    """
    Loads the time series data and ensures frequency information is set.

    Returns:
        pd.DataFrame: Time series data with a datetime index.
    """
    try:
        date_rng = pd.date_range(start="2010-01-01", end="2022-12-31", freq="M")
        data = pd.DataFrame(date_rng, columns=["date"])
        data.set_index("date", inplace=True)
        data["value"] = np.random.rand(len(data)) * 100  # Replace as needed
        data.index.freq = "M"
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


# Preprocess time series data
def preprocess_data(data):
    """
    Preprocesses the dataset by splitting it into training and testing sets.

    Args:
        data (pd.DataFrame): Time series data with a datetime index.

    Returns:
        tuple: Training and testing datasets.
    """
    try:
        train_data = data[:'2021-12-31']
        test_data = data['2022-01-01':]
        print("Training Data Shape:", train_data.shape)
        print("Testing Data Shape:", test_data.shape)
        return train_data, test_data
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        raise


# Fit a SARIMA model
def fit_sarima_model(train_data, order, seasonal_order):
    """
    Fits a SARIMA model to the training data.

    Args:
        train_data (pd.Series): Training data.
        order (tuple): Non-seasonal order (p, d, q).
        seasonal_order (tuple): Seasonal order (P, D, Q, s).

    Returns:
        SARIMAXResults: Fitted SARIMA model.
    """
    try:
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        return model_fit
    except Exception as e:
        print(f"Error fitting SARIMA model: {e}")
        raise


# Perform grid search for SARIMA parameters
def grid_search_sarima(train_data):
    """
    Performs grid search to find the best SARIMA parameters based on AIC.

    Args:
        train_data (pd.Series): Training data.

    Returns:
        tuple: Best SARIMA parameters (order, seasonal_order).
    """
    try:
        p = d = q = range(0, 2)
        seasonal_p = seasonal_d = seasonal_q = range(0, 2)
        pdq = list(product(p, d, q))
        seasonal_pdq = [
            (x[0], x[1], x[2], 12) for x in product(seasonal_p, seasonal_d,
                                                    seasonal_q)
        ]
        best_aic = float('inf')
        best_params = None

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    model_fit = fit_sarima_model(train_data, param,
                                                 param_seasonal)
                    aic = model_fit.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (param, param_seasonal)
                except:
                    continue

        print("Best SARIMA Parameters:", best_params)
        return best_params
    except Exception as e:
        print(f"Error in grid search: {e}")
        raise


# Evaluate the model
def calculate_metrics(test_data, predictions):
    """
    Calculates RMSE and MAPE for model evaluation.

    Args:
        test_data (pd.Series): True values of the test data.
        predictions (pd.Series): Predicted values from the model.

    Returns:
        tuple: RMSE and MAPE values.
    """
    try:
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        return rmse, mape
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        raise


# Main workflow
try:
    # Load and preprocess data
    data = load_data()
    train_data, test_data = preprocess_data(data)

    # Perform grid search for best parameters
    best_params = grid_search_sarima(train_data)
    best_order, best_seasonal_order = best_params

    # Fit the SARIMA model
    model_fit = fit_sarima_model(train_data, best_order, best_seasonal_order)

    # Make predictions
    start_index = test_data.index[0]
    end_index = test_data.index[-1]
    predictions = model_fit.predict(start=start_index, end=end_index)

    # Evaluate the model
    calculate_metrics(test_data["value"], predictions)

    # Forecast future bookings
    n_steps = 3  # Number of steps to forecast (next quarter = 3 months)
    forecast = model_fit.get_forecast(steps=n_steps)

    # Extract forecast mean and confidence intervals
    mean_forecast = forecast.predicted_mean
    conf_interval = forecast.conf_int()

    # Print forecasted values
    print("Forecast for the next quarter:")
    print(mean_forecast)

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data, label='Actual')
    plt.plot(test_data.index, predictions, label='Predicted')
    plt.title('Actual vs Predicted Bookings')
    plt.xlabel('Date')
    plt.ylabel('Number of Bookings')
    plt.legend()
    plt.show()

    # Plot historical data with forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, label='Historical Data')
    plt.plot(mean_forecast.index, mean_forecast, label='Forecast')
    plt.fill_between(
        conf_interval.index,
        conf_interval.iloc[:, 0],  # Lower bound
        conf_interval.iloc[:, 1],  # Upper bound
        color='lightblue',
        alpha=0.5,
        label='Confidence Interval'
    )
    plt.title('Historical Data and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Number of Bookings')
    plt.legend()
    plt.show()

except Exception as e:
    print(f"An unexpected error occurred: {e}")
