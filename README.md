# Spatiotemporal Forecasting and Predictive Modeling in Airline Operations with SARIMA

This project explores time series forecasting and predictive modeling in the context of airline operations. Using a real-world inspired airline passenger dataset, the project applies SARIMA (Seasonal ARIMA) models to forecast future passenger trends while also analyzing key patterns in passenger demographics, flight routes, and delays.

## 📊 Project Objectives

- Forecast monthly passenger traffic using SARIMA modeling.
- Perform grid search for optimal SARIMA parameters based on AIC.
- Evaluate model performance using RMSE and MAPE.
- Analyze passenger trends based on age, nationality, gender, and airport data.
- Visualize flight patterns across continents and temporal behaviors.

---

## 📁 Dataset

The dataset contains the following fields:
- `Passenger ID`, `First Name`, `Last Name`, `Gender`, `Age`, `Nationality`
- `Airport Name`, `Country Code`, `Continent`, `Departure Date`, `Arrival Airport`
- `Flight Status` (e.g., On Time, Delayed)

> **Note:** The dataset used in this project is anonymized and contains synthetic data for illustrative purposes.

---

## 🔧 Features

- **Time Series Forecasting** using SARIMA with seasonal components.
- **Grid Search Optimization** for hyperparameter tuning.
- **Evaluation Metrics**: RMSE and MAPE for model accuracy.
- **Visualizations**: Line plots, forecast intervals, demographic heatmaps.
- **EDA**: Exploratory analysis on passenger trends and delays.

---

## 🧪 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/airline-sarima-forecasting.git
cd airline-sarima-forecasting
```

### 2. Install Dependencies
Make sure you have Python 3.8+ and install required libraries:
```bash
pip install -r requirements.txt
```

### 3. Run the Main Script
```bash
python airline_forecast.py
```

---

## 💠 Requirements

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `statsmodels`

(Install with `pip install -r requirements.txt`)

---

## 📈 Sample Output

- Actual vs Predicted Plots
- Future Forecast Graph with Confidence Intervals
- Forecasted Passenger Count for the Next Quarter
- RMSE and MAPE Metrics
- Demographic Distribution Charts

---

## 📚 Folder Structure

```
.
├── airline_forecast.py       # Main forecasting script
├── Airline Dataset.csv       # Input dataset
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── plots/                    # Saved visualizations
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change or add.

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 📬 Contact

For questions, feedback, or collaboration:  
**[noor.cs2@yahoo.com](mailto:noor.cs2@yahoo.com)**  
or visit **[https://noorcs39.github.io/Nooruddin](https://noorcs39.github.io/Nooruddin)**

