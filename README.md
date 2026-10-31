# Real-Time Air Quality Prediction Using Machine Learning

**A machine learning pipeline for PM2.5 concentration prediction using historical air quality observations, ensemble regression models, and real-time pollutant data from the World Air Quality Index (WAQI) API.**

## Overview

Air pollution monitoring is essential for understanding environmental conditions and assessing potential health risks. This project implements a machine learning-based approach to predict **PM2.5 concentrations from other atmospheric pollutant measurements**, using historical air quality data to train regression models and live pollutant observations to generate predictions.

The system combines data preprocessing, exploratory data analysis (EDA), feature correlation analysis, ensemble regression, model evaluation, real-time API integration, and automated prediction refreshes in a single Python workflow.

Two regression algorithms—**Random Forest Regressor and XGBoost Regressor**—are trained and evaluated using Root Mean Squared Error (RMSE). The model with the lower test RMSE is selected for subsequent predictions.

The selected model receives real-time pollutant measurements from the WAQI API and estimates PM2.5 concentrations. The predicted concentration is then mapped to a categorical air-quality label and visualized using a color-coded gauge.

## Key Features

- **Data preprocessing:** Transforms pollutant observations from long-format CSV data into a station- and timestamp-indexed wide-format dataset.
- **Exploratory data analysis:** Examines pollutant distributions, correlations, station-level PM2.5 trends, and relationships between PM2.5 and other pollutants.
- **Ensemble regression:** Trains Random Forest and XGBoost regression models for PM2.5 estimation.
- **Automated model selection:** Selects the model with the lowest RMSE on the held-out test dataset.
- **Real-time API integration:** Retrieves live pollutant measurements from the WAQI API.
- **Feature transformation:** Applies the fitted `StandardScaler` to real-time input features before inference.
- **Air-quality classification:** Maps predicted PM2.5 concentrations to predefined categorical labels.
- **Visualization:** Generates pollutant correlation heatmaps, distribution plots, temporal trend charts, scatter plots, box plots, and a PM2.5 gauge.
- **Automated refresh:** Supports repeated predictions at configurable intervals.

## System Architecture

The implementation follows a data-to-prediction pipeline:

```text
                 HISTORICAL DATA
                        |
                        v
                CSV Data Ingestion
                        |
                        v
             Data Pivoting & Cleaning
                        |
                        v
              Exploratory Data Analysis
                        |
                        v
              Feature / Target Selection
                        |
                        v
              Missing-Value Handling
                        |
                        v
                  StandardScaler
                        |
                        v
                Train-Test Split
                        |
             +----------+----------+
             |                     |
             v                     v
      Random Forest              XGBoost
       Regressor                Regressor
             |                     |
             v                     v
         Test RMSE              Test RMSE
             |                     |
             +----------+----------+
                        |
                        v
                Best Model Selection
                        |
                        v
               Trained Model + Scaler
                        |
                        v
              WAQI Real-Time API Data
                        |
                        v
              Input Feature Extraction
                        |
                        v
                Feature Scaling
                        |
                        v
                 PM2.5 Prediction
                        |
                        v
              AQI Category Mapping
                        |
             +----------+----------+
             |                     |
             v                     v
       Console Output          Gauge Plot
                        |
                        v
               Scheduled Refresh Loop
```

**Important:** The current implementation performs a random train-test split and fits the scaler before splitting. For a production-grade evaluation, the scaler should be fitted only on the training data, and time-ordered validation should be considered for temporal forecasting tasks.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Regression algorithms | Random Forest, XGBoost |
| Feature scaling | StandardScaler |
| Model evaluation | Mean Squared Error, RMSE |
| Data visualization | Matplotlib, Seaborn |
| External data integration | Requests, WAQI API |
| Data source | Historical air quality CSV dataset and live WAQI observations |
| Development environment | Jupyter Notebook / Google Colab |

## Machine Learning Methodology

### 1. Historical Data Ingestion and Transformation

The input dataset contains pollutant measurements in long format, with fields representing the monitoring station, timestamp, pollutant identifier, and average pollutant concentration.

The `pivot_and_clean_data()` function transforms these records into a wide-format DataFrame.

**Transformation strategy:**

- Groups observations by `station` and `last_update`.
- Pivots `pollutant_id` into individual pollutant columns.
- Uses `pollutant_avg` as the measurement value.
- Normalizes column names to lowercase.
- Renames `pm2.5` to `pm25` and `ozone` to `o3`.
- Removes rows where every required pollutant measurement is missing.

The resulting dataset contains the following pollutant features:

| Feature | Description |
|---|---|
| `pm25` | Fine particulate matter with an aerodynamic diameter of 2.5 micrometers or less |
| `pm10` | Particulate matter with an aerodynamic diameter of 10 micrometers or less |
| `no2` | Nitrogen dioxide |
| `so2` | Sulfur dioxide |
| `co` | Carbon monoxide |
| `o3` | Ozone |

The transformed dataset serves as the basis for exploratory analysis and model training.

### 2. Exploratory Data Analysis

EDA is used to examine the statistical properties of the pollutant measurements and identify potential relationships relevant to PM2.5 estimation.

The project implements the following visualizations:

**Correlation Heatmap**

Computes pairwise correlations among pollutant features to examine their linear relationships.

**Pollutant Distribution Analysis**

Uses histograms and kernel density estimates to inspect pollutant distributions, concentration ranges, and potential skewness.

**Station-Level Temporal Analysis**

Plots PM2.5 measurements against timestamps, grouped by monitoring station, to inspect changes in recorded concentrations over time.

**Average Pollutant Concentrations**

Compares mean pollutant measurements across the dataset using a horizontal bar chart.

**Cross-Station Distribution Analysis**

Uses box plots to examine pollutant variability and potential outliers across the observations.

**PM2.5 Relationship Analysis**

Generates scatter plots comparing PM2.5 with PM10, NO2, CO, SO2, and O3.

These analyses provide an initial understanding of the dataset before model training.

### 3. Feature Engineering and Target Definition

The prediction task is formulated as a supervised regression problem.

**Target variable:**

```python
y = df_clean["pm25"]
```

**Input features:**

```python
X = df_clean[features].drop(columns=["pm25"])
```

The model uses the following five input features:

```text
pm10, no2, so2, co, o3
```

The target is the measured PM2.5 concentration.

Rows without a PM2.5 target value are excluded from training. Missing values in the input features are replaced with zero in the current implementation.

**Technical consideration:** Replacing missing pollutant values with zero can introduce bias because zero represents a concentration measurement rather than an unavailable observation. A production implementation should evaluate alternatives such as median imputation and preserve missingness information where appropriate.

### 4. Feature Scaling

The project uses Scikit-learn's `StandardScaler` to standardize input features.

For each feature, standardization is calculated as:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- \(x\) is the original feature value.
- \(\mu\) is the feature mean.
- \(\sigma\) is the feature standard deviation.
- \(z\) is the standardized feature value.

The fitted scaler is reused during real-time inference to ensure that incoming pollutant measurements undergo the same transformation as the training inputs.

**Implementation note:** The current code fits the scaler before the train-test split. This should be changed to fit the scaler exclusively on `X_train` to avoid data leakage during model evaluation.

### 5. Model Training and Selection

The implementation trains two tree-based ensemble regression models.

#### Random Forest Regressor

Random Forest combines predictions from multiple decision trees to estimate a continuous target.

Configuration:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

The ensemble can model nonlinear relationships between pollutant concentrations and PM2.5.

#### XGBoost Regressor

XGBoost uses gradient boosting to build an ensemble of decision trees, with successive trees improving the model's predictions.

Configuration:

```python
XGBRegressor(
    n_estimators=100,
    random_state=42,
    verbosity=0
)
```

The implementation compares the two models using the same held-out test partition.

#### Model Selection Strategy

The model with the lowest test RMSE is selected as the best-performing model.

```python
rmse = mean_squared_error(
    y_test,
    y_pred,
    squared=False
)
```

RMSE is defined as:

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

Where:

- \(y_i\) is the observed PM2.5 concentration.
- \(\hat{y}_i\) is the predicted PM2.5 concentration.
- \(n\) is the number of test observations.

A lower RMSE indicates that predictions are closer to the observed values on average, with larger errors receiving greater weight.

The selected model, fitted scaler, and input feature names are returned for reuse during inference.

**Note:** The actual RMSE values depend on the input dataset and have not been included here because they are not specified in the source code.

## Real-Time Prediction Pipeline

### 1. Live Data Acquisition

The `fetch_aqi_data()` function sends an HTTP request to the WAQI API for a specified city.

The API response is parsed to extract the available individual pollutant measurements from the `iaqi` field.

Example:

```python
live_data = fetch_aqi_data("delhi")
```

The city parameter can be changed to another city supported by the API, subject to the API's station and data availability.

### 2. Input Feature Extraction

The live pollutant measurements are arranged in the same feature order used during training.

```python
input_vector = [
    live_data.get(feature, 0)
    for feature in input_features
]
```

The resulting vector is converted into a DataFrame before scaling.

**Important:** The current implementation substitutes zero for missing API features. This can produce misleading predictions if a pollutant is unavailable. A more robust implementation should validate the input features and handle missing measurements explicitly.

### 3. PM2.5 Inference

The fitted scaler transforms the live feature vector before passing it to the selected regression model.

```python
input_scaled = scaler.transform(input_df)

predicted_pm25 = model.predict(input_scaled)[0]
```

The output is a continuous numerical estimate of PM2.5 concentration based on the trained model.

### 4. Air-Quality Category Mapping

The `aqi_category()` function maps the predicted concentration to the following predefined categories:

| Predicted PM2.5 concentration | Category |
|---|---|
| ≤ 50 | Good |
| > 50 to 100 | Satisfactory |
| > 100 to 200 | Moderate |
| > 200 to 300 | Poor |
| > 300 to 400 | Very Poor |
| > 400 | Severe |

These thresholds are implemented directly in the project.

**Interpretation limitation:** This is a custom category mapping applied to predicted PM2.5 concentration. It should not be treated as an official AQI calculation. Official AQI systems can use pollutant-specific breakpoints, concentration averaging periods, and the maximum sub-index across multiple pollutants.

The project estimates PM2.5 concentration; it does not independently forecast future air quality or calculate an official multi-pollutant AQI.

## Visualization and Monitoring

### PM2.5 Prediction Gauge

The `plot_pm25_gauge()` function plots the predicted PM2.5 value against a color-coded concentration scale.

The visualization includes:

- Predefined concentration intervals.
- Category labels.
- A vertical marker indicating the predicted concentration.
- A title displaying the prediction and mapped category.

The gauge supports quick visual interpretation of the model output.

### Automated Prediction Refresh

The `auto_refresh_predict()` function executes repeated prediction cycles at a configurable interval.

Example:

```python
auto_refresh_predict(
    model,
    scaler,
    input_features,
    city="delhi",
    interval=300,
    cycles=5
)
```

This configuration requests five prediction cycles with a 300-second delay between cycles.

The function fetches fresh API data during each cycle, runs inference, and prints the result.

The current implementation performs periodic polling rather than continuous streaming. It does not include a persistent background service, a web dashboard, or a database-backed monitoring system.

## Installation and Setup

### Prerequisites

- Python 3.9 or later is recommended.
- A historical air quality CSV dataset with the required columns.
- Internet access for live WAQI API requests.
- A valid WAQI API token.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy requests scikit-learn xgboost matplotlib seaborn jupyter
```

### 4. Configure the Historical Dataset

Update the CSV path in the notebook:

```python
csv_path = "path/to/Real-time-air-quality-index.csv"
```

The dataset should contain the following columns:

```text
station
last_update
pollutant_id
pollutant_avg
```

The pollutant identifiers must correspond to the expected measurements, such as PM2.5, PM10, NO2, SO2, CO, and ozone.

### 5. Configure the WAQI API Token

Obtain an API token from the official [World Air Quality Index project](https://aqicn.org/data-platform/token/).

Do not commit API tokens or other credentials to a public repository.

Set the token as an environment variable instead:

**Windows PowerShell**

```powershell
$env:WAQI_TOKEN="your_api_token"
```

**macOS / Linux**

```bash
export WAQI_TOKEN="your_api_token"
```

Update the API function to read the token from the environment:

```python
import os
import requests

def fetch_aqi_data(city="delhi"):
    token = os.getenv("WAQI_TOKEN")

    if not token:
        raise ValueError("WAQI_TOKEN environment variable is not configured.")

    url = f"https://api.waqi.info/feed/{city}/"

    try:
        response = requests.get(
            url,
            params={"token": token},
            timeout=15
        )
        response.raise_for_status()

        data = response.json()

        if data.get("status") != "ok":
            print("Failed to fetch:", data.get("data"))
            return None

        iaqi = data["data"].get("iaqi", {})

        return {
            key.lower(): value["v"]
            for key, value in iaqi.items()
            if isinstance(value, dict) and "v" in value
        }

    except requests.RequestException as exc:
        print(f"API request failed: {exc}")
        return None
```

This version adds environment-based credential management, a request timeout, HTTP error handling, and basic response validation.

**Security note:** If an actual API token has already been committed to a public repository, revoke or rotate it and remove the exposed credential from the code.

## Execution

Open the notebook in Jupyter:

```bash
jupyter notebook Air_Quality_Index.ipynb
```

Alternatively, upload it to Google Colab and execute the cells in order.

The execution sequence is:

1. Import dependencies.
2. Load and transform the historical dataset.
3. Perform exploratory data analysis.
4. Train and evaluate the regression models.
5. Select the model with the lowest test RMSE.
6. Fetch live pollutant measurements.
7. Generate a PM2.5 prediction.
8. Map the prediction to a category.
9. Visualize the predicted concentration.
10. Optionally run repeated predictions using the refresh loop.

## Project Structure

```text
Real-Time-Air-Quality-Prediction/
│
├── Air_Quality_Index.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── data/
    └── Real-time-air-quality-index.csv
```

This is a suggested repository structure. The `data/` directory and supporting files should be created if they are not already present.

For large datasets or datasets with redistribution restrictions, document the data source and retrieval instructions instead of committing the raw data.

## Limitations and Technical Considerations

The current implementation demonstrates an end-to-end prototype, but several considerations are important when interpreting its results.

**1. Contemporaneous estimation versus forecasting**

The model predicts PM2.5 using other pollutant measurements obtained from the same API response. This is a regression-based estimation task, not a forecast of future PM2.5 concentrations.

**2. Missing-value handling**

Missing training and API inputs are replaced with zero. This may distort the feature distribution and should be replaced with a validated imputation strategy.

**3. Data leakage during scaling**

The scaler is fitted before the train-test split. The correct procedure is to split the data first and fit preprocessing transformations only on the training partition.

**4. Temporal and station-level dependencies**

A random train-test split may place observations from the same monitoring station and nearby timestamps in both partitions. This can produce optimistic evaluation results. Time-based or station-aware validation should be considered, depending on the intended use case.

**5. AQI interpretation**

The output category is based on predefined PM2.5 thresholds. It is not a substitute for an official AQI calculation.

**6. API feature availability**

The WAQI API may return a different set of pollutant measurements depending on the station and location. Missing features must be handled carefully to avoid unreliable predictions.

**7. Reproducibility**

Model performance depends on the historical dataset, data quality, feature availability, and train-test split. RMSE values should be recorded alongside the dataset version and model configuration.

## Potential Improvements

Future development could extend the project with:

- **Leakage-free preprocessing:** Integrate imputation and scaling into a Scikit-learn `Pipeline`.
- **Time-series validation:** Evaluate generalization using chronological splits or `TimeSeriesSplit`.
- **Additional evaluation metrics:** Report MAE, RMSE, and R².
- **Hyperparameter optimization:** Tune tree depth, learning rate, number of estimators, and other model parameters.
- **Feature importance analysis:** Investigate which pollutants contribute most to predictions.
- **Robust API handling:** Add retry policies, rate-limit handling, input validation, and structured logging.
- **Model persistence:** Save the trained estimator and preprocessing pipeline using `joblib`.
- **Prediction history:** Store timestamped predictions and actual measurements for retrospective analysis.
- **Interactive dashboard:** Integrate the inference pipeline with Streamlit or a web-based monitoring interface.
- **Future PM2.5 forecasting:** Incorporate lagged pollutant measurements, meteorological variables, and historical time windows to predict concentrations at future horizons.

## Skills Demonstrated

- Data preprocessing and transformation using Pandas.
- Exploratory data analysis and statistical visualization.
- Supervised machine learning and ensemble regression.
- Random Forest and gradient-boosted tree models.
- Feature scaling and train-test evaluation.
- Regression model comparison using RMSE.
- REST API integration and JSON response processing.
- Real-time inference using a trained model.
- Automated polling and configurable prediction cycles.
- Environmental data analysis.

## Conclusion

This project demonstrates an integrated machine learning workflow for estimating PM2.5 concentrations from pollutant measurements. By combining historical data preprocessing, exploratory analysis, ensemble regression, model selection, and real-time API integration, it connects offline model development with live inference.

The implementation provides a foundation for further work in environmental data analytics, air-quality monitoring, and machine learning deployment. Improving validation methodology, missing-data handling, and temporal modeling would be important next steps toward a more reliable operational system.
