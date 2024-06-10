Temperature Analysis using ARIMA Model and Additional Techniques 

## Overview

This repository contains two main components:
1. **Temperature Analysis using ARIMA Model and Additional Techniques**
2. **PostgreSQL and pgAdmin Setup**

### 1. Temperature Analysis using ARIMA Model and Additional Techniques

This project implements a time-series analysis and forecasting model using the ARIMA (Autoregressive Integrated Moving Average) method along with additional techniques such as polynomial regression and Fourier analysis. The goal is to analyze historical temperature data and forecast future temperature trends using multiple approaches for comparison.

#### Features

- **Data Preprocessing**: The provided temperature data is preprocessed to handle missing values, outliers, and ensure stationarity if necessary.
- **ARIMA Model Implementation**: The ARIMA model is implemented to capture the temporal dependencies in the temperature data and make predictions.
- **Polynomial Regression**: Polynomial regression is applied to fit a curve to the temperature data and make predictions based on polynomial functions.
- **Fourier Analysis**: Fourier analysis is used to decompose the temperature data into its frequency components and identify seasonal patterns.
- **Forecasting**: Future temperature values are forecasted using the trained models from ARIMA, polynomial regression, and Fourier analysis.
- **Visualization**: The results, including historical temperature trends and forecasted values from different methods, are visualized using plots.
#### Usage

1. **Clone the repository to your local machine**:

    ```bash
    git clone https://github.com/mkhoder2013/TenderedFramework.git
    cd TenderedFramework
2. **Execute the ARIMA model script** `Arima.py` to train the model and generate ARIMA forecasts:

    ```bash
    python Arima.py
    ```

3. **Run the polynomial regression script** `polynomial.py` to perform polynomial regression analysis and generate forecasts:

    ```bash
    python polynomial.py
    ```

4. **Execute the Fourier analysis script** `fourier_analysis.py`:

    ```bash
    python fourier_analysis.py
    ```
    ```
    
