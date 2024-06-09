import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
file_path = r'data\IOT-temp.csv'  # Update with your file path
df = pd.read_csv(file_path)

# Rename columns for easier access
df.columns = ['id', 'room_id', 'noted_date', 'temp', 'out/in']

# Convert noted_date to datetime
df['noted_date'] = pd.to_datetime(df['noted_date'], format='%d-%m-%Y %H:%M')
df.set_index('noted_date', inplace=True)

# Filter the data for "In" location only
df_in = df[df['out/in'] == 'In']

# Ensure 'temp' is numeric
df_in['temp'] = pd.to_numeric(df_in['temp'], errors='coerce')

# Resample data by minute and take the mean temperature
df_resampled = df_in['temp'].resample('T').mean()
df_resampled.dropna(inplace=True)

# Step 2: Predictive Modeling using ARIMA
model = ARIMA(df_resampled, order=(5, 1, 0))
model_fit = model.fit()

# Save the model
with open('temp_arima_model.pkl', 'wb') as pkl:
    pickle.dump(model_fit, pkl)

# Visualize the original data and the fitted values
plt.figure(figsize=(12, 6))
plt.plot(df_resampled.index, df_resampled.values, label='Original Data (In Location)')
plt.plot(df_resampled.index, model_fit.fittedvalues, label='Fitted Values (ARIMA)', alpha=0.75)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Original Temperature Data and Fitted Values for "In" Location')
plt.show()
