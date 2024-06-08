import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Load your data
data = pd.read_csv(r'C:\temp\IOT-temp\IOT-temp.csv')
# Inspect the first few rows to understand the data format
print(data.head())

# Convert timestamps to a numeric format
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%m-%d-%Y %H:%M')
data['timestamp'] = (data['timestamp'].astype('int64') // 10**9).astype(np.int64)  # Convert to seconds since epoch

# Extract features and target variable
X = data['timestamp'].values
y = data['temperature'].values

# Create a cubic spline interpolation of the data
cs = CubicSpline(X, y)

# Generate a finer set of points for a smoother curve
X_fine = np.linspace(X.min(), X.max(), 1000)
y_fine = cs(X_fine)

# Plot the original data points
plt.scatter(X, y, color='blue', label='Original data')

# Plot the interpolated curve
plt.plot(X_fine, y_fine, color='red', label='Cubic spline interpolation')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.legend()
plt.show()

