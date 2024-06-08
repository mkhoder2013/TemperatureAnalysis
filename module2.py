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

# Fit a polynomial regression model (use a lower degree for smoothness)
degree = 10  # Adjust the degree as needed
coefficients = np.polyfit(X, y, degree)
polynomial = np.poly1d(coefficients)

# Print the polynomial equation
print("Polynomial equation coefficients (in descending order of powers):")
for i, coef in enumerate(coefficients):
    print(f"a_{degree - i} = {coef}")

# Generate a finer set of points for a smoother curve
X_fine = np.linspace(X.min(), X.max(), 1000)
y_fine = polynomial(X_fine)

# Plot the original data points
plt.scatter(X, y, color='blue', label='Original data')

# Plot the polynomial fit
plt.plot(X_fine, y_fine, color='red', label='Polynomial fit')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.legend()
plt.show()


