import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the CSV file
file_path = r'data\OneDayTemp.csv'
df = pd.read_csv(file_path)

# Ensure the DateTime column is in datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H:%M')

# Filter the data for a specific location
location = 'Room Admin'
df_filtered = df[df['Location'] == location]

# Extract time and temperature data
time = (df_filtered['DateTime'] - df_filtered['DateTime'].min()).dt.total_seconds() / 3600  # Convert time to hours
temperature = df_filtered['Temp'].dropna().values  # Drop NaN values

# Define a polynomial function to fit the data
def poly_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Fit the polynomial to the data
popt, pcov = curve_fit(poly_func, time[:len(temperature)], temperature)

# Print the coefficients of the polynomial
a, b, c, d = popt
print(f"Polynomial coefficients: a={a}, b={b}, c={c}, d={d}")

# Generate the polynomial representation
fitted_temp = poly_func(time, *popt)

# Plot the original data and the fitted polynomial
plt.plot(time, temperature, 'bo', label='Original Data')
plt.plot(time, fitted_temp, 'r-', label='Fitted Polynomial')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Store the polynomial coefficients instead of raw data
polynomial_representation = popt
print(f"Stored polynomial representation: {polynomial_representation}")
