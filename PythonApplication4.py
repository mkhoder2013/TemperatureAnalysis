import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv(r'C:\temp\IOT-temp\IOT-temp.csv')

# Inspect the first few rows to understand the data format
print(data.head())

# Convert timestamps to a numeric format
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d-%m-%Y %H:%M')
data['timestamp'] = (data['timestamp'].astype('int64') // 10**9).astype(np.int64)  # Convert to seconds since epoch

# Extract features and target variable
X = data['timestamp'].values.reshape(-1, 1)
y = data['temperature'].values

# Function to fit polynomial regression and calculate metrics
def evaluate_polynomial_regression(degree, X, y):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2, model

# Try polynomial degrees from 1 to 10 and evaluate
degrees = range(1, 11)
mse_values = []
r2_values = []
models = []

for degree in degrees:
    mse, r2, model = evaluate_polynomial_regression(degree, X, y)
    mse_values.append(mse)
    r2_values.append(r2)
    models.append(model)

# Find the best degree (the one with the lowest MSE or highest R2)
best_degree = degrees[mse_values.index(min(mse_values))]
print(f"Best degree: {best_degree}")

# Plot MSE and R2 values
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(degrees, mse_values, marker='o')
plt.xlabel('Degree of polynomial')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Polynomial Degree')

plt.subplot(1, 2, 2)
plt.plot(degrees, r2_values, marker='o')
plt.xlabel('Degree of polynomial')
plt.ylabel('R-squared')
plt.title('R2 vs. Polynomial Degree')

plt.tight_layout()
plt.show()

# Use the best degree to fit the final model
poly = PolynomialFeatures(best_degree)
X_poly = poly.fit_transform(X)
best_model = models[best_degree - 1]

# Generate predictions with the best model
y_pred = best_model.predict(X_poly)

# Plot the original data and the polynomial fit
plt.scatter(X, y, color='blue', label='Original data')

# Plot polynomial fit
# Sort the values for a smooth curve
sorted_indices = np.argsort(X.flatten())
plt.plot(X[sorted_indices], y_pred[sorted_indices], color='red', label='Polynomial fit')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Print the polynomial equation
def print_polynomial(coefficients, intercept):
    equation = f"{intercept}"
    for i, coef in enumerate(coefficients[1:], 1):
        equation += f" + ({coef}) * x^{i}"
    return equation

coefficients = best_model.coef_
intercept = best_model.intercept_
print("Best Polynomial equation:", print_polynomial(coefficients, intercept))
