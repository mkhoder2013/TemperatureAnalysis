import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

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

# Step 2: Apply Fourier Transform
fft_values = fft(df_resampled.values)
fft_freqs = np.fft.fftfreq(len(fft_values))

# Keep only the top N significant frequencies
N = 20  # Number of significant frequencies to keep
indices = np.argsort(np.abs(fft_values))[-N:]
fft_values_compressed = np.zeros_like(fft_values)
fft_values_compressed[indices] = fft_values[indices]

# Step 3: Inverse Fourier Transform to regenerate data
regenerated_data = ifft(fft_values_compressed).real

# Save the Fourier coefficients
with open('fft_coefficients.pkl', 'wb') as pkl:
    pickle.dump(fft_values_compressed, pkl)

# Step 4: Regenerate data using saved Fourier coefficients
# Load the Fourier coefficients
with open('fft_coefficients.pkl', 'rb') as pkl:
    loaded_fft_values_compressed = pickle.load(pkl)

# Inverse Fourier Transform to regenerate data
regenerated_data_from_file = ifft(loaded_fft_values_compressed).real

# Visualize the original and regenerated data
plt.figure(figsize=(12, 6))
plt.plot(df_resampled.index, df_resampled.values, label='Original Data (In Location)')
plt.plot(df_resampled.index, regenerated_data_from_file, label='Regenerated Data (Fourier Transform)', alpha=0.75)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Original and Regenerated Temperature Data for "In" Location')
plt.show()
