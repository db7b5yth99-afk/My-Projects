import yfinance as yf
import numpy as np
import pandas as pd
from numpy.ma.core import arange
from sktime.forecasting.trend import SplineTrendForecaster
from scipy.fft import rfft, rfftfreq, irfft
from matplotlib import pyplot as plt

data = yf.download("^GSPC", period="10y", interval="1d", progress=False)["Close"].dropna().squeeze()
ret = np.log(data / data.iloc[0])
forecaster = SplineTrendForecaster(n_knots=4, degree=3)
forecaster.fit(ret)
trend = forecaster.predict(ret.index)
det = ret - trend
# det.plot.line()
# plt.show()

# Frequency decompose
fft_vals = rfft(det.values)
magnitude = np.abs(fft_vals)
n_harmonics = 14                # I've tested this, the 14th component is important
top_idx = np.argsort(magnitude[1:])[-n_harmonics:] + 1

# Denoise and Reconstruct
n = len(det)
filtered = np.zeros_like(fft_vals, dtype=complex)
filtered[0] = fft_vals[0]
filtered[top_idx] = fft_vals[top_idx]
reconstructed = irfft(filtered, n=n)
# plt.plot(data.index,reconstructed)
# plt.title(f"Component count: {n_harmonics}")
# plt.show()

# Forecast
forecast_days = 15
t_fut = np.arange(n, n + forecast_days)
forecast = np.zeros(forecast_days)
freq = rfftfreq(len(det), d=1.0)
for k in top_idx:
    amp = np.abs(fft_vals[k]) / n
    phase = np.angle(fft_vals[k])
    forecast += amp * np.cos(2 * np.pi * freq[k] * t_fut + phase)
full_cycle = np.concatenate([reconstructed, forecast])

# === Bull/Bear Signal [-1, +1] ===
slope = np.gradient(full_cycle)
signal = np.tanh(slope / np.std(slope) * 4)
print(f"Latest Bull/Bear Signal: {signal[-1]:.3f}")