import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sktime.forecasting.trend import SplineTrendForecaster
from scipy.signal import find_peaks

data = yf.download(tickers="^GSPC",start="1998-01-01", interval="1d")["Close"].dropna().squeeze()
data.plot.line()
plt.title('S&P500 From 1998 to Now')
plt.show()
ret = np.log(data/data.iloc[0])    # Return based on day 1
ret.plot.line()
plt.title('Return Plot (Versus Day 1)')
plt.show()

'''
Remove Trend
'''
# Finding Trend
forecaster = SplineTrendForecaster(n_knots=4, degree=3)     # Lower to n_knots to 4, prevent killing period info
forecaster.fit(ret)
trend = forecaster.predict(ret.index)
trend.plot.line()
plt.title('Trend Plot')
plt.show()
# Removing Trend
y_detrended = ret-trend
y_detrended_plot = y_detrended.rolling(20).mean().dropna()     # For better visualize
y_detrended_plot.plot.line()
plt.title('Detrended Plot (Smoothed)')
plt.show()

'''
Stabilize Variance
'''
# First Stabilize (Quarter Volatility)
rolling_std = y_detrended.rolling(63).std().dropna()
long_term_std = rolling_std.mean()
y_detrended_stabilized = y_detrended.iloc[63:]/rolling_std*long_term_std
y_detrended_stabilized_plot = y_detrended_stabilized.rolling(20).mean().dropna()     # For better visualize
y_detrended_stabilized_plot.plot.line()
plt.title('Initial Quarterly-Stabilized Plot (Smoothed)')
plt.show()
# Second Stabilize (Annual Volatility)
rolling_std = y_detrended_stabilized.rolling(126).std().dropna()
long_term_std = rolling_std.mean()
y_detrended_stabilized = y_detrended_stabilized.iloc[126:]/rolling_std*long_term_std
y_detrended_stabilized_plot = y_detrended_stabilized.rolling(20).mean().dropna()     # For better visualize
y_detrended_stabilized_plot.plot.line()
plt.title('Sequential Semiannually-Stabilized Plot (Smoothed)')
plt.show()

'''
Turning Point Analysis
'''
# Peak Detection
y = y_detrended_stabilized.squeeze().dropna()
peak_idx, _ = find_peaks(y.values, distance=60, prominence=0.08)
trough_idx, _ = find_peaks(-y.values, distance=60, prominence=0.08)
# Output to Dataframe
peak_dates = y.index[peak_idx]
trough_dates = y.index[trough_idx]
Peak = pd.DataFrame({
    'Close_Price': list(data.loc[peak_dates]),
    'Type': ['Peak (Start of Decline)'] * len(peak_idx)
}, index=peak_dates)
Trough = pd.DataFrame({
    'Close_Price': list(data.loc[trough_dates]),
    'Type': ['Trough (Start of Rise)'] * len(trough_dates)
}, index=trough_dates)
Turning_point = pd.concat([Peak, Trough]).sort_index()

'''
Result
'''
# Useful Info
print("============= Turning Points =============")
print("### Full DataSet ###")
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{:.2f}'.format)
print(Turning_point)
print("\nKey dates")
print(Turning_point.index.strftime('%Y-%m-%d').tolist())
# Cycle Plot
fig, ax = plt.subplots(figsize=(14, 7))
y_detrended_stabilized_plot.plot(ax=ax)
ax.scatter(peak_dates, y.loc[peak_dates], color='#E63946', marker='^', s=80, zorder=5)
ax.scatter(trough_dates, y.loc[trough_dates], color='#2A9D8F', marker='v', s=80, zorder=5)
plt.title('Sequential Semiannually-Stabilized Plot (Smoothed & Labeled)')
plt.show()
# Close Price Plot
fig, ax = plt.subplots(figsize=(14, 7))
data.plot(ax=ax)
ax.scatter(peak_dates, data.loc[peak_dates], color='#E63946', marker='^', s=80, zorder=5)
ax.scatter(trough_dates, data.loc[trough_dates], color='#2A9D8F', marker='v', s=80, zorder=5)
plt.title('S&P500 From 1998 to Now (Turning Points Labeled)')
plt.show()