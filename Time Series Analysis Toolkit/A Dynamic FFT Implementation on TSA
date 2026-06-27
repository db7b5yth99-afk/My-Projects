import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from sktime.forecasting.trend import SplineTrendForecaster
from scipy.fft import rfft, rfftfreq, irfft
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def wraped_fft(data):
    """FFT decompose"""
    fft_vals = rfft(data.values)
    magnitude = np.abs(fft_vals)
    top_idx = np.argsort(magnitude[1:])[-1] + 1
    freq = rfftfreq(len(data), d=1.0)[top_idx]
    return freq, magnitude, fft_vals, top_idx


def reconstructed_fft(fft_vals, top_idx, data):
    """FFT reconstruct"""
    n = len(data)
    filtered = np.zeros_like(fft_vals, dtype=complex)
    filtered[0] = fft_vals[0]
    filtered[top_idx] = fft_vals[top_idx]
    reconstructed = irfft(filtered, n=n)
    reconstructed_series = pd.Series(reconstructed, index=data.index)
    return reconstructed_series


def extract_one_dominant_sine(data, win_length=252, step=None):
    """
    Extract the single strongest sine wave (by peak FFT magnitude) from sliding windows.
    Used for looping in later function.
    """
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    if isinstance(data, pd.DataFrame):
        raise ValueError("extract_one_dominant_sine supports only univariate data "
                         "(Series or single-column DataFrame).")
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    if step is None:
        step = win_length // 4
    candidates = []
    for i in range(0, len(data) - win_length, step):
        window_data = data.iloc[i : i + win_length]
        freq, magnitude, fft_vals, top_idx = wraped_fft(window_data)
        candidates.append({
            'magnitude': magnitude[top_idx],
            'fft_vals': fft_vals,
            'top_idx': top_idx,
            'window_dates': window_data,
            'freq': freq
        })
    candidates_df = pd.DataFrame(candidates)
    selected = candidates_df.loc[candidates_df['magnitude'].idxmax()]
    dominant_sine = reconstructed_fft(selected['fft_vals'],
                                      selected['top_idx'],
                                      selected['window_dates'])
    return dominant_sine, [1/selected['freq'],selected['magnitude']]


def extend_series(sine_wave, extension_ratio=0.2):
    """Extend length of sin wave for adding fading effect."""
    fft_vals = rfft(sine_wave.values)
    magnitude = np.abs(fft_vals)
    top_idx = np.argmax(magnitude[1:]) + 1
    freq = rfftfreq(len(sine_wave))[top_idx]
    amplitude = magnitude[top_idx]
    phase = np.angle(fft_vals[top_idx])
    n = len(sine_wave)
    ext = int(n * extension_ratio)
    t = np.arange(-ext, n + ext)
    extended_values = amplitude * np.sin(2 * np.pi * freq * t + phase)
    new_index = pd.date_range(
        start=sine_wave.index[0] - pd.Timedelta(days=ext),
        periods=len(t),
        freq='D'
    )
    return pd.Series(extended_values, index=new_index)


def fade_envelope(length, extension_ratio=0.2, sigma=1, peak_start=None, peak_end=None):
    """Create a smooth fading envelope using normal CDF."""
    t = np.arange(length)
    ext = int(length * (extension_ratio / (1 + (extension_ratio * 2))))
    envelope = np.ones(length)
    envelope[:ext] = 0
    envelope[-ext:] = 0
    if peak_start is None:
        peak_start = ext
    if peak_end is None:
        peak_end = length - ext   # FIXED: was incorrectly "t - ext" (array)
    rise_region = t < peak_start
    if np.any(rise_region):
        envelope[rise_region] = norm.cdf(t[rise_region], loc=peak_start, scale=sigma)
    fall_region = t > peak_end
    if np.any(fall_region):
        envelope[fall_region] = 1 - norm.cdf(t[fall_region], loc=peak_end, scale=sigma)
    return envelope


def target(params, sine_list, original_data):
    """Define goal seeking objective for scipy.optimize.minimize()."""
    n_waves = len(sine_list)
    reconstructed = pd.Series(0.0, index=original_data.index)
    for i in range(n_waves):
        sigma = params[i * 3]
        peak_start = params[i * 3 + 1]
        peak_end = params[i * 3 + 2]
        sine = sine_list[i]
        env = fade_envelope(len(sine), sigma=sigma, peak_start=peak_start, peak_end=peak_end)
        faded = sine.values * env
        reconstructed = reconstructed.add(pd.Series(faded, index=sine.index), fill_value=0)
    error = np.sum((reconstructed.values - original_data.values) ** 2)
    return error


def optimize_joint_fade(sine_list, original_data, maxiter=200):
    """Optimize parameter of fading effect and apply it."""
    if isinstance(original_data, pd.DataFrame):
        original_data = original_data.squeeze()
    if not isinstance(original_data, pd.Series):
        original_data = pd.Series(original_data)
    n_waves = len(sine_list)
    x0 = []
    bounds = []
    for sine in sine_list:
        length = len(sine)
        x0.extend([length * 0.15, length * 0.3, length * 0.7])
        bounds.extend([
            (length * 0.03, length * 0.5),
            (0, length * 0.6),
            (length * 0.4, length * 0.95)
        ])
    result = minimize(
        target,
        x0,
        args=(sine_list, original_data),
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': maxiter}
    )
    optimized_params = result.x.reshape(n_waves, 3)
    faded_list = []
    reconstructed = pd.Series(0.0, index=original_data.index)
    for i, sine in enumerate(sine_list):
        sigma, peak_start, peak_end = optimized_params[i]
        env = fade_envelope(len(sine), sigma, peak_start, peak_end)
        faded = sine.values * env
        faded_series = pd.Series(faded, index=sine.index)
        faded_list.append(faded_series)
        reconstructed = reconstructed.add(faded_series, fill_value=0)
    return reconstructed, optimized_params


def extract_n_dominant_sine(data, win_length_list, step_list=None, use_fade=False):
    """Main function that combine sin extraction, reconstruction with an optional fading effect"""
    if step_list is None:
        step_list = [w // 4 for w in win_length_list]  # list of int (cleaner than ndarray)
    if len(win_length_list) != len(step_list):
        print('Error: Length mismatch between win_length_list and step_list')
        return None
    dominant_sine_list = []
    reconstructed = pd.Series(0.0, index=data.index)
    residual = data.copy()
    selected = []
    if isinstance(residual, pd.DataFrame):
        residual = residual.squeeze()
    for i in range(len(win_length_list)):
        dominant_sine, selected_temp = extract_one_dominant_sine(residual, win_length_list[i], step_list[i])
        residual = residual.subtract(dominant_sine, fill_value=0).fillna(0)
        reconstructed = reconstructed.add(dominant_sine, fill_value=0).fillna(0)
        selected.append(selected_temp)
        dominant_sine_list.append(dominant_sine)
    selected = pd.DataFrame(selected, columns=['Period (Days)', 'Magnitude'])
    if use_fade:
        reconstructed, optimized_params = optimize_joint_fade(dominant_sine_list, data, maxiter=1000)
        optimized_params = pd.DataFrame(optimized_params, columns=['sigma', 'peak_start', 'peak_end'])
    return reconstructed, selected, optimized_params


# =======================================================================================================================
# =======================================================================================================================
# =======================================================================================================================
# =======================================================================================================================

# Application
print("Downloading S&P 500 data (requires internet)...")
data = yf.download("^GSPC", period='10y', interval="1d", progress=False)["Close"].dropna()
ret = np.log(data / data.iloc[0])
print("Fitting SplineTrendForecaster (n_knots=4, degree=3)...")
forecaster = SplineTrendForecaster(n_knots=4, degree=3)
forecaster.fit(ret)
trend = forecaster.predict(ret.index)
detrended = ret - trend
print("Extracting dominant cycles (use_fade=True)...")
cycle, selected, optimized_params = extract_n_dominant_sine(
    detrended,
    win_length_list=[500, 500, 252, 252, 252, 252],
    use_fade=True
)
print("Cycle extraction complete. Length of cycle series:", len(cycle))
print("Cycle parameter:\n",selected)
print("Fading parameter:\nsigma, peak_start, peak_end\n",optimized_params)
# Plot
plt.figure(figsize=(12, 6))
plt.plot(detrended)
plt.title('Cycle Plot (Raw)')
plt.xlabel('Date')
plt.ylabel('Amplitude (detrended log-return scale)')
plt.show()
plt.plot(cycle)
plt.title('Cycle Plot (Extracted Dominant Sines + Optional Fade Optimization)')
plt.xlabel('Date')
plt.ylabel('Amplitude (detrended log-return scale)')
plt.show()
print("\nScript completed successfully.")
