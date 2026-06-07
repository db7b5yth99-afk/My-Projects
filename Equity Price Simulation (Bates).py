import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Data
ticker = "MNTS"
time_horizon = "4y"
data = yf.download(ticker, period=time_horizon, interval="1d")['Close'].astype(float).dropna().squeeze()
returns = np.log(data / data.shift(1)).dropna()

# Parameter Calculation
moving_mu = returns.rolling(21).mean()
mu = moving_mu.iloc[-1]         # smoothen daily mu
moving_std = returns.rolling(63).std()
vt = moving_std.iloc[-1]        # smoothen daily std
theta = moving_std.dropna().mean()  # smoothen long term daily std
rho = np.corrcoef(moving_std.diff().iloc[-500:], returns.iloc[-500:])[0, 1]   # correlation between change in std and mean
abnr = 0.025                # hard-coded excess return

if abs(mu) > abs(returns[:-22].mean()) + abnr:
    print(f'Alert: Abnormal recent return (mu = {mu:.4f}), use long term ({time_horizon}-mu={returns.mean():.4f}) mu instead? (Y/N/set_0)')
    while True:
        uinput=input().lower()
        if uinput == 'y':
            mu = returns.mean()
            print(f'mu is set to {mu:.4f}')
            break
        elif uinput == 'set_0':
            mu = 0
            print(f'mu is set to {mu:.4f}')
            break
        elif uinput == 'n':
            print(f'mu remains {mu:.4f}')
            break
        else:
            print(f'Invalid input')

# Jump detection
jump_return = returns[(returns > (returns.mean() + theta * 3)) | (returns < (returns.mean() - theta * 3))]
lda = len(jump_return) / (len(returns))     # freq of daily jump return
mu_j = jump_return.mean()                   # jump mean
std_j = jump_return.std()                   # jump std

model = LinearRegression()
X = moving_std.iloc[-501:-2].to_numpy().reshape(-1, 1)
y = moving_std.diff().iloc[-500:-1].to_numpy()
model.fit(X, y)
kap = -model.coef_[0]                       # mean reversion speed
k = np.exp(mu_j + 0.5 * std_j**2) - 1
vov = np.std(y - model.predict(X))          # std of std

# Parameters Obtained (For reference)
k = float(k)
vt = float(vt)
theta = float(theta)
vov = float(vov)
kap = float(kap)
rho = float(rho)
mu = float(mu)
lda = float(lda)
mu_j = float(mu_j)
std_j = float(std_j)

# Simulation
total_trials = 20000
quarter_pred = np.zeros(total_trials)
forcesell = 0
fsp = 1                     # force selling price

for j in range(total_trials):
    trials = 63
    dt = 1.0
    S = data.iloc[-1]
    vol = vt

    for i in range(trials):
        # Updating Parameters
        dWt1 = np.random.normal(0, 1.0)
        dWt2 = rho * dWt1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1.0)
        dvol = kap * (theta - vol) * dt + vov * dWt2
        vol = max(0, vol + dvol)          # prevent negative vol
        dNt = np.random.poisson(lda * dt)
        Z = np.random.normal(0, 1)
        jt = np.exp(mu_j + Z * std_j) if dNt > 0 else 1.0

        # The Three Puzzles
        drift = (mu - lda * k) * dt
        diffusion = vol * dWt1
        jump_effect = (jt - 1.0) * dNt

        # Result
        S = S * (1 + drift + diffusion + jump_effect)
        S = max(S, fsp)                      # high risk force sell
        if S == fsp:
            forcesell += 1
            S = 0
            break

    quarter_pred[j] = S

# Results
print(f"Current price: ${data.iloc[-1]:.4f}")
print(f"Total force sell (close price <1$) trial out of {total_trials} attempts: {forcesell}({(forcesell/total_trials)*100:.2f}%)")
print(f"Mean price after 1 quarter:   ${np.mean(quarter_pred):.4f}")
print(f"Median price after 1 quarter: ${np.median(quarter_pred):.4f}")
print(f"5% quantile (downside):       ${np.percentile(quarter_pred, 5):.4f}")
print(f"95% quantile (upside):        ${np.percentile(quarter_pred, 95):.4f}")
print(f"Prob of loss (>0% down):      {(quarter_pred < data.iloc[-1]).mean()*100:.1f}%")
sns.boxplot(y=quarter_pred[(quarter_pred>np.percentile(quarter_pred, 5))&(quarter_pred<np.percentile(quarter_pred, 95))])
plt.title("Price after 1 Quarter")
plt.show()

# Return
'''
Current price: $14.9100
Total force sell trial out of 20000 attempts: 25
Mean price after 1 quarter:   $560.2396
Median price after 1 quarter: $245.0831
5% quantile (downside):       $36.7315
95% quantile (upside):        $1855.8697
Prob of loss (>0% down):      1.1%
'''
# Message
'''
Its not working
'''