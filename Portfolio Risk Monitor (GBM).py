import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import exp

ticker = ['VELO', 'CODA', 'OPEN', 'NMTC', 'MDAI', 'STXS']
shares = np.array([420, 110, 150, 150, 250, 200])
data = yf.download(ticker, period='4y', interval='1mo')['Close']
Portfolio = data@shares
m_return = np.log(Portfolio.squeeze()/Portfolio.squeeze().shift(1)).dropna()
m_mu = m_return.mean()
m_std = m_return.std()
last_price = Portfolio.iloc[-1].item()
trials = 2000
Simulated_Price = np.zeros(trials)
for i in range(trials):
    Simulated_Price[i] = last_price * exp((m_mu-m_std**2/2) + m_std*np.random.normal(0,1))

print(f"Current position: ${Portfolio.iloc[-1]:.4f}")
print(f"Mean price after 1 month:   ${np.mean(Simulated_Price):.4f}")
print(f"Median price after 1 month: ${np.median(Simulated_Price):.4f}")
print(f"5% quantile (downside):       ${np.percentile(Simulated_Price, 5):.4f}")
print(f"95% quantile (upside):        ${np.percentile(Simulated_Price, 95):.4f}")
print(f"Prob of loss (>0% down):      {(Simulated_Price < Portfolio.iloc[-1]).mean()*100:.1f}%")
sns.boxplot(y=Simulated_Price)
plt.title("Price after 1 Month")
plt.show()