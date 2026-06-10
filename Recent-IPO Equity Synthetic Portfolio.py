
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV,Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Prep data
tickers = [
    # Index and Risk Free Asset
    "IWC", "SHY", "IEF", "TLT",
    # Related ETFs
    "ITA", "XAR", "PPA", "GCAD", "XLI", "VIS", "IWM", "SMH", "PICK", "REMX"
]

all_prices = yf.download(tickers + ["ELMT"], start='2026-04-24', interval='1d')['Close']
cand = all_prices[tickers]
rep  = all_prices["ELMT"]
cand_ret = np.log(cand / cand.shift(1)).dropna()
rep_ret  = np.log(rep  / rep.shift(1)).dropna()

split = int(len(cand_ret) * 0.8)
X_train, X_test = cand_ret.iloc[:split], cand_ret.iloc[split:]
y_train, y_test = rep_ret.iloc[:split], rep_ret.iloc[split:]

# Model Fitting
'''
Note:
With only around 30 trading day for this ticker, I'm kinda loss here. 
Time series cross validation make no sense and I cannot think of other 
better regression pick. Maybe this is what I can do so far. I will be
back later.
'''
alphas = np.logspace(-5, 2, 100)
pipe_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=alphas))
])
pipe_ridge.fit(X_train, y_train)
y_pred_ridge = pipe_ridge.predict(X_test)
ridge = pipe_ridge.named_steps['ridge']
betas = pd.Series(ridge.coef_, index=X_train.columns)
betas = betas.sort_values(key=abs, ascending=False)
print("== Ridge Regression ==\nMetrics")
print(f"Best alpha : {pipe_ridge.named_steps['ridge'].alpha_:.6f}")
print(f"MAE        : {mean_absolute_error(y_test, y_pred_ridge):.6f}")
print(f"MSE        : {mean_squared_error(y_test, y_pred_ridge):.6f}")
print(f"R²         : {r2_score(y_test, y_pred_ridge):.4f}")
print("\nBeta (standardized)")
print(betas.round(6))
print(f"Intercept: {ridge.intercept_:.8f}")

'''
Result 6/10/2026
== Ridge Regression ==
Metrics
Best alpha : 8.697490
MAE        : 0.048319
MSE        : 0.003925
R²         : 0.5066

Beta (standardized)
Ticker
PICK    0.024970
ITA    -0.013536
REMX    0.011927
SHY    -0.010317
SMH     0.009555
XAR    -0.005729
TLT     0.005669
GCAD   -0.004761
XLI     0.003994
IWM    -0.003543
PPA    -0.002988
IWC     0.001588
IEF     0.000964
VIS     0.000148
dtype: float64
Intercept: 0.00434363
'''