# gex_eod.py — run after 16:15 ET each trading day
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import norm
import math
import json

spx = yf.Ticker("^SPX")
spot = spx.history(period="1d")["Close"].iloc[-1]
expiries = [e for e in spx.options if (pd.Timestamp(e) - pd.Timestamp.today()).days <= 7]

def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))

rows = []
for exp in expiries:
    T = max((pd.Timestamp(exp) - pd.Timestamp.today()).days, 0.5) / 365
    chain = spx.option_chain(exp)
    for right, frame in [("CALL", chain.calls), ("PUT", chain.puts)]:
        f = frame[(frame.openInterest > 0) & (frame.impliedVolatility > 0)
                  & (abs(frame.strike - spot) < 300)]
        for _, row in f.iterrows():
            g = bs_gamma(spot, row.strike, T, 0.045, row.impliedVolatility)
            gex = g * row.openInterest * 100 * spot**2 * 0.01
            sign = 1 if right == "CALL" else -1
            rows.append({"strike": row.strike, "expiry": exp, "right": right,
                         "gex": sign * gex, "oi": row.openInterest})

df = pd.DataFrame(rows)
df.to_csv(f"archive/gex_{datetime.now():%Y%m%d}.csv", index=False)
out = {"spot": spot, "net_gex_bn": df["gex"].sum() / 1e9,
       "by_strike": df.groupby("strike")["gex"].sum().div(1e9).round(2).to_dict()}
json.dump(out, open("latest_gex.json", "w"))