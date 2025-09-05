import numpy as np
import pandas as pd
from env.trading_env import TradingEnv
from portfolio.broker import PortfolioLite  # falls dein Broker so importiert wird

# --- Dummy-Daten bauen (NUR für Smoke-Test!)
assets = ["A", "B", "C"]
dates = pd.bdate_range("2020-01-01", periods=6)  # t0..t5 -> 5 Schritte möglich

# px_by_date[t] = DataFrame(index=assets, columns=['adj_open','adj_close','bid_ask_spread_corwin_schultz'])
px_by_date = {}
price = {a: 100.0 + i*10 for i, a in enumerate(assets)}
for d in dates:
    df = pd.DataFrame(index=assets)
    df["adj_open"] = [price[a] for a in assets]
    df["adj_close"] = [price[a]* (1.0 + 0.001) for a in assets]  # +0.1% intraday
    df["bid_ask_spread_corwin_schultz"] = 0.0
    px_by_date[d] = df
    # nächster Tag leicht anders
    for a in assets:
        price[a] *= 1.001  # +0.1% pro Tag

# Einfacher Obs-Builder: z. B. letztes Close / 100
def obs_builder(date):
    close = px_by_date[date]["adj_close"].values
    return (close / 100.0).astype(np.float32)

# Broker aufsetzen
broker = PortfolioLite(assets=assets, initial_cash=1_000_000.0)

# Env
env = TradingEnv(
    broker=broker,
    dates=dates,
    px_by_date=px_by_date,
    obs_builder=obs_builder,
    initial_cash=1_000_000.0,
    obs_transform=None,
    reward_mode="logret",
)

obs, info = env.reset()
print("reset obs:", obs.shape)

done = False
steps = 0
while not done:
    action = np.random.normal(size=len(assets)).astype(np.float32)
    obs, r, term, trunc, info = env.step(action)
    steps += 1
    done = term or trunc
    print(f"step {steps}: r={r:.6f}, term={term}, cash={info.get('cash'):.2f}, value={info.get('value'):.2f}")

print("OK.")
