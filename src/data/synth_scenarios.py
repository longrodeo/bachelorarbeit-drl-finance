# synth_scenarios.py
# - Deterministic seed + logging (repo helpers)
# - Parquet IO (repo parquet_io)
# - Scenario generation via bootstrapped whitened innovations + covariance rescaling
# - Sanity checks: whitening Cov(Z)~I and recoloring Cov(U_sim)~Sigma_tgt
# - Beta via OLS regression: log(volume) = a + beta*|return| + eps
#   and simulation uses volume_multiplier = exp(beta*|return|)

import numpy as np
import pandas as pd
from pathlib import Path


from src.utils.helpers import set_seed, get_logger
from src.utils.parquet_io import load_parquet, save_parquet


# -----------------------
# CONFIG
# -----------------------
INPUT_PARQUET = Path("data/interim/panel.parquet")   # <-- adjust
OUTPUT_DIR = Path("data/synth")
SEED = 42

ASSETS = ["SPY", "IEUR", "EWJ", "IEMG", "EWC", "IAU", "BTC-USD", "ETH-USD"]
EQUITIES = ["SPY", "IEUR", "EWJ", "IEMG", "EWC"]
GOLD = ["IAU"]
CRYPTO = ["BTC-USD", "ETH-USD"]

N_DAYS = 252
BLOCK_LEN = 10
N_CAND = 200

SANITY_RECOLOR_SAMPLE = 5000  # how many historical days to sample for recoloring check

# -----------------------
# REPRO + LOGGING
# -----------------------
set_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger = get_logger("synth", to_file=str(OUTPUT_DIR / "synth_scenarios.log"))
logger.info(f"seed={SEED} input={INPUT_PARQUET} out_dir={OUTPUT_DIR}")
logger.info(f"assets={ASSETS} n_days={N_DAYS} block_len={BLOCK_LEN} n_cand={N_CAND}")

# -----------------------
# LOAD + ALIGN
# -----------------------
df = load_parquet(INPUT_PARQUET)

if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["date", "asset"]:
    if "date" in df.columns and "asset" in df.columns:
        df = df.set_index(["date", "asset"])
    else:
        if isinstance(df.index, pd.MultiIndex) and len(df.index.levels) == 2:
            df.index = df.index.set_names(["date", "asset"])
        else:
            raise ValueError("Expected MultiIndex (date, asset) or columns date+asset.")

df = df.sort_index()
df = df.loc[(slice(None), ASSETS), :]

adj_close = df["adj_close"].unstack("asset")
adj_open  = df["adj_open"].unstack("asset")
high      = df["high"].unstack("asset")
low       = df["low"].unstack("asset")
close     = df["close"].unstack("asset")
volume    = df["volume"].unstack("asset")

first_valid = []
for a in ASSETS:
    ok = adj_close[a].notna() & adj_open[a].notna()
    if not ok.any():
        raise ValueError(f"No valid adj_open/adj_close for asset {a}.")
    first_valid.append(adj_close.index[ok].min())

common_start = max(first_valid)

adj_close = adj_close.loc[adj_close.index >= common_start, ASSETS].dropna()
adj_open  = adj_open.loc[adj_close.index, ASSETS].dropna()

high   = high.loc[adj_close.index, ASSETS].ffill()
low    = low.loc[adj_close.index, ASSETS].ffill()
close  = close.loc[adj_close.index, ASSETS].ffill()
volume = volume.loc[adj_close.index, ASSETS].ffill()

logger.info(f"common_start={common_start} history_days={len(adj_close)}")

# -----------------------
# HISTORICAL LIBRARIES
# -----------------------
r = np.log(adj_close / adj_close.shift(1)).dropna()
dates_r = r.index

R = r.to_numpy()  # (T x d)

# overnight gaps: log(adj_open_t / adj_close_{t-1})
g = np.log(adj_open.loc[dates_r] / adj_close.shift(1).loc[dates_r]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
G = g.to_numpy()  # (T x d)

# intraday range factor q = (high-low)/close
q = ((high.loc[dates_r] - low.loc[dates_r]) / close.loc[dates_r]).replace([np.inf, -np.inf], np.nan)
q = q.clip(lower=0.0).ffill().fillna(0.0)
Q = q.to_numpy()  # (T x d)

# log-volume
eps = 1e-8
lv = np.log(volume.loc[dates_r].clip(lower=eps))
LV = lv.to_numpy()  # (T x d)

# -----------------------
# WHITEN RETURNS
# -----------------------
mu_hist = R.mean(axis=0)
U = R - mu_hist

Sigma = np.cov(U, rowvar=False)
Sigma = 0.5 * (Sigma + Sigma.T)

L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(Sigma.shape[0]))
Z = np.linalg.solve(L, U.T).T  # (T x d) approx Cov(Z)=I

std_hist = np.sqrt(np.diag(Sigma))
C = Sigma / np.outer(std_hist, std_hist)
np.fill_diagonal(C, 1.0)

# ---- Sanity check 1: Cov(Z) ~ I ----
CovZ = np.cov(Z, rowvar=False)
diag_mean = float(np.mean(np.diag(CovZ)))
offdiag = CovZ - np.diag(np.diag(CovZ))
offdiag_maxabs = float(np.max(np.abs(offdiag)))
logger.info(f"sanity_whiten: mean(diag(CovZ))={diag_mean:.4f} max|offdiag(CovZ)|={offdiag_maxabs:.4f}")

# -----------------------
# BETA VIA REGRESSION: log(volume) = a + beta*|return| + eps
# -----------------------
absR = np.abs(R)
beta_vec = np.zeros(len(ASSETS), dtype=float)
BETA_CAP = 15.0  # cap to avoid unrealistic volume explosions

for j, a in enumerate(ASSETS):
    x = absR[:, j]
    y = LV[:, j]

    # simple outlier-robustness: clip x at 99% quantile
    q99 = np.quantile(x, 0.99)
    x = np.clip(x, 0.0, q99)

    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)

    if denom <= 1e-12:
        beta = 0.0
    else:
        beta = float(np.sum((x - x_mean) * (y - y_mean)) / denom)

    beta_vec[j] = float(np.clip(beta, 0.0, BETA_CAP))  # keep it non-negative

logger.info("beta_per_asset (OLS on history): " + ", ".join([f"{a}={beta_vec[i]:.2f}" for i, a in enumerate(ASSETS)]))
logger.info(f"beta_summary: mean={float(beta_vec.mean()):.2f} median={float(np.median(beta_vec)):.2f} max={float(beta_vec.max()):.2f}")
logger.info(f"beta_capped_count={int(np.sum(beta_vec >= BETA_CAP - 1e-12))}")


# -----------------------
# START LEVEL + SYN DATES
# -----------------------
P0 = adj_close.iloc[-1].to_numpy(dtype=float)

last_dt = adj_close.index[-1]
tz = getattr(last_dt, "tz", None)
start_dt = pd.Timestamp(last_dt) + pd.Timedelta(days=1)
syn_dates = pd.bdate_range(start=start_dt, periods=N_DAYS, tz=tz)

asset_to_idx = {a: i for i, a in enumerate(ASSETS)}
equity_idx = [asset_to_idx[a] for a in EQUITIES]
spy_i = asset_to_idx["SPY"]
spy_vol_base = float(R[:, spy_i].std(ddof=1) * np.sqrt(252))

# -----------------------
# SCENARIOS
# -----------------------
scenarios = {
    "bear_1y": {
        "k": 2.0, "alpha": 0.50,
        "mu_pa_equity_first": -0.25, "mu_pa_equity_last": -0.05,
        "mu_pa_gold": +0.05,
        "mu_pa_crypto_first": -0.70, "mu_pa_crypto_last": -0.30,
        "vol_scale": 1.6, "q_scale": 1.25,

        "target": {"ret_spy": -0.30,"ret_tol": 0.05, "vol_spy": spy_vol_base * 2.0, "mdd_spy": 0.40, "ret_crypto": -0.60,
                   "ret_crypto_tol": 0.05},
    },
    "side_lowvol_1y": {
        "k": 0.70, "alpha": 0.00,
        "mu_pa_equity_first": 0.00, "mu_pa_equity_last": 0.00,
        "mu_pa_gold": +0.02,
        "mu_pa_crypto_first": 0.00, "mu_pa_crypto_last": 0.00,
        "vol_scale": 0.95, "q_scale": 0.90,
        "target": {"ret_spy": 0.00, "ret_tol": 0.05, "vol_spy": spy_vol_base * 0.90, "mdd_spy": 0.10},
    },
    "side_highvol_1y": {
        "k": 1.80, "alpha": 0.25,
        "mu_pa_equity_first": 0.00, "mu_pa_equity_last": 0.00,
        "mu_pa_gold": 0.00,
        "mu_pa_crypto_first": 0.00, "mu_pa_crypto_last": 0.00,
        "vol_scale": 1.4, "q_scale": 1.20,
        "target": {"ret_spy": 0.00, "ret_tol": 0.05, "vol_spy": spy_vol_base * 1.80, "mdd_spy": 0.25},
    },
}

# -----------------------
# GENERATE + SELECT
# -----------------------
T = Z.shape[0]
max_start = T - BLOCK_LEN
if max_start <= 0:
    raise ValueError("Not enough history for block bootstrap. Reduce BLOCK_LEN or use more history.")

all_outputs = {}

for scen_name, spec in scenarios.items():
    k = float(spec["k"])
    alpha = float(spec["alpha"])

    # build stress correlation
    C_stress = (1.0 - alpha) * C + alpha * np.ones_like(C)
    np.fill_diagonal(C_stress, 1.0)
    C_stress = np.clip(C_stress, -0.95, 0.95)
    np.fill_diagonal(C_stress, 1.0)

    # target covariance
    sigma_tgt = k * std_hist
    Sigma_tgt = np.diag(sigma_tgt) @ C_stress @ np.diag(sigma_tgt)
    Sigma_tgt = 0.5 * (Sigma_tgt + Sigma_tgt.T)
    L_tgt = np.linalg.cholesky(Sigma_tgt + 1e-12 * np.eye(len(ASSETS)))

    # ---- Sanity check 2: recoloring Cov(Z @ L_tgt.T) ~ Sigma_tgt ----
    n_s = min(SANITY_RECOLOR_SAMPLE, T)
    idx_s = np.random.randint(0, T, size=n_s)
    U_sim = Z[idx_s, :] @ L_tgt.T
    CovU = np.cov(U_sim, rowvar=False)
    rel_err = float(np.linalg.norm(CovU - Sigma_tgt, ord="fro") / (np.linalg.norm(Sigma_tgt, ord="fro") + 1e-12))
    logger.info(f"sanity_recolor[{scen_name}]: rel_fro_err={rel_err:.4f}")

    # piecewise drift MU
    n1 = int(round(N_DAYS * 0.80))
    n2 = N_DAYS - n1

    mu_pa_first = np.zeros(len(ASSETS))
    mu_pa_last  = np.zeros(len(ASSETS))

    for a in EQUITIES:
        mu_pa_first[asset_to_idx[a]] = spec["mu_pa_equity_first"]
        mu_pa_last[asset_to_idx[a]]  = spec["mu_pa_equity_last"]
    for a in GOLD:
        mu_pa_first[asset_to_idx[a]] = spec["mu_pa_gold"]
        mu_pa_last[asset_to_idx[a]]  = spec["mu_pa_gold"]
    for a in CRYPTO:
        mu_pa_first[asset_to_idx[a]] = spec["mu_pa_crypto_first"]
        mu_pa_last[asset_to_idx[a]]  = spec["mu_pa_crypto_last"]

    MU = np.vstack([
        np.tile(mu_pa_first / 252.0, (n1, 1)),
        np.tile(mu_pa_last  / 252.0, (n2, 1)),
    ])

    best_score = np.inf
    best_pack = None
    best_metrics = None

    for _ in range(N_CAND):
        # block bootstrap Z
        idx = []
        while len(idx) < N_DAYS:
            s = np.random.randint(0, max_start + 1)
            idx.extend(range(s, s + BLOCK_LEN))
        idx = np.array(idx[:N_DAYS])

        Z_star = Z[idx, :]
        R_star = MU + Z_star @ L_tgt.T

        # mean pairwise equity correlation (off-diagonal) from synthetic returns
        corr = np.corrcoef(R_star[:, equity_idx].T)
        m = corr.shape[0]
        equity_corr_mean = float((corr.sum() - m) / (m * (m - 1)))  # average off-diagonal

        # adj_close
        logP = np.log(P0)[None, :] + np.cumsum(R_star, axis=0)
        adj_close_star = np.exp(logP)

        # gaps (multivariate blocks)
        idx_g = []
        while len(idx_g) < N_DAYS:
            s = np.random.randint(0, max_start + 1)
            idx_g.extend(range(s, s + BLOCK_LEN))
        idx_g = np.array(idx_g[:N_DAYS])
        G_star = G[idx_g, :]

        gap_scale = float(np.clip(np.sqrt(k), 0.7, 1.7))
        G_star = gap_scale * G_star

        adj_open_star = np.empty_like(adj_close_star)
        prev_close = P0.copy()
        for t in range(N_DAYS):
            adj_open_star[t, :] = prev_close * np.exp(G_star[t, :])
            prev_close = adj_close_star[t, :]

        # range (per-asset blocks)
        Q_star = np.empty((N_DAYS, len(ASSETS)))
        for j in range(len(ASSETS)):
            idx_q = []
            while len(idx_q) < N_DAYS:
                s = np.random.randint(0, max_start + 1)
                idx_q.extend(range(s, s + BLOCK_LEN))
            idx_q = np.array(idx_q[:N_DAYS])
            Q_star[:, j] = Q[idx_q, j]
        Q_star = np.clip(Q_star, 0.0, 5.0) * float(spec["q_scale"])

        mid = 0.5 * (adj_open_star + adj_close_star)
        amp = mid * (Q_star / 2.0)
        high_cand = mid + amp
        low_cand  = np.maximum(mid - amp, 1e-12)

        high_star = np.maximum.reduce([high_cand, adj_open_star, adj_close_star])
        low_star  = np.minimum.reduce([low_cand, adj_open_star, adj_close_star])

        # volume: bootstrap log-volume + stress scaling + regression-based multiplier exp(beta*|return|)
        LV_star = np.empty((N_DAYS, len(ASSETS)))
        for j in range(len(ASSETS)):
            idx_v = []
            while len(idx_v) < N_DAYS:
                s = np.random.randint(0, max_start + 1)
                idx_v.extend(range(s, s + BLOCK_LEN))
            idx_v = np.array(idx_v[:N_DAYS])
            LV_star[:, j] = LV[idx_v, j]

        vol_mult = np.exp(np.abs(R_star) * beta_vec[None, :])  # (N_DAYS x d)
        volume_star = np.exp(LV_star) * float(spec["vol_scale"]) * vol_mult
        volume_star = np.clip(volume_star, 0.0, None)

        # metrics for selection (SPY)
        spy_prices = adj_close_star[:, spy_i]
        spy_ret = float(np.exp(R_star[:, spy_i].sum()) - 1.0)
        spy_vol = float(R_star[:, spy_i].std(ddof=1) * np.sqrt(252))
        peak = np.maximum.accumulate(spy_prices)
        mdd = float(np.max(1.0 - spy_prices / peak))
        spy_daily_std = float(R_star[:, spy_i].std(ddof=1))
        spy_end_ratio = float(spy_prices[-1] / spy_prices[0] - 1.0)

        # metrics for crypto bear case
        btc_i = asset_to_idx["BTC-USD"]
        eth_i = asset_to_idx["ETH-USD"]

        btc_ret = float(np.exp(R_star[:, btc_i].sum()) - 1.0)
        eth_ret = float(np.exp(R_star[:, eth_i].sum()) - 1.0)
        crypto_ret = 0.5 * (btc_ret + eth_ret)

        tgt = spec["target"]
        score = (
            ((spy_ret - tgt["ret_spy"]) / max(1e-6, abs(tgt["ret_spy"]) + tgt["ret_tol"])) ** 2
            + ((spy_vol - tgt["vol_spy"]) / max(1e-6, tgt["vol_spy"])) ** 2
            + ((mdd - tgt["mdd_spy"]) / max(1e-6, tgt["mdd_spy"])) ** 2
        )
        if "ret_crypto" in tgt:
            score += ((crypto_ret - tgt["ret_crypto"]) / max(1e-6, abs(tgt["ret_crypto"]) + tgt["ret_crypto_tol"])) ** 2

        if score < best_score:
            best_score = score
            best_pack = {
                "open": adj_open_star.copy(),
                "high": high_star,
                "low": low_star,
                "close": adj_close_star.copy(),
                "adj_open": adj_open_star,
                "adj_close": adj_close_star,
                "volume": volume_star,
            }
            best_metrics = (spy_ret, spy_vol, mdd, equity_corr_mean, spy_daily_std, spy_end_ratio)
    
    # log chosen path stats

    spy_ret, spy_vol, mdd, eq_corr, spy_dstd, spy_end = best_metrics
    logger.info(
        f"{scen_name}: best_score={best_score:.6f} "
        f"SPY_ret={spy_ret:.3f} SPY_end={spy_end:.3f} "
        f"SPY_vol={spy_vol:.3f} SPY_dstd={spy_dstd:.4f} "
        f"SPY_mdd={mdd:.3f} EQ_corr_mean={eq_corr:.3f}"
    )

    # build long output
    out = []
    for j, a in enumerate(ASSETS):
        tmp = pd.DataFrame(
            {
                "open": best_pack["open"][:, j],
                "high": best_pack["high"][:, j],
                "low": best_pack["low"][:, j],
                "close": best_pack["close"][:, j],
                "adj_open": best_pack["adj_open"][:, j],
                "adj_close": best_pack["adj_close"][:, j],
                "volume": best_pack["volume"][:, j],
                "dividends": 0.0,
                "stock_splits": 0.0,
            },
            index=syn_dates,
        )
        tmp["asset"] = a
        out.append(tmp)

    out_df = pd.concat(out, axis=0)
    out_df.index.name = "date"
    out_df = out_df.reset_index().set_index(["date", "asset"]).sort_index()

    all_outputs[scen_name] = out_df
    logger.info(f"{scen_name}: rows={len(out_df)}")

# -----------------------
# SAVE
# -----------------------
for scen_name, out_df in all_outputs.items():
    out_path = OUTPUT_DIR / f"{scen_name}.parquet"
    save_parquet(out_df, out_path)
    logger.info(f"saved {scen_name} -> {out_path}")

logger.info("done.")
print(f"Saved synthetic scenarios to: {OUTPUT_DIR}")
