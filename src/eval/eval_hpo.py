from pathlib import Path
import glob
from statistics import median
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_ROOT = Path("C:/Dev/Bachelorarbeit/data/accounting/runs/hpo_stageA/ppo_log_state0_cpcv_20250928_110845")  # <- DEIN Run-Ordner

K_LAST = 5  # wie viele letzte Punkte mitteln / für Trend nutzen

def load_events(tb_dir):
    ev_files = sorted(glob.glob(str(tb_dir / "**/events.*"), recursive=True))
    if not ev_files:
        return None
    ea = EventAccumulator(ev_files[-1])
    ea.Reload()
    return ea

def last_k(ea, tag, k=K_LAST):
    if tag not in ea.Tags().get('scalars', []):
        return None, None
    xs = [x.step for x in ea.Scalars(tag)]
    ys = [x.value for x in ea.Scalars(tag)]
    if not ys:
        return None, None
    k = min(k, len(ys))
    return xs[-k:], ys[-k:]

def mean_last_k(ea, tag, k=K_LAST):
    xs, ys = last_k(ea, tag, k)
    return None if ys is None else float(np.mean(ys))

def slope_last_k(ea, tag, k=K_LAST):
    xs, ys = last_k(ea, tag, k)
    if ys is None or len(ys) < 2:
        return None
    # einfache lineare Regression (Steigung)
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)

rows = []
for trial in sorted([p for p in RUN_ROOT.iterdir() if p.is_dir()]):
    fold_metrics = []
    for tb in sorted(trial.glob("fold_*/tb")):
        ea = load_events(tb)
        if ea is None:
            continue

        # Reward-Level & Trend (Eval bevorzugt, sonst Rollout)
        score = mean_last_k(ea, "eval/ep_rew_mean")
        if score is None:
            score = mean_last_k(ea, "rollout/ep_rew_mean")
        trend = slope_last_k(ea, "eval/ep_rew_mean")
        if trend is None:
            trend = slope_last_k(ea, "rollout/ep_rew_mean")

        # Lern-Dynamik
        kl   = mean_last_k(ea, "train/approx_kl")
        clip = mean_last_k(ea, "train/clip_fraction")
        ent  = mean_last_k(ea, "train/entropy_loss")
        stdv = mean_last_k(ea, "train/std")
        ev   = mean_last_k(ea, "train/explained_variance")
        vlos = mean_last_k(ea, "train/value_loss")

        if score is not None:
            fold_metrics.append((score, trend, kl, clip, ent, stdv, ev, vlos))

    if fold_metrics:
        # über Folds aggregieren
        S, T, KL, CF, EN, ST, EV, VL = zip(*fold_metrics)
        iqr_S = np.subtract(*np.percentile(S, [75, 25]))
        pos_trend = sum(1 for t in T if (t is not None and t > 0))
        rows.append({
            "trial": trial.name,
            "score_med": float(median(S)),
            "score_iqr": float(iqr_S),
            "trend_med": float(median([t for t in T if t is not None])) if any(t is not None for t in T) else None,
            "kl_med":    float(median([x for x in KL if x is not None])) if any(KL) else None,
            "clip_med":  float(median([x for x in CF if x is not None])) if any(CF) else None,
            "ent_med":   float(median([x for x in EN if x is not None])) if any(EN) else None,
            "std_med":   float(median([x for x in ST if x is not None])) if any(ST) else None,
            "ev_med":    float(median([x for x in EV if x is not None])) if any(EV) else None,
            "vloss_med": float(median([x for x in VL if x is not None])) if any(VL) else None,
            "folds": len(fold_metrics),
            "pos_trend_folds": pos_trend
        })

# Ranking: erst Score, dann Trend, dann EV
rows.sort(key=lambda r: (
    r["score_med"],
    (r["trend_med"] if r["trend_med"] is not None else -1e9),
    (r["ev_med"]    if r["ev_med"]    is not None else -1e9)
), reverse=True)

print("trial | score_med | trend_med | ev_med | kl_med | clip_med | ent_med | std_med | vloss_med | score_iqr | pos_trend/ folds")
for r in rows:
    print(f'{r["trial"]:25s} {r["score_med"]:10.4f} '
          f'{str(r["trend_med"]):>10s} {str(r["ev_med"]):>8s} '
          f'{str(r["kl_med"]):>8s} {str(r["clip_med"]):>9s} '
          f'{str(r["ent_med"]):>8s} {str(r["std_med"]):>8s} '
          f'{str(r["vloss_med"]):>11s} {r["score_iqr"]:10.4f} '
          f'{r["pos_trend_folds"]}/{r["folds"]}')
