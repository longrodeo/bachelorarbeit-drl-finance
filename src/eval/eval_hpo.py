# C:\Dev\Bachelorarbeit\src\eval\eval_hpo.py
from pathlib import Path
from statistics import median
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

RUN_ROOT = Path(r"C:\Dev\Bachelorarbeit\data\accounting\runs\hpo_stageC\ppo_log_state0_cpcv_20250929_190212")
K_LAST = 5  # wie viele letzte Punkte mitteln / für Trend

def load_events_from_dir(dir_with_events: Path):
    ev_files = sorted(glob.glob(str(dir_with_events / "events.out.tfevents.*")))
    if not ev_files:
        return None
    ea = EventAccumulator(str(dir_with_events), size_guidance={"scalars": 0})
    ea.Reload()
    return ea

def last_k(ea, tag, k=K_LAST):
    tags = ea.Tags().get('scalars', [])
    if tag not in tags:
        return None, None
    vals = ea.Scalars(tag)
    if not vals:
        return None, None
    xs = [v.step for v in vals]
    ys = [v.value for v in vals]
    k = min(k, len(ys))
    return xs[-k:], ys[-k:]

def mean_last_k(ea, tag, k=K_LAST):
    xs, ys = last_k(ea, tag, k)
    return None if ys is None else float(np.mean(ys))

def slope_last_k(ea, tag, k=K_LAST):
    xs, ys = last_k(ea, tag, k)
    if ys is None or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)

def find_fold_tb_dirs(trial_dir: Path):
    """
    Finde für ein Trial alle TB-Verzeichnisse robust:
    - akzeptiert fold_*/tb, fold_*/tb_*, fold_*/sb3_log*, usw.
    """
    tb_dirs = []
    for fold_dir in sorted(trial_dir.glob("fold_*")):
        # nimm irgendeinen Unterordner, der Events enthält
        candidates = list(fold_dir.glob("**/events.out.tfevents.*"))
        if candidates:
            tb_dirs.append(candidates[-1].parent)  # Verzeichnis, das die Event-Datei enthält
    return tb_dirs

rows = []
# Trials = alle direkten Unterordner im RUN_ROOT, die keine TB-Meta-Dateien sind
for trial in sorted([p for p in RUN_ROOT.iterdir() if p.is_dir() and p.name not in ("tb",)]):
    tb_dirs = find_fold_tb_dirs(trial)
    fold_metrics = []
    for tb_dir in tb_dirs:
        ea = load_events_from_dir(tb_dir)
        if ea is None:
            continue

        # Score (Eval bevorzugt, sonst Rollout)
        score = mean_last_k(ea, "eval/ep_rew_mean")
        if score is None:
            score = mean_last_k(ea, "rollout/ep_rew_mean")
        trend = slope_last_k(ea, "eval/ep_rew_mean")
        if trend is None:
            trend = slope_last_k(ea, "rollout/ep_rew_mean")

        # Lern-Dynamik
        kl   = mean_last_k(ea, "train/approx_kl")
        clip = mean_last_k(ea, "train/clip_fraction")
        ent  = mean_last_k(ea, "train/entropy_loss") or mean_last_k(ea, "train/entropy")
        stdv = mean_last_k(ea, "train/std") or mean_last_k(ea, "train/approx_std")
        ev   = mean_last_k(ea, "train/explained_variance") or mean_last_k(ea, "rollout/explained_variance")
        vlos = mean_last_k(ea, "train/value_loss")

        if score is not None:
            fold_metrics.append((score, trend, kl, clip, ent, stdv, ev, vlos))

    if fold_metrics:
        S, T, KL, CF, EN, ST, EV, VL = zip(*fold_metrics)
        iqr_S = np.subtract(*np.percentile(S, [75, 25]))
        pos_trend = sum(1 for t in T if (t is not None and t > 0))
        rows.append({
            "trial": trial.name,
            "score_med": float(median(S)),
            "score_iqr": float(iqr_S),
            "trend_med": float(median([t for t in T if t is not None])) if any(t is not None for t in T) else None,
            "kl_med":    float(median([x for x in KL if x is not None])) if any(x is not None for x in KL) else None,
            "clip_med":  float(median([x for x in CF if x is not None])) if any(x is not None for x in CF) else None,
            "ent_med":   float(median([x for x in EN if x is not None])) if any(x is not None for x in EN) else None,
            "std_med":   float(median([x for x in ST if x is not None])) if any(x is not None for x in ST) else None,
            "ev_med":    float(median([x for x in EV if x is not None])) if any(x is not None for x in EV) else None,
            "vloss_med": float(median([x for x in VL if x is not None])) if any(x is not None for x in VL) else None,
            "folds": len(fold_metrics),
            "pos_trend_folds": pos_trend
        })

# Ranking: nach score_med, dann trend_med, dann ev_med
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
