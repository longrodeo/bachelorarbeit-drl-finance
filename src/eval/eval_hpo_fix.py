# C:\Dev\Bachelorarbeit\src\eval\eval_hpo_fix.py
from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Kandidaten-Tags (wir nehmen den ersten, der existiert)
TAG_CANDS = {
    "score":     ["eval/ep_rew_mean", "rollout/ep_rew_mean", "eval/mean_reward", "metrics/score", "score"],
    "kl_med":    ["train/approx_kl"],
    "clip_med":  ["train/clip_fraction"],
    "ent_med":   ["train/entropy_loss", "train/entropy"],
    "std_med":   ["train/std", "train/approx_std"],
    "vloss_med": ["train/value_loss"],
    "ev_med":    ["train/explained_variance", "rollout/explained_variance"],
}

RUN_ROOT = Path(r"C:\Dev\Bachelorarbeit\data\accounting\runs\hpo_stageC\ppo_log_state0_cpcv_20250929_190212")

def read_last_scalars(tb_dir: Path) -> dict:
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out = {}
    for k, opts in TAG_CANDS.items():
        for tag in opts:
            if tag in available:
                vals = ea.Scalars(tag)
                if vals:
                    out[k] = vals[-1].value
                    break
    return out

def find_trial_fold(ev_file: Path):
    """
    Ermittelt Trial & Fold robust:
    - bevorzugt .../<trial>/fold_XX/<tb*>/events...
    - sonst nähster Ordner unterhalb RUN_ROOT als 'trial'
    """
    p = ev_file.parent
    trial = fold = None
    for _ in range(6):  # bis zu 6 Ebenen hoch
        if p.name.startswith("fold_"):
            fold = p.name
            trial = p.parent.name
            break
        p = p.parent
    if trial is None:
        # Fallback: Trial = erster Ordner unterhalb RUN_ROOT
        try:
            rel = ev_file.relative_to(RUN_ROOT)
            parts = rel.parts
            trial = parts[0] if parts else ev_file.parent.name
            fold = "nofold"
        except Exception:
            trial = ev_file.parent.name
            fold = "nofold"
    return trial, fold

def main():
    events = list(RUN_ROOT.rglob("events.out.tfevents.*"))
    if not events:
        print("Keine Event-Dateien gefunden unter:", RUN_ROOT)
        raise SystemExit(1)

    rows = []
    for ev in events:
        tb_dir = ev.parent
        metrics = read_last_scalars(tb_dir)
        if metrics:
            trial, fold = find_trial_fold(ev)
            metrics.update(trial=trial, fold=fold)
            rows.append(metrics)

    if not rows:
        print("Keine Scalars gefunden.")
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    agg = (df.groupby("trial")
             .agg({c: "median" for c in df.columns if c not in ["trial", "fold"]})
             .reset_index())

    out = RUN_ROOT / "_eval_tb_quick.csv"
    agg.to_csv(out, index=False)
    print("OK ->", out)

if __name__ == "__main__":
    main()
