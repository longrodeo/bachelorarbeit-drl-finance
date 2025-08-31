# src/accounting/analysis.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from utils.parquet_io import load_parquet

def plot_nav_from_snapshots(path: Path, save: bool = False):
    """
    Liest das portfolio_snapshots.parquet und plottet den Portfolio-NAV über die Runden.
    """
    path = Path(path)
    snap_path = path / "portfolio_snapshots.parquet"
    if not snap_path.exists():
        raise FileNotFoundError(f"{snap_path} nicht gefunden. Erst AccountingRecorder.log_round() laufen lassen.")

    df = load_parquet(snap_path)

    # pro Round nur eine NAV-Zeile (einfach unique nehmen)
    navs = df[["round","t_plus_1","portfolio_value_t1"]].drop_duplicates("round").set_index("round")

    fig, ax = plt.subplots(figsize=(8,4))
    navs["portfolio_value_t1"].plot(ax=ax, marker="o")
    ax.set_title("Portfolio NAV über Runden")
    ax.set_ylabel("Value")
    ax.grid(True)

    if save:
        out_path = path / "portfolio_nav.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert unter {out_path}")
    else:
        plt.show()
