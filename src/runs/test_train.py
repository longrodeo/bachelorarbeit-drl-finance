# C:\Dev\Bachelorarbeit\src\runs\test_train.py
from pathlib import Path
import src.state.state_builder as sb
from src.splits.cpcv import iter_windows_from_yaml
from src.env.data_builder import load_data_for_windows, build_env_segment
from src.utils.paths import CONFIG_DIR

# 1) CPCV-Folds aus YAML
splits_yaml = CONFIG_DIR / "splits" / "splits_cpcv.yaml"
folds = iter_windows_from_yaml(str(splits_yaml))
print(f"[CPCV] #folds={len(folds)}  first fold:", folds[0])

# 2) Panel passend zu CPCV laden (jahresweise, automatisch aus den Folds)
panel = load_data_for_windows(folds, strategy="cpcv", features_source="features_v1")
print("[PANEL]", panel.index.min(), "→", panel.index.max(),
      "| rows:", len(panel), "| cols:", panel.shape[1])

# 3) Spec laden (als Objekt!)
spec = sb.load_spec(str(CONFIG_DIR / "state_config" / "state0.yml"))

# 4) Erstes Train-Segment als Env bauen (ohne Recorder)
seg = folds[0]["train"][0]  # ("YYYY-MM-DD","YYYY-MM-DD")
env = build_env_segment(panel, seg,
                        state_spec=spec,
                        reward_kind="log",
                        with_recorder=False,
                        out_dir=None)

base = env.unwrapped  # Gym-Warnungen vermeiden
print("[ENV]", "segment:",
      base.dates[base.start_idx], "→", base.dates[base.end_idx_exclusive-1],
      "| n_steps:", base.end_idx_exclusive - base.start_idx - 1,
      "| assets:", base.assets)

env.close()
