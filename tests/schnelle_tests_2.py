from pathlib import Path
from accounting.evaluator import compute_rewards_from_snapshots
from accounting.reward import RewardSpec

acc = Path("accounting_demo")
df_log  = compute_rewards_from_snapshots(acc, spec=RewardSpec(kind="log"), out_name="rewards_log.parquet")
df_ic   = compute_rewards_from_snapshots(acc, spec=RewardSpec(kind="icvar",    alpha=0.05, window=252, lambda_=1.0, icvar_mode="ex_ante"), out_name="rewards_icvar.parquet")
df_icdd = compute_rewards_from_snapshots(acc, spec=RewardSpec(kind="icvar_dd", alpha=0.05, window=252, lambda_=1.0, gamma=1.0, icvar_mode="ex_ante"), out_name="rewards_icvar_dd.parquet")
print(df_log.tail(2)); print(df_ic.tail(2)); print(df_icdd.tail(2))
