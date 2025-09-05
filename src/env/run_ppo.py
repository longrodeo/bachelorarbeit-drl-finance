# train/run_ppo.py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

# ... baue hier dein env wie im Smoke-Test ...
env = TradingEnv(...)

model = PPO("MlpPolicy", env, verbose=1, seed=42)
model.learn(total_timesteps=50_000)
