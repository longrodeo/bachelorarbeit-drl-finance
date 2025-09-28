# src/rl/agent_factory.py
from __future__ import annotations
from typing import Any, Dict

from src.rl.cnn_extractor import CNN1DExtractor
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

def make_agent(algo: str, env, tensorboard_log: str | None, seed: int,
               algo_kwargs: Dict[str, Any] | None = None) -> BaseAlgorithm:
    policy = "MultiInputPolicy"
    policy_kwargs: Dict[str, Any] = {}
    algo_kwargs = dict(algo_kwargs or {})

    policy_kwargs["features_extractor_class"] = CNN1DExtractor
    if algo.lower() == "ppo":
        return PPO(policy, env, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, seed=seed, verbose=1,
                   **algo_kwargs)
    if algo.lower() == "sac":
        return SAC(policy, env, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, seed=seed, verbose=1,
                   **algo_kwargs)
    raise ValueError(f"Unknown algo: {algo}")
