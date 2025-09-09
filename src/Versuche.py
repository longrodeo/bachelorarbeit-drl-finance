import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import StopTrainingOnStepThreshold
import gymnasium as gym
print("SB3:", sb3.__version__)
print("Gymnasium:", gym.__version__)
print("StopTrainingOnStepThreshold ok:", StopTrainingOnStepThreshold is not None)