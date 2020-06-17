import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.SepsisEnv import SepsisEnv
from load_data import load_data
from add_reward import add_reward_df

import pandas as pd

df = load_data()
df = add_reward_df(df)
df = df.reset_index()

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: SepsisEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
