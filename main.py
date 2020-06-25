import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.SepsisEnv import SepsisEnv
from load_data import load_data
from add_reward import add_reward_df, add_end_episode_df
import pandas as pd
from tqdm import tqdm


def train_model(df, env, model, total_timesteps, ):
    model = PPO2(MlpPolicy, env, verbose=0)
    model.learn(total_timesteps=20)
    reward_list = []
    obs = env.reset()
    patient_count = 0
    for _ in tqdm(range(200)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        reward_list.append(rewards)
        if done:
            patient_count += 1
            obs = env.reset()
        env.render()



df = load_data()
df = add_reward_df(df)
df = add_end_episode_df(df)
df = df.reset_index()

env = DummyVecEnv([lambda: SepsisEnv(df)])
model = PPO2(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=20)
reward_list = []
obs = env.reset()
patient_count = 0
for _ in tqdm(range(200)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    reward_list.append(rewards)
    if done:
        patient_count += 1
        obs = env.reset()
    env.render()

print('Total patients: ', patient_count)
print('Total reward:', sum(reward_list))