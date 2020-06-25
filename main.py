
# Filter tensorflow version warnings
import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

from tqdm import tqdm
import pandas as pd
import numpy as np
from add_reward import add_reward_df, add_end_episode_df
from load_data import load_data
from env.SepsisEnv import SepsisEnv
from stable_baselines import PPO2, A2C
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq import DQN, MlpPolicy as DQN_MlpPolicy, LnMlpPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
import gym


def train_model(env, model, total_timesteps, iterations):
    model.learn(total_timesteps=total_timesteps)
    reward_list = []
    obs = env.reset()
    patient_count = 0
    for _ in tqdm(range(iterations)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        reward_list.append(rewards)
        if done:
            patient_count += 1
            obs = env.reset()
        # env.render()
    print('Model: ', model.__class__)
    print('Policy: ', model.policy)
    print('Total patients: ', patient_count)
    print('Total reward:', sum(reward_list))

def train_baseline_models(df, iterations, constant=False):
    reward_list = []
    env = DummyVecEnv([lambda: SepsisEnv(df)])
    obs = env.reset()
    patient_count = 0
    for _ in tqdm(range(iterations)): 
        if constant:
            obs, rewards, done, info = env.step(np.array([0]))
        else:
            action = np.random.choice([0,1], size=1)
            obs, rewards, done, info = env.step(action)
        reward_list.append(rewards)
        if done:
            patient_count += 1
            obs = env.reset()
    if constant:
        print('Model: All Non-sepsis')
    else:
        print('Model: Random')
    print('Total patients: ', patient_count)
    print('Total reward:', sum(reward_list))

if __name__ == '__main__':
    df = load_data()
    df = add_reward_df(df)
    df = add_end_episode_df(df)
    df = df.reset_index()
    total_timesteps = 20_000
    iterations = 50_000

    env = DummyVecEnv([lambda: SepsisEnv(df)])

    models = [
        PPO2(MlpPolicy, env, verbose=0),
        PPO2(MlpLstmPolicy, env, nminibatches=1, verbose=0),
        PPO2(MlpLnLstmPolicy, env, nminibatches=1, verbose=0),
        A2C(MlpPolicy, env, lr_schedule='constant'),
        A2C(MlpLstmPolicy, env, lr_schedule='constant'),
        DQN(env=env,
            policy=DQN_MlpPolicy,
            learning_rate=1e-3,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            ),
        DQN(env=env,
            policy=LnMlpPolicy,
            learning_rate=1e-3,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            )
    ]

    for model in models:
        env = DummyVecEnv([lambda: SepsisEnv(df)])
        train_model(env=env, model=model,
                    total_timesteps=total_timesteps, iterations=iterations)

    train_baseline_models(df, iterations=iterations, constant=False)
    train_baseline_models(df, iterations=iterations, constant=True)