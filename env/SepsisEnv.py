import random
import gym
from gym import spaces
import pandas as pd
import numpy as np

MIN_OBSERVATION = -25.545204389483803
MAX_OBSERVATION = 335.0

class SepsisEnv(gym.Env):
    """A Sepsis environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df):
        super(SepsisEnv, self).__init__()

        self.df = df
        self.reward_range = (-2.0, 1.0)
        # index of where each patient begins
        self.patient_start_index = list(df.loc[df['index']==0].index)
        self.index = 0
        
        # Only two possible actions, 0 for non-sepsis,
        # 1 for sepsis
        n_actions = 2
        self.action_space = spaces.Discrete(n=n_actions)
        
        # Observation space is a feature vector of 41 vital signs, 
        # lab values, and other demographic information
        self.observation_space = spaces.Box(
            low=MIN_OBSERVATION, high=MAX_OBSERVATION, shape=(1, 42), dtype=np.float16)

    def _next_observation(self):
        '''Set the next observation state'''
        obs = np.array([
        self.df.loc[self.current_step, ~self.df.columns.isin(['SepsisLabel',
                            'patient', 
                            'zeros_reward',
                            'ones_reward',
                            'end_episode'])].values 
        ])

        return obs


    def step(self, action):
        '''Execute one time step within the environment'''
        
        self.current_step += 1

        done = False
        if action == 0:
            reward = self.df.loc[self.current_step, ['zeros_reward']]
        else:
            reward = self.df.loc[self.current_step, ['ones_reward']]
        
        end_episode = self.df.loc[self.current_step, ['end_episode']]
        if end_episode.values == 1:
            done = True
            self.index += 1

        obs = self._next_observation()

        return obs, reward, done, {}

    
    def reset(self):
        '''Reset the state of the environment to an initial state'''
        
        # Set the current step to an index  within the data frame
        self.current_step = self.patient_start_index[self.index]

        return self._next_observation()

    def render(self, mode='human', close=False):
        '''Render the environment to the screen'''
        
        print('current step' ,self.current_step)
        print('index', self.index)  


