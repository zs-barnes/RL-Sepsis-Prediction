# RL-Sepsis-Prediction
(Public) Data Set from https://physionet.org/content/challenge-2019/1.0.0/

We're designing a reinforcement learning environment and model to classify patients with sepsis at each hour.

The RL environment is using OpenAI's gym : https://github.com/openai/gym

Creating custom gym environment and RL training code from <br />
https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

RL algorithms from the https://github.com/hill-a/stable-baselines package.

# Install dependencies
If using conda, create an environment with python 3.7:
`conda create -n rl_sepsis python=3.7`

Activate the environment:
`conda activate rl_sepsis`

Then, install the necessary packages:
`pip install -r requirements.txt`

# Clean data
We have upload training set A from the physionet competition into the repo.
To load and clean the data, run:

`make load_data`

This will take about 10 minutes, and the progress bar will be displayed using tqdm. It will create 
a `cache\` directory, (created from the cache_em_all package) where the cleaned data will be stored.

Now, in a notebook or .py file, you can load the data with  

```
from load_data import load_data
df = load_data()
```

where df is a pandas data frame. 

Alternatively, once you clone this repo you can open up `Load_Data.ipynb` and run all the cells.  If no error is thrown, then you have loaded the data successfully.


# Add Rewards
Using the utility function provided by the competition, 
we have added two columns that correspond to the reward
recieved at each hour depending on whether predicting a zero or a one.

To create the reward columns, run:
`make add_reward`

This should only take 10-15 seconds, and will add the file "training_setA_rewards" under the `cache\`
directory.

# Train Model
To see the RL train, simply run
`make train_model`.
Currently, the output contains future warnings, and the only output from the render function from our Gym environment is the current timestep, which corresponds to the index of the pandas dataframe. The training loss is printed from the stablebaselines Multi-layer Perceptron model.
