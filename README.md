# RL-Sepsis-Prediction

We designed a reinforcement learning environment and model to classify patients with sepsis at each hour.

A video presentation of our project can be found [here]().

The step-by-step results of our project can be found in our notebook [here](https://github.com/zs-barnes/RL-Sepsis-Prediction/blob/master/Viz.ipynb).

# Introduction

Sepsis is a life-threatening condition that arises when the body's response to infection causes injury to its tissues and organs. It is the most common cause of death for people who have been hospitalized, and results in a $15.4 billion annual cost in the US.  Early detection and treatment are essential for prevention and a 1-hour delay in antibiotic treatment can lead to 4% increase in hospital mortality.  Given the nature of our data as a multivariate timeseries of patient vital signs, this makes this an ideal classification problem to apply reinforcement learning methods to.

# Data

![physionet_logo](/images/physionet_logo.jpeg)

We used a public data set from the PhysioNet Computing Challenge [which can be downloaded here](https://physionet.org/content/challenge-2019/1.0.0/).

An explanation by the PhysioNet Challenge is given below:

Data used in the competition is sourced from ICU patients in three separate hospital systems.  

The data repository contains one file per subject (ex - training/p00101.psv).  Each training data file provides a table with measurements over time. Each column of the table provides a sequence of measurements over time (ex - heart rate over several hours), where the header of the column describes the measurement. Each row of the table provides a collection of measurements at the same time (ex - heart rate and oxygen level). The table is formatted in the following way:

![physionet_data_table](/images/physionet_data_table.png)

There are 40 time-dependent variables HR, O2Sat, Temp ..., HospAdmTime, which are described here. The final column, SepsisLabel, indicates the onset of sepsis according to the Sepsis-3 definition, where 1 indicates sepsis and 0 indicates no sepsis. Entries of NaN (not a number) indicate that there was no recorded measurement of a variable at the time interval.

![timeseries](/images/multivariate_timeseries.png)

# RL Framework

Our Reinforcement Learning environment is using [OpenAI's gym](https://github.com/openai/gym).

For step-by-step instructions for how to set up your environment, see the section below on *Setup*.

To create this environment, we referenced:
* How to create a custom gym environment with RL training code [here](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e).
* Creating RL algorithms using the Stable Baselines package [here](https://github.com/hill-a/stable-baselines).

We can briefly frame our reinforcement learning problem as such:

* Environment: SepsisEnv modeled using OpenAI Gym, where we have a sequential multivariate timeseries of patients' vital signs
* Agent: A binary classifier that predicts whether patients have sepsis or not
* States: Each timestep that contains multiple patient vital signs taken at the same time
* Actions: Binary prediction of whether a patient has sepsis (1) or does not (0)
* Rewards: The calculated score between 1 and -2 based on the utility function calculated from true/false positive and true/false negative rates

The algorithm will be evaluated by its performance as a binary classifier using a utility function created by the [PhysioNet Challenge](https://physionet.org/content/challenge-2019/1.0.0/). This utility function rewards classifiers for early predictions of sepsis and penalizes them for late predictions and for predictions of sepsis in non-sepsis patients.

The PhysioNet Challenge defines a score U(s,t) for each prediction.  This will be done for each line in the data file that represents each patient s and each time interval t:

![physionet_utility](/images/physionet_utility.png)

The following figure shows the utility function for a sepsis patient with t_sepsis = 48 as an example (figure from [PhysioNet Challenge](https://physionet.org/content/challenge-2019/1.0.0/)):

![physionet_utility_plot](/images/physionet_utility_plot.png)

# Evaluation

We then compared performance across multiple algorithms.  You can check out our notebook [here](https://github.com/zs-barnes/RL-Sepsis-Prediction/blob/master/Viz.ipynb) for more.

In total, we compare:

Proximal Policy Optimization Algorithm + Multi-Layer Perceptron
Proximal Policy Optimization Algorithm + Multi-Layer Perceptron Long-Short Term Memory
Proximal Policy Optimization Algorithm + Multi-Layer Perceptron Long-Short Term Memory with Layer Normalization
Synchronous, deterministic variant of Asynchronous Advantage Actor Critic + Multi-Layer Perceptron
Synchronous, deterministic variant of Asynchronous Advantage Actor Critic + Multi-Layer Perceptron Long-Short Term Memory
Deep Q Network + Multi-Layer Perceptron
Deep Q Network + Multi-Layer Perceptron Long-Short Term Memory

The below plot nicely summarizes our results, with both versions of our Deep Q-Learning Network with Multi-Layer Perceptrons performing the best, all the combinations of A2C and Proximal Policy models performing worse than the Deep Q-Learning Networks, and our random baseline model performing the worst as expected:

![visualization_anim](/images/visualization_anim.svg)

Having both our Deep Q-Learning Networks perform the best makes sense since it combines Q-Learning with the power of deep neural networks to let RL work for complex, high-dimensional environments like our multivariate space of all the patient's vital signs.

For future direction, we will attempt to tease out the differences between the different Deep Q-Learning Networks and explore the potential benefits or pitfalls of providing layer normalization versus not. Since this is medical data where each feature is interpretable, this also lends itself quite well to feature engineering depending upon domain expertise input provided by a medical professional. We can also run this on alternative data sets to validate our work outside these three hospitals to see if this is generalizable.

-----

# Setup

## 1) Install dependencies
If using conda, create an environment with python 3.7:
`conda create -n rl_sepsis python=3.7`

Activate the environment:
`conda activate rl_sepsis`

Then, install the necessary packages:
`pip install -r requirements.txt`

## 2) Clean data
We have uploaded training set A from the physionet competition into the repo.
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


## 3) Add Rewards
Using the utility function provided by the competition, 
we have added two columns that correspond to the reward
received at each hour depending on whether predicting a zero or a one.

To create the reward columns, run:
`make add_reward`

This should only take 10-15 seconds, and will add the file "training_setA_rewards" under the `cache\`
directory.

## 4) Train Model
To see the RL train, simply run
`make train_model`.
Currently, the output contains future warnings, and the only output from the render function from our Gym environment is considered in the current timestep, which corresponds to the index of the pandas dataframe. The training loss is printed from the stable baselines Multi-layer Perceptron model.

## 5) Results
To see graphical results of performance for the different baseline models, [see our visualization notebook](https://github.com/zs-barnes/RL-Sepsis-Prediction/blob/master/Viz.ipynb).  
