Final Project Check-in 

- Name(s): Zachary Barnes & Mundy Reimer 
- Finalized Research Question (1): Can a reinforcement learning agent learn to classify sepsis patients at each hour given a multivariate timeseries of patients' vitals?
- The following working code in GitHub (3): 
    * All code [found here](https://github.com/zs-barnes/RL-Sepsis-Prediction)
    * How to setup environment, load data, create rewards, and train model [here](https://github.com/zs-barnes/RL-Sepsis-Prediction/blob/master/README.md)
    * Comparison of results of random actions vs. learning [here](https://github.com/zs-barnes/RL-Sepsis-Prediction/blob/master/gym_env_w_random.ipynb)
    - A environment (1)  
    - An agent that performs random actions in the environment (1) 
    - An agent that learns based on the environment (1)
- List of ideas to finish project (1):
    * Try out a different optimizer besides Multi-Layer Perceptron (MLP), maybe RNN-LSTM, or CNN-LSTM?
    * Figure out how to configure our training so that individual patients are separate and each patient has their own associated episodes (right now all patients' data are grouped together as one continuous multivariate timeseries)
    * Figure out when to reset the state of our environment between patients (check *done* flag in *step* function)
    * Figure out how to configure our random sampling that is currently being done on our time series per patient
    * Find optimal total time steps needed (right now it is set at 20_000)
    * Create a visualization / rendering of our learning (right now we rely on our text-formatted output of *env.render*)
    
