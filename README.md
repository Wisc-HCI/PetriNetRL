# PetriNetRL
## Description
A makeshift reinforcement learning approach for petri nets. Still a WIP.

## Setup

[Setup Stable Baselines](https://stable-baselines3.readthedocs.io/en/master/guide/install.html#prerequisites)

If you want to view the training progress: [Tensorboard Docker Image](https://hub.docker.com/r/volnet/tensorflow-tensorboard)

## Running
There are 2 environments: `deadlockenv.py` and `FullCostEnv.py`. `deadlockenv.py` trains the network to avoid deadlocking, i.e. scenarios where there are no available actions to pick from. `FullCostEnv.py` trains the network on completing the collaborative process, deciding who is involved, what they do, and the order in which they do it.

`newlearn.py` is setup to first have the network trained in `deadlockenv.py` followed by `FullCostEnv.py`. So if you want to train your network, supply the appropriate json file, update `newlearn.py` and execute that file.