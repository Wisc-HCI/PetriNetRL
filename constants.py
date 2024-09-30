import numpy as np

# Toggles for which environments to train in
DEADLOCK_TRAINING = True
EXPLORATION_TRAINING = True
FULL_COST_TRAINING = False

# Frequency for model saving
DEADLOCK_ITERATION_SAVE_INTERVAL = 5
EXPLORATION_ITERATION_SAVE_INTERVAL = 5
FULL_COST_ITERATION_SAVE_INTERVAL = 5

# Training iterations and timesteps per iteration
MAX_DEADLOCK_ITERATIONS = 30
DEADLOCK_TIMESTEPS = 10000
MAX_EXPLORATION_ITERATIONS = 100
EXPLORATION_TIMESTEPS = 10000
MAX_FULL_COST_ITERATIONS = 50
FULL_COST_TIMESTEPS = 10000
MAX_TESTING_TIMESTEPS = 1000

# Reward definitions
DEADLOCK_OCCURS_REWARD = -100000.0
INVALID_STATE_REWARD = -100000.0
GOAL_FOUND_REWARD = 10000.0
STEP_REWARD = -0.1
FIRST_TIME_ACTION_REWARD = 100.0

# Input file (petri net)
# TODO: move to argparse's
FILENAME = "cost_net.json"
OUTPUT = "output.csv"

# Directories
models_dir = f"models/PetriNet-PPO/"
logdir = f"logs/"

# 
ONE_TIME_KEY = "once"
EXTRAPOLATED_KEY = "extrapolated"
ONE_TIME_INDEX = 0
EXTRAPOLATED_INDEX = 1

# ========================
# Functions
# ========================

# Evaluate network to determine whether the goal state(s) has been satisfied
def IS_GOAL(marking, goal_state):
    places = np.where(goal_state == 1)[0]
    for i in places:
        if marking[i][0] < 1:
            return False
    return True