# Toggles for which environments to train in
DEADLOCK_TRAINING = True
EXPLORATION_TRAINING = False
PPO_TRANING = True

# Frequency for model saving
DEADLOCK_ITERATION_SAVE_INTERVAL = 5
EXPLORATION_ITERATION_SAVE_INTERVAL = 5
PPO_ITERATION_SAVE_INTERVAL = 5

# Training iteractions and timesteps per iteration
MAX_DEADLOCK_ITERATIONS = 50
DEADLOCK_TIMESTEPS = 10000
MAX_EXPLORATION_ITERATIONS = 50
EXPLORATION_TIMESTEPS = 10000
MAX_PPO_ITERATIONS = 50
PPO_TIMESTEPS = 25000
MAX_TESTING_TIMESTEPS = 100

# Multiprocessing
NUM_CPU = 4

# Input file (petri net)
FILENAME = "cost_net.json"
OUTPUT = "output.csv"

# Directories
models_dir = f"models/Deadlock-PPO/"
logdir = f"logs/"

# 
ONE_TIME_KEY = "once"
EXTRAPOLATED_KEY = "extrapolated"
ONE_TIME_INDEX = 0
EXTRAPOLATED_INDEX = 1