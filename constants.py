# Toggles for which environments to train in
DEADLOCK_TRAINING = True
PPO_TRANING = True

# Frequency for model saving
DEADLOCK_ITERATION_SAVE_INTERVAL = 5
PPO_ITERATION_SAVE_INTERVAL = 1

# Training iteractions and timesteps per iteration
MAX_DEADLOCK_ITERATIONS = 10
DEADLOCK_TIMESTEPS = 10000
MAX_PPO_ITERATIONS = 50
PPO_TIMESTEPS = 10000
MAX_TESTING_TIMESTEPS = 10000

# Multiprocessing
NUM_CPU = 4

# Input file (petri net)
FILENAME = "cost_net.json"

# 
ONE_TIME_KEY = "once"
EXTRAPOLATED_KEY = "extrapolated"
ONE_TIME_INDEX = 0
EXTRAPOLATED_INDEX = 1