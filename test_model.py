import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
import os
from petrienv import PetriEnv
from deadlockenv import DeadlockEnv
import time
import json


FILENAME = "poi_net.json"

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

models_dir = f"models/PPO/"
logdir = f"logs/"

# Create directories if not existant
if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# Load initial network from json
with open(FILENAME, encoding='utf-8') as fh:
    json_obj = json.load(fh)


# Setup model and environment
env = PetriEnv(json_obj)
# env = DeadlockEnv(json_obj)

# Mask
env.reset(0, {})
env = ActionMasker(env, mask_fn)  # Wrap to enable masking
model = MaskablePPO.load("models/Deadlock-PPO/Deadlock-PPO-5.zip")
# model = MaskablePPO.load("models/PPO/PPO-50.zip")
obs, info = env.reset()

# Non-Mask
# env.reset(0)
# model = PPO.load("models/1696260194/PPO-1.zip")
# obs, info = env.reset(0)

done = False

print("intial observation:")
for i, row in enumerate(obs):
    print(env.place_names[i], "\t", row[0])
print('---------')

cummulative_reward = 0.0
iteration = 0
max_iters = 10000
while not done:
    iteration += 1
    # Mask
    action, _states = model.predict(obs, action_masks=mask_fn(env))
    # Non-Mask
    # action, _states = model.predict(obs)
    obs, rewards, done, shortcut, info = env.step(action)
    cummulative_reward += rewards
    print("transition:", env.transition_names[action])
    # print("reward", rewards)
    # print("cumulative reward", cummulative_reward)
    # print("resulting observation:")
    # for i, row in enumerate(obs):
    #     print(env.place_names[i], "\t", row[0])
    print('---------')
    if iteration > max_iters or done:
        done = True
        print("reward", rewards)
        print("cumulative reward", cummulative_reward)
        print("resulting observation:")
        for i, row in enumerate(obs):
            print(env.place_names[i], "\t", row[0])
        print("Ending due to iteration cap or done flag")
        print("iteration", iteration, "Max", max_iters)
    #     print("cumulative reward", cummulative_reward)
