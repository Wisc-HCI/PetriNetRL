import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
import os
from petrienv import PetriEnv
from explorationenv import ExplorationEnv
from deadlockenv import DeadlockEnv
import time
import json
from constants import *
import csv

def is_sim_type(transition):
    for obj in transition["metaData"]:
        if obj["type"] == "simulation":
            return "Simulation"
    return "Setup"

def find_costs(transition):
    costs = [0] * 5
    for obj in transition["metaData"]:
        if obj["type"] == "ergoHand":
            costs[0] += obj["value"][1]
        if obj["type"] == "ergoArm":
            costs[1] += obj["value"][1]
        if obj["type"] == "ergoShoulder":
            costs[2] += obj["value"][1]
        if obj["type"] == "ergoWholeBody":
            costs[3] += obj["value"][1]
    for obj in transition["cost"]:
        if obj["category"] == "monetary":
            costs[4] += obj["value"]

    return costs

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

# Load initial network from json
with open(FILENAME, encoding='utf-8') as fh:
    json_obj = json.load(fh)

# Setup model and environment
env = PetriEnv(json_obj)
# env = DeadlockEnv(json_obj)
# env = ExplorationEnv(json_obj)

# Mask
env.reset(0, {})
env = ActionMasker(env, mask_fn)  # Wrap to enable masking
model = MaskablePPO.load("models/Deadlock-PPO/Deadlock-PPO-ppo-final.zip")
# model = MaskablePPO.load("models/PPO/PPO-50.zip")
obs, info = env.reset()

# Non-Mask
# env.reset(0)
# model = PPO.load("models/1696260194/PPO-1.zip")
# obs, info = env.reset(0)

done = False

def is_goal(marking, goal_state):
    places = np.where(goal_state == 1)[0]
    for i in places:
        if marking[i][0] < 1:
            return False
    return True

print("intial observation:")
for i, row in enumerate(obs):
    print(env.place_names[i], "\t", row[0])
print('---------')

cummulative_reward = 0.0
iteration = 0
action_sequence = []
while not done:
    iteration += 1
    # Mask
    action, _states = model.predict(obs, action_masks=mask_fn(env))
    # Non-Mask
    # action, _states = model.predict(obs)
    obs, rewards, done, shortcut, info = env.step(action)
    cummulative_reward += rewards
    # print("transition:", env.transition_names[action])
    action_sequence.append((env.transition_ids[action], action))
    # print("reward", rewards)
    # print("cumulative reward", cummulative_reward)
    # print("resulting observation:")
    # for i, row in enumerate(obs):
    #     print(env.place_names[i], "\t", row[0])
    # print('---------')
    if iteration >= MAX_TESTING_TIMESTEPS or done:
        done = True
        # print(action_sequence)
        print("reward", rewards)
        print("cumulative reward", cummulative_reward)
        # print("resulting observation:")
        # for i, row in enumerate(obs):
        #     print(env.place_names[i], "\t", row[0])
        print("is goal met? + {0}".format(is_goal(obs, env.goal_state)))
        print("Ending due to iteration cap or done flag")
        print("iteration", iteration, "Max", MAX_TESTING_TIMESTEPS)
    #     print("cumulative reward", cummulative_reward)


currentTime = 0
with open(OUTPUT, "w+", newline='') as fh:
    csv_writer = csv.writer(fh)
    csv_writer.writerow(["Action", "Type", "Agent Assigned", "Start Time (s)", "End Time (s)", "Hand Cost", "Arm Cost", "Shoulder Cost", "Whole Body Cost", "Monetary Cost"])

    for (transition_id, action) in action_sequence:
        transition = json_obj["transitions"][transition_id]
        costs = find_costs(transition)
        duration = transition["time"]
        # TODO: keep track of time, busy/working agents, who all is assigned to a task (multiple agent split)
        csv_writer.writerow([env.transition_names[action], is_sim_type(transition), "", currentTime, currentTime+duration, costs[0], costs[1], costs[2], costs[3], costs[4]])
        currentTime += duration