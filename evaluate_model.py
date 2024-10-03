import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
from fullcostenv import FullCostEnv
from explorationenv import ExplorationEnv
from deadlockenv import DeadlockEnv
from constants import *
from utils import *
import csv
import argparse

# Inspect transition metadata to determine interaction phase type (setup or simulation)
def is_sim_type(transition):
    for obj in transition["metaData"]:
        if obj["type"] == "simulation":
            return "Simulation"
    return "Setup"

# Inspect transition metadata to determine ergonomic and monetary costs
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

# Masking function, call the environment's masking function
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


def run(arguments):
    # Determine which input json file to use 
    f = FILENAME # default "cost_net.json"
    if arguments.inputfile is not None:
        f = arguments.inputfile
    
    # Determine naming scheme for model output
    outputFilename = f.replace(".json", "") + "-output.csv"
    if arguments.output is not None:
        outputFilename = arguments.output

    # Load petrinet data from json (transitions, places)
    [json_obj, weights] = LOAD_JOB_FILE(f)

    # Setup evaluation environment
    # env = FullCostEnv(json_obj, weights)
    # TODO: (remove) Temporary, testing environment
    env = ExplorationEnv(json_obj)

    # Reset and mask environment
    env.reset(0, {})
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # Load model for evaluation
    model = None
    if arguments.model is not None:
        model = MaskablePPO.load(arguments.model, weights_only=True)
    else:
        model = MaskablePPO.load("models/PetriNet-PPO/Exploration-final.zip")

    # Reset environment and get initial observation
    obs, _info = env.reset()

    done = False

    # Setup tracking
    cummulative_reward = 0.0
    iteration = 0
    action_sequence = []

    # Iterate until model finds the goal(s) or max timesteps are hit
    while not done:
        # Increase iteration count
        iteration += 1
        
        # Determine best action from current state
        action, _states = model.predict(obs, action_masks=mask_fn(env))
        
        # Step the model based on selected action
        obs, rewards, done, shortcut, info = env.step(action)

        # Update rewards
        cummulative_reward += rewards
        
        # Add selected action to sequence
        action_sequence.append((env.transition_ids[action], action))
        
        # If goal(s) are met or max timesteps are reached, mark as done and print
        if iteration >= MAX_TESTING_TIMESTEPS or done:
            done = True
            print("reward", rewards)
            print("cumulative reward", cummulative_reward)
            print("is goal met? + {0}".format(IS_GOAL(obs, env.goal_state)))
            print("Ending due to iteration cap or done flag")
            print("iteration", iteration, "Max", MAX_TESTING_TIMESTEPS)


    currentTime = 0

    # Write output to CSV
    with open(outputFilename, "w+", newline='') as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(["Action", "Type", "Agent Assigned To Task", "From Standing Location", "To Standing Location", "From Hand Location", "To Hand Location", "Start Time (s)", "End Time (s)", "Hand Cost", "Arm Cost", "Shoulder Cost", "Whole Body Cost", "Monetary Cost", "Primitives", "MVC", "Hand Distance", "Stand Distance", "Is One Handed"])

        for (transition_id, action) in action_sequence:
            transition = json_obj["transitions"][transition_id]
            costs = find_costs(transition)
            duration = transition["time"]
            primtives = ""
            mvcs = ""
            isOneHanded = ""
            standDistanceTraveled = ""
            handDistanceTraveled = ""
            a = 0
            fromHandLocation = ""
            toHandLocation = ""
            fromStandLocation = ""
            toStandLocation = ""
            
            # Iterate over transition metadata to extract information for CSV output
            for m in transition["metaData"]:
                if m["type"] == "primitiveAssignment":
                    try:
                        primtives += json_obj["nameLookup"][m["value"][1]] + ";"
                    except:
                        a = 1
                elif m["type"] == "isOneHanded":
                    isOneHanded += "True" if m["value"][1] else "False" + ";"
                elif m["type"] == "mVC":
                    mvcs += str(m["value"][1]) + ";"
                elif m["type"] == "handTravelDistance":
                    handDistanceTraveled += str(m["value"][1]) + ";"
                elif m["type"] == "standTravelDistance":
                    standDistanceTraveled += str(m["value"][1]) + ";"
                elif m["type"] == "standing":
                    fromStandLocation = json_obj["nameLookup"][m["value"][0]]
                    toStandLocation = fromStandLocation
                elif m["type"] == "hand":
                    fromHandLocation = json_obj["nameLookup"][m["value"][0]]
                    toHandLocation = fromHandLocation
                elif m["type"] == "fromHandPOI":
                    fromHandLocation = json_obj["nameLookup"][m["value"][0]]
                elif m["type"] == "toHandPOI":
                    toHandLocation = json_obj["nameLookup"][m["value"][0]]
                elif m["type"] == "fromStandingPOI":
                    fromStandLocation = json_obj["nameLookup"][m["value"][0]]
                elif m["type"] == "toStandingPOI":
                    toStandLocation = json_obj["nameLookup"][m["value"][0]]

            # TODO: keep track of time, busy/working agents, who all is assigned to a task (multiple agent split)
            csv_writer.writerow([env.transition_names[action], is_sim_type(transition), "", fromStandLocation, toStandLocation, fromHandLocation, toHandLocation, currentTime, currentTime+duration, costs[0], costs[1], costs[2], costs[3], costs[4], primtives, mvcs, handDistanceTraveled, standDistanceTraveled, isOneHanded])
            currentTime += duration



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", type=str, default=None, help="")
    parser.add_argument("--model", type=str, default=None, help="")
    parser.add_argument("--output", type=str, default=None, help="")
    args = parser.parse_args()

    run(args)